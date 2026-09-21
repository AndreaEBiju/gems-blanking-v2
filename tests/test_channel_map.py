"""The channel table, cohort inference, and the profile this project shares.

None of this needs ``GEMSBlanking``: the geometry and the profile format are ours
to hold, and the profile is plain JSON with a schema their UI also writes. Every
test passes an explicit ``root`` so no test can touch a real home directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from gems_blanking_v2.io.channel_map import (
    CUFF_CONTACTS,
    HW_TRIPOLE_CHANNEL_COUNT,
    ChannelMap,
    infer_config,
    load_geometry,
    load_profile,
    meta_path,
    profile_path,
    profiles_dir,
    resolve_channel_map,
    save_geometry,
    save_profile,
    scale_to_uv,
)
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.types import ChannelInfo


def nerve_channel(
    index: int,
    cuff: str,
    contact: int,
    *,
    rostral_end: int | None = 1,
    config: str = "independent",
) -> ChannelInfo:
    """One cuff contact."""
    return ChannelInfo(
        index=index,
        name=f"{cuff}VN{contact}",
        role="nerve",
        cuff_id=cuff,
        contact_index=contact,
        rostral_end=rostral_end,
        config=config,  # type: ignore[arg-type]
    )


def stomach_channel(index: int, n: int, config: str = "independent") -> ChannelInfo:
    """One stomach channel."""
    return ChannelInfo(
        index=index,
        name=f"ANT{n}",
        role="stomach",
        cuff_id=None,
        contact_index=None,
        rostral_end=None,
        config=config,  # type: ignore[arg-type]
    )


def new_cohort_map(animal: str = "J", *, rostral_end: int | None = 1) -> ChannelMap:
    """Return the nine-channel new cohort: two cuffs of three, three stomach."""
    channels: list[ChannelInfo] = []
    index = 0
    for cuff in ("L", "R"):
        for contact in (1, 2, 3):
            channels.append(nerve_channel(index, cuff, contact, rostral_end=rostral_end))
            index += 1
    for n in (1, 2, 3):
        channels.append(stomach_channel(index, n))
        index += 1
    return ChannelMap(animal=animal, channels=channels, units="uV")


def old_cohort_map(animal: str = "F") -> ChannelMap:
    """Return the five-channel old cohort: a pre-formed tripole per nerve."""
    channels = [
        ChannelInfo(
            index=0, name="LVN", role="nerve", cuff_id="L", contact_index=None,
            rostral_end=None, config="hw_tripole",
        ),
        ChannelInfo(
            index=1, name="RVN", role="nerve", cuff_id="R", contact_index=None,
            rostral_end=None, config="hw_tripole",
        ),
        *[stomach_channel(i, i - 1, config="hw_tripole") for i in (2, 3, 4)],
    ]
    return ChannelMap(animal=animal, channels=channels, units="uV")


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("units", "factor"), [("V", 1e6), ("mV", 1e3), ("uV", 1.0)])
def test_unit_scaling_matches_convertunits_m(units: str, factor: float) -> None:
    """``processing_new/convertUnits.m``: V->uV 1e6, mV->uV 1e3."""
    assert scale_to_uv(units) == factor  # type: ignore[arg-type]


def test_an_unknown_unit_is_rejected_rather_than_assumed() -> None:
    """Guessing here is a factor of 10^6, so there is no default."""
    with pytest.raises(ValueError, match="unknown units"):
        scale_to_uv("microvolts")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown units"):
        ChannelMap(animal="J", channels=new_cohort_map().channels, units="counts")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# config inference
# ---------------------------------------------------------------------------


def test_five_channels_without_geometry_is_the_old_cohort() -> None:
    assert infer_config(HW_TRIPOLE_CHANNEL_COUNT, has_contact_index=False) == "hw_tripole"


def test_contact_index_means_independent_whatever_the_channel_count() -> None:
    """Individually recorded contacts are the new cohort by definition."""
    for n in (5, 9, 12):
        assert infer_config(n, has_contact_index=True) == "independent"


def test_an_unfamiliar_layout_without_geometry_is_refused() -> None:
    """A wrong config silently changes what every later step reads."""
    with pytest.raises(ValueError, match="cannot infer config"):
        infer_config(7, has_contact_index=False)


def test_both_cohorts_report_their_own_config() -> None:
    assert new_cohort_map().config == "independent"
    assert old_cohort_map().config == "hw_tripole"
    assert new_cohort_map().n_channels == 9
    assert old_cohort_map().n_channels == HW_TRIPOLE_CHANNEL_COUNT


# ---------------------------------------------------------------------------
# table validation
# ---------------------------------------------------------------------------


def test_a_table_with_duplicate_indices_is_rejected() -> None:
    channels = [nerve_channel(0, "L", 1), nerve_channel(0, "L", 2)]
    with pytest.raises(ValueError, match="unique and ascending"):
        ChannelMap(animal="J", channels=channels)


def test_a_table_with_out_of_order_indices_is_rejected() -> None:
    channels = [nerve_channel(1, "L", 1), nerve_channel(0, "L", 2)]
    with pytest.raises(ValueError, match="unique and ascending"):
        ChannelMap(animal="J", channels=channels)


def test_names_differing_only_by_case_are_rejected() -> None:
    """Cross-platform rule 8's reasoning: such a pair collides silently."""
    channels = [
        ChannelInfo(0, "LVN1", "nerve", "L", 1, 1, "independent"),
        ChannelInfo(1, "lvn1", "nerve", "L", 2, 1, "independent"),
    ]
    with pytest.raises(ValueError, match="case-insensitively"):
        ChannelMap(animal="J", channels=channels)


def test_a_table_mixing_configs_is_rejected() -> None:
    channels = [
        ChannelInfo(0, "LVN1", "nerve", "L", 1, 1, "independent"),
        ChannelInfo(1, "RVN", "nerve", "R", None, None, "hw_tripole"),
    ]
    with pytest.raises(ValueError, match="one config"):
        ChannelMap(animal="J", channels=channels)


def test_contacts_are_returned_in_contact_order() -> None:
    mapping = new_cohort_map()
    assert [c.contact_index for c in mapping.contacts("L")] == [1, 2, 3]
    assert len(mapping.contacts("R")) == CUFF_CONTACTS
    assert mapping.cuffs == ["L", "R"]


# ---------------------------------------------------------------------------
# rostral_end
# ---------------------------------------------------------------------------


def test_a_known_rostral_end_makes_direction_valid() -> None:
    assert new_cohort_map(rostral_end=1).direction_valid


def test_an_absent_rostral_end_invalidates_direction_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """It cannot be reconstructed once the animal is gone, so it is never guessed."""
    mapping = new_cohort_map(rostral_end=None)
    assert not mapping.direction_valid

    with caplog.at_level(logging.WARNING, logger="gems_blanking_v2.io.channel_map"):
        mapping.warn_if_direction_unknown("gems_j_t01_bl_120000")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "rostral_end" in message
    assert "unsigned" in message
    assert "direction_valid=False" in message
    assert "gems_j_t01_bl_120000" in message


def test_a_valid_direction_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="gems_blanking_v2.io.channel_map"):
        new_cohort_map(rostral_end=1).warn_if_direction_unknown("rec")
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_the_old_cohort_has_no_direction_because_it_has_no_contacts() -> None:
    """A hardware tripole has no inter-contact delay to sign."""
    assert not old_cohort_map().direction_valid


def test_direction_needs_every_cuff_not_just_one() -> None:
    channels = [
        nerve_channel(0, "L", 1, rostral_end=1),
        nerve_channel(1, "L", 2, rostral_end=1),
        nerve_channel(2, "R", 1, rostral_end=None),
        nerve_channel(3, "R", 2, rostral_end=None),
    ]
    assert not ChannelMap(animal="J", channels=channels).direction_valid


# ---------------------------------------------------------------------------
# meta.json - the single source of truth
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    """Return a fake gems_root with the real layout."""
    return GemsStore.initialise(tmp_path / "gems")


def test_geometry_round_trips_through_meta_json(store: GemsStore) -> None:
    original = new_cohort_map("J")
    path = save_geometry(original, "t01", store, mirror_to_profile=False)

    assert path == meta_path(store, "J", "t01")
    assert store.relpath(path) == "data/J/t01/meta.json"

    loaded = load_geometry("J", "t01", store)
    assert loaded is not None
    assert loaded.channels == original.channels
    assert loaded.units == original.units
    assert loaded.config == "independent"


def test_meta_json_is_per_session_not_per_animal(store: GemsStore) -> None:
    """Two sessions of one animal can differ - a cuff may be re-implanted."""
    save_geometry(new_cohort_map("J", rostral_end=1), "t01", store, mirror_to_profile=False)
    save_geometry(new_cohort_map("J", rostral_end=None), "t02", store, mirror_to_profile=False)

    first = load_geometry("J", "t01", store)
    second = load_geometry("J", "t02", store)
    assert first is not None
    assert second is not None
    assert first.direction_valid
    assert not second.direction_valid


def test_writing_geometry_preserves_another_writers_keys(store: GemsStore) -> None:
    """Task 03A's ``apply_corrections`` writes condition and who/when to this file.

    Two writers, one document: a geometry update must not drop the condition block,
    and vice versa. Same read-modify-write discipline as the profile.
    """
    path = meta_path(store, "J", "t01")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "animal": "J",
            "session": "t01",
            "condition": "bl",
            "condition_source": "human",
            "corrections": [{"who": "andrea", "when": "2026-09-01T00:00:00Z"}],
        }),
        encoding="utf-8",
        newline="\n",
    )

    save_geometry(new_cohort_map("J"), "t01", store, mirror_to_profile=False)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["condition"] == "bl"
    assert document["condition_source"] == "human"
    assert document["corrections"] == [{"who": "andrea", "when": "2026-09-01T00:00:00Z"}]
    assert len(document["channels"]) == 9


def test_meta_json_stores_no_paths_at_all(store: GemsStore) -> None:
    """Cross-platform rule 2 by construction: there is no path to get wrong."""
    path = save_geometry(new_cohort_map("J"), "t01", store, mirror_to_profile=False)
    text = path.read_text(encoding="utf-8")
    assert str(store.root) not in text
    assert "/" not in text.replace("data/J", "")  # no path-shaped values
    assert b"\r\n" not in path.read_bytes()


def test_an_absent_rostral_end_is_an_absent_key_in_meta_json(store: GemsStore) -> None:
    mapping = new_cohort_map("J", rostral_end=None)
    path = save_geometry(mapping, "t01", store, mirror_to_profile=False)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert all("rostral_end" not in ch for ch in document["channels"])
    assert "null" not in path.read_text(encoding="utf-8")


def test_a_corrupt_meta_json_is_not_silently_overwritten(store: GemsStore) -> None:
    """It holds another writer's data; clobbering it would destroy the condition."""
    path = meta_path(store, "J", "t01")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        save_geometry(new_cohort_map("J"), "t01", store, mirror_to_profile=False)
    with pytest.raises(ValueError, match="could not be read"):
        load_geometry("J", "t01", store)


def test_meta_json_without_geometry_raises(store: GemsStore) -> None:
    """A file holding only 03A's condition block has no geometry yet."""
    path = meta_path(store, "J", "t01")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"animal": "J", "condition": "bl"}), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="has no channels"):
        load_geometry("J", "t01", store)


def test_a_missing_meta_json_is_none(store: GemsStore) -> None:
    assert load_geometry("J", "t01", store) is None


# ---------------------------------------------------------------------------
# precedence: ours wins, theirs is a mirror
# ---------------------------------------------------------------------------


def test_meta_json_wins_over_the_profile_mirror(store: GemsStore, tmp_path: Path) -> None:
    """The ruling: ours is authoritative, theirs is a regenerable copy.

    Their ``from_review_session`` rebuilds the channel list from a dialog that
    knows nothing about ``rostral_end``, so if their file won, a UI session would
    silently destroy geometry that cannot be recovered once the animal is gone.
    """
    profiles = tmp_path / "profiles"
    save_profile(new_cohort_map("J", rostral_end=None), profiles)  # a stale mirror
    save_geometry(new_cohort_map("J", rostral_end=1), "t01", store, mirror_to_profile=False)

    resolved = resolve_channel_map("J", session="t01", store=store, profiles_root=profiles)
    assert resolved is not None
    assert resolved.direction_valid, "meta.json must win over the profile"


def test_an_explicit_map_wins_over_everything(store: GemsStore, tmp_path: Path) -> None:
    save_geometry(new_cohort_map("J", rostral_end=None), "t01", store, mirror_to_profile=False)
    explicit = new_cohort_map("J", rostral_end=3)
    resolved = resolve_channel_map(
        "J", explicit=explicit, session="t01", store=store, profiles_root=tmp_path
    )
    assert resolved is explicit


def test_falling_back_to_the_mirror_warns_that_it_is_a_copy(
    store: GemsStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    profiles = tmp_path / "profiles"
    save_profile(new_cohort_map("J"), profiles)

    with caplog.at_level(logging.WARNING, logger="gems_blanking_v2.io.channel_map"):
        resolved = resolve_channel_map("J", session="t01", store=store, profiles_root=profiles)

    assert resolved is not None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "mirror" in warnings[0].getMessage()
    assert "another tool may rewrite" in warnings[0].getMessage()


def test_no_geometry_anywhere_resolves_to_none(store: GemsStore, tmp_path: Path) -> None:
    assert resolve_channel_map("J", session="t01", store=store, profiles_root=tmp_path) is None


def test_saving_geometry_refreshes_the_mirror_and_marks_it_derived(
    store: GemsStore, tmp_path: Path
) -> None:
    """The mirror says where it came from, so nobody edits it expecting it to stick."""
    profiles = tmp_path / "profiles"
    save_geometry(new_cohort_map("J"), "t01", store, profiles_root=profiles)

    document = json.loads(profile_path("J", profiles).read_text(encoding="utf-8"))
    assignment = document["channel_assignment"]
    assert assignment["mirror_of"] == "data/J/t01/meta.json"
    assert "regenerate rather than edit" in assignment["mirror_note"]
    assert len(assignment["channels"]) == 9

    # And the mirror is still readable as a channel map.
    mirrored = load_profile("J", profiles)
    assert mirrored is not None
    assert mirrored.channels == new_cohort_map("J").channels


# ---------------------------------------------------------------------------
# the profile mirror
# ---------------------------------------------------------------------------


def test_the_profile_lives_where_their_ui_reads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""``~/.detector/preprocessing_profiles/<animal>.json``.

    A deliberate exception to cross-platform rule 15: writing our own config through
    ``platformdirs`` is right, but this file is a contract with another application,
    and putting it in ``%LOCALAPPDATA%`` would mean their UI never sees it.
    """
    monkeypatch.delenv("GEMS_PROFILES_DIR", raising=False)
    assert profiles_dir() == Path.home() / ".detector" / "preprocessing_profiles"
    assert profile_path("J").name == "J.json"


def test_an_animal_name_that_cannot_be_a_filename_is_rejected() -> None:
    for bad in ("", "J/L", "J:L", "a*b"):
        with pytest.raises(ValueError, match="filename component"):
            profile_path(bad)


def test_a_profile_round_trips(tmp_path: Path) -> None:
    original = new_cohort_map(animal="J")
    save_profile(original, tmp_path)
    loaded = load_profile("J", tmp_path)

    assert loaded is not None
    assert loaded.animal == original.animal
    assert loaded.units == original.units
    assert loaded.config == original.config
    assert loaded.channels == original.channels
    assert loaded.direction_valid


def test_the_old_cohort_profile_round_trips(tmp_path: Path) -> None:
    original = old_cohort_map(animal="F")
    save_profile(original, tmp_path)
    loaded = load_profile("F", tmp_path)
    assert loaded is not None
    assert loaded.config == "hw_tripole"
    assert loaded.channels == original.channels
    assert not loaded.direction_valid


@pytest.mark.parametrize("units", ["V", "mV", "uV"])
def test_the_declared_units_survive_the_profile(tmp_path: Path, units: str) -> None:
    """Declared once per animal, so no caller has to guess a factor of 10^6."""
    mapping = ChannelMap(animal="J", channels=new_cohort_map().channels, units=units)  # type: ignore[arg-type]
    save_profile(mapping, tmp_path)
    loaded = load_profile("J", tmp_path)
    assert loaded is not None
    assert loaded.units == units
    assert loaded.scale_uv == scale_to_uv(units)


def test_an_absent_rostral_end_round_trips_as_an_absent_key(tmp_path: Path) -> None:
    """CLAUDE.md: a missing scalar serialised to JSON is an absent key, not null."""
    save_profile(new_cohort_map(rostral_end=None), tmp_path)
    document = json.loads(profile_path("J", tmp_path).read_text(encoding="utf-8"))

    for channel in document["channel_assignment"]["channels"]:
        assert "rostral_end" not in channel
    assert "null" not in profile_path("J", tmp_path).read_text(encoding="utf-8")

    loaded = load_profile("J", tmp_path)
    assert loaded is not None
    assert all(c.rostral_end is None for c in loaded.channels)
    assert not loaded.direction_valid


def test_a_null_is_read_as_absent(tmp_path: Path) -> None:
    """Accept absent and null identically; write only absent."""
    save_profile(new_cohort_map(), tmp_path)
    path = profile_path("J", tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    for channel in document["channel_assignment"]["channels"]:
        channel["rostral_end"] = None
        channel["cuff_id"] = channel.get("cuff_id")
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    loaded = load_profile("J", tmp_path)
    assert loaded is not None
    assert all(c.rostral_end is None for c in loaded.channels)


def test_saving_preserves_the_keys_their_ui_owns(tmp_path: Path) -> None:
    """Their notch block and stream selections must survive our write."""
    path = profile_path("J", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "animal_id": "J",
            "channel_assignment": {
                "raw_stream": "Raww",
                "stim_stream": "Stim",
                "vibration_stream": "Vib",
                "stim_envelope_stream": "Env",
                "channels": [{"signal_index": 0, "tdt_index": 0, "role": "nerve", "label": "old"}],
            },
            "notch": {"frequencies_filtered": [60, 120, 180], "q_factor": 30},
            "schema_version": "1.0",
            "created_at": "2026-05-01T00:00:00Z",
            "last_seen_in_batch_at": "2026-09-01T00:00:00Z",
        }),
        encoding="utf-8",
        newline="\n",
    )

    save_profile(new_cohort_map(animal="J"), tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["notch"] == {"frequencies_filtered": [60, 120, 180], "q_factor": 30}
    assert document["channel_assignment"]["raw_stream"] == "Raww"
    assert document["channel_assignment"]["stim_envelope_stream"] == "Env"
    assert document["created_at"] == "2026-05-01T00:00:00Z"
    assert document["last_seen_in_batch_at"] == "2026-09-01T00:00:00Z"
    assert len(document["channel_assignment"]["channels"]) == 9


def test_the_profile_uses_their_role_vocabulary(tmp_path: Path) -> None:
    """Theirs is nerve/stomach/other; A.1's is nerve/stomach/aux."""
    channels = [
        nerve_channel(0, "L", 1),
        nerve_channel(1, "L", 2),
        nerve_channel(2, "L", 3),
        ChannelInfo(3, "ADC2", "aux", None, None, None, "independent"),
    ]
    save_profile(ChannelMap(animal="J", channels=channels), tmp_path)
    document = json.loads(profile_path("J", tmp_path).read_text(encoding="utf-8"))
    roles = [c["role"] for c in document["channel_assignment"]["channels"]]
    assert roles == ["nerve", "nerve", "nerve", "other"]

    loaded = load_profile("J", tmp_path)
    assert loaded is not None
    assert [c.role for c in loaded.channels] == ["nerve", "nerve", "nerve", "aux"]


def test_a_missing_profile_is_none_not_an_error(tmp_path: Path) -> None:
    assert load_profile("nobody", tmp_path) is None


def test_a_corrupt_profile_raises_rather_than_reading_as_absent(tmp_path: Path) -> None:
    """Refuse a corrupt profile instead of reading it as absent.

    Their ``Profile.load`` returns None on any exception, which makes a corrupt
    profile indistinguishable from an absent one. A corrupt channel map would
    silently change which channel is which.
    """
    path = profile_path("J", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="could not be read"):
        load_profile("J", tmp_path)


def test_a_profile_with_no_channels_raises(tmp_path: Path) -> None:
    path = profile_path("J", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"animal_id": "J"}), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="no channels"):
        load_profile("J", tmp_path)


def test_a_profile_with_an_unknown_role_raises(tmp_path: Path) -> None:
    path = profile_path("J", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "animal_id": "J",
            "channel_assignment": {
                "channels": [
                    {"signal_index": i, "role": "nerve", "label": f"c{i}", "contact_index": 1}
                    for i in range(4)
                ] + [{"signal_index": 4, "role": "telepathy", "label": "x"}],
            },
        }),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="unknown role"):
        load_profile("J", tmp_path)


def test_the_profile_is_written_atomically_and_as_utf8_lf(tmp_path: Path) -> None:
    """Their own writer uses the platform default encoding; this one does not."""
    mapping = ChannelMap(
        animal="J",
        channels=[ChannelInfo(0, "σ-channel", "nerve", "L", 1, 1, "independent")],  # noqa: RUF001
    )
    path = save_profile(mapping, tmp_path)

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8")
    assert not list(tmp_path.glob("*.tmp"))
    loaded = load_profile("J", tmp_path)
    assert loaded is not None
    assert loaded.channels[0].name == "σ-channel"  # noqa: RUF001


def test_extra_keys_can_be_attached_without_disturbing_the_schema(tmp_path: Path) -> None:
    save_profile(new_cohort_map(), tmp_path, extra={"gems_blanking_v2_version": "0.1.0"})
    document = json.loads(profile_path("J", tmp_path).read_text(encoding="utf-8"))
    assert document["gems_blanking_v2_version"] == "0.1.0"
    assert document["schema_version"] == "1.0"
