"""Scanning a folder: dedupe by content, block the unclear, never overwrite a human.

Every test builds its own tree under ``tmp_path`` and its own store, so nothing
here touches a real Drive mount or a real home directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gems_blanking_v2.io.channel_map import meta_path, save_geometry
from gems_blanking_v2.io.conditions import (
    Condition,
    ConditionRule,
    Rules,
    default_rules,
)
from gems_blanking_v2.io.scan import (
    Correction,
    ScanResult,
    apply_corrections,
    assert_corpus_eligible,
    scan,
    sort_for_review,
)
from gems_blanking_v2.io.store import GemsStore, sha256_file

from test_channel_map import new_cohort_map
from test_conditions import second_token_initial


@pytest.fixture
def rules() -> Rules:
    """Default rules with animal extraction that needs no private checkout."""
    base = default_rules()
    return Rules(
        vocabulary=base.vocabulary,
        rules=base.rules,
        estim_hz=base.estim_hz,
        mstim_hz=base.mstim_hz,
        timepoint_pattern=base.timepoint_pattern,
        strip_suffixes=base.strip_suffixes,
        source=base.source,
        animal_from=second_token_initial,
    )


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    """Return a fake gems_root with the real layout."""
    return GemsStore.initialise(tmp_path / "gems")


def write_recording(path: Path, content: bytes = b"samples") -> Path:
    """Create a stand-in recording file. Only its name and bytes matter here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# walking
# ---------------------------------------------------------------------------


def test_the_scan_is_recursive_and_picks_up_both_extensions(
    tmp_path: Path, rules: Rules
) -> None:
    root = tmp_path / "data"
    write_recording(root / "a" / "E1000_JEL_E1000_bl_1315.mat")
    write_recording(root / "a" / "b" / "M100_LOL_MS2_stim_rec_2031.h5", b"other")
    write_recording(root / "notes.txt", b"not a recording")

    results = scan(root, rules)
    assert {r.path.name for r in results} == {
        "E1000_JEL_E1000_bl_1315.mat",
        "M100_LOL_MS2_stim_rec_2031.h5",
    }


def test_labelling_outputs_are_skipped(tmp_path: Path, rules: Rules) -> None:
    """A ``_blankmotion.mat`` holds ``yOut``, not ``y``; their loader refuses it too."""
    root = tmp_path / "data"
    write_recording(root / "E1000_JEL_E1000_bl_1315.mat")
    write_recording(root / "E1000_JEL_E1000_bl_1315_blankmotion.mat", b"blanked")

    results = scan(root, rules)
    assert [r.path.name for r in results] == ["E1000_JEL_E1000_bl_1315.mat"]


def test_an_empty_tree_scans_to_nothing(tmp_path: Path, rules: Rules) -> None:
    (tmp_path / "data").mkdir()
    assert scan(tmp_path / "data", rules) == []


def test_the_scan_writes_nothing_into_the_store(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """Scanning is read-only until a human confirms."""
    root = tmp_path / "data"
    write_recording(root / "E1000_JEL_E1000_bl_1315.mat")
    before = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*"))

    scan(root, rules, store=store)

    after = sorted(p.relative_to(store.root).as_posix() for p in store.root.rglob("*"))
    assert after == before


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------


def test_a_matched_recording_carries_its_rule_and_animal(tmp_path: Path, rules: Rules) -> None:
    root = tmp_path / "data"
    write_recording(root / "gems_d_t01_es1_sr_204720.mat")

    (result,) = scan(root, rules)
    assert result.status == "matched"
    assert result.condition == Condition("stim_recovery", 10.0, None, "t01")
    assert result.matched_rule == "stim_rec_new"
    assert result.animal == "D"
    assert result.session == "gems_d_t01_es1_sr_204720"
    assert result.corpus_eligible


def test_an_old_cohort_row_resolves_onto_the_same_axes(
    tmp_path: Path, rules: Rules
) -> None:
    """``E1000`` is 1000 Hz electrical, so the row parses and is eligible."""
    root = tmp_path / "data"
    write_recording(root / "E1000_FRE_E1000_stim_rec_1406.mat")

    (result,) = scan(root, rules)
    assert result.status == "matched"
    assert result.condition.epoch == "stim_recovery"
    assert result.condition.estim_hz == 1000.0
    assert result.matched_rule == "stim_rec_old"
    assert result.animal == "F"
    assert result.unparsed_stim_tokens == ()
    assert result.corpus_eligible


def test_a_token_conflict_is_carried_and_written_to_meta_json(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """Recorded rather than dropped, in the scan row and in ``meta.json``."""
    root = tmp_path / "data"
    path = write_recording(root / "M100_JEL_MS2_bl_1945.mat")

    (result,) = scan(root, rules)
    assert result.token_conflict == ("M100", "MS2")
    assert result.condition.mstim_hz == 100.0
    assert result.corpus_eligible

    (written,) = apply_corrections(
        [result],
        [Correction(path=path, condition=result.condition)],
        "andrea",
        store,
        rules,
    )
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["token_conflict"] == ["M100", "MS2"]
    assert document["condition"]["mstim_hz"] == 100.0


def test_an_unrecognised_name_is_unknown_and_blocks(tmp_path: Path, rules: Rules) -> None:
    """The regression guard, at scan level: never silently a control."""
    root = tmp_path / "data"
    write_recording(root / "X_JEL_new_protocol_2026.mat")

    (result,) = scan(root, rules)
    assert result.status == "unknown"
    assert result.condition.epoch == "unknown"
    assert result.condition.epoch != "baseline"
    assert not result.corpus_eligible
    assert result.needs_a_human


def test_an_ambiguous_name_reports_its_candidates(tmp_path: Path) -> None:
    rules = Rules(
        vocabulary=("baseline", "stim", "unknown"),
        rules=(
            ConditionRule(id="a", pattern="_x", epoch="baseline", priority=10),
            ConditionRule(id="b", pattern="_x", epoch="stim", priority=10),
        ),
        animal_from=second_token_initial,
    )
    root = tmp_path / "data"
    write_recording(root / "E1_JEL_x_01.mat")

    (result,) = scan(root, rules)
    assert result.status == "ambiguous"
    assert result.candidates == ("baseline", "stim")
    assert not result.corpus_eligible


# ---------------------------------------------------------------------------
# deduplication and known recordings
# ---------------------------------------------------------------------------


def test_the_same_content_at_two_paths_is_one_duplicate_row_naming_both(
    tmp_path: Path, rules: Rules
) -> None:
    """A folder reorganisation or a Drive conflict copy, deduped by content."""
    root = tmp_path / "data"
    first = write_recording(root / "a" / "E1000_JEL_E1000_bl_1315.mat", b"identical")
    second = write_recording(root / "b" / "E1000_JEL_E1000_bl_1315 (1).mat", b"identical")

    results = scan(root, rules)
    duplicates = [r for r in results if r.status == "duplicate"]
    assert len(duplicates) == 1
    assert duplicates[0].path in (first, second)
    assert duplicates[0].duplicate_of in (first, second)
    assert duplicates[0].duplicate_of != duplicates[0].path
    assert not duplicates[0].corpus_eligible
    assert duplicates[0].content_hash is not None


def test_different_content_at_two_paths_is_not_a_duplicate(
    tmp_path: Path, rules: Rules
) -> None:
    root = tmp_path / "data"
    write_recording(root / "a" / "E1000_JEL_E1000_bl_1315.mat", b"one")
    write_recording(root / "b" / "E1000_JEL_E1000_bl_1315.mat", b"two")

    assert [r.status for r in scan(root, rules)] == ["matched", "matched"]


def test_a_file_unique_by_size_is_not_hashed(tmp_path: Path, rules: Rules) -> None:
    """Hashing every file over a streamed mount is too slow to be usable.

    Identical content implies identical size, so a size-unique file cannot be a
    duplicate and ``content_hash`` is left absent - meaning "not needed", not
    "unknown".
    """
    root = tmp_path / "data"
    write_recording(root / "E1000_JEL_E1000_bl_1315.mat", b"unique-length-content")
    write_recording(root / "M100_LOL_MS2_stim_rec_2031.mat", b"different length here!!")

    results = scan(root, rules)
    assert all(r.content_hash is None for r in results)


def test_a_known_recording_is_reported_as_known_and_rehashed(
    tmp_path: Path, rules: Rules
) -> None:
    """Known rows are skipped for ingest but their checksum is re-verified."""
    root = tmp_path / "data"
    path = write_recording(root / "E1000_JEL_E1000_bl_1315.mat", b"known-bytes")
    digest = sha256_file(path)

    (result,) = scan(root, rules, known={digest: "E1000_JEL_E1000_bl_1315"})
    assert result.status == "known"
    assert result.content_hash == digest


def test_progress_is_reported_while_hashing(tmp_path: Path, rules: Rules) -> None:
    root = tmp_path / "data"
    write_recording(root / "a" / "E1_JEL_x_bl_1.mat", b"same")
    write_recording(root / "b" / "E1_JEL_x_bl_2.mat", b"same")

    seen: list[tuple[int, int]] = []
    scan(root, rules, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]


# ---------------------------------------------------------------------------
# review ordering
# ---------------------------------------------------------------------------


def test_rows_needing_attention_sort_to_the_top() -> None:
    """In a listing of hundreds they must not be buried among the correct ones."""
    rows = [
        ScanResult(path=Path("d.mat"), status="known"),
        ScanResult(path=Path("c.mat"), status="matched"),
        ScanResult(path=Path("b.mat"), status="ambiguous"),
        ScanResult(path=Path("a.mat"), status="unknown"),
        ScanResult(path=Path("e.mat"), status="duplicate"),
    ]
    assert [r.status for r in sort_for_review(rows)] == [
        "unknown",
        "ambiguous",
        "duplicate",
        "matched",
        "known",
    ]


def test_the_scan_returns_rows_already_sorted(tmp_path: Path, rules: Rules) -> None:
    root = tmp_path / "data"
    write_recording(root / "E1000_JEL_E1000_bl_1315.mat", b"aaa")
    write_recording(root / "Z_JEL_mystery_2026.mat", b"bbbb")

    results = scan(root, rules)
    assert results[0].status == "unknown"


# ---------------------------------------------------------------------------
# corpus eligibility
# ---------------------------------------------------------------------------


def test_an_unknown_recording_cannot_enter_a_corpus(tmp_path: Path, rules: Rules) -> None:
    root = tmp_path / "data"
    write_recording(root / "Z_JEL_mystery_2026.mat")

    results = scan(root, rules)
    with pytest.raises(ValueError, match="cannot enter a corpus") as exc:
        assert_corpus_eligible(results)
    assert "Z_JEL_mystery_2026.mat" in str(exc.value)
    assert "must not become a control by default" in str(exc.value)


def test_a_fully_resolved_scan_passes_the_corpus_check(tmp_path: Path, rules: Rules) -> None:
    root = tmp_path / "data"
    write_recording(root / "gems_d_t01_ms1_bl_164012.mat", b"aaa")
    write_recording(root / "gems_d_t01_es1_sr_204720.mat", b"bbbb")

    assert_corpus_eligible(scan(root, rules))


# ---------------------------------------------------------------------------
# corrections
# ---------------------------------------------------------------------------


def test_a_correction_records_who_and_when(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    root = tmp_path / "data"
    path = write_recording(root / "Z_JEL_mystery_2026.mat")
    results = scan(root, rules)

    (written,) = apply_corrections(
        results,
        [Correction(path=path, condition=Condition("baseline", None, None, "t01"))],
        "andrea",
        store,
        rules,
    )
    document = json.loads(written.read_text(encoding="utf-8"))

    assert document["condition"] == {"epoch": "baseline", "timepoint": "t01"}
    assert document["condition_source"] == "human"
    assert document["corrections"][0]["who"] == "andrea"
    assert document["corrections"][0]["when"].endswith("Z")
    assert document["corrections"][0]["condition"]["epoch"] == "baseline"


def test_a_human_condition_survives_a_rescan(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """However confidently the rules would classify it, the human's answer stands."""
    root = tmp_path / "data"
    path = write_recording(root / "E1000_JEL_E1000_bl_1315.mat")

    first = scan(root, rules, store=store)
    assert first[0].condition.epoch == "baseline"

    human = Condition("stim_recovery", 100.0, None, "t01")
    apply_corrections(first, [Correction(path=path, condition=human)], "andrea", store, rules)

    (rescanned,) = scan(root, rules, store=store)
    assert rescanned.condition == human
    assert rescanned.condition_source == "human"
    assert rescanned.status == "matched"
    assert rescanned.corpus_eligible


def test_a_rule_derived_condition_is_re_derived_on_rescan(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """Only a human's answer is sticky, so improving a rule improves old rows."""
    root = tmp_path / "data"
    write_recording(root / "E1000_JEL_E1000_bl_1315.mat")
    path = meta_path(store, "J", "E1000_JEL_E1000_bl_1315")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"condition": {"epoch": "stim_recovery"}, "condition_source": "rule"}),
        encoding="utf-8",
        newline="\n",
    )

    (result,) = scan(root, rules, store=store)
    assert result.condition.epoch == "baseline"
    assert result.condition_source == "rule"


def test_a_correction_outside_the_vocabulary_is_refused_at_write_time(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    root = tmp_path / "data"
    path = write_recording(root / "Z_JEL_mystery_2026.mat")
    results = scan(root, rules)

    with pytest.raises(ValueError, match="closed vocabulary"):
        apply_corrections(
            results,
            [Correction(path=path, condition=Condition("stimulation"))],
            "andrea",
            store,
            rules,
        )
    assert not meta_path(store, "J", "Z_JEL_mystery_2026").exists()


def test_a_correction_contradicting_a_matched_rule_flags_that_rule(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """How a bad rule gets caught rather than propagating."""
    root = tmp_path / "data"
    path = write_recording(root / "E1000_JEL_E1000_bl_1315.mat")
    results = scan(root, rules)
    assert results[0].matched_rule == "baseline_bl"

    (written,) = apply_corrections(
        results,
        [Correction(path=path, condition=Condition("stim_recovery"))],
        "andrea",
        store,
        rules,
    )
    entry = json.loads(written.read_text(encoding="utf-8"))["corrections"][0]
    assert entry["contradicted_rule"] == "baseline_bl"


def test_a_correction_agreeing_with_the_rule_flags_nothing(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    root = tmp_path / "data"
    path = write_recording(root / "E1000_JEL_E1000_bl_1315.mat")
    results = scan(root, rules)

    (written,) = apply_corrections(
        results,
        [Correction(path=path, condition=results[0].condition)],
        "andrea",
        store,
        rules,
    )
    entry = json.loads(written.read_text(encoding="utf-8"))["corrections"][0]
    assert "contradicted_rule" not in entry


def test_a_correction_preserves_the_geometry_block(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """Task 03 writes geometry into the same document; a correction must not drop it."""
    root = tmp_path / "data"
    path = write_recording(root / "E1000_JEL_E1000_bl_1315.mat")
    save_geometry(
        new_cohort_map("J"), "E1000_JEL_E1000_bl_1315", store, mirror_to_profile=False
    )
    results = scan(root, rules)

    (written,) = apply_corrections(
        results,
        [Correction(path=path, condition=Condition("stim_recovery"))],
        "andrea",
        store,
        rules,
    )
    document = json.loads(written.read_text(encoding="utf-8"))

    assert len(document["channels"]) == 9
    assert document["units"] == "uV"
    assert document["condition"] == {"epoch": "stim_recovery"}


def test_corrections_accumulate_rather_than_replace(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """The correction history is provenance: who changed what, and when."""
    root = tmp_path / "data"
    path = write_recording(root / "Z_JEL_mystery_2026.mat")
    results = scan(root, rules)

    apply_corrections(
        results,
        [Correction(path=path, condition=Condition("baseline"))],
        "andrea",
        store,
        rules,
    )
    (written,) = apply_corrections(
        results,
        [
            Correction(
                path=path,
                condition=Condition("stim_recovery"),
                note="misread the log",
            )
        ],
        "sam",
        store,
        rules,
    )
    document = json.loads(written.read_text(encoding="utf-8"))

    assert len(document["corrections"]) == 2
    assert [c["who"] for c in document["corrections"]] == ["andrea", "sam"]
    assert document["corrections"][1]["note"] == "misread the log"
    assert document["condition"] == {"epoch": "stim_recovery"}


def test_a_correction_can_set_the_animal_when_the_name_carries_none(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    root = tmp_path / "data"
    path = write_recording(root / "nounderscore.mat")
    results = scan(root, rules)
    assert results[0].animal is None

    (written,) = apply_corrections(
        results,
        [Correction(path=path, animal="J", condition=Condition("baseline"))],
        "andrea",
        store,
        rules,
    )
    assert store.relpath(written) == "data/J/nounderscore/meta.json"
    assert json.loads(written.read_text(encoding="utf-8"))["animal"] == "J"


def test_a_correction_without_an_animal_anywhere_is_refused(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """The directory the correction is written to is keyed on the animal."""
    root = tmp_path / "data"
    path = write_recording(root / "nounderscore.mat")
    results = scan(root, rules)

    with pytest.raises(ValueError, match="no animal for this recording"):
        apply_corrections(
            results,
            [Correction(path=path, condition=Condition("baseline"))],
            "andrea",
            store,
            rules,
        )


def test_a_correction_for_a_path_not_in_the_scan_is_refused(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    with pytest.raises(ValueError, match="not in this scan"):
        apply_corrections(
            [],
            [Correction(path=tmp_path / "ghost.mat", condition=Condition("baseline"))],
            "u",
            store,
            rules,
        )


def test_a_correction_will_not_overwrite_an_unreadable_meta_json(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    """It holds geometry and history; clobbering it would destroy both."""
    root = tmp_path / "data"
    path = write_recording(root / "E1000_JEL_E1000_bl_1315.mat")
    meta = meta_path(store, "J", "E1000_JEL_E1000_bl_1315")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text("{not json", encoding="utf-8", newline="\n")

    results = scan(root, rules)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        apply_corrections(
            results,
            [Correction(path=path, condition=Condition("stim_recovery"))],
            "andrea",
            store,
            rules,
        )


def test_the_written_meta_json_is_utf8_with_lf(
    tmp_path: Path, rules: Rules, store: GemsStore
) -> None:
    root = tmp_path / "data"
    path = write_recording(root / "Z_JEL_mystery_2026.mat")
    results = scan(root, rules)

    (written,) = apply_corrections(
        results,
        [Correction(path=path, condition=Condition("baseline"), note="σ note")],  # noqa: RUF001
        "andrea",
        store,
        rules,
    )
    raw = written.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8")
    assert not list(written.parent.glob("*.tmp"))
