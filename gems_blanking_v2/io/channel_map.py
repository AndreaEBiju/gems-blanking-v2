"""The channel table: geometry, cohort configuration, and the per-animal profile.

``GEMSBlanking``'s loader returns samples and a rate and **no channel table at
all**, so the geometry this project needs - which cuff a contact belongs to, where
it sits along the cuff, which end is rostral - has to come from somewhere else.
That somewhere is **our own store**: ``data/<animal>/<session>/meta.json`` is the
single source of truth. The copy written into ``detector-pyqt``'s per-animal
preprocessing profile is a **mirror** for their UI's benefit and may be
regenerated from ours at any time - their ``from_review_session`` rebuilds the
channel list from a dialog that knows nothing about geometry, so treating their
file as authoritative would let a UI session destroy ``rostral_end``.

Two things are never inferred:

``rostral_end``
    It cannot be reconstructed once the animal is gone. Absent means absent: the
    consequence is unsigned velocities and ``direction_valid=False``, not a guess
    from channel order or from a name.
``units``
    Nothing in the loader converts them and the files are not self-describing;
    ``processing_new/convertUnits.m`` treats the unit as declared. Guessing wrong
    is a factor of 10^6, so it is declared and stored alongside the geometry.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.types import ChannelInfo

__all__ = [
    "CUFF_CONTACTS",
    "GEOMETRY_KEYS",
    "HW_TRIPOLE_CHANNEL_COUNT",
    "META_NAME",
    "ChannelMap",
    "Units",
    "infer_config",
    "load_geometry",
    "load_profile",
    "meta_path",
    "profile_path",
    "profiles_dir",
    "resolve_channel_map",
    "save_geometry",
    "save_profile",
    "scale_to_uv",
]

log = logging.getLogger(__name__)

Units = Literal["V", "mV", "uV"]
"""The unit a recording's samples are in. Declared, never inferred."""

_UV_PER: Final[dict[str, float]] = {"V": 1e6, "mV": 1e3, "uV": 1.0}
"""Conversion to microvolts, matching ``processing_new/convertUnits.m``."""

CUFF_CONTACTS: Final = 3
"""Contacts per cuff in the v9 design: V1, V2, V3."""

HW_TRIPOLE_CHANNEL_COUNT: Final = 5
"""Old-cohort channel count: one pre-formed tripole per nerve plus stomach.

The individual contacts are unrecoverable in that cohort - the tripole was formed
in hardware - which is why ``config`` matters to every later step.
"""

# The PyQt UI's role vocabulary, which is not ours.
_ROLE_FROM_PROFILE: Final[dict[str, str]] = {
    # Their vocabulary, plus ours: meta.json stores A.1 roles, the profile stores
    # theirs, and one reader handles both.
    "nerve": "nerve",
    "stomach": "stomach",
    "other": "aux",
    "aux": "aux",
}
_ROLE_TO_PROFILE: Final[dict[str, str]] = {"nerve": "nerve", "stomach": "stomach", "aux": "other"}

PROFILE_SCHEMA_VERSION: Final = "1.0"
"""``GEMSBlanking``'s ``profiles.SCHEMA_VERSION``. Not bumped by us: the geometry
lives inside each channel dict, which their loader carries through untouched."""

def scale_to_uv(units: Units) -> float:
    """Return the factor converting ``units`` to microvolts."""
    try:
        return _UV_PER[units]
    except KeyError as exc:
        msg = f"unknown units {units!r}; expected one of {sorted(_UV_PER)}"
        raise ValueError(msg) from exc


def _validated_units(value: object, where: str) -> Units:
    """Return ``value`` as :data:`Units`, or raise naming ``where`` it came from.

    Written as an explicit match rather than a cast so the Literal is narrowed by
    the type checker rather than asserted past it.
    """
    for candidate in ("V", "mV", "uV"):
        if value == candidate:
            return candidate
    msg = f"{where} declares unknown units {value!r}; expected one of V, mV, uV"
    raise ValueError(msg)


def infer_config(
    n_channels: int, has_contact_index: bool
) -> Literal["hw_tripole", "independent"]:
    """Infer the cohort configuration from the channel count and the geometry.

    A recording whose channels carry a ``contact_index`` has individually recorded
    contacts, so it is ``independent`` whatever its channel count. Without one, a
    five-channel file is the old hardware-shorted cohort.

    Raises
    ------
    ValueError
        When neither holds - an unfamiliar layout with no geometry. Refusing is
        the point: a wrong ``config`` silently changes what every later step reads.
    """
    if has_contact_index:
        return "independent"
    if n_channels == HW_TRIPOLE_CHANNEL_COUNT:
        return "hw_tripole"
    msg = (
        f"cannot infer config: {n_channels} channels and no contact_index. "
        f"{HW_TRIPOLE_CHANNEL_COUNT} channels without geometry is the hw_tripole "
        "cohort; anything else needs an explicit channel map."
    )
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChannelMap:
    """The channel table for one animal, with the geometry later steps need.

    Attributes
    ----------
    animal
        Animal identifier; the profile mirror is keyed on it.
    channels
        One :class:`~gems_blanking_v2.types.ChannelInfo` per column of the raw
        matrix, in column order.
    units
        The unit the raw samples are in, declared by whoever built this map.
    """

    animal: str
    channels: list[ChannelInfo]
    units: Units = "uV"

    def __post_init__(self) -> None:
        """Check the table is internally consistent before anything relies on it."""
        indices = [c.index for c in self.channels]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            msg = f"channel indices must be unique and ascending, got {indices}"
            raise ValueError(msg)
        names = [c.name for c in self.channels]
        if len({n.casefold() for n in names}) != len(names):
            # Cross-platform rule 8's reasoning applied to channel names: a pair
            # differing only by case is a collision waiting to happen.
            msg = f"channel names must be unique case-insensitively, got {names}"
            raise ValueError(msg)
        configs = {c.config for c in self.channels}
        if len(configs) > 1:
            msg = f"a recording has one config, got {sorted(configs)}"
            raise ValueError(msg)
        scale_to_uv(self.units)

    @property
    def config(self) -> Literal["hw_tripole", "independent"]:
        """The cohort configuration. Uniform across the table by construction."""
        return self.channels[0].config

    @property
    def n_channels(self) -> int:
        """Number of channels in the table."""
        return len(self.channels)

    @property
    def scale_uv(self) -> float:
        """Factor converting this recording's samples to microvolts."""
        return scale_to_uv(self.units)

    @property
    def cuffs(self) -> list[str]:
        """Cuff identifiers present, sorted."""
        return sorted({c.cuff_id for c in self.channels if c.cuff_id is not None})

    def contacts(self, cuff_id: str) -> list[ChannelInfo]:
        """Return one cuff's contacts, ordered by ``contact_index``."""
        got = [c for c in self.channels if c.cuff_id == cuff_id and c.contact_index]
        return sorted(got, key=lambda c: c.contact_index or 0)

    @property
    def direction_valid(self) -> bool:
        """Whether conduction *direction* can be signed.

        False when any cuff lacks ``rostral_end``. Velocity magnitudes remain
        usable; their sign does not, and every velocity record from this recording
        must carry this flag.
        """
        cuffs = self.cuffs
        if not cuffs:
            return False
        return all(
            any(c.rostral_end is not None for c in self.contacts(cuff)) for cuff in cuffs
        )

    def warn_if_direction_unknown(self, recording_id: str) -> None:
        """Log one WARNING per recording when direction cannot be signed."""
        if not self.direction_valid:
            log.warning(
                "%s: rostral_end is unknown for animal %s, so velocities are "
                "unsigned and direction_valid=False on every velocity record. It "
                "cannot be reconstructed after the fact and must not be guessed.",
                recording_id,
                self.animal,
            )


# ---------------------------------------------------------------------------
# meta.json - the single source of truth
# ---------------------------------------------------------------------------

META_NAME: Final = "meta.json"
"""Per-session metadata file inside our own store, ``data/<animal>/<session>/``.

**This is the single source of truth for geometry.** The copy in
``detector-pyqt``'s preprocessing profile is a mirror for their UI's benefit and
may be regenerated from here at any time. Their ``from_review_session`` rebuilds
the channel list from a dialog that knows nothing about ``cuff_id``,
``contact_index`` or ``rostral_end``, so a user re-running channel assignment in
their UI would otherwise destroy ``rostral_end`` - which cannot be recovered once
the animal is gone.

The file has **more than one writer**: this module writes the geometry, and task
03A's ``apply_corrections`` writes the condition and its ``who``/``when``
provenance. Every write is therefore read-modify-write, preserving keys it does
not recognise, for the same reason the profile write is.
"""

GEOMETRY_KEYS: Final = ("channels", "units", "geometry_updated_at")
"""The keys this module owns inside ``meta.json``. Everything else is left alone."""


def meta_path(store: GemsStore, animal: str, session: str) -> Path:
    """Return the path of one session's ``meta.json`` inside the store."""
    return store.session_dir(animal, session) / META_NAME


def save_geometry(
    channel_map: ChannelMap,
    session: str,
    store: GemsStore,
    *,
    mirror_to_profile: bool = True,
    profiles_root: Path | None = None,
) -> Path:
    r"""Write the geometry to ``meta.json`` and, by default, mirror it to the profile.

    The write preserves every key this module does not own, so task 03A's condition
    block survives a geometry update and vice versa. Nothing written here contains a
    path, so cross-platform rule 2 is satisfied by construction rather than by care.

    Parameters
    ----------
    channel_map
        The geometry to record. Its ``animal`` names the directory.
    session
        Session identifier; with ``animal`` it selects the directory.
    store
        The Drive store, from task 00A.
    mirror_to_profile
        Whether to refresh ``detector-pyqt``'s profile from this. The mirror is
        derived, never authoritative.
    profiles_root
        Profile directory, for tests.

    Returns
    -------
    pathlib.Path
        The ``meta.json`` written.
    """
    path = meta_path(store, channel_map.animal, session)
    document: dict[str, Any] = {}
    if path.is_file():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            msg = f"{path} exists but could not be read; refusing to overwrite it: {exc}"
            raise ValueError(msg) from exc

    document["animal"] = channel_map.animal
    document["session"] = session
    document["units"] = channel_map.units
    document["channels"] = [_channel_to_meta_dict(c) for c in channel_map.channels]
    document["geometry_updated_at"] = _utc_now()

    body = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(path)

    if mirror_to_profile:
        save_profile(channel_map, profiles_root, mirror_of=store.relpath(path))
    return path


def load_geometry(animal: str, session: str, store: GemsStore) -> ChannelMap | None:
    """Return the session's geometry from ``meta.json``, or None if absent.

    Raises
    ------
    ValueError
        If the file exists but cannot be read or has no channels. A corrupt
        geometry would silently change which channel is which.
    """
    path = meta_path(store, animal, session)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"{path} exists but could not be read: {exc}"
        raise ValueError(msg) from exc

    raw_channels = document.get("channels") or []
    if not raw_channels:
        msg = f"{path} has no channels; geometry has not been recorded for this session"
        raise ValueError(msg)

    has_contact = any(ch.get("contact_index") is not None for ch in raw_channels)
    fallback = infer_config(len(raw_channels), has_contact)
    channels = sorted(
        (_channel_from_profile_dict(ch, fallback) for ch in raw_channels),
        key=lambda c: c.index,
    )
    units = _validated_units(document.get("units") or "uV", str(path))
    return ChannelMap(animal=str(document.get("animal") or animal), channels=channels, units=units)


def resolve_channel_map(
    animal: str,
    *,
    explicit: ChannelMap | None = None,
    session: str | None = None,
    store: GemsStore | None = None,
    profiles_root: Path | None = None,
) -> ChannelMap | None:
    """Return the geometry to use, in precedence order, or None if there is none.

    Explicit argument, then ``meta.json`` (**authoritative**), then the profile
    mirror. Falling back to the mirror logs a WARNING: it is a copy, another tool
    can rewrite it, and it is keyed per animal rather than per session, so it may
    describe a different session's geometry.
    """
    if explicit is not None:
        return explicit
    if store is not None and session is not None:
        from_meta = load_geometry(animal, session, store)
        if from_meta is not None:
            return from_meta
    mirror = load_profile(animal, profiles_root)
    if mirror is not None:
        log.warning(
            "animal %s: using the preprocessing-profile mirror because no "
            "%s holds geometry for session %s. The profile is a per-animal copy "
            "that another tool may rewrite - record geometry in the store to make "
            "it authoritative.",
            animal,
            META_NAME,
            session,
        )
    return mirror


def _channel_to_meta_dict(channel: ChannelInfo) -> dict[str, Any]:
    """Render one channel for ``meta.json``. A missing scalar is an absent key."""
    out: dict[str, Any] = {
        "signal_index": channel.index,
        "label": channel.name,
        "role": channel.role,
        "config": channel.config,
    }
    if channel.cuff_id is not None:
        out["cuff_id"] = channel.cuff_id
    if channel.contact_index is not None:
        out["contact_index"] = channel.contact_index
    if channel.rostral_end is not None:
        out["rostral_end"] = channel.rostral_end
    return out


def _utc_now() -> str:
    """Return a UTC timestamp, the same shape the other stores use."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# the per-animal profile - a mirror of the above
# ---------------------------------------------------------------------------


def profiles_dir(root: Path | None = None) -> Path:
    r"""Return the directory holding per-animal profiles.

    ``~/.detector/preprocessing_profiles``, which is where
    ``detector-pyqt/ui/widgets/channel_assignment.py`` and
    ``GEMSBlanking/detector/preprocessing/profiles.py`` already read and write.

    This is a deliberate exception to cross-platform rule 15, which says per-user
    config is written through ``platformdirs``. That rule is about *our* config; this
    file is a shared contract with another application, and writing it to
    ``%LOCALAPPDATA%`` instead would mean their UI never sees it. ``root`` exists so
    tests never touch a real home directory.
    """
    if root is not None:
        return Path(root)
    if (env := os.environ.get("GEMS_PROFILES_DIR")) is not None:
        return Path(env)
    return Path.home() / ".detector" / "preprocessing_profiles"


def profile_path(animal: str, root: Path | None = None) -> Path:
    """Return the profile path for ``animal``."""
    if not animal or any(c in animal for c in '<>:"/\\|?*'):
        msg = f"animal {animal!r} cannot be a filename component"
        raise ValueError(msg)
    return profiles_dir(root) / f"{animal}.json"


def _channel_to_profile_dict(channel: ChannelInfo) -> dict[str, Any]:
    """Render one channel in the profile's schema, geometry included.

    The four keys their UI needs keep their names and meanings; ours sit alongside.
    A missing scalar is an **absent key**, never ``null`` - the JSON half of the
    missing-value convention in ``CLAUDE.md``.
    """
    out: dict[str, Any] = {
        "signal_index": channel.index,
        "tdt_index": channel.index,
        "role": _ROLE_TO_PROFILE[channel.role],
        "label": channel.name,
        "config": channel.config,
    }
    if channel.cuff_id is not None:
        out["cuff_id"] = channel.cuff_id
    if channel.contact_index is not None:
        out["contact_index"] = channel.contact_index
    if channel.rostral_end is not None:
        out["rostral_end"] = channel.rostral_end
    return out


def _channel_from_profile_dict(
    raw: dict[str, Any], fallback_config: Literal["hw_tripole", "independent"]
) -> ChannelInfo:
    """Read one channel dict, accepting absent and ``null`` identically."""

    def opt_int(key: str) -> int | None:
        value = raw.get(key)
        return None if value is None else int(value)

    index = raw.get("signal_index", raw.get("tdt_index"))
    if index is None:
        msg = f"profile channel is missing signal_index: {raw!r}"
        raise ValueError(msg)
    role_raw = raw.get("role") or "other"
    if role_raw not in _ROLE_FROM_PROFILE:
        msg = f"profile channel has unknown role {role_raw!r}: {raw!r}"
        raise ValueError(msg)
    config = raw.get("config") or fallback_config
    if config not in ("hw_tripole", "independent"):
        msg = f"profile channel has unknown config {config!r}"
        raise ValueError(msg)
    return ChannelInfo(
        index=int(index),
        name=str(raw.get("label") or f"Ch{int(index) + 1}"),
        role=_ROLE_FROM_PROFILE[role_raw],  # type: ignore[arg-type]
        cuff_id=None if raw.get("cuff_id") is None else str(raw["cuff_id"]),
        contact_index=opt_int("contact_index"),
        rostral_end=opt_int("rostral_end"),
        config=config,
    )


def save_profile(
    channel_map: ChannelMap,
    root: Path | None = None,
    *,
    extra: dict[str, Any] | None = None,
    mirror_of: str | None = None,
) -> Path:
    r"""Write ``channel_map`` into the animal's profile, preserving everything else.

    An existing profile is **extended**, not replaced: its ``notch`` block, stream
    selections, timestamps and any key this project does not know about survive.
    The write is atomic (temp file in the same directory, then replace) and the text
    is UTF-8 with ``\\n`` endings, because their own writer uses the platform default
    and this file is read on both platforms.

    Returns
    -------
    pathlib.Path
        The profile written.
    """
    path = profile_path(channel_map.animal, root)
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))

    assignment = dict(existing.get("channel_assignment") or {})
    assignment["channels"] = [_channel_to_profile_dict(c) for c in channel_map.channels]
    # `units` lives inside channel_assignment, not at the top level, because their
    # Profile.load reads known fields one by one and Profile.save writes
    # `asdict(self)` - so a top-level key this project added would be silently
    # dropped the next time their UI saved. channel_assignment is carried through
    # opaquely, so anything inside it survives their round trip.
    assignment["units"] = channel_map.units
    if mirror_of is not None:
        # Marked as derived so nobody edits it expecting the edit to survive: the
        # store's meta.json is authoritative and this file is regenerated from it.
        assignment["mirror_of"] = mirror_of
        assignment["mirror_note"] = (
            "Derived from the gems-blanking-v2 store; regenerate rather than edit. "
            "Geometry edits made here are lost on the next sync."
        )

    document = dict(existing)
    document["animal_id"] = channel_map.animal
    document["channel_assignment"] = assignment
    document.setdefault("schema_version", PROFILE_SCHEMA_VERSION)
    if extra:
        document.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    body = json.dumps(document, indent=2, sort_keys=True) + "\n"
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


def load_profile(animal: str, root: Path | None = None) -> ChannelMap | None:
    """Return the animal's channel map, or None if no profile exists.

    Raises
    ------
    ValueError
        If a profile exists but cannot be read. ``GEMSBlanking``'s own
        ``Profile.load`` returns ``None`` on any exception, which makes a corrupt
        profile indistinguishable from an absent one; a corrupt channel map would
        silently change which channel is which, so it raises here instead.
    """
    path = profile_path(animal, root)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"profile {path} exists but could not be read: {exc}"
        raise ValueError(msg) from exc

    raw_channels = (document.get("channel_assignment") or {}).get("channels") or []
    if not raw_channels:
        msg = f"profile {path} has no channels"
        raise ValueError(msg)

    has_contact = any(ch.get("contact_index") is not None for ch in raw_channels)
    fallback = infer_config(len(raw_channels), has_contact)
    channels = [_channel_from_profile_dict(ch, fallback) for ch in raw_channels]
    channels.sort(key=lambda c: c.index)

    assignment = document.get("channel_assignment") or {}
    # Top level is read too, for any profile written before `units` moved inside.
    declared = assignment.get("units") or document.get("units") or "uV"
    units = _validated_units(declared, f"profile {path}")
    return ChannelMap(
        animal=str(document.get("animal_id") or animal), channels=channels, units=units
    )
