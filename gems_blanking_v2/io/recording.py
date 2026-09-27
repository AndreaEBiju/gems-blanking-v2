"""Load a recording through ``GEMSBlanking``'s loader and adapt it to A.1.

Their ``load_recording`` handles the formats - ``.mat``, chunked HDF5 from the
splitter, flat HDF5 - and refuses a ``_blankmotion.mat`` save-output with a clear
message. All of that is reused rather than reimplemented. What this module adds is
the conversion their type does not carry:

* ``y`` is **float32** in whatever unit the file holds; A.1 wants float64
  microvolts, so the unit is applied from the channel map, where it was declared.
* their ``Recording`` has **no channel table**; A.1 requires one, so it comes from
  ``data/<animal>/<session>/meta.json`` in our store - the single source of truth -
  or from an explicit map, and only as a last resort from their profile mirror.
* ``recording_id`` is a path stem; A.1 wants ``animal`` and ``session``. Deriving
  an animal from a filename is condition inference, which is task 03A's job, so
  ``animal`` is required here and nothing is parsed out of the name.
* ``stim_end_idx`` and ``existing_bad_intervals`` are **1-based inclusive**, the
  MATLAB convention. They are converted to seconds and returned beside the
  recording, because A.1 has nowhere to put them and widening a shared contract to
  carry one task's needs is how contracts rot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.io.chanlabels import (
    assert_labels_match,
    assert_plausible_units,
    read_channel_labels,
)
from gems_blanking_v2.io.channel_map import (
    ChannelMap,
    Units,
    meta_path,
    resolve_channel_map,
    with_rostral_end,
)
from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.io.quality import read_exclusion
from gems_blanking_v2.io.stim_split import (
    PROTOCOL_FILENAME,
    ProtocolNotCoveredError,
    ProtocolSpec,
    load_protocol_book,
)
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.io.tdt_block import session_key
from gems_blanking_v2.types import Recording

__all__ = ["LoadedRecording", "load_recording", "spans_from_matlab_intervals"]

F64 = npt.NDArray[np.float64]

_INTERVAL_COLUMNS = 2
"""An interval array is (k, 2): start and stop."""


def spans_from_matlab_intervals(
    intervals: npt.NDArray[np.int64] | None, fs: float
) -> list[tuple[float, float]]:
    """Convert 1-based inclusive sample intervals to 0-based half-open seconds.

    ``existing_bad_intervals`` and ``stim_end_idx`` come from MATLAB, where a range
    is ``[a, b]`` counting from 1. Here a span is ``[start, stop)`` counting from 0,
    so ``[a, b]`` becomes ``[(a-1)/fs, b/fs)`` - the off-by-one that would otherwise
    shift every mask by one sample and be invisible in a plot.
    """
    if intervals is None or len(intervals) == 0:
        return []
    array = np.atleast_2d(np.asarray(intervals))
    if array.shape[1] != _INTERVAL_COLUMNS:
        msg = f"intervals must be (k, 2), got {array.shape}"
        raise ValueError(msg)
    return [((int(a) - 1) / fs, int(b) / fs) for a, b in array]


def _condition_provenance(
    store: GemsStore | None, animal: str, session: str
) -> dict[str, Any]:
    """Return the condition record and any token conflict, for provenance.

    Task 03A's ruling: a row resolved by the explicit-Hz precedence rule carries
    ``token_conflict`` in ``meta.json`` **and in provenance**. Provenance is the
    only place a later reader of a model sees it, so it is copied through here
    rather than left in the store for someone to go and look up.
    """
    if store is None:
        return {}
    path = meta_path(store, animal, session)
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    out: dict[str, Any] = {}
    if isinstance(document.get("condition"), dict):
        out["condition"] = dict(document["condition"])
        out["condition_source"] = str(document.get("condition_source") or "rule")
    if document.get("token_conflict"):
        out["token_conflict"] = list(document["token_conflict"])
    return out


@dataclass(frozen=True, slots=True)
class LoadedRecording:
    """An A.1 :class:`Recording` plus what the source file said that A.1 cannot hold.

    Attributes
    ----------
    recording
        The A.1 record: float64 microvolts, with a channel table.
    channel_map
        The map used, so a caller can see the geometry and the declared units.
    stim_end_s
        End of the stimulation period in seconds, or ``None`` when the file does not
        say. Task 03B splits on this.
    existing_bad_spans_s
        Previously marked bad spans as ``[start, stop)`` seconds. Empty when none.
    direction_valid
        False when ``rostral_end`` is unknown, in which case velocities are
        unsigned. Carried here so every downstream record can inherit it.
    provenance
        The loader's provenance dict plus what this adapter did.
    """

    recording: Recording
    channel_map: ChannelMap
    stim_end_s: float | None = None
    existing_bad_spans_s: list[tuple[float, float]] = field(default_factory=list)
    direction_valid: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


def _cohort_spec(path: Path, store: GemsStore | None) -> ProtocolSpec | None:
    """Return the cohort's protocol from ``protocol.yaml``, or ``None``.

    Units and ``rostral_end`` are cohort constants (invariant 24), so they live in
    the protocol and not in 800 copies of ``meta.json``. Resolved from the scan
    root the recording sits under, which is the same lookup ``split_stim_recovery``
    uses for the stim prior - one protocol, consulted the same way by everything.

    ``None`` only when there is no store to read a protocol from (the test path)
    or the recording sits outside every scan root. Not a place to guess: the
    caller passes an explicit map carrying its own units, or the plausibility
    check refuses the ``"uV"`` fallback, and direction stays unsigned.
    """
    if store is None:
        return None
    try:
        book = load_protocol_book(store.root / PROTOCOL_FILENAME)
        rel = path.resolve().relative_to(Path(store.root).resolve()).as_posix()
        return book.for_path(rel)
    except (FileNotFoundError, ValueError, ProtocolNotCoveredError):
        return None


def load_recording(
    path: Path,
    animal: str,
    *,
    channel_map: ChannelMap | None = None,
    session: str | None = None,
    store: GemsStore | None = None,
    profiles_root: Path | None = None,
    detector_core_root: Path | None = None,
) -> LoadedRecording:
    """Load ``path`` and return it as an A.1 :class:`Recording` plus its side-car.

    Parameters
    ----------
    path
        A ``.mat``, flat HDF5 or splitter-chunked HDF5 recording. Flat HDF5 is
        preferred where a choice exists (``detector-pyqt/scripts/m1_ingest.py``
        converts to it); that claim is theirs and is not re-measured here.
    animal
        Animal identifier. Required: it selects the profile, and recovering it from
        the filename is task 03A's condition inference, not this function's guess.
    channel_map
        The geometry, overriding every stored copy. When omitted it is resolved from
        ``meta.json`` in ``store`` and, failing that, from the profile mirror with a
        warning. Without any of those this raises rather than inventing a table.
    session
        Store key. Defaults to :func:`~gems_blanking_v2.io.tdt_block.session_key`,
        the acquisition block plus its ``.tsq`` start time - the one construction
        site, shared with the generator and the scan. A file with no acquisition
        record has NO store key: its stem is kept as a provenance label only and
        the store is not consulted for its geometry, because a stem is exactly the
        dateless key the store stopped using.
    store
        The Drive store holding ``data/<animal>/<session>/meta.json``, the
        authoritative geometry.
    profiles_root
        Profile-mirror directory, for tests.
    detector_core_root
        ``GEMSBlanking`` checkout, for tests.

    Returns
    -------
    LoadedRecording

    Raises
    ------
    FileNotFoundError
        If the file, or the ``GEMSBlanking`` checkout, is missing.
    ValueError
        If no channel map is available, or the table does not match the file's
        channel count. A mismatch means the geometry describes a different
        recording, and silently truncating it would mislabel every channel.
    """
    path = Path(path)
    recording_io = import_detector_module("recording_io", detector_core_root)
    native = recording_io.load_recording(path)
    spec = _cohort_spec(path, store)
    key = session if session is not None else session_key(path)
    session_id = key if key is not None else str(native.recording_id)
    keyed_store = store if key is not None else None

    mapping = resolve_channel_map(
        animal,
        explicit=channel_map,
        session=session_id,
        store=keyed_store,
        profiles_root=profiles_root,
        units=cast("Units", spec.units) if spec is not None else "uV",
    )
    if mapping is None:
        where = (
            f"record it in data/{animal}/{session_id}/meta.json, or pass channel_map="
            if key is not None else
            "there is no .tsq beside the file, so it has no store key; pass session= "
            "or channel_map="
        )
        msg = (
            f"no geometry for animal {animal!r} session {session_id!r}: {where}. The "
            "file has no channel table, so geometry cannot be inferred from it and "
            "must not be guessed."
        )
        raise ValueError(msg)

    # The file's own labels are authoritative for ORDER. meta.json declares only
    # what the file does not state, so a disagreement here means the map
    # describes a different recording - or the same one with its cuffs
    # transposed, which inverts every left/right result silently.
    # rostral_end is declared once in protocol.yaml; the map from meta.json never
    # carries it. with_rostral_end fills nerve contacts only, leaves an explicit
    # map's own value alone, and is a no-op for hw_tripole (unsignable).
    mapping = with_rostral_end(mapping, spec.rostral_end if spec is not None else None)

    labels = read_channel_labels(path)
    assert_labels_match(mapping.channels, labels, f"{path.name}")

    n_channels = int(native.y.shape[1])
    if mapping.n_channels != n_channels:
        msg = (
            f"channel map for {animal!r} has {mapping.n_channels} channels but "
            f"{path.name} has {n_channels}; the map describes a different recording"
        )
        raise ValueError(msg)

    fs = float(native.fs)
    if not np.isfinite(fs) or fs <= 0:
        msg = f"{path.name} reports a non-physical fs of {native.fs!r}"
        raise ValueError(msg)

    # float32 -> float64 first, then scale, so the multiply happens at full width.
    # Units stay DECLARED (invariant 14) - this infers nothing. It only refuses a
    # declaration that cannot be true, which is what makes a 10^6 typo loud
    # instead of a plausible-looking signal six steps downstream.
    median_sigma_uv = assert_plausible_units(
        np.asarray(native.y, dtype=np.float64), mapping.units, mapping.scale_uv,
        f"{path.name} (animal {animal!r})",
    )
    data: F64 = np.asarray(native.y, dtype=np.float64) * mapping.scale_uv

    recording = Recording(
        fs=fs,
        data=data,
        channels=list(mapping.channels),
        animal=animal,
        session=session_id,
        path=path,
    )

    stim_end_s: float | None = None
    if getattr(native, "stim_end_idx", None) is not None:
        # 1-based inclusive end -> the exclusive boundary in seconds.
        stim_end_s = int(native.stim_end_idx) / fs

    mapping.warn_if_direction_unknown(str(native.recording_id))

    provenance: dict[str, Any] = {
        "loader": "GEMSBlanking:detector/recording_io.load_recording",
        "source_format": dict(getattr(native, "provenance", {}) or {}).get("format"),
        "rec_type": getattr(native, "rec_type", None),
        "recording_id": str(native.recording_id),
        "session": session_id,
        "session_source": (
            "explicit" if session is not None
            else "tsq" if key is not None
            else "file_stem_label_not_a_store_key"
        ),
        "declared_units": mapping.units,
        "scale_to_uv": mapping.scale_uv,
        # Invariant 26: the plausibility check computes this on every load, so
        # recording it here accumulates the cohort's sigma distribution for free
        # as labelling proceeds. A sweep to obtain it would re-read the bytes a
        # pass that had to happen has already read.
        "median_robust_sigma_uv": median_sigma_uv,
        "native_dtype": str(np.asarray(native.y).dtype),
        "config": mapping.config,
        "geometry_source": "explicit" if channel_map is not None else "stored",
        **_condition_provenance(keyed_store, animal, session_id),
        # Andrea's exclusion (2026-09-26) travels with the data. Loading is not
        # refused - QC and review still need to open it - but every consumer
        # that builds a corpus or a label set reads this and must drop it.
        **({"excluded": ex} if keyed_store is not None
           and (ex := read_exclusion(keyed_store, animal, session_id)) else {}),
        "direction_valid": mapping.direction_valid,
    }

    return LoadedRecording(
        recording=recording,
        channel_map=mapping,
        stim_end_s=stim_end_s,
        existing_bad_spans_s=spans_from_matlab_intervals(
            getattr(native, "existing_bad_intervals", None), fs
        ),
        direction_valid=mapping.direction_valid,
        provenance=provenance,
    )
