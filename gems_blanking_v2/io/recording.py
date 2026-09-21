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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.io.channel_map import ChannelMap, resolve_channel_map
from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.io.store import GemsStore
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
        Session identifier. Defaults to the file stem, which is what the loader
        already calls ``recording_id``, and which also selects the ``meta.json``.
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
    session_id = session if session is not None else str(native.recording_id)

    mapping = resolve_channel_map(
        animal,
        explicit=channel_map,
        session=session_id,
        store=store,
        profiles_root=profiles_root,
    )
    if mapping is None:
        msg = (
            f"no geometry for animal {animal!r} session {session_id!r}: record it in "
            f"data/{animal}/{session_id}/meta.json, or pass channel_map=. The file has "
            "no channel table, so geometry cannot be inferred from it and must not be "
            "guessed."
        )
        raise ValueError(msg)

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
        "declared_units": mapping.units,
        "scale_to_uv": mapping.scale_uv,
        "native_dtype": str(np.asarray(native.y).dtype),
        "config": mapping.config,
        "geometry_source": "explicit" if channel_map is not None else "stored",
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
