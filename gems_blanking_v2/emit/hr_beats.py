"""Store the pipeline's beats in Andrea's HRV analysis's own format (Andrea, 2026-09-30).

Beats are computed once, by the pipeline's beat detector (task 05's, with the
transient veto), and her post-processing-only ``HR_BR_HRVAnalysis_beats.m`` reads them
instead of running its own ``findpeaks``. The format is read from her code, not
invented: ``HR_BR_HRVAnalysis_new`` builds ``heartlocs`` - a column of 1-BASED SAMPLE
indices into the analysed channel ``signal(:, chanidx)`` - and saves it in
``<condition>_HRBR.mat``. The beats file holds exactly that plus ``fs`` (checked by
her function against the signal's) and provenance fields it does not read.

The index conversion is the MATLAB boundary (invariant 15): a beat at 0-based sample
``k`` of the epoch is ``heartlocs = k + 1``. A recording with no vetted channel gets
no file at all; her function then raises instead of falling back to peak finding.

Addendum to ruling (c), 2026-09-30: MASK-GRADE beats - a train that passes the
count gate but fails the transient veto - may place the peri-R time mask and nothing
else. They are written by :func:`write_mask_beats` under a different file name
(:data:`MASK_BEATS_SUFFIX`) and a different variable (``maskBeatlocs``, never
``heartlocs``), so her function, which loads ``heartlocs``, refuses such a file with
``badBeatsFile`` even if it is handed one.

Ruling 2026-10-02 (g) 4: per-minute storage removes the beats of rejected minutes, and
``blankSpans`` carries those spans to her function's optional ``'BlankSpans'`` input, which
merges them into her blank mask exactly as artifact blanking (``blankIdx``). Same
convention as ``blankIdx``: an N x 2 array of 1-BASED INCLUSIVE sample indices
``[start stop]`` into the epoch. Our half-open ``[a, b)`` seconds become
``[round((a - epoch) * fs) + 1, round((b - epoch) * fs)]`` - so ``[4/fs, 5/fs)``, one
sample, is ``[5 5]`` (invariant 15).

THE ORIGIN OF A TRAIN (found 2026-10-09, fix (c) of ``origin_check/REPORT.md``): the
routing wrote every train with ``epoch_start_s = 0`` while detecting its beats on the
routing REGION ``data[round(lo fs) : round(hi fs)]``. For a stim_rec recording that
region starts at 132 s, so ``heartlocs`` are 1-based into the region while the file
declares 0 - and the routing readers rely on that 0, so the files are not rewritten.
The origin is instead taken from the build record that sliced the region, by
:func:`train_origin` (the ONE place, invariant 33), and every consumer is handed it as a
0-based FILE SAMPLE, ``origin_sample0``: heartloc ``h`` is file sample
``origin_sample0 + h - 1``. :func:`beats_file_record` is the one construction site of
the mask provenance's ``extra.beats_file``, which the Night 6 wrapper slices with.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat, savemat

from gems_blanking_v2.emit.peri_r import beat_epoch_samples
from gems_blanking_v2.extent.grid import seconds_to_sample

__all__ = [
    "BEATS_FILE_PATH_KEY",
    "MASK_BEATS_SUFFIX",
    "beats_file_record",
    "read_blank_spans",
    "read_gap_after",
    "read_hr_beats",
    "read_mask_beats",
    "to_blank_spans",
    "to_heartlocs",
    "train_origin",
    "write_hr_beats",
    "write_mask_beats",
]

F64 = npt.NDArray[np.float64]

MASK_BEATS_SUFFIX = "_peri_r_beats.mat"
"""Every mask-grade beats file, and no HRV beats file, ends with this."""

BEATS_FILE_PATH_KEY: Final = {"hrv": "hrv_beats", "mask": "mask_grade_beats"}
"""``extra.beats_file`` key naming the train's store path, by grade. The Night 6 wrapper
reads only ``hrv_beats`` - a mask-grade train places the peri-R mask and nothing else."""

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def train_origin(region_s: Sequence[float], fs: float, *, what: str,
                 n_region_samples: int | None = None, declared_sample0: int | None = None,
                 declared_start_s: float | None = None) -> tuple[int, int]:
    """Return ``(origin_sample0, n_region)`` of a train from its build record's region.

    ``origin_sample0`` is the 0-based file sample of heartloc 1, through the handoff's own
    :func:`~gems_blanking_v2.emit.handoff.epoch_start_fields` (``round(lo fs)``, the rule
    the region was sliced with - invariants 15, 22); ``n_region`` is the region's length in
    samples. Refuses, naming ``what``: a malformed or negative region; a build record whose
    ``n_samples`` disagrees with the region; a file that DECLARES its origin as
    ``epochStartSample0`` (``declared_sample0``) or as a nonzero ``epochStart_s``
    (``declared_start_s``) that disagrees with the build record. A declared ``epochStart_s``
    of 0 is the routing's stored 0 (module docstring) and is not a declaration.
    """
    from gems_blanking_v2.emit.handoff import (  # noqa: PLC0415 - handoff imports heavy modules
        EPOCH_START_SAMPLE_KEY,
        epoch_start_fields,
    )
    try:
        lo, hi = (float(x) for x in region_s)
    except (TypeError, ValueError):
        msg = f"{what}: build record region_s must be [start, stop] seconds, got {region_s!r}"
        raise ValueError(msg) from None
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        msg = f"{what}: build record region_s {region_s!r} is not a finite [start, stop)"
        raise ValueError(msg)
    origin = int(epoch_start_fields(lo, fs)[EPOCH_START_SAMPLE_KEY])
    n_region = seconds_to_sample(hi, fs) - origin
    if n_region_samples is not None and int(n_region_samples) != n_region:
        msg = (f"{what}: build record n_samples {int(n_region_samples)} disagrees with its "
               f"region_s {[lo, hi]} ({n_region} samples at {fs} Hz)")
        raise ValueError(msg)
    if declared_sample0 is not None and int(declared_sample0) != origin:
        msg = (f"{what}: the beats file declares epochStartSample0 {int(declared_sample0)} but "
               f"its build record's region starts at file sample {origin}")
        raise ValueError(msg)
    if declared_start_s is not None and float(declared_start_s) != 0.0 and (
            seconds_to_sample(float(declared_start_s), fs) != origin):
        msg = (f"{what}: the beats file declares epochStart_s {float(declared_start_s)} but its "
               f"build record's region starts at file sample {origin}")
        raise ValueError(msg)
    return origin, n_region


def beats_file_record(*, grade: str, store_rel: str, sha256: str, origin_sample0: int,
                      heartlocs: npt.ArrayLike, epoch_start_sample: int, n_samples: int,
                      published: bool, read_from: str) -> dict[str, Any]:
    """Return one epoch's ``extra.beats_file``: the train, its origin and its beat count.

    ``store_rel`` is where the train is (or will be) published, POSIX and relative to
    ``gems_root`` (cross-platform rule 2); ``read_from`` is where it was read for this
    record. ``n_in_epoch`` counts the beats whose file sample
    ``origin_sample0 + h - 1`` lies in ``[epoch_start_sample, epoch_start_sample + n)``,
    through :func:`~gems_blanking_v2.emit.peri_r.beat_epoch_samples` - the count the
    Night 6 wrapper must reproduce or refuse.
    """
    if grade not in BEATS_FILE_PATH_KEY:
        msg = f"grade must be one of {sorted(BEATS_FILE_PATH_KEY)}, got {grade!r}"
        raise ValueError(msg)
    if (PurePosixPath(store_rel).is_absolute() or PureWindowsPath(store_rel).drive
            or "\\" in store_rel):
        msg = f"store_rel must be a relative POSIX path, got {store_rel!r}"
        raise ValueError(msg)
    if not _SHA256.match(sha256):
        msg = f"sha256 must be 64 lowercase hex characters, got {sha256!r}"
        raise ValueError(msg)
    for name, v in (("origin_sample0", origin_sample0), ("epoch_start_sample", epoch_start_sample),
                    ("n_samples", n_samples)):
        if isinstance(v, bool) or not isinstance(v, int | np.integer) or int(v) < 0:
            msg = f"{name} must be a non-negative integer sample count, got {v!r}"
            raise TypeError(msg)
    r = beat_epoch_samples(heartlocs, origin_sample=int(origin_sample0),
                           epoch_start_sample=int(epoch_start_sample))
    return {BEATS_FILE_PATH_KEY[grade]: store_rel, "grade": grade, "sha256": sha256,
            "origin_sample0": int(origin_sample0),
            "origin_rule": "heartloc h is 0-based file sample origin_sample0 + h - 1; the "
                           "origin is the routing build record's region start, never the "
                           "file's epochStart_s",
            "epoch_start_sample0": int(epoch_start_sample),
            "n_in_epoch": int(((r >= 0) & (r < int(n_samples))).sum()),
            "n_beats": int(r.size), "published_at_mask_time": bool(published),
            "read_from": read_from}


def to_heartlocs(beats_s: npt.ArrayLike, fs: float, epoch_start_s: float, n_samples: int) -> F64:
    """Return her ``heartlocs``: 1-based sample indices into the epoch, as a column.

    ``beats_s`` are on the recording's timeline; the epoch starts at
    ``epoch_start_s`` and holds ``n_samples``. Raises for a beat outside the epoch or
    two beats on one sample - her function requires strictly increasing indices.
    """
    t = np.sort(np.asarray(beats_s, dtype=np.float64))
    k = np.round((t - epoch_start_s) * fs).astype(np.int64)  # 0-based
    if k.size and (k.min() < 0 or k.max() >= n_samples):
        msg = f"a beat falls outside the epoch of {n_samples} samples"
        raise ValueError(msg)
    if k.size > 1 and np.any(np.diff(k) <= 0):
        msg = "two beats round to one sample; her heartlocs must be strictly increasing"
        raise ValueError(msg)
    return (k + 1).astype(np.float64).reshape(-1, 1)


def to_blank_spans(spans_s: npt.ArrayLike, fs: float, epoch_start_s: float,
                   n_samples: int) -> F64:
    """Return her ``blankSpans``: half-open ``[a, b)`` s -> N x 2 1-based inclusive samples.

    Clipped to the epoch; a span shorter than one sample after rounding is dropped. Raises
    for overlapping or unsorted spans.
    """
    sp = np.asarray(spans_s, dtype=np.float64).reshape(-1, 2)
    k0 = np.clip(np.round((sp[:, 0] - epoch_start_s) * fs).astype(np.int64), 0, n_samples)
    k1 = np.clip(np.round((sp[:, 1] - epoch_start_s) * fs).astype(np.int64), 0, n_samples)
    keep = k1 > k0
    k0, k1 = k0[keep], k1[keep]
    if k0.size > 1 and np.any(k0[1:] < k1[:-1]):
        msg = "blank spans must be sorted and must not overlap"
        raise ValueError(msg)
    return np.column_stack([k0 + 1, k1]).astype(np.float64).reshape(-1, 2)


def write_hr_beats(
    path: Path, beats_s: npt.ArrayLike, *, fs: float, epoch_start_s: float, n_samples: int,
    channel: str, source: str, gap_after: npt.ArrayLike | None = None,
    blank_spans_s: npt.ArrayLike | None = None,
) -> Path:
    """Write the beats file her ``HR_BR_HRVAnalysis_beats`` reads (a fully vetted train).

    ``gap_after`` (ruling (h) 3): per beat, in ``beats_s``'s sorted order, whether the
    interval after it spans a tagged gap - a dropped or unfilled beat. Stored as
    ``gapAfter``, a logical column the length of ``heartlocs``. Her function loads only
    ``heartlocs`` and ``fs``, so its outputs do not change; using the tags is Andrea's
    decision. A single dropped beat reads as one ~2 RR interval, inside her [100, 500] ms
    range, so without the tags HRV would take it as real.

    ``blank_spans_s`` (ruling 2026-10-02 (g) 4): half-open ``[a, b)`` s on the recording's
    timeline, stored as ``blankSpans`` (:func:`to_blank_spans`). No stored beat may fall in
    one - a rejected minute's beats are removed, never left to look valid.
    """
    if Path(path).name.endswith(MASK_BEATS_SUFFIX):
        msg = f"{Path(path).name} is a mask-grade name; HRV beats must not carry it"
        raise ValueError(msg)
    heartlocs = to_heartlocs(beats_s, fs, epoch_start_s, n_samples)
    doc = {"heartlocs": heartlocs, "fs": float(fs), "beatChannel": channel, "source": source,
           "epochStart_s": float(epoch_start_s)}
    if gap_after is not None:
        raw = np.asarray(gap_after, dtype=bool).ravel()
        if raw.size != heartlocs.shape[0]:
            msg = f"gap_after has {raw.size} entries for {heartlocs.shape[0]} beats"
            raise ValueError(msg)
        order = np.argsort(np.asarray(beats_s, dtype=np.float64), kind="stable")
        doc["gapAfter"] = raw[order].reshape(-1, 1)
    if blank_spans_s is not None:
        spans = to_blank_spans(blank_spans_s, fs, epoch_start_s, n_samples)
        locs = heartlocs.ravel()
        inside = ((locs[:, None] >= spans[None, :, 0])
                  & (locs[:, None] <= spans[None, :, 1])).any(axis=1)
        if inside.any():
            msg = f"{int(inside.sum())} stored beat(s) fall inside a blank span"
            raise ValueError(msg)
        doc["blankSpans"] = spans
    savemat(path, doc, do_compression=False)
    return path


def read_gap_after(path: Path) -> npt.NDArray[np.bool_] | None:
    """Return the file's ``gapAfter`` tags, or None when it carries none."""
    m = loadmat(path)
    if "gapAfter" not in m:
        return None
    return np.asarray(m["gapAfter"], dtype=bool).ravel()


def read_blank_spans(path: Path) -> F64 | None:
    """Return the file's blank spans as half-open ``[a, b)`` s on the epoch's timeline, or None."""
    m = loadmat(path)
    if "blankSpans" not in m:
        return None
    sp = np.asarray(m["blankSpans"], dtype=np.float64).reshape(-1, 2)
    fs = float(np.asarray(m["fs"]).squeeze())
    start = float(np.asarray(m.get("epochStart_s", 0.0)).squeeze())
    return np.column_stack([(sp[:, 0] - 1.0) / fs + start, sp[:, 1] / fs + start])


def read_hr_beats(path: Path) -> tuple[F64, float]:
    """Return ``(beat times in seconds on the epoch's timeline, fs)`` from a beats file."""
    m = loadmat(path)
    fs = float(np.asarray(m["fs"]).squeeze())
    heartlocs = np.asarray(m["heartlocs"], dtype=np.float64).ravel()
    start = float(np.asarray(m.get("epochStart_s", 0.0)).squeeze())
    return (heartlocs - 1.0) / fs + start, fs


def write_mask_beats(
    path: Path, beats_s: npt.ArrayLike, *, fs: float, epoch_start_s: float, n_samples: int,
    channel: str, source: str,
) -> Path:
    """Write a mask-grade train for the peri-R mask only; unreadable by her function.

    The name must end with :data:`MASK_BEATS_SUFFIX`. Same 1-based indices as
    :func:`to_heartlocs`, stored as ``maskBeatlocs`` with ``grade = "peri_r_mask_only"``.
    """
    if not Path(path).name.endswith(MASK_BEATS_SUFFIX):
        msg = f"a mask-grade beats file must end with {MASK_BEATS_SUFFIX!r}: {Path(path).name}"
        raise ValueError(msg)
    savemat(path, {"maskBeatlocs": to_heartlocs(beats_s, fs, epoch_start_s, n_samples),
                   "fs": float(fs), "beatChannel": channel, "source": source,
                   "grade": "peri_r_mask_only", "epochStart_s": float(epoch_start_s)},
            do_compression=False)
    return path


def read_mask_beats(path: Path) -> tuple[F64, float]:
    """Return ``(beat times s, fs)`` from a mask-grade file; raises on an HRV file."""
    m = loadmat(path)
    if "maskBeatlocs" not in m:
        msg = f"{Path(path).name} holds no maskBeatlocs: not a mask-grade beats file"
        raise ValueError(msg)
    fs = float(np.asarray(m["fs"]).squeeze())
    locs = np.asarray(m["maskBeatlocs"], dtype=np.float64).ravel()
    start = float(np.asarray(m.get("epochStart_s", 0.0)).squeeze())
    return (locs - 1.0) / fs + start, fs
