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

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat, savemat

__all__ = [
    "MASK_BEATS_SUFFIX",
    "read_hr_beats",
    "read_mask_beats",
    "to_heartlocs",
    "write_hr_beats",
    "write_mask_beats",
]

F64 = npt.NDArray[np.float64]

MASK_BEATS_SUFFIX = "_peri_r_beats.mat"
"""Every mask-grade beats file, and no HRV beats file, ends with this."""


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


def write_hr_beats(
    path: Path, beats_s: npt.ArrayLike, *, fs: float, epoch_start_s: float, n_samples: int,
    channel: str, source: str,
) -> Path:
    """Write the beats file her ``HR_BR_HRVAnalysis_beats`` reads (a fully vetted train)."""
    if Path(path).name.endswith(MASK_BEATS_SUFFIX):
        msg = f"{Path(path).name} is a mask-grade name; HRV beats must not carry it"
        raise ValueError(msg)
    savemat(path, {"heartlocs": to_heartlocs(beats_s, fs, epoch_start_s, n_samples),
                   "fs": float(fs), "beatChannel": channel, "source": source,
                   "epochStart_s": float(epoch_start_s)}, do_compression=False)
    return path


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
