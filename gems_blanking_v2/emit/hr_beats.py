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

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat, savemat

__all__ = ["read_hr_beats", "to_heartlocs", "write_hr_beats"]

F64 = npt.NDArray[np.float64]


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
    """Write the beats file her ``HR_BR_HRVAnalysis_beats`` reads."""
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
