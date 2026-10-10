"""The one conversion between the 10 ms frame grid and sample indices (invariants 15, 33).

Frame ``i`` of an epoch covers ``[i * grid_s, (i + 1) * grid_s)`` seconds from the epoch
start, which is samples ``[round(i * grid_s * fs), round((i + 1) * grid_s * fs))`` -
0-based, half-open. Every place that turns frames into samples - the sample mask, the
MATLAB blank spans, the clip test - calls :func:`frame_sample_bounds`, so they cannot
disagree about a boundary sample. The MATLAB form of ``[k0, k1)`` is ``[k0 + 1, k1]``
(1-based inclusive; :func:`to_matlab_inclusive`).

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.constants import GRID_S

__all__ = ["SAMPLE_TOLERANCE", "T0_TOLERANCE_S", "first_sample_at_or_after",
           "frame_sample_bounds", "n_grid_frames", "seconds_to_sample", "to_matlab_inclusive"]

T0_TOLERANCE_S: Final = 1e-9
"""Two time origins closer than this are the same origin (float noise only)."""

SAMPLE_TOLERANCE: Final = 1e-6
"""A product ``t fs`` within this many samples of an integer IS that integer.

Float noise only: a time written as ``k / fs`` and multiplied back is off by ~1e-9 samples
at 10^7 samples, so ``ceil`` alone would move an exact sample one later. The MATLAB twin
(``night6_first_sample.m``) uses the same tolerance.
"""


def seconds_to_sample(t_s: float, fs: float) -> int:
    """Return the 0-based sample index of ``t_s`` seconds from the epoch start: ``round(t fs)``.

    The one time-to-sample rounding: frame bounds (:func:`frame_sample_bounds`) and the
    line-distrust minute bounds (``emit.line_distrust``) both go through it.
    """
    return int(round(t_s * fs))


def first_sample_at_or_after(t_s: float, fs: float) -> int:
    """Return the first 0-based sample at or after ``t_s`` seconds: ``ceil(t fs)``.

    The one conversion of an EVENT time (a stim end, a settling point) to the first sample
    it no longer covers - the rule's ``k = ceil(t fs)`` (``round`` is for window bounds,
    :func:`seconds_to_sample`). A product within :data:`SAMPLE_TOLERANCE` of an integer is
    that integer, so ``first_sample_at_or_after(k / fs, fs) == k``. Monotone in ``t_s``.
    MATLAB: ``night6_first_sample`` (invariant 22: one canonical form on both sides).
    """
    x = float(t_s) * float(fs)
    k = round(x)
    if abs(x - k) <= SAMPLE_TOLERANCE:
        return int(k)
    return int(math.ceil(x))


def frame_sample_bounds(i0: int, i1: int, fs: float, grid_s: float = GRID_S) -> tuple[int, int]:
    """Return the 0-based half-open samples ``[k0, k1)`` covered by frames ``[i0, i1)``."""
    return seconds_to_sample(i0 * grid_s, fs), seconds_to_sample(i1 * grid_s, fs)


def n_grid_frames(n_samples: int, fs: float, grid_s: float = GRID_S) -> int:
    """Whole frames in an epoch of ``n_samples``: ``floor(duration / grid_s)``."""
    return int(math.floor(n_samples / fs / grid_s))


def to_matlab_inclusive(k0: npt.ArrayLike, k1: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """``[k0, k1)`` 0-based half-open to MATLAB's ``[k0 + 1, k1]`` 1-based inclusive, N x 2.

    ``[4, 5)`` - one sample, the fifth - is ``[5 5]``.
    """
    a = np.asarray(k0, dtype=np.int64).ravel()
    b = np.asarray(k1, dtype=np.int64).ravel()
    if np.any(b <= a):
        msg = "every span must hold at least one sample"
        raise ValueError(msg)
    return np.column_stack([a + 1, b]).astype(np.float64).reshape(-1, 2)
