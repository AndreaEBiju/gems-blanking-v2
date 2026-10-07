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

__all__ = ["T0_TOLERANCE_S", "frame_sample_bounds", "n_grid_frames", "to_matlab_inclusive"]

T0_TOLERANCE_S: Final = 1e-9
"""Two time origins closer than this are the same origin (float noise only)."""


def frame_sample_bounds(i0: int, i1: int, fs: float, grid_s: float = GRID_S) -> tuple[int, int]:
    """Return the 0-based half-open samples ``[k0, k1)`` covered by frames ``[i0, i1)``."""
    return int(round(i0 * grid_s * fs)), int(round(i1 * grid_s * fs))


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
