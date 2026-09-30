"""The gated common-mode mask for the spike consumer (ruling 2026-09-30, item 4c).

For cuffs whose event-locked firing survives correction: mask ``T`` for
:data:`MASK_HALF_S` either side of every common-mode event at which ``T`` (in the
consumer's 300-3000 Hz band) reaches :data:`GATE_SIGMA` times its own session sigma
within :data:`GATE_WINDOW_S` of the event. For the spike consumer only; never for a
cuff whose event peak is broad (ruling 3: a broad hump is physiology).

EXPOSURE ACCOUNTING IS MANDATORY: the events co-vary with breathing and heart rate,
so masked time must leave the denominator of every rate and every phase bin. The
consumer pads every NaN run by its edge buffer (``pipeline_params.edgeBufferMs`` =
10 ms), so the time it actually loses is the mask DILATED by that pad -
:func:`effective_exposure` returns it, and that, not the bare mask, is what a rate is
divided by. Masked samples are NaN, never zero (invariant 1).

OUTSIDE THE GENERATION HASH: nothing that decides a candidate imports it.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt

from gems_blanking_v2.constants import BANDS, ENG_BAND

__all__ = [
    "CONSUMER_EDGE_PAD_S",
    "GATE_SIGMA",
    "GATE_WINDOW_S",
    "MASK_HALF_S",
    "apply_mask",
    "effective_exposure",
    "gated_event_mask",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

GATE_SIGMA: Final = 4.0
"""The gate, in the consumer's own session sigma of ``T`` - below its 4.5 threshold."""

GATE_WINDOW_S: Final = 0.001
"""``T`` must reach the gate within this of the event: the event's own width."""

MASK_HALF_S: Final = 0.0015
"""Half-width of each masked span (ruling 2026-09-30, item 4c)."""

CONSUMER_EDGE_PAD_S: Final = 0.010
"""The spike consumer's pad around every NaN run (``pipeline_params.edgeBufferMs``)."""


def _eng(x: F64, fs: float) -> F64:
    lo, hi = BANDS[ENG_BAND].lo_hz, BANDS[ENG_BAND].hi_hz
    sos = butter(4, (lo, min(hi, fs / 2 - 1)), btype="bandpass", fs=fs, output="sos")
    bad = ~np.isfinite(x)
    if bad.any():  # temporary, for filtering only (invariant 8)
        idx = np.arange(x.size)
        x = np.interp(idx, idx[~bad], x[~bad])
    y = np.asarray(sosfiltfilt(sos, x), dtype=np.float64)
    y[bad] = np.nan
    return y


def gated_event_mask(t: F64, events_s: npt.ArrayLike, fs: float) -> tuple[Bool, float]:
    """Return ``(mask, sigma_uv)``: True where ``T`` is masked, and the sigma used.

    Sigma is the consumer's estimator on ``T``'s ENG band - median(|x|)/0.6745 over
    the finite samples of the whole input, one scalar (invariant 5). ``events_s``
    are sample-time seconds on ``t``'s own timeline.
    """
    e = _eng(np.asarray(t, dtype=np.float64), fs)
    sigma = float(np.nanmedian(np.abs(e)) / 0.6745)
    n = e.size
    w, h = int(round(GATE_WINDOW_S * fs)), int(round(MASK_HALF_S * fs))
    mask = np.zeros(n, dtype=bool)
    for t0 in np.asarray(events_s, dtype=np.float64):
        c = int(round(t0 * fs))
        a, b = max(c - w, 0), min(c + w + 1, n)
        if b > a and np.nanmax(np.abs(e[a:b])) >= GATE_SIGMA * sigma:
            mask[max(c - h, 0):min(c + h + 1, n)] = True
    return mask, sigma


def effective_exposure(mask: Bool, fs: float, pad_s: float = CONSUMER_EDGE_PAD_S) -> Bool:
    """Return the time the consumer actually loses: ``mask`` dilated by its NaN pad."""
    p = int(round(pad_s * fs))
    if not p or not mask.any():
        return np.asarray(mask, dtype=bool).copy()
    k = np.ones(2 * p + 1, dtype=np.int64)
    return np.asarray(np.convolve(mask.astype(np.int64), k, "same") > 0, dtype=bool)


def apply_mask(t: F64, mask: Bool) -> F64:
    """Return a copy of ``t`` with the masked samples NaN - never zero (invariant 1)."""
    out = np.array(t, dtype=np.float64)
    out[mask] = np.nan
    return out
