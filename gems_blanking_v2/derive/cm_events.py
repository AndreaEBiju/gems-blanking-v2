"""Common-mode events: the simultaneity filter of task 09's ROUND 4 MISS ruling.

Every contact of the new cohort is single-ended against one ground, so what happens
at the ground site appears on all nine channels at once. An event is a peak of the
channel-median 300-3000 Hz amplitude (in each channel's own robust sigma) of at
least :data:`CANDIDATE_SIGMA`, at least :data:`MIN_SPACING_S` from the next, at which
at least :data:`MIN_CHANNELS` channels have their extremum within
:data:`EXTREMUM_SEARCH_S`, and
(a) at the same instant: within :data:`SAME_INSTANT_S` of the channels' median time,
(b) with the majority polarity,
(c) of comparable size: :data:`AMP_RATIO` of the channels' median absolute amplitude,
and whose common-mode average (10 Hz high-pass, raw shape) is narrower than
:data:`MAX_WIDTH_S` at half height. Measured 2026-09-29/30 on the round-3/4
recordings: 124-2,900 events per minute, 30/40 injected identical 1.2 ms transients
recovered to 0.2 ms against that background, and part of the class is the
heartbeat's own sharp edge.

OUTSIDE THE GENERATION HASH: nothing that decides a candidate imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, find_peaks, sosfiltfilt

from gems_blanking_v2.types import Recording

__all__ = [
    "AMP_RATIO",
    "CANDIDATE_SIGMA",
    "EXTREMUM_SEARCH_S",
    "MAX_WIDTH_S",
    "MIN_CHANNELS",
    "MIN_SPACING_S",
    "SAME_INSTANT_S",
    "Events",
    "find_events",
]

F64 = npt.NDArray[np.float64]

CANDIDATE_SIGMA: Final = 6.0
MIN_SPACING_S: Final = 0.003
MIN_CHANNELS: Final = 7
EXTREMUM_SEARCH_S: Final = 0.0005
SAME_INSTANT_S: Final = 0.0002
AMP_RATIO: Final[tuple[float, float]] = (0.5, 2.0)
MAX_WIDTH_S: Final = 0.003
_CHUNK_S: Final = 60.0
_PAD_S: Final = 0.5


@dataclass(frozen=True)
class Events:
    """Common-mode events of one recording (times on its own timeline, seconds).

    ``cm_amp_uv``: the common-mode average's peak (10 Hz high-pass), microvolts;
    ``channel_amp_uv``: each channel's 300-3000 Hz extremum at the event, signed,
    ``(n_events, n_channels)`` in ``channels`` order; ``width_s``: at half height.
    """

    t_s: F64
    cm_amp_uv: F64
    channel_amp_uv: F64
    width_s: F64
    channels: tuple[str, ...]


def _simultaneous(e: F64, p: int, fs: float) -> tuple[npt.NDArray[np.bool_], float, F64] | None:
    """Rules (a)-(c) at candidate ``p``: which channels agree, their time and amplitude."""
    w5 = int(round(EXTREMUM_SEARCH_S * fs))
    i0, i1 = p - w5, p + w5 + 1
    if i0 < 0 or i1 > e.shape[1]:
        return None
    seg = e[:, i0:i1]
    k = np.argmax(np.abs(seg), axis=1)
    amp = seg[np.arange(seg.shape[0]), k]
    tt = (i0 + k) / fs
    tmed, pol, amed = float(np.median(tt)), np.sign(np.median(amp)), float(np.median(np.abs(amp)))
    ok = ((np.abs(tt - tmed) <= SAME_INSTANT_S) & (np.sign(amp) == pol)
          & (np.abs(amp) >= AMP_RATIO[0] * amed) & (np.abs(amp) <= AMP_RATIO[1] * amed))
    if int(ok.sum()) < MIN_CHANNELS:
        return None
    return ok, tmed, amp


def _width(cm: F64, fs: float) -> tuple[float, float]:
    """(peak, width at half height) of a common-mode average centred in ``cm``."""
    half, m = cm.size // 2, int(round(0.001 * fs))
    cm = cm - np.median(cm)
    j = half - m + int(np.argmax(np.abs(cm[half - m:half + m])))
    peak = abs(float(cm[j]))
    left, right = j, j
    while left > 0 and abs(cm[left]) > peak / 2:
        left -= 1
    while right < cm.size - 1 and abs(cm[right]) > peak / 2:
        right += 1
    return peak, (right - left) / fs


def find_events(rec: Recording) -> Events:
    """Find every common-mode event in ``rec`` (all channels, NaN-free input)."""
    fs = float(rec.fs)
    names = tuple(c.name for c in rec.channels)
    n = rec.data.shape[0]
    eng = butter(4, (300.0, min(3000.0, fs / 2 - 1)), btype="bandpass", fs=fs, output="sos")
    hp = butter(4, 10.0, btype="highpass", fs=fs, output="sos")
    ch, pad = int(_CHUNK_S * fs), int(_PAD_S * fs)

    def block(a: int, b: int) -> F64:
        return np.stack([np.asarray(rec.data[a:b, c.index], dtype=np.float64)
                         for c in rec.channels])

    parts = []  # session sigma per channel, from every 10th sample of each chunk
    for a in range(0, n, ch):
        lo, hi = max(a - pad, 0), min(a + ch + pad, n)
        e = sosfiltfilt(eng, block(lo, hi), axis=1)
        parts.append(np.abs(e[:, a - lo:min(a + ch, n) - lo][:, ::10]))
    sigma = np.median(np.concatenate(parts, axis=1), axis=1) / 0.6745
    half = int(round(0.010 * fs))
    t_out, amp_out, chan_out, width_out = [], [], [], []
    for a in range(0, n, ch):
        lo, hi = max(a - pad, 0), min(a + ch + pad, n)
        b = block(lo, hi)
        e = sosfiltfilt(eng, b, axis=1)
        score = np.median(np.abs(e) / sigma[:, None], axis=0)
        pk, _ = find_peaks(score, height=CANDIDATE_SIGMA,
                           distance=max(int(MIN_SPACING_S * fs), 1))
        pk = pk[(pk >= a - lo) & (pk < min(a + ch, n) - lo)]
        hpd = None
        for p in pk:
            found = _simultaneous(e, int(p), fs)
            if found is None:
                continue
            ok, tmed, amp = found
            hpd = sosfiltfilt(hp, b, axis=1) if hpd is None else hpd
            c0 = int(round(tmed * fs))
            if c0 - half < 0 or c0 + half >= hpd.shape[1]:
                continue
            peak, width = _width(hpd[ok, c0 - half:c0 + half].mean(axis=0), fs)
            if width >= MAX_WIDTH_S:
                continue
            t_out.append(lo / fs + tmed)
            amp_out.append(peak)
            chan_out.append(amp)
            width_out.append(width)
    order = np.argsort(t_out)
    return Events(t_s=np.asarray(t_out, dtype=np.float64)[order],
                  cm_amp_uv=np.asarray(amp_out, dtype=np.float64)[order],
                  channel_amp_uv=(np.asarray(chan_out, dtype=np.float64)[order]
                                  if chan_out else np.zeros((0, len(names)))),
                  width_s=np.asarray(width_out, dtype=np.float64)[order], channels=names)
