"""Per-spike outside-coincidence veto for the spike consumer (ruling 2026-09-30 (b), item 2).

Andrea, 2026-09-29: tell leak from nerve spikes by their characteristics. Physiology
cannot put one vagus's spike on the other cuff and the stomach at the same instant.
So, AFTER her detector has run on the ``T`` the consumer reads, each detected spike is
vetoed when the reference built from every channel OUTSIDE its cuff (300-3000 Hz, in
that reference's own robust sigma) peaks above ``theta`` within ``+/-w`` of it.

- ``w`` is the cuff's measured core half-width (the event-triggered spike histogram's
  narrow core), rounded UP to 0.1 ms (:func:`core_half_width`).
- ``theta`` is set on the first half of the recording so the veto fires at random
  times at most :data:`TARGET_CHANCE_LOSS` (:func:`calibrate_theta`); the chance loss
  is then measured on the second half (:func:`chance_loss`).
- Nothing is NaN'd and exposure is unchanged, so a rate is divided by
  ``1 - chance loss`` instead - the replacement for the withdrawn time mask, whose
  10 ms NaN pads let the spikes place the mask.

The gate is independent of the cuff under test: it reads only channels outside it.

Ruling 2026-09-30 (c), item 2: where the veto's verification fails, the cuff is
distrusted for the spike consumer in that recording (:data:`SPIKE_DISTRUSTED`,
:func:`spike_trusted`); a cuff distrusted for every consumer is distrusted here too.

OUTSIDE THE GENERATION HASH: nothing that decides a candidate imports it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.stats import poisson

from gems_blanking_v2.derive.common_mode import _eng, outside_reference
from gems_blanking_v2.derive.consumer_screen import DISTRUSTED_CUFFS
from gems_blanking_v2.types import Recording

__all__ = [
    "CORE_EXCESS_SHARE",
    "CORE_P",
    "CORE_P_WINDOW_S",
    "CORE_SEARCH_S",
    "SPIKE_DISTRUSTED",
    "TARGET_CHANCE_LOSS",
    "W_STEP_S",
    "calibrate_theta",
    "chance_loss",
    "core_half_width",
    "over_theta",
    "reference_sigma",
    "spike_trusted",
    "veto_spikes",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

TARGET_CHANCE_LOSS: Final = 0.02
"""At random times the veto fires at most this often, on the half theta is set on."""

W_STEP_S: Final = 0.0001
"""``w`` is rounded up to this: the histogram's 0.1 ms bins."""

CORE_SEARCH_S: Final = 0.002
"""The core is sought within this of zero lag; beyond it the histogram is flank."""

CORE_P_WINDOW_S: Final = 0.001
"""The window whose excess decides whether there is a core at all (+/-1 ms)."""

CORE_P: Final = 1e-3
"""The core's significance bar - the ratified lag-histogram rule's."""

CORE_EXCESS_SHARE: Final = 0.9
"""The half-width holds this share of the excess within :data:`CORE_SEARCH_S`."""


SPIKE_DISTRUSTED: Final[Mapping[tuple[str, str], str]] = {
    ("gems_h_t01_es2_bl_224901_20260905T024908Z", "L"): (
        "ruling 2026-09-30 (c) 2: fails check (iii) - an event core x3.0 survives the "
        "veto (w = 0.2 ms) - and is the low-gain cuff"),
    ("gems_a_t05_2_1_bl_183840_20260924T223845Z", "L"): (
        "ruling 2026-09-30 (c) 2: the left-right zero-lag core persists outside both "
        "cuffs' peri-R windows (x4.18, p 1.4e-6; x4.3 on all spikes)"),
    ("gems_a_t05_2_1_bl_183840_20260924T223845Z", "R"): (
        "ruling 2026-09-30 (c) 2: the left-right zero-lag core persists outside both "
        "cuffs' peri-R windows (x4.18, p 1.4e-6; x4.3 on all spikes)"),
}
"""(recording id, cuff) -> why, for the spike consumer only. Lifted only by a ruling."""


def spike_trusted(recording_id: str, cuff: str) -> tuple[bool, str]:
    """Return ``(trusted, why not)`` for the spike consumer reading ``cuff``'s ``T``.

    Distrusted when the cuff is in :data:`SPIKE_DISTRUSTED` or distrusted for every
    consumer (``consumer_screen.DISTRUSTED_CUFFS``); ``why`` is empty when trusted.
    """
    for table in (SPIKE_DISTRUSTED, DISTRUSTED_CUFFS):
        why = table.get((recording_id, cuff))
        if why is not None:
            return False, why
    return True, ""


def reference_sigma(rec: Recording, cuff: str) -> tuple[F64, float]:
    """``(|outside reference| / sigma, sigma)`` in 300-3000 Hz, sigma robust.

    The reference is the mean of every channel outside ``cuff``; sigma is
    median(|x|)/0.6745 over its finite samples - one scalar (invariant 5).
    """
    ref, _names = outside_reference(rec, cuff)
    e = _eng(ref, float(rec.fs))
    sigma = float(np.nanmedian(np.abs(e)) / 0.6745)
    return np.abs(e) / sigma, sigma


def core_half_width(counts: npt.ArrayLike, centres_s: npt.ArrayLike, flank_mean: float) -> float:
    """Return the core half-width, rounded up to :data:`W_STEP_S`; NaN when there is none.

    A core exists when the bins within :data:`CORE_P_WINDOW_S` of zero lag hold
    significantly more than the flank predicts (one-sided Poisson p <
    :data:`CORE_P`, the ratified lag-histogram rule). Its half-width is the smallest
    |lag| reach whose bins hold :data:`CORE_EXCESS_SHARE` of the excess (count minus
    ``flank_mean``) within :data:`CORE_SEARCH_S`, plus half a bin, rounded up to
    :data:`W_STEP_S`. Cumulative, so a core spread thinly over several bins - as at
    low spike counts - is still measured.
    """
    c = np.asarray(counts, dtype=np.float64)
    t = np.asarray(centres_s, dtype=np.float64)
    core = np.abs(t) <= CORE_P_WINDOW_S + 1e-12
    expected = flank_mean * int(core.sum())
    observed = float(c[core].sum())
    if expected <= 0 or float(poisson.sf(observed - 1, expected)) >= CORE_P:
        return float("nan")
    search = np.abs(t) <= CORE_SEARCH_S + 1e-12
    reaches = np.unique(np.round(np.abs(t[search]), 6))
    total = float((c[search] - flank_mean).sum())
    reach = float(reaches[-1])
    for r in reaches:
        if float((c[np.abs(t) <= r + 1e-12] - flank_mean).sum()) >= CORE_EXCESS_SHARE * total:
            reach = float(r)
            break
    half = reach + W_STEP_S / 2
    return math.ceil(round(half / W_STEP_S, 6)) * W_STEP_S


def over_theta(ref_sigma: F64, theta: float, w_s: float, fs: float) -> Bool:
    """Return, per sample, whether the reference exceeds ``theta`` within ``+/-w_s``.

    NaN never exceeds.
    """
    hit = np.asarray(np.nan_to_num(ref_sigma, nan=0.0) > theta, dtype=np.int64)
    k = max(int(round(w_s * fs)), 0)
    if k == 0:
        return hit.astype(bool)
    return np.asarray(np.convolve(hit, np.ones(2 * k + 1, dtype=np.int64), "same") > 0, dtype=bool)


def _peaks_at(ref_sigma: F64, times_s: F64, w_s: float, fs: float) -> F64:
    k = max(int(round(w_s * fs)), 0)
    n = ref_sigma.size
    out = np.empty(times_s.size)
    for i, t in enumerate(times_s):
        c = int(round(t * fs))
        seg = ref_sigma[max(c - k, 0):min(c + k + 1, n)]
        out[i] = float(np.nanmax(seg)) if seg.size and np.isfinite(seg).any() else 0.0
    return out


def calibrate_theta(
    ref_sigma: F64, w_s: float, fs: float, window_s: tuple[float, float], *,
    target: float = TARGET_CHANCE_LOSS, n: int = 20000, seed: int = 0,
) -> float:
    """Return the ``1 - target`` quantile of the reference's peak within ``+/-w_s``.

    Taken at ``n`` uniform random times in ``window_s``, so the veto fires there at
    ``target``.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(window_s[0] + w_s, window_s[1] - w_s, n)
    return float(np.quantile(_peaks_at(ref_sigma, t, w_s, fs), 1.0 - target))


def chance_loss(
    ref_sigma: F64, theta: float, w_s: float, fs: float, times_s: npt.ArrayLike,
) -> float:
    """Return the fraction of ``times_s`` (random times) at which the veto would fire."""
    t = np.asarray(times_s, dtype=np.float64)
    if t.size == 0:
        return float("nan")
    return float(np.mean(_peaks_at(ref_sigma, t, w_s, fs) > theta))


def veto_spikes(spike_times_s: npt.ArrayLike, over: Bool, fs: float) -> Bool:
    """Per spike: True where :func:`over_theta` says the outside reference coincided."""
    t = np.asarray(spike_times_s, dtype=np.float64)
    idx = np.clip(np.round(t * fs).astype(np.int64), 0, over.size - 1)
    return np.asarray(over[idx], dtype=bool)
