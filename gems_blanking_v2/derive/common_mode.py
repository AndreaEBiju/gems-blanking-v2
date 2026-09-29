"""Common-mode leak correction for a cuff's tripole: task 14's "subtract" route.

Ruling 2026-09-29. The new cohort has no reference channel: every contact is
single-ended against one ground in the abdominal wall, so whatever is at that site
enters all nine channels identically. A tripole with ``a + b = 1`` cancels an
identical signal exactly, but the contacts' gains differ by a few percent, so ``T``
keeps a leak of that size - measured, it drives up to 83 false spikes a minute on
one cuff. Masking every event would also remove real spikes that coincide with the
events, so the leak is SUBTRACTED instead:

1. the reference is the mean of every raw channel **outside** the cuff - the other
   cuff's contacts and the stomach contacts - which carry the ground-site signal
   with none of this nerve's activity;
2. ``k`` is the least-squares coefficient of ``T`` on that reference in the ENG band
   (300-3000 Hz, the band the spike consumer reads): one whole-recording scalar per
   cuff, stateless (invariants 5 and 7);
3. the corrected tripole is ``T - k * reference``, broadband, so the consumer filters
   it exactly as it filters ``T``.

The route requires verification (task 14), and this module supplies the parts that
do not need the consumer: the fit's residual, and :func:`differential_survival`,
which injects a spike present on this cuff only and measures how much of it
survives. Whether the spike detector stops firing on the events is measured with
the consumer itself.

NaN is honoured throughout (invariant 1): masked samples stay NaN in the output,
never enter the fit, and the samples within the filter's settling of a NaN run are
kept out of the fit too. Interpolation for filtering is temporary (invariant 8).

OUTSIDE THE GENERATION HASH: nothing that decides a candidate imports it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt

from gems_blanking_v2.constants import BANDS, ENG_BAND
from gems_blanking_v2.derive.derivations import NAIVE_WEIGHTS, tripole
from gems_blanking_v2.types import Recording

__all__ = [
    "FIT_ORDER",
    "SETTLE_S",
    "LeakFit",
    "differential_survival",
    "fit_leak",
    "outside_reference",
    "subtract_common_mode",
]

F64 = npt.NDArray[np.float64]

FIT_ORDER: Final = 4
"""Butterworth order of the ENG-band filter the fit reads - the spike consumer's
(``pipeline_params.filterOrder``), so the fit sees the band the consumer sees."""

SETTLE_S: Final = 0.010
"""Samples this close to a NaN run are kept out of the fit: the measured 1%-of-peak
impulse response of the 300-3000 Hz order-4 design is 5.1 ms, and the consumer pads
10 ms (``pipeline_params.edgeBufferMs``)."""


@dataclass(frozen=True, slots=True)
class LeakFit:
    """One cuff's leak fit: ``T_corrected = T - k * reference``.

    ``rms_before_uv`` / ``rms_after_uv`` are the ENG-band RMS of ``T`` over the fitted
    samples before and after subtraction; their gap is what the correction removed.
    """

    cuff: str
    k: float
    reference: tuple[str, ...]
    n_fit: int
    rms_before_uv: float
    rms_after_uv: float


def _eng(x: F64, fs: float) -> F64:
    lo, hi = BANDS[ENG_BAND].lo_hz, BANDS[ENG_BAND].hi_hz
    sos = butter(FIT_ORDER, (lo, min(hi, fs / 2 - 1)), btype="bandpass", fs=fs, output="sos")
    bad = ~np.isfinite(x)
    if bad.all():
        return np.full_like(x, np.nan)
    if bad.any():  # temporary, for filtering only, and reverted below (invariant 8)
        idx = np.arange(x.size)
        x = np.interp(idx, idx[~bad], x[~bad])
    y = np.asarray(sosfiltfilt(sos, x), dtype=np.float64)
    y[bad] = np.nan
    return y


def _cuff_columns(rec: Recording, cuff: str) -> dict[int, int]:
    cols = {c.contact_index: c.index for c in rec.channels
            if c.cuff_id == cuff and c.contact_index is not None}
    if set(cols) != {1, 2, 3}:
        msg = f"cuff {cuff!r} needs contacts 1-3 to form a tripole, has {sorted(cols)}"
        raise ValueError(msg)
    return cols


def outside_reference(rec: Recording, cuff: str) -> tuple[F64, tuple[str, ...]]:
    """Return the mean of every raw channel outside ``cuff``, and their names.

    Every channel not on this cuff - the other cuff's contacts and the stomach
    contacts - so the reference carries the shared ground's signal and none of this
    nerve's. NaN where every outside channel is NaN. Raises when there is none.
    """
    _cuff_columns(rec, cuff)
    outside = [c for c in rec.channels if c.cuff_id != cuff]
    if not outside:
        msg = f"no channel outside cuff {cuff!r}: a common-mode reference needs one"
        raise ValueError(msg)
    cols = np.column_stack([np.asarray(rec.data[:, c.index], dtype=np.float64) for c in outside])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are NaN, as meant
        ref = np.nanmean(cols, axis=1)
    return np.asarray(ref, dtype=np.float64), tuple(c.name for c in outside)


def _tripole(rec: Recording, cuff: str) -> F64:
    cols = _cuff_columns(rec, cuff)
    v = {i: np.asarray(rec.data[:, cols[i]], dtype=np.float64) for i in (1, 2, 3)}
    a, b = NAIVE_WEIGHTS
    return tripole(v[1], v[2], v[3], a, b)


def fit_leak(t: F64, ref: F64, fs: float) -> tuple[float, int, float, float]:
    """``(k, n_fit, rms_before, rms_after)``: least squares of ``T`` on ``ref``, ENG band.

    Only samples where both are finite and at least :data:`SETTLE_S` from any NaN
    enter the fit. Raises when none do.
    """
    te, re = _eng(np.asarray(t, np.float64), fs), _eng(np.asarray(ref, np.float64), fs)
    ok = np.isfinite(te) & np.isfinite(re)
    pad = int(round(SETTLE_S * fs))
    if pad and not ok.all():
        near_bad = np.convolve((~ok).astype(np.int64), np.ones(2 * pad + 1, np.int64), "same") > 0
        ok &= ~near_bad
    if not ok.any():
        msg = "no finite sample to fit the leak on"
        raise ValueError(msg)
    x, y = re[ok], te[ok]
    den = float(x @ x)
    k = float(x @ y / den) if den > 0 else 0.0
    before, after = float(np.sqrt(np.mean(y**2))), float(np.sqrt(np.mean((y - k * x) ** 2)))
    return k, int(ok.sum()), before, after


def subtract_common_mode(rec: Recording, cuff: str) -> tuple[F64, LeakFit]:
    """Return ``cuff``'s corrected tripole (broadband, uV) and its :class:`LeakFit`.

    ``T`` is formed with the applied weights (:data:`NAIVE_WEIGHTS`), exactly as the
    derivations form it. NaN wherever ``T`` or the reference is NaN.
    """
    t = _tripole(rec, cuff)
    ref, names = outside_reference(rec, cuff)
    k, n, before, after = fit_leak(t, ref, float(rec.fs))
    out = np.asarray(t - k * ref, dtype=np.float64)
    out[~(np.isfinite(t) & np.isfinite(ref))] = np.nan
    return out, LeakFit(cuff, k, names, n, before, after)


def differential_survival(
    rec: Recording, cuff: str, at_s: float, amp_uv: float, *, fwhm_s: float = 0.0006
) -> float:
    """Inject a spike on ``cuff``'s middle contact only and return how much survives.

    A spike on one contact is differential - no other channel sees it - so the
    correction must leave it: the returned value is the injected spike's ENG-band
    peak in the corrected ``T`` (with injection minus without) over the same in the
    uncorrected ``T``. The route's verification requires it within 5% of 1.
    """
    cols = _cuff_columns(rec, cuff)
    fs = float(rec.fs)
    n = rec.data.shape[0]
    tt = np.arange(n, dtype=np.float64) / fs
    width = fwhm_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spike = amp_uv * np.exp(-0.5 * ((tt - at_s) / width) ** 2)
    data = np.array(rec.data, dtype=np.float64)
    data[:, cols[2]] += spike
    injected = Recording(fs=rec.fs, data=data, channels=rec.channels, animal=rec.animal,
                         session=rec.session, path=rec.path)
    base, _fit0 = subtract_common_mode(rec, cuff)
    with_spike, _fit1 = subtract_common_mode(injected, cuff)
    raw = _eng(_tripole(injected, cuff) - _tripole(rec, cuff), fs)
    corr = _eng(with_spike - base, fs)
    i0, i1 = int((at_s - 0.005) * fs), int((at_s + 0.005) * fs)
    return float(np.nanmax(np.abs(corr[i0:i1])) / np.nanmax(np.abs(raw[i0:i1])))
