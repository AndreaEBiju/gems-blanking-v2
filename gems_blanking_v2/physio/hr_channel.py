"""``best_hr_channel`` for the new cohort (ruling 2026-09-29, item 6).

``CONSUMERS`` names ``best_hr_channel`` as the signal ``hrv`` and ``breathing`` read,
and the old cohort took it from Andrea's manual ``hrChanIdx``. The new cohort has
none, so it is defined here by task 05's ranking - template SNR with the
``implausible_frac`` and ``rescue_rate`` vetoes (:func:`rank_hr_channels`) - over
every signal an HR consumer could read:

- the nine raw channels, which carry the shared-ground common mode in full;
- each cuff's tripole ``T`` (the applied weights), which cancels it to the gain
  mismatch;
- ``stomach_ref``, the stomach common average.

Nothing is excluded in advance: the ranking is self-policing (a channel whose
"beats" are artifacts averages to a flat template), and whether the chosen channel
is harmed by the common mode is decided afterwards by the operational test - the
beat train with and without the events - not by where the channel sits. Re-picked
per recording, as task 05 requires.

OUTSIDE THE GENERATION HASH: the detection chain detects beats on ``R_T``
(``chain.BEAT_SIGNAL``) and does not import this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import butter, decimate, detrend, find_peaks, sosfiltfilt

from gems_blanking_v2.derive.cm_events import Events
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.physio.rpeaks import (
    DETECT_BAND_HZ,
    GLOBAL_RR_FRACTION,
    PROVISIONAL_MAX_IMPLAUSIBLE_FRAC,
    PROVISIONAL_MAX_RESCUE_RATE,
    RR_PLAUSIBLE_RANGE_S,
    BeatTrain,
    _prepare,
    _template_snr,
    detect_rpeaks,
    rank_hr_channels,
)
from gems_blanking_v2.types import ChannelInfo, Recording

__all__ = [
    "AC_BPM",
    "CARRIES_QRS",
    "DERIVED_CANDIDATES",
    "HUM_LOCK_BPM",
    "HUM_LOCK_MINUTES",
    "MAINS_HZ",
    "MAX_INJECTIONS",
    "PARTIAL_MIN_S",
    "PROVISIONAL_AC_MIN_PEAK",
    "PROVISIONAL_MAX_COUNT_DEV",
    "PROVISIONAL_MAX_TRANSIENT_HARM",
    "CountGate",
    "TrainGate",
    "autocorr_rate",
    "autocorr_windows",
    "best_hr_channel",
    "blank_spans_s",
    "cm_gains",
    "count_gate",
    "findpeaks_replica",
    "gated_selection",
    "harm_is_settled",
    "hr_candidates",
    "hum_grid",
    "hum_locked_persistent",
    "implausible_fraction",
    "mains_harmonics",
    "minute_gapped",
    "minute_validity",
    "peri_r_train",
    "pick_train",
    "rule2_winner",
    "transient_harm",
]

F64 = npt.NDArray[np.float64]

DERIVED_CANDIDATES: tuple[str, ...] = ("L_T", "R_T", "stomach_ref")
"""The derived signals ranked beside the raw channels, when the recording has them."""


def hr_candidates(rec: Recording) -> Recording:
    """Return ``rec`` with the derived candidates appended as extra columns.

    The raw columns are unchanged and keep their indices; each derived signal that
    this recording can form (:data:`DERIVED_CANDIDATES`) is appended with role
    ``"aux"``. A copy: the input is never written to.
    """
    signals, _weights = build_derivations(rec)
    extra = [(n, signals[n]) for n in DERIVED_CANDIDATES if n in signals]
    clash = {n for n, _ in extra} & {c.name for c in rec.channels}
    if clash:
        msg = f"derived candidate names collide with raw channels: {sorted(clash)}"
        raise ValueError(msg)
    base = np.asarray(rec.data, dtype=np.float64)
    data = np.column_stack([base, *[np.asarray(x, dtype=np.float64) for _n, x in extra]])
    n0 = base.shape[1]
    channels = list(rec.channels) + [
        ChannelInfo(n0 + k, name, "aux", None, None, None, rec.channels[0].config)
        for k, (name, _x) in enumerate(extra)]
    return replace(rec, data=data, channels=channels)


PROVISIONAL_MAX_TRANSIENT_HARM: Final = 0.01
"""PROVISIONAL (ruling 2026-09-30, item 6): a candidate is vetoed when more than this
fraction of :data:`N_INJECTIONS` injected common-mode transients cost a beat or move a
fiducial by more than :data:`FIDUCIAL_SHIFT_S`. Measured with the HR consumer's own
detector (2026-09-30): 200 transients cost 4-135 beats on a raw contact, 0-2 on T."""

N_INJECTIONS: Final = 200
TRANSIENT_FWHM_S: Final = 0.0012
FIDUCIAL_SHIFT_S: Final = 0.002
HARM_WINDOW_S: Final = 0.1
NONCARDIAC_S: Final = 0.020
"""Events this far from every beat are non-cardiac, and only their amplitudes are
injected: part of the common-mode class is the heartbeat's own sharp edge."""


def cm_gains(events: Events, keep: npt.ArrayLike | None = None) -> dict[str, float]:
    """Return each channel's common-mode gain from the events (``keep``: which ones).

    The median over the kept events of the channel's |amplitude| over that event's
    median |amplitude| across channels. Pass the non-cardiac selection: the heart is a
    second common-mode source with its own gain pattern, and the heartbeat's sharp
    edge is part of the event class.
    """
    a = np.abs(np.asarray(events.channel_amp_uv, dtype=np.float64))
    if keep is not None:
        a = a[np.asarray(keep, dtype=bool)]
    if a.size == 0:
        return dict.fromkeys(events.channels, 1.0)
    rel = a / np.median(a, axis=1, keepdims=True)
    return {name: float(np.median(rel[:, k])) for k, name in enumerate(events.channels)}


def _trains(cand: Recording) -> dict[str, BeatTrain]:
    return {c.name: detect_rpeaks(np.asarray(cand.data[:, c.index], dtype=np.float64),
                                  float(cand.fs)) for c in cand.channels}


def _noncardiac(events: Events, beats_s: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Which events are more than :data:`NONCARDIAC_S` from every beat."""
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    t_ev = np.asarray(events.t_s, dtype=np.float64)
    if b.size == 0 or t_ev.size == 0:
        return np.ones(t_ev.size, dtype=bool)
    i = np.searchsorted(b, t_ev)
    dmin = np.minimum(np.abs(t_ev - b[np.clip(i - 1, 0, b.size - 1)]),
                      np.abs(b[np.clip(i, 0, b.size - 1)] - t_ev))
    return np.asarray(dmin > NONCARDIAC_S, dtype=bool)


def _harmed(b0: F64, b1: F64, times: F64) -> int:
    """How many injections have a beat within the window that is lost or moved."""
    b1 = np.sort(b1)
    count = 0
    for t0 in times:
        near = b0[np.abs(b0 - t0) <= HARM_WINDOW_S]
        if near.size == 0:
            continue
        if b1.size == 0:
            count += 1
            continue
        j = np.searchsorted(b1, near)
        d = np.minimum(np.abs(near - b1[np.clip(j - 1, 0, b1.size - 1)]),
                       np.abs(b1[np.clip(j, 0, b1.size - 1)] - near))
        count += int(np.any(d > FIDUCIAL_SHIFT_S))
    return count


def transient_harm(
    rec: Recording, events: Events, beats_s: npt.ArrayLike, *, seed: int = 0,
    before: dict[str, BeatTrain] | None = None,
) -> dict[str, float]:
    """Return ``{candidate: fraction of injections that cost a beat or moved a fiducial}``.

    :data:`N_INJECTIONS` Gaussian transients of :data:`TRANSIENT_FWHM_S` at uniform
    random times, amplitudes drawn from the recording's own non-cardiac events
    (common-mode amplitude, more than :data:`NONCARDIAC_S` from every beat in
    ``beats_s``), added to every raw channel times its :func:`cm_gains` gain over
    those same non-cardiac events - so a
    tripole or ``stomach_ref`` carries exactly the leak its contacts' gains produce.
    An injection harms a candidate when a beat within :data:`HARM_WINDOW_S` of it has
    no beat within :data:`FIDUCIAL_SHIFT_S` after injection (lost, or moved further).
    """
    injected, times = _inject(rec, events, _noncardiac(events, beats_s), seed)
    before = before if before is not None else _trains(hr_candidates(rec))
    after = _trains(hr_candidates(injected))
    return {name: _harmed(np.asarray(tr.t_s), np.asarray(after[name].t_s), times) / N_INJECTIONS
            for name, tr in before.items()}


def best_hr_channel(
    rec: Recording, *, events: Events | None = None, seed: int = 0
) -> tuple[str, pd.DataFrame]:
    """Return ``(name, table)``: task 05's pick over :func:`hr_candidates`.

    With ``events`` (ruling 2026-09-30, item 6), candidates whose
    :func:`transient_harm` exceeds :data:`PROVISIONAL_MAX_TRANSIENT_HARM` are vetoed
    first and template SNR ranks only the survivors - SNR cannot see this harm,
    because the far-field ECG is common mode too. The beats that decide which events
    are non-cardiac come from the channel SNR ranks first (or, when task 05 vetoes
    every channel, the one with the most beats). Raises ``ValueError`` when every
    candidate is vetoed - never a silent least-bad channel.
    """
    cand = hr_candidates(rec)
    trains = _trains(cand)
    if events is None:
        return rank_hr_channels(cand, trains)
    try:
        first, _t = rank_hr_channels(cand, trains)
    except ValueError:
        first = max(trains, key=lambda k: trains[k].n_beats)
    harm = transient_harm(rec, events, trains[first].t_s, seed=seed, before=trains)
    survivors = {k: v for k, v in trains.items() if harm[k] <= PROVISIONAL_MAX_TRANSIENT_HARM}
    if not survivors:
        msg = ("every channel was vetoed for HRV by the transient test: "
               + ", ".join(f"{k}({v:.3f})" for k, v in sorted(harm.items(), key=lambda kv: kv[1])))
        raise ValueError(msg)
    best, table = rank_hr_channels(cand, survivors)
    table["transient_harm"] = table["channel"].map(harm)
    vetoed = pd.DataFrame([{"channel": k, "transient_harm": v, "eligible": False,
                            "vetoed_by": "transient_harm"}
                           for k, v in harm.items() if k not in survivors])
    return best, pd.concat([table, vetoed], ignore_index=True)


# ---------------------------------------------------------------------------
# the independent count check and the two-candidate selection (ruling 2026-09-30 (c) 4)
# ---------------------------------------------------------------------------

PROVISIONAL_MAX_COUNT_DEV: Final = 0.05
"""PROVISIONAL: a train's beats per minute must be within this fraction of the
autocorrelation rate in at least :data:`COUNT_PASS_FRACTION` of the clear minutes."""

COUNT_PASS_FRACTION: Final = 0.95

AC_BPM: Final[tuple[float, float]] = (250.0, 550.0)
"""Rat heart rates only: the autocorrelation is searched over these lags, which keeps
the octave errors (every other beat, twice per beat) out of range."""

PROVISIONAL_AC_MIN_PEAK: Final = 0.2
"""PROVISIONAL: a minute's autocorrelation is clear when its normalised peak in the
searched range reaches this and is an interior local maximum."""

AC_MIN_CLEAR_FRACTION: Final = 0.5
"""Fewer clear minutes than this fraction: the recording is unassessable and nothing
is stored."""

AC_WINDOW_S: Final = 60.0
AC_BAND_HZ: Final[tuple[float, float]] = (10.0, 150.0)
_AC_TARGET_HZ: Final = 1000.0
_MAX_DECIMATE_STAGE: Final = 13
FINDPEAKS_DISTANCE_S: Final = 0.100
FINDPEAKS_EDGE_S: Final = 0.75
DETECTORS: Final[tuple[str, str]] = ("task05", "findpeaks")


MAINS_HZ: Final = 60.0
"""Ruling 2026-10-02 (d)/(e): every autocorrelation input - the count gate's and the
cross-check references' - is notched at this and its harmonics up to the band's upper edge
(:func:`mains_harmonics`) before rectification, or rectified mains repeats every 1/120 s and
the autocorrelation locks onto lags m / 120 s (7200 / m bpm)."""
HUM_LOCK_BPM: Final = 0.3
HUM_LOCK_MINUTES: Final = 3
"""Ruling 2026-10-02 (e) 2: a minute is hum-locked only when its rate sits within
:data:`HUM_LOCK_BPM` of the SAME 7200 / m value for at least this many consecutive clear
minutes - a mains lock holds to +/-0.2 bpm for minutes; a heart rarely stays within 0.6 bpm
for three. Hum-locked minutes are not clear."""


def mains_harmonics() -> tuple[float, ...]:
    """Return :data:`MAINS_HZ` and its harmonics up to :data:`AC_BAND_HZ`'s upper edge."""
    return tuple(MAINS_HZ * k for k in range(1, int(AC_BAND_HZ[1] // MAINS_HZ) + 1))


def hum_grid(bpm: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Per rate, the m of the 7200 / m value it sits within :data:`HUM_LOCK_BPM` of; 0 if none."""
    b = np.asarray(bpm, dtype=np.float64)
    out = np.zeros(b.shape, dtype=np.int64)
    ok = np.isfinite(b) & (b > 0)
    m = np.maximum(np.round(7200.0 / b[ok]), 1).astype(np.int64)
    hit = np.abs(b[ok] - 7200.0 / m) <= HUM_LOCK_BPM
    out[np.flatnonzero(ok)[hit]] = m[hit]
    return out


def hum_locked_persistent(bpm: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Return, per minute, whether it sits in a persistent mains lock.

    A lock is a run of >= :data:`HUM_LOCK_MINUTES` consecutive clear minutes all on the same
    7200 / m value (:func:`hum_grid`); an unclear minute breaks a run.
    """
    g = hum_grid(bpm)
    out = np.zeros(g.shape, dtype=bool)
    i = 0
    while i < g.size:
        if g[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < g.size and g[j + 1] == g[i]:
            j += 1
        if j - i + 1 >= HUM_LOCK_MINUTES:
            out[i:j + 1] = True
        i = j + 1
    return out


def _decimate_to(x: F64, fs: float, target_hz: float) -> tuple[F64, float]:
    """Decimate (FIR, zero phase) in stages of at most 13 toward ``target_hz``."""
    y = np.asarray(x, dtype=np.float64)
    q = max(int(fs // target_hz), 1)
    while q > 1:
        stage = next((s for s in range(min(q, _MAX_DECIMATE_STAGE), 1, -1) if q % s == 0),
                     min(q, _MAX_DECIMATE_STAGE))
        y = np.asarray(decimate(y, stage, ftype="fir", zero_phase=True), dtype=np.float64)
        fs /= stage
        q //= stage
    return y, fs


PARTIAL_MIN_S: Final = 30.0
"""Ruling 2026-10-02 (g) 3: a trailing partial window at least this long is assessed with
the same tests (the count scaled to its duration); a shorter one is never assessed, and its
time is a tagged gap."""


def _ac_peak(seg: F64, fd: float, lag_lo: int, lag_hi: int) -> tuple[float, float]:
    """``(rate, parabola-refined rate)`` bpm of one window's autocorrelation; NaN if unclear."""
    w = seg.size
    seg = seg - seg.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(seg, n=2 * w)) ** 2)[:w]
    rate = refined = float("nan")
    if ac[0] > 0 and lag_hi + 1 < w:
        ac = ac / ac[0]
        k = lag_lo + int(np.argmax(ac[lag_lo:lag_hi + 1]))
        if (lag_lo < k < lag_hi and ac[k] >= PROVISIONAL_AC_MIN_PEAK
                and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]):
            rate = 60.0 * fd / k
            den = ac[k - 1] - 2.0 * ac[k] + ac[k + 1]
            shift = 0.5 * (ac[k - 1] - ac[k + 1]) / den if den < 0 else 0.0
            refined = 60.0 * fd / (k + shift)
    return rate, refined


def _ac_windows(x: npt.ArrayLike, fs: float, notch_hz: tuple[float, ...],
                partial: bool) -> tuple[F64, F64, F64, F64]:
    """Return ``(starts s, durations s, rate, refined rate)`` per window - the one site.

    Decimate; notch at ``notch_hz`` (none if empty); 10-150 Hz band-pass; rectify. Windows
    are the full :data:`AC_WINDOW_S` minutes from 0 and, with ``partial``, the trailing
    remainder when it is at least :data:`PARTIAL_MIN_S` long. No hum lock is applied here.
    """
    from gems_blanking_v2.derive.stomach_alt import notch  # noqa: PLC0415

    xa = np.asarray(x, dtype=np.float64)
    y, fd = _decimate_to(xa, fs, _AC_TARGET_HZ)
    if notch_hz:
        y = notch(y, fd, notch_hz)
    sos = butter(4, AC_BAND_HZ, btype="bandpass", fs=fd, output="sos")
    r = np.abs(sosfiltfilt(sos, y))
    w = int(round(AC_WINDOW_S * fd))
    lag_lo = int(np.floor(60.0 / AC_BPM[1] * fd))
    lag_hi = int(np.ceil(60.0 / AC_BPM[0] * fd))
    spans: list[tuple[int, int, float, float]] = []
    for m in range(int(r.size / fd // AC_WINDOW_S)):
        a = int(round(m * AC_WINDOW_S * fd))
        if a + w > r.size:
            break
        spans.append((a, a + w, m * AC_WINDOW_S, AC_WINDOW_S))
    if partial:
        start = len(spans) * AC_WINDOW_S
        a = int(round(start * fd))
        dur = xa.size / fs - start
        if dur >= PARTIAL_MIN_S and a < r.size:
            spans.append((a, r.size, start, dur))
    rates = [_ac_peak(r[a:b], fd, lag_lo, lag_hi) for a, b, _s, _d in spans]
    return (np.asarray([sp[2] for sp in spans], dtype=np.float64),
            np.asarray([sp[3] for sp in spans], dtype=np.float64),
            np.asarray([q[0] for q in rates], dtype=np.float64),
            np.asarray([q[1] for q in rates], dtype=np.float64))


def _locked(refined: F64, durs: F64) -> npt.NDArray[np.bool_]:
    """Return the persistent lock (:func:`hum_locked_persistent`) per window.

    Full minutes are judged among themselves, exactly as without a partial window; a
    trailing partial window joins the run test as one more window and cannot change a full
    minute's decision.
    """
    full = durs == AC_WINDOW_S
    n_full = int(full.sum())
    out = np.zeros(refined.size, dtype=bool)
    out[:n_full] = hum_locked_persistent(refined[:n_full])
    if refined.size > n_full:
        out[n_full:] = hum_locked_persistent(refined)[n_full:]
    return out


def autocorr_rate(x: npt.ArrayLike, fs: float) -> tuple[F64, F64]:
    """Return ``(minute start times s, bpm)`` from the rectified 10-150 Hz autocorrelation.

    No peak picking: per :data:`AC_WINDOW_S` window, the normalised autocorrelation of
    ``|10-150 Hz|`` is searched over the :data:`AC_BPM` lags, and the peak's lag
    gives the rate. The nearest lag (~1 ms steps) is within 0.5% at 550 bpm, a tenth
    of :data:`PROVISIONAL_MAX_COUNT_DEV`, so there is no sub-lag refinement. ``bpm``
    is NaN where the minute is not clear (:data:`PROVISIONAL_AC_MIN_PEAK`, interior
    local maximum) or hum-locked (:func:`hum_locked_persistent` on the parabola-refined
    peak - an integer lag sits up to ~1.5 bpm off 7200 / m even when locked, so the lock
    test cannot read the unrefined rate; the returned rate stays unrefined). A trailing
    partial minute is not assessed (the count gate's grid; :func:`autocorr_windows` adds it).

    Ruling 2026-10-02 (e) 1: the input is notched at :func:`mains_harmonics` after
    decimation, before the band-pass and rectification - for every source.
    """
    starts, durs, rate, fine = _ac_windows(x, fs, mains_harmonics(), partial=False)
    return starts, np.where(_locked(fine, durs), np.nan, rate)


def autocorr_windows(x: npt.ArrayLike, fs: float) -> tuple[F64, F64, F64]:
    """Return ``(starts s, durations s, bpm)``: :func:`autocorr_rate` plus the trailing window.

    Ruling 2026-10-02 (g) 3: the full minutes are exactly :func:`autocorr_rate`'s; a trailing
    remainder of at least :data:`PARTIAL_MIN_S` is one more window, judged by the same
    clarity and lock tests. Per-minute validity reads this; the count gate reads only the
    full minutes.
    """
    starts, durs, rate, fine = _ac_windows(x, fs, mains_harmonics(), partial=True)
    return starts, durs, np.where(_locked(fine, durs), np.nan, rate)


@dataclass(frozen=True)
class CountGate:
    """One train's beats per minute against the autocorrelation rate."""

    minutes: int
    clear_minutes: int
    fraction_within: float
    """Of the clear minutes, the fraction within :data:`PROVISIONAL_MAX_COUNT_DEV`."""
    median_abs_dev: float
    """Median over clear minutes of ``|count / rate - 1|``; the tie-break. NaN if none."""
    assessable: bool
    passes: bool


def count_gate(beats_s: npt.ArrayLike, starts: F64, bpm: F64) -> CountGate:
    """Return the count check of one train against :func:`autocorr_rate`'s output.

    Passes when beats per minute are within :data:`PROVISIONAL_MAX_COUNT_DEV` of the
    autocorrelation rate in at least :data:`COUNT_PASS_FRACTION` of the clear
    minutes, and at least :data:`AC_MIN_CLEAR_FRACTION` of the minutes are clear
    (otherwise unassessable, which never passes).
    """
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    clear = np.isfinite(bpm)
    n_clear = int(clear.sum())
    assessable = bool(starts.size > 0 and n_clear >= AC_MIN_CLEAR_FRACTION * starts.size)
    per_min = np.array([np.count_nonzero((b >= s) & (b < s + AC_WINDOW_S))
                        for s in starts[clear]], dtype=np.float64) * 60.0 / AC_WINDOW_S
    dev = np.abs(per_min / bpm[clear] - 1.0)
    # |count - rate| <= 5% of rate, in the form that has no rounding at the boundary
    within = np.abs(per_min - bpm[clear]) <= PROVISIONAL_MAX_COUNT_DEV * bpm[clear]
    frac = float(np.mean(within)) if n_clear else 0.0
    return CountGate(minutes=int(starts.size), clear_minutes=n_clear, fraction_within=frac,
                     median_abs_dev=float(np.median(dev)) if n_clear else float("nan"),
                     assessable=assessable,
                     passes=bool(assessable and frac >= COUNT_PASS_FRACTION))


def minute_validity(beats_s: npt.ArrayLike, starts: F64, bpm: F64,
                    bad_minutes_s: npt.ArrayLike = (),
                    durs: npt.ArrayLike | None = None) -> npt.NDArray[np.bool_]:
    """Per minute, whether it is valid (ruling 2026-10-02 (f) 2).

    Valid: clear (``bpm`` finite - :func:`autocorr_rate` already NaNs a hum-locked minute),
    beats in ``[start, start + duration)`` within :data:`PROVISIONAL_MAX_COUNT_DEV` of
    the minute's rate (:func:`count_gate`'s own test, the count scaled to the window's
    duration - ruling (g) 3), and the start not in ``bad_minutes_s`` (a pair's assessable
    minutes where cross-check (a) disagrees). ``durs``: per window; :data:`AC_WINDOW_S`
    when omitted.
    """
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    d = (np.full(starts.size, AC_WINDOW_S) if durs is None
         else np.asarray(durs, dtype=np.float64))
    per_min = np.array([np.count_nonzero((b >= s) & (b < s + w))
                        for s, w in zip(starts, d, strict=True)], dtype=np.float64) * 60.0 / d
    with np.errstate(invalid="ignore"):
        within = np.abs(per_min - bpm) <= PROVISIONAL_MAX_COUNT_DEV * bpm
    return np.asarray(np.isfinite(bpm) & within & ~_listed(starts, bad_minutes_s))


def _listed(starts: F64, minutes_s: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Per start, whether it is one of ``minutes_s`` (one minute grid, so 1 us is exact)."""
    bad = np.asarray(minutes_s, dtype=np.float64)
    if bad.size == 0:
        return np.zeros(starts.size, dtype=bool)
    return np.asarray(np.isclose(starts[:, None], bad[None, :], rtol=0.0, atol=1e-6).any(axis=1))


def minute_gapped(row: TrainGate) -> TrainGate:
    """Return ``row`` with every beat outside its valid minutes removed, as tagged gaps.

    Ruling 2026-10-02 (f) 2: wrong beats are never left in place to look valid to
    Andrea's window rule. A beat is kept when it lies in a valid window of
    ``row.minute_s`` (a trailing remainder shorter than :data:`PARTIAL_MIN_S` is never
    assessed, so never kept - ruling (g) 3). Gaps:
    between two kept beats with an invalid minute between them, and task 05's own gaps
    whose first beat is kept; each is ``(beat before, beat after)``, so :attr:`gap_after`
    marks the beat before.
    """
    b = np.sort(row.beats_s)
    starts = np.asarray(row.minute_s, dtype=np.float64)
    valid = np.asarray(row.valid, dtype=bool)
    if starts.size == 0:
        return replace(row, beats_s=b[:0], gaps_s=())
    keep = _in_window(starts, _durations(row), b, valid)
    m = np.searchsorted(starts, b, side="right") - 1
    kb, km = b[keep], m[keep]
    old = np.asarray([g[0] for g in row.gaps_s], dtype=np.float64)
    gaps = []
    for i in range(kb.size - 1):
        jump = km[i + 1] - km[i] > 1
        tagged = old.size > 0 and bool(np.isclose(kb[i], old, rtol=0.0, atol=1e-9).any())
        if jump or tagged:
            gaps.append((float(kb[i]), float(kb[i + 1])))
    return replace(row, beats_s=kb, gaps_s=tuple(gaps))


def _durations(row: TrainGate) -> F64:
    """``row``'s window durations; every window :data:`AC_WINDOW_S` when none are recorded."""
    if len(row.minute_dur_s) == len(row.minute_s):
        return np.asarray(row.minute_dur_s, dtype=np.float64)
    return np.full(len(row.minute_s), AC_WINDOW_S)


def _in_window(starts: F64, durs: F64, t: F64,
               flag: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Per time, whether it lies in a window ``[start, start + duration)`` flagged in ``flag``."""
    if starts.size == 0:
        return np.zeros(t.size, dtype=bool)
    m = np.searchsorted(starts, t, side="right") - 1
    mc = np.clip(m, 0, None)
    return np.asarray((m >= 0) & (t < starts[mc] + durs[mc]) & flag[mc])


def blank_spans_s(row: TrainGate, dur_s: float) -> tuple[tuple[float, float], ...]:
    """Return the spans of ``[0, dur_s)`` outside ``row``'s valid windows, merged, in s.

    Ruling 2026-10-02 (g) 4: what Andrea's ``BlankSpans`` blanks - every rejected window and
    any time no window assesses (a trailing remainder shorter than :data:`PARTIAL_MIN_S`).
    Half-open ``[a, b)``, on the train's timeline.
    """
    starts = np.asarray(row.minute_s, dtype=np.float64)
    durs = _durations(row)
    valid = np.asarray(row.valid, dtype=bool)
    good = sorted((float(a), float(a + d))
                  for a, d, v in zip(starts, durs, valid, strict=True) if v)
    out: list[tuple[float, float]] = []
    t = 0.0
    for a, b in good:
        if a > t:
            out.append((t, min(a, dur_s)))
        t = max(t, b)
    if t < dur_s:
        out.append((t, dur_s))
    return tuple((a, b) for a, b in out if b > a)


def findpeaks_replica(x: npt.ArrayLike, fs: float) -> F64:
    """Return the beats (s) of Andrea's ``findpeaks`` in ``HR_BR_HRVAnalysis_new``.

    10-150 Hz order-4 Butterworth, zero phase; linear detrend; peaks at least
    :data:`FINDPEAKS_DISTANCE_S` apart with no height floor; :data:`FINDPEAKS_EDGE_S`
    excluded at each end. Matched her MATLAB beats on 99.7-100% of beats on nine
    recordings (2026-09-30).
    """
    sos = butter(4, (AC_BAND_HZ[0], min(AC_BAND_HZ[1], fs / 2 - 1)), btype="bandpass",
                 fs=fs, output="sos")
    y = detrend(sosfiltfilt(sos, np.asarray(x, dtype=np.float64)))
    pk, _ = find_peaks(y, distance=round(FINDPEAKS_DISTANCE_S * fs))
    edge = round(FINDPEAKS_EDGE_S * fs)
    return np.asarray(pk[(pk >= edge) & (pk < y.size - edge)] / fs, dtype=np.float64)


def implausible_fraction(beats_s: npt.ArrayLike) -> float:
    """Return task 05's ``implausible_frac`` rule applied to any train.

    The fraction of intervals under :data:`GLOBAL_RR_FRACTION` of the median of the
    intervals inside :data:`RR_PLAUSIBLE_RANGE_S`; 1.0 when no interval is plausible.
    """
    rr = np.diff(np.sort(np.asarray(beats_s, dtype=np.float64)))
    lo, hi = RR_PLAUSIBLE_RANGE_S
    ok = rr[(rr >= lo) & (rr <= hi)]
    if ok.size == 0:
        return 1.0
    return float(np.mean(rr < GLOBAL_RR_FRACTION * float(np.median(ok))))


@dataclass(frozen=True)
class TrainGate:
    """One candidate train on one channel, with every gate it was put through."""

    channel: str
    detector: str
    beats_s: F64 = field(repr=False)
    snr: float
    count: CountGate
    transient_harm: float
    implausible_frac: float
    rescue_rate: float
    plausible: bool
    passes: bool
    source: str = "channel"
    """``"channel"`` - a train on one :func:`hr_candidates` signal; ``"pair"`` - the
    cross-site lead (``hr_pairs``, adopted by ruling 2026-10-02 (c)), ``channel`` naming it
    as ``"<plus>-<minus>"``."""
    harm_median: float = float("nan")
    """The median-pattern injection's harm - reported, never binding."""
    cross_check: str = ""
    """Pair rows only: ``"pass"``, or which of (a) rate / (c) morphology failed; empty when
    the gates already failed and the cross-check was not run."""
    gaps_s: tuple[tuple[float, float], ...] = ()
    """Task 05's unrecovered gaps ``(beat before, beat after)``; findpeaks reports none."""
    note: str = ""
    """Why this row was stored, when the choice was not the plain best-SNR one."""
    n_injections: int = N_INJECTIONS
    """How many real-pattern injections ``transient_harm`` rests on (ruling 2026-10-02 (e) 3)."""
    rate_ok: bool | None = None
    """Pair rows: cross-check (a) over the recording; None where it was not run."""
    morphology_ok: bool | None = None
    """Pair rows: cross-check (c); None where it was not run."""
    eligible: bool = False
    """Ruling 2026-10-02 (f) 2: passes the recording-level gates - the transient veto with
    precision, task 05 plausibility and, for a pair, morphology (c). Set only by
    ``gated_selection(per_minute=True)``; False otherwise."""
    minute_s: tuple[float, ...] = ()
    """Start times of the windows :func:`autocorr_windows` assessed: the count gate's minutes
    plus, when long enough, the trailing partial window (ruling (g) 3)."""
    minute_dur_s: tuple[float, ...] = ()
    """Each window's duration; :data:`AC_WINDOW_S` for every full minute."""
    valid: tuple[bool, ...] = ()
    """Per minute of ``minute_s``, :func:`minute_validity` - for a pair, with (a)'s failing
    minutes removed once the cross-check has run."""

    @property
    def n_valid(self) -> int:
        """How many minutes are valid."""
        return int(sum(self.valid))

    @property
    def longest_valid_run(self) -> int:
        """The longest run of consecutive valid minutes."""
        best = run = 0
        for v in self.valid:
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    @property
    def gap_after(self) -> npt.NDArray[np.bool_]:
        """Per beat (sorted), whether the interval after it spans a tagged gap."""
        b = np.sort(self.beats_s)
        starts = np.asarray([g[0] for g in self.gaps_s], dtype=np.float64)
        if starts.size == 0:
            return np.zeros(b.size, dtype=bool)
        return np.asarray(np.isclose(b[:, None], starts[None, :], rtol=0.0, atol=1e-9).any(axis=1))


MAX_INJECTIONS: Final = 2000
"""Ruling 2026-10-02 (e) 3: the most injections a veto decision may take."""
VETO_CONFIDENCE: Final = 0.95


def harm_is_settled(hits: int, n: int) -> bool:
    """Whether the one-sided 95% interval of ``hits / n`` excludes the veto threshold.

    At or under the threshold, the Clopper-Pearson one-sided upper bound must be under it;
    over it, the one-sided lower bound must be over it. (0 hits in 200 is NOT settled:
    the upper bound is 0.0149.)
    """
    from scipy.stats import beta  # noqa: PLC0415

    est = hits / n
    if est <= PROVISIONAL_MAX_TRANSIENT_HARM:
        upper = 1.0 if hits >= n else float(beta.ppf(VETO_CONFIDENCE, hits + 1, n - hits))
        return upper < PROVISIONAL_MAX_TRANSIENT_HARM
    lower = 0.0 if hits <= 0 else float(beta.ppf(1.0 - VETO_CONFIDENCE, hits, n - hits + 1))
    return lower > PROVISIONAL_MAX_TRANSIENT_HARM


Block = Callable[[int], tuple[F64, Callable[["TrainGate"], F64]]]
"""``k -> (block k's injection times, each row's injected signal)``."""


def _refine_harm(rows: list[TrainGate], block: Block, fs: float, *,
                 need_count: bool = True) -> list[TrainGate]:
    """Add injection blocks until every veto decision that matters is settled.

    Rows that pass the count gate and plausibility, and whose harm is not settled
    (:func:`harm_is_settled`), get further blocks of :data:`N_INJECTIONS` (``block(k)``
    returns block k's injection times and each row's injected signal) until settled or
    at :data:`MAX_INJECTIONS`; the decision is then the estimate against the threshold.
    Rows failing the count gate or plausibility fail whatever the veto, and keep 200.
    With ``need_count=False`` (per-minute storage, ruling 2026-10-02 (f) 2, where the
    count gate is not a recording-level gate) plausibility alone decides who needs it.
    """
    hits = {i: round(r.transient_harm * r.n_injections) for i, r in enumerate(rows)
            if np.isfinite(r.transient_harm)}
    n = {i: rows[i].n_injections for i in hits}
    need = [i for i in hits if (rows[i].count.passes or not need_count) and rows[i].plausible
            and not harm_is_settled(hits[i], n[i])]
    k = 1
    while need:
        times, signal = block(k)
        for i in need:
            r = rows[i]
            hits[i] += _harmed(r.beats_s, _detect(r.detector, signal(r), fs), times)
            n[i] += N_INJECTIONS
        need = [i for i in need if n[i] < MAX_INJECTIONS and not harm_is_settled(hits[i], n[i])]
        k += 1
    out = list(rows)
    for i, h in hits.items():
        if n[i] != rows[i].n_injections:
            r = rows[i]
            harm = h / n[i]
            out[i] = replace(r, transient_harm=harm, n_injections=n[i],
                             passes=bool(r.count.passes and r.plausible
                                         and harm <= PROVISIONAL_MAX_TRANSIENT_HARM))
    return out


CARRIES_QRS: Final = 0.5
"""Rule 2 of ruling 2026-10-02 (c), tightened by (g) 2: of the two trains' beats unmatched
within 20 ms, train X wins over Y only when X's unmatched beats reach the QRS-correlation
floor in a MAJORITY (> this) and Y's in a minority (< this) - :func:`rule2_winner`.
Otherwise the comparison is ambiguous: an incumbent stays; with none, neither is stored.
Measured on A t05: the lead 0.974, its L_T 0.013."""


def rule2_winner(pair_qrs: float, other_qrs: float) -> str | None:
    """Return ``"pair"`` / ``"other"`` for a clear winner (:data:`CARRIES_QRS`), else None."""
    if pair_qrs > CARRIES_QRS > other_qrs:
        return "pair"
    if other_qrs > CARRIES_QRS > pair_qrs:
        return "other"
    return None


def _inject(rec: Recording, events: Events, keep: npt.NDArray[np.bool_],
            seed: int) -> tuple[Recording, F64]:
    """Return ``(injected copy, injection times s)`` for :func:`transient_harm`."""
    fs = float(rec.fs)
    n = rec.data.shape[0]
    amps = np.asarray(events.cm_amp_uv, dtype=np.float64)[keep]
    if amps.size == 0:
        msg = "no non-cardiac common-mode event to take an amplitude from"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(1.0, n / fs - 1.0, N_INJECTIONS))
    a = rng.choice(amps, N_INJECTIONS)
    g = cm_gains(events, keep)
    width = TRANSIENT_FWHM_S / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    pulse = np.zeros(n)
    half = int(round(10 * width * fs))
    for t0, amp in zip(times, a, strict=True):
        c = int(round(t0 * fs))
        k = np.arange(max(c - half, 0), min(c + half + 1, n))
        pulse[k] += amp * np.exp(-0.5 * ((k / fs - t0) / width) ** 2)
    data = np.array(rec.data, dtype=np.float64)
    for ch in rec.channels:
        data[:, ch.index] += g.get(ch.name, 1.0) * pulse
    return replace(rec, data=data), times


def _detect(det: str, x: F64, fs: float) -> F64:
    if det == "task05":
        return np.asarray(detect_rpeaks(x, fs).t_s, dtype=np.float64)
    return findpeaks_replica(x, fs)


def _gate_row(name: str, det: str, x: F64, x_real: F64 | None, x_med: F64 | None, times: F64,
              fs: float, ac: tuple[F64, F64, F64], t05: BeatTrain | None,
              source: str) -> TrainGate:
    """Gate one train: count, plausibility, and the real-pattern veto (binding).

    ``ac``: :func:`autocorr_windows` - the count gate reads its full minutes only.
    """
    starts_w, durs, bpm_w = ac
    full = durs == AC_WINDOW_S
    starts, bpm = starts_w[full], bpm_w[full]
    if det == "task05":
        tr = t05 if t05 is not None else detect_rpeaks(x, fs)
        beats = np.asarray(tr.t_s, dtype=np.float64)
        implausible, rescue = tr.implausible_frac, tr.rescue_rate
        gaps = tuple((float(a), float(b)) for a, b, _m in tr.gaps)
    else:
        beats = findpeaks_replica(x, fs)
        implausible, rescue, gaps = implausible_fraction(beats), 0.0, ()
    y, fs_d = _prepare(x, fs, DETECT_BAND_HZ)
    snr = float(_template_snr(y, beats, fs_d)) if beats.size else float("nan")
    cg = count_gate(beats, starts, bpm)

    def harm_of(xi: F64 | None) -> float:
        if xi is None:
            return float("nan")
        return _harmed(beats, _detect(det, xi, fs), times) / N_INJECTIONS
    harm, harm_m = harm_of(x_real), harm_of(x_med)
    plausible = bool(implausible <= PROVISIONAL_MAX_IMPLAUSIBLE_FRAC
                     and rescue <= PROVISIONAL_MAX_RESCUE_RATE)
    return TrainGate(channel=name, detector=det, beats_s=beats, snr=snr, count=cg,
                     transient_harm=harm, implausible_frac=float(implausible),
                     rescue_rate=float(rescue), plausible=plausible,
                     passes=bool(cg.passes and plausible
                                 and harm <= PROVISIONAL_MAX_TRANSIENT_HARM),
                     source=source, harm_median=harm_m, gaps_s=gaps,
                     minute_s=tuple(float(t) for t in starts_w),
                     minute_dur_s=tuple(float(d) for d in durs),
                     valid=tuple(bool(v) for v in minute_validity(beats, starts_w, bpm_w,
                                                                  durs=durs)))


def gated_selection(  # noqa: PLR0915 - one pass over every source, in the ruled order
    rec: Recording, events: Events, *, seed: int = 0, incumbent: tuple[str, str] | None = None,
    pairs: bool = True, per_minute: bool = False,
) -> tuple[TrainGate | None, list[TrainGate]]:
    """Return ``(stored, every train's gates)`` - ``stored`` is None if nothing passes.

    Sources (ruling 2026-09-30 (c) 4; ruling 2026-10-02 (c) 1): per signal of
    :func:`hr_candidates`, task 05's :func:`detect_rpeaks` and :func:`findpeaks_replica`;
    with ``pairs``, every cross-site lead of ``hr_pairs.pair_candidates`` (detached
    contacts left out) with both detectors. Every row faces the same gates:
    :func:`count_gate` against its own :func:`autocorr_rate`, task 05's plausibility
    gates, and the transient veto with the REAL-PATTERN injection binding
    (``hr_pairs.inject_real_patterns``; the median pattern is reported alongside). A
    pair row must also pass the cross-check (a) ``hr_pairs.rate_check`` and (c)
    ``hr_pairs.morphology_check``. With no non-cardiac event to inject, no row passes.

    Choice: the ``incumbent`` ``(channel, detector)`` when it still passes; else the
    passing channel row with the best template SNR (exact tie: the count closest to the
    autocorrelation); a pair only where no channel row passes. Then rule 2: when a
    channel train is chosen and a pair passes, a disagreement on more than 5% of beats
    is resolved beat by beat (``hr_pairs.resolve_disagreement``) and the train that
    carries the QRS (:data:`CARRIES_QRS`) is stored; unresolved, nothing is.

    The cross-check runs on every pair row that passes the count gate and plausibility,
    whatever its veto: a mask-grade pair (:func:`peri_r_train`) must pass (a) and (c)
    too (ruling 2026-10-02 (f) 3).

    ``per_minute`` (ruling 2026-10-02 (f) 2): the count gate leaves the recording-level
    gates. A row is ``eligible`` when it passes the veto (with precision, for every
    plausible row), plausibility and, for a pair, (c); each minute is valid by
    :func:`minute_validity` - for a pair, also not failing (a) where (a) is assessable.
    Stored (:func:`_choose_minutes`): the incumbent when eligible, else the eligible row
    with the most valid minutes (ties: template SNR), then rule 2 between it and the best
    eligible row of the other source, on the minutes valid in both; the stored train is
    :func:`minute_gapped`. Timing (b) enters only through rule 2: it applies where an
    eligible train of the other source exists to compare with.
    """
    # imported here: hr_pairs imports this module
    from gems_blanking_v2.physio import hr_pairs as hp  # noqa: PLC0415

    fs = float(rec.fs)
    cand = hr_candidates(rec)
    t05 = _trains(cand)
    try:
        first, _t = rank_hr_channels(cand, t05)
    except ValueError:
        first = max(t05, key=lambda k: t05[k].n_beats)
    keep = _noncardiac(events, t05[first].t_s)
    rel = hp.event_patterns(events)
    real: Recording | None = None
    med: Recording | None = None
    cand_r: Recording | None = None
    cand_m: Recording | None = None
    times = np.zeros(0)
    if bool((keep & np.isfinite(rel).all(axis=1)).any()):  # else no row can pass the veto
        real, times = hp.inject_real_patterns(rec, events, keep, seed=seed)
        med, _tm = hp.inject_median_pattern(rec, events, keep, seed=seed)
        cand_r, cand_m = hr_candidates(real), hr_candidates(med)
    rows: list[TrainGate] = []
    for c in cand.channels:
        x = np.asarray(cand.data[:, c.index], dtype=np.float64)
        xr = None if cand_r is None else np.asarray(cand_r.data[:, c.index], dtype=np.float64)
        xm = None if cand_m is None else np.asarray(cand_m.data[:, c.index], dtype=np.float64)
        ac = autocorr_windows(x, fs)
        rows += [_gate_row(c.name, det, x, xr, xm, times, fs, ac,
                           t05[c.name] if det == "task05" else None, "channel")
                 for det in DETECTORS]
    detached: frozenset[str] = frozenset()
    leads: dict[str, hp.PairLead] = {}
    if pairs:
        detached = hp.detached_contacts(events, keep)
        acs: dict[frozenset[str], tuple[F64, F64, F64]] = {}
        for p in hp.pair_candidates(rec, detached):
            leads[p.name] = p
            x = hp.lead_signal(rec, p)
            xr = hp.lead_signal(real, p) if real is not None else None
            xm = hp.lead_signal(med, p) if med is not None else None
            key = frozenset((p.plus, p.minus))
            if key not in acs:
                acs[key] = autocorr_windows(x, fs)
            rows += [_gate_row(p.name, det, x, xr, xm, times, fs, acs[key], None, "pair")
                     for det in DETECTORS]
    if real is not None:  # veto precision (ruling 2026-10-02 (e) 3)

        def block(k: int) -> tuple[F64, Callable[[TrainGate], F64]]:
            rk, tk = hp.inject_real_patterns(rec, events, keep, seed=seed + k)
            ck = hr_candidates(rk)
            col = {c.name: c.index for c in ck.channels}

            def signal(r: TrainGate) -> F64:
                if r.source == "pair":
                    return hp.lead_signal(rk, leads[r.channel])
                return np.asarray(ck.data[:, col[r.channel]], dtype=np.float64)
            return tk, signal
        rows = _refine_harm(rows, block, fs, need_count=not per_minute)

    def veto_ok(r: TrainGate) -> bool:
        return bool(np.isfinite(r.transient_harm)
                    and r.transient_harm <= PROVISIONAL_MAX_TRANSIENT_HARM)
    cuff = {c.name: c.cuff_id for c in rec.channels}
    refs: dict[frozenset[str | None], hp.References] = {}
    for i, row in enumerate(rows):  # the cross-check: every pair a stored or mask train could be
        if row.source != "pair" or not row.plausible:
            continue
        if not (row.count.passes or (per_minute and veto_ok(row))):
            continue
        p = leads[row.channel]
        cuffs = frozenset((cuff[p.plus], cuff[p.minus]))
        if cuffs not in refs:
            refs[cuffs] = hp.rate_references(rec, p, detached)
        a = hp.rate_check(row.beats_s, rec, p, detached, refs=refs[cuffs])
        m = hp.morphology_check(row.beats_s, rec, detached=detached)
        failed = [n for n, ok in (("rate", a.passes), ("morphology", m.passes)) if not ok]
        minute_s = np.asarray(row.minute_s, dtype=np.float64)
        keep = np.asarray(row.valid, dtype=bool) & ~_listed(minute_s, a.bad_minutes_s)
        valid = tuple(bool(v) for v in keep)
        rows[i] = replace(row, passes=row.passes and not failed,
                          cross_check="+".join(failed) if failed else "pass",
                          rate_ok=a.passes, morphology_ok=m.passes, valid=valid)
    if per_minute:
        rows = [replace(r, eligible=bool(r.plausible and veto_ok(r) and np.isfinite(r.snr)
                                         and (r.source == "channel" or r.morphology_ok is True)))
                for r in rows]
        return _choose_minutes(rows, rec, detached, incumbent), rows
    return _choose(rows, rec, detached, incumbent), rows


def _choose(rows: list[TrainGate], rec: Recording, detached: frozenset[str],  # noqa: PLR0911
            incumbent: tuple[str, str] | None) -> TrainGate | None:
    """Incumbent, else best channel, else best pair; then rule 2 (see :func:`gated_selection`)."""
    from gems_blanking_v2.physio import hr_pairs as hp  # noqa: PLC0415

    channel = [r for r in rows if r.source == "channel"]
    best_pair = pick_train([r for r in rows if r.source == "pair"])
    chosen = None
    if incumbent is not None:
        chosen = next((r for r in channel
                       if (r.channel, r.detector) == incumbent and r.passes), None)
    is_incumbent = chosen is not None
    if chosen is None:
        chosen = pick_train(channel)
    if chosen is None:
        if best_pair is None:
            return None
        return replace(best_pair, note="pair lead: no channel train passes")
    if best_pair is None:
        return chosen
    t = hp.timing_check(best_pair.beats_s, chosen.beats_s)
    if not t.disagree:
        return chosen
    res = hp.resolve_disagreement(best_pair.beats_s, chosen.beats_s, rec, detached=detached,
                                  offset_s=t.offset_s)
    pq, cq = res.lead_unmatched_qrs, res.other_unmatched_qrs
    why = (f"rule 2: {best_pair.channel} {best_pair.detector} vs {chosen.channel} "
           f"{chosen.detector} disagree; unmatched beats carrying the QRS {pq:.3f} vs {cq:.3f}")
    win = rule2_winner(pq, cq)
    if win == "pair":
        return replace(best_pair, note=why + " - the pair replaces it")
    if win == "other":
        return replace(chosen, note=why + " - the channel train stays")
    if is_incumbent:
        return replace(chosen, note=why + " - ambiguous: the incumbent stays")
    return None  # ambiguous with no incumbent: neither train is stored


def _in_minutes(row: TrainGate, minutes: npt.NDArray[np.bool_]) -> F64:
    """``row``'s beats that lie in the minutes flagged in ``minutes`` (its own grid)."""
    b = np.sort(row.beats_s)
    ok = _in_window(np.asarray(row.minute_s, dtype=np.float64), _durations(row), b, minutes)
    return np.asarray(b[ok], dtype=np.float64)


def _choose_minutes(rows: list[TrainGate], rec: Recording, detached: frozenset[str],
                    incumbent: tuple[str, str] | None) -> TrainGate | None:
    """Per-minute storage (ruling 2026-10-02 (f) 2; see :func:`gated_selection`)."""
    from gems_blanking_v2.physio import hr_pairs as hp  # noqa: PLC0415

    ok = [r for r in rows if r.eligible and r.n_valid > 0]
    if not ok:
        return None

    def rank(r: TrainGate) -> tuple[int, float, float]:
        return (r.n_valid, r.snr, -r.count.median_abs_dev)
    chosen = None
    if incumbent is not None:
        chosen = next((r for r in ok if (r.channel, r.detector) == incumbent), None)
    is_incumbent = chosen is not None
    why = "per minute: the incumbent, still eligible"
    if chosen is None:
        chosen = max(ok, key=rank)
        why = f"per minute: most valid minutes ({chosen.n_valid} of {len(chosen.valid)})"
    others = [r for r in ok if r.source != chosen.source]
    if others:
        other = max(others, key=rank)
        pair, chan = (chosen, other) if chosen.source == "pair" else (other, chosen)
        assert len(pair.valid) == len(chan.valid), "every row shares one minute grid"
        both = np.asarray(pair.valid, dtype=bool) & np.asarray(chan.valid, dtype=bool)
        pb, cb = _in_minutes(pair, both), _in_minutes(chan, both)
        if pb.size and cb.size:
            t = hp.timing_check(pb, cb)
            if t.disagree:
                res = hp.resolve_disagreement(pb, cb, rec, detached=detached, offset_s=t.offset_s)
                pq, cq = res.lead_unmatched_qrs, res.other_unmatched_qrs
                why = (f"rule 2 on {int(both.sum())} shared valid minutes: {pair.channel} "
                       f"{pair.detector} vs {chan.channel} {chan.detector} disagree; unmatched "
                       f"beats carrying the QRS {pq:.3f} vs {cq:.3f}")
                win = rule2_winner(pq, cq)
                if win == "pair":
                    chosen, why = pair, why + " - the pair is stored"
                elif win == "other":
                    chosen, why = chan, why + " - the channel train is stored"
                elif is_incumbent:
                    why += " - ambiguous: the incumbent stays"
                else:
                    return None  # ambiguous with no incumbent: neither train is stored
    return minute_gapped(replace(chosen, note=why))


def pick_train(rows: list[TrainGate]) -> TrainGate | None:
    """Return the passing train with the best template SNR, or None if none passes.

    On an exact SNR tie, the one closest to the autocorrelation count
    (``count.median_abs_dev``).
    """
    passing = [r for r in rows if r.passes and np.isfinite(r.snr)]
    if not passing:
        return None
    return max(passing, key=lambda r: (r.snr, -r.count.median_abs_dev))


def peri_r_train(rows: list[TrainGate]) -> tuple[TrainGate | None, str]:
    """Return ``(train, grade)`` for the peri-R time mask (addendum to ruling (c) 1).

    The vetted train (:func:`pick_train`) when there is one, grade ``"hrv"``;
    otherwise the train with the best template SNR among those that pass the count
    gate and task 05's plausibility gates, whatever their transient harm - grade
    ``"mask"``, for the mask only. ``(None, "none")`` when no train passes the count
    gate: the recording gets no peri-R route.

    Ruling 2026-10-02 (f) 3: a pair establishes that it is cardiac only through
    cross-check (a) and (c), so a mask-grade pair must pass both (``rate_ok`` and
    ``morphology_ok``; not run counts as failing).
    """
    vetted = pick_train(rows)
    if vetted is not None:
        return vetted, "hrv"
    ok = [r for r in rows if r.count.passes and r.plausible and np.isfinite(r.snr)
          and (r.source != "pair" or (r.rate_ok is True and r.morphology_ok is True))]
    if not ok:
        return None, "none"
    return max(ok, key=lambda r: (r.snr, -r.count.median_abs_dev)), "mask"
