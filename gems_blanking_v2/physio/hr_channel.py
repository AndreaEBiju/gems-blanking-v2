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
    "PROVISIONAL_AC_MIN_PEAK",
    "PROVISIONAL_MAX_COUNT_DEV",
    "PROVISIONAL_MAX_TRANSIENT_HARM",
    "CountGate",
    "TrainGate",
    "autocorr_rate",
    "best_hr_channel",
    "cm_gains",
    "count_gate",
    "findpeaks_replica",
    "gated_selection",
    "hr_candidates",
    "implausible_fraction",
    "peri_r_train",
    "pick_train",
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


def autocorr_rate(x: npt.ArrayLike, fs: float) -> tuple[F64, F64]:
    """Return ``(minute start times s, bpm)`` from the rectified 10-150 Hz autocorrelation.

    No peak picking: per :data:`AC_WINDOW_S` window, the normalised autocorrelation of
    ``|10-150 Hz|`` is searched over the :data:`AC_BPM` lags, and the peak's lag
    gives the rate. The nearest lag (~1 ms steps) is within 0.5% at 550 bpm, a tenth
    of :data:`PROVISIONAL_MAX_COUNT_DEV`, so there is no sub-lag refinement. ``bpm``
    is NaN where the minute is not clear (:data:`PROVISIONAL_AC_MIN_PEAK`, interior
    local maximum). A trailing partial minute is not assessed.
    """
    y, fd = _decimate_to(np.asarray(x, dtype=np.float64), fs, _AC_TARGET_HZ)
    sos = butter(4, AC_BAND_HZ, btype="bandpass", fs=fd, output="sos")
    r = np.abs(sosfiltfilt(sos, y))
    w = int(round(AC_WINDOW_S * fd))
    lag_lo = int(np.floor(60.0 / AC_BPM[1] * fd))
    lag_hi = int(np.ceil(60.0 / AC_BPM[0] * fd))
    starts: list[float] = []
    bpm: list[float] = []
    for m in range(int(r.size / fd // AC_WINDOW_S)):
        a = int(round(m * AC_WINDOW_S * fd))
        if a + w > r.size:
            break
        seg = r[a:a + w] - r[a:a + w].mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(seg, n=2 * w)) ** 2)[:w]
        rate = float("nan")
        if ac[0] > 0:
            ac = ac / ac[0]
            k = lag_lo + int(np.argmax(ac[lag_lo:lag_hi + 1]))
            if (lag_lo < k < lag_hi and ac[k] >= PROVISIONAL_AC_MIN_PEAK
                    and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]):
                rate = 60.0 * fd / k
        starts.append(m * AC_WINDOW_S)
        bpm.append(rate)
    return np.asarray(starts, dtype=np.float64), np.asarray(bpm, dtype=np.float64)


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

    @property
    def gap_after(self) -> npt.NDArray[np.bool_]:
        """Per beat (sorted), whether the interval after it spans a tagged gap."""
        b = np.sort(self.beats_s)
        starts = np.asarray([g[0] for g in self.gaps_s], dtype=np.float64)
        if starts.size == 0:
            return np.zeros(b.size, dtype=bool)
        return np.asarray(np.isclose(b[:, None], starts[None, :], rtol=0.0, atol=1e-9).any(axis=1))


CARRIES_QRS: Final = 0.5
"""Rule 2 of ruling 2026-10-02 (c): of the two trains' beats unmatched within 20 ms, the
train whose unmatched beats reach the QRS-correlation floor in at least this fraction
wins - when the other's do not. Both or neither: unresolved, neither is stored. Measured
on A t05: the lead 0.974, its L_T 0.013."""


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
              fs: float, ac: tuple[F64, F64], t05: BeatTrain | None, source: str) -> TrainGate:
    """Gate one train: count, plausibility, and the real-pattern veto (binding)."""
    starts, bpm = ac
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
                     source=source, harm_median=harm_m, gaps_s=gaps)


def gated_selection(
    rec: Recording, events: Events, *, seed: int = 0, incumbent: tuple[str, str] | None = None,
    pairs: bool = True,
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
        ac = autocorr_rate(x, fs)
        rows += [_gate_row(c.name, det, x, xr, xm, times, fs, ac,
                           t05[c.name] if det == "task05" else None, "channel")
                 for det in DETECTORS]
    detached: frozenset[str] = frozenset()
    if pairs:
        detached = hp.detached_contacts(events, keep)
        acs: dict[frozenset[str], tuple[F64, F64]] = {}
        for p in hp.pair_candidates(rec, detached):
            x = hp.lead_signal(rec, p)
            xr = hp.lead_signal(real, p) if real is not None else None
            xm = hp.lead_signal(med, p) if med is not None else None
            key = frozenset((p.plus, p.minus))
            if key not in acs:
                acs[key] = autocorr_rate(x, fs)
            for det in DETECTORS:
                row = _gate_row(p.name, det, x, xr, xm, times, fs, acs[key], None, "pair")
                if row.passes:
                    a = hp.rate_check(row.beats_s, rec, p, detached)
                    m = hp.morphology_check(row.beats_s, rec, detached=detached)
                    failed = [n for n, ok in (("rate", a.passes), ("morphology", m.passes))
                              if not ok]
                    row = replace(row, passes=not failed,
                                  cross_check="+".join(failed) if failed else "pass")
                rows.append(row)
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
    if pq >= CARRIES_QRS > cq:
        return replace(best_pair, note=why + " - the pair replaces it")
    if cq >= CARRIES_QRS > pq:
        return replace(chosen, note=why + " - the channel train stays")
    return None  # unresolved: neither train is trusted


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
    """
    vetted = pick_train(rows)
    if vetted is not None:
        return vetted, "hrv"
    ok = [r for r in rows if r.count.passes and r.plausible and np.isfinite(r.snr)]
    if not ok:
        return None, "none"
    return max(ok, key=lambda r: (r.snr, -r.count.median_abs_dev)), "mask"
