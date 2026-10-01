"""The cross-site HR lead, pairs only - the design frozen for H (ruling 2026-10-01 (d)).

Why it can work (ruling (c) 2). On this rig the ground site's transients enter every
contact with nearly the same gain (measured g 0.96-1.05, 99-100% sign agreement, no
drift over a recording), while the heart's far-field projects differently onto the two
necks. A single contact cannot tell the transient from a beat, so the transient is
captured; the tripole cancels both. The difference of a left and a right neck contact
cancels the ground and keeps the heart: a bipolar ECG lead from distant contacts.

**Candidates** (:func:`pair_candidates`): every left-neck contact minus every right-neck
contact and the reverse - 9 differences in both orientations, 18 leads when both cuffs
have three contacts - each with task 05's :func:`detect_rpeaks` and the ``findpeaks``
replica. Both detectors take positive peaks, so orientation is part of the candidate.
Not in the design: weighted leads, stomach pairs, the template detector.

**Gates, unchanged**: :func:`hr_channel.count_gate` against the lead's own
:func:`hr_channel.autocorr_rate`, task 05's plausibility gates, and the transient veto
at :data:`hr_channel.PROVISIONAL_MAX_TRANSIENT_HARM`. The BINDING veto injects
**real patterns** (ruling (c) amendment 1): each of :data:`hr_channel.N_INJECTIONS`
transients carries one real non-cardiac event's signed gain pattern and amplitude
(:func:`inject_real_patterns`). The median-pattern injection is reported alongside; a
pattern-matched null cancels it by construction, so it never decides.

**Protocol** (:func:`half_split`, amendment 2 - selection must not inflate): the region
is halved; every candidate is gated on the first half and :func:`select` picks one;
only that lead and detector is gated on the second half, and only that result decides.
The single channels (:func:`hr_channel.hr_candidates`) go through the same protocol, so
the comparison holds the protocol fixed (invariant 12).

**Independent cross-check** (ruling (d) 2), on references that do not use the lead:
(a) :func:`rate_check`, (b) :func:`timing_check` where a vetted train exists,
(c) :func:`morphology_check`. A recording keeps a gain only if (a) and (c) pass, and
(b) where it applies.

OUTSIDE THE GENERATION HASH: nothing in the detection chain imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.derive.cm_events import Events, find_events
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.physio import hr_channel as hc
from gems_blanking_v2.physio.rpeaks import (
    DETECT_BAND_HZ,
    PROVISIONAL_MAX_IMPLAUSIBLE_FRAC,
    PROVISIONAL_MAX_RESCUE_RATE,
    _prepare,
    _template_snr,
    detect_rpeaks,
    rank_hr_channels,
)
from gems_blanking_v2.types import ChannelInfo, Recording

__all__ = [
    "INJECTIONS",
    "MORPH_SHUFFLES",
    "RATE_PASS_FRACTION",
    "RATE_TOL",
    "TIMING_PASS_FRACTION",
    "TIMING_TOL_S",
    "HalfSplit",
    "LeadGate",
    "MorphologyCheck",
    "PairLead",
    "RateCheck",
    "TimingCheck",
    "event_patterns",
    "gate_half",
    "half_split",
    "inject_median_pattern",
    "inject_real_patterns",
    "lead_signal",
    "morphology_check",
    "pair_candidates",
    "rate_check",
    "select",
    "timing_check",
]

F64 = npt.NDArray[np.float64]

INJECTIONS: Final[tuple[str, str]] = ("real", "median")
"""``real`` is binding; ``median`` is reported alongside."""

RATE_TOL: Final = 0.05
"""(a): the lead's beats per minute within this fraction of the reference median."""
RATE_PASS_FRACTION: Final = 0.95
"""(a): ... in at least this fraction of the minutes that have a clear reference."""
TIMING_TOL_S: Final = 0.002
"""(b): a lead beat agrees with the vetted train within this, after the offset."""
TIMING_PASS_FRACTION: Final = 0.99
"""(b): ... for at least this fraction of the lead's beats."""
TIMING_SEARCH_S: Final = 0.010
"""(b): the constant offset is the median lead-minus-vetted lag over pairs this close."""
MORPH_SHUFFLES: Final = 20
"""(c): noise-shuffled templates per contact; the contact must beat every one."""
MORPH_SHIFT_RR: Final[tuple[float, float]] = (0.25, 0.75)
"""(c): a shuffled template moves each beat by this range of the median RR - the same
count of windows, none locked to a beat."""


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairLead:
    """One candidate lead: ``plus`` minus ``minus``, raw columns ``plus_index``, ``minus_index``."""

    plus: str
    minus: str
    plus_index: int
    minus_index: int

    @property
    def name(self) -> str:
        """``"<plus>-<minus>"``, e.g. ``"LVN2-RVN1"``."""
        return f"{self.plus}-{self.minus}"


def _neck(rec: Recording, cuff: str) -> list[ChannelInfo]:
    return sorted((c for c in rec.channels
                   if c.role == "nerve" and c.cuff_id == cuff and c.contact_index is not None),
                  key=lambda c: c.contact_index or 0)


def pair_candidates(rec: Recording) -> list[PairLead]:
    """Every left-neck contact minus every right-neck contact, and the reverse.

    Contacts are the raw nerve channels with a ``cuff_id`` and a ``contact_index``;
    stomach and aux channels never enter. 18 leads when both cuffs have three contacts.
    """
    left, right = _neck(rec, "L"), _neck(rec, "R")
    out = []
    for a in left:
        for b in right:
            out.append(PairLead(a.name, b.name, a.index, b.index))
            out.append(PairLead(b.name, a.name, b.index, a.index))
    return out


def lead_signal(rec: Recording, lead: PairLead) -> F64:
    """Return the lead in microvolts: a new array; the recording is not written."""
    d = rec.data
    plus = np.asarray(d[:, lead.plus_index], dtype=np.float64)
    return plus - np.asarray(d[:, lead.minus_index], dtype=np.float64)


# ---------------------------------------------------------------------------
# the veto's injections
# ---------------------------------------------------------------------------


def event_patterns(events: Events) -> F64:
    """Signed relative gain pattern per event, ``(n_events, n_channels)``, NaN where undefined.

    Each channel's 300-3000 Hz amplitude times the event's polarity (sign of its median
    amplitude) over the event's median |amplitude|. An event whose median |amplitude|
    is zero has no pattern (0/0): its row is NaN, and it is never injected.
    """
    a = np.asarray(events.channel_amp_uv, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, len(events.channels)))
    pol = np.sign(np.median(a, axis=1))
    pol[pol == 0] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = a * pol[:, None] / np.median(np.abs(a), axis=1, keepdims=True)
    rel[~np.isfinite(rel).all(axis=1)] = np.nan
    return rel


def _pulse(n: int, fs: float, t0: float, amp: float) -> tuple[npt.NDArray[np.int64], F64]:
    width = hc.TRANSIENT_FWHM_S / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    c, half = int(round(t0 * fs)), int(round(10 * width * fs))
    k = np.arange(max(c - half, 0), min(c + half + 1, n))
    return k, amp * np.exp(-0.5 * ((k / fs - t0) / width) ** 2)


def _inject(rec: Recording, events: Events, keep: npt.NDArray[np.bool_], seed: int,
            *, median: bool) -> tuple[Recording, F64]:
    rel = event_patterns(events)
    pool = np.flatnonzero(np.asarray(keep, dtype=bool) & np.isfinite(rel).all(axis=1))
    if pool.size == 0:
        msg = "no non-cardiac event with a gain pattern to inject"
        raise ValueError(msg)
    fs, n = float(rec.fs), rec.data.shape[0]
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(1.0, n / fs - 1.0, hc.N_INJECTIONS))
    pick = rng.choice(pool, hc.N_INJECTIONS)
    amp = np.asarray(events.cm_amp_uv, dtype=np.float64)[pick]
    g = np.median(rel[pool], axis=0)
    col = {c.name: c.index for c in rec.channels}
    data = np.array(rec.data, dtype=np.float64)
    for t0, a, e in zip(times, amp, pick, strict=True):
        k, p = _pulse(n, fs, float(t0), float(a))
        pattern = g if median else rel[e]
        for name, gain in zip(events.channels, pattern, strict=True):
            if name in col:
                data[k, col[name]] += gain * p
    return replace(rec, data=data), times


def inject_real_patterns(rec: Recording, events: Events, keep: npt.ArrayLike, *,
                         seed: int = 0) -> tuple[Recording, F64]:
    """Inject the BINDING way: each transient is one real non-cardiac event, pattern and amplitude.

    :data:`hr_channel.N_INJECTIONS` Gaussian transients of
    :data:`hr_channel.TRANSIENT_FWHM_S` at uniform times; each draws one event from
    ``keep`` (non-cardiac) that has a pattern, and enters every channel the events name
    at that event's common-mode amplitude times its signed relative gain. Returns
    ``(injected copy, times s)``; the input is not written.
    """
    return _inject(rec, events, np.asarray(keep, dtype=bool), seed, median=False)


def inject_median_pattern(rec: Recording, events: Events, keep: npt.ArrayLike, *,
                          seed: int = 0) -> tuple[Recording, F64]:
    """Inject the median signed pattern on every transient (reported, never binding).

    Same times and amplitudes as :func:`inject_real_patterns` with the same seed.
    """
    return _inject(rec, events, np.asarray(keep, dtype=bool), seed, median=True)


# ---------------------------------------------------------------------------
# gating and selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadGate:
    """One lead (a pair, or a single channel for the comparison) and one detector, gated."""

    lead: str
    detector: str
    beats_s: F64 = field(repr=False)
    snr: float
    count: hc.CountGate
    implausible_frac: float
    rescue_rate: float
    plausible: bool
    harm: dict[str, float]
    """Per injection (:data:`INJECTIONS`), the fraction of injections that cost a beat."""

    @property
    def passes(self) -> bool:
        """Count gate, plausibility and the BINDING real-pattern veto."""
        return bool(self.count.passes and self.plausible
                    and self.harm["real"] <= hc.PROVISIONAL_MAX_TRANSIENT_HARM)

    @property
    def passes_median(self) -> bool:
        """The same with the median-pattern veto - reported alongside, never binding."""
        return bool(self.count.passes and self.plausible
                    and self.harm["median"] <= hc.PROVISIONAL_MAX_TRANSIENT_HARM)


def _gate(name: str, det: str, x: F64, x_inj: dict[str, F64], times: F64,
          fs: float, ac: tuple[F64, F64] | None = None) -> LeadGate:
    starts, bpm = ac if ac is not None else hc.autocorr_rate(x, fs)
    if det == "task05":
        tr = detect_rpeaks(x, fs)
        beats = np.asarray(tr.t_s, dtype=np.float64)
        implaus, rescue = float(tr.implausible_frac), float(tr.rescue_rate)
    else:
        beats = hc.findpeaks_replica(x, fs)
        implaus, rescue = hc.implausible_fraction(beats), 0.0
    y, fsd = _prepare(x, fs, DETECT_BAND_HZ)
    snr = float(_template_snr(y, beats, fsd)) if beats.size else float("nan")
    harm = {k: hc._harmed(beats, hc._detect(det, xi, fs), times) / hc.N_INJECTIONS
            for k, xi in x_inj.items()} if x_inj else dict.fromkeys(INJECTIONS, float("nan"))
    plausible = bool(implaus <= PROVISIONAL_MAX_IMPLAUSIBLE_FRAC
                     and rescue <= PROVISIONAL_MAX_RESCUE_RATE)
    return LeadGate(name, det, beats, snr, hc.count_gate(beats, starts, bpm), implaus, rescue,
                    plausible, harm)


@dataclass
class _Half:
    rec: Recording
    events: Events
    keep: npt.NDArray[np.bool_]
    injected: dict[str, Recording]
    times: F64


def _prepare_half(rec: Recording, seed: int) -> _Half:
    """Events, the non-cardiac selection (as ``gated_selection``) and both injections."""
    ev = find_events(rec)
    cand = hc.hr_candidates(rec)
    t05 = {c.name: detect_rpeaks(np.asarray(cand.data[:, c.index], dtype=np.float64), float(rec.fs))
           for c in cand.channels}
    try:
        first, _t = rank_hr_channels(cand, t05)
    except ValueError:
        first = max(t05, key=lambda k: t05[k].n_beats)
    keep = hc._noncardiac(ev, t05[first].t_s)
    if not (keep & np.isfinite(event_patterns(ev)).all(axis=1)).any():
        # nothing to inject: the veto cannot be measured, so no lead can pass it (harm NaN)
        return _Half(rec, ev, keep, {}, np.zeros(0))
    real, times = inject_real_patterns(rec, ev, keep, seed=seed)
    med, times_m = inject_median_pattern(rec, ev, keep, seed=seed)
    assert np.array_equal(times, times_m)
    return _Half(rec, ev, keep, {"real": real, "median": med}, times)


def gate_half(h: _Half, *, pairs: list[PairLead] | None = None,
              only: tuple[str, str] | None = None) -> tuple[list[LeadGate], list[LeadGate]]:
    """Gate the pair leads and the single channels of one half; ``only`` = one (lead, detector).

    Returns ``(pair rows, single-channel rows)``. A lead and its reverse share the
    autocorrelation rate (it does not see the sign), so it is computed once per pair.
    """
    fs = float(h.rec.fs)
    pairs = pair_candidates(h.rec) if pairs is None else pairs

    def want(name: str, det: str) -> bool:
        return only is None or (name, det) == only

    rows: list[LeadGate] = []
    ac: dict[frozenset[str], tuple[F64, F64]] = {}
    for p in pairs:
        dets = [d for d in hc.DETECTORS if want(p.name, d)]
        if not dets:
            continue
        x = lead_signal(h.rec, p)
        xi = {k: lead_signal(r, p) for k, r in h.injected.items()}
        key = frozenset((p.plus, p.minus))
        if key not in ac:
            ac[key] = hc.autocorr_rate(x, fs)
        rows += [_gate(p.name, d, x, xi, h.times, fs, ac[key]) for d in dets]
    singles: list[LeadGate] = []
    cand = hc.hr_candidates(h.rec)
    cand_i = {k: hc.hr_candidates(r) for k, r in h.injected.items()}
    for c in cand.channels:
        dets = [d for d in hc.DETECTORS if want(c.name, d)]
        if not dets:
            continue
        x = np.asarray(cand.data[:, c.index], dtype=np.float64)
        xi = {k: np.asarray(ci.data[:, c.index], dtype=np.float64) for k, ci in cand_i.items()}
        a = hc.autocorr_rate(x, fs)
        singles += [_gate(c.name, d, x, xi, h.times, fs, a) for d in dets]
    return rows, singles


def select(rows: list[LeadGate]) -> LeadGate | None:
    """Return the first-half choice: the passing row with the best template SNR.

    With none passing: among rows that pass the count gate and plausibility, the least
    real-pattern harm (then the best SNR) - so the second half still judges something.
    None when no row passes the count gate and plausibility.
    """
    ok = [r for r in rows if r.passes and np.isfinite(r.snr)]
    if ok:
        return max(ok, key=lambda r: r.snr)
    cp = [r for r in rows if r.count.passes and r.plausible and np.isfinite(r.snr)]
    return min(cp, key=lambda r: (r.harm["real"], -r.snr)) if cp else None


@dataclass(frozen=True)
class HalfSplit:
    """:func:`half_split`'s result."""

    first_pairs: list[LeadGate] = field(repr=False)
    first_singles: list[LeadGate] = field(repr=False)
    chosen_pair: LeadGate | None
    chosen_single: LeadGate | None
    second_pair: LeadGate | None
    second_single: LeadGate | None
    second_rec: Recording = field(repr=False)
    second_events: Events = field(repr=False)

    @property
    def pairs_passing_first(self) -> int:
        """How many first-half pair rows pass every gate."""
        return sum(r.passes for r in self.first_pairs)

    @property
    def gained(self) -> bool:
        """The chosen pair passes the second half and the chosen single channel does not."""
        return bool(self.second_pair is not None and self.second_pair.passes
                    and not (self.second_single is not None and self.second_single.passes))


def _halves(rec: Recording) -> tuple[Recording, Recording]:
    n = rec.data.shape[0] // 2
    return replace(rec, data=rec.data[:n]), replace(rec, data=rec.data[n:2 * n])


def half_split(rec: Recording, *, seed: int = 0) -> HalfSplit:
    """Run the frozen protocol: choose on the first half, judge on the second.

    Everything that selects - events, the non-cardiac pool, the gates of all 18 pair
    candidates and of every single channel, :func:`select` - uses the first half only.
    The second half has its own events and injections and gates only the chosen pair
    and the chosen single channel.
    """
    a, b = _halves(rec)
    h1 = _prepare_half(a, seed)
    pairs1, singles1 = gate_half(h1)
    cp, cs = select(pairs1), select(singles1)
    del h1
    h2 = _prepare_half(b, seed)
    sp = gate_half(h2, only=(cp.lead, cp.detector))[0][0] if cp is not None else None
    ss = gate_half(h2, pairs=[], only=(cs.lead, cs.detector))[1][0] if cs is not None else None
    return HalfSplit(pairs1, singles1, cp, cs, sp, ss, b, h2.events)


# ---------------------------------------------------------------------------
# the independent cross-check (ruling (d) 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateCheck:
    """(a): the lead's per-minute rate against references that do not use it."""

    references: tuple[str, ...]
    minutes: int
    minutes_with_reference: int
    fraction_within: float
    passes: bool


def rate_check(beats_s: npt.ArrayLike, rec: Recording, lead: PairLead) -> RateCheck:
    """Per minute, the lead's rate against the median clear reference.

    References: :func:`hr_channel.autocorr_rate` on every raw channel of a site the
    lead does not use (a left-right pair uses both necks, so the stomach contacts) and
    on ``stomach_ref``. A minute is judged when at least one reference is clear there;
    it agrees when beats per minute are within :data:`RATE_TOL` of the median clear
    reference. Passes when at least :data:`RATE_PASS_FRACTION` of judged minutes agree;
    no judged minute fails (unassessable never passes).
    """
    fs = float(rec.fs)
    used = {c.cuff_id for c in rec.channels if c.name in (lead.plus, lead.minus)}
    refs: list[tuple[str, F64]] = [
        (c.name, np.asarray(rec.data[:, c.index], dtype=np.float64)) for c in rec.channels
        if (c.role == "stomach") or (c.role == "nerve" and c.cuff_id not in used)]
    signals, _w = build_derivations(rec)
    if "stomach_ref" in signals:
        refs.append(("stomach_ref", np.asarray(signals["stomach_ref"], dtype=np.float64)))
    rates = []
    starts0: F64 | None = None
    for _name, x in refs:
        starts, bpm = hc.autocorr_rate(x, fs)
        if starts0 is None:
            starts0 = starts
        assert np.array_equal(starts, starts0), "references share one minute grid"
        rates.append(bpm)
    names = tuple(n for n, _x in refs)
    if starts0 is None or starts0.size == 0:
        return RateCheck(names, 0, 0, 0.0, passes=False)
    stack = np.vstack(rates)
    clear = np.isfinite(stack).any(axis=0)
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    judged = np.flatnonzero(clear)
    if judged.size == 0:
        return RateCheck(names, int(starts0.size), 0, 0.0, passes=False)
    ok = []
    for i in judged:
        s = starts0[i]
        per_min = np.count_nonzero((b >= s) & (b < s + hc.AC_WINDOW_S)) * 60.0 / hc.AC_WINDOW_S
        ref = float(np.nanmedian(stack[:, i]))
        ok.append(abs(per_min - ref) <= RATE_TOL * ref)
    frac = float(np.mean(ok))
    return RateCheck(names, int(starts0.size), int(judged.size), frac,
                     passes=frac >= RATE_PASS_FRACTION)


@dataclass(frozen=True)
class TimingCheck:
    """(b): the lead's beats against a vetted train, after one constant offset."""

    offset_s: float
    fraction_within: float
    passes: bool


def timing_check(beats_s: npt.ArrayLike, vetted_s: npt.ArrayLike) -> TimingCheck:
    """At least :data:`TIMING_PASS_FRACTION` of the lead's beats within :data:`TIMING_TOL_S`.

    The offset is the median lead-minus-vetted lag over lead beats with a vetted beat
    within :data:`TIMING_SEARCH_S` (a fiducial on another lead sits at a constant lag);
    every lead beat then counts, within tolerance or not. No vetted beat: fails.
    """
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    v = np.sort(np.asarray(vetted_s, dtype=np.float64))
    if b.size == 0 or v.size == 0:
        return TimingCheck(float("nan"), 0.0, passes=False)
    j = np.searchsorted(v, b)
    prev, nxt = v[np.clip(j - 1, 0, v.size - 1)], v[np.clip(j, 0, v.size - 1)]
    lag = np.where(np.abs(b - prev) <= np.abs(nxt - b), b - prev, b - nxt)
    close = np.abs(lag) <= TIMING_SEARCH_S
    if not close.any():
        return TimingCheck(float("nan"), 0.0, passes=False)
    off = float(np.median(lag[close]))
    j = np.searchsorted(v, b - off)
    prev, nxt = v[np.clip(j - 1, 0, v.size - 1)], v[np.clip(j, 0, v.size - 1)]
    d = np.minimum(np.abs(b - off - prev), np.abs(nxt - (b - off)))
    frac = float(np.mean(d <= TIMING_TOL_S))
    return TimingCheck(off, frac, passes=frac >= TIMING_PASS_FRACTION)


@dataclass(frozen=True)
class MorphologyCheck:
    """(c): per raw contact, the lead-locked template SNR against noise-shuffled templates."""

    snr: dict[str, float]
    shuffled_max: dict[str, float]
    failing: tuple[str, ...]
    passes: bool


def morphology_check(beats_s: npt.ArrayLike, rec: Recording, *, seed: int = 0) -> MorphologyCheck:
    """Every raw contact must show the QRS at the lead's beats.

    Per raw nerve and stomach contact (10-150 Hz as task 05 prepares it), the template
    SNR (:func:`rpeaks._template_snr`) at the lead's beats must exceed the SNR at each
    of :data:`MORPH_SHUFFLES` shuffles of the same beats, each beat moved by a uniform
    :data:`MORPH_SHIFT_RR` of the median RR - same count, none beat-locked. The check
    passes only when every contact does.
    """
    fs = float(rec.fs)
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    rng = np.random.default_rng(seed)
    rr = float(np.median(np.diff(b))) if b.size > 1 else float("nan")
    snr, shuf = {}, {}
    for c in rec.channels:
        if c.role not in ("nerve", "stomach"):
            continue
        y, fsd = _prepare(np.asarray(rec.data[:, c.index], dtype=np.float64), fs, DETECT_BAND_HZ)
        s = float(_template_snr(y, b, fsd)) if b.size else float("nan")
        sh = []
        for _k in range(MORPH_SHUFFLES):
            if not np.isfinite(rr):
                break
            moved = b + rng.uniform(*MORPH_SHIFT_RR, b.size) * rr
            moved = moved[moved < y.size / fsd]
            sh.append(float(_template_snr(y, moved, fsd)))
        snr[c.name] = s
        shuf[c.name] = float(np.nanmax(sh)) if sh and np.isfinite(sh).any() else float("nan")
    failing = tuple(n for n in snr
                    if not (np.isfinite(snr[n]) and np.isfinite(shuf[n]) and snr[n] > shuf[n]))
    return MorphologyCheck(snr, shuf, failing, passes=bool(snr) and not failing)
