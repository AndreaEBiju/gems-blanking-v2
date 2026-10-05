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
A pair containing a DETACHED contact is not a candidate (ruling 2026-10-02 1): first-half
|g| < :data:`DETACHED_MAX_G` or sign agreement < :data:`DETACHED_MIN_AGREEMENT`
(:func:`detached_contacts`) - a contact not seeing the shared ground cannot cancel it.
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

**Independent cross-check** (ruling (d) 2, corrected by ruling 2026-10-02), on
references that do not use the lead:
(a) :func:`rate_check` - references from :func:`refined_autocorr_rate` (parabolic
sub-lag; the count gate is unchanged); minutes whose references disagree by more than
:data:`RATE_REF_SPREAD` are unassessable; too few assessable minutes is unassessable.
(b) :func:`timing_check` where a vetted train exists and itself passes the binding veto -
beat identity within :data:`TIMING_TOL_S`, >= 99% in both directions; trains that
disagree on more than :data:`TIMING_DISAGREE_FRACTION` of beats are resolved beat by beat
by :func:`resolve_disagreement`, and neither is trusted until then.
(c) :func:`morphology_check` on every non-detached contact, at least
:data:`MORPH_MIN_CONTACTS` on :data:`MORPH_MIN_SITES` sites.
A recording keeps a gain only if (a) and (c) pass, and (b) where it applies.

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
    "DETACHED_MAX_G",
    "DETACHED_MIN_AGREEMENT",
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
    "Resolution",
    "TimingCheck",
    "detached_contacts",
    "event_patterns",
    "gate_half",
    "ground_gains",
    "half_split",
    "hum_locked",
    "inject_median_pattern",
    "inject_real_patterns",
    "lead_signal",
    "morphology_check",
    "noncardiac_events",
    "pair_candidates",
    "rate_check",
    "rate_references",
    "reference_rate",
    "reference_windows",
    "refined_autocorr_rate",
    "resolve_disagreement",
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
TIMING_TOL_S: Final = 0.005
"""(b), ruling 2026-10-02 3: a beat matches within this after the offset - far below half
an RR, so a wrong beat cannot match; identity is the question, the veto judges precision."""
TIMING_PASS_FRACTION: Final = 0.99
"""(b): ... for at least this fraction of beats, in BOTH directions."""
TIMING_DISAGREE_S: Final = 0.020
TIMING_DISAGREE_FRACTION: Final = 0.05
"""(b): more than this fraction unmatched within :data:`TIMING_DISAGREE_S`, either
direction, means at least one train is wrong: neither is trusted until
:func:`resolve_disagreement` says which carries the QRS."""
TIMING_SEARCH_S: Final = 0.010
"""(b): the constant offset is the median lead-minus-vetted lag over pairs this close."""
MORPH_SHUFFLES: Final = 20
"""(c): noise-shuffled templates per contact; the contact must beat every one."""
MORPH_SHIFT_RR: Final[tuple[float, float]] = (0.25, 0.75)
"""(c): a shuffled template moves each beat by this range of the median RR - the same
count of windows, none locked to a beat."""
MORPH_MIN_CONTACTS: Final = 4
MORPH_MIN_SITES: Final = 2
"""(c), ruling 2026-10-02 1: at least this many non-detached contacts on at least this
many sites, or the recording is unassessable (no gain)."""
DETACHED_MAX_G: Final = 0.2
DETACHED_MIN_AGREEMENT: Final = 0.75
"""Ruling 2026-10-02 1: a contact whose first-half |g| is under :data:`DETACHED_MAX_G` or
whose sign agreement with g is under :data:`DETACHED_MIN_AGREEMENT` is not seeing the
shared ground - detached. From the ground physics, not from the cross-check."""
RATE_REF_SPREAD: Final = 0.05
"""(a), ruling 2026-10-02 2: a minute whose clear references disagree among themselves by
more than this ((max - min) / median) is unassessable - nothing to judge against."""
RATE_MIN_ASSESSABLE: Final = 0.5
"""(a): assessable minutes must be at least this fraction of the minutes with any clear
reference, or the recording is unassessable (no gain)."""
MAINS_HZ: Final = hc.MAINS_HZ
HUM_LOCK_BPM: Final = hc.HUM_LOCK_BPM
"""Defined once, in ``hr_channel`` (the count gate shares them)."""
QRS_HALF_S: Final = 0.020
"""Template windows for :func:`resolve_disagreement`: +/- this around a beat."""


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


def pair_candidates(rec: Recording, detached: frozenset[str] = frozenset()) -> list[PairLead]:
    """Every left-neck contact minus every right-neck contact, and the reverse.

    Contacts are the raw nerve channels with a ``cuff_id`` and a ``contact_index``;
    stomach and aux channels never enter. 18 leads when both cuffs have three contacts.
    A pair containing a ``detached`` contact is not a candidate (ruling 2026-10-02 1).
    """
    left = [c for c in _neck(rec, "L") if c.name not in detached]
    right = [c for c in _neck(rec, "R") if c.name not in detached]
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


def ground_gains(events: Events, keep: npt.ArrayLike) -> dict[str, tuple[float, float]]:
    """Return ``{channel: (g, sign agreement)}`` over the non-cardiac events with a pattern.

    ``g`` is the median signed relative gain (:func:`event_patterns`); the agreement is
    the fraction of those events whose gain has g's sign. NaN when no event qualifies.
    """
    rel = event_patterns(events)
    ok = np.asarray(keep, dtype=bool) & np.isfinite(rel).all(axis=1)
    if not ok.any():
        return {n: (float("nan"), float("nan")) for n in events.channels}
    g = np.median(rel[ok], axis=0)
    agree = np.mean(np.sign(rel[ok]) == np.sign(g)[None, :], axis=0)
    return {n: (float(g[k]), float(agree[k])) for k, n in enumerate(events.channels)}


def detached_contacts(events: Events, keep: npt.ArrayLike) -> frozenset[str]:
    """Return the contacts not seeing the shared ground.

    Detached: |g| < :data:`DETACHED_MAX_G` or sign agreement < :data:`DETACHED_MIN_AGREEMENT`.
    With no event to measure g on, none is called detached (no evidence either way).
    """
    return frozenset(n for n, (g, a) in ground_gains(events, keep).items()
                     if np.isfinite(g) and (abs(g) < DETACHED_MAX_G or a < DETACHED_MIN_AGREEMENT))


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


def noncardiac_events(rec: Recording) -> tuple[Events, npt.NDArray[np.bool_]]:
    """Return ``(events, non-cardiac selection)`` as ``gated_selection`` takes them.

    The common-mode events of ``rec`` and, per event, whether it is clear of task 05's beats
    on the top-ranked HR candidate - the pool the ground gains (:func:`ground_gains`) and so
    the detached rule (:func:`detached_contacts`) are measured on. One construction site:
    the half-split protocol and task 18's contact exclusion both call it.
    """
    ev = find_events(rec)
    cand = hc.hr_candidates(rec)
    t05 = {c.name: detect_rpeaks(np.asarray(cand.data[:, c.index], dtype=np.float64), float(rec.fs))
           for c in cand.channels}
    try:
        first, _t = rank_hr_channels(cand, t05)
    except ValueError:
        first = max(t05, key=lambda k: t05[k].n_beats)
    return ev, hc._noncardiac(ev, t05[first].t_s)


def _prepare_half(rec: Recording, seed: int) -> _Half:
    """Events, the non-cardiac selection (as ``gated_selection``) and both injections."""
    ev, keep = noncardiac_events(rec)
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
    detached: frozenset[str]
    """Detached contacts, from the first half's ground gains (ruling 2026-10-02 1)."""
    second: _Half = field(repr=False)
    """The second half with its events and injections - what the cross-check runs on."""

    @property
    def second_rec(self) -> Recording:
        """The second half's recording."""
        return self.second.rec

    @property
    def second_events(self) -> Events:
        """The second half's events."""
        return self.second.events

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
    detached = detached_contacts(h1.events, h1.keep)
    pairs1, singles1 = gate_half(h1, pairs=pair_candidates(a, detached))
    cp, cs = select(pairs1), select(singles1)
    del h1
    h2 = _prepare_half(b, seed)
    sp = (gate_half(h2, pairs=pair_candidates(b, detached), only=(cp.lead, cp.detector))[0][0]
          if cp is not None else None)
    ss = gate_half(h2, pairs=[], only=(cs.lead, cs.detector))[1][0] if cs is not None else None
    return HalfSplit(pairs1, singles1, cp, cs, sp, ss, detached, h2)


# ---------------------------------------------------------------------------
# the independent cross-check (ruling (d) 2)
# ---------------------------------------------------------------------------


def refined_autocorr_rate(x: npt.ArrayLike, fs: float,
                          notch_hz: tuple[float, ...] = ()) -> tuple[F64, F64]:
    """:func:`hr_channel.autocorr_rate` with a parabolic sub-lag refinement of its peak.

    The same signal, windows, lag range and clarity rule - so the same minutes are clear
    and NaN - but the rate comes from the vertex of the parabola through the peak lag and
    its two neighbours. For the cross-check's references only (ruling 2026-10-02 2); the
    count gate keeps :func:`hr_channel.autocorr_rate` unchanged. ``notch_hz``: zero-phase
    notches applied after decimation, before the band-pass and the rectification.
    """
    starts, _durs, _rate, fine = hc._ac_windows(x, fs, notch_hz, partial=False)
    return starts, fine


def hum_locked(bpm: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Per rate, whether it sits within :data:`HUM_LOCK_BPM` of 7200 / m - single minutes.

    A DIAGNOSTIC since ruling 2026-10-02 (e) 2: a single minute on the grid is often the heart
    itself; the lock that makes a minute unclear is :func:`hr_channel.hum_locked_persistent`.

    Rectified 60 Hz repeats every 1/120 s, so its autocorrelation peaks at lags m / 120 s -
    rates of 7200 / m bpm (m = 15, 17, 18, 20 measured on animal A's stomach references:
    480, 423.5, 400, 360). NaN is not locked.
    """
    b = np.asarray(bpm, dtype=np.float64)
    out = np.zeros(b.shape, dtype=bool)
    ok = np.isfinite(b) & (b > 0)
    m = np.round(7200.0 / b[ok])
    out[ok] = (m >= 1) & (np.abs(b[ok] - 7200.0 / np.maximum(m, 1)) <= HUM_LOCK_BPM)
    return out


def reference_rate(x: npt.ArrayLike, fs: float) -> tuple[F64, F64]:
    """Return a cross-check rate reference: mains-notched, refined, hum-locked minutes unclear.

    Ruling 2026-10-02 (d) 1 / (e) 2: :func:`refined_autocorr_rate` with notches at
    :func:`hr_channel.mains_harmonics`, then every minute in a persistent lock
    (:func:`hr_channel.hum_locked_persistent`) is set to NaN. A single grid minute is not a
    lock. The count gate has its own (unrefined) notched rate, :func:`hr_channel.autocorr_rate`.
    """
    starts, bpm = refined_autocorr_rate(x, fs, notch_hz=hc.mains_harmonics())
    return starts, np.where(hc.hum_locked_persistent(bpm), np.nan, bpm)


def reference_windows(x: npt.ArrayLike, fs: float) -> tuple[F64, F64, F64]:
    """Return ``(starts s, durations s, bpm)``: :func:`reference_rate` plus the trailing window.

    Ruling 2026-10-02 (g) 3: the full minutes are exactly :func:`reference_rate`'s; a trailing
    remainder of at least ``hr_channel.PARTIAL_MIN_S`` is one more window, refined, notched
    and lock-tested the same way (``hr_channel._locked``).
    """
    starts, durs, _rate, fine = hc._ac_windows(x, fs, hc.mains_harmonics(), partial=True)
    return starts, durs, np.where(hc._locked(fine, durs), np.nan, fine)


@dataclass(frozen=True)
class RateCheck:
    """(a): the lead's per-minute rate against references that do not use it."""

    references: tuple[str, ...]
    minutes: int
    minutes_with_reference: int
    minutes_assessable: int
    """Minutes whose clear references agree among themselves within :data:`RATE_REF_SPREAD`."""
    fraction_within: float
    assessable: bool
    passes: bool
    bad_minutes_s: tuple[float, ...] = ()
    """Start times of the assessable minutes where the lead disagrees - the per-minute (a)
    of ruling 2026-10-02 (f) 2 (a minute that is not assessable is never listed)."""


References = tuple[tuple[str, ...], F64, F64, F64]
"""``(names, window starts s, window durations s, rates (n_references x n_windows) bpm)`` for
:func:`rate_check`."""


def rate_references(rec: Recording, lead: PairLead,
                    detached: frozenset[str] = frozenset()) -> References:
    """Return the references :func:`rate_check` judges ``lead`` against (see there).

    They depend on the lead only through the cuffs it uses, so one result serves every
    lead on the same cuffs.
    """
    fs = float(rec.fs)
    used = {c.cuff_id for c in rec.channels if c.name in (lead.plus, lead.minus)}
    refs: list[tuple[str, F64]] = [
        (c.name, np.asarray(rec.data[:, c.index], dtype=np.float64)) for c in rec.channels
        if c.name not in detached
        and ((c.role == "stomach") or (c.role == "nerve" and c.cuff_id not in used))]
    signals, _w = build_derivations(rec)
    if "stomach_ref" in signals:
        refs.append(("stomach_ref", np.asarray(signals["stomach_ref"], dtype=np.float64)))
    rates = []
    starts0: F64 | None = None
    durs0: F64 = np.zeros(0)
    for _name, x in refs:
        starts, durs, bpm = reference_windows(x, fs)
        if starts0 is None:
            starts0, durs0 = starts, durs
        assert np.array_equal(starts, starts0), "references share one minute grid"
        rates.append(bpm)
    names = tuple(n for n, _x in refs)
    if starts0 is None or starts0.size == 0:
        return names, np.zeros(0), np.zeros(0), np.zeros((len(names), 0))
    return names, starts0, durs0, np.vstack(rates)


def rate_check(beats_s: npt.ArrayLike, rec: Recording, lead: PairLead,
               detached: frozenset[str] = frozenset(), *,
               refs: References | None = None) -> RateCheck:
    """Per minute, the lead's rate against the median clear reference.

    References: :func:`reference_rate` on every raw channel of a site the lead
    does not use (a left-right pair uses both necks, so the stomach contacts) and on
    ``stomach_ref``. A minute with a clear reference is ASSESSABLE when its clear
    references agree among themselves within :data:`RATE_REF_SPREAD`; there the lead
    agrees when its beats per minute are within :data:`RATE_TOL` of their median. The
    recording is assessable when assessable minutes are at least
    :data:`RATE_MIN_ASSESSABLE` of the minutes with any clear reference, and passes when
    it is assessable and at least :data:`RATE_PASS_FRACTION` of assessable minutes agree.
    Unassessable never passes.

    POST-H AMENDMENT (ruling 2026-10-02 (c) 3): a ``detached`` contact is not a
    reference - a contact that does not see the shared ground is not a physiological
    reference, as it is left out of (c). Decided after H; it changes no H outcome.
    ``stomach_ref`` stays: it is a derived signal, not a contact.

    ``refs``: :func:`rate_references` for this lead, when already computed.

    Ruling 2026-10-02 (g) 3: a trailing partial window (:func:`reference_windows`) is judged
    the same way, its count scaled to its duration, and can be listed in ``bad_minutes_s``;
    the recording-level result - the one a mask-grade pair must pass - reads the full
    minutes only, exactly as before.
    """
    names, starts0, durs0, stack = rate_references(rec, lead, detached) if refs is None else refs
    full = durs0 == hc.AC_WINDOW_S
    if not full.any():
        return RateCheck(names, 0, 0, 0, 0.0, assessable=False, passes=False)
    clear = np.isfinite(stack).any(axis=0)
    any_clear = np.flatnonzero(clear & full)
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    ok = []
    bad: list[float] = []
    for i in np.flatnonzero(clear):
        col = stack[:, i][np.isfinite(stack[:, i])]
        ref = float(np.median(col))
        if (col.max() - col.min()) > RATE_REF_SPREAD * ref:
            continue  # the references disagree: nothing to judge against
        s, d = starts0[i], durs0[i]
        per_min = np.count_nonzero((b >= s) & (b < s + d)) * 60.0 / d
        agree = abs(per_min - ref) <= RATE_TOL * ref
        if full[i]:
            ok.append(agree)
        if not agree:
            bad.append(float(s))
    n_any = int(any_clear.size)
    assessable = bool(n_any > 0 and len(ok) >= RATE_MIN_ASSESSABLE * n_any)
    frac = float(np.mean(ok)) if ok else 0.0
    return RateCheck(names, int(full.sum()), n_any, len(ok), frac, assessable=assessable,
                     passes=bool(assessable and frac >= RATE_PASS_FRACTION),
                     bad_minutes_s=tuple(bad))


@dataclass(frozen=True)
class TimingCheck:
    """(b): the lead's beats against a vetted train, after one constant offset."""

    offset_s: float
    matched_lead: float
    """Fraction of the lead's beats with a vetted beat within :data:`TIMING_TOL_S`."""
    matched_vetted: float
    """Fraction of the vetted beats with a lead beat within :data:`TIMING_TOL_S`."""
    sd_matched_s: float
    """SD of the matched differences - reported, not a gate."""
    unmatched_lead_20ms: float
    unmatched_vetted_20ms: float
    disagree: bool
    """More than :data:`TIMING_DISAGREE_FRACTION` unmatched within :data:`TIMING_DISAGREE_S`,
    either direction: at least one train is wrong (:func:`resolve_disagreement`)."""
    passes: bool


def _signed_nearest(p: F64, q: F64) -> F64:
    """Per element of ``p``, its signed difference to the nearest element of ``q`` (sorted)."""
    j = np.searchsorted(q, p)
    prev, nxt = q[np.clip(j - 1, 0, q.size - 1)], q[np.clip(j, 0, q.size - 1)]
    return np.where(np.abs(p - prev) <= np.abs(nxt - p), p - prev, p - nxt)


def timing_check(beats_s: npt.ArrayLike, vetted_s: npt.ArrayLike) -> TimingCheck:
    """Beat identity against a vetted train: >= 99% matched within 5 ms, both directions.

    The offset is the median lead-minus-vetted lag over lead beats with a vetted beat
    within :data:`TIMING_SEARCH_S` (a fiducial on another lead sits at a constant lag).
    After it, a lead beat is matched when a vetted beat is within :data:`TIMING_TOL_S`,
    and a vetted beat when a lead beat is. Passes when at least
    :data:`TIMING_PASS_FRACTION` of each train is matched. Also reported: the SD of the
    matched differences (not a gate) and the disagreement flag. No beat: fails.
    """
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    v = np.sort(np.asarray(vetted_s, dtype=np.float64))
    nan = float("nan")
    if b.size == 0 or v.size == 0:
        return TimingCheck(nan, 0.0, 0.0, nan, 1.0, 1.0, disagree=True, passes=False)
    lag = _signed_nearest(b, v)
    close = np.abs(lag) <= TIMING_SEARCH_S
    if not close.any():
        return TimingCheck(nan, 0.0, 0.0, nan, 1.0, 1.0, disagree=True, passes=False)
    off = float(np.median(lag[close]))
    d_lead = _signed_nearest(b - off, v)
    d_vet = _signed_nearest(v, b - off)
    m_lead = float(np.mean(np.abs(d_lead) <= TIMING_TOL_S))
    m_vet = float(np.mean(np.abs(d_vet) <= TIMING_TOL_S))
    matched = d_lead[np.abs(d_lead) <= TIMING_TOL_S]
    sd = float(np.std(matched, ddof=1)) if matched.size > 1 else nan
    u_lead = float(np.mean(np.abs(d_lead) > TIMING_DISAGREE_S))
    u_vet = float(np.mean(np.abs(d_vet) > TIMING_DISAGREE_S))
    return TimingCheck(off, m_lead, m_vet, sd, u_lead, u_vet,
                       disagree=bool(max(u_lead, u_vet) > TIMING_DISAGREE_FRACTION),
                       passes=bool(min(m_lead, m_vet) >= TIMING_PASS_FRACTION))


@dataclass(frozen=True)
class Resolution:
    """:func:`resolve_disagreement`: which train's unmatched beats carry the QRS."""

    contacts: tuple[str, ...]
    n_matched: int
    floor: float
    """1st percentile of the matched beats' template correlations."""
    lead_unmatched: int
    lead_unmatched_qrs: float
    """Fraction of the lead's unmatched beats whose correlation reaches the floor."""
    other_unmatched: int
    other_unmatched_qrs: float


def resolve_disagreement(lead_s: npt.ArrayLike, other_s: npt.ArrayLike, rec: Recording, *,
                         detached: frozenset[str] = frozenset(),
                         offset_s: float = 0.0) -> Resolution:
    """Beat by beat, which train's unmatched beats carry the QRS (ruling 2026-10-02 3).

    The template is the mean multichannel window (+/-:data:`QRS_HALF_S`, 10-150 Hz as
    task 05 prepares it, every non-detached raw contact) over the beats both trains agree
    on (within :data:`TIMING_TOL_S` after ``offset_s``). Each beat's correlation with it
    is the Pearson correlation of the concatenated windows. A beat unmatched within
    :data:`TIMING_DISAGREE_S` carries the QRS when its correlation reaches the floor -
    the 1st percentile over the matched beats. Reported, not decided: the fractions say
    which train is right.
    """
    fs = float(rec.fs)
    b = np.sort(np.asarray(lead_s, dtype=np.float64))
    o = np.sort(np.asarray(other_s, dtype=np.float64)) + offset_s  # into the lead's frame
    contacts = [c for c in rec.channels
                if c.role in ("nerve", "stomach") and c.name not in detached]
    ys = []
    fsd = fs
    for c in contacts:
        y, fsd = _prepare(np.asarray(rec.data[:, c.index], dtype=np.float64), fs, DETECT_BAND_HZ)
        ys.append(y)
    mat = np.vstack(ys)
    h = int(round(QRS_HALF_S * fsd))

    def windows(t: F64) -> tuple[F64, npt.NDArray[np.bool_]]:
        i = np.round(t * fsd).astype(np.int64)
        ok = (i - h >= 0) & (i + h + 1 <= mat.shape[1])
        if not ok.any():
            return np.zeros((0, 1)), ok
        return np.stack([mat[:, k - h:k + h + 1].ravel() for k in i[ok]]), ok

    def corr(stack: F64, tpl: F64) -> F64:
        a = stack - stack.mean(axis=1, keepdims=True)
        t = tpl - tpl.mean()
        return np.asarray((a @ t) / (np.linalg.norm(a, axis=1) * np.linalg.norm(t) + 1e-300))

    d_b = _signed_nearest(b, o) if o.size else np.full(b.size, np.inf)
    d_o = _signed_nearest(o, b) if b.size else np.full(o.size, np.inf)
    matched_t = b[np.abs(d_b) <= TIMING_TOL_S]
    m_stack, _ok = windows(matched_t)
    if m_stack.shape[0] < 2:  # noqa: PLR2004 - a template needs beats
        return Resolution(tuple(c.name for c in contacts), int(m_stack.shape[0]), float("nan"),
                          int(np.sum(np.abs(d_b) > TIMING_DISAGREE_S)), float("nan"),
                          int(np.sum(np.abs(d_o) > TIMING_DISAGREE_S)), float("nan"))
    tpl = m_stack.mean(axis=0)
    floor = float(np.percentile(corr(m_stack, tpl), 1))

    def qrs_fraction(t: F64) -> tuple[int, float]:
        st, _o = windows(t)
        if st.shape[0] == 0:
            return int(t.size), float("nan")
        return int(t.size), float(np.mean(corr(st, tpl) >= floor))
    nl, fl = qrs_fraction(b[np.abs(d_b) > TIMING_DISAGREE_S])
    no, fo = qrs_fraction(o[np.abs(d_o) > TIMING_DISAGREE_S])
    return Resolution(tuple(c.name for c in contacts), int(m_stack.shape[0]), floor, nl, fl, no, fo)


@dataclass(frozen=True)
class MorphologyCheck:
    """(c): per raw contact, the lead-locked template SNR against noise-shuffled templates."""

    snr: dict[str, float]
    shuffled_max: dict[str, float]
    failing: tuple[str, ...]
    excluded: tuple[str, ...]
    """Detached contacts, left out (ruling 2026-10-02 1)."""
    assessable: bool
    """At least :data:`MORPH_MIN_CONTACTS` non-detached contacts, :data:`MORPH_MIN_SITES` sites."""
    passes: bool


def morphology_check(beats_s: npt.ArrayLike, rec: Recording, *,
                     detached: frozenset[str] = frozenset(), seed: int = 0) -> MorphologyCheck:
    """Every non-detached raw contact must show the QRS at the lead's beats.

    Per raw nerve and stomach contact not in ``detached`` (10-150 Hz as task 05 prepares
    it), the template SNR (:func:`rpeaks._template_snr`) at the lead's beats must exceed
    the SNR at each of :data:`MORPH_SHUFFLES` shuffles of the same beats, each beat moved
    by a uniform :data:`MORPH_SHIFT_RR` of the median RR - same count, none beat-locked.
    Assessable only with at least :data:`MORPH_MIN_CONTACTS` such contacts on at least
    :data:`MORPH_MIN_SITES` sites (left neck, right neck, stomach); passes when
    assessable and every such contact shows the QRS.
    """
    fs = float(rec.fs)
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    rng = np.random.default_rng(seed)
    rr = float(np.median(np.diff(b))) if b.size > 1 else float("nan")
    snr, shuf = {}, {}
    use = [c for c in rec.channels if c.role in ("nerve", "stomach") and c.name not in detached]
    sites = {c.cuff_id if c.role == "nerve" else "stomach" for c in use}
    excluded = tuple(sorted(c.name for c in rec.channels
                            if c.role in ("nerve", "stomach") and c.name in detached))
    for c in use:
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
    assessable = len(use) >= MORPH_MIN_CONTACTS and len(sites) >= MORPH_MIN_SITES
    return MorphologyCheck(snr, shuf, failing, excluded, assessable,
                           passes=bool(assessable and not failing))
