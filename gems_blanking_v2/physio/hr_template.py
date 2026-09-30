"""A template-resolved HR candidate train (rulings 2026-09-30 (f) and (g)).

One extra candidate train per channel, to be gated by the SAME gates as task 05's
detector and the ``findpeaks`` replica (count gate, transient veto, plausibility). The
existing detectors are unchanged; this builds from their peaks. It answers the
diagnosis on 9 recordings: capture (the injected transient becomes the beat) is the
veto harm on raw contacts, findpeaks adds extras at its 100 ms spacing floor, and task
05 misses beats.

Two signals: ``y`` - the channel in 10-150 Hz at ~2 kHz, exactly as task 05 prepares it
(fiducials are its positive peaks) - and ``ys`` - the same in :data:`SCORE_BAND_HZ`,
where every template and correlation is computed ((g) 1: in 10-150 Hz a 1.2 ms
transient is 10-20 ms wide and cannot be excised; here it stays ~1 ms).

1. **Candidates** - the union of task 05's beats and the replica's, snapped to the local
   maximum of ``y`` within :data:`SNAP_S`, merged within :data:`MERGE_S`.
2. **Seed** ((g) 4) - task 05's beats when at most :data:`SEED_SHORT_MAX` of its
   intervals are short (under :data:`SHORT_FRACTION` x the autocorrelation RR);
   otherwise the candidates split into their two interleaved subtrains and the one with
   the higher template SNR is kept. SNRs within :data:`AMBIGUOUS_SNR` of each other:
   ambiguous, no train.
3. **Clean** ((g) 3) - seed beats whose both adjacent intervals are within
   :data:`CLEAN_RR_TOL` of that minute's autocorrelation RR, in clear minutes only.
   The autocorrelation (``hr_channel.autocorr_rate``) is the count gate's own
   independent reference, so a train's extras cannot define what is clean.
4. **Template** - the MEDIAN of the clean beats' +/-40 ms windows of ``ys``.
5. **Masked scores** ((g) 2) - the +/-:data:`MASK_HALF_S` of EVERY common-mode event in
   the window is excised from candidate and template alike, and the best correlation
   over alignment shifts of +/-:data:`ALIGN_S` is taken. Excision leaving less than
   :data:`MIN_KEPT_FRACTION` of the window: the score is undefined, the candidate is
   dropped and its gap re-searched. Scoring only - nothing is written (invariant 8).
6. **Floor** ((f) 3, (g) 3) - the :data:`FLOOR_QUANTILE` quantile of masked scores over
   clean beats that are NOT suspects, each scored exactly as a candidate is (its own
   events excised, plus the candidate's). Fewer than :data:`MIN_FLOOR_BEATS` such beats:
   no train. There is no fallback to suspects - that is where captures hide.
7. **Every candidate meets the floor** ((h) 2) - its score, masked or not, must reach
   the floor built the same way, or it is dropped and its gap re-searched under the same
   floor. About 1% of real beats fall under a 1st-percentile floor by construction; they
   leave tagged gaps (:attr:`TemplateTrain.gap_after`), never a false beat. A candidate
   within :data:`SUSPECT_S` of an event takes its fiducial from the masked alignment.
8. **Refractory** - beats closer than ``k`` x the running median RR conflict and the
   higher template score stays, never the larger peak. The running median (centred,
   +/-:data:`RUNNING_MEDIAN_HALF_S`) is over template-matching beats only, so extras
   cannot halve it. A beat is template-matching when its score reaches the floor made
   the same way ((g) 2): the unmasked floor when no event is in its window, the masked
   floor when one is. Against the unmasked floor alone, a channel whose QRS edge
   triggers the event finder had almost no matching beats, a median several RR long,
   and lost 75% of its beats to the refractory (fixture, 2026-09-30). ``k`` in
   :data:`K_CHOICES`.
9. **Re-search** (Andrea's rule) - a gap >= :data:`GAP_MULTIPLE` x the running median is
   searched near each expected time (+/-:data:`RESEARCH_FRACTION` of RR) with RELAXED
   height and width (the development multipliers on the clean beats' 1st-percentile
   prominence and width range); a peak must still reach the floor. Nothing is
   inserted: an unfilled gap stays and is tagged.

OUTSIDE THE GENERATION HASH: nothing in the detection chain imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks, peak_prominences, peak_widths

from gems_blanking_v2.physio.rpeaks import (
    DETECT_BAND_HZ,
    TEMPLATE_HALF_WIDTH_S,
    _prepare,
    _template_snr,
    detect_rpeaks,
)

__all__ = [
    "ALIGN_S",
    "AMBIGUOUS_SNR",
    "CLEAN_RR_TOL",
    "FLOOR_QUANTILE",
    "GAP_MULTIPLE",
    "K_CHOICES",
    "MASK_HALF_S",
    "MAX_FIDUCIAL_OFFSET_S",
    "MERGE_S",
    "MIN_FLOOR_BEATS",
    "MIN_KEPT_FRACTION",
    "RESEARCH_FRACTION",
    "RUNNING_MEDIAN_HALF_S",
    "SCORE_BAND_HZ",
    "SEED_SHORT_MAX",
    "SHORT_FRACTION",
    "SNAP_S",
    "SUSPECT_S",
    "Prepared",
    "TemplateParams",
    "TemplateTrain",
    "prepare",
    "resolve",
    "running_median_rr",
    "template_train",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B = npt.NDArray[np.bool_]
S = npt.NDArray[np.str_]
Mask = tuple[int, ...]

K_CHOICES: Final[tuple[float, ...]] = (0.6, 0.65, 0.7, 0.75, 0.8)
"""(f) 4: the refractory fraction is chosen from these on A/B, capped at 0.8."""
SCORE_BAND_HZ: Final[tuple[float, float]] = (10.0, 900.0)
"""(g) 1: templates and correlations. Fixture: beats hit by a transient kept 64/71 here,
0/71 in 10-150 Hz."""
CLEAN_RR_TOL: Final = 0.10
SHORT_FRACTION: Final = 0.75
SEED_SHORT_MAX: Final = 0.25
AMBIGUOUS_SNR: Final = 0.20
MIN_FLOOR_BEATS: Final = 50
MIN_KEPT_FRACTION: Final = 0.5
RUNNING_MEDIAN_HALF_S: Final = 10.0
SUSPECT_S: Final = 0.002
MASK_HALF_S: Final = 0.002
ALIGN_S: Final = 0.001
FLOOR_QUANTILE: Final = 0.01
FLOOR_SAMPLE: Final = 1000
"""Non-suspect clean beats the floor is estimated on (seeded) - tractable when dense."""
FLOOR_SEED: Final = 20260930
SNAP_S: Final = 0.001
MERGE_S: Final = 0.005
GAP_MULTIPLE: Final = 1.5
RESEARCH_FRACTION: Final = 0.15
MAX_FIDUCIAL_OFFSET_S: Final = 0.0005
"""(f) 2: alignment fiducials must sit within this of peak fiducials, on average."""


@dataclass(frozen=True)
class TemplateParams:
    """The development parameters, fixed on A/B before the held-out run ((f) 7)."""

    k: float
    height_mult: float
    width_mult: float

    def __post_init__(self) -> None:
        """Refuse a value outside the ruled ranges."""
        if self.k not in K_CHOICES:
            msg = f"k={self.k} is not one of {K_CHOICES}"
            raise ValueError(msg)
        for name in ("height_mult", "width_mult"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                msg = f"{name}={v} must relax, so lie in (0, 1]"
                raise ValueError(msg)


@dataclass(frozen=True)
class TemplateTrain:
    """The train and what each step did; times in seconds on the channel's timeline."""

    t_s: F64
    kind: S = field(repr=False)
    """Per beat: ``peak``, ``aligned`` (a kept suspect: masked alignment) or ``researched``."""
    no_train: str = ""
    """Why this channel yields no train (empty when it yields one)."""
    seed: str = ""
    """``task05`` or ``split`` ((g) 4)."""
    split_snr: tuple[float, float] | None = None
    """The two subtrains' template SNRs, the kept one first, when the seed was a split."""
    split_jitter_s: float = float("nan")
    """SD of the RR difference between the two subtrains, when split."""
    gaps_s: list[tuple[float, float]] = field(default_factory=list)
    """Every interval the re-search could not fill - tagged, never filled ((h) 3)."""
    counts: dict[str, int] = field(default_factory=dict)
    fiducial_offset_s: float = float("nan")
    """Mean signed (alignment - peak) fiducial over clean beats; must be under 0.5 ms."""
    floor: float = float("nan")
    """The unmasked floor."""

    @property
    def gap_after(self) -> npt.NDArray[np.bool_]:
        """Per beat: whether the interval after it spans a tagged gap ((h) 3).

        HRV must not read such an interval as one long real RR.
        """
        starts = np.asarray([a for a, _b in self.gaps_s], dtype=np.float64)
        return np.isin(self.t_s, starts)  # a gap starts at a beat, the same float


@dataclass
class _Seed:
    idx: I64
    kind: str
    snr: tuple[float, float] | None = None
    jitter: float = float("nan")
    ambiguous: str = ""


def running_median_rr(idx: I64, fs: float, at: I64 | None = None,
                      half_s: float = RUNNING_MEDIAN_HALF_S) -> F64:
    """Return, per time in ``at`` (default ``idx``), the median of ``idx``'s intervals.

    Only intervals whose midpoints lie within +/-``half_s`` count. Centred, never causal;
    NaN where none reach.
    """
    at = idx if at is None else at
    if idx.size < 2:  # noqa: PLR2004
        return np.full(at.size, np.nan)
    t = idx / fs
    rr = np.diff(t)
    mid = 0.5 * (t[1:] + t[:-1])
    ta = at / fs
    lo = np.searchsorted(mid, ta - half_s)
    hi = np.searchsorted(mid, ta + half_s)
    return np.array([np.median(rr[a:b]) if b > a else np.nan for a, b in zip(lo, hi, strict=True)])


def _windows(y: F64, idx: I64, half: int) -> tuple[F64, B]:
    ok = (idx - half >= 0) & (idx + half + 1 <= y.size)
    if not ok.any():
        return np.zeros((0, 2 * half + 1)), ok
    return np.asarray([y[i - half:i + half + 1] for i in idx[ok]]), ok


def _corr_rows(w: F64, tmpl: F64, keep: B) -> F64:
    """Pearson correlation of each row with the template, over the kept columns only."""
    a = w[:, keep] - w[:, keep].mean(axis=1, keepdims=True)
    b = tmpl[keep] - tmpl[keep].mean()
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(den > 0, (a @ b) / den, 0.0), dtype=np.float64)


def _masks(ev: I64, idx: I64, reach: int) -> list[Mask]:
    """Per candidate: the offsets (samples) of every event within ``reach`` of it."""
    if ev.size == 0:
        return [()] * idx.size
    lo = np.searchsorted(ev, idx - reach)
    hi = np.searchsorted(ev, idx + reach, side="right")
    return [tuple(int(e) - int(i) for e in ev[a:b]) for i, a, b in zip(idx, lo, hi, strict=True)]


def _is_suspect(mask: Mask, r: int) -> bool:
    return any(abs(o) <= r for o in mask)


class _Scorer:
    """Masked, aligned template correlations in ``ys`` and the floors they meet."""

    def __init__(self, ys: F64, fs: float, clean: I64, ev: I64, sus_r: int) -> None:
        self.y = ys
        self.half = max(int(round(TEMPLATE_HALF_WIDTH_S * fs)), 1)
        self.m = max(int(round(MASK_HALF_S * fs)), 1)
        self.a = max(int(round(ALIGN_S * fs)), 0)
        self.reach = self.half + self.a + self.m
        w, ok = _windows(ys, clean, self.half)
        self.template = np.median(w, axis=0) if w.shape[0] else np.zeros(2 * self.half + 1)
        self.clean = np.asarray(clean[ok], dtype=np.int64)
        own = _masks(ev, self.clean, self.reach)
        nonsus = [j for j, o in enumerate(own) if not _is_suspect(o, sus_r)]
        if len(nonsus) > FLOOR_SAMPLE:
            rng = np.random.default_rng(FLOOR_SEED)
            nonsus = sorted(rng.choice(nonsus, FLOOR_SAMPLE, replace=False).tolist())
        self.floor_rows = self.clean[np.asarray(nonsus, dtype=np.int64)]
        self._own = [own[j] for j in nonsus]
        self._floors: dict[Mask, float] = {}

    def keep(self, mask: Mask) -> B:
        """Return the window columns left after excising every event in ``mask``."""
        k = np.ones(2 * self.half + 1, bool)
        for o in mask:
            c = self.half + o
            k[max(c - self.m, 0):max(c + self.m + 1, 0)] = False
        return k

    def score(self, idx: I64, masks: list[Mask]) -> tuple[F64, I64]:
        """Return the best correlation over alignment shifts and the shift; NaN if undefined.

        (g) 2: a score exists only while excision leaves :data:`MIN_KEPT_FRACTION` of the
        window, at the fiducial the score is taken at.
        """
        best = np.full(idx.size, -np.inf)
        shift = np.zeros(idx.size, dtype=np.int64)
        for s in range(-self.a, self.a + 1):
            w, ok = _windows(self.y, idx + s, self.half)
            full = np.full(idx.size, -np.inf)
            if ok.any():
                pos = np.flatnonzero(ok)
                vals = np.full(pos.size, -np.inf)
                groups: dict[Mask, list[int]] = {}
                for j, p in enumerate(pos):
                    groups.setdefault(tuple(o - s for o in masks[p]), []).append(j)
                for key, js in groups.items():
                    kp = self.keep(key)
                    if kp.mean() >= MIN_KEPT_FRACTION:
                        vals[js] = _corr_rows(w[js], self.template, kp)
                full[pos] = vals
            better = full > best
            best[better], shift[better] = full[better], s
        best[~np.isfinite(best)] = np.nan
        return best, shift

    def floor(self, mask: Mask) -> float:
        """Return the FLOOR_QUANTILE of non-suspect clean beats scored under ``mask`` too."""
        if mask not in self._floors:
            both = [tuple(sorted(set(o) | set(mask))) for o in self._own]
            sc, _s = self.score(self.floor_rows, both)
            sc = sc[np.isfinite(sc)]
            self._floors[mask] = float(np.quantile(sc, FLOOR_QUANTILE)) if sc.size else float("inf")
        return self._floors[mask]


def _snap(y: F64, idx: I64, r: int) -> I64:
    out = [max(i - r, 0) + int(np.argmax(y[max(i - r, 0):min(i + r + 1, y.size)])) for i in idx]
    return np.asarray(out, dtype=np.int64)


def _merge(y: F64, idx: I64, r: int) -> I64:
    kept: list[int] = []
    for i in np.sort(idx):
        if kept and i - kept[-1] <= r:
            if y[i] > y[kept[-1]]:
                kept[-1] = int(i)
            continue
        kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def _ac_rr(starts: F64, bpm: F64, idx: I64, fd: float, window_s: float) -> F64:
    """Per beat: its minute's autocorrelation RR (s); NaN in an unclear or partial minute."""
    if starts.size == 0:
        return np.full(idx.size, np.nan)
    t = idx / fd
    m = np.clip((t // window_s).astype(int), 0, bpm.size - 1)
    rr = 60.0 / bpm[m]
    rr[t >= starts[-1] + window_s] = np.nan
    return np.asarray(rr, dtype=np.float64)


def _clean(seed: I64, ac: F64, fd: float) -> I64:
    """(g) 3: both adjacent seed intervals within CLEAN_RR_TOL of the minute's AC RR."""
    if seed.size < 3:  # noqa: PLR2004
        return np.zeros(0, dtype=np.int64)
    rr = np.diff(seed) / fd
    before, after = np.r_[np.nan, rr], np.r_[rr, np.nan]
    with np.errstate(invalid="ignore"):
        ok = (np.abs(before / ac - 1) <= CLEAN_RR_TOL) & (np.abs(after / ac - 1) <= CLEAN_RR_TOL)
    return np.asarray(seed[ok], dtype=np.int64)


def _short_frac(train: I64, ac: F64, fd: float) -> float:
    """Fraction of intervals under SHORT_FRACTION x the AC RR, over clear minutes."""
    if train.size < 2:  # noqa: PLR2004
        return 1.0
    rr = np.diff(train) / fd
    ref = ac[1:]
    ok = np.isfinite(ref)
    return float(np.mean(rr[ok] < SHORT_FRACTION * ref[ok])) if ok.any() else 1.0


def _seed(y: F64, fd: float, idx: I64, t05: I64, ac_t05: F64) -> _Seed:
    """(g) 4: task 05's beats, or the better of the two interleaved candidate subtrains."""
    if _short_frac(t05, ac_t05, fd) <= SEED_SHORT_MAX:
        return _Seed(t05, "task05")
    a, b = idx[0::2], idx[1::2]
    sa, sb = float(_template_snr(y, a / fd, fd)), float(_template_snr(y, b / fd, fd))
    n = min(a.size, b.size)
    jitter = float(np.std(np.diff(a[:n]) / fd - np.diff(b[:n]) / fd)) if n > 2 else float("nan")  # noqa: PLR2004
    hi_snr, lo_snr = max(sa, sb), min(sa, sb)
    out = _Seed(a if sa >= sb else b, "split", (hi_snr, lo_snr), jitter)
    if not np.isfinite(hi_snr) or hi_snr <= 0 or (hi_snr - lo_snr) / hi_snr <= AMBIGUOUS_SNR:
        out.ambiguous = (f"interleaved subtrains' template SNRs {hi_snr:.1f} and {lo_snr:.1f} "
                         f"are within {AMBIGUOUS_SNR:.0%}: ambiguous")
    return out


def _judge(idx: I64, scorer: _Scorer, masks: list[Mask], sus_r: int,
           counts: dict[str, int]) -> tuple[I64, F64, S, B]:
    """Score every candidate; drop the undefined and suspects under their floor.

    Also returns, per kept beat, whether it is TEMPLATE-MATCHING: its score reaches the
    floor made the same way (its own mask; the unmasked floor when it has none) - (g) 2.
    """
    sc, shift = scorer.score(idx, masks)
    suspect = np.array([_is_suspect(m, sus_r) for m in masks], dtype=bool)
    undefined = ~np.isfinite(sc)
    floors = np.array([scorer.floor(m) if not un else np.inf
                       for m, un in zip(masks, undefined, strict=True)])
    matching = ~undefined & (sc >= floors)
    keep = ~undefined & matching  # (h) 2: the floor applies to every candidate
    counts["suspects"] = int(suspect.sum())
    counts["suspects_dropped"] = int((suspect & ~undefined & ~keep).sum())
    counts["nonsuspects_dropped"] = int((~suspect & ~undefined & ~keep).sum())
    counts["undefined_dropped"] = int(undefined.sum())
    fid = np.where(suspect, idx + shift, idx)[keep]
    kind = np.where(suspect, "aligned", "peak")[keep]
    order = np.argsort(fid, kind="stable")
    return fid[order], sc[keep][order], kind[order], matching[keep][order]


def _matched_median(fid: I64, match: B, fd: float, ac: F64) -> F64:
    """Return, per beat, the running median over consecutive template-matching beats.

    Non-matching beats are skipped (extras on every cycle cannot starve the sample), and
    an interval spanning a missed beat - at least :data:`GAP_MULTIPLE` x that minute's
    autocorrelation RR, ``ac`` per beat - is left out (a dropped beat cannot stretch
    it). Fixture 2026-09-30: stretched intervals made k x median exceed one RR and the
    refractory cascaded through real beats; requiring adjacent matches instead left
    no interval at all when a mid-cycle peak sat in every cycle.
    """
    if fid.size < 2:  # noqa: PLR2004
        return np.full(fid.size, np.nan)
    m = fid[match]
    if m.size < 2:  # noqa: PLR2004
        return np.full(fid.size, np.nan)
    t = m / fd
    rr = np.diff(t)
    ref = ac[match][1:]
    with np.errstate(invalid="ignore"):
        ok = ~(rr >= GAP_MULTIPLE * ref)
    rr, mid = rr[ok], (0.5 * (t[1:] + t[:-1]))[ok]
    tq = fid / fd
    lo = np.searchsorted(mid, tq - RUNNING_MEDIAN_HALF_S)
    hi = np.searchsorted(mid, tq + RUNNING_MEDIAN_HALF_S)
    return np.array([np.median(rr[x:y]) if y > x else np.nan for x, y in zip(lo, hi, strict=True)])


def _refractory(fid: I64, score: F64, kind: S, match: B, k: float, fd: float, ac: F64,
                counts: dict[str, int]) -> tuple[I64, F64, S, B, F64]:
    """Run two passes; the running median over template-matching beats only ((g) 4)."""
    removed = 0
    for _pass in range(2):
        med = _matched_median(fid, match, fd, ac)
        kept: list[int] = []
        for j in range(fid.size):
            if kept and np.isfinite(med[j]) and (fid[j] - fid[kept[-1]]) / fd < k * med[j]:
                removed += 1
                if score[j] > score[kept[-1]]:
                    kept[-1] = j
                continue
            kept.append(j)
        sel = np.asarray(kept, dtype=np.int64)
        fid, score, kind, match, ac = fid[sel], score[sel], kind[sel], match[sel], ac[sel]
    counts["refractory_removed"] = removed
    return fid, score, kind, match, ac


@dataclass
class _Context:
    y: F64
    fd: float
    scorer: _Scorer
    ev: I64
    sus_r: int
    params: TemplateParams
    unmasked: float


def _research(c: _Context, fid: I64, match: B,
              ac: F64) -> tuple[list[int], list[tuple[float, float]]]:
    """Search each gap near its expected times: relaxed height and width, the same floor.

    Nothing is inserted; a gap the search cannot fill is returned, tagged.
    """
    found: list[int] = []
    gaps: list[tuple[float, float]] = []
    sc, y, fd, p = c.scorer, c.y, c.fd, c.params
    if sc.clean.size == 0 or fid.size < 2:  # noqa: PLR2004
        return found, gaps
    prom_c = peak_prominences(y, sc.clean)[0]
    wid_c = peak_widths(y, sc.clean, rel_height=0.5)[0]
    h_min = max(p.height_mult * float(np.quantile(prom_c, 0.01)), np.finfo(float).tiny)
    w_lo = p.width_mult * float(np.quantile(wid_c, 0.01))
    w_hi = float(np.quantile(wid_c, 0.99)) / p.width_mult
    pk, _p = find_peaks(y, prominence=h_min)
    if pk.size:
        wd = peak_widths(y, pk, rel_height=0.5)[0]
        pk = pk[(wd >= w_lo) & (wd <= w_hi)]
    med = _matched_median(fid, match, fd, ac)
    for a, b, m in zip(fid[:-1], fid[1:], med[:-1], strict=True):
        if not np.isfinite(m) or (b - a) / fd < GAP_MULTIPLE * m:
            continue
        n_miss = max(int(round((b - a) / fd / m)) - 1, 1)
        filled, prev = 0, int(a)
        spacing = p.k * m * fd
        for j in range(1, n_miss + 1):
            e = a + j * (b - a) / (n_miss + 1)
            r = RESEARCH_FRACTION * m * fd
            cand = pk[(pk >= e - r) & (pk <= e + r) & (pk - prev >= spacing) & (b - pk >= spacing)]
            if cand.size == 0:
                continue
            o = _masks(c.ev, cand, sc.reach)
            s, sh = sc.score(cand, o)
            fl = np.array([sc.floor(oo) if np.isfinite(ss) else np.inf
                           for oo, ss in zip(o, s, strict=True)])
            good = np.isfinite(s) & (s >= fl)
            if not good.any():
                continue
            best = int(np.argmax(np.where(good, s, -np.inf)))
            prev = int(cand[best] + (sh[best] if _is_suspect(o[best], c.sus_r) else 0))
            found.append(prev)
            filled += 1
        if filled < n_miss:
            gaps.append((a / fd, b / fd))
    return found, gaps


def _no_train(reason: str, counts: dict[str, int], seed: _Seed | None = None) -> TemplateTrain:
    return TemplateTrain(np.zeros(0), np.zeros(0, dtype="<U10"), no_train=reason, counts=counts,
                         seed=seed.kind if seed else "", split_snr=seed.snr if seed else None,
                         split_jitter_s=seed.jitter if seed else float("nan"))


@dataclass
class Prepared:
    """Hold everything a channel's train needs that does not depend on the parameters.

    Candidates, seed, clean set, template, floors and the floor judgement: one
    :func:`prepare` serves every (k, height, width) of :func:`resolve`.
    """

    no_train: TemplateTrain | None
    y: F64 = field(default_factory=lambda: np.zeros(0), repr=False)
    fd: float = float("nan")
    scorer: _Scorer | None = field(default=None, repr=False)
    ev: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    sus_r: int = 0
    seed: _Seed | None = None
    fid: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)
    score: F64 = field(default_factory=lambda: np.zeros(0), repr=False)
    kind: S = field(default_factory=lambda: np.zeros(0, dtype="<U10"), repr=False)
    match: B = field(default_factory=lambda: np.zeros(0, dtype=bool), repr=False)
    ac: F64 = field(default_factory=lambda: np.zeros(0), repr=False)
    counts: dict[str, int] = field(default_factory=dict)
    fiducial_offset_s: float = float("nan")


def prepare(x: npt.ArrayLike, fs: float, events_s: npt.ArrayLike) -> Prepared:
    """Run every parameter-free step of :func:`template_train` on one channel."""
    from gems_blanking_v2.physio.hr_channel import (  # noqa: PLC0415
        AC_WINDOW_S,
        autocorr_rate,
        findpeaks_replica,
    )

    x = np.asarray(x, dtype=np.float64)
    y, fd = _prepare(x, fs, DETECT_BAND_HZ)
    counts: dict[str, int] = {}
    if y.size == 0 or not np.any(y):
        return Prepared(_no_train("flat channel", counts))
    ys, _fd = _prepare(x, fs, SCORE_BAND_HZ)
    r, mr = max(int(round(SNAP_S * fd)), 1), int(round(MERGE_S * fd))

    def snapped(t_s: F64) -> I64:
        near = np.clip(np.round(t_s * fd).astype(np.int64), 0, y.size - 1)
        return _merge(y, _snap(y, near, r), mr)

    t05 = snapped(np.asarray(detect_rpeaks(x, fs).t_s, dtype=np.float64))
    idx = snapped(np.r_[t05 / fd, findpeaks_replica(x, fs)])
    counts["candidates"] = int(idx.size)
    if idx.size < 3:  # noqa: PLR2004
        return Prepared(_no_train("fewer than 3 candidates", counts))
    starts, bpm = autocorr_rate(x, fs)
    if not np.isfinite(bpm).any():  # (g) 3: clean exists in clear minutes only
        return Prepared(_no_train("the autocorrelation is unclear in every minute: no clean beat",
                                  counts))
    seed = _seed(y, fd, idx, t05, _ac_rr(starts, bpm, t05, fd, AC_WINDOW_S))
    if seed.ambiguous:
        return Prepared(_no_train(seed.ambiguous, counts, seed))
    clean = _clean(seed.idx, _ac_rr(starts, bpm, seed.idx, fd, AC_WINDOW_S), fd)
    counts["clean"] = int(clean.size)
    ev = np.sort(np.round(np.asarray(events_s, float) * fd).astype(np.int64))
    sus_r = int(round(SUSPECT_S * fd))
    scorer = _Scorer(ys, fd, clean, ev, sus_r)
    counts["floor_beats"] = int(scorer.floor_rows.size)
    if scorer.floor_rows.size < MIN_FLOOR_BEATS:
        return Prepared(_no_train(f"{scorer.floor_rows.size} non-suspect clean beats, fewer than "
                                  f"{MIN_FLOOR_BEATS}: no floor", counts, seed))
    fid, score, kind, match = _judge(idx, scorer, _masks(ev, idx, scorer.reach), sus_r, counts)
    _c, cs = scorer.score(scorer.clean, [()] * scorer.clean.size)  # (f) 2: alignment vs peak
    return Prepared(None, y, fd, scorer, ev, sus_r, seed, fid, score, kind, match,
                    _ac_rr(starts, bpm, fid, fd, AC_WINDOW_S), counts, float(np.mean(cs)) / fd)


def resolve(p: Prepared, params: TemplateParams) -> TemplateTrain:
    """Run the parameter-dependent steps (refractory, re-search) on a prepared channel."""
    if p.no_train is not None:
        return p.no_train
    assert p.scorer is not None and p.seed is not None  # set whenever no_train is None
    counts = dict(p.counts)
    fid, _score, kind, match, ac = _refractory(p.fid, p.score, p.kind, p.match, params.k, p.fd,
                                              p.ac, counts)
    unmasked = p.scorer.floor(())
    found, gaps = _research(_Context(p.y, p.fd, p.scorer, p.ev, p.sus_r, params, unmasked),
                            fid, match, ac)
    counts["researched"] = len(found)
    t_idx = np.r_[fid, np.asarray(found, dtype=np.int64)]
    t_kind = np.r_[kind, np.full(len(found), "researched")]
    order = np.argsort(t_idx, kind="stable")
    return TemplateTrain(t_s=np.asarray(t_idx[order] / p.fd, dtype=np.float64),
                         kind=t_kind[order], seed=p.seed.kind, split_snr=p.seed.snr,
                         split_jitter_s=p.seed.jitter, gaps_s=gaps, counts=counts,
                         fiducial_offset_s=p.fiducial_offset_s, floor=unmasked)


def template_train(
    x: npt.ArrayLike, fs: float, events_s: npt.ArrayLike, params: TemplateParams,
) -> TemplateTrain:
    """Return the template-resolved train on one channel, or why there is none.

    ``events_s``: the recording's common-mode event times (``cm_events.find_events``),
    seconds, on this channel's timeline. Equivalent to ``resolve(prepare(...), params)``.
    """
    return resolve(prepare(x, fs, events_s), params)
