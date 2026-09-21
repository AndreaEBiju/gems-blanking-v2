"""R-peak detection, gap-targeted rescue, and best-channel ranking.

Two passes and no adaptive state. Pass 1 finds beats on a band-limited trace; pass 2
looks again, only inside intervals that are an integer multiple of the local rate,
with a threshold relaxed by an amount the shrunken search space pays for.

Three things here were measured rather than inherited, and each replaces a value
that was actively harmful:

``R_MIN`` **is 60 ms, not 90 ms.**
    The 90 ms figure comes from human ECG, where RR is 800-1000 ms. Animal J's
    measured RR median is 164.7 ms (HR 363-364 bpm, identical across all 9 channels
    and every band tested), so 90 ms is 0.55x RR - it sits on top of the
    plausibility threshold itself and deletes real beats. At 600 bpm it would be 90%
    of RR. 60 ms is 0.36x RR. The RR histogram is a tight unimodal spike at
    155-180 ms with essentially nothing below 145 ms. **Repeat that histogram on at
    least five files before pinning this.**

The plausibility rule uses a **global** RR, never the accepted sequence.
    The local-median form runs away: dropping a beat raises the local median, which
    raises the threshold, which drops more beats. Measured on animal J at 0.75 it
    collapses outright - RR median 330 ms, 54% of beats dropped - while the global
    form is monotone and stable to 0.85. Same failure as the adaptive QRS threshold
    in ``PIPELINE.md`` 10.1: adaptive state contaminated by the artifact it exists
    to reject.

**No beat is ever fabricated.** Measured at 9% missed beats over 300 runs,
midpoint insertion biases RMSSD, SD1 and pNN **downward** by 4-6% - the same
direction as a vagal-tone effect - because it forces the flanking intervals equal
and their successive difference to exactly zero. Re-detection in a window is
unbiased to 0.0%. An interval that cannot be recovered is recorded as a gap and
excluded, never filled.

Detection reads one channel at a time and depends on nothing above step 05. In
particular there is no motion mask here and must not be: beats lost inside an
artifact sit in spans that get masked anyway, so what pass 1 cannot recover is
information the pipeline was going to discard. That is what breaks the R-peak /
motion circularity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import butter, decimate, find_peaks, peak_widths, sosfiltfilt

from gems_blanking_v2.constants import BANDS, HR_BAND
from gems_blanking_v2.derive.derivations import robust_sigma
from gems_blanking_v2.types import Recording

__all__ = [
    "DETECT_BAND_HZ",
    "GLOBAL_RR_FRACTION",
    "INHERITED_R_MIN_S",
    "MIN_MISSING_MULTIPLE",
    "PASS1_PROMINENCE_SIGMA",
    "PASS2_MULTIPLE_TOLERANCE",
    "PASS2_PROMINENCE_SIGMA",
    "PROVISIONAL_MAX_IMPLAUSIBLE_FRAC",
    "PROVISIONAL_MAX_RESCUE_RATE",
    "R_MIN_S",
    "BeatTrain",
    "detect_rpeaks",
    "missing_beat_count",
    "rank_hr_channels",
]

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]
BeatTag = Literal["detected", "rescued"]

DETECT_BAND_HZ: Final[tuple[float, float]] = (BANDS[HR_BAND].lo_hz, BANDS[HR_BAND].hi_hz)
"""Detection band, read from :data:`~gems_blanking_v2.constants.HR_BAND`.

One source of truth: this is the hrv consumer's contamination band *and* the band
its detector reads, and the two cannot be allowed to drift apart - they already had.

**Ruled 2026-09-21 as 10-150 Hz.** The earlier ``1-100 Hz`` is gone rather than left
reachable, because a superseded band in the code is a second rule competing with the
binding one. Recorded here as history, measured over 8 seeds and 3160 true beats:

===================  ============  ===============  ================
pass 1               false beats   pass-1 misses    implausible_frac
===================  ============  ===============  ================
10-150 Hz, k=6       **0**         29 (28 rescued)  2.8%
1-100 Hz, k=3        14            0                5-6%
===================  ============  ===============  ================

On animal J the same ruling shows as template SNR 611 against 446 and a
long-interval rate of 2.0% against 3.6%. In 1-100 Hz the in-band noise floor
collapses to ~0.6 uV while a weak beat still carries ~18 uV of prominence, so noise
and signal sit on the same side of 3 sigma.
"""

DECIMATE_TARGET_HZ: Final = 2000.0
"""Decimate toward this before a 1 Hz corner is asked for, per the conventions.

It also fixes the fiducial grid at ~0.5 ms, under the 1 ms the rescue must place a
beat within, and both stages are zero-phase so a beat does not move.
"""

R_MIN_S: Final = 0.060
"""Refractory floor, seconds. **Measured, not inherited** - see the module docstring.

Provisional: the RR histogram behind it is one animal, one file.
"""

INHERITED_R_MIN_S: Final = 0.090
"""The human-ECG value this replaces. Kept so a test can show what it would cost."""

GLOBAL_RR_FRACTION: Final = 0.75
"""Operating point of the plausibility rule, as a fraction of the **global** RR.

About 124 ms at animal J's 164.7 ms RR. The global form is monotone to 0.85; the
local-median form collapses at exactly this value.
"""

RR_PLAUSIBLE_RANGE_S: Final[tuple[float, float]] = (0.080, 0.500)
"""Intervals the global RR median is computed over. 750 bpm down to 120 bpm."""

PASS1_PROMINENCE_SIGMA: Final = 6.0
"""Pass-1 peak prominence, in robust sigma of the band-limited trace.

Ruled 2026-09-21 together with :data:`DETECT_BAND_HZ`; the two are one decision. At
``k=6`` in 10-150 Hz a 60 ms T wave is not detected at all, at ``k=3`` every beat is
doubled - see ``tests.conftest.T_WAVE_WIDTH_FACTOR``.
"""

PASS2_PROMINENCE_SIGMA: Final = 1.2
"""Pass-2 prominence. Lower, and the shrunken search space pays for it.

Pass 1 has ~2180 independent opportunities over 120 s (Bonferroni z = 4.08); pass 2
has ~30 windows (z = 2.95). A ~28% lower threshold at the same family-wise
false-positive rate, before the width and template priors are counted at all.
"""

PASS2_MULTIPLE_TOLERANCE: Final = 0.40
"""How close an interval must be to an integer multiple of the local RR to qualify.

The specified 0.20 was **measured too strict**: only 18-38 of about 350 long
intervals on animal J qualified. Widened to 0.40.

What the widening buys depends entirely on RR variability, measured here on true
beat trains with 9% of beats deleted, counting how many genuinely doubled intervals
the rule admits:

========  =================  =================
RR CV     admitted at 0.20   admitted at 0.40
========  =================  =================
3.3%      161/161 (100%)     161/161 (100%)
10%       123/158 (78%)      155/158 (98%)
20%       81/156 (52%)       132/156 (85%)
========  =================  =================

So at the synthetic's own 3.3% CV the constant does nothing and **no test on the
default generator can justify it** - the two values are indistinguishable there.
At a realistic CV, 0.20 discards a third to a half of the intervals pass 2 exists
to serve, which is exactly the 18-38 of 350 seen on animal J.
:func:`tests.test_rpeaks.test_the_pass2_tolerance_only_matters_at_realistic_rr_spread`
holds the table. **0.40 itself is a judgement, not a measurement**: re-measure on
real data once the global-RR plausibility fix has lowered the RR CV, since the two
interact.
"""

PASS2_WINDOW_FRACTION: Final = 0.25
"""Half-width of a rescue window, as a fraction of the local RR."""

QRS_WIDTH_RANGE: Final[tuple[float, float]] = (0.5, 2.0)
"""Width prior on a rescued peak, as a multiple of the median detected QRS width."""

LOCAL_MEDIAN_INTERVALS: Final = 10
"""Half-window, in intervals, of the local RR median pass 2 uses.

Used only to *estimate how many beats are missing*, never to gate acceptance, so it
is not self-referential the way the withdrawn plausibility rule was.
"""

TEMPLATE_HALF_WIDTH_S: Final = 0.040
"""Half-width of the beat window the QRS template averages over."""

PROVISIONAL_MAX_IMPLAUSIBLE_FRAC: Final = 0.10
"""Veto gate: a channel disagreeing with itself this often is disqualified.

**PROVISIONAL, and the prefix is the point** - it is to be set in build-order step 9
from the cross-animal set, not from one animal or a synthetic, and the name is what
stops a provisional number quietly becoming a constant.

The calibration behind 0.10: animal J sits near 2% implausible at the binding
operating point and 14-20% at ``k=3``; the synthetic's amplitude-wander channel is
28%. So 0.10 is loose enough not to veto a usable channel and tight enough to catch
one detecting mostly artifact.
"""

PROVISIONAL_MAX_RESCUE_RATE: Final = 0.25
"""Veto gate: a channel needing this much rescuing is disqualified.

**PROVISIONAL**, on the same terms as :data:`PROVISIONAL_MAX_IMPLAUSIBLE_FRAC`. The
synthetic's no-cardiac channel sits at 0.58.
"""

MIN_BEATS_FOR_AN_INTERVAL: Final = 2
"""Two beats make one RR interval, so anything below this has no rate at all."""

MIN_MISSING_MULTIPLE: Final = 2
"""An interval must look like at least two beats before pass 2 will search it.

``m == 1`` is an ordinary interval with nothing missing from it.
"""

MIN_RESCUE_WINDOW_SAMPLES: Final = 3
"""Below this a window cannot hold an interior peak, so there is nothing to find."""


@dataclass(frozen=True, slots=True)
class BeatTrain:
    """A beat train and what it cost to get.

    Attributes
    ----------
    t_s
        Beat times in seconds, strictly ascending.
    tag
        ``"detected"`` or ``"rescued"`` per beat, same length as ``t_s``.
    gaps
        ``(t0, t1, m)`` per unrecovered interval: its bounds in seconds and how many
        beats are missing from it. **HRV must exclude every interval touching one of
        these**; rate may use ``m`` to count the missing beats.
    rescue_rate
        Rescued beats as a fraction of all beats. High means the channel needed a lot
        of help, which is a channel-quality signal rather than a success.
    implausible_frac
        Fraction of pass-1 intervals that failed the plausibility rule - a channel
        disagreeing with itself.
    global_rr_s
        The whole-file RR median the plausibility threshold came from.
    qrs_width_s
        Median detected QRS width, which sets the pass-2 width prior.
    """

    t_s: F64
    tag: npt.NDArray[np.str_]
    gaps: list[tuple[float, float, int]] = field(default_factory=list)
    rescue_rate: float = 0.0
    implausible_frac: float = 0.0
    global_rr_s: float = float("nan")
    qrs_width_s: float = float("nan")

    def __post_init__(self) -> None:
        """Check the train is ordered and self-consistent."""
        if self.t_s.size != self.tag.size:
            msg = f"t_s and tag must match in length, got {self.t_s.size} and {self.tag.size}"
            raise ValueError(msg)
        if self.t_s.size > 1 and not bool(np.all(np.diff(self.t_s) > 0)):
            msg = "beat times must be strictly increasing"
            raise ValueError(msg)

    @property
    def n_beats(self) -> int:
        """Number of beats in the train."""
        return int(self.t_s.size)

    @property
    def n_missing(self) -> int:
        """Beats known to be missing, summed over the unrecovered gaps."""
        return sum(m for _t0, _t1, m in self.gaps)

    def rr_s(self) -> F64:
        """Return every RR interval in seconds, **including** those spanning gaps."""
        if self.t_s.size < MIN_BEATS_FOR_AN_INTERVAL:
            return np.empty(0, dtype=np.float64)
        return np.asarray(np.diff(self.t_s), dtype=np.float64)

    def usable_rr_mask(self) -> npt.NDArray[np.bool_]:
        """Return which entries of :meth:`rr_s` touch no unrecovered gap.

        ``True`` here means *usable*, which is the opposite of the emitted-mask
        convention - these are not masks over the sample grid, they are a selection
        over intervals, and inverting the sense would be worse than the mismatch.

        A successive-difference statistic (RMSSD, SD1, pNN) must difference only
        **adjacent** usable intervals: ``mask[:-1] & mask[1:]``. Differencing across
        a gap inflates RMSSD, SDNN and SD1 by around 770% at 9% missed beats, so an
        interval spanning one is not a measurement of anything.
        """
        if self.t_s.size < MIN_BEATS_FOR_AN_INTERVAL:
            return np.zeros(0, dtype=bool)
        starts, stops = self.t_s[:-1], self.t_s[1:]
        keep = np.ones(starts.size, dtype=bool)
        for t0, t1, _m in self.gaps:
            keep &= ~((starts < t1) & (stops > t0))
        return keep

    def usable_rr_s(self) -> F64:
        """Return only the RR intervals that touch no unrecovered gap.

        This is what HRV reads. See :meth:`usable_rr_mask` for the adjacency caveat.
        """
        if self.t_s.size < MIN_BEATS_FOR_AN_INTERVAL:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self.rr_s()[self.usable_rr_mask()], dtype=np.float64)


def _prepare(x: F64, fs: float, band: tuple[float, float]) -> tuple[F64, float]:
    """Decimate toward 2 kHz then band-limit. Returns ``(y, fs_out)``.

    Decimation first because a 1 Hz corner at 24.4 kHz is the numerically unstable
    case the conventions call out by name. Both stages are zero-phase, so a beat does
    not move.

    A NaN span is interpolated **for filtering only**, as ``step1_bandpass.m`` does,
    and nothing interpolated is returned - only beat times are (invariant 8).
    """
    finite = np.isfinite(x)
    if not bool(finite.all()):
        if not bool(finite.any()):
            return np.zeros(0, dtype=np.float64), fs
        x = x.copy()
        idx = np.arange(x.size, dtype=np.float64)
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    factor = max(int(fs // DECIMATE_TARGET_HZ), 1)
    y = np.asarray(x, dtype=np.float64)
    if factor > 1:
        y = np.asarray(decimate(y, factor, ftype="fir", zero_phase=True), dtype=np.float64)
        fs = fs / factor

    nyq = fs / 2
    lo, hi = band
    sos = butter(4, [lo / nyq, min(hi, nyq * 0.9) / nyq], btype="bandpass", output="sos")
    return np.asarray(sosfiltfilt(sos, y), dtype=np.float64), fs


def _global_rr(peak_idx: npt.NDArray[np.int64], fs: float) -> float:
    """Return the whole-file RR median, over plausible intervals only.

    One number for the whole file, computed once. Nothing recomputes it from the
    accepted sequence - that is the runaway this replaces.
    """
    if peak_idx.size < MIN_BEATS_FOR_AN_INTERVAL:
        return float("nan")
    rr = np.diff(peak_idx) / fs
    lo, hi = RR_PLAUSIBLE_RANGE_S
    in_range = rr[(rr >= lo) & (rr <= hi)]
    return float(np.median(in_range)) if in_range.size else float("nan")


def _enforce_plausibility(
    peak_idx: npt.NDArray[np.int64],
    prominence: F64,
    fs: float,
    min_spacing_s: float,
) -> tuple[npt.NDArray[np.int64], int]:
    """Drop peaks closer together than ``min_spacing_s``, keeping the stronger one.

    Keeping the more prominent of a too-close pair matters: a noise spike just before
    a real beat would otherwise delete the beat and keep the spike.
    """
    if peak_idx.size == 0 or not np.isfinite(min_spacing_s):
        return peak_idx, 0

    spacing = max(int(round(min_spacing_s * fs)), 1)
    kept: list[int] = []
    strengths: list[float] = []
    dropped = 0
    for index, strength in zip(peak_idx.tolist(), prominence.tolist(), strict=True):
        if kept and (index - kept[-1]) < spacing:
            dropped += 1
            if strength > strengths[-1]:
                kept[-1], strengths[-1] = int(index), float(strength)
            continue
        kept.append(int(index))
        strengths.append(float(strength))
    return np.asarray(kept, dtype=np.int64), dropped


def _beat_windows(y: F64, idx: npt.NDArray[np.int64], fs: float) -> F64:
    """Stack the +/-40 ms windows about ``idx`` that fit inside ``y``."""
    half = max(int(round(TEMPLATE_HALF_WIDTH_S * fs)), 1)
    windows = [
        y[i - half : i + half + 1]
        for i in idx.tolist()
        if i - half >= 0 and i + half + 1 <= y.size
    ]
    if not windows:
        return np.zeros((0, 2 * half + 1), dtype=np.float64)
    return np.asarray(np.vstack(windows), dtype=np.float64)


def _qrs_template(y: F64, peak_idx: npt.NDArray[np.int64], fs: float) -> F64:
    """Return the mean beat-aligned window: this channel's QRS template.

    Non-time-locked content averages down as ``1/sqrt(N)``, so a channel detecting
    artifacts rather than beats produces a near-flat template. That is what makes the
    ranking self-policing.
    """
    stack = _beat_windows(y, peak_idx, fs)
    if stack.shape[0] == 0:
        half = max(int(round(TEMPLATE_HALF_WIDTH_S * fs)), 1)
        return np.zeros(2 * half + 1, dtype=np.float64)
    return np.asarray(stack.mean(axis=0), dtype=np.float64)


def _median_qrs_width_s(y: F64, peak_idx: npt.NDArray[np.int64], fs: float) -> float:
    """Return the median half-prominence width of the detected peaks, in seconds."""
    if peak_idx.size == 0:
        return float("nan")
    widths = peak_widths(y, peak_idx, rel_height=0.5)[0]
    return float(np.median(widths)) / fs


def missing_beat_count(interval_s: float, local_rr_s: float) -> int | None:
    """Return how many beats an interval holds, or ``None`` if it does not qualify.

    An interval qualifies for pass 2 only when it is within
    :data:`PASS2_MULTIPLE_TOLERANCE` of an integer multiple ``m >= 2`` of the local
    RR. ``m`` is the number of beats the interval should contain, so ``m - 1`` are
    missing from it.

    Public because the tolerance's cost is only visible through this rule, and it
    depends on RR variability rather than on anything about the signal - see
    :data:`PASS2_MULTIPLE_TOLERANCE`.

    Parameters
    ----------
    interval_s
        The observed interval, seconds.
    local_rr_s
        The local median RR, seconds. Used to *estimate* the count, never to gate
        acceptance of a peak, so it is not self-referential the way the withdrawn
        plausibility rule was.
    """
    if not np.isfinite(local_rr_s) or local_rr_s <= 0 or not np.isfinite(interval_s):
        return None
    m = int(round(interval_s / local_rr_s))
    too_far = abs(interval_s - m * local_rr_s) > PASS2_MULTIPLE_TOLERANCE * local_rr_s
    if m < MIN_MISSING_MULTIPLE or too_far:
        return None
    return m


def _local_median_rr(rr: F64, index: int) -> float:
    """Return the median RR over +/-10 intervals around ``index``."""
    lo = max(index - LOCAL_MEDIAN_INTERVALS, 0)
    hi = min(index + LOCAL_MEDIAN_INTERVALS + 1, rr.size)
    window = rr[lo:hi]
    return float(np.median(window)) if window.size else float("nan")


def _best_by_template(
    y: F64, candidates: npt.NDArray[np.int64], template: F64
) -> int | None:
    """Return the candidate whose neighbourhood correlates best with the template."""
    half = template.size // 2
    norm_template = float(np.linalg.norm(template))
    if norm_template == 0.0:
        return int(candidates[0]) if candidates.size else None

    best_index: int | None = None
    best_score = -np.inf
    for index in candidates.tolist():
        lo, hi = index - half, index + half + 1
        if lo < 0 or hi > y.size:
            continue
        window = y[lo:hi]
        denominator = float(np.linalg.norm(window)) * norm_template
        if denominator == 0.0:
            continue
        score = float(np.dot(window, template)) / denominator
        if score > best_score:
            best_score, best_index = score, int(index)
    return best_index


def _rescue(
    y: F64,
    fs: float,
    peak_idx: npt.NDArray[np.int64],
    sigma: float,
    template: F64,
    qrs_width_s: float,
) -> tuple[list[int], list[tuple[float, float, int]]]:
    """Look again inside intervals that are an integer multiple of the local RR.

    Returns ``(rescued_indices, gaps)``. A window with no qualifying candidate
    contributes to a gap record and **no beat**.
    """
    rescued: list[int] = []
    gaps: list[tuple[float, float, int]] = []
    if peak_idx.size < MIN_BEATS_FOR_AN_INTERVAL:
        return rescued, gaps

    rr = np.asarray(np.diff(peak_idx) / fs, dtype=np.float64)
    width_lo, width_hi = QRS_WIDTH_RANGE
    width_bounds = (
        (width_lo * qrs_width_s * fs, width_hi * qrs_width_s * fs)
        if np.isfinite(qrs_width_s) and qrs_width_s > 0
        else None
    )

    for j, interval in enumerate(rr.tolist()):
        local = _local_median_rr(rr, j)
        m = missing_beat_count(interval, local)
        if m is None:
            continue

        unrecovered = 0
        for q in range(1, m):
            centre = float(peak_idx[j]) + q * interval * fs / m
            half = PASS2_WINDOW_FRACTION * local * fs
            lo = max(int(round(centre - half)), 0)
            hi = min(int(round(centre + half)), y.size - 1)
            if hi - lo < MIN_RESCUE_WINDOW_SAMPLES:
                unrecovered += 1
                continue

            candidates, _properties = find_peaks(
                y[lo : hi + 1],
                prominence=PASS2_PROMINENCE_SIGMA * sigma,
                width=width_bounds,
            )
            best = (
                _best_by_template(y, lo + candidates, template) if candidates.size else None
            )
            if best is None:
                unrecovered += 1
                continue
            rescued.append(best)

        if unrecovered:
            gaps.append((float(peak_idx[j] / fs), float(peak_idx[j + 1] / fs), unrecovered))

    return rescued, gaps


def detect_rpeaks(x: F64, fs: float) -> BeatTrain:
    """Detect R-peaks in two passes, fabricating nothing.

    Parameters
    ----------
    x
        One channel, microvolts. A NaN span is interpolated for filtering and
        discarded again; only beat times are returned.
    fs
        Sample rate in Hz, read from the recording - never assumed.

    Returns
    -------
    BeatTrain
        Beats, their tags, the unrecovered gaps, and the two quality fractions the
        channel ranking gates on. Empty when the trace carries no beats.
    """
    y, fs_d = _prepare(np.asarray(x, dtype=np.float64), fs, DETECT_BAND_HZ)
    empty = BeatTrain(t_s=np.empty(0, dtype=np.float64), tag=np.empty(0, dtype="<U8"))
    sigma = robust_sigma(y) if y.size else float("nan")
    if not np.isfinite(sigma) or sigma == 0.0:
        return empty

    provisional, properties = find_peaks(
        y,
        prominence=PASS1_PROMINENCE_SIGMA * sigma,
        distance=max(int(round(R_MIN_S * fs_d)), 1),
    )
    if provisional.size == 0:
        return empty

    global_rr = _global_rr(provisional, fs_d)
    kept, dropped = _enforce_plausibility(
        provisional,
        np.asarray(properties["prominences"], dtype=np.float64),
        fs_d,
        GLOBAL_RR_FRACTION * global_rr,
    )
    implausible_frac = dropped / max(provisional.size - 1, 1)

    template = _qrs_template(y, kept, fs_d)
    qrs_width_s = _median_qrs_width_s(y, kept, fs_d)
    rescued, gaps = _rescue(y, fs_d, kept, sigma, template, qrs_width_s)

    idx = np.concatenate([kept, np.asarray(rescued, dtype=np.int64)])
    tags = np.array(["detected"] * kept.size + ["rescued"] * len(rescued), dtype="<U8")
    order = np.argsort(idx, kind="stable")
    idx, tags = idx[order], tags[order]
    unique, first = np.unique(idx, return_index=True)
    if unique.size != idx.size:
        idx, tags = unique, tags[np.sort(first)]

    return BeatTrain(
        t_s=np.asarray(idx / fs_d, dtype=np.float64),
        tag=tags,
        gaps=gaps,
        rescue_rate=len(rescued) / max(idx.size, 1),
        implausible_frac=implausible_frac,
        global_rr_s=global_rr,
        qrs_width_s=qrs_width_s,
    )


def _template_snr(y: F64, t_s: F64, fs: float) -> float:
    """Return ``peak_to_peak(template) / mean(standard error across beats)``.

    SNR here measures **reproducibility**, which is what predicts HRV reliability -
    not amplitude. A channel whose "beats" are artifacts averages to a near-flat
    template with a large spread, so it scores itself down.
    """
    stack = _beat_windows(y, np.round(t_s * fs).astype(np.int64), fs)
    if stack.shape[0] < MIN_BEATS_FOR_AN_INTERVAL:
        return float("nan")
    template = stack.mean(axis=0)
    standard_error = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    mean_se = float(np.mean(standard_error))
    if mean_se == 0.0:
        return float("inf")
    return float(np.ptp(template)) / mean_se


def rank_hr_channels(
    rec: Recording, trains: dict[str, BeatTrain]
) -> tuple[str, pd.DataFrame]:
    """Rank channels for HRV and return ``(best_name, table)``.

    Ranking is by template SNR. The two quality fractions are **vetoes, not
    weights**: a channel over :data:`PROVISIONAL_MAX_IMPLAUSIBLE_FRAC` or
    :data:`PROVISIONAL_MAX_RESCUE_RATE` is disqualified outright rather than
    penalised, because a
    channel that disagrees with itself is not a slightly worse measurement of heart
    rate, it is a measurement of something else. Beat count against the median across
    channels is reported but not used.

    Re-pick per recording and log the choice: a channel that stops being best is a
    drift signal.

    Parameters
    ----------
    rec
        The recording the trains came from. The SNR is computed on the same
        :data:`DETECT_BAND_HZ` trace detection read.
    trains
        One :class:`BeatTrain` per candidate channel name.

    Raises
    ------
    ValueError
        If ``trains`` is empty, names a channel this recording does not have, or no
        channel survives the vetoes. Picking the least-bad channel silently would
        hand HRV a trace nobody vetted.
    """
    if not trains:
        msg = "rank_hr_channels needs at least one train"
        raise ValueError(msg)

    by_name = {c.name: c for c in rec.channels}
    rows: list[dict[str, object]] = []
    for name, train in trains.items():
        if name not in by_name:
            msg = f"train for {name!r} names no channel in this recording"
            raise ValueError(msg)
        y, fs_d = _prepare(
            np.asarray(rec.data[:, by_name[name].index], dtype=np.float64),
            rec.fs,
            DETECT_BAND_HZ,
        )
        vetoed_by = [
            gate
            for gate, failed in (
                (
                    "implausible_frac",
                    train.implausible_frac > PROVISIONAL_MAX_IMPLAUSIBLE_FRAC,
                ),
                ("rescue_rate", train.rescue_rate > PROVISIONAL_MAX_RESCUE_RATE),
            )
            if failed
        ]
        rows.append(
            {
                "channel": name,
                "snr": _template_snr(y, train.t_s, fs_d),
                "n_beats": train.n_beats,
                "implausible_frac": train.implausible_frac,
                "rescue_rate": train.rescue_rate,
                "global_rr_s": train.global_rr_s,
                "n_missing": train.n_missing,
                "vetoed_by": ",".join(vetoed_by),
                "eligible": not vetoed_by,
            }
        )

    table = pd.DataFrame(rows).sort_values("snr", ascending=False, ignore_index=True)
    table["beats_vs_median"] = table["n_beats"] - table["n_beats"].median()

    eligible = table[table["eligible"]]
    if eligible.empty:
        vetoes = ", ".join(f"{r.channel}({r.vetoed_by})" for r in table.itertuples())
        msg = (
            f"every channel was vetoed for HRV: {vetoes}. Picking the least-bad one "
            "silently would hand HRV a trace nobody vetted - inspect the recording."
        )
        raise ValueError(msg)

    best = str(eligible.iloc[0]["channel"])
    log.info(
        "animal %s session %s: HRV channel %s chosen, SNR %.1f (next best %s). "
        "Re-picked per recording - a channel that stops being best is a drift signal.",
        rec.animal,
        rec.session,
        best,
        float(eligible.iloc[0]["snr"]),
        str(eligible.iloc[1]["channel"]) if len(eligible) > 1 else "none",
    )
    return best, table
