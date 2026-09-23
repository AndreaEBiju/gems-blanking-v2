"""Peri-R cardiac window: how much of each band the heartbeat actually contaminates.

The current pipeline blanks +/-15 ms around every R-peak on **all** channels, which
at ~400 bpm is ~20% of every recording. This measures what is really needed, per
channel and per band.

**The signal is averaged, not its envelope.** An envelope over ``window_s`` cannot
resolve anything shorter than ``window_s``, and at RR = 165 ms the +/-66 ms
extraction profile is narrower than the envelope window in four of the six bands -
so every frame in the profile comes from a window containing lag 0 and the profile
cannot vary. Coherent averaging of the band-passed *signal* is limited only by the
sample rate, and it is the physically right operation: the QRS is time-locked to R
and the neural signal is not, so the average isolates the cardiac contribution and
suppresses everything else by ``sqrt(n_beats)``.

**The null is an anti-phase control, not the profile's own outer thirds.** Triggering
on the midpoints between successive R-peaks gives the same beat periodicity, the same
beat count and the same filter, differing only in phase. A baseline taken from the
profile's own outer thirds sits inside the same envelope window as lag 0 whenever the
window is wide, so baseline equals peak, nothing exceeds threshold, and the method
returns "no window" when it means "cannot see" - opposite conclusions from the same
value. A jittered-trigger surrogate is the second null: where the two disagree, the
band is tracking the cardiac *rhythm* rather than the QRS.

**Three of the six bands need a different question.** At 364 bpm the heart-rate
fundamental is ~6 Hz. ``0-2`` and ``0.5-3`` sit below it, so a null result there is a
confirmation rather than a blindness. ``2-50`` contains the fundamental and its first
harmonics: the contamination is a continuous narrowband component, not an event, and
blanking +/-anything around a 6 Hz trigger would remove the whole record. That is
:data:`CardiacStatus` ``not_applicable``, and it belongs to task 13/14 as a notch or
a modelled confound.

**Units cancel.** Every quantity here is either a time or a ratio against a null
measured on the same trace, so the measurement does not need a declared amplitude
unit and invariant 14 is not engaged. Nothing in this module interprets a number as
microvolts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert, sosfiltfilt

from gems_blanking_v2.bands.envelope import (
    PAD_TYPE,
    _decimate_for,
    _decimated_rate,
    _design,
    band_envelope,
    impulse_response_length_s,
)
from gems_blanking_v2.constants import BANDS, GRID_S, MAD_TO_SIGMA, BandSpec
from gems_blanking_v2.types import Recording

__all__ = [
    "ENG_BAND_NAME",
    "MIN_BEATS_FOR_RR",
    "RESOLUTION_FLAG_FRACTION",
    "SPIKE_THRESHOLD_SIGMA",
    "BandVerdict",
    "CardiacStatus",
    "cardiac_window_report",
    "measure_cardiac_window",
]

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]

CardiacStatus = Literal["measured", "no_window", "unresolvable", "not_applicable"]
"""What the measurement is entitled to say. **Only ``measured`` may blank.**

``no_window``
    Resolved, and nothing rose above the null. A real negative.
``unresolvable``
    The method could not see. Not a negative - the opposite conclusion, and
    ``tuple | None`` could not tell them apart, which is why this enum exists.
``not_applicable``
    Peri-R blanking is the wrong operation for this band at this heart rate.
"""

SPIKE_THRESHOLD_SIGMA: Final = 4.5
"""The spike consumer's own threshold, for the ENG band's crossing-rate measure."""

RESOLUTION_FLAG_FRACTION: Final = 0.20
"""Flag an envelope extent within this fraction of ``window_s`` as unresolved.

At or below the window, the extent is the window and essentially nothing else -
reporting it would bake a resolution limit into every recording as physiology.
"""

MIN_BEATS_FOR_RR: Final = 2
"""Two R-peaks is the least that defines an interval."""

ENG_BAND_NAME: Final = "300-3000"
"""The band whose consumer is spike detection, so it gets the crossing-rate measure."""


@dataclass(frozen=True, slots=True)
class BandVerdict:
    """One ``(channel, band)`` answer, with the reasoning attached.

    Attributes
    ----------
    channel, band
        What this is about.
    status
        See :data:`CardiacStatus`. **Only ``measured`` yields a blanking extent.**
    extent_signal_s
        ``(t_start, t_stop)`` relative to R from the **coherent signal average** -
        sample-resolution, and what task 13 and ``cardiacRemoveWinMs`` consume.
        ``None`` unless ``status == "measured"``.
    extent_env_s
        The same span measured the old way, on the envelope. Kept only as the
        resolution-floor check.
    window_s
        The band's envelope window: the floor ``extent_env_s`` cannot go below.
    window_rr_ratio
        ``window_s / RR``. Above ~1 the envelope method is blind by construction.
    peak_over_null
        Peak of ``|coherent average|`` divided by the anti-phase null's threshold.
        Above 1 means something is locked to the QRS.
    jitter_agrees
        Whether the jittered-trigger null gives the same verdict as the anti-phase
        one. Disagreement means the band is tracking the beat *rhythm*, not the QRS.
    crossing_rise
        ENG band only: peri-R rate of ``|x| > 4.5 sigma`` at lag 0 over its
        off-peak rate. The measure matched to what spike detection actually does.
    reason
        Why the status is what it is, in words.
    """

    channel: str
    band: str
    status: CardiacStatus
    extent_signal_s: tuple[float, float] | None
    extent_env_s: tuple[float, float] | None
    window_s: float
    window_rr_ratio: float
    peak_over_null: float
    jitter_agrees: bool
    n_beats: int
    reason: str
    crossing_rise: float | None = None
    peak_lag_s: float = 0.0
    """Lag of the coherent average's peak relative to R. Nonzero means conduction delay."""

    @property
    def duration_s(self) -> float:
        """Blanking duration this verdict licenses. Zero unless ``measured``."""
        if self.status != "measured" or self.extent_signal_s is None:
            return 0.0
        return self.extent_signal_s[1] - self.extent_signal_s[0]

    def duty_cycle(self, rr_s: float) -> float:
        """Fraction of the record this verdict would blank at ``rr_s``."""
        return self.duration_s / rr_s if rr_s > 0 else 0.0

    @property
    def envelope_unresolved(self) -> bool:
        """Whether the envelope extent is within :data:`RESOLUTION_FLAG_FRACTION`."""
        if self.extent_env_s is None:
            return True
        span = self.extent_env_s[1] - self.extent_env_s[0]
        return abs(span - self.window_s) <= RESOLUTION_FLAG_FRACTION * self.window_s

    def to_provenance(self) -> dict[str, Any]:
        """Return a JSON-ready record. A value that is absent stays absent."""
        record: dict[str, Any] = {
            "channel": self.channel,
            "band": self.band,
            "status": self.status,
            "window_s": self.window_s,
            "window_rr_ratio": self.window_rr_ratio,
            "peak_over_null": self.peak_over_null,
            "jitter_agrees": self.jitter_agrees,
            "n_beats": self.n_beats,
            "peak_lag_s": self.peak_lag_s,
            "reason": self.reason,
            "envelope_unresolved": self.envelope_unresolved,
        }
        if self.extent_signal_s is not None:
            record["extent_signal_start_s"] = self.extent_signal_s[0]
            record["extent_signal_stop_s"] = self.extent_signal_s[1]
        if self.extent_env_s is not None:
            record["extent_env_start_s"] = self.extent_env_s[0]
            record["extent_env_stop_s"] = self.extent_env_s[1]
        if self.crossing_rise is not None:
            record["crossing_rise"] = self.crossing_rise
        return record


def _band_limit(x: F64, fs: float, spec: BandSpec) -> tuple[F64, float]:
    """Band-pass at the rate the band runs at, with task 06's design and padding.

    Reuses task 06's decimation and design so the trace measured here is the trace the
    detector sees. Returns the band-limited signal and the rate it is sampled at.
    """
    y, rate = _decimate_for(x, fs, spec.hi_hz)
    sos = _design(rate, spec.lo_hz, spec.hi_hz)
    return np.asarray(sosfiltfilt(sos, y, padtype=PAD_TYPE), dtype=np.float64), rate


def _coherent_average(x: F64, fs: float, triggers_s: F64, half_s: float) -> F64:
    """Average ``x`` over ``+/-half_s`` about every trigger. Limited only by ``fs``.

    Beats whose window would fall off either end are dropped rather than zero-padded:
    padding would pull the average toward zero at the edges and look like a decaying
    cardiac component.
    """
    half = int(round(half_s * fs))
    idx = np.round(np.asarray(triggers_s) * fs).astype(np.int64)
    idx = idx[(idx - half >= 0) & (idx + half + 1 <= x.size)]
    if idx.size == 0:
        return np.zeros(2 * half + 1, dtype=np.float64)
    offsets = np.arange(-half, half + 1)
    return np.asarray(x[idx[:, None] + offsets[None, :]].mean(axis=0), dtype=np.float64)


def _analytic(profile: F64) -> F64:
    """Return the analytic envelope of a coherent average.

    A band-passed QRS is a ringing transient: in ``300-3000`` the average oscillates
    at the carrier and dips below any threshold between lobes, so a contiguous run
    reports one lobe - 1.3 ms where the burst is several times that. The analytic
    envelope removes the carrier and nothing else. Its resolution is set by the band's
    own bandwidth (~0.4 ms at 2700 Hz, ~5 ms at 200 Hz), not by an averaging window,
    so this does **not** reintroduce the floor that :func:`_envelope_extent` measures.
    """
    return np.asarray(np.abs(hilbert(profile)), dtype=np.float64)


def _null_threshold(null_average: F64) -> float:
    """Return the level an R-triggered sample must clear: the null's own **maximum**.

    Not ``median + k * MAD``. The extent is anchored on the profile's peak, so the
    comparison is a maximum over every lag, and a per-sample threshold is the wrong
    family: a 3-sigma level over the ~3200 lags of a 300-3000 profile is cleared by
    white noise essentially always, and the measurement called pure noise "measured"
    at 1.13x. Taking the null's maximum over the same number of lags, with the same
    beat count and the same filter, makes the two sides the same family - invariant
    10b's argument, applied across lags instead of across (signal, band) pairs.

    Non-parametric, so it assumes nothing about the residual's distribution after
    averaging thousands of beats.
    """
    return float(np.max(np.abs(null_average)))


def _contiguous_about_peak(
    profile: F64, fs: float, threshold: float
) -> tuple[float, float] | None:
    """Return the contiguous span containing the peak where ``profile`` exceeds ``threshold``.

    ``profile`` is an analytic envelope from :func:`_analytic`, so it is positive and
    carries no carrier oscillation to dip below the threshold mid-burst.

    **Anchored on the peak, not on lag 0.** The window does not have to be centred on
    R: conduction delay puts the stomach channels' cardiac deflection several
    milliseconds after it, and requiring the crossing to occur at lag 0 exactly
    reported ``no_window`` for all three ANT channels at 5-6x the null - contamination
    that would then never be blanked. Taking the peak cannot find the neighbouring
    beat because the profile is capped at ``0.4 * RR`` either side.

    The returned span is relative to R and is therefore asymmetric where the physiology
    is, which is the honest answer and the thing a symmetric ``+/-15 ms`` constant
    cannot express.
    """
    centre = profile.size // 2
    above = profile > threshold
    if not bool(np.any(above)):
        return None
    peak = int(np.argmax(profile))
    start = peak
    while start > 0 and above[start - 1]:
        start -= 1
    stop = peak
    while stop < profile.size - 1 and above[stop + 1]:
        stop += 1
    return (start - centre) / fs, (stop - centre + 1) / fs


def _envelope_extent(
    column: F64, fs: float, spec: BandSpec, rpeaks_s: F64, midpoints_s: F64, half_s: float
) -> tuple[float, float] | None:
    """Repeat the measurement the old way, on the envelope, for the resolution floor.

    Kept **only** as the floor check. On the 10 ms grid with a ``window_s``-wide
    envelope, every frame within ``window_s/2`` of lag 0 draws on samples that include
    lag 0, so the profile cannot be narrower than the window however brief the real
    contamination is. Comparing this against :attr:`BandVerdict.window_s` is what shows
    that the historical +/-15 ms number was a window width, not a physiological extent.
    """
    env = band_envelope(column, fs, spec.lo_hz, spec.hi_hz, spec.window_s)
    grid_fs = 1.0 / GRID_S
    finite = np.where(np.isfinite(env), env, np.nan)
    averaged = _coherent_average_nan(finite, grid_fs, rpeaks_s, half_s)
    anti = _coherent_average_nan(finite, grid_fs, midpoints_s, half_s)
    if averaged.size == 0 or not np.any(np.isfinite(averaged)):
        return None
    baseline = float(np.nanmedian(anti))
    return _contiguous_about_peak(
        np.nan_to_num(averaged - baseline), grid_fs, _null_threshold(anti - baseline)
    )


def _coherent_average_nan(x: F64, fs: float, triggers_s: F64, half_s: float) -> F64:
    """:func:`_coherent_average`, but NaN-aware - envelope frames can be invalid."""
    half = int(round(half_s * fs))
    idx = np.round(np.asarray(triggers_s) * fs).astype(np.int64)
    idx = idx[(idx - half >= 0) & (idx + half + 1 <= x.size)]
    if idx.size == 0:
        return np.zeros(2 * half + 1, dtype=np.float64)
    offsets = np.arange(-half, half + 1)
    block = x[idx[:, None] + offsets[None, :]]
    with np.errstate(invalid="ignore"):
        return np.asarray(np.nanmean(block, axis=0), dtype=np.float64)


def _crossing_rise(
    x: F64, fs: float, triggers_s: F64, half_s: float, peak_lag_s: float, n_bins: int = 41
) -> float:
    """Return the peri-R rate of ``|x| > 4.5 sigma`` at lag 0 over its off-peak rate.

    The measurement matched to what spike detection actually does, at sample
    resolution and with no envelope anywhere. This is what falsified the original
    prediction that nothing above 300 Hz needed blanking.
    """
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(x - np.median(x))))
    if sigma <= 0:
        return float("nan")
    crossings = np.flatnonzero(np.abs(x) > SPIKE_THRESHOLD_SIGMA * sigma) / fs
    if crossings.size == 0:
        return float("nan")

    edges = np.linspace(-half_s, half_s, n_bins + 1)
    lags: list[F64] = []
    for trigger in np.asarray(triggers_s).tolist():
        lo, hi = np.searchsorted(crossings, (trigger - half_s, trigger + half_s))
        if hi > lo:
            lags.append(crossings[lo:hi] - trigger)
    if not lags:
        return float("nan")
    counts, _ = np.histogram(np.concatenate(lags), bins=edges)

    # Read at the peak lag, not at lag 0: the coherent average locates the deflection
    # independently, and in 300-3000 the energy sits on the QRS's leading edge rather
    # than on R. Reading the centre bin returned 0.0 where the rate is highest 5 ms
    # earlier. Using an independently determined lag also keeps this unbiased, which
    # taking the largest bin would not be.
    peak_bin = int(np.clip(np.searchsorted(edges, peak_lag_s) - 1, 0, n_bins - 1))
    flank = np.concatenate([counts[: n_bins // 4], counts[-(n_bins // 4) :]])
    baseline = float(np.mean(flank))
    return float(counts[peak_bin]) / baseline if baseline > 0 else float("nan")


def _applicability(
    spec: BandSpec, rr_s: float, half_s: float, fs: float
) -> tuple[CardiacStatus | None, str]:
    """Decide before measuring whether peri-R blanking is even the right question.

    Order matters: a band where the operation is wrong is reported as such even if it
    would also be unresolvable, because ``not_applicable`` routes the problem to task
    13/14 while ``unresolvable`` only says to look again with a better method.
    """
    fundamental_hz = 1.0 / rr_s
    profile_s = 2.0 * half_s
    if spec.lo_hz <= fundamental_hz <= spec.hi_hz and spec.window_s > rr_s:
        return "not_applicable", (
            f"band contains the {fundamental_hz:.1f} Hz fundamental and its window "
            f"({spec.window_s:.3f} s) exceeds RR ({rr_s:.3f} s). The contamination is "
            "a continuous narrowband component, not an event: blanking around a "
            f"{fundamental_hz:.0f} Hz trigger would remove the whole record. Task "
            "13/14 owns this as a notch at the beat rate and its harmonics, or as a "
            "modelled confound."
        )

    impz_s = impulse_response_length_s(
        spec.lo_hz, spec.hi_hz, _decimated_rate(fs, spec.hi_hz)
    )
    if impz_s > profile_s:
        return "unresolvable", (
            f"the band-pass's own impulse response is {impz_s * 1e3:.0f} ms, wider "
            f"than the whole {profile_s * 1e3:.0f} ms extraction profile, so the "
            "coherent average is a truncated view of the filter's ringing and no "
            "boundary exists inside the window to find. The profile cannot be "
            f"widened: it is capped at 0.4 x RR either side, so at RR "
            f"{rr_s * 1e3:.0f} ms this band is unresolvable by construction rather "
            "than in this recording. Invariant 19's logic, applied to a profile "
            "width instead of an edge trim."
        )
    if spec.hi_hz < fundamental_hz:
        return None, (
            f"band tops out at {spec.hi_hz:g} Hz, below the {fundamental_hz:.1f} Hz "
            "heart-rate fundamental, so a null result is a confirmation on spectral "
            "grounds rather than a failure to see"
        )
    if impz_s > 0.25 * profile_s:
        return None, (
            f"resolvable, but the band-pass's {impz_s * 1e3:.0f} ms impulse response "
            f"is a large share of any extent measured in a {profile_s * 1e3:.0f} ms "
            "profile: read the extent as what the consumer's filtered trace shows, "
            "not as the width of the underlying cardiac event"
        )
    return None, ""


def cardiac_window_report(
    signal: F64,
    fs: float,
    channel_names: list[str],
    rpeaks_s: F64,
    bands: dict[str, BandSpec] | None = None,
    half_window_frac_rr: float = 0.40,
    *,
    seed: int = 0,
) -> list[BandVerdict]:
    """Measure the peri-R window for every channel and band.

    Parameters
    ----------
    signal
        ``(n_samples, n_channels)``. Any amplitude unit: every result is a time or a
        ratio against a null on the same trace, so units cancel.
    fs
        Sample rate, Hz. Read from the recording, never assumed.
    channel_names
        One per column.
    rpeaks_s
        R-peak times in seconds, from task 05.
    bands
        Defaults to the six of A.3.
    half_window_frac_rr
        Extraction half-width as a fraction of RR. **Asserted below RR/2**, since a
        wider window overlaps the neighbouring beat and contaminates every null.
    seed
        For the jittered-trigger surrogate.
    """
    bands = BANDS if bands is None else bands
    rpeaks_s = np.asarray(rpeaks_s, dtype=np.float64)
    if rpeaks_s.size < MIN_BEATS_FOR_RR:
        msg = f"need at least two R-peaks to define RR, got {rpeaks_s.size}"
        raise ValueError(msg)

    rr_s = float(np.median(np.diff(rpeaks_s)))
    half_s = half_window_frac_rr * rr_s
    if half_s >= rr_s / 2.0:
        msg = (
            f"half_window_frac_rr={half_window_frac_rr} gives +/-{half_s * 1e3:.1f} ms "
            f"at RR {rr_s * 1e3:.1f} ms, which reaches the neighbouring beat "
            f"(RR/2 = {rr_s / 2 * 1e3:.1f} ms) and contaminates the null"
        )
        raise ValueError(msg)
    log.info(
        "RR %.1f ms, half-window +/-%.1f ms (%.2f x RR, limit %.1f ms), %d beats",
        rr_s * 1e3,
        half_s * 1e3,
        half_window_frac_rr,
        rr_s / 2 * 1e3,
        rpeaks_s.size,
    )

    midpoints_s = (rpeaks_s[:-1] + rpeaks_s[1:]) / 2.0
    rng = np.random.default_rng(seed)
    jittered_s = rpeaks_s + rng.uniform(-rr_s / 2, rr_s / 2, size=rpeaks_s.size)

    verdicts: list[BandVerdict] = []
    for band, spec in bands.items():
        forced, reason = _applicability(spec, rr_s, half_s, fs)
        for column, name in enumerate(channel_names):
            verdicts.append(
                _one(
                    np.asarray(signal[:, column], dtype=np.float64),
                    fs,
                    name,
                    band,
                    spec,
                    rpeaks_s,
                    midpoints_s,
                    jittered_s,
                    half_s,
                    rr_s,
                    forced,
                    reason,
                )
            )
    return verdicts


def _one(
    column: F64,
    fs: float,
    name: str,
    band: str,
    spec: BandSpec,
    rpeaks_s: F64,
    midpoints_s: F64,
    jittered_s: F64,
    half_s: float,
    rr_s: float,
    forced: CardiacStatus | None,
    forced_reason: str,
) -> BandVerdict:
    """Measure one ``(channel, band)``."""
    ratio = spec.window_s / rr_s
    if forced is not None:
        return BandVerdict(
            channel=name, band=band, status=forced, extent_signal_s=None,
            extent_env_s=None, window_s=spec.window_s, window_rr_ratio=ratio,
            peak_over_null=float("nan"), jitter_agrees=True,
            n_beats=int(rpeaks_s.size), reason=forced_reason,
        )

    limited, rate = _band_limit(column, fs, spec)
    averaged = _coherent_average(limited, rate, rpeaks_s, half_s)
    anti = _coherent_average(limited, rate, midpoints_s, half_s)
    jitter = _coherent_average(limited, rate, jittered_s, half_s)

    profile = _analytic(averaged)
    anti_threshold = _null_threshold(_analytic(anti))
    jitter_threshold = _null_threshold(_analytic(jitter))
    extent = _contiguous_about_peak(profile, rate, anti_threshold)
    jitter_extent = _contiguous_about_peak(profile, rate, jitter_threshold)
    agrees = (extent is None) == (jitter_extent is None)

    peak = float(np.max(profile))
    over = peak / anti_threshold if anti_threshold > 0 else float("nan")
    peak_lag = float((int(np.argmax(profile)) - profile.size // 2) / rate)

    crossing = (
        _crossing_rise(limited, rate, rpeaks_s, half_s, peak_lag)
        if band == ENG_BAND_NAME
        else None
    )
    env_extent = _envelope_extent(column, fs, spec, rpeaks_s, midpoints_s, half_s)

    edge_s = (averaged.size // 2) / rate
    if extent is not None and (
        extent[0] <= -edge_s + 1.0 / rate or extent[1] >= edge_s - 1.0 / rate
    ):
        return BandVerdict(
            channel=name, band=band, status="unresolvable", extent_signal_s=None,
            extent_env_s=env_extent, window_s=spec.window_s, window_rr_ratio=ratio,
            peak_over_null=over, jitter_agrees=agrees, n_beats=int(rpeaks_s.size),
            crossing_rise=crossing, peak_lag_s=peak_lag,
            reason=(
                f"the excursion reaches the edge of the +/-{edge_s * 1e3:.0f} ms "
                "profile, so no boundary was found inside it - the measurement has "
                "no upper bound to report and a wider window would reach the "
                "neighbouring beat. Not a negative"
            ),
        )
    if extent is None:
        status: CardiacStatus = "no_window"
        reason = forced_reason or (
            f"coherent average peaks at {over:.2f}x the anti-phase null threshold, "
            "which does not clear it - resolved, and nothing is locked to the QRS"
        )
    else:
        status = "measured"
        reason = (
            f"coherent average peaks at {over:.2f}x the anti-phase null threshold "
            f"over {(extent[1] - extent[0]) * 1e3:.1f} ms spanning lag "
            f"{peak_lag * 1e3:+.2f} ms"
        )
        if forced_reason:
            reason += f"; {forced_reason}"
        if not agrees:
            reason += "; the jittered null disagrees, so some of this is beat rhythm"

    return BandVerdict(
        channel=name, band=band, status=status, extent_signal_s=extent,
        extent_env_s=env_extent, window_s=spec.window_s, window_rr_ratio=ratio,
        peak_over_null=over, jitter_agrees=agrees, n_beats=int(rpeaks_s.size),
        reason=reason, crossing_rise=crossing, peak_lag_s=peak_lag,
    )


def measure_cardiac_window(
    rec: Recording,
    rpeaks_s: F64,
    bands: dict[str, BandSpec] | None = None,
    half_window_frac_rr: float = 0.40,
    baseline_pct: float = 50.0,
) -> dict[tuple[str, str], tuple[float, float] | None]:
    """Return ``(channel, band) -> (t_start_s, t_stop_s)`` relative to R, or ``None``.

    The task's signature. **``None`` here is ambiguous by construction** - it covers
    ``no_window``, ``unresolvable`` and ``not_applicable``, which are three different
    conclusions - so anything that acts on the result should take
    :func:`cardiac_window_report` instead and read the status. Only ``measured``
    produces a tuple.

    ``baseline_pct`` is accepted and unused: the baseline is now an anti-phase
    control rather than a percentile of the profile's own outer thirds, because the
    outer thirds sit inside the same envelope window as lag 0 whenever the window is
    wide. Kept in the signature so existing callers do not break.
    """
    del baseline_pct
    verdicts = cardiac_window_report(
        np.asarray(rec.data, dtype=np.float64),
        float(rec.fs),
        [c.name for c in rec.channels],
        rpeaks_s,
        bands,
        half_window_frac_rr,
    )
    return {(v.channel, v.band): v.extent_signal_s for v in verdicts}
