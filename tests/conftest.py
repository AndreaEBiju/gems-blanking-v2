"""Synthetic-signal generators with known ground truth. Every test uses these.

No test module may invent its own signal: a generator bug otherwise produces
confident wrong numbers in one test and nowhere else. Import them directly -
``tests`` is on ``sys.path`` under pytest's default import mode::

    from conftest import make_ecg, make_multichannel

Conventions, matching ``CLAUDE.md``: amplitude in microvolts (``float64``), time in
seconds (``float64``), frequency in Hz, ``fs`` always passed explicitly, every
generator taking an explicit ``seed``. Sigma means the robust estimate
``1.4826 * MAD``, never the sample standard deviation, because these signals all
contain the outliers whose size we are trying to state.

Design notes that matter for correctness:

* No generator filters anything. Band-limited components are built from explicit
  sinusoids or analytic envelopes instead, so a generator can never reproduce the
  ``butter(..., 'ba')`` instability the real pipeline guards against.
* :func:`make_beats` affine-corrects its drawn intervals so the realised mean and SD
  are the requested ones. i.i.d. draws cannot state their SD to 2% without ~1250
  beats, and a test that only passes for a lucky seed is not a test.
* :func:`make_ecg` draws its noise last, so the beat train for a given seed is
  identical at every ``noise_uv``, including 0. That gives tests a noiseless
  reference for the same signal.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.constants import FS_NOMINAL_HZ, MAD_TO_SIGMA
from gems_blanking_v2.types import ChannelInfo, Recording

F64 = npt.NDArray[np.float64]

ArtifactKind = Literal["excursion", "drift", "step", "tribo", "clip"]

ADDITIVE_KINDS: Final[tuple[ArtifactKind, ...]] = ("excursion", "drift", "step", "tribo")
"""The kinds for which ``amp_ratio`` is exactly recoverable as ``max|out - in|/sigma``.

``'clip'`` is excluded: it is non-linear, and its ground truth is the rail, not an
injected amplitude.
"""

QRS_WIDTH_MS: Final = 10.0
"""Default width of the QRS central lobe, milliseconds. Rat QRS is 8-12 ms."""

MIN_QRS_SAMPLES: Final = 3
"""Shortest QRS kernel that can still be triphasic with an odd central lobe."""

FLANK_FRACTION: Final = 0.15
"""Flanking-lobe amplitude as a fraction of the QRS peak.

Must stay at or below 0.2: above that the kernel's own rebound is detected as a
second peak and every beat is counted twice. That bug was introduced once already.
"""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _n_samples(fs: float, dur_s: float) -> int:
    """Return the number of samples in ``dur_s`` seconds at ``fs`` Hz."""
    if fs <= 0.0:
        msg = f"fs must be positive, got {fs}"
        raise ValueError(msg)
    if dur_s <= 0.0:
        msg = f"dur_s must be positive, got {dur_s}"
        raise ValueError(msg)
    return int(round(fs * dur_s))


def robust_sigma(x: F64) -> float:
    """Robust scale ``1.4826 * MAD`` of ``x`` in the units of ``x``, ignoring NaN.

    Used instead of the standard deviation everywhere an amplitude is stated
    relative to the host signal, because the host may already contain artifacts.
    """
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    mad = float(np.median(np.abs(finite - np.median(finite))))
    return MAD_TO_SIGMA * mad


def _tukey(n: int, alpha: float) -> F64:
    """Tukey (tapered-cosine) window of ``n`` samples, taper fraction ``alpha``.

    ``alpha = 0`` is rectangular, ``alpha = 1`` is a Hann window. Written out rather
    than imported so the generators depend on numpy alone.
    """
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float64)
    if alpha <= 0.0:
        return np.ones(n, dtype=np.float64)
    alpha = min(alpha, 1.0)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    half = alpha / 2.0
    rise = t < half
    fall = t > 1.0 - half
    w[rise] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * t[rise] / alpha - 1.0)))
    w[fall] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * (t[fall] - 1.0) / alpha + 1.0)))
    return w


def _standardise(z: F64) -> F64:
    """Rescale ``z`` to exactly zero mean and unit sample SD (``ddof=1``)."""
    if z.size < 2:
        msg = "need at least two values to standardise"
        raise ValueError(msg)
    sd = float(np.std(z, ddof=1))
    if sd == 0.0:
        msg = "cannot standardise a constant draw"
        raise ValueError(msg)
    return (z - float(np.mean(z))) / sd


# ---------------------------------------------------------------------------
# cardiac
# ---------------------------------------------------------------------------


def make_beats(
    fs: float,
    dur_s: float,
    rr_s: float = 0.150,
    sd_rr_s: float = 0.005,
    seed: int = 0,
) -> F64:
    """Ground-truth R times in seconds. Realistic rat HRV.

    The drawn RR intervals are affine-corrected so that ``np.diff`` of the returned
    times has mean ``rr_s`` and sample SD (``ddof=1``) ``sd_rr_s`` exactly, up to the
    one-sample quantisation applied at the end (41 microseconds at 24.4 kHz, 0.8% of
    a 5 ms SD). 2% accuracy on an SD estimate needs ~1250 beats
    (relative SE = 1/sqrt(2n)) and a 120 s file holds ~800, so without the correction
    the fixture could not state its own SD. Intervals are otherwise independent, so
    beat *phase* performs a random walk exactly as a real heart's does.

    **Consequence:** the intervals are not an i.i.d. draw. Do not use this fixture to
    test an estimator's own sampling behaviour - its moments are fixed by
    construction, so any estimator of them will look unnaturally good.

    Parameters
    ----------
    fs
        Sample rate in Hz. R times are snapped to this grid so that a beat sits on a
        sample a detector can actually return.
    dur_s
        Record duration in seconds. All returned times are strictly inside it.
    rr_s
        Mean RR interval in seconds. 0.150 s is ~400 bpm, a resting rat.
    sd_rr_s
        SD of the RR interval in seconds.
    seed
        Seed for the interval draw.

    Returns
    -------
    numpy.ndarray
        Strictly increasing R times in seconds, ``float64``.
    """
    if rr_s <= 0.0:
        msg = f"rr_s must be positive, got {rr_s}"
        raise ValueError(msg)
    if sd_rr_s < 0.0:
        msg = f"sd_rr_s must be non-negative, got {sd_rr_s}"
        raise ValueError(msg)
    if rr_s <= 4.0 * sd_rr_s:
        msg = f"rr_s={rr_s} must exceed 4*sd_rr_s={4 * sd_rr_s} to keep intervals positive"
        raise ValueError(msg)

    n_guess = int(dur_s / rr_s)
    # The phase random walk wanders by ~sd*sqrt(n); leave room for 5 of those plus a
    # whole interval, so the last beat is inside the record for any seed.
    margin_s = rr_s + 5.0 * sd_rr_s * math.sqrt(max(n_guess, 1))
    n_int = int((dur_s - margin_s - rr_s) / rr_s)
    if n_int < 2:
        msg = (
            f"dur_s={dur_s} is too short for rr_s={rr_s}: it holds {n_int} intervals, "
            "and at least 2 are needed to state an SD"
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    z = _standardise(np.clip(rng.standard_normal(n_int), -4.0, 4.0))
    rr = rr_s + sd_rr_s * z

    beats = rr_s + np.concatenate(([0.0], np.cumsum(rr)))
    beats = np.round(beats * fs) / fs

    if not np.all(np.diff(beats) > 0.0):
        msg = "beat times are not strictly increasing; sd_rr_s is too large for fs"
        raise ValueError(msg)
    if beats[-1] >= dur_s:
        msg = f"last beat {beats[-1]:.3f}s fell outside dur_s={dur_s}"
        raise ValueError(msg)
    return np.asarray(beats, dtype=np.float64)


def make_qrs(fs: float, width_ms: float = QRS_WIDTH_MS) -> F64:
    """Triphasic QRS kernel, dimensionless, peak exactly 1.0.

    A central positive lobe ``width_ms`` wide, flanked by two symmetric negative
    lobes half that width at :data:`FLANK_FRACTION` (0.15) of the peak. The kernel
    has odd length and its single maximum sits at the centre index, so convolving it
    onto a beat time puts the R peak on that beat time.

    The flanks must stay at or below 0.2 of the peak: above that the kernel's own
    rebound registers as a second local maximum and every beat is counted twice.

    Parameters
    ----------
    fs
        Sample rate in Hz.
    width_ms
        Width of the central positive lobe in milliseconds. Rat QRS is 8-12 ms.

    Returns
    -------
    numpy.ndarray
        Kernel of length ``w + 2*flank`` samples, where ``w`` is the odd sample count
        closest to ``width_ms`` and ``flank`` the odd count nearest ``w/2``. Both are
        odd so that each lobe has a centre sample and reaches its stated amplitude
        exactly; an even Hann window peaks between samples and would fall 0.02% short.
    """
    if width_ms <= 0.0:
        msg = f"width_ms must be positive, got {width_ms}"
        raise ValueError(msg)
    w = int(round(fs * width_ms / 1000.0))
    w = max(w | 1, MIN_QRS_SAMPLES)  # force odd so the peak has a centre sample
    flank = max((w // 2) | 1, 1)

    central = np.hanning(w)
    side = -FLANK_FRACTION * np.hanning(flank)
    return np.concatenate([side, central, side]).astype(np.float64)


class EcgSynth(NamedTuple):
    """Return of :func:`make_ecg`."""

    signal: F64
    """Microvolts, ``(n_samples,)``."""
    beats_s: F64
    """Ground-truth R times in seconds."""
    weak_idx: npt.NDArray[np.int64]
    """Indices into ``beats_s`` of the attenuated beats."""


WEAK_AMP_RANGE_UV: Final = (8.0, 13.0)
"""Amplitude range of an attenuated beat, microvolts.

At the default ``noise_uv`` of 5.0 the whole-signal threshold ``3 * 1.4826 * MAD``
is ~15 uV, so every weak beat falls below it and every normal beat is far above it.
:func:`tests.test_conftest.test_weak_beats_straddle_the_mad_threshold` holds that.

**That threshold is the broadband one.** Task 05 detects on a band-limited trace,
where MAD-sigma is 4-9x smaller (measured: 0.62 uV in 1-100 Hz, 2.89 uV in
10-150 Hz, against 5.46 uV broadband at ``noise_uv=5``), so an 8-13 uV beat is well
*above* the in-band threshold and ``weak_frac`` alone produces no missed beats.
Pass ``weak_amp_uv`` to place the attenuated beats against a band-limited threshold
instead.
"""

T_WAVE_PHASE_RANGE: Final = (0.50, 0.60)
"""Where a T wave sits after its R peak, as a fraction of the RR interval.

Measured on animal J: the short-interval population that seeded the plausibility
runaway clustered at **~90 ms, which is 0.55 x the 164.7 ms RR** - the signature of
a detector firing on the T wave as well as the R peak.
"""

T_WAVE_AMP_RANGE: Final = (0.30, 0.50)
"""T-wave amplitude as a fraction of its beat's R amplitude.

Large enough to be detected at a low threshold and to seed the runaway, small enough
that a correctly thresholded detector ignores it. Both ends matter: below ~0.3 the
short-interval population never appears, above ~0.5 every reasonable threshold
doubles every beat and the file is simply unusable rather than instructively bad.
"""

T_WAVE_WIDTH_FACTOR: Final = 6.0
"""T-wave width as a multiple of the QRS width - so 60 ms at the 10 ms default.

**Width, not amplitude, is what decides whether a T wave gets detected**, which was
not obvious and is worth recording. The band pass is a shape filter, so a narrow T
wave looks like a QRS however small it is. Measured at the spec's 30-50% amplitude,
counting peaks against 395 true beats:

======  =================  =================
width   10-150 Hz, k=6     10-150 Hz, k=3
======  =================  =================
40 ms   768 (all doubled)  768
50 ms   553                766
60 ms   **395 (correct)**  765
80 ms   395                764
======  =================  =================

60 ms is the crossover, and it is the useful place to sit: the binding detector
ignores the T wave while the superseded ``k=3`` doubles every beat. That makes one
generator serve both the "a good detector is not fooled" test and the runaway test.
Sweeping amplitude from 0.10 to 0.50 at a fixed width barely moves the count,
because sigma moves with the amplitude.
"""


def make_ecg(
    fs: float,
    dur_s: float,
    amp_uv: float = 60.0,
    weak_frac: float = 0.0,
    noise_uv: float = 5.0,
    seed: int = 0,
    weak_amp_uv: tuple[float, float] | None = None,
    rr_s: float = 0.150,
    t_wave: bool = False,
) -> EcgSynth:
    """Beat train convolved with the QRS kernel, on a noise floor.

    ``weak_frac`` of the beats are attenuated to 8-13 uV so they fall below a
    ``3 * 1.4826 * MAD`` threshold computed on the returned signal - the gap-rescue
    case. ``noise_uv`` is what makes that claim mean anything: with no background,
    MAD is 0 and nothing is below any multiple of it. At the default 5.0 the
    threshold is ~15 uV, above the 8-13 uV attenuated beats and well below the 60 uV
    normal ones. Note the real-data R-peak threshold is 6x MAD-sigma in 10-150 Hz
    (task 05), not 3x; this fixture's 3x is about the weak-beat separation only.

    The noise is the last quantity drawn from the generator, so the beat train,
    the choice of weak beats and their amplitudes depend only on ``seed``. Calling
    twice with the same seed and ``noise_uv=0.0`` yields the noiseless reference for
    the same signal.

    Parameters
    ----------
    fs
        Sample rate in Hz.
    dur_s
        Duration in seconds.
    amp_uv
        Peak amplitude of a normal beat, microvolts.
    weak_frac
        Fraction of beats attenuated into ``weak_amp_uv``. Rounded to the nearest
        whole beat.
    noise_uv
        SD of the additive white noise, microvolts. 0.0 gives a noiseless signal.
    seed
        Seed for beats, weak selection, weak amplitudes and noise.
    weak_amp_uv
        ``(lo, hi)`` amplitude range of an attenuated beat, microvolts. Defaults to
        :data:`WEAK_AMP_RANGE_UV`, which is calibrated against the **broadband** MAD
        threshold; a caller detecting in a band must set this against that band's
        own sigma, or the "weak" beats will not be weak. See
        :data:`WEAK_AMP_RANGE_UV`.
    rr_s
        Mean RR interval in seconds, passed to :func:`make_beats`. The default 0.150
        is 400 bpm, near animal J's measured 164.7 ms. Raise the rate to 600 bpm
        (``rr_s=0.100``) to reach the regime where the inherited 90 ms refractory
        deletes real beats.
    t_wave
        Add a broad second deflection per beat at :data:`T_WAVE_PHASE_RANGE` of the
        RR interval, at :data:`T_WAVE_AMP_RANGE` of that beat's R amplitude and
        :data:`T_WAVE_WIDTH_FACTOR` times the QRS width.

        **This is what makes the plausibility runaway reproducible.** Without it the
        signal has essentially no false-peak population, the local-median and global
        forms of the rule keep identical beats at every operating point, and the
        collapse measured on real data cannot be tested at all. ``beats_s`` still
        holds only the R times: a T wave is not a beat, and a detector that returns
        one has made a false detection, not a differently-placed true one.

    Returns
    -------
    EcgSynth
        ``(signal, beats_s, weak_idx)``.
    """
    if not 0.0 <= weak_frac <= 1.0:
        msg = f"weak_frac must be in [0, 1], got {weak_frac}"
        raise ValueError(msg)
    if noise_uv < 0.0:
        msg = f"noise_uv must be non-negative, got {noise_uv}"
        raise ValueError(msg)
    weak_lo, weak_hi = WEAK_AMP_RANGE_UV if weak_amp_uv is None else weak_amp_uv
    if not 0.0 < weak_lo <= weak_hi:
        msg = f"weak_amp_uv must be a positive ascending pair, got {(weak_lo, weak_hi)}"
        raise ValueError(msg)

    n = _n_samples(fs, dur_s)
    beats = make_beats(fs, dur_s, rr_s=rr_s, seed=seed)
    kernel = make_qrs(fs)
    half = kernel.size // 2

    rng = np.random.default_rng(seed + 1)
    n_weak = int(round(weak_frac * beats.size))
    weak_idx = np.sort(rng.choice(beats.size, size=n_weak, replace=False)).astype(np.int64)
    amps = np.full(beats.size, float(amp_uv), dtype=np.float64)
    if n_weak:
        amps[weak_idx] = rng.uniform(weak_lo, weak_hi, size=n_weak)

    sig = np.zeros(n, dtype=np.float64)
    for t_s, amp in zip(beats, amps, strict=True):
        centre = int(round(t_s * fs))
        lo, hi = centre - half, centre + half + 1
        k_lo, k_hi = max(0, -lo), kernel.size - max(0, hi - n)
        sig[max(lo, 0) : min(hi, n)] += amp * kernel[k_lo:k_hi]

    if t_wave:
        t_kernel = np.hanning(int(round(fs * QRS_WIDTH_MS * T_WAVE_WIDTH_FACTOR / 1000.0)) | 1)
        t_half = t_kernel.size // 2
        phases = rng.uniform(*T_WAVE_PHASE_RANGE, size=beats.size)
        fractions = rng.uniform(*T_WAVE_AMP_RANGE, size=beats.size)
        intervals = np.append(np.diff(beats), rr_s)
        for t_s, amp, phase, fraction, interval in zip(
            beats, amps, phases, fractions, intervals, strict=True
        ):
            centre = int(round((t_s + phase * interval) * fs))
            lo, hi = centre - t_half, centre + t_half + 1
            k_lo, k_hi = max(0, -lo), t_kernel.size - max(0, hi - n)
            if min(hi, n) > max(lo, 0):
                sig[max(lo, 0) : min(hi, n)] += fraction * amp * t_kernel[k_lo:k_hi]

    if noise_uv > 0.0:
        sig += rng.normal(0.0, noise_uv, size=n)
    return EcgSynth(sig, beats, weak_idx)


# ---------------------------------------------------------------------------
# neural and gastric content
# ---------------------------------------------------------------------------


class EngSynth(NamedTuple):
    """Return of :func:`make_eng`."""

    signal: F64
    """Microvolts, ``(n_samples,)``."""
    spike_times_s: F64
    """Ground-truth spike times in seconds (the positive-lobe peak)."""


SPIKE_WIDTH_MS: Final = 0.6
"""Total width of the biphasic spike waveform, milliseconds."""

STEP_EDGE_MS: Final = 0.1
"""Rise time of the ``'step'`` artifact, milliseconds - about 2.5 samples at 24.4 kHz.

A rail transition is limited by the front-end anti-alias filter, so at these rates it
is the fastest edge the sampled signal can carry. See :func:`inject_artifact` for why
this does not reproduce the 32.2% figure in A.5.
"""


def make_eng(
    fs: float,
    dur_s: float,
    rate_hz: float = 20.0,
    spike_uv: float = 80.0,
    noise_uv: float = 6.0,
    seed: int = 0,
) -> EngSynth:
    """Poisson spike train with a biphasic 0.6 ms waveform, plus white noise.

    The train is homogeneous Poisson with exponential inter-arrivals and **no
    refractory period**: two spikes may overlap. That is deliberate - a detector
    that silently assumes a refractory period should fail here rather than on real
    data.

    Parameters
    ----------
    fs
        Sample rate in Hz.
    dur_s
        Duration in seconds.
    rate_hz
        Mean firing rate in spikes per second.
    spike_uv
        Peak amplitude of the positive lobe, microvolts. The negative lobe is twice
        as long and scaled so the waveform integrates to exactly zero, which keeps
        the spike train from adding a DC offset into the slow bands.
    noise_uv
        SD of the additive white noise, microvolts.
    seed
        Seed for arrival times and noise.

    Returns
    -------
    EngSynth
        ``(signal, spike_times_s)``.
    """
    if rate_hz < 0.0:
        msg = f"rate_hz must be non-negative, got {rate_hz}"
        raise ValueError(msg)
    n = _n_samples(fs, dur_s)
    rng = np.random.default_rng(seed)

    times: list[float] = []
    if rate_hz > 0.0:
        t = float(rng.exponential(1.0 / rate_hz))
        while t < dur_s:
            times.append(t)
            t += float(rng.exponential(1.0 / rate_hz))
    spike_times = np.asarray(times, dtype=np.float64)

    # Biphasic: a positive lobe of width w/3 then a longer negative lobe, scaled so
    # the two areas cancel exactly (a trimmed Hann lobe of half-width p has area p).
    w = max(int(round(fs * SPIKE_WIDTH_MS / 1000.0)), MIN_QRS_SAMPLES)
    w_pos = max(w // 3, 1)
    w_neg = max(w - w_pos, 1)
    wave = np.concatenate(
        [
            np.hanning(2 * w_pos + 1)[1:-1],
            -(w_pos / w_neg) * np.hanning(2 * w_neg + 1)[1:-1],
        ]
    )
    peak_off = int(np.argmax(wave))

    sig = np.zeros(n, dtype=np.float64)
    for t_s in spike_times:
        start = int(round(t_s * fs)) - peak_off
        lo, hi = max(start, 0), min(start + wave.size, n)
        if hi > lo:
            sig[lo:hi] += spike_uv * wave[lo - start : hi - start]

    if noise_uv > 0.0:
        sig += rng.normal(0.0, noise_uv, size=n)
    return EngSynth(sig, spike_times)


class SlowSynth(NamedTuple):
    """Return of :func:`make_slow`."""

    signal: F64
    """Microvolts, ``(n_samples,)``."""
    peaks_s: F64
    """Ground-truth times of the wave maxima in seconds - what the ``slow_wave``
    consumer measures, and therefore what a mask must not displace."""


def make_slow(
    fs: float,
    dur_s: float,
    freq_hz: float = 0.05,
    amp_uv: float = 200.0,
    seed: int = 0,
    noise_uv: float = 0.0,
) -> SlowSynth:
    """Gastric slow wave: a sinusoid at ``freq_hz`` with a random phase.

    0.05 Hz is 3 cycles per minute, the rat gastric slow wave. The phase is the only
    random quantity unless ``noise_uv`` is set, so the peak times are exact.

    Parameters
    ----------
    fs
        Sample rate in Hz.
    dur_s
        Duration in seconds.
    freq_hz
        Slow-wave frequency in Hz.
    amp_uv
        Amplitude in microvolts; peak-to-peak is twice this.
    seed
        Seed for the initial phase and any noise.
    noise_uv
        SD of additive white noise, microvolts. Default 0 keeps the peaks exact.

    Returns
    -------
    SlowSynth
        ``(signal, peaks_s)``.
    """
    if freq_hz <= 0.0:
        msg = f"freq_hz must be positive, got {freq_hz}"
        raise ValueError(msg)
    n = _n_samples(fs, dur_s)
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    t = np.arange(n, dtype=np.float64) / fs
    sig = amp_uv * np.sin(2.0 * np.pi * freq_hz * t + phase)

    # sin peaks where 2*pi*f*t + phase == pi/2 + 2*pi*k
    k0 = math.ceil((phase - 0.5 * np.pi) / (2.0 * np.pi))
    k_max = math.floor((2.0 * np.pi * freq_hz * dur_s + phase - 0.5 * np.pi) / (2.0 * np.pi))
    ks = np.arange(k0, k_max + 1, dtype=np.float64)
    peaks = (0.5 * np.pi + 2.0 * np.pi * ks - phase) / (2.0 * np.pi * freq_hz)

    if noise_uv > 0.0:
        sig = sig + rng.normal(0.0, noise_uv, size=n)
    return SlowSynth(sig, peaks[(peaks >= 0.0) & (peaks < dur_s)])


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


class TruthSpan(NamedTuple):
    """Ground truth for one injected artifact.

    A tuple so it unpacks as ``(start_s, stop_s, ...)``, with names so a test can say
    what it means.
    """

    start_s: float
    """Span start in seconds, snapped to the sample actually written."""
    stop_s: float
    """Span end in seconds, exclusive, snapped to the sample actually written."""
    kind: ArtifactKind
    """Which waveform was injected."""
    amp_ratio: float
    """For an additive kind, the requested peak in units of the host's pre-injection
    sigma. For ``'clip'`` it is the rail in those units, which is not an amplitude:
    a *smaller* value means more damage."""
    sigma_uv: float
    """The host's pre-injection ``1.4826 * MAD``, microvolts."""
    peak_uv: float
    """Realised peak of the injected component, microvolts, equal to
    ``amp_ratio * sigma_uv`` by construction. ``nan`` for ``'clip'``, which injects
    nothing - it removes."""
    rail_uv: float
    """Clip rail in microvolts. ``nan`` for every additive kind."""
    additive: bool
    """Whether ``out - in`` is the artifact. False only for ``'clip'``."""


def inject_artifact(
    sig: F64,
    fs: float,
    t0_s: float,
    dur_s: float,
    kind: ArtifactKind,
    amp_ratio: float,
    seed: int = 0,
) -> tuple[F64, TruthSpan]:
    """Apply an artifact of known amplitude to a copy of ``sig``.

    For the four additive kinds ``amp_ratio`` is the peak of the injected component
    in units of the host's pre-injection robust sigma, so the measured ratio

    ``max|out - in|`` over the span ``/ robust_sigma(in)``

    equals ``amp_ratio`` by construction and the ground truth is exactly
    recoverable. Where the host is NaN the output stays NaN, as it must.

    ``'clip'`` is the exception and is deliberately **not additive**: it hard-limits
    the host at the rail instead of adding to it. Task 14 routes clipping around the
    classifier precisely because saturation is non-linear and the envelope can
    *understate* the damage - an additive flat-top would not exercise that path. For
    ``'clip'`` the ground truth is the span and ``rail_uv``, not ``amp_ratio``.

    Parameters
    ----------
    sig
        Host signal in microvolts. Not modified.
    fs
        Sample rate in Hz.
    t0_s, dur_s
        Span start and length in seconds. Must lie inside the record.
    kind
        ``'excursion'``
            A band-limited (1-20 Hz) tug with fast Tukey flanks. Measured >99.99%
            of its energy below 300 Hz - **smoother than real smooth motion**, which
            A.5 puts at 98.7-99.7%. It therefore under-tests a detector's ENG-band
            response to motion; use ``'tribo'`` for that.
        ``'drift'``
            One raised-cosine swell across the whole span; energy at ~1/dur_s Hz.
            >99.99% below 300 Hz. Deterministic - ``seed`` is unused.
        ``'step'``
            A flat-topped step with :data:`STEP_EDGE_MS` raised-cosine edges, the
            additive flat-top formerly called ``'saturation'``. ``seed`` is unused.
            Measured 18.4% of its energy above 300 Hz in a 10 ms window around the
            edge, 6.4% in 20 ms, 2.4% in 50 ms: a step's spectrum is 1/f, so the
            fraction is set by the analysis window, not the rise time (an
            instantaneous edge gives 18.6% in 10 ms). A.5's 32.2% quotes no window
            and is not reproducible from a step alone - it needs a window near 5 ms.
            Do not tune the edge to chase it.
        ``'tribo'``
            A burst of exponentially decaying triboelectric crackle, alternating
            polarity, tau = 2 ms. Measured 21-25% of its energy above 300 Hz,
            broadband by construction - the one additive kind that reaches the ENG
            band.
        ``'clip'``
            Hard-limits the host to ``+/- amp_ratio * sigma`` inside the span.
            ``seed`` is unused. Non-linear and **energy-removing**: the clipped span
            has a lower RMS than the original, so an envelope detector sees less
            signal, not more, which is why this kind exists.
    amp_ratio
        For the additive kinds, peak amplitude relative to the host's robust sigma.
        For ``'clip'``, the rail in the same units - so a smaller value clips harder,
        and a value above the host's peak does nothing at all.
    seed
        Seed for the stochastic kinds (``'excursion'`` and ``'tribo'``).

    Returns
    -------
    tuple
        ``(signal, truth_span)``.
    """
    if sig.ndim != 1:
        msg = f"sig must be 1-D, got shape {sig.shape}"
        raise ValueError(msg)
    n = sig.size
    i0 = int(round(t0_s * fs))
    i1 = i0 + _n_samples(fs, dur_s)
    if i0 < 0 or i1 > n:
        msg = f"span [{t0_s}, {t0_s + dur_s}) s does not fit in a {n / fs:.3f} s record"
        raise ValueError(msg)

    sigma = robust_sigma(sig)
    if not math.isfinite(sigma) or sigma == 0.0:
        msg = "host signal has zero or undefined robust sigma; amp_ratio is meaningless"
        raise ValueError(msg)
    peak_uv = amp_ratio * sigma

    m = i1 - i0
    t = np.arange(m, dtype=np.float64) / fs
    rng = np.random.default_rng(seed)

    if kind == "clip":
        # Non-linear: nothing is injected, the rails take signal away. NaN survives
        # np.clip, so a masked sample stays masked.
        out = sig.copy()
        out[i0:i1] = np.clip(out[i0:i1], -peak_uv, peak_uv)
        return out, TruthSpan(
            start_s=i0 / fs,
            stop_s=i1 / fs,
            kind=kind,
            amp_ratio=amp_ratio,
            sigma_uv=sigma,
            peak_uv=float("nan"),
            rail_uv=peak_uv,
            additive=False,
        )

    comp: F64
    if kind == "drift":
        comp = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (m / fs)))
    elif kind == "excursion":
        freqs = rng.uniform(1.0, 20.0, size=6)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=6)
        comp = np.sum(
            [np.sin(2.0 * np.pi * f * t + p) for f, p in zip(freqs, phases, strict=True)], axis=0
        ) * _tukey(m, 0.2)
    elif kind == "step":
        edge = max(int(round(STEP_EDGE_MS * fs / 1000.0)), 1)
        alpha = min(2.0 * edge / m, 1.0)
        comp = _tukey(m, alpha)
    elif kind == "tribo":
        tau_s = 0.002
        comp = np.zeros(m, dtype=np.float64)
        n_burst = max(int(round(dur_s / (4.0 * tau_s))), 1)
        starts = np.sort(rng.integers(0, m, size=n_burst))
        for j, s in enumerate(starts):
            decay = np.exp(-(np.arange(m - s, dtype=np.float64) / fs) / tau_s)
            comp[s:] += (1.0 if j % 2 == 0 else -1.0) * decay
        comp *= _tukey(m, 0.1)
    else:
        # Unreachable for a caller mypy has checked, which is exactly why it is here:
        # ArtifactKind is a Literal, and only an unchecked caller can get this far.
        msg = f"unknown artifact kind {kind!r}"  # type: ignore[unreachable]
        raise ValueError(msg)

    scale = float(np.max(np.abs(comp)))
    if scale == 0.0:
        msg = f"{kind} component came out empty"
        raise ValueError(msg)
    comp = comp * (peak_uv / scale)

    out = sig.copy()
    out[i0:i1] += comp
    return out, TruthSpan(
        start_s=i0 / fs,
        stop_s=i1 / fs,
        kind=kind,
        amp_ratio=amp_ratio,
        sigma_uv=sigma,
        peak_uv=peak_uv,
        rail_uv=float("nan"),
        additive=True,
    )


# ---------------------------------------------------------------------------
# a whole recording
# ---------------------------------------------------------------------------


class MultichannelTruth(TypedDict):
    """Ground truth accompanying :func:`make_multichannel`."""

    fs: float
    seed: int
    common_mode: F64
    """The shared artifact trace before per-contact gains, microvolts. All zeros when
    ``common_mode=False``."""
    gains: dict[str, float]
    """Per-channel gain applied to :data:`common_mode`. Unequal on purpose: equal
    gains would let a naive 0.5/0.5 tripole cancel the common mode perfectly, which
    real contact impedances never do."""
    neural: dict[str, F64]
    """Per-nerve-channel independent spike content, microvolts, before the common
    mode is added."""
    spike_times_s: dict[str, F64]
    """Ground-truth spike times per nerve channel."""
    slow: dict[str, F64]
    """Per-stomach-channel slow wave, microvolts, before the common mode."""
    slow_peaks_s: dict[str, F64]
    """Ground-truth slow-wave peak times per stomach channel."""
    artifact_spans: list[TruthSpan]
    """Artifacts injected into the common-mode trace."""
    noise_uv: float


def make_multichannel(
    fs: float,
    dur_s: float,
    n_cuff: int = 2,
    n_stomach: int = 3,
    common_mode: bool = True,
    seed: int = 0,
) -> tuple[Recording, MultichannelTruth]:
    """Build a full synthetic recording: nerve cuffs, stomach EMG, known common mode.

    Each cuff contributes three independent contacts (``V1..V3``), each with its own
    spike train, its own noise and its own gain on the shared common-mode trace. The
    default 2 cuffs and 3 stomach channels give the 9 channels of the real rig.

    The common-mode trace carries the artifacts, which is what makes this fixture
    the one to use for the tripole and cross-channel-agreement work: the artifact is
    common to every channel, the neural content is not.

    Parameters
    ----------
    fs
        Sample rate in Hz.
    dur_s
        Duration in seconds. Must hold at least a few RR intervals' worth of signal.
    n_cuff
        Number of cuffs, 1 or 2. Cuffs are named ``"L"`` then ``"R"``.
    n_stomach
        Number of stomach channels.
    common_mode
        When False the common-mode trace is identically zero and no artifact is
        injected; the ground truth still has the keys.
    seed
        Master seed. Every channel derives its own seed from it deterministically.

    Returns
    -------
    tuple
        ``(Recording, MultichannelTruth)``.
    """
    if not 1 <= n_cuff <= 2:
        msg = f"n_cuff must be 1 or 2, got {n_cuff}"
        raise ValueError(msg)
    if n_stomach < 0:
        msg = f"n_stomach must be non-negative, got {n_stomach}"
        raise ValueError(msg)

    n = _n_samples(fs, dur_s)
    rng = np.random.default_rng(seed)
    noise_uv = 6.0

    shared = np.zeros(n, dtype=np.float64)
    spans: list[TruthSpan] = []
    if common_mode:
        # Additive kinds only. A clip rail belongs to one amplifier channel, so it
        # cannot live in a trace that is shared across channels at different gains;
        # apply 'clip' to a column of rec.data if a test needs a saturated channel.
        # A slow wander the artifacts sit on, so robust_sigma of the trace is defined.
        t = np.arange(n, dtype=np.float64) / fs
        shared = 5.0 * np.sin(2.0 * np.pi * 0.3 * t + float(rng.uniform(0, 2 * np.pi)))
        schedule: list[tuple[ArtifactKind, float, float, float]] = [
            ("excursion", 0.20, 0.15, 30.0),
            ("drift", 0.50, 0.40, 12.0),
            ("step", 0.75, 0.05, 60.0),
        ]
        for k, (kind, frac, span_s, ratio) in enumerate(schedule):
            t0 = frac * dur_s
            if t0 + span_s > dur_s:
                continue
            shared, span = inject_artifact(shared, fs, t0, span_s, kind, ratio, seed=seed + k)
            spans.append(span)

    channels: list[ChannelInfo] = []
    columns: list[F64] = []
    truth: MultichannelTruth = {
        "fs": fs,
        "seed": seed,
        "common_mode": shared,
        "gains": {},
        "neural": {},
        "spike_times_s": {},
        "slow": {},
        "slow_peaks_s": {},
        "artifact_spans": spans,
        "noise_uv": noise_uv,
    }

    col = 0
    for c, cuff in enumerate(("L", "R")[:n_cuff]):
        for contact in (1, 2, 3):
            name = f"{cuff}VN{contact}"
            sub = seed + 100 * (c + 1) + contact
            eng = make_eng(fs, dur_s, noise_uv=noise_uv, seed=sub)
            gain = float(rng.uniform(0.80, 1.25))
            channels.append(
                ChannelInfo(
                    index=col,
                    name=name,
                    role="nerve",
                    cuff_id=cuff,
                    contact_index=contact,
                    rostral_end=1,
                    config="independent",
                )
            )
            columns.append(eng.signal + gain * shared)
            truth["gains"][name] = gain
            truth["neural"][name] = eng.signal
            truth["spike_times_s"][name] = eng.spike_times_s
            col += 1

    for s in range(n_stomach):
        name = f"ANT{s + 1}"
        slow = make_slow(fs, dur_s, seed=seed + 900 + s, noise_uv=noise_uv)
        gain = float(rng.uniform(0.80, 1.25))
        channels.append(
            ChannelInfo(
                index=col,
                name=name,
                role="stomach",
                cuff_id=None,
                contact_index=None,
                rostral_end=None,
                config="independent",
            )
        )
        columns.append(slow.signal + gain * shared)
        truth["gains"][name] = gain
        truth["slow"][name] = slow.signal
        truth["slow_peaks_s"][name] = slow.peaks_s
        col += 1

    rec = Recording(
        fs=fs,
        data=np.column_stack(columns).astype(np.float64),
        channels=channels,
        animal="SYNTH",
        session=f"seed{seed}",
        path=Path("synthetic") / f"seed{seed}.dat",
    )
    return rec, truth


def make_common_mode(
    fs: float,
    dur_s: float,
    lo_hz: float = 20.0,
    hi_hz: float = 300.0,
    sigma_uv: float = 50.0,
    n_components: int = 24,
    seed: int = 0,
) -> F64:
    """Band-limited common-mode trace, for the tripole-rejection tests.

    Built from summed sinusoids with random phases rather than by filtering noise,
    like every other component here, so the band limits are exact and no filter
    design can misbehave. Scaled so :func:`robust_sigma` is ``sigma_uv``.

    ``make_multichannel``'s common mode sits at 0.3 Hz plus injected artifacts,
    which is the right shape for detection tests and the wrong one for a fit that
    minimises variance in 20-300 Hz - nothing of it lands in that band. This gives
    a common mode the fit can actually see.

    Parameters
    ----------
    fs
        Sample rate in Hz.
    dur_s
        Duration in seconds.
    lo_hz, hi_hz
        Band the components are drawn from, inclusive.
    sigma_uv
        Target robust scale of the result, microvolts.
    n_components
        How many sinusoids to sum. More makes the amplitude distribution more
        Gaussian; 24 is already indistinguishable for these purposes.
    seed
        Seed for the frequencies and phases.
    """
    if not 0.0 < lo_hz < hi_hz:
        msg = f"need 0 < lo_hz < hi_hz, got {lo_hz} and {hi_hz}"
        raise ValueError(msg)
    n = _n_samples(fs, dur_s)
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / fs

    freqs = rng.uniform(lo_hz, hi_hz, size=n_components)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    trace = np.sum(
        [np.sin(2.0 * np.pi * f * t + p) for f, p in zip(freqs, phases, strict=True)], axis=0
    )
    scale = robust_sigma(trace)
    if scale == 0.0:  # pragma: no cover - needs a degenerate draw
        msg = "common-mode components cancelled; try another seed"
        raise ValueError(msg)
    return np.asarray(trace * (sigma_uv / scale), dtype=np.float64)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the stimulation monitor
# ---------------------------------------------------------------------------


class VibSynth(NamedTuple):
    """Return of :func:`make_vib`."""

    signal: F64
    """The monitor channel, arbitrary units, ``(n_samples,)``."""
    stim_start_s: F64
    """Ground-truth stim onset in seconds. A scalar array so it prints like the rest."""
    stim_stop_s: F64
    """Ground-truth stim offset in seconds, exclusive."""


VIB_CARRIER_HZ: Final = 60.0
"""Carrier frequency of the vibration monitor, Hz.

The monitor records a mechanical oscillation, so it is a carrier gated on and off
rather than a step. That matters for the envelope: a gated carrier has a 100 ms RMS
envelope with ramped edges, which is what the edge refinement exists to handle.
"""

VIB_ON_AMP: Final = 1.0
"""Carrier amplitude while stimulating, arbitrary units.

Arbitrary on purpose: the split is a contrast on the monitor's own envelope, so its
scale never enters a decision and it carries no declared units. See
:func:`tests.test_stim_split.test_the_split_is_invariant_to_the_monitors_scale`.
"""

VIB_OFF_NOISE: Final = 0.02
"""OFF-state noise, as a fraction of :data:`VIB_ON_AMP`.

Not zero, and that is deliberate: with a noiseless OFF state the envelope's bottom
decile has no spread, the ``off_level + 8 sigma`` threshold degenerates onto the OFF
level itself, and the edge refinement would be testing a fallback rather than the
rule.
"""


def make_vib(
    fs: float,
    dur_s: float,
    stim_start_s: float,
    stim_duration_s: float,
    *,
    carrier_hz: float = VIB_CARRIER_HZ,
    amp: float = VIB_ON_AMP,
    noise: float = VIB_OFF_NOISE,
    dropout: tuple[float, float] | None = None,
    contaminant: tuple[float, float, float] | None = None,
    seed: int = 0,
) -> VibSynth:
    """Build a stimulation-monitor channel: a gated carrier on a noise floor.

    The ground-truth boundaries are returned, so a detector can be scored against
    them rather than against another detector.

    Parameters
    ----------
    fs
        Monitor sample rate in Hz. Need not equal the signal's - the real monitor is
        recorded on its own clock, which is why :func:`make_vib` takes its own rate.
    dur_s
        Duration in seconds.
    stim_start_s
        Onset of the gated carrier, seconds. Pass a negative value to start the
        recording mid-stim: the carrier is then ON at the first sample, which is the
        ``clipped_start`` case.
    stim_duration_s
        How long the carrier is ON from ``stim_start_s``, seconds. The gate is
        truncated at the end of the record, which is the ``clipped_end`` case.
    carrier_hz
        Carrier frequency, Hz.
    amp
        Carrier amplitude while ON, arbitrary units.
    noise
        SD of the additive noise everywhere, in the same arbitrary units.
    dropout
        ``(start_s, duration_s)`` of a gap in the middle of the stim, in absolute
        seconds. The gap in the real data is a stimulator interruption; a matched
        filter spans it because the window width is fixed, whereas a threshold plus
        "keep the largest ON segment" keeps only one side of it.
    contaminant
        ``(start_s, duration_s, amplitude_fraction)`` of a second, quieter burst -
        handling noise, say - at ``amplitude_fraction`` of the stim carrier.

        This is what makes the old ``(p20 + p80)/2`` rule fail, and finding it took
        two attempts worth recording. With a stationary floor the old threshold is
        badly wrong - it flags 46% of the record at an 8% duty cycle - but "keep the
        largest ON segment" still recovers the onset to 74 ms, so a clean synthetic
        cannot demonstrate any failure. A drifting noise floor does not do it either,
        measured out to a 20x ramp, because the stim carrier stays the loudest thing
        in the record. What breaks "largest segment" is a *longer* supra-threshold
        run: a quieter burst lasting more than the stim duration wins on length while
        losing on amplitude, so the old rule locks onto it and the matched-width
        search does not.
    seed
        Seed for the noise.

    Returns
    -------
    VibSynth
        ``(signal, stim_start_s, stim_stop_s)``, the boundaries clipped to the record.
    """
    if dur_s <= 0.0:
        msg = f"dur_s must be positive, got {dur_s}"
        raise ValueError(msg)
    if stim_duration_s <= 0.0:
        msg = f"stim_duration_s must be positive, got {stim_duration_s}"
        raise ValueError(msg)
    if noise < 0.0:
        msg = f"noise must be non-negative, got {noise}"
        raise ValueError(msg)

    n = _n_samples(fs, dur_s)
    t_s = np.arange(n, dtype=np.float64) / fs
    gate = (t_s >= stim_start_s) & (t_s < stim_start_s + stim_duration_s)
    if dropout is not None:
        gap_start, gap_dur = dropout
        gate &= ~((t_s >= gap_start) & (t_s < gap_start + gap_dur))

    rng = np.random.default_rng(seed)
    sig = amp * gate * np.sin(2.0 * np.pi * carrier_hz * t_s)
    if contaminant is not None:
        burst_start, burst_dur, fraction = contaminant
        burst = (t_s >= burst_start) & (t_s < burst_start + burst_dur)
        sig = sig + fraction * amp * burst * np.sin(2.0 * np.pi * carrier_hz * t_s)
    sig = sig + rng.normal(0.0, noise * amp, size=n)

    return VibSynth(
        np.asarray(sig, dtype=np.float64),
        np.asarray(max(stim_start_s, 0.0), dtype=np.float64),
        np.asarray(min(stim_start_s + stim_duration_s, n / fs), dtype=np.float64),
    )


@pytest.fixture(autouse=True)
def isolate_user_config(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every test away from the developer's real per-user config and GEMS_ROOT.

    Autouse and unconditional. Without it a test that expects "no root configured"
    passes on a clean machine and fails on one where someone has run ``gems init`` -
    which is precisely the non-hermetic behaviour that makes a suite untrustworthy
    on CI. Discovered the hard way: running ``gems init`` once broke two tests.
    """
    fake_home = tmp_path_factory.mktemp("user-config")
    monkeypatch.setattr("gems_blanking_v2.io.store.config_path", lambda: fake_home / "config.toml")
    monkeypatch.setattr(
        "gems_blanking_v2.io.store.legacy_config_path", lambda: fake_home / ".gems" / "config.toml"
    )
    monkeypatch.delenv("GEMS_ROOT", raising=False)


@pytest.fixture
def fs() -> float:
    """Nominal TDT sample rate, Hz. Tests that need another rate pass it explicitly."""
    return FS_NOMINAL_HZ


@pytest.fixture
def seed() -> int:
    """Default seed. Every generator takes one explicitly; this is the shared default."""
    return 0


RUNAWAY_RECORDING_STEM: Final = "gems_j_t01_ms3_bl_230315"
"""The animal-J baseline the plausibility runaway was measured on (601 s, 3511 beats).

Named in task 05 so the real-data regression has a fixed target rather than
whichever file happens to be around.
"""


@pytest.fixture
def real_recording() -> Path | None:
    """Return :data:`RUNAWAY_RECORDING_STEM` on this machine, or ``None``.

    ``None`` rather than a skip, so the test decides and says why. The shared drive
    is not mounted in CI and usually not on a dev machine either, so a real-data test
    has to be optional or it is just red.

    Discovery goes through the store, never a hardcoded path - the root differs by
    platform and the drive name is not even the same string on Windows
    (cross-platform rules 2-4). ``GEMS_ROOT`` is unset by the autouse
    :func:`isolate_user_config` fixture, so this looks for the marker from the
    current directory upward and returns ``None`` when there is none.
    """
    from gems_blanking_v2.io.store import MARKER_NAME  # noqa: PLC0415

    root: Path | None = None
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / MARKER_NAME).exists():
            root = candidate
            break
    if root is None:
        return None

    for extension in (".mat", ".h5"):
        hits = sorted(root.rglob(f"{RUNAWAY_RECORDING_STEM}{extension}"))
        if hits:
            return hits[0]
    return None
