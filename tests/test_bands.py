"""Tests for :mod:`gems_blanking_v2.bands`: envelopes, the reference pair and z.

Every numeric claim in the modules' docstrings has a test here that would fail if it
were wrong. One of the task's required tests could not pass as specified and the
measurement that fixed it is in
:func:`test_every_band_matches_its_own_degrees_of_freedom`.
"""

from __future__ import annotations

import logging
import math
import tracemalloc
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.bands.envelope import (
    ENVELOPE_FLOOR_UV,
    MIN_SEGMENT_CYCLES,
    SETTLING_WINDOWS,
    FilterSanityError,
    analysis_epoch,
    assert_filter_sane,
    band_envelope,
    band_envelope_for,
    log_envelope,
    settling_s,
    valid_segments,
)
from gems_blanking_v2.bands.reference import (
    MIN_REFERENCE_FRAMES,
    SCALE_FLOOR,
    DeadChannelError,
    Reference,
    epoch_reference,
    is_dead,
)
from gems_blanking_v2.bands.zscore import zscore
from gems_blanking_v2.constants import BANDS, GRID_S, MAD_TO_SIGMA, REFERENCE_STATISTIC
from gems_blanking_v2.types import ChannelInfo, Recording
from scipy.signal import butter, lfilter

F64 = npt.NDArray[np.float64]

DOF_SEEDS = tuple(range(8))
"""Seeds the degrees-of-freedom estimate is pooled over. Fixed, per the testing rules."""

DOF_RECORD_S = 600.0
"""Record length for the dof check.

Ten minutes, because the slow bands hold only 80-100 independent windows in that and
a shorter record cannot resolve a 20% claim at all - the per-seed estimate for
``0.5-3`` scatters by a third.
"""


def _white(fs: float, dur_s: float, sigma_uv: float = 10.0, seed: int = 0) -> F64:
    """White noise of known sigma, microvolts.

    Not from ``conftest``: the generators there produce physiological signals with
    structure, and these tests need a process whose band-limited variance and
    effective degrees of freedom are analytically known. Every call takes a seed.
    """
    return np.asarray(
        np.random.default_rng(seed).normal(0.0, sigma_uv, size=int(round(dur_s * fs))),
        dtype=np.float64,
    )


def _measured_dof(env: F64) -> float:
    """Return the effective degrees of freedom implied by an envelope's spread.

    For a Gaussian process the power estimate over a window has relative variance
    ``2/k`` with ``k = 2*B*T``, so ``k = 2 / (var(P)/mean(P)^2)``.
    """
    power = env[np.isfinite(env)] ** 2
    return 2.0 / float(power.var() / power.mean() ** 2)


def _recording(data: F64, fs: float, name: str = "LVN1") -> Recording:
    """Build a one-channel recording, for the precondition tests."""
    return Recording(
        fs=fs,
        data=data.reshape(-1, 1),
        channels=[ChannelInfo(0, name, "nerve", "L", 1, 1, "independent")],
        animal="J",
        session="t01",
        path=Path("gems_j_t01_bl_120000.mat"),
    )


# ---------------------------------------------------------------------------
# the envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band", list(BANDS))
def test_the_envelope_matches_the_analytic_expectation(band: str, fs: float) -> None:
    """White noise of known sigma through each band, within 5%.

    For white noise of variance ``s^2`` over the full 0 to Nyquist range, the variance
    surviving a band of width ``B`` is ``s^2 * B / (fs/2)``, so the RMS envelope should
    sit at ``s * sqrt(B / nyquist)``.

    Two details the first draft of this test got wrong, both worth stating:

    The comparison uses the filter's **equivalent noise bandwidth**, not the nominal
    width. A 4th-order Butterworth through ``filtfilt`` passes about 90% of its
    nominal band, which is twice the 5% being asserted.

    And the Nyquist is the **input's**, not the decimated one. Decimation removes the
    noise above the new Nyquist but leaves the density in the retained band untouched,
    so the in-band variance is set by the rate the signal was generated at. Using the
    decimated Nyquist inflates the expectation by ``sqrt(12)`` for the slow bands.
    """
    from gems_blanking_v2.bands.envelope import DECIMATE_TARGET_HZ  # noqa: PLC0415
    from scipy.signal import sosfreqz  # noqa: PLC0415

    spec = BANDS[band]
    sigma_uv = 10.0
    envelope = band_envelope_for(_white(fs, 120.0, sigma_uv, seed=0), fs, band)

    rate = fs
    if spec.hi_hz < DECIMATE_TARGET_HZ / 2:
        rate = fs / max(int(fs // DECIMATE_TARGET_HZ), 1)
    nyquist = rate / 2.0
    hi = min(spec.hi_hz, 0.9 * nyquist)
    sos = (
        butter(4, hi / nyquist, btype="lowpass", output="sos")
        if spec.lo_hz <= 0
        else butter(4, [spec.lo_hz / nyquist, hi / nyquist], btype="bandpass", output="sos")
    )
    frequencies, response = sosfreqz(sos, worN=100_000, fs=rate)
    power = np.abs(response) ** 4
    noise_bandwidth = float(np.trapezoid(power, frequencies) / power.max())

    expected = sigma_uv * math.sqrt(noise_bandwidth / (fs / 2.0))
    measured = float(np.mean(envelope[np.isfinite(envelope)]))

    assert measured == pytest.approx(expected, rel=0.05), (
        f"{band}: envelope {measured:.4f} uV against an analytic {expected:.4f} uV"
    )


def test_every_band_matches_its_own_degrees_of_freedom(fs: float) -> None:
    """``2*B*T`` per band, within 20% - **and two things had to be right first**.

    The task asks for "the variance of the envelope matches ``2*B*T ~= 30`` within 20%
    for every band". Two corrections were needed to make that testable:

    **It is each band's own dof, not 30.** The ENG band's is 135 by design and
    ``10-150``'s is 28. ``constants.py`` says this test "must exempt" the ENG band; it
    does not need to, because comparing each band's measured dof to *its own* spec
    value is both stronger and passes everywhere.

    **The filter settling transient had to go.** Untrimmed, ``0.5-3`` measures 22.9
    against a spec 30 and individual seeds come back as low as 0.7 - the transient
    accounting for more envelope variance than the signal. See
    :data:`~gems_blanking_v2.bands.envelope.SETTLING_WINDOWS`.

    Pooled over 8 seeds of 10-minute white noise: 135.3, 30.6, 28.5, 29.9, 32.0, 31.7
    against specs of 135.0, 30.0, 28.0, 29.8, 30.0, 30.0 - every band inside 7%.
    """
    for band, spec in BANDS.items():
        estimates = [
            _measured_dof(band_envelope_for(_white(fs, DOF_RECORD_S, seed=seed), fs, band))
            for seed in DOF_SEEDS
        ]
        measured = float(np.mean(estimates))
        assert measured == pytest.approx(spec.dof, rel=0.20), (
            f"{band}: measured dof {measured:.1f} against a spec {spec.dof:.1f}"
        )


def test_trimming_the_settling_window_is_what_makes_the_slow_bands_work(fs: float) -> None:
    """The measurement behind :data:`SETTLING_WINDOWS`, held so it cannot be dropped.

    With the trim the 0.5-3 band's effective dof is 32.0; without it, 22.9, and
    individual seeds come back as low as 0.7 - the transient accounting for more
    envelope variance than the signal. The fast bands are untouched either way, which
    is why it went unnoticed until the slow bands were measured.

    The untrimmed arm is produced by setting the constant to zero rather than by
    reimplementing the envelope, so the two arms differ in exactly one thing.
    """
    from gems_blanking_v2.bands import envelope as envelope_module  # noqa: PLC0415

    band = "0.5-3"
    spec = BANDS[band]

    measured: dict[float, list[float]] = {}
    for windows in (SETTLING_WINDOWS, 0.0):
        original = envelope_module.SETTLING_WINDOWS
        try:
            envelope_module.SETTLING_WINDOWS = windows  # type: ignore[misc]
            measured[windows] = [
                _measured_dof(
                    band_envelope_for(_white(fs, DOF_RECORD_S, seed=seed), fs, band)
                )
                for seed in DOF_SEEDS
            ]
        finally:
            envelope_module.SETTLING_WINDOWS = original  # type: ignore[misc]

    trimmed = float(np.mean(measured[SETTLING_WINDOWS]))
    untrimmed = float(np.mean(measured[0.0]))

    assert trimmed == pytest.approx(32.0, rel=0.10)
    assert untrimmed == pytest.approx(22.9, rel=0.20)
    assert trimmed == pytest.approx(spec.dof, rel=0.20)
    assert untrimmed < 0.85 * spec.dof, "the transient should visibly break the estimate"
    assert min(measured[0.0]) < 5.0, "at least one untrimmed seed should collapse"
    assert min(measured[SETTLING_WINDOWS]) > 20.0, "no trimmed seed should collapse"


def test_the_fast_bands_are_unaffected_by_the_settling_trim(fs: float) -> None:
    """Which is why the transient stayed hidden: it only bites where the window is long."""
    from gems_blanking_v2.bands import envelope as envelope_module  # noqa: PLC0415

    for band in ("300-3000", "100-300"):
        both = []
        for windows in (SETTLING_WINDOWS, 0.0):
            original = envelope_module.SETTLING_WINDOWS
            try:
                envelope_module.SETTLING_WINDOWS = windows  # type: ignore[misc]
                both.append(
                    _measured_dof(band_envelope_for(_white(fs, 120.0, seed=0), fs, band))
                )
            finally:
                envelope_module.SETTLING_WINDOWS = original  # type: ignore[misc]
        assert both[0] == pytest.approx(both[1], rel=0.02), band


def test_the_settling_edges_come_back_as_nan(fs: float) -> None:
    """One analysis window at each end of a segment has no assessable envelope."""
    for band, spec in BANDS.items():
        envelope = band_envelope_for(_white(fs, 120.0, seed=0), fs, band)
        finite = np.flatnonzero(np.isfinite(envelope))
        expected = int(round(settling_s(spec.window_s) / GRID_S))

        assert finite.size > 0, band
        assert int(finite[0]) >= expected - 1, band
        assert envelope.size - 1 - int(finite[-1]) >= expected - 1, band


def test_the_envelope_is_on_the_shared_grid(fs: float) -> None:
    """``n_frames = floor(duration / 0.010)``, whatever the band (A.2)."""
    for dur_s in (30.0, 47.5, 120.0):
        expected = int(math.floor(dur_s / GRID_S))
        for band in BANDS:
            envelope = band_envelope_for(_white(fs, dur_s, seed=0), fs, band)
            assert envelope.size == expected, f"{band} at {dur_s} s"


def test_the_sample_rate_is_read_never_assumed() -> None:
    """The same physical signal at two rates gives the same envelope in microvolts.

    A sine rather than noise, deliberately. White noise of a fixed sigma is **not** the
    same signal at two rates - its power spectral density is ``sigma^2/(fs/2)``, so
    halving the rate doubles the density and the in-band envelope rises by ``sqrt(2)``.
    That is physics, not a rate bug, and a test built on it would have asserted the
    wrong thing. A 200 Hz sine of amplitude ``A`` has an RMS of ``A/sqrt(2)`` whatever
    the rate.
    """
    band = "100-300"
    amplitude_uv = 50.0
    envelopes = []
    for rate in (24414.0625, 12207.03125):
        t_s = np.arange(int(60.0 * rate)) / rate
        signal = amplitude_uv * np.sin(2.0 * np.pi * 200.0 * t_s)
        envelopes.append(band_envelope_for(signal, rate, band))

    assert envelopes[0].size == envelopes[1].size
    means = [float(np.mean(e[np.isfinite(e)])) for e in envelopes]
    assert means[0] == pytest.approx(amplitude_uv / math.sqrt(2.0), rel=0.02)
    assert means[0] == pytest.approx(means[1], rel=0.02)


# ---------------------------------------------------------------------------
# the filter-stability guard
# ---------------------------------------------------------------------------


def test_the_filter_guard_fires_on_an_ill_conditioned_ba_design(fs: float) -> None:
    """**The bug this exists for**, reproduced: ``butter(..., 'ba')`` at a 1 Hz corner.

    A 1 Hz corner at 24.4 kHz is a normalised frequency of 8e-5. The transfer-function
    form returns finite-looking coefficients and then produces garbage - during design
    this gave a MAD-sigma of 1e208. The guard is on the filter's **output range**
    rather than on its poles, because the output is what fails observably.
    """
    x = _white(fs, 5.0, seed=0)
    b, a = butter(4, [1.0 / (fs / 2), 100.0 / (fs / 2)], btype="bandpass")
    y = lfilter(b, a, x)

    assert not np.isfinite(y).all() or float(np.max(np.abs(y))) > 1e3 * float(
        np.max(np.abs(x))
    ), "the ill-conditioned design stopped misbehaving - the guard needs a new case"

    with pytest.raises(FilterSanityError, match="Decimate first"):
        assert_filter_sane(x, y, "1-100 Hz via ba")


def test_the_filter_guard_passes_a_sound_design(fs: float) -> None:
    """It must not fire on a correct filter, or it would be noise."""
    x = _white(fs, 5.0, seed=0)
    for band in BANDS:
        envelope = band_envelope_for(x, fs, band)
        assert np.isfinite(envelope[np.isfinite(envelope)]).all(), band


def test_the_guard_catches_amplification_and_non_finite_output() -> None:
    """The two ways a filter output can be impossible."""
    x = np.array([1.0, -1.0, 2.0, -2.0])

    with pytest.raises(FilterSanityError, match="cannot amplify"):
        assert_filter_sane(x, x * 1e6, "amplifying")

    with pytest.raises(FilterSanityError, match="non-finite"):
        assert_filter_sane(x, np.array([1.0, np.inf, 1.0, 1.0]), "diverging")

    assert_filter_sane(x, x * 0.5, "sound")


# ---------------------------------------------------------------------------
# NaN handling and segments
# ---------------------------------------------------------------------------


def test_a_gap_in_a_slow_band_gives_two_segments_not_one_interpolated_trace() -> None:
    """A 1 s gap is half a cycle at 0.5 Hz, so interpolating invents the measurement.

    The task's required test. The signal is a 0.05 Hz wave with a 1 s hole in the
    middle: two assessable segments, and the frames spanning the hole are ``nan``
    rather than carrying an envelope computed across it.
    """
    fs = 2000.0
    dur_s = 400.0
    t = np.arange(int(dur_s * fs)) / fs
    x = 100.0 * np.sin(2.0 * np.pi * 0.05 * t)
    x[int(200.0 * fs) : int(201.0 * fs)] = np.nan

    segments = valid_segments(x, fs, BANDS["0-2"].lo_hz, BANDS["0-2"].window_s)
    assert len(segments) == 2
    assert all(segment.assessable for segment in segments)

    envelope = band_envelope(x, fs, 0.0, 2.0, BANDS["0-2"].window_s)
    hole = slice(int(200.0 / GRID_S), int(201.0 / GRID_S))
    assert bool(np.isnan(envelope[hole]).all()), "the gap was interpolated across"
    assert bool(np.isfinite(envelope[int(100.0 / GRID_S)]))
    assert bool(np.isfinite(envelope[int(300.0 / GRID_S)]))


def test_a_short_segment_is_unassessable_rather_than_wrong() -> None:
    """Three cycles of the band's lowest frequency, or its window, whichever is longer.

    A segment too short to hold the thing being measured produces ``nan``, not a
    number computed from too little data.
    """
    fs = 2000.0
    spec = BANDS["0.5-3"]
    short = np.full(int(2.0 * fs), 1.0)
    long_enough = np.full(int(60.0 * fs), 1.0)

    assert not valid_segments(short, fs, spec.lo_hz, spec.window_s)[0].assessable
    assert valid_segments(long_enough, fs, spec.lo_hz, spec.window_s)[0].assessable

    by_cycles = MIN_SEGMENT_CYCLES / spec.lo_hz
    assert by_cycles == pytest.approx(6.0)


def test_an_unassessable_segment_produces_no_frames_even_when_it_could(fs: float) -> None:
    """Classifying a segment unassessable has to *do* something, and this is the case.

    Found by mutation: deleting the assessable check left every test passing, because
    a segment short enough to fail the cycles rule is usually also too short to survive
    the settling trim, so it produced no frames either way. The gap between the two
    rules is where it matters - in the 2-50 Hz band the cycles floor is 1.5 s while the
    window plus its settling edges is 0.93 s, so a 1.2 s segment would otherwise yield
    about 27 frames of an envelope of something that has not completed three cycles.
    """
    band = "2-50"
    spec = BANDS[band]
    by_cycles = MIN_SEGMENT_CYCLES / spec.lo_hz
    by_window = spec.window_s * (1.0 + 2.0 * SETTLING_WINDOWS)
    assert by_window < 1.2 < by_cycles, "the gap between the two floors has moved"

    rate = 2000.0
    x = np.full(int(60.0 * rate), np.nan, dtype=np.float64)
    live = slice(int(20.0 * rate), int(21.2 * rate))
    x[live] = np.random.default_rng(0).normal(0.0, 10.0, size=live.stop - live.start)

    segments = valid_segments(x, rate, spec.lo_hz, spec.window_s)
    assert len(segments) == 1
    assert not segments[0].assessable

    envelope = band_envelope(x, rate, spec.lo_hz, spec.hi_hz, spec.window_s)
    assert not np.isfinite(envelope).any(), (
        "a segment too short to hold three cycles must yield no envelope at all"
    )


def test_a_dc_band_uses_its_window_because_it_has_no_lowest_cycle() -> None:
    """``lo_hz = 0`` means there is no lowest frequency to count three cycles of.

    The floor is then the analysis window plus its two settling edges, which is the
    shortest segment that can produce even one assessable frame.
    """
    fs = 2000.0
    spec = BANDS["0-2"]
    floor_s = spec.window_s * (1.0 + 2.0 * SETTLING_WINDOWS)

    just_under = valid_segments(np.ones(int((floor_s - 1.0) * fs)), fs, 0.0, spec.window_s)
    just_over = valid_segments(np.ones(int((floor_s + 1.0) * fs)), fs, 0.0, spec.window_s)

    assert not just_under[0].assessable
    assert just_over[0].assessable


def test_a_fast_band_interpolates_a_short_gap_and_restores_it() -> None:
    """At 300-3000 Hz a 30 ms gap is many cycles, so only its own frames are lost."""
    fs = 24414.0625
    x = _white(fs, 60.0, seed=0)
    x[int(30.0 * fs) : int(30.03 * fs)] = np.nan

    envelope = band_envelope_for(x, fs, "300-3000")
    finite = np.isfinite(envelope)

    assert float(finite.mean()) > 0.95, "a 30 ms gap should not cost the whole trace"
    assert not bool(finite[int(30.0 / GRID_S) + 1])


# ---------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------


def test_the_reference_is_the_median_and_mad_of_the_log_envelope(fs: float) -> None:
    """Hard invariant 5, checked against the formula rather than a reimplementation."""
    values = log_envelope(band_envelope_for(_white(fs, 120.0, seed=0), fs, "100-300"))
    reference = epoch_reference(values, signal="L_T", band="100-300")

    assessable = values[np.isfinite(values)]
    expected_median = float(np.median(assessable))
    expected_scale = MAD_TO_SIGMA * float(np.median(np.abs(assessable - expected_median)))

    assert reference.median == pytest.approx(expected_median)
    assert reference.scale == pytest.approx(expected_scale)
    assert reference.n_frames == assessable.size
    assert reference.to_provenance()["reference_statistic"] == REFERENCE_STATISTIC


def test_there_is_no_percentile_parameter() -> None:
    """A.3: the percentile reference is gone from the code, not kept as an option.

    A second reference rule in the signature would compete with the binding one, and
    the competing rule is the one that was measured to be mis-centred.
    """
    import inspect  # noqa: PLC0415

    parameters = inspect.signature(epoch_reference).parameters
    assert "pct" not in parameters
    assert "percentile" not in parameters
    assert set(parameters) == {"log_env", "signal", "band"}


def test_the_log_form_centres_the_null_where_the_linear_form_did_not(fs: float) -> None:
    """The measurement that made the log binding, reproduced on clean frames.

    Measured 2026-09-19: the linear form (a 10th-percentile reference on the linear
    envelope) puts median ``z`` at 1.00 and p90 at 3.05, so about 10% of clean frames
    exceed ``z = 3`` **per pair** - which the union across 36 pairs turns into 40-55%
    of the file. The log form gives median 0 and p90 1.44.

    Reproduced here on white noise: the log form's median is 0 by construction and its
    p90 lands near 1.44, while the linear form's median sits near 1 and its p90 near 3.
    """
    envelope = band_envelope_for(_white(fs, 600.0, seed=0), fs, "100-300")
    finite = envelope[np.isfinite(envelope)]

    values = log_envelope(envelope)
    reference = epoch_reference(values, signal="L_T", band="100-300")
    logged = zscore(values, reference, signal="L_T", band="100-300")
    logged = logged[np.isfinite(logged)]

    linear_reference = float(np.percentile(finite, 10))
    linear_scale = MAD_TO_SIGMA * float(np.median(np.abs(finite - np.median(finite))))
    linear = (finite - linear_reference) / linear_scale

    assert float(np.median(logged)) == pytest.approx(0.0, abs=0.02)
    assert float(np.percentile(logged, 90)) == pytest.approx(1.44, abs=0.25)
    assert float(np.mean(logged > 3.0)) < 0.01

    assert float(np.median(linear)) == pytest.approx(1.0, abs=0.35)
    assert float(np.percentile(linear, 90)) > 2.0
    assert float(np.mean(linear > 3.0)) > 5.0 * float(np.mean(logged > 3.0))


def test_a_dead_channel_is_gated_not_divided_by_zero(fs: float) -> None:
    """The task's required test. A flat channel gives MAD -> 0 and z -> inf.

    Gated **before** the division, because an infinity reaching the candidate
    generator is a detection on every frame of a channel that recorded nothing.
    """
    flat = np.zeros(int(120.0 * fs), dtype=np.float64)
    values = log_envelope(band_envelope_for(flat, fs, "100-300"))

    assert is_dead(values)
    with pytest.raises(DeadChannelError, match="gate it, do not divide"):
        epoch_reference(values, signal="L_T", band="100-300")

    assert float(np.max(values[np.isfinite(values)])) == pytest.approx(
        math.log(ENVELOPE_FLOOR_UV)
    )


def test_a_reference_cannot_be_built_below_the_scale_floor() -> None:
    """Constructing the pair directly is guarded too, not only the factory."""
    with pytest.raises(DeadChannelError, match="under the floor"):
        Reference(median=0.0, scale=SCALE_FLOOR / 10, n_frames=1000, signal="L_T", band="0-2")


def test_too_few_assessable_frames_is_refused(fs: float) -> None:
    """A median and MAD from under 100 frames carry enough error to move every z."""
    values = log_envelope(band_envelope_for(_white(fs, 120.0, seed=0), fs, "100-300"))
    values[MIN_REFERENCE_FRAMES - 1 :] = np.nan

    with pytest.raises(ValueError, match="assessable frames"):
        epoch_reference(values, signal="L_T", band="100-300")


def test_nan_frames_are_excluded_from_the_reference_not_imputed(fs: float) -> None:
    """Imputing the median would narrow the MAD by the fraction that was missing."""
    values = log_envelope(band_envelope_for(_white(fs, 300.0, seed=0), fs, "100-300"))
    whole = epoch_reference(values, signal="L_T", band="100-300")

    holed = values.copy()
    holed[1000:5000] = np.nan
    punched = epoch_reference(holed, signal="L_T", band="100-300")

    imputed = np.where(np.isnan(holed), whole.median, holed)
    imputed_scale = MAD_TO_SIGMA * float(
        np.median(np.abs(imputed - float(np.median(imputed))))
    )

    assert punched.n_frames < whole.n_frames
    assert punched.scale == pytest.approx(whole.scale, rel=0.05)
    assert imputed_scale < 0.95 * whole.scale, "imputation should visibly narrow it"


def test_the_reference_is_logged_per_signal_and_band(
    fs: float, caplog: pytest.LogCaptureFixture
) -> None:
    """The acceptance asks for the scalars to be logged per (signal, band)."""
    values = log_envelope(band_envelope_for(_white(fs, 120.0, seed=0), fs, "100-300"))
    with caplog.at_level(logging.INFO, logger="gems_blanking_v2.bands.reference"):
        epoch_reference(values, signal="L_T", band="100-300")

    assert any("L_T/100-300: reference median" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# z
# ---------------------------------------------------------------------------


def test_a_reference_is_never_applied_to_another_signal(fs: float) -> None:
    """Hard invariant 3, enforced rather than trusted.

    Borrowing was measured at 1136 true detections against 19,624 false ones, which is
    not a rule to leave to the caller's discipline.
    """
    values = log_envelope(band_envelope_for(_white(fs, 120.0, seed=0), fs, "100-300"))
    reference = epoch_reference(values, signal="L_T", band="100-300")

    with pytest.raises(ValueError, match="invariant 3"):
        zscore(values, reference, signal="R_T", band="100-300")
    with pytest.raises(ValueError, match="invariant 3"):
        zscore(values, reference, signal="L_T", band="10-150")

    assert np.isfinite(zscore(values, reference, signal="L_T", band="100-300")).any()


def test_a_nan_frame_has_no_z(fs: float) -> None:
    """``nan`` stays ``nan``. Zero would read as "measured, and normal"."""
    values = log_envelope(band_envelope_for(_white(fs, 120.0, seed=0), fs, "100-300"))
    reference = epoch_reference(values, signal="L_T", band="100-300")
    result = zscore(values, reference, signal="L_T", band="100-300")

    assert np.array_equal(np.isnan(result), np.isnan(values))
    assert result.size == values.size


def test_an_injected_excursion_shows_up_as_large_z(fs: float) -> None:
    """The whole point: something loud is far from the epoch's own median."""
    x = _white(fs, 120.0, seed=0)
    start = int(60.0 * fs)
    x[start : start + int(0.2 * fs)] *= 30.0

    values = log_envelope(band_envelope_for(x, fs, "100-300"))
    reference = epoch_reference(values, signal="L_T", band="100-300")
    result = zscore(values, reference, signal="L_T", band="100-300")

    during = result[int(60.0 / GRID_S) : int(60.2 / GRID_S)]
    assert float(np.nanmax(during)) > 10.0
    assert float(np.nanmedian(result)) == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# the epoch precondition
# ---------------------------------------------------------------------------


def test_an_unsplit_stim_recovery_recording_is_refused(fs: float) -> None:
    """The contract with 03B, as a checked precondition rather than an assumption."""
    rec = _recording(_white(fs, 30.0, seed=0), fs)

    with pytest.raises(ValueError, match="has not been split"):
        analysis_epoch(rec, condition_epoch="stim_recovery")

    epoch = analysis_epoch(rec, condition_epoch="stim_recovery", source="recovery")
    assert epoch.source == "recovery"


def test_a_baseline_recording_needs_no_split(fs: float) -> None:
    """Which is the other half of making it a contract: baselines pass straight through."""
    rec = _recording(_white(fs, 30.0, seed=0), fs)
    epoch = analysis_epoch(rec, condition_epoch="baseline")

    assert epoch.source == "baseline"
    assert epoch.t0_offset_s == 0.0
    assert epoch.duration_s == pytest.approx(30.0, abs=0.01)


def test_an_unclassified_recording_is_refused(fs: float) -> None:
    """``unknown`` is not a licence to proceed - it might be either kind."""
    rec = _recording(_white(fs, 30.0, seed=0), fs)

    with pytest.raises(ValueError, match="neither baseline nor stim_recovery"):
        analysis_epoch(rec, condition_epoch="unknown")


def test_a_read_only_epoch_view_can_be_analysed(fs: float) -> None:
    """Invariant 17: epochs arrive read-only, and nothing here writes to them."""
    data = _white(fs, 120.0, seed=0).reshape(-1, 1)
    view = data[1000:]
    view.flags.writeable = False
    rec = _recording(view.ravel(), fs)

    envelope = band_envelope_for(rec.data[:, 0], fs, "100-300")
    assert np.isfinite(envelope[np.isfinite(envelope)]).all()
    assert not rec.data.flags.writeable


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def test_the_envelope_streams_rather_than_materialising_the_file(fs: float) -> None:
    """The memory claim, as the property that actually matters.

    "9 channels x 6 bands x 37 M samples is ~16 GB if materialised" - so nothing may
    hold more than one band of one channel at a time, and what is kept is the
    frame-rate envelope, 100 values a second rather than 24,414.

    Measured with ``tracemalloc``, which **undercounts**: numpy allocates large buffers
    outside Python's allocator, so this is a floor on the real figure, not an RSS
    ceiling. The portable claim asserted here is the ratio - peak allocation stays a
    small multiple of one channel's input - plus the compression the retained output
    achieves. A real RSS ceiling needs ``psutil``, which is not a dependency; **the
    number to ratify is in the task report**.
    """
    dur_s = 120.0
    x = _white(fs, dur_s, seed=0)
    input_bytes = x.nbytes

    tracemalloc.start()
    try:
        envelopes = {band: band_envelope_for(x, fs, band) for band in BANDS}
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    retained = sum(e.nbytes for e in envelopes.values())
    assert peak < 6 * input_bytes, (
        f"peak {peak / 1e6:.1f} MB against an input of {input_bytes / 1e6:.1f} MB"
    )
    assert retained < 0.05 * input_bytes * len(BANDS), "the retained output is not frame-rate"
    assert all(e.size == int(dur_s / GRID_S) for e in envelopes.values())
