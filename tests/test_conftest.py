"""Every generator in ``conftest`` must reproduce its own spec.

A generator bug produces confident wrong numbers in every downstream test, which is
exactly how the QRS double-peak was introduced during design. Each numeric claim in
a generator docstring has a test here that would fail if the claim were wrong.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.constants import FS_NOMINAL_HZ, MAD_TO_SIGMA

from conftest import (
    ADDITIVE_KINDS,
    FLANK_FRACTION,
    SPIKE_WIDTH_MS,
    WEAK_AMP_RANGE_UV,
    ArtifactKind,
    inject_artifact,
    make_beats,
    make_ecg,
    make_eng,
    make_multichannel,
    make_qrs,
    make_slow,
    robust_sigma,
)


def count_local_maxima(x: npt.NDArray[np.float64], above: float) -> int:
    """Return the number of strict interior local maxima of ``x`` exceeding ``above``."""
    interior = x[1:-1]
    is_max = (interior > x[:-2]) & (interior > x[2:]) & (interior > above)
    return int(np.count_nonzero(is_max))


# ---------------------------------------------------------------------------
# make_beats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rr_s,sd_rr_s", [(0.150, 0.005), (0.180, 0.012), (0.120, 0.003)])
def test_beats_reproduce_requested_mean_and_sd(fs: float, rr_s: float, sd_rr_s: float) -> None:
    beats = make_beats(fs, 600.0, rr_s=rr_s, sd_rr_s=sd_rr_s, seed=3)
    rr = np.diff(beats)
    assert rr.size > 1000
    assert float(np.mean(rr)) == pytest.approx(rr_s, rel=0.02)
    assert float(np.std(rr, ddof=1)) == pytest.approx(sd_rr_s, rel=0.02)


def test_beats_are_strictly_increasing_and_inside_the_record(fs: float) -> None:
    dur_s = 30.0
    beats = make_beats(fs, dur_s, seed=1)
    assert np.all(np.diff(beats) > 0.0)
    assert beats[0] > 0.0
    assert beats[-1] < dur_s


def test_beats_are_on_the_sample_grid(fs: float) -> None:
    beats = make_beats(fs, 20.0, seed=7)
    residual = np.abs(beats * fs - np.round(beats * fs))
    assert float(np.max(residual)) < 1e-6


def test_beats_are_reproducible_and_seed_dependent(fs: float) -> None:
    a = make_beats(fs, 20.0, seed=2)
    b = make_beats(fs, 20.0, seed=2)
    c = make_beats(fs, 20.0, seed=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[: min(a.size, c.size)], c[: min(a.size, c.size)])


def test_beats_reject_an_sd_that_would_make_intervals_negative(fs: float) -> None:
    with pytest.raises(ValueError, match="must exceed"):
        make_beats(fs, 20.0, rr_s=0.15, sd_rr_s=0.05)


def test_beats_reject_a_record_too_short_to_state_an_sd(fs: float) -> None:
    with pytest.raises(ValueError, match="too short"):
        make_beats(fs, 0.4, rr_s=0.15)


# ---------------------------------------------------------------------------
# make_qrs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width_ms", [8.0, 10.0, 12.0])
def test_qrs_has_exactly_one_local_maximum_above_a_fifth_of_peak(
    fs: float, width_ms: float
) -> None:
    k = make_qrs(fs, width_ms=width_ms)
    assert count_local_maxima(k, above=0.2 * float(np.max(k))) == 1


def test_qrs_flanks_are_negative_and_within_the_double_count_limit(fs: float) -> None:
    k = make_qrs(fs)
    assert float(np.max(k)) == pytest.approx(1.0)
    assert float(np.min(k)) == pytest.approx(-FLANK_FRACTION)
    assert FLANK_FRACTION <= 0.2


def test_qrs_is_odd_length_symmetric_and_peaks_at_its_centre(fs: float) -> None:
    k = make_qrs(fs)
    assert k.size % 2 == 1
    assert int(np.argmax(k)) == k.size // 2
    assert np.allclose(k, k[::-1])


@pytest.mark.parametrize("width_ms", [8.0, 10.0, 12.0])
def test_qrs_central_lobe_matches_the_requested_width(fs: float, width_ms: float) -> None:
    k = make_qrs(fs, width_ms=width_ms)
    w = int(round(fs * width_ms / 1000.0)) | 1
    # The central Hann lobe is zero at both ends, so it contributes w-2 positive samples.
    assert int(np.count_nonzero(k > 0.0)) == w - 2
    assert (1000.0 * w / fs) == pytest.approx(width_ms, abs=1000.0 / fs)


# ---------------------------------------------------------------------------
# make_ecg
# ---------------------------------------------------------------------------


def test_ecg_places_one_peak_per_beat_at_the_beat_time(fs: float) -> None:
    sig, beats, _ = make_ecg(fs, 5.0, noise_uv=0.0, seed=4)
    for t_s in beats:
        i = int(round(t_s * fs))
        assert sig[i] == pytest.approx(60.0, rel=1e-9)


def test_ecg_beat_train_does_not_depend_on_the_noise_level(fs: float) -> None:
    clean, beats_a, weak_a = make_ecg(fs, 5.0, weak_frac=0.2, noise_uv=0.0, seed=5)
    noisy, beats_b, weak_b = make_ecg(fs, 5.0, weak_frac=0.2, noise_uv=5.0, seed=5)
    assert np.array_equal(beats_a, beats_b)
    assert np.array_equal(weak_a, weak_b)
    assert float(np.std(noisy - clean, ddof=1)) == pytest.approx(5.0, rel=0.05)


def test_ecg_weak_beats_have_the_documented_amplitude(fs: float) -> None:
    clean, beats, weak = make_ecg(fs, 20.0, weak_frac=0.15, noise_uv=0.0, seed=6)
    assert weak.size == round(0.15 * beats.size)
    peaks = np.array([clean[int(round(beats[i] * fs))] for i in weak])
    assert float(np.min(peaks)) >= WEAK_AMP_RANGE_UV[0]
    assert float(np.max(peaks)) <= WEAK_AMP_RANGE_UV[1]


def test_weak_beats_straddle_the_mad_threshold(fs: float) -> None:
    """Weak beats fall below 3 x 1.4826 x MAD at the default noise; normal beats do not."""
    noisy, beats, weak = make_ecg(fs, 20.0, weak_frac=0.15, seed=6)
    clean, _, _ = make_ecg(fs, 20.0, weak_frac=0.15, noise_uv=0.0, seed=6)
    threshold = 3.0 * robust_sigma(noisy)

    assert WEAK_AMP_RANGE_UV[1] < threshold < 60.0
    weak_peaks = np.array([clean[int(round(beats[i] * fs))] for i in weak])
    strong = np.setdiff1d(np.arange(beats.size), weak)
    strong_peaks = np.array([clean[int(round(beats[i] * fs))] for i in strong])
    assert np.all(weak_peaks < threshold)
    assert np.all(strong_peaks > threshold)


def test_ecg_with_no_weak_fraction_has_no_weak_beats(fs: float) -> None:
    _, _, weak = make_ecg(fs, 5.0, seed=8)
    assert weak.size == 0


# ---------------------------------------------------------------------------
# make_eng
# ---------------------------------------------------------------------------


def test_eng_rate_matches_the_request_within_poisson_error(fs: float) -> None:
    dur_s, rate_hz = 60.0, 20.0
    _, times = make_eng(fs, dur_s, rate_hz=rate_hz, seed=9)
    expected = rate_hz * dur_s
    assert abs(times.size - expected) < 3.0 * math.sqrt(expected)


def test_eng_inter_arrivals_are_exponential(fs: float) -> None:
    """Mean and SD of an exponential are equal; that is the distribution's signature."""
    _, times = make_eng(fs, 300.0, rate_hz=20.0, seed=10)
    gaps = np.diff(times)
    assert float(np.mean(gaps)) == pytest.approx(1.0 / 20.0, rel=0.05)
    assert float(np.std(gaps, ddof=1)) == pytest.approx(float(np.mean(gaps)), rel=0.10)


def test_eng_spike_waveform_is_biphasic_zero_area_and_the_right_width(fs: float) -> None:
    sig, times = make_eng(fs, 2.0, rate_hz=2.0, spike_uv=80.0, noise_uv=0.0, seed=11)
    width_samples = int(round(fs * SPIKE_WIDTH_MS / 1000.0))
    i = int(round(times[0] * fs))
    lo, hi = i - width_samples, i + 2 * width_samples
    seg = sig[lo:hi]
    assert float(np.max(seg)) == pytest.approx(80.0, rel=1e-9)
    assert float(np.min(seg)) < 0.0
    assert float(np.sum(seg)) == pytest.approx(0.0, abs=1e-9 * 80.0 * width_samples)


def test_eng_noise_level_is_the_requested_sigma(fs: float) -> None:
    sig, _ = make_eng(fs, 10.0, rate_hz=0.0, noise_uv=6.0, seed=12)
    assert robust_sigma(sig) == pytest.approx(6.0, rel=0.02)


# ---------------------------------------------------------------------------
# make_slow
# ---------------------------------------------------------------------------


def test_slow_has_the_requested_amplitude_and_frequency() -> None:
    fs_low = 200.0  # a 0.05 Hz wave needs duration, not bandwidth
    dur_s = 400.0
    sig, _ = make_slow(fs_low, dur_s, freq_hz=0.05, amp_uv=200.0, seed=13)
    assert float(np.max(np.abs(sig))) == pytest.approx(200.0, rel=1e-3)

    spectrum = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / fs_low)
    assert float(freqs[int(np.argmax(spectrum))]) == pytest.approx(0.05, abs=1.0 / dur_s)


def test_slow_peak_times_land_on_the_maxima() -> None:
    fs_low = 200.0
    sig, peaks = make_slow(fs_low, 400.0, freq_hz=0.05, amp_uv=200.0, seed=14)
    assert peaks.size == 20  # 400 s at 0.05 Hz, up to a phase-dependent end effect
    values = np.array([sig[int(round(p * fs_low))] for p in peaks])
    assert np.all(values > 199.9)


def test_slow_peaks_are_one_period_apart() -> None:
    _, peaks = make_slow(200.0, 400.0, freq_hz=0.05, seed=15)
    assert np.allclose(np.diff(peaks), 20.0)


# ---------------------------------------------------------------------------
# inject_artifact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ADDITIVE_KINDS)
@pytest.mark.parametrize("amp_ratio", [3.0, 10.0, 50.0])
def test_injected_amplitude_ratio_matches_the_request(
    fs: float, kind: ArtifactKind, amp_ratio: float
) -> None:
    host, _ = make_eng(fs, 4.0, seed=16)
    sigma_before = robust_sigma(host)
    out, span = inject_artifact(host, fs, 1.0, 0.30, kind, amp_ratio, seed=17)

    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))
    measured = float(np.max(np.abs(out[i0:i1] - host[i0:i1]))) / sigma_before
    assert measured == pytest.approx(amp_ratio, rel=0.10)
    assert span.sigma_uv == pytest.approx(sigma_before, rel=1e-12)
    assert span.peak_uv == pytest.approx(amp_ratio * sigma_before, rel=1e-12)


def test_additive_kinds_are_marked_additive_and_carry_no_rail(fs: float) -> None:
    host, _ = make_eng(fs, 4.0, seed=16)
    for kind in ADDITIVE_KINDS:
        _, span = inject_artifact(host, fs, 1.0, 0.30, kind, 10.0, seed=17)
        assert span.additive, kind
        assert math.isnan(span.rail_uv), kind


@pytest.mark.parametrize("kind", ADDITIVE_KINDS)
def test_injection_leaves_everything_outside_the_span_untouched(
    fs: float, kind: ArtifactKind
) -> None:
    host, _ = make_eng(fs, 4.0, seed=18)
    out, span = inject_artifact(host, fs, 1.0, 0.30, kind, 20.0, seed=19)
    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))
    assert np.array_equal(out[:i0], host[:i0])
    assert np.array_equal(out[i1:], host[i1:])
    assert not np.shares_memory(out, host)


def test_injection_does_not_mutate_its_input(fs: float) -> None:
    host, _ = make_eng(fs, 2.0, seed=20)
    before = host.copy()
    inject_artifact(host, fs, 0.5, 0.20, "excursion", 10.0, seed=21)
    assert np.array_equal(host, before)


def test_injection_preserves_nan_and_never_writes_a_zero_run(fs: float) -> None:
    """Hard invariant 1: masked samples stay NaN, and nothing emits a run of zeros."""
    host, _ = make_eng(fs, 2.0, seed=22)
    host[5000:5100] = np.nan
    out, _ = inject_artifact(host, fs, 0.15, 0.10, "step", 30.0, seed=23)
    assert np.all(np.isnan(out[5000:5100]))
    assert np.count_nonzero(out == 0.0) == 0


def test_step_has_a_flat_top(fs: float) -> None:
    host, _ = make_eng(fs, 2.0, seed=24)
    out, span = inject_artifact(host, fs, 0.5, 0.20, "step", 40.0, seed=25)
    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))
    comp = out[i0:i1] - host[i0:i1]
    mid = comp[comp.size // 4 : 3 * comp.size // 4]
    assert np.all(mid == pytest.approx(span.peak_uv, rel=1e-9))


def test_drift_is_smooth_and_returns_to_baseline(fs: float) -> None:
    host, _ = make_eng(fs, 4.0, seed=26)
    out, span = inject_artifact(host, fs, 1.0, 1.00, "drift", 25.0, seed=27)
    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))
    comp = out[i0:i1] - host[i0:i1]
    assert comp[0] == pytest.approx(0.0, abs=1e-9)
    assert comp[-1] == pytest.approx(0.0, abs=span.peak_uv * 1e-4)
    assert int(np.argmax(comp)) == pytest.approx(comp.size // 2, abs=2)


# --- 'clip': the one non-linear kind -------------------------------------


def test_clip_hard_limits_the_host_at_the_rail(fs: float) -> None:
    host, _ = make_eng(fs, 4.0, seed=50)
    sigma_before = robust_sigma(host)
    out, span = inject_artifact(host, fs, 1.0, 0.50, "clip", 3.0, seed=51)
    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))

    assert span.rail_uv == pytest.approx(3.0 * sigma_before, rel=1e-12)
    assert math.isnan(span.peak_uv)
    assert not span.additive
    assert float(np.max(np.abs(out[i0:i1]))) <= span.rail_uv * (1.0 + 1e-12)
    assert float(np.max(np.abs(host[i0:i1]))) > span.rail_uv  # the host really was clipped


def test_clip_changes_only_the_samples_beyond_the_rail(fs: float) -> None:
    """It is not additive: inside the rail the host passes through untouched."""
    host, _ = make_eng(fs, 4.0, seed=52)
    out, span = inject_artifact(host, fs, 1.0, 0.50, "clip", 3.0, seed=53)
    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))

    inside = np.abs(host[i0:i1]) <= span.rail_uv
    assert np.array_equal(out[i0:i1][inside], host[i0:i1][inside])
    assert np.any(~inside)
    assert np.array_equal(out[:i0], host[:i0])
    assert np.array_equal(out[i1:], host[i1:])


def test_clip_removes_energy_so_an_envelope_understates_the_damage(fs: float) -> None:
    """Why this kind exists: saturation lowers the envelope instead of raising it.

    An additive flat-top raises RMS and any envelope detector sees it. A rail does
    the opposite, which is what task 14 routes around the classifier for.
    """
    host, _ = make_eng(fs, 4.0, seed=54)
    clipped, span = inject_artifact(host, fs, 1.0, 0.50, "clip", 3.0, seed=55)
    stepped, _ = inject_artifact(host, fs, 1.0, 0.50, "step", 3.0, seed=55)
    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))

    def rms(x: npt.NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(x**2)))

    assert rms(clipped[i0:i1]) < rms(host[i0:i1])
    assert rms(stepped[i0:i1]) > rms(host[i0:i1])


def test_clip_above_the_host_peak_is_a_no_op(fs: float) -> None:
    host, _ = make_eng(fs, 2.0, seed=56)
    out, span = inject_artifact(host, fs, 0.5, 0.20, "clip", 1e6, seed=57)
    assert np.array_equal(out, host)
    assert span.kind == "clip"


def test_clip_preserves_nan(fs: float) -> None:
    host, _ = make_eng(fs, 2.0, seed=58)
    host[5000:5100] = np.nan
    out, _ = inject_artifact(host, fs, 0.15, 0.10, "clip", 2.0, seed=59)
    assert np.all(np.isnan(out[5000:5100]))


def energy_fraction_above(x: npt.NDArray[np.float64], fs: float, f_hz: float) -> float:
    """Return the fraction of ``x``'s energy above ``f_hz``, DC removed."""
    p = np.abs(np.fft.rfft(x - float(np.mean(x)))) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / fs)
    return float(p[f > f_hz].sum() / p.sum())


def test_smooth_kinds_put_essentially_no_energy_in_the_eng_band(fs: float) -> None:
    """Measured >99.99% below 300 Hz - smoother than the 98.7-99.7% A.5 measured.

    Recorded as a generator limitation: excursion and drift will not exercise a
    detector's ENG-band response to motion.
    """
    host, _ = make_eng(fs, 4.0, seed=30)
    for kind in ("excursion", "drift"):
        out, span = inject_artifact(host, fs, 1.0, 0.30, kind, 20.0, seed=31)
        i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))
        comp = out[i0:i1] - host[i0:i1]
        assert energy_fraction_above(comp, fs, 300.0) < 1e-4, kind


def test_the_saturating_step_reaches_the_eng_band_but_not_a_third_of_the_way(
    fs: float,
) -> None:
    """A.5 says 32.2% of a saturating step's energy is above 300 Hz; this gives 18%.

    A step's spectrum falls as 1/f, so the fraction above a fixed corner is set by
    the analysis window (18.6% in 10 ms even for an instantaneous edge, 2.4% in
    50 ms). A.5 states no window, so the figure is not reproducible from a step
    alone. The generator keeps a physical rise time instead of being tuned to match.
    """
    host, _ = make_eng(fs, 4.0, seed=32)
    out, span = inject_artifact(host, fs, 1.0, 0.30, "step", 40.0, seed=33)
    edge = int(round(span.start_s * fs))
    half_window = int(round(0.005 * fs))
    comp = (out - host)[edge - half_window : edge + half_window]
    assert 0.15 < energy_fraction_above(comp, fs, 300.0) < 0.22


def test_tribo_crackle_is_broadband(fs: float) -> None:
    """21-25% of its energy above 300 Hz: the only kind that reaches the ENG band."""
    host, _ = make_eng(fs, 4.0, seed=34)
    out, span = inject_artifact(host, fs, 1.0, 0.30, "tribo", 20.0, seed=35)
    i0, i1 = int(round(span.start_s * fs)), int(round(span.stop_s * fs))
    comp = out[i0:i1] - host[i0:i1]
    assert 0.10 < energy_fraction_above(comp, fs, 300.0) < 0.40


def test_injection_rejects_a_span_outside_the_record(fs: float) -> None:
    host, _ = make_eng(fs, 1.0, seed=28)
    with pytest.raises(ValueError, match="does not fit"):
        inject_artifact(host, fs, 0.9, 0.5, "drift", 10.0)


def test_injection_rejects_a_host_with_no_scale(fs: float) -> None:
    flat = np.zeros(int(fs), dtype=np.float64)
    with pytest.raises(ValueError, match="robust sigma"):
        inject_artifact(flat, fs, 0.1, 0.1, "drift", 10.0)


def test_injection_rejects_an_unknown_kind(fs: float) -> None:
    host, _ = make_eng(fs, 1.0, seed=29)
    with pytest.raises(ValueError, match="unknown artifact kind"):
        inject_artifact(host, fs, 0.1, 0.1, "wobble", 10.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# make_multichannel
# ---------------------------------------------------------------------------


def test_multichannel_default_layout_is_the_nine_channel_rig(fs: float) -> None:
    rec, _ = make_multichannel(fs, 4.0, seed=30)
    assert rec.n_channels == 9
    assert [c.name for c in rec.channels] == [
        "LVN1",
        "LVN2",
        "LVN3",
        "RVN1",
        "RVN2",
        "RVN3",
        "ANT1",
        "ANT2",
        "ANT3",
    ]
    assert [c.index for c in rec.channels] == list(range(9))
    assert rec.data.dtype == np.float64
    assert rec.fs == fs
    # fs*dur is not an integer at 24414.0625 Hz, so the record is one sample short.
    assert rec.duration_s == pytest.approx(4.0, abs=1.0 / fs)


def test_multichannel_roles_and_cuff_metadata(fs: float) -> None:
    rec, _ = make_multichannel(fs, 2.0, seed=31)
    nerve = [c for c in rec.channels if c.role == "nerve"]
    stomach = [c for c in rec.channels if c.role == "stomach"]
    assert len(nerve) == 6
    assert len(stomach) == 3
    assert {c.cuff_id for c in nerve} == {"L", "R"}
    assert sorted(c.contact_index or 0 for c in nerve) == [1, 1, 2, 2, 3, 3]
    assert all(c.cuff_id is None and c.contact_index is None for c in stomach)
    assert all(c.config == "independent" for c in rec.channels)


def test_multichannel_common_mode_enters_every_channel_at_its_own_gain(fs: float) -> None:
    rec, truth = make_multichannel(fs, 3.0, seed=32)
    shared = truth["common_mode"]
    assert float(np.max(np.abs(shared))) > 0.0
    for ch in rec.channels:
        own = truth["neural"].get(ch.name)
        if own is None:
            own = truth["slow"][ch.name]
        residual = rec.data[:, ch.index] - own
        assert np.allclose(residual, truth["gains"][ch.name] * shared, atol=1e-9)


def test_multichannel_gains_are_unequal(fs: float) -> None:
    """Equal gains would let a naive 0.5/0.5 tripole cancel the common mode exactly."""
    _, truth = make_multichannel(fs, 2.0, seed=33)
    gains = np.array(list(truth["gains"].values()))
    assert float(np.std(gains)) > 0.01
    assert np.all((gains >= 0.80) & (gains <= 1.25))


def test_multichannel_neural_content_is_independent_per_contact(fs: float) -> None:
    _, truth = make_multichannel(fs, 4.0, seed=34)
    names = list(truth["neural"])
    for a, b in itertools.pairwise(names):
        r = float(np.corrcoef(truth["neural"][a], truth["neural"][b])[0, 1])
        assert abs(r) < 0.05
    assert all(t.size > 0 for t in truth["spike_times_s"].values())


def test_multichannel_artifacts_are_reported_with_their_spans(fs: float) -> None:
    _, truth = make_multichannel(fs, 20.0, seed=35)
    kinds = [s.kind for s in truth["artifact_spans"]]
    assert kinds == ["excursion", "drift", "step"]
    for span in truth["artifact_spans"]:
        assert 0.0 <= span.start_s < span.stop_s <= 20.0
        assert span.peak_uv == pytest.approx(span.amp_ratio * span.sigma_uv, rel=1e-12)


def test_multichannel_without_common_mode_is_artifact_free(fs: float) -> None:
    rec, truth = make_multichannel(fs, 3.0, common_mode=False, seed=36)
    assert not np.any(truth["common_mode"])
    assert truth["artifact_spans"] == []
    for ch in rec.channels:
        own = truth["neural"].get(ch.name, truth["slow"].get(ch.name))
        assert own is not None
        assert np.allclose(rec.data[:, ch.index], own)


def test_multichannel_is_finite_and_has_no_long_zero_runs(fs: float) -> None:
    """Hard invariant 1: an exact-zero run is indistinguishable from signal downstream."""
    rec, _ = make_multichannel(fs, 3.0, seed=37)
    assert np.all(np.isfinite(rec.data))
    is_zero = rec.data == 0.0
    assert not np.any(is_zero[:-2] & is_zero[1:-1] & is_zero[2:])


def test_multichannel_is_reproducible(fs: float) -> None:
    a, _ = make_multichannel(fs, 2.0, seed=38)
    b, _ = make_multichannel(fs, 2.0, seed=38)
    c, _ = make_multichannel(fs, 2.0, seed=39)
    assert np.array_equal(a.data, b.data)
    assert not np.array_equal(a.data, c.data)


def test_multichannel_channel_lookup(fs: float) -> None:
    rec, _ = make_multichannel(fs, 2.0, seed=40)
    assert rec.channel("RVN2").index == 4
    with pytest.raises(KeyError, match="no channel named"):
        rec.channel("nope")


def test_multichannel_single_cuff(fs: float) -> None:
    rec, _ = make_multichannel(fs, 2.0, n_cuff=1, n_stomach=1, seed=41)
    assert rec.n_channels == 4
    assert {c.cuff_id for c in rec.channels if c.role == "nerve"} == {"L"}


# ---------------------------------------------------------------------------
# helpers the generators rely on
# ---------------------------------------------------------------------------


def test_robust_sigma_recovers_a_known_gaussian_scale() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 7.0, size=200_000)
    assert robust_sigma(x) == pytest.approx(7.0, rel=0.01)


def test_robust_sigma_ignores_nan_and_resists_outliers() -> None:
    rng = np.random.default_rng(43)
    x = rng.normal(0.0, 7.0, size=200_000)
    x[:1000] = np.nan
    x[1000:2000] = 1e6
    assert robust_sigma(x) == pytest.approx(7.0, rel=0.02)


def test_mad_to_sigma_is_the_gaussian_consistency_constant() -> None:
    assert pytest.approx(1.4826, abs=5e-5) == MAD_TO_SIGMA


def test_the_fs_fixture_is_the_nominal_tdt_rate(fs: float) -> None:
    assert fs == FS_NOMINAL_HZ
