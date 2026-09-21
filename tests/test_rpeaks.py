"""Tests for :mod:`gems_blanking_v2.physio.rpeaks`.

Every numeric claim in the module's docstrings has a test here that would fail if it
were wrong.

The plausibility runaway is now reproducible in CI via ``make_ecg(t_wave=True)`` -
see :func:`test_the_local_median_plausibility_form_runs_away`. One claim still does
not reproduce on the synthetic and says so in its own test:
:func:`test_the_pass2_tolerance_only_matters_at_realistic_rr_spread`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.constants import BANDS, FS_NOMINAL_HZ, HR_BAND
from gems_blanking_v2.physio import rpeaks as rp
from gems_blanking_v2.physio.rpeaks import (
    DETECT_BAND_HZ,
    GLOBAL_RR_FRACTION,
    INHERITED_R_MIN_S,
    PASS1_PROMINENCE_SIGMA,
    PROVISIONAL_MAX_IMPLAUSIBLE_FRAC,
    PROVISIONAL_MAX_RESCUE_RATE,
    R_MIN_S,
    BeatTrain,
    detect_rpeaks,
    rank_hr_channels,
)
from gems_blanking_v2.types import ChannelInfo, Recording
from scipy.signal import find_peaks

from conftest import inject_artifact, make_beats, make_ecg, make_slow, robust_sigma

F64 = npt.NDArray[np.float64]

SEEDS = (11, 7, 3, 29, 101, 5, 17, 44)
"""The seed set the pooled measurements run over. Fixed, per the testing rules."""

MATCH_TOLERANCE_S = 0.030
"""How close a detected beat must be to a true one to count as that beat.

One fifth of the 150 ms synthetic RR, so it cannot accidentally match a neighbour.
"""


# ---------------------------------------------------------------------------
# HRV statistics, so the tests can check what the module's claims are about.
# The `hrv` consumer will own these properly; here they exist only to make the
# never-fabricate claim falsifiable.
# ---------------------------------------------------------------------------


def _rmssd(rr_s: F64) -> float:
    """Root mean square of successive RR differences, seconds."""
    return float(np.sqrt(np.mean(np.diff(rr_s) ** 2)))


def _sdnn(rr_s: F64) -> float:
    """Return the standard deviation of the RR intervals, seconds."""
    return float(np.std(rr_s, ddof=1))


def _sd1(rr_s: F64) -> float:
    """Poincare SD1, seconds. Identical to ``RMSSD / sqrt(2)`` by definition."""
    return _rmssd(rr_s) / float(np.sqrt(2.0))


def _rmssd_excluding_gaps(train: BeatTrain) -> float:
    """RMSSD over **adjacent** usable intervals only - the downstream contract."""
    rr = train.rr_s()
    usable = train.usable_rr_mask()
    adjacent = usable[:-1] & usable[1:]
    return float(np.sqrt(np.mean(np.diff(rr)[adjacent] ** 2)))


def _midpoint_insert(beats_s: F64) -> F64:
    """Fill every interval that looks like ``m`` beats long with ``m-1`` midpoints.

    The handling this module refuses to use, implemented here so its bias can be
    measured rather than asserted.
    """
    rr = np.diff(beats_s)
    local = float(np.median(rr))
    filled = list(beats_s)
    for j, interval in enumerate(rr.tolist()):
        m = int(round(interval / local))
        filled.extend(beats_s[j] + q * interval / m for q in range(1, m))
    return np.asarray(np.sort(np.array(filled)), dtype=np.float64)


def _matched(found_s: F64, truth_s: F64, tol_s: float = MATCH_TOLERANCE_S) -> F64:
    """Return the absolute error of every truth beat that has a match within ``tol``."""
    if found_s.size == 0:
        return np.empty(0, dtype=np.float64)
    errors = [float(np.min(np.abs(found_s - b))) for b in truth_s.tolist()]
    return np.asarray([e for e in errors if e <= tol_s], dtype=np.float64)


# ---------------------------------------------------------------------------
# pass 1 / pass 2
# ---------------------------------------------------------------------------


def test_pass1_finds_every_normal_beat_and_invents_none(fs: float) -> None:
    """Zero false beats over 3160 true ones, which is what picks the default band.

    The count is exact, not approximate: 8 seeds x 395 beats, all found, nothing
    added. See :data:`~gems_blanking_v2.physio.rpeaks.DETECT_BAND_HZ`.
    """
    total_true = 0
    total_found = 0
    total_false = 0
    for seed in SEEDS:
        sig, beats, weak = make_ecg(fs, 60.0, weak_frac=0.09, seed=seed)
        train = detect_rpeaks(sig, fs)
        strong = np.delete(beats, weak)
        total_true += strong.size
        total_found += _matched(train.t_s, strong).size
        total_false += train.n_beats - _matched(train.t_s, beats).size

    assert total_found == total_true, f"lost {total_true - total_found} normal beats"
    assert total_false == 0, f"{total_false} of {total_true} emitted beats are not beats"


def test_pass1_misses_attenuated_beats_and_pass2_recovers_them(fs: float) -> None:
    """The acceptance criterion: >=90% of pass-1 misses recovered, under 1 ms out.

    Pooled over 8 seeds because a single 60 s file only loses 3-6 beats, and a
    recovery *rate* from n=4 is not a rate. Measured: 29 pass-1 misses, 28 recovered
    (96.6%), mean placement error 0.121 ms and worst 0.369 ms - against the 0.24 ms
    mean the spec quotes for re-detection.
    """
    missed = 0
    recovered: list[float] = []
    for seed in SEEDS:
        sig, beats, weak = make_ecg(fs, 60.0, weak_frac=0.09, seed=seed)
        train = detect_rpeaks(sig, fs)
        detected = train.t_s[train.tag == "detected"]
        rescued = train.t_s[train.tag == "rescued"]
        missed += weak.size - _matched(detected, beats[weak]).size
        recovered.extend(_matched(rescued, beats[weak]).tolist())

    assert missed >= 20, f"only {missed} pass-1 misses pooled - the rate means little"
    rate = len(recovered) / missed
    assert rate >= 0.90, f"pass 2 recovered {rate:.1%} of {missed} misses"

    errors_ms = np.asarray(recovered) * 1e3
    assert float(errors_ms.max()) < 1.0, f"worst fiducial error {errors_ms.max():.3f} ms"
    assert float(errors_ms.mean()) < 0.5, f"mean fiducial error {errors_ms.mean():.3f} ms"


def test_a_rescued_beat_is_tagged_and_a_detected_one_is_not(fs: float) -> None:
    """Provenance per beat: the two tags are the only two, and both occur."""
    sig, _beats, _weak = make_ecg(fs, 60.0, weak_frac=0.09, seed=11)
    train = detect_rpeaks(sig, fs)

    assert set(train.tag.tolist()) <= {"detected", "rescued"}
    assert (train.tag == "rescued").sum() > 0, "no rescue happened, so nothing is proved"
    assert train.rescue_rate == pytest.approx(
        float((train.tag == "rescued").sum()) / train.n_beats
    )


def test_an_empty_rescue_window_yields_a_gap_and_no_beat(fs: float) -> None:
    """A window with nothing in it must produce a gap record, never a beat.

    Constructed by attenuating 9% of beats to 1-2 nV, which is 0.0008 x the band
    sigma - the beat is genuinely absent, so the correct answer is "missing", not a
    guess. Measured: 35 of 36 windows record a gap and stay empty.

    The 36th is the honest number to know: **one window in 36 yields a peak that
    passes the 1.2 sigma prominence, the width prior and the template argmax while
    containing no beat.** That is a false positive at ~2.8% per empty window, not a
    fabricated sample - the rescue only ever returns peaks that exist in the trace -
    and it is the price of the relaxed pass-2 threshold. Report it with the rescue
    rate rather than tightening the threshold to hide it.
    """
    sig, beats, weak = make_ecg(
        fs, 60.0, weak_frac=0.09, seed=11, weak_amp_uv=(0.001, 0.002)
    )
    train = detect_rpeaks(sig, fs)
    strong = np.delete(beats, weak)

    spurious = _matched(train.t_s, beats[weak]).size
    assert spurious <= 1, f"{spurious} of {weak.size} empty windows produced a beat"
    assert train.n_beats == strong.size + spurious
    assert train.n_missing >= int(0.9 * weak.size), (
        f"only {train.n_missing} of {weak.size} absent beats were recorded as missing"
    )
    for t0, t1, m in train.gaps:
        assert t1 > t0
        assert m >= 1


def test_no_gap_is_recorded_when_nothing_is_missing(fs: float) -> None:
    """A clean file must report no gaps, or HRV would discard usable intervals."""
    sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=11)
    train = detect_rpeaks(sig, fs)

    assert train.gaps == []
    assert train.n_missing == 0
    assert train.usable_rr_mask().all()
    assert train.n_beats == beats.size


# ---------------------------------------------------------------------------
# never fabricate a beat
# ---------------------------------------------------------------------------


def test_midpoint_insertion_biases_hrv_downward_by_about_six_percent(fs: float) -> None:
    """Show the reason insertion is banned, reproducing the spec's table.

    Measured over 30 trains with 9% of beats deleted: RMSSD **-6.88%** against the
    spec's -6.3%, and SD1 identically so since ``SD1 = RMSSD / sqrt(2)``. The bias is
    downward and systematic - the same direction as a vagal-tone effect - because a
    midpoint forces the two flanking intervals equal and their successive difference
    to exactly zero.

    This test is the guard: if anyone reintroduces insertion, the recovered HRV will
    carry this bias, and the test fails if the bias stops reproducing.
    """
    biases: list[float] = []
    for seed in range(30):
        _sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=seed)
        rng = np.random.default_rng(1000 + seed)
        drop = rng.choice(
            np.arange(1, beats.size - 1), size=int(round(0.09 * beats.size)), replace=False
        )
        kept = np.delete(beats, drop)
        truth = _rmssd(np.diff(beats))
        biases.append(_rmssd(np.diff(_midpoint_insert(kept))) / truth - 1.0)

    bias = float(np.mean(biases))
    assert -0.09 < bias < -0.04, f"midpoint bias measured {bias:+.2%}, expected about -6%"
    assert bias < -0.02, "the bias must be outside the 2% band the rescue achieves"


def test_leaving_the_gap_in_inflates_hrv_by_hundreds_of_percent(fs: float) -> None:
    """Why :meth:`BeatTrain.usable_rr_mask` exists at all.

    Measured over 30 trains at 9% missing: RMSSD **+902%** and SDNN **+898%**,
    against the spec's +771% and +766%. The spec's exact magnitude depends on which
    beats are dropped, so the assertion is on the order of magnitude, which is the
    part that matters: differencing across a gap is not a slightly worse measurement,
    it is a different quantity.
    """
    rmssd_bias: list[float] = []
    sdnn_bias: list[float] = []
    for seed in range(30):
        _sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=seed)
        rng = np.random.default_rng(1000 + seed)
        drop = rng.choice(
            np.arange(1, beats.size - 1), size=int(round(0.09 * beats.size)), replace=False
        )
        kept = np.delete(beats, drop)
        rmssd_bias.append(_rmssd(np.diff(kept)) / _rmssd(np.diff(beats)) - 1.0)
        sdnn_bias.append(_sdnn(np.diff(kept)) / _sdnn(np.diff(beats)) - 1.0)

    assert float(np.mean(rmssd_bias)) > 5.0, "the +771% inflation did not reproduce"
    assert float(np.mean(sdnn_bias)) > 5.0, "the +766% inflation did not reproduce"


def test_hrv_from_the_rescued_train_is_within_two_percent_of_truth(fs: float) -> None:
    """The acceptance criterion for the rescue: RMSSD and SD1 both within 2%.

    Measured pooled over 8 seeds: **+0.22%**. The comparison excludes intervals
    touching an unrecovered gap, which is the documented downstream contract, and
    differences only adjacent usable intervals.
    """
    errors: list[float] = []
    sd1_errors: list[float] = []
    for seed in SEEDS:
        sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.09, seed=seed)
        train = detect_rpeaks(sig, fs)
        truth_rr = np.diff(beats)
        errors.append(_rmssd_excluding_gaps(train) / _rmssd(truth_rr) - 1.0)
        sd1_errors.append(
            (_rmssd_excluding_gaps(train) / np.sqrt(2.0)) / _sd1(truth_rr) - 1.0
        )

    assert abs(float(np.mean(errors))) < 0.02, f"RMSSD off by {np.mean(errors):+.2%}"
    assert abs(float(np.mean(sd1_errors))) < 0.02, f"SD1 off by {np.mean(sd1_errors):+.2%}"
    assert max(abs(e) for e in errors) < 0.02, "a single seed exceeded 2%"


def test_every_emitted_beat_sits_on_a_local_maximum_of_the_trace(fs: float) -> None:
    """Invariant 8 at the level this module can be held to.

    No beat time is computed, interpolated or averaged into existence: each one is
    the index of a peak the band-limited trace actually has. If a rescue ever
    synthesised a time, it would land between samples and fail this.
    """
    sig, _beats, _weak = make_ecg(fs, 60.0, weak_frac=0.09, seed=11)
    train = detect_rpeaks(sig, fs)
    y, fs_d = rp._prepare(sig, fs, DETECT_BAND_HZ)

    idx = np.round(train.t_s * fs_d).astype(np.int64)
    assert np.allclose(train.t_s, idx / fs_d), "a beat time is not on the sample grid"
    interior = idx[(idx > 0) & (idx < y.size - 1)]
    assert np.all((y[interior] >= y[interior - 1]) & (y[interior] >= y[interior + 1]))


# ---------------------------------------------------------------------------
# R_MIN and the plausibility rule
# ---------------------------------------------------------------------------


def test_the_inherited_refractory_deletes_real_beats_at_600_bpm(fs: float) -> None:
    """Show why ``R_MIN`` is 60 ms, measured here rather than inherited.

    At 600 bpm (RR 100 ms) the inherited 90 ms refractory is 90% of RR. Measured on
    294 true beats: 60 ms finds all 294, 90 ms finds 289 - it deletes 5 real beats,
    1.7% of them, for no gain. Both thresholds see the same trace and the same peaks;
    the only difference is the refractory.
    """
    sig, beats, _weak = make_ecg(fs, 30.0, weak_frac=0.0, seed=4, rr_s=0.100)
    y, fs_d = rp._prepare(sig, fs, DETECT_BAND_HZ)
    sigma = robust_sigma(y)

    found: dict[float, int] = {}
    for r_min in (R_MIN_S, INHERITED_R_MIN_S):
        peaks, _props = find_peaks(
            y,
            prominence=PASS1_PROMINENCE_SIGMA * sigma,
            distance=max(int(round(r_min * fs_d)), 1),
        )
        found[r_min] = _matched(peaks / fs_d, beats, tol_s=0.005).size

    assert found[R_MIN_S] == beats.size, "60 ms lost a beat at 600 bpm"
    assert found[INHERITED_R_MIN_S] < beats.size, "90 ms cost nothing here - re-measure"
    assert R_MIN_S < GLOBAL_RR_FRACTION * 0.100, (
        "R_MIN must stay under the plausibility threshold, or it is the binding rule"
    )


def test_r_min_stays_clear_of_the_measured_rr_histogram() -> None:
    """The arithmetic behind 60 ms, against animal J's measured 164.7 ms RR.

    60 ms is 0.36 x RR; the inherited 90 ms is 0.55 x RR, which coincides with the
    plausibility fraction the spec originally specified - so the two rules would have
    been doing the same job, and the refractory would have won silently.
    """
    measured_rr_s = 0.1647
    assert R_MIN_S / measured_rr_s == pytest.approx(0.36, abs=0.01)
    assert INHERITED_R_MIN_S / measured_rr_s == pytest.approx(0.55, abs=0.01)
    assert R_MIN_S < 0.145, "the RR histogram has essentially nothing below 145 ms"


def test_the_plausibility_threshold_comes_from_the_whole_file(fs: float) -> None:
    """The threshold is a whole-file scalar, not a function of the accepted sequence.

    Dropping a peak cannot move it: the global RR is computed from the *provisional*
    peaks, once, before any rejection. The check is direct - the reported
    ``global_rr_s`` equals the median of the provisional intervals, and differs from
    the median of the retained ones.
    """
    sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.09, seed=11)
    train = detect_rpeaks(sig, fs)

    assert train.global_rr_s == pytest.approx(float(np.median(np.diff(beats))), abs=0.005)
    y, fs_d = rp._prepare(sig, fs, DETECT_BAND_HZ)
    provisional, _props = find_peaks(
        y,
        prominence=PASS1_PROMINENCE_SIGMA * robust_sigma(y),
        distance=max(int(round(R_MIN_S * fs_d)), 1),
    )
    assert train.global_rr_s == pytest.approx(rp._global_rr(provisional, fs_d))


def test_the_global_rr_ignores_intervals_outside_80_to_500_ms() -> None:
    """A run of impossibly short intervals must not drag the reference down."""
    fs_d = 2000.0
    good = np.arange(0, 20) * int(0.150 * fs_d)
    tail = good[-1] + np.arange(1, 40) * int(0.010 * fs_d)  # 100 Hz, far too fast
    rr = rp._global_rr(np.concatenate([good, tail]).astype(np.int64), fs_d)

    assert rr == pytest.approx(0.150, abs=0.001), "an out-of-range interval was counted"


def test_the_global_plausibility_form_converges_and_never_overshoots(fs: float) -> None:
    """The stability claim, stated as the property that distinguishes it from the local form.

    Run on a ``t_wave=True`` signal at ``k=3``, so half the provisional peaks are
    false and the rule has something to act on - at the binding threshold a clean
    file gives it nothing to drop and the test would be vacuous.

    Measured, against a true RR of 150.4 ms and a (contaminated) reference of
    85.5 ms:

    ========  ======  ============  ==================
    fraction  kept    retained RR   vs the true RR
    ========  ======  ============  ==================
    0.60      765     75.7 ms       0.50x
    0.70      765     75.7 ms       0.50x
    0.75      694     76.7 ms       0.51x
    0.85      472     146.0 ms      0.97x
    0.95      396     **150.4 ms**  **1.00x**
    ========  ======  ============  ==================

    Tightening the fraction walks the retained RR *toward* the truth and it stops
    there: monotone, and it never overshoots. The local form on the same peaks goes
    to 2x the truth and locks - see
    :func:`test_the_local_median_plausibility_form_runs_away`. That, not the absolute
    value, is why the rule is global.
    """
    sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=11, t_wave=True)
    peaks, prominences, fs_d = _provisional_peaks(sig, fs, sigma_multiple=3.0)
    global_rr = rp._global_rr(peaks, fs_d)
    true_rr = float(np.median(np.diff(beats)))

    kept_counts: list[int] = []
    retained: list[float] = []
    for fraction in (0.60, 0.70, 0.75, 0.85, 0.95):
        kept, _dropped = rp._enforce_plausibility(
            peaks, prominences, fs_d, fraction * global_rr
        )
        kept_counts.append(kept.size)
        retained.append(float(np.median(np.diff(kept))) / fs_d)
        assert retained[-1] <= 1.05 * true_rr, (
            f"at fraction {fraction} the retained RR overshot the true RR - "
            "that is the runaway, and the global form must not do it"
        )

    assert kept_counts == sorted(kept_counts, reverse=True), "not monotone in the fraction"
    assert retained == sorted(retained), "the retained RR did not converge monotonically"
    assert retained[-1] == pytest.approx(true_rr, rel=0.01), (
        "at a tight enough fraction the global form should recover the true rate"
    )


def test_a_noise_spike_next_to_a_beat_does_not_delete_the_beat() -> None:
    """The too-close pair resolves by prominence, not by arrival order."""
    fs_d = 2000.0
    peaks = np.array([100, 110, 400], dtype=np.int64)
    prominences = np.array([1.0, 50.0, 50.0], dtype=np.float64)

    kept, dropped = rp._enforce_plausibility(peaks, prominences, fs_d, 0.100)

    assert dropped == 1
    assert kept.tolist() == [110, 400], "the weaker of the pair should have gone"


# ---------------------------------------------------------------------------
# best-channel ranking
# ---------------------------------------------------------------------------


def _three_channel_recording(fs: float, dur_s: float = 40.0) -> tuple[Recording, F64]:
    """Build a clean channel, an amplitude-wandering one, and one with no cardiac.

    The third carries only noise plus six large excursions, so anything it "detects"
    is artifact-driven - the case the ranking has to put last.
    """
    clean, beats, _weak = make_ecg(fs, dur_s, amp_uv=60.0, noise_uv=5.0, seed=11)
    noisy, _b, _w = make_ecg(fs, dur_s, amp_uv=60.0, noise_uv=25.0, seed=12)
    envelope, _peaks = make_slow(fs, dur_s, freq_hz=0.3, amp_uv=1.0, seed=13)
    noisy = noisy * (1.0 + 0.8 * envelope / float(np.max(np.abs(envelope))))

    no_cardiac = np.random.default_rng(14).normal(0.0, 20.0, size=clean.size)
    for start_s in (3.0, 9.0, 15.0, 21.0, 27.0, 33.0):
        no_cardiac, _truth = inject_artifact(
            no_cardiac, fs, start_s, 0.05, "excursion", 20.0, seed=int(start_s)
        )

    channels = [
        ChannelInfo(0, "LVN1", "nerve", "L", 1, 1, "independent"),
        ChannelInfo(1, "LVN2", "nerve", "L", 2, 1, "independent"),
        ChannelInfo(2, "ANT1", "stomach", None, None, None, "independent"),
    ]
    rec = Recording(
        fs=fs,
        data=np.column_stack([clean, noisy, no_cardiac]),
        channels=channels,
        animal="J",
        session="s1",
        path=Path("synthetic.mat"),
    )
    return rec, beats


def _trains(rec: Recording) -> dict[str, BeatTrain]:
    """Detect on every channel of ``rec``."""
    return {c.name: detect_rpeaks(rec.data[:, c.index], rec.fs) for c in rec.channels}


def test_rank_hr_channels_reproduces_the_ordering(fs: float) -> None:
    """Clean > amplitude-wandering > no-cardiac, by template SNR.

    Measured: **1516 / 197 / 9.7**, against the spec's worked 234 / 27.6 / 4.7. The
    *ordering* reproduces and the separations are the same order (7.7x and 20x here,
    8.5x and 5.9x there); the absolute values do not, and cannot, because they are
    set by the synthetic's noise floor rather than by anything about the method.
    Assert the ordering and the separation, never the absolute numbers.
    """
    rec, beats = _three_channel_recording(fs)
    best, table = rank_hr_channels(rec, _trains(rec))

    assert best == "LVN1"
    assert table["channel"].tolist() == ["LVN1", "LVN2", "ANT1"]
    snr = dict(zip(table["channel"], table["snr"], strict=True))
    assert snr["LVN1"] > snr["LVN2"] > snr["ANT1"]
    assert snr["LVN1"] / snr["LVN2"] > 3.0
    assert snr["LVN2"] / snr["ANT1"] > 3.0
    assert int(table.loc[table["channel"] == "LVN1", "n_beats"].iloc[0]) == beats.size


def test_a_channel_with_no_cardiac_content_ranks_last(fs: float) -> None:
    """The self-policing claim: artifact-driven beats average to a flat template.

    ``ANT1`` has no time-locked content at all, so its template is near-flat while
    its across-beat spread is large. It ranks last on SNR *and* is vetoed on
    ``rescue_rate`` (0.58), so it could not be chosen even if it ranked first.
    """
    rec, _beats = _three_channel_recording(fs)
    _best, table = rank_hr_channels(rec, _trains(rec))
    row = table[table["channel"] == "ANT1"].iloc[0]

    assert table["channel"].iloc[-1] == "ANT1"
    assert not bool(row["eligible"])
    assert "rescue_rate" in str(row["vetoed_by"])


def test_the_gates_veto_rather_than_penalise(fs: float) -> None:
    """A vetoed channel is excluded outright, even when it ranks above an eligible one.

    The amplitude-wandering channel scores 197 - second of three - but its
    ``implausible_frac`` of 0.28 is over the gate, so it is not a candidate at all.
    """
    rec, _beats = _three_channel_recording(fs)
    _best, table = rank_hr_channels(rec, _trains(rec))
    row = table[table["channel"] == "LVN2"].iloc[0]

    assert float(row["implausible_frac"]) > PROVISIONAL_MAX_IMPLAUSIBLE_FRAC
    assert not bool(row["eligible"])
    assert "implausible_frac" in str(row["vetoed_by"])
    assert float(row["snr"]) > float(table[table["channel"] == "ANT1"]["snr"].iloc[0])


def test_the_chosen_channel_is_logged(fs: float, caplog: pytest.LogCaptureFixture) -> None:
    """Re-picked per recording and logged, so a channel that stops being best shows."""
    rec, _beats = _three_channel_recording(fs)
    with caplog.at_level(logging.INFO, logger="gems_blanking_v2.physio.rpeaks"):
        rank_hr_channels(rec, _trains(rec))

    assert any("HRV channel LVN1 chosen" in r.getMessage() for r in caplog.records)
    assert any("drift signal" in r.getMessage() for r in caplog.records)


def test_beat_count_against_the_median_is_reported_not_used(fs: float) -> None:
    """A sanity print, per the spec: present in the table, absent from the ranking."""
    rec, _beats = _three_channel_recording(fs)
    _best, table = rank_hr_channels(rec, _trains(rec))

    assert "beats_vs_median" in table.columns
    assert float(table["beats_vs_median"].iloc[0]) == pytest.approx(
        float(table["n_beats"].iloc[0] - table["n_beats"].median())
    )


def test_ranking_raises_when_every_channel_is_vetoed(fs: float) -> None:
    """No fallback chain. A recording with no usable channel is reported, not guessed."""
    rec, _beats = _three_channel_recording(fs)
    trains = _trains(rec)
    trains["LVN1"] = BeatTrain(
        t_s=trains["LVN1"].t_s,
        tag=trains["LVN1"].tag,
        implausible_frac=0.9,
        rescue_rate=0.9,
    )

    with pytest.raises(ValueError, match="every channel was vetoed"):
        rank_hr_channels(rec, trains)


def test_ranking_rejects_a_train_for_an_absent_channel(fs: float) -> None:
    """A name that is not in the recording is an error, not a skipped row."""
    rec, _beats = _three_channel_recording(fs)
    trains = _trains(rec)
    trains["RVN9"] = trains["LVN1"]

    with pytest.raises(ValueError, match="names no channel"):
        rank_hr_channels(rec, trains)


def test_ranking_rejects_an_empty_train_set(fs: float) -> None:
    """Nothing to rank is an error, not an arbitrary pick."""
    rec, _beats = _three_channel_recording(fs)

    with pytest.raises(ValueError, match="at least one train"):
        rank_hr_channels(rec, {})


# ---------------------------------------------------------------------------
# contracts and edges
# ---------------------------------------------------------------------------


def test_the_sample_rate_is_read_never_assumed(fs: float) -> None:
    """Beat times in seconds must be the same whatever the rate the file was at.

    Run the same 30 s of beats at the nominal rate and at half of it; the recovered
    beat times agree to under a millisecond, which they only can if ``fs`` is carried
    through rather than assumed.
    """
    at_nominal, _beats, _weak = make_ecg(fs, 30.0, weak_frac=0.0, seed=11)
    at_half, _beats_half, _weak_half = make_ecg(fs / 2, 30.0, weak_frac=0.0, seed=11)

    nominal = detect_rpeaks(at_nominal, fs)
    halved = detect_rpeaks(at_half, fs / 2)

    assert abs(nominal.n_beats - halved.n_beats) <= 1
    n = min(nominal.n_beats, halved.n_beats)
    assert float(np.max(np.abs(nominal.t_s[:n] - halved.t_s[:n]))) < 0.001


def test_a_nan_span_is_filtered_over_but_never_returned_as_a_beat(fs: float) -> None:
    """Interpolation for filtering is temporary; no beat may come out of a NaN span.

    Invariant 8: the interpolation exists so the filter does not smear NaN across the
    whole trace, and nothing derived from it survives. A beat landing inside the
    blanked span would be a fabricated fiducial.
    """
    sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=11)
    blank = (20.0, 24.0)
    holed = sig.copy()
    holed[int(blank[0] * fs) : int(blank[1] * fs)] = np.nan

    train = detect_rpeaks(holed, fs)
    inside = train.t_s[(train.t_s > blank[0] + 0.05) & (train.t_s < blank[1] - 0.05)]

    assert inside.size == 0, f"{inside.size} beats came out of a blanked span"

    # Without the interpolation the filter smears NaN over the whole trace and the
    # train comes back empty, which would satisfy the assertion above on its own.
    # So hold the other side too: every beat outside the blank is still found.
    outside = beats[(beats < blank[0] - 0.05) | (beats > blank[1] + 0.05)]
    assert _matched(train.t_s, outside).size == outside.size, (
        f"only {_matched(train.t_s, outside).size} of {outside.size} beats outside "
        "the blanked span survived - the NaN span was not interpolated over"
    )
    assert train.n_beats >= outside.size


def test_an_all_nan_channel_yields_an_empty_train(fs: float) -> None:
    """A dead channel is empty, not an exception and not a guess."""
    train = detect_rpeaks(np.full(int(5 * fs), np.nan), fs)

    assert train.n_beats == 0
    assert train.gaps == []
    assert train.rr_s().size == 0
    assert train.usable_rr_s().size == 0


def test_a_flat_channel_yields_an_empty_train(fs: float) -> None:
    """Zero variance means zero sigma, which is not a threshold anyone can use."""
    train = detect_rpeaks(np.zeros(int(5 * fs)), fs)

    assert train.n_beats == 0
    assert np.isnan(train.global_rr_s)


def test_usable_rr_excludes_every_interval_touching_a_gap() -> None:
    """The downstream contract, checked directly against a hand-built train."""
    train = BeatTrain(
        t_s=np.array([0.0, 0.15, 0.45, 0.60, 0.75]),
        tag=np.array(["detected"] * 5, dtype="<U8"),
        gaps=[(0.15, 0.45, 1)],
    )

    assert train.rr_s().size == 4
    assert train.usable_rr_mask().tolist() == [True, False, True, True]
    assert train.usable_rr_s().tolist() == pytest.approx([0.15, 0.15, 0.15])
    assert train.n_missing == 1


def test_a_beat_train_must_be_ordered_and_consistent() -> None:
    """Two ways to build a train that no consumer could read."""
    with pytest.raises(ValueError, match="must match in length"):
        BeatTrain(t_s=np.array([0.0, 0.1]), tag=np.array(["detected"], dtype="<U8"))

    with pytest.raises(ValueError, match="strictly increasing"):
        BeatTrain(
            t_s=np.array([0.0, 0.2, 0.1]),
            tag=np.array(["detected"] * 3, dtype="<U8"),
        )


def test_detection_is_deterministic(fs: float) -> None:
    """No adaptive state, no RNG: the same input gives the same train twice."""
    sig, _beats, _weak = make_ecg(fs, 30.0, weak_frac=0.09, seed=11)

    first, second = detect_rpeaks(sig, fs), detect_rpeaks(sig, fs)

    assert np.array_equal(first.t_s, second.t_s)
    assert np.array_equal(first.tag, second.tag)
    assert first.gaps == second.gaps


def test_the_detection_band_is_the_hrv_consumers_band() -> None:
    """One source of truth: the detector reads the band its consumer masks.

    These drifted apart once already - A.4 carried ``1-100`` for hrv after detection
    had moved - and a contamination band that is not the band the detector reads
    measures the wrong thing.
    """
    assert HR_BAND == "10-150"
    assert (BANDS[HR_BAND].lo_hz, BANDS[HR_BAND].hi_hz) == DETECT_BAND_HZ
    assert PASS1_PROMINENCE_SIGMA == 6.0
    assert not hasattr(rp, "ALGORITHM_BAND_HZ"), (
        "the superseded 1-100 band is reachable again - it competes with the ruling"
    )


def test_the_superseded_band_emits_false_beats(fs: float) -> None:
    """Why the ruling went the way it did. The stale numbers live here, not in code.

    Measured over 8 seeds, 3160 true beats, **counting pass 1 only**: ``10-150 Hz,
    k=6`` emits **0** false beats where ``1-100 Hz, k=3`` emits **14**. The superseded
    setting also misses none of the attenuated beats, so its pass 2 never runs.

    Run through ``_prepare``, ``find_peaks`` and ``_enforce_plausibility`` directly,
    so that no superseded constant has to exist in the module to keep this
    measurable.
    """
    counts: dict[str, int] = {}
    for label, band, sigma_multiple in (
        ("ruled", DETECT_BAND_HZ, PASS1_PROMINENCE_SIGMA),
        ("superseded", (1.0, 100.0), 3.0),
    ):
        false_beats = 0
        for seed in SEEDS:
            sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.09, seed=seed)
            y, fs_d = rp._prepare(sig, fs, band)
            peaks, props = find_peaks(
                y,
                prominence=sigma_multiple * robust_sigma(y),
                distance=max(int(round(R_MIN_S * fs_d)), 1),
            )
            kept, _dropped = rp._enforce_plausibility(
                peaks,
                np.asarray(props["prominences"], dtype=np.float64),
                fs_d,
                GLOBAL_RR_FRACTION * rp._global_rr(peaks, fs_d),
            )
            false_beats += kept.size - _matched(kept / fs_d, beats).size
        counts[label] = false_beats

    assert counts["ruled"] == 0
    assert counts["superseded"] == 14, "the superseded false-beat count moved"


# ---------------------------------------------------------------------------
# the T wave, and the runaway it makes testable
# ---------------------------------------------------------------------------


def _local_median_plausibility(
    peak_idx: npt.NDArray[np.int64], fs: float, fraction: float
) -> npt.NDArray[np.int64]:
    """Apply the **withdrawn** rule: a threshold from the *accepted* sequence's median.

    Implemented here, never in the module, so its failure can be measured. The
    feedback is the whole point: each acceptance decision changes the threshold the
    next one is made against.
    """
    kept = [int(peak_idx[0])]
    for index in peak_idx[1:].tolist():
        if len(kept) > 1:
            recent = np.diff(np.asarray(kept))[-(2 * rp.LOCAL_MEDIAN_INTERVALS + 1) :] / fs
            threshold = fraction * float(np.median(recent))
        else:
            threshold = fraction * 0.150
        if (index - kept[-1]) / fs < threshold:
            continue
        kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def _provisional_peaks(
    sig: F64, fs: float, sigma_multiple: float = PASS1_PROMINENCE_SIGMA
) -> tuple[npt.NDArray[np.int64], F64, float]:
    """Return pass 1's peaks before any plausibility rule, plus the decimated rate."""
    y, fs_d = rp._prepare(sig, fs, DETECT_BAND_HZ)
    peaks, props = find_peaks(
        y,
        prominence=sigma_multiple * robust_sigma(y),
        distance=max(int(round(R_MIN_S * fs_d)), 1),
    )
    return peaks, np.asarray(props["prominences"], dtype=np.float64), fs_d


def test_the_ruled_detector_is_not_fooled_by_a_t_wave(fs: float) -> None:
    """A T wave is not a beat, and at the binding operating point it is not detected.

    Measured over 5 seeds with ``t_wave=True``: exactly 395 beats against 395 true
    ones, worst fiducial error 0.29 ms, and ``implausible_frac`` exactly **0.0000**.
    The same signal at ``k=3`` doubles every beat, which is the other half of why
    the ruling went as it did.
    """
    for seed in (11, 7, 3, 29, 101):
        sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=seed, t_wave=True)
        train = detect_rpeaks(sig, fs)

        assert train.n_beats == beats.size, f"seed {seed}: T waves were counted as beats"
        assert train.implausible_frac == 0.0
        assert float(np.max(_matched(train.t_s, beats))) < 0.001

    sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=11, t_wave=True)
    at_k3, _prom, _fs_d = _provisional_peaks(sig, fs, sigma_multiple=3.0)
    assert at_k3.size > 1.9 * beats.size, "k=3 stopped doubling every beat - re-measure"


def test_a_t_wave_reproduces_the_short_interval_cluster(fs: float) -> None:
    """The animal-J signature: false peaks clustered at ~0.55 x RR, not at random.

    On animal J the short-interval population sat at ~90 ms against a 164.7 ms RR.
    Here, detected at ``k=3`` where the T wave does clear the threshold, the extra
    peaks land at a median 0.55 x RR after their beat - the T wave, by construction,
    which is what makes this the right generator for the runaway.
    """
    sig, beats, _weak = make_ecg(fs, 60.0, weak_frac=0.0, seed=11, t_wave=True)
    peaks, _prom, fs_d = _provisional_peaks(sig, fs, sigma_multiple=3.0)
    times = peaks / fs_d
    true_rr = float(np.median(np.diff(beats)))

    phases = [
        (t - beats[beats < t][-1]) / true_rr
        for t in times.tolist()
        if float(np.min(np.abs(beats - t))) > 0.010 and (beats < t).any()
    ]

    assert len(phases) > 0.9 * beats.size, "the T waves were not detected at k=3"
    assert float(np.median(phases)) == pytest.approx(0.55, abs=0.08)


def test_the_local_median_plausibility_form_runs_away(fs: float) -> None:
    """**The runaway, reproduced in CI.** This is the claim the whole rule exists for.

    Construction: ``t_wave=True`` for a false-peak population at 0.55 x RR, plus 50%
    of beats attenuated below threshold for a long-interval population. Both are
    needed - a false-peak population alone is *self-correcting* under the local rule,
    because dropping every T wave leaves exactly the true RR.

    Measured over 5 seeds, at the specified 0.75 operating point against a true
    150.4 ms RR:

    ==============  ================  ==============
    form            retained RR       peaks dropped
    ==============  ================  ==============
    global          84.5-87.0 ms      9-13%
    local (accepted) **303-364 ms**   59-63%
    ==============  ================  ==============

    The local form is **bistable**: once a run of long intervals lifts the accepted
    median above one RR, every real interval becomes "too short", which leaves
    alternate beats, which makes the median 2 x RR, which locks it in. That is the
    330 ms and 54% measured on animal J.

    **The global form is not correct here either** - see
    :func:`test_the_global_reference_is_only_as_good_as_the_provisional_peaks`. The
    difference this test holds is that its error is *bounded by the contamination*
    while the local form's compounds without bound.
    """
    ratios: list[float] = []
    for seed in (11, 7, 3, 29, 101):
        sig, _beats, _weak = make_ecg(
            fs, 60.0, weak_frac=0.5, weak_amp_uv=(3.0, 5.0), seed=seed, t_wave=True
        )
        peaks, prominences, fs_d = _provisional_peaks(sig, fs)
        global_rr = rp._global_rr(peaks, fs_d)

        by_global, _dropped = rp._enforce_plausibility(
            peaks, prominences, fs_d, GLOBAL_RR_FRACTION * global_rr
        )
        by_local = _local_median_plausibility(peaks, fs_d, GLOBAL_RR_FRACTION)

        local_rr = float(np.median(np.diff(by_local))) / fs_d
        assert local_rr > 0.250, f"seed {seed}: the local form did not run away"
        assert by_local.size < 0.45 * peaks.size, f"seed {seed}: it dropped too little"
        assert by_global.size > 0.85 * peaks.size, f"seed {seed}: the global form ran away"
        ratios.append(local_rr / global_rr)

    assert min(ratios) > 3.0, "the local form should inflate RR several-fold"


def test_the_global_reference_is_only_as_good_as_the_provisional_peaks(fs: float) -> None:
    """A limitation of the ruled rule, found by building the generator for the runaway.

    The global median assumes the provisional peaks are *mostly real*. When a
    false-peak population approaches the size of the real one, the median lands on
    the false interval instead: measured here at 84.5-87.0 ms, which is the R-to-T
    interval, against a true RR of 150.4 ms.

    This is not an argument for the local form, whose error at the same operating
    point is 303-364 ms and unbounded. It is a reason the **operating point** matters:
    at ``k=6`` on a clean file the short-interval rate is 0%, so the reference is
    safe. Worth stating in QC - a ``global_rr_s`` far from the rest of the file's
    channels means the peak set is contaminated, not that the heart rate changed.
    """
    sig, beats, _weak = make_ecg(
        fs, 60.0, weak_frac=0.5, weak_amp_uv=(3.0, 5.0), seed=11, t_wave=True
    )
    peaks, _prom, fs_d = _provisional_peaks(sig, fs)
    contaminated = rp._global_rr(peaks, fs_d)
    true_rr = float(np.median(np.diff(beats)))

    assert contaminated == pytest.approx(0.5 * true_rr, rel=0.20)

    clean_sig, clean_beats, _w = make_ecg(fs, 60.0, weak_frac=0.0, seed=11, t_wave=True)
    clean = detect_rpeaks(clean_sig, fs)
    assert clean.global_rr_s == pytest.approx(float(np.median(np.diff(clean_beats))), abs=0.005)


def test_the_runaway_against_real_data(real_recording: Path | None) -> None:
    """Regression against ``gems_j_t01_ms3_bl_230315``, where the collapse was seen.

    Skips when the file is unreachable, which is most of the time - the shared drive
    is not mounted in CI. The measurement to reproduce, animal J baseline, 601 s:

    - local form at fraction 0.75: RR median 330 ms, 54% of peaks dropped
    - global form: 3511 beats, RR 166.1 ms, short-interval 0.00%, long 2.05%

    The synthetic reproduces the *mechanism*
    (:func:`test_the_local_median_plausibility_form_runs_away`); this holds the
    measurement on the file it came from.
    """
    if real_recording is None:
        pytest.skip("gems_j_t01_ms3_bl_230315 is not reachable from this machine")

    from gems_blanking_v2.io.recording import load_recording  # noqa: PLC0415

    rec = load_recording(real_recording, animal="J").recording
    trains = {c.name: detect_rpeaks(rec.data[:, c.index], rec.fs) for c in rec.channels}
    best, _table = rank_hr_channels(rec, trains)
    train = trains[best]

    assert train.global_rr_s == pytest.approx(0.1661, abs=0.005)
    assert train.n_beats == pytest.approx(3511, rel=0.02)
    assert train.implausible_frac < 0.005

    peaks, _prom, fs_d = _provisional_peaks(rec.data[:, rec.channels[0].index], rec.fs)
    by_local = _local_median_plausibility(peaks, fs_d, GLOBAL_RR_FRACTION)
    assert float(np.median(np.diff(by_local))) / fs_d > 0.250
    assert by_local.size < 0.50 * peaks.size


def test_the_pass2_tolerance_only_matters_at_realistic_rr_spread() -> None:
    """Measure what widening 0.20 to 0.40 buys - and where it buys nothing.

    Counted on true beat trains with 9% of beats deleted, over how many genuinely
    doubled intervals :func:`missing_beat_count` admits:

    ========  =================  =================
    RR CV     admitted at 0.20   admitted at 0.40
    ========  =================  =================
    3.3%      161/161 (100%)     161/161 (100%)
    10%       123/158 (78%)      155/158 (98%)
    20%       81/156 (52%)       132/156 (85%)
    ========  =================  =================

    At the default generator's 3.3% CV the two tolerances are **indistinguishable**,
    which is why no test on that generator can justify the widening and why a
    mutation of the constant to 0.20 survives the rest of this suite. At a realistic
    spread 0.20 discards a third to a half of the intervals pass 2 exists to serve -
    the same complaint as the 18-38 of ~350 measured on animal J. So this test holds
    the *rule*, which is measurable, rather than the chosen value, which is a
    judgement.
    """
    admitted: dict[float, dict[float, tuple[int, int]]] = {}
    for sd_rr_s in (0.005, 0.015, 0.030):
        beats = make_beats(FS_NOMINAL_HZ, 300.0, sd_rr_s=sd_rr_s, seed=11)
        rng = np.random.default_rng(5)
        drop = rng.choice(
            np.arange(1, beats.size - 1), size=int(0.09 * beats.size), replace=False
        )
        rr = np.diff(np.delete(beats, drop))
        cv = float(np.std(np.diff(beats)) / np.mean(np.diff(beats)))

        admitted[cv] = {}
        for tolerance in (0.20, 0.40):
            original = rp.PASS2_MULTIPLE_TOLERANCE
            try:
                rp.PASS2_MULTIPLE_TOLERANCE = tolerance  # type: ignore[misc]
                ok = 0
                total = 0
                for j, interval in enumerate(rr.tolist()):
                    local = rp._local_median_rr(rr, j)
                    if int(round(interval / local)) < 2:
                        continue
                    total += 1
                    ok += rp.missing_beat_count(interval, local) is not None
            finally:
                rp.PASS2_MULTIPLE_TOLERANCE = original  # type: ignore[misc]
            admitted[cv][tolerance] = (ok, total)

    low_cv = min(admitted)
    assert admitted[low_cv][0.20] == admitted[low_cv][0.40], (
        "at a 3.3% RR CV the two tolerances should be indistinguishable"
    )

    high_cv = max(admitted)
    strict_ok, total = admitted[high_cv][0.20]
    wide_ok, _total = admitted[high_cv][0.40]
    assert strict_ok / total < 0.60, "0.20 stopped being too strict - re-measure"
    assert wide_ok / total > 0.80, "0.40 no longer admits most recoverable intervals"


def test_the_rescue_rate_and_implausible_frac_are_reported_per_channel(fs: float) -> None:
    """Both quality fractions are per-channel numbers, not thresholds applied here.

    ``detect_rpeaks`` never vetoes anything - it measures. Only
    :func:`rank_hr_channels` gates, which keeps the detector stateless and the policy
    in one place.
    """
    rec, _beats = _three_channel_recording(fs)
    trains = _trains(rec)

    assert trains["LVN1"].implausible_frac < PROVISIONAL_MAX_IMPLAUSIBLE_FRAC
    assert trains["LVN2"].implausible_frac > PROVISIONAL_MAX_IMPLAUSIBLE_FRAC
    assert trains["ANT1"].rescue_rate > PROVISIONAL_MAX_RESCUE_RATE
    for train in trains.values():
        assert 0.0 <= train.rescue_rate <= 1.0
        assert 0.0 <= train.implausible_frac <= 1.0
