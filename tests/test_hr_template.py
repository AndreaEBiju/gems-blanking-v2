"""Tests for :mod:`gems_blanking_v2.physio.hr_template` (rulings 2026-09-30 (f), (g)).

One fixture case per measured failure mode (``make_hr_trouble``). Each case first checks
that the fixture reproduces the failure on the existing detectors (invariant 41), then
what the template train does about it. Where the train does NOT fully repair a mode
(recall under capture, task 05's misses) the test states the measured ceiling as a
no-regression bound, not as a pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.physio import hr_template as ht
from gems_blanking_v2.physio.hr_channel import findpeaks_replica
from gems_blanking_v2.physio.rpeaks import DETECT_BAND_HZ, _prepare, detect_rpeaks
from scipy.signal import find_peaks

from tests.conftest import HrTrouble, make_hr_trouble

FS = 4000.0
DUR = 180.0
P = ht.TemplateParams(0.8, 0.5, 0.5)
F64 = npt.NDArray[np.float64]


def _dist(a: F64, b: F64) -> F64:
    b = np.sort(np.asarray(b, float))
    if b.size == 0:
        return np.full(np.asarray(a).size, np.inf)
    j = np.searchsorted(b, a)
    return np.minimum(np.abs(a - b[np.clip(j - 1, 0, b.size - 1)]),
                      np.abs(b[np.clip(j, 0, b.size - 1)] - a))


def recall(w: HrTrouble, got: F64) -> float:
    return float(np.mean(_dist(w.beats_s, got) <= 0.001))


def extras(w: HrTrouble, got: F64) -> int:
    return int((_dist(np.asarray(got, float), w.beats_s) > 0.001).sum())


def _case(**kw: float) -> HrTrouble:
    return make_hr_trouble(FS, DUR, seed=7, **kw)


@pytest.fixture(scope="module")
def clean() -> HrTrouble:
    return _case()


# --- the parameters -----------------------------------------------------------------


def test_k_is_one_of_the_ruled_choices_and_multipliers_relax() -> None:
    assert ht.K_CHOICES == (0.6, 0.65, 0.7, 0.75, 0.8)
    with pytest.raises(ValueError, match="not one of"):
        ht.TemplateParams(0.9, 0.5, 0.5)
    with pytest.raises(ValueError, match="must relax"):
        ht.TemplateParams(0.8, 1.2, 0.5)
    with pytest.raises(ValueError, match="must relax"):
        ht.TemplateParams(0.8, 0.5, 0.0)


def test_the_running_median_is_centred() -> None:
    idx = np.r_[np.arange(0, 100) * 300, 30000 + np.arange(1, 100) * 600].astype(np.int64)
    med = ht.running_median_rr(idx, 1000.0, half_s=5.0)
    assert med[10] == pytest.approx(0.3)
    assert med[-10] == pytest.approx(0.6)


# --- clean: the identity -----------------------------------------------------------------


def test_a_clean_channel_is_returned_unchanged_with_aligned_fiducials_on_the_peaks(clean) -> None:  # noqa: ANN001
    t = ht.template_train(clean.signal, FS, clean.events_s, P)
    assert t.no_train == "" and t.seed == "task05"
    assert recall(clean, t.t_s) == 1.0 and extras(clean, t.t_s) == 0
    assert abs(t.fiducial_offset_s) < ht.MAX_FIDUCIAL_OFFSET_S
    assert t.counts["floor_beats"] >= ht.MIN_FLOOR_BEATS


# --- capture ---------------------------------------------------------------------------


@pytest.mark.parametrize(("rate", "gain"), [(0.5, 0.05), (2.0, 0.10)])
def test_captures_are_dropped_extras_fall_and_recall_rises(rate: float, gain: float) -> None:
    w = _case(transients_per_s=rate)
    fp = findpeaks_replica(w.signal, FS)
    assert extras(w, fp) > 0.1 * w.beats_s.size  # the fixture captures (control)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    assert t.counts["suspects_dropped"] >= 0.99 * t.counts["suspects"]
    lone = w.transients_s[_dist(w.transients_s, w.beats_s) > 0.02]
    assert np.mean(_dist(lone, t.t_s) <= 0.002) <= 0.01  # no transient is taken as a beat
    assert extras(w, t.t_s) <= 0.1 * extras(w, fp)
    assert recall(w, t.t_s) >= recall(w, fp) + gain
    assert t.counts["researched"] > 0  # recovered beats come from the re-search


def test_an_everywhere_unclear_autocorrelation_yields_no_train() -> None:
    w = _case(transients_per_s=6.0)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    assert t.t_s.size == 0 and "unclear in every minute" in t.no_train


def test_a_real_beat_hit_by_a_transient_is_kept_by_masked_correlation() -> None:
    w = _case(transients_per_s=2.0, on_beat_frac=0.3)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    hit = w.transients_s[_dist(w.transients_s, w.beats_s) <= 0.002]
    hit_beats = w.beats_s[_dist(w.beats_s, hit) <= 0.002]
    assert hit_beats.size > 50
    assert np.mean(_dist(hit_beats, t.t_s) <= 0.001) >= 0.9


def test_the_score_band_is_10_to_900_hz() -> None:
    assert ht.SCORE_BAND_HZ == (10.0, 900.0)


# --- floor extras ------------------------------------------------------------------------


def test_extras_at_the_findpeaks_spacing_floor_are_resolved_by_template_match() -> None:
    w = _case(rr_s=0.21)
    fp = findpeaks_replica(w.signal, FS)
    assert extras(w, fp) > 0.5 * w.beats_s.size  # a noise peak at ~100 ms on most cycles
    for k in (0.6, 0.8):
        t = ht.template_train(w.signal, FS, w.events_s, ht.TemplateParams(k, 0.5, 0.5))
        assert extras(w, t.t_s) == 0 and recall(w, t.t_s) >= 0.995


# --- a mid-cycle peak ---------------------------------------------------------------------


def test_a_mid_cycle_peak_on_every_cycle_is_split_off_by_template_snr() -> None:
    w = _case(rr_s=0.21, midcycle_amp=0.6)
    t5 = np.asarray(detect_rpeaks(w.signal, FS).t_s)
    assert extras(w, t5) > 0.5 * w.beats_s.size  # task 05 takes the mid-cycle peak (control)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    assert t.seed == "split" and t.split_snr is not None
    hi, lo = t.split_snr
    assert (hi - lo) / hi > ht.AMBIGUOUS_SNR
    assert np.isfinite(t.split_jitter_s)
    assert extras(w, t.t_s) == 0 and recall(w, t.t_s) >= 0.995


def test_a_mid_cycle_wave_as_strong_as_the_qrs_is_ambiguous_and_yields_no_train() -> None:
    w = _case(rr_s=0.21, midcycle_amp=1.0, midcycle_width_ms=10.0)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    assert t.t_s.size == 0 and "ambiguous" in t.no_train
    hi, lo = t.split_snr
    assert (hi - lo) / hi <= ht.AMBIGUOUS_SNR


# --- task 05's misses ---------------------------------------------------------------------


def test_task05_misses_are_reproduced_and_not_made_worse() -> None:
    w = _case(alternans=0.3, noise_uv=12.0)
    t5 = np.asarray(detect_rpeaks(w.signal, FS).t_s)
    assert recall(w, t5) < 0.97  # task 05 misses the weak beats (control)
    t = ht.template_train(w.signal, FS, w.events_s, ht.TemplateParams(0.8, 0.2, 0.5))
    # measured ceiling, not a pass: a noise peak takes the weak beat's place, so no gap
    # opens for the re-search (reported for ruling)
    assert recall(w, t.t_s) >= recall(w, t5) - 0.005


# --- rate rises ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", ht.K_CHOICES)
def test_the_refractory_removes_no_real_beat_during_a_rate_rise(k: float) -> None:
    w = _case(rr_s=0.19, rr_end_s=0.15)
    t = ht.template_train(w.signal, FS, w.events_s, ht.TemplateParams(k, 0.5, 0.5))
    assert recall(w, t.t_s) == 1.0 and extras(w, t.t_s) == 0
    assert t.counts["refractory_removed"] == 0


# --- cardiac events on most beats ---------------------------------------------------------


def test_cardiac_events_on_most_beats_do_not_starve_the_template_or_the_median() -> None:
    w = _case(transients_per_s=0.5, cardiac_event_frac=0.9)
    alone = _case(transients_per_s=0.5)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    base = ht.template_train(alone.signal, FS, alone.events_s, P)
    assert t.counts["floor_beats"] >= ht.MIN_FLOOR_BEATS
    assert t.counts["suspects"] > 0.8 * w.beats_s.size
    assert recall(w, t.t_s) >= recall(alone, base.t_s) - 0.05
    assert extras(w, t.t_s) <= 0.01 * w.beats_s.size


def test_fewer_than_50_non_suspect_clean_beats_yield_no_train() -> None:
    w = _case(cardiac_event_frac=0.995)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    assert t.t_s.size == 0 and "no floor" in t.no_train
    assert t.counts["floor_beats"] < ht.MIN_FLOOR_BEATS


# --- excision leaving under half the window --------------------------------------------------


def test_a_score_with_under_half_the_window_left_is_undefined_and_dropped(clean) -> None:  # noqa: ANN001
    targets = clean.beats_s[100:120]
    dense = np.concatenate([b + np.arange(-0.040, 0.0405, 0.003) for b in targets])
    t = ht.template_train(clean.signal, FS, np.sort(np.r_[clean.events_s, dense]), P)
    assert t.counts["undefined_dropped"] >= targets.size
    assert np.all(_dist(targets, t.t_s) > 0.001)  # never kept, never re-found
    assert sum(any(a < b < c for a, c in t.gaps_s) for b in targets) >= 0.5 * targets.size


# --- nothing is inserted -----------------------------------------------------------------------


def test_every_beat_is_a_measured_peak_never_an_expected_time() -> None:
    w = _case(missing_frac=0.02, transients_per_s=1.0)
    t = ht.template_train(w.signal, FS, w.events_s, P)
    y, fd = _prepare(w.signal, FS, DETECT_BAND_HZ)
    pk, _ = find_peaks(y)
    near = _dist(t.t_s, pk / fd)
    reach = np.where(t.kind == "aligned", ht.ALIGN_S, 0.0) + 0.5 / fd
    assert np.all(near <= reach)


def test_hr_template_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.physio.hr_template" not in chain.generation_modules()


# --- the two robust statistics, tested directly -------------------------------------------


def test_a_dropped_beat_does_not_stretch_the_running_median() -> None:
    fd = 1000.0
    fid = np.arange(0, 60000, 150, dtype=np.int64)
    match = np.ones(fid.size, dtype=bool)
    match[1::3] = False  # every third beat does not match the template
    ac = np.full(fid.size, 0.150)
    med = ht._matched_median(fid, match, fd, ac)
    assert np.allclose(med, 0.150)  # every 300 ms span over a non-match is left out
    no_ref = ht._matched_median(fid, match, fd, np.full(fid.size, np.nan))
    assert np.nanmax(no_ref) > 0.2  # without the reference they stretch it


def test_the_median_template_resists_transients_a_mean_would_absorb() -> None:
    w = _case(transients_per_s=6.0)
    ys, fd = _prepare(w.signal, FS, ht.SCORE_BAND_HZ)
    idx = np.round(w.beats_s * fd).astype(np.int64)
    near = _dist(w.beats_s, w.transients_s) <= 0.04  # beats with a transient in the window
    sc = ht._Scorer(ys, fd, idx, np.zeros(0, dtype=np.int64), 4)
    ref = ht._Scorer(ys, fd, idx[~near], np.zeros(0, dtype=np.int64), 4).template
    windows = ht._windows(ys, idx, sc.half)[0]
    mean_t = windows.mean(axis=0)
    assert near.mean() > 0.2
    assert np.corrcoef(sc.template, ref)[0, 1] > np.corrcoef(mean_t, ref)[0, 1]
    assert np.corrcoef(sc.template, ref)[0, 1] > 0.99
