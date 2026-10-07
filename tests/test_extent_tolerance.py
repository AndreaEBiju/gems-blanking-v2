"""Task 13: extent per consumer, measured settling, the operational cardiac test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.bands.envelope import impulse_response_length_s
from gems_blanking_v2.constants import BANDS, CONSUMERS
from gems_blanking_v2.detect import chain
from gems_blanking_v2.extent import tolerance as tl
from gems_blanking_v2.types import Candidate, Event

from tests.conftest import make_band_z, make_fiducial_shift

FS = 24414.0625
TOL = tl.ToleranceTable({"spikes": 4.0, "velocity": 4.0, "mmc": 3.0, "slow_wave": 3.0,
                         "breathing": 3.0}, source="synthetic test table")

def _settle(consumer: str) -> tl.ConsumerSettling:
    s = tl.consumer_settling(consumer, FS)
    assert s is not None
    return s



def _event(start: float, stop: float, *, judgement: str = "motion",
           source: str = "human", p: float = float("nan")) -> Event:
    c = Candidate(start, stop, ("L_T",), ("300-3000",), 9.0, "electrical")
    return Event(c, judgement, p, source)  # type: ignore[arg-type]


def test_tolerance_is_outside_the_generation_hash() -> None:
    mods = chain.generation_modules()
    assert "gems_blanking_v2.extent.tolerance" not in mods
    assert "gems_blanking_v2.extent.routing" not in mods


# ---------------------------------------------------------------------------
# the consumer table
# ---------------------------------------------------------------------------


def test_the_consumer_table_carries_the_ruled_corrections() -> None:
    cons = tl.extent_consumers()
    assert "slow_c" not in cons
    assert cons["mmc"].signals == ("ANT1", "ANT2", "ANT3")
    assert cons["slow_wave"].signals == ("ANT1", "ANT2", "ANT3")
    assert set(cons) == set(tl.CONSUMER_FILTERS)


def test_the_corrections_are_still_needed_in_constants() -> None:
    """Fails once constants.py carries R8 and the strike: then remove the patch here."""
    by = {c.name: c for c in CONSUMERS}
    assert "slow_c" in by, "constants.py struck slow_c: delete STRUCK_CONSUMERS"
    assert by["slow_wave"].signals == ("stomach_ref",), (
        "constants.py now has slow_wave's signals: drop it from CONSUMER_SIGNAL_ERRATA")
    assert by["mmc"].signals == ("stomach_ref",), (
        "constants.py now has mmc's signals: delete CONSUMER_SIGNAL_ERRATA")


def test_consumer_signals_resolve_per_recording() -> None:
    assert tl.consumer_signals("spikes", cuffs=("L", "R")) == ("L_T", "R_T")
    assert tl.consumer_signals("velocity", cuffs=("L",)) == ("L_V1", "L_V3")
    assert tl.consumer_signals("hrv", best_hr_channel="RVN2") == ("RVN2",)
    with pytest.raises(ValueError, match="best HR channel"):
        tl.consumer_signals("breathing")


# ---------------------------------------------------------------------------
# settling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("consumer", sorted(tl.CONSUMER_FILTERS))
def test_impz_settling_matches_an_independent_step_response(consumer: str) -> None:
    f = tl.CONSUMER_FILTERS[consumer]
    for fs in (FS, 4000.0):
        assert tl.step_settling_s(f, fs) == pytest.approx(tl.impulse_settling_s(f, fs),
                                                          abs=2.0 / fs)


@pytest.mark.parametrize(("consumer", "lo", "hi"), [
    ("spikes", 300.0, 3000.0), ("mmc", 2.0, 50.0), ("hrv", 10.0, 150.0)])
def test_order_4_chains_measure_exactly_what_the_detection_side_measures(
    consumer: str, lo: float, hi: float
) -> None:
    """One method: where the designs coincide, the two measurements are equal."""
    f = tl.CONSUMER_FILTERS[consumer]
    assert tl.impulse_settling_s(f, FS) == impulse_response_length_s(lo, hi, FS)


def test_measured_settling_per_consumer_at_the_tdt_rate() -> None:
    """The numbers this build reports (impz, 1% of peak, at 24414.0625 Hz)."""
    got = {c: _settle(c) for c in tl.CONSUMER_FILTERS}
    assert got["spikes"].total_s == pytest.approx(0.00512, abs=1e-4)
    assert got["hrv"].total_s == pytest.approx(0.1402, abs=1e-3)
    assert got["mmc"].impulse_s == pytest.approx(0.4858, abs=1e-3)
    assert got["mmc"].total_s == 15.0  # the 30 s moving threshold's half-window (R6)
    sw = got["slow_wave"]
    assert sw.impulse_s == pytest.approx(8.167, abs=0.01)  # the spec's measured 8.17 s
    assert sw.total_s == sw.impulse_s > sw.extra_s == 2.5


def test_a_consumer_without_a_chain_has_unknown_settling() -> None:
    assert tl.consumer_settling("slow_c", FS) is None


# ---------------------------------------------------------------------------
# extent
# ---------------------------------------------------------------------------


def _z(eng_bump: float, slow_bump: float, dur: float = 60.0) -> dict[Any, Any]:
    eng = make_band_z("300-3000", dur, bumps=((20.0, 20.2, eng_bump, "L_T"),), signal="L_T")
    slow = make_band_z("0-2", dur, bumps=((19.0, 27.0, slow_bump, "ANT1"),),
                       signal="ANT1")
    return {("L_T", "300-3000"): eng.z_max, ("ANT1", "0-2"): slow.z_max}


def test_an_eng_only_event_has_a_spike_extent_and_none_for_slow_wave() -> None:
    ev = _event(19.5, 21.0)
    z = _z(eng_bump=12.0, slow_bump=1.0)
    spikes = tl.compute_extent(ev, z, "spikes", signal="L_T", tolerances=TOL, fs=FS, z_t0_s=0.0)
    slow = tl.compute_extent(ev, z, "slow_wave", signal="ANT1", tolerances=TOL, fs=FS,
                             z_t0_s=0.0)
    assert slow is None
    assert spikes is not None
    assert (spikes.core_start_s, spikes.core_stop_s) == pytest.approx((20.0, 20.2))
    pad = _settle("spikes").total_s
    assert (spikes.start_s, spikes.stop_s) == pytest.approx((20.0 - pad, 20.2 + pad))
    assert spikes.resolution_s == BANDS["300-3000"].window_s


def test_the_extent_is_not_the_candidate_interval() -> None:
    """A 1.5 s candidate whose ENG evidence is 200 ms gets a ~200 ms spike extent."""
    ev = _event(19.5, 21.0)
    ext = tl.compute_extent(ev, _z(12.0, 1.0), "spikes", signal="L_T", tolerances=TOL, fs=FS,
                            z_t0_s=0.0)
    assert ext is not None and ext.stop_s - ext.start_s < 0.25


def test_a_slow_band_extent_states_its_coarse_resolution() -> None:
    ev = _event(19.5, 21.0)
    ext = tl.compute_extent(ev, _z(12.0, 9.0), "slow_wave", signal="ANT1",
                            tolerances=TOL, fs=FS, z_t0_s=0.0)
    assert ext is not None and ext.resolution_s == 7.5
    assert ext.start_s == pytest.approx(19.0 - _settle("slow_wave").total_s)


def test_an_unjudged_or_uncalibrated_event_is_not_confirmed() -> None:
    assert tl.is_confirmed_motion(_event(1, 2))
    assert not tl.is_confirmed_motion(_event(1, 2, judgement="physiology"))
    model = _event(1, 2, judgement="unjudged", source="model", p=0.9)
    with pytest.raises(ValueError, match="calibrated"):
        tl.is_confirmed_motion(model, p_threshold=0.5)
    assert tl.is_confirmed_motion(model, p_threshold=0.5, calibrator="cal/iso.json")
    assert not tl.is_confirmed_motion(model, p_threshold=0.95, calibrator="cal/iso.json")


def test_tolerances_have_no_defaults_and_name_their_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source"):
        tl.ToleranceTable({"spikes": 4.0}, source="")
    with pytest.raises(ValueError, match="operational"):
        tl.ToleranceTable({"hrv": 3.0}, source="x")
    with pytest.raises(KeyError, match="mmc"):
        tl.ToleranceTable({"spikes": 4.0}, source="x").for_consumer("mmc")
    p = tmp_path / "tol.json"
    p.write_text(json.dumps({"z_tol": {"spikes": 4.5}}), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        tl.ToleranceTable.from_json(p)


def test_mmc_is_not_measured_within_15_s_of_any_blank() -> None:
    spans = tl.mmc_not_measured_spans([(100.0, 101.0), (110.0, 111.0), (300.0, 300.5)], 600.0)
    assert spans == [(85.0, 126.0), (285.0, 315.5)]
    assert tl.mmc_not_measured_spans([(5.0, 6.0)], 600.0) == [(0.0, 21.0)]


# ---------------------------------------------------------------------------
# the operational cardiac test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("shift_s", "caught"), [(0.002, True), (0.00005, False)])
def test_a_2_ms_fiducial_shift_is_caught_and_a_0_05_ms_one_is_not(
    shift_s: float, caught: bool
) -> None:
    sim = make_fiducial_shift(FS, 6.0, beat_index=15, shift_s=shift_s, seed=3)
    span = (sim.beat_s - 0.01, sim.beat_s + 0.01)
    v = tl.cardiac_operational_damage(sim.contaminated, FS, span, suppressed=sim.reference)
    assert v.changed is caught
    assert v.n_added == v.n_lost == 0


def test_a_shift_within_the_detector_quantisation_is_not_a_change() -> None:
    tol = tl.hrv_fiducial_tolerance_s(FS)
    assert 0.0004 < tol < 0.001  # 1.5 samples at ~2.03 kHz: under the 1-2 ms that moves RMSSD


def test_without_a_reference_a_beat_inside_the_span_counts_as_changed() -> None:
    sim = make_fiducial_shift(FS, 6.0, beat_index=15, shift_s=0.0, seed=3)
    v = tl.cardiac_operational_damage(sim.contaminated, FS, (sim.beat_s - 0.01,
                                                             sim.beat_s + 0.01))
    assert v.changed and v.n_inside_unverified == 1
    quiet = (sim.beat_s + 0.05, sim.beat_s + 0.07)  # between beats
    assert not tl.cardiac_operational_damage(sim.contaminated, FS, quiet).changed


def test_beat_train_comparison_counts_added_lost_and_moved() -> None:
    v = tl.beat_train_changed([1.0, 2.0, 3.0], [1.0, 2.003, 4.0], shift_tol_s=0.001,
                              match_s=0.02)
    assert (v.n_added, v.n_lost, v.changed) == (1, 1, True)
    assert v.max_shift_s == pytest.approx(0.003)
    assert not tl.beat_train_changed([1.0, 2.0], [1.0, 2.0005], shift_tol_s=0.001,
                                     match_s=0.02).changed


def test_hrv_extent_follows_the_operational_verdict() -> None:
    moved = make_fiducial_shift(FS, 6.0, beat_index=15, shift_s=0.002, seed=3)
    ev = _event(moved.beat_s - 0.01, moved.beat_s + 0.01)
    ext = tl.hrv_extent(ev, moved.contaminated, FS, signal="RVN2", x_t0_s=0.0,
                        suppressed=moved.reference)
    assert ext is not None and ext.consumer == "hrv" and ext.band == "10-150"
    pad = _settle("hrv").total_s
    assert ext.stop_s - ext.start_s == pytest.approx(0.02 + 2 * pad)
    still = make_fiducial_shift(FS, 6.0, beat_index=15, shift_s=0.00005, seed=3)
    assert tl.hrv_extent(ev, still.contaminated, FS, signal="RVN2", x_t0_s=0.0,
                         suppressed=still.reference) is None
    with pytest.raises(ValueError, match="operational"):
        tl.compute_extent(ev, {}, "hrv", signal="RVN2", tolerances=TOL, fs=FS, z_t0_s=0.0)


def test_no_nan_or_zero_tolerance_slips_through() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            tl.ToleranceTable({"spikes": bad}, source="x")
    assert np.isfinite(TOL.for_consumer("spikes"))


def test_settling_is_the_maximum_over_every_stage() -> None:
    """Invariant 19: a later stage longer than the filter sets the settling."""
    assert tl.ConsumerSettling("x", impulse_s=1.0, extra_s=2.5).total_s == 2.5
    assert tl.ConsumerSettling("x", impulse_s=3.0, extra_s=2.5).total_s == 3.0


def test_z_on_a_region_timeline_is_placed_by_its_origin() -> None:
    """A recovery epoch starting at 120 s: z frame 0 is 120 s on the event's timeline."""
    ev = _event(139.5, 141.0)
    z = _z(eng_bump=12.0, slow_bump=1.0)  # bump at 20.0-20.2 s on the region's timeline
    ext = tl.compute_extent(ev, z, "spikes", signal="L_T", tolerances=TOL, fs=FS,
                            z_t0_s=120.0)
    assert ext is not None
    assert (ext.core_start_s, ext.core_stop_s) == pytest.approx((140.0, 140.2))
    with pytest.raises(tl.ExtentNotAssessableError, match="not assessable"):
        tl.compute_extent(ev, z, "spikes", signal="L_T", tolerances=TOL, fs=FS, z_t0_s=0.0)


def test_every_event_gets_an_answer_for_every_consumer() -> None:
    evs = {"a": _event(19.5, 21.0), "b": _event(40.0, 41.0)}
    rows = tl.extents_for_events(evs, _z(12.0, 1.0),
                                 {"spikes": ("L_T",), "slow_wave": ("ANT1",)},
                                 tolerances=TOL, fs=FS, z_t0_s=0.0,
                                 confirmed=tl.is_confirmed_motion)
    got = {(e, c): x is not None for e, c, _s, x in rows}
    assert got == {("a", "spikes"): True, ("a", "slow_wave"): False,
                   ("b", "spikes"): False, ("b", "slow_wave"): False}


def test_an_event_outside_or_nan_in_z_is_not_assessable_not_under_tolerance() -> None:
    z = _z(12.0, 1.0)
    nan_z = {k: np.full_like(v, np.nan) for k, v in z.items()}
    ev = _event(19.5, 21.0)
    with pytest.raises(tl.ExtentNotAssessableError):
        tl.compute_extent(ev, nan_z, "spikes", signal="L_T", tolerances=TOL, fs=FS, z_t0_s=0.0)
    rows = tl.extents_for_events({"a": ev}, nan_z, {"spikes": ("L_T",)}, tolerances=TOL, fs=FS,
                                 z_t0_s=0.0, confirmed=tl.is_confirmed_motion)
    assert isinstance(rows[0][3], tl.NotAssessable)


def test_hrv_is_never_silently_skipped() -> None:
    evs = {"a": _event(19.5, 21.0)}
    with pytest.raises(ValueError, match="refusing to skip hrv"):
        tl.extents_for_events(evs, _z(12.0, 1.0), {"hrv": ("RVN2",)}, tolerances=TOL, fs=FS,
                              z_t0_s=0.0, confirmed=tl.is_confirmed_motion)
    with pytest.raises(ValueError, match="refusing to skip hrv"):
        tl.extents_for_events(evs, _z(12.0, 1.0), {"hrv": ("RVN2",)}, tolerances=TOL, fs=FS,
                              z_t0_s=0.0, confirmed=tl.is_confirmed_motion,
                              hr_signal=("RVN1", np.zeros(10), 0.0))


def test_an_unconfirmed_event_gets_no_extent() -> None:
    evs = {"a": _event(19.5, 21.0, judgement="physiology")}
    with pytest.raises(ValueError, match="not confirmed"):
        tl.extents_for_events(evs, _z(12.0, 1.0), {"spikes": ("L_T",)}, tolerances=TOL, fs=FS,
                              z_t0_s=0.0, confirmed=tl.is_confirmed_motion)


def test_hrv_extent_is_placed_by_the_signal_origin() -> None:
    """The HR channel's x[0] at 120 s: the extent comes back on the event's timeline."""
    moved = make_fiducial_shift(FS, 6.0, beat_index=15, shift_s=0.002, seed=3)
    ev = _event(120.0 + moved.beat_s - 0.01, 120.0 + moved.beat_s + 0.01)
    ext = tl.hrv_extent(ev, moved.contaminated, FS, signal="RVN2", x_t0_s=120.0,
                        suppressed=moved.reference)
    assert ext is not None and ext.core_start_s == pytest.approx(120.0 + moved.beat_s - 0.01)


def test_hrv_origin_decides_which_beats_the_span_holds() -> None:
    """Without a suppressed reference the span must land on x's samples via x_t0_s."""
    sim = make_fiducial_shift(FS, 6.0, beat_index=15, shift_s=0.0, seed=3)
    ev = _event(120.0 + sim.beat_s - 0.01, 120.0 + sim.beat_s + 0.01)
    ext = tl.hrv_extent(ev, sim.contaminated, FS, signal="RVN2", x_t0_s=120.0)
    assert ext is not None  # the beat inside the span cannot be verified -> changed
