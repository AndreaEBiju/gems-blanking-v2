"""Task 13: extent per consumer, measured settling, the operational cardiac test."""

from __future__ import annotations

import hashlib
import json
import math
import re
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
    """The numbers this build reports (impz, 1% of peak, at 24414.0625 Hz; edges measured).

    RULING 2026-10-09 (c): spikes and mmc use the MEASURED zero-phase edge settling, not
    task 13's one-way impulse response (5.1 ms) or 15 s half-window.
    """
    got = {c: _settle(c) for c in tl.CONSUMER_FILTERS}
    assert got["spikes"].impulse_s == pytest.approx(0.00512, abs=1e-4)  # task 13, superseded
    assert got["spikes"].total_s == pytest.approx(0.010282, abs=1e-12)  # 7.782 + 2.5 ms
    assert got["hrv"].total_s == pytest.approx(0.1402, abs=1e-3)
    assert got["mmc"].impulse_s == pytest.approx(0.4858, abs=1e-3)
    assert got["mmc"].extra_s == 0.0  # her moving median/MAD skip NaN ((c) 3 (b))
    assert got["mmc"].total_s == 1.5  # (c) 2: was 15.0
    sw = got["slow_wave"]
    assert sw.impulse_s == pytest.approx(8.167, abs=0.01)  # the spec's measured 8.17 s
    assert sw.total_s == sw.impulse_s > sw.extra_s == 2.5
    assert sw.edge is None and got["hrv"].edge is None  # not measured: impz stands


def test_the_edge_settlings_are_the_measured_ones_and_the_pads_round_them_up() -> None:
    """(c) 1-2: 7.782 ms + 2.5 ms before a gap (1.5 ms after) -> 10.5 ms; 1.1594 s -> 1.5 s."""
    sp, mmc = tl.EDGE_SETTLING["spikes"], tl.EDGE_SETTLING["mmc"]
    assert sp.filter_s == tl.SPIKE_STEP1_EDGE_S == 0.007782
    assert sp.before_gap_s == pytest.approx(0.007782 + 0.0025, abs=1e-15)
    assert sp.after_gap_s == pytest.approx(0.007782 + 0.0015, abs=1e-15)
    assert sp.total_s == sp.before_gap_s == sp.extent_pad_s
    assert sp.pad_s == 0.0105 == math.ceil(sp.total_s * 2e3) / 2e3  # rounded up to 0.5 ms
    assert sp.filter_s > sp.total_s - 0.003 > tl.impulse_settling_s(  # zero phase > one way
        tl.CONSUMER_FILTERS["spikes"], FS)
    assert mmc.filter_s == mmc.total_s == 1.1594
    assert mmc.pad_s == mmc.extent_pad_s == 1.5 == math.ceil(mmc.total_s * 2) / 2
    for e in tl.EDGE_SETTLING.values():
        assert e.ruling.startswith("RULING 2026-10-09") and e.files
        assert all(re.fullmatch(r"[0-9a-f]{64}", h) for _, h in e.files)


REPO = Path(__file__).resolve().parents[1]


def test_the_measurement_files_are_archived_in_the_repo_and_hash_as_recorded() -> None:
    """Every file an edge settling rests on is in the repo, byte for byte (LF), at its hash."""
    seen = set()
    for e in tl.EDGE_SETTLING.values():
        for f, h in e.files:
            assert f.startswith(tl.EDGE_MEASUREMENTS_DIR + "/"), f
            raw = (REPO / f).read_bytes().replace(b"\r\n", b"\n")  # eol=lf; never CRLF
            assert hashlib.sha256(raw).hexdigest() == h, f
            seen.add(f)
    assert len(seen) >= 9  # spikes 4, mmc 2, hr_band 3


def test_the_hr_band_edge_is_the_maximum_of_its_two_measurements() -> None:
    """0.2374 s: the beat detector's fs/12 measurement binds the full-rate trace one (0.1999 s).

    The full-rate file is measured with what Night 6 passes HR_BR_HRVAnalysis_beats: the
    file's fs (night6_run_recording.m:429, ``fs = plan.fs``) and ``H.order`` = 4.
    """
    hb = tl.EDGE_SETTLING["hr_band"]
    d = REPO / tl.EDGE_MEASUREMENTS_DIR
    full = json.loads((d / "hr_edge_settling.json").read_text(encoding="utf-8"))
    f12 = json.loads((d / "hr_edge_settling_fs12.json").read_text(encoding="utf-8"))
    assert full["fs"] == FS and full["order"] == 4 and full["band_hz"] == [10.0, 150.0]
    assert f12["fs"] == pytest.approx(FS / 12, abs=1e-9)
    assert (full["worst_s"], f12["worst_s"]) == (0.1999, 0.2374)
    assert hb.filter_s == tl.HR_BAND_EDGE_S == max(full["worst_s"], f12["worst_s"]) == 0.2374
    assert hb.before_gap_s == max(v[0] for v in f12["settling_s"].values()) == 0.2374
    assert hb.after_gap_s == max(v[1] for v in f12["settling_s"].values()) == 0.2236
    assert hb.total_s == 0.2374 > tl.impulse_settling_s(tl.CONSUMER_FILTERS["hrv"], FS)
    run = (REPO / "matlab" / "night6" / "night6_run_recording.m").read_text(encoding="utf-8")
    assert "HR_BR_HRVAnalysis_beats(o.beatsEpochFile, X, fs, H.cutoff, H.order," in run
    assert "fs = plan.fs;" in run and "P.hr = struct('cutoff', 8, 'order', 4," in run
    # the band is a stage, not a consumer: no extent is padded by it
    assert "hr_band" not in tl.CONSUMER_FILTERS
    assert _settle("hrv").edge is None and _settle("breathing").edge is None


def test_night6_declares_exactly_the_python_edge_settlings() -> None:
    """``matlab/night6/edge_settling.json`` is :func:`edge_settling_record`, byte for byte."""
    f = Path(__file__).resolve().parents[1] / "matlab" / "night6" / "edge_settling.json"
    rec = tl.edge_settling_record()
    assert json.loads(f.read_text(encoding="utf-8")) == rec
    assert f.read_bytes() == (json.dumps(rec, ensure_ascii=True, sort_keys=True, indent=1,
                                         allow_nan=False) + "\n").encode("ascii")
    assert rec["spikes"]["edge_buffer_ms"] == 10.5
    params = (f.parent / "night6_v2_params.m").read_text(encoding="utf-8")
    assert "P.edgeBufferMs = E.spikes.edge_buffer_ms;" in params  # read, never typed
    assert "'edgeBufferMs', 10.5" in params  # and asserted


def test_an_extent_is_padded_by_the_measured_edge_settling() -> None:
    """(c) 1-2: a spike extent gains 10.282 ms a side (not 5.1 ms), an mmc one 1.5 s (not 15)."""
    ev = _event(19.5, 21.0)
    z = _z(eng_bump=12.0, slow_bump=1.0)
    sp = tl.compute_extent(ev, z, "spikes", signal="L_T", tolerances=TOL, fs=FS, z_t0_s=0.0)
    assert sp is not None and sp.settling_s == pytest.approx(0.010282, abs=1e-12)
    assert sp.core_start_s - sp.start_s == pytest.approx(0.010282, abs=1e-9)
    assert sp.stop_s - sp.core_stop_s == pytest.approx(0.010282, abs=1e-9)
    mz = {("ANT1", "2-50"): make_band_z("2-50", 120.0, bumps=((50.0, 52.0, 12.0, "ANT1"),),
                                         signal="ANT1").z_max}
    m = tl.compute_extent(_event(49.5, 52.5), mz, "mmc", signal="ANT1", tolerances=TOL,
                          fs=FS, z_t0_s=0.0)
    assert m is not None and m.settling_s == 1.5
    assert m.core_start_s - m.start_s == pytest.approx(1.5, abs=1e-9)
    assert m.stop_s - m.core_stop_s == pytest.approx(1.5, abs=1e-9)


def test_settling_provenance_carries_each_measurement() -> None:
    p = tl.settling_provenance(FS)
    assert p["spikes"]["total_s"] == pytest.approx(0.010282, abs=1e-12)
    assert p["mmc"]["total_s"] == 1.5
    assert p["mmc"]["edge"] == tl.edge_settling_record()["mmc"]
    assert "edge" not in p["slow_wave"] and "edge" not in p["hrv"]
    assert p["hr_band_edge"]["total_s"] == 0.2374
    assert p["hr_band_edge"]["edge"] == tl.edge_settling_record()["hr_band"]
    assert p["hr_band_edge"]["pads_no_extent"] is True
    json.dumps(p, allow_nan=False)


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


def test_the_decision_rules_are_declared_and_anything_else_is_refused_by_name() -> None:
    """RULING 2026-10-09 (d) 1: raw P >= 0.5 for I/J/K, calibrated P >= 0.5 for A/B/H."""
    assert set(tl.DECISION_RULES) == {"calibrated_p_ge_0.5", "raw_p_ge_0.5"}
    raw = tl.decision_rule("raw_p_ge_0.5", animal="new:I")
    cal = tl.decision_rule("calibrated_p_ge_0.5", animal="new:A")
    assert (raw.score, raw.threshold, cal.score, cal.threshold) == ("raw", 0.5, "calibrated", 0.5)
    for bad in (None, "", "raw_p_ge_0.3", "calibrated", 0.5):
        with pytest.raises(ValueError, match="new:J"):
            tl.decision_rule(bad, animal="new:J")
    with pytest.raises(ValueError, match="no decision_rule"):
        tl.decision_rule(None, animal="new:K")
    with pytest.raises(ValueError, match=re.escape("unknown decision_rule 'raw_p_ge_0.3'")):
        tl.decision_rule("raw_p_ge_0.3", animal="new:K")
    # the rule decides from ITS score only, at its threshold, inclusive; NaN never confirms
    assert tl.confirms_motion(raw, p_calibrated=0.01, p_raw=0.5)
    assert not tl.confirms_motion(raw, p_calibrated=0.99, p_raw=0.4999)
    assert tl.confirms_motion(cal, p_calibrated=0.5, p_raw=0.01)
    assert not tl.confirms_motion(cal, p_calibrated=0.4999, p_raw=0.99)
    assert not tl.confirms_motion(raw, p_calibrated=0.9, p_raw=float("nan"))
    assert not tl.confirms_motion(cal, p_calibrated=float("nan"), p_raw=0.9)
    # the caveat travels with the raw rule only (RULING 2026-10-09 (d) 1)
    assert "after the I/J/K scores were seen" in raw.provenance()["caveat"]
    assert "caveat" not in cal.provenance()
    assert raw.provenance()["ruling"].startswith("RULING 2026-10-09 (d) 1")
    json.dumps(raw.provenance(), allow_nan=False)


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


def test_expected_consumers_is_the_table_without_what_is_out_of_this_build() -> None:
    assert tl.expected_consumers() == ("breathing", "hrv", "mmc", "slow_wave", "spikes")
    assert "velocity" in tl.extent_consumers()
    assert set(tl.extent_consumers()) - set(tl.expected_consumers()) == {"velocity"}
