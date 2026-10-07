"""Task 19: the headless acceptance report - every computable row on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.acceptance import report as ar
from gems_blanking_v2.detect import chain
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit.handoff import write_mask_file
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.extent.grid import n_grid_frames

FS = 2000.0
DUR_S = 60.0
N_SAMPLES = int(DUR_S * FS)
N_FRAMES = n_grid_frames(N_SAMPLES, FS)
MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/cal.json"}


def test_acceptance_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.acceptance.report" not in chain.generation_modules()


def _all_not_computable() -> list[ar.Row]:
    return [ar.row1_candidate_recall(None, None), ar.row2_class_balance(None),
            ar.row3_no_zeros(None, None), ar.row4_cardiac_scope(None),
            ar.row5_retention(None, None), ar.row6_cross_animal(None),
            ar.row7_coverage_confound(None, has_coverage=True),
            ar.row8_downstream(None, None, None), ar.row9_velocity(),
            ar.row10_mode_comparison(None, None), ar.row11_model_routing(None, None),
            ar.row12_calibration(None)]


def test_a_row_without_its_input_is_not_computable_and_names_it() -> None:
    rows = _all_not_computable()
    for r in rows:
        if r.number == 9:
            assert r.status is ar.Status.NOT_BUILT and "R5" in r.reason
            continue
        assert r.status is ar.Status.NOT_COMPUTABLE, r.number
        assert r.reason.startswith("missing input: ") and r.value is None, r.number
    rep = ar.build_report(rows, ar.Disagreements(), code_commit="c", generation_sha="g")
    assert not rep.gate_passed  # not computable never passes the gate
    doc = json.loads(rep.to_json())
    assert all("value" not in r for r in doc["rows"])  # nothing fabricated
    assert len(doc["spec_contradictions"]) >= 4 and doc["disagreements"]


def test_the_report_needs_every_row_exactly_once() -> None:
    rows = _all_not_computable()
    with pytest.raises(ValueError, match="rows 1-12"):
        ar.build_report(rows[:-1], ar.Disagreements(), code_commit="c", generation_sha="g")


def test_disagreements_are_appended_to() -> None:
    d = ar.Disagreements()
    n = len(d.items)
    d.add("z_enter", "3.0", "2.8 on animal K", "audit round 15", "2026-10-08")
    assert len(d.items) == n + 1 and d.items[-1].constant == "z_enter"
    assert {"ENG band upper corner", "spike consumer settling (task 13)"} <= {
        x.constant for x in d.items}


# row 1 ---------------------------------------------------------------------


def test_row1_counts_only_injections_above_some_tolerance() -> None:
    inj = ([ar.Injection("r", True, True)] * 98 + [ar.Injection("r", False, True)] * 2
           + [ar.Injection("r", False, False)] * 50)
    r = ar.row1_candidate_recall(inj, {"r": 2500})
    assert r.status is ar.Status.PASS and r.value == pytest.approx(0.98)
    assert r.detail["n_below_every_tolerance"] == 50
    assert ar.row1_candidate_recall(inj, {"r": 3001}).status is ar.Status.FAIL
    bad = [ar.Injection("r", True, True)] * 97 + [ar.Injection("r", False, True)] * 3
    assert ar.row1_candidate_recall(bad, {"r": 10}).status is ar.Status.FAIL


def test_row2_class_balance() -> None:
    assert ar.row2_class_balance({"a": (6, 100), "b": (4, 100)}).status is ar.Status.PASS
    assert ar.row2_class_balance({"a": (4, 100)}).status is ar.Status.FAIL


# row 3 ---------------------------------------------------------------------


def test_row3_passes_on_emitted_masks_with_tapered_boundaries() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, N_SAMPLES)
    inv = mk.mask_frames([(10.0, 11.0), (30.0, 30.5)], N_FRAMES, t0_s=0.0)
    y = mk.apply_mask(x, FS, inv)
    r = ar.row3_no_zeros([("spikes/L_T", x, y)], FS)
    assert r.status is ar.Status.PASS and r.detail["boundaries"] == 4 == r.detail["tapered"]


def test_row3_fails_on_a_zero_run_or_an_untapered_boundary() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0, 10, N_SAMPLES)
    hard = x.copy()
    hard[20000:21000] = np.nan  # NaN, but no taper beside it
    r = ar.row3_no_zeros([("hard", x, hard)], FS)
    assert r.status is ar.Status.FAIL and r.detail["tapered"] < r.detail["boundaries"]
    zeros = x.copy()
    zeros[500:600] = 0.0
    assert ar.row3_no_zeros([("zeros", x, zeros)], FS).status is ar.Status.FAIL


# row 4 ---------------------------------------------------------------------


def test_row4_flat_measured_and_blind() -> None:
    flat = [{"channel": "R_T", "band": "300-3000", "status": "no_window"},
            {"channel": "ANT1", "band": "0-2", "status": "no_window"}]
    assert ar.row4_cardiac_scope(flat).status is ar.Status.PASS
    meas = [*flat, {"channel": "L_T", "band": "300-3000", "status": "measured"}]
    assert ar.row4_cardiac_scope(meas).status is ar.Status.FAIL
    blind = [*flat, {"channel": "ANT1", "band": "0.5-3", "status": "unresolvable"}]
    assert ar.row4_cardiac_scope(blind).status is ar.Status.NOT_COMPUTABLE
    old_band = [{"channel": "R_T", "band": "300-5000", "status": "no_window"}]
    assert ar.row4_cardiac_scope(old_band).status is ar.Status.NOT_COMPUTABLE  # R8: not ENG


# row 5 ---------------------------------------------------------------------


def _masks(eng: float, slow: float) -> dict[mk.MaskKey, mk.ConsumerMask]:
    spans = [mk.MaskSpan("spikes", "L_T", 0.0, eng * DUR_S, "x"),
             mk.MaskSpan("slow_wave", "ANT1", 0.0, slow * DUR_S, "x")]
    return mk.build_masks({"spikes": ("L_T",), "slow_wave": ("ANT1",)}, spans,
                          n_frames=N_FRAMES, t0_s=0.0)


def test_row5_eng_retention_above_slow_bands_and_baseline() -> None:
    r = ar.row5_retention({"a": _masks(0.02, 0.2)}, {"a": 0.9})
    assert r.status is ar.Status.PASS and r.value == pytest.approx(0.98)
    assert r.detail["eng_minus_slow"]["0-2"] == pytest.approx(0.18)
    assert ar.row5_retention({"a": _masks(0.2, 0.02)}, {"a": 0.9}).status is ar.Status.FAIL
    assert ar.row5_retention({"a": _masks(0.2, 0.3)}, {"a": 0.9}).status is ar.Status.FAIL


# row 6 ---------------------------------------------------------------------


def test_row6_reports_i_j_k_individually_and_refuses_anything_else() -> None:
    per = {("new", "I"): (40, 42), ("new", "J"): (30, 31), ("new", "K"): (20, 25)}
    r = ar.row6_cross_animal(per)
    assert r.status is ar.Status.PASS and r.value is None  # never pooled into one number
    assert r.detail["recall_by_animal"]["new:K"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="R1"):
        ar.row6_cross_animal({**per, ("old", "J"): (5, 5)})
    with pytest.raises(ValueError, match="R1"):
        ar.row6_cross_animal({("new", "A"): (5, 5)})
    part = ar.row6_cross_animal({("new", "I"): (40, 42)})
    assert part.status is ar.Status.NOT_COMPUTABLE and "new:J" in part.reason


# row 7 ---------------------------------------------------------------------


def _confound_rows(coef: float, n: int = 80, seed: int = 3) -> list[ar.BlankRow]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        cond = "stim_recovery" if i % 2 else "baseline"
        cov = rng.uniform(0.6, 1.0)
        frac = 0.05 + coef * (cond == "stim_recovery") + 0.02 * cov + rng.normal(0, 0.005)
        rows.append(ar.BlankRow(f"r{i}", cond, "pooled", float(frac), float(cov)))
    rows.append(ar.BlankRow("low", "baseline", "pooled", 0.9, 0.3))  # excluded: coverage < 0.5
    return rows


def test_row7_recovers_a_known_condition_coefficient() -> None:
    r = ar.row7_coverage_confound(_confound_rows(0.04), has_coverage=True)
    assert r.status is ar.Status.FAIL  # blanking depends on condition: the confound
    assert r.value == pytest.approx(0.04, abs=0.004)
    lo, hi = r.detail["ci95"]
    assert lo < 0.04 < hi and r.detail["n_excluded_low_coverage"] == 1
    null = ar.row7_coverage_confound(_confound_rows(0.0), has_coverage=True)
    assert null.status is ar.Status.PASS and abs(null.value or 0) < 0.004


def test_row7_refuses_without_coverage() -> None:
    with pytest.raises(ar.CoverageMissingError, match="hasCoverage"):
        ar.row7_coverage_confound(_confound_rows(0.0), has_coverage=False)
    rows = [*_confound_rows(0.0), ar.BlankRow("x", "baseline", "pooled", 0.1, None)]
    with pytest.raises(ar.CoverageMissingError):
        ar.row7_coverage_confound(rows, has_coverage=True)


def test_row7_reports_by_mode_and_carries_mode_as_a_factor() -> None:
    rows = [ar.BlankRow(r.recording, r.condition, "adapted" if i % 3 == 0 else "pooled",
                        r.masked_motion_frac, r.coverage)
            for i, r in enumerate(_confound_rows(0.0))]
    r = ar.row7_coverage_confound(rows, has_coverage=True)
    assert r.detail["mode_is_factor"] and set(r.detail["mean_by_mode"]) == {"adapted", "pooled"}


def test_masked_motion_leaves_out_the_protocol_and_cardiac() -> None:
    spans = [mk.MaskSpan("spikes", "L_T", 0.0, 120.0, "excluded_epoch"),
             mk.MaskSpan("spikes", "L_T", 200.0, 201.0, "cardiac"),
             mk.MaskSpan("spikes", "L_T", 300.0, 306.0, "in_band_eng_sample_sigma"),
             mk.MaskSpan("spikes", "L_T", 305.0, 309.0, "clip"),
             mk.MaskSpan("spikes", "L_T", 400.0, 460.0, "per_minute_cuff_distrust")]
    assert ar.masked_motion_fraction(spans, 600.0) == pytest.approx(9.0 / 600.0)


# row 8 / 9 / 10 ------------------------------------------------------------


def test_row8_all_up_and_variance_down() -> None:
    before = dict.fromkeys(ar.DOWNSTREAM_METRICS, 1.0)
    after = dict.fromkeys(ar.DOWNSTREAM_METRICS, 2.0)
    assert ar.row8_downstream(before, after, (0.5, 0.3)).status is ar.Status.PASS
    assert ar.row8_downstream(before, after, (0.3, 0.5)).status is ar.Status.FAIL
    worse = {**after, "nRR_used": 0.5}
    assert ar.row8_downstream(before, worse, (0.5, 0.3)).status is ar.Status.FAIL


def test_row10_never_a_vs_c_and_r9_rule(tmp_path: Path) -> None:
    plot = tmp_path / "lc.png"
    plot.write_bytes(b"png")
    curves = {(m, a): plot for m in ("A", "B", "C") for a in ("H", "B")}
    assert ar.row10_mode_comparison(curves, {"f1_diff": 0.05, "ci95": [0.01, 0.09]},
                                    comparisons=[("A", "C")]).status is ar.Status.FAIL
    win = ar.row10_mode_comparison(curves, {"f1_diff": 0.05, "ci95": [0.01, 0.09]})
    assert win.detail["verdict"] == "C beats B" and win.detail["task11_investigation_triggered"]
    small = ar.row10_mode_comparison(curves, {"f1_diff": 0.02, "ci95": [0.01, 0.03]})
    assert small.detail["verdict"] == "C does not beat B"
    unsure = ar.row10_mode_comparison(curves, {"f1_diff": 0.04, "ci95": [-0.01, 0.09]})
    assert unsure.detail["verdict"] == "C does not beat B"  # R9: the CI must exclude 0
    gone = ar.row10_mode_comparison({("A", "H"): tmp_path / "missing.png"}, None)
    assert gone.status is ar.Status.NOT_COMPUTABLE


# row 11 --------------------------------------------------------------------


def _write(tmp_path: Path, name: str, model: dict[str, str]) -> Path:
    prov = MaskProvenance(model=model, thresholds={"source": "test"},
                          reference_values={"none": 0}, code_commit="c",
                          generation_sha="0133349b3ebeff80", routing_hash="r",
                          created_at="2026-10-08T05:00:00+00:00", recording=name)
    masks = _masks(0.01, 0.01)
    return write_mask_file(tmp_path / f"{name}.mat", masks, prov, fs=FS, n_samples=N_SAMPLES,
                           epoch_start_s=0.0, min_retention=0.5,
                           animal_median={"spikes|L_T|300-3000": 0.02,
                                          "slow_wave|ANT1|0-2": 0.02})


def test_row11_every_mask_names_the_logged_model_and_reruns_reapply_it(tmp_path: Path) -> None:
    files = {"r1": _write(tmp_path, "r1", MODEL), "r2": _write(tmp_path, "r2", MODEL)}
    log = {"r1": MODEL, "r2": MODEL}
    (tmp_path / "again").mkdir()
    rerun = {"r1": _write(tmp_path / "again", "r1", MODEL)}
    ok = ar.row11_model_routing(files, log, rerun)
    assert ok.status is ar.Status.PASS and ok.value == 0.0
    other = {**MODEL, "version": "0.4.0"}
    assert ar.row11_model_routing(files, {"r1": other, "r2": MODEL}).status is ar.Status.FAIL
    assert ar.row11_model_routing(files, {"r1": MODEL}).status is ar.Status.FAIL  # r2 unlogged
    bad_rerun = {"r1": _write(tmp_path / "again", "r1x", other)}
    assert ar.row11_model_routing(files, log, bad_rerun).status is ar.Status.FAIL


# row 12 --------------------------------------------------------------------


def test_row12_ece_on_held_out_data() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 20000)
    calibrated = (rng.uniform(0, 1, p.size) < p).astype(float)
    overconfident = (rng.uniform(0, 1, p.size) < 0.5).astype(float)
    ok = ar.row12_calibration({"m1": (p, calibrated)})
    assert ok.status is ar.Status.PASS and (ok.value or 1) < 0.02
    bad = ar.row12_calibration({"m1": (p, calibrated), "m2": (p, overconfident)})
    assert bad.status is ar.Status.FAIL and bad.value == pytest.approx(0.25, abs=0.02)
    ece, curve = ar.expected_calibration_error([0.9, 0.9], [1.0, 0.0])
    assert ece == pytest.approx(0.4) and curve[0]["n"] == 2


def test_a_full_synthetic_report_round_trips_to_json(tmp_path: Path) -> None:
    rows = _all_not_computable()
    rows[0] = ar.row1_candidate_recall([ar.Injection("r", True, True)] * 100, {"r": 10})
    rep = ar.build_report(rows, ar.Disagreements(), code_commit="c", generation_sha="g")
    doc = json.loads(rep.to_json())
    assert doc["rows"][0]["status"] == "pass" and doc["rows"][0]["value"] == 1.0
    assert "| 1 | candidate_recall | pass |" in rep.to_markdown()
