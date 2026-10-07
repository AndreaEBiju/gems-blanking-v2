"""Task 19: the headless acceptance report - every computable row on synthetic data."""

from __future__ import annotations

import dataclasses
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
from gems_blanking_v2.extent.tolerance import expected_consumers
from gems_blanking_v2.io.registry_log import RegistryAction, RegistryEvent
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import make_confound_rows, make_eng

FS = 2000.0
DUR_S = 60.0
N_SAMPLES = int(DUR_S * FS)
N_FRAMES = n_grid_frames(N_SAMPLES, FS)
MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/cal.json"}
IJK = {("new", "I"): [(20, 21), (20, 21)], ("new", "J"): [(15, 16), (15, 15)],
       ("new", "K"): [(10, 12), (10, 13)]}
READS: dict[str, tuple[str, ...]] = {"spikes": ("L_T",), "slow_wave": ("ANT1",),
                                     "mmc": ("ANT1",), "hrv": ("RVN2",),
                                     "breathing": ("RVN2",), "velocity": ()}
CONSUMERS = list(expected_consumers())
"""Expected consumers: the tolerance table's, velocity excepted (R5) - the one helper."""
TRAIN = [("new", "A"), ("new", "B"), ("new", "H"), ("old", "J")]


def test_acceptance_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.acceptance.report" not in chain.generation_modules()


def _all_not_computable() -> list[ar.Row]:
    return [ar.row1_candidate_recall(None, None, consumers=CONSUMERS),
            ar.row2_class_balance(None),
            ar.row3_no_zeros(None, None, None), ar.row4_cardiac_scope(None),
            ar.row5_retention(None, None), ar.row6_cross_animal(None, scoring_corpus=None),
            ar.row7_coverage_confound(None, has_coverage=True, equivalence_margin=None,
                                      consumers=CONSUMERS),
            ar.row8_downstream(None, None, None), ar.row9_velocity(),
            ar.row10_mode_comparison(None, None, comparisons=None, animals=["H"]),
            ar.row11_model_routing(None, None, None, None), ar.row12_calibration(None, None)]


def _promoted(*models: str, demoted: tuple[str, ...] = ()) -> list[RegistryEvent]:
    ev = []
    for i, m in enumerate(models):
        ev.append(RegistryEvent(f"2026-10-0{i + 1}T00:00:00+00:00", "u", RegistryAction.TRAINED, m))
        ev.append(RegistryEvent(f"2026-10-0{i + 1}T01:00:00+00:00", "u", RegistryAction.PROMOTED,
                                m))
    for m in demoted:
        ev.append(RegistryEvent("2026-10-09T00:00:00+00:00", "u", RegistryAction.DEMOTED, m))
    return ev


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
    assert len(doc["spec_contradictions"]) >= 8 and doc["disagreements"]


def test_the_report_needs_every_row_exactly_once() -> None:
    with pytest.raises(ValueError, match="rows 1-12"):
        ar.build_report(_all_not_computable()[:-1], ar.Disagreements(), code_commit="c",
                        generation_sha="g")


def test_disagreements_are_seeded_from_the_spec_and_appended_to() -> None:
    d = ar.Disagreements()
    names = {x.constant for x in d.items}
    assert {"ENG band upper corner", "R-peak threshold", "HR detection band (hrBandHz)",
            "reference statistic", "MMC cycle length", "cost model"} <= names
    tri = next(x for x in d.items if x.constant == "tripole sigma reduction")
    assert "not a constant" in tri.measured
    n = len(d.items)
    d.add("z_enter", "3.0", "2.8 on animal K", "audit round 15", "2026-10-08")
    assert len(d.items) == n + 1 and d.items[-1].constant == "z_enter"


# row 1 ---------------------------------------------------------------------


def _inj(det: bool, *consumers: str, rec: str = "r") -> ar.Injection:
    return ar.Injection(rec, det, frozenset(consumers))


def test_row1_is_per_consumer() -> None:
    inj = ([_inj(True, "spikes", "hrv")] * 98 + [_inj(False, "spikes", "hrv")] * 2
           + [_inj(False)] * 50)
    two = ["spikes", "hrv"]
    r = ar.row1_candidate_recall(inj, {"r": 2500}, consumers=two)
    assert r.status is ar.Status.PASS and r.value == pytest.approx(0.98)
    assert r.detail["n_below_every_tolerance"] == 50
    # spikes fine, slow_wave not: one consumer under 98% fails the row
    bad = inj + [_inj(False, "slow_wave")] * 5 + [_inj(True, "slow_wave")] * 95
    rb = ar.row1_candidate_recall(bad, {"r": 10}, consumers=[*two, "slow_wave"])
    assert rb.status is ar.Status.FAIL and rb.detail["recall_by_consumer"]["slow_wave"] == 0.95
    assert ar.row1_candidate_recall(inj, {"r": 3001}, consumers=two).status is ar.Status.FAIL


def test_row1_needs_a_candidate_count_for_every_injected_recording() -> None:
    inj = [_inj(True, "spikes", rec="r1"), _inj(True, "spikes", rec="r2")]
    r = ar.row1_candidate_recall(inj, {"r1": 10}, consumers=['spikes'])
    assert r.status is ar.Status.NOT_COMPUTABLE and "r2" in r.reason


def test_row2_class_balance() -> None:
    assert ar.row2_class_balance({"a": (6, 100), "b": (4, 100)}).status is ar.Status.PASS
    assert ar.row2_class_balance({"a": (4, 100)}).status is ar.Status.FAIL


# row 3 ---------------------------------------------------------------------


def test_row3_zeros_on_matlab_side_taper_on_python_side() -> None:
    x = make_eng(FS, DUR_S, seed=1).signal
    inv = mk.mask_frames([(10.0, 11.0), (30.0, 30.5)], N_FRAMES, t0_s=0.0)
    y = mk.apply_mask(x, FS, inv)
    yout = x.copy()
    yout[mk.frames_to_samples(inv, FS, N_SAMPLES)] = np.nan  # what MATLAB gets: no taper
    r = ar.row3_no_zeros([("spikes/L_T", yout)], [("spikes/L_T", x, y)], FS)
    assert r.status is ar.Status.PASS and r.detail["python_boundaries"] == 4
    zeroed = yout.copy()
    zeroed[500:600] = 0.0
    assert ar.row3_no_zeros([("z", zeroed)], [("p", x, y)], FS).status is ar.Status.FAIL
    assert ar.row3_no_zeros([("y", yout)], [("hard", x, yout)], FS).status is ar.Status.FAIL
    assert ar.row3_no_zeros([("y", yout)], None, FS).status is ar.Status.NOT_COMPUTABLE


def test_row3_ignores_boundaries_beside_pre_existing_nan() -> None:
    x = make_eng(FS, DUR_S, seed=2).signal.copy()
    x[19000:21000] = np.nan  # a dropout in the original
    inv = mk.mask_frames([(10.5, 11.0)], N_FRAMES, t0_s=0.0)
    y = mk.apply_mask(x, FS, inv)
    r = ar.row3_no_zeros([("y", y)], [("p", x, y)], FS)
    assert r.status is ar.Status.PASS


# row 4 ---------------------------------------------------------------------


def _v(band: str, status: str = "no_window") -> dict[str, str]:
    return {"channel": "R_T", "band": band, "status": status}


def test_row4_needs_every_band_and_cannot_pass_blind() -> None:
    flat = [_v("300-3000"), _v("0.5-3"), _v("0-2")]
    assert ar.row4_cardiac_scope(flat).status is ar.Status.PASS
    only_eng = ar.row4_cardiac_scope([_v("300-3000")])
    assert only_eng.status is ar.Status.NOT_COMPUTABLE and "0.5-3" in only_eng.reason
    assert ar.row4_cardiac_scope([*flat, _v("300-3000", "measured")]).status is ar.Status.FAIL
    blind = [_v("300-3000"), _v("0.5-3", "unresolvable"), _v("0-2", "unresolvable")]
    assert ar.row4_cardiac_scope(blind).status is ar.Status.NOT_COMPUTABLE
    assert ar.row4_cardiac_scope([_v("300-5000")]).status is ar.Status.NOT_COMPUTABLE  # R8


# row 5 ---------------------------------------------------------------------


def _masks(eng: float, slow: float, slow2: float | None = None
           ) -> dict[mk.MaskKey, mk.ConsumerMask]:
    spans = [mk.MaskSpan("spikes", "L_T", 0.0, eng * DUR_S, "x"),
             mk.MaskSpan("slow_wave", "ANT1", 0.0, slow * DUR_S, "x"),
             mk.MaskSpan("breathing", "RVN2", 0.0, (slow if slow2 is None else slow2) * DUR_S,
                         "x")]
    return mk.build_masks(READS, spans, n_frames=N_FRAMES, t0_s=0.0)


def test_row5_eng_retention_above_slow_bands_and_baseline() -> None:
    r = ar.row5_retention({"a": _masks(0.02, 0.2)}, {"a": 0.9})
    assert r.status is ar.Status.PASS and r.value == pytest.approx(0.98)
    assert r.detail["eng_minus_slow"]["0-2"] == pytest.approx(0.18)
    assert ar.row5_retention({"a": _masks(0.2, 0.02)}, {"a": 0.9}).status is ar.Status.FAIL
    assert ar.row5_retention({"a": _masks(0.2, 0.3)}, {"a": 0.9}).status is ar.Status.FAIL


def test_row5_without_a_slow_band_or_baseline_overlap_is_not_computable() -> None:
    eng_only = {k: v for k, v in _masks(0.02, 0.2).items() if k[0] == "spikes"}
    r = ar.row5_retention({"a": eng_only}, {"a": 0.9})
    assert r.status is ar.Status.NOT_COMPUTABLE and "slow-band" in r.reason
    r2 = ar.row5_retention({"a": _masks(0.02, 0.2)}, {"other": 0.9})
    assert r2.status is ar.Status.NOT_COMPUTABLE and "baseline" in r2.reason
    assert r2.detail["masks_without_baseline"] == ["a"]


def test_row5_names_the_mask_recordings_it_could_not_compare() -> None:
    r = ar.row5_retention({"a": _masks(0.02, 0.2), "b": _masks(0.5, 0.5)}, {"a": 0.9})
    assert r.status is ar.Status.PASS and r.detail["n_baseline"] == 1
    assert r.detail["masks_without_baseline"] == ["b"]


# row 6 ---------------------------------------------------------------------


def test_row6_reports_i_j_k_individually_with_span_bootstrap_cis() -> None:
    r = ar.row6_cross_animal(IJK, scoring_corpus=TRAIN)
    assert r.status is ar.Status.PASS and r.value is None  # never pooled
    k = r.detail["new:K"]
    assert k["recall"] == pytest.approx(20 / 25) and k["ci95"][0] <= 0.8 <= k["ci95"][1]
    assert r.detail == ar.row6_cross_animal(IJK, scoring_corpus=TRAIN).detail  # seeded


def test_row6_refuses_non_test_animals_and_a_leaky_scoring_model() -> None:
    with pytest.raises(ValueError, match="R1"):
        ar.row6_cross_animal({**IJK, ("old", "J"): [(5, 5)]}, scoring_corpus=TRAIN)
    with pytest.raises(ValueError, match="trained on test animals"):
        ar.row6_cross_animal(IJK, scoring_corpus=[*TRAIN, ("new", "J")])
    part = ar.row6_cross_animal({("new", "I"): [(40, 42)]}, scoring_corpus=TRAIN)
    assert part.status is ar.Status.NOT_COMPUTABLE and "new:J" in part.reason


def test_row6_refuses_two_keys_that_merge_once_case_is_normalised() -> None:
    with pytest.raises(ValueError, match="new:I twice"):
        ar.row6_cross_animal({**IJK, ("new", "i"): [(1, 1), (1, 1)]}, scoring_corpus=TRAIN)
    with pytest.raises(ValueError, match="new:I twice"):
        ar.row6_cross_animal({**IJK, ("NEW", "I"): [(1, 1), (1, 1)]}, scoring_corpus=TRAIN)


# row 7 ---------------------------------------------------------------------


def test_row7_recovers_a_known_coefficient_only_with_its_covariates() -> None:
    rows = make_confound_rows(0.04)
    r = ar.row7_coverage_confound(rows, has_coverage=True, equivalence_margin=None,
                                  consumers=['spikes'])
    d = r.detail["spikes"]
    assert r.status is ar.Status.FAIL  # blanking depends on condition: the confound
    assert d["coef"] == pytest.approx(0.04, abs=0.003) and d["mode_is_factor"]
    assert d["n_excluded_low_coverage"] == 1
    kept = [x for x in rows if x.recording != "low"]
    x = np.column_stack([np.ones(len(kept)),
                         [float(r.condition == "stim_recovery") for r in kept]])
    naive = np.linalg.lstsq(x, np.array([r.fraction for r in kept]), rcond=None)[0][1]
    assert abs(naive - 0.04) > 0.01  # condition alone, without its covariates: biased


def test_row7_never_passes_without_a_ruled_margin() -> None:
    null = ar.row7_coverage_confound(make_confound_rows(0.0), has_coverage=True,
                                     equivalence_margin=None, consumers=['spikes'])
    assert null.status is ar.Status.NOT_COMPUTABLE and "margin not ruled" in null.reason
    assert null.detail["spikes"]["ci95"][0] <= 0 <= null.detail["spikes"]["ci95"][1]
    ruled = ar.row7_coverage_confound(make_confound_rows(0.0), has_coverage=True,
                                      equivalence_margin=0.01, consumers=['spikes'])
    assert ruled.status is ar.Status.PASS


def test_row7_runs_per_consumer() -> None:
    rows = (make_confound_rows(0.0, consumer="spikes")
            + make_confound_rows(0.05, consumer="slow_wave", seed=4))
    r = ar.row7_coverage_confound(rows, has_coverage=True, equivalence_margin=0.01,
                                  consumers=['spikes', 'slow_wave'])
    assert r.status is ar.Status.FAIL and set(r.detail) == {"spikes", "slow_wave"}
    assert r.detail["slow_wave"]["coef"] == pytest.approx(0.05, abs=0.003)


def test_row7_reports_the_failing_consumer_not_the_largest_coefficient() -> None:
    """A noisy consumer with a larger |coef| whose CI spans 0 must not be the value."""
    rng = np.random.default_rng(11)
    noisy = [dataclasses.replace(r, motion_blank_s=r.motion_blank_s
                                 + rng.normal(0, 0.4) * (r.duration_s - r.excluded_s))
             for r in make_confound_rows(0.0, consumer="slow_wave", n=24, seed=6)]
    rows = make_confound_rows(0.01, consumer="spikes") + noisy
    r = ar.row7_coverage_confound(rows, has_coverage=True, equivalence_margin=None,
                                  consumers=["spikes", "slow_wave"])
    fail, quiet = r.detail["spikes"], r.detail["slow_wave"]
    assert quiet["ci95"][0] < 0 < quiet["ci95"][1]  # the noisy one does not fail
    assert abs(quiet["coef"]) > abs(fail["coef"])  # but has the larger coefficient
    assert r.status is ar.Status.FAIL and r.value == fail["coef"]


def test_rows_1_and_7_default_to_the_expected_consumers() -> None:
    r1 = ar.row1_candidate_recall([_inj(True, "spikes")] * 100, {"r": 10})
    assert r1.status is ar.Status.NOT_COMPUTABLE
    assert all(c in r1.reason for c in CONSUMERS if c != "spikes")
    r7 = ar.row7_coverage_confound(make_confound_rows(0.0), has_coverage=True,
                                   equivalence_margin=0.01)
    assert r7.status is ar.Status.NOT_COMPUTABLE
    assert all(c in r7.reason for c in CONSUMERS if c != "spikes")


def test_row7_refuses_without_coverage() -> None:
    with pytest.raises(ar.CoverageMissingError, match="hasCoverage"):
        ar.row7_coverage_confound(make_confound_rows(0.0), has_coverage=False,
                                  equivalence_margin=None, consumers=['spikes'])
    rows = [*make_confound_rows(0.0), ar.BlankRow("x", "spikes", "baseline", "pooled", 1.0, 100.0,
                                              0.0, None)]
    with pytest.raises(ar.CoverageMissingError):
        ar.row7_coverage_confound(rows, has_coverage=True, equivalence_margin=None,
                                  consumers=['spikes'])


def test_row7_names_confounded_terms_and_leaves_p_absent_when_exact() -> None:
    rows = make_confound_rows(0.0, modes=False)
    tied = [ar.BlankRow(x.recording, x.consumer, x.condition,
                        "adapted" if x.condition == "stim_recovery" else "pooled",
                        x.motion_blank_s, x.duration_s, x.excluded_s, x.coverage) for x in rows]
    r = ar.row7_coverage_confound(tied, has_coverage=True, equivalence_margin=None,
                                  consumers=['spikes'])
    assert r.status is ar.Status.NOT_COMPUTABLE and "confounded terms" in r.reason
    few = make_confound_rows(0.0, n=8)
    assert "dof" in ar.row7_coverage_confound(few, has_coverage=True,
                                              equivalence_margin=0.01, consumers=['spikes']).reason
    exact = [ar.BlankRow(f"r{i}", "spikes", "stim_recovery" if i % 2 else "baseline", "pooled",
                         60.0, 1200.0, 0.0, 0.6 + 0.01 * i) for i in range(30)]
    d = ar.row7_coverage_confound(exact, has_coverage=True, equivalence_margin=0.01,
                                  consumers=['spikes'])
    assert "p" not in d.detail["spikes"] or d.detail["spikes"]["se"] > 0


def test_masked_motion_uses_an_allow_list_and_the_time_at_risk() -> None:
    spans = [mk.MaskSpan("spikes", "L_T", 0.0, 120.0, "excluded_epoch"),
             mk.MaskSpan("spikes", "L_T", 200.0, 201.0, "cardiac"),
             mk.MaskSpan("spikes", "L_T", 300.0, 306.0, "in_band_eng_sample_sigma"),
             mk.MaskSpan("spikes", "L_T", 305.0, 309.0, "clip"),
             mk.MaskSpan("spikes", "L_T", 400.0, 460.0, "per_minute_cuff_distrust"),
             mk.MaskSpan("spikes", "L_T", 590.0, 700.0, "beat_train_changed")]
    assert ar.masked_motion_seconds(spans, 600.0) == pytest.approx(9.0 + 10.0)
    with pytest.raises(ValueError, match="unknown mask reason"):
        ar.masked_motion_seconds([mk.MaskSpan("spikes", "L_T", 0, 1, "x")], 600.0)
    with pytest.raises(ValueError, match="positive"):
        ar.masked_motion_seconds([], 0.0)
    row = ar.BlankRow("r", "spikes", "stim_recovery", "pooled", 54.0, 1200.0, 120.0, 0.9)
    assert row.fraction == pytest.approx(54.0 / 1080.0)


# row 8 / 10 ----------------------------------------------------------------


def test_row8_all_up_and_variance_down() -> None:
    before = dict.fromkeys(ar.DOWNSTREAM_METRICS, 1.0)
    after = dict.fromkeys(ar.DOWNSTREAM_METRICS, 2.0)
    assert ar.row8_downstream(before, after, (0.5, 0.3)).status is ar.Status.PASS
    assert ar.row8_downstream(before, after, (0.3, 0.5)).status is ar.Status.FAIL
    assert ar.row8_downstream(before, {**after, "nRR_used": 0.5}, (0.5, 0.3)).status \
        is ar.Status.FAIL


def _bvc(diff: float, lo: float, hi: float, **kw: object) -> dict[str, object]:
    return {"f1_diff": diff, "ci95": [lo, hi], "n_resamples": 1000, "cluster_unit": "recording",
            "excluded_folds": ["old:L"], "corpus": "old_cohort_loao", "cohort": "old",
            "matched_event_control": {"f1_diff": diff, "ci95": [lo, hi]}, **kw}


@pytest.fixture
def curves(tmp_path: Path) -> dict[tuple[str, str], Path]:
    plot = tmp_path / "lc.png"
    plot.write_bytes(b"png")
    return {(m, a): plot for m in ("pooled", "adapted", "per_animal") for a in ("H", "B")}


def test_row10_catches_a_vs_c_however_named(curves: dict[tuple[str, str], Path]) -> None:
    ok = [ar.Comparison("B", "C", "reported"), ar.Comparison("A", "C", "not_comparable")]
    assert ar.row10_mode_comparison(curves, _bvc(0.01, -0.02, 0.04), comparisons=ok,
                                    animals=["H", "B"]).status is ar.Status.PASS
    for a, b in (("A", "C"), ("pooled", "per_animal"), ("PER_ANIMAL", "pooled")):
        bad = [ar.Comparison(a, b, "reported")]
        assert ar.row10_mode_comparison(curves, _bvc(0.05, 0.01, 0.09), comparisons=bad,
                                        animals=["H", "B"]).status is ar.Status.FAIL


def test_row10_r9_rule_at_its_boundaries(curves: dict[tuple[str, str], Path]) -> None:
    def verdict(diff: float, lo: float) -> tuple[str, bool]:
        r = ar.row10_mode_comparison(curves, _bvc(diff, lo, 0.1), comparisons=[],
                                     animals=["H", "B"])
        assert (r.status is ar.Status.FAIL) == r.detail["task11_investigation_triggered"]
        return r.detail["verdict"], r.detail["task11_investigation_triggered"]

    assert verdict(0.05, 0.01) == ("C beats B", True)
    assert verdict(0.03, 0.001) == ("C beats B", True)  # exactly the margin wins
    assert verdict(0.029, 0.01) == ("C does not beat B", False)
    assert verdict(0.05, 0.0) == ("C does not beat B", False)  # CI touching 0 does not
    assert verdict(0.04, -0.01) == ("C does not beat B", False)


def test_row10_requires_the_full_record(curves: dict[tuple[str, str], Path]) -> None:
    r = ar.row10_mode_comparison(curves, _bvc(0.05, 0.01, 0.09), comparisons=None,
                                 animals=["H"])
    assert r.status is ar.Status.NOT_COMPUTABLE and "comparisons" in r.reason
    some = {k: v for k, v in curves.items() if k[0] != "adapted"}
    r = ar.row10_mode_comparison(some, _bvc(0.05, 0.01, 0.09), comparisons=[],
                                 animals=["H", "B"])
    assert r.status is ar.Status.NOT_COMPUTABLE and "adapted/H" in r.reason
    partial = {k: v for k, v in _bvc(0.05, 0.01, 0.09).items() if k != "excluded_folds"}
    r = ar.row10_mode_comparison(curves, partial, comparisons=[], animals=["H", "B"])
    assert r.status is ar.Status.NOT_COMPUTABLE and "excluded_folds" in r.reason
    with pytest.raises(ValueError, match="1000 resamples"):
        ar.row10_mode_comparison(curves, _bvc(0.05, 0.01, 0.09, n_resamples=200),
                                 comparisons=[], animals=["H", "B"])
    with pytest.raises(ValueError, match="R1"):
        ar.row10_mode_comparison(curves, _bvc(0.05, 0.01, 0.09, corpus="new_cohort_loao"),
                                 comparisons=[], animals=["H", "B"])


def test_row10_cohort_must_agree_with_the_corpus(curves: dict[tuple[str, str], Path]) -> None:
    with pytest.raises(ValueError, match="cannot come from cohort 'new'"):
        ar.row10_mode_comparison(curves, _bvc(0.01, -0.02, 0.04, cohort="new",
                                              cluster_unit="span"),
                                 comparisons=[], animals=["H", "B"])
    ok = _bvc(0.01, -0.02, 0.04, cohort="new", cluster_unit="span", corpus="within_animal")
    assert ar.row10_mode_comparison(curves, ok, comparisons=[],
                                    animals=["H", "B"]).status is ar.Status.PASS


@pytest.mark.parametrize("where", ["main", "control"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_row10_refuses_non_finite_differences(curves: dict[tuple[str, str], Path], where: str,
                                              bad: float) -> None:
    for field in ("f1_diff", "lo", "hi"):
        rec = _bvc(0.01, -0.02, 0.04)
        target = rec if where == "main" else rec["matched_event_control"]
        assert isinstance(target, dict)
        if field == "f1_diff":
            target["f1_diff"] = bad
        else:
            target["ci95"] = [bad, 0.04] if field == "lo" else [-0.02, bad]
        with pytest.raises(ValueError, match="finite"):
            ar.row10_mode_comparison(curves, rec, comparisons=[], animals=["H", "B"])
    rec = _bvc(0.01, -0.02, 0.04)
    rec["matched_event_control"] = {"f1_diff": 0.01, "ci95": [0.1]}
    with pytest.raises(ValueError, match="two numbers"):
        ar.row10_mode_comparison(curves, rec, comparisons=[], animals=["H", "B"])


# row 11 --------------------------------------------------------------------


def _write(folder: Path, name: str, model: dict[str, str], routing: str = "64c2e1ea") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    prov = MaskProvenance(model=model, thresholds={"source": "test"},
                          reference_values={"none": 0}, code_commit="c",
                          generation_sha="0133349b3ebeff80", routing_hash=routing,
                          created_at="2026-10-08T05:00:00+00:00", recording=name)
    masks = _masks(0.01, 0.01)
    return write_mask_file(folder / f"{name}.mat", masks, prov,
                           signals=READS,
                           fs=FS, n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={})


def test_row11_full_modelspec_routing_log_and_reruns(tmp_path: Path) -> None:
    files = {r: _write(tmp_path / "run1", r, MODEL) for r in ("r1", "r2")}
    again = {r: _write(tmp_path / "run2", r, MODEL) for r in ("r1", "r2")}
    log = {"r1": MODEL, "r2": MODEL}
    rlog = {"r1": "64c2e1ea", "r2": "64c2e1ea"}
    assert ar.row11_model_routing(files, log, rlog, again).status is ar.Status.PASS
    other_cal = {**MODEL, "calibrator": "models/y/cal.json"}
    assert ar.row11_model_routing(files, {**log, "r1": other_cal}, rlog, again).status \
        is ar.Status.FAIL
    assert ar.row11_model_routing(files, log, {**rlog, "r2": "deadbeef"}, again).status \
        is ar.Status.FAIL
    missing = ar.row11_model_routing(files, log, rlog, {"r1": again["r1"]})
    assert missing.status is ar.Status.NOT_COMPUTABLE and "r2" in missing.reason
    assert ar.row11_model_routing(files, log, rlog, None).status is ar.Status.NOT_COMPUTABLE
    drift = {"r1": _write(tmp_path / "run3", "r1", {**MODEL, "version": "0.4.0"}),
             "r2": again["r2"]}
    assert ar.row11_model_routing(files, log, rlog, drift).status is ar.Status.FAIL


# row 12 --------------------------------------------------------------------


def test_row12_every_shipped_model_needs_held_out_data() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 20000)
    calibrated = (rng.uniform(0, 1, p.size) < p).astype(float)
    overconfident = (rng.uniform(0, 1, p.size) < 0.5).astype(float)
    ok = ar.row12_calibration(_promoted("m1"), {"m1": (p, calibrated)})
    assert ok.status is ar.Status.PASS and (ok.value or 1) < 0.02
    bad = ar.row12_calibration(_promoted("m1", "m2"),
                               {"m1": (p, calibrated), "m2": (p, overconfident)})
    assert bad.status is ar.Status.FAIL and bad.value == pytest.approx(0.25, abs=0.02)
    gap = ar.row12_calibration(_promoted("m1", "m3"), {"m1": (p, calibrated)})
    assert gap.status is ar.Status.NOT_COMPUTABLE and "m3" in gap.reason
    ece, curve = ar.expected_calibration_error([0.9, 0.9], [1.0, 0.0])
    assert ece == pytest.approx(0.4) and curve[0]["n"] == 2


def test_a_full_synthetic_report_round_trips_to_json() -> None:
    rows = _all_not_computable()
    rows[0] = ar.row1_candidate_recall([_inj(True, "spikes")] * 100, {"r": 10},
                                       consumers=["spikes"])
    rep = ar.build_report(rows, ar.Disagreements(), code_commit="c", generation_sha="g")
    doc = json.loads(rep.to_json())
    assert doc["rows"][0]["status"] == "pass" and doc["rows"][0]["value"] == 1.0
    assert "| 1 | candidate_recall | pass |" in rep.to_markdown()


# ---------------------------------------------------------------------------
# re-review holes, each probed and closed
# ---------------------------------------------------------------------------


def test_row1_and_row7_name_expected_consumers_without_data() -> None:
    r1 = ar.row1_candidate_recall([_inj(True, "spikes")] * 100, {"r": 10}, consumers=CONSUMERS)
    assert r1.status is ar.Status.NOT_COMPUTABLE and "hrv" in r1.reason
    r7 = ar.row7_coverage_confound(make_confound_rows(0.0), has_coverage=True,
                                   equivalence_margin=0.01, consumers=CONSUMERS)
    assert r7.status is ar.Status.NOT_COMPUTABLE and "blank fractions for" in r7.reason
    assert "mmc" in r7.reason


def test_row2_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="0 <= motion <= judged"):
        ar.row2_class_balance({"a": (11, 10)})
    with pytest.raises(ValueError, match="0 <= motion <= judged"):
        ar.row2_class_balance({"a": (-1, 10)})


def test_row3_without_any_boundary_is_not_computable() -> None:
    x = make_eng(FS, DUR_S, seed=3).signal
    r = ar.row3_no_zeros([("y", x)], [("p", x, x)], FS)
    assert r.status is ar.Status.NOT_COMPUTABLE


def test_row4_an_unknown_status_raises() -> None:
    with pytest.raises(ValueError, match="not one of"):
        ar.row4_cardiac_scope([_v("300-3000", "error"), _v("0.5-3"), _v("0-2")])


def test_row5_uses_the_same_recordings_for_eng_and_baseline() -> None:
    masks = {"a": _masks(0.30, 0.40), "b": _masks(0.01, 0.40)}
    r = ar.row5_retention(masks, {"a": 0.75})  # only "a" has a baseline
    assert r.value == pytest.approx(0.70) and r.status is ar.Status.FAIL  # not b's 0.99


def test_row6_needs_two_spans_a_corpus_and_ignores_letter_case() -> None:
    one = {**IJK, ("new", "K"): [(10, 12)]}
    r = ar.row6_cross_animal(one, scoring_corpus=TRAIN)
    assert r.status is ar.Status.NOT_COMPUTABLE and "new:K" in r.reason
    assert ar.row6_cross_animal(IJK, scoring_corpus=[]).status is ar.Status.NOT_COMPUTABLE
    with pytest.raises(ValueError, match="trained on test animals"):
        ar.row6_cross_animal(IJK, scoring_corpus=[*TRAIN, ("NEW", "j")])
    lower = {("new", a.lower()): v for (_c, a), v in IJK.items()}
    assert ar.row6_cross_animal(lower, scoring_corpus=TRAIN).status is ar.Status.PASS


def test_motion_inside_the_excluded_epoch_is_not_counted() -> None:
    spans = [mk.MaskSpan("spikes", "L_T", 0.0, 120.0, "excluded_epoch"),
             mk.MaskSpan("spikes", "L_T", 100.0, 130.0, "clip")]
    assert ar.masked_motion_seconds(spans, 600.0) == pytest.approx(10.0)


def test_row10_needs_animals_cluster_unit_by_cohort_and_a_matched_control(
    curves: dict[tuple[str, str], Path]
) -> None:
    good = _bvc(0.01, -0.02, 0.04)
    assert ar.row10_mode_comparison(curves, good, comparisons=[],
                                    animals=[]).status is ar.Status.NOT_COMPUTABLE
    with pytest.raises(ValueError, match="clusters by"):
        ar.row10_mode_comparison(curves, {**good, "cluster_unit": "span"}, comparisons=[],
                                 animals=["H", "B"])
    with pytest.raises(ValueError, match="clusters by"):
        ar.row10_mode_comparison(curves, {**good, "cohort": "new", "corpus": "within_animal"},
                                 comparisons=[], animals=["H", "B"])
    no_ctrl = ar.row10_mode_comparison(curves, {**good, "matched_event_control": None},
                                       comparisons=[], animals=["H", "B"])
    assert no_ctrl.status is ar.Status.NOT_COMPUTABLE and "invariant 13" in no_ctrl.reason
    c_wins = ar.row10_mode_comparison(curves, _bvc(0.05, 0.01, 0.09), comparisons=[],
                                      animals=["H", "B"])
    assert c_wins.status is ar.Status.FAIL and "task 11" in c_wins.reason


def test_row11_checks_the_reruns_routing_hash_and_recording(tmp_path: Path) -> None:
    files = {r: _write(tmp_path / "run1", r, MODEL) for r in ("r1", "r2")}
    log = {"r1": MODEL, "r2": MODEL}
    rlog = {"r1": "64c2e1ea", "r2": "64c2e1ea"}
    stray = {"r1": _write(tmp_path / "run2", "r9", MODEL),
             "r2": _write(tmp_path / "run2", "r2", MODEL)}
    r = ar.row11_model_routing(files, log, rlog, stray)
    assert r.status is ar.Status.FAIL and "r9" in r.detail["problems"]["r1"]
    other_routing = {"r1": _write(tmp_path / "run3", "r1", MODEL, routing="deadbeef"),
                     "r2": stray["r2"]}
    r = ar.row11_model_routing(files, log, rlog, other_routing)
    assert r.status is ar.Status.FAIL and "routing" in r.detail["problems"]["r1"]
    r = ar.row11_model_routing(files, log, {"r2": "64c2e1ea"}, stray)
    assert "no routing-table hash logged" in r.detail["problems"]["r1"]


def test_row12_needs_promotion_and_enough_held_out_data() -> None:
    p = np.full(500, 0.5)
    y = np.tile([0.0, 1.0], 250)
    gone = ar.row12_calibration(_promoted("m1", demoted=("m1",)), {"m1": (p, y)})
    assert gone.status is ar.Status.NOT_COMPUTABLE and "promoted" in gone.reason
    tiny = ar.row12_calibration(_promoted("m1"), {"m1": ([0.9, 0.1], [1.0, 0.0])})
    assert tiny.status is ar.Status.NOT_COMPUTABLE and "m1" in tiny.reason
    short_y = ar.row12_calibration(_promoted("m1"), {"m1": (p, y[:2])})
    assert short_y.status is ar.Status.NOT_COMPUTABLE and "m1" in short_y.reason


@settings(max_examples=40, deadline=None)
@given(numbers=st.lists(st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
                        min_size=12, max_size=12),
       statuses=st.lists(st.sampled_from(list(ar.Status)), min_size=12, max_size=12))
def test_the_report_json_round_trips_without_nan(numbers: list[float | None],
                                                 statuses: list[ar.Status]) -> None:
    rows = [ar.Row(i + 1, f"r{i + 1}", statuses[i], "c", numbers[i], {"x": numbers[i]})
            for i in range(12)]
    rep = ar.build_report(rows, ar.Disagreements(), code_commit="c", generation_sha="g")
    text = rep.to_json()
    doc = json.loads(text)
    assert json.loads(json.dumps(doc, allow_nan=False)) == doc
    for r, n in zip(doc["rows"], numbers, strict=True):
        assert ("value" in r) == (n is not None and np.isfinite(n))


_FLOATS = st.floats(allow_nan=True, allow_infinity=True)
_SCALARS = st.one_of(st.none(), st.booleans(), st.integers(-10**6, 10**6), st.text(max_size=4),
                     _FLOATS, _FLOATS.map(np.float64),
                     st.floats(allow_nan=True, allow_infinity=True, width=32).map(np.float32),
                     st.integers(-1000, 1000).map(np.int64), st.booleans().map(np.bool_),
                     st.sampled_from([np.float32("nan"), np.float64("-inf"),
                                      np.float32("inf"), float("nan")]))
_DETAIL = st.recursive(_SCALARS, lambda kids: st.lists(kids, max_size=4)
                       | st.dictionaries(st.text(max_size=4), kids, max_size=4), max_leaves=16)


def _has_missing(v: object) -> bool:  # the test's own oracle, written apart from _missing
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v) or np.isinf(v))
    if isinstance(v, list):
        return any(_has_missing(x) for x in v)
    return False


def _no_null_or_nan(v: object) -> bool:
    if isinstance(v, dict):
        return all(_no_null_or_nan(x) for x in v.values())
    if isinstance(v, list):
        return all(_no_null_or_nan(x) for x in v)
    return v is not None and not (isinstance(v, float) and not np.isfinite(v))


@settings(max_examples=150, deadline=None)
@given(detail=st.dictionaries(st.text(max_size=4), _DETAIL, max_size=5))
def test_any_detail_serialises_with_missing_values_absent(detail: dict[str, object]) -> None:
    rows = [ar.Row(i + 1, f"r{i + 1}", ar.Status.NOT_COMPUTABLE, "c", None, detail)
            for i in range(12)]
    doc = json.loads(ar.build_report(rows, ar.Disagreements(), code_commit="c",
                                     generation_sha="g").to_json())
    got = doc["rows"][0].get("detail", {})
    assert _no_null_or_nan(got)
    assert set(got) == {k for k, v in detail.items() if not _has_missing(v)}


def test_nan_inside_a_list_makes_that_list_absent() -> None:
    detail = {"ci95": [0.1, float("nan")], "f32": [np.float32("nan")],
              "nested": [[1.0, [None]]], "kept": [1, np.float64(2.5), [np.int64(3)]],
              "inner": {"lo": np.float32("inf"), "n": np.int64(4)}}
    rows = [ar.Row(i + 1, f"r{i + 1}", ar.Status.NOT_COMPUTABLE, "c", None, detail)
            for i in range(12)]
    doc = json.loads(ar.build_report(rows, ar.Disagreements(), code_commit="c",
                                     generation_sha="g").to_json())
    assert doc["rows"][0]["detail"] == {"kept": [1, 2.5, [3]], "inner": {"n": 4}}
