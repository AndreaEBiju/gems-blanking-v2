"""Task 12: folds, leakage guards, the three modes, calibration, baselines and verdicts."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.model import compare as cmp
from gems_blanking_v2.model import evaluate as ev
from gems_blanking_v2.model import labels as lb
from gems_blanking_v2.model import modes as md
from gems_blanking_v2.model import train as tr
from gems_blanking_v2.types import TrainingMode
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

from tests.conftest import make_feature_table

THREADS = 2


@pytest.fixture
def record(tmp_path: Path) -> Path:
    return ev.write_run_record(tmp_path / "run_record.json", run_id="test")


@pytest.fixture
def table() -> pd.DataFrame:
    raw = make_feature_table({"new": ("A", "B", "H"), "old": ("F",)}, n_recordings=3,
                             cores_per_recording=50, seed=1)
    return md.prepare_table(raw)


# ---------------------------------------------------------------------------
# the loader (regression guard on task 10)
# ---------------------------------------------------------------------------


def test_training_loader_drops_unjudged_and_unsure() -> None:
    raw = make_feature_table({"new": ("A",)}, n_recordings=1, cores_per_recording=8, seed=2)
    raw.loc[0, "judgement"] = "unjudged"
    raw.loc[1, "judgement"] = "unsure"
    out = lb.training_rows(raw.drop(columns="y"))
    assert len(out) == len(raw) - 2
    assert set(out["judgement"]) <= {"motion", "physiology"}
    assert not out.index.isin([0, 1]).any()


# ---------------------------------------------------------------------------
# folds and leakage guards
# ---------------------------------------------------------------------------


def test_loao_folds_hold_no_animal_on_both_sides(table: pd.DataFrame) -> None:
    folds = md.loao_folds(table, ["new:A", "new:B", "new:H"])
    for f in folds:
        train = set(table["animal_key"].iloc[f.train])
        evald = set(table["animal_key"].iloc[f.evaluate])
        assert evald == {f.target}
        assert f.target not in train
        letters_train = {md.animal_letter(r) for r in table["recording"].iloc[f.train]}
        letters_eval = {md.animal_letter(r) for r in table["recording"].iloc[f.evaluate]}
        if None not in letters_eval:  # the private checkout is present: check by letter too
            assert letters_eval == {f.target[-1]}
            assert not (letters_train & letters_eval)


def test_disjointness_check_fires_on_a_leaky_fold(table: pd.DataFrame) -> None:
    (f,) = md.loao_folds(table, ["new:A"])
    leaky = dataclasses.replace(f, train=np.concatenate([f.train, f.evaluate[:1]]))
    with pytest.raises(AssertionError, match="both sides"):
        md.assert_disjoint_animals(table, leaky)
    # names with no readable letter: only the animal-key check can catch it
    anon = table.copy()
    anon["recording"] = [f"r_{i}_x" for i in pd.factorize(anon["recording"])[0]]
    with pytest.raises(AssertionError, match=r"animal\(s\) \['new:A'\] on both sides"):
        md.assert_disjoint_animals(anon, leaky)
    other = int(np.flatnonzero(table["animal_key"] == "new:B")[0])
    (c,) = md.per_animal_folds(table, "new:A")[:1]
    mixed = dataclasses.replace(c, train=np.concatenate([c.train, [other]]))
    with pytest.raises(AssertionError, match="holds animal keys"):
        md.assert_disjoint_animals(table, mixed)


def test_mode_b_never_adapts_on_a_scored_recording(table: pd.DataFrame) -> None:
    folds = md.adapted_folds(table, "new:B")
    rec = table["recording"].to_numpy()
    assert len(folds) == 3
    for f in folds:
        assert not set(rec[f.adapt]) & set(rec[f.evaluate])
        assert set(rec[f.evaluate]) == set(f.held_out)
        assert "new:B" not in set(table["animal_key"].iloc[f.train])
    scored = np.concatenate([f.evaluate for f in folds])
    assert sorted(scored) == sorted(np.flatnonzero(table["animal_key"] == "new:B"))


def test_mode_b_leakage_guard_fires(table: pd.DataFrame) -> None:
    f = md.adapted_folds(table, "new:B")[0]
    leaky = dataclasses.replace(f, adapt=np.concatenate([f.adapt, f.evaluate[:1]]))
    with pytest.raises(AssertionError, match="adapted AND scored"):
        md.assert_no_recording_leak(table, leaky)
    leaky2 = dataclasses.replace(f, train=np.concatenate([f.train, f.adapt[:1]]))
    with pytest.raises(AssertionError, match="target rows in the pooled corpus"):
        md.assert_no_recording_leak(table, leaky2)


def test_mode_c_corpus_holds_exactly_one_animal(table: pd.DataFrame) -> None:
    for f in md.per_animal_folds(table, "new:H"):
        keys = set(table["animal_key"].iloc[np.concatenate([f.train, f.evaluate])])
        assert keys == {"new:H"}
        letters = {md.animal_letter(r) for r in table["recording"].iloc[f.train]}
        assert letters in ({"H"}, {None})


def test_mode_c_refuses_animals_it_cannot_fit(table: pd.DataFrame) -> None:
    one_class = table.copy()
    one_class.loc[one_class["animal_key"] == "old:F", "y"] = 1  # old labels: positives only
    with pytest.raises(md.ModeRefusedError, match="one class"):
        md.per_animal_folds(one_class, "old:F")
    single = table[~((table["animal_key"] == "new:A")
                     & (table["recording"] != table.loc[table["animal_key"] == "new:A",
                                                        "recording"].iloc[0]))]
    with pytest.raises(md.ModeRefusedError, match=">= 2 labelled recordings"):
        md.per_animal_folds(single.reset_index(drop=True), "new:A")
    unknown = table.copy()
    unknown.loc[unknown["animal_key"] == "old:F", "animal_key"] = "old:?"
    with pytest.raises(md.ModeRefusedError, match="unknown group"):
        md.per_animal_folds(unknown, "old:?")
    with pytest.raises(ValueError, match="unknown group"):
        md.loao_folds(unknown, ["old:?"])


def test_old_and_new_letters_are_different_animals() -> None:
    raw = make_feature_table({"new": ("J",), "old": ("J",)}, n_recordings=2,
                             cores_per_recording=10, seed=3)
    t = md.prepare_table(raw)
    assert set(t["animal_key"]) == {"new:J", "old:J"}
    (f,) = md.loao_folds(t, ["new:J"])
    assert set(t["animal_key"].iloc[f.train]) == {"old:J"}


def test_a_leaky_recording_index_feature_is_rejected(table: pd.DataFrame) -> None:
    x = table[tr.feature_columns(table)].copy()
    tr.check_leakage(x, table["recording"].to_numpy())  # the real features pass
    x["recording_index"] = pd.factorize(table["recording"])[0].astype(float)
    with pytest.raises(tr.LeakyFeatureError, match="recording_index"):
        tr.check_leakage(x, table["recording"].to_numpy())
    x2 = table[tr.feature_columns(table)].copy()
    x2["start_s"] = table["start_s"]
    with pytest.raises(tr.LeakyFeatureError, match="start_s"):
        tr.check_leakage(x2, table["recording"].to_numpy())
    with pytest.raises(tr.LeakyFeatureError, match="bookkeeping"):
        tr.feature_columns(table, ["recording", "duration_s"])


# ---------------------------------------------------------------------------
# the run record (R9 written before training)
# ---------------------------------------------------------------------------


def test_training_refuses_without_a_run_record(table: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no run record"):
        md.run_modes(table, targets=["new:A"], record_path=tmp_path / "absent.json",
                     num_threads=THREADS)


def test_run_record_is_write_once_and_binding(tmp_path: Path) -> None:
    p = ev.write_run_record(tmp_path / "r.json", run_id="one")
    assert ev.require_run_record(p) == ev.R9_THRESHOLDS
    ev.write_run_record(p, run_id="one")  # idempotent
    with pytest.raises(ValueError, match="already exists"):
        ev.write_run_record(p, run_id="two")
    tampered = p.read_text(encoding="utf-8").replace('"ece_max": 0.05', '"ece_max": 0.1')
    p.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="not the binding values"):
        ev.require_run_record(p)


def test_r9_values_are_the_ruled_ones() -> None:
    r = ev.R9_THRESHOLDS
    assert (r.ece_max, r.c_beats_b_min_f1_gain, r.min_positives_deciding, r.n_bootstrap,
            r.tier_drop_f1) == (0.05, 0.03, 20, 1000, 0.02)


# ---------------------------------------------------------------------------
# the three modes, end to end
# ---------------------------------------------------------------------------


def test_all_three_modes_score_the_same_rows(table: pd.DataFrame, record: Path) -> None:
    run = md.run_modes(table, targets=["new:A", "new:B"], record_path=record,
                       num_threads=THREADS, w_adapt_grid=(1.0, 10.0), rounds=40,
                       adapt_rounds=20)
    p = run.predictions
    for t in ("new:A", "new:B"):
        rows = {m: set(p.loc[(p["target"] == t) & (p["mode"] == m), "row"])
                for m in ("pooled", "per_animal")}
        for w in (1.0, 10.0):
            rows[f"b{w}"] = set(p.loc[(p["target"] == t) & (p["mode"] == "adapted")
                                      & (p["w_adapt"] == w), "row"])
        target_rows = set(np.flatnonzero(table["animal_key"] == t))
        assert all(v == target_rows for v in rows.values())
    assert set(p["protocol"]) == {"LOAO", "LOAO_ADAPT", "LORO"}
    assert p["p_cal"].notna().mean() > 0.9
    summ = cmp.summarize(p, ev.R9_THRESHOLDS)
    assert (summ["f1"] > summ["f1_all_motion"]).all()  # separable synthetic data
    assert not run.refusals


def test_old_cohort_positives_only_are_refused_for_mode_c(record: Path) -> None:
    raw = make_feature_table({"new": ("A", "B"), "old": ("F",)}, n_recordings=2,
                             cores_per_recording=30, seed=4)
    raw.loc[raw["cohort"] == "old", "y"] = 1
    t = md.prepare_table(raw)
    run = md.run_modes(t, targets=["old:F"], record_path=record, num_threads=THREADS,
                       modes=(TrainingMode.PER_ANIMAL,), rounds=10)
    assert [r.reason for r in run.refusals] == ["labels hold one class only ([1])"]


def test_adaptation_does_not_modify_the_pooled_model(table: pd.DataFrame) -> None:
    feats = tr.feature_columns(table)
    rows = np.flatnonzero(table["animal_key"] != "new:A")
    x, y = table[feats], table["y"].to_numpy()
    base = tr.fit(x.iloc[rows], y[rows], num_threads=THREADS, rounds=20)
    before = tr.predict_raw(base, x)
    tr.fit(x, y, np.full(len(y), 3.0), num_threads=THREADS, rounds=10, init_model=base)
    assert np.array_equal(before, tr.predict_raw(base, x))


def test_learning_curve_has_at_least_five_points_per_line(table: pd.DataFrame,
                                                          record: Path) -> None:
    sizes = cmp.curve_sizes(20, 300)
    assert len(sizes) >= cmp.MIN_CURVE_POINTS
    curve = md.learning_curve(table, targets=["new:A"], record_path=record,
                              num_threads=THREADS, sizes=sizes,
                              modes=(TrainingMode.POOLED, TrainingMode.PER_ANIMAL),
                              n_resamples=50)
    a = curve[curve["line"] == "pooled"]
    assert len(a) >= cmp.MIN_CURVE_POINTS
    c = curve[curve["line"] == "per_animal"]
    # per-animal folds hold 100 training events: larger points are withheld, not faked
    assert (c["requested_size"] <= 100).all()
    assert (c["n_train_events"] >= c["requested_size"]).all()


def test_curve_sizes_refuses_too_few_points() -> None:
    with pytest.raises(ValueError, match=">= 5 points"):
        cmp.curve_sizes(10, 1000, n_points=4)
    with pytest.raises(ValueError, match="distinct"):
        cmp.curve_sizes(1, 3)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def test_crossfit_calibration_meets_the_ece_tolerance_on_held_out_data() -> None:
    rng = np.random.default_rng(5)
    n = 6000
    p_true = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p_true).astype(int)
    clusters = np.repeat(np.arange(30), n // 30)
    logit = np.log(p_true / (1 - p_true))
    distorted = {"isotonic": p_true ** 3,  # badly miscalibrated, but monotone
                 "platt": 1 / (1 + np.exp(-(0.4 * logit + 1.5)))}  # a logistic distortion
    for kind, raw in distorted.items():
        assert ev.ece(raw, y) > 0.1
        cal = ev.crossfit_calibrate(raw, y, clusters, kind=kind)  # type: ignore[arg-type]
        assert np.isfinite(cal).all()
        assert ev.ece(cal, y) <= ev.R9_THRESHOLDS.ece_max


def test_platt_does_not_diverge_on_saturated_scores() -> None:
    # LightGBM-like output: most scores pinned near 0 or 1, positives a minority
    rng = np.random.default_rng(13)
    n = 10000
    y = (rng.random(n) < 0.23).astype(int)  # noqa: PLR2004
    logit = np.where(y == 1, rng.normal(3.0, 4.0, n), rng.normal(-4.0, 4.0, n))
    raw = 1 / (1 + np.exp(-logit))
    raw[rng.random(n) < 0.2] = np.where(rng.random() < 0.5, 1e-9, 1 - 1e-9)  # noqa: PLR2004
    c = ev.Calibrator.fit(raw, y, "platt")
    assert 0.01 < c.a < 10  # noqa: PLR2004
    p = c.apply(raw)
    assert 0.05 < p.mean() < 0.5  # noqa: PLR2004
    assert ev.ece(p, y) < ev.ece(raw, y) + 0.02  # noqa: PLR2004


def test_calibrator_round_trips_as_json() -> None:
    rng = np.random.default_rng(6)
    s = rng.random(200)
    y = (rng.random(200) < s).astype(int)
    for kind in ("isotonic", "platt"):
        c = ev.Calibrator.fit(s, y, kind)
        c2 = ev.Calibrator.from_json(c.to_json())
        assert c2 == c
        assert np.allclose(c.apply(s), c2.apply(s))


def test_isotonic_calibrator_matches_sklearn() -> None:
    rng = np.random.default_rng(7)
    s = rng.random(300)
    y = (rng.random(300) < s ** 2).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    grid = np.linspace(-0.1, 1.1, 101)
    assert np.allclose(ev.Calibrator.fit(s, y).apply(grid), iso.predict(grid))


def test_a_clusters_calibrated_probability_ignores_its_own_labels() -> None:
    rng = np.random.default_rng(11)
    raw = rng.random(400)
    y = (rng.random(400) < raw).astype(int)
    clusters = np.repeat(np.arange(8), 50)
    before = ev.crossfit_calibrate(raw, y, clusters)
    y2 = y.copy()
    y2[clusters == 0] = 1 - y2[clusters == 0]
    after = ev.crossfit_calibrate(raw, y2, clusters)
    assert np.array_equal(before[clusters == 0], after[clusters == 0])
    assert not np.array_equal(before[clusters == 1], after[clusters == 1])


def test_tier_step_kept_on_a_small_significant_fall() -> None:
    r9 = ev.R9_THRESHOLDS
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, 4000)
    flip = rng.random(4000) < 0.01
    v = cmp.tier_step_verdict(y, y, np.where(flip, 1 - y, y), np.repeat(np.arange(80), 50),
                              r9, step="1+2a+2b", previous="1+2a")
    assert v.diff < 0 and v.hi < 0
    assert v.diff > -r9.tier_drop_f1
    assert v.kept


def test_crossfit_leaves_nan_where_no_calibrator_can_be_fitted() -> None:
    raw = np.array([0.1, 0.2, 0.9, 0.8])
    y = np.array([0, 0, 1, 1])
    out = ev.crossfit_calibrate(raw, y, np.array([0, 0, 1, 1]))
    assert np.isnan(out).all()  # each complement holds one class


# ---------------------------------------------------------------------------
# scores, bootstrap, baselines
# ---------------------------------------------------------------------------


def test_scores_and_undefined_ratios() -> None:
    s = ev.scores([1, 1, 0, 0], [1, 0, 1, 0])
    assert (s.precision, s.recall, s.f1, s.prevalence) == (0.5, 0.5, 0.5, 0.5)
    empty = ev.scores([0, 0], [0, 0])
    assert math.isnan(empty.f1) and math.isnan(empty.precision)


def test_cluster_bootstrap_is_wider_than_an_event_bootstrap_for_clustered_errors() -> None:
    rng = np.random.default_rng(8)
    n_cl, per = 20, 50
    y = rng.integers(0, 2, n_cl * per)
    good = np.repeat(rng.random(n_cl) < 0.7, per)  # whole clusters right or wrong
    yhat = np.where(good, y, 1 - y)
    lo_c, hi_c = ev.cluster_bootstrap_ci(y, yhat, np.repeat(np.arange(n_cl), per))
    lo_e, hi_e = ev.cluster_bootstrap_ci(y, yhat, np.arange(n_cl * per))
    assert (hi_c - lo_c) > 2 * (hi_e - lo_e)
    f1 = ev.scores(y, yhat).f1
    assert lo_c <= f1 <= hi_c


def test_paired_difference_of_identical_predictions_is_zero() -> None:
    y = np.array([1, 0, 1, 1, 0, 0] * 10)
    yhat = np.array([1, 0, 0, 1, 1, 0] * 10)
    d, lo, hi = ev.paired_diff_ci(y, yhat, yhat, np.repeat(np.arange(6), 10))
    assert (d, lo, hi) == (0.0, 0.0, 0.0)
    d2, _lo, _hi = ev.paired_diff_ci(y, yhat, y, np.repeat(np.arange(6), 10))
    assert d2 == pytest.approx(1 - ev.scores(y, yhat).f1)


def test_threshold_baseline_on_the_eng_band_ratio() -> None:
    assert ev.BASELINE_FEATURE == "band_ratio_max_c0"
    assert "5000" not in ev.BASELINE_FEATURE
    x = np.array([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    b = ev.ThresholdBaseline.fit(x, y)
    assert b.direction == 1
    assert ev.scores(y, b.predict(x)).f1 == 1.0
    assert b.predict(np.array([np.nan, 5.0])).tolist() == [0, 1]
    flipped = ev.ThresholdBaseline.fit(-x, y)
    assert flipped.direction == -1


def test_cluster_ids_are_spans_in_the_new_cohort_and_recordings_in_the_old() -> None:
    raw = make_feature_table({"new": ("A",), "old": ("F",)}, n_recordings=1,
                             cores_per_recording=2, seed=9)
    c = ev.cluster_ids(raw)
    assert c[raw["cohort"] == "new"].str.startswith("span:").all()
    assert c[raw["cohort"] == "old"].str.startswith("rec:").all()


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def _bc(gains: list[float], los: list[float], n_pos: int) -> pd.DataFrame:
    return pd.DataFrame({"target": "new:A", "comparison": "B vs C", "w_a": [1.0, 3.0][:len(gains)],
                         "n_pos": n_pos, "f1_b_minus_a": gains, "lo": los,
                         "hi": [g + 0.05 for g in gains]})


def test_c_beats_b_only_by_the_r9_margin_at_every_weight() -> None:
    r9 = ev.R9_THRESHOLDS
    (v,) = cmp.b_vs_c_verdict(_bc([0.05, 0.04], [0.01, 0.005], 50), r9)
    assert v.verdict == "C > B" and v.task11_investigation
    (v,) = cmp.b_vs_c_verdict(_bc([0.05, 0.02], [0.01, 0.005], 50), r9)
    assert v.verdict == "B >= C"  # under the margin at one weight
    (v,) = cmp.b_vs_c_verdict(_bc([0.05, 0.04], [0.01, -0.001], 50), r9)
    assert v.verdict == "B >= C"  # CI touches 0 at one weight
    (v,) = cmp.b_vs_c_verdict(_bc([0.05, 0.04], [0.01, 0.005], 19), r9)
    assert not v.deciding and not v.task11_investigation
    assert v.verdict.startswith("not deciding")


def test_tier_step_dropped_only_on_a_significant_fall() -> None:
    r9 = ev.R9_THRESHOLDS
    rng = np.random.default_rng(10)
    y = rng.integers(0, 2, 2000)
    good = y.copy()
    worse = np.where(rng.random(2000) < 0.3, 1 - y, y)
    cl = np.repeat(np.arange(40), 50)
    kept = cmp.tier_step_verdict(y, good, good, cl, r9, step="1+2a", previous="1")
    assert kept.kept
    dropped = cmp.tier_step_verdict(y, good, worse, cl, r9, step="1+2a", previous="1")
    assert not dropped.kept and dropped.diff < -r9.tier_drop_f1
    better = cmp.tier_step_verdict(y, worse, good, cl, r9, step="1+2a", previous="1")
    assert better.kept


def test_a_vs_c_is_never_a_comparison(table: pd.DataFrame, record: Path) -> None:
    run = md.run_modes(table, targets=["new:H"], record_path=record, num_threads=THREADS,
                       w_adapt_grid=(3.0,), rounds=30, adapt_rounds=10)
    m = cmp.matched_protocol_table(run.predictions, ev.R9_THRESHOLDS)
    ac = m[m["comparison"] == "A vs C"]
    assert len(ac) == 1
    assert not ac["comparable"].iloc[0]
    assert ac[["f1_b_minus_a", "lo", "hi"]].isna().all(axis=None)
    assert set(m.loc[m["comparable"], "comparison"]) == {"A vs B", "B vs C"}


def test_comparison_artifact_is_written(table: pd.DataFrame, record: Path,
                                        tmp_path: Path) -> None:
    r9 = ev.require_run_record(record)
    run = md.run_modes(table, targets=["new:A"], record_path=record, num_threads=THREADS,
                       w_adapt_grid=(1.0,), rounds=30, adapt_rounds=10)
    summ = cmp.summarize(run.predictions, r9)
    matched = cmp.matched_protocol_table(run.predictions, r9)
    curve = md.learning_curve(table, targets=["new:A"], record_path=record,
                              num_threads=THREADS, sizes=cmp.curve_sizes(20, 200),
                              modes=(TrainingMode.POOLED,), n_resamples=20)
    pq = cmp.write_comparison(tmp_path / "out", "t0", summary=summ, matched=matched,
                              verdicts=cmp.b_vs_c_verdict(matched, r9), corpus=run.corpus,
                              preds=run.predictions, curve=curve, r9=r9)
    back = pd.read_parquet(pq)
    assert set(back["mode"]) == {"pooled", "adapted", "per_animal"}
    md_text = (tmp_path / "out" / "comparison_t0.md").read_text(encoding="utf-8")
    assert "not comparable" in md_text
    assert (tmp_path / "out" / "curve_new_A.png").is_file()
