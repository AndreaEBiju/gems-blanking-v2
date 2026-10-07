"""Task 12 under rulings 2026-10-08 and 2026-10-08 (b).

The reuse record and retrain.py's parameters, nested tuning, B's inner-validated weight,
mode A's zero-label calibration, per-fold small folds, the test-animal registry rule, the
old-cohort gate and prior correction, the cohort probe and the nested tier check.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from gems_blanking_v2.constants import FS_NOMINAL_HZ
from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.model import compare as cmp
from gems_blanking_v2.model import evaluate as ev
from gems_blanking_v2.model import labels as lb
from gems_blanking_v2.model import modes as md
from gems_blanking_v2.model import params as pm
from gems_blanking_v2.model import registry as rg
from gems_blanking_v2.model import train as tr
from gems_blanking_v2.model import tuning as tu
from gems_blanking_v2.model.provenance import build_provenance, corpus_composition
from gems_blanking_v2.types import Recording, TrainingMode

from tests.conftest import make_feature_table, make_multichannel

THREADS = 2


def _detector_available(name: str) -> bool:
    try:
        import_detector_module(name)
    except (FileNotFoundError, ImportError):
        return False
    return True


@pytest.fixture
def new_table() -> pd.DataFrame:
    raw = make_feature_table({"new": ("A", "B", "H")}, n_recordings=3,
                             cores_per_recording=120, seed=41)
    return md.prepare_table(raw)


def _record(path: Path, **extra: Any) -> Path:  # noqa: ANN401
    return ev.write_run_record(path, run_id="r", extra=extra or None)


# ---------------------------------------------------------------------------
# item 1 (2026-10-08): reuse record, retrain.py's parameters
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _detector_available("model"), reason="GEMSBlanking not available")
def test_the_fixed_params_are_retrain_pys_defaults() -> None:
    v = pm.verify_retrain_defaults()
    assert v["verified"] and v["values"] == dict(pm.RETRAIN_DEFAULT_HPARAMS)
    extra = {k: pm.FIXED_PARAMS[k] for k in set(pm.FIXED_PARAMS) - set(v["values"])}
    assert extra == {"use_missing": True, "zero_as_missing": False}
    assert all(pm.FIXED_PARAMS[k] == val for k, val in v["values"].items())


def test_the_run_record_names_the_deviation_from_the_reuse_list(tmp_path: Path) -> None:
    rec = json.loads(_record(tmp_path / "r.json").read_text(encoding="utf-8"))
    reuse = rec["reuse"]
    assert reuse["ruling"] == "RULING 2026-10-08 item 1"
    assert any("retrain.py" in x for x in reuse["task12_reuse_list"])
    assert any("DEFAULT_HPARAMS" in k for k in reuse["reused"])
    assert {"detector/review.py", "detector/heldout_eval.py"} <= set(reuse["reused"])
    assert "invariant 9" in reuse["not_reused"]["detector/retrain.py retrain() training"]
    assert rec["params"] == json.loads(json.dumps(dict(pm.FIXED_PARAMS)))
    assert (rec["num_boost_round"], rec["adapt_rounds"]) == (pm.NUM_BOOST_ROUND,
                                                             pm.ADAPT_ROUNDS)


@pytest.mark.skipif(not _detector_available("heldout_eval"), reason="GEMSBlanking not available")
def test_heldout_eval_micro_f1_is_our_f1() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    yhat = np.where(rng.random(300) < 0.8, y, 1 - y)
    recs = np.repeat(["r1", "r2", "r3"], 100)
    he = ev.heldout_eval_metrics(y, yhat, recs)
    assert he["f1_heldout_eval_micro"] == pytest.approx(ev.scores(y, yhat).f1)
    per = [ev.scores(y[recs == r], yhat[recs == r]).f1 for r in ("r1", "r2", "r3")]
    assert he["f1_heldout_eval_macro"] == pytest.approx(float(np.mean(per)))


# ---------------------------------------------------------------------------
# nested tuning: the held-out animal never reaches the tuner
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SpyTuner:
    """Records every call's rows (positional table rows) and groups; returns FIXED_PARAMS."""

    calls: list[tuple[set[int], set[str]]] = dataclasses.field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {"enabled": True, "backend": "spy"}

    def __call__(self, x: pd.DataFrame, y: npt.NDArray[np.int8], w: npt.NDArray[np.float64],
                 groups: npt.NDArray[np.str_], *, num_threads: int) -> Mapping[str, Any]:
        self.calls.append((set(x.index.tolist()), set(groups.tolist())))
        return {**pm.FIXED_PARAMS, "num_leaves": 7}


def test_the_held_out_animal_never_reaches_the_tuner(new_table: pd.DataFrame,
                                                     tmp_path: Path) -> None:
    spy = _SpyTuner()
    record = _record(tmp_path / "r.json", tuning=spy.record())
    run = md.run_modes(new_table, targets=["new:A", "new:H"], record_path=record,
                       num_threads=THREADS, w_adapt_grid=(1.0,), rounds=10, adapt_rounds=5,
                       tuner=spy)
    keys = new_table["animal_key"].to_numpy()
    recs = new_table["recording"].astype(str).to_numpy()
    assert len(spy.calls) == len(run.tuned) == 2 * (1 + 3)  # per target: A/B pooled + 3 C
    for (rows, groups), t in zip(spy.calls, run.tuned, strict=True):
        if t["mode"] == "pooled":
            assert t["target"] not in groups
            assert not (rows & set(np.flatnonzero(keys == t["target"])))
        else:
            assert t["fold"] not in groups and groups <= set(recs[keys == t["target"]])
            assert not (rows & set(np.flatnonzero(recs == t["fold"])))
        assert t["params"]["num_leaves"] == 7


def test_tuning_must_be_in_the_run_record(new_table: pd.DataFrame, tmp_path: Path) -> None:
    spy = _SpyTuner()
    with pytest.raises(ValueError, match=r"extra\['tuning'\]"):
        md.run_modes(new_table, targets=["new:A"], record_path=_record(tmp_path / "r.json"),
                     num_threads=THREADS, modes=(TrainingMode.POOLED,), tuner=spy)
    with pytest.raises(ValueError, match=r"extra\['tuning'\]"):  # a record claiming tuning
        md.run_modes(new_table, targets=["new:A"], num_threads=THREADS,
                     record_path=_record(tmp_path / "t.json", tuning=spy.record()),
                     modes=(TrainingMode.POOLED,))


def test_inner_cv_scores_only_the_groups_it_is_given(new_table: pd.DataFrame) -> None:
    feats = tr.feature_columns(new_table)
    sub = new_table[new_table["animal_key"] != "new:A"]
    f1 = tu.inner_cv_f1(pm.FIXED_PARAMS, sub[feats], sub["y"], np.ones(len(sub)),
                        sub["animal_key"], rounds=20, num_threads=THREADS)
    assert 0 < f1 <= 1
    one = sub[sub["animal_key"] == "new:B"]
    assert math.isnan(tu.inner_cv_f1(pm.FIXED_PARAMS, one[feats], one["y"], np.ones(len(one)),
                                     one["animal_key"], rounds=5, num_threads=THREADS))


@pytest.mark.skipif(importlib.util.find_spec("optuna") is not None, reason="optuna installed")
def test_the_optuna_tuner_says_so_when_optuna_is_absent(new_table: pd.DataFrame) -> None:
    t = tu.OptunaInnerCV(n_trials=2)
    assert t.record()["enabled"] and "nesting" in t.record()
    with pytest.raises(ImportError, match="tuning is off by default"):
        t(new_table[tr.feature_columns(new_table)], new_table["y"].to_numpy(),
          np.ones(len(new_table)), new_table["animal_key"].to_numpy(), num_threads=1)


# ---------------------------------------------------------------------------
# item 2 (b): B's weight by inner validation, never the test fold
# ---------------------------------------------------------------------------


def test_bs_weight_never_sees_its_test_fold(new_table: pd.DataFrame, tmp_path: Path) -> None:
    record = _record(tmp_path / "r.json")
    kw: dict[str, Any] = {"targets": ["new:A"], "record_path": record, "num_threads": THREADS,
                          "modes": (TrainingMode.ADAPTED,), "w_adapt_grid": (1.0, 30.0),
                          "rounds": 20, "adapt_rounds": 10}
    base = md.run_modes(new_table, **kw).w_selection.set_index("fold")
    held = sorted(base.index)[0]
    flipped = new_table.copy()
    m = (flipped["recording"] == held).to_numpy()
    flipped.loc[m, "y"] = 1 - flipped.loc[m, "y"]  # the test fold's labels, inverted
    again = md.run_modes(flipped, **kw).w_selection.set_index("fold")
    assert again.loc[held, "w_chosen"] == base.loc[held, "w_chosen"]
    assert again.loc[held, "inner_f1"] == base.loc[held, "inner_f1"]
    assert (base["basis"].str.contains("inner leave-one-adaptation-recording-out")).all()


def test_the_w_rule_breaks_ties_and_no_evidence_to_the_smallest() -> None:
    y1 = [np.array([1, 0, 1], np.int8)]
    good, bad = [np.array([True, False, True])], [np.array([False, True, False])]
    w, basis, _f = md._select_w((1.0, 3.0), {1.0: (y1, bad), 3.0: (y1, good)}, 1)
    assert w == 3.0 and "tie" not in basis
    w, basis, _f = md._select_w((1.0, 3.0), {1.0: (y1, good), 3.0: (y1, good)}, 1)
    assert w == 1.0 and "tie" in basis
    w, basis, _f = md._select_w((3.0, 10.0), {3.0: ([], []), 10.0: ([], [])}, 0)
    assert w == 3.0 and basis.startswith("no inner evidence")


# ---------------------------------------------------------------------------
# item 2 (c): mode A calibration uses zero target labels
# ---------------------------------------------------------------------------


def _preds(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for t in ("new:A", "new:B", "new:H"):
        y = rng.integers(0, 2, 200)
        raw = np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0, 1)
        parts.append(pd.DataFrame({"mode": "pooled", "target": t, "w_adapt": np.nan,
                                   "y": y, "raw": raw, "cluster": [f"span:{t}{i % 5}"
                                                                   for i in range(200)]}))
    return pd.concat(parts, ignore_index=True)


def test_mode_a_calibration_never_reads_the_target_labels() -> None:
    p = _preds()
    p["p_cal"] = np.nan
    md._calibrate(p, "isotonic")
    q = _preds()
    a = (q["target"] == "new:A").to_numpy()
    q.loc[a, "y"] = 1 - q.loc[a, "y"]
    q["p_cal"] = np.nan
    md._calibrate(q, "isotonic")
    assert np.array_equal(p.loc[a, "p_cal"].to_numpy(), q.loc[a, "p_cal"].to_numpy())
    others = (p["target"] != "new:A").to_numpy()
    want = ev.Calibrator.fit(p.loc[others, "raw"], p.loc[others, "y"]).apply(p.loc[a, "raw"])
    assert np.allclose(p.loc[a, "p_cal"].to_numpy(), want)
    alone = p[p["target"] == "new:A"].assign(p_cal=np.nan)
    md._calibrate(alone, "isotonic")
    assert alone["p_cal"].isna().all()  # no other target: no calibration, never its own


# ---------------------------------------------------------------------------
# item 2 (b): "< 20 positives" per fold
# ---------------------------------------------------------------------------


def _fold_preds() -> pd.DataFrame:
    rows, k = [], 0
    for fold, n_pos in (("r1", 30), ("r2", 5)):
        for mode, w in (("adapted", math.nan), ("per_animal", math.nan), ("pooled", math.nan)):
            for i in range(60):
                y = int(i < n_pos)
                rows.append({"mode": mode, "target": "new:A", "w_adapt": w,
                             "fold": "all" if mode == "pooled" else fold, "row": k + i,
                             "recording": fold, "cluster": f"span:{fold}{i % 4}", "y": y,
                             "yhat": y if mode != "per_animal" or fold == "r2" else 1 - y,
                             "raw": float(y), "yhat_thr": np.nan, "protocol": "x",
                             "n_train_events": 1, "w_chosen": 3.0, "w_rule": ""})
        k += 60
    return pd.DataFrame(rows)


def test_a_fold_below_20_positives_never_decides() -> None:
    p = _fold_preds()
    r9 = ev.R9_THRESHOLDS
    m = cmp.matched_protocol_table(p, r9)
    bc = m[(m["comparison"] == "B vs C") & m["decides"]].iloc[0]
    assert (bc["n_folds"], bc["n_deciding_folds"], bc["n_rows_deciding"]) == (2, 1, 60)
    assert bc["f1_b_minus_a"] == pytest.approx(-1.0)  # r1 only: C wrong on every row there
    small = cmp.small_fold_table(p, r9)
    assert set(small["fold"]) == {"r2"} and set(small["mode"]) == {"adapted", "per_animal"}
    s = cmp.summarize(p, r9).set_index("mode")
    assert s.loc["adapted", "n_deciding_folds"] == 1 and s.loc["pooled", "deciding"]


# ---------------------------------------------------------------------------
# item 2 (d): registry - an adapted model of I/J/K only with separate labels
# ---------------------------------------------------------------------------


_TABLE = make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=40,
                            seed=11)
_FEATS = tr.feature_columns(_TABLE)


def _spec_for(booster: lgb.Booster, animal: str, *, flag: bool) -> tuple[rg.ModelSpec,
                                                                            ev.Calibrator]:
    cal = ev.Calibrator.fit(tr.predict_raw(booster, _TABLE[_FEATS]), _TABLE["y"])
    mid = rg.model_content_id(booster, cal, mode=TrainingMode.ADAPTED, animal=animal,
                              corpus_hash="c", w_adapt=3.0)
    return rg.ModelSpec(mode=TrainingMode.ADAPTED, animal=animal, version="1",
                        corpus_hash="c", calibrator=rg.calibrator_relpath(mid),
                        trained_at=datetime(2026, 10, 8, tzinfo=UTC),
                        metrics={"LOAO_ADAPT": {"f1": 0.5}}, n_train_events=10,
                        never_scores_evaluation_spans=flag), cal


def test_a_test_animal_model_exists_only_as_flagged_adapted_with_separate_labels(
        tmp_path: Path) -> None:
    booster = tr.fit(_TABLE[_FEATS], _TABLE["y"], num_threads=THREADS, rounds=10)
    with pytest.raises(ValueError, match="prospective test set"):
        _spec_for(booster, "new:J", flag=False)
    with pytest.raises(ValueError, match="cannot carry it"):
        _spec_for(booster, "new:A", flag=True)
    with pytest.raises(ValueError, match="prospective test set"):
        rg.ModelSpec(mode=TrainingMode.PER_ANIMAL, animal="new:I", version="1",
                     corpus_hash="c", calibrator=Path("models") / ("0" * 32) / "calibrator.json",
                     trained_at=datetime(2026, 10, 8, tzinfo=UTC), metrics={},
                     n_train_events=1, never_scores_evaluation_spans=True)
    spec, cal = _spec_for(booster, "new:J", flag=True)
    assert rg.ModelSpec.from_json(spec.to_json()) == spec
    assert "never_scores_evaluation_spans" not in _spec_for(booster, "new:A", flag=False)[
        0].to_dict()  # absent unless True
    reg = rg.Registry(GemsStore.initialise(tmp_path / "gems"))
    record = ev.write_run_record(tmp_path / "rr.json", run_id="t")
    prov = build_provenance(spec, corpus=corpus_composition(_TABLE, None), record_path=record,
                            w_adapt=3.0, calibration={"kind": "isotonic", "fitted_on": "t",
                                                      "protocol": "LOAO_ADAPT", "targets": [],
                                                      "n_predictions": 0},
                            code={"package_sha256": "0" * 64})
    with pytest.raises(ValueError, match="adaptation_separation"):
        reg.register(spec, booster, cal, user="t", provenance=prov)
    adapt = pd.DataFrame({"recording": ["j1", "j1"], "start_s": [1.0, 50.0],
                          "stop_s": [1.2, 50.2], "label_set": [rg.ADAPT_LABEL_SET] * 2})
    spans = {"j1": [(10.0, 40.0)]}
    with pytest.raises(ValueError, match="label_set"):
        rg.adaptation_separation(adapt.assign(label_set="test"), animal="new:J",
                                 evaluation_spans=spans)
    with pytest.raises(ValueError, match="overlap an evaluation span"):
        rg.adaptation_separation(adapt, animal="new:J", evaluation_spans={"j1": [(0.0, 1.1)]})
    with pytest.raises(ValueError, match="I/J/K"):
        rg.adaptation_separation(adapt, animal="new:A", evaluation_spans=spans)
    sep = rg.adaptation_separation(adapt, animal="new:J", evaluation_spans=spans)
    reg.register(spec, booster, cal, user="t", provenance={**prov, "adaptation_separation": sep})
    rec, _t = make_multichannel(FS_NOMINAL_HZ, 2.0, seed=12)
    rec_j: Recording = dataclasses.replace(rec, animal="J", session="s_J")
    cores = _TABLE.drop(columns="span_id")
    with pytest.raises(rg.NotApplicableError, match="never scores evaluation spans"):
        rg.run_inference(rec_j, spec, reg, cohort="new", cores=cores, chosen_by="t",
                         evaluation=True)
    with pytest.raises(rg.NotApplicableError, match="never scores evaluation spans"):
        rg.run_inference(rec_j, spec, reg, cohort="new", cores=_TABLE, chosen_by="t",
                         evaluation=False)  # the cores carry audit spans
    res = rg.run_inference(rec_j, spec, reg, cohort="new", cores=cores, chosen_by="t",
                           evaluation=False)
    assert np.isfinite(res.p_motion).all()
    with pytest.raises(rg.NotApplicableError, match="never scores evaluation spans"):
        rg.resolve_batch([rec_j], {"new:J": spec}, reg, cohort="new", evaluation=True)
    assert len(rg.resolve_batch([rec_j], {"new:J": spec}, reg, cohort="new",
                                evaluation=False)) == 1


# ---------------------------------------------------------------------------
# item 1 (b)-(c): old rows only with set A, prior-corrected weights
# ---------------------------------------------------------------------------


def _old_table(set_a: int = 10, seed: int = 50) -> pd.DataFrame:
    raw = make_feature_table({"new": ("A", "B"), "old": ("F", "L")}, n_recordings=3,
                             cores_per_recording=40, set_a_per_recording=set_a,
                             set_a_prevalence=0.25, seed=seed)
    return md.prepare_table(raw)


def test_prior_correction_sets_the_old_motion_rate_to_set_as() -> None:
    t = _old_table()
    w, corr = tr.prior_weights(t)
    assert corr is not None
    old = (t["cohort"] == "old").to_numpy()
    set_a = old & (t["basis"] == lb.SET_A_BASIS).to_numpy()
    y = t["y"].to_numpy()
    assert corr.rate == pytest.approx(y[set_a].mean())
    assert corr.n_marks == int((old & ~set_a).sum()) and corr.n_set_a == int(set_a.sum())
    eff = (w[old] * y[old]).sum() / w[old].sum()
    assert eff == pytest.approx(corr.rate)  # the effective old-cohort motion rate
    assert (w[~old] == 1).all() and (w[old & (y == 0)] == 1).all()
    lo, hi = corr.ci
    assert lo <= corr.rate <= hi and lo < hi
    assert corr.w_pos_ci[0] <= corr.w_pos <= corr.w_pos_ci[1]
    d = corr.to_dict()
    assert d["rate_ci95"] == [lo, hi] and "item 1(c)" in d["rule"]
    assert tr.prior_weights(t[t["cohort"] == "new"])[1] is None


def test_old_rows_without_set_a_negatives_are_refused() -> None:
    with pytest.raises(tr.OldRowsRefusedError, match="set A"):
        tr.prior_correction(_old_table(set_a=0))
    t = _old_table()
    neg_only_marks = t.copy()
    m = ((t["cohort"] == "old") & (t["basis"] != lb.SET_A_BASIS)).to_numpy()
    neg_only_marks.loc[np.flatnonzero(m)[:1], "y"] = 0
    with pytest.raises(ValueError, match="positives only"):
        tr.prior_correction(neg_only_marks)


def test_every_fit_with_old_rows_carries_its_correction(tmp_path: Path) -> None:
    t = _old_table()
    run = md.run_modes(t, targets=["new:A"], record_path=_record(tmp_path / "r.json"),
                       num_threads=THREADS, modes=(TrainingMode.POOLED,), rounds=10)
    (pr,) = run.priors
    assert pr["fit"] == "pooled new:A" and pr["n_marks"] > 0


def test_set_a_rows_bypass_the_tier_filter_only_they() -> None:
    raw = make_feature_table({"old": ("F",)}, n_recordings=2, cores_per_recording=5,
                             set_a_per_recording=3, seed=2)
    out = lb.training_rows(raw.drop(columns="y"), old_tiers={}, keep_tiers=("1",))
    assert (out["basis"] == lb.SET_A_BASIS).all() and len(out) == 6


# ---------------------------------------------------------------------------
# item 1 (d): the cohort probe
# ---------------------------------------------------------------------------


def test_the_cohort_probe_is_never_a_pass_by_default(new_table: pd.DataFrame,
                                                     tmp_path: Path) -> None:
    r9 = ev.R9_THRESHOLDS
    record = _record(tmp_path / "r.json")
    run = md.run_modes(new_table, targets=["new:A"], record_path=record, num_threads=THREADS,
                       w_adapt_grid=(1.0,), rounds=10, adapt_rounds=5)
    probe = cmp.cohort_probe(run.predictions, new_table, r9).set_index("mode")
    assert set(probe.index) == {"pooled", "adapted", "per_animal"}
    assert probe["passes"].isna().all()
    assert probe["status"].str.contains("no old-cohort cores judged physiology").all()
    t = _old_table()
    run = md.run_modes(t, targets=["new:A", "old:F"], record_path=_record(tmp_path / "o.json"),
                       num_threads=THREADS, modes=(TrainingMode.POOLED,), rounds=20)
    p = cmp.cohort_probe(run.predictions, t, r9).set_index("mode")
    assert p.loc["pooled", "status"] in ("pass", "fail")
    assert p.loc["pooled", "n_physiology_old"] > 0 and p.loc["pooled", "n_physiology_new"] > 0
    assert p.loc["pooled", "passes"] == (abs(p.loc["pooled", "delta"]) <= 0.10)
    for mode in ("adapted", "per_animal"):  # not run in this pass: no predictions at all
        assert p.loc[mode, "status"].startswith("not computable: no out-of-fold")
        assert math.isnan(p.loc[mode, "passes"])


def test_two_adapted_weights_never_share_a_model_id() -> None:
    # a corpus too small to split gives every weight the same booster; the id still differs
    booster = tr.fit(_TABLE[_FEATS], _TABLE["y"], num_threads=THREADS, rounds=2)
    cal = ev.Calibrator.fit(tr.predict_raw(booster, _TABLE[_FEATS]), _TABLE["y"])
    ids = {rg.model_content_id(booster, cal, mode=TrainingMode.ADAPTED, animal="new:A",
                               corpus_hash="c", w_adapt=w) for w in (1.0, 3.0, 10.0, 30.0)}
    assert len(ids) == 4
    assert rg.model_content_id(booster, cal, mode=TrainingMode.POOLED, animal=None,
                               corpus_hash="c") not in ids


# ---------------------------------------------------------------------------
# rulings (i)/(j): the nested tier check, wired for when old rows return
# ---------------------------------------------------------------------------


def test_the_tier_check_runs_when_set_a_exists_and_refuses_without(tmp_path: Path) -> None:
    t = _old_table()
    old_recs = sorted(set(t.loc[t["cohort"] == "old", "recording"]))
    tiers = {r: ("1", "2a", "2b")[i % 3] for i, r in enumerate(old_recs)}
    base = t.drop(columns=["animal_key", "cluster", "name_letter", "y"])

    def labelled(keep: tuple[str, ...]) -> pd.DataFrame:
        return lb.training_rows(base, old_tiers=tiers, keep_tiers=keep)

    record = _record(tmp_path / "r.json")
    verdicts, largest, preds = md.tier_check(labelled, targets=["new:A", "new:B"],
                                             record_path=record, num_threads=THREADS)
    assert [v.step for v in verdicts] == ["1+2a", "1+2a+2b"]
    assert largest in md.TIER_CHAIN
    assert all(len(p) == len(preds["1"]) for p in preds.values())
    with pytest.raises(ValueError, match="new-cohort audit spans only"):
        md.tier_check(labelled, targets=["old:F"], record_path=record, num_threads=THREADS)
    no_a = base[base["basis"] != lb.SET_A_BASIS]
    with pytest.raises(tr.OldRowsRefusedError):
        md.tier_check(lambda k: lb.training_rows(no_a, old_tiers=tiers, keep_tiers=k),
                      targets=["new:A", "new:B"], record_path=record, num_threads=THREADS)


def test_tier_chain_keeps_against_the_last_kept_step() -> None:
    r9 = ev.R9_THRESHOLDS
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 2000)
    bad = np.where(rng.random(2000) < 0.3, 1 - y, y)
    base = pd.DataFrame({"eid": np.arange(2000).astype(str), "target": "new:A",
                         "cluster": [f"span:{i % 40}" for i in range(2000)], "y": y})
    steps = {"1": base.assign(yhat=y), "1+2a": base.assign(yhat=bad),
             "1+2a+2b": base.assign(yhat=y)}
    verdicts, largest = cmp.tier_chain(steps, ["1", "1+2a", "1+2a+2b"], r9)
    assert [v.kept for v in verdicts] == [False, True]
    assert verdicts[1].previous == "1" and largest == "1+2a+2b"
    with pytest.raises(ValueError, match="audit spans"):
        cmp.tier_chain({**steps, "1": base.assign(yhat=y, cluster="rec:x")},
                       ["1", "1+2a"], r9)
