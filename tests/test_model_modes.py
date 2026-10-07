"""Task 12: folds, leakage guards, the three modes, calibration, baselines and verdicts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
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


def _rate(t: pd.DataFrame) -> tr.OldRateEstimate:
    """Return the combined old-rate estimate over a test table's own old rows."""
    return tr.old_motion_rate(t)


@pytest.fixture
def record(tmp_path: Path) -> Path:
    return ev.write_run_record(tmp_path / "run_record.json", run_id="test")


@pytest.fixture
def table() -> pd.DataFrame:
    raw = make_feature_table({"new": ("A", "B", "H"), "old": ("F",)}, n_recordings=3,
                             cores_per_recording=120, set_a_per_recording=40, seed=1)
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


def _renames(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "renames.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_a_name_that_reads_another_animal_needs_a_ruled_rename(tmp_path: Path) -> None:
    raw = make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=10,
                             seed=14)
    if md.animal_letter(str(raw["recording"].iloc[0])) is None:
        pytest.skip("GEMSBlanking checkout not available: no second reading of the animal")
    moved = str(raw.loc[raw["animal"] == "B", "recording"].iloc[0])  # a "gems_b_..." block
    raw.loc[raw["recording"] == moved, "animal"] = "A"  # its folder was renamed to A
    with pytest.raises(ValueError, match="reads a different animal"):
        md.prepare_table(raw)
    wrong = _renames(tmp_path, [{"recording": moved, "cohort": "new", "animal": "B",
                                 "basis": "test"}])
    with pytest.raises(ValueError, match="reads a different animal"):
        md.prepare_table(raw, renames=wrong)  # acknowledging the wrong animal
    t = md.prepare_table(raw, renames=_renames(tmp_path, [
        {"recording": moved, "cohort": "new", "animal": "A", "basis": "test"}]))
    assert set(t.loc[t["recording"] == moved, "name_letter"]) == {"A"}
    md.loao_folds(t, ["new:A", "new:B"])  # the acknowledged block no longer trips the check


def test_the_rename_record_is_validated(tmp_path: Path) -> None:
    row = {"recording": "r1", "cohort": "new", "animal": "A", "basis": "ruling"}
    assert md.read_renames(_renames(tmp_path, [row])) == {"r1": ("new", "A")}
    with pytest.raises(ValueError, match="not an object"):
        md.read_renames(_renames(tmp_path, [row, ["r2", "new", "B"]]))  # type: ignore[list-item]
    with pytest.raises(ValueError, match="declared twice"):
        md.read_renames(_renames(tmp_path, [row, {**row, "animal": "B"}]))
    with pytest.raises(ValueError, match="missing 'basis'"):
        md.read_renames(_renames(tmp_path, [{k: v for k, v in row.items() if k != "basis"}]))
    with pytest.raises(ValueError, match="animal"):
        md.read_renames(_renames(tmp_path, [{**row, "animal": "a"}]))


def test_a_rename_must_name_the_recordings_cohort(tmp_path: Path) -> None:
    raw = make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=10,
                             seed=21)
    if md.animal_letter(str(raw["recording"].iloc[0])) is None:
        pytest.skip("GEMSBlanking checkout not available: no second reading of the animal")
    moved = str(raw.loc[raw["animal"] == "B", "recording"].iloc[0])
    raw.loc[raw["recording"] == moved, "animal"] = "A"
    old = _renames(tmp_path, [{"recording": moved, "cohort": "old", "animal": "A",
                               "basis": "test"}])
    with pytest.raises(ValueError, match="wrong cohort"):
        md.prepare_table(raw, renames=old)


def test_run_modes_requires_the_renames_in_the_run_record(tmp_path: Path) -> None:
    raw = make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=20,
                             seed=22)
    path = _renames(tmp_path, [{"recording": "not_in_table", "cohort": "new", "animal": "A",
                                "basis": "test"}])
    t = md.prepare_table(raw, renames=path)
    assert md.renames_record(path)["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert md.renames_record(path)["content"] == json.loads(path.read_text(encoding="utf-8"))
    bare = ev.write_run_record(tmp_path / "bare.json", run_id="r")
    with pytest.raises(ValueError, match="extra\\['renames'\\]"):
        md.run_modes(t, old_rate=_rate(t), targets=["new:A"], record_path=bare, num_threads=THREADS,
                     modes=(TrainingMode.POOLED,), rounds=5)
    good = ev.write_run_record(tmp_path / "good.json", run_id="r",
                               extra={"renames": md.renames_record(path)})
    md.run_modes(t, old_rate=_rate(t), targets=["new:A"], record_path=good, num_threads=THREADS,
                 modes=(TrainingMode.POOLED,), rounds=5)
    path.write_text(path.read_text(encoding="utf-8").replace("test", "edited"),
                    encoding="utf-8")
    t2 = md.prepare_table(raw, renames=path)  # the file changed after the record
    with pytest.raises(ValueError, match="extra\\['renames'\\]"):
        md.run_modes(t2, old_rate=_rate(t2),
                     targets=["new:A"], record_path=good, num_threads=THREADS,
                     modes=(TrainingMode.POOLED,), rounds=5)
    plain = md.prepare_table(raw)  # no renames: a record claiming some is refused
    with pytest.raises(ValueError, match="extra\\['renames'\\]"):
        md.run_modes(plain, old_rate=_rate(plain),
                     targets=["new:A"], record_path=good, num_threads=THREADS,
                     modes=(TrainingMode.POOLED,), rounds=5)


def test_run_modes_rechecks_the_test_set(record: Path) -> None:
    t = md.prepare_table(make_feature_table({"new": ("A", "B")}, n_recordings=2,
                                            cores_per_recording=20, seed=23))
    t.loc[t["animal"] == "B", "animal"] = "J"  # altered after preparation
    with pytest.raises(ValueError, match="prospective test set"):
        md.run_modes(t, old_rate=_rate(t),
                     targets=["new:A"], record_path=record, num_threads=THREADS,
                     modes=(TrainingMode.POOLED,), rounds=5)


def test_a_target_without_scorable_recordings_is_refused() -> None:
    raw = make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=10,
                             seed=24)
    raw.loc[raw["animal"] == "A", "span_id"] = None
    t = md.prepare_table(raw)
    for fn in (md.adapted_folds, md.per_animal_folds):
        with pytest.raises(md.ModeRefusedError, match="no recording with scorable"):
            fn(t, "new:A")


def test_loao_folds_assert_span_scoring_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=10,
                             seed=25)
    raw.loc[raw.index[:3], "span_id"] = None  # animal A rows outside any span
    t = md.prepare_table(raw)
    monkeypatch.setattr(md, "_scorable", lambda tb: np.ones(len(tb), dtype=bool))
    with pytest.raises(AssertionError, match="outside an audit span"):
        md.loao_folds(t, ["new:A"])


def test_prepare_table_requires_label_source() -> None:
    raw = make_feature_table({"new": ("A",)}, n_recordings=1, cores_per_recording=4, seed=26)
    with pytest.raises(ValueError, match="label_source"):
        md.prepare_table(raw.drop(columns="label_source"))


def test_one_recording_is_one_animal_of_one_cohort() -> None:
    raw = make_feature_table({"new": ("A",)}, n_recordings=1, cores_per_recording=6, seed=15)
    raw.loc[0, "animal"] = "B"  # never silently last-wins
    with pytest.raises(ValueError, match="more than one animal or cohort"):
        md.prepare_table(raw)


def test_the_prospective_test_set_is_refused() -> None:
    raw = make_feature_table({"new": ("A", "J")}, n_recordings=1, cores_per_recording=6,
                             seed=16)
    with pytest.raises(ValueError, match="prospective test set"):
        md.prepare_table(raw)  # new-cohort J, even labelled "train"
    a_only = raw[raw["animal"] == "A"].copy()
    a_only.loc[a_only.index[0], "label_set"] = "test"
    with pytest.raises(ValueError, match="label_set != 'train'"):
        md.prepare_table(a_only)
    old_j = make_feature_table({"old": ("J",)}, n_recordings=1, cores_per_recording=6, seed=17)
    md.prepare_table(old_j)  # old JEL is a different rat: not a test animal


def test_new_cohort_targets_are_scored_on_audit_spans_only(record: Path) -> None:
    raw = make_feature_table({"new": ("A", "B")}, n_recordings=3, cores_per_recording=30,
                             seed=18)
    outside = (raw["animal"] == "A") & (raw.index % 5 == 0)
    raw.loc[outside, "span_id"] = None  # judged, but not inside an exhaustive span
    t = md.prepare_table(raw)
    out_rows = set(np.flatnonzero(outside.to_numpy()))
    (fa,) = md.loao_folds(t, ["new:A"])
    assert not out_rows & set(fa.evaluate)
    for f in md.adapted_folds(t, "new:A") + md.per_animal_folds(t, "new:A"):
        assert not out_rows & set(f.evaluate)
        assert set(t["cluster"].iloc[f.evaluate].str[:5]) == {"span:"}
    trained = set().union(*(set(f.train) for f in md.per_animal_folds(t, "new:A")))
    assert out_rows <= trained  # they still train
    run = md.run_modes(t, old_rate=_rate(t),
                       targets=["new:A"], record_path=record, num_threads=THREADS,
                       w_adapt_grid=(1.0,), rounds=10, adapt_rounds=5)
    assert not out_rows & set(run.predictions["row"].astype(int))
    bad = dataclasses.replace(fa, evaluate=np.concatenate([fa.evaluate, sorted(out_rows)[:1]]))
    with pytest.raises(AssertionError, match="outside an audit span"):
        md._assert_scored_on_spans(t, bad)


def test_unknown_old_rows_never_train_an_old_target() -> None:
    raw = make_feature_table({"new": ("A",), "old": ("F", "L")}, n_recordings=2,
                             cores_per_recording=10, seed=19)
    unknown = raw["animal"] == "L"
    raw.loc[unknown, "animal"] = "?"  # the real "?" files are "ein2_1_..." - no letter
    raw.loc[unknown, "recording"] = raw.loc[unknown, "recording"].str.replace(
        "E1000_LOL", "ein2_1_lol")
    t = md.prepare_table(raw)
    (f_old,) = md.loao_folds(t, ["old:F"])
    assert "old:?" not in set(t["animal_key"].iloc[f_old.train])
    assert f_old.n_excluded_unknown == int((t["animal_key"] == "old:?").sum()) > 0
    (f_new,) = md.loao_folds(t, ["new:A"])
    assert "old:?" in set(t["animal_key"].iloc[f_new.train])
    assert f_new.n_excluded_unknown == 0
    for f_b in md.adapted_folds(t, "old:F"):  # mode B uses the same exclusion
        assert "old:?" not in set(t["animal_key"].iloc[f_b.train])
        assert np.array_equal(f_b.train, f_old.train)
        assert f_b.n_excluded_unknown == f_old.n_excluded_unknown
    for f_b in md.adapted_folds(t, "new:A"):
        assert np.array_equal(f_b.train, f_new.train)


def test_old_and_new_letters_are_different_animals() -> None:
    raw = make_feature_table({"new": ("A",), "old": ("A",)}, n_recordings=2,
                             cores_per_recording=10, seed=3)
    t = md.prepare_table(raw)
    assert set(t["animal_key"]) == {"new:A", "old:A"}
    (f,) = md.loao_folds(t, ["new:A"])
    assert set(t["animal_key"].iloc[f.train]) == {"old:A"}


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
        md.run_modes(table, old_rate=_rate(table),
                     targets=["new:A"], record_path=tmp_path / "absent.json",
                     num_threads=THREADS)


def test_run_record_is_write_once_and_binding(tmp_path: Path) -> None:
    p = ev.write_run_record(tmp_path / "r.json", run_id="one", extra={"k": 1})
    assert ev.require_run_record(p) == ev.R9_THRESHOLDS
    ev.write_run_record(p, run_id="one", extra={"k": 1})  # idempotent
    with pytest.raises(ValueError, match="already exists"):
        ev.write_run_record(p, run_id="two", extra={"k": 1})
    with pytest.raises(ValueError, match="already exists"):
        ev.write_run_record(p, run_id="one", extra={"k": 2})
    good = p.read_text(encoding="utf-8")
    for old, new, why in (('"ece_max": 0.05', '"ece_max": 0.1', "binding values"),
                          ('"decision_p": 0.5', '"decision_p": 0.4', "decision_p"),
                          ('"ece_bins": 10', '"ece_bins": 15', "ece_bins"),
                          ('"calibration": "isotonic"', '"calibration": "platt"', "calibration"),
                          ("30.0", "31.0", "w_adapt_grid"),
                          ('"min_data_in_leaf": 100', '"min_data_in_leaf": 10', "params"),
                          ('"num_boost_round": 300', '"num_boost_round": 3000',
                           "num_boost_round"),
                          ("invariant 9", "invariant 8", "reuse"),
                          ("never on the test fold", "on the test fold", "rules")):
        assert old in good, old
        p.write_text(good.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError, match=why):
            ev.require_run_record(p)
    rules = json.loads(good)["rules"]
    assert set(rules) == {"verdict_rule", "small_fold_rule", "calibration", "registry",
                          "renames", "cohort"}
    for key, text in rules.items():  # ruled, no longer interpretations awaiting an answer
        assert "RULING 2026-10-08 (b)" in text and "INTERPRETATION" not in text, key


def test_run_modes_refuses_a_protocol_off_the_record(table: pd.DataFrame,
                                                    record: Path) -> None:
    with pytest.raises(ValueError, match="recorded kind"):
        md.run_modes(table, old_rate=_rate(table),
                     targets=["new:A"], record_path=record, num_threads=THREADS,
                     calibration="platt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="recorded grid"):
        md.run_modes(table, old_rate=_rate(table),
                     targets=["new:A"], record_path=record, num_threads=THREADS,
                     w_adapt_grid=(2.0,))


def test_r9_values_are_the_ruled_ones() -> None:
    r = ev.R9_THRESHOLDS
    assert (r.ece_max, r.c_beats_b_min_f1_gain, r.min_positives_deciding, r.n_bootstrap,
            r.tier_drop_f1) == (0.05, 0.03, 20, 1000, 0.02)


# ---------------------------------------------------------------------------
# the three modes, end to end
# ---------------------------------------------------------------------------


def test_all_three_modes_score_the_same_rows(table: pd.DataFrame, record: Path) -> None:
    run = md.run_modes(table, old_rate=_rate(table), targets=["new:A", "new:B"], record_path=record,
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
    # mode B's inner-selected series scores the same rows, one chosen w per fold
    for t in ("new:A", "new:B"):
        sel = p[(p["target"] == t) & (p["mode"] == "adapted") & p["w_adapt"].isna()]
        assert set(sel["row"]) == set(np.flatnonzero(table["animal_key"] == t))
        assert (sel["w_rule"] == "inner_selected").all()
        assert set(sel["w_chosen"]) <= {1.0, 10.0}
        assert (sel.groupby("fold")["w_chosen"].nunique() == 1).all()
    assert len(run.w_selection) == 2 * 3  # one per held-out recording per target
    # SMALL_FOLD_RULE (ruling 2026-10-08 (b) item 2): per fold as the protocol defines it
    for _i, r in summ.iterrows():
        g = cmp._sel(p, r["mode"], r["target"], r["w_adapt"])
        per = g.groupby("fold")["y"].sum()
        assert r["n_folds"] == len(per) == (1 if r["mode"] == "pooled" else 3)
        assert r["n_deciding_folds"] == int((per >= 20).sum())
        assert r["min_pos_per_fold"] == int(per.min())
        assert r["deciding"] == (r["n_deciding_folds"] > 0)
    # mode A is calibrated with zero target labels; B and C use the target's own
    assert (summ["ece_cal_uses_target_labels"] == (summ["mode"] != "pooled")).all()
    # the corpus table counts a fold's TRAINING rows, never its held-out recording
    c = run.corpus.set_index(["mode", "target"])
    c_fold = md.per_animal_folds(table, "new:A")[0]
    assert c.loc[("per_animal", "new:A"), "n_events"] == c_fold.train.size
    b_fold = md.adapted_folds(table, "new:A")[0]
    (a_fold,) = md.loao_folds(table, ["new:A"])
    assert c.loc[("adapted", "new:A"), "n_events"] == a_fold.train.size + b_fold.adapt.size


def test_the_generator_honours_r3_old_positives_only() -> None:
    raw = make_feature_table({"new": ("A",), "old": ("F",)}, n_recordings=2,
                             cores_per_recording=40, seed=20)
    assert (raw.loc[raw["cohort"] == "old", "y"] == 1).all()
    assert set(raw.loc[raw["cohort"] == "new", "y"]) == {0, 1}


def test_old_cohort_positives_only_are_refused_for_mode_c(record: Path) -> None:
    # set A's negatives exist for F; L has only its inherited marks (positives)
    raw = make_feature_table({"new": ("A", "B"), "old": ("F", "L")}, n_recordings=2,
                             cores_per_recording=30, set_a_per_recording=40, seed=4)
    raw = raw[~((raw["animal"] == "L") & (raw["basis"] == lb.SET_A_BASIS))]
    t = md.prepare_table(raw)
    run = md.run_modes(t, old_rate=_rate(t),
                       targets=["old:L"], record_path=record, num_threads=THREADS,
                       modes=(TrainingMode.PER_ANIMAL,), rounds=10)
    assert [r.reason for r in run.refusals] == ["labels hold one class only ([1])"]


def test_old_rows_never_train_without_set_a_negatives(record: Path) -> None:
    raw = make_feature_table({"new": ("A", "B"), "old": ("F",)}, n_recordings=2,
                             cores_per_recording=20, seed=4)
    t = md.prepare_table(raw)
    for targets in (["new:A"], ["old:F"]):
        with pytest.raises(tr.OldRowsRefusedError, match=r"item 1\(b\)"):
            md.run_modes(t, old_rate=_rate(t),
                         targets=targets, record_path=record, num_threads=THREADS,
                         rounds=5, adapt_rounds=5, w_adapt_grid=(1.0,))
    # new-cohort only: the provisional runs
    new_only = md.prepare_table(raw[raw["cohort"] == "new"])
    run = md.run_modes(new_only, old_rate=_rate(new_only),
                       targets=["new:A"], record_path=record, num_threads=THREADS,
                       modes=(TrainingMode.POOLED,), rounds=5)
    assert len(run.predictions) and not run.priors


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
    # new cohort only: a subsample of a corpus with old rows can draw marks without set
    # A's negatives, and such a fit is refused (ruling 2026-10-08 (b) item 1(b))
    table = md.prepare_table(table[table["cohort"] == "new"])
    curve = md.learning_curve(table, targets=["new:A"], record_path=record,
                              num_threads=THREADS, sizes=sizes,
                              modes=tuple(TrainingMode), w_adapt_grid=(3.0,),
                              n_resamples=50)
    a = curve[curve["line"] == "pooled (LOAO)"]
    b = curve[curve["mode"] == "adapted"]
    assert set(b["line"]) == {"adapted w=3 (LOAO_ADAPT; x excludes adaptation rows)"}
    assert (b["n_train_events"] == b["requested_size"]).all()  # x = the pooled part only
    assert (b["n_adapt_events"] > 0).all()
    assert len(a) >= cmp.MIN_CURVE_POINTS
    c = curve[curve["line"] == "per_animal (LORO)"]
    # per-animal folds hold 240 training events: larger points are withheld, not faked
    assert len(c) > 0
    assert (c["requested_size"] <= 240).all()
    assert (c["n_train_events"] == c["requested_size"]).all()
    assert set(curve["protocol"]) == {"LOAO", "LOAO_ADAPT", "LORO"}


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
    y = (rng.random(n) < 0.23).astype(int)
    logit = np.where(y == 1, rng.normal(3.0, 4.0, n), rng.normal(-4.0, 4.0, n))
    raw = 1 / (1 + np.exp(-logit))
    pin = rng.random(n) < 0.2
    raw[pin] = np.where(rng.random(int(pin.sum())) < 0.5, 1e-9, 1 - 1e-9)
    c = ev.Calibrator.fit(raw, y, "platt")
    assert 0.01 < c.a < 10
    p = c.apply(raw)
    assert 0.05 < p.mean() < 0.5
    assert ev.ece(p, y) < ev.ece(raw, y) + 0.02


def test_calibrator_round_trips_as_json() -> None:
    rng = np.random.default_rng(6)
    s = rng.random(200)
    y = (rng.random(200) < s).astype(int)
    for kind in ("isotonic", "platt"):
        c = ev.Calibrator.fit(s, y, kind)
        assert (len(c.x) > 0) if kind == "isotonic" else (c.a != 1.0 or c.b != 0.0)
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


def _bc(gain: float, lo: float, n_deciding: int, *, sweep: float = -0.5) -> pd.DataFrame:
    """B-vs-C rows: the inner-selected row (gain, lo) and a swept row (``sweep``)."""
    return pd.DataFrame({"target": "new:A", "comparison": "B vs C",
                         "w_a": [float("nan"), 3.0], "b_weight": ["inner-selected", "w=3"],
                         "decides": [True, False], "w_chosen_per_fold": ['{"3": 2}', ""],
                         "n_deciding_folds": n_deciding, "f1_b_minus_a": [gain, sweep],
                         "lo": [lo, sweep - 0.05], "hi": [gain + 0.05, sweep + 0.05]})


def test_c_beats_b_only_by_the_r9_margin_at_bs_inner_selected_weight() -> None:
    r9 = ev.R9_THRESHOLDS
    (v,) = cmp.b_vs_c_verdict(_bc(0.05, 0.01, 2), r9)
    assert v.verdict == "C > B" and v.task11_investigation
    (v,) = cmp.b_vs_c_verdict(_bc(0.02, 0.01, 2), r9)
    assert v.verdict == "B >= C"  # under the margin
    (v,) = cmp.b_vs_c_verdict(_bc(0.05, -0.001, 2), r9)
    assert v.verdict == "B >= C"  # CI touches 0
    (v,) = cmp.b_vs_c_verdict(_bc(0.05, 0.01, 0), r9)
    assert not v.deciding and not v.task11_investigation
    assert v.verdict.startswith("not deciding")
    # a swept weight never decides, whichever way it points
    (v,) = cmp.b_vs_c_verdict(_bc(0.0, -0.02, 2, sweep=0.3), r9)
    assert v.verdict == "B >= C" and "reported only" in v.detail
    (v,) = cmp.b_vs_c_verdict(_bc(0.0, -0.02, 2).iloc[1:], r9)
    assert v.verdict.startswith("no inner-selected B") and not v.deciding


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
    run = md.run_modes(table, old_rate=_rate(table),
                       targets=["new:H"], record_path=record, num_threads=THREADS,
                       w_adapt_grid=(3.0,), rounds=30, adapt_rounds=10)
    m = cmp.matched_protocol_table(run.predictions, ev.R9_THRESHOLDS)
    ac = m[m["comparison"] == "A vs C"]
    assert len(ac) == 1
    assert not ac["comparable"].iloc[0]
    assert ac[["f1_b_minus_a", "lo", "hi"]].isna().all(axis=None)
    assert set(m.loc[m["comparable"], "comparison"]) == {"A vs B", "B vs C"}
    assert m.loc[m["decides"], ["comparison", "b_weight"]].values.tolist() == [
        ["B vs C", "inner-selected"]]


def test_comparison_artifact_is_written(table: pd.DataFrame, record: Path,
                                        tmp_path: Path) -> None:
    r9 = ev.require_run_record(record)
    run = md.run_modes(table, old_rate=_rate(table),
                       targets=["new:A"], record_path=record, num_threads=THREADS,
                       w_adapt_grid=(1.0,), rounds=30, adapt_rounds=10)
    summ = cmp.summarize(run.predictions, r9)
    matched = cmp.matched_protocol_table(run.predictions, r9)
    curve = md.learning_curve(md.prepare_table(table[table["cohort"] == "new"]),
                              targets=["new:A"], record_path=record,
                              num_threads=THREADS, sizes=cmp.curve_sizes(20, 200),
                              modes=(TrainingMode.POOLED,), n_resamples=20)
    probe = cmp.cohort_probe(run.predictions, table, r9)
    pq = cmp.write_comparison(tmp_path / "out", "t0", summary=summ, matched=matched,
                              verdicts=cmp.b_vs_c_verdict(matched, r9), corpus=run.corpus,
                              preds=run.predictions, curve=curve, r9=r9, probe=probe,
                              small=cmp.small_fold_table(run.predictions, r9),
                              w_selection=run.w_selection)
    back = pd.read_parquet(pq)
    assert set(back["mode"]) == {"pooled", "adapted", "per_animal"}
    md_text = (tmp_path / "out" / "comparison_t0.md").read_text(encoding="utf-8")
    assert "not comparable" in md_text
    assert "zero target labels" in md_text and "Cohort probe" in md_text
    assert "inner-selected" in md_text
    cmp.write_comparison(tmp_path / "out2", "t0", summary=summ, matched=matched, verdicts=[],
                         corpus=run.corpus, preds=run.predictions)
    assert "NOT COMPUTED" in (tmp_path / "out2" / "comparison_t0.md").read_text(
        encoding="utf-8")
    assert (tmp_path / "out" / "curve_new_A.png").is_file()
    assert not list((tmp_path / "out").glob("*.tmp"))
    for bad in ("a:b", "x/y", "CON", "t0."):
        with pytest.raises(ValueError, match="path component"):
            cmp.write_comparison(tmp_path / "out", bad, summary=summ, matched=matched,
                                 verdicts=[], corpus=run.corpus, preds=run.predictions, r9=r9)
