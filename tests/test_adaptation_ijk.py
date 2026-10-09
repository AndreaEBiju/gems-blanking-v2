"""RULING 2026-10-09 items 3-5: I/J/K adaptation rows are admitted by a declared criterion.

Admitted: ``label_purpose == "adaptation"`` AND ``queue_file`` a declared adaptation queue
AND ``queue_sha256`` its declared SHA-256 AND new-cohort I/J/K - and only at least 60 s
from every evaluation interval, re-asserted at training time. Everything else of I/J/K
(set A, blind audits, the K exam) stays refused (R1).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.model import evaluate as ev
from gems_blanking_v2.model import labels as lb
from gems_blanking_v2.model import modes as md
from gems_blanking_v2.model import registry as rg
from gems_blanking_v2.model import train as tr
from gems_blanking_v2.model.provenance import read_provenance
from gems_blanking_v2.types import TrainingMode

from tests.conftest import make_feature_table

THREADS = 2
QUEUE = "setADAPT_IJK.parquet"
SHA = lb.ADAPTATION_QUEUES[QUEUE]
TOPUP = "setADAPT_J_topup.parquet"
TOPUP_SHA = "8ea02d7862c6c91803cd069194d676eb926f1b4642659ccba9dbcdbdb4b5d750"
"""The J top-up queue's sha256 (setADAPT_J_topup.json, RULING 2026-10-09 option (a))."""
K_EXAM = ("gems_k_t02_cme2_sr_214535_20260825T014539Z", 228.313, 828.313)
"""The K exam span (k_exam_span.json, RULING 2026-10-09 item 4)."""
I_SPAN = ("gems_i_t02_es2_bl_215727_20260901T015731Z", 85.0673492614171, 205.06734926141712)


def _evaluation(*extra: tuple[str, float, float]) -> pd.DataFrame:
    rows = [K_EXAM, I_SPAN, *extra]
    return pd.DataFrame(rows, columns=["recording", "start_s", "stop_s"])


def _base(seed: int = 0) -> pd.DataFrame:
    return make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=30,
                              seed=seed)


def _ijk(animal: str, *, n_rec: int = 2, purpose: str | None = lb.ADAPTATION_PURPOSE,
         queue: str = QUEUE, sha: str = SHA, seed: int = 1) -> pd.DataFrame:
    """Rows of a test animal as the screen writes them: label_set test, no audit span."""
    t = make_feature_table({"new": (animal,)}, n_recordings=n_rec, cores_per_recording=12,
                           seed=seed)
    t["label_set"] = "test"
    t["span_id"] = None
    t["label_purpose"] = purpose
    t["queue_file"] = queue
    t["queue_sha256"] = sha
    return t


def _table(*parts: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# the four required cases
# ---------------------------------------------------------------------------


def test_an_evaluation_tagged_i_row_is_refused() -> None:
    set_a = _ijk("I", purpose=None, queue="setA_random.parquet", sha="a" * 64)
    eval_purpose = _ijk("I", purpose="evaluation", seed=2)  # right queue and sha, wrong tag
    eval_purpose["recording"] = eval_purpose["recording"].str.replace("_ms", "_es", regex=False)
    for i_rows in (set_a, eval_purpose):
        t = _table(_base(), i_rows)
        out, rec = lb.admit_adaptation(t, _evaluation())
        assert rec["n_admitted"] == {}
        assert (out.loc[out["animal"] == "I", "label_set"] == "test").all()
        assert not lb.is_adaptation_row(out).any()
        assert set(lb.training_rows(out)["animal"]) == {"A", "B"}
        with pytest.raises(ValueError, match="prospective test set"):
            md.refuse_test_rows(out, evaluation=_evaluation())
        with pytest.raises(ValueError, match="prospective test set"):
            md.prepare_table(out, evaluation=_evaluation())


def test_an_adaptation_row_from_another_queue_or_with_a_wrong_sha_is_refused() -> None:
    for queue, sha in (("setB_uncertainty.parquet", SHA), (QUEUE, "0" * 64), (QUEUE, "")):
        t = _table(_base(), _ijk("I", queue=queue, sha=sha))
        with pytest.raises(ValueError, match="not a declared adaptation queue"):
            lb.admit_adaptation(t, _evaluation())
        assert not lb.is_adaptation_row(t).any()
        forged = t.assign(label_set=np.where(t["animal"] == "I", lb.ADAPT_LABEL_SET, "train"))
        assert set(lb.training_rows(forged)["animal"]) == {"A", "B"}
        with pytest.raises(ValueError, match="prospective test set"):
            md.refuse_test_rows(forged, evaluation=_evaluation())
    old_j = make_feature_table({"old": ("J",)}, n_recordings=1, cores_per_recording=4, seed=3)
    old_j = old_j.assign(label_purpose=lb.ADAPTATION_PURPOSE, queue_file=QUEUE,
                         queue_sha256=SHA)  # the old cohort's JEL is not a test animal
    with pytest.raises(ValueError, match="not a declared adaptation queue"):
        lb.admit_adaptation(_table(_base(), old_j), _evaluation())


def test_a_valid_adaptation_row_is_admitted() -> None:
    t = _table(_base(), _ijk("I"))
    out, rec = lb.admit_adaptation(t, _evaluation())
    i_rows = out["animal"] == "I"
    assert (out.loc[i_rows, "label_set"] == lb.ADAPT_LABEL_SET).all()
    assert (out.loc[~i_rows, "label_set"] == "train").all()
    assert rec["n_admitted"] == {"I": int(i_rows.sum())} and rec["checked"] is True
    rows = lb.training_rows(out)
    assert set(rows["animal"]) == {"A", "B", "I"}
    assert len(rows) == len(out)  # every judgement in the fixture is motion or physiology
    prepared = md.prepare_table(rows, evaluation=_evaluation())
    assert (prepared["animal_key"] == "new:I").sum() == int(i_rows.sum())
    with pytest.raises(ValueError, match="without the evaluation intervals"):
        md.prepare_table(rows)  # run_modes, pooled and LOAO passes still refuse I/J/K
    with pytest.raises(ValueError, match="without the evaluation intervals"):
        md.refuse_test_rows(rows, context="a SHAP review")


def test_an_adaptation_row_overlapping_the_k_exam_span_raises() -> None:
    k = _ijk("K", seed=4)
    first = k["recording"] == k["recording"].iloc[0]
    assert K_EXAM[0] in lb.ADAPTATION_EXCLUDED_RECORDINGS
    # RULING 2026-10-09 item 4: the K exam recording supplies NO adaptation core, so a row
    # anywhere on it is refused - 888.5 s (60.19 s past the span, outside the margin) and
    # 1300 s included; the margin applies to other recordings only.
    for start in (300.0, 828.0, 870.0, 888.5, 1300.0):
        moved = k.copy()
        moved.loc[first, "recording"] = K_EXAM[0]
        n = int(first.sum())
        moved.loc[first, "start_s"] = start + 0.3 * np.arange(n) / n
        moved.loc[first, "stop_s"] = moved.loc[first, "start_s"] + 0.2
        t = _table(_base(), moved)
        with pytest.raises(ValueError, match="supplies no adaptation row at any distance"):
            lb.admit_adaptation(t, _evaluation())
    # the margin still governs a span on another recording: I's span, 888.5-style offsets
    i = _ijk("I", seed=5)
    one = i["recording"] == i["recording"].iloc[0]
    for start, ok in ((150.0, False), (240.0, False), (265.5, True)):
        moved = i.copy()
        moved.loc[one, "recording"] = I_SPAN[0]
        n = int(one.sum())
        moved.loc[one, "start_s"] = start + 0.3 * np.arange(n) / n
        moved.loc[one, "stop_s"] = moved.loc[one, "start_s"] + 0.2
        t = _table(_base(), moved)
        if ok:  # 265.5 - 205.067 = 60.43 s: outside the margin
            lb.admit_adaptation(t, _evaluation())
            continue
        with pytest.raises(ValueError, match="within 60 s of an evaluation interval"):
            lb.admit_adaptation(t, _evaluation())
    # re-asserted at training time: a row moved into the exam span AFTER admission
    out, _rec = lb.admit_adaptation(_table(_base(), k), _evaluation())
    rows = lb.training_rows(out)
    late = rows.copy()
    j = late.index[late["animal"] == "K"][0]
    late.at[j, "recording"] = K_EXAM[0]
    late.at[j, "start_s"] = 500.0
    late.at[j, "stop_s"] = 500.2
    with pytest.raises(ValueError, match="supplies no adaptation row at any distance"):
        md.refuse_test_rows(late, evaluation=_evaluation())
    with pytest.raises(ValueError, match="supplies no adaptation row at any distance"):
        md.prepare_table(late, evaluation=_evaluation())
    far = rows.copy()  # far from the span but on the exam recording: refused at training too
    far.at[j, "recording"] = K_EXAM[0]
    far.at[j, "start_s"] = 1300.0
    far.at[j, "stop_s"] = 1300.2
    with pytest.raises(ValueError, match="supplies no adaptation row at any distance"):
        md.prepare_table(far, evaluation=_evaluation())
    late.at[j, "recording"] = I_SPAN[0]  # the margin still governs other recordings
    late.at[j, "start_s"] = 150.0
    late.at[j, "stop_s"] = 150.2
    with pytest.raises(ValueError, match="within 60 s of an evaluation interval"):
        md.prepare_table(late, evaluation=_evaluation())


# ---------------------------------------------------------------------------
# the J top-up queue (RULING 2026-10-09 option (a)): declared by its sha256, J only
# ---------------------------------------------------------------------------


def test_the_declared_queues_are_exactly_the_two_files_with_their_animals() -> None:
    assert dict(lb.ADAPTATION_QUEUES) == {QUEUE: SHA, TOPUP: TOPUP_SHA}
    assert {n: sorted(q.animals) for n, q in lb.ADAPTATION_QUEUE_SPECS.items()} == {
        QUEUE: ["I", "J", "K"], TOPUP: ["J"]}


def test_a_j_topup_row_with_the_right_sha_is_admitted() -> None:
    j_top = _ijk("J", queue=TOPUP, sha=TOPUP_SHA, seed=8)
    i_main = _ijk("I", seed=9)
    t = _table(_base(), j_top, i_main)
    assert lb.is_adaptation_row(t).sum() == len(j_top) + len(i_main)
    out, rec = lb.admit_adaptation(t, _evaluation())
    assert rec["n_admitted"] == {"I": len(i_main), "J": len(j_top)}
    assert rec["queue_animals"][TOPUP] == ["J"]
    assert (out.loc[out["animal"] == "J", "label_set"] == lb.ADAPT_LABEL_SET).all()
    rows = lb.training_rows(out)
    assert set(rows["animal"]) == {"A", "B", "I", "J"}
    md.refuse_test_rows(rows, evaluation=_evaluation())  # passes: declared and disjoint


def test_a_j_topup_row_with_a_wrong_sha_raises() -> None:
    for sha in (SHA, "0" * 64, TOPUP_SHA[:-1] + "1", ""):
        t = _table(_base(), _ijk("J", queue=TOPUP, sha=sha, seed=10))
        assert not lb.is_adaptation_row(t).any()
        with pytest.raises(ValueError, match="not a declared adaptation queue"):
            lb.admit_adaptation(t, _evaluation())
        forged = t.assign(label_set=np.where(t["animal"] == "J", lb.ADAPT_LABEL_SET, "train"))
        assert set(lb.training_rows(forged)["animal"]) == {"A", "B"}
        with pytest.raises(ValueError, match="prospective test set"):
            md.refuse_test_rows(forged, evaluation=_evaluation())


def test_an_i_or_k_row_claiming_the_j_topup_queue_is_refused() -> None:
    for animal in ("I", "K"):
        claim = _ijk(animal, queue=TOPUP, sha=TOPUP_SHA, seed=11)
        t = _table(_base(), claim, _ijk("J", queue=TOPUP, sha=TOPUP_SHA, seed=12))
        ok = lb.is_adaptation_row(t)
        assert not ok[(t["animal"] == animal).to_numpy()].any()
        assert ok[(t["animal"] == "J").to_numpy()].all()  # the J rows beside it qualify
        with pytest.raises(ValueError, match=r"not a declared adaptation queue.*for that "
                                             r"animal.*setADAPT_J_topup\.parquet.*\['J'\]"):
            lb.admit_adaptation(t, _evaluation())
        forged = t.assign(label_set=np.where(t["animal"].isin(["A", "B"]), "train",
                                             lb.ADAPT_LABEL_SET))
        assert set(lb.training_rows(forged)["animal"]) == {"A", "B", "J"}
        with pytest.raises(ValueError, match="prospective test set"):
            md.refuse_test_rows(forged, evaluation=_evaluation())


# ---------------------------------------------------------------------------
# supporting behaviour
# ---------------------------------------------------------------------------


def test_disjointness_refuses_an_empty_evaluation_set() -> None:
    k = _ijk("K")
    with pytest.raises(ValueError, match="vacuously"):
        lb.assert_adaptation_disjoint(k, _evaluation().iloc[:0])


def test_adjudications_carry_the_admission_columns_from_the_newest_judgement() -> None:
    cores = _ijk("I", purpose=None, queue="", sha="").drop(
        columns=["label_purpose", "queue_file", "queue_sha256"])
    cores["judgement"] = "unjudged"
    cores["core_key"] = [f"k{i}" for i in range(len(cores))]
    adj = pd.DataFrame({
        "core_key": ["k0", "k1", "k1"], "judgement": ["motion", "unsure", "physiology"],
        "at": ["2026-10-09T10:00:00", "2026-10-09T10:00:00", "2026-10-09T11:00:00"],
        "cohort": "new", "animal": "I", "label_set": "test",
        "queue_file": [QUEUE, "setA_random.parquet", QUEUE],
        "queue_sha256": [SHA, "b" * 64, SHA], "label_purpose": [lb.ADAPTATION_PURPOSE, None,
                                                                lb.ADAPTATION_PURPOSE]})
    out, _rep = lb.apply_adjudications(cores, adj, rate_sample_keys=())
    got = out.set_index("core_key")
    assert got.loc["k1", "judgement"] == "physiology"  # newest wins, and so does its trace
    assert got.loc["k1", "queue_file"] == QUEUE and got.loc["k1", "queue_sha256"] == SHA
    assert lb.is_adaptation_row(out).sum() == 2  # k0 and k1; nobody judged the rest
    assert got.loc["k2", "queue_file"] is None


# ---------------------------------------------------------------------------
# the adaptation fit
# ---------------------------------------------------------------------------


def _prior(table: pd.DataFrame) -> tuple[object, rg.ModelSpec, ev.Calibrator]:
    keys = table["animal_key"].to_numpy()
    rows = table[~np.isin(keys, ["new:I", "new:J", "new:K"])]
    feats = tr.feature_columns(rows)
    booster = tr.fit(rows[feats], rows["y"], tr.sample_weights(rows), num_threads=THREADS,
                     rounds=10)
    cal = ev.Calibrator.fit(tr.predict_raw(booster, rows[feats]), rows["y"])
    ch = tr.corpus_hash(rows)
    mid = rg.model_content_id(booster, cal, mode=TrainingMode.POOLED, animal=None,
                              corpus_hash=ch)
    spec = rg.ModelSpec(mode=TrainingMode.POOLED, animal=None, version="t", corpus_hash=ch,
                        calibrator=rg.calibrator_relpath(mid),
                        trained_at=datetime(2026, 10, 9, tzinfo=UTC), metrics={},
                        n_train_events=len(rows))
    return booster, spec, cal


def test_adapt_test_animal_registers_flagged_models_and_checks_its_prior(
        tmp_path: Path) -> None:
    out, _rec = lb.admit_adaptation(_table(_base(5), _ijk("I", n_rec=3, seed=6)),
                                    _evaluation())
    table = md.prepare_table(lb.training_rows(out), evaluation=_evaluation())
    booster, spec, cal = _prior(table)
    record = ev.write_run_record(tmp_path / "rr.json", run_id="adapt_t")
    reg = rg.Registry(GemsStore.initialise(tmp_path / "gems"))
    kw = {"target": "new:I", "prior": booster, "prior_spec": spec, "prior_calibrator": cal,
          "evaluation": _evaluation(), "target_spans": {I_SPAN[0]: [I_SPAN[1:]]},
          "record_path": record, "num_threads": THREADS, "old_rate": None, "old_tiers": None,
          "label_opt_ins": md.LabelOptIns(allow_model_labels=False, keep_tiers=("1",)),
          "user": "t", "w_adapt_grid": (1.0, 30.0), "adapt_rounds": 5}
    res = md.adapt_test_animal(table, registry=reg, **kw)  # type: ignore[arg-type]
    assert sorted(res.registered) == [1.0, 30.0] and not res.refusals
    n_i = int((table["animal_key"] == "new:I").sum())
    assert len(res.predictions) == 2 * n_i
    assert set(res.predictions["protocol"]) == {md.ADAPT_IJK_PROTOCOL}
    for mid in res.registered.values():
        s = reg.spec(mid)
        assert s.never_scores_evaluation_spans and s.animal == "new:I"
        assert s.unvalidated  # ADAPT_LORO is not an animal-level held-out protocol
        prov = read_provenance(reg.store.model_dir(mid))
        assert prov["adaptation"]["prior_model_id"] == spec.model_id
        assert prov["adaptation"]["queues"] == [f"{QUEUE}|{SHA}"]
        assert prov["adaptation_separation"]["evaluation_spans"] == {
            I_SPAN[0]: [list(I_SPAN[1:])]}
    with pytest.raises(rg.NotApplicableError, match="never scores evaluation spans"):
        rg.resolve_batch([], {"new:I": reg.spec(res.registered[1.0])}, reg, cohort="new",
                         evaluation=True)
    # the prior must be exactly the pooled model over these rows
    with pytest.raises(ValueError, match="not the prior's"):
        md.adapt_test_animal(table.drop(index=table.index[0]).reset_index(drop=True),
                             **kw)  # type: ignore[arg-type]
    other_cal = ev.Calibrator.fit([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="content id"):
        md.adapt_test_animal(table, **{**kw, "prior_calibrator": other_cal})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="each one an evaluation interval"):
        md.adapt_test_animal(table, **{**kw, "target_spans": {"x": [(0.0, 1.0)]}})  # type: ignore[arg-type]
    two = _table(lb.training_rows(out), lb.training_rows(lb.admit_adaptation(
        _table(_base(5), _ijk("J", seed=7)), _evaluation())[0]).query("animal == 'J'"))
    with pytest.raises(ValueError, match="another test animal"):
        md.adapt_test_animal(md.prepare_table(two, evaluation=_evaluation()),
                             **kw)  # type: ignore[arg-type]
    assert json.loads(record.read_text(encoding="utf-8"))["run_id"] == "adapt_t"
