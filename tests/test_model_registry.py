"""Task 12A: the registry presents options and records the choice; it never decides."""

from __future__ import annotations

import dataclasses
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.constants import FS_NOMINAL_HZ
from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.model import evaluate as ev
from gems_blanking_v2.model import registry as rg
from gems_blanking_v2.model import train as tr
from gems_blanking_v2.model.evaluate import Calibrator
from gems_blanking_v2.model.provenance import build_provenance, corpus_composition
from gems_blanking_v2.types import Recording, TrainingMode
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import make_feature_table, make_multichannel

_TABLE = make_feature_table({"new": ("A", "B")}, n_recordings=2, cores_per_recording=40,
                            seed=11)
_FEATS = tr.feature_columns(_TABLE)


@pytest.fixture(scope="module")
def booster() -> lgb.Booster:
    return tr.fit(_TABLE[_FEATS], _TABLE["y"], num_threads=2, rounds=10)


@pytest.fixture(scope="module")
def recording() -> Recording:
    rec, _truth = make_multichannel(FS_NOMINAL_HZ, 2.0, seed=12)
    return rec


def _rec(base: Recording, animal: str) -> Recording:
    return dataclasses.replace(base, animal=animal, session=f"s_{animal}")


def _register(reg: rg.Registry, booster: lgb.Booster, mode: TrainingMode,
              animal: str | None, metrics: dict[str, dict[str, float]] | None = None,
              corpus: str = "c0") -> rg.ModelSpec:
    raw = tr.predict_raw(booster, _TABLE[_FEATS])
    cal = Calibrator.fit(raw, _TABLE["y"])
    mid = rg.model_content_id(booster, cal, mode=mode, animal=animal, corpus_hash=corpus,
                              w_adapt=3.0 if mode is TrainingMode.ADAPTED else None)
    spec = rg.ModelSpec(mode=mode, animal=animal, version="0.1.0", corpus_hash=corpus,
                        calibrator=rg.calibrator_relpath(mid),
                        trained_at=datetime(2026, 10, 7, 4, 0, tzinfo=UTC),
                        metrics=metrics if metrics is not None else
                        {"LOAO": {"f1": 0.8}} if mode is TrainingMode.POOLED else
                        {"LORO": {"f1": 0.9}},
                        n_train_events=len(_TABLE))
    reg.register(spec, booster, cal, user="tester", provenance=_prov(reg, spec))
    return spec


def _prov(reg: rg.Registry, spec: rg.ModelSpec) -> dict[str, object]:
    record = ev.write_run_record(reg.store.root / "run_record.json", run_id="t")
    return build_provenance(spec, corpus=corpus_composition(_TABLE, None), record_path=record,
                            w_adapt=3.0 if spec.mode is TrainingMode.ADAPTED else None,
                            calibration={"kind": "isotonic", "fitted_on": "test",
                                         "protocol": "LOAO", "targets": [],
                                         "n_predictions": 0},
                            code={"package_sha256": "0" * 64})


@pytest.fixture
def reg(tmp_path: Path) -> rg.Registry:
    return rg.Registry(GemsStore.initialise(tmp_path / "gems"))


def test_list_applicable_never_returns_another_animals_model(
        reg: rg.Registry, booster: lgb.Booster, recording: Recording) -> None:
    pooled = _register(reg, booster, TrainingMode.POOLED, None)
    per_a = _register(reg, booster, TrainingMode.PER_ANIMAL, "new:A")
    per_b = _register(reg, booster, TrainingMode.PER_ANIMAL, "new:B")
    ad_b = _register(reg, booster, TrainingMode.ADAPTED, "new:B", {"LOAO_ADAPT": {"f1": 0.85}})
    for_a = {s.model_id for s in rg.list_applicable(_rec(recording, "A"), reg, cohort="new")}
    assert for_a == {pooled.model_id, per_a.model_id}
    for_b = {s.model_id for s in rg.list_applicable(_rec(recording, "B"), reg, cohort="new")}
    assert for_b == {pooled.model_id, per_b.model_id, ad_b.model_id}
    assert {s.model_id for s in rg.list_applicable(_rec(recording, "Z"), reg, cohort="new")} == {
        pooled.model_id}


def test_old_and_new_cohort_letters_never_share_a_model(
        reg: rg.Registry, booster: lgb.Booster, recording: Recording) -> None:
    # a non-test letter shared by both cohorts
    old_a = _register(reg, booster, TrainingMode.PER_ANIMAL, "old:A")
    new_a = _register(reg, booster, TrainingMode.ADAPTED, "new:A", {"LOAO_ADAPT": {"f1": 0.5}})
    rec_a = _rec(recording, "A")  # a Recording carries only the letter
    assert {s.model_id for s in rg.list_applicable(rec_a, reg, cohort="new")} == {
        new_a.model_id}
    assert {s.model_id for s in rg.list_applicable(rec_a, reg, cohort="old")} == {
        old_a.model_id}
    with pytest.raises(rg.NotApplicableError, match="not applicable to new:A"):
        rg.run_inference(rec_a, old_a, reg, cohort="new", cores=_TABLE, chosen_by="t",
                         evaluation=False)
    assert len(rg.resolve_batch([rec_a], {"old:A": old_a}, reg, cohort="old",
                                evaluation=False)) == 1
    with pytest.raises(rg.NotApplicableError, match="not applicable"):
        rg.resolve_batch([rec_a], {"new:A": old_a}, reg, cohort="new", evaluation=False)
    with pytest.raises(ValueError, match="cohort must be"):
        rg.list_applicable(rec_a, reg, cohort="A")
    # old J (JEL) vs new J (a test animal): only a POOLED model may score new J
    old_j = _register(reg, booster, TrainingMode.ADAPTED, "old:J", {"LOAO_ADAPT": {"f1": 0.4}})
    pooled = _register(reg, booster, TrainingMode.POOLED, None)
    rec_j = _rec(recording, "J")
    assert {s.model_id for s in rg.list_applicable(rec_j, reg, cohort="new")} == {
        pooled.model_id}
    assert {s.model_id for s in rg.list_applicable(rec_j, reg, cohort="old")} == {
        pooled.model_id, old_j.model_id}


def test_animal_keys_are_one_letter_and_never_the_test_set() -> None:
    mid = "0" * 32
    kw = {"version": "1", "corpus_hash": "h", "calibrator": rg.calibrator_relpath(mid),
          "trained_at": datetime(2026, 1, 1, tzinfo=UTC), "metrics": {},
          "n_train_events": 1}
    for bad in ("old:?", "new:a", "new:AB", "old: A", "new:"):
        for mode in (TrainingMode.PER_ANIMAL, TrainingMode.ADAPTED):
            with pytest.raises(ValueError, match="animal key"):
                rg.ModelSpec(mode=mode, animal=bad, **kw)  # type: ignore[arg-type]
    for test_animal in ("new:I", "new:J", "new:K"):
        for mode in (TrainingMode.PER_ANIMAL, TrainingMode.ADAPTED):
            with pytest.raises(ValueError, match="prospective test set"):
                rg.ModelSpec(mode=mode, animal=test_animal, **kw)  # type: ignore[arg-type]
    rg.ModelSpec(mode=TrainingMode.PER_ANIMAL, animal="old:J", **kw)  # type: ignore[arg-type]


def test_metrics_carry_their_protocol(reg: rg.Registry, booster: lgb.Booster,
                                      recording: Recording) -> None:
    _register(reg, booster, TrainingMode.POOLED, None)
    _register(reg, booster, TrainingMode.PER_ANIMAL, "new:A")
    shown = {str(s.mode): s.metrics
             for s in rg.list_applicable(_rec(recording, "A"), reg, cohort="new")}
    assert shown == {"pooled": {"LOAO": {"f1": 0.8}}, "per_animal": {"LORO": {"f1": 0.9}}}


def test_run_inference_refuses_a_model_that_is_not_applicable(
        reg: rg.Registry, booster: lgb.Booster, recording: Recording) -> None:
    per_b = _register(reg, booster, TrainingMode.PER_ANIMAL, "new:B")
    with pytest.raises(rg.NotApplicableError, match="not applicable to new:A"):
        rg.run_inference(_rec(recording, "A"), per_b, reg, cohort="new", cores=_TABLE,
                         chosen_by="t", evaluation=False)
    unregistered = dataclasses.replace(per_b, corpus_hash="other")
    with pytest.raises(rg.NotApplicableError):
        rg.run_inference(_rec(recording, "B"), unregistered, reg, cohort="new", cores=_TABLE,
                         chosen_by="t", evaluation=False)


def test_inference_entry_points_have_no_default_model() -> None:
    for fn in (rg.run_inference, rg.list_applicable, rg.resolve_batch):
        for name, par in inspect.signature(fn).parameters.items():
            assert par.default is inspect.Parameter.empty, f"{fn.__name__}({name}=...)"
    src = inspect.getsource(rg)
    assert "current_model" not in src.replace("``current_model``", "")


def test_run_inference_scores_with_the_chosen_calibrated_model(
        reg: rg.Registry, booster: lgb.Booster, recording: Recording) -> None:
    spec = _register(reg, booster, TrainingMode.PER_ANIMAL, "new:A")
    res = rg.run_inference(_rec(recording, "A"), spec, reg, cohort="new", cores=_TABLE,
                           chosen_by="andrea", evaluation=False)
    raw = tr.predict_raw(booster, _TABLE[_FEATS])
    assert np.allclose(res.raw, raw)
    assert np.allclose(res.p_motion, reg.calibrator(spec).apply(raw))
    assert res.provenance["model_id"] == spec.model_id
    assert rg.require_model_in_provenance(res.provenance) == spec


def test_provenance_round_trips_the_full_spec(reg: rg.Registry, booster: lgb.Booster,
                                              recording: Recording) -> None:
    spec = _register(reg, booster, TrainingMode.ADAPTED, "new:A", {"LOAO_ADAPT": {"f1": 0.7}})
    prov = rg.inference_provenance(spec, rec=_rec(recording, "A"), cohort="new",
                                   chosen_by="andrea")
    back = json.loads(json.dumps(prov))
    assert rg.ModelSpec.from_dict(back["model"]) == spec
    assert (back["cohort"], back["animal_key"]) == ("new", "new:A")
    with pytest.raises(ValueError, match="names no model"):
        rg.require_model_in_provenance({"chosen_by": "andrea"})


def test_an_unvalidated_model_is_selectable_and_flagged(
        reg: rg.Registry, booster: lgb.Booster, recording: Recording) -> None:
    spec = _register(reg, booster, TrainingMode.PER_ANIMAL, "new:A", metrics={})
    assert spec.unvalidated
    assert spec in rg.list_applicable(_rec(recording, "A"), reg, cohort="new")
    res = rg.run_inference(_rec(recording, "A"), spec, reg, cohort="new", cores=_TABLE,
                           chosen_by="t", evaluation=False)
    assert res.provenance["unvalidated"] is True
    train_only = dataclasses.replace(spec, metrics={"train": {"f1": 1.0}})
    assert train_only.unvalidated


def test_registry_is_append_only_and_models_immutable(
        reg: rg.Registry, booster: lgb.Booster) -> None:
    spec = _register(reg, booster, TrainingMode.POOLED, None)
    cal = reg.calibrator(spec)
    changed = dataclasses.replace(spec, version="9.9.9")
    with pytest.raises(ValueError, match="immutable"):
        reg.register(changed, booster, cal, user="tester", provenance=_prov(reg, changed))
    wrong = dataclasses.replace(spec, corpus_hash="different")
    with pytest.raises(ValueError, match="content id"):
        reg.register(wrong, booster, cal, user="tester", provenance=_prov(reg, wrong))
    other = _register(reg, booster, TrainingMode.PER_ANIMAL, "new:A")
    with pytest.raises(ValueError, match="does not carry this model's spec"):
        reg.register(spec, booster, cal, user="tester", provenance=_prov(reg, other))
    reg.promote(spec.model_id, user="tester")
    assert reg.promoted_label(TrainingMode.POOLED, None) == spec.model_id
    shards = list(reg.store.registry_dir.iterdir())
    assert shards and all(p.name.startswith("events.jsonl") for p in shards)


def test_spec_validation() -> None:
    mid = "0" * 32
    kw = {"version": "1", "corpus_hash": "h", "calibrator": rg.calibrator_relpath(mid),
          "trained_at": datetime(2026, 1, 1, tzinfo=UTC), "metrics": {},
          "n_train_events": 1}
    with pytest.raises(ValueError, match="POOLED has no animal"):
        rg.ModelSpec(mode=TrainingMode.POOLED, animal="new:A", **kw)  # type: ignore[arg-type]
    for bare in ("A", "J", "x:J", "new:", "new:J:K"):
        with pytest.raises(ValueError, match="animal key"):
            rg.ModelSpec(mode=TrainingMode.PER_ANIMAL, animal=bare, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must name one"):
        rg.ModelSpec(mode=TrainingMode.PER_ANIMAL, animal=None, **kw)  # type: ignore[arg-type]
    bad = {**kw, "calibrator": Path("/abs/models") / mid / "calibrator.json"}
    with pytest.raises(ValueError, match="relative"):
        rg.ModelSpec(mode=TrainingMode.POOLED, animal=None, **bad)  # type: ignore[arg-type]
    naive = {**kw, "trained_at": datetime(2026, 1, 1)}
    with pytest.raises(ValueError, match="timezone"):
        rg.ModelSpec(mode=TrainingMode.POOLED, animal=None, **naive)  # type: ignore[arg-type]
    nan = {**kw, "metrics": {"LOAO": {"f1": float("nan")}}}
    with pytest.raises(ValueError, match="omit it"):
        rg.ModelSpec(mode=TrainingMode.POOLED, animal=None, **nan)  # type: ignore[arg-type]
    pooled = rg.ModelSpec(mode=TrainingMode.POOLED, animal=None, **kw)  # type: ignore[arg-type]
    assert "animal" not in pooled.to_dict()  # absent, never null
    assert rg.ModelSpec.from_dict({**pooled.to_dict(), "animal": None}) == pooled


_metric = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(mode=st.sampled_from(list(TrainingMode)),
       cohort=st.sampled_from(["old", "new"]),
       letter=st.sampled_from("ABCDEFGHLMNOPQRSTUVWXYZ"),
       version=st.text(max_size=8), corpus=st.text(max_size=12),
       mid=st.text(alphabet="0123456789abcdef", min_size=32, max_size=32),
       metrics=st.dictionaries(st.sampled_from(["LOAO", "LOAO_ADAPT", "LORO", "train"]),
                               st.dictionaries(st.text(max_size=6), _metric, max_size=3),
                               max_size=3),
       secs=st.integers(min_value=0, max_value=4_000_000_000),
       n=st.integers(min_value=0, max_value=10**7))
def test_model_spec_round_trips_exactly(mode: TrainingMode, cohort: str, letter: str,
                                        version: str,
                                        corpus: str, mid: str,
                                        metrics: dict[str, dict[str, float]], secs: int,
                                        n: int) -> None:
    animal = f"{cohort}:{letter}"
    spec = rg.ModelSpec(mode=mode, animal=None if mode is TrainingMode.POOLED else animal,
                        version=version, corpus_hash=corpus,
                        calibrator=rg.calibrator_relpath(mid),
                        trained_at=datetime.fromtimestamp(secs, tz=UTC), metrics=metrics,
                        n_train_events=n)
    text = spec.to_json()
    assert text.isascii()
    back = rg.ModelSpec.from_json(text)
    assert back == spec
    assert back.to_json() == text


def test_batch_needs_one_choice_per_animal_and_flags_mixed_modes(
        reg: rg.Registry, booster: lgb.Booster, recording: Recording) -> None:
    pooled = _register(reg, booster, TrainingMode.POOLED, None)
    per_b = _register(reg, booster, TrainingMode.PER_ANIMAL, "new:B")
    recs = [_rec(recording, "A"), _rec(recording, "B")]
    with pytest.raises(rg.NotApplicableError, match="no model chosen"):
        rg.resolve_batch(recs, {"new:A": pooled}, reg, cohort="new", evaluation=False)
    table = rg.resolve_batch(recs, {"new:A": pooled, "new:B": per_b}, reg, cohort="new",
                             evaluation=False)
    assert table["mixed_modes"].all()
    assert list(table["mode"]) == ["pooled", "per_animal"]
    with pytest.raises(rg.NotApplicableError, match="not applicable"):
        rg.resolve_batch(recs, {"new:A": per_b, "new:B": per_b}, reg, cohort="new",
                         evaluation=False)
    same = rg.resolve_batch(recs, {"new:A": pooled, "new:B": pooled}, reg, cohort="new",
                            evaluation=False)
    assert not same["mixed_modes"].any()


def test_shap_top_features_via_detector_review(booster: lgb.Booster) -> None:
    try:
        import_detector_module("review")
    except (FileNotFoundError, ImportError):
        pytest.skip("GEMSBlanking checkout not available")
    top = tr.shap_top_features(booster, _TABLE[_FEATS], top_n=5)
    assert len(top) == 5
    assert top["mean_abs_shap"].is_monotonic_decreasing
    assert isinstance(top, pd.DataFrame)
