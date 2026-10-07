"""Task 12 acceptance 5/6 and 12A: final models registered with provenance; SHAP review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.io.registry_log import RegistryAction, read_events, replay
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.model import evaluate as ev
from gems_blanking_v2.model import modes as md
from gems_blanking_v2.model import provenance as pv
from gems_blanking_v2.model import registry as rg
from gems_blanking_v2.model import shap_review as sr
from gems_blanking_v2.types import TrainingMode

from tests.conftest import make_feature_table

THREADS = 2
W = (1.0, 10.0)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, md.ModeRun,
                                                          rg.Registry, Path]:
    root = tmp_path_factory.mktemp("prov")
    record = ev.write_run_record(root / "run_record.json", run_id="unit_run")
    raw = make_feature_table({"new": ("A", "B"), "old": ("F",)}, n_recordings=3,
                             cores_per_recording=40, seed=31)
    tiers = {r: ("1" if i % 2 else "2a") for i, r in
             enumerate(sorted(set(raw.loc[raw["cohort"] == "old", "recording"])))}
    table = md.prepare_table(raw)
    reg = rg.Registry(GemsStore.initialise(root / "gems"))
    out = md.run_modes(table, targets=["new:A", "new:B"], record_path=record,
                       num_threads=THREADS, w_adapt_grid=W, rounds=30, adapt_rounds=10,
                       registry=reg, user="tester", old_tiers=tiers)
    return table, out, reg, record


def test_every_evaluated_mode_registers_a_final_model(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    table, out, reg, _record = run
    specs = {s.model_id: s for s in reg.specs()}
    assert set(out.registered) == set(specs)
    kinds = sorted((str(s.mode), s.animal or "") for s in specs.values())
    assert kinds == sorted([("pooled", "")] + [("per_animal", t) for t in ("new:A", "new:B")]
                           + [("adapted", t) for t in ("new:A", "new:B") for _w in W])
    pooled = next(s for s in specs.values() if s.mode is TrainingMode.POOLED)
    assert pooled.n_train_events == len(table)
    for s in specs.values():
        (proto,) = s.metrics
        assert proto == {"pooled": "LOAO", "adapted": "LOAO_ADAPT",
                         "per_animal": "LORO"}[str(s.mode)]
        assert {"f1", "precision", "recall", "n_pos"} <= set(s.metrics[proto])
        assert not s.unvalidated
        assert s.version == "unit_run"


def test_registered_models_are_never_promoted_and_none_is_a_default(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    _t, _o, reg, _r = run
    state = replay(read_events(reg.store))
    assert {st.status for st in state.values()} == {RegistryAction.TRAINED}
    for mode in TrainingMode:
        for animal in (None, "new:A", "new:B"):
            assert reg.promoted_label(mode, animal) is None


def test_provenance_json_carries_spec_corpus_thresholds_record_and_code(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    table, _o, reg, record = run
    rec_bytes = record.read_bytes()
    rec = json.loads(rec_bytes)
    for spec in reg.specs():
        path = reg.provenance_path(spec)
        text = path.read_text(encoding="utf-8")
        assert "\r" not in text and text.isascii()
        assert "null" not in text and "NaN" not in text
        prov = pv.read_provenance(path.parent)
        assert rg.ModelSpec.from_dict(prov["model"]) == spec
        assert prov["model_id"] == spec.model_id
        assert prov["run_record"] == {"run_id": "unit_run",
                                      "sha256": hashlib.sha256(rec_bytes).hexdigest()}
        assert prov["thresholds"] == {k: rec[k] for k in ev.run_protocol()}
        assert ("w_adapt" in prov) == (spec.mode is TrainingMode.ADAPTED)
        if spec.mode is TrainingMode.ADAPTED:
            assert prov["w_adapt"] in W
        assert sum(c["n_events"] for c in prov["corpus"]) == spec.n_train_events
        for c in prov["corpus"]:
            assert ("tier" in c) == c["animal_key"].startswith("old:")
            assert c["n_motion"] + c["n_not_motion"] == c["n_events"]
        assert "package_sha256" in prov["code"]
    pooled = next(s for s in reg.specs() if s.mode is TrainingMode.POOLED)
    comp = pv.read_provenance(reg.provenance_path(pooled).parent)["corpus"]
    old = [c for c in comp if c["animal_key"] == "old:F"]
    assert {c["tier"] for c in old} == {"1", "2a"}
    assert sum(c["n_events"] for c in old) == int((table["cohort"] == "old").sum())


def test_corpus_composition_refuses_an_old_recording_without_a_tier() -> None:
    raw = make_feature_table({"new": ("A",), "old": ("F",)}, n_recordings=2,
                             cores_per_recording=5, seed=32)
    with pytest.raises(ValueError, match="without a tier"):
        pv.corpus_composition(raw, {})
    new_only = raw[raw["cohort"] == "new"]
    assert all("tier" not in c for c in pv.corpus_composition(new_only, None))


def test_provenance_is_write_once_and_refuses_missing_values(tmp_path: Path) -> None:
    pv.write_provenance(tmp_path, {"a": 1})
    pv.write_provenance(tmp_path, {"a": 1})  # identical: fine
    with pytest.raises(ValueError, match="write-once"):
        pv.write_provenance(tmp_path, {"a": 2})
    for bad in ({"a": None}, {"a": [1.0, float("nan")]}, {"a": {"b": float("inf")}}):
        with pytest.raises(ValueError, match="absent key"):
            pv.write_provenance(tmp_path / "x", bad)


def test_w_adapt_is_present_exactly_for_mode_b(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    _t, _o, reg, record = run
    pooled = next(s for s in reg.specs() if s.mode is TrainingMode.POOLED)
    adapted = next(s for s in reg.specs() if s.mode is TrainingMode.ADAPTED)
    with pytest.raises(ValueError, match="w_adapt"):
        pv.build_provenance(pooled, corpus=[], record_path=record, w_adapt=3.0)
    with pytest.raises(ValueError, match="w_adapt"):
        pv.build_provenance(adapted, corpus=[], record_path=record, w_adapt=None)


def test_registration_needs_a_user_and_a_full_calibrated_pass(tmp_path: Path) -> None:
    record = ev.write_run_record(tmp_path / "r.json", run_id="r")
    table = md.prepare_table(make_feature_table({"new": ("A", "B")}, n_recordings=2,
                                                cores_per_recording=20, seed=33))
    reg = rg.Registry(GemsStore.initialise(tmp_path / "gems"))
    kw = {"targets": ["new:A"], "record_path": record, "num_threads": THREADS,
          "registry": reg}
    with pytest.raises(ValueError, match="acting user"):
        md.run_modes(table, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only a full, calibrated pass"):
        md.run_modes(table, user="t", train_size=50, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only a full, calibrated pass"):
        md.run_modes(table, user="t", calibration=None, **kw)  # type: ignore[arg-type]
    plain = md.run_modes(table, targets=["new:A"], record_path=record, num_threads=THREADS,
                         modes=(TrainingMode.POOLED,), rounds=10)
    assert plain.registered == ()
    assert reg.specs() == []


def test_a_registered_model_is_applicable_and_runs(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    from gems_blanking_v2.constants import FS_NOMINAL_HZ  # noqa: PLC0415

    from tests.conftest import make_multichannel  # noqa: PLC0415

    table, _o, reg, _r = run
    rec, _truth = make_multichannel(FS_NOMINAL_HZ, 2.0, seed=34)
    import dataclasses  # noqa: PLC0415

    rec_a = dataclasses.replace(rec, animal="A")
    listed = rg.list_applicable(rec_a, reg, cohort="new")
    assert {str(s.mode) for s in listed} == {"pooled", "adapted", "per_animal"}
    assert all(s.animal in (None, "new:A") for s in listed)
    res = rg.run_inference(rec_a, listed[0], reg, cohort="new", cores=table, chosen_by="t")
    assert np.isfinite(res.p_motion).all()


def _detector_review_available() -> bool:
    try:
        import_detector_module("review")
    except (FileNotFoundError, ImportError):
        return False
    return True


@pytest.mark.skipif(not _detector_review_available(), reason="GEMSBlanking not available")
def test_shap_review_html_reuses_detector_review(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    table, _o, reg, _r = run
    pooled = next(s for s in reg.specs() if s.mode is TrainingMode.POOLED)
    fs = dict.fromkeys(table["recording"].astype(str), 24414.0625)
    # every core judged "not motion": each one the model calls motion is a disagreement
    cores = table.assign(y=0)
    out = sr.write_shap_review(reg, pooled, cores, fs=fs, top_k=5, top_n=7)
    assert out == reg.store.model_dir(pooled.model_id) / "shap"
    page = (out / sr.REVIEW_NAME).read_bytes()
    assert b"\r\n" not in page
    text = page.decode("utf-8")
    assert "Disagreement review" in text and pooled.model_id in text
    assert text.count("Top SHAP contributions") == 5  # top_k disagreements, each with SHAP
    assert "Disagreement 5 / 5" in text
    top = (out / sr.TOP_FEATURES_NAME).read_text(encoding="utf-8")
    assert top.count("<tr><td>") == 7
    assert "band_ratio" in top or "frac_signals_over" in top  # a synthetic signal feature
    assert not list(out.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        sr.write_shap_review(reg, pooled, table, fs=fs)


@pytest.mark.skipif(not _detector_review_available(), reason="GEMSBlanking not available")
def test_shap_review_reads_fs_never_assumes_it(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    table, _o, reg, _r = run
    per = next(s for s in reg.specs() if s.mode is TrainingMode.PER_ANIMAL)
    with pytest.raises(ValueError, match="no fs"):
        sr.write_shap_review(reg, per, table, fs={})
