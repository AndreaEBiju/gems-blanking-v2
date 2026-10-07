"""Task 12 acceptance 5/6 and 12A: final models registered with provenance; SHAP review."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PureWindowsPath

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
from gems_blanking_v2.model import train as tr
from gems_blanking_v2.model.labels import NEGATIVE_JUDGEMENTS
from gems_blanking_v2.types import TrainingMode

from tests.conftest import FEATURE_SIGNAL, make_feature_table

THREADS = 2


def _rate(t: pd.DataFrame) -> tr.OldRateEstimate:
    """Return the combined old-rate estimate over a test table's own old rows."""
    return tr.old_motion_rate(t)
W = ev.W_ADAPT_GRID
OPT_INS = md.LabelOptIns(allow_model_labels=False, keep_tiers=("1", "2a"))


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, md.ModeRun,
                                                          rg.Registry, Path]:
    root = tmp_path_factory.mktemp("prov")
    record = ev.write_run_record(root / "run_record.json", run_id="unit_run")
    raw = make_feature_table({"new": ("A", "B"), "old": ("F",)}, n_recordings=3,
                             cores_per_recording=120, set_a_per_recording=30, seed=31)
    tiers = {r: ("1" if i % 2 else "2a") for i, r in
             enumerate(sorted(set(raw.loc[raw["cohort"] == "old", "recording"])))}
    table = md.prepare_table(raw)
    reg = rg.Registry(GemsStore.initialise(root / "gems"))
    out = md.run_modes(table, old_rate=_rate(table), targets=["new:A", "new:B"], record_path=record,
                       num_threads=THREADS, w_adapt_grid=W, rounds=30, adapt_rounds=10,
                       registry=reg, user="tester", old_tiers=tiers, label_opt_ins=OPT_INS)
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
        assert {"f1", "precision", "recall", "n_pos", "f1_cal", "precision_cal",
                "recall_cal", "n_cal"} <= set(s.metrics[proto])
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
    pprov = pv.read_provenance(reg.provenance_path(pooled).parent)
    cal = pprov["calibration"]
    assert cal["kind"] == "isotonic" and cal["protocol"] == "LOAO"
    assert cal["targets"] == ["new:A", "new:B"]
    assert "per-target pooled LOAO" in cal["fitted_on"] and "NOT of this final" in cal["fitted_on"]
    assert cal["n_predictions"] == int(table["animal_key"].isin(["new:A", "new:B"]).sum())
    for spec in reg.specs():
        c = pv.read_provenance(reg.provenance_path(spec).parent)["calibration"]
        if spec.mode is not TrainingMode.POOLED:
            assert c["targets"] == [spec.animal]
    assert pprov["label_opt_ins"] == {"allow_model_labels": False, "keep_tiers": ["1", "2a"]}
    assert set(pprov["metric_basis"]) == set(md.METRIC_BASIS)
    assert "raw P(motion)" in pprov["metric_basis"]["f1/precision/recall"]
    assert pprov["metrics_note"].startswith("metrics are LOAO")
    assert pprov["metrics_note"].endswith("new:A, new:B, old:F")
    for spec in reg.specs():
        if spec.mode is not TrainingMode.POOLED:
            assert "metrics_note" not in pv.read_provenance(reg.provenance_path(spec).parent)
    comp = pprov["corpus"]
    for c in comp:
        assert sum(c["label_source"].values()) == c["n_events"]
        assert set(c["label_source"]) == {"human"}
    old = [c for c in comp if c["animal_key"] == "old:F"]
    assert {c["tier"] for c in old} == {"1", "2a", "set_a"}
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
    cal = {"kind": "isotonic", "fitted_on": "x", "protocol": "LOAO", "targets": [],
           "n_predictions": 0}
    with pytest.raises(ValueError, match="w_adapt"):
        pv.build_provenance(pooled, corpus=[], record_path=record, w_adapt=3.0,
                            calibration=cal)
    with pytest.raises(ValueError, match="w_adapt"):
        pv.build_provenance(adapted, corpus=[], record_path=record, w_adapt=None,
                            calibration=cal)
    with pytest.raises(ValueError, match="calibration disclosure"):
        pv.build_provenance(pooled, corpus=[], record_path=record, w_adapt=None,
                            calibration={"kind": "isotonic"})


def test_registration_needs_a_user_and_a_full_calibrated_pass(tmp_path: Path) -> None:
    record = ev.write_run_record(tmp_path / "r.json", run_id="r")
    table = md.prepare_table(make_feature_table({"new": ("A", "B")}, n_recordings=2,
                                                cores_per_recording=20, seed=33))
    reg = rg.Registry(GemsStore.initialise(tmp_path / "gems"))
    kw = {"targets": ["new:A"], "record_path": record, "num_threads": THREADS,
          "registry": reg}
    with pytest.raises(ValueError, match="acting user"):
        md.run_modes(table, old_rate=_rate(table), **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only a full, calibrated pass"):
        md.run_modes(table, old_rate=_rate(table),
                     user="t", train_size=50, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only a full, calibrated pass"):
        md.run_modes(table, old_rate=_rate(table),
                     user="t", calibration=None, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="all three modes over the full"):
        md.run_modes(table, old_rate=_rate(table),
                     user="t", w_adapt_grid=(1.0, 10.0), label_opt_ins=OPT_INS,
                     **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="all three modes over the full"):
        md.run_modes(table, old_rate=_rate(table),
                     user="t", modes=(TrainingMode.POOLED,), label_opt_ins=OPT_INS,
                     **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="label opt-ins"):
        md.run_modes(table, old_rate=_rate(table), user="t", **kw)  # type: ignore[arg-type]
    as_dict: object = {"allow_model_labels": False, "keep_tiers": ["1"]}  # not LabelOptIns
    with pytest.raises(ValueError, match="label opt-ins"):
        md.run_modes(table, old_rate=_rate(table),
                     user="t", label_opt_ins=as_dict, **kw)  # type: ignore[arg-type]
    plain = md.run_modes(table, old_rate=_rate(table),
                         targets=["new:A"], record_path=record, num_threads=THREADS,
                         modes=(TrainingMode.POOLED,), rounds=10)
    assert plain.registered == ()
    assert reg.specs() == []


def test_label_opt_ins_are_checked_against_the_table(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a bool"):
        md.LabelOptIns(allow_model_labels="no", keep_tiers=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="drawn from"):
        md.LabelOptIns(allow_model_labels=False, keep_tiers=("3",))
    for bad in ("1", ["1"], ("1", 2)):
        with pytest.raises(TypeError, match="tuple of str"):
            md.LabelOptIns(allow_model_labels=False, keep_tiers=bad)  # type: ignore[arg-type]
    raw = make_feature_table({"new": ("A", "B"), "old": ("F",)}, n_recordings=2,
                             cores_per_recording=10, set_a_per_recording=40, seed=36)
    old_recs = sorted(set(raw.loc[raw["cohort"] == "old", "recording"]))
    tiers = {old_recs[0]: "1", old_recs[1]: "2b"}
    t = md.prepare_table(raw)
    ok = md.LabelOptIns(allow_model_labels=False, keep_tiers=("1", "2b"))
    ok.check(t, tiers)
    with pytest.raises(ValueError, match="outside keep_tiers"):
        md.LabelOptIns(allow_model_labels=False, keep_tiers=("1",)).check(t, tiers)
    with pytest.raises(ValueError, match="outside keep_tiers"):
        ok.check(t, {old_recs[0]: "1"})  # an old recording with no tier
    mixed = t.copy()
    mixed.loc[mixed.index[0], "label_source"] = "model"
    with pytest.raises(ValueError, match="label_source"):
        ok.check(mixed, tiers)
    md.LabelOptIns(allow_model_labels=True, keep_tiers=("1", "2b")).check(mixed, tiers)
    record = ev.write_run_record(tmp_path / "r.json", run_id="r")
    reg = rg.Registry(GemsStore.initialise(tmp_path / "gems"))
    with pytest.raises(ValueError, match="outside keep_tiers"):
        md.run_modes(t, old_rate=_rate(t),
                     targets=["new:A"], record_path=record, num_threads=THREADS,
                     registry=reg, user="t", old_tiers=tiers,
                     label_opt_ins=md.LabelOptIns(allow_model_labels=False, keep_tiers=("1",)))


def test_a_final_model_that_cannot_be_fitted_records_a_refusal(tmp_path: Path) -> None:
    raw = make_feature_table({"new": ("A", "B")}, n_recordings=3, cores_per_recording=30,
                             seed=37)
    b_recs = sorted(set(raw.loc[raw["animal"] == "B", "recording"]))
    raw = raw[~raw["recording"].isin(b_recs[1:])]  # B keeps one recording: C and B refuse
    t = md.prepare_table(raw)
    record = ev.write_run_record(tmp_path / "r.json", run_id="r")
    reg = rg.Registry(GemsStore.initialise(tmp_path / "gems"))
    out = md.run_modes(t, old_rate=_rate(t), targets=["new:A", "new:B"], record_path=record,
                       num_threads=THREADS, rounds=10, adapt_rounds=5, registry=reg,
                       user="t", label_opt_ins=md.LabelOptIns(False, ()))
    final = {(r.mode, r.target) for r in out.refusals if r.fold == "final"}
    assert ("per_animal", "new:B") in final
    assert ("adapted", "new:B") in final
    assert ("per_animal", "new:A") not in final
    kinds = {(str(s.mode), s.animal) for s in reg.specs()}
    assert ("per_animal", "new:B") not in kinds and ("per_animal", "new:A") in kinds


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
    res = rg.run_inference(rec_a, listed[0], reg, cohort="new", cores=table, chosen_by="t",
                           evaluation=False)
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
    _t, _o, reg, _r = run
    pooled = next(s for s in reg.specs() if s.mode is TrainingMode.POOLED)
    # unseen judged cores whose classes overlap, so some negatives are called motion: six
    # judged negatives are given the motion signature outright, so the disagreement set is
    # never empty by the luck of a seed
    raw = make_feature_table({"new": ("A",)}, n_recordings=2, cores_per_recording=60,
                             separation=0.5, seed=35)
    neg = raw.index[raw["y"] == 0][:6]
    raw.loc[neg, list(FEATURE_SIGNAL)] += 3.0
    cores = md.prepare_table(raw)
    fs = dict.fromkeys(cores["recording"].astype(str), 24414.0625)
    p = reg.calibrator(pooled).apply(tr.predict_raw(reg.booster(pooled), cores[
        tr.feature_columns(cores)]))
    n_dis = int((cores["judgement"].isin(NEGATIVE_JUDGEMENTS).to_numpy()
                 & (p >= 0.5)).sum())
    k = min(5, n_dis)
    assert k >= 1
    out = sr.write_shap_review(reg, pooled, cores, fs=fs, top_k=5, top_n=7)
    assert out == reg.store.model_dir(pooled.model_id) / "shap"
    page = (out / sr.REVIEW_NAME).read_bytes()
    assert b"\r\n" not in page
    text = page.decode("utf-8")
    assert "Disagreement review" in text and pooled.model_id in text
    assert text.count("Top SHAP contributions") == k  # judged negatives called motion
    assert f"Disagreement {k} / {k}" in text
    top = (out / sr.TOP_FEATURES_NAME).read_text(encoding="utf-8")
    assert top.count("<tr><td>") == 7
    assert "band_ratio" in top or "frac_signals_over" in top  # a synthetic signal feature
    assert not list(out.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        sr.write_shap_review(reg, pooled, cores, fs=fs)


@pytest.mark.skipif(not _detector_review_available(), reason="GEMSBlanking not available")
def test_shap_review_refuses_unjudged_and_test_cores(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    table, _o, reg, _r = run
    per = next(s for s in reg.specs() if s.mode is TrainingMode.PER_ANIMAL)
    fs = dict.fromkeys(table["recording"].astype(str), 1000.0)
    for j in ("unjudged", "unsure"):
        bad = table.copy()
        bad.loc[bad.index[0], "judgement"] = j
        with pytest.raises(ValueError, match="not judged"):
            sr.write_shap_review(reg, per, bad, fs=fs)
    with pytest.raises(ValueError, match="judgement column"):
        sr.write_shap_review(reg, per, table.drop(columns="judgement"), fs=fs)
    test = table.copy()
    test.loc[test["animal"] == "B", "animal"] = "J"
    with pytest.raises(ValueError, match="prospective test set"):
        sr.write_shap_review(reg, per, test, fs=fs)
    assert not (reg.store.model_dir(per.model_id) / "shap").exists()


def test_review_samples_convert_half_open_seconds_to_one_based_samples() -> None:
    # [10.0, 10.2) s at 1 kHz: 0-based samples 10000..10199; centre 10100 -> 1-based 10101;
    # context [8.0, 12.2) s = 0-based [8000, 12200) = 1-based inclusive [8001, 12200]
    assert sr.review_samples(10.0, 10.2, 1000.0) == (10101, 8001, 12200)
    assert sr.review_samples(0.5, 0.6, 1000.0)[1] == 1  # clipped at the first sample
    # odd-length cores: the middle sample, never round-half-to-even of a midpoint
    assert sr.review_samples(0.0, 0.003, 1000.0)[0] == 2  # samples 0,1,2 -> middle 1
    assert sr.review_samples(5.0, 5.011, 1000.0)[0] == 5006  # 5000..5010 -> middle 5005
    assert sr.review_samples(5.0, 5.012, 1000.0)[0] == 5007  # even: upper middle 5006
    fs = 24414.0625
    c, a, b = sr.review_samples(100.0, 100.5, fs)
    s0, s1 = round(100.0 * fs), round(100.5 * fs)
    assert c == (s0 + s1) // 2 + 1
    assert (a, b) == (s0 - round(2.0 * fs) + 1, s1 + round(2.0 * fs))
    # on the 10 ms grid at 24414.0625 Hz exact half-sample midpoints occur
    for k in range(1, 400):
        lo, hi = 0.01 * k, 0.01 * k + 0.03
        m0, m1 = round(lo * fs), round(hi * fs)
        assert sr.review_samples(lo, hi, fs)[0] - 1 in range(m0, m1)
        assert sr.review_samples(lo, hi, fs)[0] - 1 == (m0 + m1) // 2


@pytest.mark.skipif(not _detector_review_available(), reason="GEMSBlanking not available")
def test_a_failed_shap_build_leaves_nothing_and_can_retry(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path],
        monkeypatch: pytest.MonkeyPatch) -> None:
    table, _o, reg, _r = run
    target = next(s for s in reg.specs() if s.mode is TrainingMode.ADAPTED)
    fs = dict.fromkeys(table["recording"].astype(str), 1000.0)
    review = import_detector_module("review")

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(review, "generate_review_html", boom)
    with pytest.raises(RuntimeError, match="generation failed"):
        sr.write_shap_review(reg, target, table, fs=fs, top_k=2)
    mdir = reg.store.model_dir(target.model_id)
    assert not (mdir / "shap").exists()
    assert not list(mdir.glob("shap.building-*"))
    monkeypatch.undo()
    other = mdir / "shap.building-otherhost-99999"  # another machine's build in progress
    other.mkdir()
    (other / "partial.html").write_text("x", encoding="utf-8")
    out = sr.write_shap_review(reg, target, table, fs=fs, top_k=2)
    assert (out / sr.REVIEW_NAME).is_file()
    assert (other / "partial.html").is_file()  # never touched


@pytest.mark.skipif(not _detector_review_available(), reason="GEMSBlanking not available")
def test_an_incomplete_shap_build_is_never_moved_into_place(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path],
        monkeypatch: pytest.MonkeyPatch) -> None:
    table, _o, reg, _r = run
    target = next(s for s in reg.specs() if s.mode is TrainingMode.PER_ANIMAL)
    fs = dict.fromkeys(table["recording"].astype(str), 1000.0)
    monkeypatch.setattr(sr, "atomic_write_text", lambda *_a, **_k: None)  # top page lost
    with pytest.raises(RuntimeError, match="lacks"):
        sr.write_shap_review(reg, target, table, fs=fs, top_k=2)
    mdir = reg.store.model_dir(target.model_id)
    assert not (mdir / "shap").exists()
    assert not list(mdir.glob("shap.building-*"))


def test_the_deepest_shap_build_path_fits_under_a_windows_style_root(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.node", lambda: "a-very-long-workstation-name.lab.example")
    monkeypatch.setattr("os.getpid", lambda: 4294967295)  # the largest Windows PID
    store = GemsStore(PureWindowsPath(  # type: ignore[arg-type]
        r"G:\Shared drives\BIONICs Lab_ Enteric Interfaces Team"))
    out_dir = store.model_dir("a" * 32) / "shap"
    build = sr.build_dir(out_dir)
    host = build.name.removeprefix("shap.building-").rsplit("-", 1)[0]
    assert len(host) <= sr.HOST_CHARS
    deepest = build / sr.DEEPEST_LEAF
    assert len(sr.DEEPEST_LEAF) == len(f".{sr.TOP_FEATURES_NAME}.") + 8 + len(".tmp")
    assert store.check_path_length(deepest) is None
    assert len(str(deepest)) < 260
    long_store = GemsStore(PureWindowsPath("G:\\" + "x" * 200))  # type: ignore[arg-type]
    msg = long_store.check_path_length(sr.build_dir(long_store.model_dir("a" * 32) / "shap")
                                       / (sr.REVIEW_NAME + ".raw"))
    assert msg is not None and "Windows limit" in msg


def test_the_deepest_leaf_bounds_the_real_atomic_write_temp_name(tmp_path: Path) -> None:
    name = sr.TOP_FEATURES_NAME
    fd, tmp = tempfile.mkstemp(dir=tmp_path, prefix=f".{name}.", suffix=".tmp")
    os.close(fd)
    assert len(Path(tmp).name) <= len(sr.DEEPEST_LEAF)


def test_hosts_sharing_a_long_prefix_get_different_build_names(
        monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = Path("models") / ("a" * 32) / "shap"
    names = set()
    for host in ("Andrea-MacBook-Pro.local", "Andrea-MacBook-Pro-2.local", "x"):
        monkeypatch.setattr("platform.node", lambda h=host: h)
        part = sr.build_dir(out_dir).name.removeprefix("shap.building-").rsplit("-", 1)[0]
        assert len(part) <= sr.HOST_CHARS
        names.add(part)
    assert len(names) == 3


@pytest.mark.skipif(not _detector_review_available(), reason="GEMSBlanking not available")
def test_an_over_long_build_path_fails_clearly_and_writes_nothing(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path],
        monkeypatch: pytest.MonkeyPatch) -> None:
    table, _o, reg, _r = run
    target = [s for s in reg.specs() if s.mode is TrainingMode.ADAPTED][-1]
    out_dir = reg.store.model_dir(target.model_id) / "shap"
    limit = len(str(sr.build_dir(out_dir) / sr.DEEPEST_LEAF)) - 1  # one char too short
    monkeypatch.setattr("gems_blanking_v2.io.store.WINDOWS_MAX_PATH", limit)
    fs = dict.fromkeys(table["recording"].astype(str), 1000.0)
    with pytest.raises(ValueError, match="Windows limit"):
        sr.write_shap_review(reg, target, table, fs=fs, top_k=2)
    mdir = reg.store.model_dir(target.model_id)
    assert not (mdir / "shap").exists()
    assert not list(mdir.glob("shap.building-*"))


@pytest.mark.skipif(not _detector_review_available(), reason="GEMSBlanking not available")
def test_shap_review_reads_fs_never_assumes_it(
        run: tuple[pd.DataFrame, md.ModeRun, rg.Registry, Path]) -> None:
    table, _o, reg, _r = run
    per = next(s for s in reg.specs() if s.mode is TrainingMode.PER_ANIMAL)
    with pytest.raises(ValueError, match="no fs"):
        sr.write_shap_review(reg, per, table, fs={})
