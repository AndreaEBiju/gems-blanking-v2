"""Night 6 builds stomach_ref from the mask provenance's recipe (Andrea, 2026-10-12).

``matlab/night6/night6_signal_recipe.m`` reads ``stomach_ref_recipe`` from the mask file's
provenance and refuses by name when it is absent or incomplete, when it names a notch,
when its weights are not its form's, or when its contacts or config are not the file's
stomach channels; ``night6_apply_recipe.m`` applies it in the derivation's order of
operations. Checked against Python, in ONE MATLAB process
(``tests/matlab/check_stomach_ref.m``), on a seeded ``make_multichannel`` recording:

* the MATLAB build equals ``build_stomach_reference`` (and the recorded recipe applied in
  Python) sample for sample - bit for bit, NaN samples included - and the hrv run's input
  is that signal in volts with NaN exactly on the hrv mask;
* stomach_ref is formed from the stomach columns only, never a nerve channel;
* every incomplete or inconsistent recipe is refused by name;
* THE GATE FIX: recipes are formed for the signals of consumers that run, only. A file
  whose QC-held hrv/breathing read a stomach_ref it carries no recipe for (K t01 cme1's
  case) still plans its mmc; the same file with hrv/breathing not held is refused.

``GEMS_STOMACHREF_REAL`` (a ``real_case_<session>.json`` written by
``stomach_ref_proof.py``) adds the same comparison on that real recording - opt-in, since
it reads the shared drive. Skips, with the reason, when MATLAB is absent.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.derive import stomach_recipe as sr
from gems_blanking_v2.derive.derivations import build_stomach_reference
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit.handoff import write_mask_file
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.extent.grid import n_grid_frames, seconds_to_sample
from gems_blanking_v2.types import Recording
from scipy.io import loadmat, savemat

from tests.conftest import make_multichannel
from tests.test_matlab_acceptance import _matlab

FS = 24414.0625
DUR_S = 6.0
NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"
GASTRIC = ("ANT1", "ANT2", "ANT3")
READS: dict[str, tuple[str, ...]] = {"spikes": (), "slow_wave": (), "mmc": GASTRIC,
                                     "hrv": ("stomach_ref",), "breathing": ("stomach_ref",),
                                     "velocity": ()}
SPANS = [mk.MaskSpan("hrv", "stomach_ref", 1.0, 1.25, "x"),
         mk.MaskSpan("hrv", "stomach_ref", 4.0, 4.1, "x"),
         mk.MaskSpan("breathing", "stomach_ref", 2.0, 2.5, "x"),
         mk.MaskSpan("mmc", "ANT2", 3.0, 3.2, "x")]
NAN_AT = ((100, ("ANT2",)), (200, ("ANT1",)), (300, GASTRIC))
"""Samples made NaN in Ynan: one contact, the positive input, every contact."""
REQUIRED = ("schema", "how", "config", "contacts", "positive", "form", "weights", "notch")


def _rec() -> Recording:
    rec, _truth = make_multichannel(FS, DUR_S, seed=5)
    return rec


def _meta(rec: Recording) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in rec.channels:
        ch: dict[str, Any] = {"label": c.name, "signal_index": c.index, "config": c.config,
                              "role": c.role}
        if c.role == "nerve":
            ch.update(cuff_id=c.cuff_id, contact_index=c.contact_index)
        out.append(ch)
    return out


def _with_nans(rec: Recording) -> Recording:
    data = rec.data.copy()
    col = {c.name: c.index for c in rec.channels}
    for row, names in NAN_AT:
        for name in names:
            data[row, col[name]] = np.nan
    return replace(rec, data=data)


def _prov(rec: Recording) -> MaskProvenance:
    return MaskProvenance(
        model={"mode": "pooled", "version": "t", "corpus_hash": "0" * 32,
               "calibrator": "models/t/cal.json"},
        thresholds={"source": "synthetic"}, reference_values={"stomach_ref|0.5-3": {"m": 1}},
        code_commit="test", generation_sha="0133349b3ebeff80", routing_hash="test",
        created_at="2026-10-12T05:00:00+00:00", recording=rec.session)


def _edit(src: Path, dst: Path, *, provenance: dict[str, Any] | None = None,
          held: dict[str, Any] | None = None) -> None:
    doc = {k: v for k, v in loadmat(src).items() if not k.startswith("__")}
    if provenance is not None:
        doc["provenance_json"] = json.dumps(provenance, sort_keys=True, ensure_ascii=True,
                                            separators=(",", ":"))
    if held is not None:
        doc["heldconsumers_json"] = json.dumps(held, sort_keys=True)
    savemat(dst, doc, do_compression=True)


def _cases(root: Path, rec: Recording) -> dict[str, Any]:
    """Write inputs.mat and every case's mask file; return what Python expects."""
    n = rec.n_samples
    cases = root / "cases"
    cases.mkdir(parents=True)
    recipe = sr.stomach_ref_recipe(rec, {})
    masks = mk.build_masks(READS, SPANS, n_frames=n_grid_frames(n, FS), t0_s=0.0)
    ok = cases / "ok.mat"
    write_mask_file(ok, masks, _prov(rec), signals=READS, fs=FS, n_samples=n,
                    epoch_start_s=0.0, min_retention=0.5, animal_median={},
                    line_distrust=None, peri_r=None,
                    not_computed={"slow_wave": "synthetic: not computed"},
                    elsewhere={"spikes": "synthetic: finished per animal in masks/m_spikes"},
                    epoch_start_sample=seconds_to_sample(0.0, FS),
                    stomach_ref_recipe=recipe, release="synthetic: spans large on purpose")
    prov = json.loads(str(loadmat(ok)["provenance_json"][0]))
    bare = {k: v for k, v in prov.items() if k != "stomach_ref_recipe"}
    held = {c: {"reasons": [f"{c}|stomach_ref|band: synthetic hold"]}
            for c in ("hrv", "breathing")}
    _edit(ok, cases / "norecipe.mat", provenance=bare)
    _edit(ok, cases / "held_norecipe.mat", provenance=bare, held=held)
    mutants: dict[str, dict[str, Any]] = {
        "notch": {"notch": {"freqs_hz": [60.0], "q": 30.0, "contacts": ["ANT1"]}},
        "weights_dot": {"weights": [0.6666666666666666, -0.3333333333333333,
                                    -0.3333333333333333]},
        "weights_swapped": {"weights": [-1.0 / 3, 1.0 - 1.0 / 3, -1.0 / 3]},
        "contacts_order": {"contacts": ["ANT1", "ANT3", "ANT2"]},
        "contacts_nerve": {"contacts": ["ANT1", "LVN2", "ANT3"]},
        "positive": {"positive": "ANT2"},
        "config": {"config": "hw_tripole"},
        "form": {"form": "positive_minus_mean"},
        "schema": {"schema": "stomach_ref_recipe/2"},
    }
    for name, change in mutants.items():
        _edit(ok, cases / f"{name}.mat",
              provenance={**prov, "stomach_ref_recipe": {**recipe, **change}})
    for key in REQUIRED:
        _edit(ok, cases / f"lacks_{key}.mat", provenance={**prov, "stomach_ref_recipe": {
            k: v for k, v in recipe.items() if k != key}})
    nan_rec = _with_nans(rec)
    savemat(root / "inputs.mat", {"Y": rec.data, "Ynan": nan_rec.data,
                                  "labels": np.array([c.name for c in rec.channels],
                                                     dtype=object),
                                  "fs": FS, "n": float(n),
                                  "meta_json": json.dumps(_meta(rec))})
    return {"recipe": recipe, "masks": masks, "nan_rec": nan_rec, "mutants": sorted(mutants)}


def _run(matlab: Path, cmd: str) -> None:
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=1800, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


@pytest.fixture(scope="module")
def night6(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    root = tmp_path_factory.mktemp("stomach_ref")
    rec = _rec()
    want = _cases(root, rec)
    out, out_mat = root / "result.json", root / "x.mat"
    _run(matlab, f"addpath('{NIGHT6.as_posix()}'); addpath('{HARNESS.as_posix()}'); "
                 f"check_stomach_ref('{root.as_posix()}', '{out.as_posix()}', "
                 f"'{out_mat.as_posix()}');")
    res = json.loads(out.read_text(encoding="utf-8"))
    arrays = loadmat(out_mat) if out_mat.is_file() else {}
    return {"rec": rec, "res": res, "arrays": arrays, **want}


def test_the_matlab_stomach_ref_is_the_python_derivation_sample_for_sample(
        night6: dict[str, Any]) -> None:
    rec: Recording = night6["rec"]
    assert night6["res"]["ok"]["error_id"] == "", night6["res"]["ok"]["message"]
    want, how = build_stomach_reference(rec)
    assert how == "common_average" and want is not None
    x = np.asarray(night6["arrays"]["x"], dtype=np.float64).ravel()
    assert x.shape == want.shape
    assert np.array_equal(x, want)                                   # bit for bit
    cols = {c.name: rec.data[:, c.index] for c in rec.channels if c.role == "stomach"}
    assert np.array_equal(x, sr.apply_stomach_ref_recipe(night6["recipe"], cols, FS))


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_nan_samples_match_too(night6: dict[str, Any]) -> None:
    want, _how = build_stomach_reference(night6["nan_rec"])
    x = np.asarray(night6["arrays"]["x_nan"], dtype=np.float64).ravel()
    assert want is not None and np.array_equal(x, want, equal_nan=True)
    assert np.isfinite(x[100]) and np.isnan(x[200]) and np.isnan(x[300])
    assert int(np.isnan(x).sum()) == 2


def test_the_hrv_input_is_that_signal_in_volts_nan_on_its_own_mask_only(
        night6: dict[str, Any]) -> None:
    rec: Recording = night6["rec"]
    want, _how = build_stomach_reference(rec)
    assert want is not None
    x = np.asarray(night6["arrays"]["X_hrv"], dtype=np.float64).ravel()
    m = night6["masks"][("hrv", "stomach_ref", "10-150")]
    nan = np.zeros(rec.n_samples, dtype=bool)
    for a, b in mk.mask_sample_spans(m.invalid, FS, rec.n_samples, m.grid_s):
        nan[a:b] = True
    assert np.array_equal(np.isnan(x), nan)
    assert np.array_equal(x[~nan], want[~nan] * 1e-6)   # declared uV -> volts, as Night 6


def test_stomach_ref_reads_the_stomach_columns_only(night6: dict[str, Any]) -> None:
    rec: Recording = night6["rec"]
    stomach = [c.index + 1 for c in rec.channels if c.role == "stomach"]   # 1-based
    assert night6["res"]["ok"]["stomach_cols"] == stomach
    assert night6["res"]["contacts_nerve"]["error_id"] == "night6:stomachRefContacts"


def test_a_mask_file_without_the_recipe_is_refused_by_name(night6: dict[str, Any]) -> None:
    r = night6["res"]["norecipe"]
    assert r["error_id"] == "night6:stomachRefRecipe"
    assert "carries no stomach_ref_recipe" in r["message"]


@pytest.mark.parametrize("key", REQUIRED)
def test_a_recipe_missing_a_field_is_refused_naming_it(night6: dict[str, Any], key: str) -> None:
    r = night6["res"][f"lacks_{key}"]
    assert r["error_id"] == "night6:stomachRefRecipe"
    assert key in r["message"]


@pytest.mark.parametrize(("case", "error_id"), [
    ("notch", "night6:stomachRefNotch"),
    ("weights_dot", "night6:stomachRefWeights"),
    ("weights_swapped", "night6:stomachRefWeights"),
    ("contacts_order", "night6:stomachRefContacts"),
    ("contacts_nerve", "night6:stomachRefContacts"),
    ("positive", "night6:stomachRefContacts"),
    ("config", "night6:stomachRefConfig"),
    ("form", "night6:stomachRefForm"),
    ("schema", "night6:stomachRefRecipe"),
])
def test_an_inconsistent_recipe_is_refused(night6: dict[str, Any], case: str,
                                           error_id: str) -> None:
    assert night6["res"][case]["error_id"] == error_id, night6["res"][case]["message"]


def test_a_held_consumers_signal_never_blocks_the_consumers_that_run(
        night6: dict[str, Any]) -> None:
    """K t01 cme1: hrv/breathing QC-held on stomach_ref; its mmc must still be planned."""
    r = night6["res"]["held_norecipe"]
    assert r["error_id"] == "", r["message"]
    assert r["statuses"] == {"hrv": "skipped_held", "breathing": "skipped_held",
                             "mmc": "to_run"}
    assert r["recipes"] == ["ANT1", "ANT2", "ANT3"]           # no stomach_ref formed
    assert night6["res"]["ok"]["recipes"] == ["stomach_ref", "ANT1", "ANT2", "ANT3"]


def test_recipes_are_formed_after_the_runs_from_the_runs() -> None:
    prep = (NIGHT6 / "night6_prepare_epoch.m").read_text(encoding="utf-8")
    body = prep[prep.index("plan.runs = struct('call'"):]
    body = body[:body.index("\nend\n")]
    assert "for k = 1:numel(plan.runs)" in body
    assert "night6_signal_recipe(s{1}, fileLabels, channels, srRecipe)" in body
    assert prep.count("night6_signal_recipe(") == 1     # one construction site
    inp = (NIGHT6 / "night6_consumer_input.m").read_text(encoding="utf-8")
    assert "night6_apply_recipe(r, Y, rows) * plan.scaleToVolts" in inp


# ---------------------------------------------------------------------------
# opt-in: a real recording, re-versioned by stomach_ref_proof.py
# ---------------------------------------------------------------------------

def test_the_matlab_stomach_ref_equals_python_on_a_real_recording(tmp_path: Path) -> None:
    case_file = os.environ.get("GEMS_STOMACHREF_REAL")
    if not case_file:
        pytest.skip("GEMS_STOMACHREF_REAL names no real_case_<session>.json (opt-in: it "
                    "reads the shared drive)")
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    from gems_blanking_v2.io.detector_core import import_detector_module  # noqa: PLC0415
    from gems_blanking_v2.io.recording import load_recording  # noqa: PLC0415
    from gems_blanking_v2.io.store import GemsStore  # noqa: PLC0415

    case = json.loads(Path(case_file).read_text(encoding="utf-8"))
    src = Path(case["gems_root"]) / case["source_path"]
    loaded = load_recording(src, case["animal"], store=GemsStore(Path(case["gems_root"])))
    units = loaded.provenance["declared_units"]
    native = import_detector_module("recording_io").load_recording(src)
    file_units = np.asarray(native.y, dtype=np.float64)          # the file's own numbers
    rec = replace(loaded.recording, data=file_units)
    del loaded
    m = loadmat(case["new_mask"], variable_names=["epochStartSample0", "nSamples",
                                                  "provenance_json"])
    i0 = int(m["epochStartSample0"].ravel()[0])
    n = int(m["nSamples"].ravel()[0])
    sub = replace(rec, data=rec.data[i0:i0 + n])
    want, _how = build_stomach_reference(sub)
    recipe = json.loads(str(m["provenance_json"][0]))["stomach_ref_recipe"]
    out, out_mat = tmp_path / "real.json", tmp_path / "x_real.mat"
    _run(matlab, f"addpath('{NIGHT6.as_posix()}'); addpath('{HARNESS.as_posix()}'); "
                 f"check_stomach_ref_real('{Path(case_file).as_posix()}', '{units}', "
                 f"'{out.as_posix()}', '{out_mat.as_posix()}');")
    res = json.loads(out.read_text(encoding="utf-8"))
    x = np.asarray(loadmat(out_mat)["x"], dtype=np.float64).ravel()
    assert want is not None and np.array_equal(x, want)              # bit for bit
    cols = {c.name: sub.data[:, c.index] for c in sub.channels if c.role == "stomach"}
    assert np.array_equal(x, sr.apply_stomach_ref_recipe(recipe, cols, float(rec.fs)))
    assert res["same_fields"] and res["differing"] == []
    assert res["labels"] == ["ANT1", "ANT2", "ANT3"]
