"""Every Night 6 result records the sha256 of the stomach_ref_recipe it applied, or "none".

Andrea, 2026-10-12. One canonical form across the MATLAB/Python boundary (invariant 22):
the recipe's canonical JSON bytes (sorted keys, ASCII escapes, "," and ":"), exactly as
``MaskProvenance.to_json`` writes them inside ``provenance_json``. Python hashes them with
``derive.stomach_recipe.stomach_ref_recipe_sha256``; MATLAB (``night6_recipe_sha256``) hashes
the same bytes where they stand in the stored text - it never re-serialises, because its
jsonencode writes 5.0 as 5 (the 5-second defect). Checked here on texts with a float that
renders differently on the two sides, a nested decoy key and a string that names the key.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from gems_blanking_v2.derive import stomach_recipe as sr
from gems_blanking_v2.emit.provenance import MaskProvenance

from tests.conftest import make_multichannel
from tests.test_matlab_acceptance import _matlab

NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"


def _recipe() -> dict[str, Any]:
    rec, _truth = make_multichannel(24414.0625, 1.0, seed=5)
    return dict(sr.stomach_ref_recipe(rec, {}))


def _prov_text(recipe: dict[str, Any] | None, extra: dict[str, Any] | None = None) -> str:
    return MaskProvenance(
        model={"mode": "pooled", "version": "t", "corpus_hash": "0" * 32,
               "calibrator": "models/t/cal.json"},
        thresholds={"source": "synthetic"}, reference_values={"stomach_ref|0.5-3": {"m": 1}},
        code_commit="test", generation_sha="0133349b3ebeff80", routing_hash="test",
        created_at="2026-10-12T05:00:00+00:00", recording="r",
        extra=extra or {}, stomach_ref_recipe=recipe or {}).to_json()


DECOY_EXTRA = {"note": 'a string naming "stomach_ref_recipe": {"x": 1}',
               "proof": {"stomach_ref_recipe": {"schema": "decoy", "weights": [9.0]}}}
"""Sorted before the top-level key, so the scan meets both decoys first."""


def _cases() -> dict[str, dict[str, Any]]:
    r = _recipe()
    five = {**r, "weights": [5.0, -2.5, 0.1]}   # 5.0 renders "5.0" in Python, 5 in MATLAB
    return {"ok": {"text": _prov_text(r), "applied": True, "recipe": r},
            "not_applied": {"text": _prov_text(r), "applied": False, "recipe": None},
            "five_point_zero": {"text": _prov_text(five), "applied": True, "recipe": five},
            "decoys": {"text": _prov_text(r, DECOY_EXTRA), "applied": True, "recipe": r},
            "absent": {"text": _prov_text(None), "applied": True, "recipe": None}}


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    tmp = tmp_path_factory.mktemp("recipe_hash")
    case = {"night6": NIGHT6.as_posix(),
            "cases": {k: {"text": v["text"], "applied": v["applied"]} for k, v in _cases().items()}}
    cf, rf = tmp / "case.json", tmp / "result.json"
    cf.write_text(json.dumps(case), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{HARNESS.as_posix()}'); "
           f"check_recipe_hash('{cf.as_posix()}', '{rf.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return dict(json.loads(rf.read_text(encoding="utf-8")))


def test_the_python_hash_is_of_the_bytes_inside_provenance_json() -> None:
    """The canonical form IS the text emit.provenance writes: a substring of to_json()."""
    for name, c in _cases().items():
        if c["recipe"] is None:
            continue
        stored = json.loads(c["text"])["stomach_ref_recipe"]
        canon = json.dumps(stored, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        assert f'"stomach_ref_recipe":{canon}' in c["text"], name
        assert sr.stomach_ref_recipe_sha256(stored) == hashlib.sha256(
            canon.encode("ascii")).hexdigest()
        assert re.fullmatch(r"[0-9a-f]{64}", sr.stomach_ref_recipe_sha256(stored))


@pytest.mark.parametrize("name", ["ok", "five_point_zero", "decoys"])
def test_matlab_and_python_give_the_same_hash(result: dict[str, Any], name: str) -> None:
    c = _cases()[name]
    stored = json.loads(c["text"])["stomach_ref_recipe"]
    assert result[name]["error_id"] == ""
    assert result[name]["hash"] == sr.stomach_ref_recipe_sha256(stored)


def test_a_float_that_matlab_would_print_differently_still_matches(
        result: dict[str, Any]) -> None:
    assert '"weights":[5.0,-2.5,0.1]' in _cases()["five_point_zero"]["text"]
    assert result["five_point_zero"]["hash"] != result["ok"]["hash"]


def test_decoys_are_never_hashed(result: dict[str, Any]) -> None:
    assert result["decoys"]["hash"] == result["ok"]["hash"]


def test_none_when_no_recipe_was_applied(result: dict[str, Any]) -> None:
    assert result["not_applied"]["hash"] == "none"


def test_applied_without_a_recipe_is_refused_by_name(result: dict[str, Any]) -> None:
    assert result["absent"]["error_id"] == "night6:recipeHash"


def test_every_night6_record_carries_the_field() -> None:
    rec = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    assert ("R.stomach_ref_recipe_sha256 = night6_recipe_sha256(R.mask_provenance_json, ...\n"
            "                                                       isfield(plan.recipes, "
            "'stomach_ref'));") in rec
    assert "run.stomach_ref_recipe_sha256 = R.stomach_ref_recipe_sha256;" in rec
