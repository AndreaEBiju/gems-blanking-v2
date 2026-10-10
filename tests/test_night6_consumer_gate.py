"""RULING 2026-10-10 items 1 and 5 on the Night 6 side: which consumers run from a mask file.

``matlab/night6/night6_consumer_gate.m`` is the one site: a consumer finished in another file
(``elsewhere_json``) is skipped there, a consumer outside the batch's subset for the recording
is skipped, and a consumer QC holds (``heldconsumers_json``, per consumer) is skipped unless
released - the others run. ``night6_consumer_subset`` checks a batch's subset and gives the
epoch folder's suffix, so two subsets of one mask folder never share outputs.
``tests/matlab/check_consumer_gate.m`` runs the cases in MATLAB; the wiring into
``night6_prepare_epoch`` / ``night6_run_recording`` / ``night6_batch`` is checked in the source.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.test_matlab_acceptance import _matlab

NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, str]]:
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    out = tmp_path_factory.mktemp("gate") / "result.json"
    cmd = (f"addpath('{NIGHT6.as_posix()}'); addpath('{HARNESS.as_posix()}'); "
           f"check_consumer_gate('{out.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=600, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res: dict[str, dict[str, str]] = json.loads(out.read_text(encoding="utf-8"))
    return res


def test_a_held_consumer_is_skipped_and_the_others_run(results: dict) -> None:
    assert results["held_unreleased"]["status"] == "skipped_held"
    assert "24.0% blanked" in results["held_unreleased"]["reason"]
    assert results["held_released"]["status"] == ""      # released: it runs
    assert results["not_held"]["status"] == ""           # never the whole recording
    assert results["empty_records"]["status"] == ""


def test_a_consumer_finished_elsewhere_is_skipped_even_when_the_batch_names_it(
        results: dict) -> None:
    for case in ("elsewhere", "elsewhere_beats_batch"):
        assert results[case]["status"] == "skipped_elsewhere"
        assert "masks/m_spikes" in results[case]["reason"]


def test_a_batch_subset_runs_only_its_consumers(results: dict) -> None:
    assert results["not_in_batch"]["status"] == "skipped_not_in_batch"
    assert "[mmc slow_wave]" in results["not_in_batch"]["reason"]
    assert results["in_batch"]["status"] == ""
    assert results["in_batch_but_held"]["status"] == "skipped_held"


def test_a_subset_is_checked_and_tags_its_epoch_folder(results: dict) -> None:
    assert results["subset_nonspike"] == {"subset": "breathing,hrv,mmc,slow_wave",
                                          "tag": "__br_hv_mm_sw"}
    assert results["subset_spikes"] == {"subset": "spikes", "tag": "__sp"}
    assert results["subset_all"] == {"subset": "", "tag": ""}
    assert results["subset_unknown"]["id"] == "night6:consumers"
    assert "heart" in results["subset_unknown"]["message"]
    assert results["subset_twice"]["id"] == "night6:consumers"


def test_the_gate_is_wired_into_the_plan_the_run_and_the_batch() -> None:
    prep = (NIGHT6 / "night6_prepare_epoch.m").read_text(encoding="utf-8")
    assert "night6_consumer_gate(cn, plan.held, plan.elsewhere, subset)" in prep
    assert re.search(r"elseif ~isempty\(gs\)\s*st\.status = gs;", prep)
    assert prep.index("elseif ~isempty(gs)") < prep.index("elseif ~any(sel)")
    assert "night6:elsewhereReads" in prep
    run = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    assert "tag = [erase(files(k).name, '_masks.mat') subsetTag];" in run
    assert "&& same_subset(R, o.Consumers)" in run
    assert "'Consumers', o.Consumers);" in run
    batch = (NIGHT6 / "night6_batch.m").read_text(encoding="utf-8")
    assert batch.count("[args, {'Consumers', subsets{i}}]") == 2
