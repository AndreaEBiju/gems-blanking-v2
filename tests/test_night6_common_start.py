"""RULING 2026-10-10 item 7 in Night 6: starts v6, each start beside the results, common start.

A v6 starts file written by the Python writer (two stim levels of one animal, one of them
reference-sensitive) is read by ``night6_recovery_start``; ``night6_recovery_lead_in`` records
every analysis's OWN start and the animal's COMMON start (and the reference-sensitive "until")
beside the results; ``night6_common_start_rec`` moves every cut to its common start for the
common-start version of the outputs, and refuses a record without one. The run writes that
version from copies of the untouched outputs (checked in the source). Skips without MATLAB or
processing_new.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from gems_blanking_v2.extent import recovery_start as rs

from tests.test_matlab_acceptance import _matlab, _processing_new

NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"
FS = 24414.0625
REF = {"window_s": [540.0, 600.0], "anchor_s": 120.0, "n_clean": 79, "n_candidates": 81,
       "after_anchor_s": {"chosen": 70.79, "clean_min": 29.864, "clean_max": 79.528}}


def _file(session: str, e: float, settle: float, m: float,
          ref: dict[str, Any] | None = None, level: str = "L1") -> dict[str, Any]:
    return rs.file_starts(session=session, animal="A", electrical_basis="from_electrical_end",
                          fs=FS, electrical_end_s=e, mechanical_end_s=m,
                          electrical_settle_s=settle, times_source="tsq",
                          electrical_source="test", reference_sensitive=ref, stim_level=level)


@pytest.fixture(scope="module")
def got(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None or pnew is None:
        pytest.skip("MATLAB or processing_new is not on this machine")
    tmp = tmp_path_factory.mktemp("common")
    starts = tmp / "starts.json"
    doc = rs.recovery_starts_document([_file("a_es1", 120.0, 124.0, 120.0, level="es1"),
                                       _file("a_es3", 120.0, 140.0, 120.0, level="es3"),
                                       _file("a_late", 120.0, 190.79, 120.0, REF,
                                             level="es5")], fs=FS)
    rs.write_recovery_starts(starts, doc)
    out = tmp / "out.json"
    cmd = (f"addpath('{Path(pnew).as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); check_common_start('{starts.as_posix()}', "
           f"'a_es1', '{out.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=600, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    res["doc"] = doc
    return res


def test_each_analysis_reports_its_own_and_the_common_start(got: dict[str, Any]) -> None:
    rec = json.loads(got["rec"])
    row = next(f for f in got["doc"]["files"] if f["session"] == "a_es1")
    assert rec["electrical_basis"] == "from_electrical_end" and rec["animal"] == "A"
    for r in row["analyses"]:
        c = rec["consumers"][r["analysis"]]
        assert c["start_sample0"] == r["start_sample0"]
        assert c["common_start_sample0"] == r["common_start_sample0"] > r["start_sample0"]
        assert c["common_start_set_by"] == "stim level es3"  # a_late is excluded


def test_the_common_version_cuts_every_output_at_its_common_start(got: dict[str, Any]) -> None:
    rec, com = json.loads(got["rec"]), json.loads(got["common"])
    assert com["version"] == "common_start"
    own = {c["cut"]: c for c in rec["cuts"]}
    for c in com["cuts"]:
        o = own[c["cut"]]
        assert c["start_sample0"] == o["common_start_sample0"] >= o["start_sample0"]
        assert c["own_start_sample0"] == o["start_sample0"]
        assert c["output_rows_before_start"] == o["output_rows_before_common"]
    assert got["refused"]["id"] == "night6:commonStart"
    assert got["refused_not_applying"]["id"] == "night6:commonStart"


def test_a_v5_starts_file_is_refused_by_night6() -> None:
    src = (NIGHT6 / "night6_recovery_start.m").read_text(encoding="utf-8")
    assert "want = 'gems-blanking-v2 recovery starts v8';" in src
    assert "check_v6(F, file);" in src


def test_the_run_writes_the_common_start_version_from_untouched_copies() -> None:
    src = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    i_copy = src.index("copyfile(fullfile(outDir, run.outputs{q}), cdir);")
    i_common = src.index("night6_common_start_rec(plan.recoveryStart)")
    i_own = src.index("run.recovery_trim = night6_recovery_trim_outputs(outDir,")
    assert i_copy < i_common < i_own  # copies taken before the own trim rewrites the files
    assert "cdir = fullfile(outDir, 'common_start');" in src
