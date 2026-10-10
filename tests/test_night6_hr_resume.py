"""Review 7 finding 5: Night 6 resumes a complete epoch only if its HR runs follow (i) 4.

RULING 2026-10-09 (i) 4 runs hrv and breathing as two HR_BR calls, each recording the
outputs it is read for (``night6_hr_outputs``). A record written before that - one shared
HR call, or no ``outputs_used`` - must rerun, never be reported done. The rule has one site,
``matlab/night6/night6_hr_runs_resumable.m``; ``tests/matlab/check_hr_resume.m`` runs it in
MATLAB on records encoded and decoded the way ``night6_run_recording`` writes and reads them,
and the resume step's use of it is checked in the source.
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
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    out = tmp_path_factory.mktemp("hr_resume") / "result.json"
    cmd = (f"addpath('{NIGHT6.as_posix()}'); addpath('{HARNESS.as_posix()}'); "
           f"check_hr_resume('{out.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=600, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res: dict[str, str] = json.loads(out.read_text(encoding="utf-8"))
    return res


@pytest.mark.parametrize("case", ["two_calls", "no_hr_run", "empty_runs", "hr_only_same_fields",
                                  "consumer_as_text"])
def test_a_record_whose_hr_runs_follow_i4_is_resumable(results: dict[str, str],
                                                       case: str) -> None:
    assert results[case] == "true", results[case]


@pytest.mark.parametrize(("case", "words"), [
    ("one_shared_call", "serves 2 consumer(s)"),
    ("no_outputs_used", "records no outputs_used"),
    ("outputs_used_of_the_other_consumer", "records no outputs_used"),
    ("no_runs_field", "no runs list"),
    ("hr_without_consumers", "names no consumers"),
])
def test_a_record_from_before_i4_is_rerun(results: dict[str, str], case: str,
                                          words: str) -> None:
    assert results[case].startswith("false: "), results[case]
    assert words in results[case], results[case]


def test_the_resume_step_requires_it() -> None:
    """The resume key skips an epoch only when the HR-run check passes, and says why not."""
    src = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    assert re.search(r"\[hrOk, hrWhy\] = night6_hr_runs_resumable\(R\);\s*"
                     r"if isfield\(R, 'status'\) && strcmp\(R\.status, 'complete'\) && same "
                     r"&& hrOk\s*records\{k\} = R;\s*todo\(k\) = false;", src), \
        "a complete record may be skipped only when night6_hr_runs_resumable passes"
    assert "complete before (i) 4 (%s), rerun" in src
    assert "'night6_hr_runs_resumable'}" in src  # hashed in function_provenance
