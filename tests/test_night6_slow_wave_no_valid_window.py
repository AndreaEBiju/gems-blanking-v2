"""RULING 2026-10-13 (slow-wave release): an epoch with no valid window, end to end.

A REAL Night 6 slow-wave run (her slowWaveAnalysis_new, decimated78, the span rule) on the
synthetic store's 10 s baseline SESSION_E (tests/test_night6_wrapper.py): no 60 s rate window
fits, so every channel has no valid window and her averages are NaN. Defect at 43048b6: the
span-rule summary copied those NaN averages into the record and the writer refused it
(night6:nonFinite), leaving outputs with no record. Now the record is written, complete, with
every channel and the run "no valid window" (never held or failed) and the averages ABSENT
(the project's JSON convention: absent, never NaN or null).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.test_matlab_acceptance import _matlab, _processing_new
from tests.test_night6_wrapper import MODEL, SESSION_E, _store

NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"
AVERAGES = ("avg_detected", "avg_kept", "avg_kept_zone_counted")


@pytest.fixture(scope="module")
def record(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    tmp = tmp_path_factory.mktemp("novalid")
    root = tmp / "store"
    _store(root)
    case = {"processing_new": pnew.as_posix(), "night6": NIGHT6.as_posix(),
            "gems_root": root.as_posix(), "out_root": (tmp / "out").as_posix(),
            "mask_folder": (root / "data" / "T" / SESSION_E / "masks" / MODEL).as_posix()}
    cf, rf = tmp / "case.json", tmp / "result.json"
    cf.write_text(json.dumps(case), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{HARNESS.as_posix()}'); "
           f"check_no_valid_window('{cf.as_posix()}', '{rf.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=1200, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res = json.loads(rf.read_text(encoding="utf-8"))
    assert res["error"] == "", res["error"]
    text = Path(res["record"]).read_text(encoding="utf-8")
    return {"res": res, "text": text, "rec": json.loads(text)}


def _walk(v: Any, path: str = "") -> list[str]:  # noqa: ANN401
    bad: list[str] = []
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        bad.append(path)
    elif isinstance(v, dict):
        for k, x in v.items():
            bad += _walk(x, f"{path}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            bad += _walk(x, f"{path}[{i}]")
    return bad


def test_the_record_is_written_complete(record: dict[str, Any]) -> None:
    assert record["res"]["status"] == "complete"
    assert record["rec"]["status"] == "complete"


def test_every_channel_and_the_run_are_no_valid_window(record: dict[str, Any]) -> None:
    sw = record["rec"]["consumers"]["slow_wave"]
    assert sw["status"] == "no valid window"
    assert set(sw["channels"]) == {"ANT1", "ANT2", "ANT3"}
    for ch, st in sw["channels"].items():
        assert st["status"] == "no valid window", ch
        assert "RULING 2026-10-13" in st["reason"]
        w = st["windows"]
        assert w["n_valid_windows"] == 0 and w["n_rate_windows"] == 0 and w["n_windows"] == 0
    statuses = [sw["status"], *(st["status"] for st in sw["channels"].values())]
    assert not any(s.startswith(("skipped_held", "failed")) for s in statuses)


def test_non_finite_averages_are_absent_not_null(record: dict[str, Any]) -> None:
    runs = [r for r in record["rec"]["runs"] if r["call"] == "slowWaveAnalysis_new"]
    assert runs and all(r["status"] == "ok" for r in runs)
    seen = 0
    for r in runs:
        for ch, s in r["span_rule"].items():
            seen += 1
            for k in AVERAGES:
                assert k not in s, (ch, k)
            assert s["n_kept"] == 0 and "zone_s" in s
            p = s["pre_span_peaks"]
            assert "median_amplitude_uV" not in p and p["n"] == 0 and p["defined"] is False
            assert r["windows"][ch]["status"] == "no valid window"
    assert seen == 3
    assert _walk(record["rec"]) == []                      # no null, NaN or Inf anywhere
    # nor as a JSON token (strings may mention the word NaN)
    assert not re.search(r"[:,\[]\s*(NaN|-?Infinity|null)\s*[,\]}]", record["text"])
