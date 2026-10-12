"""Night 6 writes its records atomically AND checked (found 2026-10-12, a full C: drive).

``night6_run_recording``'s ``write_json`` wrote ``<record>.tmp`` without checking ``fwrite``
or ``fclose`` and then renamed it over the record: on a full disk an empty ``.tmp`` replaced
``night6_record.json`` (A gems_a_t01_ms1_sr_164150 e132, 0 bytes). The write now has one
site, ``matlab/night6/night6_write_text_atomic.m``, which proves the ``.tmp`` holds exactly
the text (fwrite count, fclose status, byte-for-byte read-back) before the rename and
otherwise raises ``night6:writeFailed`` leaving the record as it was.

``tests/matlab/check_write_atomic.m`` runs the writer with ``fwrite`` / ``fclose`` shadowed
for it alone (``tests/matlab/fault_io``) to simulate each failure. ``N6_WRITER_DIR`` picks the
writer under test (revert check: ``tests/matlab/mutants/n6_unchecked_write`` is the old
code, and these tests fail against it).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.test_matlab_acceptance import _matlab

NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"
WRITER = Path(os.environ.get("N6_WRITER_DIR", str(NIGHT6)))
NEW = '{\n  "new": "µV café"\n}'


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    out = tmp_path_factory.mktemp("write_atomic") / "result.json"
    cmd = (f"addpath('{HARNESS.as_posix()}'); check_write_atomic('{out.as_posix()}', "
           f"'{WRITER.as_posix()}', '{(HARNESS / 'fault_io').as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=600, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return res


def test_a_good_write_replaces_the_record_exactly(results: dict[str, Any]) -> None:
    r = results["none"]
    assert r["error_id"] == "" and r["new_bytes_exact"] is True, r
    assert r["final_text"].encode("latin-1").decode("utf-8") == NEW
    assert r["tmp_left"] is False


@pytest.mark.parametrize("fault", ["short", "fclose", "truncate"])
def test_a_failed_write_raises_and_the_old_record_survives(results: dict[str, Any],
                                                           fault: str) -> None:
    r = results[fault]
    assert r["final_text"] == "OLD", r  # the record that was there is still there
    assert r["error_id"] == "night6:writeFailed", r
    assert "night6_record.json left unchanged" in r["message"], r
    assert r["tmp_left"] is False, r  # the failed .tmp is not left to be mistaken for output
    # each failure is caught by its own check, naming its cause (the read-back alone would
    # catch all three, but would not say which)
    n = len(NEW.encode("utf-8"))
    cause = {"short": f"0 of {n} bytes written", "fclose": f"{n} of {n} bytes written, fclose "
             "status -1", "truncate": f"reads back as 0 bytes, not the {n} written"}[fault]
    assert cause in r["message"], r


def test_a_failed_rename_raises_and_leaves_the_record_and_no_tmp(
        results: dict[str, Any]) -> None:
    """The record is held open by another handle, so movefile cannot replace it (review)."""
    r = results["locked"]
    assert r["error_id"] == "night6:writeFailed", r
    assert "cannot replace" in r["message"] and "night6_record.json" in r["message"], r
    assert r["final_text"] == "OLD", r
    assert r["tmp_left"] is False, r


def _reached_by_calls() -> set[str]:
    """Every matlab/night6 file night6_run_recording calls, directly or through another.

    A call is ``name(`` or ``@name``; computed here independently of the MATLAB derivation.
    """
    stems = {m.stem for m in NIGHT6.glob("*.m")}
    seen: set[str] = set()
    todo = ["night6_run_recording"]
    while todo:
        src = (NIGHT6 / f"{todo.pop()}.m").read_text(encoding="utf-8")
        for c in (set(re.findall(r"\b(\w+)\s*\(", src)) | set(re.findall(r"@(\w+)", src))) & stems:
            if c not in seen and c != "night6_run_recording":
                seen.add(c)
                todo.append(c)
    return seen


@pytest.fixture(scope="module")
def wrapper_functions(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    out = tmp_path_factory.mktemp("ours") / "ours.json"
    cmd = (f"addpath('{NIGHT6.as_posix()}'); f = fopen('{out.as_posix()}', 'w'); "
           f"fwrite(f, jsonencode(night6_wrapper_functions('{NIGHT6.as_posix()}')), 'char'); "
           "fclose(f);")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=600, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    got: list[str] = json.loads(out.read_text(encoding="utf-8"))
    return got


def test_every_night6_function_the_run_reaches_is_hashed_and_shadow_checked(
        wrapper_functions: list[str]) -> None:
    """function_provenance's ``ours`` (refused if shadowed, hash recorded) is derived.

    Review of night6-atomic-json: the hand-written list missed night6_write_text_atomic,
    night6_calls, night6_consumer_input, night6_mmc_opts, night6_prepare_epoch,
    night6_first_sample, night6_hold_unit and night6_signal_from_token. It is now
    night6_wrapper_functions(here); every file the run reaches by a call must be in it.
    """
    rec = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    fp = rec[rec.index("function F = function_provenance()"):]
    fp = fp[:fp.index("\nend\n")]
    assert "ours = night6_wrapper_functions(here);" in fp
    assert not re.search(r"ours = \{", fp)  # no hand list beside it
    reached = _reached_by_calls()
    assert {"night6_write_text_atomic", "night6_calls", "night6_consumer_input",
            "night6_mmc_opts", "night6_prepare_epoch", "night6_wrapper_functions",
            "process_dataset_v2"} <= reached  # the probe itself sees what it must
    missing = sorted(reached - set(wrapper_functions))
    assert not missing, f"called from night6_run_recording but not in ours: {missing}"
    assert "night6_run_recording" not in wrapper_functions
    assert wrapper_functions == sorted(wrapper_functions)


def test_every_night6_text_record_goes_through_the_checked_writer() -> None:
    """No other unchecked fopen-for-write / movefile of a text record in matlab/night6."""
    rec = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    body = rec[rec.index("function write_json(f, R)"):]
    body = body[:body.index("\nend\n")]
    assert "night6_write_text_atomic(f, jsonencode(R" in body and "fopen" not in body
    batch = (NIGHT6 / "night6_batch.m").read_text(encoding="utf-8")
    assert "night6_write_text_atomic(f, jsonencode(summary" in batch
    for m in sorted(NIGHT6.glob("*.m")):
        if m.name == "night6_write_text_atomic.m":
            continue
        src = m.read_text(encoding="utf-8")
        assert not re.search(r"fopen\([^)]*'w'", src), f"{m.name} opens a file for writing"
