"""Night 6 trimming by recovery start (RULING 2026-10-08 (k) 2), MATLAB side, checked from Python.

A synthetic stim_rec recording (the wrapper test's 9-contact store: every sample encodes
its file index, a whole-file beat train, a mask file written by the real
``emit.handoff.write_mask_file`` for an epoch starting at 2.0 s) is planned by
``night6_prepare_epoch`` with and without its recovery starts, in ONE MATLAB process
(``tests/matlab/check_recovery_start.m``). Checks:

* TRIM MEANS MASKING THE INPUT: every consumer input with starts equals the input without
  them OR the leading rows 1 .. k0 - i0, exactly - nothing else changes;
* the 1-sample boundary (invariant 15): a start at file sample i0 + 1 (0-based) masks row
  1 only, i0 + 2 rows 1-2, i0 nothing; a start before the epoch masks nothing and is
  recorded "early part deferred to add-on";
* the starts come from the Python writer (``write_recovery_starts``): an analysis with
  unknown settling keeps the epoch start labelled ``fixed_132s_settling_unknown`` while
  the others in the same file are trimmed to their own starts;
* hrv and breathing with different starts get two HR runs (one mask each, invariant 2);
  a slow-wave lead-in that swallows the only difference between the ANT masks makes them
  identical, and they share one run that keeps all three (the (j) 5 (a) shortcut, exact);
* refused by name, never a silent fallback: a stim_rec epoch with no starts file, a held
  file, a file with no row, a consumer that runs with no row, an fs mismatch, a row for a
  recording that is not stim_rec; a baseline without a row is untouched;
* the reader refuses another schema, a session twice (case-insensitively) and a
  non-integer start sample;
* ``night6_run_recording``: stim_rec without RecoveryStarts is refused; with it, every
  epoch record carries each consumer's start, basis, source and the file's SHA-256; a
  complete record made with the same file is resumed, with another file rerun.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.emit.hr_beats import beats_file_record, write_hr_beats
from gems_blanking_v2.extent import recovery_start as rs
from gems_blanking_v2.extent.grid import seconds_to_sample
from scipy.io import loadmat, savemat

from tests.test_matlab_acceptance import _matlab, _processing_new
from tests.test_night6_wrapper import (
    BEAT_BLANK_S,
    BEATS_S,
    FS,
    HARNESS,
    HR,
    LABELS,
    MODEL,
    N_FILE,
    NIGHT6,
    READS_A,
    SPANS_A,
    _channels,
    _signal,
    _write_epoch,
)

SESSION = "syn_s_t01_ms1_sr_100000_20260101T150000Z"
START = 2.0
I0 = seconds_to_sample(START, FS)
N = N_FILE - I0
CONSUMERS = ("spikes", "slow_wave", "mmc", "hrv", "breathing")


def _store(root: Path, condition: str = "stim_recovery", name: str = SESSION) -> dict[str, Any]:
    """One recording with beats and one epoch from START; returns what the harness needs."""
    sdir = root / "data" / "T" / name
    mdir = sdir / "masks" / MODEL
    mdir.mkdir(parents=True)
    (root / "rec").mkdir(exist_ok=True)
    savemat(root / "rec" / f"{name}_sig.mat",
            {"signal": _signal(name), "fs": FS,
             "chanlabels": np.array(LABELS, dtype=object).reshape(1, -1)})
    meta = {"session": name, "animal": "T", "channels": _channels(),
            "source_path": f"rec/{name}_sig.mat"}
    (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")
    bfile = sdir / f"{name}_beats.mat"
    write_hr_beats(bfile, BEATS_S, fs=FS, epoch_start_s=0.0, n_samples=N_FILE, channel=HR,
                   source="synthetic", gap_after=np.arange(BEATS_S.size) % 7 == 3,
                   blank_spans_s=BEAT_BLANK_S)
    rec = beats_file_record(
        grade="hrv", store_rel=f"data/T/{name}/{bfile.name}",
        sha256=hashlib.sha256(bfile.read_bytes()).hexdigest(), origin_sample0=0,
        heartlocs=loadmat(bfile)["heartlocs"].ravel(), epoch_start_sample=I0, n_samples=N,
        published=True, read_from=f"gems_root:data/T/{name}/{bfile.name}")
    _write_epoch(mdir, name, READS_A, SPANS_A, START, N,
                 {"condition": condition, "beats_file": rec}, None)
    return {"mask_folder": mdir, "mask_file": mdir / f"e{round(START)}_masks.mat",
            "beats_file": bfile, "record_json": json.dumps(rec, ensure_ascii=True),
            "meta_json": json.dumps(meta)}


def _row(name: str, k0: int, basis: str = rs.BASIS_MEASURED) -> dict[str, Any]:
    return {"analysis": name, "start_s": k0 / FS, "start_sample0": k0, "basis": basis,
            "source": "synthetic"}


def _starts(path: Path, files: list[dict[str, Any]], fs: float = FS) -> str:
    rs.write_recovery_starts(path, rs.recovery_starts_document(files, fs=fs))
    return path.as_posix()


BOUNDARY = {"spikes": I0 + 1, "slow_wave": I0, "mmc": I0 - 1, "hrv": I0 + 1,
            "breathing": I0 + 2}
"""0-based file samples: one sample in, at the start, one before, one in, two in."""
DEEP_SW = seconds_to_sample(8.5, FS)
"""Past the ANT1 slow-wave span (7-8 s): the three ANT masks become identical."""
TEST_TABLE = {
    "spikes": rs.Analysis("spikes", "c", (rs.Output("o", (rs.Stage(
        "w", "trailing window", "whole", "x.m:1", "x", "code", window_s=1.0),), "x"),)),
    "hrv": rs.Analysis("hrv", "c", (rs.Output("o", (rs.Stage(
        "w", "centred window", "half", "x.m:1", "x", "code", window_s=4.0),), "x"),)),
    "breathing": rs.Analysis("breathing", "c", (rs.Output("o", (rs.Stage(
        "w", "centred window", "half", "x.m:1", "x", "code", window_s=4.0),), "x"),)),
    "mmc": rs.Analysis("mmc", "c", (rs.Output("o", (rs.Stage(
        "w", "centred window", "half", "x.m:1", "x", "code", window_s=0.5),), "x"),)),
    "slow_wave": rs.Analysis("slow_wave", "c", (rs.Output("o", (rs.Stage(
        "unmeasured", "centred window", "unknown", "x.m:1", "x", "code"),), "x"),)),
}
"""For the writer-to-wrapper case: slow wave unknown, the others measured."""


def _cases(tmp: Path) -> dict[str, Any]:
    st = _store(tmp / "store")
    bl = _store(tmp / "store_bl", condition="baseline")
    common = {"labels": list(LABELS), "fs": FS, "n_file": N_FILE, "session": SESSION,
              "meta_json": st["meta_json"]}

    def plan(name: str, starts: str, store: dict[str, Any] = st,
             condition: str = "stim_recovery") -> dict[str, Any]:
        return {**common, "name": name, "starts_file": starts, "condition": condition,
                "mask_file": store["mask_file"].as_posix(),
                "beats_file": store["beats_file"].as_posix(),
                "record_json": store["record_json"]}

    file_b = {"session": SESSION, "fs": FS,
              "analyses": [_row(c, BOUNDARY[c]) for c in CONSUMERS]}
    file_deep = {"session": SESSION, "fs": FS,
                 "analyses": [_row(c, DEEP_SW if c == "slow_wave" else I0) for c in CONSUMERS]}
    py = rs.file_starts(session=SESSION, fs=FS, stim_off_s=1.0, electrical_settle_s=1.5,
                        stim_off_source="synthetic", electrical_source="synthetic",
                        table=TEST_TABLE, fixed_start_s=START)
    held = rs.file_starts(session=SESSION, fs=FS, stim_off_s=None, electrical_settle_s=None,
                          stim_off_source="a", electrical_source="b")
    other = {"session": "someone_else", "fs": FS, "analyses": [_row("spikes", I0)]}
    no_spikes = {**file_b, "analyses": [_row(c, BOUNDARY[c]) for c in CONSUMERS[1:]]}
    d = tmp / "starts"
    d.mkdir()
    s = {"boundary": _starts(d / "b.json", [file_b]),
         "deep": _starts(d / "deep.json", [file_deep]),
         "python": _starts(d / "py.json", [py]),
         "held": _starts(d / "held.json", [held]),
         "other": _starts(d / "other.json", [other]),
         "no_spikes": _starts(d / "nos.json", [no_spikes]),
         "fs": _starts(d / "fs.json", [{**file_b, "fs": FS + 1.0}], fs=FS + 1.0)}
    readers = []
    for name, edit in (("schema", lambda j: j.update(schema="v0")),
                       ("twice", lambda j: j["held"].append({"session": SESSION.upper(),
                                                             "basis": "x"})),
                       ("fractional", lambda j: j["files"][0]["analyses"][0].update(
                           start_sample0=I0 + 0.5))):
        j = json.loads(Path(s["boundary"]).read_text(encoding="utf-8"))
        edit(j)
        f = d / f"reader_{name}.json"
        f.write_text(json.dumps(j), encoding="utf-8", newline="\n")
        readers.append({"name": name, "file": f.as_posix()})
    readers.append({"name": "ok", "file": s["boundary"]})
    plans = [plan("boundary", s["boundary"]), plan("deep", s["deep"]),
             plan("python", s["python"]), plan("no_file", ""), plan("held", s["held"]),
             plan("no_row", s["other"]), plan("no_spikes", s["no_spikes"]),
             plan("fs", s["fs"]),
             plan("baseline_row", s["boundary"], bl, "baseline"),
             plan("baseline", s["other"], bl, "baseline")]
    run = {"gems_root": (tmp / "store").as_posix(), "mask_folder": st["mask_folder"].as_posix(),
           "out_a": (tmp / "out_a").as_posix(), "out_b": (tmp / "out_b").as_posix(),
           "starts_file": s["boundary"], "other_starts_file": s["deep"],
           "record_rel": f"T/{SESSION}/{MODEL}/e{round(START)}/night6_record.json"}
    return {"plans": plans, "readers": readers, "run": run, "py": py}


def _rows(runs: object) -> list[list[int]]:
    a = np.asarray(runs, dtype=np.int64)
    return a.reshape(-1, 2).tolist() if a.size else []


def _mask(runs: object) -> np.ndarray:
    m = np.zeros(N, dtype=bool)
    for a, b in _rows(runs):
        m[a - 1:b] = True
    return m


def _inputs(desc: list[dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    """(consumer, signal) -> NaN rows of that consumer's own input column."""
    out: dict[tuple[str, str], np.ndarray] = {}
    for e in desc:
        cons = e["consumers"] if isinstance(e["consumers"], list) else [e["consumers"]]
        for c in cons:
            keep = e["keep"] if isinstance(e["keep"], list) else [e["keep"]]
            if c == "slow_wave" and e["signal"] not in keep:
                continue  # a column analysed under another channel's mask (not kept)
            key = (c, e["signal"])
            assert key not in out, key
            out[key] = _mask(e["nan_runs"])
    return out


def _check_trim(got: dict[str, Any], rows: dict[str, int]) -> None:
    assert got["error"] == "", got["message"]
    base, trim = _inputs(got["base"]), _inputs(got["trim"])
    assert set(base) == set(trim)
    for (c, sig), m in trim.items():
        lead = np.zeros(N, dtype=bool)
        lead[:rows[c]] = True
        assert np.array_equal(m, base[(c, sig)] | lead), (c, sig)


def test_night6_trims_by_masking_the_input_to_the_sample(  # noqa: PLR0915 - one MATLAB run
        tmp_path: Path) -> None:
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    case = _cases(tmp_path)
    case_file, res_file = tmp_path / "rs_case.json", tmp_path / "rs_result.json"
    case_file.write_text(json.dumps({k: case[k] for k in ("plans", "readers", "run")},
                                    ensure_ascii=True), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); "
           f"check_recovery_start('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=1200, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res = json.loads(res_file.read_text(encoding="utf-8"))
    got = {p["name"]: p for p in res["plans"]}

    # the boundary: one sample in, at, one before, one in, two in
    b = got["boundary"]
    rows = {c: max(0, k - I0) for c, k in BOUNDARY.items()}
    assert rows == {"spikes": 1, "slow_wave": 0, "mmc": 0, "hrv": 1, "breathing": 2}
    _check_trim(b, rows)
    trim = _inputs(b["trim"])
    base = _inputs(b["base"])
    assert not base[("spikes", "L_T")][:2].any()
    assert trim[("spikes", "L_T")][:2].tolist() == [True, False]
    assert not base[("breathing", HR)][:3].any()
    assert trim[("breathing", HR)][:3].tolist() == [True, True, False]
    assert sum(e["call"] == "HR_BR_HRVAnalysis_beats" and e["signal"] == HR
               for e in b["trim"]) == 2  # hrv and breathing: two masks, two runs
    rec = json.loads(b["record"])
    assert rec["applies"] is True and rec["epoch_start_sample0"] == I0
    st = {c: rec["consumers"][c] for c in CONSUMERS}
    assert st["spikes"]["status"] == "trimmed" and st["spikes"]["trimmed_rows"] == 1
    assert st["slow_wave"]["status"] == "at_epoch_start" and st["slow_wave"]["trimmed_rows"] == 0
    assert st["mmc"]["status"] == "early part deferred to add-on"
    assert st["breathing"]["trimmed_rows"] == 2
    assert all(st[c]["source"] == "synthetic" and st[c]["basis"] == rs.BASIS_MEASURED
               for c in CONSUMERS)
    assert len(rec["sha256"]) == 64

    # slow wave: a lead-in past the only difference makes the three masks one shared run
    d = got["deep"]
    _check_trim(d, {c: (DEEP_SW - I0 if c == "slow_wave" else 0) for c in CONSUMERS})
    sw_base = [e for e in d["base"] if e["call"] == "slowWaveAnalysis_new"]
    sw_trim = [e for e in d["trim"] if e["call"] == "slowWaveAnalysis_new"]
    assert len({e["mask_signal"] for e in sw_base}) == 2
    assert {e["mask_signal"] for e in sw_trim} == {"ANT1"}

    # from the Python writer: slow wave unknown keeps the epoch start, labelled
    p = got["python"]
    py = {r["analysis"]: r for r in case["py"]["analyses"]}
    prow = {c: max(0, py[c]["start_sample0"] - I0) for c in CONSUMERS}
    assert prow["slow_wave"] == 0 and prow["mmc"] == 0
    assert prow["spikes"] == seconds_to_sample(2.5, FS) - I0
    assert prow["hrv"] == prow["breathing"] == seconds_to_sample(3.5, FS) - I0
    _check_trim(p, prow)
    prec = json.loads(p["record"])["consumers"]
    assert prec["slow_wave"]["basis"] == "fixed_132s_settling_unknown"
    assert prec["slow_wave"]["status"] == "at_epoch_start"
    assert prec["slow_wave"]["missing_settling"] == ["slow_wave/o: unmeasured"]
    assert prec["spikes"]["status"] == "trimmed"
    assert prec["mmc"]["status"] == "early part deferred to add-on"  # 1.75 s < 2.0 s

    # refusals, by name
    for name, ident, words in (
            ("no_file", "night6:recoveryStarts", "declared with RecoveryStarts"),
            ("held", "night6:recoveryStartHeld", rs.HELD_UNDETECTED),
            ("no_row", "night6:recoveryStartMissing", f"no recovery start for {SESSION}"),
            ("no_spikes", "night6:recoveryStartMissing", "analysis spikes"),
            ("fs", "night6:recoveryStartFs", "fs"),
            ("baseline_row", "night6:recoveryStartCondition", "not stim_recovery")):
        assert got[name]["error"] == ident, (name, got[name])
        assert words in got[name]["message"], (name, got[name]["message"])
    bl = got["baseline"]
    _check_trim(bl, dict.fromkeys(CONSUMERS, 0))
    assert json.loads(bl["record"])["applies"] is False

    readers = {r["name"]: r for r in res["readers"]}
    assert readers["ok"]["error"] == ""
    for name in ("schema", "twice", "fractional"):
        assert readers[name]["error"] == "night6:recoveryStartFile", (name, readers[name])

    # the real entry point
    run = res["run"]
    assert run["without"]["error"] == "night6:recoveryStarts"
    rec = json.loads(run["with"] if isinstance(run["with"], str) else run["with"][0])
    assert rec["status"] == "dry_run"
    rsr = rec["recovery_start"]
    assert rsr["file"].replace("\\", "/") == case["run"]["starts_file"]
    assert rsr["sha256"] == hashlib.sha256(Path(case["run"]["starts_file"]).read_bytes()
                                           ).hexdigest()
    assert {c: rsr["consumers"][c]["trimmed_rows"] for c in CONSUMERS} == rows
    spk = next(r for r in rec["runs"] if r["call"] == "process_dataset_v2")
    lt = next(i for i in spk["inputs"] if i["signal"] == "L_T")
    assert lt["first_nan_row"] == 1 and _rows(lt["nan_runs"])[0][0] == 1
    assert "night6_recovery_lead_in" in rec["functions"]
    assert run["resume_same"] == "complete"   # same starts file: skipped
    assert run["resume_other"] == "dry_run"   # another starts file: rerun


def test_run_epoch_always_hands_the_starts_to_the_planner() -> None:
    """The planner's 'Recovery' is optional for harnesses; the entry point never omits it."""
    code = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    assert code.count("night6_prepare_epoch(") == 1
    assert "'Recovery', struct('starts', o.RS, 'session', src.session)" in code
    batch = (NIGHT6 / "night6_batch.m").read_text(encoding="utf-8")
    assert "'RecoveryStarts', starts" in batch
