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

The two trim modes (review of be402a1; Andrea decides before Night 6):

* the mode is REQUIRED: a run or a batch without one, or with an unknown one, is refused
  by name (``night6:recoveryTrimMode``); it is in every record, and a stim_rec epoch made
  under the other mode reruns;
* ``mask_to_electrical_drop_outputs`` masks EVERY consumer's input on exactly rows
  1 .. ke - i0 (ke = stim-off + electrical settling, one point for all analyses) - the
  1-sample boundary on both sides - whatever each analysis's own start;
* the output cut, unit level: every stamp convention of the map (row1, sec0, sec_row1,
  sec_xchan_delay) at L - 1, L and L + 1; owners with different starts in one file;
  byproducts cut at their OWN owner's start whoever ran (an hrv-only, a breathing-only
  and an hrv+breathing run; an owner with no start refused); struct arrays, cells and
  co-indexed lists; v7.3 kept; an unmapped and an 'unknown' variable and a figure listed
  untrimmed by name; a stamp that does not fit its value refused; a file trimmed twice
  refused;
* the marker (``rs.TRIM_MARKER``) every trimmed file carries, exact for every action:
  owner, action, stamp, convention, the owner's start in seconds and samples, the first
  computed epoch row and sample, the mode, and per leaf the not-computed rows;
* the cited processing_new files are checked at run time: a stale hash, a missing file,
  and a changed copy of a cited function first on the path are refused by name
  (``night6:recoveryStartSource``); a byte-identical copy is not;
* end to end, with her real functions, on a 73 s epoch: the drop mode against an
  untrimmed run on the same masked input, through an oracle with its OWN convention table
  (not the map) - every entry before its owner's start dropped or flagged, every entry at
  or after it bit-identical, every other variable of every file untouched, the marker
  exact, and every trimmed variable with entries (and reference values) on both sides;
* ``night6_batch`` refuses at batch start - before any signal is loaded - a stim_rec mask
  with no recovery_starts, and a missing or unknown mode.
"""

from __future__ import annotations

import hashlib
import json
import re
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


def _store(root: Path, condition: str = "stim_recovery", name: str = SESSION,
           signal: np.ndarray | None = None, beats_s: np.ndarray = BEATS_S) -> dict[str, Any]:
    """One recording with beats and one epoch from START; returns what the harness needs.

    ``signal`` (microvolts) sets the file's length; by default the wrapper's encoded one.
    """
    sig = _signal(name) if signal is None else signal
    n_file = sig.shape[0]
    sdir = root / "data" / "T" / name
    mdir = sdir / "masks" / MODEL
    mdir.mkdir(parents=True)
    (root / "rec").mkdir(exist_ok=True)
    savemat(root / "rec" / f"{name}_sig.mat",
            {"signal": sig, "fs": FS,
             "chanlabels": np.array(LABELS, dtype=object).reshape(1, -1)})
    meta = {"session": name, "animal": "T", "channels": _channels(),
            "source_path": f"rec/{name}_sig.mat"}
    (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")
    bfile = sdir / f"{name}_beats.mat"
    write_hr_beats(bfile, beats_s, fs=FS, epoch_start_s=0.0, n_samples=n_file, channel=HR,
                   source="synthetic", gap_after=np.arange(beats_s.size) % 7 == 3,
                   blank_spans_s=BEAT_BLANK_S)
    rec = beats_file_record(
        grade="hrv", store_rel=f"data/T/{name}/{bfile.name}",
        sha256=hashlib.sha256(bfile.read_bytes()).hexdigest(), origin_sample0=0,
        heartlocs=loadmat(bfile)["heartlocs"].ravel(), epoch_start_sample=I0,
        n_samples=n_file - I0,
        published=True, read_from=f"gems_root:data/T/{name}/{bfile.name}")
    _write_epoch(mdir, name, READS_A, SPANS_A, START, n_file - I0,
                 {"condition": condition, "beats_file": rec}, None)
    return {"mask_folder": mdir, "mask_file": mdir / f"e{round(START)}_masks.mat",
            "beats_file": bfile, "record_json": json.dumps(rec, ensure_ascii=True),
            "meta_json": json.dumps(meta)}


def _file(rows: list[dict[str, Any]], ke: int = I0, session: str = SESSION) -> dict[str, Any]:
    return {"session": session, "fs": FS, "electrical_settle_sample0": ke, "analyses": rows}


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
             condition: str = "stim_recovery", mode: str = "mask_to_own_start") -> dict[str, Any]:
        return {**common, "name": name, "starts_file": starts, "condition": condition,
                "mode": mode,
                "mask_file": store["mask_file"].as_posix(),
                "beats_file": store["beats_file"].as_posix(),
                "record_json": store["record_json"]}

    file_b = _file([_row(c, BOUNDARY[c]) for c in CONSUMERS])
    file_deep = _file([_row(c, DEEP_SW if c == "slow_wave" else I0) for c in CONSUMERS])
    own = {c: I0 + 1000 * (k + 3) for k, c in enumerate(CONSUMERS)}
    elec = {d: _file([_row(c, own[c]) for c in CONSUMERS], ke=I0 + d) for d in (-1, 0, 1, 2)}
    py = rs.file_starts(session=SESSION, fs=FS, stim_off_s=1.0, electrical_settle_s=1.5,
                        stim_off_source="synthetic", electrical_source="synthetic",
                        table=TEST_TABLE, fixed_start_s=START)
    held = rs.file_starts(session=SESSION, fs=FS, stim_off_s=None, electrical_settle_s=None,
                          stim_off_source="a", electrical_source="b")
    other = _file([_row("spikes", I0)], session="someone_else")
    no_spikes = {**file_b, "analyses": [_row(c, BOUNDARY[c]) for c in CONSUMERS[1:]]}
    d = tmp / "starts"
    d.mkdir()
    s = {"boundary": _starts(d / "b.json", [file_b]),
         "deep": _starts(d / "deep.json", [file_deep]),
         "python": _starts(d / "py.json", [py]),
         "held": _starts(d / "held.json", [held]),
         "other": _starts(d / "other.json", [other]),
         "no_spikes": _starts(d / "nos.json", [no_spikes]),
         "fs": _starts(d / "fs.json", [{**file_b, "fs": FS + 1.0}], fs=FS + 1.0),
         **{f"elec{k}": _starts(d / f"elec{k}.json", [v]) for k, v in elec.items()}}
    readers = []
    for name, edit in (("schema", lambda j: j.update(schema="v0")),
                       ("twice", lambda j: j["held"].append({"session": SESSION.upper(),
                                                             "basis": "x"})),
                       ("fractional", lambda j: j["files"][0]["analyses"][0].update(
                           start_sample0=I0 + 0.5)),
                       ("v1", lambda j: j.update(schema="gems-blanking-v2 recovery starts v1")),
                       ("no_electrical", lambda j: j["files"][0].pop(
                           "electrical_settle_sample0")),
                       ("no_map", lambda j: j.pop("output_times")),
                       ("no_sources", lambda j: j.pop("source_files")),
                       ("sources_as_object", lambda j: j.update(
                           source_files=dict(rs.SOURCE_FILES))),
                       ("stale_source", lambda j: j["source_files"][0].update(
                           sha256="0" * 64)),
                       ("missing_source", lambda j: j["source_files"].append(
                           {"file": "no_such_function_k2f.m", "sha256": "0" * 64}))):
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
             plan("baseline", s["other"], bl, "baseline"),
             *(plan(f"elec{k}", s[f"elec{k}"], mode="mask_to_electrical_drop_outputs")
               for k in elec),
             plan("mode_missing", s["boundary"], mode=""),
             plan("mode_bad", s["boundary"], mode="mask_to_somewhere")]
    run = {"gems_root": (tmp / "store").as_posix(), "mask_folder": st["mask_folder"].as_posix(),
           "out_a": (tmp / "out_a").as_posix(), "out_b": (tmp / "out_b").as_posix(),
           "starts_file": s["boundary"], "other_starts_file": s["deep"],
           "record_rel": f"T/{SESSION}/{MODEL}/e{round(START)}/night6_record.json"}
    unit_dir = tmp / "unit"
    unit_dir.mkdir()
    unit = {"dir": unit_dir.as_posix(), "starts": s["boundary"], "fs": FS, "i0": UNIT_I0,
            **UNIT_L}
    batch = _batch_lists(tmp, st, s["boundary"])
    shadow = tmp / "shadow"
    shadow.mkdir()
    sources = {"dir": shadow.as_posix(), "starts": s["boundary"], "name": SHADOWED}
    return {"plans": plans, "readers": readers, "run": run, "py": py, "own": own,
            "unit": unit, "batch": batch, "sources": sources}


SHADOWED = "step3_detect.m"
"""The cited function the sources case shadows with a changed copy."""
UNIT_I0 = 500
"""The unit case's epoch start (0-based file sample): every start is UNIT_I0 + L."""


UNIT_L = {"L_hrv": 1000, "L_breathing": 2000, "L_mmc": 3000, "L_spikes": 4000}
"""Rows before each owner's start in the unit case (hand-built files)."""


def _batch_lists(tmp: Path, st: dict[str, Any], starts: str) -> dict[str, Any]:
    """Batch lists the batch must refuse at its start: no mode, a bad one, no starts."""
    pnew = _processing_new()
    base = {"gems_root": (tmp / "store").as_posix(), "units": "uV",
            "out_root": (tmp / "batch_out").as_posix(),
            "processing_new": "" if pnew is None else pnew.as_posix(),
            "recordings": [{"mask_folder": Path(st["mask_folder"]).relative_to(
                tmp / "store").as_posix()}]}
    lists = []
    for name, extra in (("no_mode", {"recovery_starts": starts}),
                        ("bad_mode", {"recovery_starts": starts,
                                      "recovery_trim_mode": "trim_it_all"}),
                        ("no_starts", {"recovery_trim_mode": "mask_to_electrical_drop_outputs"})):
        f = tmp / f"batch_{name}.json"
        f.write_text(json.dumps({**base, **extra}), encoding="utf-8", newline="\n")
        lists.append({"name": name, "file": f.as_posix()})
    return {"lists": lists}


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
    case_file.write_text(json.dumps({k: case[k] for k in ("plans", "readers", "run", "unit",
                                                          "batch", "sources")},
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

    # the drop mode: EVERY input masked on rows 1 .. ke - i0, whatever the own starts
    for k in (-1, 0, 1, 2):
        e = got[f"elec{k}"]
        _check_trim(e, dict.fromkeys(CONSUMERS, max(0, k)))
        erec = json.loads(e["record"])
        assert erec["mode"] == "mask_to_electrical_drop_outputs"
        assert erec["electrical_settle_sample0"] == I0 + k
        for c in CONSUMERS:
            ec = erec["consumers"][c]
            assert ec["trimmed_rows"] == max(0, k), (k, c)
            assert ec["output_rows_before_start"] == case["own"][c] - I0, (k, c)
            assert ec["output_status"] == "outputs before own start dropped"
    t1 = _inputs(got["elec1"]["trim"])
    assert t1[("spikes", "L_T")][:2].tolist() == [True, False]  # one sample, not two
    assert t1[("mmc", "ANT1")][:2].tolist() == [True, False]
    for name in ("mode_missing", "mode_bad"):
        assert got[name]["error"] == "night6:recoveryTrimMode", got[name]
    assert "mask_to_somewhere" in got["mode_bad"]["message"]

    readers = {r["name"]: r for r in res["readers"]}
    assert readers["ok"]["error"] == ""
    for name in ("schema", "twice", "fractional", "v1", "no_electrical", "no_map",
                 "no_sources", "sources_as_object"):
        assert readers[name]["error"] == "night6:recoveryStartFile", (name, readers[name])
    # the cited code is checked at run time against what MATLAB resolves (fix 2)
    first = rs.source_files_record()[0]["file"]
    assert readers["stale_source"]["error"] == "night6:recoveryStartSource"
    assert first in readers["stale_source"]["message"]
    assert readers["missing_source"]["error"] == "night6:recoveryStartSource"
    assert "no_such_function_k2f.m" in readers["missing_source"]["message"]
    src = res["sources"]
    shadow = (tmp_path / "shadow" / SHADOWED).as_posix().lower()
    assert src["same_which"].replace("\\", "/").lower() == shadow
    assert src["same"]["error"] == ""                       # identical bytes: accepted
    assert src["changed_which"].replace("\\", "/").lower() == shadow
    assert src["changed"]["error"] == "night6:recoveryStartSource", src["changed"]
    assert SHADOWED in src["changed"]["message"]
    assert src["after"]["error"] == ""                      # her file again: accepted

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
    assert run["resume_other_mode"] == "dry_run"   # same file, another trim mode: rerun
    assert run["mode_recorded"] == "mask_to_electrical_drop_outputs"
    assert rec["recovery_trim_mode"] == "mask_to_own_start"
    for name in ("no_mode", "bad_mode"):
        assert run[name]["error"] == "night6:recoveryTrimMode", run[name]

    _check_unit(res["unit"])
    b = res["batch"]
    for name, ident in (("no_mode", "night6:recoveryTrimMode"),
                        ("bad_mode", "night6:recoveryTrimMode"),
                        ("no_starts", "night6:recoveryStarts")):
        assert b[name]["error"] == ident, (name, b[name])
    assert "trim_it_all" in b["bad_mode"]["message"]
    assert "no recovery_starts declared" in b["no_starts"]["message"]
    assert not list((tmp_path / "batch_out").rglob("night6_record.json"))  # nothing ran


def _col(v: object) -> list[Any]:
    a = np.asarray(v, dtype=object).ravel().tolist()
    return [None if x is None else x for x in a]


def _check_unit(u: dict[str, Any]) -> None:  # noqa: PLR0915 - one hand-built case
    """Every convention cuts at exactly L: the entry at L - 1 goes, L and L + 1 stay."""
    o = u["out"]
    m = o["hrvm"]
    assert _col(m["RR_times"]) == pytest.approx([(UNIT_L["L_hrv"] + d + 1) / FS for d in (0, 1)])
    assert _col(m["RR_intervals"]) == [0.12, 0.13]                  # co-indexed, cut together
    assert _col(m["hrv_series"]) == [None, 2.0, 3.0]                # sec0: NaN, stamp kept
    assert _col(m["nRR_used"]) == [None, 5.0, 6.0]
    assert len(_col(m["metrics_t"])) == 3
    assert _col(m["heartlocs"]) == [UNIT_L["L_hrv"] + 1, UNIT_L["L_hrv"] + 2]  # row1
    assert m["hrv"] == 0.5 and _col(m["mystery"]) == [1, 2, 3]       # never cut
    hrvm = next(f for f in u["hrv"]["files"] if f["kind"] == "HRVMeasures")
    assert "hrv" in hrvm["epoch_scalars"]
    unk = {e["path"]: e["why"] for e in hrvm["untrimmed_time_convention_unknown"]}
    assert "not in the output time map" in unk["mystery"]
    hrbr = next(f for f in u["hrv"]["files"] if f["kind"] == "HRBR")
    unk = {e["path"]: e["why"] for e in hrbr["untrimmed_time_convention_unknown"]}
    assert "time convention unknown" in unk["RR_implausibleMask"]
    files = u["hrv"]["untrimmed_files"]
    files = [files] if isinstance(files, dict) else files
    assert [f["file"] for f in files] == ["e2_figure.png"]
    # owners: hrv cut at 1000, breathing at 2000 - in EVERY run, whoever ran (fix 4): an
    # hrv-only run's breath rate is cut at breathing's start, a breathing-only run's heart
    # rate and heartlocs at hrv's
    b1, b2, b3 = o["hrbr"], o["hrbr2"], u["out_breathing"]
    lh, lb = UNIT_L["L_hrv"], UNIT_L["L_breathing"]
    for b in (b1, b2, b3):
        assert _col(b["heartRateSeries"]) == [None, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert _col(b["breathRateSeries"]) == [None, None, None, None, 15.0, 16.0]
        assert _col(b["br_locs_true"]) == [lb + 1, lb + 2]
    assert _col(b3["heartlocs"]) == [lh + 1, lh + 2, lb, lb + 1, lb + 2]
    trimmed = {e["path"]: e for e in u["both"]["files"][0]["trimmed"]}
    assert trimmed["breathRateSeries"]["start_basis"] == "breathing"
    trimmed1 = {e["path"]: e for e in hrbr["trimmed"]}
    assert trimmed1["breathRateSeries"]["start_basis"].startswith("breathing's own start")
    assert "byproduct" in trimmed1["breathRateSeries"]["start_basis"]
    assert trimmed1["breathRateSeries"]["rows_before_start"] == lb
    trimmed3 = {e["path"]: e for e in u["breathing"]["files"][0]["trimmed"]}
    assert trimmed3["heartRateSeries"]["rows_before_start"] == lh
    assert trimmed3["heartlocs"]["start_basis"].startswith("hrv's own start")
    assert u["twice"]["error"] == "night6:trimTwice"
    assert u["no_owner"]["error"] == "night6:trimOwner"
    assert "breathRateSeries" in u["no_owner"]["message"]
    _check_unit_markers(u)
    # mmc: the delay is cut at its TRUE centre (stamp + W/2 - S), not at its stamp
    mm = o["mmc"]["mmc"]
    assert [r[0] for r in mm["delay"]] == [None, 2, 3]
    assert [r[0] for r in mm["signal"]] == [None, 2, 3]
    assert [r[0] for r in mm["firing"]["rate"]] == [None, 1, 1]
    assert mm["firing"]["events"] == [[0, 0, 0], [1, 1, 1], [1, 1, 1]]
    assert mm["firing"]["avgRate"] == [1, 2, 3]
    # spikes: struct array element by element, co-indexed drop, cells, still v7.3
    sp = o["spk"]["spikes"]
    ls = UNIT_L["L_spikes"]
    assert _col(sp[0]["centers"]) == [ls + 1, ls + 2]
    assert _col(sp[1]["centers"]) == [ls + 1, ls + 2]                 # nothing before
    assert sp[0]["waveforms"] == [[2, 2], [3, 3]] and sp[1]["waveforms"] == [[4, 4], [5, 5]]
    assert _col(sp[0]["times"]) == pytest.approx([ls / FS, (ls + 1) / FS])
    sw = o["spk"]["sigmaWin"]
    assert [_col(c) for c in sw["sigma"]] == [[None, 2.0, 3.0], [None, 5.0, 6.0]]
    assert u["spikes_still_v73"] is True
    assert u["shape"]["error"] == "night6:trimShape"


def _as_list(v: object) -> list[Any]:
    return v if isinstance(v, list) else [v]


def _check_unit_markers(u: dict[str, Any]) -> None:  # noqa: PLR0915 - one marker per file
    """Check the marker each trimmed file carries (fix 1): exact, for every action."""
    o = u["out"]
    name = u["marker_variable"]
    assert name == rs.TRIM_MARKER
    lh, lm, ls = UNIT_L["L_hrv"], UNIT_L["L_mmc"], UNIT_L["L_spikes"]

    def var(m: dict[str, Any], path: str) -> dict[str, Any]:
        hit = [v for v in _as_list(m["vars"]) if v["path"] == path]
        assert len(hit) == 1, (path, [v["path"] for v in _as_list(m["vars"])])
        return hit[0]

    def start(v: dict[str, Any], owner_l: int, conv: str, action: str) -> None:
        assert v["mode"] == "mask_to_electrical_drop_outputs"
        assert v["convention"] == conv and v["action"] == action
        assert v["convention_meaning"] == rs.CONVENTIONS[conv]
        assert v["start_sample0"] == UNIT_I0 + owner_l
        assert v["start_s"] == pytest.approx((UNIT_I0 + owner_l) / FS, abs=1e-12)
        assert v["first_computed_epoch_row"] == owner_l + 1
        assert v["first_computed_epoch_sample0"] == owner_l

    m = o["hrvm"][name]
    assert m["mode"] == "mask_to_electrical_drop_outputs"
    assert m["epoch_start_sample0"] == UNIT_I0 and m["electrical_settle_sample0"] == UNIT_I0 + 7
    assert _as_list(m["run_consumers"]) == ["hrv"]
    assert "mystery" in _as_list(m["untrimmed"]) and "hrv" in _as_list(m["epoch_scalars"])
    assert {v["path"] for v in _as_list(m["vars"])} == {
        "RR_intervals", "RR_times", "hrv_series", "nRR_used", "heartlocs"}
    v = var(m, "RR_times")
    start(v, lh, "sec_row1", "drop")
    (leaf,) = _as_list(v["leaves"])
    assert leaf["leaf"] == "RR_times" and leaf["n_entries"] == 3
    assert leaf["n_not_computed"] == leaf["n_dropped"] == 1
    assert _rows(leaf["not_computed_rows"]) == []
    v = var(m, "hrv_series")
    start(v, lh, "sec0", "nan")
    (leaf,) = _as_list(v["leaves"])
    assert leaf["n_not_computed"] == 1 and leaf["n_dropped"] == 0
    assert _rows(leaf["not_computed_rows"]) == [[1, 1]]
    # mmc: false (an event series: NOT computed, not "no event") and the delay
    mm = o["mmc"][name]
    v = var(mm, "mmc.firing.events")
    start(v, lm, "sec0", "false")
    (leaf,) = _as_list(v["leaves"])
    assert leaf["leaf"] == "mmc.firing.events" and _rows(leaf["not_computed_rows"]) == [[1, 1]]
    v = var(mm, "mmc.delay")
    start(v, lm, "sec_xchan_delay", "nan")
    assert _rows(_as_list(v["leaves"])[0]["not_computed_rows"]) == [[1, 1]]
    # spikes: a struct array and cells, leaf by leaf (v7.3 file)
    sm = o["spk"][name]
    v = var(sm, "spikes.centers")
    start(v, ls, "row1", "drop")
    assert [(x["leaf"], x["n_dropped"]) for x in _as_list(v["leaves"])] == [
        ("spikes(1).centers", 1), ("spikes(2).centers", 0)]
    v = var(sm, "sigmaWin.sigma")
    assert [(x["leaf"], _rows(x["not_computed_rows"])) for x in _as_list(v["leaves"])] == [
        ("sigmaWin.sigma{1}", [[1, 1]]), ("sigmaWin.sigma{2}", [[1, 1]])]
    # a breathing-only run: hrv's byproducts carry HRV's start
    bm = u["out_breathing"][name]
    v = var(bm, "heartRateSeries")
    start(v, lh, "sec0", "nan")
    assert v["owner"] == "hrv" and "byproduct" in v["start_basis"]
    v = var(bm, "breathRateSeries")
    start(v, UNIT_L["L_breathing"], "sec0", "nan")
    assert _rows(_as_list(v["leaves"])[0]["not_computed_rows"]) == [[1, 4]]


def test_run_epoch_always_hands_the_starts_to_the_planner() -> None:
    """The planner's 'Recovery' is optional for harnesses; the entry point never omits it."""
    code = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    assert code.count("night6_prepare_epoch(") == 1
    assert ("'Recovery', struct('starts', o.RS, 'session', src.session, ...\n"
            "                                                   'mode', o.RecoveryTrimMode)"
            in code)
    batch = (NIGHT6 / "night6_batch.m").read_text(encoding="utf-8")
    assert "'RecoveryStarts', starts" in batch
    assert "'RecoveryTrimMode', mode}" in batch


def test_the_trim_modes_are_one_list_on_both_sides() -> None:
    """Python's TRIM_MODES and night6_trim_modes.m name the same two modes (invariant 33)."""
    m = (NIGHT6 / "night6_trim_modes.m").read_text(encoding="utf-8")
    got = re.search(r"modes = \{([^}]*)\};", m)
    assert got
    assert tuple(re.findall(r"'([^']+)'", got.group(1))) == rs.TRIM_MODES


TRIM_KE = seconds_to_sample(2.5, FS)
"""The electrical settling of the end-to-end case: 0.5 s into the epoch."""
TRIM_OWN = {"spikes": 32.0, "mmc": 35.0, "hrv": 37.0, "slow_wave": 38.0, "breathing": 42.0}
"""Each analysis's own start (s): all later than the electrical settling, all different,
each placed so EVERY variable it owns has entries on both sides (her windows decide where
values exist in a 73 s epoch: the 60 s HR/BR windows at centres 30-43 s, the 30 s CV2
windows at 15/45 s, slow-wave peaks 15 s inside the ends, the 30 s mmc delay windows at
true centres 19.5 + 5k s). mmc at 33 s into the epoch sits in (34.5 - 4, 34.5]: the delay
whose STAMP is before the start and whose true centre is after it, so a delay cut at its
stamp is visible. hrv before breathing by 5 s: a breathing-only run's hrv byproducts are
cut at hrv's start, an hrv-only run's breathing byproducts at breathing's."""


TRIM_N_FILE = int(75.0 * FS)
"""75 s (a 73 s epoch): long enough for every output of her functions to hold entries on
both sides of every own start (her gap-aware DFA also needs more RR intervals than the
10 s file has - processing_new dfaGapAware.m:175, reported, not ours)."""


def _real_beats(seed: int = 6) -> np.ndarray:
    """R-peaks every 0.12 +/- 0.01 s, none in BEAT_BLANK_S, inside the file."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(0.12 + rng.uniform(-0.01, 0.01, 700)) + 0.05
    t = t[t < TRIM_N_FILE / FS - 0.1]
    return t[~((t >= BEAT_BLANK_S[0][0]) & (t < BEAT_BLANK_S[0][1]))]


def _real_signal(seed: int = 5) -> np.ndarray:
    """Noise, spikes on both cuffs' middle contacts, slow waves on ANT1-3, microvolts."""
    rng = np.random.default_rng(seed)
    n_file = TRIM_N_FILE
    t = np.arange(n_file) / FS
    y = rng.normal(0.0, 5.0, (n_file, len(LABELS)))
    shape = -80.0 * np.exp(-0.5 * (np.arange(-30, 31) / 6.0) ** 2)
    for lab in ("LVN2", "RVN2"):
        j = LABELS.index(lab)
        for s in np.sort(rng.choice(np.arange(40, n_file - 40), 3000, replace=False)):
            y[s - 30:s + 31, j] += shape
    for k, lab in enumerate(("ANT1", "ANT2", "ANT3")):
        y[:, LABELS.index(lab)] += 150.0 * np.sin(2.0 * np.pi * 0.25 * t + 0.4 * k)
    return y


def _trim_case(tmp_path: Path) -> dict[str, Any]:
    st = _store(tmp_path / "store", signal=_real_signal(), beats_s=_real_beats())
    own = {c: seconds_to_sample(s, FS) for c, s in TRIM_OWN.items()}
    d = tmp_path / "starts"
    d.mkdir()
    drop = _starts(d / "drop.json", [_file([_row(c, own[c]) for c in CONSUMERS], ke=TRIM_KE)])
    ref = _starts(d / "ref.json", [_file([_row(c, TRIM_KE) for c in CONSUMERS], ke=TRIM_KE)])
    return {"trim": {"gems_root": (tmp_path / "store").as_posix(),
                     "mask_folder": st["mask_folder"].as_posix(),
                     "out_drop": (tmp_path / "out_drop").as_posix(),
                     "out_ref": (tmp_path / "out_ref").as_posix(),
                     "starts_drop": drop, "starts_ref": ref, "fs": FS, "i0": I0,
                     "epoch_rel": f"T/{SESSION}/{MODEL}/e{round(START)}",
                     "start_sample0": own,
                     "L": {c: own[c] - I0 for c in CONSUMERS}},
            "plans": [], "readers": []}


def test_the_drop_mode_cuts_her_real_outputs_at_each_own_start(  # noqa: PLR0915 - one run
        tmp_path: Path) -> None:
    """End to end: dropped or flagged before each OWNER's start, bit-identical after it.

    The same epoch is run twice with her real functions on the same masked input (every
    input masked to the electrical settling): the drop mode, and the untrimmed reference
    (mask_to_own_start with every own start AT the electrical settling). An oracle in the
    harness with its OWN convention table (``oracle_table``, transcribed from her code, not
    read from the map) checks every trimmed variable of every output file, the marker
    each trimmed file carries, and that every other variable is untouched. Every trimmed
    variable has entries on both sides of its owner's start, with values in the reference
    on both sides - so a wrong convention, owner or action shows on the data.
    """
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    case = _trim_case(tmp_path)
    own = case["trim"]["start_sample0"]
    case_file, res_file = tmp_path / "trim_case.json", tmp_path / "trim_result.json"
    case_file.write_text(json.dumps(case, ensure_ascii=True), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); "
           f"check_recovery_start('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=2400, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    r = json.loads(res_file.read_text(encoding="utf-8"))["trim"]
    rd, rr = json.loads(r["record_drop"]), json.loads(r["record_ref"])
    assert rd["status"] == rr["status"] == "complete", (rd.get("runs"), rr.get("runs"))
    assert rd["recovery_trim_mode"] == "mask_to_electrical_drop_outputs"
    assert rr["recovery_trim_mode"] == "mask_to_own_start"
    for c in CONSUMERS:  # one input mask for all, in both runs; outputs cut only in drop
        assert rd["recovery_start"]["consumers"][c]["trimmed_rows"] == TRIM_KE - I0
        assert rr["recovery_start"]["consumers"][c]["trimmed_rows"] == TRIM_KE - I0
        assert rd["recovery_start"]["consumers"][c]["output_rows_before_start"] == own[c] - I0
        a = rd["recovery_start"]["analyses"][c]   # every analysis's start, run or not
        assert a["start_sample0"] == own[c] and a["output_rows_before_start"] == own[c] - I0
    runs = [x for x in rd["runs"] if x["status"] == "ok"]
    assert runs and all("recovery_trim" in x for x in runs)
    assert not any("recovery_trim" in x for x in rr["runs"])
    files = {f["file"]: f for f in r["files"]}
    kinds = {f["kind"] for f in files.values()}
    assert {"spikes_v2", "HRBR", "HRVMeasures", "slowWaves", "mmc"} <= kinds
    assert all(f["rest_equal"] for f in files.values()), [f for f in files.values()
                                                          if not f["rest_equal"]]
    bad = [v for v in r["vars"] if not v["ok"]]
    assert not bad, bad
    # the marker: in every trimmed file, exact for every variable it cut, nothing else
    assert r["marker_variable"] == rs.TRIM_MARKER
    trimmed_files = [f for f in files.values() if f["kind"]]
    assert all(f["has_marker"] for f in trimmed_files), trimmed_files
    assert not [f for f in trimmed_files if f["marker_extra"]]
    bad = [v for v in r["vars"] if v["why"] != "absent" and not v["marker_ok"]]
    assert not bad, bad
    # the oracle's table and the map name the same trimmed variables
    assert r["table_vs_map"] == {"map_only": [], "oracle_only": []}, r["table_vs_map"]
    # every trimmed variable: entries before AND after its owner's start, and values in
    # the reference on both sides (otherwise a wrong cut there could not show)
    cover: dict[tuple[str, str], list[int]] = {}
    for v in r["vars"]:
        if v["why"] == "absent":
            continue
        kind = files[v["file"]]["kind"]
        c = cover.setdefault((kind, v["path"]), [0, 0, 0, 0])
        for i, k in enumerate(("n_before", "n_after", "n_before_value", "n_after_value")):
            c[i] += v[k]
    want = {(x.file, x.path) for x in rs.OUTPUT_VARS if x.role == "trim"}
    assert set(cover) == want, (sorted(want - set(cover)), sorted(set(cover) - want))
    thin = {k: c for k, c in cover.items() if min(c) == 0}
    assert not thin, thin
    # fix 4: in each HR run, every output is cut at its OWNER's start, whoever ran
    per = {(v["file"].split("_", 1)[1], v["path"]): v for v in r["vars"]
           if v["why"] != "absent"}
    for want_cut in (("breathing_HRBR.mat", "heartRateSeries"),     # hrv-owned, breathing run
                     ("breathing_HRBR.mat", "heartlocs"),
                     ("breathing_HRVMeasures.mat", "RR_times"),
                     ("breathing_HRVMeasures.mat", "hrv_series"),
                     ("hrv_HRBR.mat", "breathRateSeries"),          # breathing-owned, hrv run
                     ("hrv_HRBR.mat", "br_locs_true")):
        v = per[want_cut]
        assert v["n_before"] > 0 and v["n_after"] > 0, (want_cut, v)
    for x in runs:
        if x["call"] != "HR_BR_HRVAnalysis_beats":
            continue
        for f in _as_list(x["recovery_trim"]["files"]):
            for e in _as_list(f["trimmed"]):
                assert e["rows_before_start"] == own[e["owner"]] - I0, (x["consumers"], e)
