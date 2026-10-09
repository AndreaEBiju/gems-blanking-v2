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
  sec_xchan_delay) at L - 1, L and L + 1; owners with different starts in one file; a
  byproduct; struct arrays, cells and co-indexed lists; v7.3 kept; an unmapped and an
  'unknown' variable and a figure listed untrimmed by name; a stamp that does not fit
  its value refused;
* end to end, with her real functions: the drop mode against an untrimmed run on the
  same masked input - every entry before its own start dropped or flagged, every entry
  at or after it bit-identical, every other variable of every file untouched;
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
                       ("no_map", lambda j: j.pop("output_times"))):
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
    unit = {"dir": unit_dir.as_posix(), "starts": s["boundary"], "fs": FS, **UNIT_L}
    batch = _batch_lists(tmp, st, s["boundary"])
    return {"plans": plans, "readers": readers, "run": run, "py": py, "own": own,
            "unit": unit, "batch": batch}


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
                                                          "batch")},
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
    for name in ("schema", "twice", "fractional", "v1", "no_electrical", "no_map"):
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


def _check_unit(u: dict[str, Any]) -> None:
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
    # owners: hrv cut at 1000, breathing at 2000; an hrv-only run's breath rate is a
    # byproduct cut at the run's own start
    b1, b2 = o["hrbr"], o["hrbr2"]
    assert _col(b1["heartRateSeries"]) == [None, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert _col(b1["breathRateSeries"]) == [None, 12.0, 13.0, 14.0, 15.0, 16.0]
    assert _col(b2["heartRateSeries"]) == [None, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert _col(b2["breathRateSeries"]) == [None, None, None, None, 15.0, 16.0]
    lb = UNIT_L["L_breathing"]
    assert _col(b2["br_locs_true"]) == [lb + 1, lb + 2]
    trimmed = {e["path"]: e for e in u["both"]["files"][0]["trimmed"]}
    assert trimmed["breathRateSeries"]["start_basis"] == "breathing"
    trimmed1 = {e["path"]: e for e in hrbr["trimmed"]}
    assert "byproduct" in trimmed1["breathRateSeries"]["start_basis"]
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
TRIM_OWN = {"spikes": 3.0, "slow_wave": 3.5, "mmc": 4.0, "hrv": 6.0, "breathing": 7.0}
"""Each analysis's own start (s): all later than the electrical settling, all different."""


TRIM_N_FILE = int(40.0 * FS)
"""40 s: her gap-aware DFA needs more RR intervals than the 10 s file has (it indexes an
empty scale list with 22 - processing_new dfaGapAware.m:175, reported, not ours)."""


def _real_beats(seed: int = 6) -> np.ndarray:
    """R-peaks every 0.12 +/- 0.01 s, none in BEAT_BLANK_S, inside the 40 s file."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(0.12 + rng.uniform(-0.01, 0.01, 400)) + 0.05
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
        for s in np.sort(rng.choice(np.arange(40, n_file - 40), 1600, replace=False)):
            y[s - 30:s + 31, j] += shape
    for k, lab in enumerate(("ANT1", "ANT2", "ANT3")):
        y[:, LABELS.index(lab)] += 150.0 * np.sin(2.0 * np.pi * 0.25 * t + 0.4 * k)
    return y


def test_the_drop_mode_cuts_her_real_outputs_at_each_own_start(tmp_path: Path) -> None:
    """End to end: dropped or flagged before each own start, bit-identical at and after it.

    The same epoch is run twice with her real functions on the same masked input (every
    input masked to the electrical settling): the drop mode, and the untrimmed reference
    (mask_to_own_start with every own start AT the electrical settling). An oracle in the
    harness, built from the map's definitions, checks every trimmed variable of every
    output file, and that every other variable is untouched.
    """
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    st = _store(tmp_path / "store", signal=_real_signal(), beats_s=_real_beats())
    own = {c: seconds_to_sample(s, FS) for c, s in TRIM_OWN.items()}
    d = tmp_path / "starts"
    d.mkdir()
    drop = _starts(d / "drop.json", [_file([_row(c, own[c]) for c in CONSUMERS], ke=TRIM_KE)])
    ref = _starts(d / "ref.json", [_file([_row(c, TRIM_KE) for c in CONSUMERS], ke=TRIM_KE)])
    case = {"trim": {"gems_root": (tmp_path / "store").as_posix(),
                     "mask_folder": st["mask_folder"].as_posix(),
                     "out_drop": (tmp_path / "out_drop").as_posix(),
                     "out_ref": (tmp_path / "out_ref").as_posix(),
                     "starts_drop": drop, "starts_ref": ref, "fs": FS,
                     "epoch_rel": f"T/{SESSION}/{MODEL}/e{round(START)}",
                     "L": {c: own[c] - I0 for c in CONSUMERS}},
            "plans": [], "readers": []}
    case_file, res_file = tmp_path / "trim_case.json", tmp_path / "trim_result.json"
    case_file.write_text(json.dumps(case, ensure_ascii=True), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); "
           f"check_recovery_start('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=1800, check=False)
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
    both = {(v["file"].split("_", 1)[1], v["path"]) for v in r["vars"]
            if v["n_before"] > 0 and v["n_after"] > 0}
    # (hrv's own beats before 6 s sit inside her 0.75 s buffer of the 3.5-5.5 s beat
    # blank, so its heartlocs have none before the start; the breathing run's do)
    for want in (("hrv_HRBR.mat", "heartBeatSeries"), ("breathing_HRBR.mat", "heartlocs"),
                 ("breathing_HRVMeasures.mat", "RR_times"), ("hrv_HRBR.mat", "heartRateSeries"),
                 ("mmc_in_mmc.mat", "mmc.signal"), ("mmc_in_mmc.mat", "mmc.firing.events"),
                 ("spikes_v2.mat", "spikes.centers"), ("spikes_v2.mat", "spikes.waveforms"),
                 ("spikes_v2.mat", "envelope.rms_uv"), ("spikes_v2.mat", "metrics.fr_hz"),
                 ("spikes_v2.mat", "metrics.burst.onsets"),
                 ("breathing_HRBR.mat", "br_locs_true"), ("hrv_HRBR.mat", "breathRateSeries")):
        assert want in both, (want, sorted(both))
    assert any(p == "slowWaveTimeSeries" for _, p in both)
