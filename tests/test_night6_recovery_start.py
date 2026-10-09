"""Night 6 trimming by recovery start, MATLAB side, checked from Python.

RULING 2026-10-08 (k) 2 (each analysis's start) and RULING 2026-10-09 item 6 (mode (B),
per output variable). A synthetic stim_rec recording (the wrapper test's 9-contact store:
every sample encodes its file index, a whole-file beat train, a mask file written by the
real ``emit.handoff.write_mask_file`` for an epoch starting at 2.0 s) is planned by
``night6_prepare_epoch`` with and without its recovery starts, in ONE MATLAB process
(``tests/matlab/check_recovery_start.m``). Checks:

* mode (B) masks EVERY consumer's input on exactly rows 1 .. ke - i0 (ke = stim-off +
  electrical settling, one point for all analyses), the 1-sample boundary on both sides,
  and nothing else changes; a start or electrical settling before the epoch masks nothing
  and is recorded "early part deferred to add-on";
* mode (A), ``mask_to_own_start``, is refused by name with RULING 2026-10-09 item 6 - by
  the planner, the entry point and the batch - and so are a missing and an unknown mode;
  a complete record made under (A) reruns;
* the starts come from the Python writer (``write_recovery_starts``): an analysis whose
  reach is unknown keeps the fixed start, labelled, and so does every cut it owns;
* a slow-wave lead-in that swallows the only difference between the ANT masks makes them
  identical, and they share one run that keeps all three (the (j) 5 (a) shortcut, exact);
* refused by name, never a silent fallback: a stim_rec epoch with no starts file, a held
  file, a file with no row, a consumer that runs with no row, an fs mismatch, a row for a
  recording that is not stim_rec; a baseline without a row is untouched;
* the reader refuses another schema (v2 too: no cuts), a session twice
  (case-insensitively), a non-integer start sample, a file with no cuts or a cut twice;
* ``night6_run_recording``: stim_rec without RecoveryStarts is refused; with it, every
  epoch record carries each consumer's start, basis, source, every cut and the file's
  SHA-256; a complete record made with the same file and mode is resumed, with another
  file or under (A) rerun;
* the per-variable cut, unit level, on hand-built files: every class and stamp convention
  at L - 1, L and L + 1 of ITS OWN cut (every cut different); byproducts cut at their own
  cut whoever ran; struct arrays, cells and co-indexed lists; v7.3 kept; an unmapped and
  an 'unknown' variable and a figure listed untrimmed; the valid fractions of every
  computed kind EXACT on known masks (computed here, independently); the recomputed
  averages and counts; not-computed mmc events NaN in a double series; and refused by
  name: a file trimmed twice, a cut the row lacks, no class or an unknown class, a sibling
  that exists, a stamp that does not fit;
* the marker (``rs.TRIM_MARKER``) every trimmed file carries, exact: class, cut, owner,
  action, stamp, convention, the cut point in seconds and samples, the first computed
  epoch row and sample, the mode, per leaf the not-computed rows, the valid-fraction
  variable, the added variables and every recomputed value with her original;
* the cited processing_new files are checked at run time: a stale hash, a missing file,
  and a changed copy of a cited function first on the path are refused by name
  (``night6:recoveryStartSource``); a byte-identical copy is not;
* end to end, with her real functions, on a 73 s epoch: mode (B) with a different cut for
  every cut id against a reference cut at the electrical settling on the same masked
  input, through an oracle with its OWN tables (not the map) of conventions, classes,
  cuts, windowed variables and averages - every entry before its cut dropped or flagged,
  every entry at or after it bit-identical, every valid fraction the same in both runs
  and consistent with her own validity rules, every average her expression over the kept
  values, every other variable untouched, the marker exact, and every trimmed variable
  with entries (and reference values) on both sides;
* ``night6_batch`` refuses at batch start - before any signal is loaded - a stim_rec mask
  with no recovery_starts, and a missing, unknown or withdrawn mode.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from itertools import pairwise
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
MODE_B = "mask_to_electrical_drop_outputs"
MODE_A = "mask_to_own_start"


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


CUT_IDS = tuple(rs.cut_id(*c) for c in rs.trim_cuts())
"""Every cut the map uses (owner.output.class)."""


def _cuts(cut_at: Mapping[str, int] | None, default: int) -> list[dict[str, Any]]:
    """Every cut of the map, at ``cut_at[cut]`` (a 0-based FILE sample) or ``default``."""
    out = []
    for owner, key, cls in rs.trim_cuts():
        cid = rs.cut_id(owner, key, cls)
        k = default if cut_at is None else cut_at.get(cid, default)
        out.append({"cut": cid, "owner": owner, "output_key": key, "trim_class": cls,
                    "start_s": k / FS, "start_sample0": k, "basis": rs.BASIS_CUT,
                    "source": "synthetic"})
    return out


def _file(rows: list[dict[str, Any]], ke: int = I0, session: str = SESSION,
          cut_at: Mapping[str, int] | None = None) -> dict[str, Any]:
    return {"session": session, "fs": FS, "electrical_settle_sample0": ke, "analyses": rows,
            "cuts": _cuts(cut_at, ke)}


def _row(name: str, k0: int, basis: str = rs.BASIS_MEASURED) -> dict[str, Any]:
    return {"analysis": name, "start_s": k0 / FS, "start_sample0": k0, "basis": basis,
            "source": "synthetic"}


def _starts(path: Path, files: list[dict[str, Any]], fs: float = FS) -> str:
    rs.write_recovery_starts(path, rs.recovery_starts_document(files, fs=fs))
    return path.as_posix()


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
"""For the writer-to-wrapper case: slow wave unknown, the others measured; no output is
keyed, so every cut is unknown too and keeps the fixed start, labelled."""


def _cases(tmp: Path) -> dict[str, Any]:
    st = _store(tmp / "store")
    bl = _store(tmp / "store_bl", condition="baseline")
    common = {"labels": list(LABELS), "fs": FS, "n_file": N_FILE, "session": SESSION,
              "meta_json": st["meta_json"]}

    def plan(name: str, starts: str, store: dict[str, Any] = st,
             condition: str = "stim_recovery", mode: str = MODE_B) -> dict[str, Any]:
        return {**common, "name": name, "starts_file": starts, "condition": condition,
                "mode": mode,
                "mask_file": store["mask_file"].as_posix(),
                "beats_file": store["beats_file"].as_posix(),
                "record_json": store["record_json"]}

    file_b = _file([_row(c, I0 + 1) for c in CONSUMERS])
    file_deep = _file([_row(c, DEEP_SW) for c in CONSUMERS], ke=DEEP_SW)
    own = {c: I0 + 1000 * (k + 3) for k, c in enumerate(CONSUMERS)}
    cut_own = {cid: own[cid.split(".")[0]] for cid in CUT_IDS}
    elec = {d: _file([_row(c, own[c]) for c in CONSUMERS], ke=I0 + d, cut_at=cut_own)
            for d in (-1, 0, 1, 2)}
    py = rs.file_starts(session=SESSION, fs=FS, stim_off_s=1.0, electrical_settle_s=1.5,
                        stim_off_source="synthetic", electrical_source="synthetic",
                        table=TEST_TABLE, fixed_start_s=START)
    held = rs.file_starts(session=SESSION, fs=FS, stim_off_s=None, electrical_settle_s=None,
                          stim_off_source="a", electrical_source="b")
    other = _file([_row("spikes", I0)], session="someone_else")
    no_spikes = {**file_b, "analyses": [_row(c, I0 + 1) for c in CONSUMERS[1:]]}
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
                       ("v2", lambda j: j.update(schema="gems-blanking-v2 recovery starts v2")),
                       ("no_electrical", lambda j: j["files"][0].pop(
                           "electrical_settle_sample0")),
                       ("no_cuts", lambda j: j["files"][0].pop("cuts")),
                       ("cut_twice", lambda j: j["files"][0]["cuts"].append(
                           j["files"][0]["cuts"][0])),
                       ("cut_fractional", lambda j: j["files"][0]["cuts"][0].update(
                           start_sample0=I0 + 0.5)),
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
    plans = [plan("mode_a", s["boundary"], mode=MODE_A), plan("deep", s["deep"]),
             plan("python", s["python"]), plan("no_file", ""), plan("held", s["held"]),
             plan("no_row", s["other"]), plan("no_spikes", s["no_spikes"]),
             plan("fs", s["fs"]),
             plan("baseline_row", s["boundary"], bl, "baseline"),
             plan("baseline", s["other"], bl, "baseline"),
             *(plan(f"elec{k}", s[f"elec{k}"]) for k in elec),
             plan("mode_missing", s["boundary"], mode=""),
             plan("mode_bad", s["boundary"], mode="mask_to_somewhere")]
    run = {"gems_root": (tmp / "store").as_posix(), "mask_folder": st["mask_folder"].as_posix(),
           "out_a": (tmp / "out_a").as_posix(), "out_b": (tmp / "out_b").as_posix(),
           "starts_file": s["elec1"], "other_starts_file": s["deep"],
           "record_rel": f"T/{SESSION}/{MODEL}/e{round(START)}/night6_record.json"}
    unit_dir = tmp / "unit"
    unit_dir.mkdir()
    unit = {"dir": unit_dir.as_posix(), "starts": s["boundary"], "fs": FS, "i0": UNIT_I0,
            **UNIT}
    batch = _batch_lists(tmp, st, s["boundary"])
    shadow = tmp / "shadow"
    shadow.mkdir()
    sources = {"dir": shadow.as_posix(), "starts": s["boundary"], "name": SHADOWED}
    return {"plans": plans, "readers": readers, "run": run, "py": py, "own": own,
            "unit": unit, "batch": batch, "sources": sources}


SHADOWED = "step3_detect.m"
"""The cited function the sources case shadows with a changed copy."""
UNIT_I0 = 500
"""The unit case's epoch start (0-based file sample): every cut is UNIT_I0 + L."""
UNIT_L = {cid: 1000 + 200 * j for j, cid in enumerate(CUT_IDS)}
"""Rows before each cut in the unit case: every cut different."""
UNIT_N = 6000
"""Rows of the unit case's full-rate inputs (invalidMask, mmc signal and events)."""
_LB = UNIT_L["hrv.beats.valid_only"]
_LC = UNIT_L["hrv.count_hrv.valid_only"]
_LH = UNIT_L["hrv.heart_rate.valid_only"]
_LT = UNIT_L["hrv.heart_band_trace.filled_or_filtered"]
_LBR = UNIT_L["breathing.breath_rate.valid_only"]
_LTR = UNIT_L["breathing.breath_troughs.valid_only"]
_LE = UNIT_L["mmc.mmc_events.filled_or_filtered"]
_LR = UNIT_L["mmc.mmc_rate.valid_only"]
_LD = UNIT_L["mmc.mmc_delay.filled_or_filtered"]
_LSW = UNIT_L["slow_wave.sw_rate.filled_or_filtered"]
_LST = UNIT_L["spikes.spike_times.filled_or_filtered"]
_LW = UNIT_L["spikes.spike_waveforms.filled_or_filtered"]
_LSG = UNIT_L["spikes.sigma_windows.valid_only"]
_LCV = UNIT_L["spikes.cv2.valid_only"]
UNIT = {
    "n": UNIT_N, "cut_L": [{"cut": c, "L": v} for c, v in UNIT_L.items()],
    # invalid input rows (1-based inclusive) crossing the window EDGES of the HR and
    # slow-wave stamps at L - 1, L, L + 1, so their fractions differ from 1 and from
    # each other by one sample
    "invalid": [[_LBR - 30, _LBR - 19], [_LC - 20, _LC - 11], [_LH + 20, _LH + 25],
                [_LSW - 20, _LSW - 14]],
    "hrbr_w": 40, "win_w": 24, "sw_w": 30, "mmc_w": 40, "sigma_w": 20, "cv2_w": 15.3,
    # mmc: NaN signal rows [channel, first, last]; events [channel, row]
    "mmc_nan": [[1, _LR - 25, _LR - 10], [2, _LR + 5, _LR + 30]],
    "ev_rows": [[1, 100], *([c, r] for c in (1, 2, 3) for r in (_LE, _LE + 1, _LE + 50)),
                [2, _LE + 100]],
    "spk_n": 40, "spk_invalid": [[1, 3, 5], [1, 30, 33]],
}
"""The unit case's hand-built inputs (window lengths in samples; cv2_w may be fractional
so that no CV2 bin edge falls on a sample)."""


def _batch_lists(tmp: Path, st: dict[str, Any], starts: str) -> dict[str, Any]:
    """Batch lists the batch must refuse at its start: no mode, a bad one, (A), no starts."""
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
                        ("mode_a", {"recovery_starts": starts, "recovery_trim_mode": MODE_A}),
                        ("no_starts", {"recovery_trim_mode": MODE_B})):
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


def _as_list(v: object) -> list[Any]:
    return v if isinstance(v, list) else [v]


def _run_harness(tmp_path: Path, case: dict[str, Any], timeout: int) -> dict[str, Any]:
    matlab, pnew = _matlab(), _processing_new()
    assert matlab is not None and pnew is not None
    case_file, res_file = tmp_path / "rs_case.json", tmp_path / "rs_result.json"
    case_file.write_text(json.dumps(case, ensure_ascii=True), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); "
           f"check_recovery_start('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=timeout, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return dict(json.loads(res_file.read_text(encoding="utf-8")))


def _need_matlab() -> None:
    if _matlab() is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if _processing_new() is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")


def test_night6_trims_in_mode_b_to_the_sample(  # noqa: PLR0915 - one MATLAB run
        tmp_path: Path) -> None:
    _need_matlab()
    case = _cases(tmp_path)
    res = _run_harness(tmp_path, {k: case[k] for k in ("plans", "readers", "run", "unit",
                                                       "batch", "sources")}, 1200)
    got = {p["name"]: p for p in res["plans"]}

    # (A) is withdrawn: refused by name, with the ruling
    assert got["mode_a"]["error"] == "night6:recoveryTrimMode", got["mode_a"]
    assert "RULING 2026-10-09 item 6" in got["mode_a"]["message"]
    assert MODE_A in got["mode_a"]["message"]

    # mode (B): EVERY input masked on rows 1 .. ke - i0, whatever the analyses' starts
    for k in (-1, 0, 1, 2):
        e = got[f"elec{k}"]
        _check_trim(e, dict.fromkeys(CONSUMERS, max(0, k)))
        erec = json.loads(e["record"])
        assert erec["mode"] == MODE_B
        assert erec["electrical_settle_sample0"] == I0 + k
        for c in CONSUMERS:
            ec = erec["consumers"][c]
            assert ec["trimmed_rows"] == max(0, k), (k, c)
            assert ec["output_rows_before_start"] == case["own"][c] - I0, (k, c)
            assert ec["output_status"].startswith("outputs before their own cuts"), ec
        word = {-1: "early part deferred to add-on", 0: "at_epoch_start", 1: "trimmed",
                2: "trimmed"}[k]
        assert {erec["consumers"][c]["status"] for c in CONSUMERS} == {word}
        cuts = {c["cut"]: c for c in _as_list(erec["cuts"])}
        assert set(cuts) == set(CUT_IDS)
        for cid, c in cuts.items():
            owner = cid.split(".")[0]
            assert c["start_sample0"] == case["own"][owner], cid
            assert c["output_rows_before_start"] == case["own"][owner] - I0, cid
    t1 = _inputs(got["elec1"]["trim"])
    assert t1[("spikes", "L_T")][:2].tolist() == [True, False]  # one sample, not two
    assert t1[("mmc", "ANT1")][:2].tolist() == [True, False]
    t2 = _inputs(got["elec2"]["trim"])
    assert t2[("breathing", HR)][:3].tolist() == [True, True, False]
    for name in ("mode_missing", "mode_bad"):
        assert got[name]["error"] == "night6:recoveryTrimMode", got[name]
    assert "mask_to_somewhere" in got["mode_bad"]["message"]

    # slow wave: an electrical lead-in past the only difference makes the masks one run
    d = got["deep"]
    _check_trim(d, dict.fromkeys(CONSUMERS, DEEP_SW - I0))
    sw_base = [e for e in d["base"] if e["call"] == "slowWaveAnalysis_new"]
    sw_trim = [e for e in d["trim"] if e["call"] == "slowWaveAnalysis_new"]
    assert len({e["mask_signal"] for e in sw_base}) == 2
    assert {e["mask_signal"] for e in sw_trim} == {"ANT1"}

    # from the Python writer: electrical settling before the epoch (nothing masked); slow
    # wave's own settling unknown keeps the fixed start, and so does every unknown cut
    p = got["python"]
    _check_trim(p, dict.fromkeys(CONSUMERS, 0))
    prec = json.loads(p["record"])
    pc = prec["consumers"]
    assert pc["slow_wave"]["basis"] == "fixed_132s_settling_unknown"
    assert pc["slow_wave"]["missing_settling"] == ["slow_wave/o: unmeasured"]
    assert {pc[c]["status"] for c in CONSUMERS} == {"early part deferred to add-on"}
    pcuts = _as_list(prec["cuts"])
    assert {c["cut"] for c in pcuts} == set(CUT_IDS)
    assert {c["basis"] for c in pcuts} == {rs.BASIS_FIXED}   # TEST_TABLE keys no output
    assert {c["start_sample0"] for c in pcuts} == {I0}

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
    for name in ("schema", "twice", "fractional", "v1", "v2", "no_electrical", "no_cuts",
                 "cut_twice", "cut_fractional", "no_map", "no_sources", "sources_as_object"):
        assert readers[name]["error"] == "night6:recoveryStartFile", (name, readers[name])
    assert "has no cuts" in readers["no_cuts"]["message"]
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
    assert {c: rsr["consumers"][c]["trimmed_rows"] for c in CONSUMERS} == dict.fromkeys(
        CONSUMERS, 1)
    assert {c["cut"] for c in _as_list(rsr["cuts"])} == set(CUT_IDS)
    spk = next(r for r in rec["runs"] if r["call"] == "process_dataset_v2")
    lt = next(i for i in spk["inputs"] if i["signal"] == "L_T")
    assert lt["first_nan_row"] == 1 and _rows(lt["nan_runs"])[0][0] == 1
    assert "night6_recovery_lead_in" in rec["functions"]
    assert run["resume_same"] == "complete"   # same starts file and mode: skipped
    assert run["resume_other"] == "dry_run"   # another starts file: rerun
    assert run["resume_old_mode"] == "dry_run"   # a record made under (A): rerun
    assert run["mode_recorded"] == MODE_B
    assert rec["recovery_trim_mode"] == MODE_B
    for name in ("no_mode", "bad_mode", "mode_a"):
        assert run[name]["error"] == "night6:recoveryTrimMode", run[name]
    assert "RULING 2026-10-09 item 6" in run["mode_a"]["message"]

    _check_unit(res["unit"])
    b = res["batch"]
    for name, ident in (("no_mode", "night6:recoveryTrimMode"),
                        ("bad_mode", "night6:recoveryTrimMode"),
                        ("mode_a", "night6:recoveryTrimMode"),
                        ("no_starts", "night6:recoveryStarts")):
        assert b[name]["error"] == ident, (name, b[name])
    assert "trim_it_all" in b["bad_mode"]["message"]
    assert "RULING 2026-10-09 item 6" in b["mode_a"]["message"]
    assert "no recovery_starts declared" in b["no_starts"]["message"]
    assert not list((tmp_path / "batch_out").rglob("night6_record.json"))  # nothing ran


# ---------------------------------------------------------------------------
# the unit case: an independent computation of every expected value
# ---------------------------------------------------------------------------


def _col(v: object) -> list[Any]:
    a = np.asarray(v, dtype=object).ravel().tolist()
    return [None if x is None else x for x in a]


def _mround(x: float) -> int:
    """MATLAB's round: half away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def _valid(n: int, runs: list[list[int]]) -> np.ndarray:
    v = np.ones(n, dtype=bool)
    for a, b in runs:
        v[a - 1:b] = False
    return v


def _frac(valid: np.ndarray, lo: int, hi: int) -> float:
    return float(valid[lo - 1:hi].sum()) / (hi - lo + 1)


def _hr_fracs(positions: list[int], w: int) -> list[float]:
    """HR_BR_HRVAnalysis_beats.m:835-838 windows, from the stamp's time as the file has it."""
    valid = _valid(UNIT_N, UNIT["invalid"])
    out = []
    for p in positions:
        tc, half = p / FS, (w / FS) / 2
        lo = max(1, _mround((tc - half) * FS) + 1)
        hi = min(UNIT_N, _mround((tc + half) * FS) + 1)
        out.append(_frac(valid, lo, hi))
    return out


def _nan_before(values: list[float], positions: list[float], cut: int) -> list[Any]:
    return [None if p < cut - 1e-6 else v for v, p in zip(values, positions, strict=True)]


def _p3(x: int) -> list[int]:
    return [x - 1, x, x + 1]


def _check_unit(u: dict[str, Any]) -> None:  # noqa: PLR0915 - one hand-built case
    """Every variable cut at exactly ITS cut; fractions, averages and events exact."""
    o = u["out"]
    # --- HRVMeasures: class (i) on RR (sec_row1), heartlocs (row1), hrv_series (sec0)
    m = o["hrvm"]
    assert _col(m["RR_times"]) == pytest.approx([(_LB + d + 1) / FS for d in (0, 1)])
    assert _col(m["RR_intervals"]) == [0.12, 0.13]                  # co-indexed, cut together
    assert _col(m["hrv_series"]) == [None, 2.0, 3.0]                # sec0: NaN, stamp kept
    assert _col(m["nRR_used"]) == [None, 5.0, 6.0]
    assert len(_col(m["metrics_t"])) == 3
    assert _col(m["heartlocs"]) == [_LB + 1, _LB + 2]               # row1
    assert m["hrv"] == 0.5 and _col(m["mystery"]) == [1, 2, 3]       # never cut
    want = _hr_fracs(_p3(_LC), UNIT["win_w"])
    assert _col(m["hrv_series_validFraction"]) == want              # EXACT, every row
    assert _col(m["nRR_used_validFraction"]) == want
    assert 0.0 < min(want) < 1.0 and len(set(want)) > 1
    hrvm = next(f for f in u["hrv"]["files"] if f["kind"] == "HRVMeasures")
    assert "hrv" in hrvm["epoch_scalars"]
    unk = {e["path"]: e["why"] for e in hrvm["untrimmed_time_convention_unknown"]}
    assert "not in the output time map" in unk["mystery"]
    hrbr = next(f for f in u["hrv"]["files"] if f["kind"] == "HRBR")
    unk = {e["path"]: e["why"] for e in hrbr["untrimmed_time_convention_unknown"]}
    assert "time convention unknown" in unk["RR_implausibleMask"]
    files = _as_list(u["hrv"]["untrimmed_files"])
    assert [f["file"] for f in files] == ["e2_figure.png"]
    # --- HRBR: each series at its own cut on ONE axis, in an hrv run and a breathing run
    pos = sorted(_p3(_LBR) + _p3(_LC) + _p3(_LH))
    for b in (o["hrbr"], o["hrbr_b"]):
        assert _col(b["heartRateSeries"]) == _nan_before(list(range(1, 10)), pos, _LH)
        assert _col(b["breathRateSeries"]) == _nan_before(list(range(11, 20)), pos, _LBR)
        assert _col(b["heartCountSeries"]) == _nan_before(list(range(21, 30)), pos, _LC)
        assert _col(b["heartCountRateSeries"]) == _nan_before(list(range(31, 40)), pos, _LC)
        assert _col(b["heartBeatSeries"]) == [None, 2.0, 3.0]        # class (ii) at its own
        assert _col(b["br_locs_true"]) == [_LTR + 1, _LTR + 2]
        assert _col(b["heartlocs"]) == [_LB + 1, _LB + 2]
        for v in ("heartRateSeries", "breathRateSeries"):
            assert _col(b[f"{v}_validFraction"]) == _hr_fracs(pos, UNIT["hrbr_w"]), v
        for v in ("heartCountSeries", "heartCountValidSec", "heartCountRateSeries"):
            assert _col(b[f"{v}_validFraction"]) == _hr_fracs(pos, UNIT["win_w"]), v
        # recomputed from the kept values; her originals in the marker
        assert b["avgHeartRate"] == 8.5 and b["avgBreathRate"] == 15.5
        assert b["avgHeartCount"] == 27.0 and b["avgHeartCountRate"] == 37.0
    trimmed = {e["path"]: e for e in hrbr["trimmed"]}
    assert trimmed["breathRateSeries"]["rows_before_start"] == _LBR
    assert "byproduct" in trimmed["breathRateSeries"]["start_basis"]
    trimmed_b = {e["path"]: e for e in u["breathing"]["files"][0]["trimmed"]}
    assert trimmed_b["heartRateSeries"]["rows_before_start"] == _LH
    assert "hrv's own cut" in trimmed_b["heartRateSeries"]["start_basis"]
    # --- mmc: NOT COMPUTED events are NaN in a double series; signal, rate, delay
    mm = o["mmc"]["mmc"]
    assert u["events_class"] == "double"
    ev = np.array([[np.nan if x is None else x for x in r] for r in mm["firing"]["events"]])
    want_ev = np.zeros((UNIT_N, 3))
    for c, r in UNIT["ev_rows"]:
        want_ev[r - 1, c - 1] = 1.0
    want_ev[:_LE, :] = np.nan
    assert np.array_equal(ev, want_ev, equal_nan=True)
    assert np.isnan(ev[_LE - 1]).all() and not np.isnan(ev[_LE]).any()  # the 1-sample cut
    sig = np.ones((UNIT_N, 3))
    for c, a, b2 in UNIT["mmc_nan"]:
        sig[a - 1:b2, c - 1] = np.nan
    ls = UNIT_L["mmc.mmc_signal.filled_or_filtered"]
    got_sig = np.array([[np.nan if x is None else x for x in r] for r in mm["signal"]])
    want_sig = sig.copy()
    want_sig[:ls, :] = np.nan
    assert np.array_equal(got_sig, want_sig, equal_nan=True)
    kept = np.arange(UNIT_N) >= _LE
    rate_want = [np.sum(want_ev[kept, c] == 1) / max(np.sum(kept & ~np.isnan(sig[:, c])) / FS,
                                                     np.finfo(float).eps) for c in range(3)]
    assert _col(mm["firing"]["avgRate"]) == pytest.approx(rate_want, rel=1e-15)
    # rate: rows L-2 .. L+3, cut at L; fraction over her window floor((c -/+ W/2) fs)
    w = UNIT["mmc_w"] / FS
    rate0 = np.outer(np.arange(1, 7), np.ones(3))
    rate0[1, 0] = np.nan
    rate0[4, 2] = np.nan
    want_rate = rate0.copy()
    want_rate[:2, :] = np.nan
    got_rate = np.array([[np.nan if x is None else x for x in r] for r in mm["firing"]["rate"]])
    assert np.array_equal(got_rate, want_rate, equal_nan=True)
    fr = []
    for k in range(-2, 4):
        c = (_LR + k) / FS
        lo = max(1, math.floor((c - w / 2) * FS) + 1)
        hi = min(UNIT_N, math.floor((c + w / 2) * FS))
        fr.append([_frac(~np.isnan(sig[:, ch]), lo, hi) for ch in range(3)])
    assert mm["firing"]["rate_validFraction"] == fr
    assert mm["firing"]["peakAmp_validFraction"] == fr
    assert any(0.0 < x < 1.0 for row in fr for x in row)
    # the delay is cut at its TRUE centre; its fraction is over firing-rate rows, per pair
    assert [r[0] for r in mm["delay"]] == [None, 2, 3]
    want_d = []
    for lo in (1, 2, 3):
        rows = rate0[lo - 1:lo + 3]
        want_d.append([float(np.mean(np.isfinite(rows[:, a]) & np.isfinite(rows[:, b])))
                       for a, b in ((0, 1), (0, 2), (1, 2))])
    assert mm["delay_validFraction"] == want_d
    # --- slow wave (class ii): cut, fraction over her window, recomputed average
    s = o["sw"]
    assert _col(s["slowWaveRateSeries"]) == [None, 2.0, 3.0]
    assert s["avgSlowWave"] == 2.5
    valid = _valid(UNIT_N, UNIT["invalid"])
    h = _mround(UNIT["sw_w"] / FS * FS / 2)
    want_sw = [_frac(valid, max(1, _mround(p / FS * FS) + 1 - h),
                     min(UNIT_N, _mround(p / FS * FS) + 1 + h)) for p in _p3(_LSW)]
    assert _col(s["slowWaveRateSeries_validFraction"]) == want_sw
    assert 0.0 < min(want_sw) < 1.0
    # --- spikes: struct array element by element, co-indexed drop, cells, still v7.3
    sp = o["spk"]["spikes"]
    assert _col(sp[0]["centers"]) == [_LST + 1, _LST + 2]
    assert _col(sp[1]["centers"]) == [_LST + 1, _LST + 2]             # nothing before
    assert sp[0]["waveforms"] == [[2, 2], [3, 3]] and sp[1]["waveforms"] == [[4, 4], [5, 5]]
    assert _col(sp[0]["alignedCenters"]) == [_LW + 1, _LW + 2]
    assert [x["nSpikes"] for x in sp] == [2, 2]                      # recomputed counts
    assert [x["nSpikes"] for x in o["spk"]["metrics"]] == [2, 2]
    sw = o["spk"]["sigmaWin"]
    assert [_col(c) for c in sw["sigma"]] == [[None, 2.0, 3.0], [None, 5.0, 6.0]]
    n_spk = UNIT["spk_n"]
    v1 = _valid(n_spk, [[a, b2] for c, a, b2 in UNIT["spk_invalid"] if c == 1])
    v2 = _valid(n_spk, [])
    win = 20
    want_sg = [[_frac(v, i, min(n_spk, i + win - 1)) for i in (1, 11, 21)] for v in (v1, v2)]
    assert [_col(c) for c in sw["sigma_validFraction"]] == want_sg
    assert want_sg[0] == [17 / 20, 19 / 20, 16 / 20]
    winsec = UNIT["cv2_w"] / FS
    edges = [0.0, winsec, 2 * winsec, (n_spk - 1) / FS]
    rows_t = [(i - 1) / FS for i in range(1, n_spk + 1)]
    want_cv = []
    for v in (v1, v2):
        f = []
        for e0, e1 in pairwise(edges):
            sel = [i for i, t in enumerate(rows_t, start=1) if e0 <= t < e1]
            f.append(_frac(v, sel[0], sel[-1]))
        want_cv.append(f)
    assert want_cv[0] == [13 / 16, 13 / 15, 6 / 8]
    mt = o["spk"]["metrics"]
    assert [_col(x["cv2_roll"]) for x in mt] == [[None, 2.0, 3.0], [None, 5.0, 6.0]]
    assert [_col(x["cv2_roll_validFraction"]) for x in mt] == want_cv
    assert u["spikes_still_v73"] is True
    # --- refusals, by name; a refused file is never written
    assert u["twice"]["error"] == "night6:trimTwice"
    assert u["no_cut"]["error"] == "night6:trimCut"
    assert "heartRateSeries" in u["no_cut"]["message"]
    for name in ("no_class", "bad_class"):
        assert u[name]["error"] == "night6:trimClass", u[name]
        assert "heartRateSeries" in u[name]["message"]
        assert "RULING 2026-10-09 item 6" in u[name]["message"]
    assert u["collide"]["error"] == "night6:trimShape"
    assert "heartRateSeries_validFraction" in u["collide"]["message"]
    assert u["shape"]["error"] == "night6:trimShape"
    assert u["untouched"]["nocut"] is True
    _check_unit_markers(u)


def _check_unit_markers(u: dict[str, Any]) -> None:  # noqa: PLR0915 - one marker per file
    """Check the marker each trimmed file carries: exact, for every class and addition."""
    o = u["out"]
    name = u["marker_variable"]
    assert name == rs.TRIM_MARKER

    def var(m: dict[str, Any], path: str) -> dict[str, Any]:
        hit = [v for v in _as_list(m["vars"]) if v["path"] == path]
        assert len(hit) == 1, (path, [v["path"] for v in _as_list(m["vars"])])
        return hit[0]

    def cut(v: dict[str, Any], cid: str, cls: str, conv: str, action: str) -> None:
        lc = UNIT_L[cid]
        assert v["mode"] == MODE_B and v["cut"] == cid and v["trim_class"] == cls
        assert v["trim_class_meaning"] == rs.TRIM_CLASSES[cls]
        assert v["convention"] == conv and v["action"] == action
        assert v["convention_meaning"] == rs.CONVENTIONS[conv]
        assert v["start_sample0"] == UNIT_I0 + lc
        assert v["start_s"] == pytest.approx((UNIT_I0 + lc) / FS, abs=1e-12)
        assert v["first_computed_epoch_row"] == lc + 1
        assert v["first_computed_epoch_sample0"] == lc
        if cls == "valid_only":
            assert re.match(r"^\S+\.m:\d+", v["edge_rule"]["source"]), v["edge_rule"]

    m = o["hrvm"][name]
    assert m["mode"] == MODE_B and "RULING 2026-10-09 item 6" in m["ruling"]
    assert m["epoch_start_sample0"] == UNIT_I0 and m["electrical_settle_sample0"] == UNIT_I0 + 7
    assert _as_list(m["run_consumers"]) == ["hrv"]
    assert "mystery" in _as_list(m["untrimmed"]) and "hrv" in _as_list(m["epoch_scalars"])
    assert {v["path"] for v in _as_list(m["vars"])} == {
        "RR_intervals", "RR_times", "hrv_series", "nRR_used", "heartlocs"}
    v = var(m, "RR_times")
    cut(v, "hrv.beats.valid_only", "valid_only", "sec_row1", "drop")
    (leaf,) = _as_list(v["leaves"])
    assert leaf["leaf"] == "RR_times" and leaf["n_entries"] == 3
    assert leaf["n_not_computed"] == leaf["n_dropped"] == 1
    assert _rows(leaf["not_computed_rows"]) == []
    assert v["valid_fraction"] == []                                   # an event list
    v = var(m, "hrv_series")
    cut(v, "hrv.count_hrv.valid_only", "valid_only", "sec0", "nan")
    (leaf,) = _as_list(v["leaves"])
    assert leaf["n_not_computed"] == 1 and leaf["n_dropped"] == 0
    assert _rows(leaf["not_computed_rows"]) == [[1, 1]]
    assert v["valid_fraction"]["variable"] == "hrv_series_validFraction"
    assert v["valid_fraction"]["added"] is True and v["valid_fraction"]["kind"] == "hr_window"
    assert sorted(_as_list(m["added_variables"])) == ["hrv_series_validFraction",
                                                      "nRR_used_validFraction"]
    # HRBR: class (ii) at its own cut; added fractions; recomputed with her originals
    b = o["hrbr"][name]
    v = var(b, "heartBeatSeries")
    cut(v, "hrv.heart_band_trace.filled_or_filtered", "filled_or_filtered", "sec0", "nan")
    assert _rows(_as_list(v["leaves"])[0]["not_computed_rows"]) == [[1, 1]]
    v = var(b, "heartRateSeries")
    cut(v, "hrv.heart_rate.valid_only", "valid_only", "sec0", "nan")
    assert _rows(_as_list(v["leaves"])[0]["not_computed_rows"]) == [[1, 7]]
    assert sorted(_as_list(b["added_variables"])) == sorted(
        f"{x}_validFraction" for x in ("heartRateSeries", "heartCountSeries",
                                       "heartCountValidSec", "heartCountRateSeries",
                                       "breathRateSeries"))
    rc = {e["path"]: e for e in _as_list(b["recomputed"])}
    assert set(rc) == {"avgHeartRate", "avgBreathRate", "avgHeartCount", "avgHeartCountRate"}
    for path, orig, new in (("avgHeartRate", 99, 8.5), ("avgBreathRate", 98, 15.5),
                            ("avgHeartCount", 97, 27.0), ("avgHeartCountRate", 96, 37.0)):
        (lf,) = _as_list(rc[path]["leaves"])
        assert (lf["original"], lf["recomputed"]) == (orig, new), path
        assert "RECOMPUTED" in rc[path]["label"]
    # a breathing-only run: hrv's byproducts at their own cuts
    bm = o["hrbr_b"][name]
    v = var(bm, "heartRateSeries")
    cut(v, "hrv.heart_rate.valid_only", "valid_only", "sec0", "nan")
    assert v["owner"] == "hrv" and "byproduct" in v["start_basis"]
    v = var(bm, "breathRateSeries")
    cut(v, "breathing.breath_rate.valid_only", "valid_only", "sec0", "nan")
    assert _rows(_as_list(v["leaves"])[0]["not_computed_rows"]) == [[1, 1]]
    # mmc: NaN events (not "no event"), the delay at its true centre, the average
    mm = o["mmc"][name]
    v = var(mm, "mmc.firing.events")
    cut(v, "mmc.mmc_events.filled_or_filtered", "filled_or_filtered", "sec0", "nan_events")
    (leaf,) = _as_list(v["leaves"])
    assert leaf["leaf"] == "mmc.firing.events" and _rows(leaf["not_computed_rows"]) == [[1, _LE]]
    assert leaf["class_before"] == "logical"
    v = var(mm, "mmc.delay")
    cut(v, "mmc.mmc_delay.filled_or_filtered", "filled_or_filtered", "sec_xchan_delay", "nan")
    assert _rows(_as_list(v["leaves"])[0]["not_computed_rows"]) == [[1, 1]]
    assert v["valid_fraction"]["kind"] == "mmc_delay_window"
    v = var(mm, "mmc.firing.rate")
    cut(v, "mmc.mmc_rate.valid_only", "valid_only", "sec0", "nan")
    assert v["edge_rule"]["half_valid"] is True
    (rca,) = _as_list(mm["recomputed"])
    assert rca["path"] == "mmc.firing.avgRate" and rca["kind"] == "events_per_valid_s"
    assert _col(_as_list(rca["leaves"])[0]["original"]) == [1, 2, 3]
    # spikes: a struct array and cells, leaf by leaf (v7.3 file); counts recomputed
    sm = o["spk"][name]
    v = var(sm, "spikes.centers")
    cut(v, "spikes.spike_times.filled_or_filtered", "filled_or_filtered", "row1", "drop")
    assert [(x["leaf"], x["n_dropped"]) for x in _as_list(v["leaves"])] == [
        ("spikes(1).centers", 1), ("spikes(2).centers", 0)]
    v = var(sm, "sigmaWin.sigma")
    cut(v, "spikes.sigma_windows.valid_only", "valid_only", "row1", "nan")
    assert [(x["leaf"], _rows(x["not_computed_rows"])) for x in _as_list(v["leaves"])] == [
        ("sigmaWin.sigma{1}", [[1, 1]]), ("sigmaWin.sigma{2}", [[1, 1]])]
    rc = {e["path"]: e for e in _as_list(sm["recomputed"])}
    assert set(rc) == {"spikes.nSpikes", "metrics.nSpikes"}
    assert [(x["leaf"], x["original"], x["recomputed"]) for x in
            _as_list(rc["metrics.nSpikes"]["leaves"])] == [
        ("metrics(1).nSpikes", 3, 2), ("metrics(2).nSpikes", 2, 2)]
    assert sorted(_as_list(sm["added_variables"])) == ["metrics.cv2_roll_validFraction",
                                                       "sigmaWin.sigma_validFraction"]


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
    """TRIM_MODES / WITHDRAWN_TRIM_MODES and night6_trim_modes.m name the same modes."""
    m = (NIGHT6 / "night6_trim_modes.m").read_text(encoding="utf-8")
    got = re.search(r"modes = \{([^}]*)\};", m)
    gone = re.search(r"withdrawn = \{([^}]*)\};", m)
    assert got and gone
    assert tuple(re.findall(r"'([^']+)'", got.group(1))) == rs.TRIM_MODES
    assert set(re.findall(r"'([^']+)'", gone.group(1))) == set(rs.WITHDRAWN_TRIM_MODES)


TRIM_KE = seconds_to_sample(2.5, FS)
"""The electrical settling of the end-to-end case: 0.5 s into the epoch."""
TRIM_CUT_S = {
    "spikes.sigma_windows.valid_only": 31.0,
    "spikes.spike_times.filled_or_filtered": 32.0,
    "spikes.cv2.valid_only": 32.2,
    "spikes.spike_waveforms.filled_or_filtered": 32.5,
    "spikes.envelope.valid_only": 33.0,
    "spikes.firing_rate.valid_only": 33.5,
    "spikes.spike_waveforms.valid_only": 34.0,
    "mmc.mmc_signal.filled_or_filtered": 30.0,
    "mmc.mmc_events.filled_or_filtered": 34.0,
    "mmc.mmc_delay.filled_or_filtered": 35.0,
    "mmc.mmc_rate.valid_only": 36.0,
    "hrv.heart_band_trace.filled_or_filtered": 36.0,
    "hrv.beats.valid_only": 36.5,
    "hrv.heart_rate.valid_only": 37.0,
    "hrv.count_hrv.valid_only": 38.5,
    "hrv.sampen.valid_only": 39.0,
    "slow_wave.sw_trace.filled_or_filtered": 38.0,
    "slow_wave.sw_peaks.filled_or_filtered": 39.0,
    "slow_wave.sw_rate.filled_or_filtered": 40.0,
    "breathing.breath_troughs.valid_only": 41.0,
    "breathing.breath_rate.valid_only": 42.0,
}
"""Each cut (s, file time): all later than the electrical settling, different within each
owner, and each placed so EVERY variable it cuts has entries on both sides (her windows
decide where values exist in a 73 s epoch: the 60 s HR/BR and sample-entropy windows at
centres 30-43 s, the 30 s CV2 windows at 15/45 s, slow-wave peaks 15 s inside the ends,
the 30 s mmc delay windows at true centres 19.5 + 5k s). The mmc delay at 33 s into the
epoch sits in (34.5 - 4, 34.5]: the delay whose STAMP is before its cut and whose true
centre is after it, so a delay cut at its stamp is visible. The mmc signal's cut is
before the events' (the average's validity is read on the kept span)."""


TRIM_N_FILE = int(75.0 * FS)
"""75 s (a 73 s epoch): long enough for every output of her functions to hold entries on
both sides of every cut (her gap-aware DFA also needs more RR intervals than the 10 s
file has - processing_new dfaGapAware.m:175, reported, not ours)."""


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
    assert set(TRIM_CUT_S) == set(CUT_IDS)
    cut = {cid: seconds_to_sample(s, FS) for cid, s in TRIM_CUT_S.items()}
    own = {c: max(k for cid, k in cut.items() if cid.split(".")[0] == c) for c in CONSUMERS}
    d = tmp_path / "starts"
    d.mkdir()
    drop = _starts(d / "drop.json", [_file([_row(c, own[c]) for c in CONSUMERS], ke=TRIM_KE,
                                           cut_at=cut)])
    ref = _starts(d / "ref.json", [_file([_row(c, TRIM_KE) for c in CONSUMERS], ke=TRIM_KE)])
    return {"trim": {"gems_root": (tmp_path / "store").as_posix(),
                     "mask_folder": st["mask_folder"].as_posix(),
                     "out_drop": (tmp_path / "out_drop").as_posix(),
                     "out_ref": (tmp_path / "out_ref").as_posix(),
                     "starts_drop": drop, "starts_ref": ref, "fs": FS, "i0": I0,
                     "epoch_rel": f"T/{SESSION}/{MODEL}/e{round(START)}",
                     "own": own,
                     "cuts": [{"cut": c, "L": k - I0, "sample0": k} for c, k in cut.items()]},
            "plans": [], "readers": []}


def test_mode_b_cuts_her_real_outputs_per_variable(  # noqa: PLR0915 - one run
        tmp_path: Path) -> None:
    """End to end: each variable dropped or flagged before ITS cut, bit-identical after it.

    The same epoch is run twice with her real functions on the same masked input (every
    input masked to the electrical settling): with a different cut for every cut id, and
    the reference with every cut at the electrical settling. An oracle in the harness with
    its OWN tables (``oracle_table``, ``oracle_fractions``, ``oracle_recomputed``,
    transcribed from her code, not read from the map) checks every trimmed variable of
    every output file - its class and cut included - the valid fractions, the recomputed
    averages, the marker each trimmed file carries, and that every other variable is
    untouched. Every trimmed variable has entries on both sides of its cut, with values in
    the reference on both sides - so a wrong class, cut, convention, owner or action shows
    on the data.
    """
    _need_matlab()
    case = _trim_case(tmp_path)
    own = case["trim"]["own"]
    rows = {c["cut"]: c["L"] for c in case["trim"]["cuts"]}
    r = _run_harness(tmp_path, case, 2400)["trim"]
    rd, rr = json.loads(r["record_drop"]), json.loads(r["record_ref"])
    assert rd["status"] == rr["status"] == "complete", (rd.get("runs"), rr.get("runs"))
    assert rd["recovery_trim_mode"] == rr["recovery_trim_mode"] == MODE_B
    for c in CONSUMERS:  # one input mask for all, in both runs
        assert rd["recovery_start"]["consumers"][c]["trimmed_rows"] == TRIM_KE - I0
        assert rr["recovery_start"]["consumers"][c]["trimmed_rows"] == TRIM_KE - I0
        a = rd["recovery_start"]["analyses"][c]   # every analysis's start, run or not
        assert a["start_sample0"] == own[c] and a["output_rows_before_start"] == own[c] - I0
    runs = [x for x in rd["runs"] if x["status"] == "ok"]
    assert runs and all("recovery_trim" in x for x in runs)
    assert all("recovery_trim" in x for x in rr["runs"] if x["status"] == "ok")
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
    assert all(f["added_ok"] for f in trimmed_files), [f for f in trimmed_files
                                                       if not f["added_ok"]]
    bad = [v for v in r["vars"] if v["why"] != "absent" and not v["marker_ok"]]
    assert not bad, bad
    # the oracle's tables and the map name the same variables
    assert r["table_vs_map"] == {"map_only": [], "oracle_only": []}, r["table_vs_map"]
    assert r["recomputed_vs_map"] == {"map_only": [], "oracle_only": []}
    # every trimmed variable: entries before AND after its cut, and values in the
    # reference on both sides (otherwise a wrong cut there could not show)
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
    # valid fractions: the same in both runs (from the input, never the cut), in [0, 1],
    # and consistent with her own validity rules where she has one
    frs = [f for f in r["fractions"] if f["why"] != "absent"]
    bad = [f for f in frs if not f["ok"]]
    assert not bad, bad
    seen = {f["path"] for f in frs}
    assert {"heartRateSeries", "heartCountRateSeries", "hrv_series", "sampEn_series",
            "slowWaveRateSeries", "mmc.firing.rate", "mmc.delay", "sigmaWin.sigma",
            "metrics.cv2_roll", "envelope.rms_uv", "metrics.fr_hz"} <= seen
    rule = {f["path"]: f["n_rule"] for f in frs}
    assert rule["heartCountRateSeries"] > 0 and rule["mmc.firing.rate"] > 0
    assert rule["heartCountValidSec"] > 0
    partial = {f["path"] for f in frs if f["n_partial"] > 0}
    assert {"heartCountRateSeries", "mmc.firing.rate", "sigmaWin.sigma"} <= partial
    # recomputed averages: her expression over the kept values, her original kept
    rcs = [x for x in r["recomputed"] if x["why"] != "absent"]
    bad = [x for x in rcs if not x["ok"]]
    assert not bad, bad
    changed = {x["path"] for x in rcs if x["n_changed"] > 0}
    assert {"avgHeartRate", "avgSlowWave", "mmc.firing.avgRate", "spikes.nSpikes"} <= changed
    # every variable at its OWN cut, in every HR run, whoever ran
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
        for f in _as_list(x["recovery_trim"]["files"]):
            for e in _as_list(f["trimmed"]):
                assert e["rows_before_start"] == rows[e["cut"]], (x["consumers"], e)
