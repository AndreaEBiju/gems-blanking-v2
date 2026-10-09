"""RULING 2026-10-08 (k) 2: the per-analysis settling table and the recovery-starts file.

* the table is keyed by exactly the Night 6 consumers, and every cited ``file:line`` still
  says what the table says; every cited processing_new file is the one read, by SHA-256
  (her working tree is not a commit; skipped without processing_new);
* the per-output time map (drop mode): every output file kind of every analysis is
  declared, every path under a declared container, every trimmed output names a declared
  stamp, a convention and an action, and the anchors of the stamps still say it;
* every window length the table uses is the one Andrea's code and the wrapper pass;
* an output's settling is the SUM of its cascade (a window fed by a filter reaches back
  through both), an analysis's the MAXIMUM over its outputs, a centred window counts half,
  a trailing one whole; the numbers are pinned against hand sums of their parts;
* an unknown stage makes the analysis unknown, never the known part (invariant 19), and
  that analysis alone keeps 132 s, labelled, with the figure named, while the others in
  the same file get their own starts (the user's rule);
* undetected edges and a signal that never settles hold the whole file (invariant 41);
* the start sample is the one shared rounding (``extent.grid.seconds_to_sample``), so a
  fallback lands on the epoch's own first sample; earlier / equal / later are labelled;
* the file: keys proven unique at write time (invariant 27), canonical ASCII JSON with no
  NaN and LF endings, and an exact round trip.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from gems_blanking_v2.extent import recovery_start as rs
from gems_blanking_v2.extent.grid import seconds_to_sample
from gems_blanking_v2.extent.tolerance import (
    CONSUMER_FILTERS,
    expected_consumers,
    impulse_settling_s,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_matlab_acceptance import _processing_new

FS = 24414.0625
REPO = Path(__file__).resolve().parents[1]


def _lines(path: Path, a: int, b: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[a - 1:b])


def test_the_table_is_keyed_by_the_night6_consumers() -> None:
    assert set(rs.ANALYSES) == set(expected_consumers())
    for name, a in rs.ANALYSES.items():
        assert a.name == name and a.outputs
        assert len({o.name for o in a.outputs}) == len(a.outputs)


def _anchored() -> list[tuple[str, str, str]]:
    """Return (what, source, anchor) of every stage and anchored output variable."""
    out = [(f"{a.name}/{s.what}", s.source, s.anchor) for a in rs.ANALYSES.values()
           for o in a.outputs for s in o.stages]
    out += [(f"{v.file}/{v.path}", v.source, v.anchor) for v in rs.OUTPUT_VARS if v.anchor]
    return out


def test_every_cited_line_still_says_it() -> None:
    """The anchor of every stage and output is on (or within) the line range it cites."""
    pnew = _processing_new()
    checked = 0
    for what, source, anchor in _anchored():
        m = re.match(r"^(\S+?):(\d+)(?:-(\d+))?", source)
        assert m, source
        rel, lo = m.group(1), int(m.group(2))
        hi = int(m.group(3) or lo)
        if rel.endswith(".py") or rel.startswith("matlab/"):
            path = REPO / rel
        elif pnew is None:
            continue
        else:
            assert rel in rs.SOURCE_FILES, f"{what} cites {rel}, which SOURCE_FILES does not hash"
            path = pnew / rel
        assert anchor in _lines(path, lo, hi), (what, source)
        checked += 1
    assert checked >= (4 if pnew is None else 80)


def test_every_cited_file_is_the_one_the_table_was_read_from() -> None:
    """Her working tree is not a commit: each cited file is pinned by its SHA-256."""
    pnew = _processing_new()
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    changed = []
    for name, want in sorted(rs.SOURCE_FILES.items()):
        got = hashlib.sha256((pnew / name).read_bytes()).hexdigest()
        if got != want:
            changed.append(f"{name}: sha256 {got} != {want}")
    assert not changed, (
        "processing_new changed since extent.recovery_start was read from it - re-read "
        "every line the table cites in these files, then update SOURCE_FILES:\n"
        + "\n".join(changed))


def test_the_output_time_map_is_complete_and_well_formed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    rec = rs.output_times_record()
    assert set(rec["conventions"]) == set(rs.CONVENTIONS)
    assert set(rec["actions"]) == set(rs.ACTIONS)
    kinds = {f["kind"] for f in rec["files"]}
    assert kinds == {"spikes_v2", "HRBR", "HRVMeasures", "slowWaves", "mmc"}
    trims = [v for v in rec["vars"] if v["role"] == "trim"]
    assert {v["owner"] for v in trims} == set(rs.ANALYSES)  # every analysis is cut somewhere
    assert {v["file"] for v in trims} == kinds
    assert {v["convention"] for v in trims} == set(rs.CONVENTIONS)
    by = {(v["file"], v["path"]): v for v in rec["vars"]}
    assert by[("mmc", "mmc.delay")]["convention"] == "sec_xchan_delay"
    assert by[("HRVMeasures", "RR_times")]["convention"] == "sec_row1"
    assert by[("HRBR", "breathRateSeries")]["owner"] == "breathing"
    assert by[("HRBR", "heartRateSeries")]["owner"] == "hrv"
    assert by[("HRBR", "RR_implausibleMask")]["role"] == "unknown"
    for v in trims:
        st = by[(v["file"], v["stamp"])]
        assert st["role"] in ("time_axis", "trim"), v
    # a malformed map is refused, never half applied
    orig = rs.OUTPUT_VARS
    for bad, words in (
            (rs.OutputVar("HRBR", "nowhere.x", "trim", "x", stamp="metrics_t",
                          convention="sec0", action="nan"), "not a declared container"),
            (rs.OutputVar("HRBR", "q", "trim", "x", stamp="nope", convention="sec0",
                          action="nan"), "not a declared time axis"),
            (rs.OutputVar("HRBR", "heartlocs", "input", "x"), "declared twice"),
            (rs.OutputVar("HRBR", "q", "trim", "x", owner="velocity", stamp="t",
                          convention="sec0", action="nan"), "is not an analysis")):
        monkeypatch.setattr(rs, "OUTPUT_VARS", (*orig, bad))
        with pytest.raises(ValueError, match=words):
            rs.output_times_record()


def test_the_mmc_delay_stamp_is_its_window_centre_minus_w_half_minus_s() -> None:
    """extract_mmc labels the delay window on rate ROWS: true centre = delay_t + W/2 - S."""
    pnew = _processing_new()
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    code = (pnew / "extract_mmc.m").read_text(encoding="utf-8", errors="replace")
    assert "centers = (W/2 : S : (t(end)-W/2)).';" in code   # rate row m at W/2 + (m-1) S
    assert "delay_t(s) = (lo+hi)/2 * S;" in code              # (lo+hi)/2 rows, times S
    w, s, lo, hi = 10.0, 1.0, 7, 36
    centres = [w / 2 + (m - 1) * s for m in range(1, 60)]
    true = (centres[lo - 1] + centres[hi - 1]) / 2
    assert true == (lo + hi) / 2 * s + w / 2 - s == (lo + hi) / 2 * s + 4.0


VALUES = [
    ("pipeline_params.m", r"P\.cv2WinSec\s*=\s*30;"),
    ("pipeline_params.m", r"P\.frBinSec\s*=\s*1;"),
    ("pipeline_params.m", r"P\.envBinSec\s*=\s*1;"),
    ("pipeline_params.m", r"P\.wfPreMs\s*=\s*1\.0;"),
    ("pipeline_params.m", r"P\.wfAlignSearchMs\s*=\s*0\.5;"),
    ("pipeline_params.m", r"P\.edgeBufferMs\s*=\s*10;"),
    ("slowWaveAnalysis_new.m", r"rateWinSec\s*=\s*60;"),
    ("HR_BR_HRVAnalysis_beats.m", r"sampEnWinSec = 60;\s+% fixed by design"),
    ("HR_BR_HRVAnalysis_beats.m", r"halfHrBr\s*=\s*hrBrWinSec / 2;"),
    ("HR_BR_HRVAnalysis_beats.m", r"halfWin\s*=\s*winSec / 2;"),
    ("extract_mmc.m", r"sigmaWin = g\('sigmaWin',30\)"),
    ("extract_mmc.m", r"W = g\('W',10\)"),
    ("extract_mmc.m", r"dW = g\('delayW',30\)"),
    ("extract_mmc.m", r"cardMs = g\('cardiacBlankMs',25\)"),
]
"""Each window length the table states, where her code sets it."""


def test_the_window_lengths_are_hers_and_the_wrappers() -> None:
    run = (REPO / "matlab" / "night6" / "night6_run_recording.m").read_text(encoding="utf-8")
    assert re.search(r"'winSec', 20,\s*\.\.\.\s*'stepSec', 1, 'hrBrWinSec', 60", run)
    assert re.search(r"'smoothWindow', 5, 'edgeBufferSec', 15", run)
    assert "'lowPassCutoff', 0.15, 'lowPassOrder', 2" in run
    mmc = (REPO / "matlab" / "night6" / "night6_mmc_opts.m").read_text(encoding="utf-8")
    for k in ("sigmaWin", "'W'", "delayW", "cardiacBlankMs"):  # extract_mmc defaults apply
        assert k not in mmc
    pnew = _processing_new()
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    for f, pat in VALUES:
        assert re.search(pat, (pnew / f).read_text(encoding="utf-8", errors="replace")), (f, pat)


def test_the_numbers_are_the_sums_of_their_parts() -> None:
    q = rs.beat_decimation_factor(FS)
    assert q == 12
    beats = 10 * q / FS + impulse_settling_s(CONSUMER_FILTERS["hrv"], FS / q)
    want = {
        "spikes": impulse_settling_s(CONSUMER_FILTERS["spikes"], FS) + 15.0,
        "hrv": beats + 30.0,
        "breathing": beats + 30.0,
        "slow_wave": impulse_settling_s(CONSUMER_FILTERS["slow_wave"], FS) + 2.5 + 30.0,
        "mmc": beats + 0.025 + impulse_settling_s(CONSUMER_FILTERS["mmc"], FS) + 15.0 + 15.0
        + 5.0 + 15.0,
    }
    for name, v in want.items():
        s = rs.analysis_settling(name, FS)
        assert s.settling_s == pytest.approx(v, abs=1e-12), name
        assert s.missing == ()
    assert rs.analysis_settling("spikes", FS).binding_output == "rolling CV2 (cv2WinSec 30 s bins)"
    assert rs.analysis_settling("mmc", FS).binding_output.startswith("cross-channel delay")
    assert rs.analysis_settling("slow_wave", FS).binding_output.startswith("slow-wave rate")
    # the values reported (2026-10-09, cohort fs), to the 10 microseconds
    got = {k: round(rs.analysis_settling(k, FS).settling_s or -1, 5) for k in want}
    assert got == {"spikes": 15.00512, "hrv": 30.14598, "breathing": 30.14598,
                   "slow_wave": 40.66697, "mmc": 50.65677}


def _stage(window: float | None, how: rs.How = "half") -> rs.Stage:
    return rs.Stage("w", "centred window", how, "x.m:1", "x", "code", window_s=window)


def test_centred_counts_half_trailing_counts_whole_and_a_cascade_adds() -> None:
    t = {"spikes": rs.Analysis("spikes", "c", (
        rs.Output("a", (_stage(10.0), _stage(4.0, "whole")), "x"),
        rs.Output("b", (_stage(30.0),), "x")))}
    s = rs.analysis_settling("spikes", FS, t)
    assert s.per_output_s == {"a": 9.0, "b": 15.0}
    assert s.settling_s == 15.0 and s.binding_output == "b"
    with pytest.raises(ValueError, match="window_s"):
        rs.stage_settling_s(_stage(None), FS)


TEST_TABLE = {
    "spikes": rs.Analysis("spikes", "c", (rs.Output("o", (_stage(2.0),), "x"),)),
    "hrv": rs.Analysis("hrv", "c", (rs.Output("o", (_stage(8.0),), "x"),)),
    "breathing": rs.Analysis("breathing", "c", (rs.Output("o", (_stage(8.0),), "x"),)),
    "mmc": rs.Analysis("mmc", "c", (rs.Output("o", (_stage(1.0),), "x"),)),
    "slow_wave": rs.Analysis("slow_wave", "c", (
        rs.Output("trace", (_stage(5.0),), "x"),
        rs.Output("rate", (_stage(5.0), _stage(None, "unknown"), _stage(60.0)), "x"))),
}
"""A table whose slow-wave rate has one unmeasured stage."""


def test_an_unknown_analysis_keeps_132_labelled_and_the_others_get_their_own() -> None:
    """The user's rule (2026-10-08): never stim-off + the known part, never a shared max."""
    s = rs.analysis_settling("slow_wave", FS, TEST_TABLE)
    assert s.settling_s is None and s.binding_output is None
    assert s.missing == ("slow_wave/rate: w",)
    assert s.per_output_s["trace"] == 2.5 and s.per_output_s["rate"] is None
    f = rs.file_starts(session="x", fs=FS, stim_off_s=120.8, electrical_settle_s=121.4,
                       stim_off_source="03B", electrical_source="(j) 6 (i)",
                       table=TEST_TABLE)
    rows = {r["analysis"]: r for r in f["analyses"]}
    sw = rows["slow_wave"]
    assert sw["start_s"] == 132.0 and sw["basis"] == "fixed_132s_settling_unknown"
    assert sw["start_sample0"] == seconds_to_sample(132.0, FS)
    assert sw["missing_settling"] == ["slow_wave/rate: w"]
    assert "own_settling_s" not in sw
    partial = 121.4 + 2.5  # the known output only
    assert sw["start_s"] != pytest.approx(partial)
    for name, own in (("spikes", 1.0), ("hrv", 4.0), ("breathing", 4.0), ("mmc", 0.5)):
        r = rows[name]
        assert r["basis"] == rs.BASIS_MEASURED, name
        assert r["start_s"] == pytest.approx(121.4 + own), name
        assert r["start_sample0"] == seconds_to_sample(121.4 + own, FS), name
    assert rows["mmc"]["relation"].startswith("earlier_than_fixed")  # 121.9 < 132
    assert f["electrical_s"] == pytest.approx(0.6)


def test_an_analysis_missing_from_the_table_is_unknown_not_zero() -> None:
    f = rs.file_starts(session="x", fs=FS, stim_off_s=120.8, electrical_settle_s=125.0,
                       stim_off_source="a", electrical_source="b",
                       analyses=["spikes", "velocity"])
    rows = {r["analysis"]: r for r in f["analyses"]}
    assert rows["velocity"]["basis"] == rs.BASIS_FIXED
    assert rows["velocity"]["missing_settling"] == ["velocity: not in the table"]
    assert rows["spikes"]["relation"].startswith("later_than_fixed")


def test_undetected_edges_and_no_settling_hold_the_whole_file() -> None:
    a = rs.file_starts(session="x", fs=FS, stim_off_s=None, electrical_settle_s=None,
                       stim_off_source="a", electrical_source="b")
    assert a["basis"] == rs.HELD_UNDETECTED and "analyses" not in a
    b = rs.file_starts(session="x", fs=FS, stim_off_s=120.8, electrical_settle_s=None,
                       stim_off_source="a", electrical_source="b")
    assert b["basis"] == rs.HELD_NO_SETTLING and "analyses" not in b
    with pytest.raises(ValueError, match="precedes stim-off"):
        rs.file_starts(session="x", fs=FS, stim_off_s=121.0, electrical_settle_s=120.0,
                       stim_off_source="a", electrical_source="b")


def test_the_relation_to_the_epoch_start_is_exact_to_one_sample() -> None:
    """Earlier / at / later are decided on the shared sample rounding, not on seconds."""
    i0 = seconds_to_sample(132.0, FS)
    for k, word in ((i0 - 1, "earlier_than_fixed"), (i0, "at_fixed_start"),
                    (i0 + 1, "later_than_fixed")):
        t = {"spikes": rs.Analysis("spikes", "c", (rs.Output("o", (
            rs.Stage("w", "trailing window", "whole", "x.m:1", "x", "code",
                     window_s=k / FS - 130.0),), "x"),))}
        f = rs.file_starts(session="x", fs=FS, stim_off_s=120.0, electrical_settle_s=130.0,
                           stim_off_source="a", electrical_source="b", table=t)
        r = f["analyses"][0]
        assert r["start_sample0"] == k
        assert r["relation"].startswith(word)


@settings(max_examples=200, deadline=None)
@given(off=st.floats(100.0, 140.0), el=st.floats(0.0, 60.0),
       fs=st.sampled_from([FS, 24414.0, 1000.0]))
def test_every_start_is_its_electrical_time_plus_its_own_settling(off: float, el: float,
                                                                  fs: float) -> None:
    f = rs.file_starts(session="x", fs=fs, stim_off_s=off, electrical_settle_s=off + el,
                       stim_off_source="a", electrical_source="b")
    assert f["electrical_settle_sample0"] == seconds_to_sample(off + el, fs)
    for r in f["analyses"]:
        own = rs.analysis_settling(r["analysis"], fs).settling_s
        assert own is not None
        assert r["start_s"] == off + el + own
        assert r["start_sample0"] == seconds_to_sample(r["start_s"], fs)
        assert r["own_settling_s"] == own


def _doc(*files: dict) -> dict:
    return rs.recovery_starts_document(list(files), fs=FS)


def test_the_document_refuses_duplicates_and_non_integer_samples() -> None:
    a = rs.file_starts(session="Sess_A", fs=FS, stim_off_s=120.8, electrical_settle_s=125.0,
                       stim_off_source="a", electrical_source="b")
    held = rs.file_starts(session="sess_a", fs=FS, stim_off_s=None, electrical_settle_s=None,
                          stim_off_source="a", electrical_source="b")
    with pytest.raises(ValueError, match="appears twice"):
        _doc(a, held)  # case-insensitive: macOS and Windows would collide
    twice = {**a, "analyses": [*a["analyses"], a["analyses"][0]]}
    with pytest.raises(ValueError, match="analysis appears twice"):
        _doc(twice)
    bad = json.loads(json.dumps(a))
    bad["analyses"][0]["start_sample0"] = float(bad["analyses"][0]["start_sample0"])
    with pytest.raises(TypeError, match="start_sample0"):
        _doc(bad)
    no_el = {k: v for k, v in a.items() if k != "electrical_settle_sample0"}
    with pytest.raises(TypeError, match="electrical_settle_sample0"):
        _doc(no_el)


def _refuse(token: str) -> None:
    msg = f"non-finite JSON constant {token}"
    raise AssertionError(msg)


def test_the_file_round_trips_exactly_as_canonical_ascii(tmp_path: Path) -> None:
    files = [rs.file_starts(session=s, fs=FS, stim_off_s=o, electrical_settle_s=e,
                            stim_off_source="03B — cached", electrical_source="(j) 6 (i)")
             for s, o, e in (("b_sr", 120.79, 124.2), ("a_sr", None, None),
                             ("c_sr", 121.5, None))]
    doc = rs.recovery_starts_document(files, fs=FS, extra={"run": "synthetic"})
    path = tmp_path / "recovery_starts.json"
    text = rs.write_recovery_starts(path, doc)
    raw = path.read_bytes()
    assert raw.decode("ascii") == text and b"\r" not in raw and raw.endswith(b"\n")
    json.loads(text, parse_constant=_refuse)  # no bare NaN / Infinity
    assert ": null" not in text  # a missing value is an absent key
    back = rs.read_recovery_starts(path)
    assert back == json.loads(json.dumps(doc))
    assert back["trim_modes"] == list(rs.TRIM_MODES)
    assert back["source_files"] == dict(rs.SOURCE_FILES)
    assert back["output_times"] == rs.output_times_record()
    assert back["files"][0]["electrical_settle_sample0"] == seconds_to_sample(124.2, FS)
    assert [f["session"] for f in back["files"]] == ["b_sr"]
    assert [f["session"] for f in back["held"]] == ["a_sr", "c_sr"]
    assert rs.write_recovery_starts(tmp_path / "again.json", back) == text
    bad = json.loads(text)
    bad["files"][0]["analyses"][0]["analysis"] = "velocityy"
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="not a consumer"):
        rs.read_recovery_starts(tmp_path / "bad.json")
    del bad["fs"]
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="'fs' is absent"):
        rs.read_recovery_starts(tmp_path / "bad.json")


def test_the_table_record_carries_every_stage_and_exclusion() -> None:
    rec = {r["analysis"]: r for r in rs.table_record(FS)}
    assert set(rec) == set(rs.ANALYSES)
    for name, a in rs.ANALYSES.items():
        assert rec[name]["settling_s"] == rs.analysis_settling(name, FS).settling_s
        assert len(rec[name]["excluded"]) == len(a.excluded)
        assert sum(len(o["stages"]) for o in rec[name]["outputs"]) == sum(
            len(o.stages) for o in a.outputs)
    cats = {e["category"] for r in rec.values() for e in r["excluded"]}
    assert {"selection rule", "edge guard", "epoch-wide statistic"} <= cats
