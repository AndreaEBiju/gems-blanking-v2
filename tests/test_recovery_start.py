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

RULING 2026-10-09 item 6 (mode (B), per output variable):

* (B) is the one mode and (A) is refused by name, with the ruling;
* every trimmed variable declares its class, and an absent or unknown class, a cut by an
  output its owner does not have, or a class (i) variable with no edge rule is refused;
* an INDEPENDENT table (``CLASS_TABLE``, transcribed from her code, not read from the map)
  gives every trimmed variable's class and reach, from the measured filter settlings and
  her window constants: class (i) keeps from electrical + its own input's settling (never
  half its window), class (ii) from electrical + its full reach, to the sample;
* the rules that decide the edge windows are cited and still on their lines; exactly the
  variables her >= 50 % rule governs are flagged as such;
* every windowed variable carries a valid fraction, the per-sample and event ones do not;
  the recomputed averages are exactly the ruled ones;
* every file's cuts are the map's, once each, as exact samples.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from gems_blanking_v2.extent import recovery_start as rs
from gems_blanking_v2.extent import tolerance as tl
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
    out += [(f"{v.file}/{v.path} edge", v.edge.source, v.edge.anchor) for v in rs.OUTPUT_VARS
            if v.edge is not None]
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
    assert checked >= (4 if pnew is None else 120)


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
    assert rec["marker_variable"] == rs.TRIM_MARKER == "night6_recovery_trim"
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
                          convention="sec0", action="nan"), "is not an analysis"),
            (rs.OutputVar("HRBR", rs.TRIM_MARKER, "parameter", "x"), "the trim marker")):
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
    ("pipeline_params.m", r"P\.sigmaWindowSec\s*=\s*5;"),
    ("slowWaveAnalysis_new.m", r"rateWinSec\s*=\s*60;"),
    ("HR_BR_HRVAnalysis_beats.m", r"sampEnWinSec = 60;\s+% fixed by design"),
    ("HR_BR_HRVAnalysis_beats.m", r"halfHrBr\s*=\s*hrBrWinSec / 2;"),
    ("HR_BR_HRVAnalysis_beats.m", r"halfWin\s*=\s*winSec / 2;"),
    ("extract_mmc.m", r"sigmaWin = g\('sigmaWin',30\)"),
    ("extract_mmc.m", r"W = g\('W',10\)"),
    ("extract_mmc.m", r"dW = g\('delayW',30\)"),
    ("extract_mmc.m", r"cardMs = g\('cardiacBlankMs',25\)"),
    ("extract_mmc.m", r"burstRefr = g\('burstRefractory', g\('refractory', 0\.5\)\)"),
    ("extract_mmc.m", r"spikeRefr = g\('spikeRefractory', 0\.05\)"),
    ("slowWaveAnalysis_new.m", r"minPeakDist_samp\s*=\s*round\(6 \* fs\);"),
    ("HR_BR_HRVAnalysis_beats.m", r"maxBreathRate_bpm = 170;"),
    ("HR_BR_HRVAnalysis_beats.m", r"minRR_sec\s*=\s*0\.1;"),
    ("HR_BR_HRVAnalysis_beats.m", r"maxRR_sec\s*=\s*0\.5;"),
    ("HR_BR_HRVAnalysis_beats.m",
     r"minBreathSepBeats = max\(2, round\(minBreathSepSec \* fs / meanRR_samp\)\);"),
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
    """RULING 2026-10-09 (c) 3: filled/filtered stages add, NaN-skipping ones add nothing.

    Each figure from its parts: the MEASURED edge settlings (spikes 7.782 ms, mmc 1.1594 s),
    task 13's impz where none was measured, her window constants, the look-backs of (d)
    (beat spacing 0.375 s, breath troughs 1.0 s, slow-wave peaks 6 s, mmc grouping 0.5 s),
    the kept CV2 bins and mmc delay window (c), and the mmc rate window ((d) 3, 5 s).
    """
    q = rs.beat_decimation_factor(FS)
    assert q == 12
    beats = 10 * q / FS + HR_EDGE_S + 0.375  # the HR band MEASURED at a NaN edge
    mmc_ev = beats + 0.050 + 1.1594 + 0.5
    want = {
        "spikes": 0.007782 + 15.0,                       # CV2 kept, (c) 3 (c)
        "hrv": beats,                                    # every window reads valid beats
        "breathing": beats + 1.0,                        # + trough spacing
        "slow_wave": impulse_settling_s(CONSUMER_FILTERS["slow_wave"], FS) + 2.5 + 6.0,
        "mmc": mmc_ev + 5.0 + 15.0,                      # rate window (d) 3 + delay (c)
    }
    for name, v in want.items():
        s = rs.analysis_settling(name, FS)
        assert s.settling_s == pytest.approx(v, abs=1e-12), name
        assert s.missing == ()
    assert rs.analysis_settling("spikes", FS).binding_output == "rolling CV2 (cv2WinSec 30 s bins)"
    assert rs.analysis_settling("mmc", FS).binding_output.startswith("cross-channel delay")
    assert rs.analysis_settling("slow_wave", FS).binding_output.startswith("slow-wave peak")
    # the values reported (2026-10-09, cohort fs), to the microsecond
    got = {k: round(rs.analysis_settling(k, FS).settling_s or -1, 6) for k in want}
    # hrv and breathing (and mmc, which reads the beats) moved with the HR-band edge settling
    # (0.141 s one-way impz -> 0.2374 s measured): was 0.520981 / 1.520981 / 22.230381
    assert got == {"spikes": 15.007782, "hrv": 0.617315, "breathing": 1.617315,
                   "slow_wave": 16.666973, "mmc": 22.326715}


def test_a_stage_that_skips_nan_adds_nothing_and_its_reach_is_still_reported() -> None:
    """(c) 3 (b): her 'omitnan' moving median and every valid-only window count 0."""
    med = next(s for s in rs.ANALYSES["mmc"].outputs[0].stages if "moving median" in s.what)
    assert med.skips_nan and rs.stage_settling_s(med, FS) == 15.0
    assert rs.stage_counted_s(med, FS) == 0.0
    t = {"spikes": rs.Analysis("spikes", "c", (rs.Output("o", (
        _stage(2.0), replace(_stage(30.0), skips_nan=True)), "x"),))}
    assert rs.analysis_settling("spikes", FS, t).settling_s == 1.0
    rec = {r["analysis"]: r for r in rs.table_record(FS)}
    st = [s for o in rec["mmc"]["outputs"] for s in o["stages"] if "moving median" in s["what"]]
    assert st and all(s["reach_s"] == 15.0 and s["counted_s"] == 0.0 and s["skips_nan"]
                      for s in st)
    skipped = {s.what for a in rs.ANALYSES.values() for o in a.outputs for s in o.stages
               if s.skips_nan}
    kept = {s.what for a in rs.ANALYSES.values() for o in a.outputs for s in o.stages
            if not s.skips_nan and any(k in s.what for k in ("CV2", "delay window",
                                                                "rate window W"))}
    assert len(skipped) == 10 and len(kept) == 3  # CV2 and delay (c), mmc rate (d) 3


def test_the_measured_edges_are_the_filters_and_carry_their_files() -> None:
    """(c) 1-2: spike, mmc and HR-band filters reach their MEASURED edge settling, not impz.

    The HR band (the beat detector's, read by hrv, breathing and mmc, and the heart-band
    trace) reaches its measured 0.2374 s (Andrea's item 2b answers).
    """
    rec = {r["analysis"]: r for r in rs.table_record(FS)}
    for name, value in (("spikes", 0.007782), ("mmc", 1.1594), ("hrv", None),
                        ("breathing", None)):
        st = [s for o in rec[name]["outputs"] for s in o["stages"]
              if s["kind"] == "filter at a NaN edge, measured"]
        hr = [s for s in st if "beat detector" in s["what"] or "her 10-150" in s["what"]]
        own = [s for s in st if s not in hr]
        assert (own if value is not None else hr), name
        assert all(s["counted_s"] == value and s["measurement"]["files"] for s in own)
        assert all(s["counted_s"] == HR_EDGE_S and s["measurement"]["files"] for s in hr)
        assert (name == "spikes") == (not hr), name  # every beat reader carries the band
    assert not [s for o in rec["hrv"]["outputs"] for s in o["stages"]
                if "10-150" in s["what"] and s["kind"] == "filter impulse response"]
    assert rs.BEAT_SPACING_S == 0.375
    doc = rs.recovery_starts_document([], fs=FS)
    assert doc["settling_rules"] == rs.RULING_SETTLING
    assert doc["edge_settling"]["mmc"]["pad_s"] == 1.5


def _matlab_round(x: float) -> int:
    return int(math.floor(x + 0.5))  # MATLAB round, half away from zero (x > 0)


def test_the_breath_trough_look_back_is_its_maximum_over_her_rr_range() -> None:
    """Her trough spacing in beats, in time, over her plausible RR [0.1, 0.5] s: <= 1.0 s."""
    sep = 60.0 / 170.0
    worst = max(max(2, _matlab_round(sep / rr)) * rr
                for rr in (0.1 + k * 1e-5 for k in range(40001)))
    assert worst == pytest.approx(rs.BREATH_TROUGH_SPACING_S, abs=1e-9)


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
    f = rs.file_starts(session="x", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=121.4,
                       times_source="03B", electrical_source="(j) 6 (i)",
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
    f = rs.file_starts(session="x", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=125.0,
                       times_source="a", electrical_source="b",
                       analyses=["spikes", "velocity"])
    rows = {r["analysis"]: r for r in f["analyses"]}
    assert rows["velocity"]["basis"] == rs.BASIS_FIXED
    assert rows["velocity"]["missing_settling"] == ["velocity: not in the table"]
    assert rows["spikes"]["relation"].startswith("later_than_fixed")


def test_undetected_edges_and_no_settling_hold_the_whole_file() -> None:
    a = rs.file_starts(session="x", fs=FS, electrical_end_s=None, mechanical_end_s=None,
                       electrical_settle_s=None,
                       times_source="a", electrical_source="b")
    assert a["basis"] == rs.HELD_UNDETECTED and "analyses" not in a
    b = rs.file_starts(session="x", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=None,
                       times_source="a", electrical_source="b")
    assert b["basis"] == rs.HELD_NO_SETTLING and "analyses" not in b
    with pytest.raises(ValueError, match="precedes the electrical end"):
        rs.file_starts(session="x", fs=FS, electrical_end_s=121.0, mechanical_end_s=121.0,
                       electrical_settle_s=120.0,
                       times_source="a", electrical_source="b")


def test_the_relation_to_the_epoch_start_is_exact_to_one_sample() -> None:
    """Earlier / at / later are decided on the shared sample rounding, not on seconds."""
    i0 = seconds_to_sample(132.0, FS)
    for k, word in ((i0 - 1, "earlier_than_fixed"), (i0, "at_fixed_start"),
                    (i0 + 1, "later_than_fixed")):
        t = {"spikes": rs.Analysis("spikes", "c", (rs.Output("o", (
            rs.Stage("w", "trailing window", "whole", "x.m:1", "x", "code",
                     window_s=k / FS - 130.0),), "x"),))}
        f = rs.file_starts(session="x", fs=FS, electrical_end_s=120.0, mechanical_end_s=120.0,
                           electrical_settle_s=130.0,
                           times_source="a", electrical_source="b", table=t)
        r = f["analyses"][0]
        assert r["start_sample0"] == k
        assert r["relation"].startswith(word)


@settings(max_examples=200, deadline=None)
@given(off=st.floats(100.0, 140.0), el=st.floats(0.0, 60.0),
       fs=st.sampled_from([FS, 24414.0, 1000.0]))
def test_every_start_is_its_electrical_time_plus_its_own_settling(off: float, el: float,
                                                                  fs: float) -> None:
    f = rs.file_starts(session="x", fs=fs, electrical_end_s=off, mechanical_end_s=off,
                       electrical_settle_s=off + el,
                       times_source="a", electrical_source="b")
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
    a = rs.file_starts(session="Sess_A", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=125.0,
                       times_source="a", electrical_source="b")
    held = rs.file_starts(session="sess_a", fs=FS, electrical_end_s=None, mechanical_end_s=None,
                          electrical_settle_s=None,
                          times_source="a", electrical_source="b")
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
    files = [rs.file_starts(session=s, fs=FS, electrical_end_s=o, mechanical_end_s=o,
                            electrical_settle_s=e,
                            times_source="03B — cached", electrical_source="(j) 6 (i)")
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
    # rows, not an object: MATLAB's jsondecode would mangle the keys (invariant 22)
    assert back["source_files"] == rs.source_files_record()
    assert {r["file"]: r["sha256"] for r in back["source_files"]} == dict(rs.SOURCE_FILES)
    assert [r["file"] for r in back["source_files"]] == sorted(rs.SOURCE_FILES)
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


def test_a_starts_file_from_other_edge_settlings_is_refused(tmp_path: Path) -> None:
    """Review 2026-10-09: v4+ records the edge settlings' sha256; a stale file is refused."""
    files = [rs.file_starts(session="b_sr", fs=FS, electrical_end_s=120.79, mechanical_end_s=120.79,
                            electrical_settle_s=124.2,
                            times_source="03B", electrical_source="(j) 6 (i)")]
    doc = rs.recovery_starts_document(files, fs=FS)
    assert rs.SCHEMA == "gems-blanking-v2 recovery starts v5"
    assert doc["edge_settling_sha256"] == tl.edge_settling_sha256()
    assert doc["edge_settling"] == tl.edge_settling_record()
    good = tmp_path / "good.json"
    rs.write_recovery_starts(good, doc)
    assert rs.read_recovery_starts(good)["edge_settling_sha256"] == tl.edge_settling_sha256()
    for name, change in (
            ("hash", lambda j: j.update(edge_settling_sha256="0" * 64)),
            ("content", lambda j: j["edge_settling"]["mmc"].update(pad_s=1.0)),
            ("both", lambda j: (j["edge_settling"]["mmc"].update(pad_s=1.0),
                                j.update(edge_settling_sha256="1" * 64)))):
        j = json.loads(good.read_text(encoding="utf-8"))
        change(j)
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(j), encoding="utf-8", newline="\n")
        with pytest.raises(ValueError, match="not this build's"):
            rs.read_recovery_starts(f)
    j = json.loads(good.read_text(encoding="utf-8"))
    del j["edge_settling_sha256"]
    f = tmp_path / "absent.json"
    f.write_text(json.dumps(j), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="'edge_settling_sha256' is absent"):
        rs.read_recovery_starts(f)
    j["schema"] = "gems-blanking-v2 recovery starts v3"
    f.write_text(json.dumps(j), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="expected 'gems-blanking-v2 recovery starts v5'"):
        rs.read_recovery_starts(f)


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


# ---------------------------------------------------------------------------
# RULING 2026-10-09 item 6: mode (B), per output variable
# ---------------------------------------------------------------------------


def test_b_is_the_one_mode_and_a_is_refused_by_name() -> None:
    assert rs.TRIM_MODES == ("mask_to_electrical_drop_outputs",)
    assert set(rs.WITHDRAWN_TRIM_MODES) == {"mask_to_own_start"}
    assert "RULING 2026-10-09 item 6" in rs.WITHDRAWN_TRIM_MODES["mask_to_own_start"]
    assert not set(rs.TRIM_MODES) & set(rs.WITHDRAWN_TRIM_MODES)
    assert set(rs.TRIM_CLASSES) == {"valid_only", "filled_or_filtered"}


HR_EDGE_S = 0.2374
"""The 10-150 Hz heart band at a NaN edge (edgepad/hr_edge_settling*.json), written here as a
number, not read from the map."""


def _independent_reaches() -> dict[str, float]:
    """Each reach from the MEASURED settlings and her constants (not the map).

    RULING 2026-10-09 (c) 3: NaN-skipping stages (valid-only windows, her 'omitnan' moving
    median/MAD) add nothing; look-backs (beat spacing 0.375 s, breath troughs 1.0 s, slow-
    wave peaks 6 s, mmc grouping 0.5 s) add; (d) 3: the mmc rate adds its 5 s look-back.
    """
    q = 12
    beats = 10 * q / FS + HR_EDGE_S + 0.375           # FIR + HR band at a NaN edge + spacing
    band = 0.007782                                   # step1 at a NaN edge (measured)
    mmc_sig = beats + 0.050 + 1.1594                  # whole blank + band at a NaN edge
    mmc_ev = mmc_sig + 0.0 + 0.0 + 0.5                # median, MAD skip NaN; grouping
    sw = impulse_settling_s(CONSUMER_FILTERS["slow_wave"], FS) + 5.0 / 2
    return {"band": band, "wave": band + 0.0015, "env": band + 0.010 / 2, "beats": beats,
            "troughs": beats + 1.0,
            "trace": HR_EDGE_S, "sw": sw, "cv2": band + 30.0 / 2,
            "sw_peaks": sw + 6.0, "sw_rate": sw + 6.0, "mmc_sig": mmc_sig, "mmc_ev": mmc_ev,
            "mmc_rate": mmc_ev + 10.0 / 2, "mmc_delay": mmc_ev + 10.0 / 2 + 30.0 / 2}


_I, _II = "valid_only", "filled_or_filtered"
CLASS_TABLE: dict[tuple[str, str], tuple[str, str]] = {
    # spikes_v2: detection on D.filtered (step1_bandpass.m:53-57 fills, filtfilt) is (ii);
    # the windows over validMask only (step2 :95, step3b :99, step6 :49, :149) are (i)
    ("spikes_v2", "sigmaWin.sigma"): (_I, "band"),
    **{("spikes_v2", f"spikes.{v}"): (_II, "band")
       for v in ("centers", "times", "peakAmp_uv", "threshAtSpike_uv", "artifactMask")},
    **{("spikes_v2", f"spikes.{v}"): (_II, "wave")
       for v in ("waveforms", "alignedCenters", "alignedTimes", "Vpp_uv", "width_ms")},
    **{("spikes_v2", f"envelope.{v}"): (_I, "env")
       for v in ("rms_uv", "sigmaFloor_uv", "excess_uv", "validFrac")},
    ("spikes_v2", "metrics.fr_hz"): (_I, "band"),
    ("spikes_v2", "metrics.fr_validFrac"): (_I, "band"),
    # Andrea, 2026-10-09: cv2_roll is cut at its FULL kept reach (the 30 s bins counted,
    # (c) 3 (c)), the analysis's binding 15.007782 s - and no other spike output is
    ("spikes_v2", "metrics.cv2_roll"): (_I, "cv2"),
    ("spikes_v2", "metrics.burst.onsets"): (_I, "wave"),
    ("spikes_v2", "metrics.burst.offsets"): (_I, "wave"),
    # HR_BR_HRVAnalysis_beats: the trace is filtered over the linear fill (:262, :282), (ii);
    # beats, intervals and every window over valid beats (:298, :829-928, :998) are (i)
    ("HRBR", "heartBeatSeries"): (_II, "trace"),
    **{("HRBR", v): (_I, "beats")
       for v in ("heartlocs", "heartRateSeries", "heartCountSeries", "heartCountValidSec",
                 "heartCountRateSeries")},
    ("HRBR", "breathRateSeries"): (_I, "troughs"),
    ("HRBR", "br_locs_true"): (_I, "troughs"),
    **{("HRVMeasures", v): (_I, "beats")
       for v in ("heartlocs", "RR_intervals", "RR_times", "hrv_series", "rmssd_series",
                 "pnn5_series", "sd1_series", "sd2_series", "sampEn_series", "nRR_used")},
    # slowWaveAnalysis_new: fillmissing before the low-pass (:126-133): all (ii)
    ("slowWaves", "slowWaveTimeSeries"): (_II, "sw"),
    ("slowWaves", "slowWavePeakLocs"): (_II, "sw_peaks"),
    ("slowWaves", "slowWaveRateSeries"): (_II, "sw_rate"),
    # extract_mmc: the band over the cardiac-blanked fill (:104-106), the events on it, and
    # the delay over mean-filled rate rows (:270) are (ii); the rate and peak amplitude over
    # valid samples with her 50 % rule (:284-293) are (i)
    ("mmc", "mmc.signal"): (_II, "mmc_sig"),
    **{("mmc", f"mmc.{lvl}.events"): (_II, "mmc_ev") for lvl in ("firing", "burst")},
    # RULING 2026-10-09 (d) 3: the rate (and its peakAmp) cut = its input's settling + 5 s
    **{("mmc", f"mmc.{lvl}.{v}"): (_I, "mmc_rate")
       for lvl in ("firing", "burst") for v in ("rate", "peakAmp")},
    ("mmc", "mmc.delay"): (_II, "mmc_delay"),
}
"""Every trimmed variable: its class and its reach, transcribed from her code."""


def test_every_trimmed_variable_is_cut_at_its_class_reach_to_the_sample() -> None:
    """Class (i) from electrical + own input settling, class (ii) + full reach (independent)."""
    reach = _independent_reaches()
    trims = {(v.file, v.path): v for v in rs.OUTPUT_VARS if v.role == "trim"}
    assert set(trims) == set(CLASS_TABLE), sorted(set(trims) ^ set(CLASS_TABLE))
    el = 125.0
    f = rs.file_starts(session="x", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=el,
                       times_source="a", electrical_source="b")
    cuts = {c["cut"]: c for c in f["cuts"]}
    for key, (cls, r) in CLASS_TABLE.items():
        v = trims[key]
        assert v.trim_class == cls, key
        owner = v.owner or rs.OUTPUT_FILES[v.file].owner
        c = cuts[rs.cut_id(owner, v.reach, cls)]
        assert c["trim_class"] == cls and c["basis"] == rs.BASIS_CUT, key
        assert c["reach_s"] == pytest.approx(reach[r], abs=1e-12), key
        assert c["start_sample0"] == seconds_to_sample(el + c["reach_s"], FS), key
        assert c["start_sample0"] == seconds_to_sample(el + reach[r], FS), key
    # (i) is NOT cut by half its window: heart rate's cut is its beats' settling alone
    hr = cuts[rs.cut_id("hrv", "heart_rate", _I)]
    assert hr["reach_s"] == pytest.approx(reach["beats"], abs=1e-12)
    # RULING 2026-10-09 (d) 3: the mmc rate is cut 5 s after its input's settling, which is
    # the band at a NaN edge with the NaN-skipping stages adding nothing (c) 3 (b)
    rate = cuts[rs.cut_id("mmc", "mmc_rate", _I)]
    ev = cuts[rs.cut_id("mmc", "mmc_events", _II)]
    assert rate["reach_s"] == pytest.approx(ev["reach_s"] + 5.0, abs=1e-12)
    assert rate["start_sample0"] == seconds_to_sample(el + reach["mmc_ev"] + 5.0, FS)
    assert {c["cut"] for c in f["cuts"]} == {rs.cut_id(*c) for c in rs.trim_cuts()}
    assert rs.cut_id("slow_wave", "sw_peaks", _II) in cuts  # peaks: + MinPeakDistance 6 s


SPIKE_FILES = ("spikes_v2",)


def test_only_cv2_roll_is_cut_at_15_s_every_other_spike_output_at_its_ms_reach() -> None:
    """Andrea, 2026-10-09: the 15.0 s spike cut applies ONLY to ``metrics.cv2_roll``.

    cv2_roll at electrical + 15.007782 s (one figure with the analysis); spike times and the
    firing rate from 7.782 ms after the electrical settling; any other spike variable cut at
    or beyond 1 s fails.
    """
    el = 125.0
    f = rs.file_starts(session="x", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=el,
                       times_source="a", electrical_source="b")
    cuts = {c["cut"]: c for c in f["cuts"]}
    trims = [v for v in rs.OUTPUT_VARS if v.role == "trim" and v.file in SPIKE_FILES]
    by_path = {v.path: cuts[rs.cut_id("spikes", v.reach, str(v.trim_class))] for v in trims}
    cv2 = by_path["metrics.cv2_roll"]
    assert cv2["reach_s"] == pytest.approx(15.007782, abs=1e-12)
    assert cv2["reach_s"] == rs.analysis_settling("spikes", FS).settling_s  # one figure
    assert cv2["start_sample0"] == seconds_to_sample(el + 15.007782, FS)
    for path in ("spikes.times", "spikes.centers", "metrics.fr_hz"):
        assert by_path[path]["reach_s"] == pytest.approx(0.007782, abs=1e-12), path
        assert by_path[path]["start_sample0"] == seconds_to_sample(el + 0.007782, FS), path
    long_ = sorted(p for p, c in by_path.items() if c["reach_s"] >= 1.0)
    assert long_ == ["metrics.cv2_roll"], long_
    assert all(c["reach_s"] < 0.013 for p, c in by_path.items() if p != "metrics.cv2_roll")


def test_the_hr_band_edge_moves_the_hrv_and_breathing_starts_and_the_trace_cut() -> None:
    """Andrea's item 2b answers: wherever the HR band enters a reach, it is 0.2374 s measured.

    hrv = FIR 0.0049 + band 0.2374 + beat spacing 0.375 = 0.617315 s; breathing + trough
    spacing 1.0 s = 1.617315 s; the heart-band trace cut 0.2374 s (was 0.140165 s).
    """
    q = rs.beat_decimation_factor(FS)
    assert rs.analysis_settling("hrv", FS).settling_s == pytest.approx(
        10 * q / FS + 0.2374 + 0.375, abs=1e-12)
    assert rs.analysis_settling("breathing", FS).settling_s == pytest.approx(
        10 * q / FS + 0.2374 + 0.375 + 1.0, abs=1e-12)
    tr, miss = rs.output_reach_s("hrv", "heart_band_trace", _II, FS)
    assert miss == () and tr == 0.2374
    assert tr > impulse_settling_s(CONSUMER_FILTERS["hrv"], FS)  # the one-way impz understates


@settings(max_examples=100, deadline=None)
@given(el=st.floats(120.0, 200.0), fs=st.sampled_from([FS, 24414.0, 1000.0]))
def test_every_cut_is_one_rounding_of_electrical_plus_reach(el: float, fs: float) -> None:
    f = rs.file_starts(session="x", fs=fs, electrical_end_s=119.0, mechanical_end_s=119.0,
                       electrical_settle_s=el,
                       times_source="a", electrical_source="b")
    for c in f["cuts"]:
        r, missing = rs.output_reach_s(c["owner"], c["output_key"], c["trim_class"], fs)
        assert r is not None and missing == ()
        assert c["start_s"] == el + r
        assert c["start_sample0"] == seconds_to_sample(el + r, fs)


def test_an_unknown_reach_keeps_132_labelled_never_the_known_part() -> None:
    t = {**rs.ANALYSES, "slow_wave": rs.Analysis("slow_wave", "c", (
        rs.Output("trace", (_stage(5.0),), "x", key="sw_trace"),
        rs.Output("rate", (_stage(5.0), _stage(None, "unknown"), _stage(60.0)), "x",
                  key="sw_rate", own_window=True)))}
    f = rs.file_starts(session="x", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=121.4,
                       times_source="a", electrical_source="b", table=t)
    cuts = {c["cut"]: c for c in f["cuts"]}
    rate = cuts[rs.cut_id("slow_wave", "sw_rate", _II)]
    assert rate["basis"] == rs.BASIS_FIXED and rate["start_s"] == 132.0
    assert rate["missing_settling"] == ["slow_wave/rate: w"] and "reach_s" not in rate
    trace = cuts[rs.cut_id("slow_wave", "sw_trace", _II)]
    assert trace["basis"] == rs.BASIS_CUT and trace["reach_s"] == 2.5
    gone = rs.output_reach_s("slow_wave", "no_such", _II, FS, t)
    assert gone == (None, ("slow_wave: 0 outputs keyed 'no_such'",))
    with pytest.raises(ValueError, match="unknown trim class"):
        rs.output_reach_s("hrv", "heart_rate", "probably_valid", FS)


def test_a_trimmed_variable_without_a_known_class_is_refused_by_name(
        monkeypatch: pytest.MonkeyPatch) -> None:
    orig = rs.OUTPUT_VARS
    i = next(k for k, v in enumerate(orig) if (v.file, v.path) == ("HRBR", "heartRateSeries"))
    good = orig[i]
    for bad, words in (
            (replace(good, trim_class=None), "HRBR/heartRateSeries has no trim class"),
            (replace(good, trim_class="partly"), "unknown trim class 'partly'"),
            (replace(good, reach="heart_rates"), "reach 'heart_rates' is not an output key"),
            (replace(good, reach="sw_rate"), "is not an output key of hrv"),
            (replace(good, edge=None), "must cite the rule"),
            (replace(good, valid=rs.ValidFraction("hr_window", "x", width="nope",
                                                  validity="invalidMask")), "'nope'"),
            (replace(good, valid=rs.ValidFraction("hr_window", "x", width="winSec")),
             "needs its window and its validity"),
            (replace(good, valid=rs.ValidFraction("guess", "x")),  # type: ignore[arg-type]
             "unknown valid-fraction"),
            (replace(good, valid=rs.ValidFraction("her", "x", variable="nothing")),
             "is not declared in HRBR")):
        monkeypatch.setattr(rs, "OUTPUT_VARS", (*orig[:i], bad, *orig[i + 1:]))
        with pytest.raises(ValueError, match=re.escape(words)):
            rs.output_times_record()
    j = next(k for k, v in enumerate(orig) if (v.file, v.path) == ("HRBR", "avgHeartRate"))
    for rc, words in ((rs.Recompute("mean_omitnan", "metrics_t"), "not a trimmed variable"),
                      (rs.Recompute("mean_well_sampled", "heartRateSeries"), "needs where"),
                      (rs.Recompute("events_per_valid_s", "heartRateSeries"), "its validity")):
        monkeypatch.setattr(rs, "OUTPUT_VARS",
                            (*orig[:j], replace(orig[j], recompute=rc), *orig[j + 1:]))
        with pytest.raises(ValueError, match=re.escape(words)):
            rs.output_times_record()
    monkeypatch.setattr(rs, "OUTPUT_VARS", (
        *orig, rs.OutputVar("HRBR", "heartRateSeries_validFraction", "parameter", "x")))
    with pytest.raises(ValueError, match="would overwrite a declared variable"):
        rs.output_times_record()


HALF_VALID = {("HRBR", "heartCountRateSeries"),
              *{("mmc", f"mmc.{lvl}.{v}") for lvl in ("firing", "burst")
                for v in ("rate", "peakAmp")}}
"""The class (i) variables her >= 50 %-valid window rule governs (HR_BR :802/:895,
extract_mmc :49/:290). HR (:844), HRV (:902), sample entropy (:928) and the spike rate
(step6 :149) are governed by other rules - cited as they are, a contradiction of the
ruling's examples."""


def test_exactly_her_half_valid_rule_is_flagged_as_it() -> None:
    got = {(v.file, v.path) for v in rs.OUTPUT_VARS
           if v.role == "trim" and v.trim_class == _I and v.edge and v.edge.half_valid}
    assert got == HALF_VALID
    for v in rs.OUTPUT_VARS:
        if v.role == "trim" and v.trim_class == _I:
            assert v.edge is not None and re.match(r"^\S+\.m:\d+", v.edge.source), v.path
    sw = next(v for v in rs.OUTPUT_VARS if v.path == "slowWaveRateSeries")
    assert sw.trim_class == _II and sw.edge is not None and sw.edge.half_valid  # filled data


WINDOWED = {("spikes_v2", "sigmaWin.sigma"),
            *{("spikes_v2", f"envelope.{v}") for v in ("rms_uv", "sigmaFloor_uv", "excess_uv",
                                                       "validFrac")},
            ("spikes_v2", "metrics.fr_hz"), ("spikes_v2", "metrics.fr_validFrac"),
            ("spikes_v2", "metrics.cv2_roll"),
            *{("HRBR", v) for v in ("heartRateSeries", "heartCountSeries", "heartCountValidSec",
                                    "heartCountRateSeries", "breathRateSeries")},
            *{("HRVMeasures", v) for v in ("hrv_series", "rmssd_series", "pnn5_series",
                                           "sd1_series", "sd2_series", "sampEn_series",
                                           "nRR_used")},
            ("slowWaves", "slowWaveRateSeries"),
            *{("mmc", f"mmc.{lvl}.{v}") for lvl in ("firing", "burst")
              for v in ("rate", "peakAmp")},
            ("mmc", "mmc.delay")}
"""Every windowed output: each value carries its valid fraction. The rest are samples or
events, whose validity is their own NaN or their presence."""


def test_every_windowed_value_carries_its_valid_fraction() -> None:
    rec = rs.output_times_record()
    got = {(v["file"], v["path"]) for v in rec["vars"] if "valid_fraction" in v}
    assert got == WINDOWED
    her = {(v["file"], v["path"]): v["valid_fraction"]["variable"] for v in rec["vars"]
           if v.get("valid_fraction", {}).get("kind") == "her"}
    assert her == {**{("spikes_v2", f"envelope.{v}"): "envelope.validFrac"
                      for v in ("rms_uv", "sigmaFloor_uv", "excess_uv", "validFrac")},
                   ("spikes_v2", "metrics.fr_hz"): "metrics.fr_validFrac",
                   ("spikes_v2", "metrics.fr_validFrac"): "metrics.fr_validFrac"}
    for v in rec["vars"]:
        vf = v.get("valid_fraction")
        if vf and vf["kind"] != "her":
            assert vf["sibling"] == v["path"] + rs.VALID_FRACTION_SUFFIX


RECOMPUTED = {("HRBR", "avgHeartRate"): "heartRateSeries",
              ("HRBR", "avgBreathRate"): "breathRateSeries",
              ("HRBR", "avgHeartCount"): "heartCountSeries",
              ("HRBR", "avgHeartCountRate"): "heartCountRateSeries",
              ("slowWaves", "avgSlowWave"): "slowWaveRateSeries",
              ("mmc", "mmc.firing.avgRate"): "mmc.firing.events",
              ("mmc", "mmc.burst.avgRate"): "mmc.burst.events",
              ("spikes_v2", "spikes.nSpikes"): "spikes.alignedCenters",
              ("spikes_v2", "metrics.nSpikes"): "spikes.alignedTimes",
              ("spikes_v2", "envelope.meanRMS_uv"): "envelope.rms_uv",
              ("spikes_v2", "envelope.meanExcess_uv"): "envelope.excess_uv"}


def test_the_recomputed_averages_are_the_ruled_ones() -> None:
    rec = rs.output_times_record()
    got = {(v["file"], v["path"]): v["recompute"]["of"] for v in rec["vars"]
           if v["role"] == "recomputed"}
    assert got == RECOMPUTED
    by = {(v["file"], v["path"]): v for v in rec["vars"]}
    assert by[("HRBR", "avgBreathRate")]["owner"] == "breathing"
    assert by[("mmc", "mmc.firing.avgRate")]["recompute"]["validity"] == "mmc.signal"


def test_every_file_carries_the_maps_cuts_once_as_samples() -> None:
    a = rs.file_starts(session="Sess_A", fs=FS, electrical_end_s=120.8, mechanical_end_s=120.8,
                       electrical_settle_s=125.0,
                       times_source="a", electrical_source="b")
    for edit, err, words in (
            (lambda f: f.pop("cuts"), TypeError, "cuts"),
            (lambda f: f["cuts"].append(dict(f["cuts"][0])), ValueError, "a cut appears twice"),
            (lambda f: f["cuts"].pop(), ValueError, "differ from the map"),
            (lambda f: f["cuts"][0].update(start_sample0=1.5), TypeError, "start_sample0")):
        bad = json.loads(json.dumps(a))
        edit(bad)
        with pytest.raises(err, match=words):
            _doc(bad)


# ---------------------------------------------------------------------------
# RULING 2026-10-09 (i) 3 (c): stim-off is two times (starts v5)
# ---------------------------------------------------------------------------


def _two(e_end: float, settle: float, m_end: float, fs: float = FS,
         table: dict[str, rs.Analysis] | None = None) -> dict[str, Any]:
    return rs.file_starts(session="x", fs=fs, electrical_end_s=e_end, mechanical_end_s=m_end,
                          electrical_settle_s=settle, times_source="tsq AmA / Mon",
                          electrical_source="electrical-only rule", table=table)


def test_the_later_of_the_two_stim_times_sets_every_start_and_cut() -> None:
    """max(E + settling + own, M + own): either time can bind, and both are recorded."""
    # electrical binds: AmA ended 2 s before the gate, its settling ends after the gate
    a = _two(118.69, 121.40, 120.73)
    # mechanical binds: the electrical recovery is settled before the gate closes
    b = _two(118.69, 119.10, 120.73)
    for f, t, bind in ((a, 121.40, "electrical"), (b, 120.73, "mechanical")):
        assert f["input_mask_s"] == t and f["input_mask_binding"] == bind
        assert f["input_mask_sample0"] == seconds_to_sample(t, FS)
        assert f["input_mask_sample0"] == max(f["electrical_settle_sample0"],
                                              f["mechanical_end_sample0"])
        assert f["electrical_end_s"] == 118.69 and f["mechanical_end_s"] == 120.73
        assert f["electrical_s"] == pytest.approx(f["electrical_settle_s"] - 118.69)
        assert "stim_off_s" not in f
        for r in f["analyses"]:
            own = rs.analysis_settling(r["analysis"], FS).settling_s
            assert own is not None
            assert r["start_s"] == t + own, (bind, r["analysis"])
            assert r["start_sample0"] == seconds_to_sample(t + own, FS)
            assert r["basis"] == rs.BASIS_MEASURED
        for c in f["cuts"]:
            assert c["basis"] == rs.BASIS_CUT
            assert c["start_s"] == t + c["reach_s"], (bind, c["cut"])
            assert c["start_sample0"] == seconds_to_sample(t + c["reach_s"], FS)
    same = _two(120.0, 120.5, 120.5)
    assert same["input_mask_binding"] == "both"


@settings(max_examples=300, deadline=None)
@given(e_end=st.floats(100.0, 140.0), el=st.floats(0.0, 60.0), dm=st.floats(-3.0, 3.0),
       fs=st.sampled_from([FS, 24414.0, 1000.0]))
def test_every_start_and_cut_is_the_max_of_both_times_plus_its_reach(
        e_end: float, el: float, dm: float, fs: float) -> None:
    f = _two(e_end, e_end + el, e_end + dm, fs)
    t = max(e_end + el, e_end + dm)
    assert f["input_mask_s"] == t
    assert f["input_mask_sample0"] == max(seconds_to_sample(e_end + el, fs),
                                          seconds_to_sample(e_end + dm, fs))
    for r in f["analyses"]:
        own = rs.analysis_settling(r["analysis"], fs).settling_s
        assert own is not None
        assert r["start_s"] == t + own
        assert r["start_s"] == pytest.approx(max(e_end + el + own, e_end + dm + own), abs=1e-9)
    for c in f["cuts"]:
        assert c["start_s"] == t + c["reach_s"]
        assert c["start_sample0"] >= f["input_mask_sample0"]
    rs.recovery_starts_document([f], fs=fs)  # consistent by construction


def test_a_file_missing_either_stim_time_is_held_naming_it() -> None:
    for e_end, m_end, word in ((None, 120.7, "electrical"), (118.7, None, "mechanical"),
                               (None, None, "electrical and mechanical")):
        f = rs.file_starts(session="x", fs=FS, electrical_end_s=e_end, mechanical_end_s=m_end,
                           electrical_settle_s=121.0, times_source="t", electrical_source="e")
        assert f["basis"] == rs.HELD_UNDETECTED and "analyses" not in f
        assert f"({word})" in f["why"], f["why"]
    with pytest.raises(ValueError, match="precedes the electrical end"):
        _two(120.0, 119.9, 118.0)  # settling is measured FROM the electrical end


def test_the_document_refuses_a_file_that_disagrees_with_its_two_times() -> None:
    a = _two(118.69, 119.10, 120.73)  # mechanical binds
    for edit, err, words in (
            (lambda f: f.pop("mechanical_end_sample0"), TypeError, "mechanical_end_sample0"),
            (lambda f: f.pop("mechanical_end_s"), TypeError, "mechanical_end_s"),
            (lambda f: f.pop("electrical_end_s"), TypeError, "electrical_end_s"),
            (lambda f: f.pop("input_mask_sample0"), TypeError, "input_mask_sample0"),
            (lambda f: f.update(input_mask_sample0=f["electrical_settle_sample0"]), ValueError,
             "is not max"),
            (lambda f: f["analyses"][0].update(start_sample0=f["input_mask_sample0"] - 1),
             ValueError, "precede the input mask"),
            (lambda f: f["cuts"][0].update(start_sample0=f["input_mask_sample0"] - 1),
             ValueError, "precede the input mask")):
        bad = json.loads(json.dumps(a))
        edit(bad)
        with pytest.raises(err, match=words):
            _doc(bad)
    _doc(a)


def test_a_v4_starts_file_with_one_stim_time_is_refused(tmp_path: Path) -> None:
    doc = rs.recovery_starts_document([_two(118.69, 121.4, 120.73)], fs=FS)
    j = json.loads(json.dumps(doc))
    j["schema"] = "gems-blanking-v2 recovery starts v4"
    f = tmp_path / "v4.json"
    f.write_text(json.dumps(j), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="expected 'gems-blanking-v2 recovery starts v5'"):
        rs.read_recovery_starts(f)
