"""RULING 2026-10-09 (c) 6: slow wave at masked edges, option (a), and the decimated rate.

Night 6 passes each slow-wave run's masked spans to ``slowWaveAnalysis_new`` as
``blankIdx`` as well as NaN, so her 15 s ``edgeBufferSec`` guards every masked span; and it
can run her decimated by 78 (13 x 6) behind a REQUIRED declaration
(``night6_slow_wave_rates``: ``full`` | ``decimated78``, no default). One MATLAB process
(``tests/matlab/check_slow_wave_rate.m``) checks, against independent expectations
computed here:

* refusals by name: no rate, an unknown rate, and any factor that does not divide 24414
  (7, 2 x 2, 1) or the epoch's whole-hertz rate (78 at 1000 Hz); 78 and 313 pass;
* the decimated path is the decimation check's own logic BIT FOR BIT (its
  ``decimate_masked`` and ``map_spans``, copied into the harness), and the mask rule: a
  decimated row is NaN iff any of its 78 source rows is, at the block boundaries (source
  rows 78 and 79 fall in decimated rows 1 and 2), the tail past the last whole block
  dropped, every column alike; a span that is not the input's NaN is refused;
* time mapping back with a 1-sample boundary: her decimated peak rows 1, 2, 40 become
  epoch rows 1, 79, 3043 (decimated row j = epoch row (j - 1) 78 + 1), her own rows kept;
* with her real function, at both rates: a SHARED call (two channels with one mask) and a
  per-channel call - each kept file's blankIdx is exactly its run's masked spans (mapped
  by the any-source rule when decimated), her invalidMask is exactly their cover, her
  edge mask covers 15 s on both sides of every span, the peaks are epoch rows; spans that
  are not the input's NaN are refused (``night6:blankIdx``).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.extent import recovery_start as rs
from gems_blanking_v2.extent.grid import seconds_to_sample
from scipy.io import savemat

from tests.test_matlab_acceptance import _matlab, _processing_new

FS = 24414.0625
R = 78
NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"

DEC_N = R * 400 + 5
DEC_SPANS = [[78, 78], [79, 79], [157, 234], [1000, 1100], [31201, 31203]]
"""1-based inclusive source rows: the last row of block 1, the first of block 2, exactly
block 3, a run over blocks 13-15, and a span in the 5-row tail decimation drops.
400 blocks: filtfilt needs more than 3 x the FIR order (120) at the second stage."""
DEC_SPD = [[1, 1], [2, 2], [3, 3], [13, 15]]
DEC_NAN_ROWS = [1, 2, 3, 13, 14, 15]

CALL_S = 80.0
EDGE_S = 15.0
TRIM_SESSION = "sw_rate_session"
TRIM_EL_S = 1.0
"""The trim case: a decimated run's file cut by mode (B), electrical settling at 1 s."""


def _span(a_s: float, b_s: float) -> list[int]:
    return [int(round(a_s * FS)), int(round(b_s * FS))]


SPANS_SHARED = [_span(20.0, 21.0), [int(round(21.5 * FS)) + 1, int(round(22.0 * FS))]]
SPANS_ANT2 = [_span(50.0, 51.0)]
FREE_S = {"shared": (40.0, 60.0), "own": (17.0, 33.0)}
"""Where no span's 15 s buffer and no epoch end reaches, per run."""


def _map(spans: list[list[int]], n_dec: int) -> list[list[int]]:
    """Map source rows a..b to decimated rows by the any-source rule (independently)."""
    out = [[(a - 1) // R + 1, (b - 1) // R + 1] for a, b in spans]
    return [[a, min(b, n_dec)] for a, b in out if a <= n_dec]


def _runs(a: object) -> list[list[int]]:
    x = np.asarray(a, dtype=np.int64)
    return x.reshape(-1, 2).tolist() if x.size else []


def _case(tmp: Path) -> dict[str, Any]:
    rng = np.random.default_rng(9)
    savemat(tmp / "dec.mat", {"X": rng.normal(0.0, 1e-5, (DEC_N, 3)), "fs": FS,
                              "spans": np.asarray(DEC_SPANS, dtype=np.float64)})
    n = int(CALL_S * FS)
    t = np.arange(n) / FS
    x = np.column_stack([1e-4 * np.sin(2 * np.pi * 0.1 * t + 0.5 * k)
                         + rng.normal(0.0, 1e-6, n) for k in range(3)])
    savemat(tmp / "calls.mat", {"X": x, "fs": FS,
                                "spans_shared": np.asarray(SPANS_SHARED, dtype=np.float64),
                                "spans_ant2": np.asarray(SPANS_ANT2, dtype=np.float64)})
    work = tmp / "work"
    work.mkdir()
    pnew = _processing_new()
    assert pnew is not None
    starts = tmp / "starts.json"
    rs.write_recovery_starts(starts, rs.recovery_starts_document([rs.file_starts(
        session=TRIM_SESSION, fs=FS, stim_off_s=0.5, electrical_settle_s=TRIM_EL_S,
        stim_off_source="test", electrical_source="test")], fs=FS))
    return {"processing_new": pnew.as_posix(), "night6": NIGHT6.as_posix(),
            "work": work.as_posix(), "decimate": {"input_file": (tmp / "dec.mat").as_posix()},
            "calls": {"input_file": (tmp / "calls.mat").as_posix()},
            "trim": {"starts_file": starts.as_posix(), "session": TRIM_SESSION}}


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    tmp = tmp_path_factory.mktemp("sw_rate")
    case_file, res_file = tmp / "case.json", tmp / "result.json"
    case_file.write_text(json.dumps(_case(tmp)), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{HARNESS.as_posix()}'); "
           f"check_slow_wave_rate('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=1800, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return dict(json.loads(res_file.read_text(encoding="utf-8")))


def test_the_rate_is_required_and_a_non_divisor_factor_is_refused(
        result: dict[str, Any]) -> None:
    r = result["refusals"]
    assert r["none"] == r["unknown"] == "night6:slowWaveRate"
    for k in ("f7", "f4", "f78_at_1000", "f1", "decimate_f7"):
        assert r[k] == "night6:decimationFactor", k
    assert r["f78"] == r["f313"] == ""   # 78 x 313 = 24414
    assert 24414 % 78 == 0 and 24414 % 7 and 24414 % 4
    assert result["table"]["names"] == ["full", "decimated78"]
    assert result["table"]["factor"] == R and result["table"]["factors"] == [13, 6]
    assert result["full_factor"] == 1


def test_the_decimated_path_is_the_decimation_checks_own_bit_for_bit(
        result: dict[str, Any]) -> None:
    d = result["decimate"]
    assert d["equal_to_check"] and d["spans_equal_to_check"]
    assert d["nd"] == DEC_N // R == 400
    assert d["fsd"] == pytest.approx(FS / R, rel=1e-15)


def test_a_decimated_sample_is_masked_iff_any_source_sample_is(
        result: dict[str, Any]) -> None:
    d = result["decimate"]
    assert _runs(d["spd"]) == DEC_SPD == _map(DEC_SPANS, d["nd"])
    assert np.atleast_1d(d["nan_rows"]).tolist() == DEC_NAN_ROWS
    assert d["nan_all_columns_equal"]
    src = np.zeros(DEC_N, dtype=bool)
    for a, b in DEC_SPANS:
        src[a - 1:b] = True
    want = np.flatnonzero(src[:400 * R].reshape(400, R).any(axis=1)) + 1
    assert want.tolist() == DEC_NAN_ROWS
    assert d["refused"] == "night6:decimatedMask"   # a NaN the spans do not hold


def test_her_decimated_peak_rows_map_back_to_epoch_rows_to_the_sample(
        result: dict[str, Any]) -> None:
    k = result["keep"]
    assert np.atleast_1d(k["called"]).tolist() == [1, 2, 40]
    assert np.atleast_1d(k["locs"]).tolist() == [1, 79, 3043]   # (j - 1) 78 + 1
    assert k["fs"] == pytest.approx(FS / R)   # her series stay at her rate


@pytest.mark.parametrize("rate", ["full", "decimated"])
@pytest.mark.parametrize("run", ["shared", "own"])
def test_blank_idx_is_exactly_each_runs_masked_spans(result: dict[str, Any], rate: str,
                                                     run: str) -> None:
    c = result["calls"][f"{rate}_{run}"]
    assert c["error"] == "", c["error"]
    spans = SPANS_SHARED if run == "shared" else SPANS_ANT2
    keep = ["ANT1", "ANT3"] if run == "shared" else ["ANT2"]
    n_src = int(CALL_S * FS)
    n = n_src if rate == "full" else n_src // R
    fs = FS if rate == "full" else FS / R
    want = spans if rate == "full" else _map(spans, n)
    assert c["blank_idx"]["n_spans"] == len(want) and c["blank_idx"]["fs"] == pytest.approx(fs)
    assert c["rate"]["name"] == ("full" if rate == "full" else "decimated78")
    eb = int(round(EDGE_S * fs))
    for ch in keep:   # a shared call: each kept channel holds the SHARED spans
        e = c["kept"][ch]
        assert e["channel"] == ch and e["n"] == n and e["fs"] == pytest.approx(fs)
        assert _runs(e["blankIdx"]) == want
        assert _runs(e["invalid_runs"]) == want   # her invalidMask: exactly the spans
        edge = np.zeros(n, dtype=bool)
        for a, b in _runs(e["edge_runs"]):
            edge[a - 1:b] = True
        for a, b in want:   # her edge buffer on both sides of every masked span
            assert edge[max(0, a - 1 - eb):min(n, b + eb)].all(), (ch, a, b)
        lo, hi = FREE_S[run]
        assert not edge[int(lo * fs):int(hi * fs)].any()   # nor where nothing is masked
        if rate == "decimated":
            assert _runs(e["blankIdx_source"]) == spans and e["decimation_factor"] == R
            called = np.atleast_1d(e["peaks_called"]).astype(np.int64)
            assert called.size >= 1
            assert np.atleast_1d(e["peaks"]).tolist() == ((called - 1) * R + 1).tolist()
        else:
            assert "peaks_called" not in e


@pytest.mark.parametrize("rate", ["full", "decimated"])
def test_every_slow_wave_call_records_the_amplitude_caveat_and_known_properties(
        result: dict[str, Any], rate: str) -> None:
    """RULING 2026-10-09 (h) 2 and (f) 1, at both rates, in the call's provenance."""
    for run in ("shared", "own"):
        cav = result["calls"][f"{rate}_{run}"]["caveats"]
        assert "(h) 2" in cav["amplitude"] and "0.15 Hz-filtered wave" in cav["amplitude"]
        assert "whole-epoch detrend" in cav["amplitude"]
        assert "either sampling rate" in cav["amplitude"]
        assert "(h) 1" in cav["setting"]
        assert ("low-pass 0.15 Hz, order 2, 5 s smoothing, 15 s edge" in cav["setting"]
                and "0-9 cpm" in cav["setting"])
        if rate == "full":  # review 2026-10-10 fix 5: it said "decimated x78" at full rate too
            assert "full rate (no decimation)" in cav["setting"]
            assert "decimated" not in cav["setting"]
        else:
            assert "decimated x78 (stages 13 x 6)" in cav["setting"]
        props = cav["known_properties"]
        assert len(props) == 2 and all("(f) 1" in p and "both rates" in p for p in props)
        assert "findpeaks near-ties" in props[0] and "30 s clean-length gate" in props[1]


def test_spans_that_are_not_the_inputs_nan_are_refused(result: dict[str, Any]) -> None:
    assert result["calls"]["not_the_nan"] == "night6:blankIdx"


def test_the_full_and_decimated_peaks_agree_away_from_masks(result: dict[str, Any]) -> None:
    """The same slow waves, at both rates: peak times within 3 decimated samples."""
    full = np.atleast_1d(result["calls"]["full_own"]["kept"]["ANT2"]["peaks"]) - 1
    dec = np.atleast_1d(result["calls"]["decimated_own"]["kept"]["ANT2"]["peaks"]) - 1
    assert full.size == dec.size >= 2
    assert np.max(np.abs(full - dec)) <= 3 * R
    assert math.isfinite(float(full.mean()))


def _mround(x: float) -> int:
    return int(math.floor(x + 0.5))  # MATLAB round for x >= 0


def test_a_decimated_runs_file_is_trimmed_at_her_rate(result: dict[str, Any]) -> None:
    """Mode (B) on a decimated run's file: each value at her rows, the cut in epoch rows.

    The valid fractions are over HER rows at HER rate (fs / 78), and the peaks are cut in
    epoch rows - both computed here independently from her saved mask.
    """
    tr = result["trim"]
    assert tr["error"] == "", tr["error"]
    # review 2026-10-09: her rate is checked against the DECLARED rate, never inferred
    assert tr["no_rate"] == tr["full_rate"] == tr["forged_rate"] == "night6:slowWaveRate"
    assert tr["untouched"] is True    # refused before anything was written
    assert tr["n_files"] == 1 and tr["fs"] == pytest.approx(FS / R)
    fsd, n = tr["fs"], tr["n"]
    valid = np.ones(n, dtype=bool)
    for a, b in _runs(tr["invalid_runs"]):
        valid[a - 1:b] = False
    h = _mround(tr["rate_win"] * fsd / 2)
    want = []
    for tc in np.atleast_1d(tr["rate_t"]):
        c = _mround(float(tc) * fsd) + 1
        lo, hi = max(1, c - h), min(n, c + h)
        want.append(float(valid[lo - 1:hi].mean()))
    got = np.atleast_1d(tr["fraction"]).astype(float)
    assert got.size == len(want) and np.allclose(got, want, rtol=0, atol=1e-12)
    assert 0.0 < got.min() < 1.0 == got.max()   # the masked span shows, at her rate
    cuts = {c["cut"]: int(c["rows"]) for c in tr["cuts"]}
    lp = cuts["slow_wave.sw_peaks.filled_or_filtered"]
    assert lp == seconds_to_sample(TRIM_EL_S + rs.output_reach_s(
        "slow_wave", "sw_peaks", "filled_or_filtered", FS)[0], FS)
    before = np.atleast_1d(tr["peaks_before"]).astype(np.int64)
    after = np.atleast_1d(tr["peaks_after"]).astype(np.int64)
    assert after.tolist() == [p for p in before.tolist() if p - 1 >= lp]


def test_the_caveat_text_is_built_from_the_settings_and_rate_it_is_given(
        result: dict[str, Any]) -> None:
    """Review 2026-10-10 fix 5: nothing in the caveat is written out; W and R build it."""
    w = result["settings"]
    assert (w["lowPassOn"], w["lowPassCutoff"], w["lowPassOrder"], w["smoothWindow"],
            w["edgeBufferSec"]) == (True, 0.15, 2, 5, 15)  # RULING 2026-10-09 (h) 1
    alt = result["caveats_alt"]
    for rate in ("full", "decimated"):
        s = alt[rate]["setting"]
        assert "low-pass 0.2 Hz, order 4, 10 s smoothing, 3 s edge" in s and "0-12 cpm" in s
        assert "0.15" not in s and "0-9 cpm" not in s
        assert "0.2 Hz-filtered wave" in alt[rate]["amplitude"]
    assert "full rate (no decimation)" in alt["full"]["setting"]
    assert "decimated x78" in alt["decimated"]["setting"]
    off = alt["off"]
    assert "no low-pass" in off["setting"] and "cpm" not in off["setting"]
    assert "Hz-filtered" not in off["amplitude"]


def test_the_slow_wave_settings_have_one_construction_site() -> None:
    """Invariant 33: params().slow_wave, the call's W and the caveats share one struct."""
    rec = (NIGHT6 / "night6_run_recording.m").read_text(encoding="utf-8")
    assert "P.slow_wave = night6_slow_wave_settings();" in rec
    assert "night6_call_slow_wave(X, fs, P.slow_wave," in rec
    call = (NIGHT6 / "night6_call_slow_wave.m").read_text(encoding="utf-8")
    assert "S.caveats = night6_slow_wave_caveats(W, R);" in call
    for f in sorted(NIGHT6.glob("*.m")):
        code = "\n".join(line.split("%", 1)[0]
                         for line in f.read_text(encoding="utf-8").splitlines())
        literal = re.search(r"'lowPassCutoff',\s*[\d.]", code)
        assert (literal is not None) == (f.name == "night6_slow_wave_settings.m"), f.name
        assert "0-9 cpm" not in code and "decimated x78" not in code, f.name
