"""RULING 2026-10-12 (Andrea): slow-wave peaks near masked spans (PILOT).

(a) masked spans go to ``slowWaveAnalysis_new`` as NaN only (no ``blankIdx``, so no 15 s
buffer around them; tested in ``test_night6_slow_wave.py``); after detection
(``night6_slow_wave_span_rule``):

* (b) a peak inside a masked span or less than 5 s from its edge is ignored;
* (c) two consecutive peaks less than 7 s apart are BOTH dropped when both lie within 15 s
  of a masked span (either side); elsewhere close peaks are kept;
* (d) rate and timing use only the kept peaks - ``night6_slow_wave_rate_series`` is her rate
  loop on a given peak list, checked here against an independent implementation (and, in
  every real call, against her own output bit for bit: ``night6:rateReplica``).

Synthetic peaks at the production rate, every expectation written out by hand below.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_matlab_acceptance import _matlab

FS = 24414.0625
NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"
SPAN = (100.0, 110.0)
"""One masked span, seconds; its 5 s exclusion is (95, 115) and its 15 s zone [85, 125]."""


def _row(t_s: float) -> int:
    return int(round(t_s * FS))


def _spans(*spans: tuple[float, float]) -> list[list[int]]:
    return [[_row(a), _row(b)] for a, b in spans]


CASES: dict[str, dict[str, Any]] = {
    # name: peaks (s), spans (s), expected 1-based indices kept / (b) / (c)
    "inside": {"peaks": [105.0], "spans": [SPAN], "kept": [], "b": [1], "c": [], "inside": [1]},
    "before_4p9": {"peaks": [95.1], "spans": [SPAN], "kept": [], "b": [1], "c": []},
    "before_5p1": {"peaks": [94.9], "spans": [SPAN], "kept": [1], "b": [], "c": []},
    "after_4p9": {"peaks": [114.9], "spans": [SPAN], "kept": [], "b": [1], "c": []},
    "after_5p1": {"peaks": [115.1], "spans": [SPAN], "kept": [1], "b": [], "c": []},
    "pair_6p9_in_zone": {"peaks": [116.0, 122.9], "spans": [SPAN], "kept": [], "b": [],
                         "c": [1, 2]},
    "pair_7p1_in_zone": {"peaks": [116.0, 123.1], "spans": [SPAN], "kept": [1, 2], "b": [],
                         "c": []},
    "pair_6p9_before_span": {"peaks": [87.0, 93.9], "spans": [SPAN], "kept": [],
                             "b": [], "c": [1, 2]},
    "pair_6p9_outside_zone": {"peaks": [200.0, 206.9], "spans": [SPAN], "kept": [1, 2],
                              "b": [], "c": []},
    "pair_7p1_outside_zone": {"peaks": [200.0, 207.1], "spans": [SPAN], "kept": [1, 2],
                              "b": [], "c": []},
    "pair_one_in_one_out": {"peaks": [121.0, 127.9], "spans": [SPAN], "kept": [1, 2],
                            "b": [], "c": []},
    "pair_one_out_one_in_before": {"peaks": [80.0, 86.9], "spans": [SPAN], "kept": [1, 2],
                                   "b": [], "c": []},
    "pair_far_peak_14p9": {"peaks": [118.0, 124.9], "spans": [SPAN], "kept": [], "b": [],
                           "c": [1, 2]},
    "pair_far_peak_15p1": {"peaks": [118.2, 125.1], "spans": [SPAN], "kept": [1, 2], "b": [],
                           "c": []},
    "no_spans": {"peaks": [10.0, 16.5], "spans": [], "kept": [1, 2], "b": [], "c": []},
    "chain_of_three": {"peaks": [116.0, 122.5, 129.0], "spans": [SPAN], "kept": [3], "b": [],
                       "c": [1, 2]},
    "b_then_pair_partner": {"peaks": [114.5, 121.0], "spans": [SPAN], "kept": [], "b": [1],
                            "c": [2]},
    "two_spans_union_zone": {"peaks": [118.0, 124.0], "spans": [SPAN, (130.0, 140.0)],
                             "kept": [], "b": [], "c": [1, 2]},
}
"""The 15 s zone is measured from the span's edges: 87.0 is 13 s before 100, 93.9 is 6.1 s.
chain_of_three: 129 is 19 s from the span, so (122.5, 129) is a pair with one peak out.
b_then_pair_partner: 114.5 is 4.5 s after the span ((b)); 121 is its consecutive partner in
her list, 6.5 s on, both in the zone, so (c) drops 121 as well (the literal reading); judged
on (b)'s survivors 121 would have no partner (pairs_after_b, reported only).
two_spans_union_zone: 118 is 8 s after span 1, 124 is 6 s before span 2.
pair_far_peak_14p9 / 15p1: the zone's own edge - the later peak 14.9 s (in) or 15.1 s (out)."""


def _rate_case() -> dict[str, Any]:
    fs = 10.0
    n = 1500  # 150 s
    peaks_s = [5, 15, 25, 35, 45, 55, 75, 95, 115, 125, 135, 145]  # windows hold 2 to 5
    locs = [1 + int(round(t * fs)) for t in peaks_s]
    return {"fs": fs, "n": n, "locs": locs,
            "invalid": [[701, 720]],      # 70.0-71.9 s masked (NaN)
            "unclean": [[501, 600]]}      # 50.0-59.9 s not clean (an edge zone)


def _independent_rate(c: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Her rule written from its documentation (slowWaveAnalysis_new.m header), not copied.

    Returns the rate, the centre times and the peak count of every window that passed the
    30 s gate (-1 elsewhere).
    """
    fs, n = c["fs"], c["n"]
    inv = np.zeros(n, dtype=bool)
    for a, b in c["invalid"]:
        inv[a - 1:b] = True
    clean = ~inv
    for a, b in c["unclean"]:
        clean[a - 1:b] = False
    locs = np.asarray(c["locs"])
    centres = np.arange(1, n + 1, int(round(fs)))
    t = (centres - 1) / fs
    half = int(round(60 * fs / 2))
    out = np.full(centres.size, np.nan)
    count = np.full(centres.size, -1)
    for i, (ctr, tc) in enumerate(zip(centres, t, strict=True)):
        if tc < 30 or tc > (n - 1) / fs - 30 or inv[ctr - 1]:
            continue
        lo, hi = max(1, ctr - half), min(n, ctr + half)
        rows = np.arange(lo, hi + 1)[clean[lo - 1:hi]]
        if rows.size < int(round(30 * fs)):
            continue
        k = int(np.isin(locs, rows).sum())
        count[i] = k
        if k >= 3:
            out[i] = k / (rows.size / fs / 60)
    return out, t, count


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    matlab = _matlab()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    tmp = tmp_path_factory.mktemp("sw_span_rule")
    cases = {k: {"peaks": [_row(t) for t in v["peaks"]], "spans": _spans(*v["spans"])}
             for k, v in CASES.items()}
    case = {"night6": NIGHT6.as_posix(), "fs": FS, "cases": cases, "rate": _rate_case()}
    case_file, res_file = tmp / "case.json", tmp / "result.json"
    case_file.write_text(json.dumps(case), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{HARNESS.as_posix()}'); "
           f"check_slow_wave_span_rule('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return dict(json.loads(res_file.read_text(encoding="utf-8")))


def _ints(x: object) -> list[int]:
    """Read an index list: MATLAB jsonencode writes one element as a scalar, none as []."""
    if isinstance(x, list) and not x:
        return []
    return [int(v) for v in np.atleast_1d(np.asarray(x, dtype=float)).ravel()]


def test_the_declared_rule_is_5_15_7() -> None:
    text = (NIGHT6 / "night6_slow_wave_settings.m").read_text(encoding="utf-8")
    assert "'exclusionSec', 5, 'pairZoneSec', 15, 'pairMinSec', 7" in text
    assert "NOT covered by the harm test" in text


def test_the_rule_struct_the_harness_used(result: dict[str, Any]) -> None:
    r = result["rule"]
    assert (r["exclusionSec"], r["pairZoneSec"], r["pairMinSec"]) == (5, 15, 7)
    assert r["name"] == "sw_span_rule_20261012"


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_peak_is_kept_or_dropped_by_the_rule_that_applies(result: dict[str, Any],
                                                               name: str) -> None:
    want, got = CASES[name], result["cases"][name]
    assert _ints(got["kept"]) == want["kept"], name
    assert _ints(got["b"]) == want["b"], name
    assert _ints(got["c"]) == want["c"], name
    if "inside" in want:
        assert _ints(got["inside"]) == want["inside"]
    n = len(want["peaks"])
    assert got["n"] == n and got["n_kept"] + got["n_b"] + got["n_c"] == n


def test_a_pair_is_dropped_whole_never_one_peak(result: dict[str, Any]) -> None:
    for name in ("pair_6p9_in_zone", "pair_6p9_before_span", "two_spans_union_zone"):
        p = np.asarray(result["cases"][name]["pairs"], dtype=float).reshape(-1, 2)
        assert p.tolist() == [[1.0, 2.0]], name


def test_the_other_reading_of_consecutive_is_reported_not_applied(
        result: dict[str, Any]) -> None:
    """b_then_pair_partner: (c) on her list drops 121; on (b)'s survivors there is no pair."""
    got = result["cases"]["b_then_pair_partner"]
    assert np.asarray(got["pairs"], dtype=float).reshape(-1, 2).tolist() == [[1.0, 2.0]]
    assert np.asarray(got["pairs_after_b"], dtype=float).size == 0


def test_a_pair_across_two_spans_names_both_spans(result: dict[str, Any]) -> None:
    got = result["cases"]["two_spans_union_zone"]
    assert np.asarray(got["pairs_spans"], dtype=float).reshape(-1, 2).tolist() == [[1.0, 2.0]]


def test_distances_are_measured_from_the_span_edges(result: dict[str, Any]) -> None:
    for name, want in (("before_4p9", 4.9), ("after_5p1", 5.1), ("inside", 0.0)):
        assert float(np.atleast_1d(result["cases"][name]["dist"])[0]) == pytest.approx(
            want, abs=2 / FS), name
    # no span: the distance is Inf, which MATLAB's jsonencode writes as null
    assert all(v is None for v in result["cases"]["no_spans"]["dist"])


def test_the_rate_from_a_given_peak_list_is_her_rule(result: dict[str, Any]) -> None:
    want, t, count = _independent_rate(_rate_case())
    assert {2, 3, 4} <= set(count.tolist())   # the >= 3 gate is exercised at its edge
    assert np.isnan(want[count == 2]).all() and not np.isnan(want[count == 3]).any()
    got = np.asarray([np.nan if v is None else v for v in result["rate"]["rate"]], dtype=float)
    assert np.allclose(np.asarray(result["rate"]["time"], dtype=float), t, rtol=0, atol=1e-12)
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(want)
    assert ok.sum() > 10 and np.allclose(got[ok], want[ok], rtol=0, atol=1e-9)
    assert np.isnan(got[70]) and not np.isnan(got[40])   # invalid centre / clean centre


def test_bad_inputs_are_refused_by_name(result: dict[str, Any]) -> None:
    assert result["refused"] == {"unsorted": "night6:spanRule", "reversed": "night6:spanRule",
                                 "no_rule": "night6:spanRule"}
