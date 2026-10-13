"""RULING 2026-10-12 (Andrea), v2 (decided after pilot v1): slow-wave peaks near masked spans.

(a) masked spans go to ``slowWaveAnalysis_new`` as NaN only (no ``blankIdx``, so no 15 s
buffer around them; tested in ``test_night6_slow_wave.py``); after detection
(``night6_slow_wave_span_rule``):

* (b) a peak inside a masked span or less than 10 s from its edge is ignored, and the time
  in those 10 s zones is left out of the clean time her rate pools
  (``night6_slow_wave_zone_mask``, ``night6_slow_wave_apply_rule``);
* (c) two consecutive DETECTED peaks (her list, before (b)) less than 7 s apart are BOTH
  dropped when both lie within 15 s of a masked span - the union of every span's zones,
  either side; elsewhere close peaks are kept;
* (d) rate and timing use only the kept peaks - ``night6_slow_wave_rate_series`` is her rate
  loop on a given peak list, checked here against an independent implementation (and, in
  every real call, against her own output bit for bit: ``night6:rateReplica``).

Also (Andrea 2026-10-12, go): every slow-wave result carries "slow wave is C1 at floor 7.42 and
NOT covered by the harm test", and a DIAGNOSTIC (``night6_slow_wave_pre_span_peaks``) lists
the kept peaks within 25 s before a span start whose amplitude is more than 2 x the channel's
median kept-peak amplitude - with no effect on any rate (bit-identical with it off).

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
SPAN2 = (140.0, 150.0)
"""One masked span, seconds; its 10 s exclusion is (90, 120) and its 15 s zone [85, 125]."""


def _row(t_s: float) -> int:
    return int(round(t_s * FS))


def _spans(*spans: tuple[float, float]) -> list[list[int]]:
    return [[_row(a), _row(b)] for a, b in spans]


CASES: dict[str, dict[str, Any]] = {
    # name: peaks (s), spans (s), expected 1-based indices kept / (b) / (c)
    "inside": {"peaks": [105.0], "spans": [SPAN], "kept": [], "b": [1], "c": [], "inside": [1]},
    "before_9p9": {"peaks": [90.1], "spans": [SPAN], "kept": [], "b": [1], "c": []},
    "before_10p1": {"peaks": [89.9], "spans": [SPAN], "kept": [1], "b": [], "c": []},
    "after_9p9": {"peaks": [119.9], "spans": [SPAN], "kept": [], "b": [1], "c": []},
    "after_10p1": {"peaks": [120.1], "spans": [SPAN], "kept": [1], "b": [], "c": []},
    "pair_6p9_b_then_zone": {"peaks": [114.0, 120.9], "spans": [SPAN], "kept": [], "b": [1],
                             "c": [2]},
    "pair_7p1_b_then_zone": {"peaks": [114.0, 121.1], "spans": [SPAN], "kept": [2], "b": [1],
                             "c": []},
    "pair_6p9_before_span": {"peaks": [85.5, 92.4], "spans": [SPAN], "kept": [], "b": [2],
                             "c": [1]},
    "pair_both_c_only_two_spans": {"peaks": [121.0, 127.9], "spans": [SPAN, SPAN2],
                                   "kept": [], "b": [], "c": [1, 2]},
    "pair_7p1_two_spans": {"peaks": [121.0, 128.1], "spans": [SPAN, SPAN2], "kept": [1, 2],
                           "b": [], "c": []},
    "pair_6p9_outside_zone": {"peaks": [200.0, 206.9], "spans": [SPAN], "kept": [1, 2],
                              "b": [], "c": []},
    "pair_7p1_outside_zone": {"peaks": [200.0, 207.1], "spans": [SPAN], "kept": [1, 2],
                              "b": [], "c": []},
    "pair_one_in_one_out": {"peaks": [124.0, 130.9], "spans": [SPAN], "kept": [1, 2],
                            "b": [], "c": []},
    "pair_one_out_one_in_before": {"peaks": [79.0, 85.9], "spans": [SPAN], "kept": [1, 2],
                                   "b": [], "c": []},
    "pair_far_peak_14p9": {"peaks": [118.0, 124.9], "spans": [SPAN], "kept": [], "b": [1],
                           "c": [2]},
    "pair_far_peak_15p1": {"peaks": [118.2, 125.1], "spans": [SPAN], "kept": [2], "b": [1],
                           "c": []},
    "no_spans": {"peaks": [10.0, 16.5], "spans": [], "kept": [1, 2], "b": [], "c": []},
    "chain_of_three": {"peaks": [117.0, 123.5, 130.0], "spans": [SPAN], "kept": [3], "b": [1],
                       "c": [2]},
}
"""Distances are measured from the span's edges. pair_6p9_b_then_zone: 114 is 4 s after the
span ((b)); 120.9 (10.9 s, in the zone) is its consecutive partner in her DETECTED list, 6.9 s
on, so (c) drops it too (Andrea 2026-10-12: (c) before (b)); judged on (b)'s survivors 120.9
would have no partner (pairs_after_b, reported only). pair_6p9_before_span: 85.5 is 14.5 s
before the span (zone only), 92.4 is 7.6 s ((b)). pair_both_c_only_two_spans: 121 is 11 s
after span 1 and 127.9 is 12.1 s before span 2 - neither within 10 s of a span, both in the
union of 15 s zones (Andrea 2026-10-12: a pair near two different spans counts).
chain_of_three: 117 is (b), 123.5 (13.5 s) its partner 6.5 s on, 130 (20 s) out of the zone,
so (123.5, 130) is no pair. pair_far_peak_14p9 / 15p1: the zone's own edge - the later peak
14.9 s (in) or 15.1 s (out)."""


ZONE_ROWS_S = (89.9, 90.1, 100.0, 105.0, 110.0, 119.9, 120.1)
"""Epoch times (s) around SPAN: in the < 10 s zone (or inside) iff 90 < t < 120."""
ZONE_WANT = [False, True, True, True, True, True, False]

APPLY_FACTORS = {"full": 1, "decimated": 78}
"""Apply-rule cases: her file at 10 Hz; the epoch at 10 Hz (full) or 780 Hz (decimated by 78)."""
APPLY_PEAKS_S = (25.0, 35.0, 45.0, 54.0, 62.0, 76.0, 82.5, 92.0, 102.0, 112.0, 122.0)
"""Called-rate peak times; the masked span is called rows 701-720 (70.0-71.9 s). 62 is 8 s
before it and 76 is 4.1 s after it ((b)); 82.5 is 10.6 s after it and 6.5 s after 76 ((c));
54 is 16 s before it and 7.9 s before 62 (kept); 92 is 20.1 s after it (kept)."""
APPLY_KEPT_S = (25.0, 35.0, 45.0, 54.0, 92.0, 102.0, 112.0, 122.0)
APPLY_AMP_UV = {35.0: 30.0, 45.0: 25.0, 54.0: 30.0}
"""Her trace at the peaks (uV; 10 elsewhere): the kept median is 10, the threshold 20. 45 s is
exactly 25.0 s before the span (listed: <= 25 s), 54 s is 16 s before it, 35 s is 35 s before
it (not listed)."""

PRESPAN_FS = 10.0
PRESPAN_CASES: dict[str, dict[str, Any]] = {
    # rows at 10 Hz (row = 10 t + 1), amplitudes (uV), spans (s); want rows / spans listed
    "equal_to_2x_is_not_listed": {"t": [50.0, 60.0, 90.0], "amp": [10.0, 10.0, 20.0],
                                  "spans": [(100.0, 110.0)], "want": [], "want_span": []},
    "window_25p1_is_not_listed": {"t": [50.0, 60.0, 74.9], "amp": [10.0, 10.0, 50.0],
                                  "spans": [(100.0, 110.0)], "want": [], "want_span": []},
    "window_24p9_is_listed": {"t": [50.0, 60.0, 75.1], "amp": [10.0, 10.0, 50.0],
                              "spans": [(100.0, 110.0)], "want": [75.1], "want_span": [1]},
    "the_next_span_counts": {"t": [40.0, 50.0, 60.0, 112.0, 180.0],
                             "amp": [10.0, 10.0, 10.0, 50.0, 50.0],
                             "spans": [(100.0, 110.0), (200.0, 210.0)], "want": [180.0],
                             "want_span": [2]},
    "after_the_last_span": {"t": [50.0, 60.0, 215.0], "amp": [10.0, 10.0, 50.0],
                            "spans": [(100.0, 110.0)], "want": [], "want_span": []},
    "spans_unsorted": {"t": [10.0, 20.0, 90.0], "amp": [10.0, 10.0, 50.0],
                       "spans": [(200.0, 210.0), (100.0, 110.0)], "want": [90.0],
                       "want_span": [2]},
    "median_not_positive": {"t": [50.0, 60.0, 90.0], "amp": [-10.0, -10.0, 5.0],
                            "spans": [(100.0, 110.0)], "want": [], "want_span": []},
}
"""window_25p1: 74.9 s is 25.1 s before 100 s; window_24p9: 75.1 s is 24.9 s before it.
the_next_span_counts: 112 s (after span 1) is 88 s before span 2 - not listed; 180 s is 20 s
before span 2. median_not_positive: the threshold is not defined, nothing is listed."""


def _tol(sw: dict[str, Any] | None, floor: float | None) -> dict[str, Any]:
    tol: dict[str, Any] = {}
    if sw is not None:
        tol["extent_rule"] = {"consumers": {"slow_wave": sw, "spikes": {"candidate": "C1"}}}
    if floor is not None:
        tol["cross_band_floor"] = {"0-2": floor, "300-3000": 4.35}
    return {"thresholds": {"tolerances": tol}, "extra": {}}


EXTENT_CASES: dict[str, dict[str, Any]] = {
    "declared_c1": _tol({"candidate": "C1", "gate": "i_or_ii"}, 7.42),
    "declared_c2": _tol({"candidate": "C2", "gate": "ii"}, 7.42),
    "undeclared": _tol({"declared": False, "reason": "x"}, 7.42),
    "undeclared_pilot": {**_tol({"declared": False, "reason": "x"}, 7.42),
                         "extra": {"SW_PILOT": {"what": "pilot"}}},
    "floor_7": _tol({"candidate": "C1"}, 7.0),
    "floor_7p4249": _tol({"candidate": "C1"}, 7.4249),
    "nothing_recorded": {"thresholds": {"source": "synthetic"}},
}
"""Mask provenances (the shape emit writes: thresholds.tolerances.extent_rule and
cross_band_floor, whose "0-2" key jsondecode renames). The caveat "slow wave is C1 at floor
7.42" is refused where the mask says otherwise and recorded unchecked where it says nothing."""


def _prespan_case(c: dict[str, Any]) -> dict[str, Any]:
    return {"rows": [int(round(s * PRESPAN_FS)) + 1 for s in c["t"]],
            "amps": [a * 1e-6 for a in c["amp"]],
            "spans": [[int(round(a * PRESPAN_FS)) + 1, int(round(b * PRESPAN_FS)) + 1]
                      for a, b in c["spans"]], "fs": PRESPAN_FS}


def _apply_case(factor: int) -> dict[str, Any]:
    n, fsc = 1500, 10.0
    called = [1 + int(round(s * fsc)) for s in APPLY_PEAKS_S]
    span_called = (701, 720)
    spans = [[(span_called[0] - 1) * factor + 1, span_called[1] * factor]]
    amps = [APPLY_AMP_UV.get(s, 10.0) * 1e-6 for s in APPLY_PEAKS_S]
    return {"n": n, "fs": fsc, "fs_epoch": fsc * factor, "factor": factor, "called": called,
            "amps": amps,
            "invalid": [list(span_called)], "edge": [[1, 150], [n - 149, n]], "spans": spans}


def _zone_rows(c: dict[str, Any]) -> np.ndarray:
    """Return the zone mask of the called rows, computed independently.

    Called row j is epoch row (j - 1) factor + 1; it is in the zone iff its distance to the
    span (0 inside) is < 10 s at the epoch rate.
    """
    (a, b), = c["spans"]
    r = (np.arange(c["n"]) * c["factor"] + 1).astype(float)
    d = np.where(r < a, (a - r) / c["fs_epoch"], np.where(r > b, (r - b) / c["fs_epoch"], 0.0))
    return d < 10.0


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
    case = {"night6": NIGHT6.as_posix(), "fs": FS, "cases": cases, "rate": _rate_case(),
            "zone": {"rows": [_row(s) for s in ZONE_ROWS_S], "spans": _spans(SPAN), "fs": FS},
            "apply": {k: _apply_case(f) for k, f in APPLY_FACTORS.items()},
            "prespan": {k: _prespan_case(v) for k, v in PRESPAN_CASES.items()},
            "extent": {k: json.dumps(v, sort_keys=True) for k, v in EXTENT_CASES.items()},
            "work": (tmp / "work").as_posix()}
    (tmp / "work").mkdir()
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


def test_the_declared_rule_is_10_15_7_with_zone_time_out_of_clean_time() -> None:
    text = (NIGHT6 / "night6_slow_wave_settings.m").read_text(encoding="utf-8")
    assert "'exclusionSec', 10, 'pairZoneSec', 15, 'pairMinSec', 7" in text
    assert "'exclusionFromCleanTime', true" in text
    assert "NOT covered by the harm test" in text


def test_the_rule_struct_the_harness_used(result: dict[str, Any]) -> None:
    r = result["rule"]
    assert (r["exclusionSec"], r["pairZoneSec"], r["pairMinSec"]) == (10, 15, 7)
    assert r["exclusionFromCleanTime"] is True
    assert r["name"] == "sw_span_rule_20261012_v2"


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
    for name in ("pair_6p9_b_then_zone", "pair_6p9_before_span", "pair_both_c_only_two_spans"):
        p = np.asarray(result["cases"][name]["pairs"], dtype=float).reshape(-1, 2)
        assert p.tolist() == [[1.0, 2.0]], name


def test_the_other_reading_of_consecutive_is_reported_not_applied(
        result: dict[str, Any]) -> None:
    """(c) on her detected list drops 120.9; on (b)'s survivors there is no pair."""
    got = result["cases"]["pair_6p9_b_then_zone"]
    assert np.asarray(got["pairs"], dtype=float).reshape(-1, 2).tolist() == [[1.0, 2.0]]
    assert np.asarray(got["pairs_after_b"], dtype=float).size == 0


def test_a_pair_across_two_spans_names_both_spans(result: dict[str, Any]) -> None:
    got = result["cases"]["pair_both_c_only_two_spans"]
    assert np.asarray(got["pairs_spans"], dtype=float).reshape(-1, 2).tolist() == [[1.0, 2.0]]


def test_distances_are_measured_from_the_span_edges(result: dict[str, Any]) -> None:
    for name, want in (("before_9p9", 9.9), ("after_10p1", 10.1), ("inside", 0.0)):
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


def test_the_zone_mask_is_the_10_s_distance(result: dict[str, Any]) -> None:
    got = [bool(v) for v in np.atleast_1d(result["zone"]["mask"])]
    assert got == ZONE_WANT


@pytest.mark.parametrize("variant", sorted(APPLY_FACTORS))
def test_the_rule_rate_leaves_the_10_s_zones_out_of_clean_time(result: dict[str, Any],
                                                               variant: str) -> None:
    """Rule (b) + (d) on a file of hers: the kept peaks and the rate.

    Her rate over clean = ~invalid & ~edge & ~zone, computed here independently; the v1
    reading (zone counted as clean) differs.
    """
    c, got = _apply_case(APPLY_FACTORS[variant]), result["apply"][variant]
    assert got["error"] == "", got["error"]
    kept_called = [1 + int(round(s * c["fs"])) for s in APPLY_KEPT_S]
    assert _ints(got["kept_called"]) == kept_called
    assert _ints(got["kept_epoch"]) == [(k - 1) * c["factor"] + 1 for k in kept_called]
    assert (got["n_detected"], got["n_dropped_b"], got["n_dropped_c"]) == (11, 2, 1)
    zone = _zone_rows(c)
    runs = np.flatnonzero(np.diff(np.concatenate(([0], zone.astype(int), [0]))))
    unclean = c["edge"] + [[int(a) + 1, int(b)] for a, b in runs.reshape(-1, 2)]
    want, _, _ = _independent_rate({"fs": c["fs"], "n": c["n"], "locs": kept_called,
                                    "invalid": c["invalid"], "unclean": unclean})
    rate = np.asarray([np.nan if v is None else v for v in got["rate"]], dtype=float)
    assert np.array_equal(np.isnan(rate), np.isnan(want))
    ok = ~np.isnan(want)
    assert ok.sum() > 10 and np.allclose(rate[ok], want[ok], rtol=0, atol=1e-9)
    counted, _, _ = _independent_rate({"fs": c["fs"], "n": c["n"], "locs": kept_called,
                                       "invalid": c["invalid"], "unclean": c["edge"]})
    assert not np.allclose(np.nan_to_num(counted), np.nan_to_num(want))  # the zone matters
    assert float(got["avg"]) == pytest.approx(float(np.nanmean(want)), abs=1e-9)


@pytest.mark.parametrize("name", sorted(PRESPAN_CASES))
def test_the_pre_span_diagnostic_lists_exactly_the_large_peaks_before_a_span(
        result: dict[str, Any], name: str) -> None:
    c, got = PRESPAN_CASES[name], result["prespan"][name]
    want_rows = [int(round(s * PRESPAN_FS)) + 1 for s in c["want"]]
    assert got["n"] == len(want_rows)
    assert _ints(got["row"]) == want_rows
    assert _ints(got["span"]) == c["want_span"]
    assert got["defined"] is (name != "median_not_positive")


def test_the_pre_span_diagnostic_refuses_mismatched_inputs(result: dict[str, Any]) -> None:
    assert result["prespan_refused"] == "night6:preSpan"


@pytest.mark.parametrize("variant", sorted(APPLY_FACTORS))
def test_the_diagnostic_in_the_output_file_and_the_record(result: dict[str, Any],
                                                          variant: str) -> None:
    got = result["apply"][variant]
    assert got["error"] == "", got["error"]
    pre = got["pre"]
    assert pre["n"] == 2 and got["record_pre_n"] == 2
    assert np.allclose(np.asarray(pre["t_epoch_s"], dtype=float), [45.0, 54.0], atol=1e-9)
    assert np.allclose(np.asarray(pre["amplitude_uV"], dtype=float), [25.0, 30.0], atol=1e-9)
    assert np.allclose(np.asarray(pre["dist_s"], dtype=float), [25.0, 16.0], atol=1e-9)
    assert float(pre["median"]) == pytest.approx(10e-6, abs=1e-15)


@pytest.mark.parametrize("variant", sorted(APPLY_FACTORS))
def test_rates_are_bit_identical_with_and_without_the_diagnostic(result: dict[str, Any],
                                                                 variant: str) -> None:
    got = result["apply"][variant]
    assert got["error"] == "", got["error"]
    assert got["no_diag_field"] is True
    assert got["identical_without_diagnostic"] is True   # every variable, bit for bit
    assert got["rate_bits_equal"] is True


CAVEAT = "slow wave is C1 at floor 7.42 and NOT covered by the harm test"


@pytest.mark.parametrize("variant", sorted(APPLY_FACTORS))
def test_every_slow_wave_result_says_c1_floor_742_not_harm_tested(result: dict[str, Any],
                                                                   variant: str) -> None:
    got = result["apply"][variant]
    assert CAVEAT in got["caveat"] and CAVEAT in got["record_caveat"]


@pytest.mark.parametrize(("name", "error"), [
    ("declared_c1", ""), ("declared_c2", "night6:slowWaveExtent"),
    ("undeclared", "night6:slowWaveExtent"), ("undeclared_pilot", ""),
    ("floor_7", "night6:slowWaveExtent"), ("floor_7p4249", ""), ("nothing_recorded", "")])
def test_the_caveat_is_checked_against_the_mask_provenance(result: dict[str, Any], name: str,
                                                           error: str) -> None:
    got = result["extent"][name]
    assert got["error_id"] == error
    if error:
        return
    assert got["statement"].startswith(CAVEAT)
    if name == "nothing_recorded":
        assert got["candidate_checked"].startswith("unchecked")
        assert got["floor_checked"].startswith("unchecked")
    elif name == "undeclared_pilot":
        assert got["candidate_checked"].startswith("pilot mask")
    else:
        assert got["candidate_checked"].endswith("slow_wave candidate C1")
        assert got["floor_checked"].startswith("mask provenance cross_band_floor 0-2")
