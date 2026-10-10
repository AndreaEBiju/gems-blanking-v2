"""RULING 2026-10-09 (i) 1: the tolerance -> threshold mapping, option (1), cross-band.

* threshold selection (:func:`tolerance_map.option1_threshold`): the smallest harmful kind
  VISIBLE in the consumer's own band, floored at z 1.44; 1.44 where none is visible;
  invisible kinds never set it, however low; the visibility verdict is tolmap's rule
  (:func:`tolerance_map.assess_window`, lift >= 1 z and above background);
* the inputs belong together or are refused (criterion v2 hash, the mapping made from the
  declared tolerances, gate rows only, one chain generation, every z consumer mapped), and
  the mmc rpeakUnits flag is carried;
* residual risk: harmful rows the detector cannot see are recorded per consumer, split into
  "no band, no detector" and "lifted, not detected";
* cross-band: a confirmed core whose own band stays under the threshold blanks the consumer
  over the event's DETECTED extent (the detecting pairs' runs above 1.44 overlapping the
  core, padded by the consumer's settling), and only then;
* routing cannot undo a detected extent: ``spans_from_routing`` refuses any decision but
  ``cross_band_decision`` for it - including the "correct" that ``route_event`` gives when
  the own band is under its threshold;
* review 7 finding 2: the cross-band scope is DECLARED per z consumer, with no default -
  ``all_detecting_pairs`` (the ruling), ``own_signals_only`` (detecting pairs on the
  consumer's own signals or their channels), ``off`` (none; its detected-but-invisible rows
  listed as residual risk) - and recorded in the table's record;
* review 7 finding 4: the visibility scan must record the declared tolerances as its input;
* RULING 2026-10-10 item 2: the consumers on the floor take the DECLARED clean-null percentile
  of their own band (required, no default); breathing keeps its visible-kind threshold; the
  percentiles are the archived clean null's; a cross-band pair counts while above its own
  band's percentile.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.extent import routing as rt
from gems_blanking_v2.extent import tolerance as tl
from gems_blanking_v2.extent import tolerance_map as tm
from gems_blanking_v2.types import Candidate, Event

from tests.conftest import make_band_z

FS = 24414.0625
GEN = "0133349b3ebeff80"
TOL_SHA = "a" * 64
CRIT_SHA = "c" * 64


def _kind(z: float, visible: bool | None) -> dict[str, Any]:
    frm: dict[str, Any] = {"seed": 0, "column": "RVN"}
    if visible is not None:
        frm["above_background"] = visible
    return {"z": z, "row": {"host_tag": "h", "dur_s": 0.5, "chan_set": "nerve",
                            "sensitive_end_sigma": 1.0, "mapped_z_from": frm}}


# ---------------------------------------------------------------- threshold selection

def test_the_smallest_visible_kind_sets_the_threshold() -> None:
    t = tm.option1_threshold("breathing", {"drift": _kind(7.69, True), "tribo": _kind(9.1, True),
                                           "clip": _kind(2.0, False)})
    assert t.z == 7.69 and t.setter == "drift"
    assert t.visible_kinds == ("drift", "tribo")  # clip, invisible, never sets it


def test_a_visible_kind_below_the_floor_is_floored_at_1_44() -> None:
    t = tm.option1_threshold("mmc", {"step": _kind(0.9, True), "tribo": _kind(5.0, True)})
    assert t.z == tm.Z_FLOOR == 1.44
    assert t.setter.startswith("floor 1.44") and "step" in t.setter
    at = tm.option1_threshold("mmc", {"step": _kind(1.44, True)})
    assert at.z == 1.44 and at.setter == "step"  # exactly at the floor: the kind sets it


@pytest.mark.parametrize("kinds", [{}, {"tribo": _kind(-0.92, False)},
                                   {"drift": _kind(0.5, None)}])  # absent verdict: invisible
def test_with_no_visible_kind_the_threshold_is_1_44(kinds: dict[str, Any]) -> None:
    t = tm.option1_threshold("spikes", kinds)
    assert t.z == 1.44 and "no harmful kind visible" in t.setter and t.visible_kinds == ()


def test_a_visible_kind_without_a_finite_z_is_refused() -> None:
    with pytest.raises(ValueError, match="visible in the band"):
        tm.option1_threshold("mmc", {"step": _kind(math.nan, True)})


def test_visibility_is_the_mappings_lift_rule() -> None:
    base = np.array([0.2, 0.3, 0.1, 0.4])
    lifted = tm.assess_window(np.array([0.5, 4.0, 0.2, 0.5]), base,
                              base + np.array([0.0, 1.0, 0.99, 0.0]))
    assert lifted["raised"] and lifted["n_lifted_frames"] == 1  # 1.0 lifts, 0.99 does not
    assert lifted["z_peak"] == 4.0 and lifted["above_background"] is True
    low = tm.assess_window(np.array([0.5, 1.43, 0.2, 0.5]), base, base + np.array([0, 2.0, 0, 0]))
    assert low["raised"] and low["above_background"] is False  # below 1.44
    near = tm.assess_window(np.array([0.5, 1.9, 0.2, 0.5]), base + np.array([0, 0, 0, 1.0]),
                            base + np.array([0, 2.0, 0, 1.0]))
    assert near["above_background"] is False  # within 1 z of the window's baseline peak
    assert tm.assess_window(np.array([0.1]), np.array([0.1]), np.array([0.5]))["raised"] is False
    assert tm.assess_window(*(np.array([np.nan]),) * 3) == {"assessable": False}


# ---------------------------------------------------------------- the declared inputs

def _mapping() -> dict[str, Any]:
    return {"input": {"sha256": TOL_SHA}, "rows_rule": "gate", "flags": [],
            "generation_sha256_16": GEN, "refused": {"spikes": "mapped z -0.92"},
            "consumers": {"breathing": {"kinds": {"drift": _kind(7.69, True)},
                                        "kinds_not_raised": ["clip", "tribo"]},
                          "mmc": {"kinds": {"step": _kind(-1.49, False)},
                                  "kinds_not_raised": ["drift"]},
                          "slow_wave": {"kinds": {"drift": _kind(-1.85, False)}},
                          "spikes": {"kinds": {"tribo": _kind(-0.92, False)}}}}


def _scan_row(consumer: str, kind: str, sigma: float, detectable: list[str],
              lifted: list[str]) -> dict[str, Any]:
    return {"consumer": consumer, "kind": kind, "tolerance_sigma": sigma, "host_tag": "host1",
            "dur_s": 5.0, "chan_set": "stomach", "status": "measured", "censored": False,
            "pairs_detectable": detectable, "pairs_lifted": lifted}


def _scan() -> dict[str, Any]:
    return {"generation_sha256_16": GEN, "input": {"name": "t.json", "sha256": TOL_SHA},
            "rows": [
        _scan_row("mmc", "drift", 0.0884, [], []),
        _scan_row("mmc", "tribo", 0.5, [], ["ANT1|10-150"]),
        _scan_row("mmc", "clip", 1.41, ["ANT3|300-3000"], ["ANT3|300-3000"]),
        _scan_row("slow_wave", "step", 0.125, [], [])]}


def _tolerances() -> dict[str, Any]:
    return {"criterion_v2": {"sha256": CRIT_SHA, "file": "tolv2/criterion.md"},
            "mmc_provenance_flag": {"flag": "measured with cardiac blanking off (rpeakUnits "
                                            "defect)", "applies_to": ["mmc", "mmc_burst"]}}


ALL = dict.fromkeys(("breathing", "mmc", "slow_wave", "spikes"), tl.CROSS_BAND_SCOPE_ALL)


def _build(**over: Any) -> tm.Option1Tolerances:  # noqa: ANN401 - keyword overrides
    args: dict[str, Any] = {"tolerances": _tolerances(), "mapping": _mapping(), "scan": _scan(),
                            "tolerances_sha256": TOL_SHA, "criterion_sha256": CRIT_SHA,
                            "mapping_sha256": "m" * 64, "scan_sha256": "s" * 64,
                            "generation": GEN, "cross_band_scope": ALL,
                            "extent_percentile": "p90",
                            "core_blanking": {**dict.fromkeys(
                                ("breathing", "mmc", "slow_wave", "spikes"), "always"),
                                "hrv": "off"},
                            "extent_floors": tm.EXTENT_FLOORS_PER_BAND}
    args.update(over)
    return tm.option1_tolerances(**args)


def test_the_option1_table_from_the_declared_inputs() -> None:
    t = _build()
    assert dict(t.table.z_tol) == {"breathing": 7.69, "mmc": 1.44, "slow_wave": 1.44,
                                   "spikes": 1.44}
    rec = t.record()
    assert rec["version"] == "v2" and rec["floor_z"] == 1.44
    assert rec["cross_band_scope"] == ALL
    assert "rpeakUnits" in rec["flags"]["mmc_provenance_flag"]["flag"]
    assert rec["inputs"]["mapping_refused"] == {"spikes": "mapped z -0.92"}
    assert len(t.sha256()) == 64 and t.sha256() == _build().sha256()


REPO = Path(__file__).resolve().parents[1]


def test_the_null_percentiles_are_the_archived_clean_nulls() -> None:
    """p90 is the ruled 1.44 everywhere; p95 / p99 are each band's, rounded to 0.01."""
    src = tm.NULL_PERCENTILE_SOURCE
    raw = (REPO / src["file"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == src["sha256"]
    script = (REPO / src["script"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(script).hexdigest() == src["script_sha256"]
    null = json.loads(raw.decode("utf-8"))
    assert null["session"] == src["recording"]
    assert round(null["pooled_all_signals"]["p90"], 2) == tm.Z_FLOOR == 1.44
    per_band = null["per_band_all_signals"]
    assert set(per_band) == set(tm.NULL_PERCENTILE_Z["p90"])
    for p in ("p95", "p99"):
        assert tm.NULL_PERCENTILE_Z[p] == {b: round(v[p], 2) for b, v in per_band.items()}
    assert set(tm.NULL_PERCENTILE_Z["p90"].values()) == {1.44}
    assert tm.EXTENT_PERCENTILES == ("p90", "p95", "p99")
    for b in per_band:  # the cache stage 1b reads is exact at or above 1.44
        assert tm.NULL_PERCENTILE_Z["p90"][b] <= tm.NULL_PERCENTILE_Z["p95"][b] \
            <= tm.NULL_PERCENTILE_Z["p99"][b]


@pytest.mark.parametrize("pct", ["p95", "p99"])
def test_the_floor_consumers_take_the_declared_percentile_of_their_band(pct: str) -> None:
    t = _build(extent_percentile=pct)
    want = tm.NULL_PERCENTILE_Z[pct]
    assert dict(t.table.z_tol) == {"breathing": 7.69, "mmc": want["2-50"],
                                   "slow_wave": want["0-2"], "spikes": want["300-3000"]}
    assert t.floor_consumers == ("mmc", "slow_wave", "spikes")
    rec = t.record()
    assert rec["extent_percentile"] == pct and rec["cross_band_floor"] == dict(want)
    assert rec["core_blanking"] == {**dict.fromkeys(("breathing", "mmc", "slow_wave",
                                                      "spikes"), "always"), "hrv": "off"}
    assert "DECLARED core blanking" in rec["core_pad_rule"]
    assert rec["null_percentile_source"]["sha256"] == tm.NULL_PERCENTILE_SOURCE["sha256"]
    assert f"clean null's {pct}" in rec["thresholds"]["mmc"]["setter"]
    assert rec["thresholds"]["breathing"]["setter"] == "drift"
    assert t.sha256() != _build().sha256()  # stage 1b is fresh only under the same choice


@pytest.mark.parametrize("bad", [None, "", "p50", "P95", 0.95, "p95_pooled"])
def test_the_percentile_is_required_and_never_defaulted(bad: object) -> None:
    with pytest.raises(ValueError, match="REQUIRED declaration"):
        _build(extent_percentile=bad)
    args = {"tolerances": _tolerances(), "mapping": _mapping(), "scan": _scan(),
            "tolerances_sha256": TOL_SHA, "criterion_sha256": CRIT_SHA,
            "mapping_sha256": "m" * 64, "scan_sha256": "s" * 64, "generation": GEN,
            "cross_band_scope": ALL, "extent_floors": tm.EXTENT_FLOORS_PER_BAND}
    with pytest.raises(TypeError, match="extent_percentile"):
        tm.option1_tolerances(**args)  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [None, {}, {"breathing": "always"},
                                 dict.fromkeys(("breathing", "mmc", "slow_wave", "spikes"),
                                               "sometimes"),
                                 {**dict.fromkeys(("breathing", "mmc", "slow_wave", "spikes"),
                                                  "always"), "hrv": "always"},
                                 dict.fromkeys(("breathing", "mmc", "slow_wave", "spikes"),
                                               "always")])
def test_core_blanking_is_declared_per_consumer_and_never_defaulted(bad: object) -> None:
    with pytest.raises(ValueError, match="core blanking"):
        _build(core_blanking=bad)


def test_both_core_blanking_settings_are_recorded_and_hashed() -> None:
    zc = ("breathing", "mmc", "slow_wave", "spikes")
    vis = _build(core_blanking={**dict.fromkeys(zc, "when_visible"), "hrv": "off"})
    mixed = _build(core_blanking={**dict.fromkeys(zc, "when_visible"), "breathing": "always",
                                  "hrv": "own_signals"})
    assert vis.record()["core_blanking"] == {**dict.fromkeys(zc, "when_visible"), "hrv": "off"}
    assert mixed.core_blanking["breathing"] == "always"
    assert mixed.core_blanking["hrv"] == "own_signals"
    assert len({_build().sha256(), vis.sha256(), mixed.sha256()}) == 3
    assert tm.CORE_BLANKING == ("always", "when_visible", "own_signals_or_visible",
                                "own_signals")
    assert tm.HRV_CORE_BLANKING == ("own_signals", "off")
    andrea = {"slow_wave": "when_visible", "mmc": "when_visible", "spikes": "when_visible",
              "hrv": "own_signals", "breathing": "own_signals_or_visible"}
    assert _build(core_blanking=andrea).core_blanking == andrea  # expressible (scope A)


def test_a_cross_band_pair_counts_while_above_its_own_bands_floor() -> None:
    """floor_by_band equals the decision table's shift (z - floor > 0) and the scalar floor."""
    rng = np.random.default_rng(7)
    ev = _event()
    z = {}
    for b in ("0-2", "2-50", "300-3000"):
        zz = rng.normal(0.0, 1.0, 6000)
        zz[1900:2300] = np.linspace(1.5, 9.0, 400)
        z[("R_V1", b)] = zz
    floors = {"0-2": 2.43, "2-50": 5.88, "300-3000": 1.77}
    pairs = {"e0": tuple(z)}
    by_band = tl.CrossBand(pairs=pairs, z=z, floor_z=99.0, scope={"spikes": "all_detecting_pairs"},
                           own_signals={}, floor_by_band=floors)
    shifted = tl.CrossBand(pairs=pairs, z={q: v - floors[q[1]] for q, v in z.items()},
                           floor_z=0.0, scope={"spikes": "all_detecting_pairs"}, own_signals={})
    a = tl.detected_extent(ev, by_band, "e0", "spikes", signal="L_T", fs=FS, z_t0_s=0.0)
    b = tl.detected_extent(ev, shifted, "e0", "spikes", signal="L_T", fs=FS, z_t0_s=0.0)
    assert a is not None and a == b
    flat = tl.CrossBand(pairs=pairs, z=z, floor_z=2.0, scope={"spikes": "all_detecting_pairs"},
                        own_signals={}, floor_by_band=dict.fromkeys(floors, 2.0))
    scalar = tl.CrossBand(pairs=pairs, z=z, floor_z=2.0, scope={"spikes": "all_detecting_pairs"},
                          own_signals={})
    assert (tl.detected_extent(ev, flat, "e0", "spikes", signal="L_T", fs=FS, z_t0_s=0.0)
            == tl.detected_extent(ev, scalar, "e0", "spikes", signal="L_T", fs=FS, z_t0_s=0.0))
    missing = tl.CrossBand(pairs=pairs, z=z, floor_z=2.0, scope={"spikes": "all_detecting_pairs"},
                           own_signals={}, floor_by_band={"0-2": 2.0})
    with pytest.raises(KeyError, match="no cross-band floor for band"):
        tl.detected_extent(ev, missing, "e0", "spikes", signal="L_T", fs=FS, z_t0_s=0.0)


def test_residual_risk_is_recorded_per_consumer() -> None:
    r = _build().record()["residual_risk"]
    mmc = r["mmc"]
    assert [(x["kind"], x["amplitude_sigma"]) for x in mmc["no_band_no_detector"]] == [
        ("drift", 0.0884)]
    assert mmc["no_band_no_detector"][0]["row"] == {"host_tag": "host1", "dur_s": 5.0,
                                                    "chan_set": "stomach",
                                                    "status": "measured", "censored": False}
    assert [x["kind"] for x in mmc["lifted_not_detected"]] == ["tribo"]  # some band lifts it
    assert all(x["kind"] != "clip" for v in mmc.values() for x in v)  # detected: not residual
    assert [x["kind"] for x in r["slow_wave"]["no_band_no_detector"]] == ["step"]
    assert r["spikes"] == {"no_band_no_detector": [], "lifted_not_detected": []}


@pytest.mark.parametrize(("over", "words"), [
    ({"criterion_sha256": "d" * 64}, "harm criterion"),
    ({"tolerances_sha256": "b" * 64}, "mapping was made from"),
    ({"generation": "0" * 16}, "chain generation"),
])
def test_inputs_that_do_not_belong_together_are_refused(over: dict[str, Any], words: str) -> None:
    with pytest.raises(ValueError, match=words):
        _build(**over)


def test_a_non_production_or_incomplete_mapping_is_refused() -> None:
    m = _mapping()
    m["rows_rule"] = "any-finite"
    with pytest.raises(ValueError, match="gate rows"):
        _build(mapping=m)
    m = _mapping()
    m["flags"] = ["NOT FOR PRODUCTION: rows that are not gate_eligible were used"]
    with pytest.raises(ValueError, match="gate rows"):
        _build(mapping=m)
    m = _mapping()
    del m["consumers"]["slow_wave"]
    with pytest.raises(ValueError, match="no entry for slow_wave"):
        _build(mapping=m)
    s = copy.deepcopy(_scan())
    s["generation_sha256_16"] = "f" * 16
    with pytest.raises(ValueError, match="visibility scan"):
        _build(scan=s)


@pytest.mark.parametrize("inp", [None, {"name": "t.json"}, {"sha256": "b" * 64}, "a" * 64])
def test_a_scan_that_does_not_record_the_declared_tolerances_is_refused(inp: object) -> None:
    """Review 7 finding 4: the scan's input tolerances must be the declared ones."""
    s = _scan()
    if inp is None:
        del s["input"]
    else:
        s["input"] = inp
    with pytest.raises(ValueError, match="visibility scan records input tolerances"):
        _build(scan=s)


# ---------------------------------------------------------------- the declared scope

@pytest.mark.parametrize("scope", tl.CROSS_BAND_SCOPES)
def test_each_scope_is_accepted_and_recorded(scope: str) -> None:
    t = _build(cross_band_scope={**ALL, "slow_wave": scope})
    assert t.record()["cross_band_scope"]["slow_wave"] == scope
    assert dict(t.cross_band_scope) == {**ALL, "slow_wave": scope}


def test_the_scope_is_in_the_hash_stage_1_is_fresh_under() -> None:
    hashes = {_build(cross_band_scope={**ALL, "mmc": s}).sha256() for s in tl.CROSS_BAND_SCOPES}
    assert len(hashes) == 3


@pytest.mark.parametrize(("scope", "words"), [
    ({k: v for k, v in ALL.items() if k != "slow_wave"}, r"undeclared \['slow_wave'\]"),
    ({**ALL, "hrv": tl.CROSS_BAND_SCOPE_ALL}, r"not z consumers \['hrv'\]"),
    ({**ALL, "mmc": "own_signals"}, r"unknown values \{'mmc': 'own_signals'\}"),
    ({**ALL, "mmc": None}, "unknown values"),
    (None, "is a mapping"),
    ("all_detecting_pairs", "is a mapping"),
])
def test_a_scope_not_declared_for_exactly_the_z_consumers_is_refused(scope: object,
                                                                     words: str) -> None:
    with pytest.raises(ValueError, match=words):
        _build(cross_band_scope=scope)


def test_a_scope_short_of_the_ruling_lists_what_it_leaves_unblanked() -> None:
    """off: every detected row of a kind invisible in the band; own: those seen only off it."""
    scan = _scan()
    scan["rows"] += [_scan_row("mmc", "step", 0.3, ["RVN|300-3000"], ["RVN|300-3000"]),
                     _scan_row("mmc", "drift", 0.7, ["ANT2|0-2", "LVN|0-2"], ["ANT2|0-2"])]
    off = _build(scan=scan, cross_band_scope={**ALL, "mmc": tl.CROSS_BAND_SCOPE_OFF})
    r = off.record()["residual_risk"]["mmc"]["cross_band_off"]
    assert sorted((x["kind"], x["amplitude_sigma"]) for x in r) == [
        ("clip", 1.41), ("drift", 0.7), ("step", 0.3)]  # all detected; none visible in 2-50
    own = _build(scan=scan, cross_band_scope={**ALL, "mmc": tl.CROSS_BAND_SCOPE_OWN})
    r = own.record()["residual_risk"]["mmc"]["detected_only_off_own_signals"]
    assert [(x["kind"], x["pairs_detectable"]) for x in r] == [("step", ["RVN|300-3000"])]
    assert "cross_band_off" not in _build(scan=scan).record()["residual_risk"]["mmc"]
    vis = _mapping()
    vis["consumers"]["mmc"]["kinds"]["step"] = _kind(3.0, True)  # visible: its own band blanks
    r = _build(scan=scan, mapping=vis, cross_band_scope={**ALL, "mmc": "off"}).record()
    assert [x["kind"] for x in r["residual_risk"]["mmc"]["cross_band_off"]] == ["clip", "drift"]


# ---------------------------------------------------------------- cross-band extents

DUR = 60.0
CORE = (30.0, 30.5)


def _event(start: float = CORE[0], stop: float = CORE[1]) -> Event:
    c = Candidate(start, stop, ("L_V1",), ("0-2",), 9.0, "electrical")
    return Event(c, "motion", 0.9, "model")


def _z(own_level: float) -> dict[tuple[str, str], Any]:
    """Return mmc's own band and one detecting pair.

    The own band (ANT1|2-50) is at ``own_level`` over the core; the detecting pair
    (L_V1|0-2) is over z_enter in the core and over 1.44 from 29.8 to 31.2 s.
    """
    own = make_band_z("2-50", DUR, bumps=((29.9, 30.6, own_level, "ANT1"),),
                      signal="ANT1").z_max
    det = make_band_z("0-2", DUR, bumps=((29.8, 31.2, 2.0, "L_V1"), (30.1, 30.3, 6.0, "L_V1")),
                      signal="L_V1").z_max
    return {("ANT1", "2-50"): own, ("L_V1", "0-2"): det}


def _cross(z: dict[tuple[str, str], Any], scope: str = tl.CROSS_BAND_SCOPE_ALL,
           pairs: tuple[tuple[str, str], ...] = (("L_V1", "0-2"),)) -> tl.CrossBand:
    own = frozenset({"ANT1", "ANT2", "ANT3", "stomach_ref"})
    return tl.CrossBand(pairs={"e1": pairs}, z=z, floor_z=tm.Z_FLOOR, scope={"mmc": scope},
                        own_signals={("mmc", "ANT1"): own})


def _extents(own_level: float, *, cross: bool) -> list[Any]:
    z = _z(own_level)
    table = tl.ToleranceTable({"mmc": 1.44}, "test")
    return tl.extents_for_events({"e1": _event()}, z, {"mmc": ("ANT1",)}, tolerances=table,
                                 fs=FS, z_t0_s=0.0, confirmed=lambda e: True,
                                 cross_band=_cross(z) if cross else None)


def test_a_core_invisible_in_the_own_band_blanks_over_its_detected_extent() -> None:
    (_eid, consumer, sig, ext), = _extents(1.0, cross=True)  # own band under 1.44
    pad = tl.consumer_settling("mmc", FS).total_s
    assert isinstance(ext, tl.Extent) and ext.basis == tl.EXTENT_BASIS_DETECTED
    assert (consumer, sig, ext.band) == ("mmc", "ANT1", "2-50")
    assert ext.core_start_s == pytest.approx(29.8) and ext.core_stop_s == pytest.approx(31.2)
    assert ext.start_s == pytest.approx(29.8 - pad) and ext.stop_s == pytest.approx(31.2 + pad)
    assert ext.settling_s == pad


def test_without_cross_band_the_same_core_is_under_tolerance() -> None:
    (_eid, _c, _s, ext), = _extents(1.0, cross=False)
    assert ext is None


def test_a_core_visible_in_the_own_band_keeps_its_own_band_extent() -> None:
    (_eid, _c, _s, ext), = _extents(5.0, cross=True)
    assert isinstance(ext, tl.Extent) and ext.basis == tl.EXTENT_BASIS_OWN_BAND
    assert ext.core_start_s == pytest.approx(29.9) and ext.core_stop_s == pytest.approx(30.6)


def test_the_detected_extent_reads_only_the_detecting_pairs() -> None:
    z = _z(1.0)
    z[("R_V3", "100-300")] = make_band_z("100-300", DUR, bumps=((10.0, 50.0, 2.5, "R_V3"),),
                                         signal="R_V3").z_max  # over the floor, but not detecting
    ext = tl.detected_extent(_event(), _cross(z), "e1", "mmc", signal="ANT1", fs=FS,
                             z_t0_s=0.0)
    assert ext is not None and ext.core_start_s == pytest.approx(29.8)
    with pytest.raises(KeyError, match="no detecting pairs"):
        tl.detected_extent(_event(), _cross(z), "e2", "mmc", signal="ANT1", fs=FS, z_t0_s=0.0)
    flat = {("L_V1", "0-2"): np.full(int(DUR / 0.01), 0.5)}
    assert tl.detected_extent(_event(), _cross(flat), "e1", "mmc", signal="ANT1", fs=FS,
                              z_t0_s=0.0) is None


def test_own_signals_only_takes_the_extent_from_the_consumers_own_pairs_only() -> None:
    z = _z(1.0)
    z[("ANT2", "0-2")] = make_band_z("0-2", DUR, bumps=((30.2, 30.4, 6.0, "ANT2"),),
                                     signal="ANT2").z_max
    both = (("L_V1", "0-2"), ("ANT2", "0-2"))
    kw: dict[str, Any] = {"signal": "ANT1", "fs": FS, "z_t0_s": 0.0}
    every = tl.detected_extent(_event(), _cross(z, pairs=both), "e1", "mmc", **kw)
    assert every is not None and every.core_start_s == pytest.approx(29.8)  # the nerve pair
    mine = tl.detected_extent(_event(), _cross(z, tl.CROSS_BAND_SCOPE_OWN, both), "e1", "mmc",
                              **kw)
    assert mine is not None and mine.basis == tl.EXTENT_BASIS_DETECTED
    assert (mine.core_start_s, mine.core_stop_s) == (pytest.approx(30.2), pytest.approx(30.4))
    assert tl.detected_extent(_event(), _cross(z, tl.CROSS_BAND_SCOPE_OWN), "e1", "mmc",
                              **kw) is None  # detected only on a nerve pair: not mmc's own


def test_scope_off_gives_no_detected_extent() -> None:
    z = _z(1.0)
    assert tl.detected_extent(_event(), _cross(z, tl.CROSS_BAND_SCOPE_OFF), "e1", "mmc",
                              signal="ANT1", fs=FS, z_t0_s=0.0) is None
    table = tl.ToleranceTable({"mmc": 1.44}, "test")
    ((_e, _c, _s, ext),) = tl.extents_for_events(
        {"e1": _event()}, z, {"mmc": ("ANT1",)}, tolerances=table, fs=FS, z_t0_s=0.0,
        confirmed=lambda e: True, cross_band=_cross(z, tl.CROSS_BAND_SCOPE_OFF))
    assert ext is None  # under tolerance, as without cross-band


def test_an_undeclared_or_unknown_scope_is_refused() -> None:
    z = _z(1.0)
    bare = tl.CrossBand(pairs={"e1": (("L_V1", "0-2"),)}, z=z, floor_z=tm.Z_FLOOR, scope={},
                        own_signals={})
    with pytest.raises(KeyError, match="declares no cross-band scope"):
        tl.detected_extent(_event(), bare, "e1", "mmc", signal="ANT1", fs=FS, z_t0_s=0.0)
    with pytest.raises(ValueError, match="unknown cross-band scope"):
        tl.detected_extent(_event(), _cross(z, "own"), "e1", "mmc", signal="ANT1", fs=FS,
                           z_t0_s=0.0)
    no_own = tl.CrossBand(pairs={"e1": (("L_V1", "0-2"),)}, z=z, floor_z=tm.Z_FLOOR,
                          scope={"mmc": tl.CROSS_BAND_SCOPE_OWN}, own_signals={})
    with pytest.raises(KeyError, match="own detection signals of"):
        tl.detected_extent(_event(), no_own, "e1", "mmc", signal="ANT1", fs=FS, z_t0_s=0.0)


DET = ("ANT1", "ANT2", "ANT3", "L_T", "L_V1", "L_V2", "L_V3", "R_T", "R_V1", "R_V2", "R_V3",
       "stomach_ref")
CH = {"LVN1": "L_V1", "LVN2": "L_V2", "LVN3": "L_V3", "RVN1": "R_V1", "RVN2": "R_V2",
      "RVN3": "R_V3", "ANT1": "ANT1", "ANT2": "ANT2", "ANT3": "ANT3"}
NERVE_L = {"L_T", "L_V1", "L_V2", "L_V3"}
NERVE_R = {"R_T", "R_V1", "R_V2", "R_V3"}
STOMACH = {"ANT1", "ANT2", "ANT3", "stomach_ref"}


@pytest.mark.parametrize(("signals", "want"), [
    (("ANT1", "ANT2", "ANT3"), STOMACH),          # slow_wave, mmc: the stomach channels
    (("L_T", "R_T"), NERVE_L | NERVE_R),          # the union (never a consumer's own set)
    (("R_T",), NERVE_R),                          # breathing on one cuff's tripole
    (("LVN2-RVN1",), NERVE_L | NERVE_R),          # breathing on a cross-cuff lead
    (("RVN3",), NERVE_R),                         # a raw contact
])
def test_own_detection_signals_are_the_consumers_signals_and_their_channels(
        signals: tuple[str, ...], want: set[str]) -> None:
    assert tl.own_detection_signals(signals, DET, CH) == want


SC_SIGNALS = {"spikes": ("L_T", "R_T"), "breathing": ("LVN2-RVN1",), "mmc": ("ANT1",)}


def _scoped_run(det_sig: str, signals: dict[str, tuple[str, ...]] | None = None,
                own: Any = None) -> dict[tuple[str, str], Any]:  # noqa: ANN401
    """Every own band flat under 9; one core detected on ``det_sig`` only; own_signals_only."""
    sigs = SC_SIGNALS if signals is None else signals
    bands = {c: tl.extent_consumers()[c].band for c in sigs}
    z: dict[tuple[str, str], Any] = {(s, bands[c]): make_band_z(bands[c], DUR, signal=s).z_max
                                     for c, v in sigs.items() for s in v}
    det = make_band_z("0-2", DUR, bumps=((29.8, 31.2, 2.0, det_sig), (30.1, 30.3, 6.0, det_sig)),
                      signal=det_sig).z_max
    cross = tl.CrossBand(pairs={"e1": ((det_sig, "0-2"),)}, z={(det_sig, "0-2"): det},
                         floor_z=tm.Z_FLOOR, scope=dict.fromkeys(sigs, tl.CROSS_BAND_SCOPE_OWN),
                         own_signals=(tl.own_signals_by_signal(sigs, DET, CH) if own is None
                                      else own))
    out = tl.extents_for_events({"e1": _event()}, z, sigs,
                                tolerances=tl.ToleranceTable(dict.fromkeys(sigs, 9.0), "test"),
                                fs=FS, z_t0_s=0.0, confirmed=lambda e: True, cross_band=cross)
    return {(c, s): r for _e, c, s, r in out}


def _detected(r: object) -> bool:
    return isinstance(r, tl.Extent) and r.basis == tl.EXTENT_BASIS_DETECTED


def test_own_signals_only_is_per_signal_never_per_consumer() -> None:
    """Review of r1010, fix 1: own signals are per signal, never per consumer.

    A core seen only on R_V2 never blanks L_T's spikes; one seen only on L_T never blanks
    breathing routed on the cross-cuff lead LVN2-RVN1 (HR source: L_V2 and R_V1) or on R_T.
    The consumer-level family (both cuffs for spikes; every nerve signal for a cross-cuff
    lead) blanked both.
    """
    r = _scoped_run("R_V2")
    assert _detected(r[("spikes", "R_T")]) and r[("spikes", "L_T")] is None
    assert r[("breathing", "LVN2-RVN1")] is None and r[("mmc", "ANT1")] is None
    r = _scoped_run("L_T")
    assert _detected(r[("spikes", "L_T")]) and r[("spikes", "R_T")] is None
    assert r[("breathing", "LVN2-RVN1")] is None
    on_rt = _scoped_run("L_T", {"breathing": ("R_T",)})
    assert on_rt[("breathing", "R_T")] is None
    r = _scoped_run("R_V1")  # one of the HR pair's two channels
    assert _detected(r[("breathing", "LVN2-RVN1")]) and _detected(r[("spikes", "R_T")])
    assert r[("spikes", "L_T")] is None
    assert _detected(_scoped_run("stomach_ref")[("mmc", "ANT1")])


def test_own_signals_by_signal_is_the_one_mapping_for_both_rules() -> None:
    own = tl.own_signals_by_signal({**SC_SIGNALS, "hrv": ("LVN2-RVN1",)}, DET, CH)
    assert own == {("spikes", "L_T"): NERVE_L, ("spikes", "R_T"): NERVE_R,
                   ("breathing", "LVN2-RVN1"): {"L_V2", "R_V1"}, ("mmc", "ANT1"): STOMACH,
                   ("hrv", "LVN2-RVN1"): {"L_V2", "R_V1"}}
    assert all(own[k] == tl.core_own_signals(*k, DET, CH) for k in own)
    with pytest.raises(ValueError, match="HR-source channels are unknown"):
        tl.own_signals_by_signal({"breathing": ("ECG",)}, DET, CH)
    # a consumer-keyed mapping (the old shape) is refused by name, never read as "no own"
    with pytest.raises(KeyError, match="own detection signals of"):
        _scoped_run("R_V2", own={"spikes": NERVE_L | NERVE_R})


def test_an_own_signal_with_no_detection_signal_is_refused() -> None:
    with pytest.raises(ValueError, match="own-signal family is unknown"):
        tl.own_detection_signals(("ECG",), DET, CH)
    with pytest.raises(ValueError, match="'XVN9' is no detection signal"):
        tl.own_detection_signals(("LVN2-XVN9",), DET, CH)


def test_channel_detection_signals_follow_the_channel_map() -> None:
    from types import SimpleNamespace as N  # noqa: PLC0415 - a minimal ChannelInfo stand-in
    chans = [N(name="LVN2", role="nerve", cuff_id="L", contact_index=2),
             N(name="RVN", role="nerve", cuff_id="R", contact_index=None),
             N(name="ANT3", role="stomach", cuff_id=None, contact_index=None),
             N(name="ACC", role="aux", cuff_id=None, contact_index=None)]
    assert tl.channel_detection_signals(chans) == {"LVN2": "L_V2", "RVN": "R_T", "ANT3": "ANT3"}


# ---------------------------------------------------------------- routing cannot undo it

def _evidence(z: dict[tuple[str, str], Any]) -> rt.EventEvidence:
    x = np.random.default_rng(3).normal(0.0, 5.0, int(DUR * 1000))
    return rt.EventEvidence(event_id="e1@ANT1", signal="ANT1", x=x, fs=1000.0, span_s=CORE,
                            x_t0_s=0.0, z=z, z_t0_s=0.0)


def test_routing_cannot_undo_a_detected_extent() -> None:
    (_eid, _c, _s, ext), = _extents(1.0, cross=True)
    table = tl.ToleranceTable({"mmc": 1.44}, "test")
    routed = rt.route_event(_evidence(_z(1.0)), "mmc", table)
    assert routed.route == "correct"  # the own band is under its threshold: routing would undo
    with pytest.raises(ValueError, match="routing may not undo"):
        mk.spans_from_routing([("e1@ANT1", ext)], [routed])
    forged = rt.RouteDecision("e1@ANT1", "mmc", "reject", "in band", "in_band_z")
    with pytest.raises(ValueError, match="routing may not undo"):
        mk.spans_from_routing([("e1@ANT1", ext)], [forged])
    dec = rt.cross_band_decision("e1@ANT1", "mmc")
    assert dec.masks and dec.reason_code == rt.CROSS_BAND_REASON_CODE
    (span,) = mk.spans_from_routing([("e1@ANT1", ext)], [dec])
    assert (span.consumer, span.signal) == ("mmc", "ANT1")
    assert (span.start_s, span.stop_s) == (ext.start_s, ext.stop_s)
    assert span.reason == rt.CROSS_BAND_REASON_CODE


def test_own_band_extents_still_route() -> None:
    (_eid, _c, _s, ext), = _extents(5.0, cross=True)
    ok = rt.RouteDecision("e1@ANT1", "mmc", "correct", "separable", "within_z")
    assert mk.spans_from_routing([("e1@ANT1", ext)], [ok]) == []


# ---------------------------------------------------------------------------
# Andrea, revised 2026-10-10 (scope B): per-consumer own-signal floors
# ---------------------------------------------------------------------------

OWN_SETS = {"slow_wave": ["ANT1", "ANT2", "ANT3"], "mmc": ["ANT1", "ANT2", "ANT3"],
            "spikes": ["L_V1", "L_V2", "L_V3", "R_V1", "R_V2", "R_V3"],
            "breathing": ["R_V2", "R_V3"]}


def _own_archive(level: float = 3.0) -> dict[str, Any]:
    """Return a synthetic baseline's own-signal floors, as own_signal_floors.py archives them."""
    z: dict[tuple[str, str], Any] = {}
    for c, sigs in OWN_SETS.items():
        band = tl.extent_consumers()[c].band
        for k, s in enumerate(sigs):
            z[(s, band)] = make_band_z(band, DUR, signal=s, bumps=(
                (5.0 + k, 6.0 + k, level + k, s), (20.0, 20.5, 2 * level, s))).z_max
    return {"session": "base", "per_consumer_own_signals": {
        c: [tm.own_signal_floor_record(z, consumer=c, signals=sigs, percentile=p)
            for p in tm.EXTENT_PERCENTILES] for c, sigs in OWN_SETS.items()}}


def _floors(archive: dict[str, Any], pct: str = "p99") -> dict[str, Any]:
    out = {}
    for c, rows in archive["per_consumer_own_signals"].items():
        r = next(x for x in rows if x["percentile"] == pct)
        out[c] = {"value": r["value"], "percentile": pct, "source": "test",
                  "population": {"signals": r["signals"], "band": r["band"],
                                 "source_recording": archive["session"],
                                 "n_frames": r["n_frames"],
                                 "archive_file": "measurements/x/own_floors.json",
                                 "archive_sha256": "a" * 64}}
    return out


def test_own_signal_floors_set_each_consumers_threshold_and_are_recorded() -> None:
    archive = _own_archive()
    decl = _floors(archive)
    t = _build(extent_floors=decl)
    for c in ("mmc", "slow_wave", "spikes"):  # the floor consumers take their OWN floor
        assert t.table.z_tol[c] == decl[c]["value"] > tm.Z_FLOOR
        assert "own-signal floor binds" in t.thresholds[c].setter
    assert decl["breathing"]["value"] < 7.69 and t.table.z_tol["breathing"] == 7.69
    assert "option-(1) threshold binds" in t.thresholds["breathing"].setter
    high = _floors(_own_archive(level=9.0))
    assert _build(extent_floors=high).table.z_tol["breathing"] == high["breathing"]["value"] > 7.69
    rec = t.record()
    assert rec["extent_floors"]["spikes"]["population"]["signals"] == OWN_SETS["spikes"]
    assert rec["cross_band_floor"] == tm.NULL_PERCENTILE_Z["p90"]  # cross-band stays per band
    assert _build().record()["extent_floors"] == tm.EXTENT_FLOORS_PER_BAND
    assert _build().table.z_tol == _build(extent_floors="per_band_percentile").table.z_tol
    assert t.sha256() != _build().sha256()


def test_an_own_signal_floor_is_reproducible_from_its_archive() -> None:
    archive = _own_archive()
    decl = _floors(archive)
    for c in OWN_SETS:
        tm.verify_floor_archive(c, decl[c], archive)
    with pytest.raises(ValueError, match="does not reproduce"):
        tm.verify_floor_archive("mmc", {**decl["mmc"], "value": decl["mmc"]["value"] + 0.01},
                                archive)
    with pytest.raises(ValueError, match="not the declared source recording"):
        tm.verify_floor_archive("mmc", decl["mmc"], {**archive, "session": "other"})
    # the archive's value is the pooled percentile over exactly those signals
    band = tl.extent_consumers()["spikes"].band
    z = {(s, band): make_band_z(band, DUR, signal=s, bumps=((1.0, 3.0, 4.0 + i, s),)).z_max
         for i, s in enumerate(OWN_SETS["spikes"])}
    got = tm.own_signal_floor_record(z, consumer="spikes", signals=OWN_SETS["spikes"],
                                     percentile="p99")
    pooled = np.concatenate([z[(s, band)] for s in OWN_SETS["spikes"]])
    assert got["value"] == round(float(np.percentile(pooled, 99.0)), 2)
    assert got["n_frames"] == pooled.size
    script = (REPO / "measurements/2026-10-10/extent_null/own_signal_floors.py").read_text(
        encoding="utf-8")
    assert "tm.own_signal_floor_record(" in script and "verify_floor_archive" in script


@pytest.mark.parametrize("bad", [None, {}, "per_band", "p99"])
def test_extent_floors_are_required_and_never_defaulted(bad: object) -> None:
    with pytest.raises(ValueError, match="extent_floors"):
        _build(extent_floors=bad)
    args = {"tolerances": _tolerances(), "mapping": _mapping(), "scan": _scan(),
            "tolerances_sha256": TOL_SHA, "criterion_sha256": CRIT_SHA,
            "mapping_sha256": "m" * 64, "scan_sha256": "s" * 64, "generation": GEN,
            "cross_band_scope": ALL, "extent_percentile": "p90",
            "core_blanking": {**dict.fromkeys(("breathing", "mmc", "slow_wave", "spikes"),
                                              "always"), "hrv": "off"}}
    with pytest.raises(TypeError, match="extent_floors"):
        tm.option1_tolerances(**args)  # type: ignore[call-arg]


def test_an_own_signal_floor_on_another_consumers_signals_or_band_is_refused() -> None:
    decl = _floors(_own_archive())
    cases = {"not stomach signals": ("mmc", {"signals": ["L_V1"]}),
             "not nerve signals": ("spikes", {"signals": ["ANT1"]}),
             "own band": ("slow_wave", {"band": "2-50"}),
             "lacks": ("breathing", {"n_frames": 0})}
    for words, (c, over) in cases.items():
        bad = {**decl, c: {**decl[c], "population": {**decl[c]["population"], **over}}}
        with pytest.raises(ValueError, match=words.split()[-1]):
            _build(extent_floors=bad)
    with pytest.raises(ValueError, match=r">= 1.44"):
        _build(extent_floors={**decl, "mmc": {**decl["mmc"], "value": 1.2}})
