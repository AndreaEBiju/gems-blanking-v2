"""Tests for :mod:`gems_blanking_v2.emit.routing` (addendum to ruling (c), 2026-09-30)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.emit.routing import (
    canonical_json,
    excluded_inputs,
    read_damage,
    routing_hash,
    validate_routing,
    write_damage,
    write_routing,
)
from gems_blanking_v2.io.store import GemsStore
from hypothesis import given, settings
from hypothesis import strategies as st


def _table() -> dict:
    return {"schema": 1, "recordings": {
        "gems_x_1": {"spike": {"L": {"route": "scalar", "veto": {"w_s": 0.0004, "theta_sigma": 5.1},
                                     "peri_r_ms": [-5.5, 1.5], "peri_r_beats": "hrv"},
                               "R": {"route": "distrusted", "why": "fails (iii)"}},
                     "hr": {"channel": "L_T", "detector": "findpeaks", "n_beats": 3640},
                     "stomach_ref": {"notch_hz": [60.0], "contacts": ["ANT1"]}},
        "gems_x_2": {"spike": {"L": {"route": "uncorrected"}},
                     "hr": {"none": "no count-gated train"}, "stomach_ref": {}}}}


def test_the_hash_ignores_key_order_and_a_recorded_hash() -> None:
    t = _table()
    shuffled = json.loads(json.dumps(t))
    shuffled["recordings"] = dict(reversed(list(shuffled["recordings"].items())))
    assert routing_hash(t) == routing_hash(shuffled) == routing_hash({**t, "hash": "whatever"})


def test_any_change_of_route_changes_the_hash() -> None:
    t = _table()
    u = copy.deepcopy(t)
    u["recordings"]["gems_x_1"]["spike"]["L"]["peri_r_ms"] = [-5.0, 1.5]
    assert routing_hash(t) != routing_hash(u)


@settings(max_examples=100, deadline=None)
@given(st.dictionaries(
    st.text(max_size=8),
    st.one_of(st.integers(), st.text(max_size=8), st.floats(allow_nan=False, allow_infinity=False)),
    max_size=6))
def test_canonical_json_round_trips_and_is_ascii(obj: dict) -> None:
    text = canonical_json(obj)
    assert text.isascii()
    assert json.loads(text) == obj
    assert canonical_json(json.loads(text)) == text


def test_the_schema_is_enforced_by_name() -> None:
    t = _table()
    t["recordings"]["gems_x_1"]["spike"]["R"] = {"route": "distrusted"}
    with pytest.raises(ValueError, match="gems_x_1 R: a distrusted cuff must say why"):
        validate_routing(t)
    t = _table()
    t["recordings"]["gems_x_2"]["spike"]["L"]["route"] = "masked"
    with pytest.raises(ValueError, match="not one of"):
        validate_routing(t)
    t = _table()
    del t["recordings"]["gems_x_2"]["hr"]["none"]
    with pytest.raises(ValueError, match="gems_x_2: hr names"):
        validate_routing(t)
    t = _table()
    del t["recordings"]["gems_x_1"]["spike"]["L"]["peri_r_beats"]
    with pytest.raises(ValueError, match="goes with a peri-R extent"):
        validate_routing(t)
    t = _table()
    del t["recordings"]["gems_x_1"]["stomach_ref"]
    with pytest.raises(ValueError, match="lacks stomach_ref"):
        validate_routing(t)


def test_excluded_inputs_are_the_distrusted_cuffs_and_a_missing_hr_train() -> None:
    t = _table()["recordings"]
    assert excluded_inputs(t["gems_x_1"]) == ["spike R: fails (iii)"]
    assert excluded_inputs(t["gems_x_2"]) == ["hr: no count-gated train"]


def test_a_frozen_table_is_written_once_and_damage_names_it(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    h, path = write_routing(store, _table())
    assert path.name == f"routing_{h[:16]}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["hash"] == h
    assert write_routing(store, _table()) == (h, path)  # idempotent
    f = write_damage(store, "plan_a", h, {"plan_a_s1#m0": "target"},
                     {"plan_a_s1#m0": ["hr: no count-gated train"]})
    assert read_damage(store, "plan_a", h) == {"plan_a_s1#m0": "target"}
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["routing_hash"] = "0" * 64
    f.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="computed under routing"):
        read_damage(store, "plan_a", h)
    with pytest.raises(ValueError, match="invalid damage classes"):
        write_damage(store, "plan_a", h, {"m": "maybe"}, {})


def test_a_different_table_under_the_same_name_is_refused(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _h, path = write_routing(store, _table())
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["hash"] = "f" * 64
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FileExistsError, match="different routing table"):
        write_routing(store, _table())


def test_routing_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.emit.routing" not in chain.generation_modules()


# --- schema 2: per-recording entries, append-only (ruling 2026-09-30 (d) 2) -------


def _v2():  # noqa: ANN202
    from gems_blanking_v2.emit.routing import build_table  # noqa: PLC0415

    return build_table(_table()["recordings"], "test")


def _new_entry() -> dict:
    return {"spike": {"L": {"route": "scalar"}}, "hr": {"none": "no count-gated train"},
            "stomach_ref": {}}


def test_each_entry_carries_its_own_hash_and_the_table_hashes_the_hashes() -> None:
    from gems_blanking_v2.emit.routing import entry_hash, table_hash  # noqa: PLC0415

    t = _v2()
    assert t["entry_hashes"] == {r: entry_hash(e) for r, e in t["entries"].items()}
    assert t["hash"] == table_hash(t)
    renamed = {**t, "frozen_under": "another note"}  # annotation is not routing
    assert table_hash(renamed) == t["hash"]


def test_an_append_keeps_every_old_entry_and_records_the_check() -> None:
    from gems_blanking_v2.emit.routing import append_entries  # noqa: PLC0415

    old = _v2()
    new, check = append_entries(old, {"gems_x_3": _new_entry()}, "round 6")
    assert check == {"old_hash": old["hash"], "new_hash": new["hash"], "unchanged_entries": True,
                     "changed": [], "removed": [], "added": ["gems_x_3"]}
    assert new["hash"] != old["hash"]
    assert all(new["entry_hashes"][r] == h for r, h in old["entry_hashes"].items())


def test_appending_an_already_routed_recording_is_refused() -> None:
    from gems_blanking_v2.emit.routing import append_entries  # noqa: PLC0415

    with pytest.raises(ValueError, match=r"already routed, so not an append: \['gems_x_1'\]"):
        append_entries(_v2(), {"gems_x_1": _new_entry()}, "round 6")


def test_a_changed_entry_alone_fails_the_unchanged_check() -> None:
    from gems_blanking_v2.emit.routing import build_table, compare_tables  # noqa: PLC0415

    old = _v2()
    entries = copy.deepcopy(old["entries"])
    entries["gems_x_1"]["spike"]["L"]["peri_r_narrow_ms"] = [[14.0, 14.5]]
    check = compare_tables(old, build_table(entries, "changed"))
    assert not check["unchanged_entries"]
    assert (check["changed"], check["removed"]) == (["gems_x_1"], [])


def test_a_removed_entry_alone_fails_the_unchanged_check() -> None:
    from gems_blanking_v2.emit.routing import build_table, compare_tables  # noqa: PLC0415

    old = _v2()
    entries = copy.deepcopy(old["entries"])
    del entries["gems_x_2"]
    check = compare_tables(old, build_table(entries, "removed"))
    assert not check["unchanged_entries"]
    assert (check["changed"], check["removed"]) == ([], ["gems_x_2"])


def test_recorded_entry_hashes_are_checked_against_the_entries() -> None:
    t = _v2()
    t["entries"]["gems_x_2"]["hr"] = {"none": "edited after hashing"}
    with pytest.raises(ValueError, match=r"do not match their entries: \['gems_x_2'\]"):
        validate_routing(t)


def test_a_stored_table_is_read_back_only_if_it_hashes_to_its_name(tmp_path: Path) -> None:
    from gems_blanking_v2.emit.routing import entry_hash, read_routing  # noqa: PLC0415

    store = GemsStore.initialise(tmp_path / "gems")
    h, path = write_routing(store, _v2())
    assert read_routing(store, h)["hash"] == h
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["frozen_under"] = "annotation only"  # does not change the routing, still reads
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert read_routing(store, h)["entries"] == _v2()["entries"]
    doc["entries"]["gems_x_1"]["hr"]["n_beats"] = 1
    doc["entry_hashes"]["gems_x_1"] = entry_hash(doc["entries"]["gems_x_1"])  # consistent edit
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="stored table hashes to"):
        read_routing(store, h)


def test_a_narrow_extent_needs_its_beat_source_and_may_stand_alone() -> None:
    t = _table()
    t["recordings"]["gems_x_2"]["spike"]["L"]["peri_r_narrow_ms"] = [[14.0, 14.5]]
    with pytest.raises(ValueError, match="goes with a peri-R extent"):
        validate_routing(t)
    t["recordings"]["gems_x_2"]["spike"]["L"]["peri_r_beats"] = "mask"
    validate_routing(t)  # a narrow extent without a broad one is valid
    t["recordings"]["gems_x_2"]["spike"]["L"] = {"route": "scalar", "peri_r_beats": "hrv"}
    with pytest.raises(ValueError, match="and only with one"):
        validate_routing(t)


# --- the two damage conventions (ruling (d) 1) --------------------------------------


def test_the_gate_and_the_reported_convention_are_stored_apart(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    h = "c" * 64
    g = write_damage(store, "plan_a", h, {"m": "below"}, {})
    e = write_damage(store, "plan_a", h, {"m": "target"}, {"m": ["hr: none"]},
                     convention="excluded_is_target")
    assert g != e
    assert read_damage(store, "plan_a", h) == {"m": "below"}
    assert read_damage(store, "plan_a", h, "excluded_is_target") == {"m": "target"}
    doc = json.loads(g.read_text(encoding="utf-8"))
    doc["convention"] = "excluded_is_target"
    g.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="holds convention"):
        read_damage(store, "plan_a", h)
    with pytest.raises(ValueError, match="not one of"):
        write_damage(store, "plan_a", h, {}, {}, convention="lenient")


def test_a_round_appends_and_records_old_hash_new_hash_and_the_check(tmp_path: Path) -> None:
    from gems_blanking_v2.emit.routing import (  # noqa: PLC0415
        append_for_round,
        read_routing,
        routing_check_path,
    )

    store = GemsStore.initialise(tmp_path / "gems")
    old_h, _p = write_routing(store, _v2())
    new_h, rec = append_for_round(store, "plan_r6", old_h, {"gems_x_3": _new_entry()}, "round 6")
    assert rec["old_hash"] == old_h and rec["new_hash"] == new_h and rec["unchanged_entries"]
    assert rec["added"] == ["gems_x_3"]
    stored = json.loads(routing_check_path(store, "plan_r6").read_text(encoding="utf-8"))
    assert stored == rec
    assert set(read_routing(store, new_h)["entries"]) == {"gems_x_1", "gems_x_2", "gems_x_3"}
    with pytest.raises(ValueError, match="already routed"):
        append_for_round(store, "plan_r7", new_h, {"gems_x_1": _new_entry()}, "round 7")


# --- ruling 2026-10-03: per-minute cuff trust ---------------------------------------------

_PM_HR = {"channel": "LVN1-RVN2", "detector": "findpeaks", "n_beats": 6633, "source": "pair",
          "storage": "per_minute", "valid_windows": [18, 20],
          "blank_s": [[120.0, 180.0], [600.0, 660.0]]}
_WHOLE = {"channel": "L_T", "detector": "task05", "n_beats": 3400}
_SV = {"route": "uncorrected", "w_s": 0.0015, "theta_sigma": 6.2}


def _pm_entry(spike: dict, hr: dict | None = None) -> dict:
    return {"spike": {"L": spike}, "hr": _PM_HR if hr is None else hr, "stomach_ref": {}}


def test_a_hump_only_cuff_is_recognised_and_any_other_reason_is_not() -> None:
    from gems_blanking_v2.emit.routing import HUMP_REASON, hump_only  # noqa: PLC0415

    assert hump_only({"route": "distrusted", "why": HUMP_REASON})
    iii = "veto verification (iii) fails: an event core survives"
    assert not hump_only({"route": "distrusted", "why": f"{iii}; {HUMP_REASON}"})
    assert not hump_only({"route": "distrusted", "why": "veto verification (ii) fails"})
    assert not hump_only({"route": "uncorrected"})
    assert not hump_only({"route": "uncorrected", "why": HUMP_REASON})  # a trusted cuff
    assert not hump_only({"route": "distrusted", "why": ""})


def test_the_per_minute_cuff_entry_carries_the_train_s_blank_spans_and_validates() -> None:
    from gems_blanking_v2.emit.routing import per_minute_cuff_entry, validate_entry  # noqa: PLC0415

    with_mask = per_minute_cuff_entry(_SV, _PM_HR, [2.0, 6.0])
    assert with_mask == {"route": "uncorrected", "veto": {"w_s": 0.0015, "theta_sigma": 6.2},
                         "peri_r_ms": [2.0, 6.0], "peri_r_beats": "per_minute",
                         "distrusted_spans": [[120.0, 180.0], [600.0, 660.0]]}
    no_core = per_minute_cuff_entry({"route": "scalar"}, _PM_HR, None)
    assert no_core == {"route": "scalar", "distrusted_spans": [[120.0, 180.0], [600.0, 660.0]]}
    for e in (with_mask, no_core):
        validate_entry("gems_x_3", _pm_entry(e))


def test_per_minute_cuff_trust_needs_a_per_minute_train_and_a_readable_route() -> None:
    from gems_blanking_v2.emit.routing import per_minute_cuff_entry  # noqa: PLC0415

    with pytest.raises(ValueError, match="needs a per-minute HR train"):
        per_minute_cuff_entry(_SV, _WHOLE, None)
    with pytest.raises(ValueError, match="not one the spike consumer reads"):
        per_minute_cuff_entry({"route": "distrusted"}, _PM_HR, None)


_BLANK = _PM_HR["blank_s"]
_PMB = "must be the per-minute HR"


@pytest.mark.parametrize(("spike", "hr", "match"), [
    ({"route": "distrusted", "why": "x", "distrusted_spans": _BLANK}, None, "trusted elsewhere"),
    ({"route": "uncorrected", "distrusted_spans": [[120.0, 180.0]]}, None, _PMB),
    ({"route": "uncorrected", "distrusted_spans": _BLANK}, {**_WHOLE, "blank_s": _BLANK}, _PMB),
    ({"route": "uncorrected", "peri_r_ms": [2.0, 6.0], "peri_r_beats": "per_minute"}, _WHOLE,
     "needs a per-minute HR train"),
    ({"route": "uncorrected", "peri_r_ms": [2.0, 6.0], "peri_r_beats": "whole"}, None,
     "is not one of"),
])
def test_inconsistent_per_minute_cuff_entries_are_refused(
    spike: dict, hr: dict | None, match: str
) -> None:
    from gems_blanking_v2.emit.routing import validate_entry  # noqa: PLC0415

    with pytest.raises(ValueError, match=match):
        validate_entry("gems_x_3", _pm_entry(spike, hr))


def test_distrusted_spans_must_be_sorted_disjoint_and_non_empty() -> None:
    from gems_blanking_v2.emit.routing import validate_entry  # noqa: PLC0415

    unsorted, overlapping = [[600.0, 660.0], [120.0, 180.0]], [[120.0, 180.0], [170.0, 200.0]]
    for bad in ([[180.0, 120.0]], unsorted, overlapping):
        hr = {**_PM_HR, "blank_s": bad}
        with pytest.raises(ValueError, match="sorted, disjoint"):
            validate_entry("gems_x_3", _pm_entry({"route": "uncorrected", "distrusted_spans": bad},
                                                 hr))
    touching = [[120.0, 180.0], [180.0, 200.0]]  # half-open spans may touch
    validate_entry("gems_x_3", _pm_entry({"route": "uncorrected", "distrusted_spans": touching},
                                         {**_PM_HR, "blank_s": touching}))
