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
    with pytest.raises(ValueError, match="come together"):
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
