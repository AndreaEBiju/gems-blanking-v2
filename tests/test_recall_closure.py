"""How a miss is closed (ruling 2026-09-29): fixed, not_target, accepted_limitation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.detect.recall import (
    check_next_round,
    last_fix_at,
    miss_closures,
    next_round_gate,
    record_classification,
    record_miss_closure,
    score_stored_round,
)
from gems_blanking_v2.io.store import GemsStore

from tests.conftest import make_band_z
from tests.test_recall import PID, _doc, _within_budget, _write_round

AT = "2026-09-29T09:00:00+00:00"
FIX = {"task07_fix": "chain change 0abda6b", "fixed_at": AT}
LIMIT = {"closure": "accepted_limitation", "reason": "no fix without fitting z_enter",
         "closed_at": AT}
NOT_TARGET = {"closure": "not_target", "reason": "she would not blank it", "closed_at": AT}
SAYS_NOT = {"s1#1": {"is_artifact": {"classification": "not_artifact"}}}


def test_each_closure_closes_a_miss_and_an_unclosed_miss_refuses() -> None:
    doc = _doc([None, "threshold"])
    assert next_round_gate(doc, {"s1#1": FIX})[0] is True
    assert next_round_gate(doc, {"s1#1": LIMIT})[0] is True
    assert next_round_gate(doc, {"s1#1": NOT_TARGET}, SAYS_NOT)[0] is True
    ok, why = next_round_gate(doc, {})
    assert not ok and "s1#1 - no closure recorded" in why


def test_not_target_needs_her_not_an_artifact_classification() -> None:
    doc = _doc([None, "threshold"])
    ok, why = next_round_gate(doc, {"s1#1": NOT_TARGET})
    assert not ok and "classification" in why
    says_yes = {"s1#1": {"is_artifact": {"classification": "artifact"}}}
    assert next_round_gate(doc, {"s1#1": NOT_TARGET}, says_yes)[0] is False


@pytest.mark.parametrize("broken", [
    {**LIMIT, "reason": "  "}, {k: v for k, v in LIMIT.items() if k != "closed_at"},
    {**FIX, "fixed_at": None}, {"task07_fix": "  ", "fixed_at": AT},
])
def test_a_closure_without_its_reason_or_time_does_not_close(broken: dict) -> None:
    assert next_round_gate(_doc([None, "threshold"]), {"s1#1": broken})[0] is False


def test_the_gate_names_which_closure_each_miss_got() -> None:
    doc = _doc([None, "threshold", "gated", "generator_blind_spot"])
    res = {"s1#1": FIX, "s1#2": LIMIT, "s1#3": NOT_TARGET}
    says = {"s1#3": {"is_artifact": {"classification": "not_artifact"}}}
    ok, why = next_round_gate(doc, res, says)

    assert ok and "1 fixed, 1 not_target, 1 accepted_limitation" in why
    assert "counted again if they recur: s1#2" in why
    assert {m: k for m, (k, _w) in miss_closures(doc, res, says).items()} == {
        "s1#1": "fixed", "s1#2": "accepted_limitation", "s1#3": "not_target"}


def _diagnosed_round(store: GemsStore) -> list[str]:
    """Score a round whose misses carry a diagnosis (z traces were revealed)."""
    _write_round(store, PID, [[[210, 211], [240, 241]], [[250, 252], [270, 271]]],
                 committed_at="2026-09-27T10:05:00+00:00",
                 revealed=[np.array([[210.5, 210.6]]), np.array([[251.0, 251.1]])])

    def reveal(span: dict) -> tuple:
        k = int(span["span_id"].rsplit("_s", 1)[1]) - 1
        return ([np.array([[210.5, 210.6]]), np.array([[251.0, 251.1]])][k],
                [make_band_z("100-300", 600.0)])

    score_stored_round(store, PID, reveal)
    doc = json.loads(store.audit_score_path(PID).read_text(encoding="utf-8"))
    return [a["id"] for a in doc["artifacts"] if not a["covered"]]


def test_closing_in_the_store_lets_the_next_round_be_drawn(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _within_budget(store)
    misses = _diagnosed_round(store)
    at = datetime(2026, 9, 29, 9, 0, tzinfo=UTC)
    assert len(misses) == 2 and check_next_round(store)[0] is False

    record_miss_closure(store, PID, misses[0], closure="accepted_limitation",
                        reason="reaching it would pin z_enter to the miss", at=at)
    assert "no closure recorded" in check_next_round(store)[1]
    record_classification(store, PID, misses[1], classification="not_artifact",
                          words="a heartbeat, not an artifact", by="Andrea", at=at)
    record_miss_closure(store, PID, misses[1], closure="not_target",
                        reason="Andrea: not an artifact she would blank", at=at)

    ok, why = check_next_round(store)
    assert ok and "1 not_target, 1 accepted_limitation" in why
    assert last_fix_at(store) is None  # a closure is not a generator change


def test_record_miss_closure_refuses_what_it_must(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    misses = _diagnosed_round(store)
    at = datetime(2026, 9, 29, 9, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="needs her recorded"):
        record_miss_closure(store, PID, misses[0], closure="not_target", reason="x", at=at)
    with pytest.raises(ValueError, match="not a miss"):
        record_miss_closure(store, PID, f"{PID}_s1#0", closure="accepted_limitation",
                            reason="x", at=at)
    with pytest.raises(ValueError, match="written reason"):
        record_miss_closure(store, PID, misses[0], closure="accepted_limitation",
                            reason=" ", at=at)
    with pytest.raises(ValueError, match="timezone-aware"):
        record_miss_closure(store, PID, misses[0], closure="accepted_limitation", reason="x",
                            at=datetime(2026, 9, 29))
    with pytest.raises(ValueError, match="not_target or accepted_limitation"):
        record_miss_closure(store, PID, misses[0], closure="fixed",  # type: ignore[arg-type]
                            reason="x", at=at)
    record_miss_closure(store, PID, misses[0], closure="accepted_limitation", reason="x", at=at)
    with pytest.raises(ValueError, match="never overwritten"):
        record_miss_closure(store, PID, misses[0], closure="accepted_limitation",
                            reason="y", at=at)


def test_an_undiagnosed_miss_cannot_be_accepted_as_a_limitation(tmp_path: Path) -> None:
    from tests.test_recall import CANDS, _reveal_with  # noqa: PLC0415

    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [240, 241]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    score_stored_round(store, PID, _reveal_with(CANDS))  # no traces: undiagnosed
    doc = json.loads(store.audit_score_path(PID).read_text(encoding="utf-8"))
    miss = next(a["id"] for a in doc["artifacts"] if not a["covered"])
    with pytest.raises(ValueError, match="undiagnosed"):
        record_miss_closure(store, PID, miss, closure="accepted_limitation", reason="x",
                            at=datetime(2026, 9, 29, 9, 0, tzinfo=UTC))
