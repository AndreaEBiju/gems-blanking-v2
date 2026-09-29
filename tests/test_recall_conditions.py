"""Conditions (ruling 2026-09-29): balanced by the planner, required by the gate.

The planner balances them across the eligible pool, and the gate needs at least 3
eligible spans of each.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.detect.recall import (
    GATE_CONDITIONS,
    MIN_SPANS_PER_CONDITION,
    condition_plan,
    pooled_gate,
)
from gems_blanking_v2.io.store import GemsStore
from hypothesis import given, settings
from hypothesis import strategies as st


def _round(store: GemsStore, pid: str, conditions: list[str], found: int = 30,
           covered: int | None = None, labelled: str = "2026-09-29T10:00:00+00:00",
           generation: str | None = None) -> None:
    """Write one scored gate round: its plan (the composition) and its score."""
    cov = found if covered is None else covered
    plan = store.audit_plan_path(pid)
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps({"plan_id": pid, "spans": [
        {"animal": "A", "condition": c, "recording_id": f"r{k}", "start_s": 0.0, "stop_s": 120.0}
        for k, c in enumerate(conditions)]}), encoding="utf-8")
    score = store.audit_score_path(pid)
    score.parent.mkdir(parents=True, exist_ok=True)
    score.write_text(json.dumps({
        "plan_id": pid, "mode": "gate",
        "provenance": {"labelled_at": labelled,
                       "generator": {"generation_sha256": generation or chain.generation_sha256()}},
        "per_span": [{"condition": c, "found": found, "covered": cov,
                      "chance_per_mark": [0.1] * found} for c in conditions]}), encoding="utf-8")


SR, BL = "stim_recovery", "baseline"


def test_a_bound_reached_on_one_condition_only_does_not_clear(tmp_path: Path) -> None:
    """150/150 on stim/recovery alone puts the bound at 0.9802 - and says why it fails."""
    store = GemsStore.initialise(tmp_path / "gems")
    _round(store, "plan_20260929T000000Z_0000000a", [SR] * 5)
    pg = pooled_gate(store)

    assert pg["lower_bound_one_sided_95"] >= 0.98
    assert pg["gate_cleared"] is False
    assert pg["spans_by_condition"] == {BL: 0, SR: 5}
    assert "reached on stim_recovery only" in pg["condition_note"]


def test_three_eligible_spans_of_each_condition_clear(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _round(store, "plan_20260929T000000Z_0000000a", [SR] * 3 + [BL] * 3, found=25)
    pg = pooled_gate(store)

    assert pg["lower_bound_one_sided_95"] >= 0.98 and pg["gate_cleared"] is True
    assert "condition_note" not in pg


def test_two_of_a_condition_is_one_short(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _round(store, "plan_20260929T000000Z_0000000a", [SR] * 4 + [BL] * 2, found=25)
    pg = pooled_gate(store)

    assert MIN_SPANS_PER_CONDITION == 3
    assert pg["gate_cleared"] is False and "2 baseline, 4 stim_recovery" in pg["condition_note"]


def test_the_next_rounds_conditions_balance_the_eligible_pool(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    assert condition_plan(store, 5) == (BL, SR, BL, SR, BL)  # empty pool: alternate
    _round(store, "plan_20260929T000000Z_0000000a", [SR] * 5)
    assert condition_plan(store, 5) == (BL,) * 5  # round 3's case: five baselines
    _round(store, "plan_20260929T010000Z_0000000b", [BL] * 5,
           labelled="2026-09-29T11:00:00+00:00", found=10)
    assert condition_plan(store, 5) == (BL, SR, BL, SR, BL)  # 5/5: alternate
    _round(store, "plan_20260929T020000Z_0000000c", [BL, SR, BL, SR, BL],
           labelled="2026-09-29T12:00:00+00:00", found=10)
    assert condition_plan(store, 5) == (SR, BL, SR, BL, SR)  # 8 bl / 7 sr


def test_only_eligible_rounds_count_and_their_scores_never_do(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _round(store, "plan_20260929T000000Z_0000000a", [SR] * 5, generation="f" * 64)
    assert condition_plan(store, 5) == (BL, SR, BL, SR, BL)  # another generator: not eligible

    a, b = (GemsStore.initialise(tmp_path / x) for x in ("a", "b"))
    _round(a, "plan_20260929T000000Z_0000000a", [SR, SR, BL, SR, SR], covered=30)
    _round(b, "plan_20260929T000000Z_0000000a", [SR, SR, BL, SR, SR], covered=12)
    assert condition_plan(a, 5) == condition_plan(b, 5) == (BL, BL, BL, SR, BL)


@given(st.integers(0, 20), st.integers(0, 20), st.integers(1, 10))
@settings(max_examples=300, deadline=None)
def test_each_span_takes_the_condition_with_fewer_spans(bl: int, sr: int, n: int) -> None:
    """Greedy balance: never picks the larger side, and ends within one if it can."""
    from gems_blanking_v2.detect import recall  # noqa: PLC0415

    orig = recall.pooled_gate
    try:
        recall.pooled_gate = lambda store: {"pooled_rounds": ["p"]}  # type: ignore[assignment]
        store = type("S", (), {"audit_plan_path": lambda self, pid: _Fake(bl, sr)})()
        picks = recall.condition_plan(store, n)  # type: ignore[arg-type]
    finally:
        recall.pooled_gate = orig
    counts = {BL: bl, SR: sr}
    for c in picks:
        other = SR if c == BL else BL
        assert counts[c] <= counts[other]
        counts[c] += 1
    if n >= abs(bl - sr):
        assert abs(counts[BL] - counts[SR]) <= 1
    assert set(picks) <= set(GATE_CONDITIONS)


class _Fake:
    """A plan path whose file holds ``bl`` baseline and ``sr`` stim_recovery spans."""

    def __init__(self, bl: int, sr: int) -> None:
        self.text = json.dumps({"spans": [{"condition": BL}] * bl + [{"condition": SR}] * sr})

    def is_file(self) -> bool:
        return True

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.text


@pytest.mark.parametrize("n", [1, 5])
def test_the_first_tie_goes_to_baseline(tmp_path: Path, n: int) -> None:
    assert condition_plan(GemsStore.initialise(tmp_path / "gems"), n)[0] == BL


def test_an_excluded_rounds_spans_do_not_count_toward_either_condition(tmp_path: Path) -> None:
    """Baselines scored under another generator are tuning data, not coverage."""
    store = GemsStore.initialise(tmp_path / "gems")
    _round(store, "plan_20260929T000000Z_0000000a", [BL] * 5, generation="f" * 64)
    _round(store, "plan_20260929T010000Z_0000000b", [SR] * 5, labelled="2026-09-29T11:00:00+00:00")
    pg = pooled_gate(store)

    assert pg["spans_by_condition"] == {BL: 0, SR: 5}
    assert pg["gate_cleared"] is False
