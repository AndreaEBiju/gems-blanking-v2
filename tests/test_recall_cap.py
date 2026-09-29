"""The provisional duration cap (ruling 2026-09-29).

The p99 of merged audit marks, recomputed after each round and recorded with its
source in the round report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from gems_blanking_v2.detect.recall import (
    DURATION_CAP_QUANTILE,
    DURATION_CAP_SOURCE,
    duration_cap,
    score_stored_round,
)
from gems_blanking_v2.io.store import GemsStore

from tests.test_recall import _reveal_with, _write_round

A, B = "plan_20260926T000000Z_0000000a", "plan_20260927T000000Z_0000000b"
NONE = [np.zeros((0, 2)), np.zeros((0, 2))]


def test_no_scored_round_means_no_cap(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    assert duration_cap(store) is None
    _write_round(store, A, [[[210, 211]], [[250, 252]]])  # committed but never scored
    assert duration_cap(store) is None


def test_marks_are_merged_before_their_durations_count(tmp_path: Path) -> None:
    """Overlapping marks are one artifact: 210-211 and 210.5-213 are one 3 s mark."""
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, A, [[[210, 211], [210.5, 213]], [[250, 251]]])
    score_stored_round(store, A, _reveal_with(NONE))
    cap = duration_cap(store)
    assert cap is not None and cap["n_marks"] == 2 and cap["unit"] == "merged"
    assert cap["cap_s"] == np.quantile([3.0, 1.0], DURATION_CAP_QUANTILE)


def test_the_cap_is_recomputed_over_every_scored_round(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, A, [[[210, 211]], [[250, 251]]])
    first = score_stored_round(store, A, _reveal_with(NONE))
    assert first.duration_cap is not None and first.duration_cap["rounds"] == [A]
    _write_round(store, B, [[[210, 230]], [[250, 251]]])  # a 20 s artifact arrives
    second = score_stored_round(store, B, _reveal_with(NONE))
    assert second.duration_cap is not None
    assert second.duration_cap["rounds"] == [A, B] and second.duration_cap["n_marks"] == 4
    assert second.duration_cap["cap_s"] > first.duration_cap["cap_s"]
    # the earlier round's record is not rewritten: its cap is the one it was scored with
    doc = json.loads(store.audit_score_path(A).read_text(encoding="utf-8"))
    assert doc["duration_cap"]["rounds"] == [A]


def test_the_report_and_the_record_carry_the_cap_and_its_source(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    marks = [[[200 + k, 200.1 + k] for k in range(99)] + [[300, 310]], [[250, 251]]]
    _write_round(store, A, marks)
    score = score_stored_round(store, A, _reveal_with(NONE))
    doc = json.loads(store.audit_score_path(A).read_text(encoding="utf-8"))
    cap = doc["duration_cap"]
    assert cap["source"] == DURATION_CAP_SOURCE and cap["provisional"] is True
    # linear p99 of 101 values is the 100th smallest: the 1 s mark; only 10 s is above
    assert cap["n_marks"] == 101 and cap["cap_s"] == 1.0 and cap["marks_above"] == 1
    text = score.report()
    assert f"duration cap    {cap['cap_s']:.1f} s = p99 of 101 merged marks" in text
    assert "rests on the 1 mark(s) above it   [PROVISIONAL]" in text
    assert "segment_indices.mat, does not exist" in text


def test_a_tuning_score_reports_the_cap_too(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, A, [[[210, 211]], [[250, 252]]])
    score_stored_round(store, A, _reveal_with(NONE))
    tune = score_stored_round(store, A, _reveal_with(NONE), mode="tuning")
    assert tune.duration_cap is not None and tune.duration_cap["rounds"] == [A]
