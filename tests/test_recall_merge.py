"""The merged scoring unit: one artifact = one connected run of committed marks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.detect.recall import (
    CURRENT_SCORING_UNIT,
    SpanInput,
    classification_path,
    close_pairs,
    load_round,
    merge_marks,
    record_classification,
    score_round,
    score_stored_round,
)
from gems_blanking_v2.io.store import GemsStore
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_recall import CANDS, PID, _reveal_with, _write_round


def test_overlapping_and_touching_marks_merge_and_a_gap_keeps_them_apart() -> None:
    marks = [[10.0, 11.0], [10.9, 12.0],    # overlap -> one
             [20.0, 21.0], [21.0, 21.5],    # touch (gap 0) -> one
             [30.0, 31.0], [31.0 + 1e-9, 32.0]]  # any positive gap -> two
    merged, groups = merge_marks(marks)

    np.testing.assert_array_equal(merged, [[10.0, 12.0], [20.0, 21.5], [30.0, 31.0],
                                           [31.0 + 1e-9, 32.0]])
    assert groups == [(0, 1), (2, 3), (4,), (5,)]


def test_a_chain_merges_transitively_whatever_the_committed_order() -> None:
    """A overlaps B, B overlaps C, A does not touch C: one artifact.

    Indices are the marks file's order, not time order.
    """
    merged, groups = merge_marks([[12.0, 14.0], [10.0, 11.0], [10.5, 12.5], [11.9, 13.0]])

    np.testing.assert_array_equal(merged, [[10.0, 14.0]])
    assert groups == [(0, 1, 2, 3)]


def test_close_but_separate_pairs_are_listed_not_merged() -> None:
    marks = [[1.0, 2.0], [2.05, 3.0], [3.15, 4.0], [3.9, 4.5]]
    merged, _ = merge_marks(marks)

    assert len(merged) == 3  # the 50 ms and 150 ms gaps stay separate; the overlap merges
    assert close_pairs(marks) == [(0, 1, pytest.approx(0.05))]


@st.composite
def _marks(draw: st.DrawFn) -> list[list[float]]:
    starts = draw(st.lists(st.integers(0, 400), min_size=0, max_size=25))
    out = []
    for s0 in starts:
        dur = draw(st.integers(1, 40))
        out.append([s0 / 10.0, (s0 + dur) / 10.0])
    return out


@given(_marks())
@settings(max_examples=300, deadline=None)
def test_the_merge_preserves_the_union_and_leaves_no_touching_artifacts(
    marks: list[list[float]],
) -> None:
    merged, groups = merge_marks(marks)
    a = np.asarray(marks, dtype=float).reshape(-1, 2)

    assert sorted(i for g in groups for i in g) == list(range(len(a)))  # a partition
    assert all(merged[k + 1, 0] > merged[k, 1] for k in range(len(merged) - 1))
    for (m0, m1), g in zip(merged, groups, strict=True):
        assert m0 == min(a[i, 0] for i in g) and m1 == max(a[i, 1] for i in g)
    grid = np.arange(0, 45, 0.05) + 0.025  # union preserved, sampled
    inside = lambda iv: ((grid[:, None] >= iv[:, 0]) & (grid[:, None] < iv[:, 1])).any(1)  # noqa: E731
    if len(a):
        assert np.array_equal(inside(a), inside(merged))
    again, _ = merge_marks(merged)
    np.testing.assert_array_equal(again, merged)  # idempotent


def _span(arts: list, cands: list) -> SpanInput:
    return SpanInput("p_s1", "r1", "J", "baseline", 0.0, 120.0,
                     np.asarray(arts, dtype=float).reshape(-1, 2),
                     np.asarray(cands, dtype=float).reshape(-1, 2))


def test_a_sliver_overlapping_a_covered_mark_is_not_a_miss_in_the_merged_unit() -> None:
    """The viewport case: one artifact marked as a big mark plus an edge sliver."""
    arts, cands = [[50.0, 51.0], [50.95, 51.03]], [[50.2, 50.4]]
    committed = score_round("p", [_span(arts, cands)])
    merged = score_round("p", [_span(arts, cands)], unit="merged")

    assert (committed.found, committed.covered) == (2, 1)
    assert (merged.found, merged.covered) == (1, 1)
    assert merged.artifacts[0].merged_from == (0, 1)
    assert merged.artifacts[0].miss_id == "p_s1#m0" and committed.artifacts[1].miss_id == "p_s1#1"
    assert merged.per_span[0]["marks_committed"] == 2 and merged.per_span[0]["found"] == 1
    assert "MERGED" in merged.report() and "AS COMMITTED" in committed.report()
    assert merged.to_json()["scoring_unit"] == "merged"


def test_the_merge_is_independent_of_the_candidates() -> None:
    arts = [[10.0, 11.0], [10.9, 12.0], [30.0, 31.0]]
    a = score_round("p", [_span(arts, [])], unit="merged")
    b = score_round("p", [_span(arts, [[0.0, 120.0]])], unit="merged")

    assert [(x.start_s, x.stop_s, x.merged_from) for x in a.artifacts] == \
           [(x.start_s, x.stop_s, x.merged_from) for x in b.artifacts]


def test_gate_evidence_is_scored_in_the_declared_unit_only(tmp_path: Path) -> None:
    """Rounds 1-2 declared nothing (as committed); their merged score is tuning."""
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [210.9, 212]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    with pytest.raises(ValueError, match="declared the 'as_committed' unit"):
        score_stored_round(store, PID, _reveal_with(CANDS), unit="merged")
    gate = score_stored_round(store, PID, _reveal_with(CANDS))
    tune = score_stored_round(store, PID, _reveal_with(CANDS), mode="tuning", unit="merged")

    assert gate.unit == "as_committed" and gate.found == 3
    assert tune.unit == "merged" and tune.found == 2 and tune.mode == "tuning"
    assert tune.provenance["declared_scoring_unit"] == "as_committed"


def test_a_plan_that_declares_merged_is_gate_scored_merged(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [210.9, 212]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    path = store.audit_plan_path(PID)
    plan = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**plan, "scoring_unit": CURRENT_SCORING_UNIT}), encoding="utf-8")

    score = score_stored_round(store, PID, _reveal_with(CANDS))
    assert CURRENT_SCORING_UNIT == "merged"
    assert score.unit == "merged" and score.found == 2 and score.mode == "gate"


def test_scoring_merged_never_rewrites_the_committed_marks(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [210.9, 212]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    _, spans = load_round(store, PID)
    files = [f for sp in spans
             for f in store.audit_dir(sp["animal"], sp["recording_id"]).glob("*_blind_marks.json")]
    assert files
    before = {f: f.read_bytes() for f in files}

    score_stored_round(store, PID, _reveal_with(CANDS), mode="tuning", unit="merged")
    assert {f: f.read_bytes() for f in before} == before


# --- the labeller's classification of a miss, beside the score (2026-09-29) ----------


def test_a_classification_is_recorded_beside_the_score_and_never_touches_the_marks(
    tmp_path: Path,
) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    score_stored_round(store, PID, _reveal_with(CANDS))
    _, spans = load_round(store, PID)
    marks = [f for sp in spans
             for f in store.audit_dir(sp["animal"], sp["recording_id"]).glob("*_blind_marks.json")]
    before = {f: f.read_bytes() for f in marks}
    score_before = store.audit_score_path(PID).read_bytes()
    at = datetime(2026, 9, 29, 9, 0, tzinfo=UTC)

    path = record_classification(store, PID, f"{PID}_s1#0", classification="artifact",
                                 words="a real pop, I would blank it", by="Andrea", at=at)

    doc = json.loads(path.read_text(encoding="utf-8"))
    assert path == classification_path(store, PID)
    assert doc[f"{PID}_s1#0"] == {"classification": "artifact", "by": "Andrea",
                                  "words": "a real pop, I would blank it", "at": at.isoformat()}
    assert {f: f.read_bytes() for f in marks} == before
    assert store.audit_score_path(PID).read_bytes() == score_before
    with pytest.raises(ValueError, match="already classified"):
        record_classification(store, PID, f"{PID}_s1#0", classification="not_artifact",
                              words="changed my mind", by="Andrea", at=at)
    with pytest.raises(ValueError, match="not an artifact"):
        record_classification(store, PID, f"{PID}_s9#0", classification="artifact",
                              words="x", by="Andrea", at=at)
    with pytest.raises(ValueError, match="one of"):
        record_classification(store, PID, f"{PID}_s1#1", classification="maybe",  # type: ignore[arg-type]
                              words="x", by="Andrea", at=at)
    with pytest.raises(ValueError, match="timezone-aware"):
        record_classification(store, PID, f"{PID}_s1#1", classification="separate",
                              words="x", by="Andrea", at=datetime(2026, 9, 29))
    with pytest.raises(ValueError, match="her words"):
        record_classification(store, PID, f"{PID}_s1#1", classification="separate",
                              words="  ", by="Andrea", at=at)
