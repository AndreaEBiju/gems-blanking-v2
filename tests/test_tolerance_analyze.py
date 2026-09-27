"""Censoring in T's replicate pass: which seeds need the bracket extended, and where."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tolerance_analyze import (
    BREATHING_CRITERION,
    BREATHING_SHIFT_S,
    _censored_side,
    _crossing,
    _load,
    apply_breathing_displacement,
    censored,
    placement_summary,
)

A, B, C = 1.0, 1.414214, 2.0
"""Three consecutive sqrt(2) grid points: a 3-point bracket."""


@pytest.mark.parametrize(
    ("direction", "changed", "expected"),
    [
        # up: tolerance is the lowest amplitude that changed the output
        ("up", (0, 0, 1), None),       # crossing at the TOP edge: resolved to (B, C]
        ("up", (0, 1, 1), None),       # crossing mid-bracket: resolved
        ("up", (1, 1, 1), "below"),    # changed already at A: tolerance <= A
        ("up", (0, 0, 0), "above"),    # never changed: tolerance > C
        # down (clip): tolerance is the lowest rail at which output is unchanged
        ("down", (1, 1, 0), None),     # unchanged only at the top rail: resolved
        ("down", (1, 0, 0), None),     # resolved mid-bracket
        ("down", (0, 0, 0), "below"),  # unchanged even at A: tolerance <= A
        ("down", (1, 1, 1), "above"),  # changed at every rail: tolerance > C
    ],
)
def test_only_the_bottom_edge_and_no_crossing_are_censored(
    direction: str, changed: tuple[int, int, int], expected: str | None
) -> None:
    curve = list(zip((A, B, C), changed, strict=True))
    assert _censored_side(curve, direction) == expected


def test_a_top_edge_crossing_is_pinned_to_one_step() -> None:
    """The case the first version got wrong: it is bracketed, not censored."""
    curve = [(A, 0), (B, 0), (C, 1)]
    assert _crossing(curve, "up") == C
    assert _censored_side(curve, "up") is None


def test_the_bottom_edge_is_the_seeds_own() -> None:
    """A seed whose lowest run failed has B as its bottom, and B is then censored."""
    curve = [(B, 1), (C, 1)]
    assert _censored_side(curve, "up") == "below"


def _write_replicate(
    tmp_path: Path, changed_by_seed: dict[int, tuple[int, int, int]], kind: str = "step"
) -> list[float]:
    """Write one key's 3-point bracket at grid[3:6] of a 9-point grid, T_hardware."""
    grid = [round(0.5 * 2 ** (i / 2), 6) for i in range(9)]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"kind_direction": {"step": "up", "clip": "down"}, "amp_sigma": grid}),
        encoding="utf-8",
    )
    rows = [
        {"host_tag": "host1_JEL", "kind": kind, "dur_s": 0.5, "chan_set": "nerve",
         "seed": seed, "amp_sigma": grid[3 + i], "ok": True, "T_hardware_changed": v}
        for seed, changed in changed_by_seed.items()
        for i, v in enumerate(changed)
    ]
    (tmp_path / "sweep_manifest_replicate.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    return grid


def _extension(tmp_path: Path) -> dict[str, dict[str, list]]:
    censored(tmp_path)
    return json.loads((tmp_path / "crossings_extend.json").read_text(encoding="utf-8"))


def test_a_left_censored_seed_is_extended_downward_only(tmp_path: Path) -> None:
    grid = _write_replicate(tmp_path, {1: (1, 1, 1), 2: (0, 1, 1)})
    (entry,) = _extension(tmp_path).values()
    assert entry["amps"] == grid[1:3]
    assert entry["seeds"] == [1]


def test_an_up_seed_censored_above_is_on_the_insensitive_side(tmp_path: Path) -> None:
    """Tolerance above the bracket cannot move the sensitive end - no points for it."""
    _write_replicate(tmp_path, {1: (0, 0, 0), 2: (0, 1, 1)})
    assert _extension(tmp_path) == {}


def test_a_clip_seed_censored_above_is_extended_upward(tmp_path: Path) -> None:
    """Clip's sensitive end is the HIGH rail: changed at every rail, top included."""
    grid = _write_replicate(tmp_path, {1: (1, 1, 1), 2: (1, 0, 0)}, kind="clip")
    (entry,) = _extension(tmp_path).values()
    assert entry["amps"] == grid[6:8]
    assert entry["seeds"] == [1]


def test_a_clip_seed_censored_below_is_on_the_insensitive_side(tmp_path: Path) -> None:
    _write_replicate(tmp_path, {1: (0, 0, 0), 2: (1, 0, 0)}, kind="clip")
    assert _extension(tmp_path) == {}


def test_top_edge_crossings_need_no_extension(tmp_path: Path) -> None:
    _write_replicate(tmp_path, {1: (0, 0, 1), 2: (0, 1, 1)})
    assert _extension(tmp_path) == {}


def test_a_degenerate_consumer_cannot_extend_the_bracket(tmp_path: Path) -> None:
    """The degenerate mmc criterion must not buy extension points by censoring."""
    _write_replicate(tmp_path, {1: (0, 0, 1)})
    rows = json.loads((tmp_path / "sweep_manifest_replicate.json").read_text(encoding="utf-8"))
    for r in rows:
        r["mmc_changed"] = 1  # censored below, on every seed
    (tmp_path / "sweep_manifest_replicate.json").write_text(json.dumps(rows), encoding="utf-8")
    assert _extension(tmp_path) == {}


# --- placement summary: the tolerance across placements -------------------------

G = [round(0.5 * 2 ** (i / 2), 6) for i in range(12)]


def _up_curve(t: int, lo: int, hi: int) -> list[tuple[float, int]]:
    """Build a seed that ran grid[lo..hi] with its up-crossing at index t."""
    return [(G[i], int(i >= t)) for i in range(lo, hi + 1)]


def test_the_sensitive_end_of_an_up_kind_is_its_low_tail() -> None:
    curves = [_up_curve(t, 2, 9) for t in (3, 5, 5, 6, 6, 7, 7, 8, 8, 9)]
    s = placement_summary(curves, "up")
    assert s["sensitive_end_sigma"] == G[3]
    assert s["sensitive_end_censored"] is False


def test_a_seed_censored_at_the_sensitive_end_makes_the_number_a_bound() -> None:
    """Dropping it - what the median-of-resolved did - would report a wider margin."""
    curves = [_up_curve(2, 2, 9)] + [_up_curve(t, 2, 9) for t in (5, 6, 6, 7, 7, 8, 8, 9, 9)]
    s = placement_summary(curves, "up")
    assert s["sensitive_end_sigma"] == G[2]
    assert s["sensitive_end_censored"] is True
    assert s["n_censored_below"] == 1


def test_the_sensitive_end_of_clip_is_its_high_tail() -> None:
    """Clip tolerance is the rail above which nothing changes: sensitive = HIGH rail."""
    def clip(t: int) -> list[tuple[float, int]]:  # unchanged (0) from rail t upward
        return [(G[i], int(i < t)) for i in range(2, 10)]

    curves = [clip(t) for t in (3, 4, 4, 5, 5, 5, 6, 6, 7, 9)]
    s = placement_summary(curves, "down")
    assert s["sensitive_end_sigma"] == G[8]  # lower bound of the most sensitive seed
    assert s["sensitive_end_quantile"] == 0.9


def test_the_resolved_spread_is_counted_in_sqrt2_steps() -> None:
    curves = [_up_curve(t, 2, 11) for t in (4, 8)]
    s = placement_summary(curves, "up")
    assert s["resolved_spread_sqrt2_steps"] == 4.0


def test_a_seed_resolved_by_the_extension_is_no_longer_censored(tmp_path: Path) -> None:
    """The extension continues the replicate curve; both readers must see it."""
    grid = _write_replicate(tmp_path, {1: (1, 1, 1), 2: (0, 1, 1)})
    assert _extension(tmp_path)  # seed 1 censored below
    ext = [{"host_tag": "host1_JEL", "kind": "step", "dur_s": 0.5, "chan_set": "nerve",
            "seed": 1, "amp_sigma": a, "ok": True, "T_hardware_changed": v}
           for a, v in ((grid[1], 0), (grid[2], 1))]
    (tmp_path / "sweep_manifest_extend.json").write_text(json.dumps(ext), encoding="utf-8")
    assert _extension(tmp_path) == {}


def test_five_placements_support_about_the_17th_percentile_and_say_so() -> None:
    """The order-statistics caveat travels with the number, not in a footnote."""
    curves = [_up_curve(t, 2, 9) for t in (4, 5, 6, 7, 8)]
    s = placement_summary(curves, "up")
    assert s["supported_percentile"] == 17
    assert s["label"] == "~17th percentile from the sensitive end, 5 placements"


def test_a_criterion_that_flips_back_is_counted_on_the_row() -> None:
    """Changed low, unchanged higher: conservative reading, flagged instability."""
    flipping = [(G[2], 1), (G[3], 0), (G[4], 0), (G[5], 1)]
    s = placement_summary([flipping, _up_curve(4, 2, 9)], "up")
    assert s["n_non_monotone"] == 1


def test_a_single_placement_row_is_not_gate_eligible() -> None:
    """A field the gate reads: one placement says nothing about the tail."""
    from tolerance_analyze import gate_eligible  # noqa: PLC0415

    one = placement_summary([_up_curve(5, 2, 9)], "up")
    six = placement_summary([_up_curve(t, 2, 9) for t in (3, 4, 5, 6, 7, 8)], "up")
    assert gate_eligible(one) is False
    assert gate_eligible(six) is True


# --- breathing: the displacement criterion (Andrea, 2026-09-26) -------------


def _brow(changed: int, shift: float | None = None) -> dict:
    row = {"host_tag": "h", "kind": "step", "dur_s": 1.0, "chan_set": "hr",
           "amp_sigma": 2.0, "seed": 1, "breathing_changed": changed}
    if shift is not None:
        row["breathing_maxShift"] = shift
    return row


@pytest.mark.parametrize(
    ("changed", "shift", "expected"),
    [(0, 0.08, 1), (0, 0.07, 0), (0, 0.06, 0), (2, 0.0, 2), (1, 0.5, 1)],
)
def test_a_breath_displaced_by_more_than_70_ms_is_a_change(
    changed: int, shift: float, expected: int
) -> None:
    """Half an R-R interval (~140-165 ms) separates 'same beat' from 'moved a beat'."""
    row = apply_breathing_displacement(_brow(changed, shift))
    assert row["breathing_changed"] == expected
    assert row["breathing_changed_count_only"] == changed


def test_the_displacement_rule_is_idempotent() -> None:
    row = apply_breathing_displacement(_brow(0, 0.09))
    again = apply_breathing_displacement(dict(row))
    assert again == row


def test_a_breathing_row_without_its_displacement_cannot_be_judged() -> None:
    with pytest.raises(KeyError, match="breathing_maxShift"):
        apply_breathing_displacement(_brow(0))


def test_every_reader_gets_the_criterion_through_load(tmp_path: Path) -> None:
    (tmp_path / "sweep_manifest.json").write_text(
        json.dumps([_brow(0, 0.2), {"ok": True, "mmc_changed": 0}]), encoding="utf-8")
    rows = _load(tmp_path, "sweep_manifest.json")
    assert rows[0]["breathing_changed"] == 1 and rows[1] == {"ok": True, "mmc_changed": 0}


def test_the_criterion_change_is_recorded_with_its_date() -> None:
    assert BREATHING_SHIFT_S == 0.07
    assert BREATHING_CRITERION["changed_on"] == "2026-09-26"
    assert "count only" in BREATHING_CRITERION["was"]
