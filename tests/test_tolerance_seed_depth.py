"""Bisection of one seed's crossing on T's existing amplitude grid."""

from __future__ import annotations

import math

import pytest
from tolerance_seed_depth import CONFIRM_STEPS, crossing, next_step

GRID23 = [round(0.5 * 2 ** (i / 2), 6) for i in range(23)]
GRID28 = [round(0.5 * 2 ** (-i / 2), 6) for i in range(5, 0, -1)] + GRID23


def _run(grid: list[float], past: callable) -> tuple[str, dict[float, bool], int]:  # type: ignore[valid-type]
    """Drive next_step against an oracle until the seed is terminal."""
    obs: dict[float, bool] = {}
    n = 0
    while True:
        status, amps = next_step(grid, obs)
        if status != "pending":
            return status, obs, n
        for a in amps:
            assert a not in obs, "never re-runs a known point"
            obs[a] = past(a)
            n += 1


@pytest.mark.parametrize("grid", [GRID23, GRID28], ids=["23", "28"])
def test_every_threshold_position_is_found_exactly(grid: list[float]) -> None:
    """For a monotone curve, bisection lands on the true crossing at every position."""
    bound = math.ceil(math.log2(len(grid) + 1)) + 2
    for t in range(len(grid) + 1):  # t == len(grid): never past threshold
        status, obs, n = _run(grid, lambda a, t=t: grid.index(a) >= t)
        if t == 0:
            assert status == "censored_below_grid"
        elif t == len(grid):
            assert status == "censored_above_grid"
        else:
            assert status == "resolved"
            assert crossing(grid, obs) == grid[t]
        assert n <= bound, f"t={t}: {n} evaluations, bound {bound}"


def test_a_resolved_seed_is_confirmed_both_sides() -> None:
    """The two confirmation points are CONFIRM_STEPS either side of the bracket."""
    t = 10
    _, obs, _ = _run(GRID23, lambda a: GRID23.index(a) >= t)
    assert GRID23[t - 1 - CONFIRM_STEPS] in obs
    assert GRID23[t + CONFIRM_STEPS] in obs


def test_a_curve_that_flips_back_is_non_monotone_not_resolved() -> None:
    """Past threshold at t, but not again two steps above: bisection must not trust it."""
    t = 10

    def past(a: float) -> bool:
        i = GRID23.index(a)
        return i >= t and i != t + CONFIRM_STEPS

    status, obs, _ = _run(GRID23, past)
    assert status == "non_monotone"
    assert crossing(GRID23, obs) is None


def test_reused_observations_are_not_rerun() -> None:
    """Seeds 1-5 arrive with their replicate bracket already known."""
    t = 12
    known = {GRID23[i]: i >= t for i in (11, 12, 13)}
    status, amps = next_step(GRID23, known)
    assert status == "pending"
    assert not set(amps) & set(known)
