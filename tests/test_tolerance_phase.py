"""The phase test's statistics, against inputs with a known answer."""

from __future__ import annotations

import numpy as np
from tolerance_phase import _abs_spearman, bunching, circular_linear_r, permutation_p


def test_a_threshold_that_is_a_sinusoid_of_phase_scores_near_one() -> None:
    phase = np.linspace(0, 1, 40, endpoint=False)
    x = np.cos(2 * np.pi * (phase - 0.3))  # any starting point of the cycle
    assert circular_linear_r(x, phase) > 0.99


def test_an_unrelated_threshold_scores_near_zero() -> None:
    rng = np.random.default_rng(0)
    phase, x = rng.uniform(size=2000), rng.normal(size=2000)
    assert circular_linear_r(x, phase) < 0.08


def test_the_permutation_null_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(1)
    cov = rng.uniform(size=20)
    assert permutation_p(_abs_spearman(cov, cov), cov, cov, _abs_spearman, rng) < 0.01
    noise = rng.normal(size=20)
    assert permutation_p(_abs_spearman(noise, cov), noise, cov, _abs_spearman, rng) > 0.01


def test_values_spread_smoothly_over_the_grid_are_not_called_bunched() -> None:
    grid = [0.5 * 2 ** (i / 2) for i in range(12)]
    even = [grid[i % 6 + 3] for i in range(18)]  # 3 per bin across 6 bins
    assert bunching(even, np.random.default_rng(2))["p_as_full_or_fuller"] > 0.2


def test_a_pile_in_one_bin_is_called_bunched() -> None:
    grid = [0.5 * 2 ** (i / 2) for i in range(12)]
    piled = [grid[6]] * 12 + [grid[2], grid[3], grid[9], grid[10]]
    assert bunching(piled, np.random.default_rng(3))["p_as_full_or_fuller"] < 0.05
