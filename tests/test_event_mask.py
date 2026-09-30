"""Tests for :mod:`gems_blanking_v2.derive.event_mask` (ruling 2026-09-30, item 4c)."""

from __future__ import annotations

import numpy as np
import pytest
from gems_blanking_v2.derive.event_mask import (
    CONSUMER_EDGE_PAD_S,
    apply_mask,
    effective_exposure,
    gated_event_mask,
)
from gems_blanking_v2.detect import chain

FS = 24414.0625


def _t(seed: int = 0) -> np.ndarray:
    """Band-limited noise standing in for T: 2 uV sigma in 300-3000 Hz."""
    from tests.conftest import make_common_mode  # noqa: PLC0415

    return make_common_mode(FS, 10.0, 300.0, 3000.0, sigma_uv=2.0, seed=seed)


def _spike(n: int, t0: float, amp: float) -> np.ndarray:
    tt = np.arange(n) / FS
    return amp * np.exp(-0.5 * ((tt - t0) / (0.0006 / 2.3548)) ** 2)


def test_an_event_where_t_reaches_the_gate_is_masked_one_and_a_half_ms_each_side() -> None:
    t = _t()
    t = t + _spike(t.size, 3.0, 30.0)  # ~15 sigma at the event
    mask, sigma = gated_event_mask(t, [3.0], FS)
    assert sigma == pytest.approx(2.0, rel=0.2)
    idx = np.flatnonzero(mask)
    assert idx.size == 2 * int(round(0.0015 * FS)) + 1  # the ruling's +/-1.5 ms, stated
    assert idx.min() / FS == pytest.approx(3.0 - 0.0015, abs=1 / FS)


def test_an_event_where_t_stays_below_the_gate_is_not_masked() -> None:
    t = _t()
    mask, _sigma = gated_event_mask(t, [2.0, 5.0], FS)  # noise alone: < 4 sigma there
    assert not mask.any()


def test_the_gate_looks_only_within_the_events_own_width() -> None:
    t = _t()
    t = t + _spike(t.size, 4.004, 30.0)  # 4 ms after the event
    mask, _sigma = gated_event_mask(t, [4.0], FS)
    assert not mask.any()


def test_the_consumer_loses_the_mask_dilated_by_its_edge_pad() -> None:
    mask = np.zeros(int(FS), dtype=bool)
    mask[1000:1073] = True
    eff = effective_exposure(mask, FS)
    p = int(round(CONSUMER_EDGE_PAD_S * FS))
    assert eff.sum() == 73 + 2 * p
    assert eff[1000 - p] and not eff[1000 - p - 1]


def test_masked_samples_are_nan_never_zero() -> None:
    t = _t()
    mask = np.zeros(t.size, dtype=bool)
    mask[500:600] = True
    out = apply_mask(t, mask)
    assert np.isnan(out[500:600]).all() and not np.any(out == 0.0)
    assert np.isfinite(t[500:600]).all()  # the input is never written


def test_the_mask_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.derive.event_mask" not in chain.generation_modules()


def test_sigma_is_the_consumers_robust_estimate_not_inflated_by_spikes() -> None:
    """300 large spikes inflate a standard deviation; median(|x|)/0.6745 barely moves."""
    t = _t()
    rng = np.random.default_rng(4)
    for t0 in rng.uniform(0.1, 9.9, 300):
        t = t + _spike(t.size, float(t0), 60.0)
    _mask, sigma = gated_event_mask(t, [], FS)
    assert sigma == pytest.approx(2.0, rel=0.2)
    assert float(np.std(t)) > 1.5 * sigma
