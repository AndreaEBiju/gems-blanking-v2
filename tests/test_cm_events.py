"""Tests for :mod:`gems_blanking_v2.derive.cm_events` (the simultaneity filter)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.cm_events import find_events
from gems_blanking_v2.detect import chain

from tests.conftest import make_shared_ground

FS = 24414.0625


@pytest.fixture(scope="module")
def world():  # noqa: ANN201 - a NamedTuple from conftest
    return make_shared_ground(FS, 10.0, n_transients=15, seed=13)


def _nearest(t: np.ndarray, x: float) -> float:
    return float(np.min(np.abs(t - x))) if t.size else np.inf


def test_every_identical_transient_is_found_at_its_instant(world) -> None:  # noqa: ANN001
    ev = find_events(world.rec)
    for t0 in world.transients_s:
        assert _nearest(ev.t_s, float(t0)) <= 0.0002
    assert ev.channel_amp_uv.shape == (ev.t_s.size, 9)
    assert np.all(ev.width_s < 0.003)


def _add(rec, where: list[str], t0: float, amp: float, sigma_s: float, sign=None):  # noqa: ANN001, ANN202
    data = np.array(rec.data, dtype=np.float64)
    tt = np.arange(data.shape[0]) / FS
    pulse = amp * np.exp(-0.5 * ((tt - t0) / sigma_s) ** 2)
    for k, c in enumerate(rec.channels):
        if c.name in where:
            data[:, c.index] += pulse * (1.0 if sign is None else sign[k])
    return replace(rec, data=data)


def test_a_spike_on_one_channel_is_not_an_event(world) -> None:  # noqa: ANN001
    rec = _add(world.rec, ["LVN2"], 5.3333, 400.0, 0.0005)
    assert _nearest(find_events(rec).t_s, 5.3333) > 0.002


def test_a_wide_common_transient_is_not_an_event(world) -> None:  # noqa: ANN001
    names = [c.name for c in world.rec.channels]
    rec = _add(world.rec, names, 6.6667, 400.0, 0.003)  # ~7 ms at half height
    assert _nearest(find_events(rec).t_s, 6.6667) > 0.002


def test_mixed_polarity_is_not_an_event(world) -> None:  # noqa: ANN001
    names = [c.name for c in world.rec.channels]
    rec = _add(world.rec, names, 7.7777, 400.0, 0.0005, sign=[1, -1, 1, -1, 1, -1, 1, -1, 1])
    assert _nearest(find_events(rec).t_s, 7.7777) > 0.002


def test_the_event_finder_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.derive.cm_events" not in chain.generation_modules()


def test_a_transient_wider_than_three_ms_is_not_an_event(world) -> None:  # noqa: ANN001
    """Reject a 4 ms flat-topped common pulse, which passes 300-3000 Hz on its edges.

    Its common average is 4 ms wide at half height, over the 3 ms limit.
    """
    data = np.array(world.rec.data, dtype=np.float64)
    tt = np.arange(data.shape[0]) / FS
    edge = 0.0002
    rise = 0.5 * (1 + np.tanh((tt - 3.3) / edge))
    pulse = 400.0 * (rise - 0.5 * (1 + np.tanh((tt - 3.304) / edge)))
    data += pulse[:, None]
    ev = find_events(replace(world.rec, data=data))
    assert _nearest(ev.t_s, 3.302) > 0.003 and _nearest(ev.t_s, 3.3) > 0.001


def test_channels_that_disagree_in_time_are_not_an_event(world) -> None:  # noqa: ANN001
    """Extrema staggered 0.1 ms per channel (0-0.8 ms): fewer than 7 within 0.2 ms."""
    data = np.array(world.rec.data, dtype=np.float64)
    tt = np.arange(data.shape[0]) / FS
    for k in range(9):
        data[:, k] += 400.0 * np.exp(-0.5 * ((tt - (4.4 + 0.0001 * k)) / 0.0005) ** 2)
    ev = find_events(replace(world.rec, data=data))
    assert _nearest(ev.t_s, 4.4004) > 0.001


def test_channels_of_very_different_size_are_not_an_event(world) -> None:  # noqa: ANN001
    """Five channels at 400 uV and four at 150 uV: only five within 0.5-2x of the median."""
    data = np.array(world.rec.data, dtype=np.float64)
    tt = np.arange(data.shape[0]) / FS
    pulse = np.exp(-0.5 * ((tt - 8.8) / 0.0005) ** 2)
    for k in range(9):
        data[:, k] += (400.0 if k < 5 else 150.0) * pulse
    ev = find_events(replace(world.rec, data=data))
    assert _nearest(ev.t_s, 8.8) > 0.001
