"""Tests for :mod:`gems_blanking_v2.derive.consumer_screen` (ruling 2026-09-30, item 5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.consumer_screen import (
    DISTRUSTED_CUFFS,
    ground_r2,
    raw_readable,
    screen_for_consumers,
)
from gems_blanking_v2.detect import chain

from tests.conftest import make_shared_ground

FS = 24414.0625


@pytest.fixture(scope="module")
def world():  # noqa: ANN201 - a NamedTuple from conftest
    return make_shared_ground(FS, 20.0, common_sigma_uv=10.0, seed=7)


def _without_ground(w, name: str, keep: float = 0.0):  # noqa: ANN001, ANN202
    """Take ``1 - keep`` of the ground signal back out of one contact."""
    data = np.array(w.rec.data, dtype=np.float64)
    col = next(c.index for c in w.rec.channels if c.name == name)
    data[:, col] -= (1.0 - keep) * w.gains[name] * w.common
    return replace(w.rec, data=data)


def test_healthy_contacts_all_carry_the_ground_signal(world) -> None:  # noqa: ANN001
    r2 = ground_r2(world.rec)
    assert set(r2) == {"L1", "L2", "L3", "R1", "R2", "R3"}
    assert min(r2.values()) > 0.8
    assert all(v == () for v in screen_for_consumers(r2).values())


def test_a_contact_without_the_ground_signal_is_the_suspect_one(world) -> None:  # noqa: ANN001
    rec = _without_ground(world, "LVN2")
    flags = screen_for_consumers(ground_r2(rec))
    assert flags["L2"] == ("missing_ground",)
    assert flags["L1"] == () and flags["L3"] == ()


def test_a_low_gain_contact_is_flagged_although_it_still_correlates(world) -> None:  # noqa: ANN001
    """Half the ground signal: R2 falls well below its peers' but not near zero."""
    rec = _without_ground(world, "RVN3", keep=0.4)
    r2 = ground_r2(rec)
    assert 0.2 < r2["R3"] < np.median([r2["R1"], r2["R2"]]) - 0.15
    assert screen_for_consumers(r2)["R3"] == ("low_gain",)


def test_a_distrusted_cuff_is_unreadable_for_every_contact(world) -> None:  # noqa: ANN001
    ((rid, cuff), _why), = DISTRUSTED_CUFFS.items()
    out = raw_readable(rid, ground_r2(world.rec), {})
    assert all("distrusted_cuff" in out[f"{cuff}{i}"] for i in (1, 2, 3))
    other = "R" if cuff == "L" else "L"
    assert all(out[f"{other}{i}"] == () for i in (1, 2, 3))


def test_a_hum_flagged_stomach_contact_is_read_by_no_consumer(world) -> None:  # noqa: ANN001
    out = raw_readable("any", ground_r2(world.rec), {"ANT1": ("mains_hum",), "ANT2": ()})
    assert out["ANT1"] == ("mains_hum",) and out["ANT2"] == ()


def test_the_consumer_screen_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.derive.consumer_screen" not in chain.generation_modules()


def test_the_missing_ground_bar_is_point_two() -> None:
    flags = screen_for_consumers({"L1": 0.10, "L2": 0.95, "L3": 0.95,
                                  "R1": 0.25, "R2": 0.9, "R3": 0.9})
    assert flags["L1"] == ("missing_ground",)
    assert flags["R1"] == ("low_gain",)  # above the bar, far below its peers


def test_low_gain_compares_a_contact_with_its_own_cuff_only() -> None:
    """A uniformly low cuff is not low-gain against a high cuff on the other side."""
    flags = screen_for_consumers({"L1": 0.50, "L2": 0.52, "L3": 0.45,
                                  "R1": 0.98, "R2": 0.98, "R3": 0.97})
    assert all(flags[k] == () for k in ("L1", "L2", "L3"))


def test_two_contacts_sharing_a_local_signal_and_no_ground_are_both_suspect(world) -> None:  # noqa: ANN001
    """Flag both contacts of B t02 1_3 L's picture: they agree with each other only.

    A reference that included the cuff would credit them with that shared signal.
    """
    from tests.conftest import make_common_mode  # noqa: PLC0415

    rec = _without_ground(world, "LVN2")
    w2 = world._replace(rec=rec)
    rec = _without_ground(w2, "LVN3")
    local = make_common_mode(FS, 20.0, 300.0, 3000.0, sigma_uv=40.0, seed=99)
    data = np.array(rec.data, dtype=np.float64)
    for name in ("LVN2", "LVN3"):
        data[:, next(c.index for c in rec.channels if c.name == name)] += local[: data.shape[0]]
    r2 = ground_r2(replace(rec, data=data))
    flags = screen_for_consumers(r2)
    assert flags["L2"] == ("missing_ground",) and flags["L3"] == ("missing_ground",)
    assert flags["L1"] == ()
