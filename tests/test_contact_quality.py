"""The contact screen: each rule, each threshold, and the velocity answer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.contact_quality import (
    assess_contacts,
    screened_signals,
    velocity_cuffs,
)
from gems_blanking_v2.types import Recording

from tests.conftest import make_common_mode, make_cuff_contacts

FS = 24414.0625
DUR = 4.0


def _col(rec: Recording, label: str) -> int:
    return next(c.index for c in rec.channels if f"{c.cuff_id}{c.contact_index}" == label)


def _with_column(rec: Recording, label: str, x: np.ndarray) -> Recording:
    data = np.array(rec.data, dtype=np.float64)
    data[:, _col(rec, label)] = x
    return replace(rec, data=data)


def test_healthy_contacts_pass_and_both_cuffs_support_velocity() -> None:
    """The fixture is physical: same-cuff peers share the common mode."""
    q = assess_contacts(make_cuff_contacts(FS, DUR, seed=1))

    assert not any(v.screened for v in q.values())
    assert min(min(v.r_peers) for v in q.values()) > 0.8
    assert velocity_cuffs(q) == ("L", "R")
    assert screened_signals(q) == frozenset()


def test_a_flat_contact_removes_its_v_and_its_cuffs_tripole() -> None:
    q = assess_contacts(make_cuff_contacts(FS, DUR, {"L2": "flat"}, seed=2))

    assert q["L2"].reasons[0] == "flat"
    assert [k for k, v in q.items() if v.screened] == ["L2"]
    assert screened_signals(q) == {"L_V2", "L_T"}
    assert velocity_cuffs(q) == ("R",)


def test_a_duplicated_contact_flags_both_copies() -> None:
    """Which of two identical columns is the real one is not knowable from the data."""
    q = assess_contacts(make_cuff_contacts(FS, DUR, {"R3": ("duplicate", "R1")}, seed=3))

    assert "duplicate" in q["R1"].reasons and "duplicate" in q["R3"].reasons
    assert q["R3"].identical_to == ("R1",)
    assert screened_signals(q) == {"R_V1", "R_V3", "R_T"}
    assert velocity_cuffs(q) == ("L",)


def test_a_cross_cuff_duplicate_is_caught_too() -> None:
    """Animal K's question - are two cuffs' raw channels one channel twice."""
    q = assess_contacts(make_cuff_contacts(FS, DUR, {"L1": ("duplicate", "R1")}, seed=4))

    assert q["L1"].identical_to == ("R1",) and q["R1"].identical_to == ("L1",)


def test_one_uncorrelated_contact_does_not_condemn_its_healthy_peers() -> None:
    """The first draft averaged r over peers, and flagged A's L1 and L2 with L3."""
    q = assess_contacts(make_cuff_contacts(FS, DUR, {"L3": "uncorrelated"}, seed=5))

    assert q["L3"].reasons == ("uncorrelated",)
    assert not q["L1"].screened and not q["L2"].screened
    assert screened_signals(q) == {"L_V3", "L_T"}


@pytest.mark.parametrize(("sigma", "flat"), [(0.35, True), (0.7, False)])
def test_the_flat_sigma_threshold(sigma: float, flat: bool) -> None:
    """0.5 uV: a contact at 0.35 is dead, one at 0.7 is not (either side of it)."""
    rec = make_cuff_contacts(FS, DUR, seed=6)
    x = np.random.default_rng(6).standard_normal(rec.data.shape[0]) * sigma * 1.6
    q = assess_contacts(_with_column(rec, "R2", x))

    assert q["R2"].sigma_uv == pytest.approx(sigma, rel=0.35)
    assert ("flat" in q["R2"].reasons) is flat


@pytest.mark.parametrize(("hold", "flat"), [(10, True), (100, False)])
def test_the_flat_fraction_threshold(hold: int, flat: bool) -> None:
    """5%: holding one sample in 10 (10% repeats) is flat; one in 100 (1%) is not."""
    rec = make_cuff_contacts(FS, DUR, seed=7)
    x = np.array(rec.data[:, _col(rec, "R2")], dtype=np.float64)
    x[hold::hold] = x[hold - 1::hold][: x[hold::hold].size]
    q = assess_contacts(_with_column(rec, "R2", x))

    assert ("flat" in q["R2"].reasons) is flat


@pytest.mark.parametrize(("noise", "duplicate"), [(0.3, True), (0.5, False)])
def test_the_duplicate_threshold(noise: float, duplicate: bool) -> None:
    """Residual under 1%: a copy plus 0.3 uV of noise (~0.8%) is a copy.

    Plus 0.5 uV (~1.3%) it is not, although its r is still above 0.9999.
    """
    rec = make_cuff_contacts(FS, DUR, seed=8)
    x = np.array(rec.data[:, _col(rec, "R1")], dtype=np.float64)
    x = x + noise * np.random.default_rng(8).standard_normal(x.size)
    q = assess_contacts(_with_column(rec, "L2", x))

    ratio = q["L2"].copy_residual
    assert (0.006 < ratio < 0.01) if duplicate else (0.01 < ratio < 0.016)
    assert q["L2"].best_other[0] == "R1" and q["L2"].best_other[1] > 0.9999
    assert ("duplicate" in q["L2"].reasons) is duplicate


@pytest.mark.parametrize("gain", [3.0, -1.0, -0.4])
def test_a_rescaled_or_inverted_copy_is_still_a_copy(gain: float) -> None:
    """A copy through another gain or a reversed lead is the same data."""
    rec = make_cuff_contacts(FS, DUR, seed=13)
    q = assess_contacts(_with_column(rec, "L2", gain * rec.data[:, _col(rec, "R1")]))

    assert "duplicate" in q["L2"].reasons and "duplicate" in q["R1"].reasons


def test_healthy_neighbours_correlated_past_0999_are_not_copies() -> None:
    """Healthy neighbours past r = 0.999 must pass - the rig's reach 0.99895.

    The first draft's r >= 0.999 rule would have screened them. A common mode 40x
    the contacts' own ENG gives the same regime here.
    """
    q = assess_contacts(make_cuff_contacts(FS, DUR, common_sigma_uv=250.0, seed=12))

    assert max(v.best_other[1] for v in q.values()) > 0.999
    assert not any(v.screened for v in q.values())


@pytest.mark.parametrize(("share", "uncorrelated"), [(0.17, True), (0.22, False)])
def test_the_uncorrelated_threshold(share: float, uncorrelated: bool) -> None:
    """Below 0.5 with every peer: r ~0.47 is off the cuff, r ~0.56 is on it."""
    rec = make_cuff_contacts(FS, DUR, {"L3": "uncorrelated"}, seed=9)
    shared = make_common_mode(FS, DUR, 300.0, 3000.0, sigma_uv=20.0, seed=9 + 11)
    x = np.array(rec.data[:, _col(rec, "L3")], dtype=np.float64) + share * shared
    q = assess_contacts(_with_column(rec, "L3", x))

    r = max(q["L3"].r_peers)
    assert (0.42 < r < 0.5) if uncorrelated else (0.5 < r < 0.6)
    assert ("uncorrelated" in q["L3"].reasons) is uncorrelated


def test_a_window_with_no_data_refuses_rather_than_passing_every_contact() -> None:
    rec = make_cuff_contacts(FS, DUR, seed=10)

    with pytest.raises(ValueError, match="< 1 s"):
        assess_contacts(rec, start_s=DUR - 0.5)


def test_masked_samples_are_left_out_not_propagated() -> None:
    """NaN is 'not assessable' (invariant 1): it must not turn every r into NaN."""
    rec = make_cuff_contacts(FS, DUR, seed=11)
    data = np.array(rec.data, dtype=np.float64)
    data[1000:5000, 0] = np.nan
    q = assess_contacts(replace(rec, data=data))

    assert all(np.isfinite(v.r_peers).all() and np.isfinite(v.sigma_uv) for v in q.values())
    assert not any(v.screened for v in q.values())
