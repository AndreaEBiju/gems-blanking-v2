"""Tests for :mod:`gems_blanking_v2.physio.hr_channel` (ruling 2026-09-29, item 6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.physio.hr_channel import DERIVED_CANDIDATES, best_hr_channel, hr_candidates

from tests.conftest import make_ecg, make_shared_ground

FS = 24414.0625


@pytest.fixture(scope="module")
def world():  # noqa: ANN201 - a NamedTuple from conftest
    return make_shared_ground(FS, 30.0, seed=5)


def test_the_candidates_are_every_raw_channel_plus_the_derived_signals(world) -> None:  # noqa: ANN001
    cand = hr_candidates(world.rec)
    names = [c.name for c in cand.channels]
    assert names[:9] == [c.name for c in world.rec.channels]
    assert tuple(names[9:]) == DERIVED_CANDIDATES
    assert np.array_equal(np.asarray(cand.data[:, :9]), np.asarray(world.rec.data))
    assert all(c.index == k for k, c in enumerate(cand.channels))
    assert all(c.role == "aux" for c in cand.channels[9:])


def _with_ecg(rec, where: list[str], amp_uv: float = 60.0):  # noqa: ANN001, ANN202
    ecg, beats, _weak = make_ecg(FS, rec.data.shape[0] / FS, amp_uv=amp_uv, noise_uv=0.0, seed=17)
    data = np.array(rec.data, dtype=np.float64)
    for c in rec.channels:
        if c.name in where:
            data[:, c.index] += ecg[: data.shape[0]]
    return replace(rec, data=data), beats


def test_the_channel_that_carries_the_heart_is_chosen(world) -> None:  # noqa: ANN001
    rec, _beats = _with_ecg(world.rec, ["ANT2"])
    best, table = best_hr_channel(rec)
    assert best == "ANT2"
    assert table["channel"].iloc[0] == "ANT2"  # first on SNR, not merely the survivor


def test_an_ecg_common_to_every_contact_is_cancelled_by_the_tripole(world) -> None:  # noqa: ANN001
    """Rank a raw channel above both tripoles when the ECG is on every contact.

    A far-field ECG identical on every contact is common mode: the raw channels
    carry it and the tripoles cancel it.
    """
    rec, _beats = _with_ecg(world.rec, [c.name for c in world.rec.channels])
    best, table = best_hr_channel(rec)
    assert best not in ("L_T", "R_T")
    snr = dict(zip(table["channel"], table["snr"], strict=True))
    assert snr[best] > 3.0 * max(snr["L_T"], snr["R_T"])


def test_every_candidate_vetoed_raises_rather_than_picking_the_least_bad(world) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="every channel was vetoed"):
        best_hr_channel(world.rec)  # no heart anywhere


def test_hr_channel_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.physio.hr_channel" not in chain.generation_modules()
