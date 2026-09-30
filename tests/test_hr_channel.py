"""Tests for :mod:`gems_blanking_v2.physio.hr_channel` (ruling 2026-09-29, item 6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.cm_events import Events, find_events
from gems_blanking_v2.detect import chain
from gems_blanking_v2.physio.hr_channel import (
    DERIVED_CANDIDATES,
    PROVISIONAL_MAX_TRANSIENT_HARM,
    best_hr_channel,
    cm_gains,
    hr_candidates,
)

from tests.conftest import make_ecg, make_shared_ground, make_two_source_ground

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


# --- the transient veto (ruling 2026-09-30, item 6) ----------------------------


@pytest.fixture(scope="module")
def rig_like():  # noqa: ANN201 - a NamedTuple from conftest
    """Build the rig's ratio: the heart leaks into T strongly, the transients barely."""
    return make_two_source_ground(FS, 60.0, cardiac_gain_spread=0.4, gain_spread=0.005, seed=21)


def test_raw_contacts_are_vetoed_and_a_tripole_survives(rig_like) -> None:  # noqa: ANN001
    ev = find_events(rig_like.rec)
    best, table = best_hr_channel(rig_like.rec, events=ev)
    assert best in ("L_T", "R_T")
    harm = dict(zip(table["channel"], table["transient_harm"], strict=True))
    raw = [c.name for c in rig_like.rec.channels]
    assert all(harm[n] > PROVISIONAL_MAX_TRANSIENT_HARM for n in raw)
    assert harm[best] <= PROVISIONAL_MAX_TRANSIENT_HARM
    vetoed = table[table["vetoed_by"] == "transient_harm"]["channel"].tolist()
    assert set(raw) <= set(vetoed)


def test_without_events_the_ranking_is_task_05s_alone(rig_like) -> None:  # noqa: ANN001
    _best, table = best_hr_channel(rig_like.rec)
    assert "transient_harm" not in table.columns


def test_every_candidate_vetoed_by_the_transient_test_raises() -> None:
    w = make_two_source_ground(FS, 60.0, seed=21)  # ground transients leak into T too
    with pytest.raises(ValueError, match="transient test"):
        best_hr_channel(w.rec, events=find_events(w.rec))


def test_gains_come_from_the_kept_events_only() -> None:
    names = ("A", "B", "C")
    amp = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.5], [2.0, 1.0, 0.5]])
    ev = Events(t_s=np.arange(4.0), cm_amp_uv=np.ones(4), channel_amp_uv=amp,
                width_s=np.full(4, 0.001), channels=names)
    assert cm_gains(ev, [True, True, False, False]) == {"A": 1.0, "B": 1.0, "C": 1.0}
    assert cm_gains(ev, [False, False, True, True]) == {"A": 2.0, "B": 1.0, "C": 0.5}


def test_an_event_within_twenty_ms_of_a_beat_is_cardiac() -> None:
    from gems_blanking_v2.physio.hr_channel import _noncardiac  # noqa: PLC0415

    ev = Events(t_s=np.array([1.019, 1.021, 2.5]), cm_amp_uv=np.ones(3),
                channel_amp_uv=np.ones((3, 3)), width_s=np.full(3, 0.001), channels=("A", "B", "C"))
    assert _noncardiac(ev, np.array([1.0, 2.0])).tolist() == [False, True, True]
