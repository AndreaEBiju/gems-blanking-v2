"""Tests for :mod:`gems_blanking_v2.derive.spike_veto` (ruling 2026-09-30 (b), item 2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.spike_veto import (
    TARGET_CHANCE_LOSS,
    calibrate_theta,
    chance_loss,
    core_half_width,
    over_theta,
    reference_sigma,
    veto_spikes,
)
from gems_blanking_v2.detect import chain

from tests.conftest import make_shared_ground

FS = 24414.0625


@pytest.fixture(scope="module")
def world():  # noqa: ANN201 - a NamedTuple from conftest
    return make_shared_ground(FS, 30.0, n_transients=60, seed=17)


@pytest.fixture(scope="module")
def calibrated(world):  # noqa: ANN001, ANN201
    ref, _sigma = reference_sigma(world.rec, "L")
    dur = world.rec.data.shape[0] / FS
    w = 0.0004
    theta = calibrate_theta(ref, w, FS, (0.0, dur / 2), seed=1)
    return ref, w, theta, dur


def test_a_spike_at_a_common_mode_transient_is_vetoed(world, calibrated) -> None:  # noqa: ANN001
    ref, w, theta, _dur = calibrated
    over = over_theta(ref, theta, w, FS)
    assert veto_spikes(world.transients_s, over, FS).all()


def test_theta_gives_the_target_chance_loss_on_each_half(calibrated) -> None:  # noqa: ANN001
    ref, w, theta, dur = calibrated
    rng = np.random.default_rng(5)
    first = rng.uniform(w, dur / 2 - w, 20000)
    second = rng.uniform(dur / 2 + w, dur - w, 20000)
    assert chance_loss(ref, theta, w, FS, first) == pytest.approx(TARGET_CHANCE_LOSS, abs=0.003)
    assert chance_loss(ref, theta, w, FS, second) == pytest.approx(TARGET_CHANCE_LOSS, abs=0.01)


def test_differential_spikes_are_lost_only_at_chance(calibrated) -> None:  # noqa: ANN001
    """Lose spikes on this cuff alone only at the chance rate, whatever their size.

    They never reach the outside reference; at uniform random times (ruling (ii)) the
    veto takes them exactly as often as it fires at random.
    """
    ref, w, theta, dur = calibrated
    t = np.random.default_rng(8).uniform(dur / 2 + 0.01, dur - 0.01, 3000)
    lost = veto_spikes(t, over_theta(ref, theta, w, FS), FS).mean()
    assert lost == pytest.approx(TARGET_CHANCE_LOSS, abs=0.01)


def test_the_gate_reads_only_channels_outside_the_cuff(world) -> None:  # noqa: ANN001
    before, s0 = reference_sigma(world.rec, "L")
    data = np.array(world.rec.data, dtype=np.float64)
    for c in world.rec.channels:
        if c.cuff_id == "L":
            data[:, c.index] += 1000.0 * np.random.default_rng(2).standard_normal(data.shape[0])
    after, s1 = reference_sigma(replace(world.rec, data=data), "L")
    assert s0 == s1 and np.array_equal(before, after)


def test_the_core_half_width_is_rounded_up_to_a_tenth_of_a_ms() -> None:
    centres = (np.arange(-50, 51) * 0.0001).round(6)
    counts = np.full(centres.size, 10.0)
    counts[np.abs(centres) <= 0.00035] = 80.0  # bins at 0, +/-0.1, +/-0.2, +/-0.3 ms
    assert core_half_width(counts, centres, 10.0) == pytest.approx(0.0004)
    counts2 = np.full(centres.size, 10.0)
    counts2[np.abs(centres) <= 0.00005] = 80.0  # the zero bin alone
    assert core_half_width(counts2, centres, 10.0) == pytest.approx(0.0001)


def test_no_core_means_no_half_width() -> None:
    centres = (np.arange(-50, 51) * 0.0001).round(6)
    assert np.isnan(core_half_width(np.full(centres.size, 10.0), centres, 10.0))


def test_a_thin_core_spread_over_bins_is_still_measured() -> None:
    """Measure a core of +2 over a flank of 1 in every bin out to 0.4 ms.

    No single bin clears a three-sigma bar at these counts; the +/-1 ms total does.
    """
    centres = (np.arange(-50, 51) * 0.0001).round(6)
    counts = np.full(centres.size, 1.0)
    counts[np.abs(centres) <= 0.00045] = 3.0
    assert core_half_width(counts, centres, 1.0) == pytest.approx(0.0005)


def test_an_insignificant_core_is_no_core() -> None:
    centres = (np.arange(-50, 51) * 0.0001).round(6)
    counts = np.full(centres.size, 10.0)
    counts[np.isclose(centres, 0.0)] = 14.0  # +4 over a flank of 10: p ~ 0.2
    assert np.isnan(core_half_width(counts, centres, 10.0))


def test_nan_in_the_reference_never_vetoes() -> None:
    ref = np.array([0.0, np.nan, 50.0, 0.0, 0.0, 0.0, 0.0])
    over = over_theta(ref, 10.0, 1.0 / FS, FS)
    assert over.tolist() == [False, True, True, True, False, False, False]


def test_the_veto_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.derive.spike_veto" not in chain.generation_modules()


# --- ruling 2026-09-30 (c) 2: distrust where verification fails ------------------


def test_the_cuffs_that_failed_verification_are_distrusted_for_spikes() -> None:
    from gems_blanking_v2.derive.spike_veto import spike_trusted  # noqa: PLC0415

    for rid, cuff in (("gems_h_t01_es2_bl_224901_20260905T024908Z", "L"),
                      ("gems_a_t05_2_1_bl_183840_20260924T223845Z", "L"),
                      ("gems_a_t05_2_1_bl_183840_20260924T223845Z", "R")):
        ok, why = spike_trusted(rid, cuff)
        assert not ok and "ruling 2026-09-30 (c) 2" in why
    assert spike_trusted("gems_h_t01_es2_bl_224901_20260905T024908Z", "R") == (True, "")


def test_a_cuff_distrusted_for_every_consumer_is_distrusted_for_spikes() -> None:
    from gems_blanking_v2.derive.consumer_screen import DISTRUSTED_CUFFS  # noqa: PLC0415
    from gems_blanking_v2.derive.spike_veto import spike_trusted  # noqa: PLC0415

    for key, why in DISTRUSTED_CUFFS.items():
        assert spike_trusted(*key) == (False, why)


# --- chance_times: a keyed stream per (recording, cuff) ---------------------------------

PINNED = [0.012851447728, 0.358977739408, 0.370734777879]


def test_the_same_key_gives_the_same_times_and_another_key_does_not() -> None:
    from gems_blanking_v2.derive.spike_veto import chance_times  # noqa: PLC0415

    a = chance_times(600.0, "gems_a_rec", "L")
    np.testing.assert_array_equal(a, chance_times(600.0, "gems_a_rec", "L"))
    assert a.shape == (20000,) and a.min() >= 0.0 and a.max() < 600.0
    assert not np.array_equal(a, chance_times(600.0, "gems_a_rec", "R"))
    assert not np.array_equal(a, chance_times(600.0, "gems_b_rec", "L"))
    # the parts are delimited: ("ab", "c") is not ("a", "bc")
    assert not np.array_equal(chance_times(1.0, "ab", "c"), chance_times(1.0, "a", "bc"))


def test_a_cuffs_times_do_not_depend_on_what_was_sampled_before() -> None:
    from gems_blanking_v2.derive.spike_veto import chance_times  # noqa: PLC0415

    alone = chance_times(600.0, "rec", "R")
    _ = chance_times(600.0, "rec", "L")  # another cuff drawn first, or not at all
    _ = chance_times(1200.0, "other", "L", n=5)
    np.testing.assert_array_equal(alone, chance_times(600.0, "rec", "R"))


def test_the_stream_is_pinned_across_runs_and_machines() -> None:
    from gems_blanking_v2.derive.spike_veto import chance_times  # noqa: PLC0415

    # SHA-256 of b"rec\0R\0" -> seed; a salted hash() or a changed derivation moves this
    assert chance_times(1.0, "rec", "R", n=3).round(12).tolist() == PINNED


def test_a_keyless_call_is_refused() -> None:
    from gems_blanking_v2.derive.spike_veto import chance_times  # noqa: PLC0415

    with pytest.raises(ValueError, match="needs a key"):
        chance_times(1.0)
