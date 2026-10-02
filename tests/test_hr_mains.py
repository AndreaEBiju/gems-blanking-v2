"""Ruling 2026-10-02 (e) 1-3: the notched count gate, the persistent hum lock, veto precision."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.physio import hr_channel as hc

F64 = npt.NDArray[np.float64]
FS = 24414.0625
NAN = float("nan")


# --- the persistent lock ---------------------------------------------------------------


def test_three_consecutive_minutes_on_one_grid_value_are_a_lock() -> None:
    locked = hc.hum_locked_persistent([400.0, 400.1, 399.9, 372.7, 378.9, 378.95, 378.9, 379.0])
    assert locked.tolist() == [True, True, True, False, True, True, True, True]


def test_two_minutes_different_grid_values_or_a_gap_are_not_a_lock() -> None:
    assert not hc.hum_locked_persistent([400.0, 400.1, 372.7]).any()  # only two
    assert not hc.hum_locked_persistent([400.0, 378.9, 360.0]).any()  # three grid values, not one
    assert not hc.hum_locked_persistent([400.0, 400.0, NAN, 400.0]).any()  # unclear breaks the run


def test_the_grid_tolerance_is_three_tenths_of_a_bpm() -> None:
    assert hc.hum_locked_persistent([400.29, 399.71, 400.0]).all()
    assert not hc.hum_locked_persistent([400.31, 400.0, 400.0]).any()  # first is off: run of two
    assert hc.hum_grid([400.0, 360.0, 423.5, 480.0, 372.7]).tolist() == [18, 20, 17, 15, 0]


def test_the_count_gate_input_is_notched_at_mains_and_its_harmonic() -> None:
    assert hc.mains_harmonics() == (60.0, 120.0)


def test_a_persistent_lock_is_not_clear_in_the_count_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    t = np.arange(int(185 * FS)) / FS
    x = np.zeros_like(t)
    for b in np.arange(0.1, 184.9, 0.161):
        near = np.abs(t - b) < 0.012
        x[near] += 40.0 * np.exp(-0.5 * ((t[near] - b) / 0.002) ** 2)
    _s, clear = hc.autocorr_rate(x, FS)
    assert np.isfinite(clear).all()
    monkeypatch.setattr(hc, "hum_locked_persistent", lambda b: np.ones(np.shape(b), dtype=bool))
    _s, locked = hc.autocorr_rate(x, FS)
    assert np.isnan(locked).all()


# --- veto precision ----------------------------------------------------------------------


def test_the_harm_interval_must_exclude_the_threshold_on_the_estimates_side() -> None:
    assert not hc.harm_is_settled(0, 298)  # one-sided upper 0.010002
    assert hc.harm_is_settled(0, 299)  # 0.009969
    assert not hc.harm_is_settled(2, 200)  # 0.01 itself: upper far above
    assert not hc.harm_is_settled(4, 200)  # 0.02, lower 0.0069
    assert hc.harm_is_settled(6, 200)  # 0.03, lower 0.0131


def _row(
    hits: int, *, count: bool = True, plausible: bool = True, beats: F64 | None = None
) -> hc.TrainGate:
    cg = hc.CountGate(
        minutes=1,
        clear_minutes=1,
        fraction_within=1.0 if count else 0.0,
        median_abs_dev=0.0,
        assessable=True,
        passes=count,
    )
    b = beats if beats is not None else np.arange(10.0)
    harm = hits / hc.N_INJECTIONS
    return hc.TrainGate(
        "x",
        "task05",
        b,
        1.0,
        cg,
        harm,
        0.0,
        0.0,
        plausible,
        bool(count and plausible and harm <= hc.PROVISIONAL_MAX_TRANSIENT_HARM),
    )


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> dict[int, int]:
    """Each row's beats array (by id) -> how many injections of every further block harm it."""
    per_block: dict[int, int] = {}
    monkeypatch.setattr(hc, "_detect", lambda _det, _x, _fs: np.zeros(0))
    monkeypatch.setattr(hc, "_harmed", lambda b0, _b1, _times: per_block[id(b0)])
    return per_block


def _block(_k: int):  # noqa: ANN202
    return np.zeros(0), lambda _r: np.zeros(0)


def test_a_clean_row_is_settled_by_one_more_block(scripted: dict[int, int]) -> None:
    r = _row(0)
    scripted[id(r.beats_s)] = 0
    out = hc._refine_harm([r], _block, FS)[0]
    assert out.n_injections == 400 and out.transient_harm == 0.0 and out.passes


def test_an_unsettled_row_runs_to_two_thousand_and_is_decided_on_its_estimate(
    scripted: dict[int, int],
) -> None:
    at, over = _row(2), _row(3, beats=np.arange(11.0))
    scripted[id(at.beats_s)] = 2  # stays exactly at 0.01
    scripted[id(over.beats_s)] = 3  # stays at 0.015 until its lower bound clears 0.01
    out_at, out_over = hc._refine_harm([at, over], _block, FS)
    assert out_at.n_injections == hc.MAX_INJECTIONS and out_at.transient_harm == pytest.approx(0.01)
    assert out_at.passes  # the estimate decides: 0.01 is at the threshold
    assert 200 < out_over.n_injections <= hc.MAX_INJECTIONS and not out_over.passes
    assert hc.harm_is_settled(
        round(out_over.transient_harm * out_over.n_injections), out_over.n_injections
    )


def test_settled_rows_and_rows_failing_other_gates_take_no_more_injections(
    scripted: dict[int, int],
) -> None:
    settled, no_count, implausible = _row(8), _row(0, count=False), _row(0, plausible=False)
    for r in (settled, no_count, implausible):
        scripted[id(r.beats_s)] = 0
    out = hc._refine_harm([settled, no_count, implausible], _block, FS)
    assert [r.n_injections for r in out] == [200, 200, 200]
    assert not any(r.passes for r in out)


def test_the_decision_uses_the_pooled_estimate_after_more_blocks(scripted: dict[int, int]) -> None:
    r = _row(2)  # 0.01 on 200: not settled
    scripted[id(r.beats_s)] = 0  # the further blocks harm nothing
    out = hc._refine_harm([r], _block, FS)[0]
    assert 200 < out.n_injections < hc.MAX_INJECTIONS
    assert out.transient_harm == pytest.approx(2 / out.n_injections) and out.passes


def test_the_count_gate_lock_test_reads_the_refined_peak() -> None:
    # a mains-only input at a lag between integer samples: the unrefined rate is off the grid
    t = np.arange(int(245 * FS)) / FS
    x = np.zeros_like(t)
    period = 14 / 120  # 7200/14 = 514.3 bpm, lag 118.68 samples at ~1017 Hz
    for b in np.arange(0.05, 244.9, period):
        near = np.abs(t - b) < 0.006
        x[near] += 50.0 * np.exp(-0.5 * ((t[near] - b) / 0.001) ** 2)
    orig = hc.mains_harmonics
    try:
        hc.mains_harmonics = lambda: ()  # leave the lock in: test the persistence test alone
        _s, bpm = hc.autocorr_rate(x, FS)
    finally:
        hc.mains_harmonics = orig
    assert np.isnan(bpm).all()  # every minute on 7200/14: a persistent lock, not clear
