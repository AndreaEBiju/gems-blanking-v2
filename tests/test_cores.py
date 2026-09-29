"""Tests for :mod:`gems_blanking_v2.detect.cores` (task 09, 2026-09-29)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.constants import GRID_S
from gems_blanking_v2.detect import chain
from gems_blanking_v2.detect.candidates import candidate_report
from gems_blanking_v2.detect.cores import (
    SLOW_BANDS,
    candidate_cores,
    width_breakdown,
)
from gems_blanking_v2.physio.rpeaks import BeatTrain
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import make_band_z

F64 = npt.NDArray[np.float64]
NO_BEATS = BeatTrain(t_s=np.empty(0, dtype=np.float64), tag=np.empty(0, dtype="<U8"))
FAST, SLOW = "300-3000", "0-2"


def _z(band: str, *bumps: tuple[float, float, float], dur: float = 3.0,
       signal: str = "R_T") -> F64:
    """One pair's z: flat 0.5 with rectangular ``(start_s, stop_s, z)`` bumps."""
    return make_band_z(band, dur, bumps=tuple((a, b, v, signal) for a, b, v in bumps)).z_max


def _frames(t0: float, t1: float) -> int:
    return int(round((t1 - t0) / GRID_S))


def test_the_core_is_where_some_pair_is_over_z_enter_and_the_tail_is_hysteresis() -> None:
    """A 3.5 bump with 2.0 shoulders: the core is the bump, the shoulders are the tail."""
    z = {("R_T", FAST): _z(FAST, (1.0, 1.4, 2.0), (1.4, 1.5, 3.5), (1.5, 1.8, 2.0))}
    rep = candidate_report(z, NO_BEATS)
    assert len(rep.candidates) == 1
    (cores,) = candidate_cores(z, rep)
    assert len(cores) == 1
    assert cores[0].start_s == pytest.approx(1.4) and cores[0].stop_s == pytest.approx(1.5)
    assert cores[0].pairs == (("R_T", FAST),) and cores[0].peak_z == 3.5
    b = width_breakdown(z, rep, 0)
    assert (b.core_fast, b.tail_fast, b.core_slow, b.tail_slow, b.merge_gap) == (10, 70, 0, 0, 0)


def test_a_bridged_gap_is_counted_as_merging_and_the_candidate_has_two_cores() -> None:
    """Two bumps 50 ms apart (under the 100 ms merge window) are one candidate."""
    z = {("R_T", FAST): _z(FAST, (1.0, 1.1, 4.0), (1.15, 1.25, 4.0))}
    rep = candidate_report(z, NO_BEATS)
    assert len(rep.candidates) == 1
    (cores,) = candidate_cores(z, rep)
    assert [(round(c.start_s, 3), round(c.stop_s, 3)) for c in cores] == [(1.0, 1.1), (1.15, 1.25)]
    b = width_breakdown(z, rep, 0)
    assert b.merge_gap == 5 and b.core_fast == 20 and b.total == 25


def test_slow_band_only_frames_are_attributed_to_the_slow_bands_and_fast_wins() -> None:
    z = {("R_T", FAST): _z(FAST, (1.5, 1.6, 4.0)),
         ("stomach_ref", SLOW): _z(SLOW, (1.0, 2.0, 3.5), (2.0, 2.5, 2.0), signal="stomach_ref")}
    rep = candidate_report(z, NO_BEATS)
    assert len(rep.candidates) == 1
    b = width_breakdown(z, rep, 0)
    assert b.core_fast == 10  # both over z_enter there: fast wins
    assert b.core_slow == 90 and b.tail_slow == 50 and b.tail_fast == 0 and b.merge_gap == 0
    (cores,) = candidate_cores(z, rep)
    assert len(cores) == 1
    assert cores[0].pairs == (("R_T", FAST), ("stomach_ref", SLOW))
    assert cores[0].peak_pair == ("R_T", FAST)


def test_the_slow_bands_are_the_multi_second_windows() -> None:
    assert frozenset({"0-2", "0.5-3"}) == SLOW_BANDS


def test_exactly_z_enter_is_not_a_core_as_it_is_not_a_crossing() -> None:
    z = {("R_T", FAST): _z(FAST, (1.0, 1.1, 3.0), (1.1, 1.2, 3.2))}
    rep = candidate_report(z, NO_BEATS)
    (cores,) = candidate_cores(z, rep)
    assert [(round(c.start_s, 3), round(c.stop_s, 3)) for c in cores] == [(1.1, 1.2)]


def test_the_thresholds_are_the_reports_own() -> None:
    """A report made at z_enter 2.5 puts a 2.7 plateau in the core."""
    z = {("R_T", FAST): _z(FAST, (1.0, 1.3, 2.7))}
    rep = candidate_report(z, NO_BEATS, z_enter=2.5)
    (cores,) = candidate_cores(z, rep)
    assert len(cores) == 1 and cores[0].duration_s == pytest.approx(0.3)


def test_the_breakdown_uses_the_reports_own_thresholds() -> None:
    """At z_enter 2.5 / z_exit 2.0, 2.7 plateaus are core and a 1.8 dip is a bridged gap."""
    z = {("R_T", FAST): _z(FAST, (1.0, 1.1, 2.7), (1.1, 1.15, 1.8), (1.15, 1.25, 2.7))}
    rep = candidate_report(z, NO_BEATS, z_enter=2.5, z_exit=2.0)
    b = width_breakdown(z, rep, 0)
    assert (b.core_fast, b.tail_fast, b.merge_gap) == (20, 0, 5)


def test_nan_frames_are_never_evidence() -> None:
    z = {("R_T", FAST): _z(FAST, (1.0, 1.1, 4.0), (1.1, 1.15, np.nan), (1.15, 1.25, 4.0))}
    rep = candidate_report(z, NO_BEATS)
    b = width_breakdown(z, rep, 0)
    assert b.merge_gap == 5 and b.core_fast == 20


def test_the_offset_moves_every_time_onto_the_recording_timeline() -> None:
    z = {("R_T", FAST): _z(FAST, (1.0, 1.1, 4.0))}
    rep = candidate_report(z, NO_BEATS)
    (cores,) = candidate_cores(z, rep, offset_s=100.0)
    assert cores[0].start_s == pytest.approx(101.0) and cores[0].stop_s == pytest.approx(101.1)


def test_a_report_with_cardiac_suppression_is_refused() -> None:
    z = {("R_T", "100-300"): _z("100-300", (1.0, 1.1, 4.0))}
    rep = candidate_report(z, NO_BEATS)
    object.__setattr__(rep, "cardiac_suppression", "applied")
    with pytest.raises(NotImplementedError, match="cardiac suppression"):
        candidate_cores(z, rep)
    with pytest.raises(NotImplementedError, match="cardiac suppression"):
        width_breakdown(z, rep, 0)


def test_cores_are_outside_the_generation_hash() -> None:
    """Computing cores must not move the budget key or the eligible rounds."""
    assert "gems_blanking_v2.detect.cores" not in chain.generation_modules()


LEVELS = st.sampled_from([0.0, 1.0, 1.6, 2.5, 3.0, 3.4, 5.0, float("nan")])


@given(st.lists(st.lists(LEVELS, min_size=60, max_size=60), min_size=1, max_size=4),
       st.lists(st.sampled_from(["300-3000", "100-300", "2-50", "0.5-3", "0-2"]),
                min_size=4, max_size=4))
@settings(max_examples=300, deadline=None)
def test_the_parts_are_exhaustive_and_the_cores_are_the_crossings(
    rows: list[list[float]], bands: list[str]
) -> None:
    """Every frame in exactly one part; cores are exactly the over-z_enter frames."""
    z = {(f"S{i}", bands[i]): np.repeat(np.asarray(r, dtype=np.float64), 2)
         for i, r in enumerate(rows)}
    rep = candidate_report(z, NO_BEATS)
    all_cores = candidate_cores(z, rep)
    stack = np.vstack(list(z.values()))
    with np.errstate(invalid="ignore"):
        entered = (np.isfinite(stack) & (stack > rep.z_enter)).any(axis=0)
    for i, c in enumerate(rep.candidates):
        a, b = int(round(c.start_s / GRID_S)), int(round(c.stop_s / GRID_S))
        br = width_breakdown(z, rep, i)
        assert br.total == b - a
        assert br.core_fast + br.core_slow == int(entered[a:b].sum())
        mask = np.zeros(b - a, dtype=bool)
        for core in all_cores[i]:
            s, e = int(round(core.start_s / GRID_S)) - a, int(round(core.stop_s / GRID_S)) - a
            assert 0 <= s < e <= b - a and not mask[s:e].any()
            mask[s:e] = True
        assert np.array_equal(mask, entered[a:b])
        assert all_cores[i], "hysteresis guarantees every candidate a core"
