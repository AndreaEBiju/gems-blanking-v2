"""Task 13 acceptance: the spike extent follows the mark, not the candidate.

Added 2026-09-29 (task 09, "Candidates are much wider than the marks"). Candidates
are deliberately wider than the artifacts Andrea marks: they run from z_enter down to
z_exit, merge across 100 ms gaps, take the max over every (signal, band) pair, and
the slow bands' 6-7.5 s windows smear onsets by seconds. That width is kept for
recall. It must never reach the mask: the blank is the consumer's extent - where
THAT consumer's band exceeds THAT consumer's tolerance, plus measured settling -
never the candidate interval.

What this test must do once ``extent/tolerance.py`` exists (task 13):

1. For every merged audit mark of rounds 1-4 that a candidate covers, take the
   confirmed event it belongs to and compute ``compute_extent(event, z, consumer)``
   for every consumer in ``CONSUMERS``.
2. Report, per consumer, ``extent duration / mark duration`` beside
   ``candidate width / mark width`` (the task 09 width measurement) - the
   distribution per condition, not only a median (invariant 34).
3. FAIL if the ``spikes`` extent tracks the candidate rather than the mark: a spike
   extent whose duration follows the covering candidates' width (e.g. its ratio to
   the mark sits near the candidate's ratio instead of near 1, or it is as wide as
   the candidate whenever the candidate is wide) is a task 13 failure, whatever
   else passes. The exact bound is task 13's to set against the measured
   settling time; it must be stated in this test when it is.
4. Slow-band consumers (``slow_wave``, ``breathing``) are exempt from 3 only to the
   resolution their window allows (task 13, "Slow bands"), and that resolution is
   reported alongside.

The audit marks live in the store, so the full check runs as a measurement, not a
hermetic unit test; the hermetic part is a synthetic mark inside a wide synthetic
candidate, asserting the spike extent stays with the mark.

HERMETIC PART, built with task 13 (2026-10-07): a 200 ms synthetic mark inside a 3 s
synthetic candidate whose z sits between ``z_exit`` and the tolerance everywhere but the
mark. The bound, stated against the measured settling: the spike extent may exceed the
mark by at most ``2 * settling + 2 * GRID_S`` (padding both sides, plus one frame of
quantisation at each edge) - at the TDT rate 2 x 5.1 ms + 20 ms = 30 ms - and must be
far below the candidate's width. The store-based measurement (rounds 1-4, per
consumer, per condition) needs the audit marks and is a measurement run, not a unit test.
"""

from __future__ import annotations

from typing import Any

import pytest
from gems_blanking_v2.constants import BANDS, GRID_S
from gems_blanking_v2.extent import tolerance as tl
from gems_blanking_v2.types import Candidate, Event

from tests.conftest import make_band_z

FS = 24414.0625
TOL = tl.ToleranceTable({"spikes": 4.0, "slow_wave": 3.0, "breathing": 3.0},
                        source="synthetic acceptance table")


def _case(
    cand: tuple[float, float], mark: tuple[float, float]
) -> tuple[Event, dict[tuple[str, str], Any]]:
    ev = Event(Candidate(*cand, ("L_T",), ("300-3000",), 9.0, "electrical"),
               "motion", float("nan"), "human")
    # Between z_exit (1.5) and the tolerance across the whole candidate; over it at the mark.
    eng = make_band_z("300-3000", 60.0, bumps=((cand[0], cand[1], 2.5, "L_T"),
                                               (mark[0], mark[1], 9.0, "L_T")), signal="L_T")
    slow = make_band_z("0-2", 60.0, bumps=((cand[0], cand[1], 2.0, "s"),), signal="ANT1")
    return ev, {("L_T", "300-3000"): eng.z_max, ("ANT1", "0-2"): slow.z_max}


@pytest.mark.parametrize(("cand", "mark"), [((18.0, 21.0), (19.4, 19.6)),
                                            ((10.0, 25.0), (12.0, 12.05)),
                                            ((30.0, 30.5), (30.1, 30.3))])
def test_the_spike_extent_follows_the_mark_not_the_candidate(
    cand: tuple[float, float], mark: tuple[float, float]
) -> None:
    ev, z = _case(cand, mark)
    ext = tl.compute_extent(ev, z, "spikes", signal="L_T", tolerances=TOL, fs=FS, z_t0_s=0.0)
    assert ext is not None
    settle = tl.consumer_settling("spikes", FS)
    assert settle is not None
    bound = 2 * settle.total_s + 2 * GRID_S
    mark_w, cand_w, ext_w = mark[1] - mark[0], cand[1] - cand[0], ext.stop_s - ext.start_s
    assert ext_w - mark_w <= bound + 1e-9
    assert ext.start_s <= mark[0] and mark[1] <= ext.stop_s
    if cand_w > 2 * mark_w + bound:
        assert ext_w < 0.5 * cand_w  # does not track the candidate


def test_slow_consumers_report_their_resolution() -> None:
    ev, z = _case((18.0, 21.0), (19.4, 19.6))
    assert tl.compute_extent(ev, z, "slow_wave", signal="ANT1", tolerances=TOL, fs=FS,
                             z_t0_s=0.0) is None
    assert BANDS["0-2"].window_s == 7.5  # the resolution any slow_wave extent carries
