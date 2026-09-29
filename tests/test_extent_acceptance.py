"""Task 13 acceptance on the audit marks - a STUB until task 13 is built.

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
"""

from __future__ import annotations

import pytest


def test_the_spike_extent_follows_the_mark_not_the_candidate() -> None:
    """Task 13 acceptance: see the module docstring. Skipped until task 13 lands."""
    pytest.importorskip("gems_blanking_v2.extent.tolerance",
                        reason="task 13 (extent per consumer) is not built yet")
    pytest.skip("task 13 exists: replace this stub with the acceptance in the module "
                "docstring - a spike extent that tracks candidate width is a failure")
