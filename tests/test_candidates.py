"""Tests for :mod:`gems_blanking_v2.detect.candidates`.

Every numeric claim in the module's docstrings has a test here that would fail if it
were wrong. Two of the task's inputs do not exist yet - task 02's cardiac windows and
task 17's video motion - and the tests build them to those tasks' declared shapes, so
they are contract tests until the real ones land.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.constants import BANDS, GRID_S
from gems_blanking_v2.detect.candidates import (
    CARDIAC_SUPPRESSION_BANDS,
    VIDEO_ASSISTED_Z_ENTER,
    CombineRule,
    candidate_report,
    flag_rates,
    generate_candidates,
)
from gems_blanking_v2.physio.rpeaks import BeatTrain

F64 = npt.NDArray[np.float64]
Pair = tuple[str, str]

ENG = "300-3000"
CARDIAC = "100-300"


def _quiet(n_frames: int, seed: int = 0) -> F64:
    """Return a clean z-trace: standard normal, so ``z > 3`` is rare by construction."""
    return np.asarray(
        np.random.default_rng(seed).standard_normal(n_frames), dtype=np.float64
    )


def _with_event(trace: F64, start_s: float, stop_s: float, peak: float) -> F64:
    """Raise ``trace`` to ``peak`` over ``[start_s, stop_s)``. Returns a copy."""
    out = trace.copy()
    out[int(start_s / GRID_S) : int(stop_s / GRID_S)] = peak
    return out


def _no_beats() -> BeatTrain:
    """Return an empty beat train, for cases where cardiac timing is irrelevant."""
    return BeatTrain(t_s=np.empty(0, dtype=np.float64), tag=np.empty(0, dtype="<U8"))


def _beats(dur_s: float, rr_s: float = 0.165) -> BeatTrain:
    """Return a regular beat train at animal J's measured 165 ms RR."""
    times = np.arange(rr_s, dur_s, rr_s, dtype=np.float64)
    return BeatTrain(t_s=times, tag=np.array(["detected"] * times.size, dtype="<U8"))


class _Motion:
    """A :class:`~gems_blanking_v2.detect.candidates.VideoMotion` for tests.

    Built to the Protocol this module declares, since task 17 specifies no type. If
    task 17 lands with a different shape, this is where the mismatch shows up.
    """

    def __init__(self, spans_s: list[tuple[float, float]]) -> None:
        self.spans_s = spans_s

    def exceeds_threshold(self, n_frames: int) -> npt.NDArray[np.bool_]:
        """Return which grid frames sit inside a motion span."""
        mask = np.zeros(n_frames, dtype=bool)
        for start_s, stop_s in self.spans_s:
            mask[int(start_s / GRID_S) : int(stop_s / GRID_S)] = True
        return mask


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_every_injected_artifact_produces_a_containing_candidate() -> None:
    """The recall claim: each injected span sits inside some candidate's span."""
    n = 6000
    injected = [(5.0, 5.4), (17.2, 17.5), (41.0, 42.0)]
    trace = _quiet(n)
    for start_s, stop_s in injected:
        trace = _with_event(trace, start_s, stop_s, peak=8.0)

    candidates = generate_candidates({("L_T", ENG): trace}, _no_beats(), None)

    for start_s, stop_s in injected:
        containing = [
            c for c in candidates if c.start_s <= start_s and c.stop_s >= stop_s - GRID_S
        ]
        assert containing, f"nothing contains the artifact at {start_s}-{stop_s} s"
        assert containing[0].peak_z == pytest.approx(8.0, abs=0.01)
        assert containing[0].signals == ("L_T",)
        assert containing[0].bands == (ENG,)


def test_hysteresis_keeps_a_dipping_event_as_one_candidate() -> None:
    """A z-trace dipping to 2.0 mid-event yields one candidate, not two.

    2.0 is below ``z_enter`` of 3.0 and above ``z_exit`` of 1.5, so it is exactly the
    case hysteresis exists for. Without it this is two candidates separated by a gap
    wider than ``merge_gap_s``.
    """
    trace = _quiet(4000)
    trace = _with_event(trace, 10.0, 12.0, peak=6.0)
    trace[int(10.8 / GRID_S) : int(11.2 / GRID_S)] = 2.0  # a 400 ms dip, > merge_gap_s

    candidates = generate_candidates({("L_T", ENG): trace}, _no_beats(), None)
    overlapping = [c for c in candidates if c.start_s < 12.0 and c.stop_s > 10.0]

    assert len(overlapping) == 1, f"the dip split the event into {len(overlapping)}"
    assert overlapping[0].start_s <= 10.0
    assert overlapping[0].stop_s >= 12.0


def test_a_dip_below_the_exit_threshold_does_split_the_event() -> None:
    """The other half: hysteresis must not weld genuinely separate events together."""
    trace = _quiet(4000)
    trace = _with_event(trace, 10.0, 12.0, peak=6.0)
    trace[int(10.8 / GRID_S) : int(11.2 / GRID_S)] = 0.0  # below z_exit, and > merge_gap

    candidates = generate_candidates({("L_T", ENG): trace}, _no_beats(), None)
    overlapping = [c for c in candidates if c.start_s < 12.0 and c.stop_s > 10.0]

    assert len(overlapping) == 2


def test_spans_shorter_than_the_minimum_are_discarded() -> None:
    """``min_dur_s`` is 20 ms - two frames - so a single-frame spike is not an event."""
    trace = _quiet(2000)
    trace[500] = 9.0

    assert generate_candidates({("L_T", ENG): trace}, _no_beats(), None) == []

    trace[500:503] = 9.0
    assert len(generate_candidates({("L_T", ENG): trace}, _no_beats(), None)) == 1


def test_gaps_shorter_than_the_merge_window_are_joined() -> None:
    """Two crossings 50 ms apart are one event; 200 ms apart are two."""
    for gap_s, expected in ((0.05, 1), (0.20, 2)):
        trace = _quiet(2000)
        trace = _with_event(trace, 5.0, 5.2, peak=7.0)
        trace = _with_event(trace, 5.2 + gap_s, 5.4 + gap_s, peak=7.0)

        candidates = generate_candidates({("L_T", ENG): trace}, _no_beats(), None)
        near = [c for c in candidates if 4.5 < c.start_s < 6.5]
        assert len(near) == expected, f"a {gap_s * 1000:.0f} ms gap gave {len(near)}"


def test_a_nan_frame_is_not_a_crossing() -> None:
    """Unassessable is *no evidence*, not evidence of quiet, and never enters a span.

    Task 06 emits ``nan`` for settling edges and short segments. A comparison against
    ``nan`` is False in numpy, but relying on that would be relying on a coincidence;
    the mask is explicitly intersected with ``isfinite``.
    """
    trace = _quiet(2000)
    trace[800:900] = np.nan

    candidates = generate_candidates({("L_T", ENG): trace}, _no_beats(), None)
    in_gap = [c for c in candidates if c.start_s >= 8.0 and c.stop_s <= 9.0]

    assert in_gap == []
    assert flag_rates({("L_T", ENG): trace}).n_assessable == 1900


# ---------------------------------------------------------------------------
# how the pairs combine
# ---------------------------------------------------------------------------


def test_the_union_over_many_pairs_is_far_above_any_per_pair_rate() -> None:
    """Measure hard invariant 10b rather than quoting it.

    36 independent clean pairs at ``z_enter = 3.0``: each flags ~0.13% of frames and
    the union flags ~4.6%, which is 35x the per-pair rate. That ratio is the whole
    point - a flag-rate target is family-wise, and a per-pair number is not the
    candidate rate.
    """
    n = 20_000
    z = {
        (f"S{i // 6}", list(BANDS)[i % 6]): _quiet(n, seed=i) for i in range(36)
    }
    rates = flag_rates(z, z_enter=3.0)

    assert len(rates.per_pair) == 36
    worst = max(rates.per_pair.values())
    assert worst < 0.005, f"a single clean pair flagged {worst:.3%}"
    assert rates.union > 20.0 * worst, "the union should dwarf any per-pair rate"
    assert rates.union == pytest.approx(0.046, abs=0.010)
    assert rates.combined == rates.union, "'any' is the union by definition"


@pytest.mark.parametrize(("combine", "k", "expected"), [("any", 1, 1), ("k_of_n", 2, 0)])
def test_the_combine_rule_is_an_explicit_parameter(
    combine: CombineRule, k: int, expected: int
) -> None:
    """One pair crossing is a candidate under ``any`` and not under ``2 of n``.

    Cross-channel agreement is the lever invariant 10b names for shrinking the
    family, and it has to be a parameter because task 09 sweeps it alongside
    ``z_enter``.
    """
    n = 2000
    z: dict[Pair, F64] = {("L_T", ENG): _quiet(n, 0), ("R_T", ENG): _quiet(n, 1)}
    z[("L_T", ENG)] = _with_event(z[("L_T", ENG)], 5.0, 5.5, peak=8.0)

    candidates = generate_candidates(z, _no_beats(), None, combine=combine, k=k)
    near = [c for c in candidates if 4.5 < c.start_s < 6.0]

    assert len(near) == expected


def test_k_of_n_fires_when_enough_pairs_agree() -> None:
    """And the same event on two channels does clear ``2 of n``."""
    n = 2000
    z: dict[Pair, F64] = {
        ("L_T", ENG): _with_event(_quiet(n, 0), 5.0, 5.5, peak=8.0),
        ("R_T", ENG): _with_event(_quiet(n, 1), 5.0, 5.5, peak=8.0),
    }

    candidates = generate_candidates(z, _no_beats(), None, combine="k_of_n", k=2)
    near = [c for c in candidates if 4.5 < c.start_s < 6.0]

    assert len(near) == 1
    assert set(near[0].signals) == {"L_T", "R_T"}


def test_the_report_carries_all_three_rates_for_task_09() -> None:
    """Per-pair, union and combined are three different numbers and all are reported.

    Reporting only ``combined`` would hide the family-wise problem whenever the rule
    is ``"any"``, since the two are then equal.
    """
    n = 5000
    z = {(f"S{i}", ENG): _quiet(n, seed=i) for i in range(8)}
    report = candidate_report(z, _no_beats(), combine="k_of_n", k=3)
    record = report.to_provenance()

    assert set(record["flag_rates"]["per_pair"]) == {f"S{i}|{ENG}" for i in range(8)}
    assert record["flag_rates"]["union"] > record["flag_rates"]["combined"]
    assert record["combine"] == "k_of_n"
    assert record["k"] == 3


def test_an_invalid_k_is_refused() -> None:
    """``k`` above the number of pairs can never fire, which is a bug not a setting."""
    z: dict[Pair, F64] = {("L_T", ENG): _quiet(1000)}

    with pytest.raises(ValueError, match="k must be between"):
        generate_candidates(z, _no_beats(), None, combine="k_of_n", k=5)


def test_inverted_hysteresis_is_refused() -> None:
    """``z_exit`` above ``z_enter`` is not hysteresis, it is a typo."""
    with pytest.raises(ValueError, match="not hysteresis"):
        generate_candidates(
            {("L_T", ENG): _quiet(1000)}, _no_beats(), None, z_enter=2.0, z_exit=3.0
        )


def test_pairs_on_different_grids_are_refused() -> None:
    """Every pair shares the 10 ms grid; ragged input is a caller bug."""
    z: dict[Pair, F64] = {("L_T", ENG): _quiet(1000), ("R_T", ENG): _quiet(900)}

    with pytest.raises(ValueError, match="same grid"):
        generate_candidates(z, _no_beats(), None)


# ---------------------------------------------------------------------------
# cardiac suppression is band-scoped
# ---------------------------------------------------------------------------


def test_cardiac_suppression_is_scoped_to_one_band() -> None:
    """Show suppression is band-scoped - the task's own test.

    A cardiac-only synthetic yields nothing in ``100-300`` and still yields
    candidates in ``300-3000`` where a real artifact sits.

    From A.5 the QRS puts 32.1-65.2% of its energy in 100-300 Hz and 0.00-0.54% above
    300 Hz, so the heartbeat is a confound in exactly one band. Suppressing more
    would discard real artifacts landing near a beat, which at 364 bpm is most of the
    recording.

    The band name in this test read ``300-5000`` until 2026-09-22; A.5b moved the ENG
    band and the line did not follow. Band names are the exact strings in A.3.
    """
    dur_s = 30.0
    n = int(dur_s / GRID_S)
    beats = _beats(dur_s)
    window = (-0.010, 0.030)

    cardiac_trace = _quiet(n, seed=0)
    for beat in beats.t_s.tolist():
        cardiac_trace = _with_event(cardiac_trace, beat, beat + 0.020, peak=9.0)

    eng_trace = _with_event(_quiet(n, seed=1), 12.0, 12.4, peak=8.0)

    z: dict[Pair, F64] = {("L_T", CARDIAC): cardiac_trace, ("L_T", ENG): eng_trace}
    windows: dict[Pair, tuple[float, float] | None] = {
        ("L_T", CARDIAC): window,
        ("L_T", ENG): window,
    }

    candidates = generate_candidates(z, beats, windows)

    assert all(CARDIAC not in c.bands for c in candidates), "the beats were not suppressed"

    # Near the injection only: a standard-normal trace throws the occasional 3-sigma
    # frame of its own, which is the per-pair false rate doing exactly what invariant
    # 10b says it does, not a failure of suppression.
    in_eng = [c for c in candidates if ENG in c.bands and 11.5 < c.start_s < 12.5]
    assert len(in_eng) == 1
    assert in_eng[0].start_s <= 12.0 < in_eng[0].stop_s
    assert in_eng[0].peak_z == pytest.approx(8.0, abs=0.01)
    assert CARDIAC_SUPPRESSION_BANDS == (CARDIAC,)


def test_an_eng_artifact_sitting_on_a_beat_is_not_suppressed() -> None:
    """The case that discriminates band-scoped suppression from band-wide.

    Found by mutation: removing the band check changed nothing, because the ENG
    artifact in the test above is 400 ms long and a 40 ms peri-R window cannot cover
    it. Here the artifact sits **entirely inside** a window, so band-wide suppression
    deletes it and band-scoped suppression keeps it.

    This is the failure that matters in practice. At animal J's 364 bpm a beat
    arrives every 165 ms, so a 40 ms window covers 24% of the recording; suppressing
    ``300-3000`` too would silently discard a quarter of all short ENG artifacts, and
    A.5 says the QRS puts 0.00-0.54% of its energy up there - there is nothing to
    suppress.
    """
    dur_s = 30.0
    n = int(dur_s / GRID_S)
    beat_at = 10.0
    beats = BeatTrain(
        t_s=np.array([beat_at]), tag=np.array(["detected"], dtype="<U8")
    )
    window = (-0.010, 0.030)

    # 20 ms of artifact starting on the beat: inside the window at both ends.
    eng = _with_event(_quiet(n, seed=3), beat_at, beat_at + 0.020, peak=8.0)
    cardiac = _with_event(_quiet(n, seed=4), beat_at, beat_at + 0.020, peak=9.0)

    z: dict[Pair, F64] = {("L_T", ENG): eng, ("L_T", CARDIAC): cardiac}
    windows: dict[Pair, tuple[float, float] | None] = {
        ("L_T", ENG): window,
        ("L_T", CARDIAC): window,
    }

    near = [
        c
        for c in generate_candidates(z, beats, windows)
        if beat_at - 0.1 < c.start_s < beat_at + 0.1
    ]

    assert len(near) == 1, "the ENG artifact under the beat was suppressed"
    assert near[0].bands == (ENG,), f"expected ENG only, got {near[0].bands}"
    assert CARDIAC not in near[0].bands


def test_an_artifact_away_from_any_beat_survives_in_the_cardiac_band() -> None:
    """Suppression is peri-R, not band-wide: ``100-300`` still detects between beats.

    With a 165 ms RR and a 40 ms window, 76% of the record is outside any window, so
    a band-wide suppression would be a very different and much worse rule.
    """
    dur_s = 30.0
    n = int(dur_s / GRID_S)
    beats = BeatTrain(
        t_s=np.array([1.0, 2.0, 3.0]), tag=np.array(["detected"] * 3, dtype="<U8")
    )
    trace = _with_event(_quiet(n, seed=2), 15.0, 15.3, peak=8.0)

    candidates = generate_candidates(
        {("L_T", CARDIAC): trace}, beats, {("L_T", CARDIAC): (-0.010, 0.030)}
    )

    assert len(candidates) == 1
    assert candidates[0].start_s <= 15.0 < candidates[0].stop_s


def test_unmeasured_cardiac_windows_are_recorded_as_such() -> None:
    """``None`` means nobody looked - which is not the same as no confound.

    Task 02 has not been built and A.6's build order puts it *after* this task, so
    this is the state the pipeline is actually in today. Recording it keeps a reader
    from mistaking an unsuppressed file for a clean one.
    """
    z: dict[Pair, F64] = {("L_T", CARDIAC): _quiet(2000)}

    unmeasured = candidate_report(z, _beats(20.0), cardiac_windows=None)
    measured = candidate_report(z, _beats(20.0), cardiac_windows={("L_T", CARDIAC): (-0.01, 0.03)})

    assert unmeasured.cardiac_suppression == "not measured"
    assert unmeasured.suppressed_frames == 0
    assert measured.cardiac_suppression == "applied"
    assert measured.suppressed_frames > 0
    assert unmeasured.to_provenance()["cardiac_suppression"] == "not measured"


def test_a_flat_band_contributes_no_suppression() -> None:
    """Task 02 returns ``None`` per pair where the band is flat; that is not a window."""
    z: dict[Pair, F64] = {("L_T", CARDIAC): _quiet(2000)}
    report = candidate_report(z, _beats(20.0), cardiac_windows={("L_T", CARDIAC): None})

    assert report.suppressed_frames == 0


# ---------------------------------------------------------------------------
# the duration cap
# ---------------------------------------------------------------------------


def test_the_duration_cap_routes_to_review_rather_than_dropping() -> None:
    """A sustained level shift is a finding, so it is flagged, never discarded."""
    trace = _with_event(_quiet(8000), 10.0, 40.0, peak=7.0)
    report = candidate_report({("L_T", ENG): trace}, _no_beats(), duration_cap_s=5.0)

    assert report.n_candidates >= 1
    assert report.over_cap, "a 30 s span should be over a 5 s cap"
    capped = report.candidates[report.over_cap[0]]
    assert capped.stop_s - capped.start_s > 5.0
    assert capped in report.candidates, "routed to review, not dropped"


def test_an_unmeasured_cap_is_absent_not_zero() -> None:
    """Record why ``duration_cap_s`` is unset as of 2026-09-22.

    It is the 99th percentile of labelled segment durations across the 43 existing
    recordings, computed from ``*_segment_indices.mat``. The shared drive is not
    mounted on the build machine - no ``.gems-root`` marker anywhere under the user
    profile, no ``*_segment_indices.mat`` on any local path, no label ``.mat`` in the
    GEMSBlanking checkout - so it cannot be computed and has not been guessed.

    ``None`` therefore means *no cap applied, and that is recorded*. The provenance
    key is **absent** rather than null, and ``over_cap`` is empty because no cap ran -
    which a reader must not confuse with "nothing exceeded the cap".
    """
    trace = _with_event(_quiet(8000), 10.0, 40.0, peak=7.0)
    report = candidate_report({("L_T", ENG): trace}, _no_beats(), duration_cap_s=None)

    assert report.duration_cap_s is None
    assert report.over_cap == ()
    assert "duration_cap_s" not in report.to_provenance()
    assert report.n_candidates >= 1, "an uncapped long span is still a candidate"


def test_exceeding_the_cap_is_logged_by_the_bare_generator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``generate_candidates`` cannot carry the routing, so it says so out loud.

    A.1 fixes :class:`~gems_blanking_v2.types.Candidate`'s fields and the task's
    signature returns a bare list, so there is nowhere for "route to review" to live
    on the return value. Silently accepting the parameter and doing nothing with it
    would be worse than either alternative.
    """
    trace = _with_event(_quiet(8000), 10.0, 40.0, peak=7.0)
    with caplog.at_level(logging.WARNING, logger="gems_blanking_v2.detect.candidates"):
        generate_candidates({("L_T", ENG): trace}, _no_beats(), None, duration_cap_s=5.0)

    assert any("routed to review" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


def test_video_lowers_the_bar_only_where_motion_exceeds_threshold() -> None:
    """A z=2.5 event is found under motion and missed without it.

    2.5 is above :data:`VIDEO_ASSISTED_Z_ENTER` and below the 3.0 default, so it
    exists only because video licensed the lower bar.
    """
    trace = _with_event(_quiet(4000), 20.0, 20.5, peak=2.5)
    z: dict[Pair, F64] = {("L_T", ENG): trace}

    peak = 2.5
    assert VIDEO_ASSISTED_Z_ENTER < peak < 3.0
    assert [c for c in generate_candidates(z, _no_beats(), None) if 19 < c.start_s < 21] == []

    with_video = generate_candidates(z, _no_beats(), None, video=_Motion([(19.5, 21.0)]))
    near = [c for c in with_video if 19 < c.start_s < 21]

    assert len(near) == 1
    assert near[0].provenance == "video_assisted"


def test_motion_elsewhere_does_not_lower_the_bar_here() -> None:
    """The mask is per frame, so motion at 5 s cannot rescue an event at 20 s."""
    trace = _with_event(_quiet(4000), 20.0, 20.5, peak=2.5)
    candidates = generate_candidates(
        {("L_T", ENG): trace}, _no_beats(), None, video=_Motion([(5.0, 6.0)])
    )

    assert [c for c in candidates if 19 < c.start_s < 21] == []


def test_an_event_that_clears_the_normal_bar_stays_electrical() -> None:
    """Provenance records *why* a candidate exists, not whether the animal moved.

    A span that reached ``z_enter`` on its own would have been found with the camera
    switched off, so tagging it ``video_assisted`` would make the tag a record of
    coincidence rather than of evidence.
    """
    trace = _with_event(_quiet(4000), 20.0, 20.5, peak=8.0)
    candidates = generate_candidates(
        {("L_T", ENG): trace}, _no_beats(), None, video=_Motion([(19.5, 21.0)])
    )
    near = [c for c in candidates if 19 < c.start_s < 21]

    assert len(near) == 1
    assert near[0].provenance == "electrical"


def test_without_video_nothing_is_video_assisted() -> None:
    """The tag cannot appear when there was no camera."""
    trace = _with_event(_quiet(4000), 20.0, 20.5, peak=8.0)
    candidates = generate_candidates({("L_T", ENG): trace}, _no_beats(), None)

    assert all(c.provenance == "electrical" for c in candidates)


# ---------------------------------------------------------------------------
# what this step deliberately does not do
# ---------------------------------------------------------------------------


def test_cross_band_coincidence_does_not_gate_candidates() -> None:
    """Leave cross-band coincidence ungated - it is a task 11 feature, not a rule here.

    A single-band event and a two-band event must both survive under the default
    rule. Gating on coincidence would destroy the evidence task 12 needs to learn
    which is which - a generator that has already decided leaves the classifier
    nothing to decide.
    """
    n = 4000
    single: dict[Pair, F64] = {
        ("L_T", ENG): _with_event(_quiet(n, 0), 10.0, 10.4, peak=8.0),
        ("L_T", CARDIAC): _quiet(n, 1),
    }
    both: dict[Pair, F64] = {
        ("L_T", ENG): _with_event(_quiet(n, 0), 10.0, 10.4, peak=8.0),
        ("L_T", CARDIAC): _with_event(_quiet(n, 1), 10.0, 10.4, peak=8.0),
    }

    single_near = [c for c in generate_candidates(single, _no_beats(), None) if 9 < c.start_s < 11]
    both_near = [c for c in generate_candidates(both, _no_beats(), None) if 9 < c.start_s < 11]

    assert len(single_near) == 1, "a one-band event must not be suppressed"
    assert len(both_near) == 1
    assert single_near[0].bands == (ENG,)
    assert set(both_near[0].bands) == {ENG, CARDIAC}, (
        "the coincidence must still be visible in the output for task 11 to use"
    )


def test_the_operating_point_is_recorded_not_tuned() -> None:
    """``z_enter`` is swept and pinned in task 09; this module records what it used."""
    report = candidate_report({("L_T", ENG): _quiet(2000)}, _no_beats(), z_enter=3.0)
    record = report.to_provenance()

    assert record["z_enter"] == 3.0
    assert record["z_exit"] == 1.5
    assert 2.0 <= record["z_enter"] <= 4.0, "the defensible range"


def test_the_report_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """One line a human can read in a QC table, with all three rates on it."""
    z = {(f"S{i}", ENG): _quiet(5000, seed=i) for i in range(4)}
    with caplog.at_level(logging.INFO, logger="gems_blanking_v2.detect.candidates"):
        candidate_report(z, _no_beats())

    messages = [r.getMessage() for r in caplog.records]
    assert any("candidates" in m and "union" in m and "worst pair" in m for m in messages)
    assert any("duration cap unset" in m for m in messages)


def test_an_empty_pair_set_yields_nothing() -> None:
    """No z is not an error, it is no candidates."""
    assert generate_candidates({}, _no_beats(), None) == []
    assert flag_rates({}).n_frames == 0
