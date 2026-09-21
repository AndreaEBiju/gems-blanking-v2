"""The constants must be internally consistent, and their stated rationales must hold.

Two of these tests document a disagreement with ``IMPLEMENTATION.md`` rather than
confirming it. That is deliberate: the contradiction is the finding, and a test is
where it stays visible.
"""

from __future__ import annotations

import numpy as np
import pytest
from gems_blanking_v2.constants import (
    BAND_DOF_TARGET,
    BANDS,
    CONSUMERS,
    ENG_BAND,
    ENG_BAND_DOF_IS_EXCEPTION,
    ENG_CORNER_TRADEOFF,
    GRID_S,
    HR_BAND,
    NERVE_BANNED_BANDS,
    REFERENCE_STATISTIC,
    STOMACH_BANNED_BANDS,
    frame_centre_s,
    n_frames,
)

# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------


def test_grid_is_ten_milliseconds() -> None:
    assert GRID_S == 0.010


@pytest.mark.parametrize(
    "duration_s,expected",
    [(0.0, 0), (0.009, 0), (0.010, 1), (1.005, 100), (600.0, 60_000)],
)
def test_frame_count_drops_a_trailing_partial_frame(duration_s: float, expected: int) -> None:
    assert n_frames(duration_s) == expected


def test_frame_centres_are_half_a_step_in_and_one_step_apart() -> None:
    assert frame_centre_s(0) == pytest.approx(0.005)
    assert frame_centre_s(1) - frame_centre_s(0) == pytest.approx(GRID_S)
    assert frame_centre_s(99) == pytest.approx(0.995)


def test_the_last_frame_centre_is_inside_the_record() -> None:
    duration_s = 1.005
    assert frame_centre_s(n_frames(duration_s) - 1) < duration_s


def test_grid_helpers_reject_negatives() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        n_frames(-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        frame_centre_s(-1)


# ---------------------------------------------------------------------------
# the bands
# ---------------------------------------------------------------------------


def test_there_are_six_bands_and_their_names_state_their_corners() -> None:
    assert len(BANDS) == 6
    for name, spec in BANDS.items():
        lo, hi = name.split("-")
        assert float(lo) == spec.lo_hz
        assert float(hi) == spec.hi_hz


def test_band_corners_are_ordered_and_windows_positive() -> None:
    for name, spec in BANDS.items():
        assert spec.lo_hz < spec.hi_hz, name
        assert spec.window_s > 0.0, name
        assert spec.bandwidth_hz == pytest.approx(spec.hi_hz - spec.lo_hz), name


def test_the_eng_band_is_three_kilohertz_not_five() -> None:
    """A.5b, measured 2026-09-19: event energy reaches background by ~4 kHz."""
    assert ENG_BAND in BANDS
    assert BANDS[ENG_BAND].hi_hz == 3000.0
    assert "300-5000" not in BANDS


def test_the_cardiac_band_is_ten_to_one_fifty_not_one_to_a_hundred() -> None:
    """Ruled 2026-09-21 with task 05, and ``1-100`` is gone rather than kept.

    A superseded band left in ``BANDS`` is a second rule competing with the binding
    one, exactly as with the percentile reference. The hrv consumer must name the
    band its own detector reads.
    """
    assert HR_BAND == "10-150"
    assert HR_BAND in BANDS
    assert "1-100" not in BANDS
    assert BANDS[HR_BAND].lo_hz == 10.0
    assert BANDS[HR_BAND].hi_hz == 150.0
    assert {c.band for c in CONSUMERS if c.name == "hrv"} == {HR_BAND}


def test_every_band_has_the_degrees_of_freedom_a_3_records() -> None:
    """The exact ``2*B*T`` per band, so any window or corner change trips this.

    Tighter than the aggregate check below, and the reason that one had to be
    loosened: ``10-150`` came in at 28, not 30.
    """
    assert {name: round(spec.dof, 1) for name, spec in BANDS.items()} == {
        "300-3000": 135.0,
        "100-300": 30.0,
        "10-150": 28.0,
        "2-50": 29.8,
        "0.5-3": 30.0,
        "0-2": 30.0,
    }


def test_every_band_but_the_eng_band_was_sized_for_thirty_degrees_of_freedom() -> None:
    """The dof rationale, at the **7%** tolerance ``10-150`` forced.

    **This tolerance was widened from 2% on 2026-09-21** and the reason is worth
    stating rather than hiding: the new cardiac band uses a round 100 ms window,
    which gives ``2 * 140 * 0.100 = 28``, a 6.7% shortfall. 107 ms would hit 30
    exactly. Every other band still sits inside 1%, and
    :func:`test_every_band_has_the_degrees_of_freedom_a_3_records` holds the exact
    figures, so nothing is actually less constrained than before.
    """
    for name, spec in BANDS.items():
        if name == ENG_BAND:
            continue
        assert spec.dof == pytest.approx(BAND_DOF_TARGET, rel=0.07), name


def test_the_eng_band_does_not_meet_the_stated_dof_rationale() -> None:
    """The ENG band's ``2*B*T`` is 135, as A.3 now records against that row.

    It was 235 before the corner moved to 3 kHz. Task 06's acceptance test "dof
    within 20% for every band" must exempt this band; the window is not to be
    shortened to 5.6 ms to make it pass, because 25 ms is a time-resolution choice.
    """
    assert ENG_BAND_DOF_IS_EXCEPTION
    assert BANDS[ENG_BAND].dof == pytest.approx(135.0)
    assert BANDS[ENG_BAND].dof > 4.0 * BAND_DOF_TARGET


def test_no_band_carries_a_reference_percentile() -> None:
    """A.3: the percentile reference is gone from code, not kept as history.

    A per-band percentile would be a second reference rule competing with the
    binding one, and the competing rule is the one that was measured to be wrong.
    """
    for name, spec in BANDS.items():
        assert not hasattr(spec, "ref_pct"), name


def test_the_reference_statistic_is_the_median_of_the_log_envelope() -> None:
    """Hard invariant 5: one reference rule, and this is it."""
    assert REFERENCE_STATISTIC == "median_of_log"


def test_log_transform_centres_the_null_as_invariant_5_claims() -> None:
    """The measurement behind invariant 5, reproduced on a right-skewed null.

    A chi-square-like envelope has median z = 1.0 and p90 = 3.05 under the linear
    p10 reference; on the log envelope the null is symmetric, median 0.
    """
    rng = np.random.default_rng(0)
    env = np.abs(rng.normal(size=(200_000, 30))).mean(axis=1)  # positive, right-skewed

    linear_z = (env - np.percentile(env, 10)) / (1.4826 * np.median(np.abs(env - np.median(env))))
    assert float(np.median(linear_z)) > 0.9

    log_env = np.log(env)
    log_z = (log_env - np.median(log_env)) / (
        1.4826 * np.median(np.abs(log_env - np.median(log_env)))
    )
    assert float(np.median(log_z)) == pytest.approx(0.0, abs=0.01)
    assert abs(float(np.percentile(log_z, 90)) + float(np.percentile(log_z, 10))) < 0.15


# ---------------------------------------------------------------------------
# the consumers
# ---------------------------------------------------------------------------


def test_there_are_seven_consumers_with_unique_names() -> None:
    names = [c.name for c in CONSUMERS]
    assert len(names) == 7
    assert len(set(names)) == 7


def test_every_consumer_names_a_band_that_exists() -> None:
    """The A.4 table still says ``300-5000``; using :data:`ENG_BAND` keeps it honest."""
    for c in CONSUMERS:
        assert c.band in BANDS, c.name


def test_only_velocity_reads_more_than_one_signal() -> None:
    """Only velocity reads two signals (hard invariant 2).

    Masks are never merged across consumers; the single within-consumer intersection
    is velocity needing V1 and V3 both valid.
    """
    multi = [c for c in CONSUMERS if len(c.signals) > 1]
    assert [c.name for c in multi] == ["velocity"]
    assert multi[0].signals == ("V1", "V3")


def test_spikes_and_velocity_read_the_eng_band() -> None:
    by_name = {c.name: c for c in CONSUMERS}
    assert by_name["spikes"].band == ENG_BAND
    assert by_name["velocity"].band == ENG_BAND


def test_no_nerve_consumer_uses_a_stomach_only_band() -> None:
    nerve_signals = {"T", "V1", "V2", "V3"}
    for c in CONSUMERS:
        if set(c.signals) & nerve_signals:
            assert c.band not in NERVE_BANNED_BANDS, c.name


def test_no_stomach_consumer_uses_the_eng_band() -> None:
    for c in CONSUMERS:
        if "stomach_ref" in c.signals:
            assert c.band not in STOMACH_BANNED_BANDS, c.name


def test_consumer_specs_are_immutable() -> None:
    """Thresholds are fixed and stateless (hard invariant 7)."""
    with pytest.raises((AttributeError, TypeError)):
        CONSUMERS[0].band = "0-2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# the measured table
# ---------------------------------------------------------------------------


def test_the_eng_corner_tradeoff_reproduces_the_quoted_deltas() -> None:
    """Narrowing 5 kHz -> 3 kHz lowers sigma 24% and raises the event rate 41%."""
    narrow, wide = ENG_CORNER_TRADEOFF[3000], ENG_CORNER_TRADEOFF[5000]
    sigma_drop = 1.0 - narrow["sigma_t_uv"] / wide["sigma_t_uv"]
    rate_gain = narrow["events_per_s"] / wide["events_per_s"] - 1.0
    assert sigma_drop == pytest.approx(0.24, abs=0.01)
    assert rate_gain == pytest.approx(0.41, abs=0.01)
    assert narrow["cardiac_peak_base"] < wide["cardiac_peak_base"]
