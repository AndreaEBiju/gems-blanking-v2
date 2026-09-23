"""Tests for the peri-R cardiac window measurement.

Every signal comes from ``conftest``; every test states a number a bug would move.
"""

from __future__ import annotations

import numpy as np
import pytest
from gems_blanking_v2.bands.envelope import _decimated_rate, impulse_response_length_s
from gems_blanking_v2.constants import BANDS, GRID_S
from gems_blanking_v2.physio.cardiac_window import (
    ENG_BAND_NAME,
    BandVerdict,
    cardiac_window_report,
    measure_cardiac_window,
)
from tests.conftest import make_ecg

FS = 12000.0
DUR_S = 60.0
OFFSET_S = 0.020
"""A conduction delay wider than the ENG extent, for the peak-anchoring tests."""

SHARP_QRS_MS = 1.0
"""A QRS narrow enough to carry 300-3000 Hz content.

The generator's 10 ms default is a Hanning window whose spectrum runs out by ~200 Hz,
so the ENG band is empty and every measurement in it is of noise. A real R-wave has a
near-discontinuous slope; 1 ms stands in for that.
"""
RR_S = 0.165
"""Animal J's measured RR, so every band/profile ratio is the one that matters."""

HALF_FRAC = 0.40
PROFILE_S = 2.0 * HALF_FRAC * RR_S
"""132 ms. The widest profile the RR allows, and the limit every status turns on."""


def _impz(band: str) -> float:
    """Return the band-pass's own impulse response, at the rate the band runs at."""
    spec = BANDS[band]
    return impulse_response_length_s(
        spec.lo_hz, spec.hi_hz, _decimated_rate(FS, spec.hi_hz)
    )


def _report(**kwargs: object) -> dict[str, BandVerdict]:
    """Measure one synthetic ECG channel across all six bands, keyed by band."""
    ecg = make_ecg(FS, DUR_S, rr_s=RR_S, **kwargs)  # type: ignore[arg-type]
    verdicts = cardiac_window_report(ecg.signal[:, None], FS, ["E1"], ecg.beats_s)
    return {v.band: v for v in verdicts}


# ---------------------------------------------------------------------------
# the central claim: the signal average resolves what the envelope cannot
# ---------------------------------------------------------------------------


def test_the_signal_average_beats_the_envelope_window_in_the_eng_band() -> None:
    """The ENG extent is bounded by QRS + ringing; the envelope extent is the window.

    This is the whole reason the method changed. The generator's QRS is 10 ms wide and
    ``300-3000``'s impulse response is ~5 ms, so an extent beyond their sum is
    measuring something else - and it must come in under the 25 ms envelope window, or
    the historical ``+/-15 ms`` was never falsifiable in the first place.

    The lower bound is *not* the kernel width: the band-limited extent is set by how
    much of the kernel lies above 300 Hz, which is why :data:`SHARP_QRS_MS` is needed
    at all.
    """
    v = _report(qrs_width_ms=SHARP_QRS_MS)[ENG_BAND_NAME]
    impz_ms = _impz(ENG_BAND_NAME) * 1e3

    assert v.status == "measured"
    assert v.extent_signal_s is not None
    assert v.duration_s < v.window_s, "the signal extent must beat the envelope window"
    assert v.duration_s * 1e3 <= SHARP_QRS_MS + 2 * impz_ms

    # The floor is the filter, not the event. A band-pass that rings for 5 ms cannot
    # turn a transient into a 1 ms burst, so an extent under impz means the carrier
    # oscillation was counted instead of the envelope of the burst - which is what
    # reading a contiguous run off the raw coherent average does.
    assert v.duration_s >= impz_ms / 1e3

    assert v.extent_env_s is not None
    env_span_ms = (v.extent_env_s[1] - v.extent_env_s[0]) * 1e3
    assert env_span_ms > v.duration_s * 1e3, "the envelope cannot resolve what this can"


def test_the_envelope_extent_tracks_the_window_not_the_physiology() -> None:
    """Quadruple the QRS width and the envelope measurement does not move at all.

    The direct demonstration that the old method measured a filter setting. Between a
    0.5 ms and a 2 ms QRS the coherent **signal** extent goes 6.7 -> 11.6 ms, while the
    envelope extent is 20.0 ms for both - identical to the sample, because a 25 ms
    boxcar cannot report anything narrower than itself. Every extent in the 2026-09-19
    prior was one of these.

    The signal extent scales by less than the QRS does because ~5 ms of it is the
    band-pass's own ringing, which is common to both. That floor is real and is what
    the consumer sees; it is just not 25 ms.
    """
    sharp = _report(qrs_width_ms=0.5)[ENG_BAND_NAME]
    broad = _report(qrs_width_ms=2.0)[ENG_BAND_NAME]

    assert sharp.status == broad.status == "measured"
    assert broad.duration_s > sharp.duration_s * 1.5, "the signal average sees the change"

    assert sharp.extent_env_s is not None
    assert broad.extent_env_s is not None
    sharp_env = sharp.extent_env_s[1] - sharp.extent_env_s[0]
    broad_env = broad.extent_env_s[1] - broad.extent_env_s[0]
    assert sharp_env == pytest.approx(broad_env, abs=1e-9), "the envelope sees nothing"
    assert sharp_env >= sharp.window_s - GRID_S, "pinned to the window, one frame"
    assert sharp_env > broad.duration_s


# ---------------------------------------------------------------------------
# the four statuses, and what each is allowed to say
# ---------------------------------------------------------------------------


def test_only_a_measured_verdict_yields_a_blanking_extent() -> None:
    """``no_window``, ``unresolvable`` and ``not_applicable`` blank nothing.

    Why the status exists rather than ``tuple | None``: three different conclusions
    collapsed into one ``None``, and two of them are opposites.
    """
    for v in _report().values():
        if v.status == "measured":
            assert v.extent_signal_s is not None
            assert v.duration_s > 0.0
        else:
            assert v.extent_signal_s is None, v.band
            assert v.duration_s == 0.0, v.band
            assert v.duty_cycle(RR_S) == 0.0, v.band


def test_a_band_whose_ringing_outruns_the_profile_is_unresolvable() -> None:
    """``10-150``'s 141 ms impulse response exceeds the 132 ms profile at this RR.

    Anything measured there is a truncated view of the filter's own ringing, and the
    profile cannot be widened - it is capped at ``0.4 * RR`` either side, so the band
    is unresolvable **by construction at this heart rate**, not in this recording. The
    criterion is the impulse response and not the band's period: 1/150 Hz is 6.7 ms
    and would have called this resolvable.
    """
    v = _report()["10-150"]

    assert _impz("10-150") > PROFILE_S > 1.0 / BANDS["10-150"].hi_hz
    assert v.status == "unresolvable"
    assert v.extent_signal_s is None
    assert "impulse response" in v.reason
    assert "by construction" in v.reason


def test_the_fundamental_band_is_not_applicable_before_it_is_unresolvable() -> None:
    """``2-50`` holds the 6 Hz fundamental, so peri-R blanking is the wrong operation.

    Its ringing also outruns the profile, so both rules fire; ``not_applicable`` wins
    because it routes the problem to task 13/14 as a notch, while ``unresolvable``
    would only say to look again.
    """
    v = _report()["2-50"]

    assert _impz("2-50") > PROFILE_S, "the unresolvable rule would also have fired"
    assert v.status == "not_applicable"
    assert "fundamental" in v.reason
    assert "notch" in v.reason


def test_the_two_slow_bands_are_unresolvable_not_null() -> None:
    """``0-2`` and ``0.5-3`` sit below 6 Hz **and** ring for seconds.

    Reporting ``no_window`` here would claim a negative the method cannot support. The
    reason carries the spectral argument too, so the confirmation is not lost.
    """
    for band in ("0-2", "0.5-3"):
        v = _report()[band]

        assert v.status == "unresolvable", band
        assert v.extent_signal_s is None, band
        assert "impulse response" in v.reason, band


def test_white_noise_blanks_nothing_anywhere() -> None:
    """The true negative: nothing is time-locked, so no band may produce an extent."""
    ecg = make_ecg(FS, DUR_S, amp_uv=0.0, noise_uv=5.0, rr_s=RR_S, seed=3)
    verdicts = cardiac_window_report(ecg.signal[:, None], FS, ["E1"], ecg.beats_s)

    for v in verdicts:
        assert v.status != "measured", f"{v.band}: {v.reason}"
        assert v.extent_signal_s is None, v.band
    resolved = [v for v in verdicts if v.status == "no_window"]
    assert resolved, "the two fast bands resolve, so they must report a real negative"
    for v in resolved:
        assert v.peak_over_null < 1.0, v.band


# ---------------------------------------------------------------------------
# the half-window assertion, the offset window, and the spike-consumer measure
# ---------------------------------------------------------------------------


def test_a_half_window_reaching_the_neighbouring_beat_raises() -> None:
    """Asserted, not assumed: at ``>= RR/2`` the null contains the next QRS.

    The default 0.40 gives +/-66 ms at animal J's 165 ms RR, comfortably inside. 0.50
    would put the anti-phase trigger's own neighbouring beats inside every window and
    inflate the null until nothing could exceed it - a silent false negative.
    """
    ecg = make_ecg(FS, 10.0, rr_s=RR_S)

    cardiac_window_report(ecg.signal[:, None], FS, ["E1"], ecg.beats_s, None, 0.49)
    with pytest.raises(ValueError, match="neighbouring beat"):
        cardiac_window_report(ecg.signal[:, None], FS, ["E1"], ecg.beats_s, None, 0.50)


def test_a_window_offset_from_r_is_still_found() -> None:
    """The extent is anchored on the peak, not on lag 0, so a delay does not hide it.

    Requiring the crossing at lag 0 exactly reported ``no_window`` for all three
    stomach channels of ``gems_j_t01_ms3_bl_230315`` at 5-6x the null - real
    contamination that would then never have been blanked.

    The trace is rolled by :data:`OFFSET_S`, which is **wider than the extent itself**,
    so lag 0 falls outside the deflection entirely and a lag-0 anchor has nothing to
    grow from. A smaller roll would leave lag 0 inside the extent and the old
    behaviour would survive the test.
    """
    ecg = make_ecg(FS, DUR_S, rr_s=RR_S, qrs_width_ms=SHARP_QRS_MS)

    base = cardiac_window_report(ecg.signal[:, None], FS, ["E1"], ecg.beats_s)[0]
    rolled = cardiac_window_report(
        np.roll(ecg.signal, int(round(OFFSET_S * FS)))[:, None], FS, ["E1"], ecg.beats_s
    )[0]

    assert base.status == rolled.status == "measured"
    assert base.extent_signal_s is not None
    assert rolled.extent_signal_s is not None
    assert base.duration_s < OFFSET_S, "the roll must clear the extent, or lag 0 stays in"
    assert rolled.extent_signal_s[0] > 0.0, "the whole window sits after R"
    assert rolled.peak_lag_s - base.peak_lag_s == pytest.approx(OFFSET_S, abs=0.0005)
    assert rolled.duration_s == pytest.approx(base.duration_s, abs=0.002)

    # The crossing rate is read at that same lag, not at the centre of the profile.
    assert rolled.crossing_rise is not None
    assert rolled.crossing_rise > 2.0


def test_the_spike_consumers_own_threshold_rate_rises_at_lag_zero() -> None:
    """``300-3000`` gets the measure matched to what spike detection actually does.

    Rate of ``|x| > 4.5 sigma`` at lag 0 over its off-peak rate. Only the ENG band
    carries it - no other consumer thresholds samples this way.
    """
    report = _report(qrs_width_ms=SHARP_QRS_MS)

    assert report[ENG_BAND_NAME].crossing_rise is not None
    assert report[ENG_BAND_NAME].crossing_rise > 1.0
    assert all(v.crossing_rise is None for b, v in report.items() if b != ENG_BAND_NAME)


def test_a_measured_verdict_survives_the_jittered_null_too() -> None:
    """A QRS is locked to R, so both nulls agree. Disagreement would mean rhythm."""
    v = _report(qrs_width_ms=SHARP_QRS_MS)[ENG_BAND_NAME]

    assert v.status == "measured"
    assert v.jitter_agrees


# ---------------------------------------------------------------------------
# serialisation and the legacy signature
# ---------------------------------------------------------------------------


def test_an_absent_measurement_is_an_absent_key_never_a_null() -> None:
    """The missing-scalar convention: JSON has no NaN, so the key simply is not there."""
    for v in _report().values():
        record = v.to_provenance()

        assert None not in record.values()
        assert ("extent_signal_start_s" in record) == (v.status == "measured")
        assert ("crossing_rise" in record) == (v.crossing_rise is not None)
        assert record["status"] == v.status


def test_the_dict_signature_collapses_three_conclusions_into_one_none() -> None:
    """:func:`measure_cardiac_window` keeps the task's signature and loses information.

    Documented and tested so nobody builds on it by accident: the mapping cannot tell
    ``unresolvable`` from ``not_applicable``, which is exactly the distinction task 13
    needs in order to know whether to notch or to look again.
    """
    ecg = make_ecg(FS, DUR_S, rr_s=RR_S)

    class _Chan:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Rec:
        data = ecg.signal[:, None]
        fs = FS
        channels = (_Chan("E1"),)

    mapping = measure_cardiac_window(_Rec(), ecg.beats_s)  # type: ignore[arg-type]

    assert mapping[("E1", ENG_BAND_NAME)] is not None
    assert mapping[("E1", "2-50")] is None
    assert mapping[("E1", "10-150")] is None
