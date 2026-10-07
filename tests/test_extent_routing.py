"""Task 14: correct before subtract before reject; clipping bypasses the classifier."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.bands import envelope
from gems_blanking_v2.bands.envelope import band_envelope_for, log_envelope
from gems_blanking_v2.bands.reference import epoch_reference
from gems_blanking_v2.bands.zscore import zscore
from gems_blanking_v2.extent import routing as rt
from gems_blanking_v2.extent.tolerance import ToleranceTable

from tests.conftest import inject_artifact, make_eng

FS = 8000.0
DUR_S = 60.0
TOL = ToleranceTable({"spikes": 3.0, "velocity": 3.0, "mmc": 3.0, "slow_wave": 3.0,
                      "breathing": 3.0}, source="synthetic test table")

def _zmap(x: np.ndarray, t0: float) -> dict[tuple[str, str], np.ndarray]:
    """Build detection's z for the test signal, as the pipeline would hand it over."""
    out = {}
    for band in ("0-2", "2-50", "0.5-3", "10-150", "100-300"):
        le = log_envelope(band_envelope_for(x, FS, band))
        ref = epoch_reference(le, signal="S", band=band)
        out[("S", band)] = zscore(le, ref, signal="S", band=band)
    return out


_Cached = tuple[np.ndarray, dict[tuple[str, str], np.ndarray], rt.EngTrace]
_CACHE: dict[tuple[int, float], _Cached] = {}


def _ev(eid: str, x: np.ndarray, span: tuple[float, float], t0: float = 0.0,
        **kw: Any) -> rt.EventEvidence:  # noqa: ANN401
    """EventEvidence on the recording timeline, x starting at ``t0``; z and ENG built once."""
    key = (id(x), t0)
    if key not in _CACHE or _CACHE[key][0] is not x:
        _CACHE[key] = (x, _zmap(x, t0), rt.eng_trace("S", x, FS, x_t0_s=t0))
    _x, z, eng = _CACHE[key]
    return rt.EventEvidence(eid, "S", x, FS, (span[0] + t0, span[1] + t0), t0, z=z, z_t0_s=t0,
                            eng=eng, **kw)



def _host(seed: int = 0) -> np.ndarray:
    """Broadband noise: every band has a floor to be measured against."""
    return np.random.default_rng(seed).normal(0.0, 10.0, int(DUR_S * FS))


def _drift() -> tuple[np.ndarray, tuple[float, float]]:
    x, truth = inject_artifact(_host(), FS, 30.0, 6.0, "drift", 40.0)
    return x, (truth.start_s, truth.stop_s)


def test_a_pure_drift_routes_to_correct_in_the_eng_consumer() -> None:
    x, span = _drift()
    d = rt.route_event(_ev("e1", x, span), "spikes", TOL)
    assert d.route == "correct" and not d.masks
    assert d.in_band_value < rt.SPIKE_MAX_SIGMA  # sample level, in ENG sigma


def test_the_same_drift_routes_to_reject_in_the_0_2_hz_consumer() -> None:
    x, span = _drift()
    d = rt.route_event(_ev("e1", x, span), "slow_wave", TOL)
    assert d.route == "reject" and d.masks
    assert d.in_band_value > TOL.for_consumer("slow_wave")  # the table's own log-z scale


def test_a_clipped_span_is_masked_with_no_model_call() -> None:
    x, truth = inject_artifact(_host(), FS, 10.0, 0.5, "clip", 0.5)
    frames = rt.clip_frames(x, FS, rail_uv=truth.rail_uv, frac_min=0.5)
    assert frames[int(10.1 / 0.01)] and not frames[int(20.0 / 0.01)]
    clipped = _ev("clip", x, (truth.start_s, truth.stop_s), clipped=True)

    def classifier(_ev: rt.EventEvidence) -> bool:
        raise AssertionError("the classifier was invoked on a clipped span")

    out = rt.route_events([clipped], ["spikes", "slow_wave"], TOL, classify=classifier)
    assert [(d.consumer, d.route) for d in out] == [("spikes", "clip"), ("slow_wave", "clip")]
    assert all(d.masks for d in out)


def test_unclipped_events_do_go_through_the_classifier() -> None:
    x, span = _drift()
    calls: list[str] = []

    def classifier(ev: rt.EventEvidence) -> bool:
        calls.append(ev.event_id)
        return ev.event_id == "yes"

    evs = [_ev("yes", x, span), _ev("no", x, span)]
    out = rt.route_events(evs, ["spikes"], TOL, classify=classifier)
    assert calls == ["yes", "no"] and [d.event_id for d in out] == ["yes"]


def _tribo() -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    host = _host(1)
    x, truth = inject_artifact(host, FS, 40.0, 0.4, "tribo", 30.0, seed=2)
    return x, x - host, (truth.start_s, truth.stop_s)


def test_a_failed_subtraction_falls_back_to_reject_with_its_residual() -> None:
    x, artifact, span = _tribo()
    half = _ev("t", x, span, subtract=lambda s: s - 0.5 * artifact)
    d = rt.route_event(half, "spikes", TOL)
    assert d.route == "reject" and d.residual_ratio >= 1.0
    assert d.residual is not None and d.residual.size > 0  # the residual is emitted


def test_a_verified_subtraction_keeps_the_data() -> None:
    x, artifact, span = _tribo()
    exact = _ev("t", x, span, subtract=lambda s: s - artifact)
    d = rt.route_event(exact, "spikes", TOL)
    assert d.route == "subtract" and not d.masks and d.residual_ratio < 1.0


def test_an_in_band_event_that_is_not_stereotyped_is_rejected() -> None:
    x, _artifact, span = _tribo()
    assert rt.route_event(_ev("t", x, span), "spikes", TOL).route == "reject"


@pytest.mark.parametrize("consumer", sorted(rt.LINE_NOISE_ACTIONS))
def test_line_noise_never_routes_to_reject(consumer: str) -> None:
    x, _artifact, span = _tribo()
    ev = _ev("hum", x, span, mains_dominant=True)
    d = rt.route_event(ev, consumer, TOL)
    assert d.route == "line_noise" and not d.masks
    assert d.action == rt.LINE_NOISE_ACTIONS[consumer]


def test_line_noise_actions_follow_ruling_c_item_4() -> None:
    a = rt.LINE_NOISE_ACTIONS
    assert a["spikes"] == "per_minute_cuff_distrust"
    assert a["hrv"] == "hum_lock_persistence_test"
    assert a["mmc"] == a["slow_wave"] == "ant1_notch_rule"
    assert "reject" not in a.values()


def test_mains_dominance_needs_both_fixed_thresholds() -> None:
    th = rt.LineNoiseThresholds(0.3, 0.5, source="hum inventory (synthetic)")
    assert rt.is_mains_dominant({"line_ratio_max": 0.4, "line_plv_max": 0.6}, th)
    assert not rt.is_mains_dominant({"line_ratio_max": 0.4, "line_plv_max": 0.2}, th)
    assert not rt.is_mains_dominant({"line_ratio_max": 0.4}, th)  # missing is not mains
    with pytest.raises(ValueError, match="source"):
        rt.LineNoiseThresholds(0.3, 0.5, source="")


def test_clipping_wins_over_line_noise_and_every_route_is_counted() -> None:
    x, span = _drift()
    evs = [_ev("c", x, span, clipped=True, mains_dominant=True),
           _ev("d", x, span)]
    out = rt.route_events(evs, ["spikes", "slow_wave"], TOL, classify=lambda _e: True)
    counts = rt.route_counts(out)
    assert counts["spikes"]["clip"] == 1 and counts["spikes"]["correct"] == 1
    assert counts["slow_wave"]["clip"] == 1 and counts["slow_wave"]["reject"] == 1
    assert set(counts["spikes"]) == set(rt.ROUTES)
    rows = rt.decisions_table(out)
    assert len(rows) == 4 and all("nan" not in str(r).lower() for r in rows)


def test_clip_frames_refuses_an_undeclared_rail() -> None:
    with pytest.raises(ValueError, match="rail_uv"):
        rt.clip_frames(_host(), FS, rail_uv=0.0, frac_min=0.5)


def test_a_short_large_transient_in_a_long_span_is_not_corrected() -> None:
    """1 ms at 75 sigma inside a 0.5 s span: its RMS is small, its peak is not."""
    host = _host(3)
    x, truth = inject_artifact(host, FS, 20.0, 0.001, "tribo", 75.0, seed=4)
    span = (truth.start_s - 0.25, truth.start_s + 0.25)
    for consumer in ("spikes", "velocity"):
        d = rt.route_event(_ev("t", x, span), consumer, TOL)
        assert d.route == "reject" and d.masks, d.reason
        assert d.in_band_value > rt.SPIKE_MAX_SIGMA


def test_an_excess_of_eng_crossings_is_in_band_even_under_the_cap() -> None:
    x, _artifact, span = _tribo()  # 30 sigma crackle: under the 40 sigma cap
    v = rt.in_band_verdict(_ev("v", x, span), "spikes", TOL)
    assert not v.separable and v.value < rt.SPIKE_MAX_SIGMA and "crossings" in v.detail


def test_hrv_routes_from_the_operational_verdict() -> None:
    x, span = _drift()
    with pytest.raises(ValueError, match="hrv_changed"):
        rt.route_event(_ev("h", x, span), "hrv", TOL)
    assert rt.route_event(_ev("h", x, span, hrv_changed=False), "hrv",
                          TOL).route == "correct"
    assert rt.route_event(_ev("h", x, span, hrv_changed=True), "hrv",
                          TOL).route == "reject"


def test_nan_stays_nan_and_nothing_is_invented() -> None:
    """Invariant 8: interpolation is for the filter only; the residual carries NaN."""
    x, artifact, span = _tribo()
    x = x.copy()
    gap = slice(int((span[0] + 0.1) * FS), int((span[0] + 0.15) * FS))
    x[gap] = np.nan
    d = rt.route_event(_ev("t", x, span,
                                        subtract=lambda s: s - np.nan_to_num(artifact)),
                       "spikes", TOL)
    assert d.residual is not None and np.isnan(d.residual).any()
    y, rate = rt._band(x, FS, "300-3000")
    assert np.isnan(y[int((span[0] + 0.11) * rate)])
    assert np.isfinite(y[int((span[0] + 0.3) * rate)])


def test_the_band_filters_are_the_detection_sides_own() -> None:
    """Pinned: routing must use bands.envelope's design (inside the generation hash)."""
    assert vars(rt)["_band_limit"] is vars(envelope)["_band_limit"]
    assert vars(rt)["_decimate_for"] is vars(envelope)["_decimate_for"]


def test_reason_codes_carry_no_numbers() -> None:
    x, span = _drift()
    out = rt.route_events([_ev("d", x, span)], ["spikes", "slow_wave"], TOL,
                          classify=lambda _e: True)
    assert all(d.reason_code and not any(ch.isdigit() for ch in d.reason_code) for d in out)


def test_the_artifact_cap_catches_a_transient_the_crossing_count_cannot() -> None:
    """Catch a transient on real-looking ENG with the 40 sigma cap alone.

    20 spikes/s well over 4.5 sigma: one transient adds a crossing or two - no
    significant excess - so only the consumer's own 40 sigma cap rejects it.
    """
    host = make_eng(FS, DUR_S, rate_hz=20.0, spike_uv=80.0, noise_uv=6.0, seed=5).signal
    x, truth = inject_artifact(host, FS, 20.0, 0.001, "tribo", 75.0, seed=4)
    span = (truth.start_s - 0.25, truth.start_s + 0.25)
    v = rt.in_band_verdict(_ev("v", x, span), "spikes", TOL)
    assert not v.separable and v.value > rt.SPIKE_MAX_SIGMA
    p_excess = float(re.search(r"\(p ([0-9.e+-]+)\)", v.detail).group(1))  # type: ignore[union-attr]
    assert p_excess >= rt.EXCESS_P  # the excess test alone would have passed it


def test_one_origin_convention_a_span_on_the_recording_timeline() -> None:
    """X starting at 120 s: the same event routes the same way (span on the recording)."""
    x, span = _drift()
    at0 = rt.route_event(_ev("d", x, span), "slow_wave", TOL)
    y = x.copy()
    at120 = rt.route_event(_ev("d", y, span, t0=120.0), "slow_wave", TOL)
    assert (at0.route, at0.in_band_value) == (at120.route, at120.in_band_value)
    assert rt.route_event(_ev("d", y, span, t0=120.0), "spikes", TOL).route == "correct"


def test_routing_reads_detections_z_and_never_rebuilds_it() -> None:
    x, span = _drift()
    ev = _ev("d", x, span)
    flat = {k: np.zeros_like(v) for k, v in (ev.z or {}).items()}
    assert rt.in_band_verdict(rt.EventEvidence("d", "S", x, FS, span, 0.0, z=flat, z_t0_s=0.0),
                              "slow_wave", TOL).separable  # the given z decides
    with pytest.raises(ValueError, match="detection's z"):
        rt.in_band_verdict(rt.EventEvidence("d", "S", x, FS, span, 0.0), "slow_wave", TOL)
    with pytest.raises(ValueError, match="shared ENG trace"):
        rt.in_band_verdict(rt.EventEvidence("d", "S", x, FS, span, 0.0), "spikes", TOL)


def test_a_thin_background_is_conservative() -> None:
    """No crossings outside the span: one crossing inside gives p = 0 (reject)."""
    quiet = np.random.default_rng(9).normal(0.0, 1.0, int(DUR_S * FS))
    tr = rt.eng_trace("S", quiet, FS, x_t0_s=0.0)
    span = (30.0, 30.5)
    ev = rt.EventEvidence("q", "S", quiet, FS, span, 0.0, eng=rt.EngTrace(
        "S", tr.y, tr.rate, tr.sigma, np.array([int(30.2 * tr.rate)]), 0.0))
    v = rt.in_band_verdict(ev, "spikes", TOL)
    assert not v.separable and "(p 0)" in v.detail


def test_a_subtraction_on_a_late_epoch_uses_xs_own_timeline() -> None:
    """Subtract on x starting at 120 s: the residual is taken where the event is."""
    x, artifact, span = _tribo()
    exact = _ev("t", x, span, t0=120.0, subtract=lambda s: s - artifact)
    d = rt.route_event(exact, "spikes", TOL)
    assert d.route == "subtract" and d.residual_ratio < 1.0


def test_one_signals_eng_trace_is_never_used_for_another() -> None:
    x, span = _drift()
    tr = rt.eng_trace("L_T", x, FS, x_t0_s=0.0)
    with pytest.raises(ValueError, match="invariant 3"):
        rt.in_band_verdict(rt.EventEvidence("e", "R_T", x, FS, span, 0.0, eng=tr), "spikes", TOL)
    with pytest.raises(ValueError, match="invariant 3"):
        rt.in_band_verdict(rt.EventEvidence("e", "L_T", x, FS, span, 120.0, eng=tr),
                           "spikes", TOL)
