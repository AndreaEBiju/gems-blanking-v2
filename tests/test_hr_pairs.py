"""The pairs-only cross-site HR lead (ruling 2026-10-01 (d)), on the synthetic shared-ground rig."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.derive.cm_events import Events
from gems_blanking_v2.detect import chain
from gems_blanking_v2.physio import hr_channel as hc
from gems_blanking_v2.physio import hr_pairs as hp
from gems_blanking_v2.physio.rpeaks import detect_rpeaks
from gems_blanking_v2.types import ChannelInfo, Recording

from tests.conftest import CrossSiteSynth, make_cross_site_heart

F64 = npt.NDArray[np.float64]
FS = 24414.0625
TH = hc.PROVISIONAL_MAX_TRANSIENT_HARM
NAMES3 = ("C0", "C1", "C2")
STRONG_STOMACH = {"R": 1.0, "L": -0.6, "stomach": 1.0}
"""The stomach sees the heart clearly, so the rate reference is assessable."""


@pytest.fixture(scope="module")
def rig() -> tuple[CrossSiteSynth, hp.HalfSplit]:
    w = make_cross_site_heart(FS, 130.0, seed=3)
    return w, hp.half_split(w.rec)


# --- candidates -------------------------------------------------------------------------


def test_the_candidates_are_nine_left_right_differences_in_both_orientations(rig) -> None:  # noqa: ANN001
    w, _r = rig
    names = [p.name for p in hp.pair_candidates(w.rec)]
    left, right = [f"LVN{k}" for k in (1, 2, 3)], [f"RVN{k}" for k in (1, 2, 3)]
    want = {f"{a}-{b}" for a in left for b in right} | {f"{b}-{a}" for a in left for b in right}
    assert len(names) == 18 and set(names) == want
    assert not any("ANT" in n for n in names)  # no stomach pair is a candidate


def test_a_missing_contact_takes_its_pairs_with_it(rig) -> None:  # noqa: ANN001
    w, _r = rig
    rec = replace(w.rec, channels=[c for c in w.rec.channels if c.name != "LVN3"])
    names = [p.name for p in hp.pair_candidates(rec)]
    assert len(names) == 12 and not any("LVN3" in n for n in names)


def test_a_lead_is_the_difference_and_the_recording_is_not_written(rig) -> None:  # noqa: ANN001
    w, _r = rig
    data = np.array(w.rec.data[:5000])
    data.flags.writeable = False
    rec = replace(w.rec, data=data)
    p = next(p for p in hp.pair_candidates(rec) if p.name == "LVN2-RVN1")
    x = hp.lead_signal(rec, p)
    assert np.array_equal(x, data[:, p.plus_index] - data[:, p.minus_index])
    x[:] = 0.0  # the lead is the caller's to modify
    assert np.any(data[:, p.plus_index] != 0)


# --- the injections -----------------------------------------------------------------------


def _events(rows: list[list[float]], cm: list[float], names: tuple[str, ...]) -> Events:
    k = len(rows)
    return Events(t_s=np.linspace(1.0, 2.0, k), cm_amp_uv=np.asarray(cm, float),
                  channel_amp_uv=np.asarray(rows, float), width_s=np.full(k, 0.001), channels=names)


def _flat(n_ch: int, dur_s: float = 30.0) -> Recording:
    chans = [ChannelInfo(i, f"C{i}", "nerve", "L", i + 1, None, "independent") for i in range(n_ch)]
    return Recording(fs=FS, data=np.zeros((int(dur_s * FS), n_ch)), channels=chans, animal="SYNTH",
                     session="s", path=None)


def test_patterns_are_signed_relative_gains_and_a_zero_median_event_has_none() -> None:
    ev = _events([[-2.0, -1.0, 1.0], [4.0, 2.0, 2.0], [0.0, 0.0, 5.0]], [1, 1, 1], NAMES3)
    rel = hp.event_patterns(ev)
    assert np.allclose(rel[0], [2.0, 1.0, -1.0])  # polarity is the median's: negative event
    assert np.allclose(rel[1], [2.0, 1.0, 1.0])
    assert np.isnan(rel[2]).all()  # median |amplitude| 0: no pattern


def test_each_real_injection_carries_one_kept_events_own_pattern_and_amplitude() -> None:
    rec = _flat(3)
    rows = [[1.0, 0.5, -0.25], [1.0, 1.0, 1.0], [0.0, 0.0, 9.0]]
    ev = _events(rows, [100.0, 200.0, 300.0], NAMES3)
    keep = np.array([True, False, True])  # event 1 is cardiac; event 2 has no pattern
    inj, times = hp.inject_real_patterns(rec, ev, keep, seed=4)
    assert times.size == hc.N_INJECTIONS
    i = np.round(times * FS).astype(int)
    peaks = inj.data[i]  # Gaussian peak at each time; overlaps are rare and checked below
    alone = np.r_[np.inf, np.diff(times)] > 0.01
    alone &= np.r_[np.diff(times), np.inf] > 0.01
    # relative to the event's median |amplitude| (0.5): [1, 0.5, -0.25] -> [2, 1, -0.5], at 100 uV
    assert np.allclose(peaks[alone], np.array([2.0, 1.0, -0.5]) * 100.0, rtol=1e-3)
    assert np.array_equal(rec.data, np.zeros_like(rec.data))  # input not written


def test_the_real_injection_draws_whole_patterns_from_several_events_never_their_median() -> None:
    rec = _flat(3)
    ev = _events([[1.0, 0.5, -0.25], [1.0, 2.0, 1.0]], [100.0, 100.0], ("C0", "C1", "C2"))
    inj, times = hp.inject_real_patterns(rec, ev, np.ones(2, bool), seed=5)
    i = np.round(times * FS).astype(int)
    alone = (np.r_[np.inf, np.diff(times)] > 0.01) & (np.r_[np.diff(times), np.inf] > 0.01)
    got = inj.data[i][alone] / 100.0
    rel = hp.event_patterns(ev)
    which = [int(np.argmin(np.abs(g - rel).max(axis=1))) for g in got]
    assert all(np.allclose(g, rel[k], rtol=1e-3) for g, k in zip(got, which, strict=True))
    assert set(which) == {0, 1}  # both events' patterns occur
    med = np.median(rel, axis=0)
    assert not any(np.allclose(g, med, rtol=1e-2) for g in got)


def test_the_median_injection_has_the_same_times_and_the_median_pattern() -> None:
    rec = _flat(2)
    ev = _events([[1.0, 0.5], [1.0, 1.5], [1.0, 3.0]], [100.0, 100.0, 100.0], ("C0", "C1"))
    keep = np.ones(3, bool)
    _r, t_real = hp.inject_real_patterns(rec, ev, keep, seed=1)
    med, t_med = hp.inject_median_pattern(rec, ev, keep, seed=1)
    assert np.array_equal(t_real, t_med)
    i = np.round(t_med * FS).astype(int)
    alone = (np.r_[np.inf, np.diff(t_med)] > 0.01) & (np.r_[np.diff(t_med), np.inf] > 0.01)
    rel = hp.event_patterns(ev)
    assert np.allclose(med.data[i][alone] / 100.0, np.median(rel, axis=0), rtol=1e-3)


def test_no_non_cardiac_event_with_a_pattern_refuses_by_name() -> None:
    ev = _events([[0.0, 0.0, 1.0]], [1.0], ("C0", "C1", "C2"))
    with pytest.raises(ValueError, match="no non-cardiac event with a gain pattern"):
        hp.inject_real_patterns(_flat(3, 5.0), ev, np.ones(1, bool))


# --- the protocol on the rig ------------------------------------------------------------


def test_single_contacts_are_captured_and_a_pair_is_not(rig) -> None:  # noqa: ANN001
    _w, r = rig
    raw = [g for g in r.first_singles if g.lead[:3] in ("LVN", "RVN", "ANT")]
    assert raw and all(g.harm["real"] > TH for g in raw)
    assert r.pairs_passing_first >= 1
    assert all(g.harm["real"] <= TH for g in r.first_pairs if g.passes)


def test_the_chosen_pair_is_the_first_halfs_and_is_judged_on_the_second(rig) -> None:  # noqa: ANN001
    _w, r = rig
    assert r.chosen_pair is not None and r.chosen_pair.passes
    assert r.chosen_pair.snr == max(g.snr for g in r.first_pairs if g.passes)
    assert r.second_pair is not None
    chosen = (r.chosen_pair.lead, r.chosen_pair.detector)
    assert (r.second_pair.lead, r.second_pair.detector) == chosen
    assert r.second_pair.passes and r.gained


def test_the_second_half_never_changes_the_choice_only_the_verdict(rig) -> None:  # noqa: ANN001
    w, r = rig
    n = w.rec.data.shape[0] // 2
    data = np.array(w.rec.data)
    silent = {"R": 0.0, "L": 0.0, "stomach": 0.0}
    no_heart = make_cross_site_heart(FS, 130.0, site_gain=silent, seed=8)
    data[n:] = no_heart.rec.data[n:2 * n]  # the second half keeps the ground, not the heart
    bad = hp.half_split(replace(w.rec, data=data))
    assert bad.chosen_pair is not None and r.chosen_pair is not None
    assert (bad.chosen_pair.lead, bad.chosen_pair.detector, bad.chosen_pair.snr) == \
        (r.chosen_pair.lead, r.chosen_pair.detector, r.chosen_pair.snr)
    assert bad.second_pair is not None and not bad.second_pair.passes and not bad.gained


def _gate(snr: float, *, count: bool = True, plaus: bool = True, real: float = 0.0,
          median: float = 0.0, name: str = "x") -> hp.LeadGate:
    cg = hc.CountGate(minutes=1, clear_minutes=1, fraction_within=1.0 if count else 0.0,
                      median_abs_dev=0.0, assessable=True, passes=count)
    harm = {"real": real, "median": median}
    return hp.LeadGate(name, "task05", np.zeros(0), snr, cg, 0.0, 0.0, plaus, harm)


def test_selection_is_the_best_passing_snr_and_the_veto_binding_is_the_real_pattern() -> None:
    rows = [_gate(50.0, name="a"), _gate(90.0, name="b"), _gate(99.0, real=0.02, name="c")]
    assert hp.select(rows).lead == "b"  # c has the best SNR but fails the real-pattern veto
    only_median_fails = _gate(10.0, median=0.5, name="m")
    assert only_median_fails.passes and not only_median_fails.passes_median


def test_with_none_passing_the_least_harm_is_judged_and_with_no_count_nothing() -> None:
    rows = [_gate(90.0, real=0.05, name="a"), _gate(10.0, real=0.02, name="b"),
            _gate(99.0, count=False, name="c")]
    assert hp.select(rows).lead == "b"
    assert hp.select([_gate(99.0, count=False), _gate(80.0, plaus=False)]) is None


def test_the_veto_threshold_is_inclusive_at_one_percent() -> None:
    assert _gate(1.0, real=TH).passes and not _gate(1.0, real=TH + 0.005).passes


# --- (a) rate --------------------------------------------------------------------------


def _fixed_rate(monkeypatch: pytest.MonkeyPatch, ref_bpm: list[float]) -> None:
    starts = np.arange(len(ref_bpm), dtype=float) * hc.AC_WINDOW_S
    monkeypatch.setattr(hc, "autocorr_rate", lambda _x, _fs: (starts, np.asarray(ref_bpm, float)))


def _train(bpm_per_minute: list[float]) -> F64:
    out = []
    for m, bpm in enumerate(bpm_per_minute):
        k = int(round(bpm))
        out.append(m * 60.0 + (np.arange(k) + 0.5) * 60.0 / k)
    return np.concatenate(out)


@pytest.fixture(scope="module")
def short_rig() -> CrossSiteSynth:
    return make_cross_site_heart(FS, 65.0, transients_per_s=0.5, site_gain=STRONG_STOMACH, seed=5)


def test_the_references_are_the_other_sites_raw_channels_and_stomach_ref(short_rig) -> None:  # noqa: ANN001
    p = next(p for p in hp.pair_candidates(short_rig.rec) if p.name == "LVN1-RVN1")
    rc = hp.rate_check(short_rig.beats_s, short_rig.rec, p)
    assert rc.references == ("ANT1", "ANT2", "ANT3", "stomach_ref")  # both necks are in the lead


def test_the_true_beats_agree_with_a_clear_stomach_reference(short_rig) -> None:  # noqa: ANN001
    p = next(p for p in hp.pair_candidates(short_rig.rec) if p.name == "LVN1-RVN1")
    rc = hp.rate_check(short_rig.beats_s, short_rig.rec, p)
    assert rc.minutes_with_reference >= 1 and rc.passes


def test_five_percent_off_the_reference_agrees_and_just_over_does_not(
    monkeypatch: pytest.MonkeyPatch, short_rig: CrossSiteSynth,
) -> None:
    p = hp.pair_candidates(short_rig.rec)[0]
    _fixed_rate(monkeypatch, [400.0])
    assert hp.rate_check(_train([420.0]), short_rig.rec, p).passes
    assert not hp.rate_check(_train([421.0]), short_rig.rec, p).passes


def test_ninety_five_percent_of_judged_minutes_must_agree(monkeypatch, short_rig) -> None:  # noqa: ANN001
    p = hp.pair_candidates(short_rig.rec)[0]
    _fixed_rate(monkeypatch, [400.0] * 20)
    one_off = _train([400.0] * 19 + [300.0])  # 19 / 20 = 0.95
    two_off = _train([400.0] * 18 + [300.0, 300.0])
    assert hp.rate_check(one_off, short_rig.rec, p).passes
    assert not hp.rate_check(two_off, short_rig.rec, p).passes


def test_only_clear_reference_minutes_are_judged_and_none_never_passes(
    monkeypatch: pytest.MonkeyPatch, short_rig: CrossSiteSynth,
) -> None:
    p = hp.pair_candidates(short_rig.rec)[0]
    _fixed_rate(monkeypatch, [400.0, np.nan])
    rc = hp.rate_check(_train([400.0, 100.0]), short_rig.rec, p)
    assert rc.minutes_with_reference == 1 and rc.passes  # the unclear minute is not judged
    _fixed_rate(monkeypatch, [np.nan, np.nan])
    rc = hp.rate_check(_train([400.0, 400.0]), short_rig.rec, p)
    assert rc.minutes_with_reference == 0 and not rc.passes


def test_a_lead_locked_on_another_periodic_source_fools_its_own_gate_not_the_cross_check() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=0.5, periodic_uv=300.0,
                              periodic_rr_s=0.125, site_gain=STRONG_STOMACH, seed=6)
    p = next(p for p in hp.pair_candidates(w.rec) if p.name == "LVN1-RVN1")
    x = hp.lead_signal(w.rec, p)
    beats = np.asarray(detect_rpeaks(x, FS).t_s, float)
    own = hc.count_gate(beats, *hc.autocorr_rate(x, FS))
    assert own.passes  # the lead's own gate is fooled: it agrees with itself
    rc = hp.rate_check(beats, w.rec, p)
    assert rc.minutes_with_reference >= 1 and not rc.passes


# --- (b) timing ------------------------------------------------------------------------


def test_a_constant_offset_is_removed_and_reported() -> None:
    v = np.arange(1000) * 0.155 + 1.0
    tc = hp.timing_check(v + 0.0009, v)
    assert tc.passes and tc.offset_s == pytest.approx(0.0009, abs=1e-9)


def test_ninety_nine_percent_within_two_ms_after_the_offset() -> None:
    v = np.arange(1000) * 0.155 + 1.0
    b = v + 0.0009
    b10, b11 = b.copy(), b.copy()
    b10[:10] += 0.005  # 990 / 1000 within tolerance
    b11[:11] += 0.005
    assert hp.timing_check(b10, v).passes and not hp.timing_check(b11, v).passes
    b2 = b.copy()
    b2[:5] += 0.0021 - 0.0  # just outside 2 ms after the offset
    assert hp.timing_check(b2, v).fraction_within == pytest.approx(0.995)


def test_no_vetted_beat_fails_the_timing_check() -> None:
    assert not hp.timing_check(np.arange(10.0), np.zeros(0)).passes


# --- (c) morphology --------------------------------------------------------------------


def test_the_true_beats_show_the_qrs_on_every_contact(short_rig) -> None:  # noqa: ANN001
    m = hp.morphology_check(short_rig.beats_s, short_rig.rec)
    assert m.passes and not m.failing
    assert all(m.snr[k] > m.shuffled_max[k] for k in m.snr)


def test_beats_that_are_not_the_heart_show_no_qrs(short_rig) -> None:  # noqa: ANN001
    rng = np.random.default_rng(2)
    fake = np.sort(rng.uniform(1.0, 64.0, short_rig.beats_s.size))
    assert not hp.morphology_check(fake, short_rig.rec).passes


def test_a_detached_contact_fails_and_is_named() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=0.5, open_contact="LVN3",
                              site_gain=STRONG_STOMACH, seed=7)
    m = hp.morphology_check(w.beats_s, w.rec)
    assert "LVN3" in m.failing and not m.passes
    assert set(m.failing) == {"LVN3"}


def test_a_half_without_events_cannot_pass_the_veto() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=0.5, seed=10)
    data = np.array(w.rec.data)
    rng = np.random.default_rng(11)
    data[:] = rng.normal(0.0, 3.0, data.shape)  # no event anywhere: nothing to inject
    h = hp._prepare_half(replace(w.rec, data=data), 0)
    assert h.injected == {} and h.times.size == 0
    rows, _s = hp.gate_half(h, pairs=hp.pair_candidates(w.rec)[:1], only=("LVN1-RVN1", "task05"))
    assert rows and all(np.isnan(r.harm["real"]) and not r.passes for r in rows)


def test_hr_pairs_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.physio.hr_pairs" not in chain.generation_modules()
