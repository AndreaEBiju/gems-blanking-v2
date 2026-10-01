"""The pairs-only cross-site HR lead (ruling 2026-10-01 (d)), on the synthetic shared-ground rig."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.derive.cm_events import Events, find_events
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


# --- detached contacts (ruling 2026-10-02 1) ----------------------------------------------


@pytest.fixture(scope="module")
def open_rig() -> CrossSiteSynth:
    return make_cross_site_heart(FS, 65.0, transients_per_s=2.0, open_contact="LVN3",
                                 site_gain=STRONG_STOMACH, seed=7)


def _keep_all(ev: Events) -> npt.NDArray[np.bool_]:
    return np.ones(len(ev.t_s), dtype=bool)


def test_a_contact_without_the_ground_is_detached_and_the_rest_are_not(open_rig) -> None:  # noqa: ANN001
    ev = find_events(open_rig.rec)
    gg = hp.ground_gains(ev, _keep_all(ev))
    assert abs(gg["LVN3"][0]) < hp.DETACHED_MAX_G
    assert hp.detached_contacts(ev, _keep_all(ev)) == frozenset({"LVN3"})


def _sign_column(n: int, n_negative: int) -> list[float]:
    return [-1.0 if k < n_negative else 1.0 for k in range(n)]


def test_detachment_is_g_below_two_tenths_or_agreement_below_three_quarters() -> None:
    n = 40
    names = ("U0", "U1", "U2", "C1", "C2", "C3")
    # three unit channels keep every event's median |amplitude| at 1, so the gains are as written

    def rows(col3: list[float]) -> list[list[float]]:
        # C1 just above |g| 0.2, C2 just below
        return [[1.0, 1.0, 1.0, 0.21, 0.19, col3[k]] for k in range(n)]
    ev = _events(rows(_sign_column(n, 10)), [1.0] * n, names)  # C3: 30 / 40 = 0.75 agree
    assert hp.detached_contacts(ev, np.ones(n, bool)) == frozenset({"C2"})
    ev = _events(rows(_sign_column(n, 11)), [1.0] * n, names)  # C3: 29 / 40, under three quarters
    assert hp.detached_contacts(ev, np.ones(n, bool)) == frozenset({"C2", "C3"})


def test_a_pair_with_a_detached_contact_is_not_a_candidate(open_rig) -> None:  # noqa: ANN001
    names = [p.name for p in hp.pair_candidates(open_rig.rec, frozenset({"LVN3"}))]
    assert len(names) == 12 and not any("LVN3" in n for n in names)


# --- (a) rate --------------------------------------------------------------------------


def _fixed_rates(monkeypatch: pytest.MonkeyPatch, per_ref: list[list[float]]) -> None:
    """Each reference, in call order, returns its own per-minute rates."""
    calls = iter(per_ref * 10)
    starts = np.arange(len(per_ref[0]), dtype=float) * hc.AC_WINDOW_S

    def fake(_x: object, _fs: float) -> tuple[F64, F64]:
        return starts, np.asarray(next(calls), float)
    monkeypatch.setattr(hp, "refined_autocorr_rate", fake)


def _train(bpm_per_minute: list[float]) -> F64:
    out = []
    for m, bpm in enumerate(bpm_per_minute):
        k = int(round(bpm))
        out.append(m * 60.0 + (np.arange(k) + 0.5) * 60.0 / k)
    return np.concatenate(out)


NAN3 = [np.nan] * 3


@pytest.fixture(scope="module")
def short_rig() -> CrossSiteSynth:
    return make_cross_site_heart(FS, 65.0, transients_per_s=0.5, site_gain=STRONG_STOMACH, seed=5)


def test_refinement_keeps_the_clear_minutes_and_lands_nearer_the_true_rate() -> None:
    t = np.arange(int(185 * FS)) / FS
    rr = 152.5 / (FS / 24)  # a lag of 152.5 samples at the autocorrelation's ~1017 Hz
    x = np.zeros_like(t)
    for b in np.arange(0.1, 184.9, rr):
        near = np.abs(t - b) < 0.01
        x[near] += 100.0 * np.exp(-0.5 * ((t[near] - b) / 0.002) ** 2)
    x += np.random.default_rng(1).normal(0.0, 2.0, x.size)
    s0, coarse = hc.autocorr_rate(x, FS)
    s1, fine = hp.refined_autocorr_rate(x, FS)
    assert np.array_equal(s0, s1) and np.array_equal(np.isnan(coarse), np.isnan(fine))
    true = 60.0 / rr
    clear = np.isfinite(fine)
    assert clear.any()
    err_c, err_f = np.abs(coarse - true)[clear] / true, np.abs(fine - true)[clear] / true
    assert np.all(err_c > 0.002)  # half a lag step off: ~0.3%
    assert np.all(err_f < 0.5 * err_c) and np.all(err_f < 0.001)


def test_the_references_are_the_other_sites_raw_channels_and_stomach_ref(short_rig) -> None:  # noqa: ANN001
    p = next(p for p in hp.pair_candidates(short_rig.rec) if p.name == "LVN1-RVN1")
    rc = hp.rate_check(short_rig.beats_s, short_rig.rec, p)
    assert rc.references == ("ANT1", "ANT2", "ANT3", "stomach_ref")  # both necks are in the lead


def test_the_true_beats_agree_with_a_clear_stomach_reference(short_rig) -> None:  # noqa: ANN001
    p = next(p for p in hp.pair_candidates(short_rig.rec) if p.name == "LVN1-RVN1")
    rc = hp.rate_check(short_rig.beats_s, short_rig.rec, p)
    assert rc.assessable and rc.passes


def test_five_percent_off_the_reference_agrees_and_just_over_does_not(
    monkeypatch: pytest.MonkeyPatch, short_rig: CrossSiteSynth,
) -> None:
    p = hp.pair_candidates(short_rig.rec)[0]
    _fixed_rates(monkeypatch, [[400.0]] * 4)
    assert hp.rate_check(_train([420.0]), short_rig.rec, p).passes
    assert not hp.rate_check(_train([421.0]), short_rig.rec, p).passes


def test_ninety_five_percent_of_assessable_minutes_must_agree(
    monkeypatch: pytest.MonkeyPatch, short_rig: CrossSiteSynth,
) -> None:
    p = hp.pair_candidates(short_rig.rec)[0]
    _fixed_rates(monkeypatch, [[400.0] * 20] * 4)
    assert hp.rate_check(_train([400.0] * 19 + [300.0]), short_rig.rec, p).passes  # 19 / 20
    assert not hp.rate_check(_train([400.0] * 18 + [300.0, 300.0]), short_rig.rec, p).passes


def test_a_minute_whose_references_disagree_by_over_five_percent_is_not_judged(
    monkeypatch: pytest.MonkeyPatch, short_rig: CrossSiteSynth,
) -> None:
    p = hp.pair_candidates(short_rig.rec)[0]
    # minute 2: references 400 and 424 differ by 5.8% of their median (412) - unassessable
    _fixed_rates(monkeypatch, [[400.0, 400.0, 400.0], [400.0, 424.0, 400.0], NAN3, NAN3])
    rc = hp.rate_check(_train([400.0, 100.0, 400.0]), short_rig.rec, p)
    assert rc.minutes_with_reference == 3 and rc.minutes_assessable == 2 and rc.passes
    # 400 and 420 differ by 4.9% of their median (410): the minute is judged
    _fixed_rates(monkeypatch, [[400.0, 400.0], [420.0, 400.0], [np.nan] * 2, [np.nan] * 2])
    assert hp.rate_check(_train([400.0, 400.0]), short_rig.rec, p).minutes_assessable == 2


def test_too_few_assessable_minutes_is_unassessable_and_never_passes(
    monkeypatch: pytest.MonkeyPatch, short_rig: CrossSiteSynth,
) -> None:
    p = hp.pair_candidates(short_rig.rec)[0]
    # 1 of 3 minutes assessable (< half): unassessable although that minute agrees
    _fixed_rates(monkeypatch, [[400.0, 400.0, 400.0], [400.0, 450.0, 450.0], NAN3, NAN3])
    rc = hp.rate_check(_train([400.0, 400.0, 400.0]), short_rig.rec, p)
    assert rc.minutes_assessable == 1 and not rc.assessable and not rc.passes
    # 2 of 4 (exactly half): assessable
    nan4 = [np.nan] * 4
    _fixed_rates(monkeypatch, [[400.0] * 4, [400.0, 400.0, 450.0, 450.0], nan4, nan4])
    rc = hp.rate_check(_train([400.0] * 4), short_rig.rec, p)
    assert rc.minutes_assessable == 2 and rc.assessable and rc.passes
    _fixed_rates(monkeypatch, [[np.nan] * 2] * 4)
    rc = hp.rate_check(_train([400.0, 400.0]), short_rig.rec, p)
    assert rc.minutes_with_reference == 0 and not rc.assessable and not rc.passes


def test_a_lead_locked_on_another_periodic_source_fools_its_own_gate_not_the_cross_check() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=0.5, periodic_uv=300.0,
                              periodic_rr_s=0.125, site_gain=STRONG_STOMACH, seed=6)
    p = next(p for p in hp.pair_candidates(w.rec) if p.name == "LVN1-RVN1")
    x = hp.lead_signal(w.rec, p)
    beats = np.asarray(detect_rpeaks(x, FS).t_s, float)
    own = hc.count_gate(beats, *hc.autocorr_rate(x, FS))
    assert own.passes  # the lead's own gate is fooled: it agrees with itself
    rc = hp.rate_check(beats, w.rec, p)
    assert rc.assessable and not rc.passes


# --- (b) timing ------------------------------------------------------------------------


V = np.arange(1000) * 0.155 + 1.0


def test_a_constant_offset_is_removed_and_reported() -> None:
    tc = hp.timing_check(V + 0.0009, V)
    assert tc.passes and tc.offset_s == pytest.approx(0.0009, abs=1e-9)
    assert tc.matched_lead == 1.0 and tc.matched_vetted == 1.0 and not tc.disagree


def test_identity_within_five_ms_after_the_offset() -> None:
    b = V + 0.0009
    b[:20] += 0.0049  # 4.9 ms off: still the same beat
    assert hp.timing_check(b, V).matched_lead == 1.0
    b[:20] += 0.0002  # 5.1 ms: not matched
    assert hp.timing_check(b, V).matched_lead == pytest.approx(0.98)


def test_ninety_nine_percent_matched_in_both_directions() -> None:
    extra11 = np.sort(np.r_[V, V[:11] + 0.07])  # 11 extra lead beats: 1000 / 1011 matched
    tc = hp.timing_check(extra11, V)
    assert tc.matched_vetted == 1.0 and tc.matched_lead < 0.99 and not tc.passes
    tc = hp.timing_check(V[11:], V)  # the lead lacks 11 vetted beats
    assert tc.matched_lead == 1.0 and tc.matched_vetted == pytest.approx(0.989) and not tc.passes
    assert hp.timing_check(V[10:], V).passes  # 990 / 1000 = 0.99


def test_the_spread_of_matched_differences_is_reported_not_gated() -> None:
    rng = np.random.default_rng(4)
    b = V + rng.normal(0.0, 0.0015, V.size).clip(-0.0045, 0.0045)
    tc = hp.timing_check(b, V)
    assert tc.passes and 0.001 < tc.sd_matched_s < 0.002


def test_more_than_five_percent_unmatched_at_twenty_ms_is_a_disagreement() -> None:
    b = V.copy()
    b[::19] += 0.06  # 53 of 1000 moved by 60 ms
    tc = hp.timing_check(b, V)
    assert tc.disagree and tc.unmatched_lead_20ms == pytest.approx(53 / 1000)
    b2 = V.copy()
    b2[::21] += 0.06  # 48 of 1000: under 5%
    assert not hp.timing_check(b2, V).disagree


def test_no_vetted_beat_fails_the_timing_check() -> None:
    assert not hp.timing_check(np.arange(10.0), np.zeros(0)).passes


def test_the_resolution_finds_the_train_whose_unmatched_beats_carry_no_qrs(short_rig) -> None:  # noqa: ANN001
    truth = short_rig.beats_s
    rng = np.random.default_rng(6)
    wrong = truth.copy()
    k = rng.choice(truth.size, truth.size // 5, replace=False)
    wrong[k] = truth[k] + rng.uniform(0.045, 0.11, k.size)  # 20% of beats not on a QRS
    r = hp.resolve_disagreement(truth, np.sort(wrong), short_rig.rec)
    assert r.other_unmatched >= 0.15 * truth.size and r.other_unmatched_qrs < 0.1
    assert r.lead_unmatched >= 0.15 * truth.size and r.lead_unmatched_qrs > 0.9
    flip = hp.resolve_disagreement(np.sort(wrong), truth, short_rig.rec)
    assert flip.lead_unmatched_qrs < 0.1 and flip.other_unmatched_qrs > 0.9


# --- (c) morphology --------------------------------------------------------------------


def test_the_true_beats_show_the_qrs_on_every_contact(short_rig) -> None:  # noqa: ANN001
    m = hp.morphology_check(short_rig.beats_s, short_rig.rec)
    assert m.assessable and m.passes and not m.failing
    assert all(m.snr[k] > m.shuffled_max[k] for k in m.snr)


def test_beats_that_are_not_the_heart_show_no_qrs(short_rig) -> None:  # noqa: ANN001
    rng = np.random.default_rng(2)
    fake = np.sort(rng.uniform(1.0, 64.0, short_rig.beats_s.size))
    assert not hp.morphology_check(fake, short_rig.rec).passes


def test_a_detached_contact_fails_unless_it_is_left_out(open_rig) -> None:  # noqa: ANN001
    m = hp.morphology_check(open_rig.beats_s, open_rig.rec)
    assert set(m.failing) == {"LVN3"} and not m.passes
    m2 = hp.morphology_check(open_rig.beats_s, open_rig.rec, detached=frozenset({"LVN3"}))
    assert m2.excluded == ("LVN3",) and "LVN3" not in m2.snr and m2.assessable and m2.passes


def test_fewer_than_four_contacts_or_one_site_is_unassessable(short_rig) -> None:  # noqa: ANN001
    names = {c.name for c in short_rig.rec.channels if c.role in ("nerve", "stomach")}

    def only(keep: set[str]) -> hp.MorphologyCheck:
        out = frozenset(names - keep)
        return hp.morphology_check(short_rig.beats_s, short_rig.rec, detached=out)
    three = only({"LVN1", "RVN1", "ANT1"})  # three contacts, three sites: too few contacts
    assert not three.failing and not three.assessable and not three.passes
    assert not only({"LVN1", "LVN2", "LVN3"}).assessable  # one site
    # four contacts on ONE site: a fourth left-neck contact, so only the site rule can fail it
    lvn4 = ChannelInfo(short_rig.rec.data.shape[1], "LVN4", "nerve", "L", 4, None, "independent")
    src = next(c for c in short_rig.rec.channels if c.name == "LVN2")
    data = np.column_stack([short_rig.rec.data, short_rig.rec.data[:, src.index]])
    four = replace(short_rig.rec, data=data, channels=[*short_rig.rec.channels, lvn4])
    rest = frozenset(c.name for c in four.channels
                     if c.cuff_id != "L" and c.role in ("nerve", "stomach"))
    one_site = hp.morphology_check(short_rig.beats_s, four, detached=rest)
    assert len(one_site.snr) == 4 and not one_site.failing and not one_site.assessable
    four_two = only({"LVN1", "LVN2", "RVN1", "RVN2"})  # the floor exactly: 4 contacts, 2 sites
    assert four_two.assessable and four_two.passes


def test_a_half_without_events_cannot_pass_the_veto() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=0.5, seed=10)
    data = np.array(w.rec.data)
    rng = np.random.default_rng(11)
    data[:] = rng.normal(0.0, 3.0, data.shape)  # no event anywhere: nothing to inject
    h = hp._prepare_half(replace(w.rec, data=data), 0)
    assert h.injected == {} and h.times.size == 0
    rows, _s = hp.gate_half(h, pairs=hp.pair_candidates(w.rec)[:1], only=("LVN1-RVN1", "task05"))
    assert rows and all(np.isnan(r.harm["real"]) and not r.passes for r in rows)


def test_the_protocol_never_chooses_a_detached_contact() -> None:
    w = make_cross_site_heart(FS, 130.0, open_contact="LVN2", seed=12)
    r = hp.half_split(w.rec)
    assert "LVN2" in r.detached
    assert not any("LVN2" in g.lead for g in r.first_pairs)
    assert r.chosen_pair is not None and "LVN2" not in r.chosen_pair.lead


def test_hr_pairs_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.physio.hr_pairs" not in chain.generation_modules()
