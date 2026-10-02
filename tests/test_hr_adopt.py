"""The pairs lead adopted into gated_selection (ruling 2026-10-02 (c) 1-3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.cm_events import find_events
from gems_blanking_v2.physio import hr_channel as hc
from gems_blanking_v2.physio import hr_pairs as hp

from tests.conftest import CrossSiteSynth, make_cross_site_heart

FS = 24414.0625
STRONG_STOMACH = {"R": 1.0, "L": -0.6, "stomach": 1.0}


@pytest.fixture(scope="module")
def rig() -> CrossSiteSynth:
    """No single signal can carry HR here: contacts are captured, T and stomach_ref cancel it."""
    return make_cross_site_heart(FS, 65.0, transients_per_s=2.0, site_gain=STRONG_STOMACH, seed=31)


@pytest.fixture(scope="module")
def selection(rig):  # noqa: ANN001, ANN201
    return hc.gated_selection(rig.rec, find_events(rig.rec))


def test_where_no_channel_train_passes_the_pair_lead_is_stored(rig, selection) -> None:  # noqa: ANN001
    best, rows = selection
    assert not any(r.passes for r in rows if r.source == "channel")
    assert best is not None and best.source == "pair" and best.cross_check == "pass"
    assert best.note == "pair lead: no channel train passes"
    assert abs(best.beats_s.size / rig.beats_s.size - 1.0) < hc.PROVISIONAL_MAX_COUNT_DEV


def test_every_pair_row_faces_the_same_gates_and_the_cross_check(selection) -> None:  # noqa: ANN001
    _best, rows = selection
    pairs = [r for r in rows if r.source == "pair"]
    assert len(pairs) == 2 * 18
    for r in pairs:
        gates = (
            r.count.passes and r.plausible and r.transient_harm <= hc.PROVISIONAL_MAX_TRANSIENT_HARM
        )
        assert r.passes == (gates and r.cross_check == "pass")
        # it runs behind count + plausibility, whatever the veto: a mask-grade pair needs it
        # too (ruling 2026-10-02 (f) 3)
        assert (r.cross_check == "") == (not (r.count.passes and r.plausible))
        assert (r.rate_ok is None) == (r.cross_check == "")


def test_without_the_pairs_source_nothing_is_stored_on_this_rig(rig) -> None:  # noqa: ANN001
    best, rows = hc.gated_selection(rig.rec, find_events(rig.rec), pairs=False)
    assert best is None and all(r.source == "channel" for r in rows)


def test_the_real_pattern_veto_is_binding_for_every_source(rig) -> None:  # noqa: ANN001
    p = next(p for p in hp.pair_candidates(rig.rec) if p.name == "RVN1-LVN1")
    x = hp.lead_signal(rig.rec, p)
    times = np.sort(np.random.default_rng(3).uniform(2.0, 60.0, hc.N_INJECTIONS))
    hit = x.copy()
    for t0 in times:  # a large sharp transient between beats: it is captured as a beat
        k = int(round((t0 + 0.07) * FS))
        hit[k - 30 : k + 30] += 5000.0 * np.exp(-0.5 * (np.arange(-30, 30) / 10.0) ** 2)
    ac = hc.autocorr_rate(x, FS)
    real_bad = hc._gate_row(p.name, "task05", x, hit, x, times, FS, ac, None, "pair")
    median_bad = hc._gate_row(p.name, "task05", x, x, hit, times, FS, ac, None, "pair")
    assert real_bad.count.passes and real_bad.plausible  # only the veto can decide
    assert real_bad.transient_harm > hc.PROVISIONAL_MAX_TRANSIENT_HARM and not real_bad.passes
    assert median_bad.harm_median > hc.PROVISIONAL_MAX_TRANSIENT_HARM
    assert median_bad.passes  # the median pattern never decides


def test_a_pair_that_passes_its_gates_but_not_the_cross_check_is_not_stored() -> None:
    w = make_cross_site_heart(
        FS,
        65.0,
        transients_per_s=2.0,
        periodic_uv=300.0,
        periodic_rr_s=0.125,
        site_gain=STRONG_STOMACH,
        seed=33,
    )
    best, rows = hc.gated_selection(w.rec, find_events(w.rec))
    fooled = [r for r in rows if r.source == "pair" and r.cross_check not in ("", "pass")]
    assert fooled and not any(r.passes for r in fooled)  # locked on the artifact: rate fails
    assert best is None or best.cross_check in ("", "pass")


def test_with_no_event_to_inject_no_row_passes(rig) -> None:  # noqa: ANN001
    ev = find_events(rig.rec)
    empty = replace(
        ev,
        t_s=ev.t_s[:0],
        cm_amp_uv=ev.cm_amp_uv[:0],
        channel_amp_uv=ev.channel_amp_uv[:0],
        width_s=ev.width_s[:0],
    )
    best, rows = hc.gated_selection(rig.rec, empty)
    assert best is None and not any(r.passes for r in rows)
    assert all(np.isnan(r.transient_harm) for r in rows)


# --- choice: incumbent preference and rule 2 ---------------------------------------------


def _row(
    name: str, beats: np.ndarray, snr: float, *, source: str = "channel", passes: bool = True
) -> hc.TrainGate:
    cg = hc.CountGate(
        minutes=1,
        clear_minutes=1,
        fraction_within=1.0,
        median_abs_dev=0.0,
        assessable=True,
        passes=True,
    )
    return hc.TrainGate(
        name, "task05", np.sort(beats), snr, cg, 0.0, 0.0, 0.0, True, passes, source=source
    )


def _wrong(truth: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = truth.copy()
    k = rng.choice(truth.size, truth.size // 5, replace=False)
    w[k] = truth[k] + rng.uniform(0.045, 0.11, k.size)  # a fifth of the beats off the QRS
    return np.sort(w)


def test_a_passing_incumbent_stays_even_when_another_channel_has_a_better_snr(rig) -> None:  # noqa: ANN001
    t = rig.beats_s
    rows = [_row("L_T", t, 900.0), _row("RVN1", t, 100.0)]
    assert hc._choose(rows, rig.rec, frozenset(), ("RVN1", "task05")).channel == "RVN1"
    assert hc._choose(rows, rig.rec, frozenset(), None).channel == "L_T"
    failing = [_row("L_T", t, 900.0), _row("RVN1", t, 100.0, passes=False)]
    assert hc._choose(failing, rig.rec, frozenset(), ("RVN1", "task05")).channel == "L_T"


def test_a_channel_train_beats_a_pair_when_they_agree(rig) -> None:  # noqa: ANN001
    t = rig.beats_s
    best = hc._choose(
        [_row("L_T", t, 100.0), _row("LVN1-RVN1", t, 900.0, source="pair")],
        rig.rec,
        frozenset(),
        None,
    )
    assert best.channel == "L_T" and best.note == ""


def test_rule_two_the_train_that_carries_the_qrs_wins(rig) -> None:  # noqa: ANN001
    t = rig.beats_s
    pair = _row("LVN1-RVN1", t, 900.0, source="pair")
    best = hc._choose(
        [_row("L_T", _wrong(t, 1), 100.0), pair], rig.rec, frozenset(), ("L_T", "task05")
    )
    assert best is not None and best.channel == "LVN1-RVN1" and "replaces" in best.note
    best = hc._choose(
        [_row("L_T", t, 100.0), _row("LVN1-RVN1", _wrong(t, 2), 900.0, source="pair")],
        rig.rec,
        frozenset(),
        None,
    )
    assert best is not None and best.channel == "L_T" and "stays" in best.note


def test_rule_two_unresolved_stores_nothing(rig) -> None:  # noqa: ANN001
    t = rig.beats_s
    rows = [_row("L_T", _wrong(t, 3), 100.0), _row("LVN1-RVN1", _wrong(t, 4), 900.0, source="pair")]
    assert hc._choose(rows, rig.rec, frozenset(), None) is None


def test_gap_after_tags_the_beat_before_each_unrecovered_gap() -> None:
    cg = hc.CountGate(
        minutes=1,
        clear_minutes=1,
        fraction_within=1.0,
        median_abs_dev=0.0,
        assessable=True,
        passes=True,
    )
    b = np.array([1.0, 1.15, 1.6, 1.75, 2.2])
    r = hc.TrainGate(
        "x", "task05", b, 1.0, cg, 0.0, 0.0, 0.0, True, True, gaps_s=((1.15, 1.6), (1.75, 2.2))
    )
    assert r.gap_after.tolist() == [False, True, False, True, False]
    assert not hc.TrainGate("x", "findpeaks", b, 1.0, cg, 0.0, 0.0, 0.0, True, True).gap_after.any()


# --- the post-H amendment to (a) -------------------------------------------------------------


def test_a_detached_contact_is_not_a_rate_reference(rig) -> None:  # noqa: ANN001
    p = next(p for p in hp.pair_candidates(rig.rec) if p.name == "LVN1-RVN1")
    with_all = hp.rate_check(rig.beats_s, rig.rec, p)
    without = hp.rate_check(rig.beats_s, rig.rec, p, frozenset({"ANT2"}))
    assert "ANT2" in with_all.references and "ANT2" not in without.references
    assert "stomach_ref" in without.references  # a derived signal, not a contact


def test_gated_selection_leaves_out_the_pairs_of_a_detached_contact() -> None:
    w = make_cross_site_heart(
        FS, 65.0, transients_per_s=2.0, open_contact="LVN3", site_gain=STRONG_STOMACH, seed=32
    )
    _best, rows = hc.gated_selection(w.rec, find_events(w.rec))
    pairs = [r for r in rows if r.source == "pair"]
    assert pairs and not any("LVN3" in r.channel for r in pairs)
    assert len(pairs) == 2 * 12


def test_a_stored_train_settled_its_veto_with_more_than_two_hundred_injections(selection) -> None:  # noqa: ANN001
    best, rows = selection
    assert best is not None and best.n_injections >= 400  # 0 hits in 200 is not settled
    for r in rows:
        if r.count.passes and r.plausible and np.isfinite(r.transient_harm):
            hits = round(r.transient_harm * r.n_injections)
            assert hc.harm_is_settled(hits, r.n_injections) or r.n_injections == hc.MAX_INJECTIONS
