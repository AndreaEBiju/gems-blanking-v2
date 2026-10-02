"""Tests for the HR count gate and two-candidate selection (ruling 2026-09-30 (c) 4)."""

from __future__ import annotations

import numpy as np
import pytest
from gems_blanking_v2.derive.cm_events import find_events
from gems_blanking_v2.physio import hr_channel as hc
from gems_blanking_v2.physio.hr_channel import (
    AC_WINDOW_S,
    PROVISIONAL_MAX_COUNT_DEV,
    CountGate,
    TrainGate,
    autocorr_rate,
    count_gate,
    findpeaks_replica,
    gated_selection,
    implausible_fraction,
    pick_train,
)

from tests.conftest import make_ecg, make_two_source_ground

FS = 24414.0625


def _minutes(counts: list[int], rate: float = 400.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Beats evenly spread, ``counts[k]`` in minute ``k``; the rate is ``rate`` in every minute."""
    beats = np.concatenate([k * AC_WINDOW_S + (np.arange(n) + 0.5) * AC_WINDOW_S / n
                            for k, n in enumerate(counts)])
    starts = np.arange(len(counts)) * AC_WINDOW_S
    return beats, starts, np.full(len(counts), rate)


# --- the autocorrelation rate --------------------------------------------------


@pytest.mark.parametrize("rr_s", [0.109, 0.130, 0.150, 0.200])  # 550 bpm first
def test_the_autocorrelation_recovers_the_heart_rate(rr_s: float) -> None:
    ecg = make_ecg(FS, 125.0, amp_uv=60.0, noise_uv=5.0, rr_s=rr_s, seed=3)
    starts, bpm = autocorr_rate(ecg.signal, FS)
    assert starts.tolist() == [0.0, 60.0]  # the trailing 5 s is not a minute
    b = ecg.beats_s
    true = [60.0 / np.mean(np.diff(b[(b >= s) & (b < s + 60.0)])) for s in starts]
    assert np.all(np.abs(bpm / true - 1.0) < 0.005)  # one lag step at 1 kHz


def test_noise_alone_gives_no_clear_minute() -> None:
    rng = np.random.default_rng(4)
    _starts, bpm = autocorr_rate(rng.standard_normal(int(125 * FS)) * 5.0, FS)
    assert np.all(np.isnan(bpm))


# --- the count gate ------------------------------------------------------------


def test_five_percent_is_inside_the_gate_and_just_over_is_not() -> None:
    assert PROVISIONAL_MAX_COUNT_DEV == 0.05
    ok = count_gate(*_minutes([420, 380]))
    assert ok.passes and ok.fraction_within == 1.0
    over = count_gate(*_minutes([421, 400]))
    assert not over.passes and over.fraction_within == 0.5


def test_ninety_five_percent_of_minutes_must_agree() -> None:
    assert count_gate(*_minutes([400] * 19 + [200])).passes
    assert not count_gate(*_minutes([400] * 18 + [200] * 2)).passes


def test_a_halved_or_doubled_train_fails() -> None:
    for n in (200, 800):
        g = count_gate(*_minutes([n] * 4))
        assert not g.passes
        assert g.median_abs_dev == pytest.approx(abs(n / 400 - 1))


def test_only_clear_minutes_are_judged_and_mostly_unclear_is_unassessable() -> None:
    beats, starts, bpm = _minutes([400, 400, 999, 400])
    bpm[2] = np.nan
    g = count_gate(beats, starts, bpm)
    assert g.assessable and g.passes and g.clear_minutes == 3
    bpm[:2] = np.nan
    g = count_gate(beats, starts, bpm)
    assert g == CountGate(minutes=4, clear_minutes=1, fraction_within=1.0,
                          median_abs_dev=0.0, assessable=False, passes=False)


# --- the candidates ------------------------------------------------------------


def test_the_findpeaks_replica_finds_every_beat_off_the_edges() -> None:
    ecg = make_ecg(FS, 20.0, amp_uv=60.0, noise_uv=2.0, seed=8)
    got = findpeaks_replica(ecg.signal, FS)
    truth = ecg.beats_s[(ecg.beats_s >= 0.75) & (ecg.beats_s < 20.0 - 0.75)]
    assert got.size == truth.size
    assert np.max(np.abs(got - truth)) < 0.002
    assert got.min() >= 0.75 and got.max() < 20.0 - 0.75


def test_implausible_fraction_counts_intervals_under_three_quarters_of_the_median() -> None:
    beats = np.cumsum([0.15] * 9 + [0.10] + [0.15] * 10)  # one 0.10 s in 19 intervals
    assert implausible_fraction(beats) == pytest.approx(1 / 19)
    assert implausible_fraction(np.array([0.0, 1.0, 2.0])) == 1.0  # nothing plausible


def _row(snr: float, dev: float, passes: bool = True, name: str = "x") -> TrainGate:
    cg = CountGate(minutes=1, clear_minutes=1, fraction_within=1.0, median_abs_dev=dev,
                   assessable=True, passes=True)
    return TrainGate(channel=name, detector="task05", beats_s=np.zeros(0), snr=snr, count=cg,
                     transient_harm=0.0, implausible_frac=0.0, rescue_rate=0.0,
                     plausible=True, passes=passes)


def test_the_best_snr_among_passing_trains_is_stored() -> None:
    rows = [_row(50.0, 0.0, passes=False, name="a"), _row(20.0, 0.01, name="b"),
            _row(30.0, 0.02, name="c")]
    assert pick_train(rows).channel == "c"  # "a" is better but fails


def test_an_snr_tie_goes_to_the_count_closest_to_the_autocorrelation() -> None:
    assert pick_train([_row(30.0, 0.03, name="a"), _row(30.0, 0.01, name="b")]).channel == "b"


def test_nothing_passing_stores_nothing() -> None:
    assert pick_train([_row(30.0, 0.0, passes=False)]) is None
    assert pick_train([]) is None


# --- the whole selection on a synthetic rig -------------------------------------


@pytest.fixture(scope="module")
def rig():  # noqa: ANN201 - a NamedTuple from conftest
    w = make_two_source_ground(FS, 65.0, cardiac_gain_spread=0.4, gain_spread=0.005, seed=21)
    return w, find_events(w.rec)


def test_a_stored_train_passed_every_gate_and_counts_the_true_beats(rig) -> None:  # noqa: ANN001
    w, ev = rig
    best, rows = gated_selection(w.rec, ev)
    assert best is not None and best.passes
    assert {r.detector for r in rows} == {"task05", "findpeaks"}
    channel = [r for r in rows if r.source == "channel"]
    assert len(channel) == 2 * len(hc.hr_candidates(w.rec).channels)
    assert {r.source for r in rows} == {"channel", "pair"}  # the adopted lead is a source too
    assert best.source == "channel"  # a passing channel train is preferred to a pair
    assert best.snr == max(r.snr for r in channel if r.passes)
    truth = np.asarray(w.beats_s)
    assert abs(best.beats_s.size / truth.size - 1.0) < PROVISIONAL_MAX_COUNT_DEV


def test_a_train_the_autocorrelation_disagrees_with_is_never_stored(rig, monkeypatch) -> None:  # noqa: ANN001
    w, ev = rig
    real = hc.autocorr_windows

    def doubled(x, fs):  # noqa: ANN001, ANN202
        starts, durs, bpm = real(x, fs)
        return starts, durs, bpm * 2

    monkeypatch.setattr(hc, "autocorr_windows", doubled)
    best, rows = gated_selection(w.rec, ev)
    assert best is None
    assert not any(r.count.passes for r in rows)


def test_the_gate_is_outside_the_generation_hash() -> None:
    from gems_blanking_v2.detect import chain  # noqa: PLC0415

    assert "gems_blanking_v2.physio.hr_channel" not in chain.generation_modules()


# --- the peri-R train (addendum to ruling (c) 1: mask-grade beats) --------------


def _gate(snr: float, count_ok: bool, harm: float, plausible: bool = True,
          name: str = "x") -> TrainGate:
    cg = CountGate(minutes=1, clear_minutes=1, fraction_within=1.0 if count_ok else 0.5,
                   median_abs_dev=0.01, assessable=True, passes=count_ok)
    return TrainGate(channel=name, detector="findpeaks", beats_s=np.zeros(0), snr=snr, count=cg,
                     transient_harm=harm, implausible_frac=0.0, rescue_rate=0.0,
                     plausible=plausible,
                     passes=count_ok and plausible and harm <= hc.PROVISIONAL_MAX_TRANSIENT_HARM)


def test_the_vetted_train_places_the_mask_when_there_is_one() -> None:
    rows = [_gate(90.0, True, 0.3, name="raw"), _gate(40.0, True, 0.0, name="T")]
    train, grade = hc.peri_r_train(rows)
    assert (train.channel, grade) == ("T", "hrv")


def test_without_a_vetted_train_a_count_passing_one_is_mask_grade() -> None:
    rows = [_gate(90.0, False, 0.0, name="miscount"), _gate(60.0, True, 0.4, name="raw"),
            _gate(80.0, True, 0.4, plausible=False, name="implausible")]
    train, grade = hc.peri_r_train(rows)
    assert (train.channel, grade) == ("raw", "mask")


def test_no_count_passing_train_means_no_peri_r_route() -> None:
    assert hc.peri_r_train([_gate(90.0, False, 0.0)]) == (None, "none")
