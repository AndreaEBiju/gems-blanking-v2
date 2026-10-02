"""Per-minute HR storage and mask-grade pairs (ruling 2026-10-02 (f) 2-3)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.derive.cm_events import find_events
from gems_blanking_v2.physio import hr_channel as hc
from gems_blanking_v2.physio import hr_pairs as hp

from tests.conftest import make_cross_site_heart

F64 = npt.NDArray[np.float64]
FS = 24414.0625
W = hc.AC_WINDOW_S
STRONG_STOMACH = {"R": 1.0, "L": -0.6, "stomach": 1.0}


def _beats(rates: list[float], *, t0: float = 0.0) -> F64:
    """Evenly spaced beats, ``rates[m]`` per minute in minute m, offset half an interval."""
    out = []
    for m, r in enumerate(rates):
        n = int(round(r * W / 60.0))
        out.append(t0 + m * W + (np.arange(n) + 0.5) * W / n)
    return np.concatenate(out) if out else np.zeros(0)


# --- minute_validity -----------------------------------------------------------------


def test_a_minute_is_valid_only_when_clear_and_its_count_is_within_five_percent() -> None:
    starts = np.arange(4) * W
    bpm = np.array([400.0, 400.0, np.nan, 400.0])
    v = hc.minute_validity(_beats([400.0, 380.0, 400.0, 421.0]), starts, bpm)
    assert v.tolist() == [True, True, False, False]  # 380 is exactly 5% under; NaN unclear


def test_the_five_percent_boundary_is_inclusive_and_one_beat_over_fails() -> None:
    starts, bpm = np.zeros(1), np.array([400.0])
    assert hc.minute_validity(_beats([420.0]), starts, bpm).tolist() == [True]
    assert hc.minute_validity(_beats([421.0]), starts, bpm).tolist() == [False]
    assert hc.minute_validity(_beats([379.0]), starts, bpm).tolist() == [False]


def test_a_pair_minute_failing_the_rate_check_is_not_valid() -> None:
    starts, bpm = np.arange(3) * W, np.full(3, 400.0)
    b = _beats([400.0] * 3)
    assert hc.minute_validity(b, starts, bpm, bad_minutes_s=(W,)).tolist() == [True, False, True]
    assert hc.minute_validity(b, starts, bpm, bad_minutes_s=()).tolist() == [True] * 3


# --- minute_gapped -------------------------------------------------------------------


def _row(
    beats: F64,
    valid: list[bool],
    *,
    name: str = "x",
    source: str = "channel",
    snr: float = 10.0,
    eligible: bool = True,
    gaps: tuple[tuple[float, float], ...] = (),
    mad: float = 0.01,
) -> hc.TrainGate:
    cg = hc.CountGate(
        minutes=len(valid),
        clear_minutes=len(valid),
        fraction_within=1.0,
        median_abs_dev=mad,
        assessable=True,
        passes=True,
    )
    return hc.TrainGate(
        channel=name,
        detector="findpeaks",
        beats_s=beats,
        snr=snr,
        count=cg,
        transient_harm=0.0,
        implausible_frac=0.0,
        rescue_rate=0.0,
        plausible=True,
        passes=False,
        source=source,
        gaps_s=gaps,
        eligible=eligible,
        minute_s=tuple(float(m * W) for m in range(len(valid))),
        valid=tuple(valid),
    )


def test_beats_of_invalid_minutes_are_removed_and_each_removal_is_a_tagged_gap() -> None:
    b = _beats([400.0] * 5)
    out = hc.minute_gapped(_row(b, [True, True, False, True, False]))
    kept = b[(b < 2 * W) | ((b >= 3 * W) & (b < 4 * W))]
    np.testing.assert_array_equal(out.beats_s, kept)
    last_before = kept[kept < 2 * W][-1]
    first_after = kept[kept >= 3 * W][0]
    assert out.gaps_s == ((last_before, first_after),)  # none between valid minutes 0 and 1
    ga = out.gap_after
    assert ga.sum() == 1 and out.beats_s[ga][0] == last_before


def test_a_trailing_partial_minute_is_never_kept() -> None:
    b = np.concatenate([_beats([400.0]), [W + 1.0, W + 2.0]])  # past the last assessed minute
    out = hc.minute_gapped(_row(b, [True]))
    assert out.beats_s.size == 400 and out.beats_s.max() < W


def test_task05_gaps_survive_where_their_first_beat_is_kept() -> None:
    b = _beats([400.0, 400.0])
    kept_gap = (float(b[10]), float(b[11]))
    lost_gap = (float(b[-3]), float(b[-2]))  # in minute 1, which is invalid
    out = hc.minute_gapped(_row(b, [True, False], gaps=(kept_gap, lost_gap)))
    assert out.gaps_s == (kept_gap,)


def test_no_valid_minute_keeps_no_beat() -> None:
    out = hc.minute_gapped(_row(_beats([400.0] * 2), [False, False]))
    assert out.beats_s.size == 0 and out.gaps_s == ()


def test_valid_minute_counts_and_the_longest_run() -> None:
    r = _row(np.zeros(0), [True, True, False, True, True, True, False])
    assert (r.n_valid, r.longest_valid_run) == (5, 3)
    assert (_row(np.zeros(0), []).n_valid, _row(np.zeros(0), []).longest_valid_run) == (0, 0)


# --- the per-minute choice -----------------------------------------------------------


def _choose(rows: list[hc.TrainGate], incumbent: tuple[str, str] | None = None):  # noqa: ANN202
    return hc._choose_minutes(rows, None, frozenset(), incumbent)  # type: ignore[arg-type]


def test_most_valid_minutes_wins_over_template_snr() -> None:
    b = _beats([400.0] * 4)
    few = _row(b, [True, False, False, True], name="few", snr=50.0)
    many = _row(b, [True, True, True, False], name="many", snr=5.0)
    best = _choose([few, many])
    assert best is not None and best.channel == "many"
    assert best.note.startswith("per minute: most valid minutes (3 of 4)")


def test_equal_valid_minutes_go_to_template_snr() -> None:
    b = _beats([400.0] * 2)
    lo = _row(b, [True, False], name="lo", snr=5.0)
    hi = _row(b, [False, True], name="hi", snr=6.0)
    best = _choose([lo, hi])
    assert best is not None and best.channel == "hi"


def test_an_eligible_incumbent_is_kept_and_its_invalid_minutes_become_gaps() -> None:
    b = _beats([400.0] * 3)
    inc = _row(b, [True, False, True], name="inc", snr=1.0)
    better = _row(b, [True, True, True], name="better", snr=99.0)
    best = _choose([inc, better], incumbent=("inc", "findpeaks"))
    assert best is not None and best.channel == "inc"
    assert best.note == "per minute: the incumbent, still eligible" and len(best.gaps_s) == 1


def test_an_ineligible_row_is_never_stored_whatever_its_minutes() -> None:
    b = _beats([400.0] * 2)
    vetoed = _row(b, [True, True], name="vetoed", eligible=False)
    ok = _row(b, [True, False], name="ok")
    best = _choose([vetoed, ok], incumbent=("vetoed", "findpeaks"))
    assert best is not None and best.channel == "ok"
    assert _choose([vetoed]) is None


def test_an_eligible_row_with_no_valid_minute_is_not_stored() -> None:
    assert _choose([_row(_beats([400.0]), [False])]) is None


def test_rule_2_is_judged_on_the_minutes_valid_in_both(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, F64] = {}

    def timing(p: F64, c: F64) -> SimpleNamespace:
        seen["pair"], seen["chan"] = p, c
        return SimpleNamespace(disagree=True, offset_s=0.0)

    def resolve(*_a, **_k) -> SimpleNamespace:  # noqa: ANN002, ANN003
        return SimpleNamespace(lead_unmatched_qrs=0.9, other_unmatched_qrs=0.1)

    monkeypatch.setattr(hp, "timing_check", timing)
    monkeypatch.setattr(hp, "resolve_disagreement", resolve)
    b = _beats([400.0] * 3)
    chan = _row(b, [True, True, False], name="L_T", snr=20.0)
    pair = _row(b, [False, True, True], name="LVN1-RVN2", source="pair", snr=5.0)
    best = _choose([chan, pair])
    shared = b[(b >= W) & (b < 2 * W)]
    np.testing.assert_array_equal(seen["pair"], shared)
    np.testing.assert_array_equal(seen["chan"], shared)
    assert best is not None and best.channel == "LVN1-RVN2"  # the pair carries the QRS
    np.testing.assert_array_equal(best.beats_s, b[b >= W])  # stored with ITS valid minutes
    assert "rule 2 on 1 shared valid minutes" in best.note


def test_rule_2_unresolved_stores_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hp, "timing_check", lambda _p, _c: SimpleNamespace(disagree=True, offset_s=0.0)
    )
    monkeypatch.setattr(
        hp,
        "resolve_disagreement",
        lambda *_a, **_k: SimpleNamespace(lead_unmatched_qrs=0.9, other_unmatched_qrs=0.9),
    )
    b = _beats([400.0] * 2)
    rows = [_row(b, [True, True], name="L_T"), _row(b, [True, True], name="P-Q", source="pair")]
    assert _choose(rows) is None


def test_rule_2_is_not_run_without_a_shared_valid_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k) -> None:  # noqa: ANN002, ANN003
        raise AssertionError("no shared minute: nothing to compare")

    monkeypatch.setattr(hp, "timing_check", boom)
    b = _beats([400.0] * 2)
    chan = _row(b, [True, False], name="L_T", snr=20.0)
    pair = _row(b, [False, True], name="P-Q", source="pair", snr=5.0)
    best = _choose([chan, pair])
    assert best is not None and best.channel == "L_T"  # equal minutes: SNR


# --- veto precision without the count gate ---------------------------------------------


def test_per_minute_refines_the_veto_of_a_plausible_row_that_fails_the_count_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hc, "_detect", lambda _det, _x, _fs: np.zeros(0))
    monkeypatch.setattr(hc, "_harmed", lambda _b0, _b1, _times: 0)
    r = replace(
        _row(_beats([400.0]), [True]),
        count=replace(_row(np.zeros(0), [True]).count, passes=False),
        transient_harm=2 / hc.N_INJECTIONS,  # 0.01 on 200: not settled
    )

    def block(_k: int):  # noqa: ANN202
        return np.zeros(0), lambda _r: np.zeros(0)

    assert hc._refine_harm([r], block, FS)[0].n_injections == hc.N_INJECTIONS
    assert hc._refine_harm([r], block, FS, need_count=False)[0].n_injections > hc.N_INJECTIONS


# --- mask-grade pairs need (a) and (c) -------------------------------------------------


@pytest.mark.parametrize(
    ("rate_ok", "morph_ok", "stored"),
    [(True, True, True), (False, True, False), (True, False, False), (None, None, False)],
)
def test_a_mask_grade_pair_must_pass_the_rate_and_morphology_checks(
    rate_ok: bool | None, morph_ok: bool | None, stored: bool
) -> None:
    pair = replace(
        _row(np.zeros(0), [True], name="P-Q", source="pair", snr=90.0),
        transient_harm=0.5,
        rate_ok=rate_ok,
        morphology_ok=morph_ok,
    )
    chan = replace(_row(np.zeros(0), [True], name="raw", snr=10.0), transient_harm=0.5)
    train, grade = hc.peri_r_train([pair, chan])
    assert grade == "mask"
    assert train is not None and train.channel == ("P-Q" if stored else "raw")


# --- end to end: one clear minute in three ----------------------------------------------

SHORT_W = 20.0
"""The 65 s rig on a 20 s grid: three 'minutes', so gapping can be seen end to end."""


@pytest.fixture(scope="module")
def unclear():  # noqa: ANN201
    """Minutes 1-2 are made unclear: whole-recording, the count gate is unassessable.

    The heart is present throughout (this rig's pair passes every recording-level gate,
    test_hr_adopt), so only the per-minute plumbing decides what is stored.
    """
    rig = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, site_gain=STRONG_STOMACH, seed=31)
    real_ac = hc.autocorr_rate

    def ac(x, fs):  # noqa: ANN001, ANN202
        starts, bpm = real_ac(x, fs)
        return starts, np.where(starts > 0, np.nan, bpm)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hc, "AC_WINDOW_S", SHORT_W)
        mp.setattr(hc, "autocorr_rate", ac)
        out = hc.gated_selection(rig.rec, find_events(rig.rec), per_minute=True)
    return rig, out


def test_per_minute_stores_the_clear_minute_where_the_whole_recording_gate_cannot(
    unclear,  # noqa: ANN001
) -> None:
    rig, (best, rows) = unclear
    assert not any(r.count.assessable for r in rows)  # whole-recording: nothing can pass
    assert not any(r.passes for r in rows)
    assert best is not None and best.source == "pair" and best.eligible
    assert best.valid == (True, False, False) and best.minute_s == (0.0, 20.0, 40.0)
    assert best.beats_s.size and best.beats_s.max() < SHORT_W  # minutes 1-2 removed
    truth = rig.beats_s[rig.beats_s < SHORT_W]
    assert abs(best.beats_s.size / truth.size - 1.0) <= hc.PROVISIONAL_MAX_COUNT_DEV
    assert best.note.startswith("per minute: most valid minutes (1 of 3)")


def test_per_minute_eligibility_is_the_recording_level_gates(unclear) -> None:  # noqa: ANN001
    _rig, (_best, rows) = unclear
    for r in rows:
        veto = bool(np.isfinite(r.transient_harm)) and (
            r.transient_harm <= hc.PROVISIONAL_MAX_TRANSIENT_HARM
        )
        morph = r.source == "channel" or r.morphology_ok is True
        assert r.eligible == bool(r.plausible and veto and np.isfinite(r.snr) and morph)
        if r.source == "pair" and r.plausible and veto:
            assert r.morphology_ok is not None and r.rate_ok is not None  # the cross-check ran
    assert any(r.eligible for r in rows if r.source == "pair")


def test_a_pair_minute_failing_the_rate_check_is_removed_inside_the_selection() -> None:
    rig = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, site_gain=STRONG_STOMACH, seed=31)
    real_ac, real_rc = hc.autocorr_rate, hp.rate_check

    def ac(x, fs):  # noqa: ANN001, ANN202
        starts, bpm = real_ac(x, fs)
        return starts, np.where(starts > 0, np.nan, bpm)

    def rc(*a, **k):  # noqa: ANN002, ANN003, ANN202 - (a) fails exactly in the one clear minute
        return replace(real_rc(*a, **k), bad_minutes_s=(0.0,))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hc, "AC_WINDOW_S", SHORT_W)
        mp.setattr(hc, "autocorr_rate", ac)
        mp.setattr(hp, "rate_check", rc)
        best, rows = hc.gated_selection(rig.rec, find_events(rig.rec), per_minute=True)
    pairs = [r for r in rows if r.source == "pair" and r.rate_ok is not None]
    assert pairs and all(not r.valid[0] for r in pairs)
    assert best is None or best.source == "channel"


def test_a_pair_failing_morphology_is_not_eligible_for_per_minute_storage() -> None:
    rig = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, site_gain=STRONG_STOMACH, seed=31)
    real_ac, real_mc = hc.autocorr_rate, hp.morphology_check

    def ac(x, fs):  # noqa: ANN001, ANN202
        starts, bpm = real_ac(x, fs)
        return starts, np.where(starts > 0, np.nan, bpm)

    def mc(*a, **k):  # noqa: ANN002, ANN003, ANN202 - (c) fails for every pair
        return replace(real_mc(*a, **k), passes=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hc, "AC_WINDOW_S", SHORT_W)
        mp.setattr(hc, "autocorr_rate", ac)
        mp.setattr(hp, "morphology_check", mc)
        best, rows = hc.gated_selection(rig.rec, find_events(rig.rec), per_minute=True)
    vetted = [r for r in rows if r.source == "pair" and r.morphology_ok is False]
    assert vetted and not any(r.eligible for r in vetted)
    assert best is None or best.source == "channel"
