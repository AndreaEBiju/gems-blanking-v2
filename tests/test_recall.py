"""Task 09 gate scoring, as pre-declared 2026-09-26 - on synthetic marks and candidates."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.detect.recall import (
    BUDGET_MIN_RECORDINGS,
    CANDIDATE_BUDGET,
    TUNING_LABEL,
    Z_ENTER,
    Z_EXIT,
    RevealMismatchError,
    SpanInput,
    artifacts_needed,
    budget_record,
    budget_status,
    candidate_digest,
    chance_of_cover,
    check_next_round,
    clopper_pearson,
    covered_fraction,
    diagnose_miss,
    is_covered,
    load_round,
    next_round_gate,
    one_sided_lower,
    poisson_binomial_95,
    pooled_gate,
    record_generator_change,
    score_round,
    score_stored_round,
    span_bootstrap,
    write_budget,
)
from gems_blanking_v2.io.store import GemsStore

from tests.conftest import make_band_z

FS = 24414.0
ONE_SAMPLE = 1.0 / FS


# --- coverage -----------------------------------------------------------------


def test_an_artifact_touched_by_a_single_candidate_sample_is_covered() -> None:
    art = (10.0, 11.0)
    assert is_covered(art, [[10.5, 10.5 + ONE_SAMPLE]])
    assert is_covered(art, [[11.0 - ONE_SAMPLE, 12.0]])  # one sample over the far edge
    assert is_covered(art, [[9.0, 10.0 + ONE_SAMPLE]])  # one sample over the near edge


def test_an_adjacent_candidate_that_does_not_overlap_does_not_cover() -> None:
    art = (10.0, 11.0)
    assert not is_covered(art, [[11.0, 11.5]])  # starts exactly where it ends
    assert not is_covered(art, [[9.0, 10.0]])  # ends exactly where it starts
    assert not is_covered(art, np.zeros((0, 2)))


def test_the_covered_fraction_counts_overlapping_candidates_once() -> None:
    art = (0.0, 10.0)
    assert covered_fraction(art, [[1.0, 4.0], [3.0, 6.0], [20.0, 30.0]]) == pytest.approx(0.5)
    assert covered_fraction(art, [[-5.0, 50.0]]) == pytest.approx(1.0)
    assert covered_fraction(art, [[10.0, 12.0]]) == 0.0


# --- the statistics -----------------------------------------------------------


def test_the_spec_projection_is_the_one_sided_bound() -> None:
    """0.05**(1/n) >= 0.98 -> n >= 149, as pre-declared; two-sided 95% would need 183."""
    assert one_sided_lower(149, 149) == pytest.approx(0.05 ** (1 / 149))
    assert artifacts_needed(0) == 149
    assert artifacts_needed(0, confidence=0.975) == 183
    assert clopper_pearson(149, 149)[0] == pytest.approx(0.025 ** (1 / 149))


def test_one_miss_moves_the_target_well_past_149() -> None:
    assert artifacts_needed(1) > 149
    assert one_sided_lower(29, 30) < 0.98


def test_the_bootstrap_resamples_spans_not_artifacts() -> None:
    """Two spans of 10: one all covered, one all missed.

    Resampling ARTIFACTS would give a tight interval around 0.5; resampling SPANS
    can draw either span twice, so the interval spans 0 to 1.
    """
    lo, hi, one, empty = span_bootstrap([(10, 10), (0, 10)])
    assert lo == 0.0 and hi == 1.0 and one == 0.0 and empty == 0


def test_the_bootstrap_is_seeded_and_drops_draws_with_no_artifacts() -> None:
    per = [(3, 3), (0, 0), (4, 5)]
    assert span_bootstrap(per) == span_bootstrap(per)
    *_, empty = span_bootstrap(per)
    assert empty > 0
    with pytest.raises(ValueError, match="undefined"):
        span_bootstrap([(0, 0), (0, 0)])


# --- diagnosis ----------------------------------------------------------------


def _traces(level: float, where: tuple[float, float] = (10.2, 10.4)) -> list:
    # 200 s: the traces must cover the recording timeline the artifacts sit on.
    return [make_band_z("300-3000", 200.0, bumps=((*where, level, "L_T"),)),
            make_band_z("2-50", 200.0, base=0.3)]


@pytest.mark.parametrize(
    ("level", "verdict"),
    [(Z_EXIT - 0.2, "generator_blind_spot"), (Z_EXIT, "threshold"),
     (Z_ENTER - 0.01, "threshold"), (Z_ENTER + 1.0, "gated")],
)
def test_a_miss_is_classified_by_the_generators_own_thresholds(
    level: float, verdict: str
) -> None:
    d = diagnose_miss((10.0, 11.0), _traces(level))
    assert d.verdict == verdict
    assert d.max_band == "300-3000" and d.max_signal == "L_T"
    assert d.max_z == pytest.approx(level)
    assert set(d.max_z_by_band) == {"300-3000", "2-50"}


def test_z_outside_the_artifact_does_not_count() -> None:
    d = diagnose_miss((10.0, 11.0), _traces(9.0, where=(11.0, 11.5)))
    assert d.verdict == "generator_blind_spot"


def test_a_miss_without_evidence_is_undiagnosed_not_guessed() -> None:
    assert diagnose_miss((1.0, 2.0), None).verdict == "undiagnosed"
    nan_only = [make_band_z("300-3000", 20.0, nan_before_s=5.0)]
    assert diagnose_miss((1.0, 2.0), nan_only).verdict == "undiagnosed"


# --- the round ----------------------------------------------------------------


def _span(i: int, arts: list, cands: list, traces: list | None = None) -> SpanInput:
    return SpanInput(f"plan_x_s{i}", f"rec{i}", "J", "baseline", 100.0 * i,
                     100.0 * i + 120.0, np.asarray(arts, dtype=float).reshape(-1, 2),
                     np.asarray(cands, dtype=float).reshape(-1, 2), traces)


def test_a_clean_round_reports_the_projection_and_says_draw_again() -> None:
    spans = [_span(i, [[100 * i + 10.0 * j, 100 * i + 10.0 * j + 1] for j in range(6)],
                   [[100 * i + 10.0 * j + 0.5, 100 * i + 10.0 * j + 0.6] for j in range(6)])
             for i in range(5)]
    score = score_round("plan_x", spans, seed=7)
    s = score.statistics()
    assert (s["found"], s["covered"], s["missed"]) == (30, 30, 0)
    assert s["artifacts_per_minute"] == pytest.approx(3.0)
    assert not s["gate_cleared"]
    assert s["more_artifacts_needed"] == 119 and s["more_rounds_projected"] == 4
    assert s["secondary_recall_at_half_overlap"] == 0.0  # each candidate covers 10%
    assert s["quoted_interval"] == "clopper_pearson"  # bootstrap is [1, 1] here
    assert score.next_step().startswith("zero misses")
    json.dumps(score.to_json(), allow_nan=False)


def test_a_round_with_a_miss_says_stop_and_diagnoses_it() -> None:
    traces = _traces(2.2, where=(20.1, 20.3))
    spans = [_span(0, [[5.0, 6.0], [20.0, 21.0]], [[5.5, 5.6]], traces),
             _span(1, [[105.0, 106.0]], [])]
    score = score_round("plan_x", spans)
    s = score.statistics()
    assert (s["found"], s["covered"], s["missed"]) == (3, 1, 2)
    diag = {m.miss_id: m.diagnosis.verdict for m in score.misses}  # type: ignore[union-attr]
    assert diag == {"plan_x_s0#1": "threshold", "plan_x_s1#0": "undiagnosed"}
    assert score.next_step().startswith("STOP")
    text = json.dumps(score.to_json(), allow_nan=False)  # an undiagnosed miss has no NaN
    assert "NaN" not in text
    assert "MISS plan_x_s0#1" in score.report()


# --- the sequential rule ------------------------------------------------------


def _doc(verdicts: list[str | None]) -> dict:
    arts = [{"id": f"s1#{i}", "covered": v is None,
             **({"diagnosis": {"verdict": v}} if v else {})} for i, v in enumerate(verdicts)]
    return {"artifacts": arts}


def test_the_next_round_is_refused_until_every_miss_is_diagnosed_and_fixed() -> None:
    assert next_round_gate(None, None) == (False, "the last round has not been scored")
    ok, why = next_round_gate(_doc([None, "undiagnosed"]), None)
    assert not ok and "undiagnosed" in why
    ok, why = next_round_gate(_doc([None, "threshold"]), None)
    assert not ok and "not closed" in why and "no closure recorded" in why
    ok, why = next_round_gate(_doc([None, "threshold"]), {"s1#1": {"task07_fix": "   "}})
    assert not ok
    ok, why = next_round_gate(_doc([None, "threshold"]),
                              {"s1#1": {"task07_fix": "commit abc123: z_enter 3.0 -> 2.5"}})
    assert not ok and "fixed_at" in why  # a fix must say when, or eligibility is unknowable
    ok, _ = next_round_gate(_doc([None, "threshold"]),
                            {"s1#1": {"task07_fix": "commit abc123: z_enter 3.0 -> 2.5",
                                      "fixed_at": "2026-09-28T10:00:00+00:00"}})
    assert ok
    ok, why = next_round_gate(_doc([None, None]), None)
    assert ok and "fresh recorded seed" in why


# --- chance recall and time covered: DESCRIPTIVE (ruling 2026-09-28) -----------


def test_chance_of_cover_is_the_fraction_of_legal_starts_that_touch_a_candidate() -> None:
    """Span [0, 100), candidate [40, 50), 10 s mark: starts (30, 50) of [0, 90] -> 20/90."""
    assert chance_of_cover(10.0, (0.0, 100.0), [[40.0, 50.0]]) == pytest.approx(20 / 90)
    assert chance_of_cover(10.0, (0.0, 100.0), [[0.0, 100.0]]) == 1.0
    assert chance_of_cover(10.0, (0.0, 100.0), np.zeros((0, 2))) == 0.0
    assert chance_of_cover(100.0, (0.0, 100.0), [[99.0, 99.5]]) == 1.0  # one legal start
    assert chance_of_cover(10.0, (0.0, 100.0), [[95.0, 99.0]]) == pytest.approx(5 / 90)
    assert chance_of_cover(10.0, (0.0, 100.0), [[-20.0, -10.0], [140.0, 150.0]]) == 0.0


def test_chance_of_cover_agrees_with_placing_marks_at_random() -> None:
    rng = np.random.default_rng(20260928)
    cands = np.array([[5.0, 5.2], [30.0, 41.0], [42.0, 43.0], [90.0, 99.0]])
    for length in (0.05, 0.5, 3.0):
        t = rng.uniform(0.0, 100.0 - length, 200_000)
        hit = ((t[:, None] < cands[:, 1]) & (t[:, None] + length > cands[:, 0])).any(axis=1)
        assert chance_of_cover(length, (0.0, 100.0), cands) == pytest.approx(hit.mean(),
                                                                             abs=0.004)


def test_time_covered_is_weighted_by_span_length() -> None:
    """A 30 s span fully covered and a 90 s span bare is 25% of the time, not 50%."""
    short = SpanInput("p_s1", "r1", "J", "baseline", 0.0, 30.0,
                      np.array([[1.0, 1.1]]), np.array([[0.0, 30.0]]))
    long_ = SpanInput("p_s2", "r2", "J", "baseline", 100.0, 190.0,
                      np.array([[150.0, 150.1]]), np.zeros((0, 2)))
    d = score_round("p", [short, long_]).statistics()["descriptive"]

    assert d["time_covered_fraction"] == pytest.approx(0.25)


def test_the_poisson_binomial_range_is_central_95() -> None:
    lo, hi = poisson_binomial_95([0.5] * 100)
    assert (lo, hi) == (0.4, 0.6)  # Binomial(100, 0.5): 2.5% / 97.5% quantiles 40, 60
    assert poisson_binomial_95([1.0] * 5) == (1.0, 1.0)


def test_every_round_reports_time_covered_and_chance_recall_as_description() -> None:
    """Reported, labelled, and changing no gate number or verdict."""
    full = _span(0, [[10.0, 10.1]], [[0.0, 120.0]])  # candidates everywhere
    none = _span(1, [[150.0, 150.2]], [[150.1, 150.15], [400.0, 410.0]])  # a sliver
    score = score_round("plan_x", [full, none])
    s = score.statistics()
    d = s["descriptive"]
    assert "not a gate condition" in d["note"]
    assert score.per_span[0]["time_covered"] == 1.0
    assert score.per_span[1]["time_covered"] == pytest.approx(0.05 / 120, rel=1e-6)
    assert d["time_covered_fraction"] == pytest.approx((1.0 + 0.05 / 120) / 2, rel=1e-6)
    sliver = (0.05 + 0.2) / (120.0 - 0.2)  # covering starts (149.9, 150.15)
    assert score.per_span[1]["candidates_in_span"] == 1
    assert d["chance_recall"] == pytest.approx((1.0 + sliver) / 2)
    assert d["p_all_covered_by_chance"] == pytest.approx(sliver)
    assert "DESCRIPTIVE, not a gate condition" in score.report()
    assert "time covered" in score.report()
    plain = {k: v for k, v in s.items() if k != "descriptive"}
    stripped = score_round("plan_x", [full, none])
    for sp in stripped.per_span:
        sp.pop("chance_per_mark")
    assert dict(stripped.statistics()) == plain  # gate numbers unmoved
    assert stripped.next_step() == score.next_step()


# --- a round in the store -----------------------------------------------------


def _write_round(store: GemsStore, plan_id: str, marks: list[list[list[float]]],
                 commit: int | None = None, committed_at: str | list[str] | None = None,
                 revealed: list | None = None) -> None:
    spans = [{"recording_id": f"blk{k}_20260916T030321Z", "animal": "J",
              "condition": "baseline", "start_s": 200.0, "stop_s": 320.0}
             for k in range(len(marks))]
    p = store.audit_plan_path(plan_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"plan_id": plan_id, "seed": 99, "spans": spans}), encoding="utf-8")
    for k, (sp, mk) in enumerate(zip(spans, marks, strict=True)):
        if commit is not None and k >= commit:
            continue
        d = store.audit_dir(sp["animal"], sp["recording_id"])
        d.mkdir(parents=True, exist_ok=True)
        sid = f"{plan_id}_s{k + 1}"
        (d / f"{sid}_blind_marks.json").write_text(json.dumps(
            {"span_id": sid, "marks": [{"start_s": a, "stop_s": b} for a, b in mk],
             **({"committed_at": committed_at if isinstance(committed_at, str)
                 else committed_at[k]} if committed_at else {})}),
            encoding="utf-8")
        record: dict = {"span_s": [200.0, 320.0], "assessable_regions_s": [[20.0, 580.0]]}
        if revealed is not None:  # what a window with digests writes at reveal
            record["reveal"] = {"candidates_sha256": candidate_digest(revealed[k]),
                                "n_candidates": len(revealed[k])}
        (d / f"{sid}_plan.json").write_text(json.dumps(record), encoding="utf-8")


def test_a_round_is_scored_whole_or_not_at_all(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, "plan_20260926T000000Z_00000063", [[[210, 211]], [[250, 252]]], commit=1)
    with pytest.raises(FileNotFoundError, match="span 2"):
        load_round(store, "plan_20260926T000000Z_00000063")
    ok, why = check_next_round(store)
    assert not ok and "still open" in why


def _budget_rows(n: int, candidates: float) -> list[dict]:
    return [{"animal": "AB"[i % 2], "condition": ("baseline", "stim_recovery")[i % 2],
             "session": f"s{i}", "candidates": candidates, "assessable_s": 560.0,
             "covered_s": 56.0} for i in range(n)]


def _within_budget(store: GemsStore, reveal_sha: str | None = None) -> None:
    write_budget(store, budget_record(_budget_rows(BUDGET_MIN_RECORDINGS, 800.0),
                                      reveal_sha=reveal_sha, sample_rule="test"))


def test_the_store_round_gates_the_next_round_through_its_resolutions(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _within_budget(store)
    ok, why = check_next_round(store)
    assert ok and why.startswith("no earlier round: draw round 1")
    pid = "plan_20260926T000000Z_00000063"
    _write_round(store, pid, [[[210, 211], [230, 231]], [[250, 252]]])
    assert check_next_round(store)[0] is False  # complete but unscored

    def reveal(span: dict) -> tuple:
        assert span["record"]["assessable_regions_s"] == [[20.0, 580.0]]
        traces = [make_band_z("100-300", 600.0, bumps=((230.2, 230.4, 5.0, "R_V1"),))]
        return np.array([[210.5, 210.6], [251.0, 251.1]]), traces

    score = score_stored_round(store, pid, reveal)
    assert (score.found, score.covered) == (3, 2)
    on_disk = json.loads(store.audit_score_path(pid).read_text(encoding="utf-8"))
    assert on_disk["statistics"]["missed"] == 1
    assert on_disk["artifacts"][1]["diagnosis"]["verdict"] == "gated"
    ok, why = check_next_round(store)
    assert not ok and "not closed" in why
    store.audit_resolution_path(pid).write_text(json.dumps(
        {f"{pid}_s1#1": {"task07_fix": "combine rule fixed in commit abc123",
                         "fixed_at": "2026-09-28T10:00:00+00:00"}}), encoding="utf-8")
    assert check_next_round(store)[0] is True


def test_no_statistic_is_computed_on_an_empty_round() -> None:
    score = score_round("plan_x", [_span(0, [], [])])
    s = score.statistics()
    assert s["found"] == 0 and "recall" not in s
    assert "undefined" in score.next_step()
    assert not math.isnan(s["artifacts_per_minute"])


# --- a score records exactly what it scored (ratified 2026-09-27) --------------

PID = "plan_20260926T000000Z_00000063"
CANDS = [np.array([[210.5, 210.6], [230.2, 230.3]]), np.array([[251.0, 251.1]])]


def _reveal_with(cands: list) -> object:
    def reveal(span: dict) -> tuple:
        return cands[int(span["span_id"].rsplit("_s", 1)[1]) - 1], None
    return reveal


def test_the_digest_is_canonical() -> None:
    a = np.array([[1.0, 1.5], [0.2, 0.3]])
    assert candidate_digest(a) == candidate_digest(a[::-1])  # order-free
    assert candidate_digest(a) == candidate_digest(a + 3e-7)  # float noise below 1 us
    assert candidate_digest(a) != candidate_digest(a + np.array([[0.0, 0.01], [0.0, 0.0]]))
    assert candidate_digest(np.zeros((0, 2))) == candidate_digest([])


def test_a_tampered_candidate_list_is_refused(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]], revealed=CANDS)
    tampered = [CANDS[0][:1], CANDS[1]]  # one revealed candidate missing on recompute
    with pytest.raises(RevealMismatchError, match="do not match"):
        score_stored_round(store, PID, _reveal_with(tampered))
    assert not store.audit_score_path(PID).exists()  # nothing written
    score = score_stored_round(store, PID, _reveal_with(CANDS))  # the true list scores
    assert score.warnings == []
    assert score.provenance["reveal_digests"] == {f"{PID}_s1": "verified",
                                                  f"{PID}_s2": "verified"}


def test_a_span_without_a_digest_scores_with_a_warning(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]])  # pre-digest app
    score = score_stored_round(store, PID, _reveal_with(CANDS))
    assert len(score.warnings) == 2
    assert all("without candidate digests" in w for w in score.warnings)
    assert "WARNING" in score.report()
    assert json.loads(store.audit_score_path(PID).read_text(encoding="utf-8"))["warnings"]


def test_the_score_records_the_generator_it_used(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211]], [[250, 252]]], revealed=CANDS)
    score_stored_round(store, PID, _reveal_with(CANDS))
    doc = json.loads(store.audit_score_path(PID).read_text(encoding="utf-8"))
    gen = doc["provenance"]["generator"]
    assert gen["parameters"]["z_enter"] == Z_ENTER and gen["parameters"]["z_exit"] == Z_EXIT
    assert len(gen["generation_sha256"]) == 64 and gen["gate"]["gate_recall"] == 0.98
    assert "gems_blanking_v2.detect.chain" in gen["generation_modules"]


# --- only rounds labelled after the last fix count ----------------------------


def test_a_round_labelled_before_a_recorded_fix_leaves_the_pooled_bound(
    tmp_path: Path,
) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    a, b = "plan_20260927T100000Z_0000000a", "plan_20260929T100000Z_0000000b"
    _write_round(store, a, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    score_stored_round(store, a, _reveal_with(CANDS))
    assert pooled_gate(store)["pooled_rounds"] == [a]  # no fix yet: it counts
    store.audit_resolution_path(a).write_text(json.dumps({f"{a}_s1#0": {
        "task07_fix": "commit abc", "fixed_at": "2026-09-28T09:00:00+00:00"}}),
        encoding="utf-8")
    after = pooled_gate(store)
    assert after["pooled_rounds"] == [] and after["found"] == 0
    assert after["excluded_rounds"][0]["plan_id"] == a
    assert "before the last fix" in after["excluded_rounds"][0]["reason"]
    _write_round(store, b, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-29T10:05:00+00:00", revealed=CANDS)
    score = score_stored_round(store, b, _reveal_with(CANDS))
    assert score.pooled is not None and score.pooled["pooled_rounds"] == [b]
    assert score.pooled["found"] == 3  # round b alone, not a + b
    assert "left out " + a in score.report()


def test_a_tuning_re_score_is_labelled_kept_apart_and_never_pooled(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    gate = score_stored_round(store, PID, _reveal_with(CANDS))
    fixed = [np.array([[210.5, 210.6], [230.2, 230.3]]),
             np.array([[251.0, 251.1], [300.0, 301.0]])]
    tune = score_stored_round(store, PID, _reveal_with(fixed), mode="tuning")  # not refused
    assert tune.mode == "tuning" and TUNING_LABEL.upper() in tune.report()
    tuned = list(store.audit_score_path(PID).parent.glob(f"{PID}_tuning_*.json"))
    assert len(tuned) == 1
    assert json.loads(tuned[0].read_text(encoding="utf-8"))["evidence"] == TUNING_LABEL
    on_disk = json.loads(store.audit_score_path(PID).read_text(encoding="utf-8"))
    assert on_disk["statistics"] == gate.to_json()["statistics"]  # gate score untouched
    assert pooled_gate(store)["pooled_rounds"] == [PID]  # the tuning file is not a round


def test_a_gate_score_is_not_replaced_under_a_different_generator(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211]], [[250, 252]]], revealed=CANDS)
    score_stored_round(store, PID, _reveal_with(CANDS))
    path = store.audit_score_path(PID)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["provenance"]["generator"]["generation_sha256"] = "0" * 64  # scored by older code
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RevealMismatchError, match="different generator"):
        score_stored_round(store, PID, _reveal_with(CANDS))


def test_a_round_started_before_a_fix_is_not_eligible_even_if_finished_after(
    tmp_path: Path,
) -> None:
    """Labelled = its FIRST committed span: a round straddling a fix is tuning data."""
    store = GemsStore.initialise(tmp_path / "gems")
    store.audit_resolution_path("plan_0").parent.mkdir(parents=True, exist_ok=True)
    store.audit_resolution_path("plan_0").write_text(json.dumps({"x#0": {
        "task07_fix": "commit abc", "fixed_at": "2026-09-28T09:00:00+00:00"}}),
        encoding="utf-8")
    _write_round(store, PID, [[[210, 211]], [[250, 252]]], revealed=CANDS,
                 committed_at=["2026-09-28T08:00:00+00:00", "2026-09-28T10:00:00+00:00"])
    score = score_stored_round(store, PID, _reveal_with(CANDS))
    assert score.pooled is not None and score.pooled["pooled_rounds"] == []


def test_a_generator_change_with_no_miss_still_makes_the_round_tuning_data(
    tmp_path: Path,
) -> None:
    """Round 1's case: zero misses, a changed generator, and it must leave the pool."""
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    score_stored_round(store, PID, _reveal_with(CANDS))
    assert pooled_gate(store)["pooled_rounds"] == [PID]
    store.audit_resolution_path(PID).write_text(json.dumps(
        {f"{PID}_s1#0": {"task07_fix": "earlier", "fixed_at": "2026-09-27T09:00:00+00:00"}}),
        encoding="utf-8")  # an earlier miss fix that must survive
    when = datetime(2026, 9, 28, 18, 0, tzinfo=UTC)

    key = record_generator_change(store, PID, change="contact screen", reason="coverage",
                                  fixed_at=when)

    res = json.loads(store.audit_resolution_path(PID).read_text(encoding="utf-8"))
    assert set(res) == {f"{PID}_s1#0", key} and key.startswith("generator_change:")
    assert len(res[key]["generator"]["generation_sha256"]) == 64
    after = pooled_gate(store)
    assert after["pooled_rounds"] == [] and after["last_fix_at"] == when.isoformat()
    with pytest.raises(ValueError, match="never overwritten"):
        record_generator_change(store, PID, change="x", reason="y", fixed_at=when)
    with pytest.raises(ValueError, match="timezone-aware"):
        record_generator_change(store, PID, change="x", reason="y",
                                fixed_at=datetime(2026, 9, 28, 19))
    assert next_round_gate(json.loads(store.audit_score_path(PID).read_text(
        encoding="utf-8")), res)[0] is True  # a change keyed off-miss blocks nothing


# --- the pool: generation hash, chance bound, eligible projection (2026-09-28) --

FULL = [np.array([[200.0, 320.0]]), np.array([[200.0, 320.0]])]  # every second covered


def test_a_round_under_another_generation_hash_leaves_the_pool_without_a_fix_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forgotten fix record must not leave tuning data in the pool."""
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    score_stored_round(store, PID, _reveal_with(CANDS))
    assert pooled_gate(store)["pooled_rounds"] == [PID]

    monkeypatch.setattr(chain, "generation_sha256", lambda: "f" * 64)
    after = pooled_gate(store)

    assert after["pooled_rounds"] == [] and after["last_fix_at"] is None
    assert "scored under generator" in after["excluded_rounds"][0]["reason"]


def test_a_round_whose_chance_bound_reaches_the_bar_is_not_counted_and_says_why(
    tmp_path: Path,
) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=FULL)
    score = score_stored_round(store, PID, _reveal_with(FULL))

    assert score.covered == score.found == 3  # clean on its face...
    assert score.pooled is not None and score.pooled["pooled_rounds"] == []
    why = score.pooled["excluded_rounds"][0]["reason"]
    assert "chance-recall upper 95% bound would be 1.000" in why
    assert score.next_step().startswith("NOT GATE EVIDENCE") and "1.000" in score.next_step()
    assert score.statistics()["descriptive"]["chance_margin"] == pytest.approx(0.98 - 1.0)
    assert "chance margin   -0.020" in score.report()


def test_a_round_below_the_chance_bound_counts_and_reports_its_margin(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    score = score_stored_round(store, PID, _reveal_with(CANDS))

    assert score.pooled is not None and score.pooled["pooled_rounds"] == [PID]
    assert 0.0 < score.pooled["chance_margin"] <= 0.98
    assert score.pooled["chance_margin"] == pytest.approx(
        0.98 - score.pooled["chance_recall_upper_95"])
    assert "pooled chance upper bound" in score.report()
    assert score.next_step().startswith("zero misses")


def test_the_chance_bound_is_pooled_in_labelling_order(tmp_path: Path) -> None:
    """Round a (full coverage) cannot count alone; round b (sparse) can.

    Taken in labelling order, a is refused, then b is admitted on its own chance.
    """
    store = GemsStore.initialise(tmp_path / "gems")
    a, b = "plan_20260927T100000Z_0000000a", "plan_20260928T100000Z_0000000b"
    _write_round(store, a, [[[210, 211]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=FULL)
    _write_round(store, b, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-28T10:05:00+00:00", revealed=CANDS)
    score_stored_round(store, a, _reveal_with(FULL))
    pg = score_stored_round(store, b, _reveal_with(CANDS)).pooled

    assert pg is not None and pg["pooled_rounds"] == [b]
    assert [e["plan_id"] for e in pg["excluded_rounds"]] == [a]


def test_the_bound_is_the_pools_not_each_rounds(tmp_path: Path) -> None:
    """Sparse first, then fully covered: pooled, the second counts.

    Alone it would be refused (upper bound 1.0); pooled the bound stays below 0.98.
    """
    store = GemsStore.initialise(tmp_path / "gems")
    a, b = "plan_20260927T100000Z_0000000a", "plan_20260928T100000Z_0000000b"
    sparse = [[[210, 211], [230, 231], [240, 241], [260, 261]], [[250, 252], [270, 271]]]
    _write_round(store, a, sparse, committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    _write_round(store, b, [[[210, 211]], [[250, 252]]],
                 committed_at="2026-09-28T10:05:00+00:00", revealed=FULL)
    score_stored_round(store, a, _reveal_with(CANDS))
    pg = score_stored_round(store, b, _reveal_with(FULL)).pooled

    assert pg is not None and pg["pooled_rounds"] == [a, b]
    assert pg["chance_recall_upper_95"] < 0.98


def test_a_chance_bound_exactly_at_the_bar_does_not_count(tmp_path: Path) -> None:
    """'Below 0.98': 49 certain marks and one at p = 0.02 put the upper end at 49/50."""
    store = GemsStore.initialise(tmp_path / "gems")
    path = store.audit_score_path(PID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "plan_id": PID, "mode": "gate",
        "provenance": {"generator": {"generation_sha256": chain.generation_sha256()}},
        "per_span": [{"found": 50, "covered": 50,
                      "chance_per_mark": [1.0] * 49 + [0.02]}]}), encoding="utf-8")

    pg = pooled_gate(store)
    assert poisson_binomial_95([1.0] * 49 + [0.02])[1] == 0.98
    assert pg["pooled_rounds"] == [] and "0.980" in pg["excluded_rounds"][0]["reason"]


def test_a_round_without_recorded_chance_cannot_count(tmp_path: Path) -> None:
    """Fail closed: a bound that cannot be checked is not a bound that passed."""
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    score_stored_round(store, PID, _reveal_with(CANDS))
    path = store.audit_score_path(PID)
    doc = json.loads(path.read_text(encoding="utf-8"))
    for sp in doc["per_span"]:
        del sp["chance_per_mark"]
    path.write_text(json.dumps(doc), encoding="utf-8")

    pg = pooled_gate(store)
    assert pg["pooled_rounds"] == []
    assert "chance recall not recorded" in pg["excluded_rounds"][0]["reason"]


def test_the_projection_counts_only_eligible_rounds(tmp_path: Path) -> None:
    """Round 1's report said '58 more' from a round that no longer counts: 149 fresh."""
    store = GemsStore.initialise(tmp_path / "gems")
    _write_round(store, PID, [[[210, 211], [230, 231]], [[250, 252]]],
                 committed_at="2026-09-27T10:05:00+00:00", revealed=CANDS)
    counted = score_stored_round(store, PID, _reveal_with(CANDS)).statistics()
    assert counted["artifacts_needed_at_zero_further_misses"] == 149
    assert counted["more_artifacts_needed"] == 149 - 3
    assert counted["projection_basis"] == f"eligible rounds: {PID}"

    record_generator_change(store, PID, change="x", reason="y",
                            fixed_at=datetime(2026, 9, 28, 17, 7, 1, tzinfo=UTC))
    tune = score_stored_round(store, PID, _reveal_with(CANDS), mode="tuning")
    s = tune.statistics()
    assert s["more_artifacts_needed"] == 149  # 149 fresh, not 149 - 3
    assert s["projection_basis"] == f"eligible rounds: none; {PID} is not among them"
    assert "149 more" in tune.report()


# --- the candidate budget (declared in task 09; enforced 2026-09-28) ---------


def test_no_round_is_drawn_while_the_generator_is_unmeasured(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    ok, why = check_next_round(store)
    assert not ok and "has not been measured for the current generator" in why


def test_no_round_is_drawn_while_the_generator_exceeds_its_budget(tmp_path: Path) -> None:
    """Round 1's spans scaled to ~3,000-13,000 per recording: over budget, refused."""
    store = GemsStore.initialise(tmp_path / "gems")
    rows = _budget_rows(BUDGET_MIN_RECORDINGS, 800.0)
    for r in rows[:4]:  # 4 of 30 over: the 90% quantile is over -> refused
        r["candidates"] = 13_000.0
    write_budget(store, budget_record(rows, reveal_sha=None, sample_rule="test"))
    ok, why = check_next_round(store)
    assert not ok and "exceeds the candidate budget" in why
    rows = _budget_rows(BUDGET_MIN_RECORDINGS, 800.0)
    for r in rows[:2]:  # 2 of 30 over: the 90% quantile is within -> allowed
        r["candidates"] = 13_000.0
    write_budget(store, budget_record(rows, reveal_sha=None, sample_rule="test"))
    assert check_next_round(store)[0]


def test_a_budget_on_too_few_recordings_or_another_generator_does_not_count(
    tmp_path: Path,
) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    write_budget(store, budget_record(_budget_rows(5, 100.0), reveal_sha=None,
                                      sample_rule="five spans"))
    ok, why = budget_status(store)
    assert not ok and "at least" in why
    _within_budget(store, reveal_sha="a" * 64)  # measured with a different reveal
    assert not budget_status(store, reveal_sha="b" * 64)[0]
    assert budget_status(store, reveal_sha="a" * 64)[0]


def test_the_budget_threshold_is_the_declared_one() -> None:
    assert CANDIDATE_BUDGET == 3000
