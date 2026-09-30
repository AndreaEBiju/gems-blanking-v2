"""Tests for the filtered gate (Andrea, 2026-09-30: from round 6 the gate is filtered recall)."""

from __future__ import annotations

import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.detect.recall import (
    CHANCE_BOUND_MAX,
    GATE_RECALL,
    MIN_SPANS_PER_CONDITION,
    filtered_gate,
    one_sided_lower,
    score_marks,
    span_bootstrap,
)


def _marks(n_spans: int, per_span: int, *, misses: dict[int, str] | None = None,
           below: set[int] | None = None, chance: float = 0.3,
           conditions: tuple[str, ...] = ("baseline", "stim_recovery")) -> list[dict]:
    """Build ``n_spans`` spans of ``per_span`` marks, alternating conditions.

    Mark ``k`` (numbered across spans) is missed if in ``misses`` and ``below`` if in
    ``below``; every other mark is a covered target.
    """
    out, k = [], 0
    for s in range(n_spans):
        for _ in range(per_span):
            out.append({"id": f"p_s{s}#m{k}", "span_id": f"p_s{s}",
                        "condition": conditions[s % len(conditions)],
                        "covered": k not in (misses or {}),
                        "damage_class": "below" if k in (below or set()) else "target",
                        "chance": chance})
            k += 1
    return out


def test_a_miss_below_every_tolerance_leaves_the_filtered_bound_untouched() -> None:
    m = _marks(8, 40, misses={3: ""}, below={3})
    g = filtered_gate(m)
    assert g["filtered"]["found"] == 319 and g["filtered"]["covered"] == 319
    assert g["raw"]["found"] == 320 and g["raw"]["covered"] == 319
    assert g["target_misses_unclassified"] == []
    assert g["filtered"]["lower_bound_one_sided_95"] > g["raw"]["lower_bound_one_sided_95"]


def test_the_bound_is_the_more_conservative_of_clopper_pearson_and_the_span_bootstrap() -> None:
    m = _marks(8, 40, misses={3: "", 100: ""})
    g = filtered_gate(m)["filtered"]
    cp = one_sided_lower(318, 320)
    per_span = [(40 - (s == 0) - (s == 2), 40) for s in range(8)]
    b = span_bootstrap(per_span)[2]
    assert g["lower_bound_parts"] == {"clopper_pearson": cp, "span_bootstrap": b}
    assert g["lower_bound_one_sided_95"] == min(cp, b)


def test_the_gate_clears_only_with_the_bound_the_spans_and_a_positive_margin() -> None:
    ok = filtered_gate(_marks(6, 60))
    assert ok["filtered"]["lower_bound_one_sided_95"] >= GATE_RECALL
    assert ok["chance_over_targets"]["margin"] > 0 and ok["gate_cleared"]
    few = filtered_gate(_marks(MIN_SPANS_PER_CONDITION * 2 - 1, 80))  # one condition short
    assert few["filtered"]["lower_bound_one_sided_95"] >= GATE_RECALL and not few["gate_cleared"]
    assert "3 of each are needed" in few["condition_note"]
    coverage = filtered_gate(_marks(6, 60, chance=0.999))  # coverage alone reaches the bar
    assert coverage["chance_over_targets"]["margin"] <= 0 and not coverage["gate_cleared"]
    weak = filtered_gate(_marks(6, 60, misses={0: "", 70: "", 150: ""}))
    assert weak["filtered"]["lower_bound_one_sided_95"] < GATE_RECALL and not weak["gate_cleared"]


def test_the_chance_margin_is_computed_over_target_marks_only() -> None:
    m = _marks(6, 60, below=set(range(0, 360, 2)))
    for x in m:
        x["chance"] = 0.999 if x["damage_class"] == "below" else 0.3
    g = filtered_gate(m)
    assert g["chance_over_targets"]["chance_recall"] == pytest.approx(0.3)
    assert g["chance_bound_max"] == CHANCE_BOUND_MAX


def test_a_target_miss_is_listed_as_it_is() -> None:
    g = filtered_gate(_marks(6, 60, misses={7: ""}))
    assert g["target_misses_unclassified"] == ["p_s0#m7"]


def test_a_mark_without_a_class_is_refused_by_name() -> None:
    m = _marks(2, 3)
    m[4]["damage_class"] = None
    with pytest.raises(ValueError, match=r"p_s1#m4"):
        filtered_gate(m)
    m[4]["damage_class"] = "maybe"
    with pytest.raises(ValueError, match="no damage class"):
        filtered_gate(m)


def test_score_marks_takes_each_marks_chance_in_artifact_order() -> None:
    doc = {"per_span": [{"span_id": "a", "condition": "baseline", "chance_per_mark": [0.1, 0.2]},
                        {"span_id": "b", "condition": "stim_recovery", "chance_per_mark": [0.9]}],
           "artifacts": [{"id": "a#m0", "span_id": "a", "covered": True},
                         {"id": "b#m0", "span_id": "b", "covered": False},
                         {"id": "a#m1", "span_id": "a", "covered": True}]}
    m = score_marks(doc, {"a#m0": "target", "a#m1": "below", "b#m0": "target"})
    assert [(x["id"], x["chance"], x["condition"], x["damage_class"]) for x in m] == [
        ("a#m0", 0.1, "baseline", "target"), ("b#m0", 0.9, "stim_recovery", "target"),
        ("a#m1", 0.2, "baseline", "below")]
    doc["per_span"][0]["chance_per_mark"] = [0.1]
    with pytest.raises(ValueError, match="no chance of cover for a#m1"):
        score_marks(doc, {})


def test_recall_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.detect.recall" not in chain.generation_modules()


# --- scoring on the frozen routing (addendum to ruling (c) 2) --------------------


def _scored(tmp_path, monkeypatch):  # noqa: ANN001, ANN202
    import json  # noqa: PLC0415

    from gems_blanking_v2.detect import recall  # noqa: PLC0415
    from gems_blanking_v2.io.store import GemsStore  # noqa: PLC0415

    store = GemsStore.initialise(tmp_path / "gems")
    pid = "plan_20260930T000000Z_00000001"
    doc = {"plan_id": pid,
           "per_span": [{"span_id": f"{pid}_s1", "condition": "baseline",
                         "chance_per_mark": [0.2, 0.3]}],
           "artifacts": [{"id": f"{pid}_s1#m0", "span_id": f"{pid}_s1", "covered": True},
                         {"id": f"{pid}_s1#m1", "span_id": f"{pid}_s1", "covered": False}]}
    path = store.audit_score_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(recall, "pooled_gate",
                        lambda _s: {"pooled_rounds": [pid], "excluded_rounds": []})
    return store, pid


def test_the_frozen_gate_reads_the_classes_stored_under_that_routing(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from gems_blanking_v2.detect.recall import frozen_filtered_gate  # noqa: PLC0415
    from gems_blanking_v2.emit.routing import write_damage  # noqa: PLC0415

    store, pid = _scored(tmp_path, monkeypatch)
    h = "a" * 64
    write_damage(store, pid, h, {f"{pid}_s1#m0": "target", f"{pid}_s1#m1": "below"}, {})
    g = frozen_filtered_gate(store, h)
    assert g["routing_hash"] == h
    assert g["pooled"]["filtered"]["found"] == 1 and g["pooled"]["raw"]["found"] == 2
    assert g["pooled"]["target_misses_unclassified"] == []


def test_a_round_without_classes_under_the_frozen_routing_is_refused(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from gems_blanking_v2.detect.recall import frozen_filtered_gate  # noqa: PLC0415
    from gems_blanking_v2.emit.routing import write_damage  # noqa: PLC0415

    store, pid = _scored(tmp_path, monkeypatch)
    write_damage(store, pid, "b" * 64, {f"{pid}_s1#m0": "target", f"{pid}_s1#m1": "target"}, {})
    with pytest.raises(FileNotFoundError, match="no damage classes under routing aaaaaaaa"):
        frozen_filtered_gate(store, "a" * 64)
