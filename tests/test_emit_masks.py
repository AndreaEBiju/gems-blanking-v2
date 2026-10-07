"""Task 15: per-consumer masks, the MATLAB file, provenance, QC gates."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit import qc
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError, model_spec_record
from gems_blanking_v2.extent import tolerance as tl
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs, find_zero_runs
from gems_blanking_v2.types import Candidate, Event, TrainingMode
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.io import loadmat

from tests.conftest import make_band_z

FS = 24414.0625
DUR_S = 60.0
N_FRAMES = int(DUR_S / 0.01)
TOL = tl.ToleranceTable({"spikes": 4.0, "velocity": 4.0, "mmc": 3.0, "slow_wave": 3.0,
                         "breathing": 3.0}, source="synthetic test table")
SIGNALS = {"spikes": ("L_T",), "slow_wave": ("stomach_ref",),
           "velocity": ("L_V1", "L_V3")}
MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/calibrator.json",
         "trained_at": datetime(2026, 10, 8, tzinfo=UTC), "n_train_events": 1200,
         "metrics": {"LOAO": {"f1": 0.88}}}

def _settle(consumer: str) -> tl.ConsumerSettling:
    s = tl.consumer_settling(consumer, FS)
    assert s is not None
    return s



def _prov(**kw: object) -> MaskProvenance:
    base: dict[str, Any] = {"model": MODEL, "thresholds": {"tolerances": {"spikes": 4.0},
                                                 "source": "synthetic"},
                  "reference_values": {"L_T|300-3000": [1.2, 0.3]}, "code_commit": "c0ffee",
                  "generation_sha": "0133349b3ebeff80", "routing_hash": "64c2e1ea",
                  "created_at": "2026-10-08T05:00:00+00:00", "recording": "rec1",
                  "settling_s": {"spikes": 0.00512}}
    base.update(kw)
    return MaskProvenance(**base)


def test_emit_is_outside_the_generation_hash() -> None:
    mods = chain.generation_modules()
    for m in ("masks", "qc", "provenance"):
        assert f"gems_blanking_v2.emit.{m}" not in mods


def _eng_only_masks() -> dict[Any, Any]:
    """Build masks for an event over the ENG tolerance only (no slow_wave extent)."""
    ev = Event(Candidate(19.5, 21.0, ("L_T",), ("300-3000",), 9.0, "electrical"),
               "motion", float("nan"), "human")
    z = {("L_T", "300-3000"): make_band_z("300-3000", DUR_S, bumps=((20.0, 20.2, 12.0, "L_T"),),
                                          signal="L_T").z_max,
         ("stomach_ref", "0-2"): make_band_z("0-2", DUR_S, signal="stomach_ref").z_max}
    exts = []
    decisions = []
    for consumer, sig in (("spikes", "L_T"), ("slow_wave", "stomach_ref")):
        e = tl.compute_extent(ev, z, consumer, signal=sig, tolerances=TOL, fs=FS, z_t0_s=0.0)
        if e is not None:
            exts.append(("ev1", e))
            decisions.append(RouteDecision("ev1", consumer, "reject", "in band"))
    spans = mk.spans_from_routing(exts, decisions)
    return mk.build_masks(SIGNALS, spans, N_FRAMES)


def test_masks_are_not_merged_across_consumers() -> None:
    masks = _eng_only_masks()
    spikes = masks[("spikes", "L_T", "300-3000")].invalid
    slow = masks[("slow_wave", "stomach_ref", "0-2")].invalid
    assert spikes.any() and not slow.any()
    assert not np.array_equal(spikes, slow)


def test_velocity_is_absent_unless_task_18_is_present() -> None:
    masks = _eng_only_masks()
    assert not any(k[0] == "velocity" for k in masks)
    with_v = mk.build_masks(SIGNALS, [mk.MaskSpan("velocity", "L_V1", 1.0, 2.0, "x")],
                            N_FRAMES, include_velocity=True)
    assert ("velocity", "L_V1", "300-3000") in with_v


def test_velocity_mask_is_the_intersection_of_v1_and_v3_validity() -> None:
    spans = [mk.MaskSpan("velocity", "L_V1", 1.0, 2.0, "x"),
             mk.MaskSpan("velocity", "L_V3", 5.0, 6.0, "x")]
    m = mk.build_masks(SIGNALS, spans, N_FRAMES, include_velocity=True)
    v1, v3 = m[("velocity", "L_V1", "300-3000")], m[("velocity", "L_V3", "300-3000")]
    v = mk.velocity_mask(v1, v3)
    assert np.array_equal(~v.invalid, ~v1.invalid & ~v3.invalid)
    with pytest.raises(ValueError, match="velocity"):
        mk.velocity_mask(m[("velocity", "L_V1", "300-3000")],
                         _eng_only_masks()[("spikes", "L_T", "300-3000")])


def test_only_masking_routes_mask_and_a_missing_decision_raises() -> None:
    ext = tl.Extent("spikes", "L_T", "300-3000", 1.0, 2.0, 1.0, 2.0, 0.0, 0.025)
    keep = mk.spans_from_routing([("e", ext)], [RouteDecision("e", "spikes", "correct", "x")])
    assert keep == []
    with pytest.raises(KeyError, match="no routing decision"):
        mk.spans_from_routing([("e", ext)], [])


def test_distrusted_spans_join_only_that_cuffs_spike_mask() -> None:
    entry = {"spike": {"L": {"route": "multi", "distrusted_spans": [[0.0, 60.0]]},
                       "R": {"route": "multi"}}, "hr": {"none": "x"}, "stomach_ref": {}}
    spans = mk.distrusted_spike_spans(entry, region_start_s=100.0)
    assert [(s.consumer, s.signal, s.start_s, s.stop_s) for s in spans] == [
        ("spikes", "L_T", 100.0, 160.0)]
    assert {s.consumer for s in mk.line_noise_spike_spans("R", [(0.0, 60.0)])} == {"spikes"}


def test_apply_mask_writes_nan_tapers_without_zeros_and_leaves_the_input() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(0, 10, int(DUR_S * FS))
    x.flags.writeable = False  # a loader's read-only view (invariant 17)
    invalid = mk.mask_frames([(10.0, 10.5)], N_FRAMES)
    y = mk.apply_mask(x, FS, invalid)
    i0, i1 = int(np.ceil(10.0 * FS)), int(np.ceil(10.5 * FS))
    assert np.isnan(y[i0:i1]).all() and np.isfinite(y[:i0]).all() and np.isfinite(y[i1:]).all()
    n = int(round(mk.TAPER_S * FS))
    w = y[i0 - n:i0] / x[i0 - n:i0]
    assert (w > 0).all() and (w < 1).all() and np.all(np.diff(w) < 0)
    assert np.array_equal(y[: i0 - n], x[: i0 - n])
    assert find_zero_runs(y) == []


def test_apply_mask_refuses_to_emit_a_zero_run() -> None:
    x = np.ones(int(FS))
    x[100:200] = 0.0  # zeros in the input are not NaN and must not be emitted as a run
    with pytest.raises(ValueError, match="zero"):
        mk.apply_mask(x, FS, np.zeros(100, dtype=bool))


def test_the_mask_file_has_no_zero_runs_and_one_span_set_per_consumer(tmp_path: Path) -> None:
    masks = _eng_only_masks()
    path = mk.write_mask_file(tmp_path / "rec1_masks.mat", masks, _prov(), fs=FS,
                              n_samples=int(DUR_S * FS),
                              events=[{"start": 19.5, "stop": 21.0, "judgement": "motion"}])
    m = loadmat(path)
    for name, value in m.items():
        if name.startswith("__") or value.dtype.kind in "U":
            continue
        assert_no_zero_runs(np.asarray(value, dtype=np.float64).ravel(), what=name)
    assert {"blank_spikes_L_T", "blank_slow_wave_stomach_ref"} <= set(m)
    assert m["blank_slow_wave_stomach_ref"].size == 0
    spans = m["blank_spikes_L_T"]
    pad = _settle("spikes").total_s
    a = np.floor((20.0 - pad) / 0.01) * 0.01
    assert spans[0, 0] == round(a * FS) + 1  # 1-based inclusive (invariant 15)
    prov = MaskProvenance.from_json(str(m["provenance_json"][0]))
    assert prov.model["mode"] == "pooled"


def test_a_mask_without_a_model_is_refused_on_write(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="model"):
        mk.write_mask_file(tmp_path / "x.mat", _eng_only_masks(), None, fs=FS,
                           n_samples=int(DUR_S * FS))
    with pytest.raises(ProvenanceError, match="model"):
        _prov(model=None)
    with pytest.raises(ProvenanceError, match="corpus_hash"):
        _prov(model={k: v for k, v in MODEL.items() if k != "corpus_hash"})
    assert not list(tmp_path.iterdir())


def test_provenance_names_the_model_rules() -> None:
    with pytest.raises(ProvenanceError, match="POOLED"):
        model_spec_record({**MODEL, "animal": "J"})
    with pytest.raises(ProvenanceError, match="POOLED"):
        model_spec_record({**MODEL, "mode": TrainingMode.PER_ANIMAL})
    assert model_spec_record({**MODEL, "mode": "per_animal", "animal": "J"})["animal"] == "J"
    for bad in ("C:/models/cal.json", "/abs/cal.json", "../cal.json", "G:\\x\\cal.json"):
        with pytest.raises(ProvenanceError, match="calibrator"):
            model_spec_record({**MODEL, "calibrator": bad})


def test_provenance_accepts_a_model_spec_object() -> None:
    """Task 12A's ModelSpec is not merged; any object with its attributes is accepted."""

    class FakeSpec:
        mode = TrainingMode.ADAPTED
        animal = "J"
        version = "1"
        corpus_hash = "cd" * 16
        calibrator = Path("models/a/cal.json")
        trained_at = datetime(2026, 10, 8, tzinfo=UTC)
        metrics: ClassVar = {"LOAO": {"f1": 0.9}}
        n_train_events = 5

    p = _prov(model=FakeSpec())
    assert p.model["mode"] == "adapted" and p.model["calibrator"] == "models/a/cal.json"


def _has_null(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, dict):
        return any(_has_null(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_null(x) for x in v)
    return False


json_scalars = st.one_of(st.integers(-10**6, 10**6), st.text(max_size=8),
                         st.floats(allow_nan=False, allow_infinity=False, width=32),
                         st.booleans())


@settings(max_examples=60, deadline=None)
@given(thresholds=st.dictionaries(st.text(min_size=1, max_size=6), json_scalars, min_size=1,
                                  max_size=4),
       refs=st.dictionaries(st.text(min_size=1, max_size=6),
                            st.lists(st.floats(allow_nan=False, allow_infinity=False,
                                               width=32), max_size=3),
                            min_size=1, max_size=4),
       commit=st.text(alphabet="0123456789abcdef", min_size=7, max_size=40))
def test_provenance_round_trips_exactly(
    thresholds: dict[str, Any], refs: dict[str, Any], commit: str
) -> None:
    p = _prov(thresholds=thresholds, reference_values=refs, code_commit=commit)
    text = p.to_json()
    assert text.isascii()
    assert not _has_null(json.loads(text))  # absent, never null (allow_nan=False bars NaN)
    back = MaskProvenance.from_json(text)
    assert back == p and back.to_json() == text
    assert json.loads(text)["model"]["corpus_hash"] == MODEL["corpus_hash"]


def test_provenance_absent_and_null_read_the_same_and_required_raises() -> None:
    doc = json.loads(_prov().to_json())
    doc["extra"] = None
    assert MaskProvenance.from_json(json.dumps(doc)) == _prov()
    del doc["routing_hash"]
    with pytest.raises(ProvenanceError, match="routing_hash"):
        MaskProvenance.from_json(json.dumps(doc))


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------


def test_the_retention_gate_fires_on_an_over_masked_recording() -> None:
    over = mk.build_masks(SIGNALS, [mk.MaskSpan("spikes", "L_T", 0.0, 40.0, "x")], N_FRAMES)
    v = qc.retention_gate(over, min_retention=0.5)
    assert v.flagged and list(v.below) == ["spikes|L_T|300-3000"]
    assert not qc.retention_gate(_eng_only_masks(), min_retention=0.5).flagged
    with pytest.raises(ValueError, match="min_retention"):
        qc.retention_gate(over, min_retention=0.0)


def _masked(frac: float) -> dict[Any, Any]:
    return mk.build_masks({"spikes": ("L_T",)},
                          [mk.MaskSpan("spikes", "L_T", 0.0, frac * DUR_S, "x")], N_FRAMES)


def test_a_blank_over_20_percent_is_held() -> None:
    h = qc.blank_fraction_hold(_masked(0.25), {"spikes|L_T|300-3000": 0.2})
    assert h.held and any("20%" in r for r in h.reasons)


def test_a_blank_over_3x_the_animal_median_is_held() -> None:
    h = qc.blank_fraction_hold(_masked(0.10), {"spikes|L_T|300-3000": 0.02},
                               hum_features={"line_ratio_max": 0.4})
    assert h.held and any("3 x" in r for r in h.reasons)
    assert h.hum_features == {"line_ratio_max": 0.4}
    assert not qc.blank_fraction_hold(_masked(0.05), {"spikes|L_T|300-3000": 0.02}).held


def test_an_unknown_animal_median_applies_the_20_percent_rule_alone() -> None:
    assert not qc.blank_fraction_hold(_masked(0.10), {"spikes|L_T|300-3000": None}).held
    h = qc.blank_fraction_hold(_masked(0.30), {})
    assert h.held and any("median unknown" in r for r in h.reasons)


def test_the_qc_report_leaves_missing_quantities_absent() -> None:
    r = qc.QcReport(recording="rec1", candidate_count=12,
                    blank_fraction_by_band=qc.blank_fraction_by_band(_eng_only_masks()),
                    retention=qc.retention_by_key(_eng_only_masks()), retention_flagged=False,
                    held=False, rpeak_gap_fraction=float("nan"), best_hr_channel=None)
    rec = json.loads(r.to_json())
    for absent in ("rpeak_gap_fraction", "best_hr_channel", "low_confidence_velocity_windows"):
        assert absent not in rec
    assert rec["candidate_count"] == 12 and "spikes|L_T|300-3000" in rec["retention"]


def test_the_longitudinal_table_is_one_animal_append_only(tmp_path: Path) -> None:
    row = qc.LongitudinalRow("J", "s1", "2026-10-01T03:00:00+00:00", {"L": 0.5}, {"L": 0.5},
                             {"RVN1": 18.0}, {"L_T": 2.3})
    path = tmp_path / "J_longitudinal.jsonl"
    qc.append_longitudinal(path, row)
    qc.append_longitudinal(path, row)
    raw = path.read_bytes()
    assert raw.count(b"\n") == 2 and b"\r" not in raw
    with pytest.raises(ValueError, match="another animal"):
        qc.append_longitudinal(path, qc.LongitudinalRow("H", "s1", "x", {}, {}, {}, {}))


def test_the_event_table_carries_routes_judgement_and_model() -> None:
    ev = Event(Candidate(1.0, 2.0, ("L_T",), ("300-3000", "100-300"), 9.0, "electrical"),
               "motion", float("nan"), "human")
    rows = mk.event_rows({"e": ev}, [RouteDecision("e", "spikes", "reject", "x"),
                                     RouteDecision("e", "slow_wave", "correct", "y")], _prov())
    (row,) = rows
    assert row["routes"] == {"spikes": "reject", "slow_wave": "correct"}
    assert row["bands"] == ["300-3000", "100-300"] and "p_motion" not in row
    assert row["model_version"] == "0.3.0"
    json.dumps(rows, allow_nan=False)
