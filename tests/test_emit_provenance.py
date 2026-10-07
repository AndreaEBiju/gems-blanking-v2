"""Task 15: mask provenance - names a model, round-trips exactly, refuses what it cannot name."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError, model_spec_record
from gems_blanking_v2.types import TrainingMode
from hypothesis import given, settings
from hypothesis import strategies as st

MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/calibrator.json",
         "trained_at": datetime(2026, 10, 8, tzinfo=UTC), "n_train_events": 1200,
         "metrics": {"LOAO": {"f1": 0.88}}}


def _prov(**kw: object) -> MaskProvenance:
    base: dict[str, Any] = {"model": MODEL, "thresholds": {"tolerances": {"spikes": 4.0},
                                                           "source": "synthetic"},
                            "reference_values": {"L_T|300-3000": [1.2, 0.3]},
                            "code_commit": "c0ffee", "generation_sha": "0133349b3ebeff80",
                            "routing_hash": "64c2e1ea",
                            "created_at": "2026-10-08T05:00:00+00:00", "recording": "rec1",
                            "settling_s": {"spikes": 0.00512}}
    base.update(kw)
    return MaskProvenance(**base)


def test_provenance_without_a_model_is_refused() -> None:
    with pytest.raises(ProvenanceError, match="model"):
        _prov(model=None)
    with pytest.raises(ProvenanceError, match="corpus_hash"):
        _prov(model={k: v for k, v in MODEL.items() if k != "corpus_hash"})


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
