"""Task 15: mask provenance - which model, thresholds, references and code made a mask.

Masks are regenerated as the model improves, and every downstream analysis must know
which one it used. **A mask whose provenance does not name a model is invalid**, and
that is asserted when the provenance is built and again on write
(:func:`MaskProvenance.validate`), never left to a reader.

The model slot holds task 12A's ``ModelSpec`` - mode, animal, version, corpus hash,
calibrator, trained_at, metrics, n_train_events. 12A's class is not merged yet, so the
slot accepts any object exposing those attributes (a dataclass, or the class itself
once it lands) or a mapping with those keys (:func:`model_spec_record`), and stores a
plain JSON record. When 12A merges, nothing here changes: its instances are accepted
as they are.

Serialisation follows the conventions table: canonical, ASCII-escaped JSON; a missing
value is an ABSENT key, never ``null`` or ``NaN``; on read, absent and ``null`` are the
same and a required field missing raises naming it. Paths are stored relative and POSIX
(cross-platform rule 2) - an absolute calibrator path is refused.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Final

from gems_blanking_v2.types import TrainingMode

__all__ = [
    "MODEL_REQUIRED",
    "MaskProvenance",
    "ProvenanceError",
    "model_spec_record",
]

MODEL_REQUIRED: Final[tuple[str, ...]] = ("mode", "version", "corpus_hash", "calibrator")
"""ModelSpec fields a mask's provenance cannot do without."""
MODEL_OPTIONAL: Final[tuple[str, ...]] = ("animal", "trained_at", "metrics", "n_train_events",
                                          "unvalidated")


class ProvenanceError(ValueError):
    """Provenance that does not identify what made the mask."""


def _get(obj: object, key: str) -> object:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _clean(value: object) -> Any:  # noqa: ANN401 - JSON value
    """JSON-ready: drop None/NaN (absent), datetimes to ISO, paths to POSIX, enums to str."""
    if isinstance(value, TrainingMode):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            msg = "a provenance timestamp must be timezone-aware"
            raise ProvenanceError(msg)
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items() if not _missing(v)}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"non-finite number {value!r} in provenance; leave the key absent instead"
        raise ProvenanceError(msg)
    as_posix = getattr(value, "as_posix", None)
    if callable(as_posix):
        return as_posix()
    return value


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _relative_posix(path: object, what: str) -> str:
    as_posix = getattr(path, "as_posix", None)
    text = as_posix() if callable(as_posix) else str(path)
    if (PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute()
            or PureWindowsPath(text).drive):
        msg = f"{what} must be stored relative to gems_root, got absolute {text!r}"
        raise ProvenanceError(msg)
    if ".." in PurePosixPath(text.replace("\\", "/")).parts:
        msg = f"{what} must not climb out of gems_root: {text!r}"
        raise ProvenanceError(msg)
    return text.replace("\\", "/")


def model_spec_record(spec: object) -> dict[str, Any]:
    """Return a task 12A ``ModelSpec`` (or anything with its fields) as a JSON record.

    Raises :class:`ProvenanceError` naming the field when a required one is absent or
    empty; ``animal`` must be absent for a POOLED model and present otherwise.
    """
    if spec is None:
        msg = "a mask's provenance must name a model (task 12A ModelSpec); got none"
        raise ProvenanceError(msg)
    rec: dict[str, Any] = {}
    for key in (*MODEL_REQUIRED, *MODEL_OPTIONAL):
        v = _get(spec, key)
        if _missing(v) or v == "":
            continue
        rec[key] = _relative_posix(v, "calibrator") if key == "calibrator" else _clean(v)
    lacking = [k for k in MODEL_REQUIRED if k not in rec]
    if lacking:
        msg = f"model provenance lacks required field(s) {lacking}"
        raise ProvenanceError(msg)
    try:
        mode = TrainingMode(rec["mode"])
    except ValueError as exc:
        msg = f"model mode {rec['mode']!r} is not one of {[m.value for m in TrainingMode]}"
        raise ProvenanceError(msg) from exc
    if (mode is TrainingMode.POOLED) == ("animal" in rec):
        msg = ("a POOLED model has no animal; an ADAPTED or PER_ANIMAL model names one "
               f"(mode {mode.value}, animal {rec.get('animal')!r})")
        raise ProvenanceError(msg)
    return rec


@dataclass(frozen=True)
class MaskProvenance:
    """Everything that made one recording's masks.

    ``model`` is :func:`model_spec_record`'s output. ``thresholds`` holds the tolerance
    table and routing thresholds with their sources; ``reference_values`` the whole-file
    reference scalars per ``signal|band``; ``code_commit`` the commit of this package;
    ``generation_sha`` the candidate generator's hash; ``routing_hash`` the frozen
    routing table; ``settling_s`` each consumer chain's measured settling.
    """

    model: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    reference_values: Mapping[str, Any]
    code_commit: str
    generation_sha: str
    routing_hash: str
    created_at: str
    recording: str
    settling_s: Mapping[str, float] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise to the JSON form and validate."""
        # Normalise to the JSON form once, so a provenance equals its own reparse (the
        # round trip is an identity), and a ModelSpec object becomes its record.
        object.__setattr__(self, "model", model_spec_record(self.model))
        for name in ("thresholds", "reference_values", "settling_s", "extra"):
            value = _clean(dict(getattr(self, name)))
            object.__setattr__(self, name, json.loads(json.dumps(value, allow_nan=False)))
        self.validate()

    def validate(self) -> None:
        """Raise unless the provenance names a model and every required part is present."""
        model_spec_record(self.model)  # re-checks the model slot
        for name in ("code_commit", "generation_sha", "routing_hash", "created_at",
                     "recording"):
            if not getattr(self, name):
                msg = f"mask provenance lacks {name}"
                raise ProvenanceError(msg)
        for name in ("thresholds", "reference_values"):
            if not getattr(self, name):
                msg = f"mask provenance lacks {name}"
                raise ProvenanceError(msg)
        try:
            when = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            msg = f"created_at {self.created_at!r} is not ISO-8601"
            raise ProvenanceError(msg) from exc
        if when.tzinfo is None:
            msg = "created_at must be timezone-aware"
            raise ProvenanceError(msg)

    def to_record(self) -> dict[str, Any]:
        """Return the JSON record: absent keys for missing values."""
        self.validate()
        rec = {
            "model": _clean(dict(self.model)), "thresholds": _clean(dict(self.thresholds)),
            "reference_values": _clean(dict(self.reference_values)),
            "code_commit": self.code_commit, "generation_sha": self.generation_sha,
            "routing_hash": self.routing_hash, "created_at": self.created_at,
            "recording": self.recording, "settling_s": _clean(dict(self.settling_s)),
        }
        if self.extra:
            rec["extra"] = _clean(dict(self.extra))
        return {k: v for k, v in rec.items() if v not in ({}, None)}

    def to_json(self) -> str:
        """Canonical, ASCII-escaped JSON (sorted keys, fixed separators)."""
        return json.dumps(self.to_record(), sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> MaskProvenance:
        """Parse :meth:`to_json`; absent and null are the same; required missing raises."""
        doc = json.loads(text)
        required = ("model", "thresholds", "reference_values", "code_commit",
                    "generation_sha", "routing_hash", "created_at", "recording")
        for key in required:
            if doc.get(key) is None:
                msg = f"mask provenance: required field {key!r} is absent"
                raise ProvenanceError(msg)
        return cls(model=doc["model"], thresholds=doc["thresholds"],
                   reference_values=doc["reference_values"], code_commit=doc["code_commit"],
                   generation_sha=doc["generation_sha"], routing_hash=doc["routing_hash"],
                   created_at=doc["created_at"], recording=doc["recording"],
                   settling_s=doc.get("settling_s") or {}, extra=doc.get("extra") or {})
