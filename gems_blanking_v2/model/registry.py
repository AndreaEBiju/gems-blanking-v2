"""Task 12A: the model registry and **user-selected** inference.

The registry presents the options honestly and records the choice; it never decides.

* **No automatic selection anywhere.** :func:`run_inference` has no default for the
  model (a test inspects its signature), there is no fallback chain, and the
  per-(mode, animal) promotion pointer is a *label* that nothing at inference time reads.
* **Animals are cohort-qualified.** ``ModelSpec.animal`` is an animal key
  ``"<cohort>:<letter>"`` (:func:`~gems_blanking_v2.model.labels.animal_key`): the old
  cohort's JEL and the new cohort's J share the letter "J" and are different rats. A
  :class:`~gems_blanking_v2.types.Recording` carries only the letter, so every inference
  entry point takes the recording's ``cohort`` explicitly (keyword-only, no default).
* :func:`list_applicable` excludes models that cannot legally apply - a ``PER_ANIMAL``
  or ``ADAPTED`` model of another animal key; :func:`run_inference` checks the animal
  key independently and then that the whole spec is listed, and fails loudly otherwise.
* Every :class:`ModelSpec` carries the metrics it shipped on **keyed by protocol**
  (``{"LOAO": {"f1": 0.88}, "LORO": {...}}``), so a picker can show "F1 0.88 (LOAO)"
  beside "F1 0.94 (LORO)" without inviting the invalid comparison (invariant 12). A
  spec with no held-out metrics is ``unvalidated``: still selectable, flagged in
  provenance.
* **Append-only.** The registry is the 00A log (:mod:`gems_blanking_v2.io.registry_log`);
  a model's spec, booster and calibrator are written once into ``models/<model_id>/``
  and never rewritten. Metric values cross into the log flattened as
  ``"<protocol>.<metric>"``.

Provenance (task 15): :func:`inference_provenance` embeds the full spec; a mask whose
provenance does not name a model is invalid (:func:`require_model_in_provenance`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.io.registry_log import (
    RegistryAction,
    RegistryEvent,
    append_event,
    read_events,
    replay,
)
from gems_blanking_v2.io.store import GemsStore, atomic_write_text, utc_stamp
from gems_blanking_v2.model.evaluate import Calibrator
from gems_blanking_v2.model.labels import animal_key, is_test_animal
from gems_blanking_v2.model.provenance import PROVENANCE_NAME, write_provenance
from gems_blanking_v2.model.train import feature_columns, predict_raw
from gems_blanking_v2.types import Recording, TrainingMode

__all__ = [
    "ADAPT_LABEL_SET",
    "HELD_OUT_PROTOCOLS",
    "InferenceResult",
    "ModelSpec",
    "NotApplicableError",
    "Registry",
    "adaptation_separation",
    "calibrator_relpath",
    "inference_provenance",
    "list_applicable",
    "model_content_id",
    "require_model_in_provenance",
    "resolve_batch",
    "run_inference",
]

F64 = npt.NDArray[np.float64]

HELD_OUT_PROTOCOLS: Final[frozenset[str]] = frozenset({"LOAO", "LOAO_ADAPT", "LORO"})
"""Protocols whose metrics count as held-out. A spec with none is ``unvalidated``."""

SPEC_NAME: Final = "spec.json"
BOOSTER_NAME: Final = "model.txt"
CALIBRATOR_NAME: Final = "calibrator.json"
_MODEL_ID_RE: Final = re.compile(r"^[0-9a-f]{32}$")
_ANIMAL_KEY_RE: Final = re.compile(r"^(old|new):([A-Z])$")
COHORTS: Final[frozenset[str]] = frozenset({"old", "new"})


def _check_cohort(cohort: str) -> str:
    if cohort not in COHORTS:
        msg = f"cohort must be one of {sorted(COHORTS)}, got {cohort!r}"
        raise ValueError(msg)
    return cohort


def model_content_id(booster: lgb.Booster, calibrator: Calibrator, *, mode: TrainingMode,
                     animal: str | None, corpus_hash: str,
                     w_adapt: float | None = None) -> str:
    """Content identity of a model (32 hex): booster, calibrator, mode, animal, corpus.

    The one construction site of a model id (invariant 33). Immutable and never reused.
    An ADAPTED model also hashes its ``w_adapt``: the corpus hash names the rows and labels
    but not their weights, so two weights whose boosters happen to coincide (a corpus too
    small to split) would otherwise claim one id with two different specs. Without a
    weight the id is what it always was.
    """
    h = hashlib.sha256()
    parts = [booster.model_to_string(), calibrator.to_json(), str(TrainingMode(mode)),
             animal or "", corpus_hash]
    if w_adapt is not None:
        parts.append(f"w_adapt={float(w_adapt)!r}")
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def calibrator_relpath(model_id: str) -> Path:
    """Where a model's calibrator lives, relative to gems_root: ``models/<id>/calibrator.json``.

    The one construction site of that path; :class:`ModelSpec` validates against it.
    """
    if not _MODEL_ID_RE.match(model_id):
        msg = f"model id must be 32 lowercase hex, got {model_id!r}"
        raise ValueError(msg)
    return Path("models") / model_id / CALIBRATOR_NAME


class NotApplicableError(ValueError):
    """The chosen model may not be applied to this recording."""


def _canon(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


@dataclass(frozen=True)
class ModelSpec:
    """One trained model, as the user sees it in the picker (task 12A).

    ``animal`` is ``None`` for ``POOLED`` and otherwise an animal key
    ``"<cohort>:<letter>"`` (validated). ``calibrator`` is a
    POSIX path **relative to gems_root** (cross-platform rule 2). ``metrics`` maps a
    protocol to its metric values; non-finite values are refused (JSON cannot carry
    them - a metric that could not be computed is absent).
    """

    mode: TrainingMode
    animal: str | None
    version: str
    corpus_hash: str
    calibrator: Path
    trained_at: datetime
    metrics: Mapping[str, Mapping[str, float]]
    n_train_events: int
    never_scores_evaluation_spans: bool = False
    """True exactly for an ADAPTED model of a prospective test animal (new-cohort I, J,
    K; ruling 2026-10-08 (b) item 2): trained on adaptation labels kept apart from the
    evaluation labels, and refused by :func:`run_inference` / :func:`resolve_batch` on
    evaluation spans. Serialised only when True (absent otherwise)."""

    def __post_init__(self) -> None:
        """Validate the mode/animal pairing, the path, the clock and the metrics."""
        flag: object = self.never_scores_evaluation_spans
        if not isinstance(flag, bool):
            msg = f"never_scores_evaluation_spans must be a bool, got {flag!r}"
            raise TypeError(msg)
        mode = TrainingMode(self.mode)
        if (mode is TrainingMode.POOLED) != (self.animal is None):
            msg = (f"{mode} model with animal={self.animal!r}: POOLED has no animal, "
                   "ADAPTED and PER_ANIMAL must name one")
            raise ValueError(msg)
        key = None if self.animal is None else _ANIMAL_KEY_RE.match(self.animal)
        if self.animal is not None and key is None:
            msg = (f"animal must be an animal key '<cohort>:<A-Z>' with cohort old or new, "
                   f"got {self.animal!r} (a bare letter names two rats across cohorts; the "
                   "old '?' token is never a per-animal or adapted model)")
            raise ValueError(msg)
        test = key is not None and is_test_animal(key.group(1), key.group(2))
        if test and not (mode is TrainingMode.ADAPTED and self.never_scores_evaluation_spans):
            msg = (f"no {mode} model can exist for {self.animal}: new-cohort I/J/K are the "
                   "prospective test set (R1). Only an ADAPTED model trained on adaptation "
                   "labels kept apart from the evaluation labels may, flagged "
                   "never_scores_evaluation_spans (ruling 2026-10-08 (b) item 2)")
            raise ValueError(msg)
        if self.never_scores_evaluation_spans and not test:
            msg = ("never_scores_evaluation_spans marks an ADAPTED model of new-cohort I/J/K "
                   f"only; {mode} for {self.animal} cannot carry it")
            raise ValueError(msg)
        cal = PurePosixPath(Path(self.calibrator).as_posix())
        if cal.is_absolute() or Path(self.calibrator).is_absolute() or ".." in cal.parts:
            msg = f"calibrator path must be relative to gems_root, got {self.calibrator}"
            raise ValueError(msg)
        parts = cal.parts
        if len(parts) != 3 or parts[0] != "models" or parts[2] != CALIBRATOR_NAME or (  # noqa: PLR2004
                not _MODEL_ID_RE.match(parts[1])):
            msg = f"calibrator must be models/<32-hex>/{CALIBRATOR_NAME}, got {cal.as_posix()}"
            raise ValueError(msg)
        if self.trained_at.tzinfo is None:
            msg = "trained_at must be timezone-aware (UTC)"
            raise ValueError(msg)
        for proto, vals in self.metrics.items():
            for k, v in vals.items():
                if not math.isfinite(float(v)):
                    msg = f"metric {proto}.{k} is {v!r}; omit it instead"
                    raise ValueError(msg)
        if self.n_train_events < 0:
            msg = "n_train_events must be >= 0"
            raise ValueError(msg)

    @property
    def unvalidated(self) -> bool:
        """True when no held-out protocol has metrics. Selectable, but flagged."""
        return not any(p in HELD_OUT_PROTOCOLS and len(v) > 0 for p, v in self.metrics.items())

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able form. ``animal`` is absent for a pooled model, never ``null``."""
        out: dict[str, Any] = {
            "mode": str(TrainingMode(self.mode)), "version": self.version,
            "corpus_hash": self.corpus_hash,
            "calibrator": PurePosixPath(Path(self.calibrator).as_posix()).as_posix(),
            "trained_at": self.trained_at.astimezone(UTC).isoformat(),
            "metrics": {p: {k: float(v) for k, v in m.items()} for p, m in self.metrics.items()},
            "n_train_events": int(self.n_train_events),
        }
        if self.animal is not None:
            out["animal"] = self.animal
        if self.never_scores_evaluation_spans:
            out["never_scores_evaluation_spans"] = True
        return out

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, ASCII, no NaN."""
        return _canon(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelSpec:
        """Inverse of :meth:`to_dict`. Absent and ``null`` ``animal`` read alike."""
        for key in ("mode", "version", "corpus_hash", "calibrator", "trained_at", "metrics",
                    "n_train_events"):
            if raw.get(key) is None:
                msg = f"model spec is missing required field {key!r}"
                raise ValueError(msg)
        return cls(mode=TrainingMode(raw["mode"]), animal=raw.get("animal"),
                   version=str(raw["version"]), corpus_hash=str(raw["corpus_hash"]),
                   calibrator=Path(PurePosixPath(str(raw["calibrator"]))),
                   trained_at=datetime.fromisoformat(str(raw["trained_at"])),
                   metrics={str(p): {str(k): float(v) for k, v in m.items()}
                            for p, m in dict(raw["metrics"]).items()},
                   n_train_events=int(raw["n_train_events"]),
                   never_scores_evaluation_spans=raw.get("never_scores_evaluation_spans")
                   or False)

    @classmethod
    def from_json(cls, text: str) -> ModelSpec:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    @property
    def model_id(self) -> str:
        """The model's content id: the directory its calibrator lives in."""
        return PurePosixPath(Path(self.calibrator).as_posix()).parts[1]

    def flat_metrics(self) -> dict[str, float]:
        """``{"<protocol>.<metric>": value}`` for the registry log."""
        return {f"{p}.{k}": float(v) for p, m in self.metrics.items() for k, v in m.items()}


@dataclass(frozen=True)
class Registry:
    """The model registry on a :class:`~gems_blanking_v2.io.store.GemsStore`."""

    store: GemsStore

    def register(self, spec: ModelSpec, booster: lgb.Booster, calibrator: Calibrator, *,
                 user: str, provenance: Mapping[str, Any], corpus_id: str = "") -> str:
        """Write the model once and append a ``trained`` event. Returns the model id.

        The spec's model id must be the content id of ``booster`` and ``calibrator``
        (:func:`model_content_id`), so a spec cannot point at a different model's files.
        ``provenance`` (:func:`~gems_blanking_v2.model.provenance.build_provenance`) is
        required and must carry this exact spec; it is written as ``provenance.json``
        beside the model (task 12 acceptance 6). Registering never promotes. Refuses to
        overwrite an existing model directory with different content - models are
        immutable.
        """
        if provenance.get("model") != spec.to_dict() or provenance.get("model_id") != (
                spec.model_id):
            msg = "provenance does not carry this model's spec; build it from the spec"
            raise ValueError(msg)
        if spec.never_scores_evaluation_spans:
            sep = provenance.get("adaptation_separation")
            if not isinstance(sep, Mapping) or sep.get("checked") is not True or sep.get(
                    "animal") != spec.animal:
                msg = (f"an ADAPTED model of test animal {spec.animal} is registered only with "
                       "provenance['adaptation_separation'] from adaptation_separation() for "
                       "that animal (ruling 2026-10-08 (b) item 2)")
                raise ValueError(msg)
        mid = spec.model_id
        w = provenance.get("w_adapt")
        want = model_content_id(booster, calibrator, mode=TrainingMode(spec.mode),
                                animal=spec.animal, corpus_hash=spec.corpus_hash,
                                w_adapt=None if w is None else float(w))
        if mid != want:
            msg = f"spec names model {mid} but its content id is {want}"
            raise ValueError(msg)
        d = self.store.model_dir(mid)
        cal_path = d / CALIBRATOR_NAME
        spec_path = d / SPEC_NAME
        if spec_path.exists():
            if spec_path.read_text(encoding="utf-8").strip() != spec.to_json():
                msg = f"model {mid} exists with a different spec; models are immutable"
                raise ValueError(msg)
        else:
            d.mkdir(parents=True, exist_ok=True)
            atomic_write_text(d / BOOSTER_NAME, booster.model_to_string())
            atomic_write_text(cal_path, calibrator.to_json() + "\n")
            atomic_write_text(spec_path, spec.to_json() + "\n")
        write_provenance(d, provenance)
        append_event(self.store, RegistryEvent(
            ts=utc_stamp(spec.trained_at), user=user, action=RegistryAction.TRAINED,
            model_id=mid, mode=str(TrainingMode(spec.mode)), animal=spec.animal or "",
            corpus_id=corpus_id, metrics=spec.flat_metrics()))
        return mid

    def promote(self, model_id: str, *, user: str) -> None:
        """Append a ``promoted`` event: a per-(mode, animal) *label*, never an auto-selector."""
        self.spec(model_id)
        append_event(self.store, RegistryEvent(ts=utc_stamp(), user=user,
                                               action=RegistryAction.PROMOTED,
                                               model_id=model_id))

    def spec(self, model_id: str) -> ModelSpec:
        """Return the stored spec of one model."""
        p = self.store.model_dir(model_id) / SPEC_NAME
        if not p.is_file():
            msg = f"no model {model_id} in the registry"
            raise KeyError(msg)
        return ModelSpec.from_json(p.read_text(encoding="utf-8"))

    def specs(self) -> list[ModelSpec]:
        """Every trained, non-retired model, in a deterministic order."""
        state = replay(read_events(self.store))
        out = []
        for mid, st in sorted(state.items()):
            if st.status is RegistryAction.RETIRED:
                continue
            if (self.store.model_dir(mid) / SPEC_NAME).is_file():
                out.append(self.spec(mid))
        return out

    def promoted_label(self, mode: TrainingMode, animal: str | None) -> str | None:
        """Return the model id last promoted for (mode, animal), for display only."""
        state = replay(read_events(self.store))
        best: tuple[str, str] | None = None
        for mid, st in state.items():
            if not st.is_promoted or not (self.store.model_dir(mid) / SPEC_NAME).is_file():
                continue
            s = self.spec(mid)
            if TrainingMode(s.mode) is mode and s.animal == animal:
                ev = [e.ts for e in read_events(self.store)
                      if e.model_id == mid and e.action is RegistryAction.PROMOTED]
                ts = max(ev) if ev else ""
                if best is None or ts > best[0]:
                    best = (ts, mid)
        return None if best is None else best[1]

    def provenance_path(self, spec: ModelSpec) -> Path:
        """Where a registered model's ``provenance.json`` lives."""
        return self.store.model_dir(spec.model_id) / PROVENANCE_NAME

    def booster(self, spec: ModelSpec) -> lgb.Booster:
        """Load the booster of a registered spec."""
        return lgb.Booster(model_file=str(self.store.model_dir(spec.model_id) / BOOSTER_NAME))

    def calibrator(self, spec: ModelSpec) -> Calibrator:
        """Load the calibrator of a registered spec."""
        p = self.store.abspath(PurePosixPath(Path(spec.calibrator).as_posix()).as_posix())
        return Calibrator.from_json(p.read_text(encoding="utf-8"))


ADAPT_LABEL_SET: Final = "adapt"
"""``label_set`` of a prospective test animal's adaptation labels: never ``test`` (the
evaluation labels), never ``train`` (R1: no I/J/K label enters training of the models
that are evaluated)."""


def adaptation_separation(rows: pd.DataFrame, *, animal: str,
                          evaluation_spans: Mapping[str, Sequence[tuple[float, float]]]
                          ) -> dict[str, Any]:
    """Check a test animal's adaptation labels against its evaluation labels; return evidence.

    Ruling 2026-10-08 (b) item 2 / R1: an ADAPTED model of new-cohort I, J or K may be
    registered only if trained on adaptation labels kept separate from the evaluation
    labels. ``rows`` are that animal's training rows (``recording``, ``start_s``,
    ``stop_s``, ``label_set``); ``evaluation_spans`` maps each recording to its evaluation
    (audit) spans in seconds. Raises unless every row has ``label_set == "adapt"`` and
    none overlaps an evaluation span in time; returns the record that
    :meth:`Registry.register` requires in ``provenance["adaptation_separation"]``.
    """
    m = _ANIMAL_KEY_RE.match(animal)
    if m is None or not is_test_animal(m.group(1), m.group(2)):
        msg = f"adaptation_separation is for new-cohort I/J/K animal keys, got {animal!r}"
        raise ValueError(msg)
    if rows.empty:
        msg = f"no adaptation rows for {animal}"
        raise ValueError(msg)
    bad_set = sorted(set(rows["label_set"].astype(str)) - {ADAPT_LABEL_SET})
    if bad_set:
        msg = (f"{animal}: adaptation rows carry label_set {bad_set}; only "
               f"{ADAPT_LABEL_SET!r} labels may adapt a test animal")
        raise ValueError(msg)
    overlaps = []
    for r, a, b in zip(rows["recording"].astype(str), rows["start_s"], rows["stop_s"],
                       strict=True):
        for s0, s1 in evaluation_spans.get(r, ()):
            if float(a) < float(s1) and float(s0) < float(b):
                overlaps.append((r, float(a), float(b)))
                break
    if overlaps:
        msg = (f"{animal}: {len(overlaps)} adaptation row(s) overlap an evaluation span, e.g. "
               f"{overlaps[:3]}; adaptation and evaluation labels never overlap in time (R1)")
        raise ValueError(msg)
    return {"checked": True, "animal": animal, "rule": "ruling 2026-10-08 (b) item 2; R1",
            "n_adapt_rows": len(rows), "n_recordings": int(rows["recording"].nunique()),
            "n_evaluation_spans": int(sum(len(v) for v in evaluation_spans.values()))}


def _applies(spec: ModelSpec, key: str) -> bool:
    return TrainingMode(spec.mode) is TrainingMode.POOLED or spec.animal == key


def list_applicable(rec: Recording, registry: Registry, *, cohort: str) -> list[ModelSpec]:
    """Every model that MAY be applied to ``rec``, metrics attached. Never a default.

    ``cohort`` is the recording's cohort (``"old"`` / ``"new"``), required: the recording
    is matched on ``animal_key(cohort, rec.animal)``, never on the bare letter.
    ``PER_ANIMAL`` and ``ADAPTED`` models of any other animal key are excluded.
    """
    key = animal_key(_check_cohort(cohort), rec.animal)
    return [s for s in registry.specs() if _applies(s, key)]


def _is_listed(chosen: ModelSpec, rec: Recording, registry: Registry, cohort: str) -> bool:
    """Whether ``chosen`` - the whole spec, not just its id - is in ``list_applicable``."""
    text = chosen.to_json()
    return any(s.to_json() == text
               for s in list_applicable(rec, registry, cohort=cohort))


@dataclass(frozen=True)
class InferenceResult:
    """Calibrated ``P(motion)`` per core, with the provenance that names the model."""

    p_motion: F64
    raw: F64
    provenance: dict[str, Any]


def inference_provenance(spec: ModelSpec, *, rec: Recording, cohort: str,
                         chosen_by: str) -> dict[str, Any]:
    """Return the provenance block of a run: the full spec and the user's choice."""
    return {"model": spec.to_dict(), "model_id": spec.model_id,
            "unvalidated": spec.unvalidated, "chosen_by": chosen_by,
            "animal": rec.animal, "cohort": _check_cohort(cohort),
            "animal_key": animal_key(cohort, rec.animal), "session": rec.session}


def require_model_in_provenance(prov: Mapping[str, Any]) -> ModelSpec:
    """Return the spec named by a mask's provenance; raise if it names none (task 12A)."""
    if not prov.get("model"):
        msg = "provenance names no model; a mask without one is invalid"
        raise ValueError(msg)
    return ModelSpec.from_dict(prov["model"])


def _refuse_on_evaluation(spec: ModelSpec, *, evaluation: bool, cores: pd.DataFrame | None,
                          where: str) -> None:
    """Refuse a ``never_scores_evaluation_spans`` model on evaluation spans.

    Evaluation is what the caller declares (``evaluation``, required) or what the cores
    show: any core with an audit ``span_id`` of a prospective test animal is an evaluation
    core (R1: I/J/K audit spans are test-only).
    """
    if not spec.never_scores_evaluation_spans:
        return
    in_span = (cores is not None and "span_id" in cores.columns
               and bool(cores["span_id"].notna().any()))
    if evaluation or in_span:
        msg = (f"{where}: model {spec.model_id} (ADAPTED, {spec.animal}) never scores "
               "evaluation spans of a prospective test animal (ruling 2026-10-08 (b) item 2)")
        raise NotApplicableError(msg)


def run_inference(rec: Recording, chosen: ModelSpec, registry: Registry, *, cohort: str,
                  cores: pd.DataFrame, chosen_by: str, evaluation: bool) -> InferenceResult:
    """Score ``rec``'s cores with the model the user chose. Fails if it may not apply.

    ``chosen`` has no default and there is no fallback: if the user has not chosen, the
    run does not start. ``cohort`` is the recording's cohort (required). ``cores`` holds
    the recording's core feature rows. ``evaluation`` (required) says whether these cores
    are evaluation spans; a model flagged ``never_scores_evaluation_spans`` is refused on
    them, and on any core carrying an audit ``span_id``.
    """
    _refuse_on_evaluation(chosen, evaluation=evaluation, cores=cores, where="run_inference")
    key = animal_key(_check_cohort(cohort), rec.animal)
    if TrainingMode(chosen.mode) is not TrainingMode.POOLED and chosen.animal != key:
        msg = (f"{chosen.mode} model for {chosen.animal} is not applicable to {key}; "
               "choose from list_applicable()")
        raise NotApplicableError(msg)
    if not _is_listed(chosen, rec, registry, cohort):
        msg = (f"model {chosen.model_id} ({chosen.mode}, animal={chosen.animal}) is not "
               f"in list_applicable() for {key}")
        raise NotApplicableError(msg)
    booster = registry.booster(chosen)
    x = cores[feature_columns(cores, booster.feature_name())]
    raw = predict_raw(booster, x)
    p = registry.calibrator(chosen).apply(raw)
    return InferenceResult(p_motion=p, raw=raw,
                           provenance=inference_provenance(chosen, rec=rec, cohort=cohort,
                                                           chosen_by=chosen_by))


def resolve_batch(recs: Sequence[Recording], choices: Mapping[str, ModelSpec],
                  registry: Registry, *, cohort: str, evaluation: bool) -> pd.DataFrame:
    """Return the assignment table a batch shows for confirmation: one choice per animal.

    A batch is one cohort (``cohort``, required); ``choices`` is keyed by animal key
    (``"<cohort>:<letter>"``). Raises if any animal in the batch has no choice, or a choice
    is not applicable. ``evaluation`` (required) says whether the batch scores evaluation
    spans; a model flagged ``never_scores_evaluation_spans`` is then refused.
    ``mixed_modes`` is True on every row when the batch spans more than one mode - the QC
    report must surface it (mode is then a covariate, task 19).
    """
    _check_cohort(cohort)
    for spec in choices.values():
        _refuse_on_evaluation(spec, evaluation=evaluation, cores=None, where="resolve_batch")
    keys = sorted({animal_key(cohort, r.animal) for r in recs})
    missing = [k for k in keys if k not in choices]
    if missing:
        msg = f"no model chosen for animal(s) {missing}; a batch needs one choice per animal"
        raise NotApplicableError(msg)
    rows = []
    for r in recs:
        key = animal_key(cohort, r.animal)
        spec = choices[key]
        if not _is_listed(spec, r, registry, cohort):
            msg = f"chosen model {spec.model_id} is not applicable to animal {key}"
            raise NotApplicableError(msg)
        rows.append({"animal": r.animal, "animal_key": key, "session": r.session,
                     "model_id": spec.model_id, "mode": str(TrainingMode(spec.mode)),
                     "unvalidated": spec.unvalidated})
    out = pd.DataFrame(rows)
    out["mixed_modes"] = out["mode"].nunique() > 1 if len(out) else False
    return out

