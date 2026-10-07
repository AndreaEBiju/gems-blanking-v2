r"""Task 12 acceptance 6: ``provenance.json`` for every trained model (invariant 11).

One file per model, in the model's own directory, written once with the model:

* ``model``: the full :class:`~gems_blanking_v2.model.registry.ModelSpec` record (mode,
  animal key, version, corpus hash, calibrator, trained_at, metrics, n_train_events) and
  its ``model_id``;
* ``corpus``: composition per animal key x old-cohort tier, with label counts;
* ``w_adapt``: mode B only - **absent** for any other mode (never ``null``);
* ``calibration``: what the calibrator was fitted on - always held-out predictions of
  the mode's evaluation folds, never of the final model itself (for POOLED: the
  per-target pooled LOAO models' held-out predictions, with the targets named);
* ``thresholds``: the protocol as the run record wrote it (R9, decision probability,
  ECE bins, ``w_adapt`` grid, calibration kind, the INTERPRETATION notes), read from the
  record file, not restated from code;
* ``run_record``: its run id and the SHA-256 of its bytes;
* ``code``: git commit (absent outside a checkout), ``dirty``, and the package source
  hash.

Serialisation follows the conventions table: canonical ASCII JSON, UTF-8, ``\n`` line
endings, atomic write; a value that does not exist is an ABSENT key. A non-finite number
raises rather than being written.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from gems_blanking_v2.detect.recall import source_sha256
from gems_blanking_v2.io.store import atomic_write_text
from gems_blanking_v2.model.evaluate import run_protocol
from gems_blanking_v2.model.labels import ADJUDICATED_BASIS, SET_A_BASIS, animal_key

if TYPE_CHECKING:
    from gems_blanking_v2.model.registry import ModelSpec

__all__ = [
    "PROVENANCE_NAME",
    "build_provenance",
    "code_provenance",
    "corpus_composition",
    "read_provenance",
    "write_provenance",
]

PROVENANCE_NAME: Final = "provenance.json"
SET_A_TIER: Final = "set_a"
ADJUDICATED_TIER: Final = "adjudicated"
"""The corpus-composition tier of other old cores Andrea judged on the screen."""
"""The corpus-composition tier of set A's judged random old cores."""
_PROTOCOL_KEYS: Final[tuple[str, ...]] = tuple(run_protocol())


def code_provenance() -> dict[str, Any]:
    """Return the code that trained: git commit and dirty flag (absent without git), hash."""
    package = Path(__file__).resolve().parents[1]
    out: dict[str, Any] = {"package_sha256": source_sha256(package)}
    try:
        commit = subprocess.run(["git", "-C", str(package.parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=30)
        status = subprocess.run(["git", "-C", str(package.parent), "status", "--porcelain",
                                 "--", package.name], capture_output=True, text=True,
                                check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return out
    out["commit"] = commit.stdout.strip()
    out["dirty"] = bool(status.stdout.strip())
    return out


def corpus_composition(table: pd.DataFrame, old_tiers: Mapping[str, str] | None
                       ) -> list[dict[str, Any]]:
    """Label counts (and label_source counts) per animal key x tier of a training corpus.

    New-cohort rows carry no tier (the key is absent). Set A's judged random old cores
    are Andrea's own judgments, not inherited marks: their tier is ``set_a``. Every other
    old-cohort recording must have a tier in ``old_tiers``; one without raises, naming it
    - an old label of unknown tier is never admitted silently.
    """
    keys = [animal_key(c, a) for c, a in
            zip(table["cohort"].astype(str), table["animal"].astype(str), strict=True)]
    tiers: list[str | None] = []
    missing: set[str] = set()
    basis = (table["basis"].astype(str) if "basis" in table.columns
             else pd.Series([""] * len(table), index=table.index))
    for c, r, b in zip(table["cohort"].astype(str), table["recording"].astype(str), basis,
                       strict=True):
        if c != "old":
            tiers.append(None)
            continue
        if b == SET_A_BASIS:
            tiers.append(SET_A_TIER)
            continue
        if b == ADJUDICATED_BASIS:
            tiers.append(ADJUDICATED_TIER)
            continue
        t = (old_tiers or {}).get(r)
        if t is None:
            missing.add(r)
        tiers.append(t)
    if missing:
        msg = f"old-cohort recording(s) without a tier: {sorted(missing)[:5]}"
        raise ValueError(msg)
    if "label_source" not in table.columns:
        msg = "corpus table lacks label_source; every label's source is recorded"
        raise ValueError(msg)
    d = pd.DataFrame({"animal_key": keys, "tier": tiers,
                      "recording": table["recording"].astype(str).to_numpy(),
                      "y": table["y"].astype(int).to_numpy(),
                      "label_source": table["label_source"].astype(str).to_numpy()})
    rows = []
    for (key, tier), g in d.groupby(["animal_key", "tier"], dropna=False, sort=True):
        row: dict[str, Any] = {"animal_key": str(key), "n_events": len(g),
                               "n_motion": int(g["y"].sum()),
                               "n_not_motion": int((g["y"] == 0).sum()),
                               "n_recordings": int(g["recording"].nunique()),
                               "label_source": {str(k): int(v) for k, v in
                                                g["label_source"].value_counts().items()}}
        if isinstance(tier, str):
            row["tier"] = tier
        rows.append(row)
    return rows


def _record(record_path: Path) -> tuple[dict[str, Any], str]:
    raw = Path(record_path).read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def build_provenance(spec: ModelSpec, *, corpus: list[dict[str, Any]], record_path: Path,
                     w_adapt: float | None, calibration: Mapping[str, Any],
                     code: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble one model's provenance. ``w_adapt`` must be given exactly for mode B.

    ``calibration`` (required) says what the calibrator was fitted on: ``kind``,
    ``fitted_on``, ``protocol``, ``targets`` and ``n_predictions``.
    """
    from gems_blanking_v2.types import TrainingMode  # noqa: PLC0415 - local, light

    is_b = TrainingMode(spec.mode) is TrainingMode.ADAPTED
    if is_b != (w_adapt is not None):
        msg = f"w_adapt is required for an adapted model and absent otherwise ({spec.mode})"
        raise ValueError(msg)
    need = ("kind", "fitted_on", "protocol", "targets", "n_predictions")
    gap = [k for k in need if k not in calibration]
    if gap:
        msg = f"calibration disclosure lacks {gap}"
        raise ValueError(msg)
    rec, sha = _record(record_path)
    missing = [k for k in _PROTOCOL_KEYS if k not in rec]
    if missing:
        msg = f"run record {record_path} lacks protocol field(s) {missing}"
        raise ValueError(msg)
    out: dict[str, Any] = {
        "model": spec.to_dict(), "model_id": spec.model_id, "unvalidated": spec.unvalidated,
        "corpus": list(corpus),
        "calibration": dict(calibration),
        "thresholds": {k: rec[k] for k in _PROTOCOL_KEYS},
        "run_record": {"run_id": rec["run_id"], "sha256": sha},
        "code": dict(code) if code is not None else code_provenance(),
    }
    if w_adapt is not None:
        out["w_adapt"] = float(w_adapt)
    return out


def _check_finite(obj: Any, where: str = "provenance") -> None:  # noqa: ANN401
    if isinstance(obj, float) and not math.isfinite(obj):
        msg = f"{where} holds {obj!r}; a missing value is an absent key"
        raise ValueError(msg)
    if obj is None:
        msg = f"{where} is null; a missing value is an absent key"
        raise ValueError(msg)
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _check_finite(v, f"{where}.{k}")
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            _check_finite(v, f"{where}[{i}]")


def _canonical(prov: Mapping[str, Any]) -> str:
    _check_finite(dict(prov))
    return json.dumps(prov, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


def write_provenance(model_dir: Path, prov: Mapping[str, Any]) -> Path:
    """Write ``provenance.json`` atomically, once. An existing different file raises."""
    path = Path(model_dir) / PROVENANCE_NAME
    text = _canonical(prov)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            msg = f"{path} exists with different content; provenance is write-once"
            raise ValueError(msg)
        return path
    atomic_write_text(path, text)
    return path


def read_provenance(model_dir: Path) -> dict[str, Any]:
    """Read a model's provenance; raises naming the field if a required one is missing."""
    raw = json.loads((Path(model_dir) / PROVENANCE_NAME).read_text(encoding="utf-8"))
    for key in ("model", "model_id", "corpus", "thresholds", "run_record", "code"):
        if raw.get(key) is None:
            msg = f"provenance is missing required field {key!r}"
            raise ValueError(msg)
    return dict(raw)
