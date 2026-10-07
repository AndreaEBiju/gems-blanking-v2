"""Task 15: the MATLAB handoff - write a recording's masks, refusing what QC holds.

:func:`write_mask_file` computes the QC gate FROM THE MASKS ITSELF (``emit.qc.emit_gate``:
the retention gate and the 20% / 3x blank hold), so a caller cannot hand it a gate that
says "pass" for masks that do not; ``min_retention`` and the animal medians are required
arguments. A held recording is written only with a non-blank ``release`` naming who
released it and why. ``gate_json`` records the verdict, ``min_retention``, the medians
used, the unknown-median notes and the release.

The file holds one ``blank_<consumer>_<signal>`` (N x 2, 1-based inclusive samples into
the epoch, via ``extent.grid`` - invariant 15) per mask key, never a merged one, plus
``notmeasured_mmc_<signal>`` (R6), ``provenance_json``, ``events_json``,
``retention_json`` and ``gate_json``. Provenance that does not name a model is refused.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.io import savemat

from gems_blanking_v2.emit.masks import ConsumerMask, MaskKey, mask_sample_spans, mmc_not_measured
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.emit.qc import emit_gate
from gems_blanking_v2.extent.grid import n_grid_frames, to_matlab_inclusive
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.extent.tolerance import extent_consumers
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs

__all__ = ["RecordingHeldError", "write_mask_file"]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
MATLAB_NAME_MAX: Final = 63
"""MATLAB's ``namelengthmax``."""
T0_TOLERANCE_S: Final = 1e-9
"""Two grid origins closer than this are the same origin (float noise only)."""


class RecordingHeldError(RuntimeError):
    """QC holds the recording; it is emitted only with an explicit release."""


def _matlab_name(prefix: str, consumer: str, signal: str) -> str:
    name = f"{prefix}_{consumer}_{signal}"
    if (not name.replace("_", "").isalnum() or len(name) > MATLAB_NAME_MAX
            or not name[0].isalpha()):
        msg = f"{name!r} is not a MATLAB variable name"
        raise ValueError(msg)
    return name


def _matlab_spans(invalid: Bool, fs: float, n_samples: int, grid_s: float, what: str) -> F64:
    spans = mask_sample_spans(invalid, fs, n_samples, grid_s)
    if not spans:
        return np.zeros((0, 2), dtype=np.float64)
    out = to_matlab_inclusive([a for a, _ in spans], [b for _, b in spans])
    assert_no_zero_runs(out.ravel(), what=what)
    return out


def write_mask_file(path: Path, masks: Mapping[MaskKey, ConsumerMask],
                    provenance: MaskProvenance | None, *, fs: float, n_samples: int,
                    epoch_start_s: float, min_retention: float,
                    animal_median: Mapping[str, float | None],
                    decisions: Iterable[RouteDecision] = (),
                    hum_features: Mapping[str, float] | None = None,
                    release: str | None = None,
                    events: Sequence[Mapping[str, Any]] = ()) -> Path:
    """Write the masks as MATLAB blank spans with their provenance and QC gate.

    Refuses: provenance that does not name a model; a held recording (gate computed here,
    from ``masks``) without a non-blank ``release``; a mask whose grid does not start at
    ``epoch_start_s`` or does not have ``floor(n_samples / fs / grid)`` frames. Every
    numeric array is checked for exact-zero runs (invariant 1).
    """
    if provenance is None:
        msg = "a mask file must carry provenance naming its model (task 15); none given"
        raise ProvenanceError(msg)
    provenance.validate()
    gate = emit_gate(masks, min_retention=min_retention, animal_median=animal_median,
                     decisions=decisions, hum_features=hum_features)
    if release is not None and not release.strip():
        msg = "a release must say who released the recording and why; got a blank string"
        raise ValueError(msg)
    if gate.held and not release:
        msg = ("QC holds this recording (" + "; ".join(gate.reasons) + "); emit it only "
               "with an explicit release naming who released it and why")
        raise RecordingHeldError(msg)
    doc: dict[str, Any] = {}
    retention: dict[str, float] = {}
    for (consumer, signal, band), m in sorted(masks.items()):
        if abs(m.t0_s - epoch_start_s) > T0_TOLERANCE_S:
            msg = (f"{consumer}/{signal}: mask grid starts at {m.t0_s} s but the file indexes "
                   f"the epoch from {epoch_start_s} s")
            raise ValueError(msg)
        want = n_grid_frames(n_samples, fs, m.grid_s)
        if m.invalid.size != want:
            msg = f"{consumer}/{signal}: {m.invalid.size} frames for an epoch of {want}"
            raise ValueError(msg)
        doc[_matlab_name("blank", consumer, signal)] = _matlab_spans(
            m.invalid, fs, n_samples, m.grid_s, f"blank spans {consumer}/{signal}")
        retention[f"{consumer}|{signal}|{band}"] = m.retention
    for sig, frames in sorted(mmc_not_measured(masks).items()):
        doc[_matlab_name("notmeasured", "mmc", sig)] = _matlab_spans(
            frames, fs, n_samples, masks[("mmc", sig, extent_consumers()["mmc"].band)].grid_s,
            f"not-measured spans mmc/{sig}")
    gate_doc: dict[str, Any] = {"held": gate.held, "reasons": list(gate.reasons),
                                "retention_flagged": gate.retention_flagged,
                                "blank_held": gate.blank_held,
                                "min_retention": gate.min_retention,
                                "medians_used": dict(gate.medians_used),
                                "notes": list(gate.notes)}
    if release:
        gate_doc["release"] = release
    doc["provenance_json"] = provenance.to_json()
    doc["events_json"] = json.dumps(list(events), sort_keys=True, ensure_ascii=True,
                                    allow_nan=False)
    doc["retention_json"] = json.dumps(retention, sort_keys=True, ensure_ascii=True,
                                       allow_nan=False)
    doc["gate_json"] = json.dumps(gate_doc, sort_keys=True, ensure_ascii=True)
    doc["fs"] = float(fs)
    doc["epochStart_s"] = float(epoch_start_s)
    doc["nSamples"] = float(n_samples)
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as fh:  # a handle, so savemat cannot append ".mat" to the name
            savemat(fh, doc, do_compression=True)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
