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

**Spike-consumer line distrust (RULING 2026-10-08 (d) 2)** travels BESIDE the spike
masks, never inside them: ``distrust_spikes_<cuff>_T`` (N x 2, the same 1-based inclusive
epoch samples) per spike signal, and ``linedistrust_json`` (the whole
:class:`~gems_blanking_v2.emit.line_distrust.LineDistrustRecord`: every cuff-minute's
counts, p-value and decision). ``blank_spikes_*`` stays the motion (and ruled cuff) mask;
this rule writes nothing into it. ``line_distrust`` is a required argument: a recording
whose spike consumer reads anything must carry a record covering exactly those signals,
and one that reads nothing passes ``None``. The record's provenance is copied into
``provenance_json`` (``spike_line_distrust``) here, from the record itself.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.io import savemat

from gems_blanking_v2.emit.line_distrust import LineDistrustRecord
from gems_blanking_v2.emit.masks import ConsumerMask, MaskKey, mask_sample_spans, mmc_not_measured
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.emit.qc import emit_gate
from gems_blanking_v2.extent.grid import T0_TOLERANCE_S, n_grid_frames, to_matlab_inclusive
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.extent.tolerance import (
    OUT_OF_BUILD_CONSUMERS,
    expected_consumers,
    extent_consumers,
)
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs

__all__ = ["RecordingHeldError", "write_mask_file"]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
MATLAB_NAME_MAX: Final = 63
"""MATLAB's ``namelengthmax``."""


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


def _check_coverage(masks: Mapping[MaskKey, ConsumerMask],
                    signals: Mapping[str, Sequence[str]]) -> None:
    """Raise unless ``masks`` cover exactly what the recording reads, on the right bands."""
    consumers = extent_consumers()
    required = expected_consumers()
    absent = [c for c in required if c not in signals]
    if absent:
        msg = (f"signals must name every consumer the recording could read (an empty tuple "
               f"where it reads none); missing {absent}")
        raise ValueError(msg)
    if not masks:
        msg = "refusing to write a mask file with no masks"
        raise ValueError(msg)
    unknown = sorted(({c for c, _s, _b in masks} - set(consumers))
                     | (set(signals) - set(consumers)))
    if unknown:
        msg = f"unknown consumers {unknown}; the consumers are {sorted(consumers)}"
        raise ValueError(msg)
    wrong_band = sorted(f"{c}/{s}: {b}" for c, s, b in masks if b != consumers[c].band)
    if wrong_band:
        msg = f"masks on the wrong band for their consumer: {wrong_band}"
        raise ValueError(msg)
    want_keys = {(c, s) for c, names in signals.items() if c not in OUT_OF_BUILD_CONSUMERS
                 for s in names}
    have_keys = {(c, s) for c, s, _b in masks if c not in OUT_OF_BUILD_CONSUMERS}
    if want_keys != have_keys:
        msg = (f"masks do not cover what this recording reads: missing "
               f"{sorted(want_keys - have_keys)}, unexpected {sorted(have_keys - want_keys)}")
        raise ValueError(msg)


def _with_line_distrust(provenance: MaskProvenance, line_distrust: LineDistrustRecord | None,
                        *, signals: Mapping[str, Sequence[str]], fs: float, n_samples: int,
                        epoch_start_s: float) -> MaskProvenance:
    """Check the record against this file and return the provenance that names it."""
    spike_signals = tuple(sorted(signals.get("spikes", ())))
    if line_distrust is None:
        if spike_signals:
            msg = (f"the spike consumer reads {list(spike_signals)}: the file must carry its "
                   "line-distrust record (RULING 2026-10-08 (d) 2); none given")
            raise ValueError(msg)
        if provenance.spike_line_distrust:
            msg = "provenance names a spike line-distrust rule but no record is carried"
            raise ProvenanceError(msg)
        return provenance
    if tuple(sorted(line_distrust.signals)) != spike_signals:
        msg = (f"the line-distrust record covers {list(line_distrust.signals)} but the spike "
               f"consumer reads {list(spike_signals)}")
        raise ValueError(msg)
    if (line_distrust.fs != float(fs) or line_distrust.n_samples != int(n_samples)
            or abs(line_distrust.epoch_start_s - epoch_start_s) > T0_TOLERANCE_S):
        msg = (f"the line-distrust record is for fs {line_distrust.fs}, {line_distrust.n_samples}"
               f" samples from {line_distrust.epoch_start_s} s; this file is fs {fs}, "
               f"{n_samples} samples from {epoch_start_s} s")
        raise ValueError(msg)
    if line_distrust.recording != provenance.recording:
        msg = (f"the line-distrust record is for {line_distrust.recording!r}, the provenance "
               f"for {provenance.recording!r}")
        raise ValueError(msg)
    want = json.loads(json.dumps(line_distrust.provenance, allow_nan=False))
    if provenance.spike_line_distrust and dict(provenance.spike_line_distrust) != want:
        msg = "provenance names a different spike line-distrust rule than the record carried"
        raise ProvenanceError(msg)
    return dataclasses.replace(provenance, spike_line_distrust=want)


def write_mask_file(path: Path, masks: Mapping[MaskKey, ConsumerMask],
                    provenance: MaskProvenance | None, *, signals: Mapping[str, Sequence[str]],
                    fs: float, n_samples: int,
                    epoch_start_s: float, min_retention: float,
                    animal_median: Mapping[str, float | None],
                    line_distrust: LineDistrustRecord | None,
                    decisions: Iterable[RouteDecision] = (),
                    hum_features: Mapping[str, float] | None = None,
                    release: str | None = None,
                    events: Sequence[Mapping[str, Any]] = ()) -> Path:
    """Write the masks as MATLAB blank spans with their provenance and QC gate.

    ``signals`` maps each consumer to the signals it reads in this recording (as for
    ``build_masks``); the masks must cover exactly those consumer x signal pairs (velocity
    excepted while task 18 is out), so the gate cannot be passed by leaving an
    over-blanked consumer out.

    Refuses: provenance that does not name a model; ``signals`` that does not name every
    consumer of :func:`~gems_blanking_v2.extent.tolerance.expected_consumers`; an empty
    ``masks``; a consumer (in ``masks`` or ``signals``) that is not in the tolerance
    table; a mask on a band other than its consumer's; masks that do not cover
    ``signals``; a held recording (gate computed here, from ``masks``) without a non-blank
    ``release``; a mask whose grid does not start at
    ``epoch_start_s`` or does not have ``floor(n_samples / fs / grid)`` frames; a
    ``line_distrust`` record missing while the spike consumer reads a signal, covering
    other signals than it reads, made for another epoch or recording, or whose rule
    disagrees with one the provenance already names. Every numeric array is checked for
    exact-zero runs (invariant 1).
    """
    if provenance is None:
        msg = "a mask file must carry provenance naming its model (task 15); none given"
        raise ProvenanceError(msg)
    provenance.validate()
    _check_coverage(masks, signals)
    provenance = _with_line_distrust(provenance, line_distrust, signals=signals, fs=fs,
                                     n_samples=n_samples, epoch_start_s=epoch_start_s)
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
    if line_distrust is not None:
        for sig in line_distrust.signals:
            spans = line_distrust.matlab_spans(sig)
            assert_no_zero_runs(spans.ravel(), what=f"line-distrust spans spikes/{sig}")
            doc[_matlab_name("distrust", "spikes", sig)] = spans
        doc["linedistrust_json"] = line_distrust.to_json()
    gate_doc: dict[str, Any] = {"held": gate.held, "reasons": list(gate.reasons),
                                "retention_flagged": gate.retention_flagged,
                                "blank_held": gate.blank_held,
                                "min_retention": gate.min_retention,
                                "medians_used": dict(gate.medians_used),
                                "notes": list(gate.notes),
                                "top_routes": [list(t) for t in gate.top_routes],
                                "hum_features": dict(gate.hum_features)}
    if release:
        gate_doc["release"] = release
    doc["provenance_json"] = provenance.to_json()
    doc["events_json"] = json.dumps(list(events), sort_keys=True, ensure_ascii=True,
                                    allow_nan=False)
    doc["retention_json"] = json.dumps(retention, sort_keys=True, ensure_ascii=True,
                                       allow_nan=False)
    doc["gate_json"] = json.dumps(gate_doc, sort_keys=True, ensure_ascii=True,
                                  allow_nan=False)
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
