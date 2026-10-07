"""Task 15: one mask per consumer, and the file MATLAB reads.

Rules applied:

* **One boolean mask per ``(consumer, signal, band)``** on the shared 10 ms grid,
  ``True`` = invalid. Masks are never merged across consumers (hard invariant 2): each
  key is built from that consumer's own extents and routes only. The one permitted
  merge is inside ``velocity``, which needs ``V1`` AND ``V3`` valid
  (:func:`velocity_mask`); task 18 is out of this build (ruling (b) R5), so
  :func:`build_masks` emits no velocity mask unless asked.
* **Masked samples are NaN, never 0** (hard invariant 1): :func:`apply_mask` writes NaN,
  and a cosine taper of :data:`TAPER_S` on the valid side of each boundary whose weight
  never reaches zero; every emitted signal array is checked with
  ``io.nan_interop.assert_no_zero_runs``.
* **The routing table's ``distrusted_spans``** (ruling 2026-10-03) become part of the
  spike consumer's mask for that cuff, and a ``line_noise`` route's per-minute cuff
  distrust (ruling (c) item 4) does too - for the spike consumer only.
* **The index boundary converts once**: spans written for MATLAB are 1-based inclusive
  samples via ``emit.hr_beats.to_blank_spans`` (invariant 15), the blankSpans
  convention.
* **A mask file whose provenance does not name a model is refused on write.**

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.io import savemat

from gems_blanking_v2.constants import GRID_S
from gems_blanking_v2.emit.hr_beats import to_blank_spans
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.extent.tolerance import Extent, extent_consumers
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs
from gems_blanking_v2.types import Event

__all__ = [
    "TAPER_S",
    "ConsumerMask",
    "MaskKey",
    "MaskSpan",
    "apply_mask",
    "build_masks",
    "distrusted_spike_spans",
    "event_rows",
    "frames_to_samples",
    "line_noise_spike_spans",
    "mask_frames",
    "masked_spans_s",
    "spans_from_routing",
    "velocity_mask",
    "write_mask_file",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
MaskKey = tuple[str, str, str]
"""``(consumer, signal, band)``."""

MATLAB_NAME_MAX: Final = 63
"""MATLAB's ``namelengthmax``."""

TAPER_S: Final = 0.0075
"""Cosine taper on the valid side of each mask boundary, seconds: the middle of the
spec's 5-10 ms."""


@dataclass(frozen=True, slots=True)
class MaskSpan:
    """One span to mask for one consumer on one signal, seconds, half-open, with why."""

    consumer: str
    signal: str
    start_s: float
    stop_s: float
    reason: str


@dataclass(frozen=True)
class ConsumerMask:
    """One consumer's mask on one signal: frames on the grid, ``True`` = invalid."""

    consumer: str
    signal: str
    band: str
    invalid: Bool
    grid_s: float

    @property
    def key(self) -> MaskKey:
        """``(consumer, signal, band)``."""
        return (self.consumer, self.signal, self.band)

    @property
    def retention(self) -> float:
        """Fraction of frames kept."""
        return 1.0 - float(self.invalid.mean()) if self.invalid.size else math.nan


def mask_frames(spans: Iterable[tuple[float, float]], n_frames: int,
                grid_s: float = GRID_S) -> Bool:
    """Frames overlapping any half-open span ``[a, b)``: frame ``i`` is ``[i g, (i+1) g)``."""
    out = np.zeros(n_frames, dtype=bool)
    for a, b in spans:
        if not b > a:
            continue
        i0 = max(0, int(math.floor(a / grid_s)))
        i1 = min(n_frames, int(math.ceil(b / grid_s)))
        out[i0:i1] = True
    return out


def spans_from_routing(extents: Iterable[tuple[str, Extent]],
                       decisions: Iterable[RouteDecision]) -> list[MaskSpan]:
    """Mask spans: each ``(event_id, extent)`` whose route for that consumer masks.

    An extent without a decision raises - routing is recorded per event per consumer,
    and a missing decision is a defect, not a "keep".
    """
    by = {(d.event_id, d.consumer): d for d in decisions}
    out: list[MaskSpan] = []
    for event_id, ext in extents:
        d = by.get((event_id, ext.consumer))
        if d is None:
            msg = f"no routing decision for event {event_id} on {ext.consumer}"
            raise KeyError(msg)
        if d.masks:
            out.append(MaskSpan(ext.consumer, ext.signal, ext.start_s, ext.stop_s,
                                f"{d.route}: {d.reason}"))
    return out


def distrusted_spike_spans(entry: Mapping[str, Any], region_start_s: float
                           ) -> list[MaskSpan]:
    """Return the routing entry's per-minute cuff distrust as spike-consumer spans.

    Ruling 2026-10-03 item 2: a cuff's ``distrusted_spans`` (region-relative, half-open
    seconds) become part of the spike consumer's mask for that cuff - and of no other
    consumer's. ``region_start_s`` places them on the recording's timeline.
    """
    out: list[MaskSpan] = []
    for cuff, s in sorted(entry.get("spike", {}).items()):
        for a, b in s.get("distrusted_spans", []):
            out.append(MaskSpan("spikes", f"{cuff}_T", region_start_s + float(a),
                                region_start_s + float(b),
                                "per-minute cuff distrust (ruling 2026-10-03)"))
    return out


def line_noise_spike_spans(cuff: str, minutes_s: Iterable[tuple[float, float]]
                           ) -> list[MaskSpan]:
    """Return line-noise minutes as spike-consumer mask spans for that cuff only.

    Ruling (c) item 4: the minutes where a cuff's mains-locked spike fraction exceeds its
    threshold.
    """
    return [MaskSpan("spikes", f"{cuff}_T", float(a), float(b),
                     "line noise: per-minute cuff distrust (ruling (c) 4)")
            for a, b in minutes_s]


def build_masks(signals: Mapping[str, Sequence[str]], spans: Iterable[MaskSpan],
                n_frames: int, *, grid_s: float = GRID_S,
                include_velocity: bool = False) -> dict[MaskKey, ConsumerMask]:
    """One mask per ``(consumer, signal, band)`` from that consumer's spans alone.

    ``signals`` maps each consumer to the signals it reads in this recording (every one
    gets a mask, all-valid if nothing touched it). A span for a consumer or signal not
    listed raises. Velocity is skipped unless ``include_velocity`` (task 18, R5).
    """
    specs = extent_consumers()
    by: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for sp in spans:
        if sp.consumer not in signals or sp.signal not in signals[sp.consumer]:
            msg = f"span for {sp.consumer}/{sp.signal}, which this recording does not read"
            raise KeyError(msg)
        by.setdefault((sp.consumer, sp.signal), []).append((sp.start_s, sp.stop_s))
    out: dict[MaskKey, ConsumerMask] = {}
    for consumer, names in signals.items():
        if consumer == "velocity" and not include_velocity:
            continue
        band = specs[consumer].band
        for sig in names:
            m = ConsumerMask(consumer, sig, band,
                             mask_frames(by.get((consumer, sig), []), n_frames, grid_s), grid_s)
            out[m.key] = m
    return out


def velocity_mask(v1: ConsumerMask, v3: ConsumerMask) -> ConsumerMask:
    """Velocity needs V1 AND V3 valid: invalid where either is (the one allowed merge)."""
    if v1.consumer != "velocity" or v3.consumer != "velocity":
        msg = "velocity_mask combines velocity's own two contact masks, nothing else"
        raise ValueError(msg)
    if not (v1.signal.endswith("_V1") and v3.signal.endswith("_V3")
            and v1.signal[:-3] == v3.signal[:-3]):
        msg = f"need one cuff's V1 and V3, got {v1.signal} and {v3.signal}"
        raise ValueError(msg)
    if v1.invalid.shape != v3.invalid.shape:
        msg = "V1 and V3 masks must be on the same grid"
        raise ValueError(msg)
    return ConsumerMask("velocity", f"{v1.signal[:-3]}_V1V3", v1.band, v1.invalid | v3.invalid,
                        v1.grid_s)


def frames_to_samples(invalid: Bool, fs: float, n_samples: int, grid_s: float = GRID_S
                      ) -> Bool:
    """Frame mask to sample mask: frame ``i`` covers samples ``[ceil(i g fs), ceil((i+1) g fs))``.

    Samples after the last whole frame (the dropped partial frame) take the last frame's
    state.
    """
    out = np.zeros(n_samples, dtype=bool)
    for i in np.flatnonzero(invalid):
        a = int(math.ceil(i * grid_s * fs))
        b = int(math.ceil((i + 1) * grid_s * fs))
        out[a:min(b, n_samples)] = True
    if invalid.size and invalid[-1]:
        out[int(math.ceil(invalid.size * grid_s * fs)):] = True
    return out


def apply_mask(x: npt.ArrayLike, fs: float, invalid: Bool, *, grid_s: float = GRID_S,
               taper_s: float = TAPER_S, what: str = "masked signal") -> F64:
    """Return a copy of ``x`` with masked samples NaN and a cosine taper beside each run.

    The taper weight ``0.5 (1 - cos(pi (d + 1) / (L + 1)))`` at distance ``d`` samples
    from the run (``L`` = taper length) is strictly between 0 and 1, so the taper never
    writes a zero. The output is checked for exact-zero runs (invariant 1). The input is
    never modified (it may be a read-only view, invariant 17).
    """
    out = np.array(x, dtype=np.float64, copy=True)
    bad = frames_to_samples(invalid, fs, out.size, grid_s) | ~np.isfinite(out)
    n_taper = int(round(taper_s * fs))
    if n_taper > 0 and bad.any():
        w = np.ones(out.size)
        ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(n_taper) + 1.0) / (n_taper + 1.0)))
        d = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
        for s in (int(v) for v in np.flatnonzero(d == 1)):  # taper before each run
            lo = max(0, s - n_taper)
            w[lo:s] = np.minimum(w[lo:s], ramp[::-1][n_taper - (s - lo):])
        for e in (int(v) for v in np.flatnonzero(d == -1)):  # and after it
            hi = min(out.size, e + n_taper)
            w[e:hi] = np.minimum(w[e:hi], ramp[: hi - e])
        out *= w
    out[bad] = np.nan
    assert_no_zero_runs(out, what=what, fs=fs)
    return out


def masked_spans_s(invalid: Bool, grid_s: float = GRID_S) -> list[tuple[float, float]]:
    """Return the mask's invalid runs as half-open ``[a, b)`` seconds."""
    d = np.diff(np.concatenate(([0], invalid.astype(np.int8), [0])))
    return [(a * grid_s, b * grid_s) for a, b in
            zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True)]


def event_rows(events: Mapping[str, Event], decisions: Iterable[RouteDecision],
               provenance: MaskProvenance) -> list[dict[str, Any]]:
    """Build the event table: one row per event with its routing per consumer.

    ``start``, ``stop`` (s), ``p_motion`` (absent when no model scored it), ``judgement``,
    ``bands`` affected, ``routes`` ``{consumer: route}``, and the model that made it
    (``model_version``, ``corpus_hash``) - the full provenance travels in the same file.
    JSON-ready: a missing number is an absent key.
    """
    by: dict[str, dict[str, str]] = {}
    for d in decisions:
        by.setdefault(d.event_id, {})[d.consumer] = d.route
    rows: list[dict[str, Any]] = []
    for eid, ev in events.items():
        c = ev.candidate
        row: dict[str, Any] = {"event_id": eid, "start": c.start_s, "stop": c.stop_s,
                               "judgement": ev.judgement, "source": ev.source,
                               "bands": list(c.bands), "routes": by.get(eid, {}),
                               "model_version": provenance.model["version"],
                               "corpus_hash": provenance.model["corpus_hash"]}
        if math.isfinite(ev.p_motion):
            row["p_motion"] = ev.p_motion
        rows.append(row)
    return rows


def _matlab_name(prefix: str, consumer: str, signal: str) -> str:
    name = f"{prefix}_{consumer}_{signal}"
    if (not name.replace("_", "").isalnum() or len(name) > MATLAB_NAME_MAX
            or not name[0].isalpha()):
        msg = f"{name!r} is not a MATLAB variable name"
        raise ValueError(msg)
    return name


def write_mask_file(path: Path, masks: Mapping[MaskKey, ConsumerMask],
                    provenance: MaskProvenance | None, *, fs: float, n_samples: int,
                    epoch_start_s: float = 0.0,
                    events: Sequence[Mapping[str, Any]] = ()) -> Path:
    """Write the masks as MATLAB blankSpans (1-based inclusive) with their provenance.

    One ``blank_<consumer>_<signal>`` (N x 2, 1-based inclusive samples into the epoch)
    per mask key - never a merged one - plus ``provenance_json``, ``events_json`` (the
    event table) and ``retention_json``. Refuses provenance that does not name a model,
    and checks every numeric array for exact-zero runs (invariant 1).
    """
    if provenance is None:
        msg = "a mask file must carry provenance naming its model (task 15); none given"
        raise ProvenanceError(msg)
    provenance.validate()
    doc: dict[str, Any] = {}
    retention: dict[str, float] = {}
    for (consumer, signal, band), m in sorted(masks.items()):
        spans_s = [(epoch_start_s + a, epoch_start_s + b)
                   for a, b in masked_spans_s(m.invalid, m.grid_s)]
        spans = to_blank_spans(np.asarray(spans_s, dtype=np.float64).reshape(-1, 2), fs,
                               epoch_start_s, n_samples)
        if spans.size:
            assert_no_zero_runs(spans.ravel(), what=f"blank spans {consumer}/{signal}")
        doc[_matlab_name("blank", consumer, signal)] = spans
        retention[f"{consumer}|{signal}|{band}"] = m.retention
    doc["provenance_json"] = provenance.to_json()
    doc["events_json"] = json.dumps(list(events), sort_keys=True, ensure_ascii=True,
                                    allow_nan=False)
    doc["retention_json"] = json.dumps(retention, sort_keys=True, ensure_ascii=True,
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
