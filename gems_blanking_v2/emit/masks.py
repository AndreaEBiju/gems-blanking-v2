"""Task 15: one mask per consumer, and the file MATLAB reads.

Rules applied:

* **One boolean mask per ``(consumer, signal, band)``** on the shared 10 ms grid,
  ``True`` = invalid. Masks are never merged across consumers (hard invariant 2): each
  key is built from that consumer's own extents and routes only. The one permitted
  merge is inside ``velocity``, which needs ``V1`` AND ``V3`` valid
  (:func:`velocity_mask`); task 18 is out of this build (ruling (b) R5), so
  :func:`build_masks` emits no velocity mask unless asked.
* **One timeline.** Extents, routing spans and distrusted spans are seconds on the
  RECORDING's timeline. A mask's grid starts at its own ``t0_s`` (the epoch start on
  that timeline): frame ``i`` covers ``[t0_s + i g, t0_s + (i + 1) g)``. The MATLAB file
  indexes the epoch from its first sample, so every mask written must have
  ``t0_s == epoch_start_s`` and exactly ``floor(n_samples / fs / g)`` frames - checked,
  never assumed.
* **One frame-to-sample conversion** (invariants 15, 33): ``extent.grid`` - the sample
  mask, the MATLAB spans and the clip test all use it, so they agree on every boundary
  sample, the inclusive END included.
* **Masked samples are NaN, never 0** (hard invariant 1): :func:`apply_mask` writes NaN,
  and a cosine taper of :data:`TAPER_S` on the valid side of each boundary whose weight
  never reaches zero; every emitted array is checked for exact-zero runs. **The taper is
  Python-side only**: the MATLAB file carries blank spans, which cannot carry a taper,
  and ``step1_bandpass.m`` treats every sample outside them as fully valid.
* **The routing table's ``distrusted_spans``** (ruling 2026-10-03) become part of the
  spike consumer's mask for that cuff, and a cuff distrusted outright (route
  ``distrusted``) gets a wholly invalid spike mask - for the spike consumer only.
* **Line noise never enters a mask.** The spike consumer's per-minute distrust for
  mains-locked spikes (ruling 2026-10-07 (c) item 4, decided by the test of RULING
  2026-10-08 (d) 2) is its own record (``emit.line_distrust``), never a mask span, so the
  motion accounting and the gates never see it. It is NaN in the spike consumer's input
  only ((e) Q2): the handoff adds it to ``blank_spikes_*`` and ``line_distrust.spike_input``
  to the Python-side input.
* **R6**: mmc output within +/-15 s of any of its blanks is "not measured"; those spans
  are written beside the mmc masks (``notmeasured_mmc_<signal>``), not into them.
* **The MATLAB file** is written by ``emit.handoff.write_mask_file``, which computes
  the QC gate from the masks itself and refuses a held recording without a release.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.constants import GRID_S
from gems_blanking_v2.extent.grid import frame_sample_bounds
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.extent.tolerance import (
    OUT_OF_BUILD_CONSUMERS,
    Extent,
    extent_consumers,
    mmc_not_measured_spans,
)
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs
from gems_blanking_v2.types import Event

if TYPE_CHECKING:
    from gems_blanking_v2.emit.provenance import MaskProvenance

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
    "mask_frames",
    "mask_sample_spans",
    "masked_spans_s",
    "mmc_not_measured",
    "spans_from_routing",
    "velocity_mask",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
MaskKey = tuple[str, str, str]
"""``(consumer, signal, band)``."""

TAPER_S: Final = 0.0075
"""Cosine taper on the valid side of each mask boundary, seconds: the middle of the
spec's 5-10 ms. Applied by :func:`apply_mask` only (see the module docstring)."""


@dataclass(frozen=True, slots=True)
class MaskSpan:
    """One span to mask for one consumer on one signal: recording-timeline seconds."""

    consumer: str
    signal: str
    start_s: float
    stop_s: float
    reason: str


@dataclass(frozen=True)
class ConsumerMask:
    """One consumer's mask on one signal: frames from ``t0_s``, ``True`` = invalid."""

    consumer: str
    signal: str
    band: str
    invalid: Bool
    grid_s: float
    t0_s: float

    @property
    def key(self) -> MaskKey:
        """``(consumer, signal, band)``."""
        return (self.consumer, self.signal, self.band)

    @property
    def retention(self) -> float:
        """Fraction of frames kept."""
        return 1.0 - float(self.invalid.mean()) if self.invalid.size else math.nan


def mask_frames(spans: Iterable[tuple[float, float]], n_frames: int, *, t0_s: float,
                grid_s: float = GRID_S) -> Bool:
    """Frames overlapping any span ``[a, b)`` (recording s); frame i is ``[t0 + i g, ...)``."""
    out = np.zeros(n_frames, dtype=bool)
    for a, b in spans:
        if not b > a:
            continue
        i0 = max(0, int(math.floor((a - t0_s) / grid_s)))
        i1 = min(n_frames, int(math.ceil((b - t0_s) / grid_s)))
        if i1 > i0:
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
                                d.reason_code or d.route))
    return out


def distrusted_spike_spans(entry: Mapping[str, Any], *, region_start_s: float,
                           region_stop_s: float) -> list[MaskSpan]:
    """Return the routing entry's spike distrust as spike-consumer spans (recording s).

    Ruling 2026-10-03: a cuff's ``distrusted_spans`` (region-relative, half-open s) join
    the spike consumer's mask for that cuff. A cuff whose route is ``distrusted`` outright
    is invalid for the spike consumer over the whole region. No other consumer is touched.
    """
    out: list[MaskSpan] = []
    for cuff, s in sorted(entry.get("spike", {}).items()):
        if s.get("route") == "distrusted":
            out.append(MaskSpan("spikes", f"{cuff}_T", region_start_s, region_stop_s,
                                "cuff_distrusted"))
            continue
        for a, b in s.get("distrusted_spans", []):
            out.append(MaskSpan("spikes", f"{cuff}_T", region_start_s + float(a),
                                region_start_s + float(b), "per_minute_cuff_distrust"))
    return out


def build_masks(signals: Mapping[str, Sequence[str]], spans: Iterable[MaskSpan], *,
                n_frames: int, t0_s: float, grid_s: float = GRID_S,
                include_velocity: bool = False) -> dict[MaskKey, ConsumerMask]:
    """One mask per ``(consumer, signal, band)`` from that consumer's spans alone.

    ``signals`` maps each consumer to the signals it reads in this recording (every one
    gets a mask, all-valid if nothing touched it). A span for a consumer or signal not
    listed raises, and so does a line-noise span (reason ``line_noise*``): line noise is
    never a blank (ruling 2026-10-07 (c)); its spike distrust is ``emit.line_distrust``'s
    own record. Velocity is skipped unless ``include_velocity`` (task 18, R5).
    """
    specs = extent_consumers()
    by: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for sp in spans:
        if sp.reason.startswith("line_noise"):
            msg = (f"{sp.consumer}/{sp.signal}: a line-noise span ({sp.reason}) never enters a "
                   "mask; the spike consumer's line distrust is its own record")
            raise ValueError(msg)
        if sp.consumer not in signals or sp.signal not in signals[sp.consumer]:
            msg = f"span for {sp.consumer}/{sp.signal}, which this recording does not read"
            raise KeyError(msg)
        by.setdefault((sp.consumer, sp.signal), []).append((sp.start_s, sp.stop_s))
    out: dict[MaskKey, ConsumerMask] = {}
    for consumer, names in signals.items():
        if consumer in OUT_OF_BUILD_CONSUMERS and not include_velocity:
            continue
        band = specs[consumer].band
        for sig in names:
            inv = mask_frames(by.get((consumer, sig), []), n_frames, t0_s=t0_s, grid_s=grid_s)
            m = ConsumerMask(consumer, sig, band, inv, grid_s, t0_s)
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
    if v1.invalid.shape != v3.invalid.shape or v1.t0_s != v3.t0_s:
        msg = "V1 and V3 masks must be on the same grid"
        raise ValueError(msg)
    return ConsumerMask("velocity", f"{v1.signal[:-3]}_V1V3", v1.band, v1.invalid | v3.invalid,
                        v1.grid_s, v1.t0_s)


def _runs(invalid: Bool) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], invalid.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(),
                    strict=True))


def mask_sample_spans(invalid: Bool, fs: float, n_samples: int, grid_s: float = GRID_S
                      ) -> list[tuple[int, int]]:
    """Return the invalid runs as 0-based half-open epoch samples, via ``extent.grid``.

    A run reaching the last whole frame extends to ``n_samples``: the samples of the
    dropped partial frame take the last frame's state.
    """
    out: list[tuple[int, int]] = []
    for i0, i1 in _runs(invalid):
        k0, k1 = frame_sample_bounds(i0, i1, fs, grid_s)
        if i1 == invalid.size:
            k1 = n_samples
        k0, k1 = min(k0, n_samples), min(k1, n_samples)
        if k1 > k0:
            out.append((k0, k1))
    return out


def frames_to_samples(invalid: Bool, fs: float, n_samples: int, grid_s: float = GRID_S
                      ) -> Bool:
    """Frame mask to sample mask, through the same conversion the MATLAB spans use."""
    out = np.zeros(n_samples, dtype=bool)
    for k0, k1 in mask_sample_spans(invalid, fs, n_samples, grid_s):
        out[k0:k1] = True
    return out


def apply_mask(x: npt.ArrayLike, fs: float, invalid: Bool, *, grid_s: float = GRID_S,
               taper_s: float = TAPER_S, what: str = "masked signal") -> F64:
    """Return a copy of ``x`` with masked samples NaN and a cosine taper beside each run.

    ``x[0]`` is the mask's frame-0 sample. The taper weight
    ``0.5 (1 - cos(pi (d + 1) / (L + 1)))`` at distance ``d`` samples from the run
    (``L`` = taper length) is strictly between 0 and 1, so it never writes a zero. The
    output is checked for exact-zero runs (invariant 1). The input is never modified (it
    may be a read-only view, invariant 17).
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


def masked_spans_s(m: ConsumerMask) -> list[tuple[float, float]]:
    """Return the mask's invalid runs as half-open recording-timeline seconds."""
    return [(m.t0_s + a * m.grid_s, m.t0_s + b * m.grid_s) for a, b in _runs(m.invalid)]


def mmc_not_measured(masks: Mapping[MaskKey, ConsumerMask]) -> dict[str, Bool]:
    """R6: per mmc signal, the frames within +/-15 s of any of that signal's mmc blanks."""
    out: dict[str, Bool] = {}
    for (consumer, sig, _band), m in masks.items():
        if consumer != "mmc":
            continue
        dur = m.invalid.size * m.grid_s
        rel = [(a - m.t0_s, b - m.t0_s) for a, b in masked_spans_s(m)]
        spans = [(m.t0_s + a, m.t0_s + b) for a, b in mmc_not_measured_spans(rel, dur)]
        out[sig] = mask_frames(spans, m.invalid.size, t0_s=m.t0_s, grid_s=m.grid_s)
    return out


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
