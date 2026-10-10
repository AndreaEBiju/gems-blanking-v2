"""Night 5 stage 1, split: inference once (stage 1a), extents per threshold set (stage 1b).

RULING 2026-10-10 item 4: "Model inference for wave 1 starts now, if it does not depend on
this choice." It does not - but stage 1 as built ran detection, features, the model, the
decisions AND the extents in one pass, keyed on the tolerance set and the cross-band scope,
so every threshold change re-ran inference. The split:

* **stage 1a (inference)** - load, detection, cores, features, model scores, the declared
  decision rule (and the calibrated sensitivity variants), the HRV operational extents (they
  depend on the confirmed cores only, never on a z threshold), and the inputs stage 1b needs,
  cached per recording. Its freshness keys are inference inputs only
  (:data:`STAGE1A_KEYS`, :func:`stage1a_keys`): the model choice (file sha256 and the
  resolved model ids), the decision rule and sensitivity thresholds, the HRV window, the wave
  (id and routing table sha256), the routing entry and the peri-R window. The night config
  reaches it only through :func:`inference_config`, an allowlist, so no extent field can
  enter stage 1a at all.
* **stage 1b (extents)** - reads the cache and computes, for one DECLARED threshold set,
  the extents, the cross-band extents, routing, masks, blanked time and line-noise pass 1
  (:func:`extents_routing_masks`). It refuses a stage-1a record from another wave or
  model (:func:`check_stage1a_fresh`).

What depends on the extent choice (:data:`EXTENT_DEPENDENT`) is everything stage 1b makes,
and one item that stage 1 used to make before the choice existed: **line-noise pass 1**. Its
minute test reads raw T less the spike consumer's MOTION MASK, every motion blank padded by
10.5 ms (``emit.line_distrust.minute_tests``, RULING 2026-10-09 (g) 5), so a different spike
threshold or cross-band scope changes the samples it tests and the per-stretch sigma. Stage 1a
therefore caches raw T (exactly as pass 1 read it) and the peri-R exclusion, and stage 1b
runs pass 1.

The cache is exact for every threshold at or above :data:`CACHE_FLOOR_Z` (the clean null's
p90, 1.44, below which ruling 2026-10-09 (i) 1 never goes):

* consumer z (the bands the consumers threshold) is kept whole;
* detection z, read only by the cross-band detected extent, is kept on the frames of its runs
  above the floor that overlap the core of an event it detected (:func:`sparse_detection_z`)
  and NaN elsewhere. A run above a floor ``f >= 1.44`` lies inside a run above 1.44, so every
  run ``detected_extent`` can use, and its boundary frames, are unchanged;
* the spike consumer's ENG trace (routing's sample-level verdict) is kept over each cached
  event's core - routing judges the candidate span, never the extent (:func:`eng_cache_spans`)
  - with the whole trace's sigma, crossings and finite count (``routing.CachedEngTrace``); a
  span outside the cache raises ``routing.EngCacheMissError``, never a verdict.

Stage 1b refuses a floor below the cache's, and a confirmation set outside the events the
cache was built for (:func:`extents_routing_masks`), so an input the cache cannot answer
exactly fails by name. ``tests/test_stage1_split.py`` holds the cache and the full arrays to
the same extents, decisions and masks for thresholds and scopes drawn at random.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.constants import GRID_S
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.extent import routing as rt
from gems_blanking_v2.extent import tolerance as tl
from gems_blanking_v2.extent.tolerance_map import Z_FLOOR
from gems_blanking_v2.types import Candidate, Event

__all__ = [
    "CACHE_FLOOR_Z",
    "CORE_PAD_REASON",
    "EXTENT_DEPENDENT",
    "INFERENCE_CONFIG_KEYS",
    "STAGE1A_KEYS",
    "STAGE1A_SCHEMA",
    "EpochCache",
    "Stage1aMismatchError",
    "blank_seconds",
    "check_stage1a_fresh",
    "content_sha256",
    "core_frames",
    "core_pad_spans",
    "eng_cache_spans",
    "event_from_row",
    "event_row",
    "extent_from_json",
    "extent_json",
    "extents_routing_masks",
    "hull_frames",
    "inference_config",
    "sparse_detection_z",
    "stage1a_keys",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
Pair = tuple[str, str]

CACHE_FLOOR_Z: Final = Z_FLOOR
"""The lowest threshold (own band) and cross-band floor stage 1b reproduces exactly: the clean
null's p90 (1.44), the floor ruling 2026-10-09 (i) 1 never goes below."""

STAGE1A_SCHEMA: Final = "night5-stage1a/1"
"""The stage-1a record's format; stage 1b refuses any other."""

STAGE1A_KEYS: Final[tuple[str, ...]] = (
    "schema", "generation", "feature_version", "cache_floor_z", "choice_file_sha256",
    "model_id", "evaluation_model_id", "decision_rule", "p_threshold", "sensitivity_p",
    "hrv_window_s", "wave_id", "production_routing_sha256", "routing_entry_hash",
    "routing_table", "peri_r_window_sha256",
)
"""Stage 1a's freshness keys: inference inputs only (RULING 2026-10-10 item 4). No extent
threshold, cross-band scope or hold is among them, so changing any of those never re-runs
inference."""

INFERENCE_CONFIG_KEYS: Final[tuple[str, ...]] = (
    "decision_rules", "p_threshold", "sensitivity_p", "hrv_window_s", "wave", "stand_in",
    "model_choice", "no_beat_minutes", "previous_waves",
)
"""The night-config fields stage 1a may read (:func:`inference_config`). Everything else -
``tolerances``, ``cross_band_scope``, extent thresholds, ``min_retention``, the hold - is
stage 1b's or finalize's."""

EXTENT_DEPENDENT: Final[Mapping[str, str]] = {
    "extents": "own-band extents (tolerance.compute_extent): the threshold sets the runs",
    "cross_band": "detected extents (tolerance.detected_extent): the scope picks the pairs and "
                  "the floor sets the runs; only consulted where the own band is under its "
                  "threshold, so the threshold changes which events reach it",
    "routing": "route_event's in-band verdict reads the extent's span (and, for z consumers, "
               "the threshold); decisions, route counts and each event's routes",
    "masks": "every consumer mask (spans_from_routing over the extents)",
    "blanked_time": "blanked time per consumer, the sensitivity table (decision rule and "
                    "calibrated 0.3/0.5/0.7) and the per-scope sensitivity",
    "longest_detected_extent": "the longest cross-band extent (review 7 finding 3)",
    "line_noise_pass1": "pass 1 tests raw T less the spike consumer's motion mask, each blank "
                        "padded 10.5 ms (RULING 2026-10-09 (g) 5): stage 1a caches raw T and "
                        "the peri-R exclusion, stage 1b runs it",
    "line_noise_pass2": "the animal's Holm family is pass 1's p-values (finalize)",
    "qc_hold": "the 20% / 3x-median hold and the 80% retention flag read the masks (finalize)",
    "animal_medians": "the leave-one-out medians of blank fraction (finalize)",
    "mmc_not_measured": "mmc's +/-15 s not-measured spans grow from its blanks (finalize)",
    "spike_time_lost": "line distrust beside motion per cuff (finalize)",
}
"""Exactly what the extent choice (threshold set, cross-band scope) changes."""


class Stage1aMismatchError(ValueError):
    """A stage-1a record does not match this run's inference inputs (or its schema)."""


def inference_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the night config as stage 1a may see it: :data:`INFERENCE_CONFIG_KEYS` only."""
    return {k: cfg[k] for k in INFERENCE_CONFIG_KEYS if k in cfg}


def _canon(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False)


def stage1a_keys(*, generation: str, feature_version: str, choice_file_sha256: str,
                 model_id: str, evaluation_model_id: str | None,
                 decision_rule: Mapping[str, Any], p_threshold: float,
                 sensitivity_p: Sequence[float], hrv_window_s: float, wave_id: str,
                 production_routing_sha256: str, routing_entry_hash: str, routing_table: str,
                 peri_r_window_sha256: str) -> dict[str, Any]:
    """Return a stage-1a record's freshness keys (:data:`STAGE1A_KEYS`), JSON-canonical.

    ``evaluation_model_id`` is None when the recording has no evaluation span (it is then
    absent from the dict, never null).
    """
    out: dict[str, Any] = {
        "schema": STAGE1A_SCHEMA, "generation": str(generation),
        "feature_version": str(feature_version), "cache_floor_z": float(CACHE_FLOOR_Z),
        "choice_file_sha256": str(choice_file_sha256), "model_id": str(model_id),
        "decision_rule": json.loads(_canon(dict(decision_rule))),
        "p_threshold": float(p_threshold),
        "sensitivity_p": sorted(float(x) for x in sensitivity_p),
        "hrv_window_s": float(hrv_window_s), "wave_id": str(wave_id),
        "production_routing_sha256": str(production_routing_sha256),
        "routing_entry_hash": str(routing_entry_hash), "routing_table": str(routing_table),
        "peri_r_window_sha256": str(peri_r_window_sha256)}
    if evaluation_model_id is not None:
        out["evaluation_model_id"] = str(evaluation_model_id)
    return out


def check_stage1a_fresh(have: Mapping[str, Any], want: Mapping[str, Any], *,
                        what: str) -> None:
    """Refuse a stage-1a record whose keys are not this run's; raise naming every difference.

    ``have`` is the record's ``keys``, ``want`` :func:`stage1a_keys` for this run. A key
    present on one side only is a difference (``evaluation_model_id`` is absent when the
    recording has no evaluation span, on both sides).
    """
    if have.get("schema") != STAGE1A_SCHEMA:
        msg = (f"{what}: stage-1a schema {have.get('schema')!r}, this build reads "
               f"{STAGE1A_SCHEMA!r}: rerun stage 1a")
        raise Stage1aMismatchError(msg)
    keys = sorted(set(STAGE1A_KEYS) | set(have) | set(want))
    diff = [k for k in keys if _canon(have.get(k)) != _canon(want.get(k))]
    if diff:
        shown = "; ".join(f"{k}: record {str(have.get(k))[:40]!r}, run {str(want.get(k))[:40]!r}"
                          for k in diff)
        msg = (f"{what}: stage 1a ran under other inference inputs ({shown}) - another wave, "
               "model, rule or window: rerun stage 1a for this run")
        raise Stage1aMismatchError(msg)


def content_sha256(content: Mapping[str, Any], arrays: Mapping[str, npt.ArrayLike]) -> str:
    """sha256 of a record's canonical JSON content and its arrays (name, dtype, shape, bytes).

    Run metadata (timings, memory, finish time) is not content and is kept out by the caller.
    """
    h = hashlib.sha256(_canon(dict(content)).encode("utf-8"))
    for name in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[name]))
        h.update(_canon([name, a.dtype.str, list(a.shape)]).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# events and extents as JSON
# ---------------------------------------------------------------------------


def event_row(event_id: str, event: Event, pairs: Sequence[Pair], *, raw: float
              ) -> dict[str, Any]:
    """One event as stage 1a stores it (non-finite numbers absent, never NaN)."""
    c = event.candidate
    row: dict[str, Any] = {"event_id": event_id, "start": float(c.start_s),
                           "stop": float(c.stop_s), "signals": list(c.signals),
                           "bands": list(c.bands), "peak_z": float(c.peak_z),
                           "pairs": [list(q) for q in pairs], "judgement": event.judgement,
                           "source": event.source}
    if math.isfinite(event.p_motion):
        row["p_motion"] = float(event.p_motion)
    if math.isfinite(raw):
        row["raw"] = float(raw)
    return row


def event_from_row(row: Mapping[str, Any]) -> tuple[Event, tuple[Pair, ...], float]:
    """Return ``(event, detecting pairs, raw P)`` from :func:`event_row`'s form."""
    cand = Candidate(start_s=float(row["start"]), stop_s=float(row["stop"]),
                     signals=tuple(row["signals"]), bands=tuple(row["bands"]),
                     peak_z=float(row["peak_z"]), provenance="electrical")
    ev = Event(candidate=cand, judgement=row["judgement"],
               p_motion=float(row.get("p_motion", math.nan)), source=row["source"])
    pairs = tuple((str(s), str(b)) for s, b in row["pairs"])
    return ev, pairs, float(row.get("raw", math.nan))


_EXTENT_FIELDS: Final = ("consumer", "signal", "band", "start_s", "stop_s", "core_start_s",
                         "core_stop_s", "settling_s", "resolution_s", "basis")


def extent_json(ext: tl.Extent | None) -> dict[str, Any]:
    """Return an extent (or "none": under tolerance / beat train unchanged) as JSON."""
    if ext is None:
        return {"none": True}
    return {"extent": {k: getattr(ext, k) for k in _EXTENT_FIELDS}}


def extent_from_json(doc: Mapping[str, Any]) -> tl.Extent | None:
    """Inverse of :func:`extent_json`."""
    if doc.get("none") is True:
        return None
    e = doc["extent"]
    return tl.Extent(str(e["consumer"]), str(e["signal"]), str(e["band"]),
                     float(e["start_s"]), float(e["stop_s"]), float(e["core_start_s"]),
                     float(e["core_stop_s"]), float(e["settling_s"]),
                     float(e["resolution_s"]), str(e["basis"]))


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


def core_frames(start_s: float, stop_s: float, z_t0_s: float, grid_s: float = GRID_S
                ) -> tuple[int, int]:
    """Return the core's frame window ``[i0, i1)`` as ``tolerance.compute_extent`` takes it."""
    return (int(math.floor((start_s - z_t0_s) / grid_s)),
            int(math.ceil((stop_s - z_t0_s) / grid_s)))


def _runs(over: Bool) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], over.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(),
                    strict=True))


def hull_frames(z: npt.ArrayLike, windows: Iterable[tuple[int, int]], floor: float
                ) -> list[tuple[int, int]]:
    """Return the runs of ``z > floor`` (finite) overlapping any of ``windows`` (``[i0, i1)``)."""
    zz = np.asarray(z, dtype=np.float64)
    runs = _runs(np.isfinite(zz) & (zz > floor))
    w = sorted(windows)
    return [(a, b) for a, b in runs if any(i0 < b and i1 > a for i0, i1 in w)]


def sparse_detection_z(zdet: Mapping[Pair, npt.ArrayLike],
                       windows_by_pair: Mapping[Pair, Sequence[tuple[int, int]]],
                       floor: float = CACHE_FLOOR_Z) -> dict[Pair, F64]:
    """Return detection z kept where the cross-band extent can read it, NaN elsewhere.

    ``windows_by_pair`` maps each pair to the core windows (:func:`core_frames`) of the
    events it detected. Every pair named is returned (all NaN where no run is kept), so a
    detecting pair is never missing from the cache.
    """
    out: dict[Pair, F64] = {}
    for pair, wins in windows_by_pair.items():
        zz = np.asarray(zdet[pair], dtype=np.float64)
        keep = np.zeros(zz.size, dtype=bool)
        for a, b in hull_frames(zz, wins, floor):
            keep[a:b] = True
        out[pair] = np.where(keep, zz, np.nan)
    return out


def eng_cache_spans(cores_s: Iterable[tuple[float, float]], *, trace_t0_s: float,
                    rate: float, size: int, margin: int = 2) -> list[tuple[int, int]]:
    """ENG sample spans covering every span routing can judge: the cores themselves.

    ``routing.route_event`` judges an event on its CANDIDATE span (``EventEvidence.span_s``
    is the core), never on the extent, so what the spike consumer's sample-level verdict
    reads does not depend on any threshold. Each core ``(start_s, stop_s)`` (recording
    timeline) is mapped to samples as ``routing._span_slice`` maps it and widened by
    ``margin`` samples.
    """
    out: list[tuple[int, int]] = []
    for a_s, b_s in cores_s:
        a = int(math.floor((a_s - trace_t0_s) * rate)) - margin
        b = int(math.ceil((b_s - trace_t0_s) * rate)) + margin
        out.append((max(0, a), min(size, b)))
    return out


# ---------------------------------------------------------------------------
# stage 1b
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochCache:
    """One epoch's stage-1a cache, as stage 1b reads it.

    ``events`` holds every core (calibrated P in ``p_motion``), ``detecting`` its detecting
    pairs, ``cached_events`` the events the sparse inputs were built for (every event
    confirmed under any declared variant). ``z`` is the consumers' z (whole), ``zdet`` the
    detection z (:func:`sparse_detection_z`), ``eng`` the spike consumer's ENG traces,
    ``hrv`` each cached event's operational HRV extent (``None``: beat train unchanged),
    ``signals`` what each consumer reads here, ``own_signals`` the ``own_signals_only``
    families, ``extra_spans`` the routing entry's spike distrust. Times on the recording's
    timeline; ``z_t0_s`` is the epoch start.
    """

    events: Mapping[str, Event]
    detecting: Mapping[str, tuple[Pair, ...]]
    cached_events: frozenset[str]
    z: Mapping[Pair, npt.ArrayLike]
    zdet: Mapping[Pair, npt.ArrayLike]
    eng: Mapping[str, rt.EngTrace | rt.CachedEngTrace]
    hrv: Mapping[str, tl.Extent | None]
    signals: Mapping[str, tuple[str, ...]]
    own_signals: Mapping[str, frozenset[str]]
    fs: float
    z_t0_s: float
    n_frames: int
    extra_spans: tuple[mk.MaskSpan, ...] = ()
    cache_floor_z: float = CACHE_FLOOR_Z


CORE_PAD_REASON: Final = "core_pad"
"""The mask reason of RULING 2026-10-10 item 2's core + padding span (never routed)."""


def core_pad_spans(cache: EpochCache, confirmed: Iterable[str], *,
                   core_blanking: Mapping[str, str],
                   tolerances: tl.ToleranceTable | None = None) -> list[mk.MaskSpan]:
    """Return the core + padding spans each z consumer's DECLARED core blanking gives.

    RULING 2026-10-10 item 2 ("the motion core itself and the consumer's padding are always
    blanked"), declared per consumer with no default (``tolerance_map.CORE_BLANKING``):

    * ``always`` - the ruling as written: every confirmed core, widened by the consumer's
      padding, on every signal the consumer reads;
    * ``when_visible`` - only where the consumer's own band is above its threshold over the
      core (some finite frame of the core window, as ``compute_extent`` takes it, above
      ``tolerances.for_consumer``), per signal.

    The padding is the consumer's measured settling (``tolerance.consumer_settling``), the
    span is clipped to the epoch's z record and never routed. hrv keeps its operational
    extent. A z consumer without a declaration, or with another value, is refused.
    """
    out: list[mk.MaskSpan] = []
    end = cache.z_t0_s + cache.n_frames * GRID_S
    for consumer, sigs in cache.signals.items():
        if consumer == "hrv" or not sigs:
            continue
        mode = core_blanking.get(consumer)
        if mode not in ("always", "when_visible"):
            msg = (f"{consumer}: core blanking {mode!r} is not declared as always | "
                   "when_visible (RULING 2026-10-10 item 2; nothing is defaulted)")
            raise ValueError(msg)
        if mode == "when_visible" and tolerances is None:
            msg = f"{consumer}: when_visible needs the consumer's threshold"
            raise ValueError(msg)
        band = tl.extent_consumers()[consumer].band
        settle = tl.consumer_settling(consumer, cache.fs)
        if settle is None:  # pragma: no cover - every extent consumer has a chain
            msg = f"{consumer}: settling unknown, so its padding is unknown (invariant 19)"
            raise ValueError(msg)
        pad = settle.total_s
        for eid in confirmed:
            c = cache.events[eid].candidate
            a, b = max(cache.z_t0_s, c.start_s - pad), min(end, c.stop_s + pad)
            if b <= a:
                continue
            for sig in sigs:
                if mode == "when_visible":
                    assert tolerances is not None
                    zz = np.asarray(cache.z[(sig, band)], dtype=np.float64)
                    i0, i1 = core_frames(c.start_s, c.stop_s, cache.z_t0_s)
                    w = zz[max(0, i0):max(0, min(zz.size, i1))]
                    w = w[np.isfinite(w)]
                    if not (w.size and float(w.max()) > tolerances.for_consumer(consumer)):
                        continue
                out.append(mk.MaskSpan(consumer, sig, a, b, CORE_PAD_REASON))
    return out


def _refuse_inexact(cache: EpochCache, ids: Sequence[str], tolerances: tl.ToleranceTable,
                    scopes: Mapping[str, str] | None, floors: Sequence[float]) -> None:
    """Refuse what the stage-1a cache cannot answer exactly (ValueError, naming it)."""
    missing = sorted(set(ids) - set(cache.cached_events))
    if missing:
        msg = (f"{len(missing)} confirmed event(s) {missing[:3]} are not in the stage-1a cache "
               "(it holds the events confirmed under the declared variants only)")
        raise ValueError(msg)
    if scopes is not None and min(floors) < cache.cache_floor_z:
        msg = (f"cross-band floor {min(floors)} is below the cache floor {cache.cache_floor_z}: "
               "the sparse detection z cannot answer it exactly")
        raise ValueError(msg)
    for name, v in cache.signals.items():
        if name == "hrv" or not v:
            continue
        thr = tolerances.for_consumer(name)
        if thr < cache.cache_floor_z:
            msg = (f"{name}: threshold {thr} is below the cache floor {cache.cache_floor_z}: the "
                   "stage-1a cache is exact only at or above it")
            raise ValueError(msg)


def extents_routing_masks(
    cache: EpochCache, confirmed: Iterable[str], *, tolerances: tl.ToleranceTable,
    scopes: Mapping[str, str] | None, floor_z: float = CACHE_FLOOR_Z,
    floor_by_band: Mapping[str, float] | None = None,
    core_blanking: Mapping[str, str] | None = None,
    route: Callable[[rt.EventEvidence, str, tl.ToleranceTable], rt.RouteDecision] | None = None,
) -> tuple[dict[str, Event], list[tuple[str, str, str, tl.Extent | tl.NotAssessable | None]],
           list[rt.RouteDecision], dict[mk.MaskKey, mk.ConsumerMask]]:
    """Return confirmed events, extents, routing decisions and masks for one threshold set.

    Night 4's stage-1 ``at_rule``, moved here so the cache and the full arrays can be held to
    the same answer. ``confirmed`` are event ids (stage 1a decided them); ``scopes`` each
    consumer's cross-band scope (None: no cross-band rule); ``floor_z`` the detected extent's
    floor, or ``floor_by_band`` one per detecting pair's band (RULING 2026-10-10 item 2: the
    declared clean-null percentile); ``core_blanking`` (each z consumer's declaration, None:
    no core rule) adds :func:`core_pad_spans` (RULING 2026-10-10 item 2). Refused: a confirmed
    event the cache was not built for, and a floor or threshold below the cache's
    (:class:`ValueError`, naming them).
    """
    route = rt.route_event if route is None else route
    ids = list(confirmed)
    floors = [floor_z] if floor_by_band is None else list(floor_by_band.values())
    _refuse_inexact(cache, ids, tolerances, scopes, floors)
    conf = {k: cache.events[k] for k in ids}
    members = {id(e) for e in conf.values()}
    nonhrv = {c: v for c, v in cache.signals.items() if c != "hrv"}
    cross = (tl.CrossBand(pairs={k: cache.detecting[k] for k in conf}, z=cache.zdet,
                          floor_z=floor_z, scope=dict(scopes), own_signals=cache.own_signals,
                          floor_by_band=None if floor_by_band is None else dict(floor_by_band))
             if scopes is not None else None)
    lo = cache.z_t0_s
    ext = tl.extents_for_events(conf, cache.z, nonhrv, tolerances=tolerances, fs=cache.fs,
                                z_t0_s=lo, confirmed=lambda e: id(e) in members,
                                cross_band=cross)
    hrv_sigs = cache.signals.get("hrv") or ()
    if hrv_sigs:
        for eid in conf:
            if eid not in cache.hrv:
                msg = f"event {eid}: no cached HRV extent"
                raise KeyError(msg)
            ext.append((eid, "hrv", hrv_sigs[0], cache.hrv[eid]))
    decisions: list[rt.RouteDecision] = []
    pairs: list[tuple[str, tl.Extent]] = []
    empty = np.empty(0, dtype=np.float64)  # route_event reads x only to subtract (never here)
    for eid, consumer, sig, res in ext:
        if not isinstance(res, tl.Extent):
            continue
        c = conf[eid].candidate
        key = f"{eid}@{sig}"
        if res.basis == tl.EXTENT_BASIS_DETECTED:  # (i) 1: never routed, always blanked
            decisions.append(rt.cross_band_decision(key, consumer))
            pairs.append((key, res))
            continue
        ev = rt.EventEvidence(event_id=key, signal=sig, x=empty, fs=cache.fs,
                              span_s=(c.start_s, c.stop_s), x_t0_s=lo, z=cache.z, z_t0_s=lo,
                              eng=cache.eng.get(sig),
                              hrv_changed=True if consumer == "hrv" else None)
        try:
            dec = route(ev, consumer, tolerances)
        except ValueError as e:
            dec = rt.RouteDecision(key, consumer, "reject",
                                   f"in-band verdict not assessable, rejected: {e}",
                                   "in_band_not_assessable")
        decisions.append(dec)
        pairs.append((key, res))
    spans = mk.spans_from_routing(pairs, decisions) + list(cache.extra_spans)
    if core_blanking is not None:
        spans += core_pad_spans(cache, ids, core_blanking=core_blanking, tolerances=tolerances)
    return conf, ext, decisions, mk.build_masks(cache.signals, spans, n_frames=cache.n_frames,
                                                t0_s=lo)


def blank_seconds(masks: Mapping[mk.MaskKey, mk.ConsumerMask]) -> dict[str, float]:
    """Blanked seconds per ``consumer|signal|band``."""
    return {f"{c}|{s}|{b}": float(m.invalid.sum() * m.grid_s)
            for (c, s, b), m in masks.items()}

