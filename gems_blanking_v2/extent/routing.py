"""Task 14: route each event, per consumer, to the cheapest response that works.

Rejecting data is the most expensive response, so the order is strict:

1. **correct** - the contaminant and the signal are separable in frequency *within that
   consumer's band*: after the band's own filter, what is left of the event in the span
   is at or below the consumer's tolerance on the band's robust sigma. A baseline drift
   is gone from the ENG band after its high-pass, so rejecting it there destroys good
   data for nothing; the same drift sits inside ``0-2`` and is not corrected there.
2. **subtract** - stereotyped with known timing: a subtraction is attempted and its
   residual is VERIFIED - emitted, and the event's remnant (the span's band power above
   the band's noise power, as an amplitude) required to be below the band's noise
   floor - or the route falls back to reject.
3. **reject** - only where the disturbance overlaps the signal in both time and
   frequency.

Two routes sit outside that order:

- **clipping bypasses the classifier.** Saturation is non-linear and the envelope can
  understate it, so frames whose fraction of samples at the rail reaches a fixed
  fraction are masked directly (:func:`clip_frames`), before and independent of any
  model decision: :func:`route_events` never hands a clipped event to the classifier.
- **line noise never routes to reject** (ruling 2026-10-07 (c) item 4). A core whose
  dominant evidence is mains goes to ``line_noise`` with a per-consumer action
  (:data:`LINE_NOISE_ACTIONS`): the spike consumer distrusts that cuff per minute where
  the mains-locked spike fraction exceeds the threshold, HR keeps the hum-lock
  persistence test, the stomach consumers keep the ANT1 notch rule, and every other
  consumer keeps the minute. The thresholds that decide "mains-dominant" are fixed
  inputs proposed from the Night 1 hum inventory, set before Night 2 - never defaults.

Every decision is recorded per event per consumer (:class:`RouteDecision`), and
:func:`route_counts` gives the counts by route.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.bands.envelope import _band_limit, _decimate_for
from gems_blanking_v2.constants import BANDS, GRID_S, MAD_TO_SIGMA
from gems_blanking_v2.extent.tolerance import ToleranceTable, extent_consumers

__all__ = [
    "LINE_NOISE_ACTIONS",
    "ROUTES",
    "EventEvidence",
    "LineNoiseThresholds",
    "RouteDecision",
    "band_excess",
    "clip_frames",
    "decisions_table",
    "is_mains_dominant",
    "route_counts",
    "route_event",
    "route_events",
]

F64 = npt.NDArray[np.float64]
Route = Literal["clip", "line_noise", "correct", "subtract", "reject"]

ROUTES: Final[tuple[str, ...]] = ("clip", "line_noise", "correct", "subtract", "reject")
MASKING_ROUTES: Final[frozenset[str]] = frozenset({"clip", "reject"})
"""Routes whose span is masked for the consumer. ``correct`` and a verified ``subtract``
keep the data; ``line_noise`` acts per consumer (:data:`LINE_NOISE_ACTIONS`)."""

LINE_NOISE_ACTIONS: Final[Mapping[str, str]] = {
    "spikes": "per_minute_cuff_distrust",
    "velocity": "keep_minute",
    "hrv": "hum_lock_persistence_test",
    "breathing": "hum_lock_persistence_test",
    "mmc": "ant1_notch_rule",
    "slow_wave": "ant1_notch_rule",
}
"""Ruling (c) item 4: what each consumer does with a mains-dominant core. None rejects."""


@dataclass(frozen=True, slots=True)
class LineNoiseThresholds:
    """Fixed thresholds deciding "dominant evidence is mains" (ruling (c) item 4).

    Proposed by the build from the Night 1 hum inventory and written down before Night
    2; supplied here, never defaulted. A core is mains-dominant when its line ratio and
    its line phase-locking value both reach their thresholds.
    """

    line_ratio_min: float
    line_plv_min: float
    source: str

    def __post_init__(self) -> None:
        """Refuse thresholds without a source."""
        if not self.source:
            msg = "line-noise thresholds must name their source (the hum inventory)"
            raise ValueError(msg)


def is_mains_dominant(features: Mapping[str, float], th: LineNoiseThresholds) -> bool:
    """Whether a core's mains features reach both fixed thresholds (task 11's names)."""
    ratio = float(features.get("line_ratio_max", math.nan))
    plv = float(features.get("line_plv_max", math.nan))
    return bool(ratio >= th.line_ratio_min and plv >= th.line_plv_min)


# ---------------------------------------------------------------------------
# clipping
# ---------------------------------------------------------------------------


def clip_frames(x: npt.ArrayLike, fs: float, *, rail_uv: float, frac_min: float,
                grid_s: float = GRID_S, rail_margin: float = 1e-3) -> npt.NDArray[np.bool_]:
    """Frames whose fraction of samples at the rail reaches ``frac_min``.

    A sample is at the rail when ``| |x| - rail_uv | <= rail_uv * rail_margin`` - a
    saturated amplifier holds the rail value, and an unclipped trace almost never lands
    within 0.1% of it. Computed from
    the samples alone - no envelope, no model. ``rail_uv`` is the amplifier's declared
    rail in microvolts and ``frac_min`` a fixed fraction; both are required.
    """
    if not (rail_uv > 0 and 0 < frac_min <= 1):
        msg = f"need rail_uv > 0 and 0 < frac_min <= 1, got {rail_uv}, {frac_min}"
        raise ValueError(msg)
    xx = np.asarray(x, dtype=np.float64)
    at = np.abs(np.abs(xx) - rail_uv) <= rail_uv * rail_margin
    n = int(math.floor(xx.size / fs / grid_s))
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = int(math.ceil(i * grid_s * fs)), int(math.ceil((i + 1) * grid_s * fs))
        seg = at[a:b]
        out[i] = seg.size > 0 and float(seg.mean()) >= frac_min
    return out


# ---------------------------------------------------------------------------
# frequency separability within a band
# ---------------------------------------------------------------------------


def _band(x: F64, fs: float, band: str) -> tuple[F64, float]:
    spec = BANDS[band]
    finite = np.isfinite(x)
    if not finite.all():
        idx = np.arange(x.size, dtype=np.float64)
        x = x.copy()
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    y, rate = _decimate_for(x, fs, spec.hi_hz)
    return _band_limit(y, rate, spec.lo_hz, spec.hi_hz, band), rate


def band_excess(x: npt.ArrayLike, fs: float, span_s: tuple[float, float], band: str
                ) -> tuple[float, float]:
    """``(span RMS / band sigma, band sigma)`` of ``x`` after the band's own filter.

    The band's sigma is the robust (MAD) sigma of the band-limited trace OUTSIDE the
    span - the band's noise floor. The ratio says how much of the event survives the
    band's filter, in that floor's units. The filter design is the detection side's
    (``bands.envelope``), one definition.
    """
    y, rate = _band(np.asarray(x, dtype=np.float64), fs, band)
    a = max(0, int(math.floor(span_s[0] * rate)))
    b = min(y.size, int(math.ceil(span_s[1] * rate)))
    inside = y[a:b]
    outside = np.concatenate([y[:a], y[b:]])
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(outside - np.median(outside))))
    if not (sigma > 0 and inside.size):
        msg = f"band {band}: no noise floor or empty span"
        raise ValueError(msg)
    return float(np.sqrt(np.mean(inside ** 2))) / sigma, sigma


# ---------------------------------------------------------------------------
# one event, one consumer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventEvidence:
    """What routing needs to know about one event on one consumer's signal.

    ``x`` is the consumer's signal (microvolts) around the event; ``span_s`` the event
    on ``x``'s timeline. ``clipped`` comes from :func:`clip_frames`. ``mains_dominant``
    from :func:`is_mains_dominant`. ``subtract`` is set only for a stereotyped event with
    known timing: it returns the signal with the event's estimate subtracted.
    """

    event_id: str
    x: F64
    fs: float
    span_s: tuple[float, float]
    clipped: bool = False
    mains_dominant: bool = False
    subtract: Callable[[F64], F64] | None = None


@dataclass(frozen=True)
class RouteDecision:
    """The route for one event on one consumer, and the numbers that decided it."""

    event_id: str
    consumer: str
    route: Route
    reason: str
    in_band_ratio: float = math.nan
    residual_ratio: float = math.nan
    action: str = ""
    residual: F64 | None = field(default=None, repr=False, compare=False)

    @property
    def masks(self) -> bool:
        """Whether this decision masks the event's span for this consumer."""
        return self.route in MASKING_ROUTES


def route_event(ev: EventEvidence, consumer: str, tolerances: ToleranceTable
                ) -> RouteDecision:
    """Route one event for one consumer, in the strict order of this module."""
    spec = extent_consumers()[consumer]
    if ev.clipped:
        return RouteDecision(ev.event_id, consumer, "clip",
                             "samples at the rail: masked without a model decision")
    if ev.mains_dominant:
        return RouteDecision(ev.event_id, consumer, "line_noise",
                             "dominant evidence is mains: never rejected (ruling (c) 4)",
                             action=LINE_NOISE_ACTIONS[consumer])
    tol = tolerances.for_consumer(consumer)
    ratio, sigma = band_excess(ev.x, ev.fs, ev.span_s, spec.band)
    if ratio <= tol:
        return RouteDecision(ev.event_id, consumer, "correct",
                             f"after the {spec.band} Hz filter the span is {ratio:.2f} x the "
                             f"band floor, within tolerance {tol:g}", in_band_ratio=ratio)
    if ev.subtract is not None:
        cleaned = np.asarray(ev.subtract(np.asarray(ev.x, dtype=np.float64)), dtype=np.float64)
        y, rate = _band(cleaned, ev.fs, spec.band)
        a = max(0, int(math.floor(ev.span_s[0] * rate)))
        b = min(y.size, int(math.ceil(ev.span_s[1] * rate)))
        residual = y[a:b]
        # What is left of the EVENT: the span's power above the band's own noise power.
        # A perfect subtraction leaves noise (RMS = sigma, remnant 0), so the remnant -
        # not the raw RMS, which is ~1 sigma even when nothing is left - is what has to
        # be below the floor.
        rms = float(np.sqrt(np.mean(residual ** 2)))
        res_ratio = math.sqrt(max(0.0, rms ** 2 - sigma ** 2)) / sigma
        if res_ratio < 1.0:
            return RouteDecision(ev.event_id, consumer, "subtract",
                                 f"residual {res_ratio:.2f} x the band floor (< 1): verified",
                                 in_band_ratio=ratio, residual_ratio=res_ratio,
                                 residual=residual)
        return RouteDecision(ev.event_id, consumer, "reject",
                             f"subtraction residual {res_ratio:.2f} x the band floor (>= 1): "
                             "verification failed, fell back to reject",
                             in_band_ratio=ratio, residual_ratio=res_ratio, residual=residual)
    return RouteDecision(ev.event_id, consumer, "reject",
                         f"in band ({ratio:.2f} x floor > {tol:g}) and not stereotyped",
                         in_band_ratio=ratio)


def route_events(
    evidence: Sequence[EventEvidence], consumers: Iterable[str], tolerances: ToleranceTable,
    *, classify: Callable[[EventEvidence], bool],
) -> list[RouteDecision]:
    """Route every event for every consumer.

    ``classify`` is the motion classifier's decision for an event (True = motion). It is
    called only for events that are not clipped: a clipped event is masked for every
    consumer before, and independent of, any model decision. An event the classifier
    rejects is not routed (it is not motion); an event it confirms is routed per
    consumer.
    """
    out: list[RouteDecision] = []
    names = list(consumers)
    for ev in evidence:
        if ev.clipped:
            out += [route_event(ev, c, tolerances) for c in names]
            continue
        if not classify(ev):
            continue
        out += [route_event(ev, c, tolerances) for c in names]
    return out


def route_counts(decisions: Iterable[RouteDecision]) -> dict[str, dict[str, int]]:
    """Count decisions by route, per consumer: ``{consumer: {route: n}}``, every route listed."""
    by: dict[str, Counter[str]] = {}
    for d in decisions:
        by.setdefault(d.consumer, Counter())[d.route] += 1
    return {c: {r: int(n.get(r, 0)) for r in ROUTES} for c, n in sorted(by.items())}


def decisions_table(decisions: Iterable[RouteDecision]) -> list[dict[str, Any]]:
    """JSON-ready rows (missing numbers absent, never NaN), one per event per consumer."""
    rows = []
    for d in decisions:
        row: dict[str, Any] = {"event_id": d.event_id, "consumer": d.consumer,
                               "route": d.route, "reason": d.reason}
        for k in ("in_band_ratio", "residual_ratio"):
            v = getattr(d, k)
            if math.isfinite(v):
                row[k] = v
        if d.action:
            row["action"] = d.action
        rows.append(row)
    return rows
