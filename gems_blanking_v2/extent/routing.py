"""Task 14: route each event, per consumer, to the cheapest response that works.

Rejecting data is the most expensive response, so the order is strict:

1. **correct** - the contaminant and the signal are separable in frequency *within that
   consumer's band*: after the band's own filter the event leaves nothing the consumer
   would act on. Each consumer is judged on ITS OWN scale (:func:`in_band_verdict`):

   * ``spikes`` / ``velocity`` - "4.5 sigma, sample level" (A.4): on the 300-3000 Hz
     trace in its own robust sigma, the span must hold no sample above the consumer's
     artifact cap (``P.maxThreshSigma`` = 40 sigma) and no excess of 4.5 sigma crossings
     over the rate outside the span (one-sided Poisson, p < :data:`EXCESS_P`). A 1 ms
     transient at 75 sigma is not "corrected" because it is short.
   * every other consumer - the tolerance table's own scale: the band's log-envelope
     z (``z = (log E - median log E) / (1.4826 MAD log E)``, invariant 5, computed by
     ``bands`` exactly as detection computes it, over the whole of ``x``) must stay at or
     below the consumer's tolerance throughout the span.
   * ``hrv`` - operational: correct iff the beat train is unchanged (task 13's verdict,
     passed in as ``EventEvidence.hrv_changed``).

   A baseline drift is gone from the ENG band after its high-pass, so rejecting it there
   destroys good data for nothing; the same drift sits inside ``0-2`` and is rejected.
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
  (:data:`LINE_NOISE_ACTIONS`). The thresholds that decide "mains-dominant" are fixed
  inputs proposed from the Night 1 hum inventory, set before Night 2 - never defaults.

**Timelines.** ``EventEvidence.span_s`` is on ``x``'s OWN timeline: 0 is ``x[0]``.
``x`` should be the whole epoch, so the band references are whole-epoch scalars.

**NaN** (invariant 8): band filtering interpolates across NaN for the filter only and
restores NaN (at the filtered rate) immediately after; every statistic is NaN-aware and
the emitted residual carries NaN, never a made-up value.

The band filters are ``bands.envelope``'s private ``_band_limit`` / ``_decimate_for``,
imported on purpose: one filter design, shared with detection. ``bands`` is inside the
generation hash, so they cannot be renamed public without changing it;
``test_extent_routing`` pins the import.

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
from scipy.stats import poisson

from gems_blanking_v2.bands.envelope import (
    _band_limit,
    _decimate_for,
    band_envelope_for,
    log_envelope,
)
from gems_blanking_v2.bands.reference import epoch_reference
from gems_blanking_v2.bands.zscore import zscore
from gems_blanking_v2.constants import BANDS, ENG_BAND, GRID_S, MAD_TO_SIGMA
from gems_blanking_v2.extent.grid import frame_sample_bounds, n_grid_frames
from gems_blanking_v2.extent.tolerance import ToleranceTable, extent_consumers

__all__ = [
    "EXCESS_P",
    "LINE_NOISE_ACTIONS",
    "ROUTES",
    "SPIKE_MAX_SIGMA",
    "SPIKE_REFRACTORY_S",
    "SPIKE_THRESH_SIGMA",
    "EventEvidence",
    "LineNoiseThresholds",
    "RouteDecision",
    "band_excess",
    "clip_frames",
    "decisions_table",
    "in_band_verdict",
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

SAMPLE_LEVEL_CONSUMERS: Final[frozenset[str]] = frozenset({"spikes", "velocity"})
SPIKE_THRESH_SIGMA: Final = 4.5
"""``pipeline_params.m`` ``P.threshSigma``: the spike consumer's detection threshold."""
SPIKE_MAX_SIGMA: Final = 40.0
"""``pipeline_params.m`` ``P.maxThreshSigma``: peaks above this are the consumer's own
artifacts."""
SPIKE_REFRACTORY_S: Final = 0.001
"""``pipeline_params.m`` ``P.refractoryMs``."""
EXCESS_P: Final = 0.01
"""One-sided Poisson p below which a span's 4.5 sigma crossings exceed the background
rate. Chosen here, not measured - provisional, for a ruling."""

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
    within 0.1% of it. Computed from the samples alone - no envelope, no model.
    ``rail_uv`` is the amplifier's declared rail in microvolts and ``frac_min`` a fixed
    fraction; both are required. Frames map to samples through ``extent.grid``.
    """
    if not (rail_uv > 0 and 0 < frac_min <= 1):
        msg = f"need rail_uv > 0 and 0 < frac_min <= 1, got {rail_uv}, {frac_min}"
        raise ValueError(msg)
    xx = np.asarray(x, dtype=np.float64)
    at = np.abs(np.abs(xx) - rail_uv) <= rail_uv * rail_margin
    n = n_grid_frames(xx.size, fs, grid_s)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = frame_sample_bounds(i, i + 1, fs, grid_s)
        seg = at[a:b]
        out[i] = seg.size > 0 and float(seg.mean()) >= frac_min
    return out


# ---------------------------------------------------------------------------
# a band, NaN-honest
# ---------------------------------------------------------------------------


def _band(x: F64, fs: float, band: str) -> tuple[F64, float]:
    """Band-limit ``x``; NaN in, NaN out (at the filtered rate). Invariant 8."""
    spec = BANDS[band]
    invalid = ~np.isfinite(x)
    if invalid.all():
        msg = f"band {band}: no finite samples"
        raise ValueError(msg)
    filled = x
    if invalid.any():
        idx = np.arange(x.size, dtype=np.float64)
        filled = x.copy()
        filled[invalid] = np.interp(idx[invalid], idx[~invalid], x[~invalid])
    y, rate = _decimate_for(filled, fs, spec.hi_hz)
    y = np.array(_band_limit(y, rate, spec.lo_hz, spec.hi_hz, band), dtype=np.float64)
    if invalid.any():
        q = int(round(fs / rate))
        starts = np.arange(0, invalid.size, q)
        bad = np.add.reduceat(invalid.astype(np.int64), starts) > 0
        y[bad[: y.size]] = np.nan  # revert: the interpolation was for the filter only
    return y, rate


def _span_slice(span_s: tuple[float, float], rate: float, n: int) -> slice:
    return slice(max(0, int(math.floor(span_s[0] * rate))),
                 min(n, int(math.ceil(span_s[1] * rate))))


def _sigma_outside(y: F64, sl: slice) -> float:
    outside = np.concatenate([y[: sl.start], y[sl.stop:]])
    outside = outside[np.isfinite(outside)]
    if outside.size == 0:
        return math.nan
    return MAD_TO_SIGMA * float(np.median(np.abs(outside - np.median(outside))))


def band_excess(x: npt.ArrayLike, fs: float, span_s: tuple[float, float], band: str
                ) -> tuple[float, float]:
    """``(span RMS / band sigma, band sigma)`` of ``x`` after the band's own filter.

    The band's sigma is the robust (MAD) sigma of the band-limited trace OUTSIDE the
    span - the band's noise floor. NaN-aware throughout.
    """
    y, rate = _band(np.asarray(x, dtype=np.float64), fs, band)
    sl = _span_slice(span_s, rate, y.size)
    sigma = _sigma_outside(y, sl)
    inside = y[sl][np.isfinite(y[sl])]
    if not (sigma > 0 and inside.size):
        msg = f"band {band}: no noise floor or no finite samples in the span"
        raise ValueError(msg)
    return float(np.sqrt(np.mean(inside ** 2))) / sigma, sigma


def _crossings(y: F64, thresh: float, rate: float) -> npt.NDArray[np.int64]:
    """Local maxima of ``|y|`` above ``thresh``, with the consumer's refractory period."""
    a = np.abs(np.where(np.isfinite(y), y, 0.0))
    cand = np.flatnonzero((a[1:-1] > thresh) & (a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:])) + 1
    keep: list[int] = []
    gap = max(1, int(round(SPIKE_REFRACTORY_S * rate)))
    for i in cand:
        if not keep or i - keep[-1] >= gap:
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


@dataclass(frozen=True)
class InBand:
    """What :func:`in_band_verdict` measured: separable or not, and on which scale."""

    separable: bool
    scale: str
    value: float
    limit: float
    detail: str


def in_band_verdict(x: npt.ArrayLike, fs: float, span_s: tuple[float, float], consumer: str,
                    tolerances: ToleranceTable, *, signal: str = "x") -> InBand:
    """Whether what survives the consumer's band filter in the span is below its tolerance."""
    xx = np.asarray(x, dtype=np.float64)
    if consumer in SAMPLE_LEVEL_CONSUMERS:
        y, rate = _band(xx, fs, ENG_BAND)
        sl = _span_slice(span_s, rate, y.size)
        sigma = _sigma_outside(y, sl)
        if not sigma > 0:
            msg = f"{consumer}: no ENG noise floor outside the span"
            raise ValueError(msg)
        peak = float(np.nanmax(np.abs(y[sl]))) / sigma
        cross = _crossings(y, SPIKE_THRESH_SIGMA * sigma, rate)
        n_in = int(((cross >= sl.start) & (cross < sl.stop)).sum())
        t_in = max((sl.stop - sl.start) / rate, 1.0 / rate)
        t_out = max((y.size - (sl.stop - sl.start)) / rate, 1.0 / rate)
        lam = (cross.size - n_in) / t_out
        p = float(poisson.sf(n_in - 1, lam * t_in)) if n_in > 0 else 1.0
        ok = peak <= SPIKE_MAX_SIGMA and p >= EXCESS_P
        return InBand(ok, "eng_sample_sigma", peak, SPIKE_MAX_SIGMA,
                      f"peak {peak:.1f} sigma (cap {SPIKE_MAX_SIGMA:g}); {n_in} crossings of "
                      f"{SPIKE_THRESH_SIGMA:g} sigma vs {lam * t_in:.2f} expected (p {p:.3g})")
    band = extent_consumers()[consumer].band
    tol = tolerances.for_consumer(consumer)
    le = log_envelope(band_envelope_for(xx, fs, band))
    ref = epoch_reference(le, signal=signal, band=band)
    z = zscore(le, ref, signal=signal, band=band)
    i0 = max(0, int(math.floor(span_s[0] / GRID_S)))
    i1 = min(z.size, int(math.ceil(span_s[1] / GRID_S)))
    inside = z[i0:i1][np.isfinite(z[i0:i1])]
    if inside.size == 0:
        msg = f"{consumer}: the {band} log-z is not assessable anywhere in the span"
        raise ValueError(msg)
    zmax = float(inside.max())
    return InBand(zmax <= tol, "log_envelope_z", zmax, tol,
                  f"max {band} log-z {zmax:.2f} vs tolerance {tol:g}")


# ---------------------------------------------------------------------------
# one event, one consumer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventEvidence:
    """What routing needs to know about one event on one consumer's signal.

    ``x`` is the consumer's signal (microvolts), ideally the whole epoch; ``span_s`` the
    event on ``x``'s OWN timeline (0 = ``x[0]``). ``clipped`` comes from
    :func:`clip_frames`, ``mains_dominant`` from :func:`is_mains_dominant`. ``subtract``
    is set only for a stereotyped event with known timing: it returns the signal with the
    event's estimate subtracted. ``hrv_changed`` is task 13's operational verdict, needed
    only to route ``hrv``.
    """

    event_id: str
    x: F64
    fs: float
    span_s: tuple[float, float]
    clipped: bool = False
    mains_dominant: bool = False
    subtract: Callable[[F64], F64] | None = None
    hrv_changed: bool | None = None


@dataclass(frozen=True)
class RouteDecision:
    """The route for one event on one consumer, and the numbers that decided it."""

    event_id: str
    consumer: str
    route: Route
    reason: str
    reason_code: str = ""
    in_band_value: float = math.nan
    residual_ratio: float = math.nan
    action: str = ""
    residual: F64 | None = field(default=None, repr=False, compare=False)

    @property
    def masks(self) -> bool:
        """Whether this decision masks the event's span for this consumer."""
        return self.route in MASKING_ROUTES


def route_event(ev: EventEvidence, consumer: str, tolerances: ToleranceTable  # noqa: PLR0911
                ) -> RouteDecision:
    """Route one event for one consumer, in the strict order of this module."""
    if ev.clipped:
        return RouteDecision(ev.event_id, consumer, "clip",
                             "samples at the rail: masked without a model decision", "clip")
    if ev.mains_dominant:
        return RouteDecision(ev.event_id, consumer, "line_noise",
                             "dominant evidence is mains: never rejected (ruling (c) 4)",
                             "line_noise", action=LINE_NOISE_ACTIONS[consumer])
    if consumer == "hrv":
        if ev.hrv_changed is None:
            msg = "hrv routes from task 13's operational verdict; hrv_changed was not given"
            raise ValueError(msg)
        if not ev.hrv_changed:
            return RouteDecision(ev.event_id, consumer, "correct",
                                 "beat train unchanged by the event", "beat_train_unchanged")
        return RouteDecision(ev.event_id, consumer, "reject", "beat train changed",
                             "beat_train_changed")
    verdict = in_band_verdict(ev.x, ev.fs, ev.span_s, consumer, tolerances)
    if verdict.separable:
        return RouteDecision(ev.event_id, consumer, "correct", verdict.detail,
                             f"within_{verdict.scale}", in_band_value=verdict.value)
    if ev.subtract is not None:
        band = extent_consumers()[consumer].band
        _ratio, sigma = band_excess(ev.x, ev.fs, ev.span_s, band)
        cleaned = np.asarray(ev.subtract(np.asarray(ev.x, dtype=np.float64)), dtype=np.float64)
        y, rate = _band(cleaned, ev.fs, band)
        residual = y[_span_slice(ev.span_s, rate, y.size)]
        fin = residual[np.isfinite(residual)]
        # What is left of the EVENT: the span's power above the band's own noise power.
        # A perfect subtraction leaves noise (RMS = sigma, remnant 0), so the remnant -
        # not the raw RMS, which is ~1 sigma even when nothing is left - is what has to
        # be below the floor.
        rms = float(np.sqrt(np.mean(fin ** 2))) if fin.size else math.inf
        res_ratio = math.sqrt(max(0.0, rms ** 2 - sigma ** 2)) / sigma
        if res_ratio < 1.0:
            return RouteDecision(ev.event_id, consumer, "subtract",
                                 f"residual {res_ratio:.2f} x the band floor (< 1): verified",
                                 "subtract_verified", in_band_value=verdict.value,
                                 residual_ratio=res_ratio, residual=residual)
        return RouteDecision(ev.event_id, consumer, "reject",
                             f"subtraction residual {res_ratio:.2f} x the band floor (>= 1): "
                             "verification failed, fell back to reject",
                             "subtract_failed", in_band_value=verdict.value,
                             residual_ratio=res_ratio, residual=residual)
    return RouteDecision(ev.event_id, consumer, "reject",
                         f"in band and not stereotyped: {verdict.detail}",
                         f"in_band_{verdict.scale}", in_band_value=verdict.value)


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
    """Return JSON-ready rows (missing numbers absent, never NaN), one per event per consumer."""
    rows = []
    for d in decisions:
        row: dict[str, Any] = {"event_id": d.event_id, "consumer": d.consumer,
                               "route": d.route, "reason": d.reason,
                               "reason_code": d.reason_code}
        for k in ("in_band_value", "residual_ratio"):
            v = getattr(d, k)
            if math.isfinite(v):
                row[k] = v
        if d.action:
            row["action"] = d.action
        rows.append(row)
    return rows
