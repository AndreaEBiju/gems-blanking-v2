"""Task 15: QC per recording - retention, the gates, the report, the longitudinal table.

Two gates, both refusing to emit a heavily masked recording silently:

* **Retention gate** (task 15): if any consumer's retention on any signal falls below a
  threshold, the whole recording is flagged rather than emitted as if it were fine. The
  threshold is a required argument - no number is chosen here.
* **Blank-fraction hold** (ruling 2026-10-07 (c) item 5): a recording whose blank
  exceeds :data:`HOLD_BLANK_FRACTION` (20%), or :data:`HOLD_MEDIAN_RATIO` (3x) its
  animal's median, is held - listed for Andrea with its top reasons and the hum features
  - and she releases or overrides it. "Total blank" is taken per consumer mask (masks
  are never merged, invariant 2): the recording is held if ANY consumer's blank does
  either. Medians are keyed by :func:`median_key`: ``consumer|signal|band``, except
  ``hrv`` and ``breathing``, whose signal is whichever channel the HR ranking chose in
  that recording, so they are keyed ``consumer|band``. A median is floored at
  :data:`HOLD_MEDIAN_FLOOR` before the 3x rule (otherwise an animal whose median is 0
  holds every recording with any blank). A median that is unknown (``None``, e.g. the
  animal's first recording) applies the 20% rule alone, and the report ALWAYS says so,
  held or not.

The two gates combine into :func:`emit_gate`, which ``emit.handoff.write_mask_file``
computes from the masks and enforces: a held recording is written only with an explicit
release. A cuff distrusted outright by the routing table (retention 0 for the spike
consumer on that cuff) therefore always holds its recording: a ruled distrust counts as
blank. Whether it should is a question for a ruling (it is not motion).

The QC report (:class:`QcReport`) carries what the spec lists; a quantity a recording
does not have (velocity windows while task 18 is out, R5) is ABSENT from its record,
never ``null`` or ``NaN``. The per-animal longitudinal table (tripole weights, peri-R
template amplitude, noise floor per session) is appended one canonical JSON line per
session to a file the caller owns (:func:`append_longitudinal`).

**Spike-consumer time lost per cuff (RULING 2026-10-08 (d) 3)** -
:func:`spike_time_lost` - reports the time the line-distrust rule (``emit.line_distrust``)
takes from the spike consumer beside the time its mask blanks, per spike signal. The
line distrust is NOT blank: it is not in the masks, so it never enters the gates above.
The handoff writes it into ``gate_json`` (``spike_time_lost``), and
:func:`spike_time_lost_by_animal` sums it per animal x cuff - the ruling's cost table.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from gems_blanking_v2.emit.line_distrust import LineDistrustRecord
from gems_blanking_v2.emit.masks import ConsumerMask, MaskKey, mask_frames
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.io.store import append_line

__all__ = [
    "HOLD_BLANK_FRACTION",
    "HOLD_MEDIAN_FLOOR",
    "HOLD_MEDIAN_RATIO",
    "BlankHold",
    "EmitGate",
    "LongitudinalRow",
    "QcReport",
    "RetentionVerdict",
    "append_longitudinal",
    "blank_fraction_by_band",
    "blank_fraction_hold",
    "emit_gate",
    "median_key",
    "retention_by_key",
    "retention_gate",
    "spike_time_lost",
    "spike_time_lost_by_animal",
]

HOLD_BLANK_FRACTION: Final = 0.20
"""Ruling (c) item 5: a blank above 20% of the recording is held for Andrea."""
HOLD_MEDIAN_RATIO: Final = 3.0
"""Ruling (c) item 5: a blank above 3x the animal's median is held for Andrea."""
HOLD_MEDIAN_FLOOR: Final = 0.01
"""The animal median is floored at 1% before the 3x rule: below that, 3x is under 3% of
the recording, which is not the outlier the rule exists to catch. Chosen here, not
ruled - provisional."""
HR_KEYED_BY_BAND: Final[frozenset[str]] = frozenset({"hrv", "breathing"})


def median_key(consumer: str, signal: str, band: str) -> str:
    """Return the key an animal median is stored under (see the module docstring)."""
    if consumer in HR_KEYED_BY_BAND:
        return f"{consumer}|{band}"
    return f"{consumer}|{signal}|{band}"


def _key(k: MaskKey) -> str:
    return "|".join(k)


def retention_by_key(masks: Mapping[MaskKey, ConsumerMask]) -> dict[str, float]:
    """Retention per ``consumer|signal|band``."""
    return {_key(k): m.retention for k, m in sorted(masks.items())}


@dataclass(frozen=True)
class RetentionVerdict:
    """Whether the recording is flagged, and which masks fell below the threshold."""

    flagged: bool
    min_retention: float
    below: Mapping[str, float]


def retention_gate(masks: Mapping[MaskKey, ConsumerMask], *, min_retention: float
                   ) -> RetentionVerdict:
    """Flag the whole recording if any mask keeps less than ``min_retention`` of its frames."""
    if not 0.0 < min_retention <= 1.0:
        msg = f"min_retention must be in (0, 1], got {min_retention}"
        raise ValueError(msg)
    below = {k: r for k, r in retention_by_key(masks).items() if r < min_retention}
    return RetentionVerdict(bool(below), min_retention, below)


@dataclass(frozen=True)
class BlankHold:
    """The blank-fraction hold: held or not, why, and what Andrea sees."""

    held: bool
    reasons: tuple[str, ...]
    blank_fraction: Mapping[str, float]
    top_routes: tuple[tuple[str, int], ...] = ()
    hum_features: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def blank_fraction_hold(
    masks: Mapping[MaskKey, ConsumerMask], animal_median: Mapping[str, float | None], *,
    decisions: Iterable[RouteDecision] = (), hum_features: Mapping[str, float] | None = None,
) -> BlankHold:
    """Hold the recording if any consumer mask blanks > 20% or > 3x the animal's median.

    ``animal_median`` maps :func:`median_key` to the animal's median blank fraction for
    that mask, or ``None`` when unknown. A median that is NaN, infinite or outside [0, 1]
    raises naming its key: ``max(nan, floor)`` is NaN and ``f > 3 * nan`` is always
    False, so a NaN would silently switch the 3x rule off. Top reasons are counted by
    routing reason code.
    """
    for k, h in (hum_features or {}).items():
        if not math.isfinite(float(h)):
            msg = f"hum feature {k} is {h!r}: a feature shown to Andrea must be finite"
            raise ValueError(msg)
    for k, v in animal_median.items():
        if v is not None and not (math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0):
            msg = (f"animal median for {k} is {v!r}: a blank fraction must be finite and in "
                   "[0, 1] (use None for unknown)")
            raise ValueError(msg)
    frac = {_key(k): 1.0 - m.retention for k, m in sorted(masks.items())}
    reasons: list[str] = []
    unknown: list[str] = []
    for (consumer, signal, band), m in sorted(masks.items()):
        f = 1.0 - m.retention
        k = _key((consumer, signal, band))
        if f > HOLD_BLANK_FRACTION:
            reasons.append(f"{k}: {100 * f:.1f}% blanked > {100 * HOLD_BLANK_FRACTION:.0f}%")
        mk = median_key(consumer, signal, band)
        med = animal_median.get(mk)
        if med is None:
            unknown.append(mk)
            continue
        floor = max(float(med), HOLD_MEDIAN_FLOOR)
        if f > HOLD_MEDIAN_RATIO * floor:
            reasons.append(f"{k}: {100 * f:.1f}% blanked > {HOLD_MEDIAN_RATIO:g} x the "
                           f"animal median {100 * floor:.1f}%")
    held = bool(reasons)
    notes = tuple(f"animal median unknown for {mk}: the 3x rule was not applied"
                  for mk in sorted(set(unknown)))
    routes = Counter(d.reason_code or d.route for d in decisions if d.masks)
    return BlankHold(held, tuple(reasons), frac, tuple(routes.most_common(5)),
                     dict(hum_features or {}), notes)


@dataclass(frozen=True)
class EmitGate:
    """QC's verdict for one recording, as ``emit.handoff.write_mask_file`` computes it."""

    held: bool
    reasons: tuple[str, ...]
    retention_flagged: bool
    blank_held: bool
    min_retention: float
    medians_used: Mapping[str, float]
    notes: tuple[str, ...]
    top_routes: tuple[tuple[str, int], ...] = ()
    hum_features: Mapping[str, float] = field(default_factory=dict)


def emit_gate(masks: Mapping[MaskKey, ConsumerMask], *, min_retention: float,
              animal_median: Mapping[str, float | None],
              decisions: Iterable[RouteDecision] = (),
              hum_features: Mapping[str, float] | None = None) -> EmitGate:
    """Compute both gates from ``masks`` and combine them (what the writer enforces)."""
    hold = blank_fraction_hold(masks, animal_median, decisions=decisions,
                               hum_features=hum_features)
    retention = retention_gate(masks, min_retention=min_retention)
    reasons = list(hold.reasons)
    if retention.flagged:
        reasons += [f"{k}: retention {r:.3f} < {retention.min_retention:g}"
                    for k, r in sorted(retention.below.items())]
    used = {k: float(v) for k, v in animal_median.items() if v is not None}
    return EmitGate(hold.held or retention.flagged, tuple(reasons), retention.flagged,
                    hold.held, min_retention, used, hold.notes, hold.top_routes,
                    dict(hold.hum_features))


def _present(d: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        value = _present(v) if isinstance(v, Mapping) else v
        if isinstance(v, Mapping) and not value:
            continue
        out[k] = value
    return out


@dataclass(frozen=True)
class QcReport:
    """Per-recording QC, the spec's list. Missing quantities are ``None`` -> absent."""

    recording: str
    candidate_count: int
    blank_fraction_by_band: Mapping[str, float]
    retention: Mapping[str, float]
    retention_flagged: bool
    held: bool
    hold_reasons: Sequence[str] = ()
    route_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    rpeak_gap_fraction: float | None = None
    best_hr_channel: str | None = None
    tripole_weights: Mapping[str, Sequence[float]] = field(default_factory=dict)
    sustained_review_queue: Sequence[Mapping[str, float]] = ()
    low_confidence_velocity_windows: Sequence[Mapping[str, float]] | None = None
    rescue_rate_by_channel: Mapping[str, float] = field(default_factory=dict)
    resolution_s_by_band: Mapping[str, float] = field(default_factory=dict)
    mmc_not_measured_fraction: Mapping[str, float] = field(default_factory=dict)
    hold_notes: Sequence[str] = ()
    spike_time_lost: Mapping[str, Mapping[str, float | int]] = field(default_factory=dict)
    """:func:`spike_time_lost`: per spike signal, mask blank vs line distrust (ruling (d) 3)."""

    def to_record(self) -> dict[str, Any]:
        """JSON-ready, with absent keys for anything missing."""
        rec = asdict(self)
        rec["hold_reasons"] = list(self.hold_reasons)
        rec["hold_notes"] = list(self.hold_notes)
        rec["sustained_review_queue"] = [dict(x) for x in self.sustained_review_queue]
        if self.low_confidence_velocity_windows is None:
            rec.pop("low_confidence_velocity_windows")
        return _present(rec)

    def to_json(self) -> str:
        """Canonical, ASCII-escaped JSON."""
        return json.dumps(self.to_record(), sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False)


def blank_fraction_by_band(masks: Mapping[MaskKey, ConsumerMask]) -> dict[str, float]:
    """Mean blank fraction per band, across the masks in that band (a QC summary only)."""
    by: dict[str, list[float]] = {}
    for (_c, _s, band), m in masks.items():
        by.setdefault(band, []).append(1.0 - m.retention)
    return {b: float(sum(v) / len(v)) for b, v in sorted(by.items())}


def spike_time_lost(masks: Mapping[MaskKey, ConsumerMask], line_distrust: LineDistrustRecord
                    ) -> dict[str, dict[str, float | int]]:
    """Spike-consumer time lost per spike signal: its mask's blank beside the line distrust.

    Counted in the mask's own 10 ms frames, so the three times add up on one grid:
    ``mask_blank_s`` (frames the spike mask blanks - motion and ruled cuff distrust),
    ``line_distrust_s`` (frames overlapping a distrusted minute), ``line_distrust_only_s``
    (distrusted and not already blanked) and ``total_lost_s`` (either), each also as a
    fraction of ``epoch_s``; plus the record's minute counts. The record must cover
    exactly the spike masks' signals.
    """
    spike = {sig: m for (c, sig, _b), m in masks.items() if c == "spikes"}
    if set(spike) != set(line_distrust.signals):
        msg = (f"the line-distrust record covers {sorted(line_distrust.signals)} but the spike "
               f"masks are {sorted(spike)}")
        raise ValueError(msg)
    out: dict[str, dict[str, float | int]] = {}
    for sig, m in sorted(spike.items()):
        n = m.invalid.size
        epoch_s = n * m.grid_s
        ld = mask_frames(line_distrust.distrusted_spans(sig), n, t0_s=m.t0_s, grid_s=m.grid_s)
        row: dict[str, float | int] = {
            "epoch_s": epoch_s,
            "mask_blank_s": float(m.invalid.sum()) * m.grid_s,
            "line_distrust_s": float(ld.sum()) * m.grid_s,
            "line_distrust_only_s": float((ld & ~m.invalid).sum()) * m.grid_s,
            "total_lost_s": float((ld | m.invalid).sum()) * m.grid_s}
        _add_fractions(row)
        row.update(line_distrust.counts(sig))
        out[sig] = row
    return out


_LOST_SECONDS: Final = ("mask_blank_s", "line_distrust_s", "line_distrust_only_s",
                        "total_lost_s")
_LOST_SUMMED: Final = ("epoch_s", *_LOST_SECONDS, "minutes_tested", "minutes_untested",
                       "minutes_distrusted")


def _add_fractions(row: dict[str, float | int]) -> None:
    """Each lost time as a fraction of ``epoch_s``; absent (not NaN) for an empty epoch."""
    epoch_s = float(row["epoch_s"])
    for k in _LOST_SECONDS:
        if epoch_s > 0:
            row[k.removesuffix("_s") + "_frac"] = float(row[k]) / epoch_s


def spike_time_lost_by_animal(
    rows: Iterable[tuple[str, str, Mapping[str, Mapping[str, float | int]]]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Ruling (d) 3's cost table: spike-consumer time lost per animal and cuff.

    ``rows`` are ``(animal, recording, spike_time_lost(...))`` per recording. Seconds and
    minute counts are summed per ``animal`` x spike signal (one per cuff), with
    ``n_recordings``; fractions are recomputed from the sums (time-weighted, never a mean
    of fractions), so the line-distrust loss sits beside the mask's (motion) loss. A
    recording given twice for one animal raises: a duplicate would count its time twice.
    """
    seen: set[tuple[str, str]] = set()
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for animal, recording, lost in rows:
        if (animal, recording) in seen:
            msg = f"{animal} {recording}: given twice; its time would be counted twice"
            raise ValueError(msg)
        seen.add((animal, recording))
        for sig, row in lost.items():
            agg = out.setdefault(animal, {}).setdefault(
                sig, dict.fromkeys(("n_recordings", *_LOST_SUMMED), 0))
            agg["n_recordings"] = int(agg["n_recordings"]) + 1
            for k in _LOST_SUMMED:
                agg[k] = agg[k] + row[k]
    for by_sig in out.values():
        for agg in by_sig.values():
            _add_fractions(agg)
    return {a: dict(sorted(s.items())) for a, s in sorted(out.items())}


@dataclass(frozen=True)
class LongitudinalRow:
    """Electrode drift per session: tripole weights, peri-R template, noise floor."""

    animal: str
    session: str
    acquired_at: str
    tripole_a: Mapping[str, float]
    tripole_b: Mapping[str, float]
    peri_r_template_uv: Mapping[str, float]
    noise_floor_uv: Mapping[str, float]


def append_longitudinal(path: Path, row: LongitudinalRow) -> Path:
    """Append one session's row to an animal's longitudinal table (one JSON line).

    The file is per animal and owned by the writer (append is the only mutation the
    store's design allows, on a file the local user owns exclusively). Refuses a row for
    another animal than the rows already there.
    """
    path = Path(path)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("animal") != row.animal:
                msg = f"{path.name} holds another animal's rows; one table per animal"
                raise ValueError(msg)
    append_line(path, json.dumps(_present(asdict(row)), sort_keys=True, ensure_ascii=True,
                                 separators=(",", ":"), allow_nan=False))
    return path
