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

The two gates combine into :func:`emit_gate`, which ``emit.masks.write_mask_file``
enforces: a held recording is written only with an explicit release.

The QC report (:class:`QcReport`) carries what the spec lists; a quantity a recording
does not have (velocity windows while task 18 is out, R5) is ABSENT from its record,
never ``null`` or ``NaN``. The per-animal longitudinal table (tripole weights, peri-R
template amplitude, noise floor per session) is appended one canonical JSON line per
session to a file the caller owns (:func:`append_longitudinal`).

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

from gems_blanking_v2.emit.masks import ConsumerMask, EmitGate, MaskKey
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.io.store import append_line

__all__ = [
    "HOLD_BLANK_FRACTION",
    "HOLD_MEDIAN_FLOOR",
    "HOLD_MEDIAN_RATIO",
    "BlankHold",
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
    that mask, or ``None`` when unknown. Top reasons are counted by routing reason code.
    """
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


def emit_gate(hold: BlankHold, retention: RetentionVerdict) -> EmitGate:
    """Combine the two gates into what ``write_mask_file`` enforces."""
    reasons = list(hold.reasons)
    if retention.flagged:
        reasons += [f"{k}: retention {r:.3f} < {retention.min_retention:g}"
                    for k, r in sorted(retention.below.items())]
    return EmitGate(hold.held or retention.flagged, tuple(reasons), retention.flagged,
                    hold.held)


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
