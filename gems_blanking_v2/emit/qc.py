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
  either. An animal median that is unknown (``None``, e.g. the animal's first
  recording) applies the 20% rule alone and says so.

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

from gems_blanking_v2.emit.masks import ConsumerMask, MaskKey
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.io.store import append_line

__all__ = [
    "HOLD_BLANK_FRACTION",
    "HOLD_MEDIAN_RATIO",
    "BlankHold",
    "LongitudinalRow",
    "QcReport",
    "RetentionVerdict",
    "append_longitudinal",
    "blank_fraction_by_band",
    "blank_fraction_hold",
    "retention_by_key",
    "retention_gate",
]

HOLD_BLANK_FRACTION: Final = 0.20
"""Ruling (c) item 5: a blank above 20% of the recording is held for Andrea."""
HOLD_MEDIAN_RATIO: Final = 3.0
"""Ruling (c) item 5: a blank above 3x the animal's median is held for Andrea."""


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


def blank_fraction_hold(
    masks: Mapping[MaskKey, ConsumerMask], animal_median: Mapping[str, float | None], *,
    decisions: Iterable[RouteDecision] = (), hum_features: Mapping[str, float] | None = None,
) -> BlankHold:
    """Hold the recording if any consumer mask blanks > 20% or > 3x the animal's median.

    ``animal_median`` maps ``consumer|signal|band`` to the animal's median blank fraction
    for that mask, or ``None`` when unknown.
    """
    frac = {k: 1.0 - r for k, r in retention_by_key(masks).items()}
    reasons: list[str] = []
    for k, f in frac.items():
        if f > HOLD_BLANK_FRACTION:
            reasons.append(f"{k}: {100 * f:.1f}% blanked > {100 * HOLD_BLANK_FRACTION:.0f}%")
        med = animal_median.get(k)
        if med is None:
            continue
        if f > HOLD_MEDIAN_RATIO * med:
            reasons.append(f"{k}: {100 * f:.1f}% blanked > {HOLD_MEDIAN_RATIO:g} x the "
                           f"animal median {100 * med:.1f}%")
    unknown = sorted(k for k in frac if animal_median.get(k) is None)
    held = bool(reasons)
    if held and unknown:
        reasons.append(f"animal median unknown for {unknown}: the 3x rule was not applied")
    routes = Counter(d.reason.split(":")[0] if d.route == "reject" else d.route
                     for d in decisions if d.masks)
    return BlankHold(held, tuple(reasons), frac, tuple(routes.most_common(5)),
                     dict(hum_features or {}))


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

    def to_record(self) -> dict[str, Any]:
        """JSON-ready, with absent keys for anything missing."""
        rec = asdict(self)
        rec["hold_reasons"] = list(self.hold_reasons)
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
