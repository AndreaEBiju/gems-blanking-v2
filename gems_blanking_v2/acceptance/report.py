"""Task 19: the acceptance report, computed headless (ruling 2026-10-07 (b) R4).

One function per row of task 19's table. Each takes the pipeline OUTPUT it needs as an
explicit argument and returns a :class:`Row`: ``pass`` / ``fail`` with the number, or
``not_computable`` naming the missing input (or the missing ruling), or ``not_built``
(velocity: task 18 is out of this build, R5). A row whose input does not exist yet is
never given a number. :func:`build_report` assembles the rows, the spec contradictions met
while building this (:data:`SPEC_CONTRADICTIONS`, with the reading implemented), and the
list of every place real data disagreed with a constant (:class:`Disagreements`, seeded
with :data:`KNOWN_DISAGREEMENTS` and appended to by the pipeline).

Readings implemented (each also listed in :data:`SPEC_CONTRADICTIONS`):

* Row 1 is per consumer: an injection counts for a consumer when it is above THAT
  consumer's tolerance (Step 9), and every consumer must reach 98%.
* Row 2's denominator is the cores JUDGED inside exhaustive audit spans (R3; invariant 9:
  unjudged is not negative).
* Row 3 checks zeros on the MATLAB-side ``yOut`` and the taper on the Python output; the
  consumers themselves receive untapered NaN boundaries (blank spans cannot carry one).
* Row 4 uses 300-3000 Hz (R8) and needs a verdict for that band AND for 0.5-3 and 0-2.
* Row 5: "slow bands" are 0.5-3 and 0-2 (below the heart rate); 10-150 and 2-50 are not
  counted as slow - flagged.
* Row 6 is R1's test set, new-cohort I, J, K, each with a span cluster-bootstrap CI,
  never pooled; the scoring model's corpus must exclude all three.
* Row 7 never passes without a RULED equivalence margin: with none, it reports the CI as
  not computable; a CI excluding 0 fails at any margin.
* Row 11 "re-running selects the same model" = re-running RE-APPLIES the user's recorded
  choice (12A: no automatic selection); nothing here selects.

Row 7 REFUSES (raises :class:`CoverageMissingError`) when coverage is unavailable - the
``hasCoverage`` rule (``IMPLEMENTATION.md``, ``bulk_mixed_models.m`` row).

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, get_args

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat
from scipy.stats import t as student_t

from gems_blanking_v2.constants import ENG_BAND
from gems_blanking_v2.emit.masks import TAPER_S, ConsumerMask, MaskKey, MaskSpan
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.io.nan_interop import find_zero_runs
from gems_blanking_v2.io.registry_log import RegistryEvent, replay
from gems_blanking_v2.model.labels import animal_key, is_test_animal
from gems_blanking_v2.physio.cardiac_window import CardiacStatus
from gems_blanking_v2.types import TrainingMode

__all__ = [
    "KNOWN_DISAGREEMENTS",
    "SPEC_CONTRADICTIONS",
    "AcceptanceReport",
    "BlankRow",
    "Comparison",
    "CoverageMissingError",
    "Disagreement",
    "Disagreements",
    "Injection",
    "Row",
    "Status",
    "build_report",
    "expected_calibration_error",
    "masked_motion_seconds",
    "row1_candidate_recall",
    "row2_class_balance",
    "row3_no_zeros",
    "row4_cardiac_scope",
    "row5_retention",
    "row6_cross_animal",
    "row7_coverage_confound",
    "row8_downstream",
    "row9_velocity",
    "row10_mode_comparison",
    "row11_model_routing",
    "row12_calibration",
]

F64 = npt.NDArray[np.float64]


class Status(StrEnum):
    """A row's outcome."""

    PASS = "pass"
    FAIL = "fail"
    NOT_COMPUTABLE = "not_computable"
    NOT_BUILT = "not_built"


@dataclass(frozen=True)
class Row:
    """One row of the report. ``value`` is the headline number (absent when there is none)."""

    number: int
    name: str
    status: Status
    condition: str
    value: float | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_record(self) -> dict[str, Any]:
        """JSON-ready; a missing value is an absent key."""
        rec: dict[str, Any] = {"number": self.number, "name": self.name,
                               "status": self.status.value, "condition": self.condition}
        if self.value is not None and math.isfinite(self.value):
            rec["value"] = self.value
        if self.detail:
            rec["detail"] = _clean(self.detail)
        if self.reason:
            rec["reason"] = self.reason
        return rec


def _clean(v: Any) -> Any:  # noqa: ANN401 - JSON value
    if isinstance(v, Mapping):
        return {str(k): _clean(x) for k, x in v.items() if not _missing(x)}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x) for x in (sorted(v) if isinstance(v, (set, frozenset)) else v)]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def _not_computable(number: int, name: str, condition: str, missing: str,
                    detail: Mapping[str, Any] | None = None) -> Row:
    return Row(number, name, Status.NOT_COMPUTABLE, condition, None, detail or {},
               reason=f"missing input: {missing}")


# ---------------------------------------------------------------------------
# row 1 - candidate recall on injected synthetics, per consumer
# ---------------------------------------------------------------------------

ROW1_RECALL: Final = 0.98
ROW1_MAX_CANDIDATES: Final = 3000


@dataclass(frozen=True, slots=True)
class Injection:
    """One injected synthetic: did it produce a candidate, and whose tolerance it exceeds."""

    recording: str
    detected: bool
    above_tolerance_of: frozenset[str]


def row1_candidate_recall(injections: Sequence[Injection] | None,
                          candidates_per_recording: Mapping[str, int] | None, *,
                          consumers: Sequence[str]) -> Row:
    """Per consumer, >= 98% of injections above its tolerance detected; <= 3000 candidates.

    ``consumers`` is the EXPECTED set (the tolerance table's consumers, velocity excepted
    while task 18 is out, R5); one with no injection above its tolerance is not
    computable, named - a consumer missing from the data is never silently passed.
    """
    cond = (f"for every consumer, >= {ROW1_RECALL:.0%} of injected synthetics above that "
            f"consumer's tolerance produce a candidate; <= {ROW1_MAX_CANDIDATES} "
            "candidates per injected recording")
    if not injections:
        return _not_computable(1, "candidate_recall", cond,
                               "injection results (task 09 synthetic replay)")
    if candidates_per_recording is None:
        return _not_computable(1, "candidate_recall", cond, "candidate counts per recording")
    lacking = sorted({i.recording for i in injections} - set(candidates_per_recording))
    if lacking:
        return _not_computable(1, "candidate_recall", cond,
                               f"candidate counts for injected recordings {lacking}")
    if not consumers:
        return _not_computable(1, "candidate_recall", cond, "the expected consumer set")
    seen = {c for i in injections for c in i.above_tolerance_of}
    unseen = sorted(set(consumers) - seen)
    if unseen:
        return _not_computable(1, "candidate_recall", cond,
                               f"injections above the tolerance of {unseen}")
    recall = {}
    for c in sorted(consumers):
        mine = [i for i in injections if c in i.above_tolerance_of]
        recall[c] = sum(i.detected for i in mine) / len(mine)
    worst = max(candidates_per_recording[r] for r in {i.recording for i in injections})
    low = min(recall.values())
    ok = low >= ROW1_RECALL and worst <= ROW1_MAX_CANDIDATES
    return Row(1, "candidate_recall", Status.PASS if ok else Status.FAIL, cond, low,
               {"recall_by_consumer": recall, "max_candidates_per_recording": worst,
                "n_below_every_tolerance": sum(not i.above_tolerance_of for i in injections)})


# ---------------------------------------------------------------------------
# row 2 - class balance
# ---------------------------------------------------------------------------

ROW2_MIN_FRACTION: Final = 0.05


def row2_class_balance(labelled: Mapping[str, tuple[int, int]] | None) -> Row:
    """``labelled``: recording -> (motion cores, judged cores) inside exhaustive spans.

    Judged = motion or a negative (physiology, line noise) per R3; ``unjudged`` and
    ``unsure`` cores are not in the denominator (invariant 9).
    """
    cond = (f">= {ROW2_MIN_FRACTION:.0%} of judged candidate cores in exhaustive spans are "
            "motion (R3; unjudged excluded)")
    if not labelled:
        return _not_computable(2, "class_balance", cond,
                               "judged cores in exhaustive audit spans (task 10 labels)")
    bad = sorted(r for r, (a, b) in labelled.items() if not 0 <= a <= b)
    if bad:
        msg = f"row 2 needs 0 <= motion <= judged per recording; violated for {bad}"
        raise ValueError(msg)
    tp = sum(a for a, _ in labelled.values())
    n = sum(b for _, b in labelled.values())
    if n == 0:
        return _not_computable(2, "class_balance", cond, "no judged cores")
    per = {r: a / b for r, (a, b) in labelled.items() if b}
    frac = tp / n
    return Row(2, "class_balance", Status.PASS if frac >= ROW2_MIN_FRACTION else Status.FAIL,
               cond, frac, {"n_judged": n, "n_motion": tp,
                            "min_recording_fraction": min(per.values()) if per else None})


# ---------------------------------------------------------------------------
# row 3 - no zeros (MATLAB yOut), every boundary tapered (Python output)
# ---------------------------------------------------------------------------


def _attenuated(x: F64, y: F64) -> int | None:
    """1 tapered, 0 not, ``None`` when the neighbours hold no comparable sample."""
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 0)
    if not ok.any():
        return None
    w = y[ok] / x[ok]
    return int(bool(np.all(w > 0) and np.all(w < 1)))


def _boundaries_tapered(original: F64, emitted: F64, fs: float) -> tuple[int, int]:
    """Count ``(boundaries, tapered)`` NaN-run boundaries of an emitted array.

    A boundary is tapered when the valid samples beside the NaN run are attenuated
    (0 < y / x < 1) across the taper length. Samples NaN already in the original are
    ignored; a boundary with no comparable neighbour is not counted.
    """
    bad = np.isnan(emitted) & np.isfinite(original)
    d = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    n_taper = int(round(TAPER_S * fs))
    total = tapered = 0
    for s in np.flatnonzero(d == 1):
        if s >= n_taper:
            t = _attenuated(original[s - n_taper:s], emitted[s - n_taper:s])
            if t is not None:
                total, tapered = total + 1, tapered + t
    for e in np.flatnonzero(d == -1):
        if e + n_taper <= emitted.size:
            t = _attenuated(original[e:e + n_taper], emitted[e:e + n_taper])
            if t is not None:
                total, tapered = total + 1, tapered + t
    return total, tapered


def row3_no_zeros(matlab_yout: Sequence[tuple[str, F64]] | None,
                  python_out: Sequence[tuple[str, F64, F64]] | None, fs: float | None) -> Row:
    """Zero runs checked on MATLAB's ``yOut``; the taper on the Python output.

    ``matlab_yout``: ``(name, yOut)`` as the consumers read it. ``python_out``:
    ``(name, original, emitted)`` from ``emit.masks.apply_mask``. The consumers get the
    MATLAB side, whose NaN boundaries are untapered (the blank spans cannot carry one).
    """
    cond = ("no exact-zero run (> 2 samples) in any MATLAB yOut; every boundary of the Python "
            "output tapered (MATLAB consumers receive untapered NaN boundaries)")
    if not matlab_yout:
        return _not_computable(3, "no_zeros", cond, "MATLAB yOut arrays (the consumers' input)")
    if not python_out or fs is None:
        return _not_computable(3, "no_zeros", cond, "Python emitted arrays with their originals")
    runs = {n: len(z) for n, y in matlab_yout if (z := find_zero_runs(np.asarray(y, float)))}
    bounds = taps = 0
    for _name, x, y in python_out:
        b, t = _boundaries_tapered(np.asarray(x, np.float64), np.asarray(y, np.float64), fs)
        bounds, taps = bounds + b, taps + t
    if bounds == 0:
        return Row(3, "no_zeros", Status.NOT_COMPUTABLE, cond, None,
                   {"matlab_arrays": len(matlab_yout), "arrays_with_zero_runs": runs},
                   "missing input: a Python output with at least one mask boundary")
    ok = not runs and taps == bounds
    return Row(3, "no_zeros", Status.PASS if ok else Status.FAIL, cond, float(len(runs)),
               {"matlab_arrays": len(matlab_yout), "arrays_with_zero_runs": runs,
                "python_boundaries": bounds, "tapered": taps})


# ---------------------------------------------------------------------------
# row 4 - cardiac scope
# ---------------------------------------------------------------------------

ROW4_BANDS_FLAT: Final[tuple[str, ...]] = (ENG_BAND, "0.5-3", "0-2")
CARDIAC_STATUSES: Final[frozenset[str]] = frozenset(get_args(CardiacStatus))


def row4_cardiac_scope(verdicts: Sequence[Mapping[str, str]] | None) -> Row:
    """``verdicts``: after masking, ``{channel, band, status}`` from ``physio.cardiac_window``.

    Needs a verdict for 300-3000 Hz (R8) and for each of 0.5-3 and 0-2. Flat =
    ``no_window``. ``measured`` fails. ``unresolvable`` / ``not_applicable`` cannot show
    flatness: not computable. At rat RR (~165 ms) the slow bands are unresolvable by
    construction (impulse responses 1.1 s / 4.0 s exceed 0.8 RR) - see the contradictions.
    """
    cond = "peri-R profile flat in 300-3000 Hz and below 3 Hz (0.5-3, 0-2) after masking"
    if not verdicts:
        return _not_computable(4, "cardiac_scope", cond,
                               "peri-R verdicts on masked recordings "
                               "(cardiac_window after masking)")
    have = {v["band"] for v in verdicts}
    absent = [b for b in ROW4_BANDS_FLAT if b not in have]
    if absent:
        return _not_computable(4, "cardiac_scope", cond, f"verdicts for band(s) {absent}")
    rel = [v for v in verdicts if v["band"] in ROW4_BANDS_FLAT]
    unknown = sorted({v["status"] for v in rel} - CARDIAC_STATUSES)
    if unknown:
        msg = f"cardiac verdict status {unknown} is not one of {sorted(CARDIAC_STATUSES)}"
        raise ValueError(msg)
    measured = [f"{v['channel']}/{v['band']}" for v in rel if v["status"] == "measured"]
    blind = [f"{v['channel']}/{v['band']}" for v in rel
             if v["status"] in ("unresolvable", "not_applicable")]
    if measured:
        return Row(4, "cardiac_scope", Status.FAIL, cond, float(len(measured)),
                   {"not_flat": measured, "unresolvable": blind})
    if blind:
        return Row(4, "cardiac_scope", Status.NOT_COMPUTABLE, cond, None,
                   {"unresolvable": blind},
                   "the method cannot see these bands; flatness is unshown (needs a ruling)")
    return Row(4, "cardiac_scope", Status.PASS, cond, 0.0, {"n_verdicts": len(rel)})


# ---------------------------------------------------------------------------
# row 5 - retention
# ---------------------------------------------------------------------------

SLOW_BANDS: Final[tuple[str, ...]] = ("0.5-3", "0-2")
"""Below the heart rate. 10-150 and 2-50 are NOT counted as slow (flagged)."""


def row5_retention(masks: Mapping[str, Mapping[MaskKey, ConsumerMask]] | None,
                   baseline_retention: Mapping[str, float] | None) -> Row:
    """Retention per band per channel against the all-channel full-band baseline.

    PASS when the median ENG-band retention is at least the baseline's (over the same
    recordings) and above the median of each slow band. "Substantially" has no number;
    the margins are reported, not gated.
    """
    cond = ("retention per band per channel vs the all-channel full-band baseline; ENG-band "
            "retention higher than the slow bands (0.5-3, 0-2)")
    if not masks:
        return _not_computable(5, "retention", cond, "emitted masks per recording")
    overlap = [r for r in masks if baseline_retention and r in baseline_retention]
    if not overlap or baseline_retention is None:
        return _not_computable(5, "retention", cond,
                               "baseline retention for the recordings whose masks are given")
    by_band: dict[str, list[float]] = {}
    per: dict[str, float] = {}
    for rec in overlap:  # every median over the SAME recordings as the baseline
        ms = masks[rec]
        for (c, s, b), m in ms.items():
            by_band.setdefault(b, []).append(m.retention)
            per[f"{rec}|{c}|{s}|{b}"] = m.retention
    if ENG_BAND not in by_band:
        return _not_computable(5, "retention", cond, "an ENG-band (300-3000) mask")
    absent = [b for b in SLOW_BANDS if b not in by_band]
    if absent:
        return _not_computable(5, "retention", cond, f"slow-band masks {absent}")
    med = {b: float(np.median(v)) for b, v in by_band.items()}
    base = float(np.median([baseline_retention[r] for r in overlap]))
    slow = {b: med[b] for b in SLOW_BANDS}
    ok = med[ENG_BAND] >= base and all(med[ENG_BAND] > v for v in slow.values())
    return Row(5, "retention", Status.PASS if ok else Status.FAIL, cond, med[ENG_BAND],
               {"median_by_band": med, "baseline_median": base, "n_baseline": len(overlap),
                "eng_minus_slow": {b: med[ENG_BAND] - v for b, v in slow.items()},
                "not_counted_as_slow": [b for b in med if b in ("10-150", "2-50")],
                "per_band_per_channel": per})


# ---------------------------------------------------------------------------
# row 6 - cross-animal (R1: new-cohort I/J/K, individually, span bootstrap)
# ---------------------------------------------------------------------------

MIN_SPANS: Final = 2
"""A span cluster bootstrap needs at least two spans."""
BOOTSTRAP_RESAMPLES: Final = 1000
"""R9: 1,000 cluster-bootstrap resamples, clustered by span (new cohort)."""


def _span_bootstrap(spans: Sequence[tuple[int, int]], seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    f = np.array([a for a, _ in spans], float)
    n = np.array([b for _, b in spans], float)
    idx = rng.integers(0, len(spans), size=(BOOTSTRAP_RESAMPLES, len(spans)))
    tot = n[idx].sum(axis=1)
    rec = np.where(tot > 0, f[idx].sum(axis=1) / np.where(tot > 0, tot, 1), np.nan)
    lo, hi = np.nanpercentile(rec, [2.5, 97.5])
    return float(lo), float(hi)


def row6_cross_animal(per_animal: Mapping[tuple[str, str], Sequence[tuple[int, int]]] | None,
                      *, scoring_corpus: Iterable[tuple[str, str]] | None, seed: int = 0) -> Row:
    """``per_animal``: ``(cohort, animal)`` -> per-span ``(found, total)``; never pooled.

    Only R1's test set belongs here; anything else raises. The scoring model's training
    corpus (``scoring_corpus``, ``(cohort, animal)`` pairs) must exclude every test
    animal (R1) - raises otherwise. Each animal gets its recall and a span
    cluster-bootstrap 95% CI (R9). PASS = all of new:I, new:J, new:K reported; the spec
    gives no threshold.
    """
    cond = ("blind-test event recall per never-seen animal (R1: new-cohort I, J, K) with a "
            "span cluster-bootstrap CI, never pooled")
    if not per_animal:
        return _not_computable(6, "cross_animal", cond,
                               "blind-test events per test animal from a model "
                               "trained without them")
    corpus = [(str(c).lower(), str(a).upper()) for c, a in (scoring_corpus or [])]
    if not corpus:
        return _not_computable(6, "cross_animal", cond, "the scoring model's training corpus")
    leaked = sorted(animal_key(c, a) for c, a in corpus if is_test_animal(c, a))
    if leaked:
        msg = f"the scoring model trained on test animals {leaked} (R1)"
        raise ValueError(msg)
    per_animal = {(str(c).lower(), str(a).upper()): v for (c, a), v in per_animal.items()}
    for cohort, animal in per_animal:
        if not is_test_animal(cohort, animal):
            msg = f"{animal_key(cohort, animal)} is not in R1's test set; it cannot be in row 6"
            raise ValueError(msg)
    detail: dict[str, Any] = {}
    for i, ((c, a), spans) in enumerate(sorted(per_animal.items())):
        f, n = sum(x for x, _ in spans), sum(y for _, y in spans)
        entry: dict[str, Any] = {"found": f, "total": n, "n_spans": len(spans)}
        if n and len(spans) >= MIN_SPANS:
            entry["recall"] = f / n
            entry["ci95"] = list(_span_bootstrap(spans, seed + i))
        detail[animal_key(c, a)] = entry
    missing = sorted({"new:I", "new:J", "new:K"} - {k for k, v in detail.items() if "recall" in v})
    if missing:
        return Row(6, "cross_animal", Status.NOT_COMPUTABLE, cond, None, detail,
                   f"missing input: blind-test events in >= {MIN_SPANS} spans for {missing}")
    return Row(6, "cross_animal", Status.PASS, cond, None, detail)


# ---------------------------------------------------------------------------
# row 7 - coverage confound
# ---------------------------------------------------------------------------

MIN_COVERAGE: Final = 0.5
"""``bulk_mixed_models.m``'s exclusion (``minCoverage``)."""
CONFOUND_ALPHA: Final = 0.05
MIN_DOF: Final = 10
"""Below this many residual degrees of freedom the CI is not reported as a result."""

MOTION_REASONS: Final[frozenset[str]] = frozenset({"clip", "beat_train_changed",
                                                   "subtract_failed"})
MOTION_REASON_PREFIX: Final = "in_band_"
"""Allow-list of the motion reason codes ``extent.routing`` writes (plus ``in_band_*``)."""
NON_MOTION_REASONS: Final[frozenset[str]] = frozenset({
    "cuff_distrusted", "per_minute_cuff_distrust", "line_noise_cuff_minute",
    "excluded_epoch", "cardiac"})


class CoverageMissingError(RuntimeError):
    """Row 7 refuses to run without coverage (``hasCoverage`` false)."""


def masked_motion_seconds(spans: Iterable[MaskSpan], duration_s: float) -> float:
    """Seconds of motion masking (union) in ``[0, duration_s)``, for one mask.

    Motion = the allow-listed reason codes; protocol, cardiac, cuff distrust and
    line-noise spans are left out; an unknown reason code raises (it would otherwise be
    silently counted or silently dropped). Motion lying inside an ``excluded_epoch`` span
    is subtracted: that time is not at risk (the denominator excludes it too).
    """
    if not duration_s > 0:
        msg = f"duration must be positive, got {duration_s}"
        raise ValueError(msg)
    ivs = []
    excluded = []
    for s in spans:
        if s.reason == "excluded_epoch":
            excluded.append((max(0.0, s.start_s), min(duration_s, s.stop_s)))
        if s.reason in NON_MOTION_REASONS:
            continue
        if s.reason not in MOTION_REASONS and not s.reason.startswith(MOTION_REASON_PREFIX):
            msg = f"unknown mask reason code {s.reason!r}: neither motion nor ruled out"
            raise ValueError(msg)
        a, b = max(0.0, s.start_s), min(duration_s, s.stop_s)
        if b > a:
            ivs.append((a, b))
    return _union_length(ivs) - _overlap_length(ivs, excluded)


def _merge(ivs: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(i for i in ivs if i[1] > i[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _union_length(ivs: Iterable[tuple[float, float]]) -> float:
    return sum(b - a for a, b in _merge(ivs))


def _overlap_length(ivs: Iterable[tuple[float, float]],
                    other: Iterable[tuple[float, float]]) -> float:
    m, o = _merge(ivs), _merge(other)
    return sum(max(0.0, min(b, d) - max(a, c)) for a, b in m for c, d in o)


@dataclass(frozen=True, slots=True)
class BlankRow:
    """One recording x consumer for the confound regression.

    ``motion_blank_s`` is motion masking only (:func:`masked_motion_seconds`); the
    denominator is the time at risk, ``duration_s - excluded_s`` (the stim epoch out).
    ``coverage`` is the valid fraction (``None`` = unknown).
    """

    recording: str
    consumer: str
    condition: str
    mode: str
    motion_blank_s: float
    duration_s: float
    excluded_s: float
    coverage: float | None

    @property
    def fraction(self) -> float:
        """Motion blank over time at risk."""
        return self.motion_blank_s / (self.duration_s - self.excluded_s)


def _wls(x: F64, y: F64, w: F64) -> tuple[F64, F64, int]:
    sw = np.sqrt(w)
    xw, yw = x * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = yw - xw @ beta
    dof = x.shape[0] - x.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(xw.T @ xw)
    return beta, np.sqrt(np.clip(np.diag(cov), 0.0, None)), dof


def _confound_one(rows: Sequence[BlankRow]) -> dict[str, Any]:
    kept = [r for r in rows if r.coverage is not None and r.coverage >= MIN_COVERAGE]
    conds = sorted({r.condition for r in kept})
    out: dict[str, Any] = {"n_used": len(kept), "n_excluded_low_coverage": len(rows) - len(kept)}
    if len(conds) != 2:  # noqa: PLR2004
        return {**out, "not_computable": f"both conditions needed, have {conds}"}
    modes = sorted({r.mode for r in kept})
    names = ["intercept", f"condition[{conds[1]}]", "coverage", *[f"mode[{m}]" for m in modes[1:]]]
    cols = [np.ones(len(kept)), np.array([r.condition == conds[1] for r in kept], float),
            np.array([r.coverage for r in kept], float)]
    cols += [np.array([r.mode == m for r in kept], float) for m in modes[1:]]
    x = np.column_stack(cols)
    rank = int(np.linalg.matrix_rank(x))
    if rank < x.shape[1]:
        confounded = [names[j] for j in range(x.shape[1])
                      if int(np.linalg.matrix_rank(np.delete(x, j, axis=1))) == rank]
        return {**out, "not_computable": f"design is rank-deficient; confounded terms "
                f"{confounded}"}
    dof = x.shape[0] - x.shape[1]
    if dof < MIN_DOF:
        return {**out, "not_computable": f"only {dof} residual dof (< {MIN_DOF})"}
    y = np.array([r.fraction for r in kept], float)
    w = np.array([r.coverage for r in kept], float)
    beta, se, dof = _wls(x, y, w)
    tcrit = float(student_t.ppf(1 - CONFOUND_ALPHA / 2, dof))
    coef, s = float(beta[1]), float(se[1])
    out.update(term=names[1], coef=coef, se=s, ci95=[coef - tcrit * s, coef + tcrit * s],
               dof=dof, mode_is_factor=len(modes) > 1,
               mean_by_condition={c: float(np.mean([r.fraction for r in kept if r.condition == c]))
                                  for c in conds},
               mean_by_mode={m: float(np.mean([r.fraction for r in kept if r.mode == m]))
                             for m in modes})
    if s > 0:
        out["p"] = float(2 * student_t.sf(abs(coef / s), dof))
    return out


def row7_coverage_confound(rows: Sequence[BlankRow] | None, *, has_coverage: bool,  # noqa: PLR0911
                           equivalence_margin: float | None, consumers: Sequence[str]) -> Row:
    """Regress motion blank fraction on condition, per consumer, with coverage (and mode).

    REFUSES (raises :class:`CoverageMissingError`) when ``has_coverage`` is false or any
    row lacks coverage. Per consumer (invariant 2), as ``bulk_mixed_models.m``: coverage
    < 0.5 excluded, weighted by coverage, coverage a covariate; mode a factor when it
    varies. FAIL if any consumer's condition CI excludes 0. Otherwise PASS only within a
    RULED ``equivalence_margin`` (none ruled yet: not computable, CI reported).
    """
    cond = ("motion blank fraction (over time at risk) does not depend on condition, per "
            "consumer (WLS on condition + coverage [+ mode], coverage-weighted, >= 0.5)")
    if not has_coverage or (rows and any(r.coverage is None for r in rows)):
        msg = ("row 7 refuses to run without coverage (hasCoverage is false): a confound check "
               "without its confound covariate reports 'no confound' for the wrong reason")
        raise CoverageMissingError(msg)
    if not rows:
        return _not_computable(7, "coverage_confound", cond,
                               "motion blank per recording x consumer with condition and mode")
    if not consumers:
        return _not_computable(7, "coverage_confound", cond, "the expected consumer set")
    unseen = sorted(set(consumers) - {r.consumer for r in rows})
    if unseen:
        return _not_computable(7, "coverage_confound", cond, f"blank fractions for {unseen}")
    per = {c: _confound_one([r for r in rows if r.consumer == c]) for c in sorted(consumers)}
    blocked = {c: d["not_computable"] for c, d in per.items() if "not_computable" in d}
    if blocked:
        return Row(7, "coverage_confound", Status.NOT_COMPUTABLE, cond, None, per,
                   f"not computable for {sorted(blocked)}: {blocked}")
    worst = max(per.values(), key=lambda d: abs(d["coef"]))
    failing = [d for d in per.values() if d["ci95"][0] > 0 or d["ci95"][1] < 0]
    if failing:
        top = max(failing, key=lambda d: abs(d["coef"]))
        return Row(7, "coverage_confound", Status.FAIL, cond, top["coef"], per)
    if equivalence_margin is None:
        return Row(7, "coverage_confound", Status.NOT_COMPUTABLE, cond, worst["coef"], per,
                   "equivalence margin not ruled: the CI includes 0, which is not evidence "
                   "of no dependence")
    inside = all(-equivalence_margin <= d["ci95"][0] and d["ci95"][1] <= equivalence_margin
                 for d in per.values())
    return Row(7, "coverage_confound", Status.PASS if inside else Status.NOT_COMPUTABLE, cond,
               worst["coef"], {**per, "equivalence_margin": equivalence_margin},
               "" if inside else "CI wider than the ruled margin: underpowered")


# ---------------------------------------------------------------------------
# row 8 - downstream (MATLAB consumer outputs)
# ---------------------------------------------------------------------------

DOWNSTREAM_METRICS: Final[tuple[str, ...]] = (
    "fracISIclean", "slow_wave_non_nan_fraction", "dfa_alpha2_available", "nRR_used",
    "step5f_median_epoch_s")


def row8_downstream(before: Mapping[str, float] | None, after: Mapping[str, float] | None,
                    between_animal_var: tuple[float, float] | None) -> Row:
    """All five downstream metrics up, and the between-animal endpoint variance down."""
    cond = ("fracISIclean, slow-wave non-NaN fraction, DFA alpha2 availability, nRR_used, "
            "median step5f epoch length all up; between-animal endpoint variance down")
    if not before or not after or between_animal_var is None:
        return _not_computable(8, "downstream", cond,
                               "MATLAB consumer outputs before and after the new masks "
                               f"({', '.join(DOWNSTREAM_METRICS)}) and between-animal variance")
    lacking = [m for m in DOWNSTREAM_METRICS if m not in before or m not in after]
    if lacking:
        return _not_computable(8, "downstream", cond, f"metrics {lacking}")
    up = {m: after[m] > before[m] for m in DOWNSTREAM_METRICS}
    var_down = between_animal_var[1] < between_animal_var[0]
    ok = all(up.values()) and var_down
    return Row(8, "downstream", Status.PASS if ok else Status.FAIL, cond,
               float(sum(up.values())), {"up": up, "between_animal_variance": list(
                   between_animal_var), "variance_down": var_down})


# ---------------------------------------------------------------------------
# row 9 - velocity (task 18 is out of this build, R5)
# ---------------------------------------------------------------------------


def row9_velocity() -> Row:
    """Not built: task 18 (conduction velocity) is out of this build (ruling (b) R5)."""
    return Row(9, "velocity", Status.NOT_BUILT,
               "peak-ratio confidence with every estimate; unsigned with a warning where "
               "rostral_end is missing",
               reason="task 18 (conduction velocity) is out of this build "
                      "(ruling 2026-10-07 (b) R5)")


# ---------------------------------------------------------------------------
# row 10 - mode comparison
# ---------------------------------------------------------------------------

MODE_C_MARGIN: Final = 0.03
"""R9: mode C beats mode B only if its F1 is higher by >= 0.03 and the CI excludes 0."""
MODE_LETTERS: Final[Mapping[str, TrainingMode]] = {
    "A": TrainingMode.POOLED, "B": TrainingMode.ADAPTED, "C": TrainingMode.PER_ANIMAL}
B_VS_C_CORPORA: Final[frozenset[str]] = frozenset({"old_cohort_loao", "within_animal"})
CLUSTER_UNIT: Final[Mapping[str, str]] = {"old": "recording", "new": "span"}
"""R9: the bootstrap cluster per cohort."""
"""R1: mode choice uses only old-cohort LOAO and within-animal held-out data."""


def _mode(name: str) -> TrainingMode:
    if name in MODE_LETTERS:
        return MODE_LETTERS[name]
    return TrainingMode(name.lower())


@dataclass(frozen=True, slots=True)
class Comparison:
    """A comparison the build reported between two modes, and its status.

    ``status`` is ``reported`` (a verdict was drawn) or ``not_comparable`` (shown side by
    side, explicitly not compared - what task 12 wants for A vs C).
    """

    a: str
    b: str
    status: str


def row10_mode_comparison(curves: Mapping[tuple[str, str], Path] | None,  # noqa: PLR0911, PLR0912
                          b_vs_c: Mapping[str, Any] | None, *,
                          comparisons: Sequence[Comparison] | None,
                          animals: Sequence[str]) -> Row:
    """Learning curves for every mode x animal; B-vs-C by R9; A-vs-C never reported.

    ``comparisons`` is REQUIRED; modes are normalised through :class:`TrainingMode`
    (A = pooled, B = adapted, C = per_animal), so a pooled-vs-per_animal comparison is
    caught however named. ``b_vs_c`` must carry ``f1_diff`` (C - B), ``ci95``,
    ``n_resamples`` (1000, R9), ``cluster_unit``, ``excluded_folds`` (folds with < 20
    positives, R9) and ``corpus`` (old_cohort_loao or within_animal, R1).
    """
    cond = ("learning curve per mode per animal; A-vs-C never compared; B-vs-C verdict "
            "(R9: C wins iff F1 +0.03 and CI excludes 0), task 11 triggered if C wins")
    if comparisons is None:
        return _not_computable(10, "mode_comparison", cond, "the list of reported comparisons")
    if not animals:
        return _not_computable(10, "mode_comparison", cond, "the animals the modes cover")
    for cmp in comparisons:
        pair = {_mode(cmp.a), _mode(cmp.b)}
        if cmp.status not in ("reported", "not_comparable"):
            msg = f"comparison status must be reported or not_comparable, got {cmp.status!r}"
            raise ValueError(msg)
        if pair == {TrainingMode.POOLED, TrainingMode.PER_ANIMAL} and cmp.status == "reported":
            return Row(10, "mode_comparison", Status.FAIL, cond,
                       reason="a pooled-vs-per_animal (A-vs-C) comparison was reported "
                              "(invariant 12)")
    if not curves:
        return _not_computable(10, "mode_comparison", cond, "learning curves per mode per animal")
    have = {(_mode(m), a) for (m, a) in curves}
    need = {(m, a) for m in TrainingMode for a in animals}
    absent = sorted(f"{m.value}/{a}" for m, a in need - have)
    gone = sorted(f"{m}/{a}" for (m, a), p in curves.items() if not Path(p).is_file())
    if absent or gone:
        return _not_computable(10, "mode_comparison", cond,
                               f"learning curves {absent + gone}")
    if not b_vs_c:
        return _not_computable(10, "mode_comparison", cond, "the B-vs-C comparison record")
    fields = ("f1_diff", "ci95", "n_resamples", "cluster_unit", "excluded_folds", "corpus",
              "cohort", "matched_event_control")
    lacking = [f for f in fields if f not in b_vs_c]
    if lacking:
        return _not_computable(10, "mode_comparison", cond, f"B-vs-C fields {lacking}")
    if int(b_vs_c["n_resamples"]) != BOOTSTRAP_RESAMPLES or b_vs_c["corpus"] not in B_VS_C_CORPORA:
        msg = (f"B-vs-C must be {BOOTSTRAP_RESAMPLES} resamples on {sorted(B_VS_C_CORPORA)} "
               f"(R9, R1); got {b_vs_c['n_resamples']} on {b_vs_c['corpus']!r}")
        raise ValueError(msg)
    unit = CLUSTER_UNIT.get(str(b_vs_c["cohort"]))
    if unit is None or b_vs_c["cluster_unit"] != unit:
        msg = (f"R9 clusters by recording (old cohort) or span (new cohort); got "
               f"{b_vs_c['cluster_unit']!r} for cohort {b_vs_c['cohort']!r}")
        raise ValueError(msg)
    control = b_vs_c["matched_event_control"]
    if not isinstance(control, Mapping) or "f1_diff" not in control or "ci95" not in control:
        return _not_computable(10, "mode_comparison", cond,
                               "a matched-event-count control for B vs C (invariant 13)")
    diff = float(b_vs_c["f1_diff"])
    lo, hi = (float(x) for x in b_vs_c["ci95"])
    c_wins = diff >= MODE_C_MARGIN and lo > 0
    if c_wins:
        return Row(10, "mode_comparison", Status.FAIL, cond, diff,
                   {"verdict": "C beats B", "ci95": [lo, hi], "matched_event_control": control,
                    "task11_investigation_triggered": True},
                   "PER_ANIMAL beating ADAPTED is a task 11 feature-invariance bug, not a "
                   "result to ship (CLAUDE.md, task 12)")
    return Row(10, "mode_comparison", Status.PASS, cond, diff,
               {"verdict": "C beats B" if c_wins else "C does not beat B", "ci95": [lo, hi],
                "n_curves": len(curves), "cluster_unit": b_vs_c["cluster_unit"],
                "excluded_folds": list(b_vs_c["excluded_folds"]), "corpus": b_vs_c["corpus"],
                "matched_event_control": control, "task11_investigation_triggered": False})


# ---------------------------------------------------------------------------
# row 11 - model routing
# ---------------------------------------------------------------------------

MODEL_FIELDS: Final[tuple[str, ...]] = ("mode", "animal", "version", "corpus_hash",
                                        "calibrator")


def _provenance_of(mask_file: Path) -> MaskProvenance:
    m = loadmat(mask_file)
    if "provenance_json" not in m:
        msg = f"{Path(mask_file).name}: no provenance"
        raise ProvenanceError(msg)
    return MaskProvenance.from_json(str(m["provenance_json"][0]))


def row11_model_routing(mask_files: Mapping[str, Path] | None,  # noqa: PLR0912
                        assignments: Mapping[str, Mapping[str, Any]] | None,
                        routing_log: Mapping[str, str] | None,
                        rerun_files: Mapping[str, Path] | None) -> Row:
    """Every mask names the logged ModelSpec and routing table; every re-run re-applies it.

    ``assignments``: recording -> the model the user chose (compared on
    :data:`MODEL_FIELDS`). ``routing_log``: recording -> the routing-table hash logged for
    it. ``rerun_files``: a second run's mask file for EVERY recording (else not
    computable, naming the ones without).
    """
    cond = ("every emitted mask names its full ModelSpec and the logged routing table; the "
            "user's choice is logged per recording; every re-run re-applies it (12A)")
    if not mask_files:
        return _not_computable(11, "model_routing", cond, "emitted mask files")
    if not assignments:
        return _not_computable(11, "model_routing", cond, "the per-recording model assignment log")
    if not routing_log:
        return _not_computable(11, "model_routing", cond, "the per-recording routing-hash log")
    norerun = sorted(set(mask_files) - set(rerun_files or {}))
    if norerun:
        return _not_computable(11, "model_routing", cond, f"re-run mask files for {norerun}")
    assert rerun_files is not None
    problems: dict[str, str] = {}
    for rec, path in mask_files.items():
        try:
            prov = _provenance_of(path)
            again = _provenance_of(rerun_files[rec])
        except (ProvenanceError, OSError, ValueError) as exc:
            problems[rec] = f"mask provenance: {exc}"
            continue
        chosen = assignments.get(rec)
        model = {k: prov.model.get(k) for k in MODEL_FIELDS}
        if chosen is None:
            problems[rec] = "no logged model choice"
        elif model != {k: chosen.get(k) for k in MODEL_FIELDS}:
            problems[rec] = "mask names a different model from the logged choice"
        elif rec not in routing_log:
            problems[rec] = "no routing-table hash logged for this recording"
        elif prov.routing_hash != routing_log[rec]:
            problems[rec] = "mask names a different routing table from the log"
        elif not prov.recording == again.recording == rec:
            problems[rec] = (f"provenance names recording {prov.recording!r} / re-run "
                             f"{again.recording!r}, not {rec!r}")
        elif {k: again.model.get(k) for k in MODEL_FIELDS} != model:
            problems[rec] = "a re-run used a different model"
        elif again.routing_hash != routing_log[rec]:
            problems[rec] = "a re-run used a different routing table"
    return Row(11, "model_routing", Status.FAIL if problems else Status.PASS, cond,
               float(len(problems)), {"n_recordings": len(mask_files), "problems": problems})


# ---------------------------------------------------------------------------
# row 12 - calibration
# ---------------------------------------------------------------------------

ECE_MAX: Final = 0.05
"""R9: expected calibration error <= 0.05 on held-out data."""
ECE_BINS: Final = 10
MIN_HELD_OUT: Final = 100
"""Fewest held-out cores per model for an ECE to mean anything (10 per bin). Provisional."""


def expected_calibration_error(p: npt.ArrayLike, y: npt.ArrayLike, n_bins: int = ECE_BINS
                               ) -> tuple[float, list[dict[str, float]]]:
    """ECE over equal-width bins of P(motion), and the reliability curve."""
    pp, yy = np.asarray(p, float), np.asarray(y, float)
    if pp.size == 0 or pp.size != yy.size:
        msg = "need equal-length, non-empty predictions and labels"
        raise ValueError(msg)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pp, edges[1:-1]), 0, n_bins - 1)
    ece, curve = 0.0, []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        conf, acc = float(pp[sel].mean()), float(yy[sel].mean())
        ece += sel.mean() * abs(conf - acc)
        curve.append({"bin_lo": float(edges[b]), "confidence": conf, "accuracy": acc,
                      "n": int(sel.sum())})
    return float(ece), curve


def row12_calibration(registry_events: Sequence[RegistryEvent] | None,
                      held_out: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]] | None) -> Row:
    """Every SHIPPED model has ECE <= 0.05 on held-out data (R9).

    Shipped = promoted after replaying the registry log (``io.registry_log.replay``).
    A model with fewer than :data:`MIN_HELD_OUT` held-out cores is not computable: with
    two samples the ECE can be 0.
    """
    cond = f"every shipped model calibrated on held-out data; ECE <= {ECE_MAX} (R9)"
    shipped = sorted(m for m, st in replay(list(registry_events or [])).items()
                     if st.is_promoted)
    if not shipped:
        return _not_computable(12, "calibration", cond, "a promoted model in the registry log")
    lacking = sorted(m for m in shipped if not held_out or m not in held_out
                     or np.asarray(held_out[m][0]).size < MIN_HELD_OUT)
    if lacking:
        return _not_computable(12, "calibration", cond, f"held-out data for {lacking}")
    assert held_out is not None
    eces, curves = {}, {}
    for model in shipped:
        p, y = held_out[model]
        eces[model], curves[model] = expected_calibration_error(p, y)
    worst = max(eces.values())
    return Row(12, "calibration", Status.PASS if worst <= ECE_MAX else Status.FAIL, cond, worst,
               {"ece": eces, "reliability": curves})


# ---------------------------------------------------------------------------
# disagreements and contradictions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    """One place real data (or the code it ran) disagreed with a constant in the spec."""

    constant: str
    spec_value: str
    measured: str
    source: str
    date: str


KNOWN_DISAGREEMENTS: Final[tuple[Disagreement, ...]] = (
    Disagreement("ENG band upper corner", "300-5000 Hz", "300-3000 Hz (A.5b; R8 erratum)",
                 "A.5b measurement, animal J", "2026-09-19"),
    Disagreement("sigma reduction, 300-5000 -> 300-3000", "24% (implied by A.5b)",
                 "12-14% measured", "IMPLEMENTATION.md A.5", ""),
    Disagreement("spike consumer settling (task 13)", "expect 30-50 ms",
                 "5.1 ms (impz, 1% of peak, 300-3000 Hz order-4 at 24414 Hz)",
                 "extent.tolerance.consumer_settling", "2026-10-07"),
    Disagreement("tripole sigma reduction", "~6x and 2.5-2.8x (both carried as constants)",
                 "not a constant: under 2 to ~10 as common mode varies; report per recording",
                 "IMPLEMENTATION.md task 04 (common-mode sweep)", ""),
    Disagreement("R-peak threshold", "3x MAD-sigma", "6x MAD-sigma in 10-150 Hz (task 05)",
                 "task 05 measurement", "2026-09-21"),
    Disagreement("HR detection band (hrBandHz)", "1-100 Hz", "10-150 Hz (ruled 2026-09-21)",
                 "task 05 / constants.HR_BAND", "2026-09-21"),
    Disagreement("R-peak minimum interval R_MIN", "90 ms", "60 ms (animal J RR median 164.7 ms)",
                 "task 05 measurement", "2026-09-21"),
    Disagreement("reference statistic", "10th-percentile linear envelope",
                 "mis-centred (median z 1.00, p90 3.05); median of the log envelope adopted",
                 "invariant 5", "2026-09-19"),
    Disagreement("MMC cycle length", "90-120 min (human and dog)", "~17.5 min in the rat",
                 "IMPLEMENTATION.md mmc section", ""),
    Disagreement("host bad_fraction", "0.046 (as proposed)", "0.0012 (blanked fraction)",
                 "IMPLEMENTATION.md step 9, the old-cohort hardware tripole host", ""),
    Disagreement("cost model", "unbiased", "biased 2x (measured)",
                 "IMPLEMENTATION.md step 9 cost model", ""),
    Disagreement("mmc / slow_wave input", "stomach_ref",
                 "raw ANT1-3 (extract_mmc.m; batch_process.m swData = signal(:, 3:5))",
                 "ruling 2026-10-06 item 3; R8", "2026-10-06"),
)
"""Seed: disagreements already recorded in the spec before this report. ``date`` is empty
where the spec records none."""


@dataclass
class Disagreements:
    """The append-only list the pipeline adds to as it runs (seeded with the known ones)."""

    items: list[Disagreement] = field(default_factory=lambda: list(KNOWN_DISAGREEMENTS))

    def add(self, constant: str, spec_value: str, measured: str, source: str,
            date: str | None = None) -> Disagreement:
        """Append one disagreement; returns it."""
        d = Disagreement(constant, spec_value, measured, source,
                         date or datetime.now(UTC).date().isoformat())
        self.items.append(d)
        return d


SPEC_CONTRADICTIONS: Final[tuple[dict[str, str], ...]] = (
    {"row": "1", "conflict": "row 1 demands >= 98% of injected synthetics; Step 9 defines an "
     "artifact that matters per consumer, by that consumer's tolerance",
     "implemented": "per consumer: recall over injections above that consumer's tolerance; "
     "every consumer must reach 98%"},
    {"row": "3", "conflict": "row 3 'every mask boundary tapered' vs the MATLAB handoff, whose "
     "blank spans cannot carry a taper", "implemented": "the consumers receive UNTAPERED NaN "
     "boundaries; the taper exists only in the Python output and is checked there"},
    {"row": "4", "conflict": "row 4 says 300-5000 Hz; R8 / A.5b put the ENG band at 300-3000",
     "implemented": "300-3000 Hz"},
    {"row": "4", "conflict": "'flat below 3 Hz' cannot be shown: at rat RR (~165 ms) the 0.5-3 "
     "and 0-2 Hz impulse responses (4.0 s, 1.1 s) exceed 0.8 RR, so cardiac_window returns "
     "unresolvable by construction", "implemented": "not computable until ruled"},
    {"row": "5", "conflict": "'substantially higher' has no number; 'slow bands' is not "
     "defined", "implemented": "ENG median above baseline and above 0.5-3 and 0-2; margins "
     "reported; 10-150 and 2-50 not counted as slow"},
    {"row": "6", "conflict": "row 6 says 'the four never-seen animals'; R1's prospective test "
     "set is new-cohort I, J, K (three); no recall threshold is given",
     "implemented": "new:I, new:J, new:K, each with a span bootstrap CI; PASS = reported"},
    {"row": "7", "conflict": "'must not depend on condition' needs an equivalence margin; the "
     "spec gives none", "implemented": "FAIL if a CI excludes 0; otherwise not computable "
     "until a margin is ruled"},
    {"row": "11", "conflict": "row 11 'routing rule ... re-running selects the same model' vs "
     "12A 'no automatic selection, no default model'",
     "implemented": "the user's recorded choice is re-applied; nothing selects"},
)


@dataclass(frozen=True)
class AcceptanceReport:
    """Every row, the contradictions met, and the disagreements list."""

    rows: tuple[Row, ...]
    disagreements: tuple[Disagreement, ...]
    created_at: str
    code_commit: str
    generation_sha: str

    @property
    def gate_passed(self) -> bool:
        """The gate passes only when every BUILT row passes (not_computable does not)."""
        return all(r.status is Status.PASS for r in self.rows if r.status is not Status.NOT_BUILT)

    def to_record(self) -> dict[str, Any]:
        """JSON-ready report."""
        return {"rows": [r.to_record() for r in self.rows],
                "gate_passed": self.gate_passed,
                "spec_contradictions": [dict(c) for c in SPEC_CONTRADICTIONS],
                "disagreements": [{k: v for k, v in asdict(d).items() if v}
                                  for d in self.disagreements],
                "created_at": self.created_at, "code_commit": self.code_commit,
                "generation_sha": self.generation_sha}

    def to_json(self) -> str:
        """Canonical, ASCII-escaped JSON; no NaN."""
        return json.dumps(self.to_record(), sort_keys=True, ensure_ascii=True, indent=1,
                          allow_nan=False)

    def to_markdown(self) -> str:
        """Render a readable table: row, status, number, reason."""
        lines = ["| # | row | status | value | reason |", "|---|---|---|---|---|"]
        for r in self.rows:
            v = "" if r.value is None or not math.isfinite(r.value) else f"{r.value:.4g}"
            lines.append(f"| {r.number} | {r.name} | {r.status.value} | {v} | {r.reason} |")
        lines += ["", f"Gate passed: {self.gate_passed}", "", "Spec contradictions:"]
        lines += [f"- row {c['row']}: {c['conflict']} -> {c['implemented']}"
                  for c in SPEC_CONTRADICTIONS]
        lines += ["", "Real data vs constants:"]
        lines += [f"- {d.constant}: spec {d.spec_value}; measured {d.measured} ({d.source}"
                  + (f", {d.date})" if d.date else ")") for d in self.disagreements]
        return "\n".join(lines) + "\n"


def build_report(rows: Sequence[Row], disagreements: Disagreements, *, code_commit: str,
                 generation_sha: str) -> AcceptanceReport:
    """Assemble the report; every one of the twelve rows must be present exactly once."""
    numbers = sorted(r.number for r in rows)
    if numbers != list(range(1, 13)):
        msg = f"the report needs rows 1-12 exactly once, got {numbers}"
        raise ValueError(msg)
    return AcceptanceReport(tuple(sorted(rows, key=lambda r: r.number)),
                            tuple(disagreements.items), datetime.now(UTC).isoformat(),
                            code_commit, generation_sha)
