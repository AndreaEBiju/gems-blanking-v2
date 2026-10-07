"""Task 19: the acceptance report, computed headless (ruling 2026-10-07 (b) R4).

One function per row of task 19's table. Each takes the pipeline OUTPUT it needs as an
explicit argument and returns a :class:`Row`: ``pass`` / ``fail`` with the number, or
``not_computable`` naming the missing input, or ``not_built`` (velocity: task 18 is out
of this build, R5). A row whose input does not exist yet is never given a number.
:func:`build_report` assembles the rows, the spec contradictions met while building this
(:data:`SPEC_CONTRADICTIONS`, and which reading was implemented), and the list of every
place real data disagreed with a constant (:class:`Disagreements`, seeded with
:data:`KNOWN_DISAGREEMENTS` and appended to by the pipeline).

Readings implemented where the spec contradicts itself (each listed in the report):

* Row 1 counts only injections above at least one consumer's tolerance (Step 9's
  definition of an artifact that matters); the >= 98% and <= 3000 candidates/recording
  stand.
* Row 4 uses the ENG band 300-3000 Hz (R8 erratum), not the row's 300-5000.
* Row 6 is the prospective test set, new-cohort I, J and K (R1) - three animals, not the
  row's "four" - each reported individually and never pooled.
* Row 11 "re-running selects the same model" is read as "re-running RE-APPLIES the
  user's recorded choice" (12A: no automatic selection); nothing here selects a model.

Row 7 REFUSES (raises :class:`CoverageMissingError`) when coverage is unavailable - the
``hasCoverage`` rule (``IMPLEMENTATION.md``, ``bulk_mixed_models.m`` row): a confound
check run without its confound covariate reports "no confound" for the wrong reason.

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
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat
from scipy.stats import t as student_t

from gems_blanking_v2.constants import ENG_BAND
from gems_blanking_v2.emit.masks import TAPER_S, ConsumerMask, MaskKey, MaskSpan
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.io.nan_interop import find_zero_runs
from gems_blanking_v2.model.labels import animal_key, is_test_animal

__all__ = [
    "KNOWN_DISAGREEMENTS",
    "SPEC_CONTRADICTIONS",
    "AcceptanceReport",
    "BlankRow",
    "CoverageMissingError",
    "Disagreement",
    "Disagreements",
    "Injection",
    "Row",
    "Status",
    "build_report",
    "expected_calibration_error",
    "masked_motion_fraction",
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
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def _not_computable(number: int, name: str, condition: str, missing: str) -> Row:
    return Row(number, name, Status.NOT_COMPUTABLE, condition,
               reason=f"missing input: {missing}")


# ---------------------------------------------------------------------------
# row 1 - candidate recall on injected synthetics
# ---------------------------------------------------------------------------

ROW1_RECALL: Final = 0.98
ROW1_MAX_CANDIDATES: Final = 3000


@dataclass(frozen=True, slots=True)
class Injection:
    """One injected synthetic: did it produce a candidate, and does it matter to anyone."""

    recording: str
    detected: bool
    above_some_tolerance: bool


def row1_candidate_recall(injections: Sequence[Injection] | None,
                          candidates_per_recording: Mapping[str, int] | None) -> Row:
    """>= 98% of injections above some consumer's tolerance detected, <= 3000 candidates."""
    cond = (f">= {ROW1_RECALL:.0%} of injected synthetics above at least one consumer's "
            f"tolerance produce a candidate, with <= {ROW1_MAX_CANDIDATES} candidates/recording")
    if not injections:
        return _not_computable(1, "candidate_recall", cond,
                               "injection results (task 09 synthetic replay)")
    if not candidates_per_recording:
        return _not_computable(1, "candidate_recall", cond, "candidate counts per recording")
    counted = [i for i in injections if i.above_some_tolerance]
    if not counted:
        return _not_computable(1, "candidate_recall", cond,
                               "no injection is above any consumer's tolerance")
    recall = sum(i.detected for i in counted) / len(counted)
    worst = max(candidates_per_recording.values())
    ok = recall >= ROW1_RECALL and worst <= ROW1_MAX_CANDIDATES
    return Row(1, "candidate_recall", Status.PASS if ok else Status.FAIL, cond, recall,
               {"n_counted": len(counted),
                "n_below_every_tolerance": len(injections) - len(counted),
                "max_candidates_per_recording": worst})


# ---------------------------------------------------------------------------
# row 2 - class balance
# ---------------------------------------------------------------------------

ROW2_MIN_FRACTION: Final = 0.05


def row2_class_balance(labelled: Mapping[str, tuple[int, int]] | None) -> Row:
    """``labelled``: recording -> (true-positive candidates, candidates), labelled recordings."""
    cond = f">= {ROW2_MIN_FRACTION:.0%} of candidates are true positives on labelled recordings"
    if not labelled:
        return _not_computable(2, "class_balance", cond,
                               "candidates judged on labelled recordings (task 10 / labelling A)")
    tp = sum(a for a, _ in labelled.values())
    n = sum(b for _, b in labelled.values())
    if n == 0:
        return _not_computable(2, "class_balance", cond, "no candidates on labelled recordings")
    frac = tp / n
    per = {r: a / b for r, (a, b) in labelled.items() if b}
    return Row(2, "class_balance", Status.PASS if frac >= ROW2_MIN_FRACTION else Status.FAIL,
               cond, frac, {"n_candidates": n, "n_true_positive": tp,
                            "min_recording_fraction": min(per.values()) if per else None})


# ---------------------------------------------------------------------------
# row 3 - no zeros, every boundary tapered
# ---------------------------------------------------------------------------


def _boundaries_tapered(original: F64, emitted: F64, fs: float) -> tuple[int, int]:
    """Count ``(boundaries, tapered)`` NaN-run boundaries of an emitted array.

    A boundary is tapered when the valid samples beside the NaN run are attenuated
    (0 < |y / x| < 1) across the taper length.
    """
    bad = np.isnan(emitted)
    d = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    n_taper = int(round(TAPER_S * fs))
    total = tapered = 0
    for s in np.flatnonzero(d == 1):
        if s >= n_taper:
            total += 1
            tapered += _attenuated(original[s - n_taper:s], emitted[s - n_taper:s])
    for e in np.flatnonzero(d == -1):
        if e + n_taper <= emitted.size:
            total += 1
            tapered += _attenuated(original[e:e + n_taper], emitted[e:e + n_taper])
    return total, tapered


def _attenuated(x: F64, y: F64) -> int:
    nz = np.abs(x) > 0
    if not nz.any() or not np.isfinite(y).all():
        return 0
    w = y[nz] / x[nz]
    return int(bool(np.all(w > 0) and np.all(w < 1)))


def row3_no_zeros(emitted: Sequence[tuple[str, F64, F64]] | None, fs: float | None) -> Row:
    """Check emitted yOut arrays: no zero runs, every boundary tapered.

    ``emitted``: ``(name, original signal, emitted yOut)``.

    The MATLAB handoff carries blank spans, which cannot carry a taper; the taper is
    checked on the emitted signal arrays (``emit.masks.apply_mask``).
    """
    cond = "no exact-zero run (> 2 samples) in any emitted yOut; every mask boundary tapered"
    if not emitted or fs is None:
        return _not_computable(3, "no_zeros", cond, "emitted yOut arrays with their originals")
    runs, bounds, taps = {}, 0, 0
    for name, x, y in emitted:
        zr = find_zero_runs(np.asarray(y, dtype=np.float64))
        if zr:
            runs[name] = len(zr)
        b, t = _boundaries_tapered(np.asarray(x, np.float64), np.asarray(y, np.float64), fs)
        bounds += b
        taps += t
    ok = not runs and taps == bounds
    return Row(3, "no_zeros", Status.PASS if ok else Status.FAIL, cond, float(len(runs)),
               {"arrays": len(emitted), "arrays_with_zero_runs": runs, "boundaries": bounds,
                "tapered": taps})


# ---------------------------------------------------------------------------
# row 4 - cardiac scope
# ---------------------------------------------------------------------------

ROW4_BANDS_FLAT: Final[tuple[str, ...]] = (ENG_BAND, "0.5-3", "0-2")


def row4_cardiac_scope(verdicts: Sequence[Mapping[str, str]] | None) -> Row:
    """``verdicts``: after masking, ``{channel, band, status}`` from ``physio.cardiac_window``.

    Flat = ``no_window`` (resolved, nothing above the null). ``measured`` fails.
    ``unresolvable`` cannot show flatness, so any such band makes the row not computable.
    ENG is 300-3000 Hz (R8), not the row's 300-5000.
    """
    cond = "peri-R profile flat in 300-3000 Hz and below 3 Hz after masking"
    if not verdicts:
        return _not_computable(4, "cardiac_scope", cond,
                               "peri-R verdicts on masked recordings "
                               "(cardiac_window after masking)")
    rel = [v for v in verdicts if v["band"] in ROW4_BANDS_FLAT]
    if not rel:
        return _not_computable(4, "cardiac_scope", cond, "no verdict for 300-3000, 0.5-3 or 0-2 Hz")
    measured = [f"{v['channel']}/{v['band']}" for v in rel if v["status"] == "measured"]
    blind = [f"{v['channel']}/{v['band']}" for v in rel
             if v["status"] in ("unresolvable", "not_applicable")]
    if measured:
        return Row(4, "cardiac_scope", Status.FAIL, cond, float(len(measured)),
                   {"not_flat": measured, "unresolvable": blind})
    if blind:
        return Row(4, "cardiac_scope", Status.NOT_COMPUTABLE, cond, None,
                   {"unresolvable": blind}, "the method cannot see these bands; flatness unshown")
    return Row(4, "cardiac_scope", Status.PASS, cond, 0.0, {"n_verdicts": len(rel)})


# ---------------------------------------------------------------------------
# row 5 - retention
# ---------------------------------------------------------------------------

SLOW_BANDS: Final[tuple[str, ...]] = ("0.5-3", "0-2")


def row5_retention(masks: Mapping[str, Mapping[MaskKey, ConsumerMask]] | None,
                   baseline_retention: Mapping[str, float] | None) -> Row:
    """Retention per band per channel against the all-channel full-band baseline.

    ``masks``: recording -> its consumer masks. ``baseline_retention``: recording -> the
    retention of the current all-channel blanking. PASS when the median ENG-band
    retention is at least the baseline's and above the median of every slow band.
    "Substantially" has no number in the spec; the margins are reported, not gated.
    """
    cond = ("retention per band per channel vs the all-channel full-band baseline; "
            "ENG-band retention higher than the slow bands")
    if not masks:
        return _not_computable(5, "retention", cond, "emitted masks per recording")
    if not baseline_retention:
        return _not_computable(5, "retention", cond,
                               "baseline retention (current all-channel blanking) per recording")
    by_band: dict[str, list[float]] = {}
    per: dict[str, float] = {}
    for rec, ms in masks.items():
        for (c, s, b), m in ms.items():
            by_band.setdefault(b, []).append(m.retention)
            per[f"{rec}|{c}|{s}|{b}"] = m.retention
    med = {b: float(np.median(v)) for b, v in by_band.items()}
    if ENG_BAND not in med:
        return _not_computable(5, "retention", cond, "no ENG-band mask")
    base = float(np.median([baseline_retention[r] for r in masks if r in baseline_retention]))
    slow = {b: med[b] for b in SLOW_BANDS if b in med}
    ok = med[ENG_BAND] >= base and all(med[ENG_BAND] > v for v in slow.values())
    return Row(5, "retention", Status.PASS if ok else Status.FAIL, cond, med[ENG_BAND],
               {"median_by_band": med, "baseline_median": base,
                "eng_minus_slow": {b: med[ENG_BAND] - v for b, v in slow.items()},
                "per_band_per_channel": per})


# ---------------------------------------------------------------------------
# row 6 - cross-animal (R1: new-cohort I/J/K, individually)
# ---------------------------------------------------------------------------


def row6_cross_animal(per_animal: Mapping[tuple[str, str], tuple[int, int]] | None) -> Row:
    """``per_animal``: ``(cohort, animal) -> (found, total)`` blind-test events, never pooled.

    Only R1's prospective test set (``is_test_animal``) belongs here; anything else
    raises. PASS when every one of new:I, new:J, new:K is reported. The spec gives no
    recall threshold for this row; the numbers are reported individually.
    """
    cond = "blind-test event recall per never-seen animal (R1: new-cohort I, J, K), never pooled"
    if not per_animal:
        return _not_computable(6, "cross_animal", cond,
                               "blind-test events per test animal from a model "
                               "trained without them")
    for cohort, animal in per_animal:
        if not is_test_animal(cohort, animal):
            msg = f"{animal_key(cohort, animal)} is not in R1's test set; it cannot be in row 6"
            raise ValueError(msg)
    recall = {animal_key(c, a): (f / n if n else math.nan) for (c, a), (f, n) in per_animal.items()}
    missing = sorted({"new:I", "new:J", "new:K"} - set(recall))
    detail = {"recall_by_animal": recall,
              "counts": {animal_key(c, a): [f, n] for (c, a), (f, n) in per_animal.items()}}
    if missing or any(math.isnan(v) for v in recall.values()):
        return Row(6, "cross_animal", Status.NOT_COMPUTABLE, cond, None, detail,
                   "missing input: blind-test events for "
                   f"{missing or 'an animal with zero events'}")
    return Row(6, "cross_animal", Status.PASS, cond, None, detail)


# ---------------------------------------------------------------------------
# row 7 - coverage confound
# ---------------------------------------------------------------------------

MIN_COVERAGE: Final = 0.5
"""``bulk_mixed_models.m``'s exclusion (``minCoverage``)."""
CONFOUND_ALPHA: Final = 0.05


class CoverageMissingError(RuntimeError):
    """Row 7 refuses to run without coverage (``hasCoverage`` false)."""


@dataclass(frozen=True, slots=True)
class BlankRow:
    """One recording for the confound regression.

    ``masked_motion_frac`` is the blank fraction from MOTION masking only - the stim
    ``excluded_epoch`` and cardiac windows are not in it. ``coverage`` is the valid fraction
    the coverage covariate needs (``None`` = unknown).
    """

    recording: str
    condition: str
    mode: str
    masked_motion_frac: float
    coverage: float | None


NON_MOTION_REASONS: Final[frozenset[str]] = frozenset({
    "cuff_distrusted", "per_minute_cuff_distrust", "line_noise_cuff_minute",
    "excluded_epoch", "cardiac"})
"""Mask-span reason codes that are NOT motion: they stay out of row 7's numerator."""


def masked_motion_fraction(spans: Iterable[MaskSpan], duration_s: float) -> float:
    """Fraction of ``duration_s`` covered by motion mask spans (union), for one mask.

    Spans whose reason is in :data:`NON_MOTION_REASONS` (stim epoch, cardiac windows, cuff
    distrust, line-noise minutes) are left out, so the regression measures the artifact
    and not the protocol.
    """
    ivs = sorted((s.start_s, s.stop_s) for s in spans if s.reason not in NON_MOTION_REASONS)
    total, end = 0.0, -math.inf
    for a0, b in ivs:
        a = max(a0, end)
        if b > a:
            total += b - a
            end = b
    return total / duration_s


def _wls(x: F64, y: F64, w: F64) -> tuple[F64, F64, int]:
    sw = np.sqrt(w)
    xw, yw = x * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = yw - xw @ beta
    dof = x.shape[0] - x.shape[1]
    if dof <= 0:
        msg = "too few recordings for the confound regression"
        raise ValueError(msg)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(xw.T @ xw)
    return beta, np.sqrt(np.diag(cov)), dof


def row7_coverage_confound(rows: Sequence[BlankRow] | None, *, has_coverage: bool) -> Row:
    """Regress masked-motion blank fraction on condition, with coverage, by mode too.

    REFUSES (raises :class:`CoverageMissingError`) when ``has_coverage`` is false or any
    row lacks coverage. As ``bulk_mixed_models.m``: rows with coverage < 0.5 are excluded,
    the regression is weighted by coverage, and coverage is a covariate. Mode enters as a
    factor when it varies. PASS when the condition coefficient's 95% CI includes 0; the
    spec gives no equivalence margin, so the CI is reported for a ruling.
    """
    cond = ("masked-motion blank fraction does not depend on condition "
            "(WLS on condition + coverage [+ mode], coverage-weighted, coverage >= 0.5)")
    if not has_coverage or (rows and any(r.coverage is None for r in rows)):
        msg = ("row 7 refuses to run without coverage (hasCoverage is false): a confound check "
               "without its confound covariate reports 'no confound' for the wrong reason")
        raise CoverageMissingError(msg)
    if not rows:
        return _not_computable(7, "coverage_confound", cond,
                               "masked-motion blank fraction per recording with condition and mode")
    kept = [r for r in rows if r.coverage is not None and r.coverage >= MIN_COVERAGE]
    conds = sorted({r.condition for r in kept})
    if len(conds) != 2:  # noqa: PLR2004
        return _not_computable(7, "coverage_confound", cond,
                               f"both conditions after the coverage exclusion (have {conds})")
    modes = sorted({r.mode for r in kept})
    cols = [np.ones(len(kept)), np.array([r.condition == conds[1] for r in kept], float),
            np.array([r.coverage for r in kept], float)]
    cols += [np.array([r.mode == m for r in kept], float) for m in modes[1:]]
    x = np.column_stack(cols)
    y = np.array([r.masked_motion_frac for r in kept], float)
    w = np.array([r.coverage for r in kept], float)
    beta, se, dof = _wls(x, y, w)
    tcrit = float(student_t.ppf(1 - CONFOUND_ALPHA / 2, dof))
    coef, s = float(beta[1]), float(se[1])
    ci = (coef - tcrit * s, coef + tcrit * s)
    p = float(2 * student_t.sf(abs(coef / s), dof)) if s > 0 else 0.0
    by_cond = {c: float(np.mean([r.masked_motion_frac for r in kept if r.condition == c]))
               for c in conds}
    by_mode = {m: float(np.mean([r.masked_motion_frac for r in kept if r.mode == m]))
               for m in modes}
    ok = ci[0] <= 0.0 <= ci[1]
    return Row(7, "coverage_confound", Status.PASS if ok else Status.FAIL, cond, coef,
               {"term": f"condition[{conds[1]} vs {conds[0]}]", "se": s, "ci95": list(ci),
                "p": p, "dof": dof, "n_used": len(kept), "n_excluded_low_coverage":
                len(rows) - len(kept), "mean_by_condition": by_cond, "mean_by_mode": by_mode,
                "mode_is_factor": len(modes) > 1})


# ---------------------------------------------------------------------------
# row 8 - downstream (MATLAB consumer outputs)
# ---------------------------------------------------------------------------

DOWNSTREAM_METRICS: Final[tuple[str, ...]] = (
    "fracISIclean", "slow_wave_non_nan_fraction", "dfa_alpha2_available", "nRR_used",
    "step5f_median_epoch_s")


def row8_downstream(before: Mapping[str, float] | None, after: Mapping[str, float] | None,
                    between_animal_var: tuple[float, float] | None) -> Row:
    """All five downstream metrics up, and the between-animal endpoint variance down.

    ``before``/``after``: metric -> value under the current and the new blanking (MATLAB
    consumer outputs); ``between_animal_var``: ``(before, after)``.
    """
    cond = "fracISIclean, slow-wave non-NaN fraction, DFA alpha2 availability, nRR_used, " \
           "median step5f epoch length all up; between-animal endpoint variance down"
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


def row10_mode_comparison(curves: Mapping[tuple[str, str], Path] | None,
                          b_vs_c: Mapping[str, Any] | None, *,
                          comparisons: Iterable[tuple[str, str]] = ()) -> Row:
    """Learning curves per (mode, animal); B-vs-C verdict by R9; never A vs C.

    ``b_vs_c``: ``{"f1_diff": C - B, "ci95": [lo, hi]}``. ``comparisons`` lists every
    comparison the build reported; an A-vs-C one fails the row. C > B triggers task 11.
    """
    cond = ("learning curve per mode per animal; A-vs-C never compared; B-vs-C verdict "
            "stated (R9: C wins iff F1 +0.03 and CI excludes 0), task 11 triggered if C > B")
    if any({a, b} == {"A", "C"} for a, b in comparisons):
        return Row(10, "mode_comparison", Status.FAIL, cond,
                   reason="an A-vs-C comparison was reported (invariant 12)")
    if not curves:
        return _not_computable(10, "mode_comparison", cond, "learning curves per mode per animal")
    absent = sorted(f"{m}/{a}" for (m, a), p in curves.items() if not Path(p).is_file())
    if absent:
        return _not_computable(10, "mode_comparison", cond, f"learning-curve plots {absent}")
    if not b_vs_c or "f1_diff" not in b_vs_c or "ci95" not in b_vs_c:
        return _not_computable(10, "mode_comparison", cond, "the B-vs-C F1 difference and its CI")
    diff, (lo, hi) = float(b_vs_c["f1_diff"]), (float(x) for x in b_vs_c["ci95"])
    c_wins = diff >= MODE_C_MARGIN and lo > 0
    verdict = "C beats B" if c_wins else "C does not beat B"
    return Row(10, "mode_comparison", Status.PASS, cond, diff,
               {"verdict": verdict, "ci95": [lo, hi], "n_curves": len(curves),
                "task11_investigation_triggered": diff > 0})


# ---------------------------------------------------------------------------
# row 11 - model routing
# ---------------------------------------------------------------------------


def _model_of(mask_file: Path) -> dict[str, Any]:
    m = loadmat(mask_file)
    if "provenance_json" not in m:
        msg = f"{Path(mask_file).name}: no provenance"
        raise ProvenanceError(msg)
    return dict(MaskProvenance.from_json(str(m["provenance_json"][0])).model)


def row11_model_routing(mask_files: Mapping[str, Path] | None,
                        assignments: Mapping[str, Mapping[str, Any]] | None,
                        rerun_files: Mapping[str, Path] | None = None) -> Row:
    """Every mask names its ModelSpec; the user's choice is logged; a re-run re-applies it.

    ``mask_files``: recording -> mask file. ``assignments``: recording -> the model the
    user chose for it (the logged routing rule). ``rerun_files``: a second run's files.
    Read as 12A requires: nothing selects a model; the recorded choice is re-applied.
    """
    cond = ("every emitted mask names its ModelSpec; the user's model choice is logged per "
            "recording; re-running re-applies the same recorded choice (12A: no auto-selection)")
    if not mask_files:
        return _not_computable(11, "model_routing", cond, "emitted mask files")
    if not assignments:
        return _not_computable(11, "model_routing", cond, "the per-recording model assignment log")
    problems: dict[str, str] = {}
    for rec, path in mask_files.items():
        try:
            model = _model_of(path)
        except (ProvenanceError, OSError, ValueError) as exc:
            problems[rec] = f"mask provenance: {exc}"
            continue
        chosen = assignments.get(rec)
        if chosen is None:
            problems[rec] = "no logged model choice"
        elif {k: model.get(k) for k in ("mode", "version", "corpus_hash")} != {
                k: chosen.get(k) for k in ("mode", "version", "corpus_hash")}:
            problems[rec] = "mask names a different model from the logged choice"
        if rerun_files and rec in rerun_files:
            try:
                again = _model_of(rerun_files[rec])
            except (ProvenanceError, OSError, ValueError) as exc:
                problems[rec] = f"re-run provenance: {exc}"
                continue
            if again != model:
                problems[rec] = "a re-run used a different model"
    rerun_note = {} if rerun_files else {"rerun": "no re-run supplied; re-application unchecked"}
    return Row(11, "model_routing", Status.FAIL if problems else Status.PASS, cond,
               float(len(problems)), {"n_recordings": len(mask_files), "problems": problems,
                                      **rerun_note})


# ---------------------------------------------------------------------------
# row 12 - calibration
# ---------------------------------------------------------------------------

ECE_MAX: Final = 0.05
"""R9: expected calibration error <= 0.05 on held-out data."""
ECE_BINS: Final = 10


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


def row12_calibration(held_out: Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]] | None) -> Row:
    """Every shipped model's ECE on held-out data <= 0.05 (R9); reliability curve reported."""
    cond = f"every shipped model calibrated on held-out data; ECE <= {ECE_MAX} (R9)"
    if not held_out:
        return _not_computable(12, "calibration", cond,
                               "held-out P(motion) and labels per shipped model (task 12)")
    eces, curves = {}, {}
    for model, (p, y) in held_out.items():
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
    Disagreement("spike consumer settling (task 13)", "expect 30-50 ms",
                 "5.1 ms (impz, 1% of peak, 300-3000 Hz order-4 at 24414 Hz)",
                 "extent.tolerance.consumer_settling", "2026-10-07"),
    Disagreement("tripole sigma reduction", "~6x", "2.5-2.8x", "A.5 measurement", "2026-09-19"),
    Disagreement("mmc / slow_wave input", "stomach_ref",
                 "raw ANT1-3 (extract_mmc.m; batch_process.m swData = signal(:, 3:5))",
                 "ruling 2026-10-06 item 3; R8", "2026-10-06"),
    Disagreement("R-peak minimum interval R_MIN", "90 ms", "60 ms (animal J RR median 164.7 ms)",
                 "task 05 measurement", "2026-09-21"),
)
"""Seed: disagreements already established in the record before this report."""


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
     "artifact that matters as one above a consumer's tolerance",
     "implemented": "recall over injections above at least one consumer's tolerance"},
    {"row": "4", "conflict": "row 4 says 300-5000 Hz; R8 / A.5b put the ENG band at 300-3000",
     "implemented": "300-3000 Hz"},
    {"row": "6", "conflict": "row 6 says 'the four never-seen animals'; R1's prospective test "
     "set is new-cohort I, J, K (three)", "implemented": "new:I, new:J, new:K, individually"},
    {"row": "11", "conflict": "row 11 'routing rule ... re-running selects the same model' vs "
     "12A 'no automatic selection, no default model'",
     "implemented": "the user's recorded choice is re-applied; nothing selects"},
    {"row": "3", "conflict": "row 3 'every mask boundary tapered' vs the MATLAB handoff, whose "
     "blank spans cannot carry a taper", "implemented": "taper checked on emitted signal "
     "arrays; the MATLAB side is NaN spans only"},
    {"row": "5", "conflict": "'substantially higher' has no number",
     "implemented": "ENG median above baseline and above each slow band; margins reported"},
    {"row": "7", "conflict": "'must not depend on condition' needs an equivalence margin; the "
     "spec gives none", "implemented": "95% CI of the condition coefficient includes 0; CI "
     "reported for a ruling"},
    {"row": "6", "conflict": "no recall threshold is given for the cross-animal row",
     "implemented": "PASS = reported for each of I, J, K; numbers for a ruling"},
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
                "disagreements": [asdict(d) for d in self.disagreements],
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
        lines += [f"- {d.constant}: spec {d.spec_value}; measured {d.measured} ({d.source}, "
                  f"{d.date})" for d in self.disagreements]
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
