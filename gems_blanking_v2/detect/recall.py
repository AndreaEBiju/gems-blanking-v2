"""Task 09 gate scoring: the candidate generator's recall against blind audit marks.

**Everything here was declared before any audit mark existed** (IMPLEMENTATION.md,
task 09, "PRE-DECLARED 2026-09-26"), so nothing about the scoring can be chosen
after seeing Andrea's marks:

* **Unit.** One artifact = one interval committed in a blind span.
* **Covered.** At least one candidate overlaps the artifact by ANY amount (a
  positive-length intersection of half-open intervals; touching end to end is
  not overlap). The two-stage design needs a candidate *proposed* inside the
  artifact; task 13's extent sets the boundaries.
* **Statistic.** ``recall = covered / found``, with an exact Clopper-Pearson 95%
  interval and a span-level bootstrap interval (artifacts within one span are
  not independent, so the bootstrap resamples SPANS). The wider is quoted.
* **Secondary only.** The fraction of each artifact's duration that candidates
  cover, and recall at >= 50% overlap.
* **Sequential rounds.** Any miss: stop, diagnose each from the z-traces, fix task
  07, and only then label more. Zero misses: draw another round with a fresh
  recorded seed and the same stratification, until the lower bound clears 98%.

**Two choices this module had to make, made here before any marks, and reported
for ratification:**

1. *Which lower bound decides the gate.* The spec quotes a two-sided 95% interval
   but projects the artifacts needed from ``0.05**(1/n) >= 0.98 -> n >= 149``,
   which is the ONE-sided 95% bound (two-sided 95% would need n >= 183). The gate
   and the projection use the one-sided 95% bound, as the projection does, and
   the MORE conservative of the Clopper-Pearson and bootstrap versions of it;
   the two-sided intervals are reported beside it.
2. *Where "z low" ends.* A miss is a **generator blind spot** when every band's
   z stays below the generator's own ``z_exit`` (it never even sustains), a
   **threshold** miss when z reaches ``z_exit`` but not ``z_enter``, and
   **gated** when z reached ``z_enter`` and still no candidate was proposed - a
   case the two named classes do not cover: something after thresholding
   (combine rule, cardiac suppression, duration or merge rules) rejected it. Both
   numbers are read from ``candidate_report``'s own defaults, not restated.

Both were ratified 2026-09-27, with two additions (IMPLEMENTATION.md, task 09):

* **Only rounds labelled after the last fix count toward the gate.** Once a miss
  has changed task 07, the marks that revealed it are tuning data. A round's
  "labelled at" is its first committed span; a fix's time is the ``fixed_at``
  recorded with it. :func:`pooled_gate` pools gate-mode scores of rounds
  labelled after the last fix and says which it pooled and which it left out. A
  ``tuning`` re-score (the current generator on an old round) is written to its
  own file, labelled "tuning check, not gate evidence", and never pooled.
* **A score records exactly what it scored.** It carries the generator's
  parameters and the package commit plus a hash of the package source
  (:func:`generator_provenance`). The window writes :func:`candidate_digest` of
  what it revealed into the span record; a gate-mode score REFUSES a span whose
  recomputed candidates do not match (:class:`RevealMismatchError`), and scores a span
  with no digest (an app started before digests existed) with a warning saying so.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy.stats import beta

from gems_blanking_v2.detect.candidates import candidate_report
from gems_blanking_v2.io.store import GemsStore, atomic_write_text

__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "BUDGET_MIN_RECORDINGS",
    "BUDGET_QUANTILE",
    "CANDIDATE_BUDGET",
    "CONFIDENCE",
    "DIGEST_RULE",
    "GATE_RECALL",
    "HALF_OVERLAP",
    "TUNING_LABEL",
    "Z_ENTER",
    "Z_EXIT",
    "ArtifactScore",
    "BandZ",
    "MissDiagnosis",
    "RevealMismatchError",
    "RoundScore",
    "SpanInput",
    "artifacts_needed",
    "budget_key",
    "budget_record",
    "budget_status",
    "candidate_digest",
    "check_next_round",
    "clopper_pearson",
    "covered_fraction",
    "diagnose_miss",
    "generator_provenance",
    "is_covered",
    "last_fix_at",
    "load_round",
    "next_round_gate",
    "one_sided_lower",
    "pooled_gate",
    "score_round",
    "score_stored_round",
    "source_sha256",
    "span_bootstrap",
    "span_id",
    "write_budget",
]

F64 = npt.NDArray[np.float64]
Verdict = Literal["generator_blind_spot", "threshold", "gated", "undiagnosed"]

GATE_RECALL: Final = 0.98
"""The recall the gate must PROVE: its lower bound, not its point estimate."""

CONFIDENCE: Final = 0.95
"""Confidence of every interval and bound here."""

HALF_OVERLAP: Final = 0.5
"""The secondary recall's bar: at least this fraction of an artifact covered."""

BOOTSTRAP_DRAWS: Final = 10_000
BOOTSTRAP_SEED: Final = 20_260_926
"""Fixed and declared, so a re-score reproduces the same bootstrap interval."""

_DEFAULTS = inspect.signature(candidate_report).parameters
Z_ENTER: Final[float] = float(_DEFAULTS["z_enter"].default)
"""The generator's entry threshold, read from ``candidate_report`` (invariant 39)."""
Z_EXIT: Final[float] = float(_DEFAULTS["z_exit"].default)
"""The generator's hysteresis floor: below it a pair is not even sustaining."""

TUNING_LABEL: Final = "tuning check, not gate evidence"
"""What a re-score of an old round with the current generator says it is."""

DIGEST_RESOLUTION_S: Final = 1e-6
DIGEST_RULE: Final = (
    "sha256 of canonical JSON {intervals_us: sorted [[round(start/1e-6), "
    "round(stop/1e-6)], ...]}, ASCII, sorted keys, separators (',', ':')"
)
"""How a revealed candidate list is digested - ONE function for the window that
writes it and the scorer that checks it (invariant 22). Candidates sit on the 10 ms
frame grid, so microsecond rounding is exact for them and immune to float noise."""


class RevealMismatchError(ValueError):
    """Recomputed candidates differ from what the window revealed: refuse to score."""


def candidate_digest(intervals: npt.ArrayLike) -> str:
    """Return the canonical digest of a candidate list (:data:`DIGEST_RULE`)."""
    a = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    rows = sorted(tuple(r) for r in np.round(a / DIGEST_RESOLUTION_S).astype(np.int64).tolist())
    payload = json.dumps({"intervals_us": rows}, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                             check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def source_sha256(package_dir: Path) -> str:
    """Hash every ``.py`` under ``package_dir``, in sorted relative-path order.

    Each file contributes its relative POSIX path and its LF-normalised bytes, so the
    hash is identical on Windows and macOS whatever git did to line endings.
    """
    h = hashlib.sha256()
    for f in sorted(package_dir.rglob("*.py"), key=lambda q: q.relative_to(package_dir).as_posix()):
        h.update(f.relative_to(package_dir).as_posix().encode("utf-8") + b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def generator_provenance() -> dict[str, Any]:
    """Return the generator a score used: its parameters, gate settings and code.

    ``source_sha256`` hashes the whole ``gems_blanking_v2`` package, so it names the
    code exactly whether or not it is committed; ``commit`` and ``dirty`` say how it
    relates to git.
    """
    package = Path(__file__).resolve().parents[1]
    params = {k: v.default for k, v in _DEFAULTS.items()
              if v.default is not inspect.Parameter.empty
              and isinstance(v.default, (int, float, str, bool, type(None)))}
    commit = _git(package.parent, "rev-parse", "HEAD")
    status = _git(package.parent, "status", "--porcelain", "--", package.name)
    out: dict[str, Any] = {
        "generator": "gems_blanking_v2.detect.candidates.candidate_report",
        "parameters": params,
        "gate": {"gate_recall": GATE_RECALL, "confidence": CONFIDENCE,
                 "bound": "one-sided, min(Clopper-Pearson, span bootstrap)",
                 "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                 "half_overlap": HALF_OVERLAP},
        "source_sha256": source_sha256(package),
    }
    if commit:
        out["commit"] = commit
        out["dirty"] = bool(status)
    return out


class BandZ(Protocol):
    """One band's per-frame z, as the audit window's reveal produces it.

    ``z_max[i]`` is the largest z over the band's signals in frame ``i`` (``nan`` =
    not assessable) and ``winner[i]`` names the signal that carried it.
    """

    band: str
    z_max: F64
    winner: tuple[str, ...]
    grid_s: float


@dataclass(frozen=True, slots=True)
class SpanInput:
    """One committed blind span: its marks, and what the reveal proposed there.

    ``artifacts`` and ``candidates`` are ``(n, 2)`` ``[start, stop)`` seconds on
    the recording's own timeline. ``traces`` may be ``None`` when the z-traces
    are unavailable; misses in such a span are then ``undiagnosed``, never guessed.
    """

    span_id: str
    recording_id: str
    animal: str
    condition: str
    start_s: float
    stop_s: float
    artifacts: F64
    candidates: F64
    traces: Sequence[BandZ] | None = None

    @property
    def minutes(self) -> float:
        """Span length, minutes."""
        return (self.stop_s - self.start_s) / 60.0


@dataclass(frozen=True, slots=True)
class MissDiagnosis:
    """Why the generator missed one artifact, from the z inside it.

    ``max_z_by_band`` maps band -> ``(max z inside the artifact, signal carrying
    it)``; a band with no assessable frame there is absent.
    """

    verdict: Verdict
    max_z: float
    max_band: str | None
    max_signal: str | None
    max_z_by_band: dict[str, tuple[float, str]]
    z_enter: float
    z_exit: float

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready record; ``max_*`` absent when nothing was assessable."""
        out: dict[str, Any] = {
            "verdict": self.verdict, "z_enter": self.z_enter, "z_exit": self.z_exit,
            "max_z_by_band": {b: {"z": z, "signal": s}
                              for b, (z, s) in sorted(self.max_z_by_band.items())},
        }
        if math.isfinite(self.max_z):  # absent, never NaN, when nothing was assessable
            out.update(max_z=self.max_z, max_band=self.max_band, max_signal=self.max_signal)
        return out


@dataclass(frozen=True, slots=True)
class ArtifactScore:
    """One artifact: covered or not, how much, and - if missed - why."""

    span_id: str
    index: int
    start_s: float
    stop_s: float
    covered: bool
    covered_fraction: float
    diagnosis: MissDiagnosis | None = None

    @property
    def miss_id(self) -> str:
        """``<span_id>#<index>``: the key a resolution is recorded under."""
        return f"{self.span_id}#{self.index}"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready record of this artifact."""
        out: dict[str, Any] = {
            "id": self.miss_id, "span_id": self.span_id, "start_s": self.start_s,
            "stop_s": self.stop_s, "covered": self.covered,
            "covered_fraction": self.covered_fraction,
        }
        if self.diagnosis is not None:
            out["diagnosis"] = self.diagnosis.to_json()
        return out


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def _intervals(x: npt.ArrayLike) -> F64:
    a = np.asarray(x, dtype=np.float64).reshape(-1, 2)
    if a.size and bool(np.any(a[:, 1] < a[:, 0])):
        msg = "an interval ends before it starts"
        raise ValueError(msg)
    return a


def is_covered(artifact: tuple[float, float], candidates: npt.ArrayLike) -> bool:
    """Whether any candidate overlaps ``artifact`` by a positive amount.

    Half-open intervals: ``[0, 1)`` and ``[1, 2)`` touch but do not overlap, so an
    adjacent candidate does not cover. Any positive intersection does - a candidate
    one sample long inside the artifact covers it.
    """
    a0, a1 = artifact
    c = _intervals(candidates)
    if not c.size:
        return False
    return bool(np.any(np.minimum(c[:, 1], a1) - np.maximum(c[:, 0], a0) > 0.0))


def covered_fraction(artifact: tuple[float, float], candidates: npt.ArrayLike) -> float:
    """Fraction of the artifact's duration inside the UNION of candidates.

    Overlapping candidates are merged first, so time covered twice counts once.
    """
    a0, a1 = artifact
    if a1 <= a0:
        return 0.0
    c = _intervals(candidates)
    clipped = [(max(s, a0), min(e, a1)) for s, e in c if min(e, a1) > max(s, a0)]
    total, end = 0.0, -math.inf
    for s, e in sorted(clipped):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total / (a1 - a0)


# ---------------------------------------------------------------------------
# the statistics
# ---------------------------------------------------------------------------


def clopper_pearson(k: int, n: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion ``k / n``."""
    if not 0 <= k <= n or n < 1:
        msg = f"need 0 <= k <= n and n >= 1, got k={k}, n={n}"
        raise ValueError(msg)
    tail = (1.0 - confidence) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return lo, hi


def one_sided_lower(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """Exact one-sided lower bound. At ``k == n`` this is ``(1 - confidence)**(1/n)``."""
    if not 0 <= k <= n or n < 1:
        msg = f"need 0 <= k <= n and n >= 1, got k={k}, n={n}"
        raise ValueError(msg)
    return 0.0 if k == 0 else float(beta.ppf(1.0 - confidence, k, n - k + 1))


def span_bootstrap(
    per_span: Sequence[tuple[int, int]], *, draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED, confidence: float = CONFIDENCE,
) -> tuple[float, float, float, int]:
    """Resample SPANS with replacement; return ``(lo, hi, one_sided_lo, n_empty)``.

    ``per_span`` is ``[(covered, found), ...]``. Each draw's recall is
    ``sum(covered) / sum(found)`` over the drawn spans; a draw whose spans hold no
    artifact at all has no recall and is dropped, and ``n_empty`` says how many.
    Percentile intervals: two-sided at ``confidence``, and the one-sided lower
    bound at the same confidence.
    """
    counts = np.asarray(per_span, dtype=np.float64).reshape(-1, 2)
    if not counts.size or counts[:, 1].sum() == 0:
        msg = "no artifacts in any span: recall is undefined"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, counts.shape[0], size=(draws, counts.shape[0]))
    cov = counts[idx, 0].sum(axis=1)
    found = counts[idx, 1].sum(axis=1)
    keep = found > 0
    r = cov[keep] / found[keep]
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(r, [tail, 1.0 - tail])
    one = np.quantile(r, 1.0 - confidence)
    return float(lo), float(hi), float(one), int((~keep).sum())


def artifacts_needed(misses: int, *, gate: float = GATE_RECALL,
                     confidence: float = CONFIDENCE, limit: int = 100_000) -> int:
    """Fewest artifacts, with ``misses`` misses and none more, for the bound to clear.

    At zero misses this is the spec's ``0.05**(1/n) >= 0.98 -> n >= 149``.
    """
    n = max(misses, 1)
    while n <= limit:
        if one_sided_lower(n - misses, n, confidence) >= gate:
            return n
        n += 1
    msg = f"no n <= {limit} clears {gate} with {misses} misses"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# diagnosis
# ---------------------------------------------------------------------------


def diagnose_miss(
    artifact: tuple[float, float], traces: Sequence[BandZ] | None, *,
    z_enter: float = Z_ENTER, z_exit: float = Z_EXIT,
) -> MissDiagnosis:
    """Classify one miss from the largest z inside it, per band.

    A frame counts when it overlaps the artifact. ``nan`` frames (not assessable)
    are ignored; with no traces, or no assessable frame anywhere, the verdict is
    ``undiagnosed`` - a miss is never classified without evidence.
    """
    a0, a1 = artifact
    by_band: dict[str, tuple[float, str]] = {}
    for t in traces or ():
        z = np.asarray(t.z_max, dtype=np.float64)
        i0 = max(int(math.floor(a0 / t.grid_s)), 0)
        i1 = min(int(math.ceil(a1 / t.grid_s)), z.size)
        window = z[i0:i1]
        if not window.size or not bool(np.isfinite(window).any()):
            continue
        j = i0 + int(np.nanargmax(window))
        by_band[t.band] = (float(z[j]), str(t.winner[j]) if j < len(t.winner) else "")
    if not by_band:
        return MissDiagnosis("undiagnosed", math.nan, None, None, {}, z_enter, z_exit)
    band = max(by_band, key=lambda b: by_band[b][0])
    zmax, signal = by_band[band]
    verdict: Verdict
    if zmax >= z_enter:
        verdict = "gated"
    elif zmax >= z_exit:
        verdict = "threshold"
    else:
        verdict = "generator_blind_spot"
    return MissDiagnosis(verdict, zmax, band, signal, by_band, z_enter, z_exit)


# ---------------------------------------------------------------------------
# the round
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoundScore:
    """One audit round, scored as pre-declared. ``to_json`` / ``report`` render it."""

    plan_id: str
    seed: int | None
    artifacts: list[ArtifactScore]
    per_span: list[dict[str, Any]]
    minutes: float
    rounds_scored: int = 1
    provenance: dict[str, Any] = field(default_factory=dict)
    mode: Literal["gate", "tuning"] = "gate"
    warnings: list[str] = field(default_factory=list)
    pooled: dict[str, Any] | None = None

    @property
    def found(self) -> int:
        """Artifacts committed in the round."""
        return len(self.artifacts)

    @property
    def covered(self) -> int:
        """Artifacts at least one candidate overlaps."""
        return sum(a.covered for a in self.artifacts)

    @property
    def misses(self) -> list[ArtifactScore]:
        """The artifacts no candidate overlaps."""
        return [a for a in self.artifacts if not a.covered]

    def statistics(self) -> dict[str, Any]:
        """Every number the round report quotes; recall keys absent at zero found."""
        out: dict[str, Any] = {
            "found": self.found, "covered": self.covered, "missed": len(self.misses),
            "minutes_labelled": self.minutes,
            "artifacts_per_minute": self.found / self.minutes if self.minutes else 0.0,
        }
        if not self.found:
            return out
        k, n = self.covered, self.found
        cp = clopper_pearson(k, n)
        cp1 = one_sided_lower(k, n)
        blo, bhi, b1, empty = span_bootstrap(
            [(s["covered"], s["found"]) for s in self.per_span])
        quoted = "clopper_pearson" if (cp[1] - cp[0]) >= (bhi - blo) else "span_bootstrap"
        gate_lower = min(cp1, b1)
        need = artifacts_needed(len(self.misses))
        per_round = n / self.rounds_scored
        more = max(need - n, 0)
        out.update(
            recall=k / n,
            clopper_pearson_95=[cp[0], cp[1]],
            span_bootstrap_95=[blo, bhi],
            bootstrap_draws_without_artifacts=empty,
            quoted_interval=quoted,
            quoted_95=list(cp) if quoted == "clopper_pearson" else [blo, bhi],
            gate_lower_bound_one_sided_95=gate_lower,
            gate_lower_bound_parts={"clopper_pearson": cp1, "span_bootstrap": b1},
            gate_cleared=gate_lower >= GATE_RECALL,
            artifacts_needed_at_zero_further_misses=need,
            more_artifacts_needed=more,
            more_rounds_projected=math.ceil(more / per_round) if more else 0,
            more_minutes_projected=(more / out["artifacts_per_minute"]
                                    if more and out["artifacts_per_minute"] else 0.0),
            secondary_mean_covered_fraction=float(np.mean(
                [a.covered_fraction for a in self.artifacts])),
            secondary_recall_at_half_overlap=sum(
                a.covered_fraction >= HALF_OVERLAP for a in self.artifacts) / n,
        )
        return out

    def next_step(self) -> str:
        """Return the sequential rule's verdict for this round.

        Clearing the gate is judged on the POOLED bound over eligible rounds
        (:func:`pooled_gate`), never on one round alone; a tuning re-score never
        decides anything.
        """
        if self.mode == "tuning":
            return (f"{TUNING_LABEL.upper()}: shows whether the current generator recovers "
                    "these marks; the gate and the sequential rule are unaffected")
        if not self.found:
            return "no artifacts found: recall is undefined; draw another round"
        if self.misses:
            return ("STOP: diagnose each miss, fix task 07, and record the fix before "
                    "any more labelling")
        if self.pooled is not None and self.pooled.get("gate_cleared"):
            return "gate cleared: the pooled lower bound is above 98%; no further round needed"
        return "zero misses: draw another round (fresh recorded seed, same stratification)"

    def to_json(self) -> dict[str, Any]:
        """Return the round's score record, as written beside its plan."""
        return {
            "plan_id": self.plan_id, **({"seed": self.seed} if self.seed is not None else {}),
            "mode": self.mode,
            "evidence": TUNING_LABEL if self.mode == "tuning" else "gate",
            "warnings": list(self.warnings),
            **({"pooled_gate": self.pooled} if self.pooled is not None else {}),
            "scoring": "PRE-DECLARED 2026-09-26, ratified 2026-09-27 (IMPLEMENTATION.md, task 09)",
            "gate_recall": GATE_RECALL, "confidence": CONFIDENCE,
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                          "unit": "span"},
            "statistics": self.statistics(),
            "next_step": self.next_step(),
            "per_span": self.per_span,
            "artifacts": [a.to_json() for a in self.artifacts],
            "provenance": self.provenance,
        }

    def report(self) -> str:
        """Return the round report, as text."""
        s = self.statistics()
        lines = [f"Audit round {self.plan_id}" + (f"  (seed {self.seed})" if self.seed else "")
                 + (f"   ** {TUNING_LABEL.upper()} **" if self.mode == "tuning" else ""),
                 f"  labelled        {s['minutes_labelled']:.1f} min over "
                 f"{len(self.per_span)} spans",
                 f"  artifacts       found {s['found']}, covered {s['covered']}, "
                 f"missed {s['missed']}   ({s['artifacts_per_minute']:.2f} per minute)"]
        if self.found:
            q = s["quoted_95"]
            lines += [
                f"  recall          {s['recall']:.3f}   95% interval {q[0]:.3f}-{q[1]:.3f} "
                f"(quoted: {s['quoted_interval'].replace('_', ' ')}, the wider)",
                f"                  Clopper-Pearson {s['clopper_pearson_95'][0]:.3f}-"
                f"{s['clopper_pearson_95'][1]:.3f}; span bootstrap "
                f"{s['span_bootstrap_95'][0]:.3f}-{s['span_bootstrap_95'][1]:.3f}",
                f"  this round      one-sided 95% lower bound "
                f"{s['gate_lower_bound_one_sided_95']:.3f} vs {GATE_RECALL} (description; "
                "the gate pools eligible rounds)",
                f"  projection      {s['artifacts_needed_at_zero_further_misses']} artifacts "
                f"needed at zero further misses: {s['more_artifacts_needed']} more, "
                f"~{s['more_rounds_projected']} more round(s), "
                f"~{s['more_minutes_projected']:.0f} min of labelling",
                f"  secondary       mean covered fraction "
                f"{s['secondary_mean_covered_fraction']:.2f}; recall at >=50% overlap "
                f"{s['secondary_recall_at_half_overlap']:.3f}",
            ]
        for m in self.misses:
            d = m.diagnosis
            head = (f"  MISS {m.miss_id}  {m.start_s:.2f}-{m.stop_s:.2f} s  "
                    f"covered {m.covered_fraction:.0%}  ->  "
                    f"{d.verdict if d else 'undiagnosed'}")
            lines.append(head)
            if d and d.max_z_by_band:
                lines.append(f"       max z {d.max_z:.2f} in {d.max_band} on {d.max_signal} "
                             f"(z_exit {d.z_exit:g}, z_enter {d.z_enter:g}); per band: "
                             + ", ".join(f"{b} {z:.1f}" for b, (z, _) in
                                         sorted(d.max_z_by_band.items())))
        if self.pooled is not None:
            pg = self.pooled
            bound = pg.get("lower_bound_one_sided_95")
            lines.append(
                f"  pooled gate     rounds {', '.join(pg['pooled_rounds']) or 'none'}: "
                + (f"{pg['covered']}/{pg['found']}, lower bound {bound:.3f} -> "
                   f"{'CLEARED' if pg['gate_cleared'] else 'not cleared'}"
                   if bound is not None else "no eligible artifacts yet"))
            for ex in pg["excluded_rounds"]:
                lines.append(f"                  left out {ex['plan_id']}: {ex['reason']}")
        for w in self.warnings:
            lines.append(f"  WARNING         {w}")
        lines.append(f"  NEXT            {self.next_step()}")
        return "\n".join(lines)


def score_round(
    plan_id: str, spans: Sequence[SpanInput], *, seed: int | None = None,
    z_enter: float = Z_ENTER, z_exit: float = Z_EXIT, rounds_scored: int = 1,
    provenance: dict[str, Any] | None = None,
) -> RoundScore:
    """Score one round of committed blind spans."""
    arts: list[ArtifactScore] = []
    per_span: list[dict[str, Any]] = []
    for sp in spans:
        a = _intervals(sp.artifacts)
        c = _intervals(sp.candidates)
        cov = 0
        for i, (a0, a1) in enumerate(a):
            hit = is_covered((a0, a1), c)
            cov += hit
            arts.append(ArtifactScore(
                sp.span_id, i, float(a0), float(a1), hit, covered_fraction((a0, a1), c),
                None if hit else diagnose_miss((a0, a1), sp.traces,
                                               z_enter=z_enter, z_exit=z_exit)))
        per_span.append({"span_id": sp.span_id, "recording_id": sp.recording_id,
                         "animal": sp.animal, "condition": sp.condition,
                         "found": len(a), "covered": cov, "minutes": sp.minutes,
                         "candidates": len(c), "traces": sp.traces is not None})
    return RoundScore(plan_id, seed, arts, per_span, sum(sp.minutes for sp in spans),
                      rounds_scored, dict(provenance or {}))


def next_round_gate(
    score: dict[str, Any] | None, resolutions: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Whether another round may be drawn, from a round's score and its resolutions.

    Refused when the round is unscored, when any miss is undiagnosed, and - the
    spec's rule, stricter than "diagnosed" - when any miss has no recorded task 07
    fix in ``resolutions`` (``{miss_id: {"task07_fix": "..."}}``). Labelling more
    against a generator already known to miss wastes the labeller's time.
    """
    if score is None:
        return False, "the last round has not been scored"
    misses = [a for a in score.get("artifacts", []) if not a.get("covered")]
    undiagnosed = [a["id"] for a in misses
                   if a.get("diagnosis", {}).get("verdict", "undiagnosed") == "undiagnosed"]
    if undiagnosed:
        return False, f"undiagnosed miss(es): {', '.join(undiagnosed)}"
    res = resolutions or {}
    unfixed = [a["id"] for a in misses
               if not str(res.get(a["id"], {}).get("task07_fix", "")).strip()]
    if unfixed:
        return False, (f"miss(es) with no recorded task 07 fix: {', '.join(unfixed)} - "
                       "fix the generator before labelling more")
    untimed = [a["id"] for a in misses if _parse_time(res[a["id"]].get("fixed_at")) is None]
    if untimed:
        return False, (f"fix(es) with no fixed_at time: {', '.join(untimed)} - the gate "
                       "needs it to tell which rounds were labelled after the fix")
    if misses:
        return True, "every miss diagnosed and fixed in task 07: draw the next round"
    return True, "zero misses: draw the next round with a fresh recorded seed"


# ---------------------------------------------------------------------------
# the store: a committed round in, a score out
# ---------------------------------------------------------------------------

RevealFn = Callable[[dict[str, Any]], tuple[npt.ArrayLike, Sequence[BandZ] | None]]
"""``span record -> (candidate intervals, band traces)``.

The runner supplies the SAME function the audit window revealed with, so the
candidates scored are the candidates shown. The window keeps its reveal in memory
only, so for a round already committed they are recomputed - deterministically,
from the same code and data - rather than read back.
"""


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _parse_time(value: object) -> datetime | None:
    """Return an aware datetime from an ISO string, or ``None`` (absent/null/malformed)."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        t = datetime.fromisoformat(value)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def span_id(plan: dict[str, Any], k: int) -> str:
    """Return a plan's k-th span id, ``<plan_id>_s<k+1>`` - the audit window uses this."""
    return f"{plan['plan_id']}_s{k + 1}"


def load_round(store: GemsStore, plan_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(plan, spans)`` for a round whose every span is committed.

    Each span dict carries the plan's span fields plus ``span_id``, ``marks``
    (``(n, 2)`` seconds) and ``record`` (the window's per-span record, whose
    ``assessable_regions_s`` fixes the region the reveal used).

    Raises
    ------
    FileNotFoundError
        If the plan is missing, or any span has no committed marks - a round is
        scored whole or not at all.
    """
    plan = _read_json(store.audit_plan_path(plan_id))
    if plan is None:
        msg = f"no audit plan {plan_id!r} in the store"
        raise FileNotFoundError(msg)
    spans = []
    for k, sp in enumerate(plan["spans"]):
        sid = span_id(plan, k)
        folder = store.audit_dir(sp["animal"], sp["recording_id"])
        marks = _read_json(folder / f"{sid}_blind_marks.json")
        if marks is None:
            msg = f"span {k + 1} of {plan_id} ({sid}) has no committed marks yet"
            raise FileNotFoundError(msg)
        record = _read_json(folder / f"{sid}_plan.json") or {}
        spans.append({**sp, "span_id": sid, "record": record,
                      "committed_at": marks.get("committed_at"),
                      "marks": np.asarray([[m["start_s"], m["stop_s"]] for m in marks["marks"]],
                                          dtype=np.float64).reshape(-1, 2)})
    return plan, spans


def _labelled_at(plan: dict[str, Any], spans: list[dict[str, Any]]) -> datetime | None:
    """When a round was labelled: its FIRST committed span (plan creation as fallback)."""
    times = [t for t in (_parse_time(sp.get("committed_at")) for sp in spans) if t]
    return min(times) if times else _parse_time(plan.get("created_at"))


def score_stored_round(
    store: GemsStore, plan_id: str, reveal: RevealFn, *,
    mode: Literal["gate", "tuning"] = "gate",
    provenance: dict[str, Any] | None = None, write: bool = True,
) -> RoundScore:
    """Score a committed round from the store and write the score.

    ``gate`` (the default) is the round's evidence: every span with a reveal digest
    must reproduce it exactly or :class:`RevealMismatchError` is raised; a span with no
    digest is scored with a warning. It is written to ``audit_score_path`` - and
    refuses to replace a gate score made under a different generator, since that
    would swap gate evidence for a re-score. ``tuning`` scores the round with the
    CURRENT generator, whatever it revealed: labelled :data:`TUNING_LABEL`, written
    to its own file, never pooled.
    """
    plan, spans = load_round(store, plan_id)
    generator = generator_provenance()
    inputs, warnings = [], []
    digest_state: dict[str, str] = {}
    for sp in spans:
        candidates, traces = reveal(sp)
        cand = _intervals(candidates)
        shown = (sp["record"].get("reveal") or {}).get("candidates_sha256")
        now = candidate_digest(cand)
        sid = sp["span_id"]
        if shown is None:
            digest_state[sid] = "absent"
            if mode == "gate":
                warnings.append(f"{sid}: revealed by an app without candidate digests - the "
                                "recomputed candidates are assumed, NOT verified, to equal "
                                "what was shown")
        elif shown == now:
            digest_state[sid] = "verified"
        else:
            digest_state[sid] = "differs"
            if mode == "gate":
                n_shown = sp["record"]["reveal"].get("n_candidates", "?")
                msg = (f"{sid}: the recomputed candidates ({len(cand)}) do not match the "
                       f"{n_shown} revealed - the generator or its input changed since the "
                       "reveal. Refusing to score it as gate evidence; a re-score with the "
                       "current generator is a tuning check (mode='tuning').")
                raise RevealMismatchError(msg)
        inputs.append(SpanInput(sid, sp["recording_id"], sp["animal"], sp["condition"],
                                float(sp["start_s"]), float(sp["stop_s"]), sp["marks"],
                                cand, traces))
    labelled = _labelled_at(plan, spans)
    score = score_round(plan_id, inputs, seed=plan.get("seed"), provenance={
        "plan_created_at": plan.get("created_at"),
        **({"labelled_at": labelled.isoformat()} if labelled else {}),
        "scored_at": datetime.now(UTC).isoformat(),
        "generator": generator,
        "reveal_digests": digest_state,
        **(provenance or {})})
    score = replace(score, mode=mode, warnings=warnings)
    if not write:
        return score
    if mode == "tuning":
        path = store.audit_tuning_path(plan_id, datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(score.to_json(), indent=1, sort_keys=True) + "\n")
        return score
    path = store.audit_score_path(plan_id)
    old = _read_json(path)
    old_src = ((old or {}).get("provenance", {}).get("generator", {}) or {}).get("source_sha256")
    if old_src is not None and old_src != generator["source_sha256"]:
        msg = (f"{plan_id} already has a gate score made under a different generator "
               f"({old_src[:12]} vs {generator['source_sha256'][:12]}); replacing it would "
               "swap gate evidence for a re-score. Use mode='tuning'.")
        raise RevealMismatchError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(score.to_json(), indent=1, sort_keys=True) + "\n")
    score = replace(score, pooled=pooled_gate(store))
    atomic_write_text(path, json.dumps(score.to_json(), indent=1, sort_keys=True) + "\n")
    return score


def last_fix_at(store: GemsStore) -> datetime | None:
    """Return the latest recorded task 07 fix time, over every round's resolutions."""
    folder = store.audit_resolution_path("x").parent
    times = []
    for f in sorted(folder.glob("*_resolutions.json")) if folder.is_dir() else []:
        for entry in json.loads(f.read_text(encoding="utf-8")).values():
            t = _parse_time(entry.get("fixed_at")) if isinstance(entry, dict) else None
            if t:
                times.append(t)
    return max(times) if times else None


def pooled_gate(store: GemsStore) -> dict[str, Any]:
    """Pool the gate-mode rounds labelled AFTER the last fix into the cumulative gate.

    Returns which rounds were pooled, which were left out and why, and the pooled
    one-sided 95% lower bound - the more conservative of Clopper-Pearson and the
    span bootstrap over every pooled span - against :data:`GATE_RECALL`.
    """
    fix = last_fix_at(store)
    folder = store.audit_score_path("x").parent
    pooled, excluded, per_span = [], [], []
    for f in sorted(folder.glob("plan_*_score.json")) if folder.is_dir() else []:
        doc = json.loads(f.read_text(encoding="utf-8"))
        pid = doc["plan_id"]
        labelled = _parse_time(doc.get("provenance", {}).get("labelled_at"))
        if doc.get("mode", "gate") != "gate":
            excluded.append({"plan_id": pid, "reason": "not a gate-mode score"})
        elif fix is not None and labelled is None:
            excluded.append({"plan_id": pid, "reason": "labelling time unknown"})
        elif fix is not None and labelled is not None and labelled <= fix:
            excluded.append({"plan_id": pid, "reason": f"labelled {labelled.isoformat()}, "
                             f"before the last fix at {fix.isoformat()} - tuning data now"})
        else:
            pooled.append(pid)
            per_span += [(s["covered"], s["found"]) for s in doc.get("per_span", [])]
    out: dict[str, Any] = {"last_fix_at": fix.isoformat() if fix else None,
                           "pooled_rounds": pooled, "excluded_rounds": excluded,
                           "gate_recall": GATE_RECALL, "gate_cleared": False}
    k, n = sum(c for c, _ in per_span), sum(f for _, f in per_span)
    out.update(found=n, covered=k)
    if n == 0:
        out["lower_bound_one_sided_95"] = None
        return out
    cp1 = one_sided_lower(k, n)
    b1 = span_bootstrap(per_span)[2]
    lower = min(cp1, b1)
    need = artifacts_needed(n - k)
    out.update(lower_bound_one_sided_95=lower,
               lower_bound_parts={"clopper_pearson": cp1, "span_bootstrap": b1},
               gate_cleared=lower >= GATE_RECALL,
               artifacts_needed_at_zero_further_misses=need,
               more_artifacts_needed=max(need - n, 0))
    return out


# ---------------------------------------------------------------------------
# the candidate budget (task 09, declared before any marks; enforced 2026-09-28)
# ---------------------------------------------------------------------------

CANDIDATE_BUDGET: Final = 3000
"""Candidates per recording the pinned generator may produce (task 09: class balance
of >= 5% true positives at ~150 real artifacts per recording). Round 1 showed why
it matters: at 93-96% of a span covered, recall is met by coverage alone."""

BUDGET_QUANTILE: Final = 0.9
"""The pool statistic: the generator is within budget when this quantile of the
measured candidates-per-recording is <= :data:`CANDIDATE_BUDGET` - at most 10% of
recordings over. The declared rule is per recording; which pool statistic decides
is this module's choice, reported for ratification (2026-09-28)."""

BUDGET_MIN_RECORDINGS: Final = 30
"""A budget is measured on many recordings, not a handful of spans."""


def budget_key(generator_sha: str, reveal_sha: str | None) -> str:
    """Key a budget measurement to the exact generator it measured."""
    return f"{generator_sha[:16]}_{(reveal_sha or 'noreveal')[:16]}"


def budget_record(rows: Sequence[dict[str, Any]], *, reveal_sha: str | None,
                  sample_rule: str) -> dict[str, Any]:
    """Summarise per-recording measurements into a budget record for this generator.

    Each row carries ``candidates`` (per recording, over its assessable regions),
    ``assessable_s`` and ``covered_s``, plus ``animal`` and ``condition``.
    """
    generator = generator_provenance()
    counts = np.asarray([r["candidates"] for r in rows], dtype=float)
    frac = np.asarray([r["covered_s"] / r["assessable_s"] for r in rows if r["assessable_s"] > 0])
    q = float(np.quantile(counts, BUDGET_QUANTILE)) if counts.size else float("nan")
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(f"{r['animal']}|{r['condition']}", []).append(r)
    cells = {k: {"n": len(v),
                 "median_candidates": float(np.median([x["candidates"] for x in v])),
                 "max_candidates": float(max(x["candidates"] for x in v)),
                 "median_time_covered": float(np.median(
                     [x["covered_s"] / x["assessable_s"] for x in v if x["assessable_s"] > 0]))}
             for k, v in sorted(by_cell.items())}
    return {
        "key": budget_key(generator["source_sha256"], reveal_sha),
        "generator": generator, "reveal_source_sha256": reveal_sha,
        "measured_at": datetime.now(UTC).isoformat(), "sample_rule": sample_rule,
        "budget": CANDIDATE_BUDGET, "quantile": BUDGET_QUANTILE,
        "min_recordings": BUDGET_MIN_RECORDINGS, "n_recordings": len(rows),
        "candidates_quantile": q,
        "candidates_median": float(np.median(counts)) if counts.size else float("nan"),
        "fraction_over_budget": float(np.mean(counts > CANDIDATE_BUDGET)) if counts.size else 1.0,
        "time_covered_median": float(np.median(frac)) if frac.size else float("nan"),
        "within_budget": bool(len(rows) >= BUDGET_MIN_RECORDINGS and q <= CANDIDATE_BUDGET),
        "by_animal_condition": cells, "recordings": list(rows),
    }


def write_budget(store: GemsStore, record: dict[str, Any]) -> Path:
    """Write a budget record at its generator's key."""
    path = store.audit_budget_path(record["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


def budget_status(store: GemsStore, reveal_sha: str | None = None) -> tuple[bool, str]:
    """Return whether the CURRENT generator is measured and within the candidate budget."""
    key = budget_key(generator_provenance()["source_sha256"], reveal_sha)
    rec = _read_json(store.audit_budget_path(key))
    if rec is None:
        return False, (f"the candidate budget has not been measured for the current "
                       f"generator ({key}); measure it on the eligible pool first")
    if rec["n_recordings"] < BUDGET_MIN_RECORDINGS:
        return False, (f"the budget measurement covers {rec['n_recordings']} recordings; "
                       f"at least {BUDGET_MIN_RECORDINGS} are needed")
    if not rec["within_budget"]:
        return False, (f"the generator exceeds the candidate budget: the "
                       f"{rec['quantile']:.0%} quantile is {rec['candidates_quantile']:.0f} "
                       f"candidates per recording against {CANDIDATE_BUDGET} "
                       f"({rec['fraction_over_budget']:.0%} of {rec['n_recordings']} over) - "
                       "a round now would clear recall by coverage alone")
    return True, (f"within budget: {rec['quantile']:.0%} quantile "
                  f"{rec['candidates_quantile']:.0f} <= {CANDIDATE_BUDGET}")


def check_next_round(store: GemsStore, *, reveal_sha: str | None = None) -> tuple[bool, str]:
    """May another audit round be drawn now? ``(allowed, reason)``.

    Every round, the first included, needs the current generator measured and
    within :data:`CANDIDATE_BUDGET` (:func:`budget_status`). After the first, the
    latest round must also be complete and scored, and :func:`next_round_gate` must
    pass on its score and the human-recorded resolutions.
    """
    folder = store.audit_plan_path("x").parent
    plans = sorted(folder.glob("plan_*.json")) if folder.is_dir() else []
    if plans:
        plan = json.loads(plans[-1].read_text(encoding="utf-8"))
        try:
            load_round(store, plan["plan_id"])
        except FileNotFoundError as exc:
            return False, f"round {plan['plan_id']} is still open: {exc}"
        ok, why = next_round_gate(_read_json(store.audit_score_path(plan["plan_id"])),
                                  _read_json(store.audit_resolution_path(plan["plan_id"])))
        if not ok:
            return ok, why
    else:
        why = "no earlier round: draw round 1"
    within, budget_why = budget_status(store, reveal_sha)
    if not within:
        return False, budget_why
    return True, f"{why}; {budget_why}"
