"""Step 9: test whether WHERE an artifact lands, relative to physiology, sets its threshold.

The seed-depth diagnostic found the threshold moves 4-16x with placement, in
deterministic kinds as much as in tribo. If that spread is explained by the
artifact's position relative to the consumer's own physiological events, the
tolerance is a FUNCTION of phase and its sensitive end is the worst phase -
measurable with a few aimed placements rather than many random ones. This tests
that on data that already exists: no new evaluation.

**One covariate per cell, declared here before looking** - a covariate chosen after
seeing four would be a garden of forking paths:

=============  ===============================================================
slow_wave      phase of the artifact MIDPOINT in the slow-wave cycle (stomach
               channel 1 peaks; circular)
hrv            phase of the artifact ONSET in the R-R cycle (circular)
T_hardware     number of baseline spikes inside the artifact window (both nerve
               channels; linear, the event's own unit)
mmc_burst      distance in seconds from the artifact window to the nearest burst
               peak on any stomach channel, 0 when one falls inside (linear)
=============  ===============================================================

Statistics: circular-linear correlation (Mardia) for phases, Spearman for the
linear covariates, each against a 10,000-shuffle permutation null with a fixed
seed. The threshold enters as its rank, so a placement censored below the grid
ranks lowest and one censored above ranks highest - no value is invented for it.

**The bunching check comes first.** Bisection on a discrete grid returns grid
values only, so "8 of 14 at 2.0 sigma" means 8 of 14 fell in the interval ending
there. A normal fitted to log2(threshold) predicts the count per sqrt(2) bin; a
parametric bootstrap asks how often a smooth distribution puts as many in its
fullest bin.

Usage::

    python tolerance_phase.py <scratch>
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from tolerance_analyze import _seed_interval
from tolerance_generate import KIND_DIRECTION
from tolerance_seed_depth import CELLS, SEEDS, observations

N_PERM: Final = 10_000
MIN_FOR_FIT: Final = 2
"""A normal needs at least three resolved values to fit a spread to."""
PERM_SEED: Final = 20260926
F64 = npt.NDArray[np.float64]


def circular_linear_r(x: F64, phase: F64) -> float:
    """Mardia's circular-linear correlation between linear ``x`` and ``phase`` (cycles).

    0 when x is unrelated to phase, 1 when x is an exact sinusoid of it. Invariant
    to where the cycle starts, which a linear correlation on phase is not.
    """
    c, s = np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)
    rxc = np.corrcoef(x, c)[0, 1]
    rxs = np.corrcoef(x, s)[0, 1]
    rcs = np.corrcoef(c, s)[0, 1]
    r2 = (rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2)
    return float(math.sqrt(max(r2, 0.0)))


def permutation_p(
    stat: float, x: F64, cov: F64, fn: Callable[[F64, F64], float], rng: np.random.Generator
) -> float:
    """One-sided p: how often a shuffled ``x`` gives a statistic at least as large."""
    hits = sum(fn(rng.permutation(x), cov) >= stat for _ in range(N_PERM))
    return (hits + 1) / (N_PERM + 1)


def _abs_spearman(x: F64, cov: F64) -> float:
    return float(abs(stats.spearmanr(x, cov).statistic))


def bunching(resolved: list[float], rng: np.random.Generator) -> dict[str, float]:
    """Test whether the fullest grid bin is fuller than a smooth distribution makes it."""
    v = np.log2(np.asarray(resolved)) - 0.25  # bin midpoint: (x/sqrt2, x] in log2
    mu, sd = float(v.mean()), float(v.std(ddof=1)) or 0.25
    counts = np.unique(np.round(np.log2(resolved) * 2), return_counts=True)[1]
    observed = int(counts.max())
    sims = rng.normal(mu, sd, size=(N_PERM, len(v)))
    sim_max = np.array([np.unique(np.ceil(row * 2), return_counts=True)[1].max() for row in sims])
    return {"n": len(v), "fullest_bin": observed,
            "expected_fullest_bin_median": float(np.median(sim_max)),
            "p_as_full_or_fuller": float((np.sum(sim_max >= observed) + 1) / (N_PERM + 1))}


def _phase(t: float, events: F64) -> float:
    i = int(np.searchsorted(events, t))
    if i == 0 or i >= len(events):
        return math.nan
    return (t - events[i - 1]) / (events[i] - events[i - 1])


def covariate(consumer: str, t0: float, dur: float, fid: dict) -> float:  # type: ignore[type-arg]
    """Return the one pre-declared covariate for this consumer's cell (module doc)."""
    if consumer == "slow_wave":
        return _phase(t0 + dur / 2, np.sort(np.asarray(fid["slow_wave"][0], float)))
    if consumer == "hrv":
        return _phase(t0, np.sort(np.asarray(fid["hrv"], float).ravel()))
    if consumer == "T_hardware":
        spikes = np.concatenate([np.asarray(c, float) for c in fid["T_hardware"]])
        return float(np.sum((spikes >= t0) & (spikes < t0 + dur)))
    bursts = np.sort(np.concatenate([np.asarray(c, float) for c in fid["mmc_burst"]]))
    inside = np.any((bursts >= t0) & (bursts < t0 + dur))
    return 0.0 if inside else float(np.min(np.minimum(np.abs(bursts - t0),
                                                      np.abs(bursts - (t0 + dur)))))


def _placements(scratch: Path) -> dict[tuple, dict[int, float]]:  # type: ignore[type-arg]
    """(cell) -> seed -> t0 in seconds, from every manifest that placed it."""
    t0: dict[tuple, dict[int, float]] = {c: {} for c in CELLS}  # type: ignore[type-arg]
    want = {(h, k, round(d, 6), cs): (h, k, d, cs, co) for h, k, d, cs, co in CELLS}
    rounds = sorted(scratch.glob("manifest_seeddepth_r*.json"))
    for f in [scratch / "manifest_replicate.json", *rounds]:
        for p in json.loads(f.read_text(encoding="utf-8"))["points"]:
            cell = want.get((p["host_tag"], p["kind"], round(float(p["dur_s"]), 6), p["chan_set"]))
            if cell is not None:
                t0[cell][int(p["seed"])] = float(p["t0_s"])
    return t0


def main(scratch: Path) -> None:
    """Run the bunching check and the phase test per cell; write phase_test.json."""
    rng = np.random.default_rng(PERM_SEED)
    fid = json.loads((scratch / "baseline_host1_JEL_fid.json").read_text(encoding="utf-8"))
    obs = observations(scratch)
    t0s = _placements(scratch)
    report = []
    for cell in CELLS:
        host, kind, dur, cs, cons = cell
        direction = KIND_DIRECTION[kind]
        ranks_in, covs, resolved = [], [], []
        for seed in SEEDS:
            past = obs[cell].get(seed, {})
            curve = sorted((a, int(p) if direction == "up" else int(not p))
                           for a, p in past.items())
            lo, hi = _seed_interval(curve, direction)
            value = -math.inf if lo == 0.0 else (math.inf if math.isinf(hi) else math.log2(hi))
            if math.isfinite(value):
                resolved.append(hi)
            cv = covariate(cons, t0s[cell][seed], dur, fid)
            if math.isnan(cv):
                continue
            ranks_in.append(value)
            covs.append(cv)
        x = stats.rankdata(np.asarray(ranks_in))
        cov = np.asarray(covs)
        circular = cons in ("slow_wave", "hrv")
        fn = circular_linear_r if circular else _abs_spearman
        stat = fn(x, cov)
        row = {
            "cell": f"{host}|{kind}|{dur:g}|{cs}", "consumer": cons,
            "covariate": {"slow_wave": "slow-wave phase of artifact midpoint",
                          "hrv": "R-R phase of artifact onset",
                          "T_hardware": "baseline spikes inside the window",
                          "mmc_burst": "seconds to nearest burst"}[cons],
            "test": "circular-linear r" if circular else "|Spearman rho|",
            "n": len(x), "statistic": round(stat, 3),
            "permutation_p": round(permutation_p(stat, x, cov, fn, rng), 4),
            "bunching": bunching(resolved, rng) if len(resolved) > MIN_FOR_FIT else None,
            "pairs": [[round(float(c), 4), float(v)] for c, v in zip(cov, ranks_in, strict=True)],
        }
        report.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "pairs"}))
    (scratch / "phase_test.json").write_text(json.dumps(report, indent=1), encoding="utf-8",
                                             newline="\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
