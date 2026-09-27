"""Step 9 seed-depth diagnostic: measure how wide the placement distribution is.

The replicate pass censored 71-75% of seed-rows in every kind, deterministic ones
included, with 62 of 155 rows censored on BOTH sides of a 3-point bracket. That
says where the artifact lands moves the tolerance by more than the 2x a bracket
spans - so widening the bracket chases the tails of a distribution whose width is
the result. This measures the width directly, on a few cells, before anything is
spent on the extension.

**Bisection, not the full grid.** 20 seeds across the whole grid is 23-28 points
per seed; for the ``slow_wave`` cell alone that is ~12.6 worker-hours. The curves
were measured monotone (1 of 220 ``slow_wave`` seed-curves, 0 of 260 for
``T_hardware``/``hrv``), so each seed's crossing is located on the existing grid
by bisection, then CONFIRMED two sqrt(2) steps either side. A confirmation that
disagrees marks the seed ``non_monotone`` rather than trusting the bisection.

**Seeds 1-5 are reused, not re-run** (invariant 26): the replicate pass already
ran them at the same placements. That reuse is only valid if the placements are
the same, so it is asserted - every reused seed's ``t0`` must equal the replicate
manifest's.

Stateless: each round recomputes what is known from every result file on disk,
so an interrupted round is simply re-planned.

Usage::

    python tolerance_seed_depth.py <scratch> next     # plan one round, or say done
    python tolerance_seed_depth.py <scratch> report   # the distribution per cell
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).parent))
from tolerance_analyze import bracket_key
from tolerance_generate import (
    CHANNEL_SETS,
    KIND_DIRECTION,
    OBSERVED_MAX_UV,
    Point,
    _amps_for,
    _inject_window,
    _point_cost_s,
    _seed_for,
    _t0_for,
)

CELLS: Final[tuple[tuple[str, str, float, str, str], ...]] = (
    ("host1_JEL", "step", 0.5, "common_mode", "slow_wave"),
    ("host1_JEL", "step", 0.05, "nerve", "T_hardware"),
    ("host1_JEL", "clip", 0.5, "stomach", "mmc_burst"),
    ("host1_JEL", "tribo", 0.5, "nerve", "hrv"),
)
"""(host, kind, duration s, channel set, consumer). Two deterministic kinds, one
with the inverted axis, and ``tribo``; four consumers; one host, so the placement
distributions are comparable. Each cell was censored on both sides in the
replicate pass - the widest rows, chosen because width is the question."""

SEEDS: Final = tuple(range(1, 21))
REUSED_SEEDS: Final = frozenset(range(1, 6))
CONFIRM_STEPS: Final = 2
IDX_BASE: Final = 200_000
"""Keeps sweep point ids, and the per-point run directories named from them,
clear of the locate and replicate passes."""

PREFIX: Final = "manifest_seeddepth_r"
T0_MATCH_S: Final = 1e-9
"""Reused placements must agree to float precision, not approximately."""


def _amp_key(a: float) -> float:
    """Grid amplitudes are written rounded to 6 dp on both sides of the boundary."""
    return round(float(a), 6)


def grid_for(host: str, chan_set: str) -> list[float]:
    """Return the locate grid for this host and channel set - the existing grid."""
    key = bracket_key(host, "step", 0.5, chan_set)
    return [_amp_key(a) for a in _amps_for(host, chan_set, False, {}, key)]


def next_step(
    grid: list[float], obs: dict[float, bool]
) -> tuple[str, list[float]]:
    """Classify one seed from what is known, and name the amplitudes to run next.

    ``obs`` maps amplitude -> "past threshold": changed for ``up`` kinds,
    UNchanged for ``down`` (clip), so the tolerance is always the lowest grid
    point that is past threshold, as in ``tolerance_analyze._crossing``.

    Returns ``(status, amps)``; ``amps`` is empty unless status is ``pending``.
    Terminal statuses: ``resolved``, ``censored_below_grid`` (past threshold at
    the lowest grid point), ``censored_above_grid`` (never past threshold),
    ``non_monotone``.
    """
    at = {grid.index(a): p for a, p in obs.items() if a in grid}
    lo = max((i for i, p in at.items() if not p), default=-1)
    hi = min((i for i, p in at.items() if p), default=len(grid))
    if lo > hi:
        return "non_monotone", []
    if hi - lo > 1:
        return "pending", [grid[(lo + hi) // 2]]
    confirm = [
        i for i in (lo - CONFIRM_STEPS, hi + CONFIRM_STEPS)
        if 0 <= i < len(grid) and i not in at
    ]
    if confirm:
        return "pending", [grid[i] for i in confirm]
    if hi == len(grid):
        return "censored_above_grid", []
    if lo == -1:
        return "censored_below_grid", []
    return "resolved", []


def crossing(grid: list[float], obs: dict[float, bool]) -> float | None:
    """Return the resolved tolerance, or ``None`` if the seed is not ``resolved``."""
    status, _ = next_step(grid, obs)
    if status != "resolved":
        return None
    return min(a for a, p in obs.items() if p and a in grid)


def _result_files(scratch: Path) -> list[Path]:
    rounds = sorted(scratch.glob(f"sweep_{PREFIX}*.json"))
    return [scratch / "sweep_manifest_replicate.json", *rounds]


def observations(scratch: Path) -> dict[tuple, dict[int, dict[float, bool]]]:
    """(cell) -> seed -> amplitude -> past-threshold, from every result on disk.

    Cells are matched through ``bracket_key`` - the one canonical form - never by
    comparing a duration float formatted on the MATLAB side (invariant 22).
    """
    want = {bracket_key(h, k, d, c): (h, k, d, c, cons) for h, k, d, c, cons in CELLS}
    out: dict[tuple, dict[int, dict[float, bool]]] = {cell: {} for cell in CELLS}
    for f in _result_files(scratch):
        if not f.exists():
            continue
        rows = json.loads(f.read_text(encoding="utf-8"))
        for r in rows if isinstance(rows, list) else [rows]:
            if not r.get("ok"):
                continue
            cell = want.get(bracket_key(r["host_tag"], r["kind"], r["dur_s"], r["chan_set"]))
            if cell is None or int(r["seed"]) not in SEEDS:
                continue
            v = r.get(f"{cell[4]}_changed")
            if v is None:
                continue
            past = int(v) > 0 if KIND_DIRECTION[cell[1]] == "up" else int(v) == 0
            out[cell].setdefault(int(r["seed"]), {})[_amp_key(r["amp_sigma"])] = past
    return out


def _check_reuse(scratch: Path, span_s: float) -> None:
    """Reused seeds must sit where the replicate pass put them, or reuse is void."""
    man = json.loads((scratch / "manifest_replicate.json").read_text(encoding="utf-8"))
    want = {bracket_key(h, k, d, c): (h, k, d, c) for h, k, d, c, _ in CELLS}
    checked = 0
    for p in man["points"]:
        cell = want.get(bracket_key(p["host_tag"], p["kind"], p["dur_s"], p["chan_set"]))
        if cell is None:
            continue
        h, k, d, c = cell
        rng = np.random.default_rng(_seed_for(h, k, d, c, int(p["seed"])))
        t0 = _t0_for(int(p["seed"]), d, span_s, rng)
        if abs(t0 - float(p["t0_s"])) > T0_MATCH_S:
            msg = (f"{bracket_key(h, k, d, c)} seed {p['seed']}: t0 {t0} here vs "
                   f"{p['t0_s']} in the replicate manifest - reusing its results "
                   "would mix two placements under one seed")
            raise SystemExit(msg)
        checked += 1
    print(f"reuse check: {checked} replicate points sit at the recomputed placements")


def plan_next(scratch: Path) -> None:
    """Write the next round's manifest and injected windows, or report done."""
    obs = observations(scratch)
    todo: list[tuple[tuple, int, float]] = []
    for cell in CELLS:
        grid = grid_for(cell[0], cell[3])
        for seed in SEEDS:
            status, amps = next_step(grid, obs[cell].get(seed, {}))
            if status == "pending":
                todo += [(cell, seed, a) for a in amps]
    if not todo:
        print("DONE - every seed is terminal; run `report`")
        return

    prepare = json.loads((scratch / "prepare.json").read_text(encoding="utf-8"))
    hosts = prepare["hosts"] if isinstance(prepare["hosts"], list) else [prepare["hosts"]]
    host = next(h for h in hosts if h["tag"] == CELLS[0][0])
    assert all(c[0] == host["tag"] for c in CELLS), "one host per diagnostic"
    fs = float(host["fs"])
    # MATLAB 1-based inclusive -> 0-based half-open
    i0, i1 = int(host["startSample"]) - 1, int(host["stopSample"])
    sigma_v = np.atleast_1d(np.asarray(host["sigmaUV_broadband"], np.float64)) * 1e-6
    y = np.asarray(loadmat(host["file"], variable_names=["yOut"])["yOut"], np.float64)
    span = np.ascontiguousarray(y[i0:i1, :])
    del y
    span_s = span.shape[0] / fs
    _check_reuse(scratch, span_s)

    k = len(list(scratch.glob(f"{PREFIX}*.json"))) + 1
    outdir = scratch / "injected"
    points: list[Point] = []
    for n, ((h, kind, dur, cs, cons), seed, amp) in enumerate(todo):
        idx = IDX_BASE + 1000 * k + n
        cols = CHANNEL_SETS[cs]
        rng = np.random.default_rng(_seed_for(h, kind, dur, cs, seed))
        t0 = _t0_for(seed, dur, span_s, rng)
        win, s0, ns = _inject_window(span, fs, cols, t0, dur, kind, amp, seed)
        name = f"seeddepth_{idx:06d}.mat"
        savemat(outdir / name, {"win": win, "start0": s0, "cols1": np.asarray(cols) + 1},
                do_compression=False)
        amp_uv = [float(amp * sigma_v[c] * 1e6) for c in cols]
        points.append(Point(
            idx=idx, host_tag=h, consumers=[cons], est_cost_s=_point_cost_s(h, [cons]),
            kind=kind, dur_s=float(dur), amp_sigma=float(amp), chan_set=cs, seed=seed,
            pass_name=f"seed_depth_r{k}", t0_s=t0, start_sample0=s0, n_samples=ns,
            cols1=[c + 1 for c in cols], file=name, amp_uv_per_chan=amp_uv,
            beyond_observed=max(amp_uv) > OBSERVED_MAX_UV,
        ))
    points.sort(key=lambda q: q.est_cost_s, reverse=True)
    path = scratch / f"{PREFIX}{k}.json"
    path.write_text(json.dumps({
        "kind_direction": dict(KIND_DIRECTION), "pass": f"seed_depth_r{k}",
        "cells": [list(c) for c in CELLS], "seeds": list(SEEDS),
        "n_points_written": len(points),
        "est_worker_s": round(sum(q.est_cost_s for q in points), 1),
        "points": [asdict(q) for q in points],
    }, indent=1), encoding="utf-8", newline="\n")
    print(f"round {k}: {len(points)} points, est {sum(q.est_cost_s for q in points):.0f} "
          f"worker-s -> {path.name}")


def report(scratch: Path) -> None:
    """Per cell: terminal statuses and the distribution of resolved tolerances."""
    obs = observations(scratch)
    out = []
    for cell in CELLS:
        grid = grid_for(cell[0], cell[3])
        statuses: dict[str, int] = {}
        xs: list[float] = []
        for seed in SEEDS:
            o = obs[cell].get(seed, {})
            status, _ = next_step(grid, o)
            statuses[status] = statuses.get(status, 0) + 1
            x = crossing(grid, o)
            if x is not None:
                xs.append(x)
        row: dict[str, object] = {"cell": bracket_key(*cell[:4]), "consumer": cell[4],
                                  "direction": KIND_DIRECTION[cell[1]],
                                  "statuses": statuses, "n_resolved": len(xs)}
        if xs:
            a = np.asarray(sorted(xs))
            steps = np.log(a / a.min()) / np.log(np.sqrt(2))
            row |= {"min_sigma": float(a.min()), "p10_sigma": float(np.quantile(a, 0.1)),
                    "median_sigma": float(np.median(a)), "p90_sigma": float(np.quantile(a, 0.9)),
                    "max_sigma": float(a.max()), "max_over_min": float(a.max() / a.min()),
                    "spread_sqrt2_steps": float(steps.max()),
                    "resolved_sigma": [float(v) for v in a]}
        out.append(row)
        print(json.dumps(row))
    (scratch / "seed_depth_report.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    N_ARGV = 3
    modes = {"next": plan_next, "report": report}
    if len(sys.argv) != N_ARGV or sys.argv[2] not in modes:
        raise SystemExit("usage: tolerance_seed_depth.py <scratch> next|report")
    modes[sys.argv[2]](Path(sys.argv[1]))
