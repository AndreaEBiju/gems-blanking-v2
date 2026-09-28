"""Step 9 stage 4: find the crossings, then assemble ``consumer_tolerances.json``.

Two modes::

    python tolerance_analyze.py <scratch> crossings   # after the locate sweep
    python tolerance_analyze.py <scratch> final       # after the replicate sweep

**The amplitude axis does not point the same way for every kind.** ``step``,
``drift`` and ``tribo`` scale an additive artifact, so damage increases with
amplitude and the tolerance is the lowest amplitude that changes the output.
``clip``'s ``amp_ratio`` IS the rail: a lower rail clips harder, damage
*decreases* with amplitude, and a rail above the host's peak does nothing at
all. A crossing finder scanning upward returns the bottom of the grid for every
clip row and looks entirely reasonable doing it. The direction is read from the
manifest's ``kind_direction``, never assumed, and the sign convention is stated
in the output.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

CONSUMERS: Final[tuple[str, ...]] = (
    "T_hardware", "mmc", "mmc_burst", "slow_wave", "breathing", "hrv",
)

BREATHING_SHIFT_S: Final = 0.07
"""A breath displaced by more than this counts as changed, seconds.

Breaths are sampled at heartbeat locations, so detected breath times are
quantised to R-R (~140-165 ms); half an R-R interval separates "same beat" from
"moved a beat".
"""

BREATHING_CRITERION: Final[dict[str, str]] = {
    "criterion": f"changed = added + lost + displaced by more than {BREATHING_SHIFT_S} s",
    "was": "count only - added or lost (A.4, pre-registered)",
    "changed_on": "2026-09-26",
    "decided_by": "Andrea",
    "reason": (
        "Breath TIMING is used downstream, not only rate, so a breath moved by one "
        "R-R interval (~140 ms) is damage. A change to a pre-registered criterion, "
        "made on Andrea's statement of what the output is used for."
    ),
    "how_applied": (
        "Offline, from the breathing_maxShift every sweep point already records (a "
        "point changes iff a breath was added or lost OR its largest paired "
        "displacement exceeds the threshold) - no sweep re-run. tolerance_criteria.m "
        "now computes it natively, so the two agree on new points."
    ),
}


def apply_breathing_displacement(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the 2026-09-26 breathing criterion to one sweep row, in place.

    A row whose count-only ``breathing_changed`` is 0 but whose
    ``breathing_maxShift`` exceeds :data:`BREATHING_SHIFT_S` becomes changed.
    The count-only value is kept as ``breathing_changed_count_only`` so the
    difference stays auditable. A breathing row without ``breathing_maxShift``
    cannot be judged under the new criterion and raises rather than being read
    as unchanged. Idempotent.
    """
    changed = row.get("breathing_changed")
    if changed is None or "breathing_changed_count_only" in row:
        return row
    if "breathing_maxShift" not in row:
        msg = (f"a breathing row has no breathing_maxShift, so the displacement "
               f"criterion cannot be applied to it: {_key_or_repr(row)}")
        raise KeyError(msg)
    row["breathing_changed_count_only"] = changed
    shift = row["breathing_maxShift"]
    if changed == 0 and isinstance(shift, (int, float)) and shift > BREATHING_SHIFT_S:
        row["breathing_changed"] = 1
    return row


def _key_or_repr(row: dict[str, Any]) -> str:
    try:
        return repr((row["host_tag"], row["kind"], row["dur_s"], row["chan_set"],
                     row.get("amp_sigma"), row.get("seed")))
    except KeyError:
        return repr(sorted(row))[:200]

DEGENERATE: Final[dict[str, str]] = {
    "mmc": (
        "NOT a tolerance. The exact set difference matches burst peaks at sample "
        "resolution (41 us) in a band whose impulse response is 486 ms - four "
        "orders of magnitude tighter than the signal supports - so it fires at "
        "the grid floor by construction. That is a property of the comparison, "
        "not of the consumer. mmc_burst carries the number task 14 routes on. "
        "The downward extension to 0.0884 sigma is retained because it confirms "
        "the flatness cheaply."
    ),
}
"""Rows emitted with a status and no value, the way task 02 handles unresolvable."""

REPLICATE_SCOPE: Final[dict[str, str]] = {
    "host1_JEL": "all",
    "host2_ORE_long": "slow_wave",
    "host2_ORE": "tribo",
}
"""Which keys get the 5-seed replicate pass, and why.

Measured cost forced this: the locate pass ran ~7 h, not the 2.5 h estimated, so
replicating every key on every host would have been ~7 h more rather than the
approved ~25%.

``host1_JEL`` is the primary and gets everything. ``host2_ORE_long`` gets its
``slow_wave`` keys, which is the only reason that span exists. ``host2_ORE``
gets ``tribo`` keys ONLY - tribo is the stochastic kind, and cross-host seed
variance in it is the one question that matters beyond this task, because task
09 uses the same four kinds for its recall sweep and would inherit the same
variance. Everything else on host 2 is **locate-only** and is labelled as such
in the output, so no row implies a seed range that was never measured."""

BRACKET: Final = 1
"""Grid points either side of the crossing, giving 3 in total.

Narrowed from 2 on measured cost: the 5-point bracket unioned across 5-6
consumers averaged 10.6 amplitudes per key and would have run 13.5 h, longer
than the locate pass itself. Three still brackets the crossing on both sides,
and what the replicate pass measures is the seed-to-seed RANGE - the shape of
the curve near the crossing is already in the locate pass at full sqrt(2)
resolution, at one seed.

The censoring this risks is handled by the adaptive extension: a 3-point
bracket censors any seed whose crossing lands on an edge, and that happens
precisely when seed variance is large - the case the pass exists to detect.
"""

BRACKET_EXTEND: Final = 2
MIN_BRACKET_POINTS: Final = 2
"""Fewer than two amplitudes run means there are no edges to be censored by."""
"""Grid steps to add on each side when a seed is censored by the bracket edge."""

REPLICATE_EXCLUDE: Final[frozenset[str]] = frozenset({"mmc"})
"""Consumers whose bracket must not drive replication.

``mmc`` is ``criterion_degenerate`` - it yields no tolerance - so replicating
around its crossing buys nothing. The measured saving is only 8%, because its
crossings sit almost on top of ``mmc_burst``'s, but running a consumer five more
times to produce a number that is explicitly not a number is indefensible at any
price.
"""

N_ARGV: Final = 3

HARDWARE_CAVEAT: Final = (
    "Derived on the old cohort, which is hardware-referenced: the nerve tripole "
    "is formed by shorting the outer contacts before the amplifier, and the "
    "stomach channels are referenced in hardware. The new cohort references in "
    "SOFTWARE - three separately digitised channels summed after the ADC - which "
    "raises noise by about sqrt(2) and leaves residual common mode from "
    "inter-channel gain and phase mismatch. These tolerances are PROVISIONAL for "
    "the new cohort and must be re-derived there; invariant 12 forbids comparing "
    "across the two."
)

VELOCITY_BY_REFERENCE: Final[dict[str, Any]] = {
    "consumer": "velocity",
    "status": "recorded_by_reference",
    "source": "A.5, measured previously - NOT re-derived here",
    "tolerance": {"raw": "~1x", "band_limited": "~5x", "broadband": "~1.4x"},
    "note": "task 18 is not built; its tolerance was already measured.",
}


def _coverage(host: str, kind: str, cons: str) -> str:
    """Whether this row's threshold has a measured seed range, or one seed only.

    A row marked ``locate_only`` has a single realisation behind it. Saying so on
    the row itself stops a reader treating its number as having the same standing
    as a replicated one.
    """
    scope = REPLICATE_SCOPE.get(host)
    if scope == "all":
        return "replicated_5_seeds"
    if scope == "slow_wave" and cons == "slow_wave":
        return "replicated_5_seeds"
    if scope == "tribo" and kind == "tribo":
        return "replicated_5_seeds"
    return "locate_only_1_seed"


def _key(r: dict) -> tuple:
    return (r["host_tag"], r["kind"], r["dur_s"], r["chan_set"])


def _placement_rows(scratch: Path) -> list[dict]:  # type: ignore[type-arg]
    """Replicate rows plus every extension run of them - one source for both readers.

    The extension continues the replicate seeds' curves at more amplitudes, so a
    reader that took the replicate file alone would still see every extended seed
    as censored. ``censored`` and ``final`` both come through here for that reason.
    """
    rows = [r for r in _load(scratch, "sweep_manifest_replicate.json") if r.get("ok")]
    for f in sorted(scratch.glob("sweep_manifest_extend*.json")):
        rows += [r for r in _load(scratch, f.name) if r.get("ok")]
    return rows


def _load(scratch: Path, name: str) -> list[dict]:
    """Read one sweep manifest.

    The ONE read path, so every mode applies the same breathing criterion (see
    :func:`apply_breathing_displacement`).
    """
    p = scratch / name
    if not p.exists():
        return []
    rows = json.loads(p.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else [rows]
    return [apply_breathing_displacement(r) if isinstance(r, dict) else r for r in rows]


def _crossing(curve: list[tuple[float, int]], direction: str) -> float | None:
    """Return the tolerance amplitude, honouring the kind's direction.

    ``up``   lowest amplitude at which the output changed.
    ``down`` lowest rail at which the output is STILL unchanged, i.e. the rail
             above which nothing happens. Scanning upward here would return the
             bottom of the grid every time.
    """
    pts = sorted(curve)
    if direction == "up":
        for amp, changed in pts:
            if changed > 0:
                return amp
        return None
    for amp, changed in pts:
        if changed == 0:
            return amp
    return None


def _censored_side(curve: list[tuple[float, int]], direction: str) -> str | None:
    """Which side of the bracket a seed's tolerance lies beyond, or ``None``.

    ``_crossing`` returns the grid point ``x`` with the tolerance in
    ``(previous, x]``. That is resolved whenever ``x`` has a run point below it,
    **including when ``x`` is the top edge**: for ``up`` the lower amplitudes
    were unchanged and the top one changed; for ``down`` the lower rails changed
    and the top one did not. Either way the threshold is pinned to one sqrt(2)
    step. Only two cases are censored, and each needs extending on ONE side:

    * ``x`` is the bottom edge -> the tolerance may be lower: ``"below"``.
    * no crossing at all -> for ``up`` nothing changed even at the top; for
      ``down`` every rail changed including the top. Either way ``"above"``.

    Treating a top-edge crossing as censored, and extending both sides for every
    censored seed, was the first version of this and roughly doubled the cost.

    Judged against the seed's OWN lowest point, not the key's: a seed whose
    bottom run failed would otherwise have a lowest-point crossing mistaken for
    a resolved one.
    """
    x = _crossing(curve, direction)
    if x is None:
        return "above"
    if x == min(a for a, _ in curve):
        return "below"
    return None


SENSITIVE_SIDE: Final[dict[str, str]] = {"up": "below", "down": "above"}
"""Which censored side holds the sensitive end: low amplitudes for additive kinds,
HIGH rails for clip (damage even at a high rail is the sensitive placement)."""

MIN_GATE_PLACEMENTS: Final = 2
"""A row the 09 gate may read needs at least this many placements (replicated)."""

SENSITIVE_Q: Final = 0.10
"""The placement quantile reported as the tolerance: the most-sensitive 10%."""


def _seed_interval(
    curve: list[tuple[float, int]], direction: str
) -> tuple[float, float]:
    """Return ``(lo, hi]`` bounding one seed's tolerance, from the points it ran.

    Resolved -> (previous run point, crossing]; censored below -> (0, lowest];
    censored above -> (highest, inf). Uses only points this seed actually ran.
    """
    amps = sorted(a for a, _ in curve)
    side = _censored_side(curve, direction)
    if side == "below":
        return 0.0, amps[0]
    if side == "above":
        return amps[-1], math.inf
    x = _crossing(curve, direction)
    assert x is not None
    return amps[amps.index(x) - 1], x


def placement_summary(
    curves: list[list[tuple[float, int]]], direction: str
) -> dict[str, object]:
    """Summarise a tolerance ACROSS PLACEMENTS, most-sensitive end first.

    Sensitivity varies more than 2x with where an artifact lands, so a scalar is a
    property of one arbitrary placement. This reports the ``SENSITIVE_Q`` quantile
    at the sensitive end - low tolerances for ``up`` kinds, HIGH rails for ``down``
    (clip damages even at a high rail when it is sensitive) - taken from each
    seed's interval so censored seeds count instead of being dropped:

    * ``up``: at least k of n seeds have tolerance <= the k-th smallest upper
      bound, so the q-quantile is <= ``sensitive_end_sigma``.
    * ``down``: at least k seeds have tolerance > the k-th largest lower bound, so
      the (1-q)-quantile is >= ``sensitive_end_sigma``.

    ``sensitive_end_censored`` is true when the seed supplying that bound was
    itself censored at the sensitive end: the tolerance lies beyond the grid
    point named, and the number is a bound rather than a measurement.
    """
    iv = [_seed_interval(c, direction) for c in curves]
    n = len(iv)
    k = max(1, math.ceil(SENSITIVE_Q * n))
    if direction == "up":
        pick = sorted(iv, key=lambda t: t[1])[k - 1]
        value, censored = pick[1], pick[0] == 0.0
    else:
        pick = sorted(iv, key=lambda t: t[0], reverse=True)[k - 1]
        value, censored = pick[0], math.isinf(pick[1])
    resolved = [hi for lo, hi in iv if lo > 0.0 and not math.isinf(hi)]
    # A placement whose output changes, then stops changing, at higher amplitude.
    # Its "lowest changed" reading is conservative, but the criterion is flipping
    # rather than crossing a threshold - measured 19% of breathing seed-curves
    # after the extension - and the gate needs to see that on the row.
    non_monotone = 0
    for c in curves:
        past = [(v > 0) if direction == "up" else (v == 0) for _, v in sorted(c)]
        non_monotone += any(past[i] and not past[i + 1] for i in range(len(past) - 1))
    # What the k-th of n order statistics actually estimates: the k/(n+1)
    # quantile in expectation. With 5 placements the most sensitive one is the
    # ~17th percentile, not the 10th, and reaches the true 10th only 41% of the
    # time (1 - 0.9^5); it takes 22 placements to reach 90%. Carried on every row.
    supported = round(100 * k / (n + 1))
    out: dict[str, object] = {
        "n_placements": n,
        "sensitive_end_quantile": SENSITIVE_Q if direction == "up" else 1 - SENSITIVE_Q,
        "supported_percentile": supported if direction == "up" else 100 - supported,
        "label": f"~{supported}th percentile from the sensitive end, {n} placements",
        "sensitive_end_sigma": value,
        "sensitive_end_censored": censored,
        "n_resolved": len(resolved),
        "n_censored_below": sum(1 for lo, _ in iv if lo == 0.0),
        "n_censored_above": sum(1 for _, hi in iv if math.isinf(hi)),
        "n_non_monotone": non_monotone,
    }
    if resolved:
        out["resolved_min_sigma"] = min(resolved)
        out["resolved_max_sigma"] = max(resolved)
        out["resolved_spread_sqrt2_steps"] = round(
            math.log(max(resolved) / min(resolved), math.sqrt(2.0)), 3)
    return out


def gate_eligible(summary: dict[str, object]) -> bool:
    """Whether the 09 gate may read a measured/bounded row: replicated placements."""
    return int(summary["n_placements"]) >= MIN_GATE_PLACEMENTS  # type: ignore[call-overload]


HOST_EXCLUSIONS: Final[dict[tuple[str, str], dict[str, str]]] = {
    ("host1_JEL", "T_hardware"): {
        "reason": "mains_dominated_host",
        "ruling": (
            "2026-09-28 (task 09, polarity result): the JEL host's spike events are "
            "mains impulses - event-train peaks at 30/60/90 Hz, median inter-event "
            "16.6 ms, widths 0.12-0.16 ms - so its T_hardware rows measured the "
            "sensitivity of a mains-impulse count, not of spike detection. ORE, clean "
            "in every train, is the old-cohort T_hardware host. JEL rows for other "
            "consumers are unaffected."
        ),
    },
}
"""(host, consumer) pairs whose rows the gate may not read, whatever their
replication, with the reason written onto each row."""


def host_exclusion(host: str, consumer: str) -> str | None:
    """Return why ``(host, consumer)`` rows are not gate-eligible, or None."""
    entry = HOST_EXCLUSIONS.get((host, consumer))
    return entry["reason"] if entry else None


def bracket_key(host: str, kind: str, dur_s: float, chan_set: str) -> str:
    """Canonical key shared by the analyser and the generator.

    ``:g`` on a float, deliberately and on BOTH sides. MATLAB's ``jsonencode``
    writes 5.0 as ``5``, so the analyser read the duration back as int 5 and
    built ``...|5|...`` while the generator formatted 5.000 as ``...|5.0|...``.
    The lookup missed and the ENTIRE 5 s duration axis produced zero replicate
    points - 21 of 67 keys - while everything looked like it had worked. 0.05
    and 0.5 format identically either way, which is why only the 5 s rows were
    lost and why the failure was quiet.
    """
    return f"{host}|{kind}|{float(dur_s):g}|{chan_set}"


def _in_replicate_scope(host: str, kind: str, consumer: str) -> bool:
    """Whether this (host, kind, consumer) row may drive a replicate bracket."""
    if consumer in REPLICATE_EXCLUDE:
        return False
    scope = REPLICATE_SCOPE.get(host)
    if scope is None:
        return False
    if scope == "slow_wave":
        return consumer == "slow_wave"
    if scope == "tribo":
        return kind == "tribo"
    return True


def crossings(scratch: Path) -> None:
    """Locate pass -> crossings.json, the bracket the replicate pass will fill."""
    man = json.loads((scratch / "manifest.json").read_text(encoding="utf-8"))
    direction = man["kind_direction"]
    amps_all = sorted(set(man["amp_sigma"]) | set(man.get("amp_sigma_low", [])))
    rows = [r for r in _load(scratch, "sweep_manifest.json") if r.get("ok")]
    print(f"{len(rows)} usable locate rows")

    curves: dict[tuple, dict[str, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        for c in CONSUMERS:
            f = f"{c}_changed"
            if f in r and r[f] is not None:
                curves[_key(r)][c].append((float(r["amp_sigma"]), int(r[f])))

    out: dict[str, list[float]] = {}
    masks: dict[str, dict[str, list[str]]] = {}
    per_amp: dict[tuple, dict[float, set[str]]] = defaultdict(dict)
    summary = []
    n_keys = 0
    n_no_crossing = 0
    n_out_of_scope = 0
    for key, per in sorted(curves.items()):
        host, kind, dur, cs = key
        n_keys += 1
        scope = REPLICATE_SCOPE.get(host)
        want: set[float] = set()
        for c, curve in per.items():
            x = _crossing(curve, direction[kind])
            summary.append({
                "host_tag": host, "kind": kind, "dur_s": dur, "chan_set": cs,
                "consumer": c, "direction": direction[kind],
                "crossing_sigma": x,
                "n_points": len(curve),
                "max_changed": max(v for _, v in curve),
            })
            # A key with no crossing inside the grid has nothing to bracket.
            if x is None or not _in_replicate_scope(host, kind, c):
                continue
            grid = sorted({a for a, _ in curve})
            i = grid.index(x)
            for a in grid[max(0, i - BRACKET) : i + BRACKET + 1]:
                want.add(a)
                # PER-CONSUMER MASK. At an amplitude only one consumer's bracket
                # contains, running the other five measures nothing: their rows
                # fall outside their own brackets and are discarded. They are
                # deterministic, so skipping them is exact, not approximate.
                # Measured: 51% of the replicate pass, and 291 of 363 points
                # need exactly one consumer.
                per_amp[key].setdefault(a, set()).add(c)
        if want:
            name = bracket_key(host, kind, dur, cs)
            out[name] = sorted(want)
            masks[name] = {
                f"{a:g}": sorted(per_amp[key][a]) for a in sorted(want)
            }
        elif scope is None or (scope == "tribo" and kind != "tribo"):
            n_out_of_scope += 1
        else:
            n_no_crossing += 1

    (scratch / "crossings.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8", newline="\n"
    )
    (scratch / "crossings_mask.json").write_text(
        json.dumps(masks, indent=1), encoding="utf-8", newline="\n"
    )
    (scratch / "locate_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8", newline="\n"
    )
    nfound = sum(1 for s in summary if s["crossing_sigma"] is not None)
    n_amp = sum(len(v) for v in out.values())
    print(f"{nfound}/{len(summary)} (key, consumer) rows have a crossing")
    print(f"{n_keys} keys total: {len(out)} to replicate, "
          f"{n_no_crossing} with no crossing inside the grid, "
          f"{n_out_of_scope} out of replicate scope")
    print(f"replicate cost: {n_amp} amplitudes x 5 seeds = {n_amp * 5} points")
    print(f"grid spans {min(amps_all)} .. {max(amps_all)} sigma")


def _extension_for(
    seeds: dict[int, list[tuple[float, int]]],
    direction: str,
    grid: list[float],
    amps_run: list[float],
) -> tuple[set[float], list[int]]:
    """Amplitudes to add, and the seeds to run them for, for one (key, consumer).

    **Only seeds censored on the SENSITIVE side** - below for ``up`` kinds, above
    for clip's inverted axis (a rail that still changes the output at the top of
    the bracket is the sensitive end). The reported tolerance is the sensitive-end
    quantile (:func:`placement_summary`); a seed censored on the insensitive side
    cannot move it, so extending it buys nothing (spend decision (a), 2026-09-26:
    550 points instead of 1510). Each censored seed extends only that one side.
    """
    sensitive = SENSITIVE_SIDE[direction]
    i_lo, i_hi = grid.index(amps_run[0]), grid.index(amps_run[-1])
    want: set[float] = set()
    cens_seeds: list[int] = []
    for seed, curve in seeds.items():
        side = _censored_side(curve, direction)
        if side != sensitive:
            continue
        cens_seeds.append(seed)
        if side == "below":
            want |= set(grid[max(0, i_lo - BRACKET_EXTEND) : i_lo])
        else:
            want |= set(grid[i_hi + 1 : i_hi + 1 + BRACKET_EXTEND])
    return want, cens_seeds


def censored(scratch: Path) -> None:
    """Replicate pass -> crossings_extend.json, for the seeds the bracket clipped.

    A 3-point bracket censors any seed whose crossing lands on an edge: all we
    learn is "at or beyond here", which is not a threshold. That happens exactly
    when seed variance is large - the case the replicate pass exists to detect,
    and the tribo question specifically - so leaving it censored would report
    "narrow range" for the widest rows.

    Narrow variance costs nothing here: no seed lands on an edge and this writes
    an empty file. Wide variance gets the bracket extended outward and only the
    censored seeds re-run.
    """
    man = json.loads((scratch / "manifest.json").read_text(encoding="utf-8"))
    direction = man["kind_direction"]
    grid = sorted(set(man["amp_sigma"]) | set(man.get("amp_sigma_low", [])))
    rows = _placement_rows(scratch)
    if not rows:
        print("no replicate rows yet")
        return

    per: dict[tuple, dict[int, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        for c in CONSUMERS:
            f = f"{c}_changed"
            if f in r and r[f] is not None:
                per[(*_key(r), c)][int(r["seed"])].append(
                    (float(r["amp_sigma"]), int(r[f]))
                )

    out: dict[str, dict[str, object]] = {}
    n_cens = 0
    for k, seeds in sorted(per.items()):
        host, kind, dur, cs, cons = k
        # The same gate that built the bracket decides who may extend it. Without
        # it mmc - criterion_degenerate, excluded from the bracket for exactly
        # that reason - came back in through the extension with 125 rows.
        if not _in_replicate_scope(host, kind, cons):
            continue
        amps_run = sorted({a for curve in seeds.values() for a, _ in curve})
        if len(amps_run) < MIN_BRACKET_POINTS:
            continue
        want, cens_seeds = _extension_for(seeds, direction[kind], grid, amps_run)
        if want and cens_seeds:
            n_cens += len(cens_seeds)
            name = bracket_key(host, kind, dur, cs)
            entry = out.setdefault(name, {"amps": [], "seeds": [], "consumers": []})
            entry["amps"] = sorted(set(entry["amps"]) | want)
            entry["seeds"] = sorted(set(entry["seeds"]) | set(cens_seeds))
            entry["consumers"] = sorted(set(entry["consumers"]) | {cons})

    (scratch / "crossings_extend.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8", newline="\n"
    )
    n_pts = sum(len(v["amps"]) * len(v["seeds"]) for v in out.values())
    print(f"{n_cens} censored (key, consumer, seed) rows across {len(out)} keys")
    print(f"extension costs {n_pts} points")
    if not out:
        print("no seed landed on a bracket edge - the 3-point bracket sufficed")


def final(scratch: Path) -> None:
    """Replicate pass -> consumer_tolerances.json, the deliverable."""
    man = json.loads((scratch / "manifest.json").read_text(encoding="utf-8"))
    direction = man["kind_direction"]
    prep = json.loads((scratch / "prepare.json").read_text(encoding="utf-8"))
    hosts = prep["hosts"] if isinstance(prep["hosts"], list) else [prep["hosts"]]
    rows = [r for r in _load(scratch, "sweep_manifest.json") if r.get("ok")]
    rows += _placement_rows(scratch)
    # Seed-depth rounds add placements (seeds 6-20) to the cells they cover.
    for f in sorted(scratch.glob("sweep_manifest_seeddepth_r*.json")):
        rows += [r for r in _load(scratch, f.name) if r.get("ok")]

    # per (key, consumer, seed) -> crossing, then median and range over seeds
    per_seed: dict[tuple, dict[int, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    uv: dict[tuple, dict[float, float]] = defaultdict(dict)
    beyond: dict[tuple, dict[float, bool]] = defaultdict(dict)
    for r in rows:
        for c in CONSUMERS:
            f = f"{c}_changed"
            if f in r and r[f] is not None:
                k = (*_key(r), c)
                per_seed[k][int(r["seed"])].append(
                    (float(r["amp_sigma"]), int(r[f]))
                )
                uv[k][float(r["amp_sigma"])] = float(r.get("amp_uv_max", float("nan")))
                beyond[k][float(r["amp_sigma"])] = bool(r.get("beyond_observed", False))

    results = []
    for k, seeds in sorted(per_seed.items()):
        host, kind, dur, cs, cons = k
        if cons in DEGENERATE:
            flat = all(
                ch > 0 for curve in seeds.values() for _, ch in curve
            )
            results.append({
                "host_tag": host, "kind": kind, "dur_s": dur, "chan_set": cs,
                "consumer": cons, "direction": direction[kind],
                "status": "criterion_degenerate",
                "gate_eligible": False,
                "reason": DEGENERATE[cons],
                "changed_at_every_grid_point": flat,
                "grid_floor_sigma": min(
                    a for curve in seeds.values() for a, _ in curve
                ),
            })
            continue
        summary = placement_summary(list(seeds.values()), direction[kind])
        value = float(summary["sensitive_end_sigma"])  # type: ignore[arg-type]
        results.append({
            "host_tag": host, "kind": kind, "dur_s": dur, "chan_set": cs,
            "consumer": cons, "direction": direction[kind],
            # "measured" needs at least one placement pinned to a grid step;
            # otherwise every placement is a bound and the row says so.
            "status": "measured" if summary["n_resolved"] else "bounded",
            "seed_coverage": _coverage(host, kind, cons),
            **summary,
            "sensitive_end_uv": uv[k].get(value),
            "beyond_observed_range": beyond[k].get(value, False),
            # A FIELD the gate reads, not a rule it has to remember: a single
            # placement says nothing about the tail when placement moves the
            # threshold 4-16x. Eligible only once replicated - and never for a
            # (host, consumer) the host could not measure (HOST_EXCLUSIONS).
            "gate_eligible": gate_eligible(summary) and host_exclusion(host, cons) is None,
            **({"gate_ineligible_reason": host_exclusion(host, cons)}
               if host_exclusion(host, cons) else {}),
        })

    doc = {
        "generated_by": "step 9, tolerance_analyze.py final",
        "sign_convention": {
            "up": "tolerance = lowest amplitude at which the output changed "
                  "(step, drift, tribo: amp_ratio scales an additive artifact)",
            "down": "tolerance = lowest rail at which the output is STILL "
                    "unchanged, i.e. the rail ABOVE which nothing happens "
                    "(clip: amp_ratio IS the rail, so damage decreases with it)",
        },
        "kind_direction": direction,
        "gate_eligible_rule": (
            f"gate_eligible is true only for a measured or bounded row with >= "
            f"{MIN_GATE_PLACEMENTS} placements. Locate-only rows (1 placement) and "
            "criterion_degenerate rows are false; a row becomes eligible once replicated. "
            "A (host, consumer) in host_exclusions is false whatever its replication, "
            "with gate_ineligible_reason on the row."
        ),
        "host_exclusions": {f"{h}|{c}": v for (h, c), v in HOST_EXCLUSIONS.items()},
        "breathing_criterion": BREATHING_CRITERION,
        "non_monotone_rule": (
            "For a placement whose output changes, then stops changing, as amplitude "
            "rises, the sensitive end is the LOWEST amplitude observed to break it - "
            "not a bisection crossing. n_non_monotone counts such placements per row."
        ),
        "tolerance_shape": (
            "A tolerance is reported ACROSS PLACEMENTS, not as one number. "
            "Where an artifact lands moves the threshold by more than 2x (seed "
            "depth diagnostic), so a scalar would be a property of one arbitrary "
            "placement. sensitive_end_sigma is the most-sensitive-end quantile "
            f"(q = {SENSITIVE_Q} for up kinds, 1 - q for clip), bounded "
            "conservatively from every placement's interval - censored ones "
            "included - so the task 09 gate errs toward rejecting blanking that "
            "would fail at an unlucky placement. The resolved spread is a result "
            "in its own right: sensitivity depends on where an artifact falls "
            "about as much as on how large it is."
        ),
        "observed_max_uv": man["observed_max_uv"],
        "amplitude_units": "multiples of the host channel's broadband robust "
                           "sigma; microvolts are a derived column",
        "criteria": "pre-registered in processing_new/tolerance_criteria.m before "
                    "any injected signal was generated",
        "hosts": [
            {kk: h[kk] for kk in ("tag", "file", "why", "startSec", "durSec")}
            for h in hosts
        ],
        "caveats": {
            "hardware_vs_software_referencing": HARDWARE_CAVEAT,
            "applies_to": ["T_hardware", "mmc", "mmc_burst", "slow_wave"],
            "breathing_0p5_3_bounded_by_sweep":
                "the 0.5-3 Hz impulse response is 3988 ms and the longest "
                "injection is 5 s, so this row is bounded by the sweep rather "
                "than resolved",
            "mmc_downstream_of_hrv":
                "extract_mmc consumes the _HRBR R-peaks to remove cardiac "
                "artifact, so a NERVE injection reaches mmc without touching a "
                "stomach channel. Compare the nerve-set mmc rows against the "
                "stomach-set ones to see it measured.",
            "mmc_two_criteria":
                "'mmc' is emitted as criterion_degenerate with no value - see "
                "its reason field. 'mmc_burst' matches burst peaks within the "
                "0.5 s burst refractory and is the row task 14 should route on.",
            "mmc_fed_state":
                "The old cohort was FED (ad-lib food), and the fed pattern "
                "REPLACES the MMC with continuous irregular activity: no phase I "
                "quiescence, no phase III bursts. So extract_mmc was developed on "
                "recordings in which the MMC does not exist, and the continuous "
                "low-level 2-50 Hz activity seen here is what a fed stomach should "
                "look like. The mmc_burst tolerance is therefore a FED-STATE "
                "tolerance and must be re-derived on a fasted recording before it "
                "is applied to one. This supersedes an earlier reading of the same "
                "data as evidence that the burst detector was broken. CARRY THIS "
                "WITH THE NUMBER: mmc_burst is a FED-STATE tolerance measured "
                "under the 30 s moving threshold, so it counts near-threshold "
                "NOISE CROSSINGS (0.02 s median) rather than episodes. It must be "
                "re-derived on fasted data with a fixed reference before being "
                "applied to anything.",
            "mmc_burst_moving_threshold_defect":
                "WITHDRAWN AND REPLACED: an earlier reading of this data called "
                "the problem over-fragmentation and proposed burstRefractory "
                "0.5 s -> 3 s. That is wrong. Those groups were counted under "
                "extract_mmc's 30 s MOVING MAD, which on the fed recording was "
                "emitting ~90 'bursts' of 0.02 s median duration - near-threshold "
                "noise crossings, not fragments of real bursts. Grouping them at "
                "3 s to land near one per slow-wave cycle was coincidence. "
                "Measured on fasted new-cohort data (gems_j_t01_ms3_bl_230315, "
                "ANT1-ANT3): the moving threshold finds 3 episodes totalling 5 s "
                "(1% duty) where a FIXED reference at 2x the quiescent floor "
                "finds 40 episodes, median 2.6 s, 23% duty. With a fixed "
                "reference the median inter-episode gap is 5.0 s, so grouping "
                "stays small. The defect is the THRESHOLD, not the refractory - "
                "an adaptive baseline adapting to the thing it measures, which is "
                "the step2_noise_sigma defect for the third time. Task 08 row. "
                "burstRefractory is now SETTLED at 0.5-1 s and needs no change: "
                "measured on fasted recovery data (gems_j_t01_ms3_sr_231323) with "
                "a fixed threshold at 2x the quiescent floor, episodes are 1.4 s "
                "median (IQR 0.8-2.6) with a 4.0 s median inter-episode gap, "
                "0.73 per slow-wave cycle at 4.30 cpm (1.2 per cycle on the "
                "baseline recording by the same method). The refractory must sit "
                "well below the gap, so the withdrawn 3 s proposal would have "
                "MERGED adjacent episodes. Discrete bursts are unambiguous when "
                "fasted - ~100 uV against a +/-20 uV baseline, 1-2 s long - and "
                "absent when fed, which is the fed/fasted explanation confirmed "
                "directly rather than inferred.",
            "mmc_burst_phase_locking_null":
                "Burst times are not phase-locked to the gastric slow wave on the "
                "same channel: pooled over all hosts and channels at the current "
                "0.5 s grouping, n = 949, R = 0.029, Rayleigh p = 0.45, against a "
                "detection floor of R ~ 0.056 at that n - a powered null, not a "
                "power failure. TWO LIMITS. (a) In a FED animal there is no MMC "
                "to lock to, so this is the expected result and is not evidence "
                "about the detector. (b) Phase is measured against "
                "slowWaveAnalysis_new's own peak train, which warns that 23-38% "
                "of its intervals exceed 8 cpm on these very spans; a noisy phase "
                "reference dilutes R toward zero, so the null is conditional on a "
                "reference that is itself flagged. (c) The slow wave must be "
                "taken in a GASTRIC band, 0.033-0.20 Hz / 2-12 cpm; the 0-2 Hz "
                "band used here is dominated by a ~1 Hz component coherent at "
                "1.00 with the nerve channels, i.e. respiration - so the phase "
                "reference may not be gastric at all.",
            "span_length_vs_rat_mmc":
                "A rat antral MMC cycle is 17.5 +/- 5.8 min (phase I 5.4, II 7.1, "
                "III 3.2, IV 1.8) - not the 90-120 min of human and dog. The T "
                "spans are 180-420 s and therefore sit inside a single phase, so "
                "the absence of a quiescent period in them is expected and is not "
                "evidence about MMC in either direction.",
        },
        "cost_and_benchmarking": (
            "The locate pass was estimated at 2.5 h and ran ~7 h - a 3x "
            "underestimate. The estimate came from a 6-point SERIAL smoke test "
            "that was host1-only and half 'nerve', the cheapest channel set. It "
            "therefore benchmarked a different job: it excluded the 552 points on "
            "the 420 s span (~2.3x each) and the common_mode points that run all "
            "five consumers, and it did not capture eight MATLAB workers "
            "contending for memory bandwidth on 4.4M-sample filtfilt calls. "
            "LESSON FOR TASK 19: an acceptance-run estimate must be benchmarked "
            "on the most expensive cell of the grid under the intended worker "
            "count, not on a cheap serial sample. A 3x miss there is an overnight "
            "surprise."
        ),
        "replicate_scope": {
            "host1_JEL": "all keys with a crossing inside the grid, 5 seeds",
            "host2_ORE_long": "slow_wave keys only, 5 seeds",
            "host2_ORE": "tribo keys only, 5 seeds - the cross-host question that "
                         "matters, since task 09 uses the same four kinds",
            "everything_else_on_host2": "LOCATE ONLY, one seed. Rows carry "
                                        "seed_coverage = locate_only_1_seed.",
        },
        "velocity": VELOCITY_BY_REFERENCE,
        "results": results,
    }
    (scratch / "consumer_tolerances.json").write_text(
        json.dumps(doc, indent=1), encoding="utf-8", newline="\n"
    )
    wide = [r for r in results if float(r.get("resolved_spread_sqrt2_steps", 0.0)) > 1.0]
    bound = [r for r in results if r.get("sensitive_end_censored")]
    print(f"wrote consumer_tolerances.json: {len(results)} rows, {len(wide)} with a "
          f"resolved placement spread wider than one sqrt(2) step, {len(bound)} whose "
          "sensitive end is a bound (censored there), not a measurement")


if __name__ == "__main__":
    modes = {"crossings": crossings, "censored": censored, "final": final}
    if len(sys.argv) != N_ARGV or sys.argv[2] not in modes:
        raise SystemExit(f"usage: tolerance_analyze.py <scratch> {'|'.join(modes)}")
    modes[sys.argv[2]](Path(sys.argv[1]))
