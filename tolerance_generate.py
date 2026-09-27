"""Step 9 stage 2: generate the injected spans for the consumer-tolerance sweep.

**Python generates, MATLAB consumes.** The artifact library is Python and stays
Python: reimplementing artifact morphology in MATLAB to keep everything in one
language would repeat, one level down, the exact error the "never reimplement
the thing you are measuring" rule exists to prevent. Crossing the language
boundary with files is the same handoff tasks 08 and 15 already use.

This script does **not** choose the clean span. MATLAB's ``tolerance_prepare``
does, using the pipeline's own definitions of clean, and exports the sample
range in ``prepare.json``. Two implementations of "which span is clean" would
give two answers, and the tolerance would belong to whichever one ran here.

Amplitude is gridded in **sigma**, not microvolts: a fixed microvolt amplitude
is a different multiple of sigma on each channel, and every consumer thresholds
in sigma. Microvolts are carried as a derived column.

**Only the injected window is written, not the whole span.** A 180 s five-channel
span is 176 MB; one file per grid point would be ~48 GB for one host. The
injectors confine their change to ``[t0, t0+dur)``, which is asserted here, so
each point stores just that window and MATLAB splices it into the baseline span.

Usage, after ``tolerance_prepare`` and before ``tolerance_sweep``::

    python tolerance_generate.py <scratch_dir>                # locate pass
    python tolerance_generate.py <scratch_dir> --replicate    # after crossings known
"""

from __future__ import annotations

import json
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).parent))
from tests.conftest import inject_artifact
from tolerance_analyze import bracket_key

KINDS: Final[tuple[str, ...]] = ("step", "clip", "drift", "tribo")
"""The four that already exist. ``inject_transplant`` is excluded deliberately:
T needs a known amplitude and a transplanted real chunk does not have one."""

DURATIONS_S: Final[tuple[float, ...]] = (0.050, 0.500, 5.000)
"""The second axis. T is a surface, not a curve - the slow consumers cannot
resolve a brief injection (measured impulse responses: 2-50 486 ms, 0-2 1146 ms,
0.5-3 3988 ms)."""

AMP_SIGMA: Final[tuple[float, ...]] = tuple(
    round(0.5 * 2 ** (i / 2), 6) for i in range(23)
)
"""0.5 to 2048 sigma in sqrt(2) steps. Covers the specified 10 uV - 50 mV against
a ~18.6 uV host sigma. A scalar read off a coarser grid is a grid artifact - this
project has produced three wrong constants that way."""

AMP_SIGMA_LOW: Final[tuple[float, ...]] = tuple(
    round(0.5 * 2 ** (-i / 2), 6) for i in range(5, 0, -1)
)
"""Five extra sqrt(2) steps below 0.5 sigma, down to 0.0884 sigma.

Originally for the channel sets that drive ``mmc`` (now every 180 s set - see
``_amps_for``). mmc's strict criterion compares the
burst fiducial as an exact sample set with no matching tolerance, so a burst peak
moving one sample counts as one lost plus one added and the criterion fires at
arbitrarily small amplitude. Extending downward LOCATES that number instead of
reporting it as "below the grid" - a consumer that changes at 0.09 sigma is a
finding about the consumer."""

KIND_DIRECTION: Final[dict[str, str]] = {
    "step": "up", "drift": "up", "tribo": "up", "clip": "down",
}
"""**clip's amplitude axis is inverted, and this is a trap.**

For ``step``, ``drift`` and ``tribo``, ``amp_ratio`` scales an additive artifact:
larger is worse. For ``clip`` it IS the rail, so a LOWER rail clips harder and
damage DECREASES with amplitude; a rail above the host's peak does nothing at
all. A crossing finder scanning upward for "the first amplitude at which the
output changes" returns the bottom of the grid for every clip row and looks
entirely reasonable doing it.

``clip``'s tolerance is therefore the rail **above** which nothing changes.
Carried in the manifest so the analysis reads the direction rather than assuming
it, and stated in the output as a sign convention."""

OBSERVED_MAX_UV: Final = 500.0
"""Largest |sample| anywhere in the host. The sweep runs past it and the output
MARKS where the curve leaves the physically observed range rather than
truncating: a tolerance existing only above anything the electrode has ever seen
is a finding about the consumer, not a number to use."""

CHANNEL_SETS: Final[dict[str, tuple[int, ...]]] = {
    "nerve": (0, 1),
    "stomach": (2, 3, 4),
    "common_mode": (0, 1, 2, 3, 4),
}
"""Per-consumer targets, plus the common-mode condition. Common mode is the
motion case and the only one that exercises the hardware tripole's rejection;
without it the sweep measures differential artifact only."""

LOCATE_SEED: Final = 0
REPLICATE_SEEDS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)
"""One grid point is one realisation. ``tribo`` is stochastic, and the result
depends on where the artifact landed - a 500 ms injection on a breath peak is
not the same experiment as one between breaths. The replicate pass varies both
the waveform and ``t0``."""

CONSUMER_COST_S: Final[dict[str, float]] = {
    "T_hardware": 12.2, "hrv": 6.6, "breathing": 0.0,
    "slow_wave": 49.9, "mmc_burst": 11.2,
}
"""Serial cost per call on a 180 s span, measured with ``maxNumCompThreads(1)``.

**Single-threaded, because that is what a pool worker is.** MATLAB gives each
parpool worker one computational thread; the client gets one per physical core
(24 here). A serial control run in the client therefore measures a different
machine. It went unnoticed for three consumers, whose hot paths are not
implicitly multithreaded, and was a factor of 6 for ``slow_wave``, whose Gaussian
``smoothdata`` over a 122,070-sample window is (15.1 s on 24 threads, 89.9 s on
one). That factor was first read as contention.

``slow_wave`` 49.9 s is measured AFTER Andrea's 2026-09-26 edit removing the
duplicate ``smoothdata`` (bit-identical output): 92.3 s before, 49.9 s after on
host1_JEL, -46%; host2_ORE 93.9 -> 49.8 s.

``breathing`` is 0.0 because it comes out of the same ``HR_BR_HRVAnalysis_new``
call as ``hrv``; ``_point_cost_s`` charges that call once when either is wanted.
"""

CONTENTION: Final[dict[int, dict[str, float]]] = {
    8: {"T_hardware": 1.08, "hrv": 1.0, "slow_wave": 1.10, "mmc_burst": 1.13},
    10: {"T_hardware": 1.20, "hrv": 1.20, "slow_wave": 1.30, "mmc_burst": 1.18},
}
"""Parallel / single-threaded-serial wall time per call, BY WORKER COUNT.

8 workers: the 1710-point replicate sweep against the serial control (machine 26%
busy, 0 run-queue, 99% disk idle). 10 workers: the six seed-depth rounds, 434
points (slow_wave n=116, T_hardware 101, mmc_burst 107, hrv 110). Contention rose
with the count - the first evidence that adding processes costs something - so
the factor is per count, and an unmeasured count raises rather than borrowing a
neighbour's number (re-measure when the count changes).
"""

N_WORKERS: Final = 10
"""Workers the next MATLAB run will use: floor(0.8 x 54.7 GB / 4.21 GB peak RSS)
for 180 s spans (invariant 37). The Processes profile defaults to 8 (invariant
39); the pool is opened on an in-session cluster object with NumWorkers raised."""

LONG_SPAN_FACTOR: Final = 111.9 / 49.9
"""slow_wave's measured cost ratio, 420 s span over 180 s span, single-threaded
(post-edit). The span ratio is 2.33; smoothing cost is linear in span at a fixed
window, and the fixed per-call overheads bring the measured ratio to 2.24."""

EDGE_KEEP_S: Final = 20.0
"""Keep the injection this far from either end of the span, so no consumer's
edge buffer straddles it."""


@dataclass(frozen=True, slots=True)
class Point:
    """One sweep point: everything MATLAB needs to run it and label the result."""

    idx: int
    host_tag: str
    consumers: list[str]
    est_cost_s: float
    kind: str
    dur_s: float
    amp_sigma: float
    chan_set: str
    seed: int
    pass_name: str
    t0_s: float
    start_sample0: int
    n_samples: int
    cols1: list[int]
    file: str
    amp_uv_per_chan: list[float]
    beyond_observed: bool


def _point_cost_s(host_tag: str, consumers: list[str], n_workers: int = N_WORKERS) -> float:
    """Estimated worker-seconds, for longest-processing-time-first dispatch.

    Masking made the workload heterogeneous - 291 points need one consumer and a
    few need three - so manifest order would leave workers idle at the tail while
    one long point finishes. Longest-first is within 4/3 of optimal for identical
    machines and costs a single sort.
    """
    cons = set(consumers)
    total = 0.0

    def charge(name: str, factor: float = 1.0) -> None:
        nonlocal total
        total += CONSUMER_COST_S[name] * CONTENTION[n_workers][name] * factor

    if "T_hardware" in cons:
        charge("T_hardware")
    if cons & {"hrv", "breathing"}:
        charge("hrv")
    if "mmc_burst" in cons:
        charge("mmc_burst")
    if "slow_wave" in cons:
        charge("slow_wave", LONG_SPAN_FACTOR if host_tag.endswith("_long") else 1.0)
    return total


def _seed_for(tag: str, kind: str, dur_s: float, chan_set: str, seed: int) -> int:
    """Deterministic across runs and machines.

    ``hash()`` on a str is salted per interpreter process, so it would give a
    different waveform every run - CLAUDE.md requires an explicit seed, and an
    irreproducible one is worse than none because it looks reproducible.
    """
    key = f"{tag}|{kind}|{dur_s}|{chan_set}|{seed}".encode()
    return zlib.crc32(key) & 0xFFFFFFFF


def _t0_for(seed: int, dur_s: float, span_s: float, rng: np.random.Generator) -> float:
    """Injection position. Fixed at the midpoint for locate, varied for replicate."""
    lo = EDGE_KEEP_S
    hi = span_s - dur_s - EDGE_KEEP_S
    if hi <= lo:
        return max(0.0, (span_s - dur_s) / 2.0)
    if seed == LOCATE_SEED:
        return (lo + hi) / 2.0
    return float(rng.uniform(lo, hi))


def _inject_window(
    span: np.ndarray, fs: float, cols: tuple[int, ...], t0_s: float,
    dur_s: float, kind: str, amp: float, seed: int,
) -> tuple[np.ndarray, int, int]:
    """Return (window, start_sample0, n) holding only the samples that changed."""
    i0 = int(round(t0_s * fs))
    n = int(round(dur_s * fs))
    win = np.empty((n, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        out, _ = inject_artifact(span[:, c], fs, t0_s, dur_s, kind, amp, seed=seed)
        changed = np.flatnonzero(out != span[:, c])
        if changed.size:
            assert changed[0] >= i0 and changed[-1] < i0 + n, (
                f"{kind} changed samples outside [t0, t0+dur): "
                f"{changed[0]}..{changed[-1]} vs {i0}..{i0 + n - 1}. "
                "The window-only optimisation assumes the injectors are confined."
            )
        win[:, j] = out[i0 : i0 + n]
    return win, i0, n


def _amps_for(tag: str, chan_set: str, replicate: bool, crossings: dict, key: str) -> tuple:
    """Amplitude list for one (kind, duration, channel-set).

    Every set on the 180 s hosts gets the downward extension to 0.0884 sigma. The
    nerve set used to stop at 0.5 sigma on the reasoning that mmc never reads it;
    the seed-depth diagnostic then put 6 of 20 T_hardware placements (step, 0.05 s,
    nerve) BELOW 0.5 sigma, so the floor was censoring the sensitive end of a
    nerve consumer. The long host runs slow_wave only and keeps the main grid.
    """
    if replicate:
        return tuple(crossings.get(key, []))
    if not tag.endswith("_long"):
        return AMP_SIGMA_LOW + AMP_SIGMA
    return AMP_SIGMA


def generate(scratch: Path, replicate: bool) -> None:  # noqa: PLR0912, PLR0915
    """Write one .mat per sweep point plus a manifest."""
    prepare = json.loads((scratch / "prepare.json").read_text(encoding="utf-8"))
    hosts = prepare["hosts"]
    if isinstance(hosts, dict):
        hosts = [hosts]
    outdir = scratch / "injected"
    outdir.mkdir(parents=True, exist_ok=True)

    if replicate:
        crossings = json.loads((scratch / "crossings.json").read_text(encoding="utf-8"))
        masks = json.loads((scratch / "crossings_mask.json").read_text(encoding="utf-8"))
    else:
        crossings = None
        masks = {}

    points: list[Point] = []
    idx = 0
    for host in hosts:
        tag = host["tag"]
        fs = float(host["fs"])
        i0 = int(host["startSample"]) - 1  # MATLAB 1-based inclusive -> 0-based
        i1 = int(host["stopSample"])  # -> 0-based half-open [i0, i1)
        sigma_v = np.atleast_1d(np.asarray(host["sigmaUV_broadband"], np.float64)) * 1e-6

        print(f"[{tag}] loading host ...", flush=True)
        y = np.asarray(loadmat(host["file"], variable_names=["yOut"])["yOut"], np.float64)
        span = np.ascontiguousarray(y[i0:i1, :])
        del y
        span_s = span.shape[0] / fs
        assert np.isfinite(span).all(), f"{tag}: span contains NaN - it is not clean"
        print(f"[{tag}] span {span.shape}, {span_s:.1f} s", flush=True)

        # slow_wave is the only consumer needing the long span; the rest run on
        # the 180 s spans so the two hosts stay comparable.
        sets = ("stomach", "common_mode") if tag.endswith("_long") else tuple(CHANNEL_SETS)

        for kind in KINDS:
            for dur_s in DURATIONS_S:
                if dur_s >= span_s - 2 * EDGE_KEEP_S:
                    continue
                for chan_set in sets:
                    cols = CHANNEL_SETS[chan_set]
                    key = bracket_key(tag, kind, dur_s, chan_set)
                    amps = _amps_for(tag, chan_set, replicate, crossings or {}, key)
                    seeds = REPLICATE_SEEDS if replicate else (LOCATE_SEED,)
                    pass_name = "replicate" if replicate else "locate"
                    for seed in seeds:
                        rng = np.random.default_rng(
                            _seed_for(tag, kind, dur_s, chan_set, seed)
                        )
                        t0_s = _t0_for(seed, dur_s, span_s, rng)
                        for amp in amps:
                            win, s0, n = _inject_window(
                                span, fs, cols, t0_s, dur_s, kind, float(amp), seed
                            )
                            name = f"{pass_name}_{idx:06d}.mat"
                            savemat(
                                outdir / name,
                                {"win": win, "start0": s0, "cols1": np.asarray(cols) + 1},
                                do_compression=False,
                            )
                            amp_uv = [float(amp * sigma_v[c] * 1e6) for c in cols]
                            # Only the consumers whose OWN bracket contains this
                            # amplitude. The rest would produce rows that fall
                            # outside their brackets and get discarded, and they
                            # are deterministic so skipping them is exact.
                            cons = (
                                masks.get(key, {}).get(f"{amp:g}", [])
                                if replicate else []
                            )
                            points.append(Point(
                                idx=idx, host_tag=tag, consumers=list(cons),
                                est_cost_s=_point_cost_s(tag, list(cons)),
                                kind=kind, dur_s=float(dur_s),
                                amp_sigma=float(amp), chan_set=chan_set, seed=seed,
                                pass_name=pass_name, t0_s=t0_s, start_sample0=s0,
                                n_samples=n, cols1=[c + 1 for c in cols], file=name,
                                amp_uv_per_chan=amp_uv,
                                beyond_observed=max(amp_uv) > OBSERVED_MAX_UV,
                            ))
                            idx += 1
                            if idx % 100 == 0:
                                print(f"  {idx} points", flush=True)

    # Longest-processing-time-first. Only meaningful once points carry a mask,
    # so the locate pass (every point full cost) is left in generation order.
    if replicate:
        points.sort(key=lambda q: q.est_cost_s, reverse=True)

    # RECONCILIATION, recorded rather than merely performed (invariant 23).
    # The 5 s key-format bug produced 1195 points where 1710 were costed, and it
    # was caught only because the arithmetic disagreed with a cost model built
    # beforehand. A count that is checked but not written down cannot be checked
    # again by anyone else, or by the same person a week later.
    expected = None
    if replicate:
        expected = sum(len(v) for v in (crossings or {}).values()) * len(REPLICATE_SEEDS)

    name = "manifest_replicate.json" if replicate else "manifest.json"
    _write_manifest(scratch / name, points, replicate, expected)
    _require_reconciled(len(points), expected)
    print(f"wrote {len(points)} points + {name}", flush=True)


def _require_reconciled(written: int, expected: int | None) -> None:
    """Refuse to hand the driver a manifest that does not match its cost model."""
    if expected is None or expected == written:
        return
    msg = (
        f"manifest holds {written} points but the bracket table implies "
        f"{expected}. A shortfall here is silent data loss - the 5 s duration "
        "axis vanished exactly this way once, because MATLAB writes 5.0 as 5 and "
        "the two sides built different keys. Fix the mismatch; do not run."
    )
    raise SystemExit(msg)


EXTEND_IDX_BASE: Final = 300_000
"""Keeps extension point ids - and the run directories named from them - clear of
the locate, replicate and seed-depth passes."""


def generate_extension(scratch: Path) -> None:
    """Write the sensitive-side extension from ``crossings_extend.json``.

    Each key names the amplitudes to add and the censored seeds to run them for,
    with the consumers whose seeds were censored as the mask. Placements are the
    replicate pass's own (same seed -> same ``t0``), so an extension point
    continues a seed's curve rather than starting a new placement. Only the hosts
    the extension touches are loaded.
    """
    ext = json.loads((scratch / "crossings_extend.json").read_text(encoding="utf-8"))
    prepare = json.loads((scratch / "prepare.json").read_text(encoding="utf-8"))
    hosts = prepare["hosts"] if isinstance(prepare["hosts"], list) else [prepare["hosts"]]
    outdir = scratch / "injected"
    outdir.mkdir(parents=True, exist_ok=True)
    by_host: dict[str, list[tuple[str, dict]]] = {}  # type: ignore[type-arg]
    for key, entry in ext.items():
        by_host.setdefault(key.split("|")[0], []).append((key, entry))

    points: list[Point] = []
    idx = EXTEND_IDX_BASE
    for host in hosts:
        tag = host["tag"]
        if tag not in by_host:
            continue
        fs = float(host["fs"])
        i0 = int(host["startSample"]) - 1  # MATLAB 1-based inclusive -> 0-based
        i1 = int(host["stopSample"])  # -> 0-based half-open [i0, i1)
        sigma_v = np.atleast_1d(np.asarray(host["sigmaUV_broadband"], np.float64)) * 1e-6
        y = np.asarray(loadmat(host["file"], variable_names=["yOut"])["yOut"], np.float64)
        span = np.ascontiguousarray(y[i0:i1, :])
        del y
        span_s = span.shape[0] / fs
        for key, entry in sorted(by_host[tag]):
            _, kind, dur_txt, chan_set = key.split("|")
            dur_s = float(dur_txt)
            cols = CHANNEL_SETS[chan_set]
            cons = list(entry["consumers"])
            for seed in entry["seeds"]:
                rng = np.random.default_rng(_seed_for(tag, kind, dur_s, chan_set, int(seed)))
                t0_s = _t0_for(int(seed), dur_s, span_s, rng)
                for amp in entry["amps"]:
                    win, s0, n = _inject_window(
                        span, fs, cols, t0_s, dur_s, kind, float(amp), int(seed)
                    )
                    name = f"extend_{idx:06d}.mat"
                    savemat(outdir / name,
                            {"win": win, "start0": s0, "cols1": np.asarray(cols) + 1},
                            do_compression=False)
                    amp_uv = [float(amp * sigma_v[c] * 1e6) for c in cols]
                    points.append(Point(
                        idx=idx, host_tag=tag, consumers=cons,
                        est_cost_s=_point_cost_s(tag, cons), kind=kind, dur_s=dur_s,
                        amp_sigma=float(amp), chan_set=chan_set, seed=int(seed),
                        pass_name="extend", t0_s=t0_s, start_sample0=s0, n_samples=n,
                        cols1=[c + 1 for c in cols], file=name, amp_uv_per_chan=amp_uv,
                        beyond_observed=max(amp_uv) > OBSERVED_MAX_UV,
                    ))
                    idx += 1
    points.sort(key=lambda q: q.est_cost_s, reverse=True)
    expected = sum(len(e["amps"]) * len(e["seeds"]) for e in ext.values())
    _require_reconciled(len(points), expected)
    (scratch / "manifest_extend.json").write_text(json.dumps({
        "kind_direction": dict(KIND_DIRECTION), "pass": "extend",
        "dispatch": "longest_processing_time_first",
        "n_points_expected": expected, "n_points_written": len(points),
        "est_worker_s": round(sum(q.est_cost_s for q in points), 1),
        "cost_model": {"serial_1thread_s": dict(CONSUMER_COST_S),
                       "contention_by_workers": {str(k): v for k, v in CONTENTION.items()},
                       "n_workers": N_WORKERS, "long_span_factor": LONG_SPAN_FACTOR},
        "points": [asdict(q) for q in points],
    }, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {len(points)} extension points (expected {expected}), est "
          f"{sum(q.est_cost_s for q in points):.0f} worker-s -> manifest_extend.json")


def _write_manifest(
    path: Path, points: list[Point], replicate: bool, expected: int | None = None
) -> None:
    """Serialise the manifest the MATLAB driver consumes."""
    path.write_text(
        json.dumps({
            "kinds": list(KINDS), "durations_s": list(DURATIONS_S),
            "amp_sigma": list(AMP_SIGMA),
            "amp_sigma_low": list(AMP_SIGMA_LOW),
            "kind_direction": dict(KIND_DIRECTION),
            "channel_sets": {k: list(v) for k, v in CHANNEL_SETS.items()},
            "observed_max_uv": OBSERVED_MAX_UV,
            "generator": "tests.conftest.inject_artifact",
            "pass": "replicate" if replicate else "locate",
            "dispatch": "longest_processing_time_first" if replicate else "generated",
            "n_points_expected": expected,
            "n_points_written": len(points),
            "reconciled": expected is None or expected == len(points),
            "est_worker_s": round(sum(q.est_cost_s for q in points), 1),
            "cost_model": {"serial_1thread_s": dict(CONSUMER_COST_S),
                           "contention_by_workers": {str(k): v for k, v in CONTENTION.items()},
                           "n_workers": N_WORKERS,
                           "long_span_factor": LONG_SPAN_FACTOR},
            "points": [asdict(q) for q in points],
        }, indent=1),
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(
            "usage: python tolerance_generate.py <scratch_dir> [--replicate | --extend]")
    if "--extend" in args:
        generate_extension(Path(args[0]))
    else:
        generate(Path(args[0]), "--replicate" in args)
