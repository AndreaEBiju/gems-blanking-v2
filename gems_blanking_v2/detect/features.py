"""Task 11: one feature row per candidate **core** (ruling 2026-10-07 (b) R3).

The classifier judges cores, not candidates: a core is labelled motion only when a mark
covers at least half of it, so long candidates are not learned as wholly motion. Every
feature here is computed on the same region and the same 10 ms grid that
:func:`~gems_blanking_v2.detect.chain.detect_region` used to propose the core, on the
**common signal set** (:func:`region_inputs`; a test asserts its z traces are
detection's own for those pairs).

**The common signal set (ruling 2026-10-08 (b) item 1(a), P1).** Every recording, of
either cohort, is featurised on the signals the old cohort has: each cuff's tripole
``<cuff>_T`` and the raw stomach channels ``ANT1``-``ANT3``. The new cohort's raw
contacts (``V1``-``V3``) and the derived ``stomach_ref`` are detection inputs only,
never feature inputs. :func:`inputs_from_signals` refuses any other signal name, so a
feature row cannot be built from a different set. Consequence: the contact-only
families (:data:`DROPPED_FAMILIES`) cannot exist on this set and are not features - they
were ``nan`` on every old-cohort row, so their missingness alone named the cohort.

Hard constraints (CLAUDE.md invariant 10):

* **Channel-count independent, by measurement.** A statistic over signals must not
  change with how many signals there are. ``max`` over n signals does (the expected
  maximum of n draws grows with n), and so do ``log(max/min)`` and the common-mode
  fraction of n signals (about 1/n for independent noise). Every such aggregate is the
  **pair statistic** instead: the mean over all unordered pairs of signals of the pair's
  max (min, spread, ...) - a U-statistic whose expectation is that of 2 signals whatever
  n is, and which equals the plain ``max`` when n = 2 (the common set has 2 nerve
  signals). Medians, means and fractions are kept. A statistic over every signal is
  taken per **group** (nerve, stomach) and then combined - the max of the group pair
  statistics, or the mean of the group fractions - so the mix of groups does not move
  it either. A group with fewer than 2 finite signals has no pair statistic (``nan``,
  R2). ``tests/test_features.py`` checks every feature at 2 against 8 nerve signals of
  identical per-signal statistics.
* **Animal-invariant.** Ratios, correlations, rates and robust z only. Absolute
  microvolts never enter a feature; scaling every channel by 10x changes nothing
  (tested). There are no explicit amplitude features.
* **Missing is missing (ruling (b) R2).** A feature whose input is absent is ``nan``,
  never imputed. LightGBM handles ``nan`` natively.

The ENG band is 300-3000 Hz (ruling (b) R8, A.5b); the motion ratio is
``100-300 / 300-3000``.

**Definitions.** ``N`` = nerve signals (``<cuff>_T``), ``S`` = stomach signals
(``ANT<k>``), ``G`` = the groups {N, S}. ``pmax_X(v)`` / ``pmin_X(v)`` = mean over the
unordered pairs of signals in X of the pair's max / min of the per-signal value ``v``
(:func:`pair_max`; ``nan`` below 2 finite values). ``gmax(v)`` = max over G of
``pmax_g(v)``; ``gmin`` likewise with ``pmin``; ``gmean(v)`` = mean over G of the
within-group mean of ``v``. ``W`` = the window: the core widened by the context
(suffix ``_c0``, ``_c100``, ``_c250``, ``_c500`` = +-0, 100, 250, 500 ms each side).
``P_b(s)`` = mean square of signal ``s`` band-passed to band ``b`` over W; ``e(s)`` =
its 300-3000 Hz RMS envelope at ~1 kHz; ``z(s, b)`` = detection's 10 ms z.

==========================  =========================================================
feature (per context)       definition over W
==========================  =========================================================
band_ratio_max / _median    pmax_N / median_N of ``log10(P_100-300 / P_300-3000)``
frac_signals_over           gmean of [any band of s has a frame with z > z_enter]
frac_pairs_over             gmean of the fraction of s's bands with a frame over z_enter
env_corr_mean               mean over nerve pairs of corr(log e(a), log e(b))
power_rel_max               mean over nerve pairs of max(P_a, P_b) / mean(P_a, P_b),
                            ``P`` = P_300-3000
power_rel_spread            mean over nerve pairs of ``|log10(P_a / P_b)|``
onset_rate_max              gmax of s's max over bands and frames of d(log env)/dt
offset_rate_min             gmin of s's min over bands and frames of d(log env)/dt
level_rate_corr             median over the group holding the event (larger gmax peak
                            z) of corr(level, rate) on each signal's own peak band
env_slope_max               pmax_N of max|de/dt| / mean(e)
slew_max / _median          pmax_N / median_N of max|dx/dt| / RMS (raw, <= 1 s window)
kurtosis_max / _median      pmax_N / median_N of the excess kurtosis (raw)
line_length_max / _median   pmax_N / median_N of sum|dx| / (RMS * n) (raw)
spec_entropy_median         median_N of the normalised spectral entropy (raw)
spec_edge_median            median_N of the 95% spectral edge, Hz (raw)
clip_frac                   gmax of the share of samples on a flat run at s's rail
peak_z                      gmax of s's max over bands and frames of z
z_mean                      mean over frames of [max over G of pmax_g of s's
                            max-over-bands z in that frame]
==========================  =========================================================

==========================  =========================================================
feature (once per core)     definition
==========================  =========================================================
duration_s                  the core's width, seconds
slow_pair_share             share of the core's over-threshold pairs on the common set
                            whose band is slow (``SLOW_BANDS``); ``nan`` with none
line_ratio_max              pmax_N of the mains-harmonic (+-1 Hz) share of 1-3000 Hz
                            power, in a 1-2 s window centred on the core
line_plv_max                pmax_N of the 60 Hz phase-locking value of e, same window
cm_fraction                 mean over nerve pairs of mean(((a+b)/2)^2) /
                            mean((a^2+b^2)/2), 300-3000 Hz, same window
==========================  =========================================================

The once-per-core line features let the classifier tell stationary mains-locked energy
from transient motion (ruling 2026-10-07 (c) item 2).

:data:`FEATURE_VERSION` names these definitions. A feature table carries it in a
``feature_version`` column and :func:`~gems_blanking_v2.model.modes.prepare_table`
refuses a table of any other version, so rows computed under different definitions are
never trained or scored together.

Outside the generation hash: :mod:`~gems_blanking_v2.detect.chain` does not import this
module, so building or changing features never changes what was proposed.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

from gems_blanking_v2.bands.envelope import band_envelope_for, log_envelope
from gems_blanking_v2.bands.reference import epoch_reference
from gems_blanking_v2.bands.zscore import zscore
from gems_blanking_v2.constants import BANDS, ENG_BAND, GRID_S
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.detect.cores import SLOW_BANDS
from gems_blanking_v2.types import Recording

__all__ = [
    "CONTEXTS_S",
    "DROPPED_FAMILIES",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FEATURE_VERSION_COLUMN",
    "MAINS_HZ",
    "NERVE_RE",
    "STOMACH_RE",
    "RegionInputs",
    "common_signal_set",
    "core_features",
    "feature_matrix",
    "inputs_from_signals",
    "missing_rates",
    "pair_max",
    "pair_max_cols",
    "pair_min",
    "prepare",
    "region_inputs",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
Pair = tuple[str, str]

CONTEXTS_S: Final[tuple[float, ...]] = (0.0, 0.100, 0.250, 0.500)
"""Context half-widths added on each side of the core, seconds (task 11)."""

LOW_BAND: Final = "100-300"
"""The motion band of the ratio; ``ENG_BAND`` is its denominator."""

ENV_RATE_HZ: Final = 1000.0
"""Nominal rate of the fine envelopes used for correlations and slopes (10 ms frames
hold too few points for a correlation inside a 20 ms core). The actual rate is
``fs / round(fs / ENV_RATE_HZ)`` (1017.25 Hz at 24414 Hz), carried as
``_Prepared.env_rate``: timing anything at the nominal rate drifts 1.7%."""

SHAPE_MAX_S: Final = 1.0
"""Longest raw window the shape and spectral features read, centred on the window. It
bounds the cost of a long core; the shape of 1 s of it is representative."""

LINE_WINDOW_S: Final = 1.0
"""Shortest window for the line features: 1 Hz resolution separates the mains
harmonics. A shorter core is read inside a window of this length around its centre."""

LINE_MAX_S: Final = 2.0
"""Longest window for the line features."""

MAINS_HZ: Final = 60.0
"""Mains fundamental (US). The line inventory (ruling (c) item 1) reports drift; the
features use the nominal value and a +-1 Hz bin, which holds the measured drift."""

LINE_HALF_BIN_HZ: Final = 1.0
"""Half-width of each harmonic's bin in the line ratio."""

RAIL_FRACTION: Final = 0.98
"""A sample is at the rail if ``|x| >= RAIL_FRACTION * max|x|`` over the region."""

FLAT_RUN: Final = 3
"""Consecutive identical samples that make a run flat (a saturated ADC repeats)."""

SPECTRAL_EDGE: Final = 0.95
"""Cumulative power fraction that defines the spectral edge frequency."""

NERVE_RE: Final = re.compile(r"^[A-Z]_T$")
"""A common-set nerve signal: one cuff's tripole, ``<cuff>_T``."""

STOMACH_RE: Final = re.compile(r"^ANT[0-9]+$")
"""A common-set stomach signal: a raw stomach channel, ``ANT<k>``."""


def _sfx(ctx: float) -> str:
    return f"_c{round(ctx * 1000)}"


_PER_CONTEXT: Final[tuple[str, ...]] = (
    "band_ratio_max", "band_ratio_median",
    "frac_signals_over", "frac_pairs_over",
    "env_corr_mean", "power_rel_max", "power_rel_spread",
    "onset_rate_max", "offset_rate_min", "level_rate_corr",
    "env_slope_max", "slew_max", "slew_median", "kurtosis_max", "kurtosis_median",
    "line_length_max", "line_length_median", "spec_entropy_median", "spec_edge_median",
    "clip_frac",
    "peak_z", "z_mean",
)
_ONCE: Final[tuple[str, ...]] = (
    "duration_s", "slow_pair_share", "line_ratio_max", "line_plv_max", "cm_fraction",
)

DROPPED_FAMILIES: Final[Mapping[str, str]] = {
    "cm_resid_max": ("common mode over residual power per cuff: needs a cuff's raw "
                     "contacts, which the common signal set (P1) does not carry"),
    "within_minus_across": ("within- minus across-cuff envelope agreement: needs two "
                            "signals of one cuff; the common set has one (T) per cuff"),
}
"""Families that cannot exist on the common signal set, and why (ruling 2026-10-08 (b)
item 1(a)). Before P1 they were ``nan`` on every old-cohort row and their missingness
alone separated the cohorts (AUC 1.000, cohort audit 2026-10-07)."""

FEATURE_VERSION: Final = "p1-pair-2026-10-07"
"""The version of every feature definition above. Bump it in the same change as any
edit that alters a feature value (a docstring edit does not); a table of another version
is refused for training and inference. Night 1's tables carry no stamp and are refused."""

FEATURE_VERSION_COLUMN: Final = "feature_version"
"""The column of a feature table that carries :data:`FEATURE_VERSION`."""

FEATURE_NAMES: Final[tuple[str, ...]] = _ONCE + tuple(
    f"{name}{_sfx(c)}" for c in CONTEXTS_S for name in _PER_CONTEXT)
"""Every feature column, in order. The same for every recording and every cohort."""


# ---------------------------------------------------------------------------
# region inputs: the common signal set, with detection's own z
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionInputs:
    """The common-set inputs of one region, on the region's own timeline.

    ``signals`` are each cuff's tripole ``<cuff>_T`` and the raw stomach channels
    ``ANT<k>`` (:func:`common_signal_set`). ``z`` and ``log_env`` are on the 10 ms grid,
    frame 0 at the region start, each signal against its own reference - identical to
    detection's z for the same pairs.
    """

    fs: float
    signals: Mapping[str, F64]
    z: Mapping[Pair, F64]
    log_env: Mapping[Pair, F64]
    z_enter: float


def common_signal_set(rec: Recording) -> dict[str, F64]:
    """Return the common signal set of ``rec``: each cuff's ``T`` plus the raw ``ANT<k>``.

    ``T`` comes from :func:`~gems_blanking_v2.derive.derivations.build_derivations` (the
    hardware tripole on the old cohort, the software tripole on the new); the stomach
    channels are the raw ``role == "stomach"`` channels, never the derived
    ``stomach_ref``. A stomach channel not named ``ANT<k>`` raises: the set is matched
    by name across cohorts, and a silent mismatch would change it.
    """
    signals, _w = build_derivations(rec)
    out = {k: np.asarray(v, dtype=np.float64) for k, v in signals.items()
           if NERVE_RE.match(k)}
    for c in rec.channels:
        if c.role != "stomach":
            continue
        if not STOMACH_RE.match(c.name):
            msg = (f"stomach channel {c.name!r} is not named ANT<k>; the common signal set "
                   "is matched by name across cohorts")
            raise ValueError(msg)
        out[c.name] = np.asarray(rec.data[:, c.index], dtype=np.float64)
    return out


def region_inputs(rec: Recording, region: tuple[float, float], *, z_enter: float) -> RegionInputs:
    """Build the common-set inputs of ``region`` (P1), z exactly as ``detect_region``.

    The region is cut as the chain cuts it and each signal is referenced against its own
    whole-region log envelope, so every z here equals detection's z for that pair
    (``tests/test_features.py`` asserts it) - without importing into the generation hash.
    """
    lo, hi = region
    i0, i1 = round(lo * rec.fs), round(hi * rec.fs)
    sub = replace(rec, data=rec.data[i0:i1])
    return inputs_from_signals(common_signal_set(sub), float(rec.fs), z_enter=z_enter)


def inputs_from_signals(signals: Mapping[str, F64], fs: float, *,
                        z_enter: float) -> RegionInputs:
    """Build :class:`RegionInputs` from common-set signals: z and log envelopes per pair.

    Each signal against its own whole-region log-envelope reference (invariants 3, 5),
    exactly as ``chain.z_by_pair``. The one construction site for features' z. Raises
    on a signal outside the common set (neither ``<cuff>_T`` nor ``ANT<k>``), naming it.
    """
    bad = sorted(s for s in signals if not (NERVE_RE.match(s) or STOMACH_RE.match(s)))
    if bad:
        msg = (f"signal(s) {bad} are outside the common signal set (<cuff>_T, ANT<k>; "
               "ruling 2026-10-08 (b) P1)")
        raise ValueError(msg)
    z: dict[Pair, F64] = {}
    log_env: dict[Pair, F64] = {}
    for name in sorted(signals):
        x = np.asarray(signals[name], dtype=np.float64)
        for band in BANDS:
            le = log_envelope(band_envelope_for(x, fs, band))
            ref = epoch_reference(le, signal=name, band=band)
            log_env[(name, band)] = le
            z[(name, band)] = zscore(le, ref, signal=name, band=band)
    return RegionInputs(fs=float(fs), signals=dict(signals), z=z, log_env=log_env,
                        z_enter=float(z_enter))


# ---------------------------------------------------------------------------
# channel-count-independent aggregates
# ---------------------------------------------------------------------------


def pair_max_cols(a: npt.ArrayLike) -> F64:
    """Column-wise pair-max of ``a`` (signals x columns): mean over signal pairs of the max.

    The i-th smallest of n finite values (0-based) is the max of exactly i of the
    C(n, 2) pairs, so the statistic is ``sum_i i * x_(i) / C(n, 2)``. Non-finite entries
    are ignored; a column with fewer than 2 finite values is ``nan``.
    """
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    fin = np.isfinite(arr)
    srt = np.sort(np.where(fin, arr, np.inf), axis=0)  # non-finite sort last
    n = fin.sum(axis=0).astype(np.float64)
    idx = np.arange(arr.shape[0], dtype=np.float64)[:, None]
    used = idx < n[None, :]
    num = np.sum(np.where(used, idx * np.where(used, srt, 0.0), 0.0), axis=0)
    pairs = n * (n - 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / pairs
    return np.asarray(np.where(pairs > 0, out, np.nan), dtype=np.float64)


def pair_max(values: Iterable[float]) -> float:
    """Mean over all unordered pairs of the pair's max: a channel-count-independent max.

    Its expectation over n exchangeable values is that of the max of 2, for every n, and
    it equals ``max`` when n = 2. Non-finite values are ignored; fewer than 2 is ``nan``.
    """
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if a.size < 2:  # noqa: PLR2004
        return math.nan
    return float(pair_max_cols(a)[0])


def pair_min(values: Iterable[float]) -> float:
    """Mean over all unordered pairs of the pair's min (see :func:`pair_max`)."""
    m = pair_max([-float(v) for v in values])
    return -m if np.isfinite(m) else math.nan


def _pair_mean(values: Iterable[float], stat: Callable[[float, float], float]) -> float:
    """Mean over unordered pairs of finite values of ``stat(a, b)`` (``nan`` below 2)."""
    a = [float(v) for v in values if np.isfinite(v)]
    return _nanstat([stat(x, y) for x, y in itertools.combinations(a, 2)], "mean")


def _group(name: str) -> str:
    return "nerve" if NERVE_RE.match(name) else "stomach"


def _over_groups(per_signal: Mapping[str, float], how: str) -> float:
    """Combine one value per signal across the groups so group sizes cancel.

    ``max``: the largest group pair-max; ``min``: the smallest group pair-min; ``mean``:
    the mean of the group means (used for fractions).
    """
    groups: dict[str, list[float]] = {}
    for s, v in per_signal.items():
        groups.setdefault(_group(s), []).append(v)
    if how == "max":
        return _nanstat([pair_max(v) for v in groups.values()], "max")
    if how == "min":
        return _nanstat([pair_min(v) for v in groups.values()], "min")
    return _nanstat([_nanstat(v, "mean") for v in groups.values()], "mean")


# ---------------------------------------------------------------------------
# precomputation
# ---------------------------------------------------------------------------


def _is_nerve(name: str) -> bool:
    return NERVE_RE.match(name) is not None


def _bandpass(x: F64, fs: float, lo: float, hi: float) -> F64:
    ny = fs / 2.0
    sos = butter(4, [lo / ny, min(hi, 0.9 * ny) / ny], btype="bandpass", output="sos")
    finite = np.isfinite(x)
    if finite.all():
        return np.asarray(sosfiltfilt(sos, x), dtype=np.float64)
    # NaN samples (masked) stay NaN; the filter runs on a temporary interpolation
    # that is reverted at once (invariant 8).
    idx = np.arange(x.size)
    xi = np.interp(idx, idx[finite], x[finite]) if finite.any() else np.zeros_like(x)
    y = np.asarray(sosfiltfilt(sos, xi), dtype=np.float64)
    y[~finite] = np.nan
    return y


def _env_step(fs: float) -> int:
    return max(1, int(round(fs / ENV_RATE_HZ)))


def _fine_envelope(y: F64, fs: float) -> F64:
    """RMS of ``y`` in blocks of ``_env_step(fs)`` samples (the last partial block dropped)."""
    step = _env_step(fs)
    n = (y.size // step) * step
    blocks = y[:n].reshape(-1, step)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.sqrt(np.nanmean(blocks**2, axis=1)), dtype=np.float64)


def _flat_rail(x: F64) -> Bool:
    """Mark samples on a flat run (``FLAT_RUN`` identical samples) at the channel's rail."""
    fin = np.isfinite(x)
    if not fin.any():
        return np.zeros(x.size, dtype=bool)
    rail = float(np.nanmax(np.abs(x)))
    at = fin & (np.abs(x) >= RAIL_FRACTION * rail) if rail > 0 else np.zeros(x.size, bool)
    eq = np.diff(x) == 0  # eq[i]: x[i+1] repeats x[i]
    edges = np.diff(np.concatenate([[0], eq.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    same = np.zeros(x.size, dtype=bool)
    for a, b in zip(starts, stops, strict=True):  # a run of repeats covers samples a..b
        if b - a + 1 >= FLAT_RUN:
            same[a:b + 1] = True
    return at & same


@dataclass
class _Prepared:
    inputs: RegionInputs
    n: int
    nerve: list[str]
    signals: list[str]
    env_rate: float
    eng: dict[str, F64] = field(default_factory=dict)
    low: dict[str, F64] = field(default_factory=dict)
    env: dict[str, F64] = field(default_factory=dict)
    log_fine: dict[str, F64] = field(default_factory=dict)
    clip: dict[str, Bool] = field(default_factory=dict)
    zstack: F64 = field(default_factory=lambda: np.zeros((0, 0)))
    pairs: list[Pair] = field(default_factory=list)
    zsig: F64 = field(default_factory=lambda: np.zeros((0, 0)))
    """Per signal (rows in ``signals`` order), the max over its bands of z, per frame."""


def prepare(inputs: RegionInputs) -> _Prepared:
    """Precompute the band-limited signals, fine envelopes and rail masks of a region."""
    fs = inputs.fs
    names = sorted(inputs.signals)
    n = min(np.asarray(inputs.signals[s]).size for s in names)
    p = _Prepared(inputs=inputs, n=n, nerve=[s for s in names if _is_nerve(s)],
                  signals=names, env_rate=fs / _env_step(fs))
    eng, low = BANDS[ENG_BAND], BANDS[LOW_BAND]
    for s in names:
        x = np.asarray(inputs.signals[s], dtype=np.float64)[:n]
        if _is_nerve(s):
            p.eng[s] = _bandpass(x, fs, eng.lo_hz, eng.hi_hz)
            p.low[s] = _bandpass(x, fs, low.lo_hz, low.hi_hz)
            p.env[s] = _fine_envelope(p.eng[s], fs)
            with np.errstate(divide="ignore"):
                p.log_fine[s] = np.log(np.maximum(p.env[s], 1e-12))
        p.clip[s] = _flat_rail(x)
    p.pairs = sorted(inputs.z)
    p.zstack = np.vstack([np.asarray(inputs.z[q], np.float64) for q in p.pairs])
    rows = []
    for s in names:
        zs = p.zstack[[i for i, q in enumerate(p.pairs) if q[0] == s]]
        with np.errstate(all="ignore"):
            m = np.max(np.where(np.isfinite(zs), zs, -np.inf), axis=0)
        rows.append(np.where(np.isfinite(m), m, np.nan))
    p.zsig = np.vstack(rows)
    return p


# ---------------------------------------------------------------------------
# per-window statistics
# ---------------------------------------------------------------------------


def _nanstat(values: Iterable[float], how: str) -> float:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return math.nan
    if how == "max":
        return float(np.max(a))
    if how == "min":
        return float(np.min(a))
    if how == "median":
        return float(np.median(a))
    return float(np.mean(a))


def _power(y: F64) -> float:
    with np.errstate(invalid="ignore"):
        v = float(np.nanmean(y * y)) if y.size else math.nan
    return v


def _corr(a: F64, b: F64) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:  # noqa: PLR2004
        return math.nan
    a, b = a[ok], b[ok]
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0 or sb == 0:
        return math.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _spectral(x: F64, fs: float) -> tuple[float, float]:
    """(normalised spectral entropy, 95% spectral edge in Hz) of a raw window."""
    x = x[np.isfinite(x)]
    if x.size < 16:  # noqa: PLR2004
        return math.nan, math.nan
    x = x - x.mean()
    pw = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    pw = pw[1:]
    tot = float(pw.sum())
    if tot <= 0:
        return math.nan, math.nan
    q = pw / tot
    nz = q[q > 0]
    ent = float(-(nz * np.log(nz)).sum() / math.log(q.size)) if q.size > 1 else math.nan
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)[1:]
    edge = float(freqs[min(int(np.searchsorted(np.cumsum(q), SPECTRAL_EDGE)), freqs.size - 1)])
    return ent, edge


def _shape(x: F64, fs: float) -> tuple[float, float, float, float, float]:
    """(slew/RMS, excess kurtosis, line length/RMS, spectral entropy, spectral edge)."""
    x = x[np.isfinite(x)]
    if x.size < 8:  # noqa: PLR2004
        return (math.nan,) * 5
    xc = x - x.mean()
    rms = float(np.sqrt(np.mean(xc * xc)))
    if rms == 0:
        return (math.nan,) * 5
    d = np.abs(np.diff(xc))
    slew = float(d.max()) * fs / rms
    kurt = float(kurtosis(xc, fisher=True, bias=True))
    ll = float(d.sum()) / (rms * xc.size)
    ent, edge = _spectral(x, fs)
    return slew, kurt, ll, ent, edge


def _span(t0: float, t1: float, rate: float, n: int | None = None) -> slice:
    """Return the indices of ``[t0, t1)`` at ``rate``: at least one, clipped to ``[0, n)``."""
    a = int(math.floor(t0 * rate))
    b = max(int(math.ceil(t1 * rate)), a + 1)
    return slice(max(0, a), b if n is None else min(n, b))


def _samples(t0: float, t1: float, fs: float, n: int) -> slice:
    return _span(t0, t1, fs, n)


def _capped(t0: float, t1: float, cap: float) -> tuple[float, float]:
    if t1 - t0 <= cap:
        return t0, t1
    mid = 0.5 * (t0 + t1)
    return mid - cap / 2, mid + cap / 2


def _power_rel(a: float, b: float) -> float:
    return max(a, b) / (0.5 * (a + b))


def _log_ratio(a: float, b: float) -> float:
    return abs(math.log10(a / b))


def _spatial(p: _Prepared, sl: slice, fsl: slice, zwin: F64, out: dict[str, float]) -> None:
    """Band ratio, fractions over threshold, envelope agreement and relative power."""
    ratios, peng = [], []
    for s in p.nerve:
        pe, pl = _power(p.eng[s][sl]), _power(p.low[s][sl])
        peng.append(pe)
        ratios.append(math.log10(pl / pe) if pe > 0 and pl > 0 else math.nan)
    out["band_ratio_max"] = pair_max(ratios)
    out["band_ratio_median"] = _nanstat(ratios, "median")

    with np.errstate(invalid="ignore"):
        over_pair = np.nanmax(np.where(np.isfinite(zwin), zwin, -np.inf), axis=1) > p.inputs.z_enter
    sig_over: dict[str, float] = {}
    pair_frac: dict[str, list[float]] = {}
    for (sig, _b), o in zip(p.pairs, over_pair, strict=True):
        sig_over[sig] = max(sig_over.get(sig, 0.0), float(o))
        pair_frac.setdefault(sig, []).append(float(o))
    out["frac_signals_over"] = _over_groups(sig_over, "mean") if sig_over else math.nan
    out["frac_pairs_over"] = (_over_groups({s: float(np.mean(v)) for s, v in pair_frac.items()},
                                           "mean") if pair_frac else math.nan)

    envs = {s: p.log_fine[s][fsl] for s in p.nerve}
    corrs = [_corr(envs[a], envs[b]) for a, b in itertools.combinations(p.nerve, 2)]
    out["env_corr_mean"] = _nanstat(corrs, "mean")
    pos = [v for v in peng if np.isfinite(v) and v > 0]
    out["power_rel_max"] = _pair_mean(pos, _power_rel)
    out["power_rel_spread"] = _pair_mean(pos, _log_ratio)


def _level_rate(p: _Prepared, zf: slice, zwin: F64, peak_of: Mapping[str, float]) -> float:
    """Level-rate correlation of the group holding the event, count-independently.

    Per signal, the correlation of log-envelope level with its rate on that signal's
    own peak band; the feature is the median over the group whose pair-max peak z is
    the larger (the group the event is in). Taking the single peak pair of all signals
    instead made the choice of pair - and the value - depend on how many signals
    competed for it (measured: median 0.40 at 2 nerve signals, 0.11 at 8, +-500 ms).
    """
    with np.errstate(all="ignore"):
        row_peak = np.max(np.where(np.isfinite(zwin), zwin, -np.inf), axis=1)
    best: dict[str, int] = {}
    for i, (s, _b) in enumerate(p.pairs):
        if np.isfinite(row_peak[i]) and (s not in best or row_peak[i] > row_peak[best[s]]):
            best[s] = i
    corr: dict[str, list[float]] = {}
    for s, i in best.items():
        seg = np.asarray(p.inputs.log_env[p.pairs[i]], np.float64)[max(zf.start - 1, 0):zf.stop]
        c = _corr(seg[1:], np.diff(seg)) if seg.size >= 4 else math.nan  # noqa: PLR2004
        corr.setdefault(_group(s), []).append(c)
    def group_peak(g: str) -> float:
        v = pair_max([z for s, z in peak_of.items() if _group(s) == g])
        return v if np.isfinite(v) else -math.inf

    for g in sorted(corr, key=group_peak, reverse=True):
        v = _nanstat(corr[g], "median")
        if np.isfinite(v):
            return v
    return math.nan


def _rates_and_z(p: _Prepared, zf: slice, zwin: F64, out: dict[str, float]) -> None:
    """Onset and offset rates, level-rate coupling, peak z and mean frame z."""
    inp = p.inputs
    rmax: dict[str, list[float]] = {}
    rmin: dict[str, list[float]] = {}
    for q in p.pairs:
        le = np.asarray(inp.log_env[q], np.float64)
        seg = le[max(zf.start - 1, 0):zf.stop]
        if seg.size >= 2:  # noqa: PLR2004
            d = np.diff(seg) / GRID_S
            if np.isfinite(d).any():
                rmax.setdefault(q[0], []).append(float(np.nanmax(d)))
                rmin.setdefault(q[0], []).append(float(np.nanmin(d)))
    out["onset_rate_max"] = _over_groups({s: max(v) for s, v in rmax.items()}, "max")
    out["offset_rate_min"] = _over_groups({s: min(v) for s, v in rmin.items()}, "min")
    if zwin.size and np.isfinite(zwin).any():
        zs = p.zsig[:, zf]
        with np.errstate(all="ignore"):
            peaks = np.max(np.where(np.isfinite(zs), zs, -np.inf), axis=1)
        peak_of = {s: float(v) if np.isfinite(v) else math.nan
                   for s, v in zip(p.signals, peaks, strict=True)}
        out["peak_z"] = _over_groups(peak_of, "max")
        out["level_rate_corr"] = _level_rate(p, zf, zwin, peak_of)
        per_group = []
        for g in ("nerve", "stomach"):
            rows = [i for i, s in enumerate(p.signals) if _group(s) == g]
            if len(rows) >= 2:  # noqa: PLR2004
                per_group.append(pair_max_cols(zs[rows]))
        stacked = np.vstack(per_group) if per_group else np.full((1, zs.shape[1]), np.nan)
        frame = np.max(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0)
        frame = frame[np.isfinite(frame)]
        out["z_mean"] = float(frame.mean()) if frame.size else math.nan
    else:
        out["level_rate_corr"] = out["peak_z"] = out["z_mean"] = math.nan


def _context_features(p: _Prepared, t0: float, t1: float) -> dict[str, float]:
    fs = p.inputs.fs
    sl = _samples(t0, t1, fs, p.n)
    er = p.env_rate
    fsl = _span(t0, t1, er)
    zf = _span(t0, t1, 1.0 / GRID_S, p.zstack.shape[1])
    zwin = p.zstack[:, zf]
    out: dict[str, float] = {}
    _spatial(p, sl, fsl, zwin, out)
    _rates_and_z(p, zf, zwin, out)

    slopes, shapes = [], []
    s0, s1 = _capped(t0, t1, SHAPE_MAX_S)
    ssl = _samples(s0, s1, fs, p.n)
    for s in p.nerve:
        e = p.env[s][fsl]
        if e.size >= 3 and np.isfinite(e).sum() >= 3 and np.nanmean(e) > 0:  # noqa: PLR2004
            slopes.append(float(np.nanmax(np.abs(np.diff(e)))) * er / float(np.nanmean(e)))
        shapes.append(_shape(np.asarray(p.inputs.signals[s], np.float64)[ssl], fs))
    out["env_slope_max"] = pair_max(slopes)
    cols: list[tuple[float, ...]] = (list(zip(*shapes, strict=True)) if shapes
                                     else [(), (), (), (), ()])
    out["slew_max"], out["slew_median"] = pair_max(cols[0]), _nanstat(cols[0], "median")
    out["kurtosis_max"] = pair_max(cols[1])
    out["kurtosis_median"] = _nanstat(cols[1], "median")
    out["line_length_max"] = pair_max(cols[2])
    out["line_length_median"] = _nanstat(cols[2], "median")
    out["spec_entropy_median"] = _nanstat(cols[3], "median")
    out["spec_edge_median"] = _nanstat(cols[4], "median")

    out["clip_frac"] = _over_groups({s: float(np.mean(m[sl])) if m[sl].size else math.nan
                                     for s, m in p.clip.items()}, "max")
    return out


def _cm_fraction(a: F64, b: F64) -> float:
    """Common-mode fraction of one pair: power of the pair mean over mean power."""
    mean_p = float(np.mean(a * a) + np.mean(b * b)) / 2
    cm = 0.5 * (a + b)
    return float(np.mean(cm * cm)) / mean_p if mean_p > 0 else math.nan


def _line_features(p: _Prepared, t0: float, t1: float) -> dict[str, float]:
    fs = p.inputs.fs
    mid = 0.5 * (t0 + t1)
    half = min(max(t1 - t0, LINE_WINDOW_S), LINE_MAX_S) / 2
    sl = _samples(mid - half, mid + half, fs, p.n)
    er = p.env_rate
    fsl = _span(mid - half, mid + half, er)
    ratios, plvs = [], []
    for s in p.nerve:
        x = np.asarray(p.inputs.signals[s], np.float64)[sl]
        x = x[np.isfinite(x)]
        if x.size >= int(0.5 * fs):
            x = x - x.mean()
            pw = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            f = np.fft.rfftfreq(x.size, 1.0 / fs)
            band = (f >= 1.0) & (f <= BANDS[ENG_BAND].hi_hz)
            harm = np.zeros(f.size, dtype=bool)
            for k in range(1, int(BANDS[ENG_BAND].hi_hz // MAINS_HZ) + 1):
                harm |= np.abs(f - k * MAINS_HZ) <= LINE_HALF_BIN_HZ
            tot = float(pw[band].sum())
            ratios.append(float(pw[band & harm].sum()) / tot if tot > 0 else math.nan)
        e = p.env[s][fsl]
        if e.size >= int(0.5 * er) and np.isfinite(e).all() and e.sum() > 0:
            t = (np.arange(e.size) + fsl.start + 0.5) / er
            ec = e - e.mean()
            den = float(np.abs(ec).sum())
            lock = float(np.abs((ec * np.exp(2j * np.pi * MAINS_HZ * t)).sum()))
            plvs.append(lock / den if den > 0 else math.nan)
    cms = [_cm_fraction(p.eng[a][sl], p.eng[b][sl])
           for a, b in itertools.combinations(p.nerve, 2)
           if np.isfinite(p.eng[a][sl]).all() and np.isfinite(p.eng[b][sl]).all()]
    return {"line_ratio_max": pair_max(ratios), "line_plv_max": pair_max(plvs),
            "cm_fraction": _nanstat(cms, "mean")}


def core_features(p: _Prepared, start_s: float, stop_s: float,
                  pairs: Sequence[Pair] = ()) -> dict[str, float]:
    """Every feature of one core, ``[start_s, stop_s)`` on the region's timeline.

    ``pairs`` are the core's over-threshold pairs (``Core.pairs``), for the slow share;
    only pairs of common-set signals are counted, and with none it is ``nan``.
    Returns a dict keyed exactly by :data:`FEATURE_NAMES`.
    """
    if not stop_s > start_s:
        msg = f"a core must have positive duration, got [{start_s}, {stop_s})"
        raise ValueError(msg)
    common = [b for s, b in pairs if s in p.inputs.signals]
    row: dict[str, float] = {
        "duration_s": float(stop_s - start_s),
        "slow_pair_share": float(np.mean([b in SLOW_BANDS for b in common])) if common
        else math.nan,
    }
    row.update(_line_features(p, start_s, stop_s))
    for ctx in CONTEXTS_S:
        for k, v in _context_features(p, start_s - ctx, stop_s + ctx).items():
            row[f"{k}{_sfx(ctx)}"] = v
    return {k: row[k] for k in FEATURE_NAMES}


def feature_matrix(p: _Prepared,
                   cores: Iterable[tuple[float, float, Sequence[Pair]]]) -> pd.DataFrame:
    """One row per core, columns :data:`FEATURE_NAMES` (``nan`` where an input is absent).

    ``cores`` yields ``(start_s, stop_s, pairs)`` on the region's timeline.
    """
    rows = [core_features(p, a, b, q) for a, b, q in cores]
    return pd.DataFrame(rows, columns=list(FEATURE_NAMES), dtype=np.float64)


def missing_rates(df: pd.DataFrame) -> dict[str, float]:
    """Per-feature fraction of ``nan`` - task 11's acceptance asks for it."""
    return {c: float(df[c].isna().mean()) for c in df.columns}
