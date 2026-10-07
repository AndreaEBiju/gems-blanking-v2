"""Task 11: one feature row per candidate **core** (ruling 2026-10-07 (b) R3).

The classifier judges cores, not candidates: a core is labelled motion only when a mark
covers at least half of it, so long candidates are not learned as wholly motion. Every
feature here is computed on the same region, the same 10 ms grid and the same detection
signals that :func:`~gems_blanking_v2.detect.chain.detect_region` used to propose the
core (:func:`region_inputs`, with a test that the z traces are identical).

Hard constraints (CLAUDE.md invariant 10):

* **Channel-count independent.** Every per-signal statistic is reduced across signals
  (max, median, fraction). There is never one column per channel, so one model serves
  the 5-channel old cohort and the 9-channel new cohort.
* **Animal-invariant.** Ratios, correlations, rates and robust z only. Absolute
  microvolts never enter a feature; scaling every channel by 10x changes nothing
  (tested). There are no explicit amplitude features.
* **Missing is missing (ruling (b) R2).** Where a cohort lacks the input - common-mode
  and within-cuff features on the old hardware tripole, which has no raw contacts -
  the feature is ``nan``, never imputed. LightGBM handles ``nan`` natively.

The ENG band is 300-3000 Hz (ruling (b) R8, A.5b); the motion ratio is
``100-300 / 300-3000``.

Families, each at ``CONTEXTS_S`` (the core itself, then +-100, 250, 500 ms):

* band ratio ``log10(P[100-300] / P[300-3000])`` per nerve signal: max, median;
* spatial: fraction of signals and of (signal, band) pairs over ``z_enter``; mean
  pairwise 300-3000 log-envelope correlation; each nerve signal's 300-3000 power over
  the median across them (max, log spread); common-mode over residual power per cuff
  (max; raw contacts only); within-cuff minus across-cuff envelope agreement;
* onset rate: the derivative of the **log** envelope (per second), max and min over
  every pair, and the correlation of level with rate on the peak pair (brief events
  couple them, sustained ones decouple);
* shape on the broadband detection signal: 300-3000 envelope slope over its mean,
  slew over RMS, excess kurtosis, line length over RMS, spectral entropy and 95%
  spectral edge (max and median over nerve signals);
* clipping fraction: share of samples on a flat run at the channel's rail;
* z summary: peak z, mean over frames of the max-over-pairs z.

Once per core (stationary, so not per context; ruling 2026-10-07 (c) item 2): the mains
line ratio, the mains phase-locking value of the 300-3000 envelope, and the common-mode
fraction across nerve signals - so the classifier can tell stationary mains-locked
energy from transient motion. Plus the core's duration and its slow-band share.

Outside the generation hash: :mod:`~gems_blanking_v2.detect.chain` does not import this
module, so building or changing features never changes what was proposed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
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
    "FEATURE_NAMES",
    "MAINS_HZ",
    "RegionInputs",
    "core_features",
    "feature_matrix",
    "inputs_from_signals",
    "missing_rates",
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


def _sfx(ctx: float) -> str:
    return f"_c{round(ctx * 1000)}"


_PER_CONTEXT: Final[tuple[str, ...]] = (
    "band_ratio_max", "band_ratio_median",
    "frac_signals_over", "frac_pairs_over",
    "env_corr_mean", "power_rel_max", "power_rel_spread",
    "cm_resid_max", "within_minus_across",
    "onset_rate_max", "offset_rate_min", "level_rate_corr",
    "env_slope_max", "slew_max", "slew_median", "kurtosis_max", "kurtosis_median",
    "line_length_max", "line_length_median", "spec_entropy_median", "spec_edge_median",
    "clip_frac",
    "peak_z", "z_mean",
)
_ONCE: Final[tuple[str, ...]] = (
    "duration_s", "slow_pair_share", "line_ratio_max", "line_plv_max", "cm_fraction",
)

FEATURE_NAMES: Final[tuple[str, ...]] = _ONCE + tuple(
    f"{name}{_sfx(c)}" for c in CONTEXTS_S for name in _PER_CONTEXT)
"""Every feature column, in order. The same for every recording and every cohort."""


# ---------------------------------------------------------------------------
# region inputs: exactly what detect_region read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionInputs:
    """The detection inputs of one region, on the region's own timeline.

    ``signals`` are the derived cuff signals plus the raw stomach channels - the
    detection set of :func:`~gems_blanking_v2.detect.chain.detect_region`.
    ``contacts`` maps cuff id to its raw contacts (``{}`` for a hardware tripole).
    ``z`` and ``log_env`` are on the 10 ms grid, frame 0 at the region start.
    """

    fs: float
    signals: Mapping[str, F64]
    contacts: Mapping[str, Mapping[str, F64]]
    z: Mapping[Pair, F64]
    log_env: Mapping[Pair, F64]
    z_enter: float


def region_inputs(rec: Recording, region: tuple[float, float], *, z_enter: float) -> RegionInputs:
    """Build the detection inputs of ``region`` exactly as ``detect_region`` does.

    Mirrors the chain's assembly (derivations plus raw stomach channels, each signal
    against its own whole-region log-envelope reference) without importing into the
    generation hash. ``tests/test_features.py`` asserts the z traces are identical.
    """
    lo, hi = region
    i0, i1 = round(lo * rec.fs), round(hi * rec.fs)
    sub = replace(rec, data=rec.data[i0:i1])
    fs = float(rec.fs)
    signals, _w = build_derivations(sub)
    raw_stomach = {c.name: np.asarray(sub.data[:, c.index], dtype=np.float64)
                   for c in sub.channels if c.role == "stomach"}
    detection = {**signals, **raw_stomach}
    contacts: dict[str, dict[str, F64]] = {}
    for c in sub.channels:
        if c.cuff_id is not None and c.contact_index is not None and c.config == "independent":
            contacts.setdefault(c.cuff_id, {})[f"V{c.contact_index}"] = np.asarray(
                sub.data[:, c.index], dtype=np.float64)
    return inputs_from_signals(detection, fs, contacts, z_enter=z_enter)


def inputs_from_signals(signals: Mapping[str, F64], fs: float,
                        contacts: Mapping[str, Mapping[str, F64]], *,
                        z_enter: float) -> RegionInputs:
    """Build :class:`RegionInputs` from detection signals: z and log envelopes per pair.

    Each signal against its own whole-region log-envelope reference (invariants 3, 5),
    exactly as ``chain.z_by_pair``. The one construction site for features' z.
    """
    z: dict[Pair, F64] = {}
    log_env: dict[Pair, F64] = {}
    for name in sorted(signals):
        x = np.asarray(signals[name], dtype=np.float64)
        for band in BANDS:
            le = log_envelope(band_envelope_for(x, fs, band))
            ref = epoch_reference(le, signal=name, band=band)
            log_env[(name, band)] = le
            z[(name, band)] = zscore(le, ref, signal=name, band=band)
    return RegionInputs(fs=float(fs), signals=dict(signals), contacts=contacts, z=z,
                        log_env=log_env, z_enter=float(z_enter))


# ---------------------------------------------------------------------------
# precomputation
# ---------------------------------------------------------------------------


def _is_nerve(name: str) -> bool:
    return name[:2] in ("L_", "R_")


def _cuff(name: str) -> str:
    return name[0]


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
    env_rate: float
    eng: dict[str, F64] = field(default_factory=dict)
    low: dict[str, F64] = field(default_factory=dict)
    env: dict[str, F64] = field(default_factory=dict)
    log_fine: dict[str, F64] = field(default_factory=dict)
    cm_eng: dict[str, tuple[F64, F64]] = field(default_factory=dict)
    clip: dict[str, Bool] = field(default_factory=dict)
    zstack: F64 = field(default_factory=lambda: np.zeros((0, 0)))
    pairs: list[Pair] = field(default_factory=list)


def prepare(inputs: RegionInputs) -> _Prepared:
    """Precompute the band-limited signals, fine envelopes and rail masks of a region."""
    fs = inputs.fs
    names = sorted(inputs.signals)
    n = min(np.asarray(inputs.signals[s]).size for s in names)
    p = _Prepared(inputs=inputs, n=n, nerve=[s for s in names if _is_nerve(s)],
                  env_rate=fs / _env_step(fs))
    eng, low = BANDS[ENG_BAND], BANDS[LOW_BAND]
    for s in names:
        x = np.asarray(inputs.signals[s], dtype=np.float64)[:n]
        p.eng[s] = _bandpass(x, fs, eng.lo_hz, eng.hi_hz)
        p.low[s] = _bandpass(x, fs, low.lo_hz, low.hi_hz)
        p.env[s] = _fine_envelope(p.eng[s], fs)
        with np.errstate(divide="ignore"):
            p.log_fine[s] = np.log(np.maximum(p.env[s], 1e-12))
        p.clip[s] = _flat_rail(x)
    for cuff, cts in inputs.contacts.items():
        if len(cts) < 2:  # noqa: PLR2004 - a common mode needs two contacts
            continue
        stack = np.vstack([_bandpass(np.asarray(v, np.float64)[:n], fs, eng.lo_hz, eng.hi_hz)
                           for _k, v in sorted(cts.items())])
        cm = np.nanmean(stack, axis=0)
        p.cm_eng[cuff] = (cm, stack - cm)
        for k, v in cts.items():
            p.clip[f"{cuff}:{k}"] = _flat_rail(np.asarray(v, np.float64)[:n])
    p.pairs = sorted(inputs.z)
    p.zstack = np.vstack([np.asarray(inputs.z[q], np.float64) for q in p.pairs])
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


def _context_features(p: _Prepared, t0: float, t1: float) -> dict[str, float]:  # noqa: PLR0915
    fs, inp = p.inputs.fs, p.inputs
    sl = _samples(t0, t1, fs, p.n)
    er = p.env_rate
    fsl = _span(t0, t1, er)
    nframes = p.zstack.shape[1]
    zf = _span(t0, t1, 1.0 / GRID_S, nframes)
    out: dict[str, float] = {}

    ratios, peng = [], {}
    for s in p.nerve:
        pe, pl = _power(p.eng[s][sl]), _power(p.low[s][sl])
        peng[s] = pe
        ratios.append(math.log10(pl / pe) if pe > 0 and pl > 0 else math.nan)
    out["band_ratio_max"] = _nanstat(ratios, "max")
    out["band_ratio_median"] = _nanstat(ratios, "median")

    zwin = p.zstack[:, zf]
    with np.errstate(invalid="ignore"):
        over_pair = np.nanmax(np.where(np.isfinite(zwin), zwin, -np.inf), axis=1) > inp.z_enter
    sig_over: dict[str, bool] = {}
    for (sig, _b), o in zip(p.pairs, over_pair, strict=True):
        sig_over[sig] = sig_over.get(sig, False) or bool(o)
    out["frac_signals_over"] = float(np.mean(list(sig_over.values()))) if sig_over else math.nan
    out["frac_pairs_over"] = float(np.mean(over_pair)) if over_pair.size else math.nan

    envs = {s: p.log_fine[s][fsl] for s in p.nerve}
    corrs = [_corr(envs[a], envs[b]) for a, b in itertools.combinations(p.nerve, 2)]
    out["env_corr_mean"] = _nanstat(corrs, "mean")
    pos = [v for v in peng.values() if np.isfinite(v) and v > 0]
    if len(pos) >= 2:  # noqa: PLR2004
        med = float(np.median(pos))
        out["power_rel_max"] = float(max(pos) / med)
        out["power_rel_spread"] = float(math.log10(max(pos) / min(pos)))
    else:
        out["power_rel_max"] = out["power_rel_spread"] = math.nan
    cmr = []
    for cm, resid in p.cm_eng.values():
        pc, pr = _power(cm[sl]), _power(resid[:, sl])
        cmr.append(math.log10(pc / pr) if pc > 0 and pr > 0 else math.nan)
    out["cm_resid_max"] = _nanstat(cmr, "max")
    combos = list(itertools.combinations(p.nerve, 2))
    within = [_corr(envs[a], envs[b]) for a, b in combos if _cuff(a) == _cuff(b)]
    across = [_corr(envs[a], envs[b]) for a, b in combos if _cuff(a) != _cuff(b)]
    w, a_ = _nanstat(within, "mean"), _nanstat(across, "mean")
    out["within_minus_across"] = w - a_ if np.isfinite(w) and np.isfinite(a_) else math.nan

    rates_max, rates_min = [], []
    for q in p.pairs:
        le = np.asarray(inp.log_env[q], np.float64)
        lo_f, hi_f = max(zf.start - 1, 0), zf.stop
        seg = le[lo_f:hi_f]
        if seg.size >= 2:  # noqa: PLR2004
            d = np.diff(seg) / GRID_S
            if np.isfinite(d).any():
                rates_max.append(float(np.nanmax(d)))
                rates_min.append(float(np.nanmin(d)))
    out["onset_rate_max"] = _nanstat(rates_max, "max")
    out["offset_rate_min"] = _nanstat(rates_min, "min")
    if zwin.size and np.isfinite(zwin).any():
        peak_row = int(np.nanargmax(np.nanmax(np.where(np.isfinite(zwin), zwin, -np.inf), axis=1)))
        le = np.asarray(inp.log_env[p.pairs[peak_row]], np.float64)
        lo_f = max(zf.start - 1, 0)
        seg = le[lo_f:zf.stop]
        out["level_rate_corr"] = _corr(seg[1:], np.diff(seg)) if seg.size >= 4 else math.nan  # noqa: PLR2004
        out["peak_z"] = float(np.nanmax(zwin))
        with np.errstate(all="ignore"):
            colmax = np.nanmax(np.where(np.isfinite(zwin), zwin, -np.inf), axis=0)
        colmax = colmax[np.isfinite(colmax)]
        out["z_mean"] = float(colmax.mean()) if colmax.size else math.nan
    else:
        out["level_rate_corr"] = out["peak_z"] = out["z_mean"] = math.nan

    slopes, shapes = [], []
    s0, s1 = _capped(t0, t1, SHAPE_MAX_S)
    ssl = _samples(s0, s1, fs, p.n)
    for s in p.nerve:
        e = p.env[s][fsl]
        if e.size >= 3 and np.isfinite(e).sum() >= 3 and np.nanmean(e) > 0:  # noqa: PLR2004
            slopes.append(float(np.nanmax(np.abs(np.diff(e)))) * er / float(np.nanmean(e)))
        shapes.append(_shape(np.asarray(inp.signals[s], np.float64)[ssl], fs))
    out["env_slope_max"] = _nanstat(slopes, "max")
    cols: list[tuple[float, ...]] = (list(zip(*shapes, strict=True)) if shapes
                                     else [(), (), (), (), ()])
    out["slew_max"], out["slew_median"] = _nanstat(cols[0], "max"), _nanstat(cols[0], "median")
    out["kurtosis_max"] = _nanstat(cols[1], "max")
    out["kurtosis_median"] = _nanstat(cols[1], "median")
    out["line_length_max"] = _nanstat(cols[2], "max")
    out["line_length_median"] = _nanstat(cols[2], "median")
    out["spec_entropy_median"] = _nanstat(cols[3], "median")
    out["spec_edge_median"] = _nanstat(cols[4], "median")

    clips = [float(np.mean(m[sl])) for m in p.clip.values() if m[sl].size]
    out["clip_frac"] = _nanstat(clips, "max")
    return out


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
    cmf = math.nan
    if len(p.nerve) >= 2:  # noqa: PLR2004
        stack = np.vstack([p.eng[s][sl] for s in p.nerve])
        if np.isfinite(stack).all():
            mean_p = float(np.mean(stack * stack))
            cm = stack.mean(axis=0)
            cmf = float(np.mean(cm * cm)) / mean_p if mean_p > 0 else math.nan
    return {"line_ratio_max": _nanstat(ratios, "max"), "line_plv_max": _nanstat(plvs, "max"),
            "cm_fraction": cmf}


def core_features(p: _Prepared, start_s: float, stop_s: float,
                  pairs: Sequence[Pair] = ()) -> dict[str, float]:
    """Every feature of one core, ``[start_s, stop_s)`` on the region's timeline.

    ``pairs`` are the core's over-threshold pairs (``Core.pairs``), for the slow share.
    Returns a dict keyed exactly by :data:`FEATURE_NAMES`.
    """
    if not stop_s > start_s:
        msg = f"a core must have positive duration, got [{start_s}, {stop_s})"
        raise ValueError(msg)
    row: dict[str, float] = {
        "duration_s": float(stop_s - start_s),
        "slow_pair_share": (float(np.mean([b in SLOW_BANDS for _s, b in pairs]))
                            if pairs else math.nan),
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
