"""Settling of HR_BR_HRVAnalysis_beats' heart-band filter at a NaN edge (read only; invariant 19).

HR_BR_HRVAnalysis_beats.m:262 and :278-282: xFill = fillmissing(x, 'linear', 'EndValues',
'nearest'); butter(order, [10 150]/(fs/2), 'bandpass') -> zp2sos -> filtfilt(sos, g, xFill);
heartBeatSeries(invalidMask) = NaN (:302-303). The filter runs at the fs the caller passes.

Night 6 (matlab/night6/night6_run_recording.m:429) passes ``fs = plan.fs`` - the mask file's
fs, held equal to the recording's (night6_prepare_epoch.m:78, :93), i.e. the FULL rate
24414.0625 Hz, and ``H.order`` = 4 (night6_run_recording.m:471). So this measurement runs at
FS = 24414.0625 Hz, order 4, band [10 150] Hz. The first measurement (hr_edge_settling.json of
13:25, 2026-10-09) ran at fs/12 = 2034.5 Hz, which is not what Night 6 passes; it is kept as
hr_edge_settling_fs12.json and reproduced here with ``--fs12`` for comparison.

Method: edgepad/mmc_edge_settling.py's. Settling = distance from the gap until |error| stays
below 1% of its peak (task 13's tolerance). error = filtered(with gap, filled) -
filtered(without gap), outside the gap. Worst case over input, gap length and side. scipy's
sosfiltfilt stands in for MATLAB filtfilt(sos, g): they differ only at the array ends.

Usage: python hr_edge_settling.py [--fs12]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS_FULL = 24414.0625
FS = FS_FULL / 12.0 if "--fs12" in sys.argv else FS_FULL
ORDER = 4
BAND_HZ = (10.0, 150.0)
SOS = butter(ORDER, [BAND_HZ[0] / (FS / 2), min(BAND_HZ[1], FS / 2 - 1) / (FS / 2)],
             btype="bandpass", output="sos")
N = int(60 * FS)
MID = N // 2
TOL = 0.01
GAPS_S = (0.025, 0.1, 1.0, 10.0)
SEED = 20261009


def fill_linear(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    bad = ~np.isfinite(y)
    i = np.arange(y.size)
    y[bad] = np.interp(i[bad], i[~bad], y[~bad])
    return y


DEGENERATE_REL = 1e-6
PEAK_REL = [0.0]  # the last call's error peak relative to the reference output's peak


def settling(x: np.ndarray, gap_s: float) -> tuple[float, float]:
    g = int(gap_s * FS)
    a, b = MID - g // 2, MID - g // 2 + g
    ref = sosfiltfilt(SOS, x)
    xg = x.copy()
    xg[a:b] = np.nan
    err = sosfiltfilt(SOS, fill_linear(xg)) - ref
    err[a:b] = 0.0
    pk = np.abs(err).max()
    PEAK_REL[0] = float(pk / max(np.abs(ref).max(), 1e-300))
    out = []
    for side in ("before", "after"):
        e = np.abs(err[:a][::-1]) if side == "before" else np.abs(err[b:])
        over = np.flatnonzero(e > TOL * pk)
        out.append(0.0 if over.size == 0 else (over[-1] + 1) / FS)
    return out[0], out[1]


QRS_SIGMA_S = (0.002, 0.004, 0.008)
QRS_OFFSETS_S = (-0.005, 0.0, 0.005, 0.010, 0.020, 0.050)


def qrs_like(t_c: float, t: np.ndarray, s: float) -> np.ndarray:
    """A QRS-like complex: Ricker wavelet of width ``s`` (s), centred at ``t_c`` (s)."""
    u = (t - t_c) / s
    return (1.0 - u * u) * np.exp(-0.5 * u * u)


def main() -> None:
    rng = np.random.default_rng(SEED)
    t = np.arange(N) / FS
    t_mid = MID / FS
    inputs = {
        "step": np.where(t < t_mid, 0.0, 1.0),
        "noise_white": rng.standard_normal(N),
        "sine_20hz": np.sin(2 * np.pi * 20 * t),
        "sine_100hz": np.sin(2 * np.pi * 100 * t),
    }
    res = {}
    rel: dict[str, float] = {}
    for name, x in inputs.items():
        for gap in GAPS_S:
            res[f"{name}|gap{gap}s"] = settling(x, gap)
            rel[f"{name}|gap{gap}s"] = PEAK_REL[0]
    # a QRS at the edge: a Ricker complex (sigma 2, 4, 8 ms) centred QRS_OFFSETS_S before the
    # START of each gap (negative = inside the gap) - the worst placement is searched, not assumed
    for sig in QRS_SIGMA_S:
        for off in QRS_OFFSETS_S:
            for gap in GAPS_S:
                g = int(gap * FS)
                a = MID - g // 2
                x = qrs_like(a / FS - off, t, sig)
                k = f"qrs_s{sig * 1e3:g}ms_off{off * 1e3:g}ms|gap{gap}s"
                res[k] = settling(x, gap)
                rel[k] = PEAK_REL[0]
    # a row whose error peak is below DEGENERATE_REL of the output's peak removed nothing (the
    # QRS lies wholly outside the gap): its "settling" is the 1 % point of rounding noise, so it
    # is reported but excluded from the worst case
    ok = {k: v for k, v in res.items() if rel[k] >= DEGENERATE_REL}
    degenerate = sorted(k for k in res if k not in ok)
    worst_all = max(max(v) for v in res.values())
    res_all = res
    res = ok
    worst = max(max(v) for v in res.values())
    worst_before = max(v[0] for v in res.values())
    worst_after = max(v[1] for v in res.values())
    out = {"filter": f"butter({ORDER},[{BAND_HZ[0]:g} {BAND_HZ[1]:g}]) bandpass SOS filtfilt, "
                     "linear fill (HR_BR_HRVAnalysis_beats.m:262, :278-282)",
           "fs": FS, "order": ORDER, "band_hz": list(BAND_HZ),
           "fs_basis": ("full rate: night6_run_recording.m:429 passes fs = plan.fs, the mask "
                        "file's fs (night6_prepare_epoch.m:78, :93)") if FS == FS_FULL else
                       "fs/12: NOT what Night 6 passes; comparison only",
           "tolerance": "1% of peak error", "seed": SEED, "gaps_s": list(GAPS_S),
           "qrs_sigma_s": list(QRS_SIGMA_S), "qrs_offsets_before_gap_start_s": list(QRS_OFFSETS_S),
           "settling_s": {k: [round(a, 4), round(b, 4)] for k, (a, b) in res_all.items()},
           "degenerate_rel_threshold": DEGENERATE_REL,
           "degenerate_rows_excluded": degenerate,
           "worst_including_degenerate_s": round(worst_all, 4),
           "error_peak_rel_to_output_peak": {k: float(f"{v:.3g}") for k, v in rel.items()},
           "worst_before_gap_s": round(worst_before, 4),
           "worst_after_gap_s": round(worst_after, 4),
           "worst_s": round(worst, 4)}
    name = "hr_edge_settling.json" if FS == FS_FULL else "hr_edge_settling_fs12_rerun.json"
    p = Path(__file__).with_name(name)
    tmp = p.with_name(f".{name}.tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(p)
    for k, v in res_all.items():
        print(f"{k:28s} before {v[0]:.4f} s  after {v[1]:.4f} s")
    print("FS", FS, "WORST", round(worst, 4), "before", round(worst_before, 4), "after",
          round(worst_after, 4))


if __name__ == "__main__":
    main()
