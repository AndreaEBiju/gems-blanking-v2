"""Settling of extract_mmc's conditioning filter at a NaN edge (read only; Andrea's item 5(a)).

extract_mmc.m:89-107: butter(4,[2 50]/(fs/2),'bandpass') -> zp2sos -> fillmissing(linear,
EndValues nearest) across NaN -> filtfilt(sos,g) -> NaN restored. The moving median and MAD
that follow (:231-232) use 'omitnan', so blanked samples never enter them; only this
filter, run on filled data, reaches past a blank edge.

Settling = distance from the gap until |error| stays below 1% of its peak (task 13's
tolerance, the same rule as edgepad/analyse_settling.py). error = filtered(with gap,
filled) - filtered(without gap), outside the gap. Worst case over input type, gap length
and side. scipy's sosfiltfilt stands in for MATLAB filtfilt(sos,g): the two differ only at
the array ends (padding), and the gaps here sit far from the ends.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 24414.0625
SOS = butter(4, [2 / (FS / 2), 50 / (FS / 2)], btype="bandpass", output="sos")
N = int(60 * FS)
MID = N // 2
TOL = 0.01


def fill_linear(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    bad = ~np.isfinite(y)
    i = np.arange(y.size)
    y[bad] = np.interp(i[bad], i[~bad], y[~bad])
    return y


def settling(x: np.ndarray, gap_s: float) -> tuple[float, float]:
    g = int(gap_s * FS)
    a, b = MID - g // 2, MID - g // 2 + g
    ref = sosfiltfilt(SOS, x)
    xg = x.copy()
    xg[a:b] = np.nan
    err = sosfiltfilt(SOS, fill_linear(xg)) - ref
    err[a:b] = 0.0
    pk = np.abs(err).max()
    out = []
    for side in ("before", "after"):
        e = np.abs(err[:a][::-1]) if side == "before" else np.abs(err[b:])
        over = np.flatnonzero(e > TOL * pk)
        out.append(0.0 if over.size == 0 else (over[-1] + 1) / FS)
    return out[0], out[1]


def main() -> None:
    rng = np.random.default_rng(20261009)
    t = np.arange(N) / FS
    inputs = {
        "step": np.where(t < t[MID], 0.0, 1.0),
        "noise_white": rng.standard_normal(N),
        "sine_5hz": np.sin(2 * np.pi * 5 * t),
        "sine_20hz": np.sin(2 * np.pi * 20 * t),
        "burst_at_edge": np.zeros(N),
    }
    inputs["burst_at_edge"][MID - int(0.05 * FS):MID] = rng.standard_normal(int(0.05 * FS)) * 10
    res = {}
    for name, x in inputs.items():
        for gap in (0.025, 0.1, 1.0, 10.0):
            res[f"{name}|gap{gap}s"] = settling(x, gap)
    worst = max(max(v) for v in res.values())
    out = {"tolerance": "1% of peak error", "fs": FS, "filter": "butter(4,[2 50]) bandpass, SOS, filtfilt, linear fill",
           "settling_s": {k: [round(a, 4), round(b, 4)] for k, (a, b) in res.items()},
           "worst_s": round(worst, 4)}
    Path(__file__).with_name("mmc_edge_settling.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    for k, v in res.items():
        print(f"{k:28s} before {v[0]:.3f} s  after {v[1]:.3f} s")
    print("WORST", worst)


if __name__ == "__main__":
    main()
