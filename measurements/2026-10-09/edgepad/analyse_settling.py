"""Edge-pad measurement, step 2: settling of step1_bandpass at a NaN edge, per input and side.

Settling = the task-13 instrument (gems_blanking_v2/extent/tolerance.py:240-245, IMPULSE_DECAY_FRACTION
= 0.01 from bands/envelope.py:132): the last distance from the edge at which the response still
exceeds 1% of its peak. Here the "response" is the edge error on VALID samples, and the distance d
counts samples from the NaN core (the valid sample adjacent to the gap is d = 1), so step2's pad of
p samples (movmax [p p], step2_noise_sigma.m:41, 77) removes exactly d <= p: the pad needed is S.

  deterministic inputs (step, spike, bump, biphasic, sub-band kink): one gap per column;
      e = Y - 0 ("total": what detection sees; background is zero) and e = Y - Yref ("gap-induced");
      tol = 1% of max |e| over both sides' valid samples.
  noise and real snippets: many gaps; e = Y - Yref (Yref = her step1 on the same data without
      gaps); per side, the RMS of e over edges at each d; tol = 1% of that envelope's max over both
      sides. Also sigma-relative (envelope / std(Yref)) and per-edge (each edge's own peak).
Writes edgepad/settling.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
FS = 24414.0625
TOL = 0.01
DMAX = int(round(0.060 * FS))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate(([False], mask.astype(bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def side_windows(gap: np.ndarray) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
    """Per gap: (trailing idx ordered d=1.., leading idx ordered d=1..), clipped half-way to neighbours."""
    rs = runs(gap)
    n = gap.size
    out = []
    for k, (a, b) in enumerate(rs):
        prev_end = rs[k - 1][1] if k else 0
        next_start = rs[k + 1][0] if k + 1 < len(rs) else n
        lt = min(DMAX, (a - prev_end) // 2 if k else a)
        ll = min(DMAX, (next_start - b) // 2 if k + 1 < len(rs) else n - b)
        trail = np.arange(a - 1, a - 1 - lt, -1) if lt > 0 else None
        lead = np.arange(b, b + ll) if ll > 0 else None
        out.append((trail, lead))
    return out


def last_above(v: np.ndarray, tol: float) -> int:
    """Largest d (1-based) with |v[d-1]| > tol; 0 if none."""
    idx = np.flatnonzero(np.abs(v) > tol)
    return int(idx[-1] + 1) if idx.size else 0


def ms(d: int) -> float:
    return round(d / FS * 1e3, 3)


def ceil_half(x_ms: float) -> float:
    return math.ceil(x_ms * 2 - 1e-9) / 2


def deterministic() -> dict:
    m = loadmat(HERE / "out_syn.mat")
    names = [str(np.asarray(x).squeeze()) for x in m["names"].ravel()]
    Y, Yref, Yraw, GAP = m["Y"], m["Yref"], m["Yraw"], m["GAP"].astype(bool)
    res = {}
    for k, nm in enumerate(names):
        (tr, ld), = side_windows(GAP[:, k])
        row = {}
        for kind, e in (("total", Y[:, k]), ("gap_induced", Y[:, k] - Yref[:, k])):
            if nm.startswith("kink") and kind == "total":
                continue  # a sub-band sinusoid's own output is not an edge effect
            et, el = e[tr], e[ld]
            peak = float(max(np.max(np.abs(et)), np.max(np.abs(el))))
            if peak == 0:
                continue
            r = {"peak_valid": peak,
                 "trail_ms": ms(last_above(et, TOL * peak)),
                 "lead_ms": ms(last_above(el, TOL * peak)),
                 "trail_ms_at_0.1pct": ms(last_above(et, 1e-3 * peak)),
                 "lead_ms_at_0.1pct": ms(last_above(el, 1e-3 * peak))}
            if kind == "gap_induced":
                inside = (Yraw[:, k] - Yref[:, k])[GAP[:, k]]
                r["peak_inside_gap"] = float(np.max(np.abs(inside))) if inside.size else 0.0
            row[kind] = r
        res[nm] = row
    return res


def stochastic(file: str, cols: list[str]) -> dict:
    m = loadmat(HERE / file)
    Y, Yref, GAP = m["Y"], m["Yref"], m["GAP"].astype(bool)
    res = {}
    for k, nm in enumerate(cols):
        e = Y[:, k] - Yref[:, k]
        sig = float(np.std(Yref[~GAP[:, k], k]))
        T, L = [], []
        for tr, ld in side_windows(GAP[:, k]):
            T.append(tr)
            L.append(ld)
        out = {"n_edges_trail": 0, "n_edges_lead": 0, "sigma_ref": sig}
        env = {}
        per_edge = {"trail": [], "lead": []}
        for side, lst in (("trail", T), ("lead", L)):
            lst = [i for i in lst if i is not None]
            dlen = min(i.size for i in lst)
            E = np.stack([e[i[:dlen]] for i in lst])
            out[f"n_edges_{side}"] = int(E.shape[0])
            out[f"min_window_ms_{side}"] = ms(dlen)
            env[side] = np.sqrt(np.mean(E ** 2, axis=0))
            for row in E:
                pk = float(np.max(np.abs(row)))
                per_edge[side].append(last_above(row, TOL * pk) if pk > 0 else 0)
            out[f"env_at_d1_over_sigma_{side}"] = round(float(env[side][0]) / sig, 4)
            out[f"err_max_over_sigma_{side}"] = round(float(np.max(np.abs(E))) / sig, 4)
        peak = max(float(env["trail"].max()), float(env["lead"].max()))
        for side in ("trail", "lead"):
            out[f"{side}_ms"] = ms(last_above(env[side], TOL * peak))
            out[f"{side}_ms_sigma1pct"] = ms(last_above(env[side], TOL * sig))
            pe = np.asarray(per_edge[side])
            out[f"{side}_ms_per_edge_p50_p95_max"] = [ms(int(np.percentile(pe, 50))),
                                                     ms(int(np.percentile(pe, 95))), ms(int(pe.max()))]
        res[nm] = out
    return res


def main() -> None:
    det = deterministic()
    noise = stochastic("out_noise.mat", ["noise_G21", "noise_G1", "noise_G100", "noise_G1000"])
    mr = loadmat(HERE / "out_real.mat")
    rnames = [str(np.asarray(x).squeeze()) for x in mr["rnames"].ravel()]
    real = stochastic("out_real.mat", rnames)
    info = loadmat(HERE / "out_syn.mat")["info"]
    finfo = {n: (str(np.asarray(info[n][0, 0]).squeeze())) for n in info.dtype.names}

    # worst case per side over every input (headline: deterministic 'total', stochastic envelope)
    cand = []
    for nm, r in det.items():
        for kind, v in r.items():
            cand.append((v["trail_ms"], "trail", nm, kind))
            cand.append((v["lead_ms"], "lead", nm, kind))
    for grp in (noise, real):
        for nm, v in grp.items():
            cand.append((v["trail_ms"], "trail", nm, "envelope"))
            cand.append((v["lead_ms"], "lead", nm, "envelope"))
    worst = {s: max((c for c in cand if c[1] == s), key=lambda c: c[0]) for s in ("trail", "lead")}
    WF_REACH = {"trail": 2.0 + 0.5, "lead": 1.0 + 0.5}   # step4: wfPost/wfPre + wfAlignSearch (ms)
    terms = {
        "step1_trail_ms": worst["trail"][0], "step1_lead_ms": worst["lead"][0],
        "step1_plus_step4_trail_ms": round(worst["trail"][0] + WF_REACH["trail"], 3),
        "step1_plus_step4_lead_ms": round(worst["lead"][0] + WF_REACH["lead"], 3),
    }
    binding = max(terms, key=terms.get)
    doc = {
        "fs": FS, "tolerance": "1% of peak (task 13: extent/tolerance.py:240-245; IMPULSE_DECAY_FRACTION "
                               "bands/envelope.py:132)",
        "distance": "samples from the NaN core; d=1 is the valid sample adjacent to the gap; pad p "
                    "removes d<=p (step2_noise_sigma.m:41,77)",
        "path": finfo,
        "deterministic": det, "noise": noise, "real": real,
        "worst": {s: {"ms": w[0], "input": w[2], "kind": w[3]} for s, w in worst.items()},
        "step4_reach_ms": WF_REACH, "terms": terms, "binding": binding,
        "settling_ms": terms[binding],
        "candidate_edgeBufferMs": ceil_half(terms[binding]),
        "candidate_edgeBufferMs_detection_only": ceil_half(max(worst["trail"][0], worst["lead"][0])),
    }
    (HERE / "settling.json").write_text(json.dumps(doc, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: doc[k] for k in ("worst", "terms", "binding", "settling_ms",
                                         "candidate_edgeBufferMs",
                                         "candidate_edgeBufferMs_detection_only")}, indent=1))
    for nm, r in det.items():
        print(nm, {k: (v["trail_ms"], v["lead_ms"], v["trail_ms_at_0.1pct"], v["lead_ms_at_0.1pct"],
                       round(v["peak_valid"], 5)) for k, v in r.items()})
    for grp in (noise, real):
        for nm, v in grp.items():
            print(nm, {k: v[k] for k in v if k.endswith(("_ms", "pct", "max", "sigma_trail",
                                                          "sigma_lead", "n_edges_trail"))})


if __name__ == "__main__":
    main()
