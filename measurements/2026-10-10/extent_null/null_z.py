"""RULING 2026-10-10 item 4: the clean null behind the 1.44 floor, and its 95th / 99th values.

The floor (bands.zscore.NULL_P90_Z = 1.44, NULL_P99_Z = 4.21) was measured 2026-09-19 on
"animal J baseline, 601 s" (IMPLEMENTATION.md, MEASURED 2026-09-19): robust z on the LOG
envelope against the file's own median reference, 6 bands, 9 raw contacts, every frame of the
uninjected file. The J baseline of 601 s named in the spec is gems_j_t01_ms3_bl_230315
(IMPLEMENTATION.md line ~2839, the runaway recording). This recomputes that null with the
package's own detection chain (chain.detect_region -> z_by_pair: the same z every threshold
reads) and reports p90 / p95 / p99 pooled over the 9 raw contacts (the 09-19 definition), per
band, and over every detection signal. Read only; writes extent_table/null.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parent
sys.path.insert(0, str(S / "night4"))
SESSION = "gems_j_t01_ms3_bl_230315_20260916T030321Z"
Q = (50, 90, 95, 99)


def pct(v: np.ndarray) -> dict[str, float]:
    v = v[np.isfinite(v)]
    return {f"p{q}": round(float(np.percentile(v, q)), 4) for q in Q} | {"n": int(v.size)}


def main() -> None:
    import psutil

    import night4 as n4
    from gems_blanking_v2.detect import chain
    st = n4.store()
    t0 = time.perf_counter()
    rec = n4._load(st, {"animal": "J", "session": SESSION})  # noqa: SLF001
    dur = rec.n_samples / float(rec.fs)
    det = chain.detect_region(rec, (0.0, dur))
    raw_stomach = {c.name for c in rec.channels if c.role == "stomach"}
    is_raw = {s: bool(re.fullmatch(r"[LR]_V\d", s)) or s in raw_stomach for s in det.signals}
    bands = sorted({b for _s, b in det.z})
    raw_pairs = [k for k in det.z if is_raw[k[0]]]
    out = {"session": SESSION, "duration_s": dur, "fs": float(rec.fs),
           "signals": list(det.signals), "raw_contacts": sorted(s for s in det.signals if is_raw[s]),
           "bands": bands,
           "pooled_raw_contacts": pct(np.concatenate([np.asarray(det.z[k], float) for k in raw_pairs])),
           "pooled_all_signals": pct(np.concatenate([np.asarray(v, float) for v in det.z.values()])),
           "per_band_raw_contacts": {b: pct(np.concatenate([np.asarray(det.z[k], float)
                                                            for k in raw_pairs if k[1] == b]))
                                     for b in bands},
           "per_band_all_signals": {b: pct(np.concatenate([np.asarray(v, float)
                                                           for k, v in det.z.items() if k[1] == b]))
                                    for b in bands},
           "per_pair": {f"{s}|{b}": pct(np.asarray(v, float)) for (s, b), v in sorted(det.z.items())},
           "recorded_constants": {"NULL_P90_Z": 1.44, "NULL_P99_Z": 4.21}}
    mem = psutil.Process().memory_info()
    out["rss_peak_gb"] = round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)
    out["t_s"] = round(time.perf_counter() - t0, 1)
    (HERE / "null.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8",
                                    newline="\n")
    print(json.dumps({k: out[k] for k in ("pooled_raw_contacts", "per_band_raw_contacts",
                                          "rss_peak_gb", "t_s")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
