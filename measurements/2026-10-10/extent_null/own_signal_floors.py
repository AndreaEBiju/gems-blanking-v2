"""Per-consumer own-signal floors on the archived clean baseline (Andrea, revised 2026-10-11).

Invariant 3 and Andrea's definition: each consumer's floor pools EXACTLY its own signals - the
channel it analyses plus every contact that channel is computed from - in its own band, from the
same baseline as null.json (gems_j_t01_ms3_bl_230315, the 1.44 null). The populations come from
extent.tolerance.own_signal_population (the one own-signal construction site) on the baseline's
consumer signals (tolerance.consumer_signals with its cuffs and, for breathing / hrv, its routed
HR channel); the value from extent.tolerance_map.own_signal_floor_record. A declaration in the
night config (``extent_floors``) is reproducible from the archive this writes
(tolerance_map.verify_floor_archive).

    python own_signal_floors.py --night4 <folder of night4.py> [--hr-channel R_T]

``--hr-channel``: the baseline's routed HR channel, declared, never guessed. Without it breathing
and hrv have NO population on this baseline (its routing entry routes no HR channel) and are
recorded as such. Read only; writes own_floors.json beside this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESSION = "gems_j_t01_ms3_bl_230315_20260916T030321Z"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--night4", required=True, help="folder holding night4.py (the loader)")
    ap.add_argument("--hr-channel")
    a = ap.parse_args()
    sys.path.insert(0, a.night4)
    import night4 as n4

    from gems_blanking_v2.detect import chain
    from gems_blanking_v2.extent import tolerance as tl
    from gems_blanking_v2.extent import tolerance_map as tm
    rec = n4._load(n4.store(), {"animal": "J", "session": SESSION})  # noqa: SLF001
    det = chain.detect_region(rec, (0.0, rec.n_samples / float(rec.fs)))
    chmap = tl.channel_detection_signals(rec.channels)
    cuffs = sorted(s[0] for s in det.signals if s.endswith("_T"))
    signals = {c: (tl.consumer_signals(c, cuffs=cuffs, best_hr_channel=a.hr_channel)
                   if c not in ("hrv", "breathing") or a.hr_channel else ())
               for c in tl.expected_consumers()}
    out: dict = {"session": SESSION, "fs": float(rec.fs), "consumer_signals": signals,
                 "detection_signals": list(det.signals), "populations": {},
                 "per_consumer_own_signals": {}, "no_population": {}}
    for c in ("slow_wave", "mmc", "spikes", "breathing"):
        try:
            pop = tl.own_signal_population(c, signals, det.signals, chmap)
        except ValueError as ex:
            out["no_population"][c] = str(ex)
            continue
        out["populations"][c] = list(pop)
        out["per_consumer_own_signals"][c] = [
            tm.own_signal_floor_record(det.z, consumer=c, signals=pop, percentile=p)
            for p in tm.EXTENT_PERCENTILES]
    (HERE / "own_floors.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                          encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("populations", "per_consumer_own_signals",
                                          "no_population")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
