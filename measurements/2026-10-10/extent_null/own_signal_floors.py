"""Per-consumer own-signal floors on the archived clean baseline (Andrea, revised 2026-10-10).

Invariant 3: each consumer's floor is computed on ITS OWN signals in its own band - the
stomach contacts for slow wave (0-2 Hz) and mmc (2-50 Hz), the nerve contacts for spikes
(300-3000 Hz), the HR-source channels for breathing - from the same baseline as null.json
(gems_j_t01_ms3_bl_230315, the 1.44 null). The computation is
extent.tolerance_map.own_signal_floor_record, so a declaration in the night config
(``extent_floors``) is reproducible from this file (tolerance_map.verify_floor_archive).

    python own_signal_floors.py --hr-source R_V2 [--hr-source R_V3]

``--hr-source``: the baseline recording's HR-source channels as detection signals (its routed
HR channel, or both channels of its pair) - declared, never guessed. Read only; writes
own_floors.json beside this script. NOT run by the builder (a data pass): the values are
measured separately.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESSION = "gems_j_t01_ms3_bl_230315_20260916T030321Z"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-source", action="append", required=True)
    ap.add_argument("--night4", required=True, help="folder holding night4.py (the loader)")
    a = ap.parse_args()
    sys.path.insert(0, a.night4)
    import night4 as n4

    from gems_blanking_v2.detect import chain
    from gems_blanking_v2.extent import tolerance_map as tm
    rec = n4._load(n4.store(), {"animal": "J", "session": SESSION})  # noqa: SLF001
    det = chain.detect_region(rec, (0.0, rec.n_samples / float(rec.fs)))
    stomach = sorted(c.name for c in rec.channels if c.role == "stomach")
    nerve = sorted(s for s in det.signals if re.fullmatch(r"[LR]_V\d", s))
    sets = {"slow_wave": stomach, "mmc": stomach, "spikes": nerve,
            "breathing": sorted(a.hr_source)}
    out = {"session": SESSION, "fs": float(rec.fs), "sets": sets,
           "per_consumer_own_signals": {
               c: [tm.own_signal_floor_record(det.z, consumer=c, signals=sigs, percentile=p)
                   for p in tm.EXTENT_PERCENTILES] for c, sigs in sets.items()}}
    (HERE / "own_floors.json").write_text(json.dumps(out, indent=1, sort_keys=True),
                                          encoding="utf-8", newline="\n")
    print(json.dumps(out["per_consumer_own_signals"], indent=1), flush=True)


if __name__ == "__main__":
    main()
