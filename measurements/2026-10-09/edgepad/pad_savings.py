"""Edge-pad measurement, step 3: spike time saved per animal by a smaller P.edgeBufferMs, edges (a).

(a) peri-R NaN spans on each recording's routed train (perir_k1/perir_train.resolve_train - the
ONE resolver night4.py uses), window per RULING 2026-10-09 (b) 1: [R - 11.5 ms, R + 9.5 ms),
except gems_a_t02_2_3_bl_215610 (16.0 / 9.5); a recording whose routed peri_r_ms (or narrow
window) is wider on either side uses its own on that side. Span -> samples with the package's
seconds_to_sample (round(t fs)); step2's pad p = round(ms 1e-3 fs) both sides
(step2_noise_sigma.m:41, 77: movmax [p p]); exact union after padding, clipped to the routing
region. Saved(p) = union(pad 10 ms) - union(pad p), seconds, over the routing region.

Sources: the same as perir_k1/perir_window.py (routing 64c2e1ea, pr/production_routing.json,
pr/b*/routing_entries.json) - copied, not imported, because its main() writes perir_k1/.
Read only. Writes edgepad/pad_savings.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parent
sys.path.insert(0, str(S / "perir_k1"))
from perir_train import PR, resolve_train  # noqa: E402

ROUTING_HASH = "64c2e1eafa318d3330e1cd10d861d4bbdb550f0bfd1365e532d5a7e4aa92aea9"
DEFAULT = (11.5, 9.5)
EXCEPTION = {"gems_a_t02_2_3_bl_215610": (16.0, 9.5)}
PADS_MS = [10.0, 10.5, 9.5, 8.0, 7.0, 6.0]


def rj(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def union_len(starts: np.ndarray, stops: np.ndarray, n: int) -> int:
    s, e = np.clip(starts, 0, n), np.clip(stops, 0, n)
    keep = e > s
    s, e = s[keep], e[keep]
    if not s.size:
        return 0
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    ce = np.maximum.accumulate(e)
    new = np.concatenate(([True], s[1:] > ce[:-1]))
    starts_m = s[new]
    ends_m = np.concatenate((ce[:-1][new[1:]], ce[-1:]))
    return int(np.sum(ends_m - starts_m))


def union_len_loop(starts, stops, n):  # independent check of the vectorised merge
    s, e = np.clip(starts, 0, n), np.clip(stops, 0, n)
    keep = e > s
    s, e = s[keep], e[keep]
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    tot, cs, ce = 0, int(s[0]), int(e[0])
    for a, b in zip(s[1:].tolist(), e[1:].tolist(), strict=True):
        if a <= ce:
            ce = max(ce, b)
        else:
            tot += ce - cs
            cs, ce = a, b
    return tot + ce - cs


def main() -> None:
    from gems_blanking_v2.emit import routing as er
    from gems_blanking_v2.extent.grid import seconds_to_sample
    from gems_blanking_v2.io.store import GemsStore, find_gems_root

    st = GemsStore(find_gems_root())
    t14 = er.read_routing(st, ROUTING_HASH)
    prod = rj(PR / "production_routing.json")
    if prod.get("parent_ruleset") != ROUTING_HASH or er.table_hash(prod) != prod.get("hash"):
        raise SystemExit("production_routing.json fails its hash/parent check")
    plan = rj(PR / "plan.json")
    sources: dict[str, tuple[str, dict]] = {rid: ("64c2e1ea", e) for rid, e in t14["entries"].items()}
    for rid, e in prod["entries"].items():
        sources.setdefault(rid, ("production_routing.json", e))
    n_batch_files = 0
    for b in sorted(plan["batches"]):
        f = PR / b / "routing_entries.json"
        if not f.is_file():
            continue
        n_batch_files += 1
        for rid, e in rj(f)["entries"].items():
            if rid not in sources:
                er.validate_entry(rid, e)
                sources[rid] = (f"{b}/routing_entries.json", e)

    rows, no_train, errors, own = [], [], {}, []
    for rid, (src, e) in sorted(sources.items()):
        animal = rid.split("_")[1].upper()
        try:
            tr = resolve_train(st.root, animal, rid, e)
        except Exception as ex:  # noqa: BLE001 - listed, not hidden
            errors[rid] = repr(ex)[:200]
            continue
        if tr is None:
            no_train.append(rid)
            continue
        stem = "_".join(rid.split("_")[:-1])
        before, after = EXCEPTION.get(stem, DEFAULT)
        rb, ra = [], []
        for s in e["spike"].values():
            if "peri_r_ms" in s:
                rb.append(-float(s["peri_r_ms"][0]))
                ra.append(float(s["peri_r_ms"][1]))
            for a, b in s.get("peri_r_narrow_ms", []):
                rb.append(-float(a))
                ra.append(float(b))
        if rb and (max(rb) > before or max(ra) > after):
            own.append({"recording": rid, "routed_before_ms": max(rb), "routed_after_ms": max(ra)})
            before, after = max(before, max(rb)), max(after, max(ra))
        fs = tr.fs
        n = seconds_to_sample(tr.region_s[1], fs) - seconds_to_sample(tr.region_s[0], fs)
        r = tr.heartlocs.astype(np.int64) - 1          # 0-based in the region
        nb, na = seconds_to_sample(before / 1e3, fs), seconds_to_sample(after / 1e3, fs)
        lost = {}
        for pm in PADS_MS:
            p = seconds_to_sample(pm / 1e3, fs)
            lost[pm] = union_len(r - nb - p, r + na + p, n)
        core = union_len(r - nb, r + na, n)
        if len(rows) < 5:  # spot-check the vectorised merge against the loop
            assert lost[10.0] == union_len_loop(r - nb - 244, r + na + 244, n)
        # exact vs naive: 2 * dpad * beats
        rows.append({"recording": rid, "animal": animal, "fs": fs, "n_region": n, "n_beats": int(r.size),
                     "window_ms": [before, after], "core": core, "lost": lost})

    by = defaultdict(lambda: {"n_rec": 0, "beats": 0, "region_s": 0.0, "core_s": 0.0,
                              **{f"lost_s@{pm}": 0.0 for pm in PADS_MS}})
    for x in rows:
        a = by[x["animal"]]
        a["n_rec"] += 1
        a["beats"] += x["n_beats"]
        a["region_s"] += x["n_region"] / x["fs"]
        a["core_s"] += x["core"] / x["fs"]
        for pm in PADS_MS:
            a[f"lost_s@{pm}"] += x["lost"][pm] / x["fs"]
    tot = {"n_rec": 0, "beats": 0, "region_s": 0.0, "core_s": 0.0, **{f"lost_s@{pm}": 0.0 for pm in PADS_MS}}
    for a in by.values():
        for k in tot:
            tot[k] += a[k]
    table = {}
    for name, a in [*sorted(by.items()), ("ALL", tot)]:
        valid10 = a["region_s"] - a["lost_s@10.0"]
        t = {"n_rec": a["n_rec"], "beats": a["beats"], "region_h": round(a["region_s"] / 3600, 3),
             "beats_per_s": round(a["beats"] / a["region_s"], 3),
             "pct_lost_core": round(100 * a["core_s"] / a["region_s"], 3),
             "pct_lost_core_plus_pad10": round(100 * a["lost_s@10.0"] / a["region_s"], 3)}
        for pm in PADS_MS[1:]:
            saved = a["lost_s@10.0"] - a[f"lost_s@{pm}"]
            naive = 2 * (10.0 - pm) / 1e3 * a["beats"]
            t[f"pad{pm}"] = {"saved_s": round(saved, 1), "pct_of_region": round(100 * saved / a["region_s"], 3),
                             "pct_of_valid_at_10": round(100 * saved / valid10, 3),
                             "naive_2dpad_beats_s": round(naive, 1)}
        table[name] = t
    doc = {"definition": __doc__, "pads_ms": PADS_MS, "n_sources": len(sources),
           "n_batch_files": n_batch_files, "n_with_train": len(rows), "no_train": len(no_train),
           "errors": errors, "own_window": own, "by_animal": table}
    (HERE / "pad_savings.json").write_text(json.dumps(doc, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: doc[k] for k in ("n_sources", "n_with_train", "no_train", "errors", "own_window")},
                     indent=1))
    for k, t in table.items():
        print(k, json.dumps(t))


if __name__ == "__main__":
    main()
