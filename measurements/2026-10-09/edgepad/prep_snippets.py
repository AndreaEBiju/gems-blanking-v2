"""Edge-pad measurement, step 0: three 10 s real tripole snippets from the cardiac cache.

Read only: scratchpad/cardiac/cache/<rid>.npz (T_<cuff>, microvolts) and the routed HRV train
the cache json names (bl recordings: origin 0). No G: access. Writes edgepad/snippets.mat.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
S = HERE.parent
CACHE = S / "cardiac" / "cache"
PICKS = [  # (rid, cuff) - three animals, bl, routed spike cuffs
    ("gems_a_t04_3_1_bl_190347_20260910T230351Z", "R"),
    ("gems_b_t03_2_1_bl_174858_20260923T214905Z", "L"),
    ("gems_h_t01_1_2_bl_214349_20260914T014355Z", "R"),
]
T0_S, DUR_S = 300.0, 10.0

out: dict[str, object] = {}
meta = []
for i, (rid, cuff) in enumerate(PICKS, start=1):
    info = json.loads((CACHE / f"{rid}.json").read_text(encoding="utf-8"))
    fs = float(info["fs"])
    assert abs(fs - 24414.0625) < 1e-6
    t = np.load(CACHE / f"{rid}.npz")[f"T_{cuff}"]
    i0 = int(round(T0_S * fs))
    n = int(round(DUR_S * fs))
    x = np.asarray(t[i0:i0 + n], dtype=np.float64).copy()
    del t
    assert np.all(np.isfinite(x)), rid
    m = loadmat(info["beats_file"])
    h = np.asarray(m["heartlocs"], dtype=np.int64).ravel()        # 1-based, origin 0 (bl)
    r0 = h - 1 - i0                                                 # 0-based within the snippet
    r0 = r0[(r0 >= 0) & (r0 < n)]
    out[f"x{i}"] = x
    out[f"r{i}"] = r0.astype(np.float64)
    meta.append({"rid": rid, "cuff": cuff, "start_s": T0_S, "dur_s": DUR_S, "units": "uV",
                 "n_beats_in_snippet": int(r0.size), "beats_file": Path(info["beats_file"]).name})
    print(rid, cuff, x.size, "beats", r0.size, "std", float(np.std(x)))
savemat(HERE / "snippets.mat", out)
(HERE / "snippets.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
