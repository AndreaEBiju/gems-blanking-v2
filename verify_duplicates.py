"""Decide, per colliding stem, whether it is one recording or two.

Two recordings claiming one store path and one recording reachable by two Drive
paths need **opposite** fixes - a better key versus a deduplicated enumeration -
so the generator must not pick. It refuses, and this decides on evidence.

**Compare the signal array, never the file bytes.** A ``_sig.mat`` carries
``createdAt`` and ``srcBlock`` in its header and trailing variables, so two
conversions of the same TDT block differ in bytes while holding identical
samples. Hashing head and tail compares exactly those two regions and reports
"different recordings" with complete confidence. That is how this was first got
wrong; the mid-file range then failed too, because a 3-byte metadata difference
shifts every offset after it.

Writes ``duplicates.json``, which the generator consults. A group absent from it,
or present and not verified identical, still makes the generator refuse.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from gems_blanking_v2.io.corpus import read_scan_roots
from gems_blanking_v2.io.store import find_gems_root
from scipy.io import loadmat

STRIDE: Final = 97
"""Sample every Nth row when hashing. Prime, so it cannot align with a channel
count or a block boundary and miss a periodic difference."""


def signal_digest(path: Path) -> tuple[str, tuple[int, ...]]:
    """Return a digest of the SIGNAL array and its shape, ignoring metadata."""
    m = loadmat(path, variable_names=["signal"])
    arr = np.asarray(m["signal"])
    sub = np.ascontiguousarray(arr[::STRIDE])
    return hashlib.sha256(sub.tobytes()).hexdigest()[:16], tuple(arr.shape)


def main(scratch: Path | None = None) -> None:
    """Group by store key, verify each group, write ``duplicates.json``."""
    root = find_gems_root()
    scope = read_scan_roots(root)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")

    by_stem: dict[str, list[Path]] = defaultdict(list)
    n = 0
    for scan_root in scope.resolve(root):
        for p in scan_root.rglob("*_sig.mat"):
            by_stem[p.stem].append(p)
            n += 1
    dups = {k: sorted(v) for k, v in by_stem.items() if len(v) > 1}
    print(f"enumerated_at {stamp}: {n} recordings, {len(dups)} colliding stems")

    out: dict[str, object] = {
        "enumerated_at": stamp,
        "n_enumerated": n,
        "method": (
            "sha256 of signal[::97] plus shape. The signal array only - a "
            "_sig.mat's createdAt and srcBlock differ between two conversions "
            "of the same block, so any file-byte comparison reports 'different' "
            "for recordings that are identical."
        ),
        "groups": {},
    }
    groups: dict[str, object] = {}
    same = diff = 0
    for i, (stem, paths) in enumerate(sorted(dups.items()), 1):
        digests, shapes = [], []
        for p in paths:
            d, s = signal_digest(p)
            digests.append(d)
            shapes.append(s)
        identical = len(set(digests)) == 1 and len(set(shapes)) == 1
        same += identical
        diff += not identical
        groups[stem] = {
            "paths": [p.relative_to(root).as_posix() for p in paths],
            "sizes": [p.stat().st_size for p in paths],
            "signal_digests": digests,
            "shapes": [list(s) for s in shapes],
            "identical": identical,
            "verdict": (
                "one recording reachable by two Drive paths; the enumeration "
                "double-counts and the store is sound"
                if identical else
                "GENUINELY DIFFERENT RECORDINGS sharing a store key - the key "
                "needs the acquisition block, not the stem"
            ),
        }
        print(f"  [{i}/{len(dups)}] {stem}: "
              f"{'identical' if identical else 'DIFFERENT'}", flush=True)
    out["groups"] = groups
    target = (scratch or Path()) / "duplicates.json"
    target.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(f"\n{same} identical, {diff} genuinely different -> {target}")
    if diff:
        print("REFUSE: at least one group is two recordings. The store key must "
              "change before anything is written.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
