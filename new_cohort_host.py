"""Adapt a new-cohort recording to the five-column input the MATLAB consumers take.

The consumers (and T's harness) were written for the OLD cohort's
``*_blankmotion.mat``: ``yOut`` N x 5 in volts, columns ``RVN, LVN, ANT1, ANT2,
ANT3``, where the two nerve columns are HARDWARE tripoles formed before the
amplifier. The new cohort records nine independent contacts, so the adapter:

1. loads through the project's loader (store ``meta.json`` for geometry, the
   cohort protocol for units, the file's own chanlabels for order);
2. forms each cuff's tripole with the project's own derivation (task 04,
   ``T = a*V1 + b*V3 - V2``, a = b = 0.5) - a SOFTWARE tripole, which is the
   hardware-vs-software caveat T's output already carries;
3. writes ``yOut = [R_T, L_T, ANT1, ANT2, ANT3]`` in VOLTS, with ``fs`` and the
   provenance of every column.

Nothing is blanked here: new-cohort recordings have no ``removedSegmentIdx`` yet,
and ``tolerance_baseline`` applies the pipeline's own clean-span criterion (no NaN,
no sample above 40 sigma, no flat run) before any consumer sees a span.

Usage::

    python new_cohort_host.py <recording _sig.mat> <out.mat>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.io.recording import load_recording
from gems_blanking_v2.io.store import GemsStore, find_gems_root
from gems_blanking_v2.io.tdt_block import session_key
from scipy.io import savemat

COLUMNS: Final = ("R_T", "L_T", "ANT1", "ANT2", "ANT3")
"""The old cohort's column order (``RVN, LVN, ANT1, ANT2, ANT3``), which the
consumers index by position: nerve 1:2, stomach 3:5, hrChanIdx = 1."""

UV_PER_V: Final = 1e6


def five_column(recording: Path, store: GemsStore) -> tuple[np.ndarray, float, dict[str, str]]:
    """Return ``(yOut in volts, fs, provenance)`` for one new-cohort recording."""
    animal = recording.parent.name.split("_")[1].upper()
    loaded = load_recording(recording, animal=animal, store=store)
    rec = loaded.recording
    signals, _weights = build_derivations(rec)
    by_name = {c.name.upper(): c.index for c in rec.channels}
    cols = [signals["R_T"], signals["L_T"],
            *(rec.data[:, by_name[f"ANT{i}"]] for i in (1, 2, 3))]
    y = np.column_stack(cols) / UV_PER_V
    prov = {
        "source": recording.name,
        "session": str(session_key(recording)),
        "columns": ",".join(COLUMNS),
        "nerve_columns": "software tripole T = 0.5*V1 + 0.5*V3 - V2 (task 04)",
        "units": "V (loader returns uV; divided by 1e6)",
    }
    return y, float(rec.fs), prov


def main(recording: Path, out: Path) -> None:
    """Write one adapted host file."""
    y, fs, prov = five_column(recording, GemsStore(find_gems_root()))
    savemat(out, {"yOut": y, "fs": fs, "chanlabels": np.array(COLUMNS, dtype=object),
                  "provenance": prov}, do_compression=False)
    print(f"{out.name}: {y.shape[0]} x {y.shape[1]}, {y.shape[0] / fs:.1f} s at {fs:g} Hz")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
