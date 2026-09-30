"""Which raw contacts a consumer may read, in the new cohort (ruling 2026-09-30, item 5).

Trust in ``T`` for the spike consumer is decided by ruling 4's verification, not here.
This screen is for the consumers that read RAW contacts (HR, velocity, and the
stomach consumers' raw contacts), with two new-cohort rules on top of
``contact_quality``'s flat / duplicate / uncorrelated / mains-hum:

- **missing ground signal**: every contact is single-ended against one shared ground
  with no local reference, so a healthy contact MUST carry the ground-site signal. A
  contact whose 300-3000 Hz R-squared against the mean of the channels outside its
  cuff is below :data:`PROVISIONAL_MIN_GROUND_R2` is the suspect one, whatever its
  peer correlation says (B t02 1_3 L: L2 and L3 at 0.04 and 0.01, correlated 0.90
  with each other, while the screen flagged L1, which carries the ground at 0.94);
- **low gain**: a contact whose R-squared sits more than
  :data:`PROVISIONAL_MAX_R2_GAP` below the median of its cuff peers' (H t01 es2 L1:
  0.67 against 0.98), which passes the correlation rule while its cuff's ``T`` is 75%
  common mode.

Cuffs distrusted by ruling until explained are listed in :data:`DISTRUSTED_CUFFS`.
A stomach contact the stomach screen flags (mains hum, flat, duplicate) is read by no
consumer; detection keeps reading every signal (invariant 43 forbids hiding a signal
from detection alone, not the reverse).

OUTSIDE THE GENERATION HASH: ``contact_quality`` is inside it (the chain imports it),
so these rules live here and change no candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.derive.common_mode import _eng, outside_reference
from gems_blanking_v2.types import Recording

__all__ = [
    "DISTRUSTED_CUFFS",
    "PROVISIONAL_MAX_R2_GAP",
    "PROVISIONAL_MIN_GROUND_R2",
    "ground_r2",
    "raw_readable",
    "screen_for_consumers",
]

F64 = npt.NDArray[np.float64]

PROVISIONAL_MIN_GROUND_R2: Final = 0.20
"""PROVISIONAL (ruling 2026-09-30): below this R-squared against the outside
reference, a contact is missing the ground signal. Measured: healthy contacts
0.67-0.99, the B t02 1_3 L2/L3 pair 0.04/0.01. Set from the cross-animal set before
it is trusted."""

PROVISIONAL_MAX_R2_GAP: Final = 0.15
"""PROVISIONAL (ruling 2026-09-30): a contact this far below its cuff peers' median
R-squared is low-gain. Measured: H t01 es2 L1 0.67 against 0.98 (gap 0.31); within
healthy cuffs the spread is up to ~0.10 (A t05 L 0.69-0.78)."""

DISTRUSTED_CUFFS: Final[Mapping[tuple[str, str], str]] = {
    ("gems_b_t02_1_3_bl_210203_20260910T010207Z", "L"): (
        "ruling 2026-09-30: L2 and L3 lack the ground signal (R2 0.04, 0.01) and agree "
        "with each other (r 0.90) at ~5x the other contacts' sigma - a short or a local "
        "source; the whole cuff is distrusted for every consumer until explained"),
}
"""(recording id, cuff) -> why. Every consumer; lifted only by a ruling."""


def ground_r2(rec: Recording) -> dict[str, float]:
    """Each nerve contact's 300-3000 Hz R-squared against its outside reference.

    The reference is the mean of every channel outside the contact's cuff
    (:func:`~gems_blanking_v2.derive.common_mode.outside_reference`). Keys are
    ``<cuff><contact>`` labels, as ``contact_quality`` uses.
    """
    fs = float(rec.fs)
    out: dict[str, float] = {}
    cuffs = sorted({c.cuff_id for c in rec.channels if c.cuff_id is not None})
    for cf in cuffs:
        ref, _names = outside_reference(rec, cf)
        re = _eng(ref, fs)
        for c in rec.channels:
            if c.cuff_id != cf or c.contact_index is None:
                continue
            y = _eng(np.asarray(rec.data[:, c.index], dtype=np.float64), fs)
            ok = np.isfinite(y) & np.isfinite(re)
            x, yy = re[ok], y[ok]
            k = float(x @ yy / (x @ x)) if x @ x > 0 else 0.0
            out[f"{cf}{c.contact_index}"] = 1.0 - float(np.var(yy - k * x) / np.var(yy))
    return out


def screen_for_consumers(r2: Mapping[str, float]) -> dict[str, tuple[str, ...]]:
    """``{label: reasons}`` from :func:`ground_r2`'s values; ``()`` means readable."""
    out: dict[str, tuple[str, ...]] = {}
    for label, v in r2.items():
        peers = [r2[k] for k in r2 if k != label and k[0] == label[0]]
        why = []
        if v < PROVISIONAL_MIN_GROUND_R2:
            why.append("missing_ground")
        elif peers and float(np.median(peers)) - v > PROVISIONAL_MAX_R2_GAP:
            why.append("low_gain")
        out[label] = tuple(why)
    return out


def raw_readable(
    recording_id: str, r2: Mapping[str, float], stomach_reasons: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """``{contact: why not}`` for every raw contact; ``()`` means a consumer may read it.

    Nerve contacts: :func:`screen_for_consumers`, and every contact of a cuff in
    :data:`DISTRUSTED_CUFFS`. Stomach contacts: any reason from the stomach screen
    (mains hum, flat, duplicate) keeps every consumer off it.
    """
    out = dict(screen_for_consumers(r2))
    for label, reasons in list(out.items()):
        if (recording_id, label[0]) in DISTRUSTED_CUFFS:
            out[label] = (*reasons, "distrusted_cuff")
    for label, reasons in stomach_reasons.items():
        out[label] = tuple(reasons)
    return out
