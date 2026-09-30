"""A stomach reference built only from the stomach contacts a consumer may read.

Ruling 2026-09-30 (b), item 6: in the three recordings where ANT1 is mains-dominated,
``stomach_ref`` (ANT1 minus the stomach common average, built in ``derivations`` -
inside the generation hash, and not touched) still carries ANT1. This builds the same
construction over the readable contacts only: the first readable contact minus the
mean of the readable ones. With ANT1 excluded that is ``ANT2 - mean(ANT2, ANT3)``.
Used only to MEASURE whether the stomach consumers' outputs change; they switch to it
only if they do (invariant 43 permits distrusting a derived quantity).

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.types import Recording

__all__ = ["MIN_CONTACTS", "readable_stomach_ref"]

F64 = npt.NDArray[np.float64]

MIN_CONTACTS = 2
"""A common average of fewer contacts is identically zero."""


def readable_stomach_ref(rec: Recording, exclude: Collection[str]) -> tuple[F64, tuple[str, ...]]:
    """Return ``(reference, contacts used)`` over the stomach contacts not in ``exclude``.

    Raises when fewer than two remain: a common average of one contact is zero, and a
    reference that is identically zero would read as a flat, perfect signal.
    """
    keep = [c for c in rec.channels if c.role == "stomach" and c.name not in exclude]
    if len(keep) < MIN_CONTACTS:
        msg = f"{len(keep)} readable stomach contact(s): a common average needs two"
        raise ValueError(msg)
    cols = np.column_stack([np.asarray(rec.data[:, c.index], dtype=np.float64) for c in keep])
    return np.asarray(cols[:, 0] - cols.mean(axis=1), dtype=np.float64), tuple(c.name for c in keep)
