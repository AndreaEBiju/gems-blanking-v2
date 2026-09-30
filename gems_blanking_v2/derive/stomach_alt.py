"""A stomach reference built only from the stomach contacts a consumer may read.

Ruling 2026-09-30 (b), item 6: in the three recordings where ANT1 is mains-dominated,
``stomach_ref`` (ANT1 minus the stomach common average, built in ``derivations`` -
inside the generation hash, and not touched) still carries ANT1. This builds the same
construction over the readable contacts only: the first readable contact minus the
mean of the readable ones. With ANT1 excluded that is ``ANT2 - mean(ANT2, ANT3)``.
Used only to MEASURE whether the stomach consumers' outputs change; they switch to it
only if they do (invariant 43 permits distrusting a derived quantity).

Ruling 2026-09-30 (c), item 5, replaces that alternative for the consumers: in those
three recordings they read ``stomach_ref`` built from a NOTCHED ANT1
(:func:`notched_stomach_ref`) - the same derivation with the hum removed. Detection
keeps the un-notched ANT1 (invariant 43).

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import iirnotch, sosfiltfilt, tf2sos

from gems_blanking_v2.derive.derivations import build_stomach_reference
from gems_blanking_v2.types import Recording

__all__ = ["MIN_CONTACTS", "NOTCH_Q", "notch", "notched_stomach_ref", "readable_stomach_ref"]

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


NOTCH_Q: Final = 30.0
"""Quality factor of each mains notch. One pass is -3 dB at f0 +/- f0 / (2Q); applied
forward and backward (zero phase) the gain there is 0.5 - at 59 and 61 Hz for 60 Hz."""


def notch(x: npt.ArrayLike, fs: float, freqs_hz: Sequence[float], q: float = NOTCH_Q) -> F64:
    """Return ``x`` with a zero-phase notch at each of ``freqs_hz`` (Hz).

    Second-order sections, ``sosfiltfilt``. A NaN span is interpolated for filtering
    only and is NaN again in the output (invariant 8).
    """
    y = np.array(x, dtype=np.float64)
    bad = ~np.isfinite(y)
    if bad.all():
        return y
    if bad.any():
        idx = np.arange(y.size, dtype=np.float64)
        y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    for f0 in freqs_hz:
        if not 0.0 < f0 < fs / 2:
            msg = f"notch at {f0} Hz is outside (0, {fs / 2}) Hz"
            raise ValueError(msg)
        b, a = iirnotch(f0, q, fs)
        y = sosfiltfilt(tf2sos(b, a), y)
    y[bad] = np.nan
    return np.asarray(y, dtype=np.float64)


def notched_stomach_ref(
    rec: Recording, freqs_hz: Sequence[float], contacts: Sequence[str] = ("ANT1",)
) -> F64:
    """Return ``stomach_ref`` as ``derivations`` builds it, from ``contacts`` notched.

    Only the named stomach contacts are filtered (:func:`notch` at ``freqs_hz``);
    every other column, and the construction itself, are the derivation's. The input
    is never written to. Ruling (c) 5 notches ANT1; its addendum extends the notch to
    every input of a recording where the residual changes mmc.
    """
    data = np.array(rec.data, dtype=np.float64)
    for name in contacts:
        col = next((c.index for c in rec.channels if c.name == name and c.role == "stomach"),
                   None)
        if col is None:
            msg = f"no stomach contact named {name!r}"
            raise ValueError(msg)
        data[:, col] = notch(data[:, col], float(rec.fs), freqs_hz)
    ref, _how = build_stomach_reference(replace(rec, data=data))
    if ref is None:
        msg = "the recording has no stomach reference"
        raise ValueError(msg)
    return ref
