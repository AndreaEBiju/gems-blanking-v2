"""Robust z on the shared grid: the log envelope against its own epoch's reference.

``z = (log E - median) / scale``, with both scalars from
:func:`~gems_blanking_v2.bands.reference.epoch_reference` for **this** signal and
**this** band. Hard invariant 3: every signal is thresholded against its own sigma,
never a borrowed one. The measured cost of borrowing is 1136 true detections against
19,624 false ones, so the check that the reference belongs to the trace is made here
rather than left to the caller's discipline.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.bands.reference import Reference

__all__ = ["NULL_P90_Z", "NULL_P99_Z", "zscore"]

F64 = npt.NDArray[np.float64]

NULL_P90_Z: Final = 1.44
"""p90 of ``z`` on clean frames under the log form, measured 2026-09-19.

With the superseded linear form it was 3.05, which is what put about 10% of clean
frames over ``z = 3`` per pair. Recorded so the shape of the null can be regression
-tested rather than assumed; see :data:`NULL_P99_Z`.
"""

NULL_P99_Z: Final = 4.21
"""p99 of ``z`` on clean frames under the log form, measured 2026-09-19.

Note what this means for invariant 10b: 1% of clean frames exceed 4.21 **per pair**,
and there are 36 pairs. Any flag-rate target is family-wise, and this constant is one
pair's contribution to it, never the candidate rate.
"""


def zscore(log_env: F64, reference: Reference, *, signal: str, band: str) -> F64:
    """Return robust ``z`` for one signal and band against its own epoch reference.

    Parameters
    ----------
    log_env
        ``(n_frames,)`` log envelope, from
        :func:`~gems_blanking_v2.bands.envelope.log_envelope`. Takes the log envelope
        rather than the linear one so the log happens exactly once, in one place.
    reference
        The pair for this signal and band.
    signal, band
        Checked against the reference's own names. Passing them again looks redundant
        and is not: it is what turns invariant 3 from a rule people remember into one
        the code enforces.

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)``. ``nan`` frames stay ``nan`` - a frame with no assessable data
        has no ``z``, and giving it 0 would read as "measured, and normal".

    Raises
    ------
    ValueError
        If the reference belongs to a different signal or band.
    """
    if reference.signal != signal or reference.band != band:
        msg = (
            f"refusing to apply the {reference.signal}/{reference.band} reference to "
            f"{signal}/{band}. Every signal is thresholded against its own sigma "
            "(invariant 3); borrowing was measured at 1136 true detections against "
            "19,624 false ones."
        )
        raise ValueError(msg)

    values = np.asarray(log_env, dtype=np.float64)
    return np.asarray((values - reference.median) / reference.scale, dtype=np.float64)
