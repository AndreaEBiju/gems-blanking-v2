"""The whole-epoch reference pair, one per (signal, band).

Hard invariant 5: ``z = (log E - median(log E)) / (1.4826 * MAD(log E))``, computed
**on the log envelope**, as one scalar pair over the whole epoch, from that epoch
alone. No running baseline, no transfer between files, no iteration.

**Why the log.** The earlier linear form - a 10th-percentile reference on the linear
envelope, scaled by its MAD - is mis-centred: measured 2026-09-19 it puts median
``z`` at 1.00 and p90 at 3.05, so about 10% of frames exceed ``z = 3`` *per pair*,
which the union across 36 pairs turns into 40-55% of the file. Envelopes are positive
and right-skewed; the log makes the null symmetric, giving median 0, p90 1.44 and p99
4.21. There is no percentile parameter here, and that is deliberate - a second
reference rule in the code would compete with the binding one, and the competing rule
is the one that was measured to be wrong.

**Why a whole-epoch scalar and not a running window.** A running baseline adapts to
slow change, and in a post-stim recovery file the slow change *is the measurement*. A
rising baseline would require artifacts to be larger to cross threshold during
recovery than at baseline, manufacturing a condition confound out of the analysis
itself. It also has no edge problem, which matters because the post-stim dynamics
live in the first window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.constants import MAD_TO_SIGMA, REFERENCE_STATISTIC

__all__ = [
    "MIN_REFERENCE_FRAMES",
    "SCALE_FLOOR",
    "DeadChannelError",
    "Reference",
    "epoch_reference",
    "is_dead",
]

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]

SCALE_FLOOR: Final = 1e-6
"""Smallest usable scale, in log units.

Below this the channel is not quiet, it is dead: ``1e-6`` in log units is a part per
million of relative spread, which no real recording produces. Dividing by it would
turn rounding noise into ``z`` in the thousands, so a channel that reaches it is
gated by :func:`is_dead` **before** any division rather than clamped after one.
"""

MIN_REFERENCE_FRAMES: Final = 100
"""Fewest assessable frames a reference may be computed from.

A.3's own note on the slow bands: a 10-minute file holds only 80-100 independent
frames at 0.5-3 Hz, and a median and MAD taken from fewer than that carry a standard
error large enough to move every ``z`` downstream. Below this the reference is
refused rather than returned with a quiet caveat.

Counted in *frames*, not independent windows, so it is a floor on the arithmetic
rather than a statement about resolution - the independent-frame problem is real but
belongs to whoever is choosing band windows, not here.
"""


class DeadChannelError(ValueError):
    """A channel has no usable spread, so no reference can be computed from it."""


@dataclass(frozen=True, slots=True)
class Reference:
    """The scalar pair for one ``(signal, band)`` over one epoch.

    Attributes
    ----------
    median
        ``median(log E)``, in log microvolts.
    scale
        ``1.4826 * MAD(log E)``, in log microvolts. Never below :data:`SCALE_FLOOR`;
        a channel that would need it to be is refused instead.
    n_frames
        How many assessable frames it was computed from. Carried because a reference
        from 120 frames and one from 60,000 are not the same measurement.
    signal, band
        What it belongs to. **A reference is never applied to another signal** - hard
        invariant 3 - and carrying the names is what lets that be checked rather than
        trusted.
    """

    median: float
    scale: float
    n_frames: int
    signal: str
    band: str

    def __post_init__(self) -> None:
        """Check the pair is usable and self-describing."""
        if not np.isfinite(self.median) or not np.isfinite(self.scale):
            msg = (
                f"{self.signal}/{self.band}: a reference must be finite, got "
                f"median={self.median!r} scale={self.scale!r}"
            )
            raise ValueError(msg)
        if self.scale < SCALE_FLOOR:
            msg = (
                f"{self.signal}/{self.band}: scale {self.scale!r} is under the floor "
                f"{SCALE_FLOOR}; gate the channel with is_dead() rather than clamping"
            )
            raise DeadChannelError(msg)

    def to_provenance(self) -> dict[str, Any]:
        """Return the pair as a provenance record.

        Everything emitted carries provenance (invariant 11), and a ``z`` trace is
        meaningless without the two scalars it was made with.
        """
        return {
            "signal": self.signal,
            "band": self.band,
            "reference_statistic": REFERENCE_STATISTIC,
            "median_log_uv": self.median,
            "scale_log_uv": self.scale,
            "n_frames": self.n_frames,
        }


def is_dead(log_env: F64) -> bool:
    """Return whether a log envelope has too little spread to give a reference.

    Gating happens **before** the division, not after: a flat channel gives
    ``MAD -> 0`` and ``z -> inf``, and an infinity that reaches the candidate
    generator is a detection on every frame of a dead channel.
    """
    finite = np.asarray(log_env, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return True
    median = float(np.median(finite))
    return MAD_TO_SIGMA * float(np.median(np.abs(finite - median))) < SCALE_FLOOR


def epoch_reference(log_env: F64, *, signal: str, band: str) -> Reference:
    """Return the ``(median, scale)`` pair for one signal and band over one epoch.

    Takes the **log** envelope, not the linear one, and says so in the signature: the
    log is the whole of invariant 5, and a function that took a linear envelope and
    logged it internally would be a second place for that decision to live.

    ``nan`` frames - gaps, short segments, filter settling edges - are excluded rather
    than imputed. A frame with no assessable data contributes nothing to a whole-epoch
    statistic, and giving it the median would narrow the MAD by exactly the fraction
    of the epoch that was missing.

    Parameters
    ----------
    log_env
        ``(n_frames,)`` log envelope from
        :func:`~gems_blanking_v2.bands.envelope.log_envelope`.
    signal, band
        Names, carried into the returned pair so it cannot be applied to another
        signal by accident (invariant 3).

    Raises
    ------
    DeadChannelError
        If the channel has no usable spread.
    ValueError
        If fewer than :data:`MIN_REFERENCE_FRAMES` frames are assessable.
    """
    values = np.asarray(log_env, dtype=np.float64)
    assessable = values[np.isfinite(values)]

    if is_dead(assessable):
        msg = (
            f"{signal}/{band}: the channel has no usable spread over this epoch "
            f"({assessable.size} assessable frames). A flat channel gives MAD -> 0 and "
            "z -> inf, which is a detection on every frame; gate it, do not divide."
        )
        raise DeadChannelError(msg)
    if assessable.size < MIN_REFERENCE_FRAMES:
        msg = (
            f"{signal}/{band}: only {assessable.size} assessable frames, under the "
            f"{MIN_REFERENCE_FRAMES} a reference needs. A median and MAD from fewer "
            "carry enough standard error to move every z downstream."
        )
        raise ValueError(msg)

    median = float(np.median(assessable))
    scale = MAD_TO_SIGMA * float(np.median(np.abs(assessable - median)))
    reference = Reference(
        median=median,
        scale=scale,
        n_frames=int(assessable.size),
        signal=signal,
        band=band,
    )
    log.info(
        "%s/%s: reference median %.4f, scale %.4f log uV over %d frames (%s)",
        signal,
        band,
        median,
        scale,
        assessable.size,
        REFERENCE_STATISTIC,
    )
    return reference
