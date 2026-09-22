"""Shared constants: the analysis grid, the bands, the consumers, the measured table.

Every number here is either a definition (the grid, the band corners) or a value
already measured on real data (Part A.5 of ``IMPLEMENTATION.md``). Nothing in this
module is re-derived at run time, and nothing downstream may redefine any of it.

Units are named in each symbol: ``_S`` seconds, ``_HZ`` hertz, ``_UV`` microvolts,
``_MM`` millimetres, ``_DB`` decibels, ``_FRAC`` a dimensionless fraction in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# A.2 - the grid
# ---------------------------------------------------------------------------

GRID_S: Final = 0.010
"""Envelope / z / mask frame step, seconds. Shared by every band."""

FS_NOMINAL_HZ: Final = 24414.0625
"""Nominal TDT sample rate. Documentation only - always read ``fs`` from the file."""


def n_frames(duration_s: float) -> int:
    """Return the number of whole 10 ms frames in ``duration_s`` seconds.

    Parameters
    ----------
    duration_s
        Record duration in seconds.

    Returns
    -------
    int
        ``floor(duration_s / GRID_S)``. A trailing partial frame is dropped, so a
        1.005 s record has 100 frames, not 101.
    """
    if duration_s < 0.0:
        msg = f"duration_s must be non-negative, got {duration_s}"
        raise ValueError(msg)
    return int(math.floor(duration_s / GRID_S))


def frame_centre_s(index: int) -> float:
    """Return the centre time of frame ``index``, seconds.

    Frame ``i`` spans ``[i*GRID_S, (i+1)*GRID_S)`` and is centred at
    ``(i + 0.5)*GRID_S``. A band's analysis window is centred here and may be much
    longer than the frame; overlap between neighbouring windows is intended.
    """
    if index < 0:
        msg = f"frame index must be non-negative, got {index}"
        raise ValueError(msg)
    return (index + 0.5) * GRID_S


# ---------------------------------------------------------------------------
# A.3 - the bands
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BandSpec:
    """One analysis band.

    Attributes
    ----------
    lo_hz, hi_hz
        Band corners in Hz. ``lo_hz == 0.0`` means low-pass.
    window_s
        Length of the RMS-envelope window, seconds, centred on the frame centre.

    There is deliberately **no reference percentile** on a band. The whole-file
    reference is :data:`REFERENCE_STATISTIC` for every band; a per-band percentile
    in code would be a second reference rule competing with the binding one.
    """

    lo_hz: float
    hi_hz: float
    window_s: float

    @property
    def bandwidth_hz(self) -> float:
        """Width of the band in Hz."""
        return self.hi_hz - self.lo_hz

    @property
    def dof(self) -> float:
        """Effective degrees of freedom of the envelope window, ``2*B*T``.

        Five of the six bands were sized for ``dof ~= 30``; the ENG band was not.
        See :data:`BAND_DOF_TARGET` and :data:`ENG_BAND_DOF_IS_EXCEPTION`.
        """
        return 2.0 * self.bandwidth_hz * self.window_s


BANDS: Final[dict[str, BandSpec]] = {
    #                      lo       hi     window_s      2*B*T
    "300-3000": BandSpec(300.0, 3000.0, 0.025),  # 135 - a time-resolution choice
    "100-300": BandSpec(100.0, 300.0, 0.075),  # 30
    "10-150": BandSpec(10.0, 150.0, 0.100),  # 28 - was 1-100/150 ms; see HR_BAND
    "2-50": BandSpec(2.0, 50.0, 0.310),  # 29.8
    "0.5-3": BandSpec(0.5, 3.0, 6.000),  # 30
    "0-2": BandSpec(0.0, 2.0, 7.500),  # 30
}
"""The six analysis bands, keyed by name.

The ENG band is ``300-3000``, **not** the historical ``300-5000``: measured
2026-09-19 (A.5b), event energy reaches background by ~4 kHz, and narrowing the
corner lowers sigma(T) by 24%, raises the event rate 41% and improves
event-to-threshold separation. This breaks comparability with previously processed
data - sigma changes, so the 4.5-sigma threshold changes, so every historical spike
count changes. Reprocess rather than mix, and record the band in provenance.

The cardiac band is ``10-150``, **not** the historical ``1-100``: ruled 2026-09-21
with task 05. 1-100 was the earlier R-peak detection band and A.4's hrv row carried
the same stale number, so once detection moved the band had no consumer at all. A
contamination band has to be the band its consumer's detector actually reads. See
:data:`HR_BAND`.
"""

HR_BAND: Final = "10-150"
"""The band the R-peak detector reads, and therefore the ``hrv`` contamination band.

Ruled 2026-09-21. On animal J, 10-150 Hz at 6 MAD-sigma gives template SNR 611
against 446 and a long-interval rate of 2.0% against 3.6%; on the synthetic it emits
0 false beats over 3160 true ones where 1-100 Hz at 3 sigma emits 25. In 1-100 the
in-band noise floor collapses to ~0.6 uV while a weak beat still carries ~18 uV of
prominence, so noise and signal sit on the same side of 3 sigma.

Its ``dof`` is 28, not 30: the 100 ms window is a round number rather than the
107 ms that would hit the target exactly. **Task 02's peri-R measurement was made
at 1-100 and needs re-measuring here.**
"""

ENG_BAND: Final = "300-3000"
"""Name of the band spike detection and velocity read.

Kept as a symbol so the 2026-09-19 corner change does not have to be chased through
string literals a second time.
"""

BAND_DOF_TARGET: Final = 30.0
"""Design target for ``2*B*T``, the effective degrees of freedom of an envelope window."""

ENG_BAND_DOF_IS_EXCEPTION: Final = True
"""The ENG band does not meet :data:`BAND_DOF_TARGET` and was never going to.

``2*B*T`` for ``300-3000`` at a 25 ms window is 135 (it was 235 at ``300-5000``); the
other five bands land at 28.0-30.0. The 25 ms window is a *time-resolution* choice -
a burst must not be smeared across a frame - not a dof choice, and the band is not to
be reshaped to hit 30.

**It needs no exemption from task 06's dof check**, and the earlier claim here that it
did was wrong. That check compares each band's *measured* effective dof against **its
own** spec value, not against :data:`BAND_DOF_TARGET`; the ENG band measures 135.3
against a spec 135.0 and passes on the nose. Comparing every band to a global 30 would
have been the thing needing an exemption - for two bands, not one, since ``10-150``
is 28.
"""

REFERENCE_STATISTIC: Final = "median_of_log"
"""Binding definition of the whole-file reference (hard invariant 5).

``z = (log E - median(log E)) / (1.4826 * MAD(log E))``, one scalar pair per
(signal, band), computed from that file alone. Envelopes are positive and
right-skewed; the log makes the null symmetric (median 0, p90 1.44).

This is the **only** reference rule in the codebase. The per-band percentile it
replaced is recorded in A.5 as history and is deliberately absent from
:class:`BandSpec`: two reference rules in code means the wrong one eventually gets
used, and the percentile form is the one that was measured to be mis-centred.
"""

# ---------------------------------------------------------------------------
# A.4 - the consumers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsumerSpec:
    """One downstream analysis that receives its own mask.

    Attributes
    ----------
    name
        Consumer identifier, used as the mask key.
    signals
        Signal names the consumer reads, without a cuff prefix. Always a tuple:
        ``velocity`` reads two contacts and needs both valid (the single permitted
        intersection under hard invariant 2). Every other consumer reads one.
    band
        Key into :data:`BANDS`.
    tolerance
        Prose statement of what "the consumer's output changed" means. The numeric
        tolerance curve is derived per consumer in task 09, not stated here.
    """

    name: str
    signals: tuple[str, ...]
    band: str
    tolerance: str


CONSUMERS: Final[tuple[ConsumerSpec, ...]] = (
    ConsumerSpec("spikes", ("T",), ENG_BAND, "4.5 sigma, sample level"),
    ConsumerSpec("slow_c", ("T",), "100-300", "own sigma"),
    ConsumerSpec("velocity", ("V1", "V3"), ENG_BAND, "peak ratio > 1"),
    ConsumerSpec("mmc", ("stomach_ref",), "2-50", "3 x moving MAD"),
    ConsumerSpec("slow_wave", ("stomach_ref",), "0-2", "peak displacement"),
    ConsumerSpec("breathing", ("best_hr_channel",), "0.5-3", "peak inserted or lost"),
    ConsumerSpec("hrv", ("best_hr_channel",), HR_BAND, "operational: beat train unchanged"),
)
"""The seven mask consumers.

``spikes`` and ``velocity`` are listed against ``300-5000`` in ``IMPLEMENTATION.md``
A.4; that is stale - A.3 and A.5b moved the ENG band to ``300-3000`` and A.4 was not
updated with them. :data:`ENG_BAND` is used here so the two cannot drift again.

No 0-2 or 2-50 Hz mask on nerve signals. No ENG-band mask on stomach signals. Masks
are never merged across consumers (hard invariant 2).
"""

NERVE_SIGNALS: Final[tuple[str, ...]] = ("V1", "V2", "V3", "T")
"""Per-cuff signal names, before the cuff prefix (``"L_V1"``, ``"R_T"``, ...)."""

STOMACH_SIGNALS: Final[tuple[str, ...]] = ("stomach_ref",)
"""Signal names carried by the stomach EMG channels."""

NERVE_BANNED_BANDS: Final[frozenset[str]] = frozenset({"0-2", "2-50"})
"""Bands that carry no nerve mask."""

STOMACH_BANNED_BANDS: Final[frozenset[str]] = frozenset({ENG_BAND})
"""Bands that carry no stomach mask."""

# ---------------------------------------------------------------------------
# A.5 - constants measured already; do not re-derive
# ---------------------------------------------------------------------------

CUFF_V9_PITCH_MM: Final = 1.50
"""Centre-to-centre contact pitch of the v9 cuff, millimetres."""

CUFF_V9_APERTURE_MM: Final = 3.00
"""V1-to-V3 span of the v9 cuff, millimetres. Two pitches."""

NOTCH_60HZ_RING_UV_PER_MV: Final = 0.221
"""Ringing a 60 Hz notch leaks into the ENG band, microvolts per mV of excursion."""

BANDPASS_GAIN_60HZ_DB: Final = -36.4
"""Gain of the ENG bandpass at 60 Hz through ``filtfilt``, dB."""

QRS_ENERGY_ABOVE_300HZ_FRAC: Final = (0.0000, 0.0054)
"""Fraction of QRS energy above 300 Hz, (min, max) over measured beats."""

QRS_ENERGY_BELOW_3HZ_FRAC: Final = 0.0
"""Fraction of QRS energy below 3 Hz.

Measured 0.00%: the cardiac artifact does not reach the slow-wave bands.
"""

QRS_ENERGY_100_300HZ_FRAC: Final = (0.321, 0.652)
"""Fraction of QRS energy in 100-300 Hz for an 8-12 ms QRS, (min, max)."""

SMOOTH_MOTION_ENERGY_BELOW_300HZ_FRAC: Final = (0.987, 0.997)
"""Fraction of smooth-motion artifact energy below 300 Hz, (min, max)."""

SATURATING_STEP_ENERGY_ABOVE_300HZ_FRAC: Final = 0.322
"""Fraction of a saturating step's energy above 300 Hz.

A step is *not* a smooth motion artifact - a third of it lands in the ENG band.
"""

VELOCITY_ARTIFACT_TOLERANCE_RATIO: Final[dict[str, float]] = {
    "raw": 1.0,
    "band_limited": 5.0,
    "broadband": 1.4,
}
"""Artifact amplitude, in units of host sigma, that the velocity estimate tolerates."""

VELOCITY_RESOLUTION_FRAC: Final[dict[float, float]] = {
    0.5: 0.04,
    1.0: 0.07,
    2.0: 0.14,
    5.0: 0.35,
}
"""Fractional velocity resolution ``dv/v ~= v/(B*L)`` at a few conduction speeds, m/s."""

# --- A.5b: the ENG corner measurement --------------------------------------

ENG_EVENT_BACKGROUND_2X_CROSSING_HZ: Final = 2374.0
"""Frequency where the event/background power ratio falls through 2x (healthy cuff)."""

ENG_EVENT_BACKGROUND_1P2X_CROSSING_HZ: Final = 3730.0
"""Frequency where the event/background power ratio falls through 1.2x."""

ENG_CORNER_TRADEOFF: Final[dict[int, dict[str, float]]] = {
    2000: {
        "sigma_t_uv": 1.83,
        "events_per_s": 10.5,
        "median_amp_sigma": 6.09,
        "p90_amp_sigma": 10.88,
        "cardiac_peak_base": 2.93,
    },
    3000: {
        "sigma_t_uv": 2.26,
        "events_per_s": 8.2,
        "median_amp_sigma": 5.84,
        "p90_amp_sigma": 9.77,
        "cardiac_peak_base": 3.56,
    },
    5000: {
        "sigma_t_uv": 2.96,
        "events_per_s": 5.8,
        "median_amp_sigma": 5.62,
        "p90_amp_sigma": 8.43,
        "cardiac_peak_base": 4.31,
    },
}
"""Measured cost of each candidate ENG upper corner. One recording, animal J.

Single-recording evidence: confirm on the cross-animal set before pinning, and
revisit 2000 Hz if that set agrees.
"""

# --- A.5c: per-cuff health check -------------------------------------------

CUFF_HEALTH_PROBE_HZ: Final = 1000.0
"""Frequency at which the event/background ratio is reported as a cuff diagnostic."""

CUFF_HEALTHY_RATIO_AT_1KHZ: Final = 36.0
"""Event/background ratio at 1 kHz on a cuff with real neural content (cuff R)."""

CUFF_FAILED_RATIO_AT_1KHZ: Final = 1.16
"""Same ratio on a cuff with none (cuff L).

A cuff near 1 has no spectral signature of neural events and must be flagged before
its data reaches any analysis.
"""

CUFF_HEALTHY_PERI_R_RISE_FRAC: Final = (0.013, 0.029)
"""ENG-band peri-R envelope rise on a healthy cuff, (min, max) fraction."""

CUFF_FAILED_PERI_R_RISE_FRAC: Final = (6.88, 8.67)
"""Same on the failed cuff - two orders of magnitude larger."""

# --- measured elsewhere in the spec, needed by more than one task ----------

TRIPOLE_SIGMA_REDUCTION: Final = (2.5, 2.8)
"""Measured sigma(V) / sigma(T), (min, max). **Not** the ~6x originally assumed."""

TRIPOLE_NAIVE_WEIGHTS: Final = (0.5, 0.5)
"""``(a, b)`` in ``T = a*V1 + b*V3 - V2``. Measured 2026-09-19: do not fit."""

SIGMA_BORROWING_COST: Final = (1136, 19624)
"""(true, false) detections when one signal's sigma is applied to another.

Hard invariant 3 exists because of this measurement.
"""

MAD_TO_SIGMA: Final = 1.4826
"""Consistency constant: ``sigma_hat = MAD_TO_SIGMA * MAD`` for Gaussian data."""
