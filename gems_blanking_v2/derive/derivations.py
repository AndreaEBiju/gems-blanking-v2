"""Derived signals: the raw contacts, the tripole ``T``, and the stomach reference.

Both representations are kept because they do different jobs. The raw contacts
carry the artifact evidence and the inter-contact delay; the tripole is what spike
detection reads.

    T = a*V1 + b*V3 - V2,   subject to a + b = 1

**The applied weights are the naive 0.5/0.5. Do not fit.** Measured 2026-09-19 on
animal J: on cuff L both variance-minimising fits went degenerate (``a -> 1.0``,
i.e. the tripole collapsed to a bipolar) and gave sigma(T) = 3.35 uV against
2.92 uV for 0.5/0.5; on cuff R fitted and naive differ by under 1%. A bounded
search over ``a`` in [0.2, 0.8] lands on 0.50 and 0.59, no better than naive.
Minimising 20-300 Hz variance reduces *that* band 32-36x but does not improve, and
can degrade, the 300-3000 Hz noise floor - which is the band that matters.

The fit is still **computed, as a diagnostic**, because the acceptance criterion
asks for it: ``(a, b)`` should land near (0.5, 0.5) and be stable across sessions
for one animal, and drift across weeks is an electrode-degradation signal (task
15). Reporting only the applied weights would make that check vacuous - they are
0.5/0.5 by construction and could never drift. So each cuff reports both, and
which one was applied is never in doubt: see :class:`CuffWeights`.

**Do not quote a sigma-reduction factor as a constant.** Both figures this
document has carried - ~6x and 2.5-2.8x - are wrong as constants: the ratio is a
property of how much common mode a recording contains, not of the tripole.
Holding the contact gains fixed and sweeping only the common-mode amplitude moves
it from under 2 to nearly 10. So it is measured per recording and reported as a QC
number by :func:`sigma_reduction`; nothing downstream may depend on a fixed value.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, decimate, sosfiltfilt

from gems_blanking_v2.constants import MAD_TO_SIGMA
from gems_blanking_v2.types import ChannelInfo, Recording

__all__ = [
    "FIT_BAND_HZ",
    "MAD_TO_SIGMA",
    "NAIVE_WEIGHTS",
    "CuffWeights",
    "StomachReference",
    "build_derivations",
    "build_stomach_reference",
    "fit_tripole_weight",
    "robust_sigma",
    "sigma_reduction",
    "tripole",
]

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]

NAIVE_WEIGHTS: Final[tuple[float, float]] = (0.5, 0.5)
"""The weights actually applied. Measured: fitting is not an improvement."""

FIT_BAND_HZ: Final[tuple[float, float]] = (20.0, 300.0)
"""Band the diagnostic fit minimises variance over: motion dominates, neural
content is minimal."""

FIT_PLAUSIBLE_RANGE: Final[tuple[float, float]] = (0.2, 0.8)
"""A fitted ``a`` outside this collapsed the tripole toward a bipolar.

Measured on cuff L, where the fit went to ``a -> 1.0``. Used to mark a fit
degenerate so a drift log does not read degeneracy as drift.
"""

MIN_FIT_SECONDS: Final = 1.0
"""Shortest finite run the diagnostic fit will use. Below this it reports ``nan``."""

DECIMATE_TARGET_HZ: Final = 2000.0
"""Decimate toward this before the 20-300 Hz filter.

``CLAUDE.md``: decimate before any sub-100 Hz filtering. A 20 Hz corner at
24.4 kHz is a normalised frequency of 1.6e-3; decimating to ~2.4 kHz first puts
both corners comfortably mid-band.
"""

StomachReference = Literal["hardware", "common_average", "single_channel"]
"""How ``stomach_ref`` was produced. Recorded, never assumed."""


@dataclass(frozen=True, slots=True)
class CuffWeights:
    """The tripole weights for one cuff: what was applied, and what a fit says.

    Attributes
    ----------
    a, b
        The weights **actually applied** to produce ``T``. Naive 0.5/0.5 for an
        independent cuff; ``nan`` for a hardware tripole, which arrives pre-formed.
    fitted_a, fitted_b
        The diagnostic fit, for the drift check in the acceptance criterion. Never
        applied. ``nan`` when there was not enough finite signal to fit, or for a
        hardware tripole.
    source
        Which path produced ``T``.

    The task's signature says this map is ``{cuff_id: (a, b)}``. A pair cannot carry
    both the applied and the fitted weights, and the acceptance criterion needs the
    fitted ones while the algorithm mandates the applied ones - so this is a record
    rather than a tuple, and which is which is explicit.
    """

    a: float
    b: float
    fitted_a: float
    fitted_b: float
    source: Literal["naive", "hw_tripole"]

    def __post_init__(self) -> None:
        """Hold ``a + b == 1`` exactly, to floating point."""
        if not math.isnan(self.a) and (self.a + self.b) != 1.0:
            msg = f"a + b must be exactly 1, got {self.a} + {self.b} = {self.a + self.b}"
            raise ValueError(msg)

    @property
    def degenerate(self) -> bool:
        """Whether the *fit* collapsed the tripole toward a bipolar.

        True when ``fitted_a`` falls outside :data:`FIT_PLAUSIBLE_RANGE`. This is a
        statement about the fit, not about the applied weights, which are fixed.
        """
        if math.isnan(self.fitted_a):
            return False
        lo, hi = FIT_PLAUSIBLE_RANGE
        return not (lo <= self.fitted_a <= hi)


def robust_sigma(x: F64) -> float:
    """Return ``1.4826 * MAD`` of the finite samples, in the units of ``x``.

    The same estimator the synthetic generators use, and for the same reason: these
    traces contain the outliers whose size is being measured, so a standard
    deviation would be pulled around by them. Returns ``nan`` when nothing is
    finite. Task 06 may want this lifted into ``bands/`` when it needs it too.
    """
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    mad = float(np.median(np.abs(finite - np.median(finite))))
    return MAD_TO_SIGMA * mad


def sigma_reduction(signals: Mapping[str, F64], cuff_id: str) -> float:
    """Return ``sigma(V1) / sigma(T)`` for one cuff - a **QC number, not a constant**.

    This is what the tripole bought on *this* recording. It is a property of how
    much common mode the recording contained, not of the derivation: with the
    contact gains fixed, sweeping only the common-mode amplitude moves it from under
    2 to nearly 10. Report it, watch it across sessions, and never let a threshold
    depend on it.

    Returns ``nan`` for a hardware tripole, which has no contacts to compare
    against.
    """
    v1 = signals.get(f"{cuff_id}_V1")
    tripole_trace = signals.get(f"{cuff_id}_T")
    if v1 is None or tripole_trace is None:
        return float("nan")
    denominator = robust_sigma(tripole_trace)
    if not math.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return robust_sigma(v1) / denominator


def tripole(v1: F64, v2: F64, v3: F64, a: float = 0.5, b: float | None = None) -> F64:
    """Return ``a*V1 + b*V3 - V2``, with ``b`` defaulting to ``1 - a``.

    NaN propagates, which is correct: a masked contact masks the tripole at that
    sample, and inventing a value there would be fabricating a sample (invariant 8).
    """
    weight_b = (1.0 - a) if b is None else b
    if (a + weight_b) != 1.0:
        msg = f"a + b must be exactly 1, got {a} + {weight_b}"
        raise ValueError(msg)
    return a * v1 + weight_b * v3 - v2


def _longest_finite_run(*traces: F64) -> slice | None:
    """Return the longest span where every trace is finite, or None if none is."""
    finite = np.ones(traces[0].size, dtype=bool)
    for trace in traces:
        finite &= np.isfinite(trace)
    if not finite.any():
        return None
    padded = np.concatenate(([False], finite, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[::2], edges[1::2]
    widest = int(np.argmax(stops - starts))
    return slice(int(starts[widest]), int(stops[widest]))


def _band_limited(trace: F64, fs: float, band: tuple[float, float]) -> tuple[F64, float]:
    """Decimate then band-limit, returning ``(filtered, fs_out)``.

    Decimation first, per the convention: a 20 Hz corner at 24.4 kHz is a
    normalised frequency of 1.6e-3, and ``sos`` is asked to work far from that.
    """
    factor = max(int(fs // DECIMATE_TARGET_HZ), 1)
    if factor > 1:
        trace = np.asarray(decimate(trace, factor, ftype="fir", zero_phase=True), dtype=np.float64)
        fs = fs / factor
    lo, hi = band
    nyq = fs / 2
    sos = butter(4, [lo / nyq, min(hi, nyq * 0.9) / nyq], btype="bandpass", output="sos")
    return np.asarray(sosfiltfilt(sos, trace), dtype=np.float64), fs


def fit_tripole_weight(
    v1: F64, v2: F64, v3: F64, fs: float, band: tuple[float, float] = FIT_BAND_HZ
) -> float:
    """Return the ``a`` minimising ``var(T)`` in ``band`` - a diagnostic only.

    Closed form, not an optimiser. With ``b = 1 - a``::

        T(a) = a*(V1 - V3) + (V3 - V2) = a*d + c

    which is a one-parameter least-squares problem, minimised at
    ``a* = -cov(d, c) / var(d)``. The task says to solve it in closed form or with
    a 1-D minimiser and not to reach for a general optimiser; this is the closed
    form.

    Returns ``nan`` when there is less than :data:`MIN_FIT_SECONDS` of jointly
    finite signal, or when ``V1`` and ``V3`` are identical so ``d`` has no variance.
    """
    span = _longest_finite_run(v1, v2, v3)
    if span is None or (span.stop - span.start) < MIN_FIT_SECONDS * fs:
        return float("nan")

    d_band, _ = _band_limited(v1[span] - v3[span], fs, band)
    c_band, _ = _band_limited(v3[span] - v2[span], fs, band)

    var_d = float(np.var(d_band))
    if var_d <= 0.0:
        return float("nan")
    covariance = float(np.mean((d_band - d_band.mean()) * (c_band - c_band.mean())))
    return -covariance / var_d


def _cuff_contacts(rec: Recording, cuff_id: str) -> dict[int, ChannelInfo]:
    """Return ``{contact_index: channel}`` for one cuff."""
    return {
        c.contact_index: c
        for c in rec.channels
        if c.cuff_id == cuff_id and c.contact_index is not None
    }


def build_stomach_reference(
    rec: Recording, config: str | None = None
) -> tuple[F64 | None, StomachReference | None]:
    """Return ``(stomach_ref, how)``, or ``(None, None)`` when there is no stomach.

    The old cohort was referenced in TDT hardware, so its stomach channel is passed
    through. The new cohort is raw and has to be referenced here, and **the design
    does not say against what** - so a common average across the stomach array is
    used and the choice is recorded rather than hidden.

    A warning goes with it, because the choice is not neutral: the gastric slow wave
    is largely common across the array, so a common average removes part of the very
    signal the ``slow_wave`` and ``mmc`` consumers read. If the intended reference
    was a particular electrode, that is a different signal and this must be changed.
    """
    if config is None:
        config = rec.channels[0].config if rec.channels else "independent"
    stomach = [c for c in rec.channels if c.role == "stomach"]
    if not stomach:
        return None, None

    columns = np.column_stack([rec.data[:, c.index] for c in stomach])
    if config == "hw_tripole":
        return np.asarray(columns[:, 0], dtype=np.float64), "hardware"
    if len(stomach) == 1:
        return np.asarray(columns[:, 0], dtype=np.float64), "single_channel"

    log.warning(
        "animal %s: stomach_ref built as a common average over %d channels, because "
        "the design does not state what the new cohort is referenced against. A "
        "common average attenuates signal that is shared across the array - which "
        "the gastric slow wave largely is - so confirm the intended reference "
        "before the 0-2 and 2-50 Hz consumers are trusted.",
        rec.animal,
        len(stomach),
    )
    reference = np.nanmean(columns, axis=1)
    return np.asarray(columns[:, 0] - reference, dtype=np.float64), "common_average"


def build_derivations(
    rec: Recording,
) -> tuple[dict[str, F64], dict[str, CuffWeights]]:
    """Build every derived signal for one recording.

    Returns
    -------
    signals
        ``{name: trace}``, cuff-prefixed: ``L_V1``, ``L_V2``, ``L_V3``, ``L_T``,
        ``R_...``, plus ``stomach_ref`` when the recording has stomach channels.
        The raw contacts are kept alongside ``T`` because detection reads them -
        the tripole is *defined* by removal of the common mode, which is the best
        artifact evidence there is (invariant 6).
    weights
        ``{cuff_id: CuffWeights}``. ``a``/``b`` are what was applied; ``fitted_a``/
        ``fitted_b`` are the diagnostic for the drift check.

    Raises
    ------
    ValueError
        If a cuff has some but not all of contacts 1-3. A partial cuff cannot form
        a tripole, and quietly skipping it would drop a nerve from the analysis.
    """
    signals: dict[str, F64] = {}
    weights: dict[str, CuffWeights] = {}

    config = rec.channels[0].config if rec.channels else "independent"
    cuffs = sorted({c.cuff_id for c in rec.channels if c.cuff_id is not None})

    for cuff in cuffs:
        contacts = _cuff_contacts(rec, cuff)

        if config == "hw_tripole" or not contacts:
            # The tripole arrives pre-formed and the contacts are unrecoverable.
            nerve = [
                c for c in rec.channels if c.cuff_id == cuff and c.role == "nerve"
            ]
            if not nerve:
                continue
            signals[f"{cuff}_T"] = np.asarray(rec.data[:, nerve[0].index], dtype=np.float64)
            weights[cuff] = CuffWeights(
                a=float("nan"),
                b=float("nan"),
                fitted_a=float("nan"),
                fitted_b=float("nan"),
                source="hw_tripole",
            )
            continue

        missing = {1, 2, 3} - set(contacts)
        if missing:
            msg = (
                f"cuff {cuff!r} has contacts {sorted(contacts)} and is missing "
                f"{sorted(missing)}; a tripole needs all three. Fix the channel map "
                "rather than dropping the cuff."
            )
            raise ValueError(msg)

        v1 = np.asarray(rec.data[:, contacts[1].index], dtype=np.float64)
        v2 = np.asarray(rec.data[:, contacts[2].index], dtype=np.float64)
        v3 = np.asarray(rec.data[:, contacts[3].index], dtype=np.float64)

        signals[f"{cuff}_V1"] = v1
        signals[f"{cuff}_V2"] = v2
        signals[f"{cuff}_V3"] = v3

        applied_a, applied_b = NAIVE_WEIGHTS
        signals[f"{cuff}_T"] = tripole(v1, v2, v3, applied_a, applied_b)

        fitted_a = fit_tripole_weight(v1, v2, v3, rec.fs)
        record = CuffWeights(
            a=applied_a,
            b=applied_b,
            fitted_a=fitted_a,
            fitted_b=float("nan") if math.isnan(fitted_a) else 1.0 - fitted_a,
            source="naive",
        )
        weights[cuff] = record
        if record.degenerate:
            log.warning(
                "animal %s cuff %s: the diagnostic fit gives a = %.3f, outside "
                "%s - the tripole would collapse toward a bipolar. The applied "
                "weights are unaffected (0.5/0.5); treat this as an electrode "
                "asymmetry signal, not as drift.",
                rec.animal,
                cuff,
                fitted_a,
                FIT_PLAUSIBLE_RANGE,
            )

    for cuff in cuffs:
        ratio = sigma_reduction(signals, cuff)
        if math.isfinite(ratio):
            log.info(
                "animal %s session %s cuff %s: sigma(V1)/sigma(T) = %.2f. QC only - "
                "this is how much common mode this recording contained, not a "
                "property of the tripole, so nothing downstream may depend on it.",
                rec.animal,
                rec.session,
                cuff,
                ratio,
            )

    stomach_ref, how = build_stomach_reference(rec, config)
    if stomach_ref is not None and how is not None:
        signals["stomach_ref"] = stomach_ref
        # "Record which path was taken" - the two-tuple return has no slot for it,
        # so it is logged here and `build_stomach_reference` is public, which is
        # what task 15 calls to put it in provenance.
        log.info(
            "animal %s session %s: stomach_ref built by %s over %d channel(s)",
            rec.animal,
            rec.session,
            how,
            sum(1 for c in rec.channels if c.role == "stomach"),
        )

    return signals, weights
