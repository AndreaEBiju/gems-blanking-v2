"""Band envelopes on the shared 10 ms grid, and the epoch precondition they need.

One band, one signal, one epoch at a time. The envelope is the RMS of the
band-limited trace over the band's own analysis window, sampled at the frame centres
of the 10 ms grid - so windows overlap, which is intended (A.2).

**This step consumes an epoch, not a file.** The reference computed downstream is a
whole-*epoch* scalar, and a ``stim_recovery`` recording that has not been split still
has the stim artifacts in it - the largest excursions anywhere in the file - which
inflate both the median and the MAD and suppress detection during recovery. That is
the failure the 03B ordering exists to prevent, so :func:`analysis_epoch` makes it a
checked precondition rather than an assumption: the dependency becomes a contract,
and a baseline recording needs no split at all.

**Decimate before filtering, and it matters for more than stability.** A 1 Hz corner
at 24.4 kHz is a normalised frequency of 8e-5, where ``butter(..., 'ba')`` returns
numerical garbage - during design this produced a MAD-sigma of 1e208. That is what
:func:`assert_filter_sane` guards. But decimation also fixes the *accuracy* of the
envelope: filtering 2-50 Hz at the full rate gives a measured relative variance 2.65x
the ``2*B*T`` prediction, an effective dof of 11 against 29.8, while decimating first
brings every band within 11% of its own spec dof. Step 1 of the task frames
decimation as a stability measure; it is both.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, decimate, sosfiltfilt

from gems_blanking_v2.constants import BANDS, GRID_S, BandSpec, n_frames
from gems_blanking_v2.types import Recording

__all__ = [
    "DECIMATE_TARGET_HZ",
    "ENVELOPE_FLOOR_UV",
    "FILTER_SANITY_GAIN",
    "MIN_SEGMENT_CYCLES",
    "SETTLING_WINDOWS",
    "AnalysisEpoch",
    "FilterSanityError",
    "Segment",
    "analysis_epoch",
    "assert_filter_sane",
    "band_envelope",
    "band_envelope_for",
    "log_envelope",
    "settling_s",
    "valid_segments",
]

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]

DECIMATE_TARGET_HZ: Final = 2000.0
"""Decimate toward this before filtering any band whose corners fit under it.

Keeps every corner at a workable normalised frequency and, measured, keeps the
envelope's effective degrees of freedom within 11% of the band's spec value.
"""

FILTER_SANITY_GAIN: Final = 1e3
"""Largest output-to-input amplitude ratio a band pass may produce before it is
treated as numerically broken.

A band pass cannot amplify: a correct one has a peak gain of 1. Three orders of
magnitude is far past any legitimate transient and far below the 1e208 an
ill-conditioned transfer-function design produced, so the guard catches garbage
without firing on a sharp filter ringing at an edge.
"""

ENVELOPE_FLOOR_UV: Final = 1e-12
"""Floor applied before taking logs, microvolts.

``log(0)`` is ``-inf`` and would poison the median and the MAD. A floor twelve orders
below a microvolt cannot affect any real measurement, and a channel quiet enough to
reach it is dead and gated separately.
"""

SETTLING_WINDOWS: Final = 1.0
"""Filter settling time at a segment edge, in analysis windows.

**Measured here, and it answers a question 03B had to leave open.** ``sosfiltfilt``
leaves a transient at each end of a segment, and in the slow bands it is large enough
to dominate the whole-epoch envelope statistics. Effective degrees of freedom over
8 seeds of 10-minute white noise, against each band's own spec value:

==========  ===========  ==================  ========
band        untrimmed    one window trimmed  spec dof
==========  ===========  ==================  ========
300-3000    135.3        135.3               135.0
100-300     30.5         30.6                30.0
10-150      28.5         28.5                28.0
2-50        27.8         29.9                29.8
**0.5-3**   **22.9**     **32.0**            30.0
**0-2**     **26.3**     **31.7**            30.0
==========  ===========  ==================  ========

Untrimmed, individual seeds in the 0.5-3 band came back with an effective dof as low
as 0.7 - the transient alone accounting for more envelope variance than the signal.
Trimming exactly one analysis window from each end removes it, the fast bands are
untouched, and every band lands within 7% of its own dof.

So the settling width for *this* filter is one analysis window, per band, and
:func:`settling_s` reports it. 03B carries ``unassessable_head_s = None`` because
task 13 owns the general number; this is the envelope's part of it, measured rather
than assumed, and a caller can hand it to ``Epoch.with_settling``.
"""

MIN_SEGMENT_CYCLES: Final = 3.0
"""Shortest usable segment, in cycles of the band's lowest frequency.

A segment shorter than this cannot support an envelope of the thing being measured -
at 0.5 Hz, three cycles is 6 s - so it is marked unassessable rather than producing a
number. The rule is stated in cycles rather than seconds because it means the same
thing in every band.
"""


class FilterSanityError(RuntimeError):
    """A filter returned output that cannot be a band-limited version of its input."""


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous run of finite samples, in sample indices into the epoch.

    Called a *segment*, not an epoch. "Epoch" already means two other things in this
    pipeline - ``Condition.epoch`` is baseline or stim_recovery, and
    ``stim_split.Epoch`` is the stim or recovery slice - and this is neither.

    Attributes
    ----------
    start, stop
        ``[start, stop)`` in samples.
    assessable
        False when the segment is shorter than :data:`MIN_SEGMENT_CYCLES` of the
        band's lowest frequency, in which case its frames carry ``nan`` rather than a
        number computed from too little data.
    """

    start: int
    stop: int
    assessable: bool

    @property
    def n_samples(self) -> int:
        """Length of the segment in samples."""
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class AnalysisEpoch:
    """A recording span this step is allowed to compute a reference over.

    The only way to get one is :func:`analysis_epoch`, which refuses an unsplit
    ``stim_recovery`` recording. Holding one is therefore evidence that the
    precondition was checked, rather than a promise that it was.

    Attributes
    ----------
    recording
        The span. May be a read-only view from :mod:`gems_blanking_v2.io.stim_split`
        (hard invariant 17).
    source
        ``"baseline"`` when the whole recording is the epoch, ``"recovery"`` or
        ``"stim"`` when it came from a split. Recorded so provenance can show which.
    t0_offset_s
        Where this epoch starts in the original recording, seconds.
    """

    recording: Recording
    source: str
    t0_offset_s: float = 0.0

    @property
    def duration_s(self) -> float:
        """Duration of the epoch in seconds."""
        return float(self.recording.data.shape[0] / self.recording.fs)


def analysis_epoch(
    recording: Recording,
    *,
    condition_epoch: str,
    source: str = "baseline",
    t0_offset_s: float = 0.0,
) -> AnalysisEpoch:
    """Wrap ``recording`` as an epoch this step may analyse, or refuse it.

    Parameters
    ----------
    recording
        The span to analyse.
    condition_epoch
        This recording's :attr:`gems_blanking_v2.io.conditions.Condition.epoch` -
        ``"baseline"``, ``"stim_recovery"`` or ``"unknown"``.
    source
        ``"baseline"``, or ``"recovery"`` / ``"stim"`` when the caller has already
        split. Passing a split source is what allows a ``stim_recovery`` condition
        through.
    t0_offset_s
        Where the span starts in the original recording, seconds.

    Raises
    ------
    ValueError
        If the condition is ``stim_recovery`` and ``source`` is still ``"baseline"``,
        i.e. the file has not been split; or if the condition is ``unknown``, which is
        not a licence to proceed - an unclassified recording might be either, and
        guessing costs a contaminated reference.
    """
    if condition_epoch == "stim_recovery" and source == "baseline":
        msg = (
            f"{recording.path.name} has condition stim_recovery and has not been "
            "split. Computing a reference over the whole file puts the stim "
            "artifacts - the largest excursions in the recording - into the median "
            "and the MAD, which suppresses detection during recovery. Split it with "
            "io.stim_split.split_stim_recovery and pass the recovery epoch."
        )
        raise ValueError(msg)
    if condition_epoch not in {"baseline", "stim_recovery"}:
        msg = (
            f"{recording.path.name} has condition epoch {condition_epoch!r}, which is "
            "neither baseline nor stim_recovery. An unclassified recording is not a "
            "licence to proceed: it might need splitting, and guessing costs a "
            "contaminated reference. Classify it first (task 03A)."
        )
        raise ValueError(msg)
    return AnalysisEpoch(recording=recording, source=source, t0_offset_s=t0_offset_s)


def assert_filter_sane(x: F64, y: F64, name: str) -> None:
    """Raise when ``y`` cannot be a band-limited version of ``x``.

    The explicit guard step 1 of the task asks for. A band pass has a peak gain of 1,
    so its output cannot exceed its input by more than a transient factor, and it
    cannot contain non-finite samples that the input did not have.

    This is checked on the **output range** rather than on the filter's poles because
    that is what fails observably: the ill-conditioned design that motivated it
    returned finite coefficients and produced a MAD-sigma of 1e208.

    Raises
    ------
    FilterSanityError
        If the output has non-finite samples the input did not, or exceeds the input's
        peak amplitude by more than :data:`FILTER_SANITY_GAIN`.
    """
    finite_in = np.isfinite(x)
    if not bool(np.isfinite(y[np.isfinite(y)]).all()) or (
        int(np.count_nonzero(~np.isfinite(y))) > int(np.count_nonzero(~finite_in))
    ):
        msg = (
            f"{name}: the filter produced non-finite output from finite input - the "
            "design is numerically unstable at these corners. Decimate first."
        )
        raise FilterSanityError(msg)

    peak_in = float(np.max(np.abs(x[finite_in]))) if bool(finite_in.any()) else 0.0
    peak_out = float(np.max(np.abs(y))) if y.size else 0.0
    if peak_in > 0.0 and peak_out > FILTER_SANITY_GAIN * peak_in:
        msg = (
            f"{name}: the filter output peaks at {peak_out:.3g} from an input peaking "
            f"at {peak_in:.3g}, a gain of {peak_out / peak_in:.3g}. A band pass cannot "
            "amplify; this design is numerically unstable at these corners. "
            "Decimate first, and design with output='sos'."
        )
        raise FilterSanityError(msg)


def _decimate_for(x: F64, fs: float, hi_hz: float) -> tuple[F64, float]:
    """Decimate toward :data:`DECIMATE_TARGET_HZ` when the band's top allows it."""
    if hi_hz >= DECIMATE_TARGET_HZ / 2.0:
        return x, fs
    factor = max(int(fs // DECIMATE_TARGET_HZ), 1)
    if factor == 1:
        return x, fs
    return np.asarray(decimate(x, factor, ftype="fir", zero_phase=True)), fs / factor


def _band_limit(x: F64, fs: float, lo_hz: float, hi_hz: float, name: str) -> F64:
    """Band-limit with ``sos`` and check the result is not numerical garbage."""
    nyquist = fs / 2.0
    hi = min(hi_hz, 0.9 * nyquist)
    if lo_hz <= 0.0:
        sos = butter(4, hi / nyquist, btype="lowpass", output="sos")
    else:
        sos = butter(4, [lo_hz / nyquist, hi / nyquist], btype="bandpass", output="sos")
    y = np.asarray(sosfiltfilt(sos, x), dtype=np.float64)
    assert_filter_sane(x, y, name)
    return y


def settling_s(window_s: float) -> float:
    """Return the filter settling width at a segment edge, seconds.

    One analysis window - see :data:`SETTLING_WINDOWS` for the measurement. Frames
    inside it come back ``nan`` from :func:`band_envelope` rather than carrying a
    number computed partly from a transient.
    """
    return SETTLING_WINDOWS * window_s


def valid_segments(x: F64, fs: float, lo_hz: float, window_s: float) -> list[Segment]:
    """Split ``x`` into contiguous finite runs, marking the too-short ones.

    Interpolating across a NaN gap is fine where the gap is short relative to the
    band's period - 30 ms at 300-3000 Hz is a fraction of a cycle. In the slow bands
    it is not: a 1 s gap is half a cycle of a 0.5 Hz signal, and interpolating it
    invents the very thing being measured. So the slow bands are processed segment by
    segment, and a segment shorter than :data:`MIN_SEGMENT_CYCLES` of the band's
    lowest frequency is marked unassessable rather than producing a number.

    Parameters
    ----------
    x
        The signal, with NaN marking invalid samples.
    fs
        Sample rate, Hz.
    lo_hz
        The band's lower corner.
    window_s
        The band's analysis window, which is a second floor: a segment that cannot
        fill one window cannot produce an envelope sample either. It is also the
        *only* floor for a DC-coupled band, where ``lo_hz = 0`` means there is no
        lowest cycle to count three of.
    """
    finite = np.isfinite(x)
    padded = np.concatenate([[False], finite, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1).tolist()
    stops = np.flatnonzero(edges == -1).tolist()

    # A segment must hold its settling edges plus at least one clear window between
    # them, or every frame in it would be inside a transient.
    by_cycles = MIN_SEGMENT_CYCLES / lo_hz if lo_hz > 0.0 else 0.0
    minimum = int(math.ceil(max(by_cycles, window_s * (1.0 + 2.0 * SETTLING_WINDOWS)) * fs))
    return [
        Segment(start=start, stop=stop, assessable=(stop - start) >= minimum)
        for start, stop in zip(starts, stops, strict=True)
    ]


def _rms_at(y: F64, fs: float, window_s: float, centres_s: F64) -> F64:
    """Return the RMS of ``y`` over a window centred on each time in ``centres_s``.

    ``centres_s`` is in seconds from the start of ``y``, so a caller working on a
    segment converts the global frame centres once and this function never needs to
    know where the segment sits.

    One cumulative sum rather than a loop, so this is O(n) in the samples and never
    materialises a window-by-frame matrix - which is what keeps a 25 minute 9 channel
    file inside memory. A window is truncated at the ends of ``y`` rather than padded,
    so no fabricated sample enters an envelope (invariant 8).
    """
    half = window_s * fs / 2.0
    centres = centres_s * fs
    lo = np.clip(np.round(centres - half).astype(np.int64), 0, y.size)
    hi = np.clip(np.round(centres + half).astype(np.int64), 0, y.size)
    hi = np.maximum(hi, lo + 1)

    cumulative = np.concatenate([[0.0], np.cumsum(y**2, dtype=np.float64)])
    mean_square = (cumulative[hi] - cumulative[lo]) / (hi - lo)
    return np.asarray(np.sqrt(np.maximum(mean_square, 0.0)), dtype=np.float64)


def band_envelope(
    x: F64,
    fs: float,
    lo_hz: float,
    hi_hz: float,
    window_s: float,
    grid_s: float = GRID_S,
) -> F64:
    """Return the RMS envelope of one band on the shared grid, in microvolts.

    Decimates toward :data:`DECIMATE_TARGET_HZ` when the band allows it, band-limits
    with ``sos``, checks the filter output, then takes the RMS over the band's own
    window centred on each frame centre. Windows overlap and may be much longer than
    the grid step; that is intended (A.2).

    NaN handling follows the band: in the fast bands a gap is interpolated for
    filtering and the frames it touches are restored to ``nan``; in the slow bands,
    where a gap is a meaningful fraction of a cycle, the signal is processed segment
    by segment and short segments come back as ``nan``. See :func:`valid_segments`.

    Parameters
    ----------
    x
        One signal, microvolts, with ``nan`` marking invalid samples.
    fs
        Sample rate, Hz. Read from the recording, never assumed.
    lo_hz, hi_hz
        Band corners. ``lo_hz = 0`` gives a low pass.
    window_s
        The band's analysis window. Sets the effective degrees of freedom,
        ``2 * bandwidth * window``.
    grid_s
        Frame spacing. The shared 10 ms grid unless a caller has a reason.

    Returns
    -------
    numpy.ndarray
        ``(n_frames,)`` in microvolts, where ``n_frames = floor(duration / grid_s)``.
        Frames with no assessable data are ``nan``.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(fs) or fs <= 0.0:
        msg = f"fs must be positive and finite, got {fs!r}"
        raise ValueError(msg)
    if window_s <= 0.0 or grid_s <= 0.0:
        msg = f"window_s and grid_s must be positive, got {window_s!r} and {grid_s!r}"
        raise ValueError(msg)

    frames = (
        n_frames(x.size / fs)
        if grid_s == GRID_S
        else int(math.floor(x.size / fs / grid_s))
    )
    if frames <= 0:
        return np.empty(0, dtype=np.float64)

    name = f"{lo_hz:g}-{hi_hz:g} Hz"
    envelope = np.full(frames, np.nan, dtype=np.float64)
    for segment in valid_segments(x, fs, lo_hz, window_s):
        if not segment.assessable:
            continue
        piece = x[segment.start : segment.stop]
        decimated, rate = _decimate_for(piece, fs, hi_hz)
        limited = _band_limit(decimated, rate, lo_hz, hi_hz, name)

        # Only frames whose centre falls inside the segment get a value: a frame
        # centred in a gap has no data, and borrowing the nearest segment's would be
        # inventing one.
        # Skip the settling window at each edge: sosfiltfilt leaves a transient
        # there, and in the slow bands it dominates the epoch's statistics entirely.
        settle = settling_s(window_s)
        first = int(math.ceil((segment.start / fs + settle) / grid_s - 0.5))
        last = int(math.floor((segment.stop / fs - settle) / grid_s - 0.5)) + 1
        span = slice(max(first, 0), min(last, frames))
        if span.stop <= span.start:
            continue
        centres_s = (
            np.arange(span.start, span.stop, dtype=np.float64) + 0.5
        ) * grid_s - segment.start / fs
        envelope[span] = _rms_at(limited, rate, window_s, centres_s)
    return envelope


def band_envelope_for(x: F64, fs: float, band: str, grid_s: float = GRID_S) -> F64:
    """Return :func:`band_envelope` for a named band of :data:`~constants.BANDS`."""
    if band not in BANDS:
        msg = f"{band!r} is not a band; the six are {sorted(BANDS)}"
        raise ValueError(msg)
    spec: BandSpec = BANDS[band]
    return band_envelope(x, fs, spec.lo_hz, spec.hi_hz, spec.window_s, grid_s)


def log_envelope(env: F64) -> F64:
    """Return ``log(max(env, floor))`` - the quantity the reference is taken on.

    A separate, explicit step rather than something the reference and the z both do
    internally, because two places that take a log are two places that can stop
    agreeing. Envelopes are positive and right-skewed; the log is what makes the null
    symmetric, which is the whole of hard invariant 5.

    ``nan`` frames stay ``nan``: a frame with no assessable data has no log.
    """
    env = np.asarray(env, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.log(np.maximum(env, ENVELOPE_FLOOR_UV)), dtype=np.float64)
