"""Task 13: the extent of a confirmed motion event, per consumer.

One event does not have one duration: it has one per consumer, because a 200 ms
excursion destroys spike detection and is invisible to the slow-wave analysis. The
extent for a consumer is the run of frames where **that consumer's band**, on **that
consumer's signal**, exceeds **that consumer's tolerance**, padded by the consumer
chain's measured settling time. The candidate interval is never the extent.

Settling (hard invariant 19)
----------------------------
The consumer chains are MATLAB filters in ``processing_new``, different from the
detection bands. :data:`CONSUMER_FILTERS` declares each one as read from Andrea's code
(file and parameter named in ``source``), and :func:`consumer_settling` measures the
one-way impulse response of that exact design at the recording's ``fs`` (the rate
MATLAB filters at), where it falls below :data:`IMPULSE_DECAY_FRACTION` of peak -
``impz`` in Python. The method is the detection side's (``bands.envelope``) and a test
holds the two equal on every order-4 design they share; the design is generalised here
only because the slow-wave low-pass is order 2. Where a chain has a further stage that
touches the edge (slow wave's 5 s Gaussian ``smoothdata``), the settling is the
maximum of the terms. A consumer with no declared chain has settling ``None``.

Where the chain's settling at a blanked edge has been MEASURED as v2 runs it
(:data:`EDGE_SETTLING`, RULING 2026-10-09 (c)), the measurement replaces the impulse
response: her chains fill across the NaN and filter zero phase, which the one-way
impulse response understates. Spikes: 7.78 ms (``step1_bandpass``) + 2.5 ms
(``step4_waveforms`` before a gap) = 10.28 ms, against task 13's 5.1 ms. mmc: 1.16 s
measured, padded 1.5 s, against task 13's 15 s (its moving median and MAD skip NaN).

The cardiac tolerance is operational
------------------------------------
For ``hrv`` the question is not an amplitude: suppress the span, re-run the beat
detector, and ask whether the beat train changed (:func:`cardiac_operational_damage`).

Slow bands
----------
An extent is no finer than its band's envelope window (``0.5-3``: 6 s; ``0-2``: 7.5 s).
Every :class:`Extent` carries that ``resolution_s``; nothing pretends otherwise.

Tolerances are an explicit input (:class:`ToleranceTable`), never a default: they come
from task T's measured tolerance surfaces (MATLAB, step 9), with their source recorded.

Consumer table: built from ``constants.CONSUMERS`` with the two corrections the spec
already ruled and ``constants.py`` does not yet carry (:data:`STRUCK_CONSUMERS`,
:data:`CONSUMER_SIGNAL_ERRATA`); a test fails once ``constants.py`` is corrected, so
the patch is removed rather than outliving its reason.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt, unit_impulse

from gems_blanking_v2.bands.envelope import IMPULSE_DECAY_FRACTION
from gems_blanking_v2.constants import BANDS, CONSUMERS, GRID_S, ConsumerSpec
from gems_blanking_v2.physio.rpeaks import DECIMATE_TARGET_HZ, detect_rpeaks
from gems_blanking_v2.types import Event

__all__ = [
    "CONSUMER_FILTERS",
    "CONSUMER_SIGNAL_ERRATA",
    "EDGE_SETTLING",
    "MMC_NOT_MEASURED_HALF_S",
    "OUT_OF_BUILD_CONSUMERS",
    "SPIKE_STEP1_EDGE_S",
    "SPIKE_WAVEFORM_AFTER_PEAK_S",
    "SPIKE_WAVEFORM_BEFORE_PEAK_S",
    "STRUCK_CONSUMERS",
    "CardiacVerdict",
    "ConsumerFilter",
    "ConsumerSettling",
    "EdgeSettling",
    "Extent",
    "ExtentNotAssessableError",
    "NotAssessable",
    "ToleranceTable",
    "beat_train_changed",
    "cardiac_operational_damage",
    "compute_extent",
    "consumer_settling",
    "consumer_signals",
    "edge_settling_record",
    "expected_consumers",
    "extent_consumers",
    "extents_for_events",
    "hrv_extent",
    "hrv_fiducial_tolerance_s",
    "impulse_settling_s",
    "is_confirmed_motion",
    "mmc_not_measured_spans",
    "settling_provenance",
    "step_settling_s",
]

F64 = npt.NDArray[np.float64]

# ---------------------------------------------------------------------------
# the consumer table (constants.CONSUMERS with the ruled corrections)
# ---------------------------------------------------------------------------

STRUCK_CONSUMERS: Final[frozenset[str]] = frozenset({"slow_c"})
"""Struck from A.4 on 2026-09-23 (never implemented anywhere); ``constants.py`` still
lists it."""
OUT_OF_BUILD_CONSUMERS: Final[frozenset[str]] = frozenset({"velocity"})
"""Consumers with an extent but no masks in this build: velocity, because task 18 is out
(ruling 2026-10-07 (b) R5). They are still in :func:`extent_consumers`."""

CONSUMER_SIGNAL_ERRATA: Final[Mapping[str, tuple[str, ...]]] = {
    "mmc": ("ANT1", "ANT2", "ANT3"),
    "slow_wave": ("ANT1", "ANT2", "ANT3"),
}
"""Ruling 2026-10-07 (b) R8 item 2 (and 2026-10-06 item 3): ``extract_mmc.m`` reads
ANT1-3 directly; ``stomach_ref`` is a detection input only. The same holds for
``slow_wave``: the invariant-43 ruling (``IMPLEMENTATION.md``: "slow_wave and mmc read
them raw") and ``batch_process.m`` (``swData = signal(:, 3:5)``, the raw ANT1-3) agree,
though R8's text names only mmc - a spec inconsistency to flag. ``constants.py`` still
says ``stomach_ref`` for both."""


def extent_consumers() -> dict[str, ConsumerSpec]:
    """Return the consumers that get an extent, keyed by name, with the ruled corrections."""
    out: dict[str, ConsumerSpec] = {}
    for c in CONSUMERS:
        if c.name in STRUCK_CONSUMERS:
            continue
        signals = CONSUMER_SIGNAL_ERRATA.get(c.name, c.signals)
        out[c.name] = ConsumerSpec(c.name, signals, c.band, c.tolerance)
    return out


def expected_consumers() -> tuple[str, ...]:
    """Return the consumers every mask file and acceptance row must cover, sorted.

    :func:`extent_consumers` without :data:`OUT_OF_BUILD_CONSUMERS`. This is the one
    construction site for that set (invariant 33): the mask-file coverage check, the
    acceptance rows and their tests all call it, so they cannot drift apart.
    """
    return tuple(sorted(c for c in extent_consumers() if c not in OUT_OF_BUILD_CONSUMERS))


def consumer_signals(consumer: str, *, cuffs: Sequence[str] = (),
                     best_hr_channel: str | None = None) -> tuple[str, ...]:
    """Return the concrete signal names one consumer reads in one recording.

    Nerve signals are cuff-prefixed (``"L_T"``); ``best_hr_channel`` is the channel the
    HR ranking chose. A consumer that needs an input the recording lacks raises.
    """
    spec = extent_consumers()[consumer]
    names: list[str] = []
    for s in spec.signals:
        if s in ("T", "V1", "V2", "V3"):
            if not cuffs:
                msg = f"{consumer} reads per-cuff {s}; no cuffs given"
                raise ValueError(msg)
            names += [f"{cuff}_{s}" for cuff in cuffs]
        elif s == "best_hr_channel":
            if not best_hr_channel:
                msg = f"{consumer} reads the best HR channel; none was chosen"
                raise ValueError(msg)
            names.append(best_hr_channel)
        else:
            names.append(s)
    return tuple(names)


# ---------------------------------------------------------------------------
# the consumer chains and their settling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConsumerFilter:
    """One consumer's filter as Andrea's MATLAB applies it (``filtfilt``, sos form).

    ``extra_edge_s`` is any further stage that reaches across an edge (seconds), with
    its own ``extra_source``.
    """

    consumer: str
    kind: Literal["bandpass", "lowpass"]
    lo_hz: float
    hi_hz: float
    order: int
    source: str
    extra_edge_s: float = 0.0
    extra_source: str = ""


@dataclass(frozen=True, slots=True)
class EdgeSettling:
    """A consumer chain's settling at a NaN (blanked) edge, MEASURED as v2 runs the chain.

    RULING 2026-10-09 (c): the one-way impulse response (``impz``, :func:`impulse_settling_s`)
    understates the edge: her chains ``fillmissing`` across the NaN and then ``filtfilt``
    (zero phase), so the edge itself is filtered, in both directions. These figures are
    the measured settling of that, by task 13's rule (the last distance from the edge at
    which the error still exceeds 1 % of its peak), worst case over input and side.

    ``filter_s`` - the chain's filter alone, zero phase over her fill, worst side;
    ``before_gap_s`` / ``after_gap_s`` - every stage that touches the edge, on the valid
    data BEFORE a gap (the gap is ahead of it) and AFTER a gap (the recovery-start side);
    ``total_s`` - their maximum (invariant 19);
    ``pad_s`` - the pad Andrea ruled at a blanked edge (``total_s`` rounded up);
    ``extent_pad_s`` - what an extent is padded by (:func:`consumer_settling`): the ruling
    says which of the two;
    ``files`` - ``(measurement file, sha256)``, relative to the build session's scratchpad
    (where they were made; they are not in the repo), so the record names exactly the
    measurement it rests on.
    """

    consumer: str
    filter_s: float
    before_gap_s: float
    after_gap_s: float
    pad_s: float
    extent_pad_s: float
    ruling: str
    method: str
    files: tuple[tuple[str, str], ...]

    @property
    def total_s(self) -> float:
        """The worse side (invariant 19: a maximum over every stage that touches the edge)."""
        return max(self.before_gap_s, self.after_gap_s)


SPIKE_WAVEFORM_BEFORE_PEAK_S: Final = 0.0015
"""``step4_waveforms.m:44-46``: wfPreMs 1 ms + wfAlignSearchMs 0.5 ms read before an aligned
peak - the part of the waveform window that reaches back into a gap BEHIND the spike
(the recovery-start side). Measured file: ``settling.json`` ``step4_reach_ms.lead``."""
SPIKE_WAVEFORM_AFTER_PEAK_S: Final = 0.0025
"""wfPostMs 2 ms + wfAlignSearchMs 0.5 ms read after it - into a gap AHEAD of the spike
(``settling.json`` ``step4_reach_ms.trail``)."""
SPIKE_STEP1_EDGE_S: Final = 0.007782
"""``step1_bandpass`` (300-3000 Hz order 4, ``fillmissing`` + ``filtfilt``) at a NaN edge,
as v2 calls it: 7.782 ms on either side (``settling.json`` ``worst``, input
``bump_*_G21``). Task 13's one-way impulse response gave 5.1 ms."""

EDGE_SETTLING: Final[Mapping[str, EdgeSettling]] = {
    "spikes": EdgeSettling(
        "spikes", SPIKE_STEP1_EDGE_S,
        before_gap_s=SPIKE_STEP1_EDGE_S + SPIKE_WAVEFORM_AFTER_PEAK_S,
        after_gap_s=SPIKE_STEP1_EDGE_S + SPIKE_WAVEFORM_BEFORE_PEAK_S,
        pad_s=0.0105, extent_pad_s=SPIKE_STEP1_EDGE_S + SPIKE_WAVEFORM_AFTER_PEAK_S,
        ruling="RULING 2026-10-09 (c) 1: P.edgeBufferMs = 10.5 ms (invariant 19, rounded up "
               "to 0.5 ms); the measured zero-phase edge settling replaces task 13's 5.1 ms "
               "wherever spike settling is used",
        method="edgepad/edgepad_run.m runs her step1_bandpass (night6_v2_params) on "
               "synthetic, noise and real snippets with NaN gaps; edgepad/analyse_settling.py "
               "takes the last distance from the gap at which the error exceeds 1 % of its "
               "peak (task 13's rule), worst over input and side: step1 7.782 ms; step4 reads "
               "2.5 ms further before a gap and 1.5 ms after one: 10.282 ms in total",
        files=(("edgepad/settling.json",
                "908feb679d6bd378e2d2849ba9a0e706008f8c70dbcb4c0292b71ffe48e2c4e9"),
               ("edgepad/pad_savings.json",
                "3995aef33b143cac47d3e0f6f9ecd5e39eefe38bcd3c56bb195f5b4147d3b4a3"),
               ("edgepad/analyse_settling.py",
                "658b7b31f7104224ff30f5e8888e22fc7be905e92c8e447151dc0fca615d7fee"),
               ("edgepad/edgepad_run.m",
                "e1cb67e1c79a7ad3de25688129d8bfbcd5a0e879e2e2ebe5907731ac06663ef6"))),
    "mmc": EdgeSettling(
        "mmc", 1.1594, before_gap_s=1.1594, after_gap_s=1.1594, pad_s=1.5, extent_pad_s=1.5,
        ruling="RULING 2026-10-09 (c) 2: mmc padding at a blanked edge 1.5 s, replacing task "
               "13's 15 s extra_edge_s",
        method="edgepad/mmc_edge_settling.py: extract_mmc.m:103-107 (butter(4, [2 50]) -> "
               "zp2sos -> fillmissing linear -> filtfilt, NaN restored); error = filtered "
               "with the gap filled - filtered without the gap; last distance above 1 % of "
               "its peak, worst over input (step, white noise, 5 and 20 Hz sines, a burst at "
               "the edge), gap (25 ms - 10 s) and side: 1.1594 s. Her moving median and MAD "
               "(:231-232) are 'omitnan': they skip the blank and add nothing",
        files=(("edgepad/mmc_edge_settling.json",
                "321aae64d6847dc3d5acee10fe90c939dcb7fba38a47e127afb6f462f65b1f2b"),
               ("edgepad/mmc_edge_settling.py",
                "0adf4782392246fba90d50c575aa991ef68d8956ffb2e25160930399b4e974cf"))),
}
"""THE measured edge settlings (invariant 33): :func:`consumer_settling`, the recovery-start
table (``extent.recovery_start``) and the Night 6 declaration
(``matlab/night6/edge_settling.json``, held equal to :func:`edge_settling_record` by a test)
all read them from here."""


def edge_settling_record() -> dict[str, Any]:
    """Return :data:`EDGE_SETTLING` as JSON-ready data (what Night 6 declares and records).

    Seconds AND, for the spike pad, the ``P.edgeBufferMs`` value in ms that
    ``night6_v2_params`` sets, so MATLAB never converts. Files are ``[{file, sha256}]`` rows
    (MATLAB's ``jsondecode`` would mangle a file name used as a key, invariant 22).
    """
    out: dict[str, Any] = {"schema": 1, "ruling": "RULING 2026-10-09 (c) 1-2"}
    for name, e in sorted(EDGE_SETTLING.items()):
        out[name] = {"filter_s": e.filter_s, "before_gap_s": e.before_gap_s,
                     "after_gap_s": e.after_gap_s, "total_s": e.total_s, "pad_s": e.pad_s,
                     "extent_pad_s": e.extent_pad_s, "ruling": e.ruling, "method": e.method,
                     "files": [{"file": f, "sha256": h} for f, h in e.files]}
    out["spikes"]["edge_buffer_ms"] = round(EDGE_SETTLING["spikes"].pad_s * 1e3, 6)
    return out


CONSUMER_FILTERS: Final[Mapping[str, ConsumerFilter]] = {
    "spikes": ConsumerFilter(
        "spikes", "bandpass", 300.0, 3000.0, 4,
        "pipeline_params.m P.bandpassLow/High/filterOrder; step1_bandpass.m butter+filtfilt"),
    "velocity": ConsumerFilter(
        "velocity", "bandpass", 300.0, 3000.0, 4,
        "the ENG band (A.4); task 18 is out of this build (ruling (b) R5)"),
    "mmc": ConsumerFilter(
        "mmc", "bandpass", 2.0, 50.0, 4, "extract_mmc.m: butter(4, [2 50]) + filtfilt",
        extra_source="none: extract_mmc.m's moving median and MAD (:231-232, sigmaWin 30 s) "
                     "are 'omitnan', so they skip a blank and add nothing at its edge "
                     "(RULING 2026-10-09 (c) 3 (b)); task 13's 15 s half-window is withdrawn "
                     "by (c) 2 - the edge is EDGE_SETTLING['mmc']"),
    "slow_wave": ConsumerFilter(
        "slow_wave", "lowpass", 0.0, 0.15, 2,
        "batch_process.m P.sw_lowPassCutoff=0.15, P.sw_lowPassOrder=2; "
        "slowWaveAnalysis_new.m butter+filtfilt (run_continuous.m uses 2 Hz order 4)",
        extra_edge_s=2.5,
        extra_source="slowWaveAnalysis_new.m smoothdata gaussian, P.sw_smoothWindow=5 s "
                     "(half-window reaches across the edge)"),
    "hrv": ConsumerFilter(
        "hrv", "bandpass", 10.0, 150.0, 4,
        "HR_BR_HRVAnalysis_new.m hrBandHz=[10 150], order=P.hr_order=4"),
    "breathing": ConsumerFilter(
        "breathing", "bandpass", 10.0, 150.0, 4,
        "HR_BR_HRVAnalysis_new.m: breaths are troughs at the 10-150 Hz beats; no other filter"),
}


def _sos(f: ConsumerFilter, fs: float) -> npt.NDArray[np.float64]:
    nyq = fs / 2.0
    if f.kind == "lowpass":
        return np.asarray(butter(f.order, f.hi_hz / nyq, btype="lowpass", output="sos"))
    hi = min(f.hi_hz, 0.9 * nyq)  # the detection design's clamp; MATLAB clamps at nyq - 1
    return np.asarray(butter(f.order, [f.lo_hz / nyq, hi / nyq], btype="bandpass",
                             output="sos"))


def _response(f: ConsumerFilter, fs: float, *, step: bool) -> F64:
    """Return the impulse (or step) response, long enough to have decayed."""
    sos = _sos(f, fs)
    scale = f.lo_hz if f.lo_hz > 0.0 else f.hi_hz
    n = int(max(1024, round(40.0 / scale * fs)))
    for _ in range(4):
        x = np.ones(n) if step else unit_impulse(n)
        y = np.asarray(sosfilt(sos, x), dtype=np.float64)
        tail = y[-max(1, n // 10):]
        ref = 1.0 if (step and f.kind == "lowpass") else 0.0
        peak = float(np.max(np.abs(y - ref)))
        if peak > 0 and float(np.max(np.abs(tail - ref))) < IMPULSE_DECAY_FRACTION * peak:
            return y
        n *= 2
    msg = f"{f.consumer}: response had not decayed within {n / fs:.1f} s"
    raise RuntimeError(msg)


@functools.lru_cache(maxsize=64)
def impulse_settling_s(f: ConsumerFilter, fs: float) -> float:
    """Where the one-way impulse response falls below 1% of peak, seconds (``impz``)."""
    y = np.abs(_response(f, fs, step=False))
    above = np.flatnonzero(y > IMPULSE_DECAY_FRACTION * float(y.max()))
    return float(above[-1] + 1) / fs


@functools.lru_cache(maxsize=64)
def step_settling_s(f: ConsumerFilter, fs: float) -> float:
    """Independent check of :func:`impulse_settling_s`: from the STEP response.

    A different input (``ones``, not ``unit_impulse``) through the same design; the
    impulse response is recovered as the step response's first difference and its 1%
    decay located the same way. It must agree with ``impz`` to within a sample or two -
    a disagreement means the design or the decay search is wrong, not the filter.
    """
    y = _response(f, fs, step=True)
    h = np.abs(np.diff(np.concatenate(([0.0], y))))
    above = np.flatnonzero(h > IMPULSE_DECAY_FRACTION * float(h.max()))
    return float(above[-1] + 1) / fs


@dataclass(frozen=True, slots=True)
class ConsumerSettling:
    """A consumer chain's settling: each term, and the binding figure.

    ``edge`` is the chain's MEASURED zero-phase settling at a blanked edge
    (:data:`EDGE_SETTLING`) where one exists: it REPLACES the one-way impulse response,
    which measures the same filter less faithfully (RULING 2026-10-09 (c)), and the extent
    is padded by its ``extent_pad_s``. Elsewhere the settling is the maximum of the impulse
    response and any further stage that touches the edge (invariant 19).
    """

    consumer: str
    impulse_s: float
    extra_s: float
    edge: EdgeSettling | None = None

    @property
    def total_s(self) -> float:
        """The measured edge pad where there is one, else ``max`` over the stages."""
        if self.edge is not None:
            return self.edge.extent_pad_s
        return max(self.impulse_s, self.extra_s)


def consumer_settling(consumer: str, fs: float) -> ConsumerSettling | None:
    """Measured settling of ``consumer``'s chain at ``fs``; ``None`` if no chain is declared."""
    f = CONSUMER_FILTERS.get(consumer)
    if f is None:
        return None
    return ConsumerSettling(consumer, impulse_settling_s(f, float(fs)), f.extra_edge_s,
                            EDGE_SETTLING.get(consumer))


def settling_provenance(fs: float) -> dict[str, Any]:
    """Every consumer's settling at ``fs`` as JSON-ready provenance, with its basis.

    Per consumer: ``total_s`` (what an extent is padded by), the impulse and extra terms,
    and, where the edge was measured, the :func:`edge_settling_record` entry (ruling,
    method and the measurement files' SHA-256). Missing values are absent keys. A mask
    writer records this beside ``MaskProvenance.settling_s`` so the numbers carry their
    measurement.
    """
    rec = edge_settling_record()
    out: dict[str, Any] = {}
    for name in sorted(CONSUMER_FILTERS):
        s = consumer_settling(name, fs)
        assert s is not None
        row: dict[str, Any] = {"total_s": s.total_s, "impulse_s": s.impulse_s,
                               "extra_s": s.extra_s}
        if s.edge is not None:
            row["edge"] = rec[name]
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToleranceTable:
    """Per-consumer tolerance in that consumer band's robust z, with its source.

    No defaults: the numbers come from task T's measurement and are recorded with it.
    ``hrv`` is operational (:func:`cardiac_operational_damage`) and carries no z.
    """

    z_tol: Mapping[str, float]
    source: str

    def __post_init__(self) -> None:
        """Validate every entry and the source."""
        if not self.source:
            msg = "a tolerance table must name its source (task T's measurement)"
            raise ValueError(msg)
        known = set(extent_consumers())
        for name, value in self.z_tol.items():
            if name not in known:
                msg = f"tolerance for unknown consumer {name!r}; known: {sorted(known)}"
                raise ValueError(msg)
            if name == "hrv":
                msg = "hrv's tolerance is operational (beat train unchanged), not a z"
                raise ValueError(msg)
            if not (math.isfinite(value) and value > 0):
                msg = f"tolerance for {name} must be finite and positive, got {value}"
                raise ValueError(msg)

    def for_consumer(self, consumer: str) -> float:
        """Return the consumer's tolerance; raises naming it when the table lacks one."""
        if consumer not in self.z_tol:
            msg = f"no tolerance for {consumer!r} in table from {self.source}"
            raise KeyError(msg)
        return float(self.z_tol[consumer])

    @classmethod
    def from_json(cls, path: Path) -> ToleranceTable:
        """Read ``{"source": ..., "z_tol": {consumer: z}}``."""
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("source", "z_tol"):
            if doc.get(key) is None:
                msg = f"{Path(path).name}: required field {key!r} is absent"
                raise ValueError(msg)
        return cls({str(k): float(v) for k, v in doc["z_tol"].items()}, str(doc["source"]))


# ---------------------------------------------------------------------------
# extent
# ---------------------------------------------------------------------------


def is_confirmed_motion(event: Event, *, p_threshold: float | None = None,
                        calibrator: str | None = None) -> bool:
    """Whether ``event`` is a confirmed motion event.

    A human ``motion`` judgement confirms it. A model's needs ``p_motion`` at or above
    ``p_threshold`` from a CALIBRATED model (``calibrator`` named): thresholding raw
    LightGBM output is mode-dependent, so an uncalibrated score never confirms.
    """
    if event.source == "human":
        return event.judgement == "motion"
    if event.source == "model":
        if not calibrator:
            msg = "P(motion) must come from a calibrated model (task 12); no calibrator named"
            raise ValueError(msg)
        if p_threshold is None:
            msg = "a model-scored event needs an explicit p_threshold"
            raise ValueError(msg)
        return bool(np.isfinite(event.p_motion) and event.p_motion >= p_threshold)
    return False


@dataclass(frozen=True, slots=True)
class Extent:
    """One event's extent for one consumer on one signal, seconds, half-open.

    ``core_*`` is where the band exceeded the tolerance; ``start_s``/``stop_s`` add the
    settling. ``resolution_s`` is the band's envelope window: nothing finer is real.
    """

    consumer: str
    signal: str
    band: str
    start_s: float
    stop_s: float
    core_start_s: float
    core_stop_s: float
    settling_s: float
    resolution_s: float


class ExtentNotAssessableError(ValueError):
    """The consumer's z cannot be judged anywhere in the event's span.

    The span is outside the z record, or z is NaN there. Distinct from "under
    tolerance", which is ``None``.
    """


@dataclass(frozen=True, slots=True)
class NotAssessable:
    """Record of an :class:`ExtentNotAssessableError` in :func:`extents_for_events`."""

    consumer: str
    signal: str
    reason: str


def _runs(over: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], over.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(),
                    strict=True))


def compute_extent(event: Event, z: Mapping[tuple[str, str], npt.ArrayLike], consumer: str,
                   *, signal: str, tolerances: ToleranceTable, fs: float, z_t0_s: float,
                   grid_s: float = GRID_S) -> Extent | None:
    """Return the event's extent for ``consumer`` on ``signal``; ``None`` if under tolerance.

    ``z`` maps ``(signal, band)`` to the robust z on the shared grid, whose frame 0
    starts at ``z_t0_s`` on the event's timeline - required, because detection returns z
    on its REGION's timeline (a stim/recovery file's recovery epoch) while candidates
    are on the recording's, and a silent offset would shift every extent by the stim
    epoch. The extent is the union of the runs of frames over the consumer's tolerance
    that overlap the event's span, padded each side by the consumer chain's settling and
    clipped to the z record.
    The caller passes confirmed motion events only (:func:`extents_for_events` enforces
    it). Raises :class:`ExtentNotAssessableError` when the span lies outside the z record or z
    is NaN on every frame of it - that is not "under tolerance".
    """
    if consumer == "hrv":
        msg = "hrv's extent is operational: use hrv_extent"
        raise ValueError(msg)
    spec = extent_consumers()[consumer]
    key = (signal, spec.band)
    if key not in z:
        msg = f"no z for {key} (consumer {consumer})"
        raise KeyError(msg)
    zz = np.asarray(z[key], dtype=np.float64)
    tol = tolerances.for_consumer(consumer)
    over = np.isfinite(zz) & (zz > tol)
    c = event.candidate
    i0 = int(math.floor((c.start_s - z_t0_s) / grid_s))
    i1 = int(math.ceil((c.stop_s - z_t0_s) / grid_s))
    lo_i, hi_i = max(0, i0), min(zz.size, i1)
    if hi_i <= lo_i or not np.isfinite(zz[lo_i:hi_i]).any():
        msg = (f"{consumer}/{signal}: z is not assessable in [{c.start_s:.3f}, "
               f"{c.stop_s:.3f}) s (z covers [{z_t0_s:.3f}, {z_t0_s + zz.size * grid_s:.3f}))")
        raise ExtentNotAssessableError(msg)
    hits = [(a, b) for a, b in _runs(over) if a < i1 and b > i0]
    if not hits:
        return None
    a, b = hits[0][0], hits[-1][1]
    settle = consumer_settling(consumer, fs)
    if settle is None:  # pragma: no cover - every extent consumer has a chain
        msg = f"{consumer}: settling unknown, so its extent is unknown (invariant 19)"
        raise ValueError(msg)
    pad = settle.total_s
    lo, hi = z_t0_s + a * grid_s, z_t0_s + b * grid_s
    return Extent(consumer, signal, spec.band, max(z_t0_s, lo - pad),
                  min(z_t0_s + zz.size * grid_s, hi + pad), lo, hi, pad,
                  BANDS[spec.band].window_s)


def extents_for_events(
    events: Mapping[str, Event], z: Mapping[tuple[str, str], npt.ArrayLike],
    signals: Mapping[str, Sequence[str]], *, tolerances: ToleranceTable, fs: float,
    z_t0_s: float, confirmed: Callable[[Event], bool],
    hr_signal: tuple[str, npt.ArrayLike, float] | None = None,
) -> list[tuple[str, str, str, Extent | NotAssessable | None]]:
    """Every event x consumer x signal: ``(event_id, consumer, signal, result)``.

    ``result`` is an :class:`Extent`, ``None`` (under this consumer's tolerance) or
    :class:`NotAssessable`, so the output says something about every event for every
    consumer (task 13's acceptance). ``signals`` maps consumer to the signals it reads
    here (:func:`consumer_signals`).

    ``confirmed`` decides which events are motion (normally :func:`is_confirmed_motion`
    with the calibrated model's threshold); an unconfirmed event raises - extents are
    for confirmed motion only. This is intended: an ``inherited`` judgement is never a
    confirmation by itself, so it gets an extent only if ``confirmed`` says so.

    ``hrv`` is operational and needs ``hr_signal = (name, samples, x_t0_s)`` for the
    channel ``signals["hrv"]`` names; reading hrv without it raises rather than leaving
    hrv silently unmasked.
    """
    if "hrv" in signals:
        need = tuple(signals["hrv"])
        if hr_signal is None or need != (hr_signal[0],):
            have = None if hr_signal is None else hr_signal[0]
            msg = f"hrv reads {need} but hr_signal is {have!r}: refusing to skip hrv"
            raise ValueError(msg)
    out: list[tuple[str, str, str, Extent | NotAssessable | None]] = []
    for eid, ev in events.items():
        if not confirmed(ev):
            msg = f"event {eid} is not confirmed motion; extents are for confirmed events"
            raise ValueError(msg)
        for consumer, names in signals.items():
            for sig in names:
                if consumer == "hrv":
                    assert hr_signal is not None
                    out.append((eid, consumer, sig, hrv_extent(ev, hr_signal[1], fs,
                                                               signal=sig,
                                                               x_t0_s=hr_signal[2])))
                    continue
                try:
                    res: Extent | NotAssessable | None = compute_extent(
                        ev, z, consumer, signal=sig, tolerances=tolerances, fs=fs,
                        z_t0_s=z_t0_s)
                except ExtentNotAssessableError as exc:
                    res = NotAssessable(consumer, sig, str(exc))
                out.append((eid, consumer, sig, res))
    return out


MMC_NOT_MEASURED_HALF_S: Final = 15.0
"""Ruling 2026-10-07 (b) R6: mmc output within +/-15 s of ANY blank is reported "not
measured" - the half-width of its 30 s moving threshold (2026-10-06). A sensitivity
carried beside the mask, not a fix; ``extract_mmc.m`` is unchanged.

The same derivation as task 13's withdrawn 15 s ``extra_edge_s`` (the moving threshold's
half-window), but a different use: this is a reported span, never blanked, and R6 rules
it separately. RULING 2026-10-09 (c) 2 replaces only the PADDING (now
``EDGE_SETTLING['mmc']``, 1.5 s), so this stays 15 s until Andrea rules on it; (c) 3 (b)
(the moving median and MAD skip NaN) undercuts its basis - flagged, not changed."""


def mmc_not_measured_spans(blanks_s: Sequence[tuple[float, float]], duration_s: float
                           ) -> list[tuple[float, float]]:
    """Return the spans where mmc's output is "not measured": every blank +/- 15 s, merged."""
    spans = sorted((max(0.0, a - MMC_NOT_MEASURED_HALF_S),
                    min(duration_s, b + MMC_NOT_MEASURED_HALF_S)) for a, b in blanks_s)
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# the operational cardiac test
# ---------------------------------------------------------------------------


def hrv_fiducial_tolerance_s(fs: float) -> float:
    """Return the smallest fiducial shift counted as a change: 1.5 detector samples.

    The beat detector (``physio.rpeaks``) decimates toward 2 kHz, so its fiducials are
    quantised to one decimated sample (~0.49 ms at 24.4 kHz). A shift within that cannot
    be told from quantisation; 1.5 samples (~0.74 ms) is the least that can, and stays
    under the 1-2 ms at which RMSSD and SD1 move (task 13). Provisional - Andrea's call.
    """
    factor = max(int(fs // DECIMATE_TARGET_HZ), 1)
    return 1.5 * factor / fs


@dataclass(frozen=True, slots=True)
class CardiacVerdict:
    """Whether suppressing a span changed the beat train, and how."""

    changed: bool
    n_added: int
    n_lost: int
    max_shift_s: float
    shift_tol_s: float
    n_inside_unverified: int = 0
    """Beats inside the span that could not be compared (no suppressed reference): each
    counts as a change, the conservative direction for HRV."""


def beat_train_changed(reference_s: npt.ArrayLike, test_s: npt.ArrayLike, *,
                       shift_tol_s: float, match_s: float) -> CardiacVerdict:
    """Compare two beat trains: any beat added, lost, or moved by more than ``shift_tol_s``.

    Beats pair when they are within ``match_s``; an unpaired beat is added or lost.
    """
    ref = np.sort(np.asarray(reference_s, dtype=np.float64))
    test = np.sort(np.asarray(test_s, dtype=np.float64))
    used = np.zeros(test.size, dtype=bool)
    shifts: list[float] = []
    lost = 0
    for t in ref:
        if test.size == 0:
            lost += 1
            continue
        j = int(np.argmin(np.abs(test - t) + np.where(used, np.inf, 0.0)))
        if not used[j] and abs(test[j] - t) <= match_s:
            used[j] = True
            shifts.append(abs(float(test[j] - t)))
        else:
            lost += 1
    added = int((~used).sum())
    max_shift = max(shifts) if shifts else 0.0
    changed = bool(added or lost or max_shift > shift_tol_s)
    return CardiacVerdict(changed, added, lost, max_shift, shift_tol_s)


def cardiac_operational_damage(
    x: npt.ArrayLike, fs: float, span_s: tuple[float, float], *,
    suppressed: npt.ArrayLike | None = None,  # span_s is on x's own timeline (0 = x[0])
    detector: Callable[[F64, float], Any] | None = None,
    shift_tol_s: float | None = None,
) -> CardiacVerdict:
    """Suppress the span, re-run the beat detector, and ask whether the train changed.

    ``suppressed`` is the signal with the event's contribution removed when that is
    known (a verified subtraction, or a synthetic test); then every beat is compared, and
    a 2 ms fiducial shift is a change while a 0.05 ms one is not.

    Without it the span is NaN'd - the detector interpolates for filtering only - and the
    beats outside the span are compared. A beat INSIDE the span then has no reference to
    be compared with, so it counts as changed (``n_inside_unverified``): an unverifiable
    fiducial is treated as moved, which over-masks rather than letting a shifted beat
    into HRV.
    """
    if detector is None:
        detector = detect_rpeaks
    tol = hrv_fiducial_tolerance_s(fs) if shift_tol_s is None else float(shift_tol_s)
    xx = np.asarray(x, dtype=np.float64)
    with_event = np.asarray(detector(xx, fs).t_s, dtype=np.float64)
    a, b = span_s
    if suppressed is not None:
        without = np.asarray(detector(np.asarray(suppressed, dtype=np.float64), fs).t_s)
        keep_w, keep_wo = with_event, without
    else:
        sup = xx.copy()
        sup[max(0, int(math.floor(a * fs))):int(math.ceil(b * fs))] = np.nan
        without = np.asarray(detector(sup, fs).t_s, dtype=np.float64)
        keep_w = with_event[(with_event < a) | (with_event >= b)]
        keep_wo = without[(without < a) | (without >= b)]
        inside = int(((with_event >= a) & (with_event < b)).sum())
        v = beat_train_changed(keep_w, keep_wo, shift_tol_s=tol, match_s=0.020)
        return CardiacVerdict(v.changed or inside > 0, v.n_added, v.n_lost, v.max_shift_s,
                              tol, inside)
    # Pair within a third of the shortest plausible RR (rpeaks' R_MIN is 60 ms).
    return beat_train_changed(keep_w, keep_wo, shift_tol_s=tol, match_s=0.020)


def hrv_extent(event: Event, x: npt.ArrayLike, fs: float, *, signal: str, x_t0_s: float,
               suppressed: npt.ArrayLike | None = None,
               detector: Callable[[F64, float], Any] | None = None) -> Extent | None:
    """``hrv``'s extent: the event's span, padded by settling, if the beat train changed.

    ``x[0]`` is at ``x_t0_s`` on the event's timeline (required, as ``z_t0_s`` is for
    :func:`compute_extent`). The extent is returned on the event's timeline.
    """
    c = event.candidate
    span = (c.start_s - x_t0_s, c.stop_s - x_t0_s)
    verdict = cardiac_operational_damage(x, fs, span, suppressed=suppressed, detector=detector)
    if not verdict.changed:
        return None
    settle = consumer_settling("hrv", fs)
    assert settle is not None
    pad = settle.total_s
    end = x_t0_s + np.asarray(x).shape[0] / fs
    band = extent_consumers()["hrv"].band
    return Extent("hrv", signal, band, max(x_t0_s, c.start_s - pad), min(end, c.stop_s + pad),
                  c.start_s, c.stop_s, pad, BANDS[band].window_s)
