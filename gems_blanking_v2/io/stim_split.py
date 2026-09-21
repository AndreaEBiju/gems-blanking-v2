"""Stim / recovery split and epoch exclusion.

A ``stim_recovery`` recording holds a stimulation epoch followed by a recovery
epoch. The stim epoch is excluded from all downstream processing and blanking; only
recovery is analysed. This module finds the boundary and hands back both epochs as
separate processing units, each knowing where it sits in the original file.

**Why this has to run before the band references.** Task 06 computes a whole-file
median and MAD of the log envelope per signal per band. Stim artifacts are the
largest excursions anywhere in the recording, so leaving them in the array inflates
both, and an inflated reference means an artifact must be *larger* to reach ``z > 3``
during recovery - detection is suppressed exactly in the window the post-stim
dynamics live in. Splitting first makes the recovery reference a recovery-only
statistic. Task 06 refuses an unsplit ``stim_recovery`` recording for this reason,
which turns the dependency into a checked precondition.

**The stim duration is known, and that changes the method.** The protocol is 2 min
stim then 20 min recovery, so detection is not "find the ON region" with onset and
offset both unknown; it is "find the onset of a known-width window", one unknown,
with the duration left over as a free validity check. The duration comes from
:class:`ProtocolSpec`, read from the shared config - never hardcoded, because a
silently wrong 120 s would be worse than no prior at all.

**What replaced the old threshold, and why.** ``splitStimRecovery.m`` set its
threshold at ``(p20 + p80)/2`` of the envelope, which is only meaningful when the
stim occupies roughly 20-80% of the record. At a fixed 120 s the stim is 20% of a
10-minute file and **8% of a 25-minute one**, so both percentiles fall inside the
OFF distribution and the threshold lands in the OFF noise. The matched-width search
here has no duty-cycle assumption at all, spans a mid-stim dropout of any length
because the width is fixed, and has lower onset variance - which matters directly,
since every recovery analysis is expressed as time since stim offset.

``minDurSec = 10`` is deleted; the known width supersedes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import numpy.typing as npt
import yaml

from gems_blanking_v2.io.store import atomic_write_text
from gems_blanking_v2.types import Recording

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

__all__ = [
    "AUDIT_COLUMNS",
    "DEFAULT_PROTOCOL_YAML",
    "EDGE_REFINE_PAD_S",
    "ENVELOPE_RMS_S",
    "ENVELOPE_SMOOTH_S",
    "OFF_ANCHOR_SIGMA",
    "PROTOCOL_FILENAME",
    "SECONDARY_EPOCH_FRACTION",
    "VIB_NAME_PATTERN",
    "Epoch",
    "ProtocolSpec",
    "SplitReport",
    "SplitStatus",
    "audit_stim_splits",
    "blanking_fraction",
    "find_stim_window",
    "load_protocol",
    "matlab_boundary",
    "protocol_path",
    "split_stim_recovery",
    "vib_envelope",
    "write_protocol",
]

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]

SplitStatus = Literal["pass", "review", "fail"]
"""``pass`` proceeds, ``review`` needs a human, ``fail`` has no recovery epoch."""

EpochName = Literal["stim", "recovery"]
ExclusionCategory = Literal["excluded_epoch"]
"""The only category this module emits, and it is **not** ``masked_motion``.

Kept distinct everywhere - QC, retention, and the coverage-confound regression in
task 19. A deterministic protocol exclusion counted as model-driven blanking would
corrupt exactly the check meant to catch confounds.
"""

PROTOCOL_FILENAME: Final = "protocol.yaml"
"""Lives at ``<gems_root>/protocol.yaml`` so every lab member splits identically."""

ENVELOPE_SMOOTH_S: Final = 0.010
"""Light pre-smoothing of the vib channel, seconds. As ``splitStimRecovery.m``."""

ENVELOPE_RMS_S: Final = 0.100
"""Moving-RMS window of the vib envelope, seconds. As ``splitStimRecovery.m``.

This sets the resolution of everything downstream: the ON and OFF edges become
ramps about this wide, so the matched window's argmax can sit up to half of it away
from the true edge. That is what the edge refinement is for.
"""

EDGE_REFINE_PAD_S: Final = 2.0
"""How far outside the matched window an edge may be followed, seconds.

As ``splitStimRecovery.m``, but doing a different job. **The spec's refinement
searches +/-2 s around ``onset + W`` and takes the crossing it finds there, which
means the reported duration can never differ from the prior by more than the pad** -
so "duration becomes a check, not an output" would be structurally unable to catch
the deviations it exists for. A 95 s stim, one of the spec's own required test cases,
is 25 s short: more than twelve pads away, and it would be reported as 120 s.

So the extent is measured from the **first and last threshold crossing inside the
matched window**, which reports any shorter duration exactly, and each edge is then
followed *outward* through contiguous supra-threshold samples so a longer one is
reported too. This pad is the gap the outward walk may bridge. The matched filter
still locates the epoch - that is what is robust at an 8% duty cycle and across a
dropout - and the OFF-anchored threshold, now crossed by 0.0075% of OFF samples,
measures its extent.
"""

OFF_ANCHOR_SIGMA: Final = 8.0
"""Edge-refinement threshold, in robust sigma above the OFF level.

Anchored to the OFF state rather than to a percentile of the whole record, which is
the assumption that failed at an 8% duty cycle. Used **only** for edge refinement
and as a reported cross-check, never as the primary detector.

**The OFF level is taken outside the matched window, not from the envelope's bottom
decile as specified**, and the difference is not cosmetic. The bottom decile is the
*lower tail* of the OFF distribution, so its median and MAD are both biased low -
measured on a 120 s stim in a 25 minute record: median 0.00452 against 0.00615 and
sigma 0.000321 against 0.001119, a 3.5x underestimate of the spread. The resulting
threshold is crossed by **21% of OFF samples** instead of 0.0075%, so the "first
crossing within +/-2 s" refinement lands on OFF noise at the edge of its search
window and pushes both boundaries outward by the full pad. Anchoring on the whole
OFF population costs nothing - the matched window has already said which samples
are OFF - and removes the bias.
"""

SECONDARY_EPOCH_FRACTION: Final = 0.10
"""How long an ON run outside the matched window must be to count as a second epoch.

As a fraction of ``stim_duration_s``, so 12 s at the 120 s protocol. **This rule is
not in the spec**, which deletes ``minDurSec`` and then asks for an epoch count: with
no minimum duration a noisy envelope yields thousands of ON runs, and counting runs
*inside* the matched window would make a mid-stim dropout look like two epochs. So
the count asks the question that actually matters - is there stim-like activity
elsewhere in the file - and derives its one threshold from the protocol rather than
introducing a free constant.
"""

VIB_NAME_PATTERN: Final = r"vib|stim|trig|dig"
"""Names that identify a stimulation-monitor channel, matched case-insensitively.

Case-insensitively because this lab's own data mixes case for the same entity
(cross-platform rule 7). A file with more than one match is ambiguous and raises
rather than picking - guessing which channel monitors the stimulator is the kind of
mistake that is invisible afterwards.
"""

DEFAULT_PROTOCOL_YAML: Final = """\
# Stimulation protocol, shared through the Drive so every lab member splits
# identically. Per cohort; override per recording only with a recorded reason.
#
# The stim duration is a PRIOR, and the whole method depends on it: detection is a
# matched-width search, so a wrong duration moves every boundary. Change it
# deliberately, and reprocess rather than mixing.

stim_duration_s: 120.0      # 2 min stim, fixed by protocol
recovery_duration_s: 1200.0 # 20 min recovery, the independent second check
stim_tolerance_s: 12.0      # start/stop routinely consumes several seconds
"""


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """The stimulation protocol's known durations, in seconds.

    Attributes
    ----------
    stim_duration_s
        Width of the matched search window. The prior the whole method rests on.
    recovery_duration_s
        Expected recovery duration. A second, **independent** check: a file whose
        recovery epoch is far from this is suspect whatever the stim epoch measured.
    stim_tolerance_s
        Half-width of the accepted band around ``stim_duration_s``. 12 s rather than
        a tight few seconds because recording start and stop routinely consume
        several seconds at the edges, and a narrow band would flag normal captures.
    """

    stim_duration_s: float
    recovery_duration_s: float
    stim_tolerance_s: float

    def __post_init__(self) -> None:
        """Check every duration is positive and finite."""
        for name in ("stim_duration_s", "recovery_duration_s", "stim_tolerance_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                msg = f"{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)

    @property
    def secondary_epoch_min_s(self) -> float:
        """Shortest ON run outside the window that counts as a second epoch."""
        return SECONDARY_EPOCH_FRACTION * self.stim_duration_s

    def to_json(self) -> dict[str, float]:
        """Return the spec for provenance. Every field is present and finite."""
        return {
            "stim_duration_s": self.stim_duration_s,
            "recovery_duration_s": self.recovery_duration_s,
            "stim_tolerance_s": self.stim_tolerance_s,
        }


@dataclass(frozen=True, slots=True)
class Epoch:
    """One processing unit: a slice of a recording that knows where it came from.

    Attributes
    ----------
    name
        ``"stim"`` or ``"recovery"``.
    recording
        The slice, with its own ``t = 0``. Recovery's local zero is meaningful - the
        post-stim dynamics start there. ``data`` is a **view**, not a copy.
    t0_offset_s
        Where this epoch starts in the original recording, so absolute time is never
        lost.
    duration_s
        This epoch's own duration. **Every valid-duration denominator downstream uses
        this**, never the original file's, or every rate is wrong by the stim
        fraction.
    excluded
        True for the stim epoch: excluded from all downstream processing and
        blanking.
    category
        ``"excluded_epoch"`` when excluded, else ``None``. Never ``masked_motion``.
    unassessable_head_s, unassessable_tail_s
        Filter-settling windows at the epoch edges, which must not be filtered
        across. ``None`` means **not yet measured** - task 13 measures the settling
        time, and this module will not invent one (invariant 8).
    """

    name: EpochName
    recording: Recording
    t0_offset_s: float
    duration_s: float
    excluded: bool
    category: ExclusionCategory | None = None
    unassessable_head_s: float | None = None
    unassessable_tail_s: float | None = None

    def __post_init__(self) -> None:
        """Check the exclusion category matches the exclusion flag."""
        if self.excluded and self.category is None:
            msg = f"epoch {self.name!r} is excluded but carries no category"
            raise ValueError(msg)
        if not self.excluded and self.category is not None:
            msg = f"epoch {self.name!r} is not excluded but carries {self.category!r}"
            raise ValueError(msg)
        if self.duration_s <= 0.0:
            msg = f"epoch {self.name!r} has a non-positive duration {self.duration_s}"
            raise ValueError(msg)

    @property
    def t1_offset_s(self) -> float:
        """Where this epoch ends in the original recording, seconds."""
        return self.t0_offset_s + self.duration_s

    def with_settling(self, settling_s: float) -> Epoch:
        """Return a copy carrying ``settling_s`` as the unassessable edge windows.

        Task 13 supplies the number. Called with a settling time longer than half the
        epoch this raises, because an epoch that is entirely settling carries no
        assessable data and saying so is better than emitting one.
        """
        if not np.isfinite(settling_s) or settling_s < 0.0:
            msg = f"settling_s must be finite and non-negative, got {settling_s!r}"
            raise ValueError(msg)
        if 2.0 * settling_s >= self.duration_s:
            msg = (
                f"a settling window of {settling_s} s leaves nothing assessable in a "
                f"{self.duration_s} s epoch"
            )
            raise ValueError(msg)
        return replace(
            self, unassessable_head_s=float(settling_s), unassessable_tail_s=float(settling_s)
        )


@dataclass(frozen=True, slots=True)
class SplitReport:
    """Everything the split decided, and everything needed to audit it.

    Attributes
    ----------
    onset_s, offset_s
        The stim boundary in the **original recording's** seconds.
    detected_duration_s
        ``offset_s - onset_s``. A check, not an output: compared against the
        protocol's known width.
    status
        ``"pass"``, ``"review"`` or ``"fail"``. ``"fail"`` means the file holds no
        recovery epoch at all.
    clipped_start, clipped_end
        Whether the vib channel was ON at the first / last sample. Evaluated
        **independently of the tolerance check**, because the two overlap: an
        8 s clipped start gives a 112 s duration, which is inside a 12 s tolerance.
        Clipped at start is benign and common; clipped at end is the fail.
    duty_cycle
        Fraction of the vib record inside the stim window. Reported because it is
        the quantity the old percentile threshold implicitly assumed.
    epoch_count
        1, plus stim-like activity found elsewhere in the file. More than 1 is a
        protocol mismatch - see :data:`SECONDARY_EPOCH_FRACTION`.
    threshold_crosscheck_s
        What the OFF-anchored threshold alone would have called the onset. A
        cross-check, never the decision.
    method
        ``"matched"`` or ``"manual"``.
    protocol
        The spec used, for provenance.
    reason
        Why the status is what it is, in words, for a human reading a QC table.
    """

    onset_s: float
    offset_s: float
    detected_duration_s: float
    status: SplitStatus
    clipped_start: bool
    clipped_end: bool
    duty_cycle: float
    epoch_count: int
    threshold_crosscheck_s: float | None
    method: Literal["matched", "manual"]
    protocol: ProtocolSpec
    reason: str
    vib_channel: str | None = None
    fs_vib: float | None = None
    recovery_duration_s: float = float("nan")

    def to_provenance(self) -> dict[str, Any]:
        """Return a JSON-ready provenance record.

        A value that could not be computed is **absent**, never ``null`` and never a
        sentinel - the serialised-missing convention.
        """
        record: dict[str, Any] = {
            "onset_s": self.onset_s,
            "offset_s": self.offset_s,
            "detected_duration_s": self.detected_duration_s,
            "status": self.status,
            "clipped_start": self.clipped_start,
            "clipped_end": self.clipped_end,
            "duty_cycle": self.duty_cycle,
            "epoch_count": self.epoch_count,
            "method": self.method,
            "reason": self.reason,
            "protocol": self.protocol.to_json(),
            "exclusion_category": "excluded_epoch",
            "settling_measured": False,
        }
        for name in ("threshold_crosscheck_s", "vib_channel", "fs_vib"):
            value = getattr(self, name)
            if value is not None:
                record[name] = value
        if np.isfinite(self.recovery_duration_s):
            record["recovery_duration_s"] = self.recovery_duration_s
        return record


# ---------------------------------------------------------------------------
# protocol config
# ---------------------------------------------------------------------------


def protocol_path(gems_root: Path) -> Path:
    """Return ``<gems_root>/protocol.yaml``."""
    return Path(gems_root) / PROTOCOL_FILENAME


def load_protocol(path: Path) -> ProtocolSpec:
    """Load and validate ``protocol.yaml``.

    Raises
    ------
    FileNotFoundError
        If the file is absent. **It is not defaulted.** The stim duration is the
        prior the whole matched-width search rests on, so a value that silently
        appeared would split differently on two machines - the same reason
        ``conditions.yaml`` refuses to default.
    ValueError
        If the document is malformed or a duration is missing or non-physical.
    """
    path = Path(path)
    if not path.is_file():
        msg = (
            f"no protocol at {path}. Write one with "
            "write_protocol(default_protocol(), path) and commit it to the shared "
            "drive so everyone splits identically. It is not defaulted: the stim "
            "duration is a prior, and a silently wrong one moves every boundary."
        )
        raise FileNotFoundError(msg)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        msg = f"{path} could not be parsed: {exc}"
        raise ValueError(msg) from exc
    return _protocol_from_document(document or {}, str(path))


def default_protocol() -> ProtocolSpec:
    """Return the starting protocol: 120 s stim, 1200 s recovery, 12 s tolerance."""
    return _protocol_from_document(yaml.safe_load(DEFAULT_PROTOCOL_YAML), "<default>")


def write_protocol(protocol: ProtocolSpec, path: Path) -> Path:
    r"""Write a protocol atomically as UTF-8 with ``\n`` endings.

    Round-trips through :func:`load_protocol`, so a written file is always loadable.
    """
    body = yaml.safe_dump(
        protocol.to_json(), sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    path = Path(path)
    atomic_write_text(path, body)
    return path


def _protocol_from_document(document: object, source: str) -> ProtocolSpec:
    """Build a :class:`ProtocolSpec` from a parsed YAML document."""
    if not isinstance(document, dict):
        msg = f"{source}: expected a mapping at the top level, got {type(document).__name__}"
        raise ValueError(msg)

    values: dict[str, float] = {}
    for name in ("stim_duration_s", "recovery_duration_s", "stim_tolerance_s"):
        raw = document.get(name)
        if raw is None:
            msg = f"{source}: {name} is required and is absent"
            raise ValueError(msg)
        try:
            values[name] = float(raw)
        except (TypeError, ValueError) as exc:
            msg = f"{source}: {name} must be a number, got {raw!r}"
            raise ValueError(msg) from exc
    return ProtocolSpec(**values)


# ---------------------------------------------------------------------------
# the vib envelope, and the matched-width search
# ---------------------------------------------------------------------------


def _moving_mean(x: F64, window: int) -> F64:
    """Return the centred moving mean of ``x`` over ``window`` samples.

    Centred, and edge windows are shortened rather than padded, so no fabricated
    sample enters the envelope (invariant 8).
    """
    window = max(int(window), 1)
    if window == 1:
        return np.asarray(x, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    half = window // 2
    idx = np.arange(x.size)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx - half + window, x.size)
    return np.asarray((cumulative[hi] - cumulative[lo]) / (hi - lo), dtype=np.float64)


def vib_envelope(vib: F64, fs_vib: float) -> F64:
    """Return the RMS envelope of a stimulation-monitor channel.

    A 10 ms smooth then a 100 ms moving RMS, as ``splitStimRecovery.m``. The vib
    channel's **amplitude units never enter a decision** - every rule below is a
    contrast on this envelope against itself - so unlike a physiological channel it
    needs no declared units (invariant 14 applies to what gets interpreted as
    microvolts).

    Parameters
    ----------
    vib
        The monitor channel, any scale.
    fs_vib
        Its own sample rate in Hz, which need not equal the signal's.
    """
    x = np.asarray(vib, dtype=np.float64).ravel()
    if x.size == 0:
        msg = "the vib channel is empty"
        raise ValueError(msg)
    if not np.isfinite(fs_vib) or fs_vib <= 0:
        msg = f"fs_vib must be positive and finite, got {fs_vib!r}"
        raise ValueError(msg)

    finite = np.isfinite(x)
    if not bool(finite.all()):
        if not bool(finite.any()):
            msg = "the vib channel is entirely non-finite"
            raise ValueError(msg)
        x = x.copy()
        idx = np.arange(x.size, dtype=np.float64)
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    smoothed = _moving_mean(x, max(int(round(ENVELOPE_SMOOTH_S * fs_vib)), 5))
    return np.asarray(
        np.sqrt(_moving_mean(smoothed**2, max(int(round(ENVELOPE_RMS_S * fs_vib)), 5))),
        dtype=np.float64,
    )


def _off_anchored_threshold(env: F64, onset: int, width: int) -> float:
    """Return ``off_level + 8 * off_sigma`` from the samples outside the stim window.

    See :data:`OFF_ANCHOR_SIGMA` for why the OFF population rather than the
    envelope's bottom decile: the decile is a truncated tail and underestimates the
    spread 3.5-fold, which pushes both refined edges outward by the whole search pad.

    When the OFF state has no spread at all - a noiseless synthetic - ``8 * 0`` would
    put the threshold on the OFF level itself and every sample would cross it. The
    midpoint between the OFF level and the window mean is used instead, which is the
    only information available, and is stated rather than quietly degenerating.
    """
    outside = np.ones(env.size, dtype=bool)
    outside[onset : onset + width] = False
    off = env[outside] if bool(outside.any()) else env

    off_level = float(np.median(off))
    off_sigma = 1.4826 * float(np.median(np.abs(off - off_level)))
    if off_sigma > 0.0:
        return off_level + OFF_ANCHOR_SIGMA * off_sigma
    window_mean = float(np.mean(env[onset : onset + width]))
    return off_level + 0.5 * max(window_mean - off_level, 0.0)


def _measure_extent(
    env: F64, onset: int, width: int, pad: int, threshold: float
) -> tuple[int, int]:
    """Return the ``[start, stop)`` extent of the stim epoch the matched window found.

    First and last crossing **inside** the window, so a stim shorter than the prior is
    measured exactly, then each edge is followed outward through contiguous
    supra-threshold samples bridging gaps up to ``pad``, so a longer one is too. See
    :data:`EDGE_REFINE_PAD_S` for why the spec's fixed-pad search cannot do this.
    """
    stop_window = min(onset + width, env.size)
    inside = np.flatnonzero(env[onset:stop_window] > threshold)
    if inside.size == 0:
        return onset, stop_window

    start = onset + int(inside[0])
    stop = onset + int(inside[-1]) + 1

    on = env > threshold
    while start > 0:
        lo = max(start - pad, 0)
        reachable = np.flatnonzero(on[lo:start])
        if reachable.size == 0:
            break
        start = lo + int(reachable[0])
    while stop < env.size:
        hi = min(stop + pad, env.size)
        reachable = np.flatnonzero(on[stop:hi])
        if reachable.size == 0:
            break
        stop = stop + int(reachable[-1]) + 1
    return start, stop


def _on_runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return the ``[start, stop)`` sample bounds of every ``True`` run."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1).tolist()
    stops = np.flatnonzero(edges == -1).tolist()
    return list(zip(starts, stops, strict=True))


@dataclass(frozen=True, slots=True)
class _Window:
    """Internal: the located stim window on the vib time base."""

    onset: int
    offset: int
    threshold: float
    contrast: float
    crosscheck_onset: int | None
    epoch_count: int
    extras: list[tuple[float, float]] = field(default_factory=list)


def find_stim_window(env: F64, fs_vib: float, protocol: ProtocolSpec) -> _Window:
    """Locate the stim epoch by sliding a boxcar of the known width over ``env``.

    ``score(t) = mean(env[t:t+W]) - mean(env outside)``. Worth being precise about
    what the subtraction does: for fixed ``W`` and ``N`` the score is a strictly
    increasing function of ``mean(env[t:t+W])``, so **the argmax is identical to that
    of a plain boxcar mean** - the contrast makes the returned value interpretable,
    it does not change where the window lands. "No duty-cycle assumption" is a
    property of fixing the width, not of subtracting the outside. Both are computed
    from one cumulative sum, so this is O(N) rather than O(N*W).

    Both edges are then refined locally against the OFF-anchored threshold, within
    :data:`EDGE_REFINE_PAD_S`, because the 100 ms RMS window turns each edge into a
    ramp and the boxcar argmax can sit up to half of it away from the true edge.

    Raises
    ------
    ValueError
        If the record is shorter than the stim window - a protocol mismatch that
        cannot be resolved by moving a boundary.
    """
    width = int(round(protocol.stim_duration_s * fs_vib))
    if width < 1:
        msg = f"a {protocol.stim_duration_s} s window is under one sample at {fs_vib} Hz"
        raise ValueError(msg)
    if env.size < width:
        msg = (
            f"the vib record is {env.size / fs_vib:.1f} s but the protocol's stim is "
            f"{protocol.stim_duration_s:.1f} s - this file cannot hold one"
        )
        raise ValueError(msg)

    cumulative = np.concatenate([[0.0], np.cumsum(env, dtype=np.float64)])
    inside = (cumulative[width:] - cumulative[:-width]) / width
    total = float(cumulative[-1])
    outside_count = env.size - width
    outside = (
        (total - inside * width) / outside_count
        if outside_count > 0
        else np.zeros_like(inside)
    )
    score = inside - outside
    onset = int(np.argmax(score))
    contrast = float(score[onset])

    threshold = _off_anchored_threshold(env, onset, width)
    pad = max(int(round(EDGE_REFINE_PAD_S * fs_vib)), 1)
    refined_onset, refined_offset = _measure_extent(env, onset, width, pad, threshold)
    if refined_offset <= refined_onset:
        refined_onset, refined_offset = onset, min(onset + width, env.size)

    on = env > threshold
    crosscheck = _on_runs(on)
    crosscheck_onset = None
    if crosscheck:
        longest = max(crosscheck, key=lambda run: run[1] - run[0])
        crosscheck_onset = longest[0]

    minimum = int(round(protocol.secondary_epoch_min_s * fs_vib))
    extras = [
        (start / fs_vib, stop / fs_vib)
        for start, stop in crosscheck
        if (stop - start) >= minimum and (stop <= refined_onset or start >= refined_offset)
    ]

    return _Window(
        onset=refined_onset,
        offset=refined_offset,
        threshold=threshold,
        contrast=contrast,
        crosscheck_onset=crosscheck_onset,
        epoch_count=1 + len(extras),
        extras=extras,
    )


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------


def _find_vib_channel(rec: Recording) -> str:
    """Return the name of the recording's stimulation-monitor channel.

    Raises
    ------
    ValueError
        If there is no match, or more than one. Neither is guessed: inferring stim
        timing from the neural channels would be circular, and picking between two
        candidate monitors is a mistake that is invisible afterwards.
    """
    import re  # noqa: PLC0415

    pattern = re.compile(VIB_NAME_PATTERN, re.IGNORECASE)
    matches = [c.name for c in rec.channels if pattern.search(c.name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        msg = (
            f"{rec.path.name} has condition stim_recovery but no stimulation-monitor "
            f"channel matching {VIB_NAME_PATTERN!r}; channels are "
            f"{[c.name for c in rec.channels]}. Stim timing is not inferred from the "
            "neural channels - that is circular, and unnecessary when a monitor "
            "channel exists."
        )
        raise ValueError(msg)
    msg = (
        f"{rec.path.name} has {len(matches)} candidate stimulation-monitor channels "
        f"{matches}; pass vib_channel= to say which one monitors the stimulator"
    )
    raise ValueError(msg)


def _slice(rec: Recording, start: int, stop: int) -> Recording:
    """Return ``rec`` restricted to ``[start, stop)``. ``data`` stays a view."""
    return Recording(
        fs=rec.fs,
        data=rec.data[start:stop],
        channels=list(rec.channels),
        animal=rec.animal,
        session=rec.session,
        path=rec.path,
    )


def _status(
    detected_s: float,
    protocol: ProtocolSpec,
    *,
    clipped_start: bool,
    clipped_end: bool,
    epoch_count: int,
) -> tuple[SplitStatus, str]:
    """Resolve the status table, in priority order.

    The spec's rows overlap - an 8 s clipped start gives a 112 s duration, which is
    *inside* a 12 s tolerance, so it matches both "clean capture" and "clipped at
    start". Clipping is therefore evaluated independently and reported as a flag;
    only clipped-at-end is a status, because only it means the file holds no recovery
    epoch.
    """
    off_by = abs(detected_s - protocol.stim_duration_s)
    if clipped_end:
        return "fail", (
            "the vib channel is still ON at the last sample, so the recording stopped "
            "during stimulation - there is no recovery epoch in this file"
        )
    if epoch_count != 1:
        return "review", (
            f"{epoch_count} stim-like epochs found where the protocol has 1 - "
            "protocol mismatch, needs a human"
        )
    if off_by <= protocol.stim_tolerance_s:
        clipped = " (clipped at start, which is benign)" if clipped_start else ""
        return "pass", (
            f"detected {detected_s:.1f} s against the protocol's "
            f"{protocol.stim_duration_s:.0f} s, inside the "
            f"{protocol.stim_tolerance_s:.0f} s tolerance{clipped}"
        )
    if clipped_start:
        return "pass", (
            f"detected {detected_s:.1f} s, short of the protocol's "
            f"{protocol.stim_duration_s:.0f} s, but the vib channel is ON at the first "
            "sample: the recording started mid-stim, so the offset is still valid and "
            "the recovery epoch is intact"
        )
    return "review", (
        f"detected {detected_s:.1f} s against the protocol's "
        f"{protocol.stim_duration_s:.0f} s, outside the "
        f"{protocol.stim_tolerance_s:.0f} s tolerance and not clipped at either edge - "
        "something is wrong, do not proceed silently"
    )


def split_stim_recovery(
    rec: Recording,
    *,
    protocol: ProtocolSpec,
    vib: F64 | None = None,
    fs_vib: float | None = None,
    vib_channel: str | None = None,
    stim_end_s: float | None = None,
) -> tuple[Epoch | None, Epoch, SplitReport]:
    """Split ``rec`` into an excluded stim epoch and an analysed recovery epoch.

    Returns ``(stim, recovery, report)``. The stim epoch is **kept**, not deleted -
    that is what lets the deferred within-stim artifact work happen later - and is
    ``None`` only when the recording started mid-stim with nothing before the
    boundary.

    Parameters
    ----------
    rec
        The recording to split.
    protocol
        Durations from the shared config. Never defaulted; see :func:`load_protocol`.
    vib
        The monitor channel, when it lives outside ``rec.data`` at its own rate.
        Requires ``fs_vib``. Omit to use one of ``rec``'s own channels.
    fs_vib
        The monitor channel's sample rate in Hz. ``Recording`` carries one ``fs`` for
        the whole matrix, so a monitor sampled at a different rate cannot live in it
        and must come in this way.
    vib_channel
        Name of the monitor channel within ``rec``. Found by
        :data:`VIB_NAME_PATTERN` when omitted.
    stim_end_s
        A human's or the source file's declared stim end, in seconds. When given, the
        boundary is taken from it and ``method`` is ``"manual"`` - no search runs.
        This is the ``splitStimRecoveryManual.m`` path, and the loader's
        ``stim_end_s`` feeds it.

    Raises
    ------
    ValueError
        If there is no monitor channel, if the monitor is ambiguous, if the record is
        shorter than the stim window, or if the recording stopped during stimulation
        so that no recovery epoch exists.
    """
    duration_s = rec.data.shape[0] / rec.fs

    if stim_end_s is not None:
        return _manual_split(rec, protocol, float(stim_end_s))

    if vib is not None:
        if fs_vib is None:
            msg = "an explicit vib channel needs its own fs_vib"
            raise ValueError(msg)
        monitor = np.asarray(vib, dtype=np.float64).ravel()
        rate = float(fs_vib)
        name = vib_channel
    else:
        name = vib_channel or _find_vib_channel(rec)
        by_name = {c.name: c for c in rec.channels}
        if name not in by_name:
            msg = f"{name!r} is not a channel of {rec.path.name}"
            raise ValueError(msg)
        monitor = np.asarray(rec.data[:, by_name[name].index], dtype=np.float64)
        rate = float(fs_vib) if fs_vib is not None else rec.fs

    env = vib_envelope(monitor, rate)
    window = find_stim_window(env, rate, protocol)

    # The one index-convention boundary in this module: the window is found on the
    # vib time base in samples, converted to seconds once, and only then mapped onto
    # the signal grid. Going straight from vib samples to signal samples would bake
    # the rate ratio into an integer division (invariant 15).
    onset_s = window.onset / rate
    offset_s = window.offset / rate
    detected_s = offset_s - onset_s

    clipped_start = bool(env[0] > window.threshold)
    clipped_end = bool(env[-1] > window.threshold)
    status, reason = _status(
        detected_s,
        protocol,
        clipped_start=clipped_start,
        clipped_end=clipped_end,
        epoch_count=window.epoch_count,
    )

    report = SplitReport(
        onset_s=onset_s,
        offset_s=offset_s,
        detected_duration_s=detected_s,
        status=status,
        clipped_start=clipped_start,
        clipped_end=clipped_end,
        duty_cycle=detected_s / duration_s,
        epoch_count=window.epoch_count,
        threshold_crosscheck_s=(
            None if window.crosscheck_onset is None else window.crosscheck_onset / rate
        ),
        method="matched",
        protocol=protocol,
        reason=reason,
        vib_channel=name,
        fs_vib=rate,
        recovery_duration_s=max(duration_s - offset_s, 0.0),
    )

    if status == "fail":
        msg = f"{rec.path.name}: {reason}"
        raise ValueError(msg)

    log.info(
        "%s session %s: stim %.2f-%.2f s (%.1f s detected, %s), recovery %.1f s, "
        "epochs %d, duty %.1f%%, excluded as excluded_epoch not masked_motion",
        rec.animal,
        rec.session,
        onset_s,
        offset_s,
        detected_s,
        status,
        report.recovery_duration_s,
        window.epoch_count,
        100.0 * report.duty_cycle,
    )
    return _epochs(rec, onset_s, offset_s, report)


def _manual_split(
    rec: Recording, protocol: ProtocolSpec, stim_end_s: float
) -> tuple[Epoch | None, Epoch, SplitReport]:
    """Split at a declared stim end, the ``splitStimRecoveryManual.m`` path.

    The manual form assumes the stim starts at sample 0, which is
    ``clipped_start`` by construction - stated here rather than left implicit as it
    is in the MATLAB.
    """
    duration_s = rec.data.shape[0] / rec.fs
    if not np.isfinite(stim_end_s) or not 0.0 < stim_end_s < duration_s:
        msg = (
            f"a declared stim end of {stim_end_s!r} s is outside this "
            f"{duration_s:.1f} s recording"
        )
        raise ValueError(msg)

    status, reason = _status(
        stim_end_s,
        protocol,
        clipped_start=True,
        clipped_end=False,
        epoch_count=1,
    )
    report = SplitReport(
        onset_s=0.0,
        offset_s=stim_end_s,
        detected_duration_s=stim_end_s,
        status=status,
        clipped_start=True,
        clipped_end=False,
        duty_cycle=stim_end_s / duration_s,
        epoch_count=1,
        threshold_crosscheck_s=None,
        method="manual",
        protocol=protocol,
        reason=f"declared stim end, no search run: {reason}",
        recovery_duration_s=duration_s - stim_end_s,
    )
    return _epochs(rec, 0.0, stim_end_s, report)


def _epochs(
    rec: Recording, onset_s: float, offset_s: float, report: SplitReport
) -> tuple[Epoch | None, Epoch, SplitReport]:
    """Build the two epochs from a boundary in the original recording's seconds."""
    n = rec.data.shape[0]
    start = max(int(round(onset_s * rec.fs)), 0)
    stop = min(int(round(offset_s * rec.fs)), n)

    stim: Epoch | None = None
    if stop > start:
        stim = Epoch(
            name="stim",
            recording=_slice(rec, start, stop),
            t0_offset_s=start / rec.fs,
            duration_s=(stop - start) / rec.fs,
            excluded=True,
            category="excluded_epoch",
        )

    if stop >= n:
        msg = (
            f"{rec.path.name}: the stim epoch runs to the end of the recording, so "
            "there is no recovery epoch to analyse"
        )
        raise ValueError(msg)

    recovery = Epoch(
        name="recovery",
        recording=_slice(rec, stop, n),
        t0_offset_s=stop / rec.fs,
        duration_s=(n - stop) / rec.fs,
        excluded=False,
    )
    return stim, recovery, report


AUDIT_COLUMNS: Final = (
    "path",
    "animal",
    "session",
    "detected_duration_s",
    "expected_duration_s",
    "duration_error_s",
    "status",
    "clipped_start",
    "clipped_end",
    "epoch_count",
    "duty_cycle",
    "onset_s",
    "offset_s",
    "matlab_onset_s",
    "matlab_offset_s",
    "onset_difference_s",
    "offset_difference_s",
    "contaminated_recovery_s",
    "error",
)
"""Columns of the acceptance table, in order. Fixed so the report is diffable."""


def matlab_boundary(recovery_path: Path, fs: float) -> tuple[float, float] | None:
    """Read the boundary the MATLAB split produced, from its ``_recovery.mat``.

    ``splitStimRecovery.m`` saves ``recoveryMask`` on the signal time base, so the
    stim epoch is the leading ``False`` run. Returns ``None`` when the file is absent
    or holds no mask - a file the old pipeline never split has nothing to diff
    against, which is itself worth reporting rather than treating as agreement.
    """
    if not recovery_path.is_file():
        return None
    try:
        from scipy.io import loadmat  # noqa: PLC0415

        contents = loadmat(recovery_path, variable_names=["recoveryMask"])
    except (OSError, ValueError, NotImplementedError):
        return None

    mask = contents.get("recoveryMask")
    if mask is None:
        return None
    recovery = np.asarray(mask, dtype=bool).ravel()
    stim = ~recovery
    if not bool(stim.any()):
        return 0.0, 0.0
    indices = np.flatnonzero(stim)
    return float(indices[0]) / fs, float(indices[-1] + 1) / fs


def audit_stim_splits(
    recordings: Iterable[tuple[Path, Recording]],
    protocol: ProtocolSpec,
    *,
    figure_path: Path | None = None,
) -> pd.DataFrame:
    """Re-split every recording and diff the boundary against the MATLAB output.

    The acceptance step, and the first subtask: **the old auto-threshold was out of
    range on every stim recording** - at a fixed 120 s stim the duty cycle is 20% of a
    10-minute file and 8% of a 25-minute one, at or below the bottom of the band
    ``(p20 + p80)/2`` needs - so where the two methods disagree by more than the
    edge-refinement window, that file's recovery reference has been contaminated with
    stim, or that much recovery was thrown away.

    A file that raises is reported as a row with ``error`` set, not skipped: a split
    that failed is the most interesting row in the table.

    Parameters
    ----------
    recordings
        ``(path, recording)`` pairs. Taking already-loaded recordings keeps this
        independent of the loader and testable without the Drive.
    protocol
        The protocol to score against.
    figure_path
        Where to write the detected-duration histogram. The distribution should be a
        tight spike at ``stim_duration_s`` with a short tail of clipped captures;
        anything else is a finding. Omitted, no figure is written.

    Returns
    -------
    pandas.DataFrame
        One row per recording, columns :data:`AUDIT_COLUMNS`.
    """
    import pandas as pd  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    for path, rec in recordings:
        row: dict[str, Any] = dict.fromkeys(AUDIT_COLUMNS)
        row |= {
            "path": path.as_posix(),
            "animal": rec.animal,
            "session": rec.session,
            "expected_duration_s": protocol.stim_duration_s,
        }
        try:
            _stim, _recovery, report = split_stim_recovery(rec, protocol=protocol)
        except ValueError as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue

        row |= {
            "detected_duration_s": report.detected_duration_s,
            "duration_error_s": report.detected_duration_s - protocol.stim_duration_s,
            "status": report.status,
            "clipped_start": report.clipped_start,
            "clipped_end": report.clipped_end,
            "epoch_count": report.epoch_count,
            "duty_cycle": report.duty_cycle,
            "onset_s": report.onset_s,
            "offset_s": report.offset_s,
        }
        legacy = matlab_boundary(path.with_name(path.stem + "_recovery.mat"), rec.fs)
        if legacy is not None:
            row["matlab_onset_s"], row["matlab_offset_s"] = legacy
            row["onset_difference_s"] = report.onset_s - legacy[0]
            row["offset_difference_s"] = report.offset_s - legacy[1]
            # Positive means the old split ended early and left stim in recovery.
            row["contaminated_recovery_s"] = report.offset_s - legacy[1]
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))
    if figure_path is not None:
        _write_duration_histogram(table, protocol, Path(figure_path))
    return table


def _write_duration_histogram(
    table: pd.DataFrame, protocol: ProtocolSpec, figure_path: Path
) -> None:
    """Write the detected-duration distribution. ``Agg`` only - never a GUI backend."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    durations = table["detected_duration_s"].dropna().to_numpy(dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    if durations.size:
        axis.hist(durations, bins=40)
    axis.axvline(protocol.stim_duration_s, linestyle="--", label="protocol")
    for sign in (-1.0, 1.0):
        axis.axvline(
            protocol.stim_duration_s + sign * protocol.stim_tolerance_s,
            linestyle=":",
            label="tolerance" if sign > 0 else None,
        )
    axis.set_xlabel("detected stim duration (s)")
    axis.set_ylabel("recordings")
    axis.set_title(f"n = {durations.size}; expected a spike at {protocol.stim_duration_s:.0f} s")
    axis.legend()
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)


def blanking_fraction(epoch: Epoch, masked_s: float) -> float:
    """Return motion-blanked seconds as a fraction of **this epoch's** duration.

    The denominator is the epoch, never the original file, or every rate downstream
    is wrong by the stim fraction. There is deliberately no way to pass an excluded
    epoch's duration in here: ``excluded_epoch`` and ``masked_motion`` are separate
    categories, and summing them would corrupt the coverage-confound regression that
    exists to catch exactly that.

    Raises
    ------
    ValueError
        If ``epoch`` is excluded. An excluded epoch has no blanking fraction; it was
        not analysed.
    """
    if epoch.excluded:
        msg = (
            f"epoch {epoch.name!r} is an {epoch.category} and has no motion-blanking "
            "fraction - a protocol exclusion is not model-driven blanking"
        )
        raise ValueError(msg)
    if not np.isfinite(masked_s) or masked_s < 0.0:
        msg = f"masked_s must be finite and non-negative, got {masked_s!r}"
        raise ValueError(msg)
    return float(masked_s) / epoch.duration_s
