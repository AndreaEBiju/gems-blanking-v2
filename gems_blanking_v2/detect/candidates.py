"""Candidate generation: hysteresis on robust z across every (signal, band) pair.

**This step's only job is recall.** Precision belongs to task 12, and candidate count
is not review burden - the classifier judges every candidate and humans label a
sample. So the defaults here are the recall-maximising ones and they are the right
*starting* point, not the right answer: ``z_enter`` and ``combine`` are both swept
and pinned in task 09, against the rates this module reports rather than against an
estimate.

**How the pairs combine is explicit, because leaving it implicit makes it a union.**
Hard invariant 10b: a per-pair clean flag rate of 0.2-3.4% becomes 40-55% over the
union of 36 pairs. :data:`CombineRule` is a parameter, and
:class:`FlagRates` reports the per-pair rates, the union, and what the chosen rule
actually produced - three different numbers that are routinely conflated.

**Cross-band coincidence is not a suppression rule here.** It is a task 11 classifier
feature. Gating candidates on it would throw away exactly the evidence task 12 needs
to learn from, and a generator that has already made the decision leaves the
classifier nothing to decide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.constants import GRID_S
from gems_blanking_v2.physio.rpeaks import BeatTrain
from gems_blanking_v2.types import Candidate

__all__ = [
    "CARDIAC_SUPPRESSION_BANDS",
    "VIDEO_ASSISTED_Z_ENTER",
    "CandidateReport",
    "CombineRule",
    "FlagRates",
    "VideoMotion",
    "candidate_report",
    "flag_rates",
    "generate_candidates",
]

log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
Pair = tuple[str, str]

CombineRule = Literal["any", "k_of_n"]
"""How the per-pair crossings combine into one decision per frame.

``"any"`` is the recall-maximising default and this step's only job is recall.
``"k_of_n"`` requires ``k`` pairs to agree, which is the lever invariant 10b names
for reducing the family - cross-channel agreement, not a tighter per-pair threshold.
"""

CARDIAC_SUPPRESSION_BANDS: Final[tuple[str, ...]] = ("100-300",)
"""Bands where a peri-R window suppresses candidates. **Only this one.**

From A.5: the QRS puts 32.1-65.2% of its energy in 100-300 Hz, 0.00-0.54% above
300 Hz and 0.00% below 3 Hz. So the heartbeat is a confound in exactly one band, and
suppressing it anywhere else would discard real artifacts that happen to land near a
beat - which at 364 bpm is a large fraction of the recording.
"""

VIDEO_ASSISTED_Z_ENTER: Final = 2.0
"""Lowered entry threshold where video motion exceeds its own threshold.

Video is independent evidence: cable, connector and headstage sources enter
*downstream of the electrode*, where no montage can reject them, so a coincident
mechanical event licenses a lower electrical bar. Candidates that only clear this
bar are tagged ``video_assisted`` and are distinguishable downstream for that reason.
"""


class VideoMotion(Protocol):
    """What candidate generation needs from task 17's video features.

    **Task 17 specifies no such type.** It describes ROIs, sync and drift and never
    names a dataclass, while this task's signature references one. This Protocol is
    the minimum contract needed here, declared structurally so task 17 can satisfy it
    without importing anything from ``detect`` - and so the shape is visible for
    adjudication rather than invented inside a function.
    """

    def exceeds_threshold(self, n_frames: int) -> Bool:
        """Return, per frame of the shared 10 ms grid, whether motion is above threshold.

        ``n_frames`` is the z-trace length, so the implementation resamples or pads to
        the grid rather than this module guessing at the video's own rate.
        """
        ...


@dataclass(frozen=True, slots=True)
class FlagRates:
    """Per-pair, union and combined frame flag rates - **three different numbers**.

    Invariant 10b exists because these get conflated. Task 09 sweeps ``z_enter`` and
    ``combine`` against *these*, measured on the recording in hand, not against the
    0.2-3.4% per-pair estimate.

    Attributes
    ----------
    per_pair
        Fraction of assessable frames where that pair alone exceeded ``z_enter``.
    union
        Fraction where **any** pair did. The family-wise rate, and the thing that
        runs 40-55% when a per-pair 0.2-3.4% is quoted as if it were the answer.
    combined
        Fraction after the chosen :data:`CombineRule`. Equals ``union`` when the rule
        is ``"any"``, which is why reporting only this one would hide the problem.
    n_frames
        Frames in the trace, assessable or not.
    n_assessable
        Frames with at least one finite pair. The denominator for every rate above.
    """

    per_pair: dict[Pair, float]
    union: float
    combined: float
    n_frames: int
    n_assessable: int

    def to_provenance(self) -> dict[str, Any]:
        """Return a JSON-ready record. Pair keys become ``"signal|band"`` strings."""
        return {
            "per_pair": {
                f"{signal}|{band}": rate
                for (signal, band), rate in self.per_pair.items()
            },
            "union": self.union,
            "combined": self.combined,
            "n_frames": self.n_frames,
            "n_assessable": self.n_assessable,
        }


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Candidates plus what task 09 needs to sweep against.

    Attributes
    ----------
    candidates
        Every candidate, in time order.
    flag_rates
        See :class:`FlagRates`.
    duration_cap_s
        The cap applied, or ``None`` when it has not been measured. ``None`` means
        **no cap was applied and that is recorded**, which is a different statement
        from "no candidate exceeded the cap".
    over_cap
        Indices into ``candidates`` of spans longer than the cap. They are **routed
        to review and never auto-masked**; they are not dropped, because a sustained
        level shift is a finding.

        Carried here rather than on the candidate because A.1 fixes
        :class:`~gems_blanking_v2.types.Candidate`'s fields, and routing is a
        property of this run rather than of the span.
    z_enter, z_exit, combine, k
        The operating point, for provenance.
    cardiac_suppression
        ``"applied"``, or ``"not measured"`` when ``cardiac_windows`` was ``None`` -
        so a reader can tell a file with no cardiac confound from one where nobody
        looked.
    """

    candidates: list[Candidate]
    flag_rates: FlagRates
    duration_cap_s: float | None
    over_cap: tuple[int, ...]
    z_enter: float
    z_exit: float
    combine: CombineRule
    k: int
    cardiac_suppression: Literal["applied", "not measured"]
    suppressed_frames: int = 0

    @property
    def n_candidates(self) -> int:
        """How many candidates were generated."""
        return len(self.candidates)

    def to_provenance(self) -> dict[str, Any]:
        """Return a JSON-ready record of the operating point and what it produced."""
        record: dict[str, Any] = {
            "n_candidates": self.n_candidates,
            "flag_rates": self.flag_rates.to_provenance(),
            "z_enter": self.z_enter,
            "z_exit": self.z_exit,
            "combine": self.combine,
            "k": self.k,
            "cardiac_suppression": self.cardiac_suppression,
            "suppressed_frames": self.suppressed_frames,
            "n_over_cap": len(self.over_cap),
        }
        # Serialised-missing convention: an unmeasured cap is an absent key, never
        # null and never a sentinel that would read as "no cap needed".
        if self.duration_cap_s is not None:
            record["duration_cap_s"] = self.duration_cap_s
        return record


def _crossing_masks(z: dict[Pair, F64], threshold: float) -> dict[Pair, Bool]:
    """Return, per pair, which frames exceed ``threshold``. ``nan`` is never a crossing."""
    masks: dict[Pair, Bool] = {}
    for pair, trace in z.items():
        values = np.asarray(trace, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            masks[pair] = np.asarray(values > threshold, dtype=bool) & np.isfinite(values)
    return masks


def _combine(masks: dict[Pair, Bool], combine: CombineRule, k: int, n_frames: int) -> Bool:
    """Reduce the per-pair masks to one decision per frame."""
    if not masks:
        return np.zeros(n_frames, dtype=bool)
    stacked = np.vstack([masks[pair] for pair in sorted(masks)])
    if combine == "any":
        return np.asarray(stacked.any(axis=0), dtype=bool)
    return np.asarray(stacked.sum(axis=0) >= k, dtype=bool)


def _runs(mask: Bool) -> list[tuple[int, int]]:
    """Return the ``[start, stop)`` bounds of every ``True`` run."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    return list(
        zip(
            np.flatnonzero(edges == 1).tolist(),
            np.flatnonzero(edges == -1).tolist(),
            strict=True,
        )
    )


def _hysteresis(entered: Bool, sustained: Bool) -> list[tuple[int, int]]:
    """Return runs of ``sustained`` that contain at least one ``entered`` frame.

    Hysteresis, expressed as containment rather than as a state machine: a run of
    frames above ``z_exit`` is a candidate exactly when it reaches ``z_enter``
    somewhere. An event whose z dips to 2.0 mid-way stays one candidate, because the
    dip never leaves the sustaining set.
    """
    return [(start, stop) for start, stop in _runs(sustained) if bool(entered[start:stop].any())]


def _cardiac_mask(
    cardiac_windows: dict[Pair, tuple[float, float] | None],
    beats: BeatTrain,
    n_frames: int,
) -> dict[Pair, Bool]:
    """Return, per suppressed pair, which frames sit inside a peri-R window.

    The windows are ``(t_start_s, t_stop_s)`` **relative to R** per task 02, so each
    beat stamps one window onto the grid. Only
    :data:`CARDIAC_SUPPRESSION_BANDS` is suppressed.
    """
    masks: dict[Pair, Bool] = {}
    for pair, window in cardiac_windows.items():
        _signal, band = pair
        if band not in CARDIAC_SUPPRESSION_BANDS or window is None:
            continue
        start_s, stop_s = window
        mask = np.zeros(n_frames, dtype=bool)
        for beat in beats.t_s.tolist():
            lo = max(int(np.floor((beat + start_s) / GRID_S)), 0)
            hi = min(int(np.ceil((beat + stop_s) / GRID_S)) + 1, n_frames)
            if hi > lo:
                mask[lo:hi] = True
        masks[pair] = mask
    return masks


def generate_candidates(
    z: dict[Pair, F64],
    beats: BeatTrain,
    cardiac_windows: dict[Pair, tuple[float, float] | None] | None,
    video: VideoMotion | None = None,
    z_enter: float = 3.0,
    z_exit: float = 1.5,
    min_dur_s: float = 0.020,
    merge_gap_s: float = 0.100,
    duration_cap_s: float | None = None,
    combine: CombineRule = "any",
    k: int = 1,
) -> list[Candidate]:
    """Propose spans from robust z, maximising recall.

    Parameters
    ----------
    z
        ``(signal, band) -> (n_frames,)`` robust z on the shared 10 ms grid, from
        task 06. ``nan`` frames are unassessable and never count as a crossing.
    beats
        The beat train, for stamping peri-R windows onto the grid.
    cardiac_windows
        ``(channel, band) -> (t_start_s, t_stop_s)`` relative to R, from task 02, or
        ``None`` per pair where the band is flat. Pass ``None`` for the whole mapping
        when task 02 has not run; suppression is then skipped and
        :attr:`CandidateReport.cardiac_suppression` records that nobody looked.
    video
        Task 17's motion, or ``None``. Where it exceeds its own threshold the entry
        bar drops to :data:`VIDEO_ASSISTED_Z_ENTER`.
    z_enter, z_exit
        Hysteresis thresholds. 3.0 is a starting value with a defensible range of
        2-4; **task 09 sweeps and pins it**, this module does not tune it.
    min_dur_s, merge_gap_s
        Discard shorter, merge closer.
    duration_cap_s
        Spans longer than this are routed to review, never auto-masked. **This
        function cannot express that routing** - it returns a bare list per the
        task's signature, and :class:`~gems_blanking_v2.types.Candidate` has no
        field for it - so exceeding the cap is logged here and carried properly by
        :attr:`CandidateReport.over_cap`. ``None`` means unmeasured, not uncapped.
    combine, k
        How the pairs combine. See :data:`CombineRule`.

    Returns
    -------
    list[Candidate]
        In time order. Spans over ``duration_cap_s`` are **included**: routing is the
        caller's, and dropping a sustained level shift would hide it.
    """
    if not z:
        return []
    if combine == "k_of_n" and not 1 <= k <= len(z):
        msg = f"k must be between 1 and the {len(z)} pairs supplied, got {k}"
        raise ValueError(msg)
    if z_exit > z_enter:
        msg = f"z_exit {z_exit} must not exceed z_enter {z_enter} - that is not hysteresis"
        raise ValueError(msg)

    lengths = {trace.size for trace in z.values()}
    if len(lengths) != 1:
        msg = f"every pair must be on the same grid, got lengths {sorted(lengths)}"
        raise ValueError(msg)
    n_frames = lengths.pop()

    enter_masks = _crossing_masks(z, z_enter)
    exit_masks = _crossing_masks(z, z_exit)

    if video is not None:
        moving = np.asarray(video.exceeds_threshold(n_frames), dtype=bool)
        assisted = _crossing_masks(z, VIDEO_ASSISTED_Z_ENTER)
        enter_masks = {
            pair: mask | (assisted[pair] & moving) for pair, mask in enter_masks.items()
        }

    if cardiac_windows:
        for pair, suppressed in _cardiac_mask(cardiac_windows, beats, n_frames).items():
            if pair in enter_masks:
                enter_masks[pair] = enter_masks[pair] & ~suppressed
                exit_masks[pair] = exit_masks[pair] & ~suppressed

    entered = _combine(enter_masks, combine, k, n_frames)
    sustained = _combine(exit_masks, combine, k, n_frames)
    spans = _hysteresis(entered, sustained)

    merge_frames = max(int(round(merge_gap_s / GRID_S)), 0)
    spans = _merge(spans, merge_frames)
    min_frames = max(int(round(min_dur_s / GRID_S)), 1)
    spans = [(start, stop) for start, stop in spans if (stop - start) >= min_frames]

    candidates = [
        _candidate(span, z, enter_masks, z_enter=z_enter, assisted=video is not None)
        for span in spans
    ]

    if duration_cap_s is not None:
        over = sum(
            1 for c in candidates if (c.stop_s - c.start_s) > duration_cap_s
        )
        if over:
            log.warning(
                "%d of %d candidates exceed the %.1f s duration cap and must be routed "
                "to review, never auto-masked. This function returns a bare list and "
                "cannot carry that routing - use candidate_report() to get it.",
                over,
                len(candidates),
                duration_cap_s,
            )
    return candidates


def _merge(spans: list[tuple[int, int]], merge_frames: int) -> list[tuple[int, int]]:
    """Join spans separated by fewer than ``merge_frames``.

    Merging crosses unassessable frames as readily as quiet ones. That is deliberate:
    the gap rule is about one event being broken into pieces, and a gap is no evidence
    of a second event whether it is quiet or unmeasured. The alternative reading -
    that an unassessable gap should block merging - is defensible and would produce
    more, shorter candidates.
    """
    if not spans:
        return []
    merged = [spans[0]]
    for start, stop in spans[1:]:
        last_start, last_stop = merged[-1]
        if start - last_stop < merge_frames:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _candidate(
    span: tuple[int, int],
    z: dict[Pair, F64],
    enter_masks: dict[Pair, Bool],
    *,
    z_enter: float,
    assisted: bool,
) -> Candidate:
    """Build one :class:`~gems_blanking_v2.types.Candidate` from a frame span.

    ``provenance`` is ``video_assisted`` only when the span exists **because** the
    bar was lowered - that is, no pair reached the unassisted ``z_enter`` anywhere
    in it. A span that would have been found electrically is electrical whether or
    not the animal happened to be moving, which is what makes the tag informative
    rather than a record of coincidence.
    """
    start, stop = span
    crossed = sorted(pair for pair, mask in enter_masks.items() if bool(mask[start:stop].any()))

    peaks = [
        float(np.nanmax(z[pair][start:stop]))
        for pair in crossed
        if bool(np.isfinite(z[pair][start:stop]).any())
    ]
    peak = max(peaks) if peaks else float("nan")

    provenance: Literal["electrical", "video_assisted"] = "electrical"
    if assisted and crossed:
        reached = any(
            bool(
                (
                    np.asarray(z[pair][start:stop], dtype=np.float64) > z_enter
                ).any()
            )
            for pair in crossed
        )
        if not reached:
            provenance = "video_assisted"

    return Candidate(
        start_s=start * GRID_S,
        stop_s=stop * GRID_S,
        signals=tuple(dict.fromkeys(signal for signal, _band in crossed)),
        bands=tuple(dict.fromkeys(band for _signal, band in crossed)),
        peak_z=peak,
        provenance=provenance,
    )


def candidate_report(
    z: dict[Pair, F64],
    beats: BeatTrain,
    cardiac_windows: dict[Pair, tuple[float, float] | None] | None = None,
    video: VideoMotion | None = None,
    z_enter: float = 3.0,
    z_exit: float = 1.5,
    min_dur_s: float = 0.020,
    merge_gap_s: float = 0.100,
    duration_cap_s: float | None = None,
    combine: CombineRule = "any",
    k: int = 1,
) -> CandidateReport:
    """Generate candidates and the flag rates task 09 sweeps against.

    The task's signature returns a bare list, and it is kept that way; this wrapper
    adds what "record the per-pair and union flag rates in the candidate report"
    asks for, so the two requirements do not have to fight.

    ``duration_cap_s`` is the 99th percentile of labelled segment durations in the
    43 existing recordings, computed from ``*_segment_indices.mat`` and pinned.
    **It is unmeasured as of 2026-09-22** - the shared drive is not mounted on the
    build machine, there is no ``.gems-root`` marker and no label file is reachable -
    so it defaults to ``None``, meaning *no cap applied and that fact recorded*.
    Guessing a number here would put a silent ceiling on event duration.
    """
    candidates = generate_candidates(
        z,
        beats,
        cardiac_windows,
        video=video,
        z_enter=z_enter,
        z_exit=z_exit,
        min_dur_s=min_dur_s,
        merge_gap_s=merge_gap_s,
        duration_cap_s=duration_cap_s,
        combine=combine,
        k=k,
    )

    rates = flag_rates(z, z_enter=z_enter, combine=combine, k=k)
    over_cap = (
        ()
        if duration_cap_s is None
        else tuple(
            index
            for index, candidate in enumerate(candidates)
            if (candidate.stop_s - candidate.start_s) > duration_cap_s
        )
    )

    suppression: Literal["applied", "not measured"] = (
        "applied" if cardiac_windows else "not measured"
    )
    suppressed = 0
    if cardiac_windows and z:
        n_frames = next(iter(z.values())).size
        masks = _cardiac_mask(cardiac_windows, beats, n_frames)
        if masks:
            suppressed = int(np.count_nonzero(np.vstack(list(masks.values())).any(axis=0)))

    report = CandidateReport(
        candidates=candidates,
        flag_rates=rates,
        duration_cap_s=duration_cap_s,
        over_cap=over_cap,
        z_enter=z_enter,
        z_exit=z_exit,
        combine=combine,
        k=k,
        cardiac_suppression=suppression,
        suppressed_frames=suppressed,
    )
    log.info(
        "%d candidates, %.2f%% of frames flagged (union %.2f%%, worst pair %.2f%%) at "
        "z_enter=%.1f combine=%s k=%d; cardiac suppression %s; duration cap %s",
        report.n_candidates,
        100.0 * rates.combined,
        100.0 * rates.union,
        100.0 * max(rates.per_pair.values(), default=0.0),
        z_enter,
        combine,
        k,
        suppression,
        "unset" if duration_cap_s is None else f"{duration_cap_s:.1f} s",
    )
    return report


def flag_rates(
    z: dict[Pair, F64], *, z_enter: float = 3.0, combine: CombineRule = "any", k: int = 1
) -> FlagRates:
    """Return the per-pair, union and combined frame flag rates.

    Three numbers, because invariant 10b is about them being confused. The
    denominator is the **assessable** frames - those with at least one finite pair -
    so a file with a long unmeasured stretch does not report an artificially low rate.
    """
    if not z:
        return FlagRates(per_pair={}, union=0.0, combined=0.0, n_frames=0, n_assessable=0)

    n_frames = next(iter(z.values())).size
    finite_any = np.zeros(n_frames, dtype=bool)
    for trace in z.values():
        finite_any |= np.isfinite(np.asarray(trace, dtype=np.float64))
    n_assessable = int(np.count_nonzero(finite_any))
    denominator = float(n_assessable) if n_assessable else 1.0

    masks = _crossing_masks(z, z_enter)
    per_pair = {
        pair: float(np.count_nonzero(mask)) / denominator for pair, mask in masks.items()
    }
    union = float(np.count_nonzero(_combine(masks, "any", 1, n_frames))) / denominator
    combined = float(np.count_nonzero(_combine(masks, combine, k, n_frames))) / denominator

    return FlagRates(
        per_pair=per_pair,
        union=union,
        combined=combined,
        n_frames=n_frames,
        n_assessable=n_assessable,
    )
