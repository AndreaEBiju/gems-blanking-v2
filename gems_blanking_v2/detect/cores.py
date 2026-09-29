"""Candidate cores: where inside a candidate the evidence actually is.

A candidate runs from ``z_enter`` down to ``z_exit``, is merged across short gaps, is
the union over every (signal, band) pair, and inherits the slow bands' multi-second
envelope windows - so it is much wider than the artifact a person marks (task 09,
"Candidates are much wider than the marks", 2026-09-29). That width is the
generator's design and is kept for recall; this module says where inside it the
evidence sits, for the classifier's features and for candidate adjudication.

A candidate's **core** is each run of frames inside it where some pair is over
``z_enter``, with the pairs that are over it and the peak. Its width breaks down,
frame by frame and exhaustively, into:

- ``core_fast``: some pair of a fast band is over ``z_enter``;
- ``core_slow``: only pairs of slow bands (:data:`SLOW_BANDS`) are over ``z_enter``;
- ``tail_fast`` / ``tail_slow``: no pair is over ``z_enter`` but one is over ``z_exit``
  - the hysteresis tail - fast wins over slow as above;
- ``merge_gap``: no pair is over ``z_exit``, so the frame is in the candidate only
  because a gap shorter than the merge window was bridged.

OUTSIDE THE GENERATION HASH, deliberately: nothing that decides a candidate imports
this module (a test holds that), so computing cores leaves the budget key and the
eligible rounds untouched. It reads the thresholds from the
:class:`~gems_blanking_v2.detect.candidates.CandidateReport` it describes rather than
restating them (invariant 39), and uses the same crossing rule as the generator: a
frame crosses a threshold when its z is finite and strictly above it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.constants import BANDS, GRID_S
from gems_blanking_v2.detect.candidates import CandidateReport

__all__ = [
    "SLOW_BANDS",
    "SLOW_WINDOW_S",
    "Core",
    "WidthBreakdown",
    "candidate_cores",
    "width_breakdown",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
Pair = tuple[str, str]

SLOW_WINDOW_S: Final = 1.0
"""A band whose envelope window is at least this long is a slow band: its window
alone smears an onset by longer than any artifact in the audit is wide."""

SLOW_BANDS: Final[frozenset[str]] = frozenset(
    name for name, spec in BANDS.items() if spec.window_s >= SLOW_WINDOW_S)
"""Derived from :data:`~gems_blanking_v2.constants.BANDS`, not listed: ``0.5-3`` (6 s)
and ``0-2`` (7.5 s)."""


@dataclass(frozen=True, slots=True)
class Core:
    """One run of frames inside a candidate where some pair is over ``z_enter``.

    Times are seconds on the same timeline as the candidate plus ``offset_s``;
    ``pairs`` are every (signal, band) over ``z_enter`` somewhere in the run, sorted;
    ``peak_pair`` is the pair holding ``peak_z``.
    """

    start_s: float
    stop_s: float
    pairs: tuple[Pair, ...]
    peak_z: float
    peak_pair: Pair

    @property
    def duration_s(self) -> float:
        """Width in seconds."""
        return self.stop_s - self.start_s


@dataclass(frozen=True, slots=True)
class WidthBreakdown:
    """A candidate's frames, each counted in exactly one part (see the module doc)."""

    core_fast: int
    core_slow: int
    tail_fast: int
    tail_slow: int
    merge_gap: int

    @property
    def total(self) -> int:
        """Every frame of the candidate: the parts are exhaustive and disjoint."""
        return self.core_fast + self.core_slow + self.tail_fast + self.tail_slow + self.merge_gap

    def seconds(self) -> dict[str, float]:
        """Each part in seconds, on the shared 10 ms grid."""
        return {"core_fast": self.core_fast * GRID_S, "core_slow": self.core_slow * GRID_S,
                "tail_fast": self.tail_fast * GRID_S, "tail_slow": self.tail_slow * GRID_S,
                "merge_gap": self.merge_gap * GRID_S}


def _check(report: CandidateReport) -> None:
    if report.cardiac_suppression == "applied":
        msg = ("cores follow the unsuppressed crossings; this report applied cardiac "
               "suppression, which the cores do not reproduce - extend this module first")
        raise NotImplementedError(msg)


def _frames(report: CandidateReport, index: int) -> tuple[int, int]:
    c = report.candidates[index]
    return int(round(c.start_s / GRID_S)), int(round(c.stop_s / GRID_S))


def _over(z: Mapping[Pair, F64], pairs: list[Pair], threshold: float, a: int, b: int) -> Bool:
    """Per pair (rows, in ``pairs`` order) and frame: finite and strictly above."""
    if not pairs:
        return np.zeros((0, b - a), dtype=bool)
    stack = np.vstack([np.asarray(z[p], dtype=np.float64)[a:b] for p in pairs])
    with np.errstate(invalid="ignore"):
        return np.asarray(np.isfinite(stack) & (stack > threshold), dtype=bool)


def _runs(mask: Bool) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(),
                    strict=True))


def candidate_cores(
    z: Mapping[Pair, F64], report: CandidateReport, *, offset_s: float = 0.0
) -> list[tuple[Core, ...]]:
    """Each candidate's cores, in candidate order; ``offset_s`` is added to every time.

    For :func:`~gems_blanking_v2.detect.chain.detect_region`'s result pass its
    ``z``, ``report`` and ``offset_s=region[0]``. A candidate with no core is
    returned as ``()`` - it can only arise from a video-lowered bar, which the
    chain does not use. Raises ``NotImplementedError`` for a report that applied
    cardiac suppression.
    """
    _check(report)
    pairs = sorted(z)
    out: list[tuple[Core, ...]] = []
    for i in range(len(report.candidates)):
        a, b = _frames(report, i)
        over = _over(z, pairs, report.z_enter, a, b)
        cores = []
        for s, e in _runs(over.any(axis=0) if over.size else np.zeros(b - a, bool)):
            rows = np.flatnonzero(over[:, s:e].any(axis=1))
            best, best_pair = -np.inf, pairs[rows[0]]
            for r in rows:
                v = float(np.nanmax(np.asarray(z[pairs[r]], dtype=np.float64)[a + s:a + e]))
                if v > best:
                    best, best_pair = v, pairs[r]
            cores.append(Core(start_s=(a + s) * GRID_S + offset_s,
                              stop_s=(a + e) * GRID_S + offset_s,
                              pairs=tuple(pairs[r] for r in rows), peak_z=best,
                              peak_pair=best_pair))
        out.append(tuple(cores))
    return out


def width_breakdown(z: Mapping[Pair, F64], report: CandidateReport, index: int) -> WidthBreakdown:
    """Split candidate ``index``'s frames into the five parts of the module doc."""
    _check(report)
    a, b = _frames(report, index)
    pairs = sorted(z)
    slow = [p for p in pairs if p[1] in SLOW_BANDS]
    fast = [p for p in pairs if p[1] not in SLOW_BANDS]

    def anyover(ps: list[Pair], thr: float) -> Bool:
        m = _over(z, ps, thr, a, b)
        return m.any(axis=0) if m.size else np.zeros(b - a, dtype=bool)

    ef, es = anyover(fast, report.z_enter), anyover(slow, report.z_enter)
    xf, xs = anyover(fast, report.z_exit), anyover(slow, report.z_exit)
    core_fast = ef
    core_slow = es & ~ef
    core = ef | es
    tail_fast = ~core & xf
    tail_slow = ~core & ~xf & xs
    gap = ~core & ~xf & ~xs
    return WidthBreakdown(int(core_fast.sum()), int(core_slow.sum()), int(tail_fast.sum()),
                          int(tail_slow.sum()), int(gap.sum()))
