"""Enforcement for hard invariant 1: a masked sample is ``NaN``, never ``0``.

The MATLAB consumer, ``processing_new/step1_bandpass.m``, decides what is masked
with one line::

    invalid = isnan(x);

There is no test for zero anywhere in it. A zero-filled gap is therefore filtered
as if it were signal: a hard step to 0 V at each boundary, rung through an
order-4 Butterworth applied twice by ``filtfilt``, leaking into adjacent *valid*
samples on both sides because the filter is zero-phase and so non-causal.

The writer side of this was fixed upstream - ``GEMSBlanking`` commit ``a95d1ff``
set ``labeled_save.BLANK_FILL_VALUE`` to ``NaN`` - so new exports are correct. What
remains is (a) keeping anything this package emits correct, and (b) recognising
the files written before that commit, which still contain zeroed gaps and which
nothing currently detects.

**This module never converts zeros to NaN.** Zero is a legal signal value, and
inferring invalidity from it is precisely the bug; a file that needs converting
needs re-exporting from its labels instead. The audit here only reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "MAX_LEGAL_ZERO_RUN",
    "ZeroRun",
    "ZeroRunReport",
    "assert_no_zero_runs",
    "audit_array",
    "find_zero_runs",
    "masked_count",
]

F64 = npt.NDArray[np.float64]

_MAX_RUNS_SHOWN: Final = 10
"""How many zero-run spans a summary lists before truncating."""

MAX_LEGAL_ZERO_RUN: Final = 2
"""Longest exact-zero run an emitted array may contain (hard invariant 1).

Two consecutive zeros occur in real signal often enough to be unremarkable; a
third is, in practice, a fill. The threshold is a convention, not a measurement -
it is the line invariant 1 draws, and this module exists to hold it.
"""


@dataclass(frozen=True, slots=True)
class ZeroRun:
    """One run of exact zeros found in a channel.

    Attributes
    ----------
    channel
        Column index the run was found in.
    start, stop
        Sample bounds, ``stop`` exclusive.
    """

    channel: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        """Run length in samples."""
        return self.stop - self.start

    def duration_s(self, fs: float) -> float:
        """Run length in seconds at ``fs`` Hz."""
        return self.length / fs


def find_zero_runs(x: F64, min_len: int = MAX_LEGAL_ZERO_RUN + 1) -> list[ZeroRun]:
    """Return every run of at least ``min_len`` exact zeros in ``x``.

    Parameters
    ----------
    x
        1-D or 2-D ``(n_samples, n_channels)`` array. NaN is not zero and is
        ignored; a run is broken by any non-zero value, NaN included.
    min_len
        Shortest run to report. The default reports exactly what invariant 1
        forbids.

    Returns
    -------
    list[ZeroRun]
        In channel order, then sample order. Empty when the array is clean.
    """
    if min_len < 1:
        msg = f"min_len must be at least 1, got {min_len}"
        raise ValueError(msg)
    if x.ndim not in (1, 2):
        msg = f"x must be 1-D or 2-D, got shape {x.shape}"
        raise ValueError(msg)
    matrix = x[:, None] if x.ndim == 1 else x

    runs: list[ZeroRun] = []
    for ch in range(matrix.shape[1]):
        is_zero = matrix[:, ch] == 0.0
        if not is_zero.any():
            continue
        # Edges of the True runs, via a padded diff - no Python loop over samples.
        padded = np.concatenate(([False], is_zero, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, stop in zip(changes[::2], changes[1::2], strict=True):
            if stop - start >= min_len:
                runs.append(ZeroRun(channel=ch, start=int(start), stop=int(stop)))
    return runs


def masked_count(x: F64) -> int:
    """Return the number of NaN samples in ``x``, summed over all channels.

    This is the quantity that must survive a write and a read: the MATLAB side
    counts the same thing with ``nnz(isnan(...))``.
    """
    return int(np.count_nonzero(np.isnan(x)))


def assert_no_zero_runs(x: F64, *, what: str = "array", fs: float | None = None) -> None:
    """Raise if ``x`` violates hard invariant 1.

    Call this on anything this package emits, before it is written.

    Raises
    ------
    ValueError
        Naming the offending channel and span, and the duration when ``fs`` is
        given, because "there is a zero run somewhere" is not actionable.
    """
    runs = find_zero_runs(x)
    if not runs:
        return
    worst = max(runs, key=lambda r: r.length)
    extent = f"{worst.length} samples"
    if fs is not None:
        extent += f" ({1000.0 * worst.duration_s(fs):.1f} ms)"
    msg = (
        f"{what} contains {len(runs)} exact-zero run(s) longer than "
        f"{MAX_LEGAL_ZERO_RUN} samples, which hard invariant 1 forbids: longest is "
        f"{extent} in channel {worst.channel} at samples [{worst.start}, {worst.stop}). "
        "A masked sample must be NaN - step1_bandpass.m tests isnan only, so a zeroed "
        "gap is filtered as signal."
    )
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ZeroRunReport:
    """What an audit found. Diagnostic only - nothing here converts anything."""

    source: str
    n_samples: int
    n_channels: int
    nan_count: int
    zero_runs: list[ZeroRun]

    @property
    def is_clean(self) -> bool:
        """Whether the array satisfies invariant 1."""
        return not self.zero_runs

    @property
    def predates_the_nan_fix(self) -> bool:
        """Whether this looks like an export from before ``GEMSBlanking`` ``a95d1ff``.

        A file with long zero runs and **no** NaN at all was written by the old
        zero-filling writer. A file with both has either been partly repaired or
        contains genuine zero-valued signal, and needs a human to look at it, which
        is why this is reported rather than acted on.
        """
        return bool(self.zero_runs) and self.nan_count == 0

    def summary(self, fs: float | None = None) -> str:
        """Return a human-readable report."""
        lines = [
            f"source:     {self.source}",
            f"shape:      {self.n_samples} x {self.n_channels}",
            f"NaN:        {self.nan_count}",
            f"zero runs:  {len(self.zero_runs)} longer than {MAX_LEGAL_ZERO_RUN} samples",
        ]
        for run in self.zero_runs[:_MAX_RUNS_SHOWN]:
            extent = f"{run.length} samples"
            if fs is not None:
                extent += f" ({1000.0 * run.duration_s(fs):.1f} ms)"
            lines.append(f"    ch {run.channel}: [{run.start}, {run.stop})  {extent}")
        if len(self.zero_runs) > _MAX_RUNS_SHOWN:
            lines.append(f"    ... and {len(self.zero_runs) - _MAX_RUNS_SHOWN} more")
        if self.predates_the_nan_fix:
            lines.append(
                "verdict:    zero-filled and NaN-free - written before the NaN fix. "
                "Re-export it from its labels; do not convert the zeros in place, "
                "because zero is a legal signal value."
            )
        elif self.zero_runs:
            lines.append(
                "verdict:    zero runs alongside NaN - inspect by hand. This is either "
                "a partly repaired file or genuine zero-valued signal."
            )
        else:
            lines.append("verdict:    clean")
        return "\n".join(lines)


def audit_array(x: F64, source: str | Path = "array") -> ZeroRunReport:
    """Report the zero runs and NaN count of an array, without modifying it."""
    matrix = x[:, None] if x.ndim == 1 else x
    return ZeroRunReport(
        source=str(source),
        n_samples=int(matrix.shape[0]),
        n_channels=int(matrix.shape[1]),
        nan_count=masked_count(matrix),
        zero_runs=find_zero_runs(matrix),
    )
