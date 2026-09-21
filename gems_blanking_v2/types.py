"""Core data structures shared by every stage of the pipeline (Part A.1).

These live here rather than in :mod:`gems_blanking_v2.io` because the synthetic
generators in ``tests/conftest.py`` must build a :class:`Recording` before the real
loader exists (task 03), and two definitions of the same contract is how the two
halves drift apart.

Units: time in seconds (``float64``) in public APIs, sample indices (``int64``) with
an explicit ``fs`` internally; amplitude in microvolts (``float64``); frequency in Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "Candidate",
    "ChannelInfo",
    "Event",
    "Recording",
    "TrainingMode",
]


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """One acquired channel and where it sits on the animal.

    Attributes
    ----------
    index
        Column in the raw sample matrix.
    name
        Acquisition name, e.g. ``"RVN"``, ``"ANT1"``.
    role
        What the channel records.
    cuff_id
        ``"L"`` / ``"R"`` for nerve channels; ``None`` for stomach and aux.
    contact_index
        1..3 along the cuff; ``None`` if the channel is not a cuff contact.
    rostral_end
        Which ``contact_index`` is rostral; ``None`` when unknown. Never guessed -
        an unknown orientation makes the sign of a conduction delay unknown too.
    config
        ``"hw_tripole"`` when the tripole was formed in hardware (the old cohort, in
        which the individual contacts are unrecoverable), ``"independent"`` when the
        contacts were recorded separately.
    """

    index: int
    name: str
    role: Literal["nerve", "stomach", "aux"]
    cuff_id: str | None
    contact_index: int | None
    rostral_end: int | None
    config: Literal["hw_tripole", "independent"]


@dataclass(frozen=True, slots=True, eq=False)
class Recording:
    """One loaded recording.

    ``eq`` is disabled: the payload is an array, and dataclass equality on an array
    raises rather than answering.

    Attributes
    ----------
    fs
        Sample rate in Hz. Nominally 24414.0625 on TDT hardware - read it from the
        file, never hardcode it.
    data
        ``(n_samples, n_channels)`` ``float64`` in microvolts. Masked samples are
        ``NaN``, never ``0`` (hard invariant 1).
    channels
        One entry per column of ``data``, in column order.
    animal
        Animal identifier, e.g. ``"F"``, ``"J"``, ``"L"``, ``"O"``.
    session
        Session identifier within the animal.
    path
        Local absolute path to the source file. Runtime only - anything written to
        the shared drive stores a POSIX path relative to ``gems_root`` instead
        (cross-platform rule 2).
    """

    fs: float
    data: npt.NDArray[np.float64]
    channels: list[ChannelInfo]
    animal: str
    session: str
    path: Path

    @property
    def n_samples(self) -> int:
        """Number of samples in the recording."""
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels in the recording."""
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration in seconds, ``n_samples / fs``."""
        return self.n_samples / self.fs

    def channel(self, name: str) -> ChannelInfo:
        """Return the channel called ``name``.

        Raises
        ------
        KeyError
            If no channel has that name.
        """
        for ch in self.channels:
            if ch.name == name:
                return ch
        msg = f"no channel named {name!r}; have {[c.name for c in self.channels]}"
        raise KeyError(msg)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A span proposed by the detector, before any judgement.

    Attributes
    ----------
    start_s, stop_s
        Span bounds in seconds from the start of the recording.
    signals
        Derived signal names that crossed threshold, cuff-prefixed (``"L_V1"``).
    bands
        Band names that crossed, keys of :data:`gems_blanking_v2.constants.BANDS`.
    peak_z
        Largest robust z reached anywhere in the span, across the crossing pairs.
    provenance
        How the span was proposed.
    """

    start_s: float
    stop_s: float
    signals: tuple[str, ...]
    bands: tuple[str, ...]
    peak_z: float
    provenance: Literal["electrical", "video_assisted"]

    @property
    def duration_s(self) -> float:
        """Span length in seconds."""
        return self.stop_s - self.start_s


class TrainingMode(StrEnum):
    """Which corpus a model was trained on.

    The three modes share every step of the pipeline and differ only in training
    corpus and evaluation protocol. There is no default and no fallback chain: the
    user picks the model for every inference run and the choice is recorded in
    provenance.
    """

    POOLED = "pooled"
    """All animals except the target. The mandatory mode - the only one that can
    process an animal with no labels."""

    ADAPTED = "adapted"
    """Pooled prior plus the target animal's own labelled events. Matches deployment."""

    PER_ANIMAL = "per_animal"
    """The target animal only."""


@dataclass(frozen=True, slots=True)
class Event:
    """A :class:`Candidate` after judgement (step 08).

    Attributes
    ----------
    candidate
        The span this judgement is about.
    judgement
        ``"unjudged"`` means no human looked at it. It is **not** ``"physiology"``
        and never enters training as a clean example (hard invariant 9).
    p_motion
        Model probability that the span is motion; ``nan`` when no model scored it.
    source
        Who produced the judgement: a human, a model, or inheritance from an
        overlapping judged span.
    """

    candidate: Candidate
    judgement: Literal["motion", "physiology", "unsure", "unjudged"]
    p_motion: float
    source: Literal["human", "model", "inherited"]
