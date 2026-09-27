"""Which recordings the blind recall audit may offer, and which seconds of each.

The audit (task 16 Change 2) is the human critical path to the 09 gate, and its
validity rests on two exclusions made BEFORE a span is drawn:

* **Excluded recordings are never offered.** Andrea's ruling (``_BAD`` /
  ``_INCOMPLETE``, short by duration, and their pairs) is recorded in each
  ``meta.json``; a recording carrying it does not appear in the pool at all,
  rather than appearing and being refused later.
* **Stim epochs are removed from stim/recovery recordings.** A span that overlaps
  stimulation shows the labeller stimulation artifact, they mark it as motion,
  and recall looks fine while measuring the wrong thing. The epoch comes from the
  protocol's stim window at the start of the file (0-132 s): the monitor channel
  that shows stimulation differs by modality, so a single-channel split can find
  a spurious epoch (see :func:`stim_epoch_s`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from gems_blanking_v2.io.quality import EXCLUSION_KEY, pair_key
from gems_blanking_v2.io.stim_split import ProtocolSpec
from gems_blanking_v2.io.store import GemsStore
from gems_blanking_v2.types import Recording

__all__ = [
    "PLANNABLE_EPOCHS",
    "EligibleRecording",
    "assessable_regions",
    "eligible_recordings",
    "planned_regions",
    "stim_epoch_s",
]

EPOCH_BY_INFIX: Final = {"bl": "baseline", "sr": "stim_recovery"}


@dataclass(frozen=True, slots=True)
class EligibleRecording:
    """One recording the audit may offer. ``source`` is resolved on this machine."""

    animal: str
    session: str
    folder_name: str
    source: Path
    epoch: str
    acquired_at: str
    duration_s: float | None = None
    """From ``meta.json`` (sample count / fs). None for an entry written before
    the generator recorded it; such a recording cannot be planned, only opened."""


def eligible_recordings(store: GemsStore) -> list[EligibleRecording]:
    """Every non-excluded recording in the store, sorted by animal then time.

    Reads each ``meta.json``. A recording is left out when it carries an
    exclusion, has no ``source_path``, or its source is not present on this
    machine - never offered and then refused.
    """
    out: list[EligibleRecording] = []
    data = store.root / "data"
    if not data.is_dir():
        return out
    for meta in sorted(data.glob("*/*/meta.json")):
        doc = json.loads(meta.read_text(encoding="utf-8"))
        if doc.get(EXCLUSION_KEY) or not doc.get("source_path"):
            continue
        source = store.root / doc["source_path"]
        if not source.is_file():
            continue
        folder = str(doc.get("folder_name") or source.parent.name)
        parsed = pair_key(folder)
        out.append(EligibleRecording(
            animal=str(doc["animal"]), session=str(doc["session"]), folder_name=folder,
            source=source, epoch=EPOCH_BY_INFIX.get(parsed[1], "unknown") if parsed else "unknown",
            acquired_at=str(doc.get("acquired_at", "")),
            duration_s=None if doc.get("duration_s") is None else float(doc["duration_s"]),
        ))
    out.sort(key=lambda r: (r.animal, r.acquired_at))
    return out


def stim_epoch_s(
    rec: Recording | None, source: Path, epoch: str, protocol: ProtocolSpec
) -> tuple[tuple[float, float] | None, str]:
    """Return ``((start_s, stop_s) | None, method)`` for the recording's stim epoch.

    ``None`` for a baseline. For stim/recovery, the protocol's stim window at the
    START of the file, widened by its tolerance: ``(0, stim_duration_s +
    stim_tolerance_s)`` - 0-132 s for the chronic protocol.

    **Why not a monitor-based split, measured 2026-09-26.** The stimulation shows
    on a DIFFERENT monitor channel by modality: ``ms`` recordings on ``ADC2`` only,
    ``es`` recordings on ``vib``/``adc1`` only, combined conditions on all three.
    A split reading ``vib`` alone found a spurious "stim epoch" at 1163-1283 s in
    ``gems_j_t01_ms1_sr_165543``, whose ``vib`` is flat throughout while ``ADC2``
    shows the stimulation at 0-120 s. Across a 10-file sample every file's
    stimulation sat at 0-120 s on some channel. The protocol window matches all of
    them and cannot pick the wrong channel; excluding a few seconds too many is
    safe, a span on real stimulation is not. ``rec`` and ``source`` are kept in the
    signature for a monitor split once task 03B chooses the channel by modality.
    """
    if epoch != "stim_recovery":
        return None, "not_stim_recovery"
    stop = float(protocol.stim_duration_s + protocol.stim_tolerance_s)
    return (0.0, stop), "protocol_window"


PLANNABLE_EPOCHS: Final = ("baseline", "stim_recovery")
"""Only classified recordings enter an audit plan: an ``unknown`` condition cannot
enter a corpus (scan.corpus_eligible), so recall measured on it reads for nothing."""


def planned_regions(
    rec: EligibleRecording, protocol: ProtocolSpec
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]] | None:
    """``(assessable regions, excluded)`` from the store alone, or ``None``.

    ``None`` when the recording cannot be planned: no recorded duration, or a
    condition outside :data:`PLANNABLE_EPOCHS`. Uses the same stim rule as an
    opened span (:func:`stim_epoch_s`), so planning and marking cannot disagree
    about where stimulation is.
    """
    if rec.duration_s is None or rec.epoch not in PLANNABLE_EPOCHS:
        return None
    stim, _method = stim_epoch_s(None, rec.source, rec.epoch, protocol)
    regions = assessable_regions(rec.duration_s, stim)
    return regions, ((stim,) if stim is not None else ())


def assessable_regions(
    duration_s: float, stim: tuple[float, float] | None
) -> tuple[tuple[float, float], ...]:
    """``[0, duration)`` with the stim epoch removed, as sorted disjoint regions."""
    if stim is None:
        return ((0.0, float(duration_s)),)
    lo, hi = max(0.0, stim[0]), min(float(duration_s), stim[1])
    regions = []
    if lo > 0.0:
        regions.append((0.0, lo))
    if hi < duration_s:
        regions.append((hi, float(duration_s)))
    return tuple(regions)
