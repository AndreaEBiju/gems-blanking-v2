"""Quality flags from folder names, session pairing, and the exclusion they imply.

**Andrea's rulings, 2026-09-26.** The instrument's block name is the recording's
IDENTITY (the store key never moves); the folder name is authoritative for its
MEANING - animal, condition and quality - because she renames folders to correct
them. A folder marked ``_BAD`` or ``_INCOMPLETE`` is excluded, **and so is its
paired baseline or stim/recovery recording**: out of labelling, training, T hosts
and every corpus. Excluded is not deleted - the flag and the reason are recorded
in the store and nothing is removed from the Drive.

**The pairing rule, stated once and tested.** A pair is the baseline and the
stim/recovery recording of one session: same animal, same trial token, same
condition token, differing only in the ``bl`` / ``sr`` infix, read from the
CORRECTED folder name with any quality flag removed::

    gems_j_t01_ms3_bl_230315   <->   gems_j_t01_ms3_sr_231323

The trailing time-of-day is not part of the key (baseline and stim start minutes
apart). **Names only propose a pair; the acquisition times confirm it** (Andrea,
2026-09-26): baseline starts first and stim/recovery between
``pair_gap_min_lo`` and ``pair_gap_min_hi`` minutes later (5-20), read from
protocol.yaml and the ``.tsq`` headers. The ambiguity check is what makes the
wide window safe: two candidates inside it still refuse.

``sr``, ``stim_rec`` and ``stim_recovery`` are ONE infix (Andrea, 2026-09-26:
older files use the long name, newer the short; each holds 2 min stim then 20 min
recovery), normalised to ``sr`` before matching. A name-matched candidate
outside the window is NOT a partner - the name is a human annotation, the
timestamp is measured (invariant 30's shape).

**Several baselines: the ~10 min one is the baseline** (Andrea, 2026-09-26). A
session can hold aborted starts beside its real baseline, all inside the sr's
window. :func:`pair_sessions` chooses the usable one, and an aborted start takes
no partner with it - before this, four good stim/recovery files were excluded as
the "partner" of a 0.0-1.2 min aborted start.

Every flagged recording ends in exactly one state: PAIRED (its partner chosen by
:func:`pair_sessions`), SUPERSEDED (a baseline inside an sr's window that is not
its baseline - excluded alone), AMBIGUOUS (the choice cannot be made), UNCONFIRMED
(name matches exist, none inside - listed with their gaps for Andrea) or MISSING
(no name match). Only PAIRED excludes a partner; nothing is resolved by guessing.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, NamedTuple

from gems_blanking_v2.io.channel_map import meta_path
from gems_blanking_v2.io.store import GemsStore

__all__ = [
    "BORDERLINE_FRACTION",
    "EXCLUSION_KEY",
    "QUALITY_FLAGS",
    "ExclusionPlan",
    "PairKey",
    "Pairing",
    "duration_verdict",
    "pair_key",
    "pair_sessions",
    "plan_exclusions",
    "quality_flags",
    "read_exclusion",
]

QUALITY_FLAGS: Final = ("BAD", "INCOMPLETE")
"""Folder-name tokens Andrea uses to mark a recording unusable. Whole tokens,
case-insensitive: ``_BAD_``, ``_incomplete`` at the end, ``_INCOMPLETE_sr``."""

_FLAG_RE: Final = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(QUALITY_FLAGS) + r")(?![A-Za-z0-9])", re.IGNORECASE
)

_SESSION_RE: Final = re.compile(
    r"^gems_(?P<animal>[a-z])_(?P<trial>t\d+)_(?P<condition>.+?)_"
    r"(?P<infix>bl|sr|stim_recovery|stim_rec)_\d{6}$",
    re.IGNORECASE,
)
_INFIX_EQUIVALENT: Final = {"bl": "bl", "sr": "sr", "stim_rec": "sr", "stim_recovery": "sr"}
"""One condition, three spellings. Longest-first in the regex so the long form
is not read as a condition token followed by a truncated infix."""
"""``gems_<animal>_<trial>_<condition>_<bl|sr>_<hhmmss>``; the condition may itself
contain underscores (``1_2``, ``3_3``), hence the lazy group."""

EXCLUSION_KEY: Final = "excluded"
"""The ``meta.json`` key holding an exclusion record. Absent means not excluded."""


def quality_flags(folder_name: str) -> tuple[str, ...]:
    """Return the quality flags in a folder name, upper-cased, in order found."""
    return tuple(m.group(1).upper() for m in _FLAG_RE.finditer(folder_name))


def _strip_flags(folder_name: str) -> str:
    stripped = _FLAG_RE.sub("", folder_name)
    return re.sub(r"_+", "_", stripped).strip("_")


class PairKey(NamedTuple):
    """What two recordings of one session share. ``infix`` is what differs."""

    animal: str
    trial: str
    condition: str


def pair_key(folder_name: str) -> tuple[PairKey, str] | None:
    """Return ``(key, infix)`` for a corrected folder name, or ``None``.

    ``None`` when the name does not follow the session convention at all - such a
    recording cannot be paired, and a flagged one is reported as MISSING.
    """
    m = _SESSION_RE.match(_strip_flags(folder_name))
    if m is None:
        return None
    key = PairKey(m.group("animal").upper(), m.group("trial").lower(),
                  m.group("condition").lower())
    return key, _INFIX_EQUIVALENT[m.group("infix").lower()]


@dataclass(frozen=True, slots=True)
class ExclusionPlan:
    """Who is excluded and why, plus the pairings that could not be resolved.

    ``excluded`` maps a session key to its record (written to ``meta.json``).
    ``ambiguous`` / ``unconfirmed`` / ``missing`` list flagged sessions whose
    partner could not be confirmed; the flagged session itself is still excluded.
    ``unconfirmed`` maps to ``[(candidate, gap_min), ...]`` - name matches whose
    start-time gap fell outside the window.
    """

    excluded: dict[str, dict[str, Any]] = field(default_factory=dict)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    unconfirmed: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    superseded: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    """Flagged or short baselines inside a stim/recovery's window that are NOT its
    baseline, because a full one is: ``{aborted bl: [(sr, the sr's baseline)]}``.
    Excluded alone - an aborted start takes no partner with it."""


def _gap_min(bl: datetime, sr: datetime) -> float:
    """Minutes from the baseline's start to the stim/recovery's start."""
    return (sr - bl).total_seconds() / 60.0


@dataclass(frozen=True, slots=True)
class Pairing:
    """Each stim/recovery session's baseline, chosen by time.

    ``baseline_of`` maps an ``sr`` session to its baseline. ``ambiguous`` maps an
    ``sr`` session to the in-window baselines it could not choose between.
    ``in_window`` maps every ``sr`` session to all its in-window baselines (chosen
    or not), so a baseline that lost the choice can say which ``sr`` it lost.
    """

    baseline_of: dict[str, str] = field(default_factory=dict)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    in_window: dict[str, list[str]] = field(default_factory=dict)


def pair_sessions(
    folders: dict[str, str],
    started: dict[str, datetime],
    *,
    gap_lo_min: float,
    gap_hi_min: float,
    unusable: frozenset[str] | set[str] = frozenset(),
) -> Pairing:
    """Choose each stim/recovery session's baseline: same key, by acquisition time.

    Candidates share the :class:`PairKey`, start first, and start
    ``[gap_lo_min, gap_hi_min]`` minutes before the ``sr``. **When several are in
    the window, the ~10 min one is the baseline** (Andrea, 2026-09-26): a session
    can hold aborted starts - 0.0-8 min files the duration rule marks short, or
    folders flagged ``_INCOMPLETE`` / ``_BAD`` - beside the real baseline, and the
    ``sr`` belongs to the usable one. ``unusable`` names those sessions. So:

    * one candidate: it, full or not (an ``sr`` whose only baseline is an aborted
      start genuinely has no baseline, and is paired with it so the exclusion
      follows);
    * several: the single one NOT in ``unusable``;
    * several usable ones, or several all unusable: ambiguous, listed, not guessed.

    Trial tokens are used as written (Andrea: the ``gems_d`` ``t03`` -> ``t01``
    restarts keep their folder names).
    """
    baselines: dict[PairKey, list[str]] = defaultdict(list)
    for session, folder in folders.items():
        parsed = pair_key(folder)
        if parsed is not None and parsed[1] == "bl":
            baselines[parsed[0]].append(session)

    pairing = Pairing()
    for session, folder in sorted(folders.items()):
        parsed = pair_key(folder)
        if parsed is None or parsed[1] != "sr":
            continue
        window = sorted(
            b for b in baselines.get(parsed[0], [])
            if gap_lo_min <= _gap_min(started[b], started[session]) <= gap_hi_min
        )
        if not window:
            continue
        pairing.in_window[session] = window
        full = [b for b in window if b not in unusable]
        if len(window) == 1:
            pairing.baseline_of[session] = window[0]
        elif len(full) == 1:
            pairing.baseline_of[session] = full[0]
        else:
            pairing.ambiguous[session] = full or window
    return pairing


def _exclusion_record(
    folder: str, flags: tuple[str, ...], short: dict[str, Any] | None,
    ruled: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the ``meta.json`` exclusion record for a flagged, ruled or short session.

    Precedence is flag, then ruling, then duration; whatever else also applies is
    kept on the record (``also_ruled``, ``also_short``), never dropped.
    """
    if flags:
        record: dict[str, Any] = {
            "reason": "quality_flag",
            "flags": list(flags),
            "source": f"folder name {folder!r}",
            "ruling": "Andrea 2026-09-26: exclude _BAD/_INCOMPLETE and its pair",
        }
        if ruled is not None:
            record["also_ruled"] = dict(ruled)
        if short is not None:
            record["also_short"] = dict(short)
        return record
    if ruled is not None:
        record = dict(ruled)
        if short is not None:
            record["also_short"] = dict(short)
        return record
    assert short is not None
    return {
        "reason": "duration_below_threshold",
        **short,
        "ruling": "Andrea 2026-09-26: a baseline < 10 min or a stim/recovery "
                  "file < 22 min is incomplete; excluded with its pair",
    }


def _resolve_partner(
    session: str, infix: str, confirmed: list[str], pairing: Pairing
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Return ``(partners, tied, lost)`` for a flagged session, from the pairing.

    An ``sr``'s partner is its chosen baseline. A baseline's partners are the
    ``sr`` sessions that chose it; an in-window ``sr`` that chose another
    baseline is ``lost`` to it, and one that could not choose is ``tied``.
    """
    if infix == "sr":
        chosen = pairing.baseline_of.get(session)
        return ([chosen] if chosen else []), pairing.ambiguous.get(session, []), []
    partners: list[str] = []
    tied: list[str] = []
    lost: list[tuple[str, str]] = []
    for sr in confirmed:
        if pairing.baseline_of.get(sr) == session:
            partners.append(sr)
        elif sr in pairing.baseline_of:
            lost.append((sr, pairing.baseline_of[sr]))
        elif session in pairing.ambiguous.get(sr, []):
            tied.append(sr)
        else:
            lost.append((sr, " | ".join(pairing.ambiguous.get(sr, []))))
    return partners, tied, lost


def _check_ruled(ruled: dict[str, dict[str, Any]], folders: dict[str, str]) -> None:
    """Refuse a ruled exclusion that names no known session or lacks its reason."""
    for session, record in ruled.items():
        if session not in folders:
            msg = f"a ruled exclusion names {session!r}, which is not a known session"
            raise KeyError(msg)
        missing_fields = {"reason", "ruling"} - set(record)
        if missing_fields:
            msg = f"the ruled exclusion of {session!r} has no {sorted(missing_fields)}"
            raise ValueError(msg)


def plan_exclusions(
    folders: dict[str, str],
    started: dict[str, datetime],
    *,
    gap_lo_min: float,
    gap_hi_min: float,
    short: dict[str, dict[str, Any]] | None = None,
    ruled: dict[str, dict[str, Any]] | None = None,
) -> ExclusionPlan:
    """Decide exclusions from ``{session_key: corrected folder name}``.

    Every flagged session is excluded with its flags. Its partner - the one
    candidate with the same :class:`PairKey`, the OTHER infix, and a start-time
    gap in ``[gap_lo_min, gap_hi_min]`` with baseline first - is excluded too,
    naming the flagged session. ``started`` holds each session's ``.tsq`` start.
    The window is required, not defaulted: pairing on names alone is what this
    replaced.

    ``short`` maps sessions that fail the duration rule (see
    :func:`duration_verdict`) to their measurement. They are excluded exactly like
    a folder flag, partner included - the rule ADDS to the flags and never removes
    one: a flagged folder that meets the duration stays excluded.

    ``ruled`` maps sessions excluded by a named ruling to their record, which must
    carry ``reason`` and ``ruling`` (e.g. ``no_stim_monitor``). Excluded like a
    flag, partner included; the partner's record repeats the ruling.
    """
    short = short or {}
    ruled = ruled or {}
    _check_ruled(ruled, folders)
    by_key: dict[tuple[PairKey, str], list[str]] = defaultdict(list)
    for session, folder in folders.items():
        parsed = pair_key(folder)
        if parsed is not None:
            by_key[parsed].append(session)
    # The partner is decided by the pairing, not by "any candidate in the
    # window": an aborted start 14 min before an sr is in its window too, and
    # took the real pair's sr down with it before this (4 sessions).
    flagged_or_short = ({s for s, f in folders.items() if quality_flags(f)}
                        | set(short) | set(ruled))
    pairing = pair_sessions(folders, started, gap_lo_min=gap_lo_min,
                            gap_hi_min=gap_hi_min, unusable=flagged_or_short)

    plan = ExclusionPlan()
    for session, folder in sorted(folders.items()):
        flags = quality_flags(folder)
        if not flags and session not in short and session not in ruled:
            continue
        plan.excluded[session] = _exclusion_record(folder, flags, short.get(session),
                                                   ruled.get(session))
        parsed = pair_key(folder)
        if parsed is None:
            plan.missing.append(session)
            continue
        key, infix = parsed
        other = "sr" if infix == "bl" else "bl"
        candidates = sorted(by_key.get((key, other), []))
        gaps = {
            c: _gap_min(started[session], started[c]) if infix == "bl"
            else _gap_min(started[c], started[session])
            for c in candidates
        }
        confirmed = [c for c in candidates if gap_lo_min <= gaps[c] <= gap_hi_min]
        partners, tied, lost = _resolve_partner(session, infix, confirmed, pairing)
        if not candidates:
            plan.missing.append(session)
        elif not confirmed:
            plan.unconfirmed[session] = [(c, round(gaps[c], 1)) for c in candidates]
        elif len(partners) > 1:
            plan.ambiguous[session] = partners
        elif not partners and tied:
            plan.ambiguous[session] = tied
        elif not partners:
            # A baseline in the sr's window that is not its baseline: the sr has
            # a usable one. The aborted start is excluded alone.
            plan.superseded[session] = lost
        else:
            partner = partners[0]
            plan.excluded.setdefault(partner, {
                "reason": "partner_of_flagged",
                "partner_of": session,
                "source": f"paired with {folders[session]!r} "
                          f"({key.animal}/{key.trial}/{key.condition}, {infix}<->{other}, "
                          f"start gap {gaps[partner]:.1f} min)",
                "ruling": (ruled[session]["ruling"] if session in ruled and not flags
                           else "Andrea 2026-09-26: exclude _BAD/_INCOMPLETE and its pair"),
            })
    return plan


BORDERLINE_FRACTION: Final = 0.10
"""Within this fraction BELOW a duration threshold a recording is borderline:
listed for Andrea rather than decided ("around 10 min" means edge cases exist)."""


def duration_verdict(
    infix: str, duration_min: float, *, min_baseline_min: float, min_sr_min: float
) -> str:
    """Return ``"ok"``, ``"borderline"`` or ``"short"`` for one recording.

    ``short`` (below 90% of its threshold) is incomplete and excluded; within 10%
    below is ``borderline`` - listed, not decided; at or above is ``ok``.
    """
    threshold = min_baseline_min if infix == "bl" else min_sr_min
    if duration_min >= threshold:
        return "ok"
    if duration_min >= threshold * (1.0 - BORDERLINE_FRACTION):
        return "borderline"
    return "short"


def read_exclusion(store: GemsStore, animal: str, session: str) -> dict[str, Any] | None:
    """Return a session's exclusion record from ``meta.json``, or ``None``.

    Absent and ``null`` read identically. An unreadable ``meta.json`` raises
    rather than reading as "not excluded": the safe failure for an exclusion
    check is to stop, not to admit the recording.
    """
    path = meta_path(store, animal, session)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    record = document.get(EXCLUSION_KEY)
    return dict(record) if isinstance(record, dict) and record else None
