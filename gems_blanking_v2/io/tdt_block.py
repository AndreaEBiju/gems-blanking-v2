"""When a recording was acquired, read from the instrument rather than the folder.

**The folder date is not evidence.** 27 TDT blocks sit under ``09042026`` whose
own ``.tsq`` start timestamp says ``2026-09-03`` - they are duplicates of the
09/03 blocks, filed under a date that is not theirs. A field derived from a
directory name is a second source of truth for something the instrument already
recorded, and this one is already known to disagree with reality.

Nothing else in the name supplies a date either. The tank stamp
(``ME_STIM_Andrea-260824-155430``) is shared by **every** block in this corpus,
so it identifies the session, not the block; and the stem's trailing digits are a
time of day only (``gems_h_pre01_160301`` -> 16:03:01). Name plus time-of-day
with no date means two genuine recordings - same animal, same condition, same
second on different days - can collide, and the corpus is past 886 and growing.

Reading this costs a 40-byte seek. No data is decoded.
"""

from __future__ import annotations

import re
import struct
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

from gems_blanking_v2.io.store import validate_component

__all__ = [
    "SESSION_KEY_SUFFIX",
    "TSQ_RECORD",
    "AcquisitionRecordError",
    "acquired_at",
    "block_name",
    "directory_disagrees",
    "find_tsq",
    "local_date",
    "session_key",
]


class AcquisitionRecordError(ValueError):
    """The acquisition record beside a recording contradicts itself."""


_TANK_BLOCK: Final = re.compile(r"^(?P<tank>.+?-\d{6}-\d{6})_(?P<block>.+)$")
"""``ME_STIM_Andrea-260824-155430_gems_j_t01_bl_120000`` -> block
``gems_j_t01_bl_120000``. Every tank in both cohorts has this form."""

TSQ_RECORD: Final = 40
"""Bytes per ``.tsq`` record. Record 0 is the file header; record 1 is the first
event and carries the block's start timestamp as a Unix epoch double."""

_TSQ_FORMAT: Final = "<iiiHHdqif"
"""size, type, code, channel, sortcode, timestamp, ev, format, fs."""

_TIMESTAMP_FIELD: Final = 5


def find_tsq(recording: Path) -> Path | None:
    """Return the ``.tsq`` beside a converted recording, or ``None``.

    ``None`` is a real answer: a recording may be reachable without its source
    block - the old cohort has no TDT block at all - and the caller decides
    whether that is acceptable rather than this function guessing a date.
    """
    found = sorted(Path(recording).parent.glob("*.tsq"))
    return found[0] if found else None


def acquired_at(recording: Path) -> str | None:
    """Return the block's start time as a UTC ISO-8601 string, or ``None``.

    UTC, and stated as such, because the correspondence with the stem is only
    obvious in local time and that is exactly the kind of implicit conversion
    that produces an off-by-one-day. ``gems_j_t01_ms3_bl_230315`` reads
    ``2026-09-16T03:03:21Z``, which is 23:03 on the 15th in EDT - the folder
    ``09152026`` is right and the UTC date differs, so neither can be compared
    to the other without saying which zone it is in.
    """
    tsq = find_tsq(recording)
    if tsq is None:
        return None
    try:
        with tsq.open("rb") as f:
            f.seek(TSQ_RECORD)
            record = f.read(TSQ_RECORD)
    except OSError:
        return None
    if len(record) != TSQ_RECORD:
        return None
    ts = struct.unpack(_TSQ_FORMAT, record)[_TIMESTAMP_FIELD]
    if not ts or ts <= 0:
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


def block_name(recording: Path) -> str | None:
    """Return the TDT block a recording file belongs to, or ``None``.

    A block directory holds every file derived from one acquisition - ``_sig``,
    ``_notched``, ``_stim``, ``_vib`` - so the block, not the file stem, is what
    they share. TDT names the ``.tsq`` ``<tank>_<block>`` with a tank of the form
    ``<experiment>-YYMMDD-HHMMSS``, and the block name is read from THAT - the
    instrument's record - never from the directory (invariant 30).

    **The directory is not the block name, measured.** 37 of 921 block
    directories in the new cohort were renamed after acquisition: status flags
    (``_INCOMPLETE``, ``_BAD``), condition edits (``ms2`` -> ``ms3``), even the
    animal letter. Renames are ongoing, so a key read from the directory would
    change under a recording - orphaning its ``meta.json`` and, later, its labels
    - the next time someone annotates a folder. The instrument's name does not
    move. Whether a rename was a CORRECTION is a separate question for the
    person who made it; see :func:`directory_disagrees`.

    Raises
    ------
    AcquisitionRecordError
        If the directory holds more than one ``.tsq``, or the ``.tsq`` name is not
        ``<tank>-YYMMDD-HHMMSS_<block>``.
    """
    found = sorted(Path(recording).parent.glob("*.tsq"))
    if not found:
        return None
    where = Path(recording).parent
    if len(found) > 1:
        msg = (f"{where}: {len(found)} .tsq files; one block per directory is what "
               "makes the directory a block. Refusing to pick one.")
        raise AcquisitionRecordError(msg)
    m = _TANK_BLOCK.match(found[0].stem)
    if m is None:
        msg = (f"{found[0].name}: cannot read a block name - expected "
               "<experiment>-YYMMDD-HHMMSS_<block>.tsq")
        raise AcquisitionRecordError(msg)
    return m.group("block")


def directory_disagrees(recording: Path) -> tuple[str, str] | None:
    """Return ``(instrument block, directory)`` when they differ, else ``None``.

    A difference is information, not an error: it is usually a human annotation
    made after acquisition, and it is reported so the person who made it can say
    whether it corrects the instrument's record or only annotates it.
    """
    block = block_name(recording)
    folder = Path(recording).parent.name
    if block is None or block.casefold() == folder.casefold():
        return None
    return block, folder


SESSION_STAMP: Final = "%Y%m%dT%H%M%SZ"
"""UTC instant in the store key. Compact because ``:`` is illegal in a Windows
path, and a full instant rather than a date so the key never has to decide which
"day" a recording belongs to (invariant 31)."""

SESSION_KEY_SUFFIX: Final = re.compile(r"_\d{8}T\d{6}Z$")
"""Matches what :data:`SESSION_STAMP` appends, so a directory can be recognised as
keyed or not without re-deriving the format elsewhere. A test holds the two
together."""


def session_key(recording: Path) -> str | None:
    """Return the store key for the acquisition a recording came from, or ``None``.

    ``<block>_<UTC start, YYYYMMDDTHHMMSSZ>``. The ONE place a session key is
    built (invariant 33): the generator, the loader and the scan all call this,
    so a file and the ``meta.json`` describing it cannot be keyed two ways.

    Why both parts: the block name's trailing digits are a time of day with no
    date, so two genuine recordings started at the same clock second on
    different days share it. The start time comes from the ``.tsq`` - never the
    folder, which is wrong for 27 blocks (invariant 30) - and is identical
    across a copied block, so duplicates still collapse onto one key.

    ``None`` when there is no acquisition record: such a file has no store key,
    and a stem is not a substitute - it is exactly the dateless key this replaces.
    """
    block = block_name(recording)
    when = acquired_at(recording) if block is not None else None
    if block is None or when is None:
        return None
    stamp = datetime.fromisoformat(when).astimezone(UTC).strftime(SESSION_STAMP)
    return validate_component(f"{block}_{stamp}")


def local_date(acquired_at_utc: str, zone: str) -> date:
    """Return the calendar date a recording was made, in the lab's own zone.

    **The convention for every grouping by day is the LOCAL date.** Session
    grouping, baseline/recovery pairing, per-day motility state and LORO splits
    all mean the day the animal was run. ``gems_j_t01_ms3_bl_230315`` is 23:03
    EDT on the 15th and 03:03 UTC on the 16th; grouping on the UTC date files
    every evening recording under the next day (invariant 31). ``zone`` is an
    IANA name, never an offset: a fixed -4 is wrong after the November change.
    """
    return datetime.fromisoformat(acquired_at_utc).astimezone(ZoneInfo(zone)).date()
