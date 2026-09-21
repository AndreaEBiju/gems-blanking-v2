"""The model registry: an append-only log on a sync layer that has no locking.

Current state is computed by **replaying** the log, never by reading a field in a
mutable pointer file. Each client appends to its own shard, readers take the union
of all shards, and union-of-lines is order-independent and idempotent - so a Drive
conflict copy merges correctly by construction rather than by luck.

What this does *not* do is make Drive transactional. Two users training the same
corpus at once produce two valid models and two log lines; a human decides which is
promoted. That is correct for a research tool, and the UI says so rather than
pretending the conflict cannot happen.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from gems_blanking_v2.io.store import (
    GemsStore,
    append_line,
    atomic_write_text,
    read_lines,
    safe_component,
    utc_stamp,
)

__all__ = [
    "ModelState",
    "RegistryAction",
    "RegistryEvent",
    "append_event",
    "compact",
    "iter_shard_paths",
    "read_events",
    "replay",
    "shard_path",
]

LOG_NAME: Final = "events.jsonl"
"""The compacted log. Shards are ``events.jsonl.<user>.<utc>`` beside it."""

_CONFLICT_RE: Final = re.compile(r"^events\.jsonl(\.[^/\\]*)?( \(\d+\))?$")
"""Matches the compacted log, any shard, and any Drive conflict copy of either.

Drive for desktop resolves a clashing write by appending ``" (1)"`` to the name. A
conflict copy holds real lines that a reader must not drop, so it is a shard too.
"""


def _optional_str(raw: dict[str, Any], key: str) -> str:
    """Read an optional string field, treating absent and ``null`` alike.

    ``str(raw.get(key, ""))`` looks equivalent and is not: when the key is present
    and ``null`` it yields the four-character string ``"None"``, inventing a value
    and making the same fact fail to dedupe across two spellings of "unset".
    """
    value = raw.get(key)
    return "" if value is None else str(value)


def _required_str(raw: dict[str, Any], key: str, line: str) -> str:
    """Read a required string field, or raise naming it.

    A required field that is absent or ``null`` makes the line malformed. It is
    never defaulted: a registry line with no ``model_id`` is not an event about the
    empty model, it is a corrupt line, and inventing a value would put a fact in the
    log that nobody wrote.
    """
    value = raw.get(key)
    if value is None:
        msg = f"registry line is missing required field {key!r}: {line[:120]!r}"
        raise ValueError(msg)
    return str(value)


class RegistryAction(StrEnum):
    """What happened to a model. State is the replay of these, in timestamp order."""

    TRAINED = "trained"
    PROMOTED = "promoted"
    DEMOTED = "demoted"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class RegistryEvent:
    """One line of the registry log.

    Attributes
    ----------
    ts
        UTC timestamp, ``YYYYmmddTHHMMSSZ`` - the same filename-safe form the shard
        names use, so a line and its file never disagree about a colon.
    user
        Filename-safe acting user (see ``store.resolve_user_id``).
    action
        One of :class:`RegistryAction`.
    model_id
        Content hash of the model; immutable and never reused.
    mode
        Training mode: ``pooled`` / ``adapted`` / ``per_animal``.
    animal
        Target animal, or ``""`` where the action is not animal-specific.
    corpus_id
        Corpus the model was trained on.
    metrics
        Free-form metrics dict, carried so the registry can present options without
        opening every model directory. Always a dict, never None: two spellings of
        "no metrics" would make ``from_json_line(to_json_line(e)) == e`` false, and
        that round-trip is what lets a conflict copy dedupe against its original.

    Events are compared and deduplicated by their canonical line, not by hash - the
    ``metrics`` dict makes an instance unhashable, which is deliberate.
    """

    ts: str
    user: str
    action: RegistryAction
    model_id: str
    mode: str = ""
    animal: str = ""
    corpus_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a non-finite metric, which JSON cannot represent.

        ``json.dumps`` emits bare ``NaN`` / ``Infinity`` - accepted by Python, invalid
        JSON for every other reader - and ``nan != nan``, so such an event does not
        even equal its own reparse. The project convention that a missing scalar is
        ``np.nan`` stops at the edge of a JSON file: a metric that could not be
        computed is **omitted**, and an absent key reads back as absent.
        """
        for key, value in self.metrics.items():
            if not math.isfinite(value):
                msg = (
                    f"metric {key!r} is {value!r}, which JSON cannot represent; "
                    "omit the key instead of storing a non-finite value"
                )
                raise ValueError(msg)

    def to_json_line(self) -> str:
        r"""Return the canonical one-line JSON form.

        Canonical means sorted keys, no incidental whitespace and ASCII escaping, so
        two clients logging the same fact produce byte-identical lines and the union
        dedupes them. Without that, a conflict copy would double every entry it
        contains.

        ``ensure_ascii`` is left at its default **on purpose**. With it off, a text
        field containing U+2028, U+2029 or U+0085 lands in the line literally, and
        although this module's own reader splits on ``\n`` alone, any reader using
        ``str.splitlines()`` - or another language's line splitter - would see one
        event as two malformed lines. A JSONL file on a shared drive is read by other
        people's tools, so the line stays pure ASCII; unicode still round-trips
        exactly, escaped as ``\uXXXX``.
        """
        payload: dict[str, Any] = {
            "ts": self.ts,
            "user": self.user,
            "action": str(self.action),
            "model_id": self.model_id,
            "mode": self.mode,
            "animal": self.animal,
            "corpus_id": self.corpus_id,
            "metrics": dict(self.metrics),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json_line(cls, line: str) -> RegistryEvent:
        """Parse one log line.

        Raises
        ------
        ValueError
            If the line is not valid JSON or names an unknown action. A malformed
            line is never skipped silently - a half-synced log that quietly loses
            entries is the failure this design exists to prevent.
        """
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"malformed registry line: {line[:120]!r}"
            raise ValueError(msg) from exc
        if not isinstance(raw, dict):
            msg = f"registry line is not a JSON object: {line[:120]!r}"
            raise ValueError(msg)
        try:
            action = RegistryAction(raw["action"])
        except (KeyError, ValueError) as exc:
            msg = f"registry line has an unknown action: {line[:120]!r}"
            raise ValueError(msg) from exc
        return cls(
            ts=_required_str(raw, "ts", line),
            user=_optional_str(raw, "user"),
            action=action,
            model_id=_required_str(raw, "model_id", line),
            mode=_optional_str(raw, "mode"),
            animal=_optional_str(raw, "animal"),
            corpus_id=_optional_str(raw, "corpus_id"),
            metrics=dict(raw.get("metrics") or {}),
        )


def shard_path(store: GemsStore, user: str, stamp: str | None = None) -> Path:
    """Return this client's own shard path, ``events.jsonl.<user>.<utc>``.

    Both components are sanitised: a git ``user.name`` is usually ``First Last``,
    and an ISO timestamp would carry colons, which Windows forbids in a filename.
    """
    return store.registry_dir / f"{LOG_NAME}.{safe_component(user)}.{stamp or utc_stamp()}"


def append_event(store: GemsStore, event: RegistryEvent, *, stamp: str | None = None) -> Path:
    """Append ``event`` to the acting user's own shard and return the shard path.

    Two users appending at once produce two files, not a conflict. Nothing is ever
    rewritten in place.
    """
    path = shard_path(store, event.user, stamp)
    append_line(path, event.to_json_line())
    return path


def iter_shard_paths(store: GemsStore) -> list[Path]:
    """Return every file a reader must union: the log, its shards, conflict copies."""
    if not store.registry_dir.is_dir():
        return []
    return sorted(
        p for p in store.registry_dir.iterdir() if p.is_file() and _CONFLICT_RE.match(p.name)
    )


def read_events(store: GemsStore) -> list[RegistryEvent]:
    """Return the union of all shards, deduplicated, in a deterministic order.

    Deduplication is on the canonical line, so the same event appearing in a shard
    and in its Drive conflict copy counts once. Ordering is by ``(ts, model_id,
    action, user)`` rather than by file, so the result does not depend on which
    client's shard was read first.
    """
    seen: set[str] = set()
    events: list[RegistryEvent] = []
    for path in iter_shard_paths(store):
        for line in read_lines(path):
            event = RegistryEvent.from_json_line(line)
            canonical = event.to_json_line()
            if canonical in seen:
                continue
            seen.add(canonical)
            events.append(event)
    return sorted(events, key=lambda e: (e.ts, e.model_id, str(e.action), e.user))


@dataclass(frozen=True, slots=True)
class ModelState:
    """Replayed state of one model.

    Attributes
    ----------
    model_id
        The model this state is about.
    status
        The last action applied to it, in timestamp order.
    trained_by, trained_at
        Provenance of the ``trained`` event, if one was logged.
    mode, animal, corpus_id, metrics
        Carried from the ``trained`` event so the registry can list options without
        opening the model directory.
    """

    model_id: str
    status: RegistryAction
    trained_by: str = ""
    trained_at: str = ""
    mode: str = ""
    animal: str = ""
    corpus_id: str = ""
    metrics: dict[str, float] | None = None

    @property
    def is_promoted(self) -> bool:
        """Whether the model's last action left it promoted."""
        return self.status is RegistryAction.PROMOTED


def replay(events: list[RegistryEvent]) -> dict[str, ModelState]:
    """Compute current state by replaying the log. Pure, order-independent, idempotent.

    Events are sorted before application, so feeding the same facts in a different
    merge order - or twice - yields the same mapping. There is no default model and
    no fallback chain here: the registry presents options, the user picks one.
    """
    ordered = sorted(events, key=lambda e: (e.ts, e.model_id, str(e.action), e.user))
    state: dict[str, ModelState] = {}
    for event in ordered:
        current = state.get(event.model_id)
        if event.action is RegistryAction.TRAINED:
            state[event.model_id] = ModelState(
                model_id=event.model_id,
                status=event.action,
                trained_by=event.user,
                trained_at=event.ts,
                mode=event.mode,
                animal=event.animal,
                corpus_id=event.corpus_id,
                metrics=dict(event.metrics or {}),
            )
        elif current is None:
            # A promotion whose 'trained' line has not synced yet. Keep the fact;
            # dropping it would make state depend on sync order.
            state[event.model_id] = ModelState(model_id=event.model_id, status=event.action)
        else:
            state[event.model_id] = ModelState(
                model_id=current.model_id,
                status=event.action,
                trained_by=current.trained_by,
                trained_at=current.trained_at,
                mode=current.mode,
                animal=current.animal,
                corpus_id=current.corpus_id,
                metrics=dict(current.metrics or {}),
            )
    return state


def compact(store: GemsStore) -> int:
    """Merge every shard into ``events.jsonl`` and remove the shards. Manual only.

    Run by one person, never automatically: this is the single point in the design
    that rewrites a shared file, which cross-platform rule 12 otherwise forbids. The
    merged file is written atomically and read back before any shard is unlinked, so
    an interrupted compaction loses nothing - at worst it leaves the shards in place
    and the next run repeats it.

    Returns
    -------
    int
        Number of unique events in the compacted log.
    """
    events = read_events(store)
    target = store.registry_dir / LOG_NAME
    body = "".join(e.to_json_line() + "\n" for e in events)
    atomic_write_text(target, body)

    # Verify the merged file alone reproduces every event before dropping shards.
    merged = [RegistryEvent.from_json_line(line) for line in read_lines(target)]
    if {e.to_json_line() for e in merged} != {e.to_json_line() for e in events}:  # pragma: no cover
        msg = "compacted log does not reproduce the shard union; shards left in place"
        raise RuntimeError(msg)

    for path in iter_shard_paths(store):
        if path != target:
            path.unlink()
    return len(events)
