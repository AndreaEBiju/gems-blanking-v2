"""Scan a folder for recordings, propose animal and condition, block what is unclear.

Scanning is **read-only**: nothing is written into ``gems_root`` until a human
confirms, and then only through :func:`apply_corrections`.

Two things here are shaped by the shared drive rather than by taste:

*Hashing is lazy.* Deduplication is by content, not by path - a folder
reorganisation or a Drive conflict copy puts the same recording at two paths - but
hashing every file over a streamed mount is far too slow. Identical content
implies identical size, so files are grouped by size first and only groups larger
than one are hashed. A row with ``content_hash=None`` was unique by size, which
means no duplicate of it can exist.

*A human's answer is final.* A condition a person set is stored with
``source: "human"`` and is never overwritten by a later scan, however confidently
the rules would classify it.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from gems_blanking_v2.io.channel_map import meta_path
from gems_blanking_v2.io.conditions import Classification, Condition, Rules
from gems_blanking_v2.io.store import GemsStore, sha256_file

__all__ = [
    "RECORDING_EXTENSIONS",
    "Correction",
    "ScanResult",
    "ScanStatus",
    "apply_corrections",
    "assert_corpus_eligible",
    "read_human_condition",
    "scan",
    "sort_for_review",
]

log = logging.getLogger(__name__)

ScanStatus = Literal["matched", "ambiguous", "unknown", "duplicate", "known"]
"""``matched``/``ambiguous``/``unknown`` come from the rules; the scan adds
``duplicate`` (same content at another path) and ``known`` (already ingested)."""

RECORDING_EXTENSIONS: tuple[str, ...] = (".mat", ".h5")
"""Extensions treated as recordings. Configurable per scan."""

_SKIP_STEM_SUFFIXES: tuple[str, ...] = ("_blankmotion", "_blankmotion_stim")
"""Labelling *outputs*, not sources. Their loader refuses these for the same
reason: a ``_blankmotion.mat`` holds ``yOut``, the blanked signal, not ``y``."""


@dataclass(frozen=True, slots=True)
class ScanResult:
    """One candidate recording, and what could be established about it.

    Attributes
    ----------
    path
        Where the file is, on this machine.
    content_hash
        sha256 of the bytes, or ``None`` when the file was unique by size and
        therefore could not be a duplicate of anything. Absent means "not needed",
        never "unknown".
    animal
        Proposed animal letter, or ``None`` when the name does not carry one.
    session
        Proposed session identifier: the stripped stem.
    condition
        The proposed four-field record. Its ``epoch`` is ``"unknown"`` when no rule
        matched; it is never defaulted to a real level.
    status
        See :data:`ScanStatus`.
    matched_rule
        Id of the rule that proposed ``condition``, so a correction can flag it.
    candidates
        The tied epochs, for ``ambiguous``.
    unparsed_stim_tokens
        Stimulation-shaped tokens that resolve to no level on either axis. Non-empty
        blocks the row.
    token_conflict
        Tokens that disagreed on one axis, where the explicit-Hz form won. Recorded
        rather than dropped, and does **not** block.
    duplicate_of
        For ``duplicate``, the path already seen with this content.
    condition_source
        ``"human"`` when the condition came from a stored correction rather than
        from a rule, in which case no rule may overwrite it.
    """

    path: Path
    content_hash: str | None = None
    animal: str | None = None
    session: str | None = None
    condition: Condition = field(default_factory=Condition)
    status: ScanStatus = "unknown"
    matched_rule: str | None = None
    candidates: tuple[str, ...] = ()
    unparsed_stim_tokens: tuple[str, ...] = ()
    token_conflict: tuple[str, ...] = ()
    duplicate_of: Path | None = None
    condition_source: Literal["rule", "human"] = "rule"

    @property
    def needs_a_human(self) -> bool:
        """Whether this row blocks until someone resolves it."""
        return self.status in ("unknown", "ambiguous") or bool(self.unparsed_stim_tokens)

    @property
    def corpus_eligible(self) -> bool:
        """Whether this recording may enter a corpus.

        ``unknown`` and ``ambiguous`` may not. Blocking is the point: a recording
        nobody has classified must not quietly become a control.
        """
        return not self.needs_a_human and self.status != "duplicate"


@dataclass(frozen=True, slots=True)
class Correction:
    """A human's answer for one recording.

    ``condition`` is the whole four-field record: an epoch alone cannot express
    "10 Hz electrical at t02", and a partial answer written as a label is how the
    frequency axes would get flattened back into categories.
    """

    path: Path
    animal: str | None = None
    condition: Condition | None = None
    note: str = ""


def _candidate_files(
    root: Path, extensions: Sequence[str]
) -> list[tuple[Path, int]]:
    """Return ``(path, size)`` for every candidate under ``root``. Metadata only.

    ``os.walk`` plus ``stat`` and nothing else: no file is opened, so this stays
    usable over a streamed mount.
    """
    wanted = {e.lower() for e in extensions}
    out: list[tuple[Path, int]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() not in wanted:
                continue
            if any(path.stem.lower().endswith(s) for s in _SKIP_STEM_SUFFIXES):
                continue
            try:
                size = path.stat().st_size
            except OSError:  # pragma: no cover - vanished or unreadable mid-walk
                log.warning("skipping %s: could not stat it", path)
                continue
            out.append((path, size))
    return sorted(out, key=lambda pair: str(pair[0]))


def _hash_one(path: Path) -> str | None:
    """Hash one file, or warn and return None if it cannot be read."""
    try:
        return sha256_file(path)
    except OSError:  # pragma: no cover - unreadable or vanished mid-scan
        log.warning("could not hash %s; leaving it unhashed", path)
        return None


def _hash_lazily(
    candidates: Sequence[tuple[Path, int]],
    *,
    rehash_all: bool,
    progress: Callable[[int, int], None] | None,
) -> dict[Path, str]:
    """Hash only what has to be hashed.

    Identical content implies identical size, so only a size collision can hide a
    duplicate and everything else can be left unhashed. ``rehash_all`` covers the
    case where a caller passed a ``known`` registry: those checksums are
    re-verified rather than trusted, which means hashing every candidate.
    """
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path, size in candidates:
        by_size[size].append(path)

    if rehash_all:
        wanted = [path for path, _size in candidates]
    else:
        wanted = [p for paths in by_size.values() if len(paths) > 1 for p in paths]

    hashes: dict[Path, str] = {}
    for done, path in enumerate(wanted, start=1):
        if (digest := _hash_one(path)) is not None:
            hashes[path] = digest
        if progress is not None:
            progress(done, len(wanted))
    return hashes


def read_human_condition(
    store: GemsStore, animal: str, session: str
) -> tuple[Condition, str] | None:
    """Return ``(condition, source)`` from ``meta.json``, or None if not recorded.

    Only a ``source`` of ``"human"`` binds a later scan; a stored rule-derived
    condition is re-derived, so improving a rule improves old rows too.
    """
    path = meta_path(store, animal, session)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        log.warning("%s could not be read, ignoring its condition: %s", path, exc)
        return None
    raw = document.get("condition")
    if not isinstance(raw, dict) or not raw:
        return None
    return Condition.from_json(raw), str(document.get("condition_source") or "rule")


def scan(
    root: Path,
    rules: Rules,
    known: Mapping[str, str] | None = None,
    *,
    extensions: Sequence[str] = RECORDING_EXTENSIONS,
    store: GemsStore | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[ScanResult]:
    """Find every recording under ``root`` and propose what can be proposed.

    Parameters
    ----------
    root
        Folder to walk recursively. Defaults elsewhere to the shared drive's
        ``data/`` tree.
    rules
        The shared rule set.
    known
        ``{content_hash: recording_id}`` of recordings already ingested. Their rows
        come back as ``known``, and their checksum is re-verified rather than
        trusted.
    extensions
        Which extensions count as recordings.
    store
        When given, stored human corrections are honoured: a condition a person set
        is never overwritten by a rule.
    progress
        Called as ``(done, total)`` while hashing, which is the slow part.

    Returns
    -------
    list[ScanResult]
        Sorted for review by :func:`sort_for_review`. Nothing is written anywhere.
    """
    known = known or {}
    candidates = _candidate_files(Path(root), extensions)
    hashes = _hash_lazily(candidates, rehash_all=bool(known), progress=progress)

    results: list[ScanResult] = []
    seen_hash: dict[str, Path] = {}
    for path, _size in candidates:
        digest = hashes.get(path)
        stem = path.stem
        classification: Classification = rules.classify(stem)
        animal = rules.animal(stem)
        session = rules.core_of(stem)

        result = ScanResult(
            path=path,
            content_hash=digest,
            animal=animal,
            session=session,
            condition=classification.condition,
            status=classification.status,
            matched_rule=classification.matched_rule,
            candidates=classification.candidates,
            unparsed_stim_tokens=classification.unparsed_stim_tokens,
            token_conflict=classification.token_conflict,
        )

        if digest is not None and digest in seen_hash:
            results.append(
                replace(result, status="duplicate", duplicate_of=seen_hash[digest])
            )
            continue
        if digest is not None:
            seen_hash[digest] = path

        if digest is not None and digest in known:
            results.append(replace(result, status="known"))
            continue

        if store is not None and animal is not None:
            stored = read_human_condition(store, animal, session)
            if stored is not None and stored[1] == "human":
                result = replace(
                    result,
                    condition=stored[0],
                    status="matched",
                    matched_rule=None,
                    candidates=(),
                    unparsed_stim_tokens=(),
                    token_conflict=(),
                    condition_source="human",
                )
        results.append(result)

    return sort_for_review(results)


def sort_for_review(results: Iterable[ScanResult]) -> list[ScanResult]:
    """Sort so the rows needing attention come first.

    ``unknown`` and ``ambiguous`` lead, because in a listing of hundreds the rows
    that need a decision must not be buried among the correct ones.
    """
    order = {"unknown": 0, "ambiguous": 1, "duplicate": 2, "matched": 3, "known": 4}
    return sorted(results, key=lambda r: (order.get(r.status, 9), str(r.path)))


def assert_corpus_eligible(results: Iterable[ScanResult]) -> None:
    """Raise if any row may not enter a corpus.

    Raises
    ------
    ValueError
        Naming every blocked recording and why. An ``unknown`` recording entering a
        corpus as a control is the failure this whole task exists to prevent, so it
        is refused here rather than warned about.
    """
    blocked = [r for r in results if not r.corpus_eligible]
    if not blocked:
        return
    lines = "\n".join(f"  - {r.path.name}: {r.status}" for r in blocked)
    msg = (
        f"{len(blocked)} recording(s) cannot enter a corpus until resolved:\n{lines}\n"
        "An unclassified recording must not become a control by default."
    )
    raise ValueError(msg)


def apply_corrections(
    results: Sequence[ScanResult],
    corrections: Iterable[Correction],
    user: str,
    store: GemsStore,
    rules: Rules,
) -> list[Path]:
    """Write human corrections into each recording's ``meta.json``.

    Every correction records ``who`` and ``when``, and marks the condition
    ``source: "human"`` so no later scan overwrites it. The condition is validated
    against the closed vocabulary **at write time** - a typed correction is exactly
    where a free-text level would otherwise enter.

    The write is read-modify-write: task 03's geometry block lives in the same
    document and must survive.

    Returns
    -------
    list[pathlib.Path]
        The ``meta.json`` files written, in order.
    """
    by_path = {r.path: r for r in results}
    written: list[Path] = []
    when = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    for correction in corrections:
        result = by_path.get(correction.path)
        if result is None:
            msg = f"correction names {correction.path}, which is not in this scan"
            raise ValueError(msg)

        animal = correction.animal or result.animal
        if animal is None:
            msg = (
                f"{correction.path.name}: no animal for this recording. Set one in the "
                "correction - the directory it is written to is keyed on it."
            )
            raise ValueError(msg)
        session = result.session or correction.path.stem

        path = meta_path(store, animal, session)
        document: dict[str, Any] = {}
        if path.is_file():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                msg = f"{path} exists but could not be read; refusing to overwrite it: {exc}"
                raise ValueError(msg) from exc

        document["animal"] = animal
        document["session"] = session
        if result.token_conflict:
            # Recorded, not dropped: the precedence rule made the row parseable and
            # this is what makes the choice auditable afterwards.
            document["token_conflict"] = list(result.token_conflict)
        if correction.condition is not None:
            document["condition"] = rules.validate_condition(correction.condition).to_json()
            document["condition_source"] = "human"
        entry: dict[str, Any] = {"who": user, "when": when}
        if correction.condition is not None:
            entry["condition"] = correction.condition.to_json()
        if correction.animal is not None:
            entry["animal"] = correction.animal
        if correction.note:
            entry["note"] = correction.note
        # A correction that contradicts a rule that *did* match flags that rule for
        # review: this is how a bad rule is caught rather than propagated.
        contradicts = (
            result.matched_rule is not None
            and correction.condition is not None
            and correction.condition != result.condition
        )
        if contradicts:
            entry["contradicted_rule"] = result.matched_rule
        document.setdefault("corrections", []).append(entry)

        body = json.dumps(document, indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(body, encoding="utf-8", newline="\n")
        tmp.replace(path)
        written.append(path)

    return written
