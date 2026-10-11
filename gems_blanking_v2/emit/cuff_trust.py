"""RULING 2026-10-10 item 6: which distrusted cuffs the spike consumer keeps.

A routing entry distrusts a cuff for the spike consumer outright (``spike[cuff].route ==
"distrusted"``) with free-text clauses in ``why``. Ruling 2026-10-09 (i) 5 asked which of them
come only from a missing heartbeat reference; review 6 classified every such cuff into four
readings (``review6_fix/cuff_release_readings.py``), and RULING 2026-10-10 item 6 decided:

* **R1** - the only clause is "QRS hump and no count-gated train": the leak is known to be
  present, so the cuff STAYS DISTRUSTED until the leak recovery of (g) 2 places its peri-R
  spans;
* **R2** - the only clause is "check (i) fails and no peri-R window exists to rerun it
  outside", and the peri-R pass found no count-gated train;
* **R3** - the same single clause, and a train existed (J's case);
* **R4** - both clauses, and no count-gated train.

R2, R3 and R4 are RELEASED: kept for the spike consumer (the whole-region distrust span is
not applied) and flagged per minute "no heartbeat reference" over every minute of the epoch -
no peri-R span can be placed on that cuff, as for the no-beat minutes of (g) 1. Any other
clause set (another reason, or both clauses with a train) keeps its distrust: "distrust for
any other reason stays". This is applied at mask emission; the routing tables are unchanged.

Andrea 2026-10-11 item 4: the released cuff-epochs that FAILED the peri-R test (a declared,
hashed list, :func:`perir_failed_list`) are distrusted for the PRIMARY spike analysis (whole
recording, :data:`PERIR_FAILED_RULE`) and kept as released only in the SENSITIVITY scope, so
results can also be reported with them included. Untestable releases (no routed beats) and
passing ones are unchanged.

"No count-gated train" is the peri-R pass row's own verdict for that cuff (``result`` starting
"no count-gated train", or ``beat_grade == "none"``), exactly as the review-6 script read it;
a cuff with no row has a train by that rule (R3, not R2), as the script read it too.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "NO_WINDOW_CLAUSE",
    "PERIR_FAILED_RULE",
    "QRS_CLAUSE",
    "READINGS",
    "RELEASED_READINGS",
    "RELEASE_FLAG",
    "RELEASE_RULE",
    "SCOPES",
    "CuffTrust",
    "PerirFailedList",
    "cuff_trust",
    "no_count_gated_train",
    "perir_failed_list",
    "reading",
]

QRS_CLAUSE: Final = "QRS hump and no count-gated train (addendum to ruling (c) 1)"
NO_WINDOW_CLAUSE: Final = "check (i) fails and no peri-R window exists to rerun it outside"
READINGS: Final[Mapping[str, str]] = {
    "R1": "only 'QRS hump and no count-gated train'",
    "R2": "only 'check (i) fails, no peri-R window' AND the peri-R pass found no count-gated "
          "train",
    "R3": "only 'check (i) fails, no peri-R window' but a train existed (J's case)",
    "R4": "both clauses, no count-gated train",
}
RELEASED_READINGS: Final = frozenset({"R2", "R3", "R4"})
RELEASE_FLAG: Final = "no heartbeat reference"
RELEASE_RULE: Final = (
    "RULING 2026-10-10 item 6: a cuff distrusted only for having no count-gated beat train or "
    "no detectable heartbeat leak (readings R2, R3, R4) is kept for the spike consumer and "
    "flagged 'no heartbeat reference' per minute; R1 (a visible QRS hump, no count-gated train) "
    "stays distrusted until the leak recovery of (g) 2 places its peri-R spans; distrust for "
    "any other reason stays. Applied at mask emission; routing tables unchanged")


PERIR_FAILED_RULE: Final = (
    "Andrea 2026-10-11 item 4: a released cuff-epoch that failed the peri-R test (Poisson excess "
    "of peri-R spike lags, p < 0.01) is DISTRUSTED for the primary spike analysis (whole "
    "recording); its spike outputs are kept in the sensitivity scope only, labelled, never "
    "mixed into primary. No peri-R window changes")
SCOPES: Final = ("primary", "sensitivity")
"""The two spike scopes item 4 creates: the failing cuff-epochs distrusted / released."""
PERIR_LIST_SCHEMA: Final = "gems-blanking-v2 perir-failed cuffs v1"


@dataclass(frozen=True)
class PerirFailedList:
    """The declared, hashed list of released cuff-epochs that failed the peri-R test."""

    sha256: str
    source_file: str
    source_sha256: str
    keys: frozenset[tuple[str, str, str]]

    def failing(self, session: str, epoch: str) -> frozenset[str]:
        """Return the cuffs of ``session`` x ``epoch`` on the list."""
        return frozenset(c for s, c, e in self.keys if s == session and e == epoch)

    def record(self) -> dict[str, Any]:
        """Return what provenance carries: the rule, the list sha256, its source and size."""
        return {"rule": PERIR_FAILED_RULE, "sha256": self.sha256, "source_file": self.source_file,
                "source_sha256": self.source_sha256, "n_cuff_epochs": len(self.keys)}


def perir_failed_list(doc: Mapping[str, Any], *, sha256: str) -> PerirFailedList:
    """Return the declared list, checked; raise ValueError naming the defect.

    ``doc`` = {"schema", "ruling" (must name Andrea 2026-10-11 item 4), "source": {"file",
    "sha256"}, "entries": [{"session", "cuff" (L | R), "epoch"}]}; ``sha256`` is the list
    file's own hash (recorded in provenance). A (session, cuff, epoch) given twice is refused
    (invariant 27), as is an entry with another field.
    """
    if doc.get("schema") != PERIR_LIST_SCHEMA:
        msg = f"peri-R failed list: schema {doc.get('schema')!r}, expected {PERIR_LIST_SCHEMA!r}"
        raise ValueError(msg)
    if "Andrea 2026-10-11 item 4" not in str(doc.get("ruling", "")):
        msg = "peri-R failed list: its ruling must name 'Andrea 2026-10-11 item 4'"
        raise ValueError(msg)
    src = doc.get("source")
    if not isinstance(src, Mapping) or not src.get("file") or len(str(src.get("sha256", ""))) != 64:  # noqa: PLR2004
        msg = "peri-R failed list: source must be {file, sha256 (64 hex)}"
        raise ValueError(msg)
    keys: set[tuple[str, str, str]] = set()
    for e in doc.get("entries") or []:
        if set(e) != {"session", "cuff", "epoch"} or e["cuff"] not in ("L", "R"):
            msg = f"peri-R failed list: entry {e!r} must be exactly {{session, cuff (L|R), epoch}}"
            raise ValueError(msg)
        k = (str(e["session"]), str(e["cuff"]), str(e["epoch"]))
        if k in keys:
            msg = f"peri-R failed list: {k} appears twice (invariant 27)"
            raise ValueError(msg)
        keys.add(k)
    return PerirFailedList(str(sha256), str(src["file"]), str(src["sha256"]), frozenset(keys))


def _clauses(why: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in why.split(";") if c.strip())


def no_count_gated_train(row: Mapping[str, Any] | None) -> bool:
    """Whether the peri-R pass found no count-gated train for the cuff (review 6's rule)."""
    r = row or {}
    return str(r.get("result", "")).startswith("no count-gated train") or \
        r.get("beat_grade") == "none"


def reading(why: str, *, no_train: bool) -> str | None:
    """Return the reading (``R1``-``R4``) of a distrusted cuff's ``why``, or None for any other."""
    cl = _clauses(why)
    if cl == (QRS_CLAUSE,):
        return "R1"
    if cl == (NO_WINDOW_CLAUSE,):
        return "R2" if no_train else "R3"
    if len(cl) == 2 and set(cl) == {QRS_CLAUSE, NO_WINDOW_CLAUSE} and no_train:  # noqa: PLR2004
        return "R4"
    return None


@dataclass(frozen=True)
class CuffTrust:
    """One cuff's spike-consumer trust under item 6."""

    cuff: str
    distrusted_by_routing: bool
    reading: str | None
    released: bool
    why: str = ""
    perir_failed: bool = False
    """On the item-4 list (failed the peri-R test): distrusted in primary, released in
    sensitivity."""

    def record(self) -> dict[str, Any]:
        """JSON form (absent, never null)."""
        out: dict[str, Any] = {"cuff": self.cuff, "released": self.released,
                               "distrusted_by_routing": self.distrusted_by_routing}
        if self.perir_failed:
            out["perir_failed"] = PERIR_FAILED_RULE
        if self.reading is not None:
            out["reading"] = self.reading
        if self.why:
            out["why"] = self.why
        if self.released:
            out["flag"] = RELEASE_FLAG
        return out


def cuff_trust(entry: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any] | None], *,
               perir_failed: Collection[str] = (), scope: str = "primary"
               ) -> dict[str, CuffTrust]:
    """Each cuff of a routing entry's spike section, classified (``rows``: cuff -> peri-R row).

    ``perir_failed``: the cuffs of this recording x epoch on the item-4 list
    (:meth:`PerirFailedList.failing`). In ``scope`` "primary" a released cuff on it is NOT
    released (distrusted, :data:`PERIR_FAILED_RULE`); in "sensitivity" it stays released. Either
    way it is marked ``perir_failed``. A listed cuff that the item-6 rule does not release is
    refused: the list names releases only.
    """
    if scope not in SCOPES:
        msg = f"cuff trust scope {scope!r} is none of {list(SCOPES)}"
        raise ValueError(msg)
    out: dict[str, CuffTrust] = {}
    for cuff, s in sorted((entry.get("spike") or {}).items()):
        if s.get("route") != "distrusted":
            out[cuff] = CuffTrust(cuff, False, None, False)
            continue
        why = str(s.get("why") or "")
        r = reading(why, no_train=no_count_gated_train(rows.get(cuff)))
        released = r in RELEASED_READINGS
        failed = cuff in perir_failed
        if failed and not released:
            msg = f"cuff {cuff} is on the peri-R failed list but item 6 does not release it ({r})"
            raise ValueError(msg)
        out[cuff] = CuffTrust(cuff, True, r, released and not (failed and scope == "primary"),
                              why, failed)
    stray = sorted(set(perir_failed) - set(out))
    if stray:
        msg = f"cuffs {stray} are on the peri-R failed list but not in the routing entry"
        raise ValueError(msg)
    return out
