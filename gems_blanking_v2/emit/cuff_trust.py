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

"No count-gated train" is the peri-R pass row's own verdict for that cuff (``result`` starting
"no count-gated train", or ``beat_grade == "none"``), exactly as the review-6 script read it;
a cuff with no row has a train by that rule (R3, not R2), as the script read it too.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "NO_WINDOW_CLAUSE",
    "QRS_CLAUSE",
    "READINGS",
    "RELEASED_READINGS",
    "RELEASE_FLAG",
    "RELEASE_RULE",
    "CuffTrust",
    "cuff_trust",
    "no_count_gated_train",
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

    def record(self) -> dict[str, Any]:
        """JSON form (absent, never null)."""
        out: dict[str, Any] = {"cuff": self.cuff, "released": self.released,
                               "distrusted_by_routing": self.distrusted_by_routing}
        if self.reading is not None:
            out["reading"] = self.reading
        if self.why:
            out["why"] = self.why
        if self.released:
            out["flag"] = RELEASE_FLAG
        return out


def cuff_trust(entry: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any] | None]
               ) -> dict[str, CuffTrust]:
    """Each cuff of a routing entry's spike section, classified (``rows``: cuff -> peri-R row)."""
    out: dict[str, CuffTrust] = {}
    for cuff, s in sorted((entry.get("spike") or {}).items()):
        if s.get("route") != "distrusted":
            out[cuff] = CuffTrust(cuff, False, None, False)
            continue
        why = str(s.get("why") or "")
        r = reading(why, no_train=no_count_gated_train(rows.get(cuff)))
        out[cuff] = CuffTrust(cuff, True, r, r in RELEASED_READINGS, why)
    return out
