"""Contacts task 18 (conduction velocity) must not use (ruling 2026-10-05, item 2).

Velocity needs three good contacts per cuff, and a contact that does not see the shared ground
is open or detached (A t04 LVN3, g -0.03; B t01 1_1 bl LVN3, g 0.08 with sign agreement 0.51).
The exclusion is mechanical: in each recording, every contact the detached rule flags - first-half
|g| < ``hr_pairs.DETACHED_MAX_G`` or sign agreement < ``hr_pairs.DETACHED_MIN_AGREEMENT`` - measured
exactly as the HR pair lead measures it (``hr_pairs.half_split``: the first half's events and
non-cardiac pool). Per recording, never per animal: the same contact can be detached in one
recording and healthy in the next (B LVN3: 0.08 in t01 1_1 bl, 0.985 in t03 2_2 sr). No list is
stored - a stored list would be a second source of truth beside the rule.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

from gems_blanking_v2.physio import hr_pairs as hp
from gems_blanking_v2.types import Recording

__all__ = ["excluded_contacts"]


def excluded_contacts(rec: Recording) -> frozenset[str]:
    """Return the contacts task 18 must not use in ``rec``: the first half's detached contacts."""
    first, _second = hp._halves(rec)
    return hp.detached_contacts(*hp.noncardiac_events(first))
