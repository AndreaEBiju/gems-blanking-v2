"""The frozen routing table and the damage classes computed under it.

Andrea, 2026-09-30: the damage rule and its routing are frozen; a change to either means
recomputing every pooled mark. The routing says, per recording and per consumer, what
each consumer reads, and every set of damage classes names the routing it was computed
under.

Per recording (keys are recording ids)::

    spike:       {cuff: {route: multi | scalar | uncorrected | distrusted,
                         why?: str,                     # distrusted only
                         veto?: {w_s, theta_sigma},     # the cuff has a core
                         peri_r_ms?: [a, b], peri_r_beats?: hrv | mask,
                         peri_r_narrow_ms?: [[a, b], ...]}}   # ruling (d) 3
    hr:          {channel, detector, n_beats} | {none: str}
    stomach_ref: {notch_hz: [...], contacts: [...]} | {}   # {} = the derivation as-is

**Schema 2 (ruling 2026-09-30 (d) 2):** the table is per-recording entries, each hashed
on its own (:func:`entry_hash`); the table hash is the hash of the entry hashes. A new
round APPENDS entries (:func:`append_entries`) and every existing entry must stay
byte-identical - :func:`compare_tables` is the check each score records. Schema 1 (one
hash over the whole document, :func:`routing_hash`) is the table frozen before (d); it
is kept readable, never rewritten.

**Damage conventions (ruling (d) 1):** ``run`` - only consumers that run count, the
gate; ``excluded_is_target`` - an excluded input makes the mark target (the corrected
addendum 3), reported beside it every round. :func:`excluded_inputs` lists the inputs a
recording's routing excludes.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from gems_blanking_v2.io.store import GemsStore, atomic_write_text

__all__ = [
    "CONVENTIONS",
    "GATE_CONVENTION",
    "SPIKE_ROUTES",
    "append_entries",
    "append_for_round",
    "build_table",
    "canonical_json",
    "compare_tables",
    "damage_path",
    "entry_hash",
    "excluded_inputs",
    "read_damage",
    "read_routing",
    "routing_check_path",
    "routing_hash",
    "routing_path",
    "table_hash",
    "validate_entry",
    "validate_routing",
    "write_damage",
    "write_routing",
]

SPIKE_ROUTES: Final[tuple[str, ...]] = ("multi", "scalar", "uncorrected", "distrusted")
CONVENTIONS: Final[tuple[str, ...]] = ("run", "excluded_is_target")
GATE_CONVENTION: Final = "run"
"""Ruling (d) 1: the gate counts only consumers that run."""
_TOP: Final = ("spike", "hr", "stomach_ref")


def canonical_json(obj: object) -> str:
    """Sorted keys, ASCII-escaped, fixed separators: one text per value, on any machine."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False)


def _sha(obj: object) -> str:
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()


def validate_entry(rid: str, entry: Mapping[str, Any]) -> None:
    """Raise, naming the recording and field, on an entry outside the schema."""
    missing = [k for k in _TOP if k not in entry]
    if missing:
        msg = f"{rid}: routing entry lacks {', '.join(missing)}"
        raise ValueError(msg)
    for cuff, s in entry["spike"].items():
        if s.get("route") not in SPIKE_ROUTES:
            msg = f"{rid} {cuff}: spike route {s.get('route')!r} is not one of {SPIKE_ROUTES}"
            raise ValueError(msg)
        if s["route"] == "distrusted" and not s.get("why"):
            msg = f"{rid} {cuff}: a distrusted cuff must say why"
            raise ValueError(msg)
        has_extent = "peri_r_ms" in s or "peri_r_narrow_ms" in s
        if has_extent != ("peri_r_beats" in s):
            msg = f"{rid} {cuff}: peri_r_beats goes with a peri-R extent, and only with one"
            raise ValueError(msg)
    hr = entry["hr"]
    if ("none" in hr) == ("channel" in hr):
        msg = f"{rid}: hr names either a stored train or why there is none"
        raise ValueError(msg)


def _entries(table: Mapping[str, Any]) -> Mapping[str, Any]:
    key = "entries" if table.get("schema") == 2 else "recordings"  # noqa: PLR2004
    recs = table.get(key)
    if not isinstance(recs, Mapping) or not recs:
        msg = "routing table has no recordings"
        raise ValueError(msg)
    return recs


def validate_routing(table: Mapping[str, Any]) -> None:
    """Validate every entry; for schema 2, also that recorded entry hashes match."""
    recs = _entries(table)
    for rid, entry in recs.items():
        validate_entry(rid, entry)
    recorded = table.get("entry_hashes")
    if recorded is not None:
        bad = sorted(r for r in recs if recorded.get(r) != entry_hash(recs[r]))
        extra = sorted(set(recorded) - set(recs))
        if bad or extra:
            msg = f"entry hashes do not match their entries: {bad + extra}"
            raise ValueError(msg)


def routing_hash(table: Mapping[str, Any]) -> str:
    """Return schema 1's hash: of the whole table's canonical JSON, less any ``hash``."""
    return _sha({k: v for k, v in table.items() if k != "hash"})


def entry_hash(entry: Mapping[str, Any]) -> str:
    """SHA-256 of one recording's entry, canonical JSON."""
    return _sha(entry)


def table_hash(table: Mapping[str, Any]) -> str:
    """Return the hash that names a table: schema 2 - of its entry hashes; 1 - of it all."""
    if table.get("schema") != 2:  # noqa: PLR2004
        return routing_hash(table)
    hashes = {r: entry_hash(e) for r, e in _entries(table).items()}
    return _sha({"schema": 2, "entry_hashes": hashes})


def build_table(entries: Mapping[str, Mapping[str, Any]], frozen_under: str) -> dict[str, Any]:
    """Return a schema-2 table - entries, their hashes and the table hash - validated."""
    table: dict[str, Any] = {"schema": 2, "frozen_under": frozen_under,
                             "entries": {r: dict(e) for r, e in entries.items()}}
    table["entry_hashes"] = {r: entry_hash(e) for r, e in table["entries"].items()}
    validate_routing(table)
    table["hash"] = table_hash(table)
    return table


def compare_tables(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    """Return the check a score records: old hash, new hash, unchanged entries.

    ``unchanged_entries`` is true when every old entry is in the new table and hashes
    the same - byte-identical canonical JSON.
    """
    o, n = _entries(old), _entries(new)
    changed = sorted(r for r in o if r in n and entry_hash(o[r]) != entry_hash(n[r]))
    removed = sorted(set(o) - set(n))
    return {"old_hash": table_hash(old), "new_hash": table_hash(new),
            "unchanged_entries": not changed and not removed,
            "changed": changed, "removed": removed, "added": sorted(set(n) - set(o))}


def append_entries(old: Mapping[str, Any], new_entries: Mapping[str, Mapping[str, Any]],
                   frozen_under: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Append new recordings' entries to a schema-2 table; return ``(table, check)``.

    Raises if any recording is already in the table - changing an entry is a routing
    change (recompute every pooled mark), never an append.
    """
    if old.get("schema") != 2:  # noqa: PLR2004
        msg = "append only to a schema-2 table"
        raise ValueError(msg)
    clash = sorted(set(new_entries) & set(_entries(old)))
    if clash:
        msg = f"already routed, so not an append: {clash}"
        raise ValueError(msg)
    table = build_table({**_entries(old), **new_entries}, frozen_under)
    check = compare_tables(old, table)
    assert check["unchanged_entries"], check  # construction guarantees it
    return table, check


def excluded_inputs(entry: Mapping[str, Any]) -> list[str]:
    """Return the inputs a recording's routing excludes (distrusted cuffs, no HR train)."""
    out = [f"spike {cuff}: {s['why']}" for cuff, s in sorted(entry["spike"].items())
           if s["route"] == "distrusted"]
    if "none" in entry["hr"]:
        out.append(f"hr: {entry['hr']['none']}")
    return out


def _labels(store: GemsStore) -> Path:
    return store.audit_score_path("x").parent.parent


def routing_path(store: GemsStore, hash_: str) -> Path:
    """Where a frozen routing table lives: beside the audit scores, keyed by its hash."""
    return _labels(store) / "blind_audit_routing" / f"routing_{hash_[:16]}.json"


def damage_path(store: GemsStore, plan_id: str, hash_: str,
                convention: str = GATE_CONVENTION) -> Path:
    """Where one round's damage classes under one routing and one convention live."""
    if convention not in CONVENTIONS:
        msg = f"convention {convention!r} is not one of {CONVENTIONS}"
        raise ValueError(msg)
    tail = "" if convention == GATE_CONVENTION else f"_{convention}"
    return _labels(store) / "blind_audit_damage" / f"{plan_id}_damage_{hash_[:16]}{tail}.json"


def write_routing(store: GemsStore, table: Mapping[str, Any]) -> tuple[str, Path]:
    """Validate, hash and write the table; return ``(hash, path)``.

    Never overwrites a different table under the same hash prefix.
    """
    validate_routing(table)
    h = table_hash(table)
    path = routing_path(store, h)
    doc = {**{k: v for k, v in table.items() if k != "hash"}, "hash": h}
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("hash") != h:
            msg = f"{path.name} already holds a different routing table ({old.get('hash')})"
            raise FileExistsError(msg)
        return h, path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    return h, path


def read_routing(store: GemsStore, hash_: str) -> dict[str, Any]:
    """Return the stored table named by ``hash_``; raises if its content does not hash to it."""
    doc: dict[str, Any] = json.loads(routing_path(store, hash_).read_text(encoding="utf-8"))
    validate_routing(doc)
    if table_hash(doc) != hash_:
        msg = f"stored table hashes to {table_hash(doc)}, not {hash_}"
        raise ValueError(msg)
    return doc


def write_damage(store: GemsStore, plan_id: str, hash_: str,
                 classes: Mapping[str, str], reasons: Mapping[str, list[str]],
                 convention: str = GATE_CONVENTION) -> Path:
    """Write one round's damage classes, naming their routing and convention."""
    bad = {k: v for k, v in classes.items() if v not in ("target", "below")}
    if bad:
        msg = f"invalid damage classes: {bad}"
        raise ValueError(msg)
    path = damage_path(store, plan_id, hash_, convention)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"plan_id": plan_id, "routing_hash": hash_, "convention": convention,
           "classes": dict(classes), "reasons": {k: list(v) for k, v in reasons.items()}}
    atomic_write_text(path, json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    return path


def read_damage(store: GemsStore, plan_id: str, hash_: str,
                convention: str = GATE_CONVENTION) -> dict[str, str]:
    """Return ``{mark id: class}``; raises if the file names another routing or convention."""
    path = damage_path(store, plan_id, hash_, convention)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("routing_hash") != hash_:
        msg = f"{path.name} was computed under routing {doc.get('routing_hash')}, not {hash_}"
        raise ValueError(msg)
    if doc.get("convention") != convention:
        msg = f"{path.name} holds convention {doc.get('convention')!r}, not {convention!r}"
        raise ValueError(msg)
    return dict(doc["classes"])


def routing_check_path(store: GemsStore, plan_id: str) -> Path:
    """Where a round records the routing it was scored on (ruling (d) 2)."""
    return _labels(store) / "blind_audit_routing" / f"{plan_id}_routing_check.json"


def append_for_round(store: GemsStore, plan_id: str, old_hash: str,
                     new_entries: Mapping[str, Mapping[str, Any]],
                     frozen_under: str) -> tuple[str, dict[str, Any]]:
    """Append a round's new recordings to the stored table ``old_hash``; return ``(hash, check)``.

    Stores the appended table and the round's check - old hash, new hash, unchanged
    entries - at :func:`routing_check_path`. Raises if the check fails; it cannot, by
    construction, but a round must never be scored on a table that changed an entry.
    """
    old = read_routing(store, old_hash)
    table, check = append_entries(old, new_entries, frozen_under)
    if not check["unchanged_entries"]:  # pragma: no cover - append_entries guarantees it
        msg = f"appending for {plan_id} changed entries: {check['changed'] + check['removed']}"
        raise ValueError(msg)
    h, _path = write_routing(store, table)
    record = {"plan_id": plan_id, **check}
    path = routing_check_path(store, plan_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    return h, record
