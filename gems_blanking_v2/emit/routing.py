"""The frozen routing table and the damage classes computed under it (addendum to ruling (c) 2).

Andrea, 2026-09-30: the damage rule and its routing are frozen; a change to either means
recomputing every pooled mark. So the routing is a single document - per recording, per
consumer, what each consumer reads - identified by the SHA-256 of its canonical JSON,
and every set of damage classes names the routing hash it was computed under. Round 6
is scored on the frozen table.

Per recording (keys are recording ids)::

    spike:       {cuff: {route: multi | scalar | uncorrected | distrusted,
                         why?: str,                     # distrusted only
                         veto?: {w_s, theta_sigma},     # the cuff has a core
                         peri_r_ms?: [a, b], peri_r_beats?: hrv | mask}}
    hr:          {channel, detector, n_beats} | {none: str}
    stomach_ref: {notch_hz: [...], contacts: [...]} | {}   # {} = the derivation as-is

An input the routing excludes (a distrusted cuff, no HR train) is unassessable, so the
damage rule calls the mark target (addendum 3): :func:`excluded_inputs` lists them.

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
    "SPIKE_ROUTES",
    "canonical_json",
    "damage_path",
    "excluded_inputs",
    "read_damage",
    "routing_hash",
    "routing_path",
    "validate_routing",
    "write_damage",
    "write_routing",
]

SPIKE_ROUTES: Final[tuple[str, ...]] = ("multi", "scalar", "uncorrected", "distrusted")
_TOP: Final = ("spike", "hr", "stomach_ref")


def canonical_json(obj: object) -> str:
    """Sorted keys, ASCII-escaped, fixed separators: one text per value, on any machine."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False)


def validate_routing(table: Mapping[str, Any]) -> None:
    """Raise, naming the recording and field, on any entry outside the schema."""
    recs = table.get("recordings")
    if not isinstance(recs, Mapping) or not recs:
        msg = "routing table has no recordings"
        raise ValueError(msg)
    for rid, entry in recs.items():
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
            if ("peri_r_ms" in s) != ("peri_r_beats" in s):
                msg = f"{rid} {cuff}: peri_r_ms and peri_r_beats come together"
                raise ValueError(msg)
        hr = entry["hr"]
        if ("none" in hr) == ("channel" in hr):
            msg = f"{rid}: hr names either a stored train or why there is none"
            raise ValueError(msg)


def routing_hash(table: Mapping[str, Any]) -> str:
    """SHA-256 of the table's canonical JSON, with any recorded ``hash`` field left out."""
    body = {k: v for k, v in table.items() if k != "hash"}
    return hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest()


def excluded_inputs(entry: Mapping[str, Any]) -> list[str]:
    """Return the inputs a recording's routing excludes - each makes its marks target."""
    out = [f"spike {cuff}: {s['why']}" for cuff, s in sorted(entry["spike"].items())
           if s["route"] == "distrusted"]
    if "none" in entry["hr"]:
        out.append(f"hr: {entry['hr']['none']}")
    return out


def routing_path(store: GemsStore, table_hash: str) -> Path:
    """Where a frozen routing table lives: beside the audit scores, keyed by its hash."""
    return store.audit_score_path("x").parent.parent / "blind_audit_routing" / (
        f"routing_{table_hash[:16]}.json")


def damage_path(store: GemsStore, plan_id: str, table_hash: str) -> Path:
    """Where one round's damage classes under one routing table live."""
    return store.audit_score_path("x").parent.parent / "blind_audit_damage" / (
        f"{plan_id}_damage_{table_hash[:16]}.json")


def write_routing(store: GemsStore, table: Mapping[str, Any]) -> tuple[str, Path]:
    """Validate, hash and write the table; return ``(hash, path)``.

    Never overwrites a different table under the same hash prefix.
    """
    validate_routing(table)
    h = routing_hash(table)
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


def write_damage(store: GemsStore, plan_id: str, table_hash: str,
                 classes: Mapping[str, str], reasons: Mapping[str, list[str]]) -> Path:
    """Write one round's damage classes, naming the routing they were computed under."""
    bad = {k: v for k, v in classes.items() if v not in ("target", "below")}
    if bad:
        msg = f"invalid damage classes: {bad}"
        raise ValueError(msg)
    path = damage_path(store, plan_id, table_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"plan_id": plan_id, "routing_hash": table_hash, "classes": dict(classes),
           "reasons": {k: list(v) for k, v in reasons.items()}}
    atomic_write_text(path, json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
    return path


def read_damage(store: GemsStore, plan_id: str, table_hash: str) -> dict[str, str]:
    """Return ``{mark id: class}``; raises if the file names another routing table."""
    path = damage_path(store, plan_id, table_hash)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("routing_hash") != table_hash:
        msg = f"{path.name} was computed under routing {doc.get('routing_hash')}, not {table_hash}"
        raise ValueError(msg)
    return dict(doc["classes"])
