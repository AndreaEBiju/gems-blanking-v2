"""RULING 2026-10-09 (i) 1: tolerance -> threshold mapping, option (1), cross-band.

The extent threshold of each z-thresholded consumer is the SMALLEST HARMFUL KIND VISIBLE IN
ITS OWN BAND, never below the clean null's 90th percentile (z 1.44,
:data:`~gems_blanking_v2.bands.zscore.NULL_P90_Z`); where no harmful kind is visible in the
band the threshold is 1.44 (:func:`option1_threshold`). Thresholds are applied only around
cores the motion model decides are motion (``tolerance.compute_extent``); a harmful kind
invisible in the band but caught by the detector blanks the consumer over the event's
DETECTED extent (``tolerance.detected_extent``, never routed); harmful kinds the detector
cannot see are residual risk, recorded in provenance per consumer and never blanked
(:func:`residual_risk`).

"Visible" is the tolerance mapping's own rule (``tolmap.py``), held here as the ONE
construction site (invariant 33): :func:`assess_window` decides whether an injection LIFTS a
frame (its log-envelope under the baseline reference exceeds the same frame's baseline z by
at least :data:`RISE_MIN_Z` = 1 z) and whether the z over the lifted frames is clearly above
the host's background (>= 1.44 and >= the window's baseline peak + 1 z). The mapping report
records that verdict per kind (``mapped_z_from.above_background``) and
:func:`kind_visible` reads it.

Inputs are DECLARED and hashed by the caller (night4's night config ``tolerances`` entry):
the v2 tolerances (``consumer_tolerances_v2.json``, harm criterion v2 hashed in its
``criterion_v2``), the tolerance mapping made FROM that file (its ``input.sha256`` must equal
the tolerances' sha256) and the visibility scan of every harmful row. :func:`option1_tolerances`
checks they belong together and builds the table; ``mmc_provenance_flag`` (the rpeakUnits
defect the mmc rows were measured under) is carried into every record.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.bands.zscore import NULL_P90_Z
from gems_blanking_v2.extent.tolerance import (
    TOLERANCES_NOT_PRODUCTION,
    ToleranceTable,
    expected_consumers,
)

__all__ = [
    "OPTION1_RULE",
    "RISE_MIN_Z",
    "TOLERANCE_VERSION",
    "Z_FLOOR",
    "ConsumerThreshold",
    "Option1Tolerances",
    "assess_window",
    "kind_visible",
    "option1_threshold",
    "option1_tolerances",
    "residual_risk",
    "z_consumers",
]

RISE_MIN_Z: Final = 1.0
"""tolmap's lift: a frame is lifted when the injection raises its log-envelope z (under the
BASELINE reference) by at least one robust sigma of the host's log-envelope."""
Z_FLOOR: Final = NULL_P90_Z
"""z 1.44, the clean null's 90th percentile under the log form (invariant 5): the threshold
never goes below it, and a kind whose z is below it is not "above background"."""
TOLERANCE_VERSION: Final = "v2"
"""RULING 2026-10-09 (i) 1: "Tolerances are recorded as v2" (harm criterion v2)."""
OPTION1_RULE: Final = (
    "RULING 2026-10-09 (i) 1, option (1), cross-band: threshold = max(1.44, smallest z of a "
    "harmful kind visible in the consumer's own band), 1.44 where none is visible; applied "
    "only around model-decided motion cores (blanked until the band returns below it, plus the "
    "consumer's measured settling); a harmful kind invisible in the band but caught by the "
    "detector blanks the consumer over the event's detected extent, never routed; harmful "
    "kinds the detector cannot see are residual risk, recorded, not blanked")

F64 = npt.NDArray[np.float64]


def assess_window(z_inj: npt.ArrayLike, z_base: npt.ArrayLike,
                  z_inj_baseref: npt.ArrayLike) -> dict[str, Any]:
    """One (placement, column) window: which frames the injection LIFTED, and the z there.

    ``z_inj`` is the injected signal's z under its OWN whole-span reference, ``z_base`` the
    clean host's z, ``z_inj_baseref`` the injected log-envelope under the BASELINE reference,
    all over the same frames. A frame is lifted when ``z_inj_baseref - z_base >=``
    :data:`RISE_MIN_Z`; ``z_peak`` is the max of ``z_inj`` over the lifted frames ONLY (never
    the window's peak, which can be host background), and ``above_background`` says whether
    it is >= :data:`Z_FLOOR` and >= the window's baseline peak + :data:`RISE_MIN_Z`. This is
    tolmap's rule (review 2026-10-09), moved here as its one construction site.
    """
    zi = np.asarray(z_inj, dtype=np.float64)
    zb = np.asarray(z_base, dtype=np.float64)
    zr = np.asarray(z_inj_baseref, dtype=np.float64)
    if not zi.shape == zb.shape == zr.shape:
        msg = f"window shapes differ: {zi.shape}, {zb.shape}, {zr.shape}"
        raise ValueError(msg)
    ok = np.isfinite(zi) & np.isfinite(zb) & np.isfinite(zr)
    if not ok.any():
        return {"assessable": False}
    lift = np.where(ok, zr - zb, -np.inf)
    lifted = ok & (lift >= RISE_MIN_Z)
    peak_base = float(np.max(zb[ok]))
    out: dict[str, Any] = {"assessable": True, "z_peak_baseline": peak_base,
                           "max_lift": float(np.max(lift[ok])),
                           "n_lifted_frames": int(lifted.sum()), "raised": bool(lifted.any())}
    if lifted.any():
        z = float(np.max(zi[lifted]))
        out["z_peak"] = z
        out["above_background"] = bool(z >= Z_FLOOR and z - peak_base >= RISE_MIN_Z)
    return out


def kind_visible(kind: Mapping[str, Any]) -> bool:
    """Whether a mapped kind is VISIBLE in the consumer's band (tolmap's above_background).

    ``kind`` is one ``consumers[c]["kinds"][k]`` entry of the mapping report: its setting
    row's ``mapped_z_from.above_background`` is :func:`assess_window`'s verdict. Absent is
    not visible (never assumed).
    """
    row = kind.get("row") or {}
    frm = row.get("mapped_z_from") or {}
    return frm.get("above_background") is True


@dataclass(frozen=True)
class ConsumerThreshold:
    """One consumer's option-(1) threshold and what set it."""

    consumer: str
    z: float
    setter: str
    visible_kinds: tuple[str, ...]
    kinds: tuple[dict[str, Any], ...]
    kinds_not_raised: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        """JSON form for provenance (absent, never null)."""
        return {"consumer": self.consumer, "z": self.z, "setter": self.setter,
                "visible_kinds": list(self.visible_kinds), "kinds": list(self.kinds),
                "kinds_not_raised": list(self.kinds_not_raised)}


def option1_threshold(consumer: str, kinds: Mapping[str, Mapping[str, Any]],
                      kinds_not_raised: Sequence[str] = ()) -> ConsumerThreshold:
    """Return ``consumer``'s threshold: max(1.44, min z over visible kinds), else 1.44.

    ``kinds`` maps a harmful kind to its mapping-report entry (``z`` and its setting ``row``,
    whose ``mapped_z_from.above_background`` decides visibility, :func:`kind_visible`). A
    visible kind with a non-finite z is refused (ValueError): visibility means a z was read.
    """
    allk: list[dict[str, Any]] = []
    vis: list[tuple[float, str]] = []
    for name in sorted(kinds):
        k = kinds[name]
        z = float(k["z"])
        visible = kind_visible(k)
        if visible and not math.isfinite(z):
            msg = f"{consumer}/{name}: visible in the band but its mapped z is {z}"
            raise ValueError(msg)
        row = k.get("row") or {}
        allk.append({"kind": name, "z": z, "visible": visible,
                     "row": {f: row[f] for f in ("host_tag", "dur_s", "chan_set",
                                                 "sensitive_end_sigma") if f in row}})
        if visible:
            vis.append((z, name))
    if vis:
        zmin, name = min(vis)
        z_thr = max(Z_FLOOR, zmin)
        setter = name if zmin >= Z_FLOOR else f"floor {Z_FLOOR:g} (smallest visible: {name})"
    else:
        z_thr, setter = Z_FLOOR, f"floor {Z_FLOOR:g} (no harmful kind visible in the band)"
    return ConsumerThreshold(consumer, float(z_thr), setter, tuple(n for _z, n in sorted(vis)),
                             tuple(allk), tuple(sorted(kinds_not_raised)))


def residual_risk(scan: Mapping[str, Any], consumer: str) -> dict[str, list[dict[str, Any]]]:
    """Harmful rows the detector cannot see, for ``consumer`` (never blanked, recorded).

    From the visibility scan (every still-harmful gate row re-injected at its v2 tolerance,
    every column x band checked): a row with no DETECTABLE pair (lifted and over z_enter) never
    makes a core, so nothing blanks it. ``no_band_no_detector`` lists those no band lifts
    either (the ruling's residual risk, ~0.088-0.18 sigma); ``lifted_not_detected`` those some
    band lifts but the detector still misses - equally unblanked, listed so nothing is hidden.
    Each item: kind, amplitude (sigma, the v2 tolerance) and the row (host, duration, channel
    set), plus the lifted pairs.
    """
    out: dict[str, list[dict[str, Any]]] = {"no_band_no_detector": [],
                                            "lifted_not_detected": []}
    for r in scan["rows"]:
        if r["consumer"] != consumer or r["pairs_detectable"]:
            continue
        item = {"kind": r["kind"], "amplitude_sigma": float(r["tolerance_sigma"]),
                "row": {"host_tag": r["host_tag"], "dur_s": float(r["dur_s"]),
                        "chan_set": r["chan_set"], "status": r["status"],
                        "censored": bool(r["censored"])},
                "pairs_lifted": list(r["pairs_lifted"])}
        out["no_band_no_detector" if not r["pairs_lifted"] else "lifted_not_detected"].append(item)
    for v in out.values():
        v.sort(key=lambda x: (x["kind"], x["amplitude_sigma"], x["row"]["host_tag"],
                              x["row"]["dur_s"], x["row"]["chan_set"]))
    return out


def z_consumers() -> tuple[str, ...]:
    """Return the consumers that take a z threshold here: every expected one but hrv."""
    return tuple(c for c in expected_consumers() if c != "hrv")


@dataclass(frozen=True)
class Option1Tolerances:
    """The option-(1) table: thresholds, residual risk, flags and the inputs it came from."""

    table: ToleranceTable
    thresholds: Mapping[str, ConsumerThreshold]
    residual: Mapping[str, Mapping[str, list[dict[str, Any]]]]
    flags: Mapping[str, Any]
    inputs: Mapping[str, Any]

    cross_band: bool = True
    """Option (1): a core whose own band stays under the threshold blanks over its detected
    extent (``tolerance.detected_extent``)."""

    def record(self) -> dict[str, Any]:
        """Everything a mask's provenance carries about its thresholds (JSON-ready)."""
        return {"version": TOLERANCE_VERSION, "rule": OPTION1_RULE, "cross_band": True,
                "z_tol": dict(self.table.z_tol), "floor_z": Z_FLOOR, "rise_min_z": RISE_MIN_Z,
                "thresholds": {c: t.record() for c, t in sorted(self.thresholds.items())},
                "residual_risk": {c: {k: list(v) for k, v in r.items()}
                                  for c, r in sorted(self.residual.items())},
                "flags": dict(self.flags), "inputs": dict(self.inputs)}

    def sha256(self) -> str:
        """Canonical hash of :meth:`record` (stage 1 is fresh only under the same table)."""
        text = json.dumps(self.record(), sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode("ascii")).hexdigest()


def option1_tolerances(tolerances: Mapping[str, Any], mapping: Mapping[str, Any],
                       scan: Mapping[str, Any], *, tolerances_sha256: str,
                       criterion_sha256: str, mapping_sha256: str, scan_sha256: str,
                       generation: str) -> Option1Tolerances:
    """Build the option-(1) table from the declared, hashed v2 inputs, or raise ValueError.

    Refused by name: tolerances without the v2 harm criterion, or whose ``criterion_v2``
    sha256 is not the declared one; a mapping not made from these tolerances (``input.sha256``)
    or of another chain generation, made on non-gate rows or flagged
    :data:`~gems_blanking_v2.extent.tolerance.TOLERANCES_NOT_PRODUCTION`, or missing a
    consumer; a scan of another generation. ``mmc_provenance_flag`` is carried as a flag.
    """
    crit = tolerances.get("criterion_v2")
    if not isinstance(crit, Mapping) or crit.get("sha256") != criterion_sha256:
        got = crit.get("sha256") if isinstance(crit, Mapping) else None
        msg = (f"the tolerances' harm criterion is {got!r}, not the declared v2 criterion "
               f"{criterion_sha256} (RULING 2026-10-09 (i) 1: tolerances are recorded as v2)")
        raise ValueError(msg)
    inp = mapping.get("input") or {}
    if inp.get("sha256") != tolerances_sha256:
        msg = (f"the tolerance mapping was made from {inp.get('sha256')!r}, not the declared "
               f"tolerances {tolerances_sha256}")
        raise ValueError(msg)
    text = json.dumps(dict(mapping), sort_keys=True, ensure_ascii=True, default=str)
    if (mapping.get("rows_rule") != "gate" or mapping.get("flags")
            or TOLERANCES_NOT_PRODUCTION in text):
        msg = (f"the tolerance mapping used rows {mapping.get('rows_rule')!r} with flags "
               f"{mapping.get('flags')!r}: only gate rows, unflagged, are production")
        raise ValueError(msg)
    for what, doc in (("mapping", mapping), ("visibility scan", scan)):
        if doc.get("generation_sha256_16") != generation:
            msg = (f"the {what} is of chain generation {doc.get('generation_sha256_16')!r}, "
                   f"not {generation}")
            raise ValueError(msg)
    thresholds: dict[str, ConsumerThreshold] = {}
    residual: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for c in z_consumers():
        entry = (mapping.get("consumers") or {}).get(c)
        if not isinstance(entry, Mapping):
            msg = f"the tolerance mapping has no entry for {c}: its threshold is unknown"
            raise ValueError(msg)
        thresholds[c] = option1_threshold(c, entry.get("kinds") or {},
                                          entry.get("kinds_not_raised") or ())
        residual[c] = residual_risk(scan, c)
    table = ToleranceTable({c: t.z for c, t in thresholds.items()},
                           f"{OPTION1_RULE}; tolerances {tolerances_sha256[:16]} (criterion v2 "
                           f"{criterion_sha256[:16]}), mapping {mapping_sha256[:16]}")
    flags: dict[str, Any] = {}
    if "mmc_provenance_flag" in tolerances:
        flags["mmc_provenance_flag"] = tolerances["mmc_provenance_flag"]
    inputs = {"tolerances_sha256": tolerances_sha256, "criterion_sha256": criterion_sha256,
              "mapping_sha256": mapping_sha256, "visibility_scan_sha256": scan_sha256,
              "generation": generation, "mapping_refused": dict(mapping.get("refused") or {})}
    return Option1Tolerances(table, thresholds, residual, flags, inputs)
