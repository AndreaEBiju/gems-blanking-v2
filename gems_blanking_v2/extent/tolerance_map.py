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
the tolerances' sha256) and the visibility scan of every harmful row, made from the same file
(its ``input.sha256`` too, review 7 finding 4). :func:`option1_tolerances` checks they belong
together and builds the table; ``mmc_provenance_flag`` (the rpeakUnits defect the mmc rows
were measured under) is carried into every record.

Cross-band scope (review 7 finding 2): each z consumer DECLARES how far option (1)'s
cross-band rule reaches - ``all_detecting_pairs`` (the ruling as written), ``own_signals_only``
or ``off`` (``tolerance.CROSS_BAND_SCOPES``) - with no default (:func:`check_cross_band_scope`).
The declaration is part of the table's record, so it is in every mask's provenance and in the
hash stage 1 is fresh under; the harmful rows a scope leaves unblanked are listed as residual
risk (:func:`scope_residual`).

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import dataclasses
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
    CROSS_BAND_SCOPE_ALL,
    CROSS_BAND_SCOPE_OFF,
    CROSS_BAND_SCOPE_OWN,
    CROSS_BAND_SCOPES,
    TOLERANCES_NOT_PRODUCTION,
    ToleranceTable,
    expected_consumers,
    extent_consumers,
)

__all__ = [
    "CORE_BLANKING",
    "CORE_PAD_RULE",
    "EXTENT_PERCENTILES",
    "NULL_PERCENTILE_SOURCE",
    "NULL_PERCENTILE_Z",
    "OPTION1_RULE",
    "RISE_MIN_Z",
    "SCOPE_OWN_HOST_COLUMNS",
    "TOLERANCE_VERSION",
    "Z_FLOOR",
    "ConsumerThreshold",
    "Option1Tolerances",
    "assess_window",
    "check_core_blanking",
    "check_cross_band_scope",
    "check_extent_percentile",
    "kind_visible",
    "option1_threshold",
    "option1_tolerances",
    "residual_risk",
    "scope_residual",
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

EXTENT_PERCENTILES: Final[tuple[str, ...]] = ("p90", "p95", "p99")
"""RULING 2026-10-10 item 2: the candidate extent thresholds for the consumers on the floor -
the clean null's 90th, 95th and 99th percentiles. Andrea picks one from the decision table; it
is a REQUIRED declaration (:func:`check_extent_percentile`), never defaulted."""

NULL_PERCENTILE_Z: Final[Mapping[str, Mapping[str, float]]] = {
    "p90": {"0-2": 1.44, "0.5-3": 1.44, "2-50": 1.44, "10-150": 1.44, "100-300": 1.44,
            "300-3000": 1.44},
    "p95": {"0-2": 2.43, "0.5-3": 2.20, "2-50": 2.52, "10-150": 1.61, "100-300": 1.81,
            "300-3000": 2.46},
    "p99": {"0-2": 7.42, "0.5-3": 6.67, "2-50": 5.88, "10-150": 3.46, "100-300": 3.11,
            "300-3000": 4.35},
}
"""The clean null's percentiles per band, z on the log envelope (invariant 5), exactly as the
extent decision table uses them (RULING 2026-10-10 item 4): p90 is the ruled floor 1.44 in every
band (:data:`Z_FLOOR`); p95 and p99 are each band's percentile over every detection signal of
the clean null that gave 1.44, rounded to 0.01 (:data:`NULL_PERCENTILE_SOURCE`)."""

NULL_PERCENTILE_SOURCE: Final[Mapping[str, Any]] = {
    "file": "measurements/2026-10-10/extent_null/null.json",
    "sha256": "2e25d4dfd451f7f4824bbfb3612acb1f27c1076a12320fa3da236b3d6e1fec33",
    "script": "measurements/2026-10-10/extent_null/null_z.py",
    "script_sha256": "d5b3c7e0c6b29018e48af947598648fc4eb9f958b795f2e3e9f120b7464b1f5f",
    "recording": "gems_j_t01_ms3_bl_230315_20260916T030321Z",
    "what": ("the J baseline of 601 s that IMPLEMENTATION.md 'MEASURED 2026-09-19' measured 1.44 "
             "on: robust z of the log envelope against the file's own median reference, every "
             "frame, 6 bands, today's chain.detect_region; per band over every detection signal "
             "(9 raw contacts, L_T, R_T, stomach_ref). Pooled over every signal: p90 1.441 "
             "(reproduces 1.44), p95 2.149, p99 5.308 (the recorded p99 4.21 is NOT reproduced: "
             "the 09-19 code differed in details not recorded)"),
}
"""Where :data:`NULL_PERCENTILE_Z` comes from; ``tests/test_tolerance_map.py`` holds the two to
each other through the archived file."""

CORE_BLANKING: Final[tuple[str, ...]] = ("always", "when_visible", "own_signals_or_visible")
"""The core-blanking declaration every z consumer makes, with no default (coordinator, after
the decision table showed the clause decides the holds by itself: slow wave held 10/10 at p99
with it, 1/10 without): ``always`` - the ruling as written, the core plus padding always
blanked; ``when_visible`` - only where the consumer's own band is above its threshold over the
core; ``own_signals_or_visible`` (Andrea, 2026-10-10) - when the core was detected on the
consumer's OWN signals (``tolerance.core_own_signals``) or its own band is above its threshold
over the core (``extent.stage1.core_pad_spans``)."""

CORE_PAD_RULE: Final = (
    "RULING 2026-10-10 item 2: the motion core and the consumer's padding (its measured "
    "settling) are blanked per each z consumer's DECLARED core blanking - 'always' (the ruling "
    "as written), 'when_visible' (only where its own band is above its threshold over the "
    "core) or 'own_signals_or_visible' (Andrea 2026-10-10: also when the core was detected on "
    "the consumer's own signals); beyond the core the extent grows only "
    "while the consumer's own band is above its threshold (the declared clean-null percentile "
    "for the consumers on the floor, the visible-kind threshold otherwise - breathing 7.69), "
    "and a cross-band extent counts a detecting pair while it is above the declared percentile "
    "of its own band")

F64 = npt.NDArray[np.float64]


def check_core_blanking(decl: object) -> dict[str, str]:
    """Return each z consumer's declared core blanking, or raise ValueError (no default)."""
    if not isinstance(decl, Mapping):
        msg = (f"core blanking is a mapping z consumer -> one of {list(CORE_BLANKING)}, declared "
               f"for every z consumer {list(z_consumers())}; got {decl!r}")
        raise ValueError(msg)
    missing = [c for c in z_consumers() if c not in decl]
    extra = sorted(str(c) for c in decl if c not in z_consumers())
    bad = {str(c): v for c, v in decl.items() if v not in CORE_BLANKING}
    if missing or extra or bad:
        msg = (f"core blanking: undeclared {missing}, not z consumers {extra}, unknown values "
               f"{bad}; each of {list(z_consumers())} declares one of {list(CORE_BLANKING)}, "
               "nothing is defaulted (RULING 2026-10-10 item 2)")
        raise ValueError(msg)
    return {c: str(decl[c]) for c in z_consumers()}


def check_extent_percentile(value: object) -> str:
    """Return the declared extent percentile, or raise ValueError (required, no default)."""
    if value not in EXTENT_PERCENTILES:
        msg = (f"the extent percentile is a REQUIRED declaration, one of "
               f"{list(EXTENT_PERCENTILES)} (RULING 2026-10-10 item 2: Andrea picks it from the "
               f"decision table); got {value!r} - nothing is defaulted")
        raise ValueError(msg)
    return str(value)


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


SCOPE_OWN_HOST_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "spikes": ("LVN", "RVN"),
    "breathing": ("LVN", "RVN"),
    "mmc": ("ANT1", "ANT2", "ANT3"),
    "slow_wave": ("ANT1", "ANT2", "ANT3"),
}
"""The tolerance sweep's host columns (tolmap ``host_columns``: RVN, LVN, ANT1-3) that count as
a consumer's OWN signals when the visibility scan is read for ``own_signals_only``: the nerve
columns for spikes and breathing (in this cohort the HR channel breathing reads is a nerve
signal or a cross-cuff nerve lead), the stomach columns for mmc and slow_wave. A host-level
reading of ``tolerance.own_detection_signals``, used only to LIST residual risk."""


def check_cross_band_scope(scopes: object) -> dict[str, str]:
    """Return the declared cross-band scope per z consumer, or raise ValueError.

    Review 7 finding 2: every z consumer (:func:`z_consumers`) declares exactly one of
    ``tolerance.CROSS_BAND_SCOPES``. Nothing is defaulted: a missing consumer, an unknown
    consumer (``hrv`` included - its extent is operational), or an unknown value is refused
    by name.
    """
    if not isinstance(scopes, Mapping):
        msg = (f"the cross-band scope is a mapping consumer -> one of {list(CROSS_BAND_SCOPES)}"
               f", declared for every z consumer {list(z_consumers())}; got {scopes!r}")
        raise ValueError(msg)
    missing = [c for c in z_consumers() if c not in scopes]
    extra = sorted(str(c) for c in scopes if c not in z_consumers())
    bad = {str(c): v for c, v in scopes.items() if v not in CROSS_BAND_SCOPES}
    if missing or extra or bad:
        msg = (f"cross-band scope: undeclared {missing}, not z consumers {extra}, unknown "
               f"values {bad}; each of {list(z_consumers())} declares one of "
               f"{list(CROSS_BAND_SCOPES)}, nothing is defaulted (review 7 finding 2)")
        raise ValueError(msg)
    return {c: str(scopes[c]) for c in z_consumers()}


def scope_residual(scan: Mapping[str, Any], consumer: str, scope: str,
                   visible_kinds: Sequence[str]) -> list[dict[str, Any]]:
    """Harmful rows the DETECTOR sees but ``scope`` leaves unblanked for ``consumer``.

    Under ``all_detecting_pairs`` none. Under ``off``, every detectable row (some pair over
    z_enter) of a kind that is not visible in the consumer's own band: cross-band would have
    blanked it, so it becomes residual risk (review 7 finding 2). Under ``own_signals_only``,
    those of them whose detectable pairs all lie off the consumer's own host columns
    (:data:`SCOPE_OWN_HOST_COLUMNS`). Recorded, never blanked.
    """
    if scope not in CROSS_BAND_SCOPES:
        msg = f"{consumer}: unknown cross-band scope {scope!r}"
        raise ValueError(msg)
    if scope == CROSS_BAND_SCOPE_ALL:
        return []
    own = SCOPE_OWN_HOST_COLUMNS.get(consumer, ())
    out = []
    for r in scan["rows"]:
        if r["consumer"] != consumer or not r["pairs_detectable"] or r["kind"] in visible_kinds:
            continue
        cols = {str(q).split("|", 1)[0] for q in r["pairs_detectable"]}
        if scope == CROSS_BAND_SCOPE_OWN and cols & set(own):
            continue
        out.append({"kind": r["kind"], "amplitude_sigma": float(r["tolerance_sigma"]),
                    "row": {"host_tag": r["host_tag"], "dur_s": float(r["dur_s"]),
                            "chan_set": r["chan_set"], "status": r["status"],
                            "censored": bool(r["censored"])},
                    "pairs_detectable": list(r["pairs_detectable"]),
                    "why": ("cross-band off: detected, invisible in the band, not blanked"
                            if scope == CROSS_BAND_SCOPE_OFF else
                            "own_signals_only: detected only off the consumer's own signals, "
                            "not blanked")})
    out.sort(key=lambda x: (x["kind"], x["amplitude_sigma"], x["row"]["host_tag"],
                            x["row"]["dur_s"], x["row"]["chan_set"]))
    return out


@dataclass(frozen=True)
class Option1Tolerances:
    """The option-(1) table: thresholds, residual risk, flags and the inputs it came from."""

    table: ToleranceTable
    thresholds: Mapping[str, ConsumerThreshold]
    residual: Mapping[str, Mapping[str, list[dict[str, Any]]]]
    flags: Mapping[str, Any]
    inputs: Mapping[str, Any]

    cross_band_scope: Mapping[str, str]
    """Each z consumer's DECLARED cross-band scope (:func:`check_cross_band_scope`): how far a
    core whose own band stays under the threshold blanks over its detected extent
    (``tolerance.detected_extent``)."""

    extent_percentile: str = "p90"
    """The DECLARED clean-null percentile (:func:`check_extent_percentile`; the builder
    requires it, this default only keeps the dataclass constructible field by field)."""

    core_blanking: Mapping[str, str] = dataclasses.field(default_factory=dict)
    """Each z consumer's DECLARED core blanking (:func:`check_core_blanking`; required by the
    builder)."""

    @property
    def cross_band_floor(self) -> dict[str, float]:
        """The cross-band floor per band: the declared percentile's value in each band."""
        return dict(NULL_PERCENTILE_Z[self.extent_percentile])

    @property
    def floor_consumers(self) -> tuple[str, ...]:
        """The consumers on the floor: no harmful kind visible in their band at or over 1.44."""
        return tuple(c for c, t in sorted(self.thresholds.items())
                     if t.setter.startswith("floor"))

    def record(self) -> dict[str, Any]:
        """Everything a mask's provenance carries about its thresholds (JSON-ready)."""
        return {"version": TOLERANCE_VERSION, "rule": OPTION1_RULE,
                "core_pad_rule": CORE_PAD_RULE,
                "core_blanking": dict(sorted(self.core_blanking.items())),
                "extent_percentile": self.extent_percentile,
                "floor_consumers": list(self.floor_consumers),
                "cross_band_floor": self.cross_band_floor,
                "null_percentile_source": dict(NULL_PERCENTILE_SOURCE),
                "cross_band_scope": dict(sorted(self.cross_band_scope.items())),
                "cross_band_scope_source": "declared per consumer in the night config "
                                           "(review 7 finding 2; no default)",
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
                       generation: str, cross_band_scope: object,
                       extent_percentile: object, core_blanking: object) -> Option1Tolerances:
    """Build the option-(1) table from the declared, hashed v2 inputs, or raise ValueError.

    Refused by name: tolerances without the v2 harm criterion, or whose ``criterion_v2``
    sha256 is not the declared one; a mapping not made from these tolerances (``input.sha256``)
    or of another chain generation, made on non-gate rows or flagged
    :data:`~gems_blanking_v2.extent.tolerance.TOLERANCES_NOT_PRODUCTION`, or missing a
    consumer; a scan of another generation, or one that does not record these tolerances as
    its input (``input.sha256``, review 7 finding 4); a cross-band scope that is not declared
    for exactly the z consumers (:func:`check_cross_band_scope`); an extent percentile that is not
    declared (:func:`check_extent_percentile`). ``mmc_provenance_flag`` is carried as a flag.

    RULING 2026-10-10 item 2: a consumer on the floor (no harmful kind visible in its band)
    takes the declared clean-null percentile of its own band (:data:`NULL_PERCENTILE_Z`) as
    its threshold; a consumer with a visible kind keeps its option-(1) threshold (breathing
    7.69). Each consumer's core and padding are blanked per its declared core blanking
    (:func:`check_core_blanking`, :data:`CORE_PAD_RULE`; ``extent.stage1.core_pad_spans``).
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
    raw_input = scan.get("input")
    sinp: Mapping[str, Any] = raw_input if isinstance(raw_input, Mapping) else {}
    if sinp.get("sha256") != tolerances_sha256:
        msg = (f"the visibility scan records input tolerances {sinp.get('sha256')!r}, not the "
               f"declared tolerances {tolerances_sha256} the mapping was made from (review 7 "
               "finding 4): its residual risk would describe other tolerances")
        raise ValueError(msg)
    scopes = check_cross_band_scope(cross_band_scope)
    pct = check_extent_percentile(extent_percentile)
    cores = check_core_blanking(core_blanking)
    thresholds: dict[str, ConsumerThreshold] = {}
    residual: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for c in z_consumers():
        entry = (mapping.get("consumers") or {}).get(c)
        if not isinstance(entry, Mapping):
            msg = f"the tolerance mapping has no entry for {c}: its threshold is unknown"
            raise ValueError(msg)
        t = option1_threshold(c, entry.get("kinds") or {}, entry.get("kinds_not_raised") or ())
        if t.setter.startswith("floor"):
            band = extent_consumers()[c].band
            t = dataclasses.replace(t, z=float(NULL_PERCENTILE_Z[pct][band]),
                                    setter=f"{t.setter}; extent at the clean null's {pct} of "
                                           f"band {band} (RULING 2026-10-10 item 2)")
        thresholds[c] = t
        residual[c] = residual_risk(scan, c)
        if scopes[c] != CROSS_BAND_SCOPE_ALL:
            key = ("cross_band_off" if scopes[c] == CROSS_BAND_SCOPE_OFF
                   else "detected_only_off_own_signals")
            residual[c][key] = scope_residual(scan, c, scopes[c], thresholds[c].visible_kinds)
    table = ToleranceTable({c: t.z for c, t in thresholds.items()},
                           f"{OPTION1_RULE}; tolerances {tolerances_sha256[:16]} (criterion v2 "
                           f"{criterion_sha256[:16]}), mapping {mapping_sha256[:16]}")
    flags: dict[str, Any] = {}
    if "mmc_provenance_flag" in tolerances:
        flags["mmc_provenance_flag"] = tolerances["mmc_provenance_flag"]
    inputs = {"tolerances_sha256": tolerances_sha256, "criterion_sha256": criterion_sha256,
              "mapping_sha256": mapping_sha256, "visibility_scan_sha256": scan_sha256,
              "generation": generation, "mapping_refused": dict(mapping.get("refused") or {})}
    return Option1Tolerances(table, thresholds, residual, flags, inputs, scopes, pct, cores)
