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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
    "BREATHING_FLOOR_LIMITATION",
    "CORE_BLANKING",
    "CORE_PAD_RULE",
    "EXTENT_FLOORS_PER_BAND",
    "EXTENT_GATES",
    "EXTENT_PERCENTILES",
    "HRV_GATES",
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
    "check_extent_floors",
    "check_extent_percentile",
    "check_extent_rule",
    "kind_visible",
    "not_emitted",
    "option1_threshold",
    "option1_tolerances",
    "own_signal_floor_record",
    "provisional_consumers",
    "residual_risk",
    "scope_residual",
    "tested_candidate_mismatches",
    "verify_floor_archive",
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
consumer's OWN signals (``tolerance.own_signals_by_signal``) or its own band is above its threshold
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


EXTENT_GATES: Final[Mapping[str, str]] = {
    "i": "rule (i) only: a confirmed core blanks the consumer x signal only when it was detected "
         "on that signal's own signals (tolerance.own_signals_by_signal)",
    "ii": "rule (ii) only: only when the consumer's own band is above its threshold in ANY finite "
          "frame of the core window (stage1.core_frames); for hrv, when the beat train changed",
    "i_or_ii": "rule (i) OR rule (ii)",
}
"""The gate of the declared extent rule (the measurement candidates' semantics, candidate set
v2). A core that passes its consumer x signal's gate blanks: core + the consumer's padding, the
own-band extent (routed, as before), and the detected extent over its OWN detecting pairs (each
pair while above its band's declared clean-null percentile); for hrv, core + padding and the
operational beat-train extent. A core that fails blanks nothing for it."""
HRV_GATES: Final[tuple[str, ...]] = ("i", "i_or_ii")
"""hrv has no z band: rule (ii) for hrv is the beat-train change, so 'ii' alone is not offered."""
EXTENT_RULE_SOURCE_KEYS: Final[tuple[str, ...]] = ("ruling", "decision_files", "candidate_set")
"""``candidate_set`` = {file, sha256}: the tested candidates the declaration must equal
(:func:`tested_candidate_mismatches`; the night driver loads, hashes and compares it)."""


def _check_provisional(src: Mapping[str, Any], out: Mapping[str, Mapping[str, Any]]) -> None:
    """``provisional: true`` entries need ONE ``source.provisional_caveat`` {text, files}.

    The caveat text is written once (here, in the declaration's source) and applied to every
    provisional consumer by :func:`provisional_consumers`; ``files`` = [{file, sha256}, ...] are
    the published decision files it comes from. A caveat with no provisional consumer is refused
    too (a dangling caveat reads as if it applied to something).
    """
    prov = [c for c, e in out.items() if "provisional" in e]
    bad = [c for c in prov if out[c]["provisional"] is not True or out[c].get("declared") is False]
    if bad:
        msg = (f"extent_rule.{bad}: provisional is true on a declared consumer (the caveat text "
               "is source.provisional_caveat, written once)")
        raise ValueError(msg)
    cav = src.get("provisional_caveat")
    if not prov:
        if cav is not None:
            msg = "extent_rule.source.provisional_caveat applies to no provisional consumer"
            raise ValueError(msg)
        return
    files = cav.get("files") if isinstance(cav, Mapping) else None
    if (not isinstance(cav, Mapping) or not str(cav.get("text", "")).strip()
            or not isinstance(files, list) or not files
            or not all(isinstance(f, Mapping) and f.get("file")
                       and re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", "")))
                       for f in files)):
        msg = (f"extent_rule: {prov} are provisional, so source.provisional_caveat must be "
               f"{{text, files: [{{file, sha256 (64 hex)}}, ...]}}; got {cav!r}")
        raise ValueError(msg)


def provisional_consumers(rule: Mapping[str, Any]) -> dict[str, str]:
    """Consumer -> its provisional caveat, from the ONE declared text (+ an entry's own caveat).

    The only place a provisional caveat is assembled (provenance records it per consumer).
    """
    cav = rule["source"].get("provisional_caveat") or {}
    src = ", ".join(f"{Path(str(f['file'])).name} (sha256 {str(f['sha256'])[:16]})"
                    for f in cav.get("files") or [])
    return {c: "; ".join([str(cav["text"]), *([str(e["caveat"])] if e.get("caveat") else []),
                          f"source {src}"])
            for c, e in sorted(rule["consumers"].items()) if e.get("provisional") is True}


def check_extent_rule(decl: object) -> dict[str, Any]:
    """Return the declared per-consumer extent rule, or raise ValueError naming the gap.

    ``{"source": {"ruling": ..., "decision_files": [{file, sha256}, ...], "candidate_set":
    {file, sha256}}, "consumers":
    {consumer: {"gate": i | ii | i_or_ii, ...} | {"declared": false, "reason": ...}}}`` for EVERY
    expected consumer (hrv included; hrv's gate is one of :data:`HRV_GATES`). A consumer declared
    ``false`` gets no masks at all (stage 1b marks it not computed, Night 6 lists leave it out);
    a consumer with no entry is refused - nothing is defaulted. Other fields of an entry
    (candidate, caveat) are carried as they are; ``provisional`` must be ``true`` and then the
    source carries the one caveat text (:func:`_check_provisional`).
    """
    if not isinstance(decl, Mapping) or not isinstance(decl.get("consumers"), Mapping):
        msg = f"extent_rule must be {{source, consumers: {{consumer: entry}}}}; got {decl!r}"
        raise ValueError(msg)
    src = decl.get("source")
    if not isinstance(src, Mapping) or any(not src.get(k) for k in EXTENT_RULE_SOURCE_KEYS):
        msg = f"extent_rule.source must name {list(EXTENT_RULE_SOURCE_KEYS)}"
        raise ValueError(msg)
    for where, f in [*(("decision_files", f) for f in src["decision_files"]),
                     ("candidate_set", src["candidate_set"])]:
        if not isinstance(f, Mapping) or not f.get("file") or len(str(f.get("sha256", ""))) != 64:  # noqa: PLR2004
            msg = f"extent_rule.source.{where}: {f!r} must be {{file, sha256 (64 hex)}}"
            raise ValueError(msg)
    cons = decl["consumers"]
    want = tuple(expected_consumers())
    missing = [c for c in want if c not in cons]
    extra = sorted(str(c) for c in cons if c not in want)
    if missing or extra:
        msg = (f"extent_rule: undeclared {missing}, not consumers {extra}; every consumer declares "
               "a gate or declared: false with a reason - nothing is defaulted")
        raise ValueError(msg)
    out: dict[str, dict[str, Any]] = {}
    for c in want:
        e = cons[c]
        if not isinstance(e, Mapping):
            msg = f"extent_rule.{c} must be a mapping"
            raise ValueError(msg)
        if e.get("declared") is False:
            if not str(e.get("reason", "")).strip() or "gate" in e:
                msg = f"extent_rule.{c}: declared false needs a reason and no gate"
                raise ValueError(msg)
        else:
            ok = HRV_GATES if c == "hrv" else tuple(EXTENT_GATES)
            if e.get("gate") not in ok or e.get("declared", True) is not True:
                msg = f"extent_rule.{c}: gate {e.get('gate')!r} is none of {list(ok)}"
                raise ValueError(msg)
        out[c] = json.loads(json.dumps(dict(e), sort_keys=True, allow_nan=False))
    if not any(e.get("declared", True) for e in out.values()):
        msg = "extent_rule declares no consumer"
        raise ValueError(msg)
    _check_provisional(src, out)
    return {"source": json.loads(json.dumps(dict(src), sort_keys=True)), "consumers": out}


def tested_candidate_mismatches(rule: Mapping[str, Any], floors: Mapping[str, Any],
                                scopes: Mapping[str, str], extent_percentile: str,
                                candidate_set: Mapping[str, Any]) -> list[str]:
    """Return every way a declared extent configuration differs from the candidate it names.

    Andrea 2026-10-11 night ("Spikes C1, as the rule gives"): each declared consumer must be
    EXACTLY the candidate the harm test tested. ``candidate_set`` is the measurement candidates'
    declaration (candidate set v2: ``version``, ``common`` {cross_band_scope, own_signals},
    ``candidates`` {name: {gates, percentile, floor_populations}}); ``rule`` / ``floors`` /
    ``scopes`` / ``extent_percentile`` the checked night-config declarations. Compared per emitted
    consumer: the candidate exists; its gate; its detected-pair percentile; the cross-band scope
    (own signals only); the own-signal definition (candidate set version 2 = the package's
    ``tolerance.own_signals_by_signal``); and the floor - a candidate floor population that is
    'none' must be declared none, an own-signal population (a signal list) must be the declared
    own-signal floor with the same signals, band, frames and value, a per-band population must be
    the declared per-band floor with the same band, frames, percentile and value. Empty: the
    declaration is the tested configuration.
    """
    out: list[str] = []
    if int(candidate_set.get("version", 0)) != 2:  # noqa: PLR2004
        out.append(f"candidate set version {candidate_set.get('version')!r}: the package's own "
                   "signals are the version-2 definition")
    common = candidate_set.get("common") or {}
    if not str(common.get("cross_band_scope", "")).startswith(CROSS_BAND_SCOPE_OWN):
        out.append(f"candidate set cross-band scope {common.get('cross_band_scope')!r}")
    for c, e in sorted(rule["consumers"].items()):
        if e.get("declared") is False:
            continue
        name = e.get("candidate")
        cand = (candidate_set.get("candidates") or {}).get(name)
        if cand is None:
            out.append(f"{c}: names candidate {name!r}, which the candidate set does not hold")
            continue
        if cand["gates"].get(c) != e["gate"]:
            out.append(f"{c}: gate {e['gate']!r}, candidate {name} tested {cand['gates'].get(c)!r}")
        if c == "hrv":
            continue
        if cand.get("percentile") != extent_percentile:
            out.append(f"{c}: detected-pair percentile {extent_percentile!r}, candidate {name} "
                       f"tested {cand.get('percentile')!r}")
        if scopes.get(c) != CROSS_BAND_SCOPE_OWN:
            out.append(f"{c}: cross-band scope {scopes.get(c)!r}, tested {CROSS_BAND_SCOPE_OWN!r}")
        fp = (cand.get("floor_populations") or {}).get(c) or {}
        f = floors.get(c) if isinstance(floors, Mapping) else None
        out += _floor_mismatch(c, name, fp, f, cand.get("percentile"))
    return out


def _floor_mismatch(c: str, name: str, fp: Mapping[str, Any], f: Any, pct: Any) -> list[str]:  # noqa: ANN401
    """One consumer's declared floor against the tested candidate's floor population."""
    if f is None:
        return [f"{c}: no per-consumer floor declared; candidate {name} tested {fp}"]
    if "none" in fp:
        return [] if "none" in f else [f"{c}: declares a floor, candidate {name} tested none"]
    pop = f.get("population") or {}
    tested_value = round(float(fp.get("p99", math.nan)), 2)
    if isinstance(fp.get("signals"), list):
        same = ("per_band_percentile" not in f and "none" not in f
                and sorted(pop.get("signals") or []) == sorted(fp["signals"])
                and pop.get("band") == fp.get("band") and pop.get("n_frames") == fp.get("n_frames")
                and round(float(f.get("value", math.nan)), 2) == tested_value)
        kind = "own-signal"
    else:
        same = (f.get("per_band_percentile") == pct and pop.get("band") == fp.get("band")
                and pop.get("n_frames") == fp.get("n_frames")
                and round(float(f.get("value", math.nan)), 2) == tested_value)
        kind = "per-band"
    return [] if same else [f"{c}: declared floor {f.get('value', f.get('none'))} "
                            f"({'per-band' if 'per_band_percentile' in f else 'own-signal'}), "
                            f"candidate {name} tested the {kind} floor {tested_value} "
                            f"on {fp.get('band')} / {fp.get('n_frames')} frames"]


def not_emitted(rule: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the consumers a checked extent rule declares false (no masks emitted)."""
    return tuple(c for c, e in sorted(rule["consumers"].items()) if e.get("declared") is False)


BREATHING_FLOOR_LIMITATION: Final = (
    "Andrea 2026-10-11: breathing keeps 7.69 and the pooled floor is NOT adopted - no "
    "independent held-out clean population with an HR channel exists for breathing; the only "
    "candidate (I t02 es2 bl blind span, ~70 s) is also a held-out harm-test recording; the I "
    "span gives lower floors than J on every consumer (pooled-rule info: G:/My Drive/"
    "gems_figures/2026-10-11_breathing_floor/, proposal.md sha256 c48966e465a04f68, floors.csv "
    "sha256 fbf76cb7540cfda1)")
"""THE construction site of the breathing-floor limitation text (Andrea 2026-10-11): every
declaration that leaves breathing without an own-signal floor carries it
(:func:`check_extent_floors`, the candidates' provenance)."""

EXTENT_FLOORS_PER_BAND: Final = "per_band_percentile"
"""The ``extent_floors`` value that keeps the per-band clean-null values
(:data:`NULL_PERCENTILE_Z` at the declared ``extent_percentile``, pooled over every detection
signal) as the floor consumers' thresholds - chosen explicitly, never defaulted."""

OWN_SIGNAL_FAMILY: Final[Mapping[str, str]] = {
    "slow_wave": "stomach", "mmc": "stomach", "spikes": "nerve", "breathing": "nerve"}
"""Which signals a consumer's own-signal floor may be computed on (Andrea, revised 2026-10-10;
invariant 3): the stomach contacts for slow wave and mmc, the nerve contacts for spikes, the
HR-source channels (nerve contacts or a pair's two) for breathing. A stomach signal carries no
cuff prefix; a nerve signal does (``L_V2``, ``R_T``)."""

FLOOR_POPULATION_KEYS: Final[tuple[str, ...]] = (
    "signals", "band", "source_recording", "n_frames", "archive_file", "archive_sha256")


def _cuffed(sig: str) -> bool:
    head, _sep, tail = sig.partition("_")
    return bool(_sep) and head in ("L", "R") and tail in ("V1", "V2", "V3", "T")


def check_extent_floors(decl: object, *, skip: Sequence[str] = ()
                        ) -> str | dict[str, dict[str, Any]]:
    """Return the declared extent floors, or raise ValueError naming the gap (no default).

    Either :data:`EXTENT_FLOORS_PER_BAND` (the per-band values, chosen explicitly) or, per z
    consumer, ``{value, percentile, population: {signals, band, source_recording, n_frames,
    archive_file, archive_sha256}, source}`` (Andrea, revised 2026-10-10): the floor computed
    on THAT consumer's own signals (invariant 3, :data:`OWN_SIGNAL_FAMILY`) in its own band,
    at a percentile of :data:`EXTENT_PERCENTILES`, from one archived baseline. A value below
    :data:`Z_FLOOR` (the stage-1a cache's exact floor) is refused. A consumer that has no
    own-signal population on the baseline (breathing on a baseline with no routed HR channel)
    declares ``{"none": <why>}``: its option-(1) threshold stands, which is allowed only where a
    visible kind sets it (never for a consumer on the floor). ``skip``: consumers the extent rule
    declares false - they must be ABSENT from a per-consumer declaration.
    """
    if decl == EXTENT_FLOORS_PER_BAND:
        return EXTENT_FLOORS_PER_BAND
    if not isinstance(decl, Mapping):
        msg = (f"extent_floors is REQUIRED: {EXTENT_FLOORS_PER_BAND!r} or a mapping z consumer "
               "-> {value, percentile, population, source}; nothing is defaulted; got "
               f"{decl!r}")
        raise ValueError(msg)
    want = [c for c in z_consumers() if c not in skip]
    missing = [c for c in want if c not in decl]
    extra = sorted(str(c) for c in decl if c not in want)
    if missing or extra:
        msg = (f"extent_floors: undeclared {missing}, not z consumers or not emitted {extra}")
        raise ValueError(msg)
    return {c: (_check_per_band_entry(c, decl[c])
                if isinstance(decl[c], Mapping) and "per_band_percentile" in decl[c]
                else _check_floor_entry(c, decl[c])) for c in want}


SHA256_HEX: Final = 64


def _check_per_band_entry(c: str, e: Mapping[str, Any]) -> dict[str, Any]:
    """Check a per-band clean-null floor for ONE consumer (its band's NULL_PERCENTILE_Z).

    ``{"per_band_percentile", "value", "source", "population": {signals (a description: every
    detection signal), band, source_recording, n_frames, archive_file, archive_sha256}}`` - the
    floor the measurement candidates' C1 used (candidate set v2). The value must be exactly the
    table's value for the consumer's band at that percentile.
    """
    pop = e.get("population")
    gap = [k for k in ("value", "source", "population") if not e.get(k)]
    gap += ([f"population.{k}" for k in FLOOR_POPULATION_KEYS if not pop.get(k)]
            if isinstance(pop, Mapping) else [])
    if gap:
        msg = f"extent_floors.{c} lacks {gap} (required, never defaulted)"
        raise ValueError(msg)
    assert isinstance(pop, Mapping)
    pct, band = e["per_band_percentile"], extent_consumers()[c].band
    if pct not in EXTENT_PERCENTILES or pop["band"] != band:
        msg = (f"extent_floors.{c}: per-band floor at {pct!r} of band {pop['band']!r}; the "
               f"consumer's band is {band}, percentiles {list(EXTENT_PERCENTILES)}")
        raise ValueError(msg)
    if float(e["value"]) != float(NULL_PERCENTILE_Z[pct][band]):
        msg = (f"extent_floors.{c}: value {e['value']} is not the clean null's {pct} of band "
               f"{band} ({NULL_PERCENTILE_Z[pct][band]})")
        raise ValueError(msg)
    return dict(json.loads(json.dumps(dict(e), sort_keys=True, allow_nan=False)))


def _check_floor_entry(c: str, e: object) -> dict[str, Any]:
    """One consumer's own-signal floor declaration, checked (:func:`check_extent_floors`)."""
    if not isinstance(e, Mapping):
        msg = f"extent_floors.{c} must be {{value, percentile, population, source}}"
        raise ValueError(msg)
    if set(e) <= {"none", "limitation"} and "none" in e:
        if not str(e["none"]).strip():
            msg = f"extent_floors.{c}: 'none' must say why the consumer has no population"
            raise ValueError(msg)
        if c == "breathing" and e.get("limitation") != BREATHING_FLOOR_LIMITATION:
            msg = ("extent_floors.breathing: a 'none' declaration carries the recorded "
                   "limitation (tolerance_map.BREATHING_FLOOR_LIMITATION), verbatim")
            raise ValueError(msg)
        return {k: str(v) for k, v in e.items()}
    pop = e.get("population")
    gap = [k for k in ("value", "percentile", "population", "source") if not e.get(k)]
    gap += ([f"population.{k}" for k in FLOOR_POPULATION_KEYS if not pop.get(k)]
            if isinstance(pop, Mapping) else [])
    if gap:
        msg = f"extent_floors.{c} lacks {gap} (required, never defaulted)"
        raise ValueError(msg)
    assert isinstance(pop, Mapping)
    v = float(e["value"])
    if not (math.isfinite(v) and v >= Z_FLOOR):
        msg = (f"extent_floors.{c}: value {e['value']!r} must be finite and >= {Z_FLOOR} "
               "(the stage-1a cache is exact only at or above it)")
        raise ValueError(msg)
    if e["percentile"] not in EXTENT_PERCENTILES:
        msg = f"extent_floors.{c}: percentile {e['percentile']!r} not in {EXTENT_PERCENTILES}"
        raise ValueError(msg)
    band = extent_consumers()[c].band
    if pop["band"] != band:
        msg = (f"extent_floors.{c}: population band {pop['band']!r} is not {c}'s own band "
               f"{band} (invariant 3)")
        raise ValueError(msg)
    sigs = pop["signals"]
    if not isinstance(sigs, Sequence) or isinstance(sigs, str) or not sigs:
        msg = f"extent_floors.{c}: population.signals must be a non-empty list"
        raise ValueError(msg)
    want_nerve = OWN_SIGNAL_FAMILY[c] == "nerve"
    wrong = [str(x) for x in sigs if _cuffed(str(x)) != want_nerve]
    if wrong:
        msg = (f"extent_floors.{c}: {wrong} are not {OWN_SIGNAL_FAMILY[c]} signals - a "
               f"floor is computed on the consumer's OWN signals (invariant 3)")
        raise ValueError(msg)
    n = pop["n_frames"]
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        msg = f"extent_floors.{c}: population.n_frames must be an int > 0"
        raise ValueError(msg)
    sha = str(pop["archive_sha256"])
    if len(sha) != SHA256_HEX or any(ch not in "0123456789abcdef" for ch in sha):
        msg = f"extent_floors.{c}: population.archive_sha256 must be 64 lowercase hex"
        raise ValueError(msg)
    return dict(json.loads(json.dumps(dict(e), sort_keys=True, allow_nan=False)))


def own_signal_floor_record(z: Mapping[tuple[str, str], npt.ArrayLike], *, consumer: str,
                            signals: Sequence[str], percentile: str) -> dict[str, Any]:
    """Return one own-signal floor as an archive stores it (``value`` rounded to 0.01).

    The percentile of every finite frame of ``z[(signal, band)]`` pooled over ``signals`` in
    ``consumer``'s own band - the one computation the archive script and the reproducibility
    test share. ``z`` is detection's robust z on the log envelope of the archived baseline.
    """
    if percentile not in EXTENT_PERCENTILES:
        msg = f"percentile {percentile!r} not in {EXTENT_PERCENTILES}"
        raise ValueError(msg)
    band = extent_consumers()[consumer].band
    absent = [s for s in signals if (s, band) not in z]
    if absent:
        msg = f"{consumer}: no z for {absent} in band {band}"
        raise KeyError(msg)
    v = np.concatenate([np.asarray(z[(s, band)], dtype=np.float64).ravel() for s in signals])
    v = v[np.isfinite(v)]
    if v.size == 0:
        msg = f"{consumer}: no finite frame on {list(signals)} in band {band}"
        raise ValueError(msg)
    return {"consumer": consumer, "percentile": percentile, "band": band,
            "signals": sorted(str(s) for s in signals), "n_frames": int(v.size),
            "value": round(float(np.percentile(v, float(percentile[1:]))), 2)}


def verify_floor_archive(consumer: str, entry: Mapping[str, Any], archive: Mapping[str, Any]
                         ) -> None:
    """Refuse a declared floor its archive does not reproduce (ValueError).

    ``archive`` is the hashed file the declaration names. Own-signal floor: its
    ``per_consumer_own_signals`` rows (:func:`own_signal_floor_record`); the declared value,
    percentile, band, signals, n_frames and source recording must be one of its rows exactly.
    Per-band floor: the clean-null archive (null.json) of the declared source recording, whose
    ``per_band_all_signals`` row for the band gives the value (to 0.01) on exactly n_frames.
    """
    pop = entry["population"]
    if "per_band_percentile" in entry:  # the null.json per-band row of every detection signal
        if archive.get("session") != pop["source_recording"]:
            msg = (f"extent_floors.{consumer}: archive is of {archive.get('session')!r}, not the "
                   f"declared source recording {pop['source_recording']!r}")
            raise ValueError(msg)
        row = (archive.get("per_band_all_signals") or {}).get(pop["band"]) or {}
        pct = entry["per_band_percentile"]
        if (round(float(row.get(pct, math.nan)), 2) != round(float(entry["value"]), 2)
                or int(row.get("n", -1)) != int(pop["n_frames"])):
            msg = (f"extent_floors.{consumer}: the archive's band {pop['band']} {pct} row {row} "
                   f"does not reproduce {entry['value']} on {pop['n_frames']} frames")
            raise ValueError(msg)
        return
    want = {"consumer": consumer, "percentile": entry["percentile"], "band": pop["band"],
            "signals": sorted(str(s) for s in pop["signals"]), "n_frames": int(pop["n_frames"]),
            "value": round(float(entry["value"]), 2)}
    if archive.get("session") != pop["source_recording"]:
        msg = (f"extent_floors.{consumer}: archive is of {archive.get('session')!r}, not the "
               f"declared source recording {pop['source_recording']!r}")
        raise ValueError(msg)
    rows = (archive.get("per_consumer_own_signals") or {}).get(consumer) or []
    if want not in rows:
        msg = (f"extent_floors.{consumer}: the archive does not reproduce {want} (rows "
               f"{rows[:3]})")
        raise ValueError(msg)


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


def check_cross_band_scope(scopes: object, *, skip: Sequence[str] = ()) -> dict[str, str]:
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
    want = [c for c in z_consumers() if c not in skip]
    missing = [c for c in want if c not in scopes]
    extra = sorted(str(c) for c in scopes if c not in want)
    bad = {str(c): v for c, v in scopes.items() if v not in CROSS_BAND_SCOPES}
    if missing or extra or bad:
        msg = (f"cross-band scope: undeclared {missing}, not z consumers {extra}, unknown "
               f"values {bad}; each of {want} declares one of "
               f"{list(CROSS_BAND_SCOPES)}, nothing is defaulted (review 7 finding 2)")
        raise ValueError(msg)
    return {c: str(scopes[c]) for c in want}


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

    extent_floors: str | Mapping[str, Mapping[str, Any]] = EXTENT_FLOORS_PER_BAND
    """The DECLARED extent floors (:func:`check_extent_floors`; required by the builder)."""

    extent_rule: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    """The DECLARED per-consumer extent rule (:func:`check_extent_rule`), when the night config
    declares one instead of ``core_blanking`` (Andrea 2026-10-11 night); empty otherwise."""

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
                **({"extent_rule": dict(self.extent_rule),
                    "extent_gates": dict(EXTENT_GATES),
                    "provisional": provisional_consumers(self.extent_rule)}
                   if self.extent_rule else
                   {"core_pad_rule": CORE_PAD_RULE,
                    "core_blanking": dict(sorted(self.core_blanking.items()))}),
                "extent_percentile": self.extent_percentile,
                "extent_floors": (self.extent_floors if isinstance(self.extent_floors, str)
                                  else {c: dict(v) for c, v in
                                        sorted(self.extent_floors.items())}),
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


def _floored(c: str, t: ConsumerThreshold, floors: str | Mapping[str, Mapping[str, Any]],
             pct: str) -> ConsumerThreshold:
    """Apply the declared floors to one option-(1) threshold (:func:`option1_tolerances`)."""
    if not isinstance(floors, str):
        f = floors[c]
        if "per_band_percentile" in f:
            if not t.setter.startswith("floor"):
                return t  # a visible kind sets it (as the per-band rule leaves it)
            return dataclasses.replace(
                t, z=float(f["value"]),
                setter=(f"{t.setter}; extent at the clean null's {f['per_band_percentile']} of "
                        f"band {f['population']['band']} (per-band floor, declared per "
                        "consumer)"))
        if "none" in f:
            if t.setter.startswith("floor"):
                msg = (f"extent_floors.{c} declares no own-signal population, but {c} is on the "
                       "floor (no visible kind sets its threshold): it needs a population")
                raise ValueError(msg)
            return dataclasses.replace(t, setter=f"{t.setter}; no own-signal floor population "
                                                 f"({f['none']}): the option-(1) threshold stands")
        fv = float(f["value"])
        pop = f["population"]
        binds = "own-signal floor" if fv >= t.z else "option-(1) threshold"
        return dataclasses.replace(
            t, z=max(float(t.z), fv),
            setter=(f"{t.setter}; max with the own-signal floor {fv} ({f['percentile']} of "
                    f"{c}'s own signals {list(pop['signals'])} in band {pop['band']}, "
                    f"{pop['source_recording']}; Andrea, revised 2026-10-10): the {binds} "
                    "binds"))
    if t.setter.startswith("floor"):
        band = extent_consumers()[c].band
        return dataclasses.replace(t, z=float(NULL_PERCENTILE_Z[pct][band]),
                                   setter=f"{t.setter}; extent at the clean null's {pct} of "
                                          f"band {band} (RULING 2026-10-10 item 2)")
    return t


def _declarations(core_blanking: object, extent_rule: object, cross_band_scope: object,
                  extent_floors: object) -> tuple[Any, ...]:
    """Check the extent declarations together (:func:`option1_tolerances`)."""
    if (core_blanking is None) == (extent_rule is None):
        msg = ("declare exactly one of core blanking (core_blanking, RULING 2026-10-10 item 2) "
               "and extent_rule (Andrea 2026-10-11 night); nothing is defaulted")
        raise ValueError(msg)
    rule = check_extent_rule(extent_rule) if extent_rule is not None else {}
    skip = not_emitted(rule) if rule else ()
    scopes = check_cross_band_scope(cross_band_scope, skip=skip)
    bad = {c: v for c, v in scopes.items() if v != CROSS_BAND_SCOPE_OWN}
    if rule and bad:
        msg = (f"under an extent rule every emitted z consumer's cross-band scope is "
               f"{CROSS_BAND_SCOPE_OWN!r}; got {bad}")
        raise ValueError(msg)
    cores = check_core_blanking(core_blanking) if core_blanking is not None else {}
    return rule, skip, scopes, cores, check_extent_floors(extent_floors, skip=skip)


def option1_tolerances(tolerances: Mapping[str, Any], mapping: Mapping[str, Any],
                       scan: Mapping[str, Any], *, tolerances_sha256: str,
                       criterion_sha256: str, mapping_sha256: str, scan_sha256: str,
                       generation: str, cross_band_scope: object,
                       extent_percentile: object, extent_floors: object,
                       core_blanking: object = None,
                       extent_rule: object = None) -> Option1Tolerances:
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

    Exactly ONE of ``core_blanking`` (the 2026-10-10 rule) and ``extent_rule`` (Andrea 2026-10-11
    night, :func:`check_extent_rule`) is declared; neither or both is refused. Under an extent
    rule every emitted z consumer's cross-band scope must be ``own_signals_only`` (the gate model
    takes the detected extent over the core's own detecting pairs); a consumer declared false
    has no scope, floor, threshold or residual-risk row.

    ``extent_floors`` (Andrea, revised 2026-10-10; :func:`check_extent_floors`, required):
    :data:`EXTENT_FLOORS_PER_BAND` keeps the rule above; per-consumer own-signal floors make
    every z consumer's threshold max(its option-(1) threshold, its own floor) - the floor
    consumers take their own floor, breathing keeps 7.69 while that is above its floor. The
    cross-band floors stay per band (``extent_percentile``).
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
    rule, skip, scopes, cores, floors = _declarations(core_blanking, extent_rule,
                                                      cross_band_scope, extent_floors)
    pct = check_extent_percentile(extent_percentile)
    thresholds: dict[str, ConsumerThreshold] = {}
    residual: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for c in (x for x in z_consumers() if x not in skip):
        entry = (mapping.get("consumers") or {}).get(c)
        if not isinstance(entry, Mapping):
            msg = f"the tolerance mapping has no entry for {c}: its threshold is unknown"
            raise ValueError(msg)
        t = option1_threshold(c, entry.get("kinds") or {}, entry.get("kinds_not_raised") or ())
        thresholds[c] = t = _floored(c, t, floors, pct)
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
    return Option1Tolerances(table, thresholds, residual, flags, inputs, scopes, pct, cores,
                             floors, rule)
