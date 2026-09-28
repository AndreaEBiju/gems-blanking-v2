"""Write ``meta.json`` for the reachable new-cohort recordings.

Labelling cannot start without these: the loader refuses to infer geometry, so
audit mode cannot open a new-cohort recording until each session has one.

**Order is read from the file, never invented here.** Every table is built from
that recording's own ``chanlabels``, so this script cannot introduce the
transposition it exists to prevent.

**Cohort constants are not written per recording.** ``units``, ``config`` and
``channel_order_source`` live in ``protocol.yaml`` (invariant 24) - they cannot
legitimately differ between two recordings of one cohort, and 800 copies of a
constant are 800 chances for one to drift - and ``rostral_end`` is one too
(Andrea, 2026-09-26: contact 1 is rostral), so it is declared in the protocol and
never written here. ``meta.json`` carries ``animal``, ``session``, ``channels``,
``geometry_updated_at``, ``acquired_at``, ``folder_name`` and ``block_name``, plus
``excluded`` when Andrea's quality ruling applies (``_BAD`` / ``_INCOMPLETE`` and
its paired recording; see :mod:`gems_blanking_v2.io.quality`).

**Meaning from the folder, identity from the block** (Andrea, 2026-09-26): the key
is the instrument's block name; animal, condition and quality are read from the
folder name, because her renames are corrections. Every disagreement is logged.

**The store key is the acquisition block plus its ``.tsq`` start time**
(:func:`~gems_blanking_v2.io.tdt_block.session_key`), not the file stem. The stem's
digits are a time of day with no date, and every file of one block (``_sig``,
``_notched``, ``_stim``) shares one ``meta.json`` rather than keying three.


Usage::

    python write_new_cohort_meta.py [--apply] [--force]

Dry-run by default. Incremental: a recording whose geometry already matches is
left alone, so re-running as the corpus grows touches only what is new.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from gems_blanking_v2.io.chanlabels import channels_from_labels, read_channel_labels
from gems_blanking_v2.io.channel_map import ChannelMap, meta_path, save_geometry
from gems_blanking_v2.io.corpus import read_scan_roots
from gems_blanking_v2.io.quality import (
    EXCLUSION_KEY,
    ExclusionPlan,
    duration_verdict,
    pair_key,
    plan_exclusions,
)
from gems_blanking_v2.io.stim_split import (
    PROTOCOL_FILENAME,
    ProtocolBook,
    ProtocolSpec,
    load_protocol_book,
)
from gems_blanking_v2.io.store import GemsStore, atomic_write_text, find_gems_root
from gems_blanking_v2.io.tdt_block import (
    SESSION_KEY_SUFFIX,
    AcquisitionRecordError,
    acquired_at,
    directory_disagrees,
    session_key,
)
from scipy.io import loadmat, whosmat

ANIMAL_RE: Final = re.compile(r"^gems_([a-z])_", re.IGNORECASE)
"""``gems_j_t01_ms3_bl_230315`` -> animal ``J``. The cohort's own convention."""

NEW_COHORT_CHANNELS: Final = 9
ENUMERATED: Final = "*_sig.mat"
DUPLICATES: Final = "duplicates.json"


class CollidingStorePathsError(RuntimeError):
    """Two recordings claim one ``meta.json`` path."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """What happened to one recording, and why."""

    path: str
    source: str
    animal: str | None
    session: str
    status: str
    detail: str = ""
    instrument_block: str = ""
    """Set when the directory was renamed after acquisition: what TDT recorded."""
    duration_min: float | None = None
    """The recording's own length: sample count / fs. None only if unreadable."""


def plan(scan_root: Path, probe_config: str) -> list[Outcome]:
    """Decide, per recording, whether a ``meta.json`` can be written."""
    out: list[Outcome] = []
    files = sorted(scan_root.rglob(ENUMERATED))
    print(f"{len(files)} {ENUMERATED} under {scan_root.name}", flush=True)
    t0 = time.time()
    for n, p in enumerate(files, 1):
        rel = p.relative_to(scan_root).as_posix()
        try:
            key = session_key(p)
        except AcquisitionRecordError as exc:
            out.append(Outcome(rel, str(p), None, p.stem, "skipped_bad_block", str(exc)))
            continue
        if key is None:
            out.append(Outcome(rel, str(p), None, p.stem, "skipped_no_tsq",
                               "no acquisition record beside the file, so no store "
                               "key; the stem carries no date and is not one"))
            continue
        session = key
        # Andrea, 2026-09-26: the block name is IDENTITY (the key above, which
        # never moves); the FOLDER name is authoritative for MEANING - animal,
        # condition, quality - because renames are her corrections. So meaning is
        # read from the folder, and every disagreement is kept in the plan so a
        # future rename is visible rather than silently absorbed.
        renamed = directory_disagrees(p)
        instrument_block = renamed[0] if renamed else ""
        m = ANIMAL_RE.match(p.parent.name)
        if m is None:
            out.append(Outcome(rel, str(p), None, session, "skipped_no_animal",
                               "folder does not match gems_<letter>_; the animal "
                               "cannot be identified and must not be guessed",
                               instrument_block))
            continue
        animal = m.group(1).upper()
        labels = read_channel_labels(p)
        if labels is None:
            out.append(Outcome(rel, str(p), animal, session, "skipped_no_chanlabels",
                               "no chanlabels, so the order would have to be "
                               "declared - which is what the rule forbids"))
            continue
        if len(labels) != NEW_COHORT_CHANNELS:
            out.append(Outcome(rel, str(p), animal, session, "skipped_channel_count",
                               f"{len(labels)} channels, not "
                               f"{NEW_COHORT_CHANNELS}: {list(labels)}"))
            continue
        try:
            channels_from_labels(labels, rostral_end=None, config=probe_config)
        except ValueError as exc:
            out.append(Outcome(rel, str(p), animal, session, "skipped_bad_label", str(exc)))
            continue
        out.append(Outcome(rel, str(p), animal, session, "ready", ",".join(labels),
                           instrument_block, _duration_min(p)))
        if n % 200 == 0:
            print(f"  {n}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def _duration_min(path: Path) -> float | None:
    """Length in minutes from the file's own sample count and fs - no array loaded.

    Read in the pass that already opens every file for its chanlabels (invariant
    26), rather than a second sweep over the Drive.
    """
    try:
        shapes = {name: shape for name, shape, _ in whosmat(path)}
        fs = float(np.asarray(loadmat(path, variable_names=["fs"])["fs"]).squeeze())
        return round(int(shapes["signal"][0]) / fs / 60.0, 3)
    except (OSError, KeyError, ValueError, NotImplementedError):
        return None


MAX_DURATION_EXCLUSION_FRACTION: Final = 0.10
"""If the duration rule would exclude more than this share of the corpus, stop and
report before applying (Andrea, 2026-09-26): a rule removing a large share of the
data deserves her look at the list first."""


def resolve_collisions(
    ready: list[Outcome], duplicates: dict[str, dict[str, Any]]
) -> tuple[list[Outcome], list[Outcome]]:
    """Split ready recordings into those to write and those deduplicated.

    **Raises rather than warning.** A warning line in a summary is read once, by
    someone who already believes the run worked - 27 of them scrolled past in
    this very script and 27 files were overwritten anyway. A store path that two
    recordings both claim is not a path, it is a bug in the key.

    A group collapses only when ``duplicates.json`` proves the recordings share
    a SIGNAL array. Absent or unverified, this raises.
    """
    by_key: dict[tuple[str | None, str], list[Outcome]] = {}
    for o in ready:
        by_key.setdefault((o.animal, o.session), []).append(o)

    write_these: list[Outcome] = []
    deduped: list[Outcome] = []
    unproven: list[str] = []
    for (_animal, session), group in sorted(by_key.items()):
        if len(group) == 1:
            write_these.append(group[0])
            continue
        # duplicates.json is keyed by file stem: it records what was compared,
        # which is the files, not the store keys built from them.
        stems = {Path(o.source).stem for o in group}
        entry = duplicates.get(next(iter(stems))) if len(stems) == 1 else None
        if entry and entry.get("identical"):
            write_these.append(group[0])
            deduped.extend(group[1:])
            continue
        unproven.append(
            f"{session}: {len(group)} recordings claim it -> "
            + ", ".join(o.path for o in group)
        )
    if unproven:
        detail = "\n  ".join(unproven)
        msg = (
            f"{len(unproven)} store path(s) are claimed by more than one "
            f"recording and are NOT proven identical:\n  {detail}\n\n"
            "Run verify_duplicates.py first. The key already includes the block's "
            ".tsq start time, so two recordings sharing it started in the same "
            "block at the same second - a copy, or an acquisition record that "
            "contradicts itself. Nothing is written until this is zero."
        )
        # DO NOT downgrade this to a warning or a skip, and do not delete it now
        # that the key includes acquired_at. It reads like migration scaffolding
        # for the 27 duplicated blocks, and it is not. The key is block + .tsq
        # start time, so a group here is either a copied block (identical signal,
        # proven in duplicates.json, collapsed) or two files that claim the same
        # acquisition and differ - a corrupted copy, or a .tsq copied into the
        # wrong directory. Either way one recording would silently overwrite
        # another's identity and channel order. A better key is not a reason to
        # remove the thing that catches the case where it is not (spec, re-key).
        raise CollidingStorePathsError(msg)
    return write_these, deduped


def _extras(o: Outcome, exclusion: dict[str, Any] | None, store: GemsStore) -> dict[str, Any]:
    """Return the per-recording fields this generator owns beyond the geometry.

    ``folder_name`` and ``block_name`` are both kept: the folder is authoritative
    for meaning, the block for identity, and a quality flag typed into a folder
    name is a label that must not be discarded because the key does not need it.
    """
    folder = Path(o.source).parent.name
    out: dict[str, Any] = {
        "folder_name": folder,
        "block_name": o.instrument_block or folder,
        # Where the recording is, relative to gems_root in POSIX form (cross-
        # platform rule 2): an absolute path saved here would not resolve on the
        # other OS. The audit opens a recording from its store entry through this.
        "source_path": store.relpath(Path(o.source)),
    }
    if o.duration_min is not None:
        # The recording's own length (sample count / fs), so the blind audit can
        # plan spans from the store alone without opening 800 files.
        out["duration_s"] = round(o.duration_min * 60.0, 3)
    if exclusion is not None:
        out[EXCLUSION_KEY] = exclusion
    return out


def write(
    outcomes: list[Outcome], spec: ProtocolSpec, store: GemsStore, *, force: bool,
    exclusions: dict[str, dict[str, Any]], apply: bool = True,
) -> dict[str, str]:
    """Write ``meta.json``. Returns ``{session: "written" | "unchanged"}``.

    With ``apply=False`` nothing is written, but the same comparison runs, so a
    dry run reports what WOULD change instead of calling every file changed -
    which is what makes a dry run usable as the idempotency check.

    Incremental unless ``force``: a file already holding this channel table AND
    these extras is left alone, so re-running as recordings arrive touches only
    what is new. Rewriting 800 files on a shared Drive every run would also make
    every mtime a lie about when the geometry was decided.
    """
    state: dict[str, str] = {}
    for o in outcomes:
        labels = tuple(o.detail.split(","))
        cmap = ChannelMap(
            animal=o.animal or "?",
            channels=channels_from_labels(labels, rostral_end=None,
                                          config=spec.config),
            units=spec.units,  # type: ignore[arg-type]
        )
        extras = _extras(o, exclusions.get(o.session), store)
        when = acquired_at(Path(o.source))
        if when is not None:
            extras["acquired_at"] = when
        existing = meta_path(store, cmap.animal, o.session)
        if not force and existing.is_file():
            try:
                doc = json.loads(existing.read_text(encoding="utf-8"))
                same = (
                    [c["label"] for c in doc.get("channels", [])] == list(labels)
                    and all(doc.get(k) == v for k, v in extras.items())
                    and (EXCLUSION_KEY in extras) == (EXCLUSION_KEY in doc)
                    and not any("rostral_end" in c for c in doc.get("channels", []))
                )
            except (OSError, json.JSONDecodeError, KeyError):
                same = False
            if same:
                state[o.session] = "unchanged"
                continue
        if not apply:
            state[o.session] = "written"
            continue
        target = save_geometry(cmap, o.session, store, mirror_to_profile=False)
        # acquired_at comes from the instrument's own .tsq header, never from the
        # containing folder: 27 blocks sit under a date that is not theirs.
        # save_geometry preserves keys it does not own, so these survive a later
        # geometry update; an exclusion no longer implied is removed here.
        doc = json.loads(target.read_text(encoding="utf-8"))
        doc.pop(EXCLUSION_KEY, None)
        doc.update(extras)
        atomic_write_text(target, json.dumps(doc, indent=2, sort_keys=True) + "\n")
        state[o.session] = "written"
    return state


def _labels(path: Path) -> list[str]:
    return [c["label"] for c in json.loads(path.read_text(encoding="utf-8"))["channels"]]


def retire_stem_keys(
    to_write: list[Outcome], store: GemsStore, *, apply: bool,
    held: tuple[Outcome, ...] = (),
) -> tuple[int, list[str]]:
    """Remove each pre-re-key ``data/<animal>/<stem>/``, once its replacement is proven.

    Retired only when the new ``meta.json`` exists with the SAME channel table
    and the old directory holds nothing but ``meta.json`` - it is regenerable
    output, but a directory holding anything else is not ours to remove. Returns
    ``(retired, unexplained)``; ``unexplained`` lists every stem-keyed directory
    still in the store afterwards, which the caller treats as a failure -
    except the old directories of ``held`` recordings, which stay by design until
    their question is answered.
    """
    data = store.root / "data"
    kept = {q for o_h in held for q in data.glob(f"*/{Path(o_h.source).stem}")}
    retired = 0
    for o in to_write:
        old = meta_path(store, o.animal or "?", Path(o.source).stem)
        if not old.is_file():
            continue
        if not apply:
            retired += 1
            continue
        new = meta_path(store, o.animal or "?", o.session)
        try:
            old_labels = _labels(old)
            new_labels = _labels(new)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if old_labels != new_labels or {q.name for q in old.parent.iterdir()} != {old.name}:
            continue
        old.unlink()
        old.parent.rmdir()
        retired += 1
    unexplained = sorted(
        d.relative_to(store.root).as_posix()
        for d in data.glob("*/*")
        if d.is_dir() and not SESSION_KEY_SUFFIX.search(d.name) and d not in kept
    )
    return retired, unexplained


def _report_plan(outcomes: list[Outcome], stamp: str) -> list[Outcome]:
    """Print distinct value-tuples with counts, not one line per recording."""
    ready = [o for o in outcomes if o.status == "ready"]
    print("\n--- distinct outcomes (enumerated_at " + stamp + ") ---")
    tuples = Counter((o.status, o.animal or "-", o.detail) for o in outcomes)
    for (status, animal, detail), n in tuples.most_common():
        print(f"  {n:5d}  {status:22} animal={animal:3} {detail[:78]}")
    print(f"  animals ready: {sorted({o.animal for o in ready if o.animal})}")
    return ready


def _reconcile(
    stamp: str, n_enum: int, written: int, unchanged: int,
    n_dedup: int, n_skip: int, n_excluded: int = 0,
) -> None:
    """Every enumerated recording lands in exactly one term, or this raises.

    ``excluded`` recordings still get a ``meta.json`` - it records the flag and
    the reason - but they are counted in their own term, not in written or
    unchanged, so the exclusion is visible in the arithmetic.
    """
    accounted = written + unchanged + n_dedup + n_skip + n_excluded
    residual = n_enum - accounted
    print("\n--- reconciliation ---")
    print(f"  enumerated_at                {stamp}")
    print(f"  recordings enumerated        {n_enum}")
    print(f"    written (new or changed)   {written}")
    print(f"    unchanged (incremental)    {unchanged}")
    print(f"    deduplicated (same signal) {n_dedup}")
    print(f"    skipped (reasons above)    {n_skip}")
    print(f"    excluded (recorded, kept)  {n_excluded}")
    print(f"    {'-' * 30}")
    print(f"    accounted for              {accounted}")
    print(f"    RESIDUAL                   {residual}")
    if residual:
        msg = (
            f"{residual} recordings are unaccounted for. Every enumerated "
            "recording must land in exactly one term; a residual means a branch "
            "drops one silently, which is the only evidence such a bug gives."
        )
        raise RuntimeError(msg)


RULED_EXCLUSIONS: Final[dict[str, dict[str, str]]] = {
    "gems_i_t03_es2_sr_223947": {
        "reason": "no_stim_monitor",
        "source": "all three stimulation monitor channels (adc1, adc2, vib) are "
                  "exactly zero for the whole file; 03B refuses a constant monitor "
                  "(invariant 41)",
        "ruling": "Andrea 2026-09-26: not sure stimulation happened in this session; "
                  "exclude it and its baseline partner",
    },
}
"""Exclusions by a named ruling, keyed by CORRECTED folder name (authoritative for
meaning). Each is excluded with its partner, like a folder flag. A key that
matches no enumerated recording raises: a ruling that silently applies to
nothing is the failure this exists to prevent."""


RULED_ANIMAL_EXCLUSIONS: Final[dict[str, dict[str, str]]] = {
    "D": {
        "reason": "animal_excluded",
        "source": "new-cohort animal D (every gems_d_... recording): its signals "
                  "are noise (found in the blind audit, round 1)",
        "ruling": "Andrea 2026-09-27: remove animal D from everything - audit, "
                  "training, corpora - as if the data did not exist",
    },
}
"""Whole-animal exclusions by a named ruling, keyed by the animal letter read from
the CORRECTED folder name (``gems_<letter>_...``). Every recording of the animal is
excluded - its partners are the same animal, so pairing adds nothing. An animal
that matches no enumerated recording raises, like a folder ruling."""


def _ruled(folders: dict[str, str]) -> dict[str, dict[str, str]]:
    """Map :data:`RULED_EXCLUSIONS` and :data:`RULED_ANIMAL_EXCLUSIONS` onto session keys.

    A session named by both keeps its folder ruling (the more specific) and carries
    the animal ruling as ``also_animal``.
    """
    by_folder: dict[str, list[str]] = {}
    for session, folder in folders.items():
        by_folder.setdefault(folder, []).append(session)
    out: dict[str, dict[str, str]] = {}
    for folder, record in RULED_EXCLUSIONS.items():
        sessions = by_folder.get(folder, [])
        if len(sessions) != 1:
            msg = (f"ruled exclusion {folder!r} matches {len(sessions)} recordings "
                   f"(expected exactly 1): {sessions}")
            raise SystemExit(msg)
        out[sessions[0]] = dict(record)
    for animal, record in RULED_ANIMAL_EXCLUSIONS.items():
        matched = [s for s, f in folders.items()
                   if (m := ANIMAL_RE.match(f)) and m.group(1).upper() == animal]
        if not matched:
            msg = f"ruled animal exclusion {animal!r} matches no recording"
            raise SystemExit(msg)
        for session in matched:
            if session in out:
                out[session] = {**out[session], "also_animal": animal}
            else:
                out[session] = {**record, "animal": animal}
    return out


def _report_exclusions(excl: ExclusionPlan, folders: dict[str, str]) -> None:
    """Print every exclusion with its reason, and every pairing left unresolved."""
    by_reason: dict[str, dict[str, dict[str, Any]]] = {}
    for s, r in excl.excluded.items():
        by_reason.setdefault(r["reason"], {})[s] = r
    flagged = by_reason.get("quality_flag", {})
    short = by_reason.get("duration_below_threshold", {})
    partners = by_reason.get("partner_of_flagged", {})
    ruled = {s: r for s, r in excl.excluded.items()
             if r["reason"] not in ("quality_flag", "duration_below_threshold",
                                    "partner_of_flagged")}
    print(f"\n--- exclusions (Andrea 2026-09-26): {len(flagged)} flagged, {len(ruled)} ruled, "
          f"{len(short)} short (not flagged), {len(partners)} partners ---")
    for s, r in sorted(ruled.items()):
        print(f"  RULED    {folders[s]:40} {r['reason']}: {r['ruling']}")
    for s, r in sorted(flagged.items()):
        also = (f"  (also short: {r['also_short']['duration_min']:.2f} min)"
                if "also_short" in r else "")
        print(f"  FLAGGED  {folders[s]:40} {'/'.join(r['flags'])}{also}")
    for s, r in sorted(short.items()):
        print(f"  SHORT    {folders[s]:40} {r['duration_min']:.2f} min < {r['threshold_min']:g}")
    for s, r in sorted(partners.items()):
        print(f"  PARTNER  {folders[s]:40} of {folders[r['partner_of']]}")
    for s, cands in sorted(excl.ambiguous.items()):
        print(f"  AMBIGUOUS partner for {folders[s]}: {[folders[c] for c in cands]}")
    for s, pairs in sorted(excl.unconfirmed.items()):
        listed = ", ".join(f"{folders[c]} ({g:+.1f} min)" for c, g in pairs)
        print(f"  UNCONFIRMED {folders[s]}: name match(es) outside the window: {listed}")
    for s in excl.missing:
        print(f"  MISSING  partner for {folders[s]}")
    for s, lost in sorted(excl.superseded.items()):
        listed = ", ".join(f"{folders[sr]} pairs with {folders.get(bl, bl)}" for sr, bl in lost)
        print(f"  SUPERSEDED {folders[s]}: not the baseline ({listed}); excluded alone")


def _duration_rule(
    to_write: list[Outcome], spec: ProtocolSpec
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Apply completeness-by-duration: ``(short, borderline, no_infix)``."""
    if spec.min_baseline_min is None or spec.min_sr_min is None:
        msg = "protocol.yaml declares no min_baseline_min / min_sr_min"
        raise SystemExit(msg)
    short: dict[str, dict[str, Any]] = {}
    borderline: list[dict[str, Any]] = []
    no_infix: list[str] = []
    for o in to_write:
        parsed = pair_key(Path(o.source).parent.name)
        if parsed is None or o.duration_min is None:
            no_infix.append(Path(o.source).parent.name)
            continue
        infix = parsed[1]
        thr = spec.min_baseline_min if infix == "bl" else spec.min_sr_min
        verdict = duration_verdict(infix, o.duration_min, min_baseline_min=spec.min_baseline_min,
                                   min_sr_min=spec.min_sr_min)
        entry = {"duration_min": o.duration_min, "threshold_min": thr, "infix": infix}
        if verdict == "short":
            short[o.session] = entry
        elif verdict == "borderline":
            borderline.append({"folder": Path(o.source).parent.name, **entry})
    return short, borderline, no_infix


def _report_duration(
    excl: ExclusionPlan, short: dict[str, dict[str, Any]], borderline: list[dict[str, Any]],
    no_infix: list[str], folders: dict[str, str], spec: ProtocolSpec, n_total: int,
) -> tuple[list[str], bool]:
    """Print the duration rule's outcome; return (excluded by it, over the stop limit)."""
    by_duration = [s for s, r in excl.excluded.items()
                   if r["reason"] == "duration_below_threshold"
                   or (r["reason"] == "partner_of_flagged" and r["partner_of"] in short)]
    print(f"\n--- completeness by duration (bl >= {spec.min_baseline_min:g} min, "
          f"sr >= {spec.min_sr_min:g} min) ---")
    print(f"  short: {len(short)}  | excluded by the rule incl. partners: {len(by_duration)} "
          f"of {n_total} ({100 * len(by_duration) / n_total:.1f}%)")
    print(f"  borderline (within 10% below, listed not decided): {len(borderline)}")
    print(f"  no bl/sr infix, rule not applicable: {len(no_infix)}")
    for b in sorted(borderline, key=lambda x: x["folder"]):
        print(f"    BORDERLINE {b['folder']:40} {b['duration_min']:6.2f} min "
              f"(threshold {b['threshold_min']:g})")
    for s_, e in sorted(short.items()):
        print(f"    SHORT      {folders[s_]:40} {e['duration_min']:6.2f} min "
              f"(threshold {e['threshold_min']:g})")
    too_many = len(by_duration) > MAX_DURATION_EXCLUSION_FRACTION * n_total
    return by_duration, too_many


def main(apply: bool, force: bool) -> None:  # noqa: PLR0915 - the orchestrator
    """Plan, report, reconcile, and optionally write."""
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    root = find_gems_root()
    store = GemsStore(root)
    scope = read_scan_roots(root)
    book: ProtocolBook = load_protocol_book(root / PROTOCOL_FILENAME)
    spec = book.for_scan_root(scope.scan_roots[0])
    print(f"cohort: units={spec.units} config={spec.config} "
          f"order={spec.channel_order_source}")

    outcomes: list[Outcome] = []
    for scan_root in scope.resolve(root):
        outcomes += plan(scan_root, spec.config)
    ready = _report_plan(outcomes, stamp)

    dupfile = Path(DUPLICATES)
    duplicates: dict[str, dict[str, Any]] = {}
    if dupfile.is_file():
        duplicates = json.loads(dupfile.read_text(encoding="utf-8")).get("groups", {})
    to_write, deduped = resolve_collisions(ready, duplicates)

    # Pairing and exclusion, from the CORRECTED folder names (Andrea's ruling 2),
    # over distinct recordings only - a deduplicated copy is not a second partner.
    folders = {o.session: Path(o.source).parent.name for o in to_write}
    if spec.pair_gap_min_lo is None or spec.pair_gap_min_hi is None:
        msg = ("protocol.yaml declares no pair_gap_min_lo / pair_gap_min_hi; "
               "pairs are confirmed by acquisition time, never by names alone")
        raise SystemExit(msg)
    started = {}
    for o in to_write:
        when = acquired_at(Path(o.source))
        assert when is not None, f"{o.source}: keyed, so it has a .tsq start"
        started[o.session] = datetime.fromisoformat(when)
    short, borderline, no_infix = _duration_rule(to_write, spec)
    excl = plan_exclusions(folders, started, gap_lo_min=spec.pair_gap_min_lo,
                           gap_hi_min=spec.pair_gap_min_hi, short=short,
                           ruled=_ruled(folders))
    by_duration, too_many = _report_duration(excl, short, borderline, no_infix, folders,
                                             spec, len(to_write))
    _report_exclusions(excl, folders)
    renames = [
        {"session": o.session, "block_name": o.instrument_block,
         "folder_name": Path(o.source).parent.name}
        for o in outcomes if o.instrument_block
    ]
    print(f"\n  block/folder disagreements logged: {len(renames)} "
          "(folder is authoritative for meaning; block name is the key)")

    Path("new_cohort_meta_plan.json").write_text(
        json.dumps({"enumerated_at": stamp, "n_enumerated": len(outcomes),
                    "outcomes": [asdict(o) for o in outcomes],
                    "renames": renames,
                    "exclusions": excl.excluded,
                    "pairing_window": {"gap_lo_min": spec.pair_gap_min_lo,
                                       "gap_hi_min": spec.pair_gap_min_hi,
                                       "order": "baseline first"},
                    "pairing_ambiguous": excl.ambiguous,
                    "pairing_unconfirmed": excl.unconfirmed,
                    "pairing_missing": excl.missing,
                    "pairing_superseded": excl.superseded,
                    "duration_rule": {"short": short, "borderline": borderline,
                                      "no_infix": no_infix,
                                      "excluded_by_rule": by_duration}}, indent=1),
        encoding="utf-8", newline="\n",
    )

    n_skip = len(outcomes) - len(ready)
    n_excl = sum(1 for o in to_write if o.session in excl.excluded)
    if too_many and apply:
        msg = (f"the duration rule would exclude {len(by_duration)} of {len(to_write)} "
               f"recordings (> {MAX_DURATION_EXCLUSION_FRACTION:.0%}); stopping for Andrea "
               "to see the list first (new_cohort_meta_plan.json, duration_rule)")
        raise SystemExit(msg)
    if not apply:
        would = write(to_write, spec, store, force=force, exclusions=excl.excluded,
                      apply=False)
        kept = [s for s in would if s not in excl.excluded]
        _reconcile(stamp, len(outcomes),
                   sum(1 for s in kept if would[s] == "written"),
                   sum(1 for s in kept if would[s] == "unchanged"),
                   len(deduped), n_skip, n_excl)
        print(f"  excluded files that would change  "
              f"{sum(1 for s in would if s in excl.excluded and would[s] == 'written')}")
        n_retire, _ = retire_stem_keys(to_write, store, apply=False)
        print(f"  stem-keyed directories to retire after writing: {n_retire}")
        print("DRY RUN - pass --apply to write meta.json to the Drive")
        return
    state = write(to_write, spec, store, force=force, exclusions=excl.excluded)
    kept = [s for s in state if s not in excl.excluded]
    _reconcile(stamp, len(outcomes),
               sum(1 for s in kept if state[s] == "written"),
               sum(1 for s in kept if state[s] == "unchanged"),
               len(deduped), n_skip, n_excl)
    held = tuple(o for o in outcomes if o.status.startswith("held_"))
    retired, unexplained = retire_stem_keys(to_write, store, apply=True, held=held)
    print(f"  stem-keyed directories retired  {retired}")
    print(f"  stem-keyed directories left     {len(unexplained)}")
    if unexplained:
        msg = (f"{len(unexplained)} stem-keyed directories remain, each a second "
               "meta.json for a recording that now has a stamped one: "
               + ", ".join(unexplained[:10]))
        raise RuntimeError(msg)


if __name__ == "__main__":
    args = sys.argv[1:]
    main("--apply" in args, "--force" in args)
