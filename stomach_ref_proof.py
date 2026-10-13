"""Prove a mask folder's stomach_ref, then write a NEW provenance version carrying its recipe.

Andrea, 2026-10-12: Night 6 builds ``stomach_ref`` from the recipe the mask provenance
records. Masks emitted before ``stomach_ref_recipe`` existed (A gems_a_t03_1_1_bl_204153,
K gems_k_t01_cme1_bl_210313) do not carry one. This script, run by the coordinator's ops
agent and never by the store writer's code path, establishes which stomach_ref their
masks were made on and, only if that is proved, writes a new version of each mask folder
whose provenance carries the recipe and the proof. Per recording:

1. the mask folder that carries a ``stomach_ref`` mask (``blank_*_stomach_ref``) is found
   under ``--masks-root`` (default: the store, read only); its provenance must round-trip
   exactly through ``MaskProvenance`` and must not already carry a recipe;
2. the routing entry is taken from ``--routing`` and must hash (``emit.routing.entry_hash``)
   to the provenance's ``extra.routing_entry_hash`` - its ``stomach_ref`` field names the
   variant (``{}`` = the derivation as-is; a notched variant is refused here: its masks'
   reference values were computed on the un-notched derivation, so they cannot prove it);
3. this checkout's candidate-generation hash must be the provenance's ``generation_sha``;
4. the raw recording is loaded through ``meta.json``'s ``source_path`` as stage 1a loads it
   (``io.recording.load_recording``), the epoch is cut as stage 1a cuts it (the
   provenance's ``extra.region``, its stop clipped to the loaded length, one
   ``seconds_to_sample`` rounding) and must be the file's ``epochStartSample0`` /
   ``nSamples``; ``stomach_ref`` is ``build_derivations(epoch)["stomach_ref"]``, exactly
   what stage 1a read, and the recipe (``derive.stomach_recipe.stomach_ref_recipe``)
   applied to the epoch must equal it bit for bit;
5. every recorded ``stomach_ref|<band>`` reference value is recomputed as stage 1a did
   (``log_envelope(band_envelope_for(x, fs, band))`` -> ``epoch_reference``) and must
   equal the recorded median, scale, frame count and statistic EXACTLY (JSON floats are
   the full float64; the stored precision is all of it).

Any failure refuses that recording and writes only a refusal record. On success it writes,
under ``--out`` (refused if it lies inside the gems root or the masks root, and nothing is
ever overwritten), ``data/<animal>/<session>/masks/<folder>/`` with every file of the
original folder: each ``e*_masks.mat`` with every variable identical except
``provenance_json``, which gains ``stomach_ref_recipe`` and
``extra.stomach_ref_recipe_proof``; ``e*_provenance.json`` and ``provenance.json`` with
the same additions in their own format; every other file byte-identical. The written
files are re-read and checked before the script reports success. A manifest
(``stomach_ref_proof_<session>_<run stamp>.json``, a new name every run) records expected vs
recomputed values and every file's sha256, and ``real_case_<session>.json`` is what
``tests/test_night6_stomach_ref.py`` reads (``GEMS_STOMACHREF_REAL``) to hold the MATLAB
build to Python's on this recording.

**The proven recipe** - the recipe and every epoch's proof - is stored once per recording
at ``data/<animal>/<session>/stomach_ref_recipe_proven.json`` under ``--out``
(``derive.stomach_recipe.proven_recipe_path``; the store writer places it beside the
session's ``meta.json``). A finalise driver, which loads no recording, reads it with
``derive.stomach_recipe.proven_stomach_ref_recipe``, which re-checks the proof.

**Nothing is overwritten on a re-run into the same ``--out``** (review 2026-10-12): the
report's name carries the run's UTC stamp, and a recording whose ``real_case`` file or new
mask folder already exists is refused BEFORE anything is loaded (:func:`claim_outputs`).
Every file is created exclusively (:func:`write_once`), so even a race cannot replace one.

Usage::

    python -u stomach_ref_proof.py --routing <production_routing.json> --out <dir>
        [--masks-root <dir>] [--gems-root <dir>] [--recording ANIMAL:SESSION ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from gems_blanking_v2.bands.envelope import band_envelope_for, log_envelope
from gems_blanking_v2.bands.reference import epoch_reference
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.derive.stomach_recipe import (
    STOMACH_REF,
    apply_stomach_ref_recipe,
    proven_recipe_path,
    proven_recipe_record,
    stomach_ref_recipe,
)
from gems_blanking_v2.detect.chain import generation_sha256
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.emit.routing import entry_hash, table_hash
from gems_blanking_v2.extent.grid import seconds_to_sample
from gems_blanking_v2.io.channel_map import meta_path
from gems_blanking_v2.io.recording import load_recording
from gems_blanking_v2.io.store import GemsStore, atomic_write_text, find_gems_root
from gems_blanking_v2.types import Recording
from scipy.io import loadmat, savemat

DEFAULT_RECORDINGS: Final[tuple[str, ...]] = (
    "A:gems_a_t03_1_1_bl_204153_20260831T004157Z",
    "K:gems_k_t01_cme1_bl_210313_20260825T010319Z",
)
PROOF_KEY: Final = "stomach_ref_recipe_proof"
FS_TOL_HZ: Final = 1e-9
"""The mask file's fs and the recording's must agree to this (as night6_prepare_epoch)."""
RULE: Final = ("Andrea 2026-10-12: Night 6 builds stomach_ref from the recipe the mask "
               "provenance records; the masks are never moved onto a nerve signal")


class ProofRefusedError(RuntimeError):
    """The recording's stomach_ref could not be proved; nothing is written for it.

    ``detail`` carries what was established before the refusal (expected vs recomputed
    reference values, when it got that far), for the refusal record.
    """

    def __init__(self, msg: str, detail: Mapping[str, Any] | None = None) -> None:
        """Keep the message and the detail."""
        super().__init__(msg)
        self.detail = dict(detail or {})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pretty(doc: Mapping[str, Any]) -> str:
    """Return the mask folder's JSON sidecar text (night4: indent 1, sorted, ASCII, LF)."""
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def _code() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True,
                              check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "gems_blanking_v2",
                              Path(__file__).name)),
            "script_sha256": _sha256(Path(__file__).resolve())}


MAX_PATH: Final = 259
"""Cross-platform rule 5: the longest path this script will create (Windows MAX_PATH 260)."""


def check_path_length(path: Path) -> None:
    """Refuse, by name, a path a Windows machine without long paths could not create."""
    n = len(str(path))
    if n > MAX_PATH:
        msg = (f"output path would be {n} characters (> {MAX_PATH}, Windows MAX_PATH; "
               f"cross-platform rule 5): {path}; choose a shorter --out")
        raise ProofRefusedError(msg)


def write_once(path: Path, text: str) -> None:
    """Create ``path`` with ``text`` (UTF-8, LF), atomically; refuse if it exists.

    The final name is claimed exclusively first (an empty file, ``open('x')``), the text
    is written to a temporary file beside it, and ``Path.replace`` (``os.replace``) moves it
    onto the claimed name - so the name is never taken from another writer, and a reader
    never sees a
    partial file (an interrupted write leaves the claim empty, which no JSON reader
    accepts). Never overwrites.
    """
    check_path_length(path.with_name(f".{path.name}.tmp"))
    try:
        with path.open("x", encoding="utf-8"):
            pass
    except FileExistsError as exc:
        msg = f"{path} exists; nothing is ever overwritten"
        raise ProofRefusedError(msg) from exc
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    tmp.replace(path)          # os.replace: atomic on both platforms


def claim_outputs(out: Path, animal: str, session: str, stamp: str
                  ) -> tuple[Path, Path, Path]:
    """Return ``(report, real_case, proven recipe)`` paths for one recording, or refuse.

    The report is named for the run (``stamp``), so a re-run writes a new one; the
    ``real_case`` file and the proven recipe (``derive.stomach_recipe.proven_recipe_path``)
    name no run, so an existing one refuses the recording before any work. Raises
    :class:`ProofRefusedError` - the report path is then still usable.
    """
    report = out / f"stomach_ref_proof_{session}_{stamp}.json"
    case = out / f"real_case_{session}.json"
    proven = proven_recipe_path(out, animal, session)
    for p in (report, case, proven):
        check_path_length(p.with_name(f".{p.name}.tmp"))
    if report.exists():
        msg = f"{report} exists; nothing is ever overwritten"
        raise ProofRefusedError(msg)
    for p in (case, proven):
        if p.exists():
            msg = (f"{p} exists from an earlier run into this --out; refused before any "
                   "work (choose a new --out, or move the earlier run aside - nothing is "
                   "deleted)")
            raise ProofRefusedError(msg)
    return report, case, proven


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _mask_folder(masks_root: Path, animal: str, session: str) -> Path:
    base = masks_root / "data" / animal / session / "masks"
    hits = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for f in sorted(d.glob("e*_masks.mat")):
            if any(k.startswith("blank_") and k.endswith(f"_{STOMACH_REF}")
                   for k in _mat_names(f)):
                hits.append(d)
                break
    if len(hits) != 1:
        msg = (f"{session}: {len(hits)} mask folders under {base} carry a {STOMACH_REF} mask "
               f"({[h.name for h in hits]}); expected exactly one")
        raise ProofRefusedError(msg)
    return hits[0]


def _mat_names(path: Path) -> list[str]:
    from scipy.io import whosmat  # noqa: PLC0415
    return [w[0] for w in whosmat(path)]


def _load_mat(path: Path) -> dict[str, Any]:
    return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}


def _text(v: Any) -> str:  # noqa: ANN401 - a loadmat char array
    return str(np.asarray(v).ravel()[0])


def _refs(x: np.ndarray, fs: float, recorded: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every recorded stomach_ref|band reference exactly as stage 1a did."""
    out: dict[str, Any] = {}
    for key in sorted(recorded):
        signal, band = key.split("|", 1)
        if signal != STOMACH_REF:
            continue
        le = log_envelope(band_envelope_for(x, fs, band))
        ref = epoch_reference(le, signal=signal, band=band)
        out[key] = {"median": float(ref.median), "scale": float(ref.scale),
                    "n_frames": int(ref.n_frames), "statistic": "median_of_log"}
    if not out:
        msg = f"the provenance records no {STOMACH_REF} reference value to prove against"
        raise ProofRefusedError(msg)
    return out


def _compare(expected: Mapping[str, Any], got: Mapping[str, Any]) -> list[str]:
    bad = []
    for key, g in got.items():
        e = expected[key]
        for field in ("median", "scale", "n_frames", "statistic"):
            if e.get(field) != g[field]:
                bad.append(f"{key} {field}: recorded {e.get(field)!r}, recomputed {g[field]!r}")
    return bad


def _open_epoch(mat: Path, entry: Mapping[str, Any]
                ) -> tuple[dict[str, Any], MaskProvenance, dict[str, Any], str, str]:
    """Check the epoch file's provenance, routing entry and code identity.

    Returns ``(mat variables, provenance, routing stomach_ref variant, entry hash, code
    generation hash)``.
    """
    m = _load_mat(mat)
    text = _text(m["provenance_json"])
    prov = MaskProvenance.from_json(text)
    if prov.to_json() != text:
        msg = f"{mat.name}: its provenance does not round-trip exactly through MaskProvenance"
        raise ProofRefusedError(msg)
    if prov.stomach_ref_recipe:
        msg = f"{mat.name}: the provenance already carries a stomach_ref_recipe"
        raise ProofRefusedError(msg)
    want_entry = prov.extra.get("routing_entry_hash")
    got_entry = entry_hash(entry)
    if got_entry != want_entry:
        msg = (f"{mat.name}: the routing entry hashes to {got_entry}, the provenance records "
               f"{want_entry}: not the entry the masks were made under")
        raise ProofRefusedError(msg)
    variant = entry.get("stomach_ref")
    if variant is None:
        msg = f"{mat.name}: the routing entry has no stomach_ref field"
        raise ProofRefusedError(msg)
    if variant:
        msg = (f"{mat.name}: the routing entry names stomach_ref {variant}; a notched "
               "recording's reference values were computed on the un-notched derivation, so "
               "they cannot prove the notched signal - refused, not attempted")
        raise ProofRefusedError(msg)
    gen = generation_sha256()[:16]
    if gen != prov.generation_sha:
        msg = (f"{mat.name}: this checkout's generation hash is {gen}, the masks' "
               f"{prov.generation_sha}: the recompute would not be the code that made them")
        raise ProofRefusedError(msg)
    return m, prov, dict(variant), got_entry, gen


def prove_epoch(rec: Recording, mat: Path, entry: Mapping[str, Any], routing_file: Path
                ) -> tuple[dict[str, Any], dict[str, Any], MaskProvenance]:
    """Prove one epoch file; return ``(recipe, proof, provenance)``, or refuse."""
    m, prov, variant, got_entry, gen = _open_epoch(mat, entry)
    fs = float(rec.fs)
    if abs(float(np.asarray(m["fs"]).ravel()[0]) - fs) > FS_TOL_HZ:
        msg = f"{mat.name}: mask fs {m['fs']} is not the recording's {fs}"
        raise ProofRefusedError(msg)
    lo, hi_planned = (float(v) for v in prov.extra["region"])
    hi = min(hi_planned, rec.n_samples / fs)            # stage 1a: the epoch is what loaded
    i0, i1 = seconds_to_sample(lo, fs), seconds_to_sample(hi, fs)
    start0 = int(np.asarray(m["epochStartSample0"]).ravel()[0])
    n = int(np.asarray(m["nSamples"]).ravel()[0])
    if (i0, i1 - i0) != (start0, n):
        msg = (f"{mat.name}: the region {prov.extra['region']} cuts samples [{i0}, {i1}) but "
               f"the file is {n} samples from {start0}")
        raise ProofRefusedError(msg)
    sub = dataclasses.replace(rec, data=rec.data[i0:i1])
    derived, _w = build_derivations(sub)
    x = np.asarray(derived[STOMACH_REF], dtype=np.float64)   # stage 1a's _hr_samples
    recipe = stomach_ref_recipe(sub, variant)               # verified on sub, bit for bit
    cols = {c.name: sub.data[:, c.index] for c in sub.channels if c.role == "stomach"}
    if not np.array_equal(apply_stomach_ref_recipe(recipe, cols, fs), x):
        msg = f"{mat.name}: the recipe applied to the epoch is not stage 1a's stomach_ref"
        raise ProofRefusedError(msg)
    expected = {k: v for k, v in prov.reference_values.items()
                if k.split("|", 1)[0] == STOMACH_REF}
    recomputed = _refs(x, fs, prov.reference_values)
    bad = _compare(expected, recomputed)
    proof = {
        "rule": RULE, "match": not bad, "mismatches": bad,
        "expected": expected, "recomputed": recomputed,
        "routing_entry_hash": {"provenance": prov.extra.get("routing_entry_hash"),
                               "recomputed": got_entry, "table_file": routing_file.name,
                               "table_file_sha256": _sha256(routing_file)},
        "routing_stomach_ref": variant,
        "generation_sha": {"provenance": prov.generation_sha, "code": gen},
        "epoch": {"region": [lo, hi], "epoch_start_sample0": i0, "n_samples": n},
        "signal": ("build_derivations(epoch)['stomach_ref'] - what stage 1a read "
                   "(night4_stage1a._hr_samples); the recipe applied to the epoch equals it "
                   "bit for bit"),
        "reference": ("log_envelope(band_envelope_for(x, fs, band)) -> epoch_reference, as "
                      "stage 1a for a signal outside the features' common set"),
        "proved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": _code(),
    }
    if bad:
        raise ProofRefusedError(f"{mat.name}: " + "; ".join(bad), proof)
    return recipe, proof, prov


Proved = tuple[dict[str, Any], dict[str, Any], MaskProvenance]


def _with_recipe(prov: MaskProvenance, recipe: Mapping[str, Any], proof: Mapping[str, Any]
                 ) -> MaskProvenance:
    return dataclasses.replace(prov, stomach_ref_recipe=recipe,
                               extra={**prov.extra, PROOF_KEY: proof})


def _new_mat(f: Path, out: Path, proved: Proved) -> str:
    recipe, proof, prov = proved
    doc = _load_mat(f)
    for k, v in doc.items():
        if np.asarray(v).dtype.kind == "U" and not _text(v).isascii():
            msg = f"{f.name}: {k} is not ASCII; it would not re-save faithfully"
            raise ProofRefusedError(msg)
    doc["provenance_json"] = _with_recipe(prov, recipe, proof).to_json()
    with out.open("wb") as fh:
        savemat(fh, doc, do_compression=True)
    _check_mat(f, out, prov, recipe, proof)
    return f"provenance_json gains stomach_ref_recipe and extra.{PROOF_KEY}"


def _new_epoch_json(f: Path, out: Path, proved: Proved) -> str:
    recipe, proof, prov = proved
    old_text = f.read_bytes().decode("utf-8")
    old = json.loads(old_text)
    if _pretty(old) != old_text or old != json.loads(prov.to_json()):
        msg = f"{f.name}: not the pretty copy of the epoch's provenance_json"
        raise ProofRefusedError(msg)
    atomic_write_text(out, _pretty(json.loads(_with_recipe(prov, recipe, proof).to_json())))
    return "the same additions as its masks file"


def _new_folder_json(f: Path, out: Path, stems: Mapping[str, Proved]) -> str:
    old_text = f.read_bytes().decode("utf-8")
    old = json.loads(old_text)
    if _pretty(old) != old_text:
        msg = f"{f.name}: its format is not the one this script reproduces"
        raise ProofRefusedError(msg)
    add = {"stomach_ref_recipe": {s: v[0] for s, v in sorted(stems.items())},
           PROOF_KEY: {s: v[1] for s, v in sorted(stems.items())}}
    clash = sorted(set(add) & set(old))
    if clash:
        msg = f"{f.name} already has {clash}"
        raise ProofRefusedError(msg)
    atomic_write_text(out, _pretty({**old, **add}))
    return f"gains stomach_ref_recipe and {PROOF_KEY} per epoch"


def _copy(f: Path, out: Path) -> str:
    out.write_bytes(f.read_bytes())
    if out.read_bytes() != f.read_bytes():
        msg = f"{f.name}: the copy is not byte-identical"
        raise ProofRefusedError(msg)
    return "byte-identical copy"


def _write_new_version(src: Path, dst: Path, proved: Mapping[str, Proved]) -> dict[str, Any]:
    """Write the folder's new version; re-read and check it; return the file manifest."""
    if dst.exists():
        msg = f"{dst} exists; nothing is ever overwritten"
        raise ProofRefusedError(msg)
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        msg = f"{tmp} exists (an earlier attempt); move it aside first - nothing is deleted"
        raise ProofRefusedError(msg)
    tmp.mkdir(parents=True)
    files: dict[str, Any] = {}
    stems = {name.removesuffix("_masks.mat"): v for name, v in proved.items()}
    for f in sorted(p for p in src.iterdir() if p.is_file()):
        out = tmp / f.name
        stem = f.name.split("_", 1)[0]
        if f.name.endswith("_masks.mat") and stem in stems:
            change = _new_mat(f, out, stems[stem])
        elif f.name.endswith("_provenance.json") and stem in stems:
            change = _new_epoch_json(f, out, stems[stem])
        elif f.name == "provenance.json":
            change = _new_folder_json(f, out, stems)
        else:
            change = _copy(f, out)
        files[f.name] = {"source_sha256": _sha256(f), "new_sha256": _sha256(out),
                         "change": change}
    tmp.replace(dst)
    return files


def _check_mat(old: Path, new: Path, prov: MaskProvenance, recipe: Mapping[str, Any],
               proof: Mapping[str, Any]) -> None:
    a, b = _load_mat(old), _load_mat(new)
    if sorted(a) != sorted(b):
        msg = f"{new.name}: variables {sorted(set(a) ^ set(b))} differ"
        raise ProofRefusedError(msg)
    for k in a:
        if k == "provenance_json":
            continue
        va, vb = np.asarray(a[k]), np.asarray(b[k])
        if va.dtype != vb.dtype or va.shape != vb.shape or not np.array_equal(va, vb):
            msg = f"{new.name}: {k} is not identical to the original"
            raise ProofRefusedError(msg)
    got = json.loads(_text(b["provenance_json"]))
    want = json.loads(prov.to_json())
    want["stomach_ref_recipe"] = json.loads(json.dumps(dict(recipe)))
    want["extra"] = {**want.get("extra", {}), PROOF_KEY: json.loads(json.dumps(dict(proof)))}
    if got != want:
        msg = f"{new.name}: provenance_json is not the original plus the recipe and proof"
        raise ProofRefusedError(msg)


def _prove_recording(st: GemsStore, masks_root: Path, out: Path, table: Mapping[str, Any],
                     routing_file: Path, animal: str, session: str, case_path: Path,
                     proven_path: Path) -> dict[str, Any]:
    """Prove one recording and write its new version; return the report, or refuse."""
    report: dict[str, Any] = {}
    entry = table["entries"].get(session)
    if entry is None:
        msg = f"{session}: no entry in {routing_file}"
        raise ProofRefusedError(msg)
    if table["entry_hashes"].get(session) != entry_hash(entry):
        msg = f"{session}: the table's entry_hashes disagree with the entry"
        raise ProofRefusedError(msg)
    folder = _mask_folder(masks_root, animal, session)
    report["mask_folder"] = folder.relative_to(masks_root).as_posix()
    dst = out / folder.relative_to(masks_root)
    longest = max((f.name for f in folder.iterdir()), key=len)
    check_path_length(dst.with_name(dst.name + ".partial") / f".{longest}.tmp")
    if dst.exists() or dst.with_name(dst.name + ".partial").exists():
        msg = f"{dst} (or its .partial) exists; refused before any work - nothing is overwritten"
        raise ProofRefusedError(msg)
    meta = json.loads(meta_path(st, animal, session).read_text(encoding="utf-8"))
    report["source_path"] = meta["source_path"]
    print(f"[proof] {session}: loading {meta['source_path']}", flush=True)
    rec = load_recording(st.root / meta["source_path"], animal, store=st).recording
    proved = {mat.name: prove_epoch(rec, mat, entry, routing_file)
              for mat in sorted(folder.glob("e*_masks.mat"))}
    report["epochs"] = {k: v[1] for k, v in proved.items()}
    recipes = [v[0] for v in proved.values()]
    if any(r != recipes[0] for r in recipes):
        msg = f"{session}: its epochs prove different recipes; none is stored"
        raise ProofRefusedError(msg)
    record = proven_recipe_record(session, recipes[0], {k: v[1] for k, v in proved.items()})
    report["files"] = _write_new_version(folder, dst, proved)
    proven_path.parent.mkdir(parents=True, exist_ok=True)
    write_once(proven_path, _pretty(record))
    report["proven_recipe"] = proven_path.relative_to(out).as_posix()
    report["new_folder"] = dst.as_posix()
    first = sorted(proved)[0]
    case = {"gems_root": st.root.as_posix(), "animal": animal, "session": session,
            "source_path": meta["source_path"],
            "meta_file": meta_path(st, animal, session).as_posix(),
            "old_mask": (folder / first).as_posix(), "new_mask": (dst / first).as_posix()}
    write_once(case_path, _pretty(case))
    report["real_case"] = case_path.name
    return report


def _print(report: Mapping[str, Any]) -> None:
    session = report["recording"]
    print(f"[proof] {session}: {report['result']}"
          + (f" - {report['reason']}" if "reason" in report else ""), flush=True)
    epochs = dict(report.get("epochs") or {})
    if report.get("refused_detail"):
        epochs["(refused)"] = report["refused_detail"]
    for ep, proof in epochs.items():
        for key, e in (proof.get("expected") or {}).items():
            g = proof["recomputed"][key]
            print(f"[proof]   {ep} {key}: recorded median {e['median']!r} scale {e['scale']!r} "
                  f"n {e['n_frames']} | recomputed median {g['median']!r} scale "
                  f"{g['scale']!r} n {g['n_frames']}", flush=True)


def run(args: argparse.Namespace) -> int:
    """Prove each recording; 0 when all were proved and written, 1 when any was refused."""
    root = find_gems_root(Path(args.gems_root) if args.gems_root else None)
    st = GemsStore(root)
    masks_root = Path(args.masks_root) if args.masks_root else root
    out = Path(args.out)
    for guarded in (root, masks_root):
        if _inside(out, guarded):
            print(f"refused: --out {out} lies inside {guarded}; this script never writes there")
            return 2
    routing_file = Path(args.routing)
    table = json.loads(routing_file.read_text(encoding="utf-8"))
    if table_hash(table) != table.get("hash"):
        print(f"refused: {routing_file} does not hash to its own 'hash' field")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    status = 0
    for item in args.recording or DEFAULT_RECORDINGS:
        animal, session = item.split(":", 1)
        report: dict[str, Any] = {"recording": session, "animal": animal, "run": stamp}
        report_path = out / f"stomach_ref_proof_{session}_{stamp}.json"
        t0 = time.perf_counter()
        try:
            report_path, case_path, proven_path = claim_outputs(out, animal, session, stamp)
            report.update(_prove_recording(st, masks_root, out, table, routing_file,
                                           animal, session, case_path, proven_path))
            report["result"] = "proved; new provenance version written"
        except ProofRefusedError as exc:
            report.update(result="REFUSED", reason=str(exc), refused_detail=exc.detail)
            status = 1
        report["seconds"] = round(time.perf_counter() - t0, 1)
        try:
            write_once(report_path, _pretty(report))
        except ProofRefusedError as exc:      # the stamped name is taken: say so, write nothing
            print(f"[proof] {session}: report not written - {exc}", flush=True)
            status = 1
        _print(report)
    return status


def main() -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="Prove a mask folder's stomach_ref and write a "
                                             "new provenance version carrying its recipe.")
    ap.add_argument("--routing", required=True, help="production_routing.json (schema 2)")
    ap.add_argument("--out", required=True, help="output root; never inside the store")
    ap.add_argument("--masks-root", help="root holding data/<animal>/<session>/masks (default: "
                                         "the gems root, read only)")
    ap.add_argument("--gems-root", help="the gems root (default: discovered)")
    ap.add_argument("--recording", action="append", help="ANIMAL:SESSION (default: A and K)")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
