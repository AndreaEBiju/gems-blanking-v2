"""stomach_ref_proof.py never overwrites what an earlier run into the same --out wrote.

Review 2026-10-12: a re-run must not replace the per-recording report or the real_case JSON.
The report is named for the run (UTC stamp); an existing real_case file or new mask folder
refuses the recording before any work; every file is created exclusively. The proof itself
needs the shared drive and is exercised by the dry run, not here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import stomach_ref_proof as proof
from gems_blanking_v2.derive import stomach_recipe as sr
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.detect.chain import generation_sha256
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit.handoff import write_mask_file
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.emit.routing import entry_hash, table_hash
from gems_blanking_v2.extent.grid import n_grid_frames

from tests.conftest import make_multichannel

SESSION = "syn_a_t01_bl_100000_20260101T150000Z"


def test_write_once_refuses_an_existing_file_and_leaves_it_alone(tmp_path: Path) -> None:
    f = tmp_path / "r.json"
    proof.write_once(f, "first\n")
    with pytest.raises(proof.ProofRefusedError, match="nothing is ever overwritten"):
        proof.write_once(f, "second\n")
    assert f.read_bytes() == b"first\n"


def test_claim_outputs_names_the_report_for_the_run_and_refuses_an_earlier_case(
        tmp_path: Path) -> None:
    report, case, proven = proof.claim_outputs(tmp_path, "A", SESSION, "20261012T200000Z")
    assert report.name == f"stomach_ref_proof_{SESSION}_20261012T200000Z.json"
    assert case.name == f"real_case_{SESSION}.json"
    assert proven == tmp_path / "data" / "A" / SESSION / "stomach_ref_recipe_proven.json"
    for earlier in (case, proven):
        earlier.parent.mkdir(parents=True, exist_ok=True)
        earlier.write_text("{}\n", encoding="utf-8")
        with pytest.raises(proof.ProofRefusedError, match="refused before any work"):
            proof.claim_outputs(tmp_path, "A", SESSION, "20261012T200001Z")
        earlier.rename(earlier.with_name(earlier.name + ".moved"))


def _table(tmp_path: Path) -> Path:
    entry = {"spike": {}, "hr": {"none": "synthetic"}, "stomach_ref": {}}
    t: dict[str, Any] = {"schema": 2, "entries": {SESSION: entry},
                         "entry_hashes": {SESSION: entry_hash(entry)}}
    t["hash"] = table_hash(t)
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(t), encoding="utf-8")
    return path


def test_a_rerun_into_the_same_out_overwrites_nothing(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    root, out = tmp_path / "root", tmp_path / "out"
    root.mkdir()
    calls: list[str] = []

    def fake_prove(st: Any, masks_root: Path, out_: Path, table: Any, routing_file: Path,  # noqa: ANN401
                   animal: str, session: str, case_path: Path, proven: Path) -> dict[str, Any]:
        calls.append(session)
        proof.write_once(case_path, f"case written by call {len(calls)}\n")
        return {"note": f"call {len(calls)}"}

    monkeypatch.setattr(proof, "find_gems_root", lambda *_a, **_k: root)
    monkeypatch.setattr(proof, "GemsStore", lambda r: SimpleNamespace(root=r))
    monkeypatch.setattr(proof, "_prove_recording", fake_prove)
    args = argparse.Namespace(gems_root=None, masks_root=None, out=str(out),
                              routing=str(_table(tmp_path)), recording=[f"A:{SESSION}"])
    assert proof.run(args) == 0
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert len(first) == 2 and f"real_case_{SESSION}.json" in first
    assert proof.run(args) == 1                            # refused: the case exists
    assert calls == [SESSION]                              # refused before any work
    for name, data in first.items():
        assert (out / name).read_bytes() == data           # nothing overwritten
    reports = sorted(p for p in out.iterdir() if p.name.startswith("stomach_ref_proof_"))
    if len(reports) == 2:                                  # a second stamp: its refusal
        assert json.loads(reports[-1].read_text(encoding="utf-8"))["result"] == "REFUSED"


# ---------------------------------------------------------------------------
# proved -> stored -> read at finalise (derive.stomach_recipe.proven_stomach_ref_recipe)
# ---------------------------------------------------------------------------

FS = 24414.0625
ENTRY: dict[str, Any] = {"spike": {}, "hr": {"none": "synthetic"}, "stomach_ref": {}}
READS: dict[str, tuple[str, ...]] = {"spikes": (), "slow_wave": (), "mmc": ("ANT1",),
                                     "hrv": (), "breathing": (), "velocity": ()}
FINALISE_READS = {**READS, "hrv": ("stomach_ref",), "breathing": ("stomach_ref",)}


@pytest.fixture(scope="module")
def proved(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Prove a pre-recipe mask file of a synthetic recording with the script's own code."""
    root = tmp_path_factory.mktemp("proved")
    rec, _t = make_multichannel(FS, 30.0, seed=4)
    n = rec.n_samples
    x = build_derivations(rec)[0]["stomach_ref"]
    refs = proof._refs(np.asarray(x), FS, {"stomach_ref|0.5-3": {}})
    gen = generation_sha256()[:16]
    prov = MaskProvenance(
        model={"mode": "pooled", "version": "t", "corpus_hash": "0" * 32,
               "calibrator": "models/t/cal.json"},
        thresholds={"source": "synthetic"}, reference_values=refs, code_commit="test",
        generation_sha=gen, routing_hash="test", created_at="2026-10-12T05:00:00+00:00",
        recording=rec.session, extra={"region": [0.0, n / FS],
                                      "routing_entry_hash": entry_hash(ENTRY)})
    mat = root / "e0_masks.mat"
    masks = mk.build_masks(READS, [], n_frames=n_grid_frames(n, FS), t0_s=0.0)
    write_mask_file(mat, masks, prov, signals=READS, fs=FS, n_samples=n, epoch_start_s=0.0,
                    epoch_start_sample=0, min_retention=0.5, animal_median={},
                    line_distrust=None, peri_r=None,
                    not_computed=dict.fromkeys(("slow_wave", "hrv", "breathing"), "synthetic"),
                    elsewhere={"spikes": "synthetic: per animal"})
    recipe, pr, _prov = proof.prove_epoch(rec, mat, ENTRY, _table(root))
    record = sr.proven_recipe_record(rec.session, recipe, {"e0_masks.mat": pr})
    return {"root": root, "session": rec.session, "recipe": recipe, "record": record,
            "gen": gen, "rec": rec}


def _store(root: Path, session: str, record: dict[str, Any]) -> None:
    path = sr.proven_recipe_path(root, "A", session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _read(p: dict[str, Any], root: Path, **kw: Any) -> dict[str, Any]:  # noqa: ANN401
    args: dict[str, Any] = {"signals": FINALISE_READS, "routing_entry": ENTRY,
                            "generation_sha": p["gen"], **kw}
    return sr.proven_stomach_ref_recipe(root, "A", p["session"], **args)


def test_the_proof_reproduces_the_recorded_reference_exactly(proved: dict[str, Any]) -> None:
    pr = proved["record"]["proofs"]["e0_masks.mat"]
    assert pr["match"] and pr["expected"] == pr["recomputed"]
    assert proved["recipe"] == sr.stomach_ref_recipe(proved["rec"], {})


def test_finalise_reads_the_stored_proven_recipe(proved: dict[str, Any], tmp_path: Path) -> None:
    _store(tmp_path, proved["session"], proved["record"])
    assert _read(proved, tmp_path) == proved["recipe"]


@pytest.mark.parametrize(("tamper", "match"), [
    (lambda r: r["proofs"]["e0_masks.mat"]["recomputed"]["stomach_ref|0.5-3"].update(
        median=1.0), "median: recorded"),
    (lambda r: r["proofs"]["e0_masks.mat"]["recomputed"]["stomach_ref|0.5-3"].update(
        n_frames=1), "n_frames: recorded"),
    (lambda r: r["proofs"]["e0_masks.mat"].update(match=False), "does not say match"),
    (lambda r: r["proofs"]["e0_masks.mat"]["routing_entry_hash"].update(recomputed="0" * 64),
     "routing entry hash"),
    (lambda r: r["proofs"]["e0_masks.mat"]["generation_sha"].update(code="ffff"),
     "generation sha"),
    (lambda r: r["proofs"].clear(), "carries no proof"),
    (lambda r: r["recipe"].pop("weights"), "lacks weights"),
    (lambda r: r.update(recording="other"), "is for 'other'"),
    (lambda r: r.update(schema="stomach_ref_recipe_proven/0"), "schema"),
])
def test_a_proven_recipe_whose_proof_does_not_check_is_refused_by_name(
        proved: dict[str, Any], tmp_path: Path, tamper: Any, match: str) -> None:  # noqa: ANN401
    record = json.loads(json.dumps(proved["record"]))
    tamper(record)
    _store(tmp_path, proved["session"], record)
    with pytest.raises(sr.StomachRecipeError, match=match):
        _read(proved, tmp_path)


def test_finalise_refuses_another_routing_entry_or_generation(proved: dict[str, Any],
                                                               tmp_path: Path) -> None:
    _store(tmp_path, proved["session"], proved["record"])
    with pytest.raises(sr.StomachRecipeError, match="this finalise runs under"):
        _read(proved, tmp_path, routing_entry={**ENTRY, "hr": {"none": "other"}})
    with pytest.raises(sr.StomachRecipeError, match="this finalise runs"):
        _read(proved, tmp_path, generation_sha="0000000000000000")
    with pytest.raises(sr.StomachRecipeError, match="no stomach_ref field"):
        _read(proved, tmp_path, routing_entry={k: v for k, v in ENTRY.items()
                                               if k != "stomach_ref"})


def test_an_absent_proven_recipe_is_refused_by_name(proved: dict[str, Any],
                                                    tmp_path: Path) -> None:
    with pytest.raises(sr.StomachRecipeError, match="no proven recipe"):
        _read(proved, tmp_path)
