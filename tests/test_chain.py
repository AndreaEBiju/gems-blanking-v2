"""The detection chain: one construction site, and a hash scoped to exactly it."""

from __future__ import annotations

import ast
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gems_blanking_v2.derive.contact_quality import assess_contacts, screened_signals
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.detect import candidates as cand_mod
from gems_blanking_v2.detect import chain
from gems_blanking_v2.detect.candidates import candidate_report
from gems_blanking_v2.io import detector_core
from gems_blanking_v2.physio.rpeaks import detect_rpeaks
from gems_blanking_v2.types import Recording

from tests.conftest import make_cuff_contacts

FS = 24414.0625
PACKAGE = Path(chain.__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rec() -> Recording:
    return make_cuff_contacts(FS, 30.0, {"L3": "uncorrelated"}, seed=21)


def test_the_chain_is_exactly_its_parts_composed_by_hand(rec: Recording) -> None:
    """The move out of the audit bridge changed nothing: same parts, same order."""
    region = (4.0, 28.0)
    got = chain.detect_region(rec, region)

    sub = replace(rec, data=rec.data[round(4.0 * FS):round(28.0 * FS)])
    signals, _ = build_derivations(sub)
    drop = screened_signals(assess_contacts(sub))
    names = sorted(n for n in signals if n not in drop)
    z = chain.z_by_pair(np.column_stack([signals[n] for n in names]), FS, names)
    report = candidate_report(z, detect_rpeaks(signals["R_T"], FS))
    want = np.array([[c.start_s + 4.0, c.stop_s + 4.0] for c in report.candidates]).reshape(-1, 2)

    assert got.signals == tuple(names) and got.screened == drop
    np.testing.assert_array_equal(got.intervals, want)
    assert set(got.z) == set(z)
    assert all(np.array_equal(got.z[k], z[k], equal_nan=True) for k in z)


def test_a_screened_contact_leaves_the_max_with_its_cuffs_tripole(rec: Recording) -> None:
    got = chain.detect_region(rec, (0.0, 30.0))

    assert got.screened == {"L_V3", "L_T"}
    assert "L_V3" not in got.signals and "L_T" not in got.signals
    assert {"L_V1", "L_V2", "R_T", "R_V1", "R_V2", "R_V3"} <= set(got.signals)
    assert not any(sig in got.screened for sig, _band in got.z)


def test_z_enter_is_the_generators_own_unless_given(
    rec: Recording, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never restated here (invariant 39): no z_enter reaches candidate_report by default."""
    seen: list[dict] = []
    real = cand_mod.candidate_report

    def spy(z: dict, beats: object, **kw: object) -> object:
        seen.append(kw)
        return real(z, beats, **kw)

    monkeypatch.setattr(cand_mod, "candidate_report", spy)
    chain.detect_region(rec, (0.0, 30.0))
    chain.detect_region(rec, (0.0, 30.0), z_enter=2.5)

    assert seen == [{}, {"z_enter": 2.5}]


def test_the_generation_scope_is_the_chain_and_its_load_path_and_nothing_else() -> None:
    mods = set(chain.generation_modules())

    assert {"gems_blanking_v2.detect.chain", "gems_blanking_v2.derive.derivations",
            "gems_blanking_v2.derive.contact_quality", "gems_blanking_v2.bands.envelope",
            "gems_blanking_v2.bands.reference", "gems_blanking_v2.bands.zscore",
            "gems_blanking_v2.detect.candidates", "gems_blanking_v2.physio.rpeaks",
            "gems_blanking_v2.constants", "gems_blanking_v2.types",
            # the load path (widened 2026-09-28): file -> the array detect_region gets
            "gems_blanking_v2.io.recording", "gems_blanking_v2.io.channel_map",
            "gems_blanking_v2.io.chanlabels", "gems_blanking_v2.io.stim_split"} <= mods
    assert not mods & {"gems_blanking_v2.detect.recall", "gems_blanking_v2.io.audit_pool",
                       "gems_blanking_v2.cli", "gems_blanking_v2.io.scan"}


def test_the_external_list_is_every_detector_module_the_loader_imports() -> None:
    """Held equal by a test, since the import closure cannot see outside the package."""
    tree = ast.parse((PACKAGE / "io" / "recording.py").read_bytes())
    called = {node.args[0].value for node in ast.walk(tree)
              if isinstance(node, ast.Call) and getattr(node.func, "id", "") ==
              "import_detector_module" and node.args and isinstance(node.args[0], ast.Constant)}

    assert called == set(chain.EXTERNAL_MODULES)


def test_the_file_reader_outside_the_package_is_hashed(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """detector.recording_io reads the file itself, so an edit to it moves the hash.

    A machine that cannot import it gets a different hash, never the same one.
    """
    real = chain.generation_sha256()
    copy = tmp_path / "recording_io.py"
    src = Path(str(detector_core.import_detector_module("recording_io").__file__))
    copy.write_bytes(src.read_bytes())
    monkeypatch.setattr(detector_core, "import_detector_module",
                        lambda name, root=None: SimpleNamespace(__file__=str(copy)))
    assert chain.generation_sha256() == real
    lf = src.read_bytes().replace(b"\r\n", b"\n")
    copy.write_bytes(lf)
    unix = chain.generation_sha256()
    copy.write_bytes(lf.replace(b"\n", b"\r\n"))  # a Windows checkout of the same file
    assert chain.generation_sha256() == unix == real
    copy.write_bytes(lf)
    with copy.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n# a reader change\n")
    assert chain.generation_sha256() != real

    def missing(name: str, root: object = None) -> object:
        raise FileNotFoundError(name)

    monkeypatch.setattr(detector_core, "import_detector_module", missing)
    assert chain.generation_sha256() != real


def _copy(tmp_path: Path) -> Path:
    dst = tmp_path / "gems_blanking_v2"
    shutil.copytree(PACKAGE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def test_a_scorer_or_report_edit_leaves_the_hash_and_a_chain_edit_moves_it(tmp_path: Path) -> None:
    pkg = _copy(tmp_path)
    before = chain.generation_sha256(pkg)
    assert before == chain.generation_sha256()  # the copy is the real package

    with (pkg / "detect" / "recall.py").open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n# a report wording change\n")
    assert chain.generation_sha256(pkg) == before

    with (pkg / "derive" / "contact_quality.py").open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n# a screen change\n")
    assert chain.generation_sha256(pkg) != before
    loader = chain.generation_sha256(pkg)
    with (pkg / "io" / "channel_map.py").open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n# a channel-order change reaches the array detect_region gets\n")
    assert chain.generation_sha256(pkg) != loader
    moved = chain.generation_sha256(pkg)
    with (pkg / "bands" / "__init__.py").open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n# a package init runs whenever a band module is imported\n")
    assert chain.generation_sha256(pkg) != moved


def test_a_new_import_in_the_chain_joins_the_scope_by_itself(tmp_path: Path) -> None:
    """Derived, not listed: a module the chain starts using is hashed with no edit here."""
    pkg = _copy(tmp_path)
    before = chain.generation_sha256(pkg)
    path = pkg / "detect" / "chain.py"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("import ast\n", "import ast\n\nfrom gems_blanking_v2.io "
                                 "import scan  # noqa: F401\n", 1),
                    encoding="utf-8", newline="\n")

    assert "gems_blanking_v2.io.scan" in chain.generation_modules(pkg)
    after = chain.generation_sha256(pkg)
    with (pkg / "io" / "scan.py").open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n# now part of the chain\n")
    assert chain.generation_sha256(pkg) not in (before, after)


def test_the_hash_is_the_same_whatever_git_did_to_line_endings(tmp_path: Path) -> None:
    pkg = _copy(tmp_path)
    before = chain.generation_sha256(pkg)
    path = pkg / "detect" / "candidates.py"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    assert chain.generation_sha256(pkg) == before
