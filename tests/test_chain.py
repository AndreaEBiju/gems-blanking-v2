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
    return make_cuff_contacts(FS, 30.0, {"L3": "uncorrelated"}, n_stomach=3, seed=21)


def test_the_chain_is_exactly_its_parts_composed_by_hand(rec: Recording) -> None:
    """Every derived signal plus the raw stomach contacts, and nothing removed."""
    region = (4.0, 28.0)
    got = chain.detect_region(rec, region)

    sub = replace(rec, data=rec.data[round(4.0 * FS):round(28.0 * FS)])
    signals, _ = build_derivations(sub)
    raw = {c.name: np.asarray(sub.data[:, c.index], float) for c in sub.channels
           if c.role == "stomach"}
    detection = {**signals, **raw}
    names = sorted(detection)
    z = chain.z_by_pair(np.column_stack([detection[n] for n in names]), FS, names)
    report = candidate_report(z, detect_rpeaks(signals["R_T"], FS))
    want = np.array([[c.start_s + 4.0, c.stop_s + 4.0] for c in report.candidates]).reshape(-1, 2)

    assert got.signals == tuple(names)
    assert got.untrusted == screened_signals(assess_contacts(sub))
    np.testing.assert_array_equal(got.intervals, want)
    assert set(got.z) == set(z)
    assert all(np.array_equal(got.z[k], z[k], equal_nan=True) for k in z)


def test_a_screened_contact_stays_in_the_max_and_is_reported_untrusted(rec: Recording) -> None:
    """Invariant 43: every consumer still reads it, so detection must too."""
    got = chain.detect_region(rec, (0.0, 30.0))

    assert got.untrusted == {"L_V3", "L_T"}
    assert {"L_V1", "L_V2", "L_V3", "L_T", "R_T", "R_V1", "R_V2", "R_V3"} <= set(got.signals)
    assert any(sig == "L_V3" for sig, _band in got.z)


def test_the_raw_stomach_contacts_are_read(rec: Recording) -> None:
    """slow_wave and mmc read ANT1-3 raw; a shared artifact cancels in stomach_ref."""
    got = chain.detect_region(rec, (0.0, 30.0))

    assert {"ANT1", "ANT2", "ANT3", "stomach_ref"} <= set(got.signals)
    assert {("ANT1", "0-2"), ("ANT3", "300-3000")} <= set(got.z)


def test_a_pop_on_a_screened_contact_is_detected() -> None:
    """A pop on a distrusted contact is still a candidate (invariant 43).

    Round 2: rail-scale pops on a broken contact the screen had removed were 9 of 14
    misses. The contact is still distrusted, and its pop is still detected.
    """
    rec = make_cuff_contacts(FS, 30.0, {"L3": "uncorrelated"}, n_stomach=3, seed=22)
    col = next(c.index for c in rec.channels if c.cuff_id == "L" and c.contact_index == 3)
    data = np.array(rec.data, dtype=np.float64)
    data[round(18.00 * FS):round(18.05 * FS), col] -= 2.0e5  # a 50 ms rail-scale pop
    got = chain.detect_region(replace(rec, data=data), (0.0, 30.0))

    assert "L_V3" in got.untrusted
    assert any(a <= 18.05 and b >= 18.0 for a, b in got.intervals)


def test_a_raw_stomach_name_that_collides_with_a_derived_signal_refuses() -> None:
    rec = make_cuff_contacts(FS, 30.0, n_stomach=1, seed=23)
    channels = [replace(c, name="R_T") if c.role == "stomach" else c for c in rec.channels]

    with pytest.raises(ValueError, match="collide"):
        chain.detect_region(replace(rec, channels=channels), (0.0, 30.0))


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
