"""Tests for scan scope and the protocol book - what is in the corpus, and by whose rule."""

from __future__ import annotations

from pathlib import Path

import pytest
from gems_blanking_v2.io.corpus import (
    CORPUS_FILENAME,
    DEFAULT_SCAN_ROOTS,
    ScanScope,
    corpus_path,
    default_scope,
    read_scan_roots,
    write_scan_roots,
)
from gems_blanking_v2.io.stim_split import (
    PROTOCOL_FILENAME,
    ProtocolNotCoveredError,
    default_protocol_book,
    load_protocol_book,
    write_protocol_book,
)

CHRONIC = "August-September Chronic Recordings"
BALLOON = "081526BalloonTrial"


# ---------------------------------------------------------------------------
# scan scope
# ---------------------------------------------------------------------------


def test_the_root_is_not_the_scan_scope(tmp_path: Path) -> None:
    """``GEMS-Andrea`` holds code trees as well as data, so the two differ.

    ``processing_new``, ``TDTMatlabSDK``, ``nerve-processing`` and
    ``IACUC Inspection 092026`` all sit under the root. Walking everything under it
    would put a MATLAB checkout in the corpus and count it against
    ``SHARED_DRIVE_ITEM_CAP``.
    """
    scope = default_scope()

    assert scope.scan_roots == DEFAULT_SCAN_ROOTS == (CHRONIC,)
    assert scope.covers(f"{CHRONIC}/09152026/gems_j_t01_ms3_bl_230315") == CHRONIC
    for outside in ("processing_new/step1_bandpass.m", "TDTMatlabSDK/x", BALLOON):
        assert scope.covers(outside) is None, outside


def test_a_prefix_match_is_not_a_path_match() -> None:
    """``August-September Chronic Recordings 2`` is a different folder."""
    scope = ScanScope(scan_roots=(CHRONIC,))

    assert scope.covers(CHRONIC) == CHRONIC
    assert scope.covers(f"{CHRONIC}/x") == CHRONIC
    assert scope.covers(f"{CHRONIC} 2/x") is None


def test_scan_roots_round_trip_and_carry_no_absolute_path(tmp_path: Path) -> None:
    r"""Rule 2: relative and POSIX, so the file reads the same on both platforms."""
    path = write_scan_roots(tmp_path, default_scope())

    assert path == corpus_path(tmp_path)
    assert path.name == CORPUS_FILENAME
    assert read_scan_roots(tmp_path) == default_scope()

    body = path.read_text(encoding="utf-8")
    assert "G:" not in body
    assert "\\" not in body
    assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize(
    "bad", ["/absolute/path", r"Windows\Style", "C:/drive", ""]
)
def test_a_scan_root_that_cannot_resolve_on_the_other_platform_raises(bad: str) -> None:
    """An absolute or Windows-style entry breaks the moment a Mac reads it."""
    with pytest.raises(ValueError, match="relative POSIX path"):
        ScanScope(scan_roots=(bad,))


def test_an_empty_scope_raises() -> None:
    """Scanning nothing is a configuration mistake, not a configuration."""
    with pytest.raises(ValueError, match="nothing would ever be scanned"):
        ScanScope(scan_roots=())


def test_the_scan_scope_is_never_defaulted(tmp_path: Path) -> None:
    """Absent means absent: guessing which subfolder holds data is worse than stopping."""
    with pytest.raises(FileNotFoundError, match="not defaulted"):
        read_scan_roots(tmp_path)


def test_scan_roots_resolve_against_the_local_root(tmp_path: Path) -> None:
    """The one place the machine-specific root gets joined on."""
    resolved = default_scope().resolve(tmp_path)

    assert resolved == [tmp_path / CHRONIC]
    assert resolved[0].is_absolute()


# ---------------------------------------------------------------------------
# the protocol book
# ---------------------------------------------------------------------------


def test_an_uncovered_scan_root_is_refused_not_defaulted(tmp_path: Path) -> None:
    """Refuse an uncovered scan root - the balloon trials must not inherit 120 s.

    They are a different experiment. Splitting one against a protocol that does not
    describe it produces a confident, wrong boundary and a clean-looking status,
    which is worse than stopping - the same rule as a missing ``vib`` channel.
    """
    book = default_protocol_book()

    assert book.for_scan_root(CHRONIC).name == "chronic_2min_20min"
    assert book.for_scan_root(CHRONIC).stim_duration_s == 120.0

    with pytest.raises(ProtocolNotCoveredError, match="refusal, not a default"):
        book.for_scan_root(BALLOON)


def test_a_recording_path_finds_its_protocol(tmp_path: Path) -> None:
    """Lookup by the gems_root-relative path, which is what a scan actually holds."""
    book = default_protocol_book()
    covered = f"{CHRONIC}/09152026/gems_j_t01_ms3_sr_231500/x_sig.mat"

    assert book.for_path(covered).name == "chronic_2min_20min"

    with pytest.raises(ProtocolNotCoveredError):
        book.for_path(f"{BALLOON}/baseline/baseline_sig.mat")


def test_the_protocol_book_round_trips(tmp_path: Path) -> None:
    r"""Written atomically as UTF-8 with ``\n`` endings, and always loadable."""
    path = write_protocol_book(default_protocol_book(), tmp_path / PROTOCOL_FILENAME)
    book = load_protocol_book(path)

    assert [s.name for s in book.protocols] == ["chronic_2min_20min"]
    assert book.protocols[0].applies_to == (CHRONIC,)
    assert book.protocols[0].recovery_duration_s == 1200.0
    assert book.protocols[0].stim_tolerance_s == 12.0
    assert b"\r\n" not in path.read_bytes()


def test_the_protocol_book_is_never_defaulted(tmp_path: Path) -> None:
    """Same rule as the single-spec form: a silently appearing prior moves boundaries."""
    with pytest.raises(FileNotFoundError, match="no protocol at"):
        load_protocol_book(tmp_path / PROTOCOL_FILENAME)


def test_a_malformed_protocol_book_raises(tmp_path: Path) -> None:
    """A document with no ``protocols`` list is not a book."""
    path = tmp_path / PROTOCOL_FILENAME
    path.write_text("stim_duration_s: 120\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="'protocols' list is required"):
        load_protocol_book(path)


def test_every_protocols_scan_root_is_in_scope() -> None:
    """A protocol governing a folder nobody scans is dead config.

    Not an error - the drive may carry protocols for scopes this machine has not
    enabled - but the two defaults must agree, or the shipped configuration refuses
    the only folder it was written for.
    """
    book = default_protocol_book()
    scope = default_scope()

    governed = {root for spec in book.protocols for root in spec.applies_to}
    assert governed == set(scope.scan_roots)
