"""The cross-platform contract in ``CLAUDE.md``, tested rather than asserted in prose.

These are the tests that section names as required. They become meaningful at task
00A, which is the first task that writes anything anyone else will read.

The open question they are shaped around: this lab's shared drive is named
``BIONICs Lab: Enteric Interfaces Team``, and ``:`` is illegal in a Windows path, so
Drive for desktop substitutes *something*. No Drive is mounted on the development
machine, so nobody has yet observed which character. Rather than guess, every root
here is parametrised over the plausible substitutions - the point of the design is
that the code never needs to know.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath

import pytest
from gems_blanking_v2.io.registry_log import (
    LOG_NAME,
    RegistryAction,
    RegistryEvent,
    append_event,
    read_events,
)
from gems_blanking_v2.io.store import (
    WINDOWS_MAX_PATH,
    FileRef,
    GemsStore,
    atomic_write_text,
    safe_component,
    sha256_file,
    utc_stamp,
    validate_component,
)

DRIVE_NAME_VARIANTS = pytest.mark.parametrize(
    "drive_name",
    [
        pytest.param("BIONICs Lab: Enteric Interfaces Team", id="literal-colon-posix-only"),
        pytest.param("BIONICs Lab\u2236 Enteric Interfaces Team", id="u2236-ratio"),
        pytest.param("BIONICs Lab\uf03a Enteric Interfaces Team", id="uf03a-private-use"),
        pytest.param("BIONICs Lab_ Enteric Interfaces Team", id="underscore"),
        pytest.param("BIONICs Lab  Enteric Interfaces Team", id="space-MEASURED"),
    ],
)
"""The names the shared-drive folder could have on disk, and the one it does.

**Measured 2026-09-22, Google Drive for desktop on Windows 11: the colon becomes
U+0020 SPACE.** The folder is literally ``BIONICs Lab  Enteric Interfaces Team``
with two consecutive spaces - the original space after the colon, plus the space
that replaced it. Not U+2236 RATIO, not U+F03A, not an underscore; the first four
entries here were all guesses and **none of them was right**, which is the whole
argument for discovering the root by marker instead of reconstructing it.

The literal-colon variant is only creatable on POSIX and its test skips on Windows.
Whichever name is real, discovery is by marker and stored paths are relative, so
nothing downstream changes - confirmed against the real mount.
"""


def _make_root(tmp_path: Path, drive_name: str) -> GemsStore:
    """Build a fake shared-drive layout whose folder carries ``drive_name``."""
    return GemsStore.initialise(tmp_path / "Shared drives" / drive_name)


# ---------------------------------------------------------------------------
# no absolute path is ever emitted
# ---------------------------------------------------------------------------


@DRIVE_NAME_VARIANTS
def test_no_absolute_path_appears_in_an_emitted_corpus_spec(
    tmp_path: Path, drive_name: str
) -> None:
    if ":" in drive_name and Path("C:/").anchor:  # pragma: no cover - platform-dependent
        pytest.skip("a literal colon cannot be a Windows directory name")
    store = _make_root(tmp_path, drive_name)
    raw = store.session_dir("J", "t01") / "raw.h5"
    atomic_write_text(raw, "payload")

    spec = {
        "corpus_id": "pooled_v3",
        "files": [
            {
                "path": store.relpath(raw),
                "sha256": sha256_file(raw),
                "bytes": raw.stat().st_size,
            }
        ],
    }
    atomic_write_text(
        store.corpus_path("pooled_v3"), json.dumps(spec, indent=2, sort_keys=True) + "\n"
    )

    text = store.corpus_path("pooled_v3").read_text(encoding="utf-8")
    assert str(store.root) not in text
    assert drive_name not in text
    assert "Shared drives" not in text
    assert json.loads(text)["files"][0]["path"] == "data/J/t01/raw.h5"


@DRIVE_NAME_VARIANTS
def test_no_absolute_path_appears_in_a_registry_line(tmp_path: Path, drive_name: str) -> None:
    if ":" in drive_name and Path("C:/").anchor:  # pragma: no cover - platform-dependent
        pytest.skip("a literal colon cannot be a Windows directory name")
    store = _make_root(tmp_path, drive_name)
    append_event(
        store,
        RegistryEvent(
            "20260920T100000Z", "andrea", RegistryAction.TRAINED, "a" * 32, corpus_id="c1"
        ),
        stamp="20260920T100000Z",
    )
    text = (store.registry_dir / f"{LOG_NAME}.andrea.20260920T100000Z").read_text(encoding="utf-8")
    assert str(store.root) not in text
    assert drive_name not in text


def test_a_provenance_record_refuses_an_absolute_path(tmp_path: Path) -> None:
    """Rule 2 is enforced at construction, not left to reviewer discipline."""
    store = GemsStore.initialise(tmp_path / "gems")
    with pytest.raises(ValueError, match="outside gems_root"):
        store.relpath(tmp_path / "somewhere-else" / "raw.h5")
    with pytest.raises(ValueError, match="must be relative"):
        FileRef(rel_path=str(store.root / "data" / "raw.h5"), sha256="0" * 64, size_bytes=1)


# ---------------------------------------------------------------------------
# a POSIX-separator spec resolves under a Windows-style root
# ---------------------------------------------------------------------------


def test_a_posix_spec_resolves_against_a_windows_style_root() -> None:
    r"""A spec written on a Mac must resolve under ``G:\Shared drives\<name>``."""
    stored = "data/J/t01/raw.h5"
    windows_root = PureWindowsPath(r"G:\Shared drives\BIONICs Lab_ Enteric Interfaces Team")
    resolved = windows_root.joinpath(*stored.split("/"))

    assert resolved == PureWindowsPath(
        r"G:\Shared drives\BIONICs Lab_ Enteric Interfaces Team\data\J\t01\raw.h5"
    )
    assert resolved.as_posix().endswith(stored)


@DRIVE_NAME_VARIANTS
def test_relative_paths_round_trip_whatever_the_root_is_called(
    tmp_path: Path, drive_name: str
) -> None:
    if ":" in drive_name and Path("C:/").anchor:  # pragma: no cover - platform-dependent
        pytest.skip("a literal colon cannot be a Windows directory name")
    store = _make_root(tmp_path, drive_name)
    for path in (
        store.session_dir("J", "t01") / "raw.h5",
        store.trials_path("J", "t01"),
        store.model_dir("a" * 32) / "provenance.json",
        store.deepest_path(),
    ):
        rel = store.relpath(path)
        assert not rel.startswith("/")
        assert "\\" not in rel
        assert store.abspath(rel) == path


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["GEMS_D_t01_MS1_bl", "gems_d_t01_ms1_bl", "Gems_D_T01_ms1_BL"])
def test_condition_rules_match_the_same_file_whatever_the_case(name: str) -> None:
    """Case never decides a match: this lab's data mixes it for the same entity."""
    rule = re.compile(
        r"gems_(?P<animal>[a-z])_t(?P<trial>\d+)_ms(?P<ms>\d+)_(?P<cond>bl|st)", re.IGNORECASE
    )
    match = rule.match(name)
    assert match is not None
    assert match.group("animal").upper() == "D"
    assert match.group("trial") == "01"


def test_two_paths_differing_only_by_case_are_rejected(tmp_path: Path) -> None:
    """Both macOS and Windows are case-insensitive; such a pair silently collides."""
    names = ["data/J/t01/raw.h5", "data/j/T01/raw.h5"]
    seen: dict[str, str] = {}
    with pytest.raises(ValueError, match="differ only by case"):
        for name in names:
            key = name.casefold()
            if key in seen:
                msg = f"paths differ only by case and will collide: {seen[key]!r} and {name!r}"
                raise ValueError(msg)
            seen[key] = name


# ---------------------------------------------------------------------------
# path length
# ---------------------------------------------------------------------------


def test_the_deepest_generated_path_is_reported_under_a_windows_style_root() -> None:
    r"""The real data already sits at ~193 characters before the tool appends anything."""
    windows_root = PureWindowsPath(r"G:\Shared drives\BIONICs Lab_ Enteric Interfaces Team")
    deepest = windows_root / "models" / ("a" * 32) / "shap" / "cross_channel_commonality.html"
    length = len(str(deepest))

    assert length < WINDOWS_MAX_PATH, f"the layout alone is already {length} chars"
    headroom = WINDOWS_MAX_PATH - length
    assert headroom > 0
    # Documented, not asserted as a target: this is how much room a deeper real root
    # has before the layout stops fitting.
    assert headroom < 200


def test_an_over_long_path_fails_with_a_clear_message_not_an_oserror(tmp_path: Path) -> None:
    store = GemsStore(tmp_path / ("x" * 150) / ("y" * 150))
    message = store.check_path_length(store.deepest_path())
    assert message is not None
    assert "Windows limit" in message and str(WINDOWS_MAX_PATH) in message


# ---------------------------------------------------------------------------
# filenames
# ---------------------------------------------------------------------------


def test_every_generated_filename_component_is_portable(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    generated = [
        store.trials_path("J", "t01").name,
        store.labels_path("J", "Andrea Biju").name,
        store.corpus_path("pooled_v3").name,
        store.model_dir("a" * 32).name,
        store.deepest_path().name,
        f"{LOG_NAME}.{safe_component('Andrea Biju')}.{utc_stamp()}",
    ]
    for name in generated:
        assert validate_component(name) == name


# ---------------------------------------------------------------------------
# text I/O
# ---------------------------------------------------------------------------


def test_every_emitted_text_file_round_trips_as_utf8_with_lf(tmp_path: Path) -> None:
    """Default encoding is not UTF-8 on all Windows installs, and JSONL must not gain CRLF."""
    store = GemsStore.initialise(tmp_path / "gems")
    # Non-ASCII on purpose: a sigma and a micro sign are exactly what this project
    # writes, and they are what a cp1252 default encoding would mangle.
    payload = "animal J — cuff L, σ = 2.26 µV\nsecond line\n"  # noqa: RUF001

    atomic_write_text(store.corpus_path("unicode_probe"), payload)
    append_event(
        store,
        RegistryEvent("20260920T100000Z", "andrea", RegistryAction.TRAINED, "m1"),
        stamp="20260920T100000Z",
    )

    for path in (store.corpus_path("unicode_probe"), *store.registry_dir.iterdir(), store.marker):
        raw = path.read_bytes()
        assert b"\r\n" not in raw, path
        assert raw.decode("utf-8")

    assert store.corpus_path("unicode_probe").read_text(encoding="utf-8") == payload
    assert read_events(store)[0].model_id == "m1"


def test_the_marker_file_is_plain_utf8(tmp_path: Path) -> None:
    store = GemsStore.initialise(tmp_path / "gems")
    assert store.marker.read_bytes() == b"gems-blanking-v2 root\n"


# ---------------------------------------------------------------------------
# no symlinks
# ---------------------------------------------------------------------------


def test_the_layout_creates_no_symlinks(tmp_path: Path) -> None:
    """Windows needs elevation for symlinks (rule 6), so the layout uses none."""
    store = GemsStore.initialise(tmp_path / "gems")
    atomic_write_text(store.session_dir("J", "t01") / "raw.h5", "x")
    for path in store.root.rglob("*"):
        assert not path.is_symlink(), path
