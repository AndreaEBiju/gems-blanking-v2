"""The shared-drive store: discovery, layout, integrity, preflight.

Every test runs against a temporary fake ``gems_root`` with the real layout. No
Google Drive is mounted on the development machine, so the behaviour against a real
synced root - in particular what Drive substitutes for the colon in this lab's
drive name - is still unverified. See ``test_portability.py``, which parametrises
the root's name over the plausible substitutions so the code is indifferent to it.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from gems_blanking_v2.io.store import (
    MARKER_NAME,
    SHARED_DRIVE_ITEM_CAP,
    WINDOWS_MAX_PATH,
    FileRef,
    GemsStore,
    append_line,
    atomic_write_bytes,
    atomic_write_text,
    cache_dir,
    find_gems_root,
    read_lines,
    resolve_user_id,
    safe_component,
    sha256_file,
    utc_stamp,
    validate_component,
)


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    """Return a fake gems_root with the real layout, on the local filesystem."""
    return GemsStore.initialise(tmp_path / "gems")


# ---------------------------------------------------------------------------
# names, stamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["a<b", "a>b", 'a"b', "a/b", "a\\b", "a|b", "a?b", "a*b", "a:b"])
def test_validate_rejects_every_windows_illegal_character(bad: str) -> None:
    with pytest.raises(ValueError, match="illegal characters"):
        validate_component(bad)


@pytest.mark.parametrize("bad", ["trailing.", "trailing ", "trailing. "])
def test_validate_rejects_trailing_dots_and_spaces(bad: str) -> None:
    with pytest.raises(ValueError, match="dot or space"):
        validate_component(bad)


@pytest.mark.parametrize(
    "bad", ["CON", "con", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "nul.txt"]
)
def test_validate_rejects_windows_reserved_device_names(bad: str) -> None:
    with pytest.raises(ValueError, match="reserved device name"):
        validate_component(bad)


def test_validate_rejects_empty_and_control_characters() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        validate_component("")
    with pytest.raises(ValueError, match="illegal characters"):
        validate_component("a\x01b")


@pytest.mark.parametrize("good", ["J", "t01_ms1_bl_164012", "pooled-v3", "a.b.c"])
def test_validate_accepts_ordinary_names(good: str) -> None:
    assert validate_component(good) == good


def test_safe_component_makes_a_git_user_name_into_a_filename() -> None:
    """``First Last`` is the common case and it contains a space."""
    assert safe_component("Andrea Biju") == "Andrea_Biju"
    assert safe_component("a:b/c") == "a_b_c"
    assert safe_component("   ") == "unknown"
    assert safe_component("CON") == "unknown"


def test_utc_stamp_has_no_colon_so_it_can_live_in_a_filename() -> None:
    """An ISO stamp (``2026-09-20T18:45:00Z``) would make the shard name unwritable."""
    stamp = utc_stamp(datetime(2026, 9, 20, 18, 45, 0, tzinfo=UTC))
    assert stamp == "20260920T184500Z"
    assert ":" not in stamp
    assert validate_component(stamp) == stamp


def test_utc_stamp_defaults_to_now_and_stays_filename_safe() -> None:
    assert validate_component(utc_stamp()).endswith("Z")


# ---------------------------------------------------------------------------
# integrity and atomic writes
# ---------------------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = os.urandom(3_000_000)  # larger than the 1 MiB chunk
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_atomic_write_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "a.json"
    atomic_write_text(target, "{}\n")
    assert target.read_text(encoding="utf-8") == "{}\n"
    assert [p.name for p in target.parent.iterdir()] == ["a.json"]


def test_atomic_write_replaces_rather_than_truncating(tmp_path: Path) -> None:
    """A reader never sees a half-written shared file."""
    target = tmp_path / "a.txt"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"


def test_atomic_write_cleans_up_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.bin"

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_bytes(target, b"x")
    assert list(tmp_path.iterdir()) == []


def test_append_and_read_round_trip_with_lf(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    append_line(path, "one")
    append_line(path, "two\n")
    assert path.read_bytes() == b"one\ntwo\n"
    assert read_lines(path) == ["one", "two"]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discovery_prefers_an_explicit_root(store: GemsStore) -> None:
    assert find_gems_root(store.root, scan=False) == store.root


def test_discovery_reads_the_env_var(store: GemsStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMS_ROOT", str(store.root))
    assert find_gems_root(scan=False) == store.root


def test_discovery_requires_the_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory that merely looks right is not a root."""
    plausible = tmp_path / "Shared drives" / "BIONICs Lab_ Enteric Interfaces Team"
    (plausible / "data").mkdir(parents=True)
    monkeypatch.setenv("GEMS_ROOT", str(plausible))
    with pytest.raises(FileNotFoundError, match=MARKER_NAME):
        find_gems_root(scan=False)


def test_discovery_error_names_what_it_tried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMS_ROOT", raising=False)
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError) as exc:
        find_gems_root(missing, scan=False)
    assert str(missing) in str(exc.value)


def test_initialise_is_idempotent_and_creates_the_layout(tmp_path: Path) -> None:
    root = tmp_path / "gems"
    first = GemsStore.initialise(root)
    marker_before = first.marker.read_bytes()
    second = GemsStore.initialise(root)
    assert first == second
    assert first.marker.read_bytes() == marker_before
    for sub in ("data", "trials", "labels", "corpora", "models", "registry"):
        assert (root / sub).is_dir()


# ---------------------------------------------------------------------------
# layout and paths
# ---------------------------------------------------------------------------


def test_layout_matches_the_spec(store: GemsStore) -> None:
    assert store.relpath(store.session_dir("J", "t01")) == "data/J/t01"
    assert store.relpath(store.trials_path("J", "t01")) == "trials/J/t01/trials.jsonl"
    assert store.relpath(store.corpus_path("pooled_v3")) == "corpora/pooled_v3.json"
    assert store.relpath(store.model_dir("a" * 32)) == f"models/{'a' * 32}"
    assert store.relpath(store.labels_path("J", "Andrea Biju", "20260920T184500Z")) == (
        "labels/J/events_Andrea_Biju_20260920T184500Z.parquet"
    )


def test_trials_is_one_jsonl_per_session_not_one_file_per_trial(store: GemsStore) -> None:
    """One file per trial is how a layout reaches the 500,000-item cap."""
    assert store.trials_path("J", "t01").name == "trials.jsonl"


def test_relpath_is_posix_on_every_platform(store: GemsStore) -> None:
    rel = store.relpath(store.model_dir("a" * 32) / "shap" / "onset_rate.html")
    assert "\\" not in rel
    assert rel == f"models/{'a' * 32}/shap/onset_rate.html"


def test_relpath_refuses_a_path_outside_the_root(store: GemsStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside gems_root"):
        store.relpath(tmp_path / "elsewhere" / "x.json")


def test_abspath_round_trips_and_rejects_escapes(store: GemsStore) -> None:
    path = store.corpus_path("pooled_v3")
    assert store.abspath(store.relpath(path)) == path
    for bad in ("/etc/passwd", "../outside.json", "a/../../b"):
        with pytest.raises(ValueError, match="must be relative"):
            store.abspath(bad)


def test_layout_validates_the_names_it_is_given(store: GemsStore) -> None:
    with pytest.raises(ValueError, match="illegal characters"):
        store.session_dir("J", "t01:ms1")
    with pytest.raises(ValueError, match="reserved device name"):
        store.model_dir("CON")


# ---------------------------------------------------------------------------
# path length
# ---------------------------------------------------------------------------


def test_deepest_path_is_the_shap_page(store: GemsStore) -> None:
    deepest = store.deepest_path()
    assert store.relpath(deepest).startswith("models/")
    assert deepest.suffix == ".html"
    assert "shap" in deepest.parts


def test_over_long_path_is_reported_not_raised_as_oserror(tmp_path: Path) -> None:
    """The machine that creates an over-long path is usually not the one that fails."""
    deep = tmp_path / ("d" * 120) / ("e" * 120)
    store = GemsStore(deep)
    message = store.check_path_length(store.deepest_path())
    assert message is not None
    assert str(WINDOWS_MAX_PATH) in message
    assert "long paths" in message


def test_a_short_root_passes_the_length_check(store: GemsStore) -> None:
    assert store.check_path_length(store.deepest_path()) is None


def test_an_over_long_path_makes_preflight_refuse(tmp_path: Path) -> None:
    """Rule 5: fail with a clear message, not an OSError on someone else's machine.

    Blocking rather than advisory, and on every platform: a root this long produces
    artifacts a Windows colleague cannot open, and the machine that creates the path
    is never the one that fails on it.
    """
    # Not initialised on purpose: Windows cannot even create this directory, which
    # is the point - the check has to happen before anything touches the filesystem.
    store = GemsStore(tmp_path / ("d" * 120) / ("e" * 100))
    report = store.preflight()
    assert not report.ok
    assert report.path_too_long
    assert "path too long" in report.summary()
    assert "REFUSE" in report.summary()


def test_a_short_root_preflights_clean(store: GemsStore) -> None:
    report = store.preflight()
    assert report.path_too_long == ""
    assert report.ok


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_is_outside_the_root(store: GemsStore) -> None:
    store.assert_cache_is_outside()
    assert store.root.resolve() not in cache_dir().resolve().parents


def test_a_cache_inside_the_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Syncing 16 GB of envelopes to every lab member is the failure being prevented."""
    root = tmp_path / "gems"
    store = GemsStore.initialise(root)
    monkeypatch.setattr("gems_blanking_v2.io.store.cache_dir", lambda: root / "cache")
    with pytest.raises(ValueError, match="would be synced"):
        store.assert_cache_is_outside()


def test_no_layout_accessor_points_into_a_cache_directory(store: GemsStore) -> None:
    """Nothing in the layout writes to ``gems_root/cache``."""
    paths = [
        store.session_dir("J", "t01"),
        store.trials_path("J", "t01"),
        store.labels_path("J", "u", "20260920T184500Z"),
        store.corpus_path("c"),
        store.model_dir("a" * 32),
        store.registry_dir,
        store.deepest_path(),
    ]
    assert not any("cache" in p.parts for p in paths)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_explicit_identity_wins_and_is_filename_safe() -> None:
    identity = resolve_user_id("Andrea Biju")
    assert identity.user_id == "Andrea_Biju"
    assert identity.source == "explicit"
    assert identity.is_confident


def test_identity_without_configuration_is_marked_unconfident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent fallback would put the wrong name on scientific metadata."""
    monkeypatch.setattr("gems_blanking_v2.io.store._git_config", lambda _key: None)
    monkeypatch.setenv("USERNAME", "User")
    identity = resolve_user_id()
    assert identity.source == "fallback"
    assert not identity.is_confident


# ---------------------------------------------------------------------------
# integrity checks and preflight
# ---------------------------------------------------------------------------


def _write_corpus_file(store: GemsStore, rel: str, payload: bytes) -> FileRef:
    """Write a file into the root and return its manifest entry."""
    path = store.abspath(rel)
    atomic_write_bytes(path, payload)
    return FileRef(rel_path=rel, sha256=sha256_file(path), size_bytes=len(payload))


def test_fileref_refuses_an_absolute_path() -> None:
    for bad in ("/data/J/raw.h5", "C:/data/J/raw.h5", "../escape.h5"):
        with pytest.raises(ValueError, match="must be relative"):
            FileRef(rel_path=bad, sha256="0" * 64, size_bytes=1)


def test_preflight_passes_on_a_complete_corpus(store: GemsStore) -> None:
    refs = [_write_corpus_file(store, "data/J/t01/raw.h5", b"payload" * 100)]
    report = store.preflight(refs)
    assert report.ok
    assert report.missing == []
    assert "OK" in report.summary()


def test_preflight_names_the_missing_file(store: GemsStore) -> None:
    refs = [FileRef("data/J/t01/raw.h5", "0" * 64, 10)]
    report = store.preflight(refs)
    assert not report.ok
    assert report.missing == ["data/J/t01/raw.h5"]
    assert "data/J/t01/raw.h5" in report.summary()


def test_a_truncated_file_fails_the_checksum(store: GemsStore) -> None:
    """Drive can present a partially synced file as complete."""
    ref = _write_corpus_file(store, "labels/J/events_u_20260920T184500Z.parquet", b"x" * 5000)
    store.abspath(ref.rel_path).write_bytes(b"x" * 4000)
    assert store.verify(ref) is not None
    report = store.preflight([ref])
    assert not report.ok
    assert report.corrupt and "size" in report.corrupt[0]


def test_a_same_size_corruption_still_fails(store: GemsStore) -> None:
    ref = _write_corpus_file(store, "corpora/c.json", b"a" * 100)
    store.abspath(ref.rel_path).write_bytes(b"b" * 100)
    why = store.verify(ref)
    assert why is not None
    assert "sha256" in why


def test_a_placeholder_is_reported_as_syncing_not_as_corrupt(store: GemsStore) -> None:
    ref = _write_corpus_file(store, "data/J/t01/raw.h5", b"payload" * 100)
    store.abspath(ref.rel_path).write_bytes(b"")
    report = store.preflight([ref])
    assert not report.ok
    assert report.syncing == ["data/J/t01/raw.h5"]
    assert report.corrupt == []


def test_preflight_flags_a_stored_absolute_path(store: GemsStore) -> None:
    report = store.preflight(
        stored_paths=["data/J/t01/raw.h5", "C:/Users/User/raw.h5", "/Users/a/raw.h5"]
    )
    assert not report.ok
    assert len(report.absolute_paths) == 2


def test_preflight_reports_platform_root_and_longest_path(store: GemsStore) -> None:
    report = store.preflight(platform_name="Windows 11")
    text = report.summary()
    assert "Windows 11" in text
    assert str(store.root) in text
    assert str(report.longest_path_len) in text


def test_probe_writable_is_true_on_a_writable_root_and_leaves_nothing(store: GemsStore) -> None:
    before = sorted(p.name for p in store.root.iterdir())
    assert store.probe_writable()
    assert sorted(p.name for p in store.root.iterdir()) == before


def test_a_read_only_root_produces_the_contributor_hint(
    store: GemsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive for desktop makes a Contributor read-only; say that, not 'permission denied'."""
    monkeypatch.setattr(GemsStore, "probe_writable", lambda _self: False)
    report = store.preflight()
    assert not report.writable
    assert "Content manager" in report.summary()


def test_item_count_is_bounded_and_reports_incompleteness(store: GemsStore) -> None:
    for i in range(30):
        atomic_write_text(store.root / "data" / f"f{i}.txt", "x")
    count, complete = store.count_items(cap=10)
    assert (count, complete) == (10, False)
    count, complete = store.count_items(cap=SHARED_DRIVE_ITEM_CAP)
    assert complete
    assert count >= 30
