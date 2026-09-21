"""Shared storage on a Google shared drive: discovery, layout, integrity, preflight.

The repo holds code only. Data, labels, models and the registry live in a synced
shared-drive folder, and ``gems_root`` is the one thing each user sets locally.

Google Drive is a **sync layer, not a database**: no atomic rename across clients,
no locking, and a silent ``file (1).json`` when two people write the same path.
Everything here is therefore write-once, content-addressed and append-only, and
every read verifies a checksum because Drive can present a partially synced file as
complete.

Units: sizes in bytes, times in UTC. Paths stored anywhere inside ``gems_root`` are
POSIX and relative to it (cross-platform rule 2); absolute paths appear only in
per-user local config, which is never committed and never synced.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

import platformdirs

__all__ = [
    "FileRef",
    "GemsStore",
    "PreflightReport",
    "cache_dir",
    "find_gems_root",
    "read_config_root",
    "resolve_user_id",
    "safe_component",
    "sha256_file",
    "utc_stamp",
    "validate_component",
    "write_config_root",
]

# ---------------------------------------------------------------------------
# constants - shared-drive limits, from the vendor's published numbers
# ---------------------------------------------------------------------------

MARKER_NAME: Final = ".gems-root"
"""Marker file that identifies a directory as a gems root. Discovery never
reconstructs the root from the drive's name - see :func:`find_gems_root`."""

SHARED_DRIVE_ITEM_CAP: Final = 500_000
"""Items per shared drive, counting files, folders, shortcuts **and trash**."""

ITEM_CAP_WARN_FRACTION: Final = 0.80
"""Fraction of :data:`SHARED_DRIVE_ITEM_CAP` above which ``gems doctor`` warns."""

WINDOWS_MAX_PATH: Final = 260
"""Windows path limit unless long paths are enabled. The real data already sits at
~193 characters before this tool appends anything."""

MAX_FOLDER_DEPTH: Final = 100
"""Shared-drive folder nesting limit."""

UPLOAD_BYTES_PER_USER_PER_DAY: Final = 750 * 1024**3
"""Shared-drive upload quota per user per 24 h. ~570 recordings/day at 1.3 GB."""

MAX_FILE_BYTES: Final = 5 * 1024**4
"""Shared-drive maximum single-file size."""

_MIN_PRINTABLE_ORD: Final = 32
"""Characters below this code point are control characters and never legal."""

_MIN_QUOTED_LEN: Final = 2
"""Shortest quoted TOML value: a pair of quotes."""

_ILLEGAL_CHARS: Final = '<>:"/\\|?*'
"""Characters a Windows path may not contain (cross-platform rule 9)."""

_RESERVED_NAMES: Final[frozenset[str]] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
"""Windows device names, reserved with or without an extension."""

_CONTRIBUTOR_HINT: Final = (
    "you appear to be a Contributor; Drive for desktop makes that read-only - "
    "ask a Manager for Content manager"
)

_HASH_CHUNK_BYTES: Final = 1 << 20


# ---------------------------------------------------------------------------
# names and stamps
# ---------------------------------------------------------------------------


def validate_component(name: str) -> str:
    r"""Return ``name`` unchanged, or raise if it cannot be a portable path segment.

    Rejects the Windows-illegal characters ``<>:"/\\|?*``, control characters,
    trailing dots or spaces, the reserved device names, and the empty string.
    Use this on anything that arrives from outside; use :func:`safe_component` to
    build a segment from text that was never meant to be one.
    """
    if not name:
        msg = "path component must not be empty"
        raise ValueError(msg)
    bad = sorted({c for c in name if c in _ILLEGAL_CHARS or ord(c) < _MIN_PRINTABLE_ORD})
    if bad:
        msg = f"path component {name!r} contains illegal characters: {bad}"
        raise ValueError(msg)
    if name != name.rstrip(". "):
        msg = f"path component {name!r} ends with a dot or space, which Windows strips"
        raise ValueError(msg)
    if name.split(".", maxsplit=1)[0].upper() in _RESERVED_NAMES:
        msg = f"path component {name!r} is a Windows reserved device name"
        raise ValueError(msg)
    return name


def safe_component(text: str, fallback: str = "unknown") -> str:
    """Return a portable path segment built from arbitrary ``text``.

    Illegal characters, control characters and whitespace collapse to ``_``. A git
    ``user.name`` is typically ``First Last``, which is why this exists: that string
    goes into a registry shard filename.
    """
    cleaned = re.sub(rf"[{re.escape(_ILLEGAL_CHARS)}\s\x00-\x1f]+", "_", text).strip("_. ")
    if not cleaned or cleaned.split(".")[0].upper() in _RESERVED_NAMES:
        return fallback
    return cleaned


def utc_stamp(when: datetime | None = None) -> str:
    """Return a UTC timestamp usable **inside a filename**: ``20260920T184500Z``.

    Deliberately not ISO 8601: ``2026-09-20T18:45:00Z`` contains colons, which are
    illegal in a Windows path. The registry shard name embeds this stamp, so an ISO
    one would make the layout unwritable on half the lab's machines.
    """
    moment = when or datetime.now(tz=UTC)
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# integrity and atomic writes
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the lowercase hex sha256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: temp file in the same directory, then replace.

    Same directory matters - ``os.replace`` is only atomic within a filesystem, and
    a temp file elsewhere would cross one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    r"""Write text atomically as UTF-8 with ``\\n`` endings (cross-platform rule 10)."""
    atomic_write_bytes(path, text.encode("utf-8"))


def append_line(path: Path, line: str) -> None:
    r"""Append one ``\\n``-terminated UTF-8 line to ``path``, creating it if needed.

    Append is the only mutation this design permits on a shared file, and only to a
    file the local user owns exclusively (its own registry shard).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line.rstrip("\n") + "\n")


def read_lines(path: Path) -> list[str]:
    r"""Read a text file as UTF-8 and return its non-empty lines, ``\\n`` normalised."""
    with path.open("r", encoding="utf-8", newline="\n") as fh:
        return [ln.strip("\n\r") for ln in fh if ln.strip()]


# ---------------------------------------------------------------------------
# root discovery and per-user config
# ---------------------------------------------------------------------------

CONFIG_APP: Final = "gems-blanking-v2"
CONFIG_NAME: Final = "config.toml"


def config_path() -> Path:
    """Return the per-user config file path, via ``platformdirs``.

    Cross-platform rule 15: not a hardcoded ``~/.gems``. ``IMPLEMENTATION.md`` 00A
    names ``~/.gems/config.toml``; that is still read (see :func:`read_config_root`)
    but new writes go here.
    """
    return Path(platformdirs.user_config_dir(CONFIG_APP, appauthor=False)) / CONFIG_NAME


def legacy_config_path() -> Path:
    """Return the pre-``platformdirs`` config location, read for compatibility."""
    return Path.home() / ".gems" / CONFIG_NAME


def _parse_config_root(text: str) -> Path | None:
    """Pull ``gems_root`` out of a minimal TOML document, or return None."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("gems_root"):
            continue
        _, _, value = line.partition("=")
        value = value.strip()
        quoted = len(value) >= _MIN_QUOTED_LEN and value[0] == value[-1]
        if quoted and value[0] in "'\"":
            return Path(value[1:-1])
    return None


def read_config_root() -> Path | None:
    """Return the configured ``gems_root``, or None if the user has not set one."""
    for candidate in (config_path(), legacy_config_path()):
        if candidate.is_file():
            root = _parse_config_root(candidate.read_text(encoding="utf-8"))
            if root is not None:
                return root
    return None


def write_config_root(root: Path) -> Path:
    """Persist ``gems_root`` to the per-user config and return the file written.

    The stored path is **absolute and local** - correct here, because this file is
    per-user, never committed and never synced. Rule 2 governs what goes *into*
    ``gems_root``, not what points at it.
    """
    text = str(root)
    if "'" in text:
        msg = f"cannot store a root containing a single quote: {text!r}"
        raise ValueError(msg)
    body = f"# gems-blanking-v2 per-user config. Not committed, not synced.\ngems_root = '{text}'\n"
    atomic_write_text(config_path(), body)
    return config_path()


def _windows_drive_candidates() -> list[Path]:
    """Return plausible Windows shared-drive mount points, cheaply."""
    out: list[Path] = []
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        base = Path(f"{letter}:/")
        try:
            if not base.is_dir():
                continue
        except OSError:  # pragma: no cover - a disconnected letter can raise
            continue
        shared = base / "Shared drives"
        out.extend([shared] if shared.is_dir() else [])
    return out


def _macos_drive_candidates() -> list[Path]:
    """Return plausible macOS Drive-for-desktop mount points."""
    cloud = Path.home() / "Library" / "CloudStorage"
    if not cloud.is_dir():
        return []
    return [
        p / "Shared drives" for p in cloud.glob("GoogleDrive-*") if (p / "Shared drives").is_dir()
    ]


def _scan_for_marker() -> Path | None:
    """Look for a ``.gems-root`` marker one level under each mounted shared drive.

    The drive's own folder name is never reconstructed: this lab's drive is called
    ``BIONICs Lab: Enteric Interfaces Team`` and ``:`` is illegal in a Windows path,
    so Drive for desktop substitutes it and the folder is **not** that string on
    Windows. Whatever it is called, the marker identifies it.
    """
    for shared in _windows_drive_candidates() + _macos_drive_candidates():
        try:
            entries = sorted(shared.iterdir())
        except OSError:  # pragma: no cover - unmounted mid-scan
            continue
        for drive in entries:
            if (drive / MARKER_NAME).is_file():
                return drive
    return None


def find_gems_root(explicit: Path | None = None, *, scan: bool = True) -> Path:
    """Resolve ``gems_root``, in order: argument, ``GEMS_ROOT``, config, marker scan.

    Parameters
    ----------
    explicit
        A caller-supplied root, checked like any other.
    scan
        Whether to look under mounted shared drives when nothing else answers.
        Tests pass False to stay off the real filesystem.

    Returns
    -------
    pathlib.Path
        The resolved root, guaranteed to contain a :data:`MARKER_NAME` file.

    Raises
    ------
    FileNotFoundError
        If no candidate carries the marker. The message lists what was tried -
        a silent wrong root is the failure this whole module exists to prevent.
    """
    tried: list[str] = []
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if (env := os.environ.get("GEMS_ROOT")) is not None:
        candidates.append(Path(env))
    if (cfg := read_config_root()) is not None:
        candidates.append(cfg)

    for candidate in candidates:
        if (candidate / MARKER_NAME).is_file():
            return candidate
        tried.append(str(candidate))

    if scan and (found := _scan_for_marker()) is not None:
        return found
    if scan:
        tried.append("mounted shared drives")

    # Listed one per line, not as a repr: a repr doubles every backslash, so a
    # Windows user cannot paste the path back out of their own error message.
    listed = "\n".join(f"  - {t}" for t in tried) or "  - (nothing to try)"
    msg = (
        f"no gems_root found: no {MARKER_NAME} marker in any of:\n{listed}\n"
        f"Set one with GEMS_ROOT, or run 'gems init --root <path>' to write {config_path()}."
    )
    raise FileNotFoundError(msg)


def cache_dir() -> Path:
    """Return the **local** content-addressed cache directory.

    Never inside ``gems_root``: envelopes and feature matrices are large and
    regenerable, and syncing 16 GB of them to every lab member is a real risk.
    :meth:`GemsStore.assert_cache_is_outside` enforces it against a concrete root.
    """
    return Path(platformdirs.user_cache_dir(CONFIG_APP, appauthor=False))


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def _git_config(key: str) -> str | None:
    """Return a git config value, or None if git is absent or the key is unset."""
    try:
        done = subprocess.run(
            ["git", "config", "--get", key],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git installed
        return None
    value = done.stdout.strip()
    return value or None


@dataclass(frozen=True, slots=True)
class UserIdentity:
    """Who is acting, and how confidently we know.

    Attributes
    ----------
    user_id
        The filename-safe identifier recorded in labels and registry lines.
    source
        Where it came from. ``"fallback"`` means nothing identified the user and the
        OS account name was used - ``gems doctor`` warns, because "who labelled this"
        is scientific metadata, not a formality.
    """

    user_id: str
    source: str

    @property
    def is_confident(self) -> bool:
        """Whether the identity came from a deliberate setting rather than the OS."""
        return self.source != "fallback"


def resolve_user_id(explicit: str | None = None) -> UserIdentity:
    """Resolve the acting user from an explicit setting, then git, then the OS."""
    if explicit:
        return UserIdentity(safe_component(explicit), "explicit")
    if (email := _git_config("user.email")) is not None:
        return UserIdentity(safe_component(email.split("@")[0]), "git user.email")
    if (name := _git_config("user.name")) is not None:
        return UserIdentity(safe_component(name), "git user.name")
    return UserIdentity(
        safe_component(os.environ.get("USERNAME") or os.environ.get("USER") or ""), "fallback"
    )


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FileRef:
    """One file inside ``gems_root``, with the checksum a reader must verify.

    Checksums live in the document that *references* a file - a corpus spec, a
    model's ``provenance.json`` - rather than in a sidecar next to it. A sidecar per
    file would double the item count against the 500,000 shared-drive cap, which is
    the one limit this layout can realistically hit.

    Attributes
    ----------
    rel_path
        POSIX path relative to ``gems_root``. Never absolute (rule 2).
    sha256
        Lowercase hex digest of the file's bytes.
    size_bytes
        Expected size; a cheap first check before hashing a 1.3 GB recording.
    """

    rel_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Reject an absolute or escaping path at construction time."""
        pure = PurePosixPath(self.rel_path)
        if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", self.rel_path):
            msg = f"rel_path must be relative and inside the root, got {self.rel_path!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Outcome of a preflight check. ``ok`` is the only thing a caller may act on."""

    root: Path
    platform: str
    missing: list[str] = field(default_factory=list)
    """Relative paths the manifest names that are not present."""
    corrupt: list[str] = field(default_factory=list)
    """Relative paths whose size or checksum disagrees with the manifest."""
    syncing: list[str] = field(default_factory=list)
    """Paths that look like Drive placeholders - present but not yet materialised."""
    absolute_paths: list[str] = field(default_factory=list)
    """Stored paths that are absolute, which cannot resolve on another machine."""
    longest_path: str = ""
    longest_path_len: int = 0
    path_too_long: str = ""
    """Why the deepest generated path will not fit, or ``""`` if it fits.

    Blocking, not advisory: a root this long produces artifacts a Windows colleague
    cannot open, and cross-platform rule 5 says to fail with a clear message rather
    than let it surface as an ``OSError`` on someone else's machine. It blocks on
    every platform for the same reason - the machine that creates the path is not
    the one that fails on it.
    """
    writable: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the run may start. Incomplete corpora never pass."""
        return not (
            self.missing
            or self.corrupt
            or self.syncing
            or self.absolute_paths
            or self.path_too_long
        )

    def summary(self) -> str:
        """Return a human-readable multi-line summary, for ``gems doctor``."""
        lines = [
            f"root:        {self.root}",
            f"platform:    {self.platform}",
            f"writable:    {'yes' if self.writable else f'NO - {_CONTRIBUTOR_HINT}'}",
            f"longest path: {self.longest_path_len} chars  {self.longest_path}",
        ]
        for label, items in (
            ("missing", self.missing),
            ("checksum/size mismatch", self.corrupt),
            ("still syncing", self.syncing),
            ("absolute path stored", self.absolute_paths),
        ):
            if items:
                lines.append(f"{label} ({len(items)}):")
                lines.extend(f"    {i}" for i in items)
        if self.path_too_long:
            lines.append(f"path too long: {self.path_too_long}")
        lines.extend(f"note: {n}" for n in self.notes)
        lines.append(f"status:      {'OK' if self.ok else 'REFUSE'}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GemsStore:
    """The shared-drive layout, rooted at a discovered ``gems_root``.

    Every accessor returns an absolute local path; every path *stored* in a file
    goes through :meth:`relpath` first.
    """

    root: Path

    @classmethod
    def discover(cls, explicit: Path | None = None, *, scan: bool = True) -> GemsStore:
        """Construct a store at the resolved :func:`find_gems_root`."""
        return cls(find_gems_root(explicit, scan=scan))

    @classmethod
    def initialise(cls, root: Path) -> GemsStore:
        """Create the layout and the marker at ``root``. Safe to re-run."""
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "trials", "labels", "corpora", "models", "registry"):
            (root / sub).mkdir(exist_ok=True)
        marker = root / MARKER_NAME
        if not marker.is_file():
            atomic_write_text(marker, "gems-blanking-v2 root\n")
        return cls(root)

    # -- layout ------------------------------------------------------------

    @property
    def marker(self) -> Path:
        """Path of the ``.gems-root`` marker file."""
        return self.root / MARKER_NAME

    @property
    def registry_dir(self) -> Path:
        """Directory holding the append-only registry log and its shards."""
        return self.root / "registry"

    def session_dir(self, animal: str, session: str) -> Path:
        """Raw data directory for one session."""
        return self.root / "data" / validate_component(animal) / validate_component(session)

    def trials_path(self, animal: str, session: str) -> Path:
        """One JSONL per session, appended per trial by the acquisition machine.

        Not one file per trial: that is how a layout reaches the 500,000-item cap,
        and many tiny files is also the worst case for Drive sync.
        """
        return (
            self.root
            / "trials"
            / validate_component(animal)
            / validate_component(session)
            / "trials.jsonl"
        )

    def labels_path(self, animal: str, user: str, stamp: str | None = None) -> Path:
        """Per-user, write-once label file for one animal."""
        name = f"events_{safe_component(user)}_{stamp or utc_stamp()}.parquet"
        return self.root / "labels" / validate_component(animal) / name

    def corpus_path(self, corpus_id: str) -> Path:
        """Immutable named corpus spec."""
        return self.root / "corpora" / f"{validate_component(corpus_id)}.json"

    def model_dir(self, model_id: str) -> Path:
        """Immutable model directory, named by content hash."""
        return self.root / "models" / validate_component(model_id)

    # -- paths -------------------------------------------------------------

    def relpath(self, path: Path) -> str:
        """Return ``path`` as a POSIX string relative to the root (rule 2).

        Raises
        ------
        ValueError
            If ``path`` is outside the root. Storing such a path would produce a
            record that cannot resolve on anyone else's machine.
        """
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError as exc:
            msg = f"{path} is outside gems_root {self.root}; refusing to store it"
            raise ValueError(msg) from exc

    def abspath(self, rel_path: str) -> Path:
        """Resolve a stored POSIX relative path against this machine's root."""
        pure = PurePosixPath(rel_path)
        if pure.is_absolute() or ".." in pure.parts:
            msg = f"stored path must be relative and inside the root, got {rel_path!r}"
            raise ValueError(msg)
        return self.root.joinpath(*pure.parts)

    def deepest_path(
        self, model_id: str = "0" * 32, feature: str = "cross_channel_commonality"
    ) -> Path:
        """Return the longest path this layout can generate, for the MAX_PATH check.

        The deepest known leaf is a SHAP page inside a model directory:
        ``models/<32-hex>/shap/<feature>.html``.
        """
        return self.model_dir(model_id) / "shap" / f"{feature}.html"

    def check_path_length(self, path: Path) -> str | None:
        """Return a message if ``path`` exceeds the Windows limit, else None.

        Reported on every platform, not just Windows: the point is that the machine
        which *creates* an over-long path is usually not the one that fails on it.
        """
        length = len(str(path))
        if length > WINDOWS_MAX_PATH:
            return (
                f"path is {length} characters, over the Windows limit of {WINDOWS_MAX_PATH}: "
                f"{path}. Shorten gems_root or enable long paths before onboarding Windows."
            )
        return None

    def assert_cache_is_outside(self) -> None:
        """Raise if the local cache would land inside ``gems_root``."""
        cache = cache_dir().resolve()
        root = self.root.resolve()
        if cache == root or root in cache.parents:
            msg = f"local cache {cache} is inside gems_root {root}; it would be synced to everyone"
            raise ValueError(msg)

    # -- integrity ---------------------------------------------------------

    def is_placeholder(self, path: Path) -> bool:
        """Whether ``path`` looks like a Drive file that has not materialised yet.

        Drive for desktop can list a streamed file whose bytes are not local. A
        zero-byte stand-in for a file a manifest says is larger is the observable
        symptom; the manifest comparison in :meth:`preflight` is what makes it
        detectable at all.
        """
        try:
            return path.is_file() and path.stat().st_size == 0
        except OSError:  # pragma: no cover - vanished mid-check
            return True

    def probe_writable(self) -> bool:
        """Attempt a real write at the root and clean it up.

        There is no filesystem API for a Drive *role*, so this is a probe, not a
        query: a Contributor sees a read-only mount and this returns False.
        """
        probe = self.root / f".gems-write-probe-{os.getpid()}"
        try:
            probe.write_bytes(b"")
            probe.unlink()
        except OSError:
            return False
        return True

    def count_items(self, cap: int = 50_000) -> tuple[int, bool]:
        """Count files and directories under the root, stopping at ``cap``.

        Returns ``(count, complete)``. A full walk of a 500,000-item streamed shared
        drive is far too slow for a ``doctor`` run, so the default is a bound: when
        ``complete`` is False the real number is only known to be at least ``count``.
        Trash is **not** visible through the filesystem and is not counted, although
        it does count against the cap - the shortfall is reported as a note.
        """
        count = 0
        for _dirpath, dirnames, filenames in os.walk(self.root):
            count += len(dirnames) + len(filenames)
            if count >= cap:
                return cap, False
        return count, True

    def verify(self, ref: FileRef) -> str | None:
        """Verify one manifest entry; return None if clean, else why it is not.

        A mismatch is an error, never a warning: training on a truncated parquet is
        indistinguishable from training on a bad one.
        """
        path = self.abspath(ref.rel_path)
        if not path.is_file():
            return "missing"
        size = path.stat().st_size
        if size != ref.size_bytes:
            return f"size {size} != manifest {ref.size_bytes}"
        digest = sha256_file(path)
        if digest != ref.sha256:
            return f"sha256 {digest[:12]}... != manifest {ref.sha256[:12]}..."
        return None

    def preflight(
        self,
        manifest: list[FileRef] | None = None,
        *,
        stored_paths: list[str] | None = None,
        platform_name: str | None = None,
    ) -> PreflightReport:
        """Check everything a run needs before it starts.

        Parameters
        ----------
        manifest
            Files the run will read, with their checksums. Every one is verified.
        stored_paths
            Paths taken from records that will be written or read; any absolute one
            is reported, because it cannot resolve on another machine.
        platform_name
            Overridable for tests; defaults to the running platform.

        Returns
        -------
        PreflightReport
            ``ok`` is False if anything is missing, corrupt, still syncing, or
            absolute. Refusing here is the point: a model trained on a half-synced
            dataset is indistinguishable from a bad model.
        """
        report_missing: list[str] = []
        report_corrupt: list[str] = []
        report_syncing: list[str] = []

        for ref in manifest or []:
            path = self.abspath(ref.rel_path)
            if not path.exists():
                report_missing.append(ref.rel_path)
                continue
            if ref.size_bytes > 0 and self.is_placeholder(path):
                report_syncing.append(ref.rel_path)
                continue
            if (why := self.verify(ref)) is not None:
                report_corrupt.append(f"{ref.rel_path}: {why}")

        absolute = [
            p
            for p in (stored_paths or [])
            if PurePosixPath(p).is_absolute() or re.match(r"^[A-Za-z]:", p) or "\\" in p
        ]

        deepest = self.deepest_path()
        return PreflightReport(
            root=self.root,
            platform=platform_name or f"{platform.system()} {platform.release()}",
            missing=report_missing,
            corrupt=report_corrupt,
            syncing=report_syncing,
            absolute_paths=absolute,
            longest_path=str(deepest),
            longest_path_len=len(str(deepest)),
            path_too_long=self.check_path_length(deepest) or "",
            writable=self.probe_writable(),
            notes=[],
        )
