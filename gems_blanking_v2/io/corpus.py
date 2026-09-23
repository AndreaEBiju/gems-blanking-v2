"""What counts as the corpus: the scan roots, separate from ``gems_root``.

**The root is a path anchor; it is not a licence to walk everything under it.**
``GEMS-Andrea`` contains ``processing_new``, ``TDTMatlabSDK``, ``nerve-processing``
and ``IACUC Inspection 092026`` - code and admin, not data. A recording outside
every scan root is not in the corpus, and adding a folder is an edit to this list
rather than an automatic consequence of someone dropping files on the drive. That
also keeps the ``SHARED_DRIVE_ITEM_CAP`` accounting honest, since the code trees
stop counting toward it.

**This lives on the drive, not in the per-user config, and that is a deviation.**
00A's example shows ``gems_root`` and ``scan_roots`` in one YAML document, but those
two settings have different scopes: ``gems_root`` is an absolute path that differs
per machine and per platform and must stay per-user, while ``scan_roots`` *defines
what the corpus is* and must be the same for everyone. Stored per-user, two lab
members would scan different folders and neither would know. Stored here, the list
is POSIX-relative to ``gems_root`` (rule 2) and shared. One line to move it back if
that is not wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from gems_blanking_v2.io.store import atomic_write_text

__all__ = [
    "CORPUS_FILENAME",
    "DEFAULT_SCAN_ROOTS",
    "ScanScope",
    "corpus_path",
    "default_scope",
    "read_scan_roots",
    "write_scan_roots",
]

CORPUS_FILENAME: Final = "corpus.yaml"
"""Lives at ``<gems_root>/corpus.yaml``, beside ``protocol.yaml``."""

DEFAULT_SCAN_ROOTS: Final[tuple[str, ...]] = ("August-September Chronic Recordings",)
"""The only folder in scope as of 2026-09-22.

The balloon trials are a different experiment with a different protocol and are
deliberately out of scope until someone adds them here on purpose.
"""


@dataclass(frozen=True, slots=True)
class ScanScope:
    """The folders under ``gems_root`` that hold recordings.

    Attributes
    ----------
    scan_roots
        POSIX-relative paths under ``gems_root``. Relative and POSIX because this
        file is read on both platforms and the root's own spelling differs between
        them (rule 2).
    """

    scan_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check every entry is a usable relative POSIX path."""
        if not self.scan_roots:
            msg = "scan_roots is empty: nothing would ever be scanned"
            raise ValueError(msg)
        for entry in self.scan_roots:
            if not entry or entry.startswith("/") or "\\" in entry or ":" in entry:
                msg = (
                    f"scan_root {entry!r} must be a relative POSIX path under "
                    "gems_root - an absolute or Windows-style path cannot resolve "
                    "on the other platform (rule 2)"
                )
                raise ValueError(msg)

    def resolve(self, gems_root: Path) -> list[Path]:
        """Return the absolute scan roots on this machine, in order."""
        return [Path(gems_root) / entry for entry in self.scan_roots]

    def covers(self, relative: str) -> str | None:
        """Return the scan root containing ``relative``, or ``None`` if outside all."""
        posix = Path(relative).as_posix()
        for entry in self.scan_roots:
            if posix == entry or posix.startswith(entry + "/"):
                return entry
        return None


def default_scope() -> ScanScope:
    """Return the scope decided on 2026-09-22: the chronic recordings only."""
    return ScanScope(scan_roots=DEFAULT_SCAN_ROOTS)


def corpus_path(gems_root: Path) -> Path:
    """Return ``<gems_root>/corpus.yaml``."""
    return Path(gems_root) / CORPUS_FILENAME


def read_scan_roots(gems_root: Path) -> ScanScope:
    """Load the scan scope.

    Raises
    ------
    FileNotFoundError
        If the file is absent. Not defaulted: silently scanning everything under the
        root is exactly what this file exists to prevent, and silently scanning one
        guessed folder would be worse.
    ValueError
        If the document is malformed or an entry is not a relative POSIX path.
    """
    path = corpus_path(gems_root)
    if not path.is_file():
        msg = (
            f"no {CORPUS_FILENAME} at {path}. Write one with "
            "write_scan_roots(root, default_scope()). It is not defaulted: the root "
            "holds code trees as well as data, so scanning everything under it is "
            "wrong, and guessing which subfolder to scan is worse."
        )
        raise FileNotFoundError(msg)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        msg = f"{path} could not be parsed: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(document, dict) or "scan_roots" not in document:
        msg = f"{path}: scan_roots is required and is absent"
        raise ValueError(msg)
    raw = document["scan_roots"]
    if not isinstance(raw, list) or not all(isinstance(e, str) for e in raw):
        msg = f"{path}: scan_roots must be a list of strings, got {raw!r}"
        raise ValueError(msg)
    return ScanScope(scan_roots=tuple(raw))


def write_scan_roots(gems_root: Path, scope: ScanScope) -> Path:
    r"""Write the scan scope atomically as UTF-8 with ``\n`` endings."""
    body = (
        "# What counts as the corpus. POSIX-relative to gems_root, shared by every\n"
        "# lab member - the root itself is per-machine and lives in the per-user\n"
        "# config, because its spelling differs between Windows and macOS.\n"
        "#\n"
        "# A recording outside every entry here is not in the corpus. Adding a folder\n"
        "# is a deliberate edit, never a consequence of someone dropping files on the\n"
        "# drive.\n"
        + yaml.safe_dump(
            {"scan_roots": list(scope.scan_roots)},
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    )
    path = corpus_path(gems_root)
    atomic_write_text(path, body)
    return path
