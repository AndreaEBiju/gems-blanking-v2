"""Locate and import ``GEMSBlanking``'s ``detector`` package, without forking it.

``CLAUDE.md`` says to import the reused modules rather than copy them, and to
verify every path before importing. Two facts shape how that is done here:

* **It is not pip-installable into this environment.** ``GEMSBlanking``'s
  ``pyproject.toml`` declares ``detector-core`` with ``numpy>=1.26,<2`` and
  ``requires-python >=3.12``; this project runs numpy 2.x and supports 3.11. An
  editable install would downgrade numpy underneath the whole test suite and drag
  in lightgbm, shap and optuna. So the sibling checkout goes on ``sys.path``
  instead, which was verified to work: ``detector.recording_io`` needs only numpy
  and h5py, and ``detector/__init__.py`` wraps its ``hdf5plugin`` import in
  ``try/except ImportError``.
* **It is private, so CI cannot see it.** Every import here is lazy and every
  failure is a clear message, so this package imports and its tests run on a
  machine that has never heard of ``GEMSBlanking``.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

__all__ = [
    "DETECTOR_CORE_ENV",
    "SIBLING_NAME",
    "detector_core_available",
    "find_detector_core",
    "import_detector_module",
]

DETECTOR_CORE_ENV: Final = "GEMS_DETECTOR_CORE"
"""Environment variable naming the ``GEMSBlanking`` checkout, if it is elsewhere."""

SIBLING_NAME: Final = "GEMSBlanking"
"""Directory name to look for beside this repository."""

_MARKER: Final = Path("detector") / "recording_io.py"
"""What makes a directory a ``GEMSBlanking`` checkout. Checked rather than assumed,
because a wrong root produces an ``ImportError`` three frames deep."""


def _is_checkout(root: Path) -> bool:
    """Whether ``root`` really is a ``GEMSBlanking`` checkout."""
    return (root / _MARKER).is_file()


def _search_roots() -> list[Path]:
    """Return the roots the *implicit* search tries, in order."""
    repo_root = Path(__file__).resolve().parents[2]
    return [
        # Beside this repository: .../Documents/GEMSBlanking for .../Documents/<repo>.
        repo_root.parent / SIBLING_NAME,
        # The submodule path the PyQt UI uses, empty unless someone initialised it.
        repo_root.parent / "detector-pyqt" / "detector-core",
    ]


def find_detector_core(explicit: Path | None = None) -> Path:
    """Return the ``GEMSBlanking`` checkout root.

    An ``explicit`` argument or ``$GEMS_DETECTOR_CORE`` is **authoritative**: if it
    is given and is not a checkout, this raises rather than searching on. Naming a
    checkout and silently getting a different one is the failure mode worth
    preventing - a caller who points at a specific tree, perhaps an older one, has
    to be told it was not used rather than discover it from the results.

    With neither given, the search tries the sibling directory and then
    ``detector-pyqt/detector-core``.

    Raises
    ------
    FileNotFoundError
        Naming what was checked. The message says what to do, because this failure
        is normal on a machine without access to a private repository.
    """
    sources = (
        ("explicit root", explicit),
        (f"${DETECTOR_CORE_ENV}", os.environ.get(DETECTOR_CORE_ENV)),
    )
    for source, value in sources:
        if value is None:
            continue
        root = Path(value)
        if _is_checkout(root):
            return root
        msg = (
            f"{source} {root} is not a {SIBLING_NAME} checkout "
            f"({_MARKER.as_posix()} not found under it). Refusing to fall back to "
            "another checkout: you asked for this one."
        )
        raise FileNotFoundError(msg)

    searched: list[str] = []
    for root in _search_roots():
        if _is_checkout(root):
            return root
        searched.append(str(root))
    listed = "\n".join(f"  - {s}" for s in searched)
    msg = (
        f"no {SIBLING_NAME} checkout found (looked for {_MARKER.as_posix()} under):\n"
        f"{listed}\n"
        f"Clone it beside this repository or set {DETECTOR_CORE_ENV} to its path. "
        "It is a private repository, so this is expected to fail in CI."
    )
    raise FileNotFoundError(msg)


def detector_core_available(explicit: Path | None = None) -> bool:
    """Whether the checkout can be found. For skipping tests, not for control flow."""
    try:
        find_detector_core(explicit)
    except FileNotFoundError:
        return False
    return True


def import_detector_module(name: str, root: Path | None = None) -> ModuleType:
    """Import ``detector.<name>`` from the ``GEMSBlanking`` checkout.

    Parameters
    ----------
    name
        Submodule name, e.g. ``"recording_io"``.
    root
        Checkout root; discovered when omitted.

    Returns
    -------
    types.ModuleType
        The imported module. An already-installed ``detector`` package wins, so a
        real install is respected rather than shadowed.

    Raises
    ------
    FileNotFoundError
        If no checkout can be found.
    ImportError
        If the module exists but will not import - for instance because a
        dependency of *that* package is missing here. The message names the
        checkout, so it is clear which tree failed.
    """
    module = f"detector.{name}"
    if module in sys.modules:
        return sys.modules[module]

    checkout = str(find_detector_core(root))
    if checkout not in sys.path:
        # Appended, not inserted: an installed `detector` takes precedence.
        sys.path.append(checkout)
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - needs a broken checkout
        msg = f"found {SIBLING_NAME} at {checkout} but could not import {module}: {exc}"
        raise ImportError(msg) from exc
