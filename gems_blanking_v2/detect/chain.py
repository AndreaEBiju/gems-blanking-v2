"""The detection chain: derivations -> contact screen -> z -> candidates.

ONE construction site for "what is a candidate" (invariant 33). Production and
the recall audit both call :func:`detect_region`; the audit's bridge only reshapes
its result for display. Until 2026-09-28 the chain was composed inside the audit
app, so the gate measured a composition production did not share.

:func:`generation_sha256` hashes exactly the source this chain executes - this
module and every ``gems_blanking_v2`` module it imports, transitively - and
nothing else. A budget record or a gate round is keyed to it, so an edit to the
scorer or a report cannot invalidate either, while any edit that could change a
candidate does. The scope is DERIVED from the imports, not listed, so it cannot
drift from the code (invariant 28).
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.bands.envelope import band_envelope_for, log_envelope
from gems_blanking_v2.bands.reference import epoch_reference
from gems_blanking_v2.bands.zscore import zscore
from gems_blanking_v2.constants import BANDS
from gems_blanking_v2.derive import contact_quality
from gems_blanking_v2.derive.contact_quality import ContactQuality, screened_signals
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.detect import candidates as _candidates
from gems_blanking_v2.detect.candidates import CandidateReport
from gems_blanking_v2.physio.rpeaks import detect_rpeaks
from gems_blanking_v2.types import Recording

_log = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]
Pair = tuple[str, str]

BEAT_SIGNAL = "R_T"
"""The derived signal beats are detected on: the right-cuff tripole."""


def z_by_pair(
    data: npt.ArrayLike, fs: float, names: list[str], bands: tuple[str, ...] | None = None
) -> dict[Pair, F64]:
    """Robust z for every ``(signal, band)`` pair, each against ITS OWN reference.

    Each signal is thresholded against its own whole-region log-envelope reference
    (invariants 3 and 5): borrowing one signal's sigma for another was measured at
    1136 true / 19,624 false detections.
    """
    arr = np.asarray(data, dtype=np.float64)
    use = bands if bands is not None else tuple(BANDS)
    out: dict[Pair, F64] = {}
    for col, name in enumerate(names):
        x = arr[:, col]
        for band in use:
            log_env = log_envelope(band_envelope_for(x, fs, band))
            ref = epoch_reference(log_env, signal=name, band=band)
            out[(name, band)] = zscore(log_env, ref, signal=name, band=band)
    return out


@dataclass(frozen=True)
class RegionDetection:
    """What the chain found in one region. Times in seconds.

    ``intervals`` are on the RECORDING's timeline (offset by ``region[0]``);
    ``z`` is on the region's own 10 ms grid, frame 0 at ``region[0]``.
    """

    region: tuple[float, float]
    intervals: F64
    report: CandidateReport
    z: dict[Pair, F64]
    signals: tuple[str, ...]
    """The detection signals that entered the max, sorted."""
    screened: frozenset[str]
    """Signals the contact screen removed (a failed contact's V and its cuff's T)."""
    quality: dict[str, ContactQuality]


def detect_region(
    rec: Recording, region: tuple[float, float], *, z_enter: float | None = None
) -> RegionDetection:
    """Run the chain over ``region`` of ``rec``.

    Everything is computed on the REGION - the baseline, or a stim/recovery file's
    recovery epoch - so the stim epoch never enters a reference. The contact screen
    reads the region's first 120 s. ``z_enter`` defaults to ``candidate_report``'s
    own default: the value is never restated here (invariant 39).
    """
    lo, hi = region
    i0, i1 = round(lo * rec.fs), round(hi * rec.fs)
    sub = replace(rec, data=rec.data[i0:i1])
    signals, _weights = build_derivations(sub)
    quality = contact_quality.assess_contacts(sub)
    dropped = screened_signals(quality)
    if dropped:
        why = {k: q.reasons for k, q in quality.items() if q.screened}
        _log.warning("contact screen removed %s from detection (%s)", sorted(dropped), why)
    names = sorted(n for n in signals if n not in dropped)
    z = z_by_pair(np.column_stack([signals[n] for n in names]), float(rec.fs), names)
    beats = detect_rpeaks(signals[BEAT_SIGNAL], float(rec.fs))
    kw: dict[str, Any] = {} if z_enter is None else {"z_enter": z_enter}
    report = _candidates.candidate_report(z, beats, **kw)
    spans = [(c.start_s + lo, c.stop_s + lo) for c in report.candidates]
    intervals = (np.asarray(spans, dtype=np.float64) if spans
                 else np.zeros((0, 2), dtype=np.float64))
    return RegionDetection(region=(float(lo), float(hi)), intervals=intervals, report=report,
                           z=z, signals=tuple(names), screened=dropped, quality=quality)


# ---------------------------------------------------------------------------
# the generation hash (scope ruled 2026-09-28)
# ---------------------------------------------------------------------------

PACKAGE = "gems_blanking_v2"
ENTRY = f"{PACKAGE}.detect.chain"


def _module_file(package_dir: Path, module: str) -> Path | None:
    parts = module.split(".")[1:]
    if not parts:
        top = package_dir / "__init__.py"
        return top if top.is_file() else None
    rel = Path(*parts)
    for cand in (package_dir / rel.with_suffix(".py"), package_dir / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _imports(path: Path, module: str) -> set[str]:
    """Package modules ``path`` imports, absolute or relative, at any depth."""
    tree = ast.parse(path.read_bytes())
    here = module.split(".")
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here[: len(here) - node.level + (path.name == "__init__.py")]
                mod = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                mod = node.module or ""
            out.add(mod)
            out |= {f"{mod}.{a.name}" for a in node.names}  # "from pkg import module"
    return {m for m in out if m == PACKAGE or m.startswith(PACKAGE + ".")}


def generation_modules(package_dir: Path | None = None, entry: str = ENTRY) -> list[str]:
    """Every package module the chain executes: ``entry`` and its import closure.

    Each module's parent packages are included too, since importing a submodule
    runs its packages' ``__init__``. Sorted, as dotted names.
    """
    root = package_dir or Path(__file__).resolve().parents[1]
    seen: set[str] = set()
    todo = [entry]
    while todo:
        mod = todo.pop()
        parts = mod.split(".")
        for k in range(1, len(parts) + 1):
            name = ".".join(parts[:k])
            if name in seen:
                continue
            path = _module_file(root, name)
            if path is None:
                continue  # "from pkg import function": not a module
            seen.add(name)
            todo.extend(_imports(path, name))
    return sorted(seen)


def generation_sha256(package_dir: Path | None = None, entry: str = ENTRY) -> str:
    """SHA-256 over the chain's own source: :func:`generation_modules`, LF-normalised.

    Each file contributes its package-relative POSIX path and its bytes with CRLF
    folded to LF, so Windows and macOS checkouts agree.
    """
    root = package_dir or Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for name in generation_modules(root, entry):
        path = _module_file(root, name)
        assert path is not None
        h.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()
