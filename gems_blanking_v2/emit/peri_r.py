"""RULING 2026-10-08 (k) 1: peri-R NaN spans in the spike consumer's mask.

**The rule.** One constant window for every file: the spans ``[R - before, R + after)``
around every beat of the recording's routed beat train are NaN in the spike consumer's
input, and in no other consumer's (hard invariant 2). ``before`` and ``after`` are the
MAXIMUM of the routed ``peri_r_ms`` over every routing table, each side measured
separately, so no file is under-blanked. A recording with no train has no peri-R spans;
its spike mask is exactly what it would be without this module.

**The window is a declared input**, never a literal (invariant 33): the JSON file written
by the window measurement (``schema`` :data:`WINDOW_SCHEMA`, ``constant_window.before_ms``
and ``after_ms``), read by :func:`load_peri_r_window`, which keeps the file's sha256 and
name. Both travel into each mask's provenance (``spike_peri_r``).

**Samples, not frames.** The spans are sample-exact. They cannot live on the 10 ms motion
grid - a 25 ms window would round to 30-40 ms per beat - so, like the line distrust, they
join ``blank_spikes_<signal>`` in the handoff (``emit.handoff``) and keep their own record
(``perir_spikes_<signal>``, ``perir_json``). The motion masks, their retention and the
20% / 3x hold are unchanged.

**One conversion each** (invariants 15, 22):

* a beat is a 1-based sample index ``h`` from its train's origin sample ``o``; it is the
  0-based epoch sample ``r = h - 1 + o - i0`` for an epoch starting at sample ``i0``
  (:func:`beat_epoch_samples` - the only place the 1-based index is converted);
* the widths are ``nb = seconds_to_sample(before_ms / 1000, fs)`` and
  ``na = seconds_to_sample(after_ms / 1000, fs)``, the same for every beat;
* the span is ``[r - nb, r + na)``, 0-based half-open, clipped to ``[0, n_samples)``
  (a beat just outside the epoch still blanks the part of its span inside), and
  overlapping or touching spans are merged (:func:`peri_r_sample_spans`). In MATLAB's
  1-based inclusive form that is ``[r - nb + 1, r + na]``.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.extent.grid import seconds_to_sample, to_matlab_inclusive

__all__ = [
    "PERI_R_RULE",
    "WINDOW_SCHEMA",
    "PeriRRecord",
    "PeriRWindow",
    "beat_epoch_samples",
    "build_peri_r",
    "load_peri_r_window",
    "peri_r_sample_spans",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

WINDOW_SCHEMA: Final = "perir_window/1"
"""The window file's schema tag (written by the measurement, ``perir_window.py``)."""
PERI_R_RULE: Final = ("RULING 2026-10-08 (k) 1: NaN [R - before, R + after) around every beat "
                      "of the recording's routed train, spike consumer only; one constant "
                      "window, the maximum of the routed peri_r_ms on each side")


@dataclass(frozen=True)
class PeriRWindow:
    """The constant peri-R window, milliseconds, with the file it was declared in."""

    before_ms: float
    after_ms: float
    source: str
    """The window file's name (never an absolute path: cross-platform rule 2)."""
    sha256: str

    def __post_init__(self) -> None:
        """Refuse a non-finite or empty window."""
        for name in ("before_ms", "after_ms"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                msg = f"peri-R window {name} must be a finite number, got {v!r}"
                raise ValueError(msg)
        if not self.before_ms + self.after_ms > 0:
            msg = (f"peri-R window [-{self.before_ms}, {self.after_ms}) ms holds no time: "
                   "before + after must be > 0")
            raise ValueError(msg)
        if len(self.sha256) != 64:  # noqa: PLR2004 - a SHA-256 hex digest
            msg = f"peri-R window sha256 must be 64 hex characters, got {self.sha256!r}"
            raise ValueError(msg)

    def samples(self, fs: float) -> tuple[int, int]:
        """Return ``(nb, na)``: the widths before and after R in samples (one rounding)."""
        return (seconds_to_sample(self.before_ms / 1000.0, fs),
                seconds_to_sample(self.after_ms / 1000.0, fs))

    def provenance(self) -> dict[str, Any]:
        """Return the window as it travels into a mask's provenance."""
        return {"before_ms": float(self.before_ms), "after_ms": float(self.after_ms),
                "file": self.source, "sha256": self.sha256}


def load_peri_r_window(path: Path) -> PeriRWindow:
    """Read the declared window file; its sha256 is of the bytes read.

    Raises, naming the field, when the schema is not :data:`WINDOW_SCHEMA` or
    ``constant_window.before_ms`` / ``after_ms`` is absent or not a finite number.
    """
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("schema") != WINDOW_SCHEMA:
        msg = f"{Path(path).name}: schema {doc.get('schema')!r} is not {WINDOW_SCHEMA!r}"
        raise ValueError(msg)
    cw = doc.get("constant_window")
    if not isinstance(cw, Mapping):
        msg = f"{Path(path).name}: required field 'constant_window' is absent"
        raise ValueError(msg)
    for key in ("before_ms", "after_ms"):
        if cw.get(key) is None:
            msg = f"{Path(path).name}: required field 'constant_window.{key}' is absent"
            raise ValueError(msg)
    return PeriRWindow(before_ms=cw["before_ms"], after_ms=cw["after_ms"],
                       source=Path(path).name, sha256=hashlib.sha256(raw).hexdigest())


def beat_epoch_samples(heartlocs: npt.ArrayLike, *, origin_sample: int,
                       epoch_start_sample: int) -> I64:
    """Return 1-based beat samples from ``origin_sample`` as 0-based epoch samples.

    ``heartlocs`` are the beats file's 1-based indices into the signal that starts at
    file sample ``origin_sample`` (0-based); the epoch starts at file sample
    ``epoch_start_sample``. The result may lie outside the epoch; the caller clips the
    spans, not the beats. Raises for a non-integer or non-positive index.
    """
    h = np.asarray(heartlocs, dtype=np.float64).ravel()
    if h.size and (np.any(h != np.round(h)) or h.min() < 1):
        msg = "heartlocs must be integer 1-based sample indices (>= 1)"
        raise ValueError(msg)
    return h.astype(np.int64) - 1 + int(origin_sample) - int(epoch_start_sample)


def peri_r_sample_spans(r: npt.ArrayLike, window: PeriRWindow, *, fs: float,
                        n_samples: int) -> list[tuple[int, int]]:
    """Return the merged ``[r - nb, r + na)`` spans, 0-based half-open, clipped to the epoch."""
    nb, na = window.samples(fs)
    rr = np.sort(np.asarray(r, dtype=np.int64).ravel())
    a = np.clip(rr - nb, 0, n_samples)
    b = np.clip(rr + na, 0, n_samples)
    out: list[tuple[int, int]] = []
    for k0, k1 in zip(a.tolist(), b.tolist(), strict=True):
        if k1 <= k0:
            continue
        if out and k0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], k1))
        else:
            out.append((k0, k1))
    return out


@dataclass(frozen=True)
class PeriRRecord:
    """One epoch's peri-R spans for the spike consumer, with what made them.

    ``signals`` are the spike consumer's signals in this epoch (the spans apply to each).
    ``train`` is the routed train's provenance record, or None when the recording has no
    train - then ``spans`` is empty and the spike mask is unchanged.
    """

    recording: str
    signals: tuple[str, ...]
    window: PeriRWindow
    fs: float
    n_samples: int
    epoch_start_s: float
    epoch_start_sample: int
    spans: tuple[tuple[int, int], ...]
    train: Mapping[str, Any] | None
    n_beats_in_epoch: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Refuse spans without a train, and spans outside the epoch or unsorted."""
        if self.train is None and self.spans:
            msg = f"{self.recording}: peri-R spans without a routed train"
            raise ValueError(msg)
        prev = 0
        for k0, k1 in self.spans:
            if not (prev <= k0 < k1 <= self.n_samples):
                msg = f"{self.recording}: peri-R spans must be sorted, disjoint, inside the epoch"
                raise ValueError(msg)
            prev = k1

    def sample_spans(self, signal: str) -> list[tuple[int, int]]:
        """Return the spans for ``signal``: all of them for a spike signal, none otherwise."""
        return list(self.spans) if signal in self.signals else []

    def matlab_spans(self) -> F64:
        """Return the spans as MATLAB rows ``[k0 + 1, k1]``, 1-based inclusive (N x 2)."""
        if not self.spans:
            return np.zeros((0, 2), dtype=np.float64)
        return to_matlab_inclusive([a for a, _ in self.spans], [b for _, b in self.spans])

    def provenance(self) -> dict[str, Any]:
        """Return the provenance: rule, window (with its file's hash), train, counts."""
        nb, na = self.window.samples(self.fs)
        rec: dict[str, Any] = {"rule": PERI_R_RULE, "window": self.window.provenance(),
                               "window_samples": {"before": nb, "after": na},
                               "signals": list(self.signals),
                               "n_beats_in_epoch": int(self.n_beats_in_epoch),
                               "n_spans": len(self.spans),
                               "samples_blanked": int(sum(b - a for a, b in self.spans))}
        if self.train is None:
            rec["train"] = "none: the routing gives this recording no beat train; no spans"
        else:
            rec["train"] = dict(self.train)
        if self.notes:
            rec["notes"] = list(self.notes)
        return rec

    def to_json(self) -> str:
        """Canonical, ASCII-escaped JSON of the provenance and the spans (0-based half-open)."""
        doc = {**self.provenance(), "recording": self.recording, "fs": self.fs,
               "n_samples": self.n_samples, "epoch_start_s": self.epoch_start_s,
               "epoch_start_sample": self.epoch_start_sample,
               "spans_0based_halfopen": [list(s) for s in self.spans]}
        return json.dumps(doc, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                          allow_nan=False)


def build_peri_r(*, recording: str, signals: Sequence[str], window: PeriRWindow, fs: float,
                 n_samples: int, epoch_start_s: float, epoch_start_sample: int,
                 heartlocs: npt.ArrayLike | None, origin_sample: int | None,
                 train: Mapping[str, Any] | None,
                 notes: Sequence[str] = ()) -> PeriRRecord:
    """Return one epoch's :class:`PeriRRecord`.

    ``heartlocs`` (1-based, from ``origin_sample``) and ``train`` (its provenance) are both
    given or both None. ``epoch_start_sample`` must equal ``seconds_to_sample`` of
    ``epoch_start_s`` - the handoff's own rule - or this raises.
    """
    if (heartlocs is None) != (train is None) or (heartlocs is None) != (origin_sample is None):
        msg = "heartlocs, origin_sample and train are given together or not at all"
        raise ValueError(msg)
    if seconds_to_sample(epoch_start_s, fs) != int(epoch_start_sample):
        msg = (f"epoch_start_sample {epoch_start_sample} disagrees with round({epoch_start_s} s "
               f"x {fs} Hz)")
        raise ValueError(msg)
    if heartlocs is None or origin_sample is None or train is None:
        return PeriRRecord(recording, tuple(signals), window, float(fs), int(n_samples),
                           float(epoch_start_s), int(epoch_start_sample), (), None, 0,
                           tuple(notes))
    r = beat_epoch_samples(heartlocs, origin_sample=origin_sample,
                           epoch_start_sample=epoch_start_sample)
    spans = peri_r_sample_spans(r, window, fs=fs, n_samples=n_samples)
    inside = int(((r >= 0) & (r < n_samples)).sum())
    return PeriRRecord(recording, tuple(signals), window, float(fs), int(n_samples),
                       float(epoch_start_s), int(epoch_start_sample), tuple(spans), dict(train),
                       inside, tuple(notes))
