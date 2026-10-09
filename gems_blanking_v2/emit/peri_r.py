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

**Per recording class (RULING 2026-10-09 (b) 1, replaces (k) 1's "maximum over all
files").** The window file (schema :data:`WINDOW_SCHEMA_V2`, read by
:func:`load_peri_r_windows`) declares a DEFAULT window (11.5 ms before R, 9.5 ms after) and,
per recording, the windows that differ from it: the ruled exception
(gems_a_t02_2_3_bl_215610, both cuffs, 16.0 / 9.5 ms) and every recording whose routed extent
exceeds the default, at its own extent - listed, never silently under-blanked.
:meth:`PeriRWindowTable.for_recording` returns the window that applies to one recording, and
the (k) 1 refusal (:func:`routed_outside_window`) checks the recording against THAT window.
The file's sha256 travels into provenance.

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
    "PERI_R_RULE_BY_CLASS",
    "WINDOW_CLASSES",
    "WINDOW_SCHEMA",
    "WINDOW_SCHEMA_V2",
    "PeriRRecord",
    "PeriRWindow",
    "PeriRWindowTable",
    "beat_epoch_samples",
    "build_peri_r",
    "load_peri_r_window",
    "load_peri_r_windows",
    "peri_r_sample_spans",
    "routed_outside_window",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

WINDOW_SCHEMA: Final = "perir_window/1"
"""The window file's schema tag (written by the measurement, ``perir_window.py``)."""
PERI_R_RULE: Final = ("RULING 2026-10-08 (k) 1: NaN [R - before, R + after) around every beat "
                      "of the recording's routed train, spike consumer only; one constant "
                      "window, the maximum of the routed peri_r_ms on each side")
WINDOW_SCHEMA_V2: Final = "perir_window/2"
"""The per-recording-class window file (RULING 2026-10-09 (b) 1)."""
WINDOW_CLASSES: Final = ("default", "ruled_exception", "own_routed_extent")
"""``default``: 11.5 / 9.5 ms; ``ruled_exception``: a window Andrea named for one recording;
``own_routed_extent``: a recording whose routed extent exceeds the default, at that extent."""
PERI_R_RULE_BY_CLASS: Final = (
    "RULING 2026-10-09 (b) 1 (replaces (k) 1's maximum over all files): NaN [R - before, "
    "R + after) around every beat of the recording's routed train, spike consumer only; the "
    "window is 11.5 ms before R and 9.5 ms after for every recording, except "
    "gems_a_t02_2_3_bl_215610 (both cuffs, 16.0 / 9.5 ms) and any recording whose routed "
    "extent exceeds the default, which uses its own routed extent (listed in the window "
    "file); a recording whose routed extent exceeds the window that applies to it is refused")


@dataclass(frozen=True)
class PeriRWindow:
    """The constant peri-R window, milliseconds, with the file it was declared in."""

    before_ms: float
    after_ms: float
    source: str
    """The window file's name (never an absolute path: cross-platform rule 2)."""
    sha256: str
    window_class: str = "constant"
    """``constant`` (a schema-1 file) or one of :data:`WINDOW_CLASSES`."""
    recording: str = ""
    """The recording a per-recording window applies to ("" for a constant one)."""

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
        out: dict[str, Any] = {"before_ms": float(self.before_ms),
                               "after_ms": float(self.after_ms),
                               "file": self.source, "sha256": self.sha256}
        if self.window_class != "constant":
            out |= {"class": self.window_class, "recording": self.recording,
                    "rule": PERI_R_RULE_BY_CLASS}
        return out


@dataclass(frozen=True)
class PeriRWindowTable:
    """The declared windows by recording class (RULING 2026-10-09 (b) 1), with their file.

    ``recordings`` maps a recording id to ``(before_ms, after_ms, class, why)`` for every
    recording whose window is NOT the default; every other recording takes the default.
    """

    default_before_ms: float
    default_after_ms: float
    recordings: Mapping[str, tuple[float, float, str, str]]
    source: str
    sha256: str

    def __post_init__(self) -> None:
        """Refuse an unknown class, a non-default class named default, and a bad window."""
        PeriRWindow(self.default_before_ms, self.default_after_ms, self.source, self.sha256)
        for rid, (b, a, cls, why) in self.recordings.items():
            if cls not in WINDOW_CLASSES or cls == "default":
                msg = f"{rid}: window class {cls!r} is not one of {WINDOW_CLASSES[1:]}"
                raise ValueError(msg)
            if not why:
                msg = f"{rid}: a {cls} window must say why"
                raise ValueError(msg)
            PeriRWindow(b, a, self.source, self.sha256)

    def for_recording(self, recording: str) -> PeriRWindow:
        """Return the window that applies to ``recording`` (the default unless listed)."""
        if recording in self.recordings:
            b, a, cls, _why = self.recordings[recording]
            return PeriRWindow(b, a, self.source, self.sha256, cls, recording)
        return PeriRWindow(self.default_before_ms, self.default_after_ms, self.source,
                           self.sha256, "default", recording)

    def provenance(self) -> dict[str, Any]:
        """Return the table's identity as it travels into provenance: file, hash, classes."""
        by: dict[str, int] = {}
        for _b, _a, cls, _w in self.recordings.values():
            by[cls] = by.get(cls, 0) + 1
        return {"file": self.source, "sha256": self.sha256, "schema": WINDOW_SCHEMA_V2,
                "rule": PERI_R_RULE_BY_CLASS,
                "default": {"before_ms": float(self.default_before_ms),
                            "after_ms": float(self.default_after_ms)},
                "listed_by_class": by}


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


def load_peri_r_windows(path: Path) -> PeriRWindowTable:
    """Read a per-recording-class window file (schema :data:`WINDOW_SCHEMA_V2`).

    ``default_window.before_ms`` / ``after_ms`` and ``recording_windows`` (``{recording:
    {before_ms, after_ms, class, why}}``) are required; a missing field raises naming it. The
    sha256 is of the bytes read.
    """
    raw = Path(path).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    name = Path(path).name
    if doc.get("schema") != WINDOW_SCHEMA_V2:
        msg = f"{name}: schema {doc.get('schema')!r} is not {WINDOW_SCHEMA_V2!r}"
        raise ValueError(msg)
    dw = doc.get("default_window")
    if not isinstance(dw, Mapping):
        msg = f"{name}: required field 'default_window' is absent"
        raise ValueError(msg)
    for key in ("before_ms", "after_ms"):
        if dw.get(key) is None:
            msg = f"{name}: required field 'default_window.{key}' is absent"
            raise ValueError(msg)
    rw = doc.get("recording_windows")
    if not isinstance(rw, Mapping):
        msg = f"{name}: required field 'recording_windows' is absent"
        raise ValueError(msg)
    recs: dict[str, tuple[float, float, str, str]] = {}
    for rid, w in rw.items():
        for key in ("before_ms", "after_ms", "class", "why"):
            if not isinstance(w, Mapping) or w.get(key) is None:
                msg = f"{name}: required field 'recording_windows.{rid}.{key}' is absent"
                raise ValueError(msg)
        recs[str(rid)] = (w["before_ms"], w["after_ms"], str(w["class"]), str(w["why"]))
    return PeriRWindowTable(dw["before_ms"], dw["after_ms"], recs, name,
                            hashlib.sha256(raw).hexdigest())


def routed_outside_window(entry: Mapping[str, Any], window: PeriRWindow) -> list[str]:
    """Return every routed peri-R window of one routing entry that ``window`` does not contain.

    The routed windows are each cuff's ``spike[cuff].peri_r_ms = [a, b]`` and every
    ``peri_r_narrow_ms`` window, milliseconds relative to R (negative = before R). One is
    contained when ``-before_ms <= a`` and ``b <= after_ms``. The constant window is the
    hull of every routed window ((k) 1: "no file is under-blanked"), so a routed window it
    does not contain is a file the constant window would under-blank: the caller refuses
    that recording by name. Each item reads ``"<cuff> peri_r_ms [a, b]"`` (or
    ``peri_r_narrow_ms``); the list is empty when every routed window is contained.
    """
    out: list[str] = []
    for cuff, s in sorted(dict(entry.get("spike") or {}).items()):
        wins = [("peri_r_ms", s["peri_r_ms"])] if "peri_r_ms" in s else []
        wins += [("peri_r_narrow_ms", w) for w in s.get("peri_r_narrow_ms", [])]
        for name, (a, b) in wins:
            if not (float(a) >= -float(window.before_ms) and float(b) <= float(window.after_ms)):
                out.append(f"{cuff} {name} [{float(a)}, {float(b)}]")
    return out


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
        rule = PERI_R_RULE if self.window.window_class == "constant" else PERI_R_RULE_BY_CLASS
        rec: dict[str, Any] = {"rule": rule, "window": self.window.provenance(),
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
