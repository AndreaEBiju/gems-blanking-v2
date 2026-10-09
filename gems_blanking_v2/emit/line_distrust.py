"""RULINGS 2026-10-08 (d) item 2 and (e) Q1-Q5: spike-consumer distrust for mains-locked spikes.

**The rule.** A cuff-minute is distrusted **for the spike consumer only** when its
mains-locked spike excess is significant after Holm correction at :data:`ALPHA` = 0.01
over **the animal's family** (per animal and cohort, every recording first - (e) Q1).
Distrusted minutes are NaN in the spike consumer's input only ((e) Q2); no other
consumer is touched. Line noise is never motion blanking (ruling 2026-10-07 (c)).

**The test** is the one that produced ruling (d)'s table (Night 1 ``hum_inventory._spikes``
and ``line_dist.py``), run on the cuff's tripole **T** - never a contact
(:func:`minute_tests` refuses any signal not named ``<cuff>_T``) - generalised by (e) to
what the spike consumer actually sees. Per minute:

1. **What is read** ((e) Q4, Q5): the minute's samples that are finite in RAW T and not
   blanked by the spike consumer's own motion mask. Its maximal runs are the minute's
   **stretches**; a stretch too short for ``sosfiltfilt`` (at most :data:`PADLEN`
   samples, ~1 ms) is dropped. If the stretches together hold less than
   :data:`MIN_VALID_S` (30 s) the minute is ``untested_short_valid``.
2. Each stretch is band-passed 300-3000 Hz on its own (Butterworth order 4, ``output='sos'``,
   ``sosfiltfilt``). One robust sigma per minute: ``1.4826 * MAD`` of all the filtered
   stretches pooled.
3. Spikes are negative local minima below ``-4.5 sigma``, with a 1 ms refractory period
   (greedy, earliest kept), within each stretch. A spike in a motion-blanked span is
   never detected, because those samples are not read (Q5). The samples in the 7.5 ms
   taper beside a blank are read at full amplitude, where the consumer sees them
   attenuated.
4. Fewer than :data:`MIN_SPIKES` (10) spikes in the minute: ``untested_few_spikes``.
5. **Intervals are within a stretch only:** an interval across a gap (missing samples or
   a motion blank) is not an interval. ``m = sum_s (n_s - 1)`` over stretches with at
   least two spikes; ``k`` of those lie within +/-1 ms of 1/60 s or 1/30 s.
6. **Chance rate:** the interval MLE ``lam = m / sum_s (t_last,s - t_first,s)`` over the
   stretches holding at least two spikes. It differs from the inventory measurement's
   ``n / (t_last - t_first)`` by ``n / (n - 1)`` on an unsplit minute; the measurement's
   form, pooled as ``sum n_s / sum span_s``, is biased high by ``(m + S) / m`` for ``S``
   stretches and grows anti-conservative as motion splits a minute (review of a374ed3:
   null P(p < 0.01) = 0.040 at 100 Hz with 0.25 s stretches). Then
   ``p0 = sum_P [exp(-lam (P - 1 ms)) - exp(-lam (P + 1 ms))]`` over both periods and
   ``p = P(X >= k | m, p0)``, one-sided binomial. With no interval at all the minute is
   ``untested_few_spikes``.

Caveat carried from the measurement: the chance model treats spikes as Poisson; the
refractory period and real bursting make it approximate.

**Minutes** are the recording's own: minute ``m`` is ``[60 m, 60 (m + 1))`` s of the file,
cut to the emitted epoch. A minute whose in-epoch part is shorter than 30 s is
``not_assessable_short``; the stim-to-recovery boundary minute follows the same rule as
the trailing partial minute ((e) Q3). A flat minute is ``untested_no_signal``
(invariant 41). None of these is distrusted.

**T must be RAW.** Motion enters only through the motion mask argument; a T that was
already masked (interior NaN runs all on the 10 ms grid) is refused.

**The Holm family is the animal's** ((e) Q1): an :class:`AnimalPTable` keyed by
``animal_key`` = ``"cohort:animal"``, built from pass 1 over every recording of that
animal and cohort; pass 2 decides each recording against the whole table
(:func:`decide_animal`). It is a required argument with no default, and the table must
hold exactly the recording's own p-values. A p-value is keyed by recording AND epoch
(its start time) - a recording's stim and recovery epochs share the boundary minute -
and a key given twice raises (invariant 27).

**Deliberate differences from the measurement:** minute edges round
(``extent.grid.seconds_to_sample``) where it floored (at most one sample); a raw flat
check before filtering; and (e)'s stretches, motion exclusion and boundary minute.

**Representation.** :class:`LineDistrustRecord` holds every cuff-minute's counts, valid
time, p-value and decision, with provenance (rules, test version, alpha, family,
parameters, cleaner). In the handoff the distrusted minutes join the spike consumer's
blank spans - the NaN it reads - while the record, its sidecar spans and the accounting
keep motion and distrust apart (``emit.qc.spike_time_lost``). :func:`spike_input` is the
Python-side input.

**Cleaner hook (ruling (b) 6, (d) 3).** None is adopted. A ``cleaner`` re-runs the test on
cleaned T and is named in the record.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binom

from gems_blanking_v2.constants import GRID_S, MAD_TO_SIGMA
from gems_blanking_v2.emit.masks import ConsumerMask, apply_mask, frames_to_samples
from gems_blanking_v2.extent.grid import (
    T0_TOLERANCE_S,
    frame_sample_bounds,
    n_grid_frames,
    seconds_to_sample,
    to_matlab_inclusive,
)

__all__ = [
    "ALPHA",
    "LOCK_PERIODS_S",
    "LOCK_TOLERANCE_S",
    "MINUTE_S",
    "MIN_SPIKES",
    "MIN_VALID_S",
    "RULE",
    "SPIKE_SIGMA",
    "TEST_VERSION",
    "AnimalPTable",
    "LineDistrustRecord",
    "MainsCleaner",
    "MinuteResult",
    "MinuteTest",
    "Pass1",
    "decide",
    "decide_animal",
    "detect_spikes",
    "holm_reject",
    "lock_test",
    "lock_test_parameters",
    "minute_tests",
    "pass1",
    "spike_input",
]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

RULE: Final = "RULING 2026-10-08 (d) item 2; (e) Q1-Q5"
TEST_VERSION: Final = "mains_lock_binom_v3"
"""v2 = (e): animal x cohort family, stretches, motion exclusion, boundary minute;
v3 = the chance rate is the interval MLE ``m / sum span``."""
ALPHA: Final = 0.01
MIN_SPIKES: Final = 10
MINUTE_S: Final = 60.0
MIN_VALID_S: Final = 30.0
"""A minute is tested only if its in-epoch part, and its read stretches, reach 30 s."""
BAND_HZ: Final = (300.0, 3000.0)
FILTER_ORDER: Final = 4
SPIKE_SIGMA: Final = 4.5
REFRACTORY_S: Final = 0.001
LOCK_PERIODS_S: Final[tuple[float, ...]] = (1.0 / 60.0, 1.0 / 30.0)
LOCK_TOLERANCE_S: Final = 0.001
PADLEN: Final = 27
"""``sosfiltfilt``'s default padding for this order-4 band-pass (``3 (2 sections + 1)``
less trailing zeros): a stretch of at most this many samples cannot be filtered."""

Status = Literal["tested", "untested_few_spikes", "untested_short_valid",
                 "untested_no_signal", "not_assessable_short"]
STATUSES: Final[tuple[str, ...]] = ("tested", "untested_few_spikes", "untested_short_valid",
                                    "untested_no_signal", "not_assessable_short")


class MainsCleaner(Protocol):
    """A mains cleaner the test may re-run on (ruling (d) 3). None is adopted yet."""

    name: str

    def __call__(self, x: F64, fs: float) -> F64:
        """Return the cleaned signal (same length and rate; the input is not modified)."""
        ...


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------


def _sos(fs: float) -> npt.NDArray[np.float64]:
    return np.asarray(butter(FILTER_ORDER, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)],
                             btype="bandpass", output="sos"), dtype=np.float64)


def _runs(ok: Bool) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(),
                    strict=True))


def _stretches(ok: Bool) -> list[tuple[int, int]]:
    """Maximal runs of read samples long enough to filter."""
    return [(a, b) for a, b in _runs(ok) if b - a > PADLEN]


def _mad(x: F64) -> float:
    return float(np.median(np.abs(x - np.median(x))))


def _detect(x: F64, stretches: Sequence[tuple[int, int]], fs: float) -> list[F64] | None:
    """Spike times (s from ``x[0]``) per stretch, or None if the read samples are flat."""
    if not stretches:
        return None
    raw = np.concatenate([x[a:b] for a, b in stretches])
    if not _mad(raw) > 0:  # deliberate: stricter than the measurement (module docstring)
        return None
    sos = _sos(fs)
    ys = [sosfiltfilt(sos, x[a:b]) for a, b in stretches]
    sig = MAD_TO_SIGMA * _mad(np.concatenate(ys))
    if not sig > 0:
        return None
    gap = int(REFRACTORY_S * fs)
    out: list[F64] = []
    for (a, _b), y in zip(stretches, ys, strict=True):
        idx = np.flatnonzero((y[1:-1] < -SPIKE_SIGMA * sig) & (y[1:-1] <= y[:-2])
                             & (y[1:-1] <= y[2:])) + 1
        keep: list[int] = []
        last = -(10 ** 12)
        for i in idx.tolist():
            if i - last >= gap:
                keep.append(i)
                last = i
        out.append((a + np.asarray(keep, dtype=np.float64)) / fs)
    return out


def detect_spikes(y_minute: npt.ArrayLike, fs: float) -> F64 | None:
    """Spike times (s) on one gap-free stretch of T, or None if flat (the measurement's case)."""
    x = np.asarray(y_minute, dtype=np.float64)
    st = _stretches(np.ones(x.size, dtype=bool))
    found = _detect(x, st, fs)
    return None if found is None else np.concatenate(found)


@dataclass(frozen=True, slots=True)
class LockStat:
    """The mains-lock statistic: spikes, intervals, locked intervals, chance and p."""

    n_spikes: int
    m_intervals: int
    k_locked: int
    p0: float
    p: float

    @property
    def tested(self) -> bool:
        """Whether there were enough spikes and intervals to test."""
        return math.isfinite(self.p)


def lock_test(stretches: Sequence[npt.ArrayLike]) -> LockStat:
    """One-sided binomial test of mains-locked intervals, within stretches (docstring 4-6)."""
    ts = [np.sort(np.asarray(t, dtype=np.float64)) for t in stretches]
    n = int(sum(t.size for t in ts))
    m = k = 0
    span = 0.0
    for t in ts:
        if t.size < 2:  # noqa: PLR2004
            continue
        iei = np.diff(t)
        near = np.zeros(iei.size, dtype=bool)
        for per in LOCK_PERIODS_S:
            near |= np.abs(iei - per) <= LOCK_TOLERANCE_S
        m += int(iei.size)
        k += int(near.sum())
        span += float(t[-1] - t[0])
    if n < MIN_SPIKES or m == 0 or not span > 0:
        return LockStat(n, m, k, math.nan, math.nan)
    lam = m / span  # the interval MLE (docstring 6)
    p0 = sum(math.exp(-lam * (per - LOCK_TOLERANCE_S)) - math.exp(-lam * (per + LOCK_TOLERANCE_S))
             for per in LOCK_PERIODS_S)
    return LockStat(n, m, k, float(p0), float(binom.sf(k - 1, m, p0)))


def holm_reject(p: npt.ArrayLike, alpha: float) -> npt.NDArray[np.bool_]:
    """Holm step-down: reject the sorted p-values while ``p_(r) <= alpha / (M - r)``."""
    pv = np.asarray(p, dtype=np.float64)
    if not np.isfinite(pv).all():
        msg = "Holm needs finite p-values; untested minutes are not in the family"
        raise ValueError(msg)
    rej = np.zeros(pv.size, dtype=bool)
    for rank, i in enumerate(np.argsort(pv, kind="stable").tolist()):
        if pv[i] <= alpha / (pv.size - rank):
            rej[i] = True
        else:
            break
    return rej


# ---------------------------------------------------------------------------
# pass 1: per cuff-minute tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinuteTest:
    """One cuff-minute's test (pass 1). Times on the recording's timeline, seconds.

    ``start_s``/``stop_s`` are the minute's part inside the epoch; ``valid_s`` the time
    the test read (its stretches). Counts are None and numbers NaN where not computed.
    """

    signal: str
    minute: int
    start_s: float
    stop_s: float
    status: Status
    n_spikes: int | None = None
    m_intervals: int | None = None
    k_locked: int | None = None
    p0: float = math.nan
    p: float = math.nan
    valid_s: float = math.nan

    def __post_init__(self) -> None:
        """Refuse an inconsistent row; every NaN is the ``math.nan`` object.

        Equal rows then compare equal (tuple comparison checks identity before ``==``).
        """
        for name in ("p0", "p", "valid_s"):
            if math.isnan(getattr(self, name)):
                object.__setattr__(self, name, math.nan)
        if self.status not in STATUSES:
            msg = f"unknown status {self.status!r}"
            raise ValueError(msg)
        if (self.status == "tested") != math.isfinite(self.p):
            msg = f"{self.signal} minute {self.minute}: a p-value iff tested ({self.status})"
            raise ValueError(msg)


def _sample(t_s: float, epoch_start_s: float, fs: float, n_samples: int) -> int:
    return min(max(seconds_to_sample(t_s - epoch_start_s, fs), 0), n_samples)


def minute_sample_bounds(start_s: float, stop_s: float, *, epoch_start_s: float, fs: float,
                         n_samples: int) -> tuple[int, int]:
    """0-based half-open epoch samples of ``[start_s, stop_s)``, via ``extent.grid``."""
    return (_sample(start_s, epoch_start_s, fs, n_samples),
            _sample(stop_s, epoch_start_s, fs, n_samples))


def _on_mask_grid(k: int, fs: float) -> bool:
    i = int(round(k / (GRID_S * fs)))
    return frame_sample_bounds(i, i, fs)[0] == k


def _refuse_masked(name: str, x: F64, fs: float) -> None:
    """Refuse T whose interior NaN runs all start and end on the mask grid: it was masked."""
    bad = ~np.isfinite(x)
    if not bad.any():
        return
    runs = [(a, b) for a, b in _runs(bad) if a > 0 and b < x.size]
    if runs and all(_on_mask_grid(a, fs) and _on_mask_grid(b, fs) for a, b in runs):
        msg = (f"{name}: every NaN run starts and ends on the 10 ms mask grid - this T looks "
               "masked. The test reads RAW T and takes motion from the motion mask")
        raise ValueError(msg)


def _check_fs(fs: float) -> None:
    if not fs > 2 * BAND_HZ[1]:
        msg = f"fs {fs} Hz cannot carry the {BAND_HZ[0]:g}-{BAND_HZ[1]:g} Hz spike band"
        raise ValueError(msg)


def _check_signal_names(names: Iterable[str]) -> None:
    for name in names:
        if not (name.endswith("_T") and len(name) > 2):  # noqa: PLR2004
            msg = (f"{name!r}: the mains-lock rule runs on the cuff's tripole T (the spike "
                   "consumer's signal, '<cuff>_T'), never on a contact")
            raise ValueError(msg)


def _motion_samples(name: str, mask: ConsumerMask, fs: float, n: int,
                    epoch_start_s: float) -> Bool:
    if mask.consumer != "spikes" or mask.signal != name:
        msg = f"{name}: the motion mask must be the spike consumer's own, got {mask.key}"
        raise ValueError(msg)
    if (abs(mask.t0_s - epoch_start_s) > T0_TOLERANCE_S
            or mask.invalid.size != n_grid_frames(n, fs, mask.grid_s)):
        msg = (f"{name}: the motion mask ({mask.invalid.size} frames from {mask.t0_s} s) is "
               f"not on this epoch ({n} samples from {epoch_start_s} s)")
        raise ValueError(msg)
    return frames_to_samples(mask.invalid, fs, n, mask.grid_s)


def _excluded_samples(name: str, spans: Sequence[tuple[int, int]], n: int) -> Bool:
    out = np.zeros(n, dtype=bool)
    for a, b in spans:
        if not (0 <= int(a) < int(b) <= n):
            msg = f"{name}: excluded span [{a}, {b}) is not inside the epoch's {n} samples"
            raise ValueError(msg)
        out[int(a):int(b)] = True
    return out


def minute_tests(raw_t: Mapping[str, npt.ArrayLike], fs: float, *, epoch_start_s: float,
                 motion: Mapping[str, ConsumerMask],
                 cleaner: MainsCleaner | None = None,
                 exclude: Mapping[str, Sequence[tuple[int, int]]] | None = None,
                 ) -> tuple[MinuteTest, ...]:
    """Pass 1: test every cuff-minute overlapping the epoch, on RAW T less motion blanks.

    ``raw_t`` maps ``<cuff>_T`` to the raw, unmasked tripole over the emitted epoch
    (microvolts; sample 0 at ``epoch_start_s``). ``motion`` maps the same names to the
    spike consumer's motion masks (required; an all-valid mask where nothing blanked).
    ``exclude`` maps a name to further 0-based half-open epoch-sample spans the spike
    consumer never reads - RULING 2026-10-09 (b) 2: the test runs on the samples spike
    detection actually uses, so the peri-R spans (and their edge pad, as the caller
    declares) are left out. A name absent from ``exclude`` excludes nothing; a name not in
    ``raw_t`` is refused. Inputs are never modified (invariant 17); a ``cleaner`` receives
    a copy.
    """
    _check_signal_names(raw_t)
    _check_fs(fs)
    if set(motion) != set(raw_t):
        msg = f"motion masks for {sorted(motion)}, T for {sorted(raw_t)}: they must match"
        raise ValueError(msg)
    extra = dict(exclude or {})
    if set(extra) - set(raw_t):
        msg = f"exclusions for {sorted(set(extra) - set(raw_t))}, which have no T here"
        raise ValueError(msg)
    out: list[MinuteTest] = []
    for name in sorted(raw_t):
        x = np.asarray(raw_t[name], dtype=np.float64)
        _refuse_masked(name, x, fs)
        if cleaner is not None:
            x = np.asarray(cleaner(np.array(x, copy=True), fs), dtype=np.float64)
        n = int(x.size)
        read = np.isfinite(x) & ~_motion_samples(name, motion[name], fs, n, epoch_start_s)
        if name in extra:
            read &= ~_excluded_samples(name, extra[name], n)
        e0, e1 = float(epoch_start_s), float(epoch_start_s) + n / fs
        for m in range(int(math.floor(e0 / MINUTE_S)), int(math.ceil(e1 / MINUTE_S))):
            ia, ib = max(m * MINUTE_S, e0), min((m + 1) * MINUTE_S, e1)
            if not ib > ia:
                continue
            if ib - ia < MIN_VALID_S:
                out.append(MinuteTest(name, m, ia, ib, "not_assessable_short"))
                continue
            k0, k1 = minute_sample_bounds(ia, ib, epoch_start_s=e0, fs=fs, n_samples=n)
            stretches = _stretches(read[k0:k1])
            valid_s = sum(b - a for a, b in stretches) / fs
            if valid_s < MIN_VALID_S:
                out.append(MinuteTest(name, m, ia, ib, "untested_short_valid",
                                      valid_s=valid_s))
                continue
            found = _detect(x[k0:k1], stretches, fs)
            if found is None:
                out.append(MinuteTest(name, m, ia, ib, "untested_no_signal", valid_s=valid_s))
                continue
            st = lock_test(found)
            if not st.tested:
                out.append(MinuteTest(name, m, ia, ib, "untested_few_spikes", st.n_spikes,
                                      st.m_intervals, valid_s=valid_s))
                continue
            out.append(MinuteTest(name, m, ia, ib, "tested", st.n_spikes, st.m_intervals,
                                  st.k_locked, st.p0, st.p, valid_s))
    return tuple(out)


@dataclass(frozen=True)
class Pass1:
    """One recording's pass-1 output and the epoch it was tested on (no samples kept)."""

    tests: tuple[MinuteTest, ...]
    fs: float
    epoch_start_s: float
    n_samples: int
    cleaner: str | None = None
    exclusion: str | None = None
    """What ``exclude`` left out of the tested samples (None: nothing beyond motion)."""


def pass1(raw_t: Mapping[str, npt.ArrayLike], fs: float, *, epoch_start_s: float,
          motion: Mapping[str, ConsumerMask], cleaner: MainsCleaner | None = None,
          exclude: Mapping[str, Sequence[tuple[int, int]]] | None = None,
          exclusion: str | None = None) -> Pass1:
    """Run :func:`minute_tests` on one recording; keep only what pass 2 needs.

    ``exclude`` (see :func:`minute_tests`) needs ``exclusion``, the words that say what it
    is, and the other way round, so the record never carries one without the other.
    """
    if (exclude is None) != (exclusion is None):
        msg = "exclude and exclusion are given together or not at all"
        raise ValueError(msg)
    lengths = {int(np.asarray(x).size) for x in raw_t.values()}
    if len(lengths) != 1:
        msg = f"every cuff's T must cover the same epoch; got lengths {sorted(lengths)}"
        raise ValueError(msg)
    tests = minute_tests(raw_t, fs, epoch_start_s=epoch_start_s, motion=motion,
                         cleaner=cleaner, exclude=exclude)
    return Pass1(tests, float(fs), float(epoch_start_s), lengths.pop(),
                 None if cleaner is None else cleaner.name, exclusion)


# ---------------------------------------------------------------------------
# the family: animal x cohort
# ---------------------------------------------------------------------------

PKey = tuple[str, float, str, int]
"""``(recording, epoch_start_s, signal, minute)``: the epoch keeps a recording's stim
and recovery boundary minutes apart."""


def _canonical(doc: object) -> str:
    return json.dumps(doc, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False)


def _check_animal_key(key: str) -> None:
    parts = key.split(":")
    if len(parts) != 2 or not all(p.strip() for p in parts):  # noqa: PLR2004
        msg = f"animal_key must be 'cohort:animal', got {key!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class AnimalPTable:
    """Pass 1's p-values for every tested cuff-minute of one animal and cohort.

    The Holm family ((e) Q1). ``animal_key`` is ``"cohort:animal"``. May be empty (no
    tested minute anywhere: nothing can be distrusted).
    """

    animal_key: str
    pvalues: Mapping[PKey, float]
    test_version: str = TEST_VERSION
    cleaner: str | None = None

    def __post_init__(self) -> None:
        """Refuse a malformed key and p-values outside [0, 1]."""
        _check_animal_key(self.animal_key)
        bad = [k for k, v in self.pvalues.items() if not (math.isfinite(v) and 0 <= v <= 1)]
        if bad:
            msg = f"{self.animal_key}: p-values must be finite and in [0, 1]: {bad[:3]}"
            raise ValueError(msg)

    @classmethod
    def from_tests(cls, animal_key: str,
                   tests: Iterable[tuple[str, float, Sequence[MinuteTest]]], *,
                   cleaner: str | None = None) -> AnimalPTable:
        """Build the table from pass 1 (``(recording, epoch_start_s, tests)``), tested rows.

        A key given twice raises: it would overwrite another epoch's p-value (invariant 27).
        """
        pv: dict[PKey, float] = {}
        for rid, e0, rows in tests:
            for r in rows:
                if r.status != "tested":
                    continue
                key = (rid, float(e0), r.signal, r.minute)
                if key in pv:
                    msg = f"{animal_key}: p-value key {key} given twice"
                    raise ValueError(msg)
                pv[key] = r.p
        return cls(animal_key, pv, TEST_VERSION, cleaner)

    def to_record(self) -> dict[str, Any]:
        """JSON-ready (rows sorted; ``cleaner`` absent when raw)."""
        rec: dict[str, Any] = {
            "animal_key": self.animal_key, "test_version": self.test_version,
            "rows": [[r, e, s, m, p] for (r, e, s, m), p in sorted(self.pvalues.items())]}
        if self.cleaner is not None:
            rec["cleaner"] = self.cleaner
        return rec

    def to_json(self) -> str:
        """Canonical, ASCII-escaped JSON."""
        return _canonical(self.to_record())

    @classmethod
    def from_json(cls, text: str) -> AnimalPTable:
        """Parse :meth:`to_json`; a required field absent or null raises naming it."""
        doc = json.loads(text)
        for key in ("animal_key", "test_version", "rows"):
            if doc.get(key) is None:
                msg = f"animal p-value table: required field {key!r} is absent"
                raise ValueError(msg)
        pv = {(str(r), float(e), str(s), int(m)): float(p) for r, e, s, m, p in doc["rows"]}
        if len(pv) != len(doc["rows"]):
            msg = "animal p-value table: duplicate (recording, signal, minute) rows"
            raise ValueError(msg)
        return cls(doc["animal_key"], pv, doc["test_version"], doc.get("cleaner"))

    @property
    def sha256(self) -> str:
        """Hash of the canonical JSON: names the exact family in provenance."""
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# pass 2: the decision and its record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinuteResult:
    """A cuff-minute's test and the decision: ``distrusted`` only if tested and Holm-rejected."""

    test: MinuteTest
    distrusted: bool

    def __post_init__(self) -> None:
        """Refuse a distrusted minute that was never tested."""
        if self.distrusted and self.test.status != "tested":
            msg = f"{self.test.signal} minute {self.test.minute}: untested, cannot be distrusted"
            raise ValueError(msg)


def lock_test_parameters() -> dict[str, Any]:
    """Return the test's parameters and rules as structured provenance fields."""
    return {"signal": "T", "input": "raw, unmasked; motion from the spike consumer's mask",
            "band_hz": list(BAND_HZ),
            "filter": {"design": "butter", "order": FILTER_ORDER, "output": "sos",
                       "apply": "sosfiltfilt", "span": "stretch", "min_stretch_samples":
                       PADLEN + 1},
            "sigma": "1.4826 * MAD of the minute's filtered stretches, pooled",
            "spike_polarity": "negative", "spike_threshold_sigma": SPIKE_SIGMA,
            "refractory_s": REFRACTORY_S,
            "lock_periods_s": list(LOCK_PERIODS_S), "lock_tolerance_s": LOCK_TOLERANCE_S,
            "intervals": "within a stretch only; none across a gap or a motion blank",
            "motion": "spikes in the spike consumer's motion-blanked spans are not read",
            "chance": ("poisson, interval MLE lam = m / sum (t_last_s - t_first_s) over "
                       "stretches with >= 2 spikes; differs from the inventory measurement's "
                       "n / (t_last - t_first) by n / (n - 1) on an unsplit minute"),
            "test": "one-sided binomial, P(X >= k | m, p0)", "correction": "holm",
            "family": "animal x cohort, all recordings first",
            "min_spikes": MIN_SPIKES, "minute_s": MINUTE_S, "min_valid_s": MIN_VALID_S,
            "boundary_minute": "in-epoch part tested when >= min_valid_s"}


_ROW_FLOATS: Final = ("start_s", "stop_s", "p0", "p", "valid_s")
_ROW_INTS: Final = ("n_spikes", "m_intervals", "k_locked")


def _row(r: MinuteResult) -> dict[str, Any]:
    t = r.test
    row: dict[str, Any] = {"signal": t.signal, "minute": t.minute, "status": t.status,
                           "distrusted": r.distrusted}
    for k in _ROW_FLOATS:
        v = getattr(t, k)
        if math.isfinite(v):
            row[k] = v
    for k in _ROW_INTS:
        v = getattr(t, k)
        if v is not None:
            row[k] = v
    return row


def _opt_int(row: Mapping[str, Any], key: str) -> int | None:
    return None if row.get(key) is None else int(row[key])


def _opt_float(row: Mapping[str, Any], key: str) -> float:
    return math.nan if row.get(key) is None else float(row[key])


def _unrow(row: Mapping[str, Any]) -> MinuteResult:
    for key in ("signal", "minute", "status", "distrusted", "start_s", "stop_s"):
        if row.get(key) is None:
            msg = f"line-distrust row: required field {key!r} is absent"
            raise ValueError(msg)
    t = MinuteTest(row["signal"], int(row["minute"]), float(row["start_s"]),
                   float(row["stop_s"]), row["status"], _opt_int(row, "n_spikes"),
                   _opt_int(row, "m_intervals"), _opt_int(row, "k_locked"),
                   _opt_float(row, "p0"), _opt_float(row, "p"), _opt_float(row, "valid_s"))
    return MinuteResult(t, bool(row["distrusted"]))


@dataclass(frozen=True)
class LineDistrustRecord:
    """One recording's spike-consumer line distrust: every cuff-minute, decided, with provenance.

    ``family``: ``kind`` (``animal_x_cohort``), ``animal_key``, ``size`` and ``sha256``.
    """

    recording: str
    fs: float
    epoch_start_s: float
    n_samples: int
    family: Mapping[str, Any]
    minutes: tuple[MinuteResult, ...]
    cleaner: str | None = None
    test_version: str = TEST_VERSION
    alpha: float = ALPHA
    rule: str = RULE
    signals: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        """Derive ``signals`` from the rows when not given, and refuse an unnamed family."""
        if not self.signals:
            object.__setattr__(self, "signals",
                               tuple(sorted({r.test.signal for r in self.minutes})))
        _check_signal_names(self.signals)
        if self.family.get("kind") != "animal_x_cohort":
            msg = f"family kind must be 'animal_x_cohort' (ruling (e) Q1), got {self.family!r}"
            raise ValueError(msg)

    @property
    def epoch_stop_s(self) -> float:
        """End of the epoch on the recording's timeline."""
        return self.epoch_start_s + self.n_samples / self.fs

    @property
    def provenance(self) -> dict[str, Any]:
        """What made the decision: rules, test version, alpha, family, parameters, cleaner."""
        rec: dict[str, Any] = {
            "rule": self.rule, "test_version": self.test_version, "alpha": self.alpha,
            "family": dict(self.family), "consumer": "spikes",
            "parameters": lock_test_parameters()}
        if self.cleaner is not None:
            rec["cleaner"] = self.cleaner
        return rec

    def distrusted_spans(self, signal: str) -> list[tuple[float, float]]:
        """Distrusted minutes of ``signal`` as recording-timeline ``[a, b)`` seconds."""
        if signal not in self.signals:
            msg = f"{signal} is not in this record ({list(self.signals)})"
            raise KeyError(msg)
        return [(r.test.start_s, r.test.stop_s) for r in self.minutes
                if r.distrusted and r.test.signal == signal]

    def sample_spans(self, signal: str) -> list[tuple[int, int]]:
        """Distrusted minutes as 0-based half-open epoch samples."""
        return [s for s in (minute_sample_bounds(a, b, epoch_start_s=self.epoch_start_s,
                                                 fs=self.fs, n_samples=self.n_samples)
                            for a, b in self.distrusted_spans(signal)) if s[1] > s[0]]

    def matlab_spans(self, signal: str) -> F64:
        """Distrusted minutes as MATLAB 1-based inclusive epoch samples (N x 2)."""
        spans = self.sample_spans(signal)
        if not spans:
            return np.zeros((0, 2), dtype=np.float64)
        return to_matlab_inclusive([a for a, _ in spans], [b for _, b in spans])

    def counts(self, signal: str) -> dict[str, int]:
        """Minutes tested, untested (incl. not assessable) and distrusted for ``signal``."""
        rows = [r for r in self.minutes if r.test.signal == signal]
        return {"minutes_tested": sum(r.test.status == "tested" for r in rows),
                "minutes_untested": sum(r.test.status != "tested" for r in rows),
                "minutes_distrusted": sum(r.distrusted for r in rows)}

    def to_record(self) -> dict[str, Any]:
        """JSON-ready; a missing number is an absent key."""
        rec = {"recording": self.recording, "fs": self.fs, "epoch_start_s": self.epoch_start_s,
               "n_samples": self.n_samples, "signals": list(self.signals),
               "provenance": self.provenance, "minutes": [_row(r) for r in self.minutes]}
        out: dict[str, Any] = json.loads(_canonical(rec))  # refuses NaN on the way
        return out

    def to_json(self) -> str:
        """Canonical, ASCII-escaped JSON."""
        return _canonical(self.to_record())

    @classmethod
    def from_json(cls, text: str) -> LineDistrustRecord:
        """Parse :meth:`to_json`; absent and null are the same; required missing raises."""
        doc = json.loads(text)
        for key in ("recording", "fs", "epoch_start_s", "n_samples", "provenance", "minutes",
                    "signals"):
            if doc.get(key) is None:
                msg = f"line-distrust record: required field {key!r} is absent"
                raise ValueError(msg)
        prov = doc["provenance"]
        for key in ("rule", "test_version", "alpha", "family", "parameters"):
            if prov.get(key) is None:
                msg = f"line-distrust provenance: required field {key!r} is absent"
                raise ValueError(msg)
        rec = cls(doc["recording"], float(doc["fs"]), float(doc["epoch_start_s"]),
                  int(doc["n_samples"]), prov["family"],
                  tuple(_unrow(r) for r in doc["minutes"]), prov.get("cleaner"),
                  prov["test_version"], float(prov["alpha"]), prov["rule"],
                  tuple(doc["signals"]))
        if prov["parameters"] != lock_test_parameters():
            msg = (f"line-distrust record made with other test parameters than "
                   f"{TEST_VERSION}'s: {prov['parameters']!r}")
            raise ValueError(msg)
        return rec


P_MATCH_RTOL: Final = 1e-12
"""Relative tolerance for a pass-2 p-value to match the animal table's (float noise only)."""


def _check_tests(tests: Sequence[MinuteTest], *, fs: float, epoch_start_s: float,
                 n_samples: int) -> None:
    """Refuse an impossible rate, an empty epoch, duplicate rows and rows outside the epoch."""
    _check_fs(fs)
    if not n_samples > 0:
        msg = f"an epoch must hold samples, got n_samples={n_samples}"
        raise ValueError(msg)
    e1 = epoch_start_s + n_samples / fs
    seen: set[tuple[str, int]] = set()
    for t in tests:
        key = (t.signal, t.minute)
        if key in seen:
            msg = f"{t.signal} minute {t.minute} appears twice"
            raise ValueError(msg)
        seen.add(key)
        if not (t.start_s >= epoch_start_s - T0_TOLERANCE_S
                and t.stop_s <= e1 + T0_TOLERANCE_S and t.stop_s > t.start_s):
            msg = (f"{t.signal} minute {t.minute} [{t.start_s}, {t.stop_s}) is not inside the "
                   f"epoch [{epoch_start_s}, {e1})")
            raise ValueError(msg)


def decide(tests: Sequence[MinuteTest], *, recording: str, fs: float, epoch_start_s: float,
           n_samples: int, family: AnimalPTable, cleaner: str | None = None
           ) -> LineDistrustRecord:
    """Pass 2: Holm at :data:`ALPHA` over the animal's table; distrust this recording's rejections.

    ``family`` is required, with no default. The table must hold exactly this recording's
    tested p-values (to a relative 1e-12), under the same test version and cleaner. Also
    refuses an ``fs`` below the band, an empty epoch, a ``(signal, minute)`` row twice
    and a row outside the epoch.
    """
    given: object = family  # checked at run time too: the type is not a guarantee
    if not isinstance(given, AnimalPTable):
        msg = (f"family must be the animal x cohort AnimalPTable (ruling (e) Q1), got "
               f"{given!r}")
        raise TypeError(msg)
    _check_tests(tests, fs=fs, epoch_start_s=epoch_start_s, n_samples=n_samples)
    if family.test_version != TEST_VERSION or family.cleaner != cleaner:
        msg = (f"the animal table was made under {family.test_version} / cleaner "
               f"{family.cleaner!r}; this pass is {TEST_VERSION} / {cleaner!r}")
        raise ValueError(msg)
    e0 = float(epoch_start_s)
    tested = {(recording, e0, t.signal, t.minute): t.p for t in tests if t.status == "tested"}
    mine = {k: v for k, v in family.pvalues.items() if k[0] == recording and k[1] == e0}
    differ = sorted(k for k in set(mine) & set(tested)
                    if not math.isclose(mine[k], tested[k], rel_tol=P_MATCH_RTOL,
                                        abs_tol=0.0))[:3]
    if set(mine) != set(tested) or differ:
        missing = sorted(set(tested) - set(mine))[:3]
        extra = sorted(set(mine) - set(tested))[:3]
        msg = (f"{recording}: the animal table does not hold exactly this recording's "
               f"tested p-values (missing {missing}, extra {extra}, differing {differ}); "
               "pass 1 and pass 2 did not test the same thing")
        raise ValueError(msg)
    keys = sorted(family.pvalues)
    pv = [family.pvalues[k] for k in keys]
    rejected = ({k for k, r in zip(keys, holm_reject(pv, ALPHA), strict=True) if r}
                if keys else set())
    fam = {"kind": "animal_x_cohort", "animal_key": family.animal_key, "size": len(keys),
           "sha256": family.sha256}
    rows = tuple(MinuteResult(t, t.status == "tested"
                              and (recording, e0, t.signal, t.minute) in rejected)
                 for t in tests)
    return LineDistrustRecord(recording, float(fs), float(epoch_start_s), int(n_samples), fam,
                              rows, cleaner)


def decide_animal(animal_key: str, passes: Iterable[tuple[str, Pass1]]
                  ) -> tuple[AnimalPTable, dict[tuple[str, float], LineDistrustRecord]]:
    """Decide one animal x cohort - the production path: every pass 1 first, then Holm.

    ``passes`` lists ``(recording, pass1(...))`` for every epoch of every recording of the
    animal (and cohort); records come back keyed ``(recording, epoch_start_s)``. Pass 1
    keeps no samples, so recordings can be loaded one at a time. All passes must share
    one cleaner (or none); a ``(recording, epoch)`` given twice raises.
    """
    passes = list(passes)
    keys = [(r, p.epoch_start_s) for r, p in passes]
    if len(set(keys)) != len(keys):
        msg = f"{animal_key}: a (recording, epoch) is given twice: {sorted(keys)}"
        raise ValueError(msg)
    cleaners = {p.cleaner for _r, p in passes}
    if len(cleaners) > 1:
        msg = (f"{animal_key}: recordings tested under different cleaners "
               f"{sorted(map(str, cleaners))}")
        raise ValueError(msg)
    cleaner = cleaners.pop() if cleaners else None
    table = AnimalPTable.from_tests(animal_key,
                                    [(r, p.epoch_start_s, p.tests) for r, p in passes],
                                    cleaner=cleaner)
    return table, {(r, p.epoch_start_s): decide(p.tests, recording=r, fs=p.fs,
                                                epoch_start_s=p.epoch_start_s,
                                                n_samples=p.n_samples, family=table,
                                                cleaner=cleaner)
                   for r, p in passes}


def spike_input(x: npt.ArrayLike, fs: float, motion: ConsumerMask,
                record: LineDistrustRecord) -> F64:
    """Return the spike consumer's input on one cuff: motion AND distrusted minutes NaN.

    Python-side twin of the handoff's ``blank_spikes_*`` ((e) Q2): distrusted minutes are
    set NaN first, then ``emit.masks.apply_mask`` NaNs the motion frames and tapers beside
    every NaN run (never writing a zero). ``x`` starts at the record's epoch start; the
    input is never modified.
    """
    if motion.consumer != "spikes" or motion.signal not in record.signals:
        msg = f"spike_input needs the spike consumer's mask of a recorded cuff, got {motion.key}"
        raise ValueError(msg)
    y = np.array(x, dtype=np.float64, copy=True)
    if y.size != record.n_samples:
        msg = f"{y.size} samples, but the record's epoch has {record.n_samples}"
        raise ValueError(msg)
    for a, b in record.sample_spans(motion.signal):
        y[a:b] = np.nan
    return apply_mask(y, fs, motion.invalid, grid_s=motion.grid_s,
                      what=f"spike input {motion.signal}")
