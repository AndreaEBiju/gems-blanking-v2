"""RULING 2026-10-08 (d) item 2: per-cuff-minute spike-consumer distrust for mains-locked spikes.

**The rule.** A cuff-minute is distrusted **for the spike consumer only** when its
mains-locked spike excess is significant after Holm correction at :data:`ALPHA` = 0.01.
Nothing else is affected: it is never motion blanking, and every other consumer keeps
the minute (ruling 2026-10-07 (c): line noise is never a reason to blank).

**The test** is the one that produced the ruling's table (Night 1 ``hum_inventory._spikes``
and ``line_dist.py``), run on the cuff's tripole **T** - the spike consumer's own signal,
never a contact. The table approximated with contacts in the new cohort; the rule is on T,
and :func:`minute_tests` refuses any signal not named ``<cuff>_T``. Per minute:

1. T over the minute is band-passed 300-3000 Hz (Butterworth order 4, ``output='sos'``,
   ``sosfiltfilt``) - filtered per minute, as measured.
2. Spikes are negative local minima below ``-4.5`` x the minute's robust sigma
   (``1.4826 * MAD``), with a 1 ms refractory period (greedy, earliest kept).
3. Fewer than :data:`MIN_SPIKES` (10) spikes: **untested**, and an untested minute is
   never distrusted.
4. ``m = n - 1`` inter-spike intervals; ``k`` of them lie within +/-1 ms of 1/60 s or
   1/30 s. The chance rate is Poisson: ``lam = n / (t_last - t_first)`` and
   ``p0 = sum_P [exp(-lam (P - 1 ms)) - exp(-lam (P + 1 ms))]`` over both periods.
5. One-sided binomial ``p = P(X >= k | m, p0)``.

Caveat carried from the measurement: the chance model treats spikes as Poisson; the
refractory period and real bursting make it approximate.

**Minutes** are the recording's own: minute ``m`` is ``[60 m, 60 (m + 1))`` s of the file.
Only minutes that lie inside the emitted epoch are assessable (the stim epoch is excluded
as routing excludes it): a minute starting before the epoch - one inside or straddling the
stim epoch - is ``not_assessable_epoch_edge``, and a trailing piece shorter than 30 s is
``not_assessable_short`` (the measurement's partial-minute rule). Neither is tested, so
neither is distrusted. A minute holding any non-finite sample is ``untested_non_finite``
(the measurement skipped those too), and a flat one ``untested_no_signal`` (invariant 41:
no statistic is reported from an input carrying no signal).

**T must be RAW - unmasked.** The test reads T as the consumer would before any blank:
a masked T (NaN over motion spans) silently turns every minute holding a blank into
``untested_non_finite``, so the rule would never see them. The parameter is named
``raw_t``, and an input whose NaN runs all start and end exactly on the 10 ms mask grid
(what ``emit.masks.apply_mask`` produces, and a genuine acquisition gap almost never
does) is refused as masked - a cheap check, not a proof.

**Two deliberate differences from the measurement**, both documented here so they are
not mistaken for drift:

* **Minute edges round, the measurement floored.** ``hum_inventory`` cut minute ``m`` at
  ``int(60 m fs)`` samples from the file start; here a time becomes a sample through
  ``extent.grid.seconds_to_sample`` (``round``), the one conversion every emitted span
  uses (invariant 33). An edge can therefore sit one sample later (``60 fs = 1464843.75``
  -> 1464844, not 1464843); one sample in 1.46 million changes no count.
* **A raw flat check.** Before filtering, a minute whose own MAD is 0 is
  ``untested_no_signal``. The measurement had only the filtered-sigma check, which a
  constant minute passes: filtering turns its rounding noise into a tiny but positive
  sigma and then into hundreds of "spikes" (found in testing).

**The Holm family is an explicit, required argument** (:data:`Family`) with no default,
because the table that produced the ruling used every tested channel-minute of the
animal, and at emission time that needs all of the animal's recordings first:

* ``"recording"`` - Holm over this recording's tested cuff-minutes (both cuffs);
* an :class:`AnimalPTable` - the animal's precomputed p-values (two passes: pass 1 runs
  :func:`minute_tests` on every recording of the animal and builds the table with
  :meth:`AnimalPTable.from_tests`; pass 2 decides each recording against the whole table).
  The table must hold exactly this recording's tested p-values, or the two passes did not
  test the same thing and the decision raises.

Which family applies is a question for Andrea; nothing here chooses.

**Representation.** The result is a :class:`LineDistrustRecord`: its own record, with
provenance (rule, test version, alpha, family, input signal, cleaner) and every cuff-minute's
counts and p-value. It is carried BESIDE the spike consumer's mask in the MATLAB handoff
(``distrust_spikes_<cuff>_T`` spans and ``linedistrust_json``), never merged into it: the
mask stays the motion (and ruled cuff) mask, and this rule writes no NaN into it.

**Cleaner hook (ruling (b) 6, ruling (d) 3).** No mains cleaner is adopted; adoption is
Andrea's decision. If one is adopted, pass it as ``cleaner`` (:class:`MainsCleaner`): the
test then re-runs on the cleaned T, minutes no longer significant regain trust, and the
record names the cleaner. Without one the test runs on raw T and the record says so
(``cleaner`` absent).

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
from gems_blanking_v2.extent.grid import (
    T0_TOLERANCE_S,
    frame_sample_bounds,
    seconds_to_sample,
    to_matlab_inclusive,
)

__all__ = [
    "ALPHA",
    "LOCK_PERIODS_S",
    "LOCK_TOLERANCE_S",
    "MINUTE_S",
    "MIN_SPIKES",
    "RULE",
    "SPIKE_SIGMA",
    "TEST_VERSION",
    "AnimalPTable",
    "Family",
    "LineDistrustRecord",
    "MainsCleaner",
    "MinuteResult",
    "MinuteTest",
    "cuff_minute_distrust",
    "decide",
    "detect_spikes",
    "holm_reject",
    "lock_test",
    "lock_test_parameters",
    "minute_tests",
]

F64 = npt.NDArray[np.float64]

RULE: Final = "RULING 2026-10-08 (d) item 2"
TEST_VERSION: Final = "mains_lock_binom_v1"
"""Bump when anything in the test changes: a record names the version it was made under."""
ALPHA: Final = 0.01
"""Holm family-wise alpha (ruling (d) 2)."""
MIN_SPIKES: Final = 10
"""Fewer spikes than this in a minute: untested, never distrusted."""
MINUTE_S: Final = 60.0
MIN_PARTIAL_S: Final = 30.0
"""A trailing minute shorter than this is not assessable (the measurement's rule)."""
BAND_HZ: Final = (300.0, 3000.0)
FILTER_ORDER: Final = 4
SPIKE_SIGMA: Final = 4.5
"""Negative peaks below ``-SPIKE_SIGMA`` robust sigma (``P.threshSigma``)."""
REFRACTORY_S: Final = 0.001
LOCK_PERIODS_S: Final[tuple[float, ...]] = (1.0 / 60.0, 1.0 / 30.0)
LOCK_TOLERANCE_S: Final = 0.001

Status = Literal["tested", "untested_few_spikes", "untested_non_finite", "untested_no_signal",
                 "not_assessable_epoch_edge", "not_assessable_short"]
STATUSES: Final[tuple[str, ...]] = ("tested", "untested_few_spikes", "untested_non_finite",
                                    "untested_no_signal", "not_assessable_epoch_edge",
                                    "not_assessable_short")
_EDGE_TOL_S: Final = 1e-9


class MainsCleaner(Protocol):
    """A mains cleaner the test may re-run on (ruling (d) 3). None is adopted yet.

    ``name`` identifies it in provenance; calling it returns a cleaned copy of T
    (microvolts, same length and rate). It must not modify its input.
    """

    name: str

    def __call__(self, x: F64, fs: float) -> F64:
        """Return the cleaned signal."""
        ...


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------


def detect_spikes(y_minute: npt.ArrayLike, fs: float) -> F64 | None:
    """Spike times (s from the minute's first sample) on one minute of T, or None if flat.

    Band-pass 300-3000 Hz (order 4, sos, filtfilt) over the minute, robust sigma of the
    filtered minute, negative local minima below ``-4.5 sigma``, 1 ms refractory (greedy).
    """
    x = np.asarray(y_minute, dtype=np.float64)
    if x.size < 3 or not float(np.median(np.abs(x - np.median(x)))) > 0:  # noqa: PLR2004
        # Flat (or empty). Deliberately stricter than the measurement (module docstring):
        # filtering would turn rounding noise into a positive sigma and then "spikes".
        return None
    sos = butter(FILTER_ORDER, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)],
                 btype="bandpass", output="sos")
    y = sosfiltfilt(sos, x)
    sig = MAD_TO_SIGMA * float(np.median(np.abs(y - np.median(y))))
    if not sig > 0:
        return None
    idx = np.flatnonzero((y[1:-1] < -SPIKE_SIGMA * sig) & (y[1:-1] <= y[:-2])
                         & (y[1:-1] <= y[2:])) + 1
    gap = int(REFRACTORY_S * fs)
    keep: list[int] = []
    last = -(10 ** 12)
    for i in idx.tolist():
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.float64) / fs


@dataclass(frozen=True, slots=True)
class LockStat:
    """The mains-lock statistic of one spike train: counts, chance and p (NaN if untested)."""

    n_spikes: int
    k_locked: int
    p0: float
    p: float

    @property
    def tested(self) -> bool:
        """Whether there were enough spikes to test."""
        return self.n_spikes >= MIN_SPIKES


def lock_test(t_s: npt.ArrayLike) -> LockStat:
    """One-sided binomial test of mains-locked inter-spike intervals (module docstring 4-5).

    Fewer than :data:`MIN_SPIKES` spikes: ``k``, ``p0`` and ``p`` are NaN/0 - untested.
    """
    t = np.sort(np.asarray(t_s, dtype=np.float64))
    n = int(t.size)
    if n < MIN_SPIKES:
        return LockStat(n, 0, math.nan, math.nan)
    iei = np.diff(t)
    near = np.zeros(iei.size, dtype=bool)
    for per in LOCK_PERIODS_S:
        near |= np.abs(iei - per) <= LOCK_TOLERANCE_S
    span = float(t[-1] - t[0])
    if not span > 0:
        msg = "spike times span no time: cannot estimate a Poisson rate"
        raise ValueError(msg)
    lam = n / span
    p0 = sum(math.exp(-lam * (per - LOCK_TOLERANCE_S)) - math.exp(-lam * (per + LOCK_TOLERANCE_S))
             for per in LOCK_PERIODS_S)
    m, k = n - 1, int(near.sum())
    p = float(binom.sf(k - 1, m, p0))
    return LockStat(n, k, float(p0), p)


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
    """One cuff-minute's test (pass 1). Times are on the recording's timeline, seconds.

    ``start_s``/``stop_s`` are the minute's part inside the epoch. ``n_spikes`` is None
    where nothing was detected (not assessable, non-finite, flat); ``k_locked``, ``p0``
    and ``p`` are NaN/None unless ``status == "tested"``.
    """

    signal: str
    minute: int
    start_s: float
    stop_s: float
    status: Status
    n_spikes: int | None = None
    k_locked: int | None = None
    p0: float = math.nan
    p: float = math.nan

    def __post_init__(self) -> None:
        """Refuse an inconsistent row, and make every NaN the ``math.nan`` object.

        Equal rows then compare equal: tuple comparison checks identity before ``==``,
        and NaN != NaN.
        """
        for name in ("p0", "p"):
            if math.isnan(getattr(self, name)):
                object.__setattr__(self, name, math.nan)
        if self.status not in STATUSES:
            msg = f"unknown status {self.status!r}"
            raise ValueError(msg)
        if (self.status == "tested") != math.isfinite(self.p):
            msg = f"{self.signal} minute {self.minute}: a p-value iff tested ({self.status})"
            raise ValueError(msg)


def _sample(t_s: float, epoch_start_s: float, fs: float, n_samples: int) -> int:
    """Sample index of recording time ``t_s`` in the epoch, clipped to ``[0, n_samples]``."""
    return min(max(seconds_to_sample(t_s - epoch_start_s, fs), 0), n_samples)


def minute_sample_bounds(start_s: float, stop_s: float, *, epoch_start_s: float, fs: float,
                         n_samples: int) -> tuple[int, int]:
    """0-based half-open epoch samples of ``[start_s, stop_s)``, via ``extent.grid``.

    Rounds, where the measurement floored: an edge may sit one sample later (module
    docstring).
    """
    return (_sample(start_s, epoch_start_s, fs, n_samples),
            _sample(stop_s, epoch_start_s, fs, n_samples))


def _on_mask_grid(k: int, fs: float) -> bool:
    """Whether sample ``k`` is a 10 ms frame boundary under ``extent.grid``'s conversion."""
    i = int(round(k / (GRID_S * fs)))
    return frame_sample_bounds(i, i, fs)[0] == k


def _refuse_masked(name: str, x: F64, fs: float) -> None:
    """Refuse T whose interior NaN runs all start and end on the mask grid: it was masked."""
    bad = ~np.isfinite(x)
    if not bad.any():
        return
    d = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    runs = [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1),
                                             strict=True) if a > 0 and b < x.size]
    if runs and all(_on_mask_grid(a, fs) and _on_mask_grid(b, fs) for a, b in runs):
        msg = (f"{name}: every NaN run starts and ends on the 10 ms mask grid - this T looks "
               "masked. The mains-lock test reads RAW T; a masked one makes every blanked "
               "minute untested")
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


def minute_tests(raw_t: Mapping[str, npt.ArrayLike], fs: float, *, epoch_start_s: float,
                 cleaner: MainsCleaner | None = None) -> tuple[MinuteTest, ...]:
    """Pass 1: test every cuff-minute that overlaps the epoch, on RAW T.

    ``raw_t`` maps ``<cuff>_T`` to the cuff's tripole over the emitted epoch, **raw and
    unmasked** (microvolts; sample 0 at ``epoch_start_s`` on the recording's timeline):
    a masked T would make every blanked minute untested, so input that looks masked is
    refused (module docstring). The inputs are never modified (invariant 17); a
    ``cleaner`` receives a copy.
    """
    _check_signal_names(raw_t)
    _check_fs(fs)
    out: list[MinuteTest] = []
    for name in sorted(raw_t):
        x = np.asarray(raw_t[name], dtype=np.float64)
        _refuse_masked(name, x, fs)
        if cleaner is not None:
            x = np.asarray(cleaner(np.array(x, copy=True), fs), dtype=np.float64)
        n = int(x.size)
        e0, e1 = float(epoch_start_s), float(epoch_start_s) + n / fs
        for m in range(int(math.floor(e0 / MINUTE_S)), int(math.ceil(e1 / MINUTE_S))):
            a, b = m * MINUTE_S, (m + 1) * MINUTE_S
            ia, ib = max(a, e0), min(b, e1)
            if not ib > ia:
                continue
            if a < e0 - _EDGE_TOL_S:
                out.append(MinuteTest(name, m, ia, ib, "not_assessable_epoch_edge"))
                continue
            if ib - ia < MIN_PARTIAL_S:
                out.append(MinuteTest(name, m, ia, ib, "not_assessable_short"))
                continue
            k0, k1 = minute_sample_bounds(ia, ib, epoch_start_s=e0, fs=fs, n_samples=n)
            seg = x[k0:k1]
            if not np.isfinite(seg).all():
                out.append(MinuteTest(name, m, ia, ib, "untested_non_finite"))
                continue
            t = detect_spikes(seg, fs)
            if t is None:
                out.append(MinuteTest(name, m, ia, ib, "untested_no_signal"))
                continue
            st = lock_test(t)
            if not st.tested:
                out.append(MinuteTest(name, m, ia, ib, "untested_few_spikes", st.n_spikes))
                continue
            out.append(MinuteTest(name, m, ia, ib, "tested", st.n_spikes, st.k_locked, st.p0,
                                  st.p))
    return tuple(out)


# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------

PKey = tuple[str, str, int]
"""``(recording, signal, minute)``."""


def _canonical(doc: object) -> str:
    return json.dumps(doc, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False)


@dataclass(frozen=True)
class AnimalPTable:
    """Pass 1's p-values for every tested cuff-minute of one animal (the Holm family).

    ``cleaner`` is the cleaner every row was tested under (None = raw T); a table mixing
    cleaners or test versions cannot be built.
    """

    animal: str
    pvalues: Mapping[PKey, float]
    test_version: str = TEST_VERSION
    cleaner: str | None = None

    def __post_init__(self) -> None:
        """Refuse an empty table, an unnamed animal and non-finite p-values."""
        if not self.animal:
            msg = "an animal p-value table must name its animal"
            raise ValueError(msg)
        if not self.pvalues:
            msg = f"{self.animal}: an empty p-value table is not a family"
            raise ValueError(msg)
        bad = [k for k, v in self.pvalues.items() if not (math.isfinite(v) and 0 <= v <= 1)]
        if bad:
            msg = f"{self.animal}: p-values must be finite and in [0, 1]: {bad[:3]}"
            raise ValueError(msg)

    @classmethod
    def from_tests(cls, animal: str, tests: Mapping[str, Sequence[MinuteTest]], *,
                   cleaner: str | None = None) -> AnimalPTable:
        """Build the table from pass 1 (``{recording: minute_tests(...)}``), tested rows only."""
        pv: dict[PKey, float] = {}
        for rid, rows in tests.items():
            for r in rows:
                if r.status == "tested":
                    pv[(rid, r.signal, r.minute)] = r.p
        return cls(animal, pv, TEST_VERSION, cleaner)

    def to_record(self) -> dict[str, Any]:
        """JSON-ready (rows sorted; ``cleaner`` absent when raw)."""
        rec: dict[str, Any] = {
            "animal": self.animal, "test_version": self.test_version,
            "rows": [[r, s, m, p] for (r, s, m), p in sorted(self.pvalues.items())]}
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
        for key in ("animal", "test_version", "rows"):
            if doc.get(key) is None:
                msg = f"animal p-value table: required field {key!r} is absent"
                raise ValueError(msg)
        pv = {(str(r), str(s), int(m)): float(p) for r, s, m, p in doc["rows"]}
        if len(pv) != len(doc["rows"]):
            msg = "animal p-value table: duplicate (recording, signal, minute) rows"
            raise ValueError(msg)
        return cls(doc["animal"], pv, doc["test_version"], doc.get("cleaner"))

    @property
    def sha256(self) -> str:
        """Hash of the canonical JSON: names the exact family in provenance."""
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()


Family = Literal["recording"] | AnimalPTable
"""The Holm family: this recording's tested cuff-minutes, or the animal's whole table."""


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
    """Return the test's parameters as structured provenance fields (JSON-ready)."""
    return {"signal": "T", "input": "raw, unmasked",
            "band_hz": list(BAND_HZ),
            "filter": {"design": "butter", "order": FILTER_ORDER, "output": "sos",
                       "apply": "sosfiltfilt", "span": "minute"},
            "sigma": "1.4826 * MAD of the filtered minute",
            "spike_polarity": "negative", "spike_threshold_sigma": SPIKE_SIGMA,
            "refractory_s": REFRACTORY_S,
            "lock_periods_s": list(LOCK_PERIODS_S), "lock_tolerance_s": LOCK_TOLERANCE_S,
            "chance": "poisson, lam = n / (t_last - t_first)",
            "test": "one-sided binomial, P(X >= k | n - 1, p0)", "correction": "holm",
            "min_spikes": MIN_SPIKES, "minute_s": MINUTE_S,
            "min_partial_minute_s": MIN_PARTIAL_S}


_ROW_FLOATS: Final = ("start_s", "stop_s", "p0", "p")
_ROW_INTS: Final = ("n_spikes", "k_locked")


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
    if t.status == "tested":
        row["m_intervals"] = int(t.n_spikes or 0) - 1
    return row


def _unrow(row: Mapping[str, Any]) -> MinuteResult:
    for key in ("signal", "minute", "status", "distrusted", "start_s", "stop_s"):
        if row.get(key) is None:
            msg = f"line-distrust row: required field {key!r} is absent"
            raise ValueError(msg)
    t = MinuteTest(row["signal"], int(row["minute"]), float(row["start_s"]),
                   float(row["stop_s"]), row["status"],
                   None if row.get("n_spikes") is None else int(row["n_spikes"]),
                   None if row.get("k_locked") is None else int(row["k_locked"]),
                   math.nan if row.get("p0") is None else float(row["p0"]),
                   math.nan if row.get("p") is None else float(row["p"]))
    return MinuteResult(t, bool(row["distrusted"]))


@dataclass(frozen=True)
class LineDistrustRecord:
    """One recording's spike-consumer line distrust: every cuff-minute, decided, with provenance.

    ``family`` records the Holm family: ``kind`` (``recording`` | ``animal``), ``size``
    (tests in it), and for an animal table its ``animal`` and ``sha256``. ``cleaner`` is
    None for raw T. Carried beside the spike mask, never merged into it.
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
        if self.family.get("kind") not in ("recording", "animal"):
            msg = f"family kind must be 'recording' or 'animal', got {self.family!r}"
            raise ValueError(msg)

    @property
    def epoch_stop_s(self) -> float:
        """End of the epoch on the recording's timeline."""
        return self.epoch_start_s + self.n_samples / self.fs

    @property
    def provenance(self) -> dict[str, Any]:
        """What made the decision: rule, test version, alpha, family, parameters, cleaner."""
        rec: dict[str, Any] = {
            "rule": self.rule, "test_version": self.test_version, "alpha": self.alpha,
            "family": dict(self.family), "consumer": "spikes", "parameters": lock_test_parameters()}
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
        return [minute_sample_bounds(a, b, epoch_start_s=self.epoch_start_s, fs=self.fs,
                                     n_samples=self.n_samples)
                for a, b in self.distrusted_spans(signal)]

    def matlab_spans(self, signal: str) -> F64:
        """Distrusted minutes as MATLAB 1-based inclusive epoch samples (N x 2)."""
        spans = [s for s in self.sample_spans(signal) if s[1] > s[0]]
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
           n_samples: int, family: Family, cleaner: str | None = None) -> LineDistrustRecord:
    """Pass 2: Holm at :data:`ALPHA` over ``family``; distrust this recording's rejections.

    ``family`` has no default (see the module docstring). With an :class:`AnimalPTable`,
    the table must hold exactly this recording's tested p-values (to a relative 1e-12),
    under the same test version and cleaner, or this raises. Also refuses an ``fs`` that
    cannot carry the band, an empty epoch, a ``(signal, minute)`` row twice, and a row
    outside the epoch.
    """
    given: object = family  # checked at run time too: the type is not a guarantee
    if not (isinstance(given, AnimalPTable) or given == "recording"):
        msg = (f"family must be 'recording' or an AnimalPTable, got {given!r}: which one "
               "applies is Andrea's call, so there is no default")
        raise ValueError(msg)
    _check_tests(tests, fs=fs, epoch_start_s=epoch_start_s, n_samples=n_samples)
    tested = {(recording, t.signal, t.minute): t.p for t in tests if t.status == "tested"}
    if isinstance(family, AnimalPTable):
        if family.test_version != TEST_VERSION or family.cleaner != cleaner:
            msg = (f"the animal table was made under {family.test_version} / cleaner "
                   f"{family.cleaner!r}; this pass is {TEST_VERSION} / {cleaner!r}")
            raise ValueError(msg)
        mine = {k: v for k, v in family.pvalues.items() if k[0] == recording}
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
        fam: dict[str, Any] = {"kind": "animal", "animal": family.animal,
                               "size": len(keys), "sha256": family.sha256}
        pv = [family.pvalues[k] for k in keys]
    else:
        keys = sorted(tested)
        fam = {"kind": "recording", "size": len(keys)}
        pv = [tested[k] for k in keys]
    rejected = ({k for k, r in zip(keys, holm_reject(pv, ALPHA), strict=True) if r}
                if keys else set())
    rows = tuple(MinuteResult(t, t.status == "tested"
                              and (recording, t.signal, t.minute) in rejected) for t in tests)
    return LineDistrustRecord(recording, float(fs), float(epoch_start_s), int(n_samples), fam,
                              rows, cleaner)


def cuff_minute_distrust(raw_t: Mapping[str, npt.ArrayLike], fs: float, *, recording: str,
                         epoch_start_s: float, family: Family,
                         cleaner: MainsCleaner | None = None) -> LineDistrustRecord:
    """Both passes for one recording: :func:`minute_tests` (on RAW T) then :func:`decide`.

    With an :class:`AnimalPTable` family, the table must come from pass 1 over the
    animal's recordings, this one included (see :func:`decide`).
    """
    _check_signal_names(raw_t)
    lengths = {int(np.asarray(x).size) for x in raw_t.values()}
    if len(lengths) != 1:
        msg = f"every cuff's T must cover the same epoch; got lengths {sorted(lengths)}"
        raise ValueError(msg)
    tests = minute_tests(raw_t, fs, epoch_start_s=epoch_start_s, cleaner=cleaner)
    return decide(tests, recording=recording, fs=fs, epoch_start_s=epoch_start_s,
                  n_samples=lengths.pop(), family=family,
                  cleaner=None if cleaner is None else cleaner.name)
