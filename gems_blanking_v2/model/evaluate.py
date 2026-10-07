"""Task 12: scores, cluster-bootstrap intervals, calibration, baselines and the run record.

Everything here works on **one judgment per core** (ruling 2026-10-07 (b) R3), never on
windows. ``y`` is 1 for ``motion``; ``unjudged`` and ``unsure`` rows never reach this
module (``labels.training_rows`` drops them).

R9 (ruling 2026-10-07 (b)) is fixed **before any training**: the thresholds live in
:data:`R9_THRESHOLDS` and are written to the run record by :func:`write_run_record`,
which every training entry point requires (:func:`require_run_record`). A record is
write-once: re-writing it with different numbers raises, so a threshold cannot be moved
after results are seen.

Uncertainty is a **cluster** bootstrap (R9): resample whole clusters with replacement -
recordings for the old cohort, audit spans for the new cohort - because cores within a
recording or span are not independent, and an event-level bootstrap would report
intervals that are too narrow.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

from gems_blanking_v2.io.store import atomic_write_text

__all__ = [
    "BASELINE_FEATURE",
    "R9_THRESHOLDS",
    "Calibrator",
    "R9Thresholds",
    "Scores",
    "ThresholdBaseline",
    "cluster_bootstrap_ci",
    "cluster_ids",
    "crossfit_calibrate",
    "ece",
    "paired_diff_ci",
    "reliability_curve",
    "require_run_record",
    "scores",
    "write_run_record",
]

F64 = npt.NDArray[np.float64]
I8 = npt.NDArray[np.int8]

BASELINE_FEATURE: Final = "band_ratio_max_c0"
"""The single-feature baseline: ``log10(P[100-300] / P[300-3000])``, max over nerve
signals, on the core itself. The ENG band is 300-3000 (ruling (b) R8), never 300-5000."""

N_BOOTSTRAP: Final = 1000
"""Cluster-bootstrap resamples (R9)."""

ECE_BINS: Final = 10
"""Equal-width probability bins for the expected calibration error."""


# ---------------------------------------------------------------------------
# R9 - fixed before any training, written to the run record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R9Thresholds:
    """The decision thresholds of ruling 2026-10-07 (b) R9. Never changed after results."""

    ece_max: float = 0.05
    """Calibration passes when expected calibration error on held-out data is at most this."""
    c_beats_b_min_f1_gain: float = 0.03
    """Mode C beats B only if its F1 is higher by at least this ..."""
    c_beats_b_ci: float = 0.95
    """... and the interval of this coverage on the difference excludes 0."""
    min_positives_deciding: int = 20
    """A fold with fewer positives is reported separately and never decides the mode."""
    n_bootstrap: int = N_BOOTSTRAP
    """Cluster-bootstrap resamples."""
    cluster_old: str = "recording"
    """Bootstrap cluster in the old cohort."""
    cluster_new: str = "span"
    """Bootstrap cluster in the new cohort (an audit span)."""
    tier_drop_f1: float = 0.02
    """Rulings (i)/(j): a tier step is dropped if audit-span F1 falls by at least this
    with the 95% interval of the drop excluding 0."""


R9_THRESHOLDS: Final = R9Thresholds()
"""The binding values. A record carrying anything else is refused."""


def write_run_record(path: Path, *, run_id: str, extra: Mapping[str, Any] | None = None
                     ) -> Path:
    """Write the run record with the R9 thresholds, **before** any training.

    Write-once: if ``path`` exists it must carry exactly these thresholds and this
    ``run_id``, otherwise this raises - an existing record is never overwritten.
    ``extra`` (corpus description, params, notes) is stored beside the thresholds.
    """
    body: dict[str, Any] = {"run_id": run_id, "r9": asdict(R9_THRESHOLDS),
                            "written_before_training": True}
    if extra:
        body["extra"] = dict(extra)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("r9") != body["r9"] or old.get("run_id") != run_id:
            msg = (f"run record {path} already exists with different thresholds or run_id; "
                   "R9 thresholds are never changed after a run starts")
            raise ValueError(msg)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(body, indent=1, sort_keys=True, ensure_ascii=True,
                                       allow_nan=False) + "\n")
    return path


def require_run_record(path: Path | None) -> R9Thresholds:
    """Return the thresholds of an existing run record, or raise.

    Training entry points call this first, so no model can be trained before the record
    exists, and a record whose thresholds differ from :data:`R9_THRESHOLDS` is refused.
    """
    if path is None or not Path(path).is_file():
        msg = ("no run record: write the R9 thresholds with write_run_record() before "
               "training (ruling 2026-10-07 (b) R9)")
        raise RuntimeError(msg)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    r9 = raw.get("r9")
    if r9 != asdict(R9_THRESHOLDS):
        msg = f"run record {path} carries R9 thresholds {r9}, not the binding values"
        raise ValueError(msg)
    return R9Thresholds(**r9)


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    """Binary scores of one evaluation set. ``nan`` where a ratio is undefined."""

    n: int
    n_pos: int
    n_pred_pos: int
    prevalence: float
    precision: float
    recall: float
    f1: float


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0 else math.nan


def _f1_counts(tp: float, fp: float, fn: float) -> float:
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else math.nan


def scores(y: npt.ArrayLike, yhat: npt.ArrayLike) -> Scores:
    """Precision, recall, F1 and prevalence of predictions ``yhat`` against ``y``."""
    yt = np.asarray(y).astype(bool)
    yp = np.asarray(yhat).astype(bool)
    if yt.shape != yp.shape:
        msg = f"y and yhat differ in shape: {yt.shape} vs {yp.shape}"
        raise ValueError(msg)
    tp = float(np.sum(yt & yp))
    fp = float(np.sum(~yt & yp))
    fn = float(np.sum(yt & ~yp))
    n = int(yt.size)
    return Scores(n=n, n_pos=int(yt.sum()), n_pred_pos=int(yp.sum()),
                  prevalence=_ratio(float(yt.sum()), n), precision=_ratio(tp, tp + fp),
                  recall=_ratio(tp, tp + fn), f1=_f1_counts(tp, fp, fn))


def cluster_ids(table: pd.DataFrame) -> pd.Series:
    """Return the bootstrap cluster of every row (R9): audit span (new), recording (old).

    A new-cohort row without a ``span_id`` falls back to its recording, so a cluster is
    never empty or shared across recordings.
    """
    rec = table["recording"].astype(str)
    if "span_id" not in table.columns:
        return "rec:" + rec
    span = table["span_id"]
    use_span = (table["cohort"] == "new") & span.notna()
    out = "rec:" + rec
    out.loc[use_span] = "span:" + span.loc[use_span].astype(str)
    return out


def _cluster_counts(clusters: npt.ArrayLike, cols: list[npt.NDArray[np.bool_]]
                    ) -> tuple[npt.NDArray[np.int64], F64]:
    """Per-cluster sums of each boolean column: returns (cluster codes, (k, n_cols))."""
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters)), sort=True)
    k = len(uniq)
    sums = np.zeros((k, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        sums[:, j] = np.bincount(codes, weights=c.astype(np.float64), minlength=k)
    return np.asarray(codes, dtype=np.int64), sums


def _boot_weights(k: int, n_resamples: int, seed: int) -> F64:
    """Multinomial cluster counts per resample: (n_resamples, k)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_resamples, k))
    w = np.zeros((n_resamples, k), dtype=np.float64)
    rows = np.repeat(np.arange(n_resamples), k)
    np.add.at(w, (rows, idx.ravel()), 1.0)
    return w


_STATS: Final[dict[str, Callable[[F64, F64, F64], F64]]] = {
    "f1": lambda tp, fp, fn: np.where(2 * tp + fp + fn > 0,
                                      2 * tp / np.maximum(2 * tp + fp + fn, 1e-300), np.nan),
    "precision": lambda tp, fp, _fn: np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-300),
                                              np.nan),
    "recall": lambda tp, _fp, fn: np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-300),
                                           np.nan),
}


def _confusion(y: npt.ArrayLike, yhat: npt.ArrayLike
               ) -> list[npt.NDArray[np.bool_]]:
    yt = np.asarray(y).astype(bool)
    yp = np.asarray(yhat).astype(bool)
    return [yt & yp, ~yt & yp, yt & ~yp]


def cluster_bootstrap_ci(y: npt.ArrayLike, yhat: npt.ArrayLike, clusters: npt.ArrayLike, *,
                         stat: Literal["f1", "precision", "recall"] = "f1",
                         n_resamples: int = N_BOOTSTRAP, level: float = 0.95,
                         seed: int = 0) -> tuple[float, float]:
    """Percentile interval of ``stat`` over cluster resamples (R9).

    Resamples where the statistic is undefined (no positives drawn) are skipped; if more
    than half are undefined the interval is ``(nan, nan)``.
    """
    _codes, sums = _cluster_counts(clusters, _confusion(y, yhat))
    if sums.shape[0] == 0:
        return (math.nan, math.nan)
    w = _boot_weights(sums.shape[0], n_resamples, seed)
    tp, fp, fn = (w @ sums[:, j] for j in range(3))
    vals = _STATS[stat](tp, fp, fn)
    ok = vals[np.isfinite(vals)]
    if ok.size < n_resamples / 2:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    return (float(np.quantile(ok, a)), float(np.quantile(ok, 1 - a)))


def paired_diff_ci(y: npt.ArrayLike, yhat_a: npt.ArrayLike, yhat_b: npt.ArrayLike,
                   clusters: npt.ArrayLike, *, n_resamples: int = N_BOOTSTRAP,
                   level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """F1(b) - F1(a) on the **same** rows, with a paired cluster-bootstrap interval.

    Both predictions are scored on each resample together, so the interval is of the
    difference, not two overlapping marginal intervals. Returns ``(diff, lo, hi)``.
    """
    cols = _confusion(y, yhat_a) + _confusion(y, yhat_b)
    _codes, sums = _cluster_counts(clusters, cols)
    full = sums.sum(axis=0)
    f1 = _STATS["f1"]
    diff = float(f1(full[3:4], full[4:5], full[5:6])[0] - f1(full[0:1], full[1:2],
                                                             full[2:3])[0])
    w = _boot_weights(sums.shape[0], n_resamples, seed)
    s = w @ sums
    d = f1(s[:, 3], s[:, 4], s[:, 5]) - f1(s[:, 0], s[:, 1], s[:, 2])
    ok = d[np.isfinite(d)]
    if ok.size < n_resamples / 2:
        return (diff, math.nan, math.nan)
    a = (1 - level) / 2
    return (diff, float(np.quantile(ok, a)), float(np.quantile(ok, 1 - a)))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def reliability_curve(p: npt.ArrayLike, y: npt.ArrayLike, n_bins: int = ECE_BINS
                      ) -> pd.DataFrame:
    """Per equal-width bin: lower edge, mean predicted, observed fraction, count."""
    pp = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    yy = np.asarray(y).astype(np.float64)
    b = np.minimum((pp * n_bins).astype(np.int64), n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        n = int(m.sum())
        rows.append({"bin_lo": k / n_bins, "mean_p": float(pp[m].mean()) if n else math.nan,
                     "observed": float(yy[m].mean()) if n else math.nan, "count": n})
    return pd.DataFrame(rows)


def ece(p: npt.ArrayLike, y: npt.ArrayLike, n_bins: int = ECE_BINS) -> float:
    """Return the expected calibration error: mean |observed - predicted| over bins.

    The mean is weighted by bin count; empty bins are skipped.
    """
    rc = reliability_curve(p, y, n_bins)
    rc = rc[rc["count"] > 0]
    n = float(rc["count"].sum())
    if n == 0:
        return math.nan
    return float((rc["count"] * (rc["observed"] - rc["mean_p"]).abs()).sum() / n)


_EPS: Final = 1e-6
_NEWTON_TOL: Final = 1e-10


def _logit(p: F64) -> F64:
    q = np.clip(p, _EPS, 1 - _EPS)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


@dataclass(frozen=True)
class Calibrator:
    """A monotone map from raw model score to calibrated ``P(motion)``.

    ``isotonic``: piecewise-linear through ``(x, y)`` breakpoints, clipped outside them
    (what ``IsotonicRegression(out_of_bounds="clip").predict`` computes). ``platt``:
    ``sigmoid(a * logit(s) + b)``. Serialised as plain JSON - never a pickle - so it is
    portable and inspectable.
    """

    kind: Literal["isotonic", "platt"]
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, raw: npt.ArrayLike, y: npt.ArrayLike,
            kind: Literal["isotonic", "platt"] = "isotonic") -> Calibrator:
        """Fit on held-out raw scores and their labels."""
        s = np.asarray(raw, dtype=np.float64)
        t = np.asarray(y).astype(np.float64)
        if s.size == 0:
            msg = "cannot fit a calibrator on zero rows"
            raise ValueError(msg)
        if kind == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(s, t)
            return cls(kind, x=tuple(float(v) for v in iso.X_thresholds_),
                       y=tuple(float(v) for v in iso.y_thresholds_))
        a, b = _platt_fit(_logit(s), t)
        return cls("platt", a=a, b=b)

    def apply(self, raw: npt.ArrayLike) -> F64:
        """Calibrated probabilities for raw scores."""
        s = np.asarray(raw, dtype=np.float64)
        if self.kind == "isotonic":
            if len(self.x) == 1:
                return np.full(s.shape, self.y[0], dtype=np.float64)
            return np.asarray(np.interp(s, self.x, self.y), dtype=np.float64)
        return np.asarray(expit(self.a * _logit(s) + self.b), dtype=np.float64)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, ASCII, no NaN)."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> Calibrator:
        """Inverse of :meth:`to_json`."""
        raw = json.loads(text)
        return cls(kind=raw["kind"], x=tuple(raw["x"]), y=tuple(raw["y"]),
                   a=float(raw["a"]), b=float(raw["b"]))


def _platt_fit(z: F64, t: F64, iters: int = 100) -> tuple[float, float]:
    """Logistic regression of ``t`` on ``z``: damped Newton with a backtracking line search.

    A plain Newton step diverges on real LightGBM scores (many near 0 and 1, logits at
    the +-13.8 clip): measured on a provisional fold it returned ``a = 5e8``. Every step
    here must lower the log-loss, so the fit cannot run away.
    """

    def loss(a: float, b: float) -> float:
        u = a * z + b
        return float(np.sum(np.logaddexp(0.0, u) - t * u))

    a, b = 1.0, 0.0
    cur = loss(a, b)
    for _ in range(iters):
        p = expit(a * z + b)
        w = np.maximum(p * (1 - p), 1e-12)
        g = np.array([np.sum((p - t) * z), np.sum(p - t)])
        h = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        h += 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        lr = 1.0
        while lr > _NEWTON_TOL:
            na, nb = a - lr * float(step[0]), b - lr * float(step[1])
            new = loss(na, nb)
            if new <= cur:
                break
            lr /= 2
        else:
            break
        moved = max(abs(na - a), abs(nb - b))
        a, b, cur = na, nb, new
        if moved < _NEWTON_TOL:
            break
    return a, b


def crossfit_calibrate(raw: npt.ArrayLike, y: npt.ArrayLike, clusters: npt.ArrayLike, *,
                       kind: Literal["isotonic", "platt"] = "isotonic") -> F64:
    """Return cross-fitted calibrated probabilities, held out from their own calibration.

    Each cluster is mapped by a calibrator fitted on the **other** clusters, so ECE
    computed on the result is a held-out ECE.

    A cluster whose complement holds only one class gets ``nan`` (no calibrator can be
    fitted without both classes), never an invented value.
    """
    s = np.asarray(raw, dtype=np.float64)
    t = np.asarray(y).astype(np.int8)
    c = np.asarray(clusters)
    out = np.full(s.shape, np.nan, dtype=np.float64)
    for k in pd.unique(pd.Series(c)):
        m = c == k
        rest = ~m
        if rest.sum() == 0 or np.unique(t[rest]).size < 2:  # noqa: PLR2004
            continue
        out[m] = Calibrator.fit(s[rest], t[rest], kind).apply(s[m])
    return out


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdBaseline:
    """Single-feature threshold: motion when ``direction * (x - threshold) >= 0``.

    Fitted on training rows only (threshold and direction maximise training F1). A row
    whose feature is missing is predicted not-motion and counted in ``n_missing``.
    """

    feature: str
    threshold: float
    direction: int

    @classmethod
    def fit(cls, x: npt.ArrayLike, y: npt.ArrayLike, feature: str = BASELINE_FEATURE
            ) -> ThresholdBaseline:
        """Choose threshold and direction on the training rows."""
        v = np.asarray(x, dtype=np.float64)
        t = np.asarray(y).astype(bool)
        ok = np.isfinite(v)
        v, t = v[ok], t[ok]
        if v.size == 0 or t.sum() == 0:
            msg = f"cannot fit a threshold on {feature}: no finite values or no positives"
            raise ValueError(msg)
        best = (-1.0, 0.0, 1)
        cands = np.unique(np.quantile(v, np.linspace(0, 1, 201)))
        for d in (1, -1):
            for th in cands:
                pred = d * (v - th) >= 0
                f = _f1_counts(float(np.sum(t & pred)), float(np.sum(~t & pred)),
                               float(np.sum(t & ~pred)))
                if np.isfinite(f) and f > best[0]:
                    best = (f, float(th), d)
        return cls(feature=feature, threshold=best[1], direction=best[2])

    def predict(self, x: npt.ArrayLike) -> I8:
        """0/1 predictions; missing values are 0."""
        v = np.asarray(x, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            pred = np.isfinite(v) & (self.direction * (v - self.threshold) >= 0)
        return pred.astype(np.int8)
