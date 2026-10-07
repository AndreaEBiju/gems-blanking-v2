"""Task 12: the binary LightGBM classifier (``y = 1`` for motion) on core features.

* **Native missing values (ruling (b) R2).** A feature whose input is absent stays
  ``nan``; LightGBM routes it natively. Nothing is imputed.
* **Fixed hyperparameters (RULING 2026-10-08 item 1).** :data:`FIXED_PARAMS` is
  ``retrain.py``'s LightGBM defaults (:mod:`~gems_blanking_v2.model.params`) and is used
  for every mode, so "the same hyperparameter search" (task 12) holds. Tuning is allowed
  only nested inside a training fold and is off by default
  (:mod:`~gems_blanking_v2.model.tuning`); the values used are in the run record.
* **Leakage check.** :func:`check_leakage` refuses any feature that identifies the
  recording - a deliberately leaky "recording index" is the test - and the only columns
  ever trained on are :data:`~gems_blanking_v2.detect.features.FEATURE_NAMES`.
* **Video-assisted positives are weighted up** by ``w_video`` when the table carries a
  ``provenance`` column (task 12). Video is out of this build (ruling (b) R5), so in
  practice every weight is 1; the mechanism and its cost measurement wait for task 17.
* **No temporal smoothing** (PIPELINE.md 10.4).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.detect.features import FEATURE_NAMES, FEATURE_VERSION, FEATURE_VERSION_COLUMN
from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.model.evaluate import W_ADAPT_GRID
from gems_blanking_v2.model.labels import HUMAN_OLD_BASES, SET_A_BASIS
from gems_blanking_v2.model.params import ADAPT_ROUNDS, FIXED_PARAMS, NUM_BOOST_ROUND

__all__ = [
    "ADAPT_ROUNDS",
    "FIXED_PARAMS",
    "MIN_SET_A_OLD_NEGATIVES",
    "NUM_BOOST_ROUND",
    "W_ADAPT_GRID",
    "FeatureVersionError",
    "LeakyFeatureError",
    "OldRowsRefusedError",
    "PriorCorrection",
    "check_feature_version",
    "check_leakage",
    "corpus_hash",
    "event_id",
    "feature_columns",
    "fit",
    "predict_raw",
    "prior_correction",
    "prior_weights",
    "sample_weights",
    "shap_top_features",
]

F64 = npt.NDArray[np.float64]

W_VIDEO: Final = 3.0
"""Up-weight of ``provenance == 'video_assisted'`` positives."""

LEAK_ETA2: Final = 0.999
"""A feature whose between-recording variance share reaches this is an identifier."""

LEAK_MIN_GROUPS: Final = 3
"""Below this many recordings with values, the share is not informative."""

NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset({
    "recording", "animal", "cohort", "start_s", "stop_s", "judgement", "source", "basis",
    "label_set", "label_source", "span_id", "peak_signal", "peak_band", "y", "cluster",
    "animal_key", "provenance", "folder", FEATURE_VERSION_COLUMN,
})
"""Bookkeeping columns that must never be trained on."""


class FeatureVersionError(ValueError):
    """A feature table was computed under other feature definitions (or carries no stamp)."""


def check_feature_version(table: pd.DataFrame, *, context: str = "training") -> None:
    """Refuse a table whose ``feature_version`` is absent or not :data:`FEATURE_VERSION`.

    Rows computed under different definitions are never trained or scored together - a
    Night 1 table (unstamped, contact families, count-dependent maxima) is refused.
    """
    if FEATURE_VERSION_COLUMN not in table.columns:
        msg = (f"the table for {context} carries no {FEATURE_VERSION_COLUMN!r} column; feature "
               f"tables must be stamped with the definitions they were computed under "
               f"(current: {FEATURE_VERSION!r})")
        raise FeatureVersionError(msg)
    found = sorted(set(table[FEATURE_VERSION_COLUMN].astype(str)))
    if found != [FEATURE_VERSION]:
        msg = (f"the table for {context} was computed under feature version(s) {found}; this "
               f"code is {FEATURE_VERSION!r}")
        raise FeatureVersionError(msg)


class LeakyFeatureError(ValueError):
    """A feature identifies the recording (or is bookkeeping) and would leak labels."""


def feature_columns(table: pd.DataFrame,
                    names: Sequence[str] = FEATURE_NAMES) -> list[str]:
    """Return the trainable columns: exactly ``names``, all of which must be present.

    Raises naming the missing columns, or any name that is a bookkeeping column.
    """
    bad = sorted(set(names) & NON_FEATURE_COLUMNS)
    if bad:
        msg = f"bookkeeping columns cannot be features: {bad}"
        raise LeakyFeatureError(msg)
    missing = [c for c in names if c not in table.columns]
    if missing:
        msg = f"feature table is missing columns: {missing[:10]}{'...' if len(missing) > 10 else ''}"  # noqa: E501, PLR2004
        raise ValueError(msg)
    return list(names)


def check_leakage(x: pd.DataFrame, recordings: npt.ArrayLike) -> None:
    """Refuse features that identify the recording.

    A feature leaks when its value is (nearly) constant within each recording and
    differs between recordings: its between-recording share of variance (eta squared,
    on non-missing rows) reaches :data:`LEAK_ETA2` across at least
    :data:`LEAK_MIN_GROUPS` recordings that hold two or more rows (a singleton has no
    within-recording spread to compare against). Such a column lets a model memorise which
    recordings were labelled motion-heavy instead of learning motion. Any column named
    in :data:`NON_FEATURE_COLUMNS` is refused outright.

    Raises
    ------
    LeakyFeatureError
        Naming every offending column.
    """
    bad = sorted(set(x.columns) & NON_FEATURE_COLUMNS)
    rec = pd.Series(np.asarray(recordings), index=x.index)
    for col in x.columns:
        if col in bad:
            continue
        v = pd.to_numeric(x[col], errors="coerce")
        ok = v.notna()
        if ok.sum() < LEAK_MIN_GROUPS:
            continue
        sizes = rec[ok].value_counts()
        multi = sizes[sizes >= 2].index  # noqa: PLR2004 - a singleton has no spread
        if len(multi) < LEAK_MIN_GROUPS:
            continue
        keep = ok & rec.isin(multi)
        g = rec[keep]
        vv = v[keep].astype(np.float64)
        total = float(((vv - vv.mean()) ** 2).sum())
        if total <= 0:
            continue
        within = float(((vv - vv.groupby(g).transform("mean")) ** 2).sum())
        if 1 - within / total >= LEAK_ETA2:
            bad.append(col)
    if bad:
        msg = (f"leaky features (identify the recording): {sorted(bad)}; remove them "
               "before training")
        raise LeakyFeatureError(msg)


def sample_weights(table: pd.DataFrame, *, w_video: float = W_VIDEO) -> F64:
    """Per-row weight: ``w_video`` for video-assisted positives, 1 otherwise."""
    w = np.ones(len(table), dtype=np.float64)
    if "provenance" in table.columns and "y" in table.columns:
        m = (table["provenance"] == "video_assisted").to_numpy() & (table["y"] == 1).to_numpy()
        w[m] = w_video
    return w


class OldRowsRefusedError(ValueError):
    """Old-cohort rows reached training without set A's judged old negatives."""


MIN_SET_A_OLD_NEGATIVES: Final = 20
"""Fewest set-A old negatives with which old-cohort rows may train (PROPOSED, a question
for Andrea): the bar R9 sets for a deciding fold's positives, applied to the only source
of old negatives. Below it the old motion rate's CI is too wide to weight anything."""

PRIOR_CI_RESAMPLES: Final = 1000
"""Cluster-bootstrap resamples (by recording) for the CI of set A's old motion rate."""


@dataclass(frozen=True)
class PriorCorrection:
    """The prior-corrected weight of old-cohort positives (ruling 2026-10-08 (b) 1(c)).

    The old marks are positives only, so taken at face value they make the old cohort
    look nearly all motion and teach "old cohort means motion". Set A's random old sample
    is unbiased and holds both classes: its motion rate ``rate`` (``k_set_a / n_set_a``,
    unsure and unjudged excluded) estimates the old cohort's. Every old-cohort POSITIVE
    row (set A's, other adjudicated cores' and the marks) gets weight ``w_pos``, chosen so
    the effective old-cohort motion rate of the training corpus,
    ``w_pos * n_pos / (w_pos * n_pos + n_neg)``, equals ``rate``; old negatives (set A's
    and other adjudicated ones) keep weight 1. Only set A's random rows estimate the rate;
    a hum add-on or other targeted draw never does. ``ci`` is the 95% cluster-bootstrap
    interval of ``rate``
    (resampling recordings, R9's old-cohort cluster) and ``w_pos_ci`` the weights at its
    ends - carried into provenance.
    """

    n_set_a: int
    k_set_a: int
    n_marks: int
    rate: float
    ci: tuple[float, float]
    w_pos: float
    w_pos_ci: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """JSON form for provenance (a value that is undefined is absent)."""
        out: dict[str, Any] = {
            "rule": "ruling 2026-10-08 (b) item 1(c): effective old motion rate = set A's",
            "n_set_a": self.n_set_a, "k_set_a": self.k_set_a, "n_marks": self.n_marks,
            "rate": self.rate, "w_pos": self.w_pos}
        if all(map(math.isfinite, self.ci)):
            out["rate_ci95"] = list(self.ci)
            if all(map(math.isfinite, self.w_pos_ci)):
                out["w_pos_at_ci"] = list(self.w_pos_ci)
            else:
                out["w_pos_at_ci_omitted"] = (
                    "the CI reaches a rate of 1, where no finite weight gives that rate")
        return out


def _w_pos(rate: float, n_pos: float, n_neg: float) -> float:
    """Multiplier on each old positive so that w * n_pos / (w * n_pos + n_neg) == rate.

    ``n_pos`` / ``n_neg`` are the base-weighted sums of the old positives / negatives.
    """
    if n_pos == 0 or rate <= 0:
        return 0.0
    if rate >= 1:
        return math.inf
    return rate * n_neg / ((1 - rate) * n_pos)


def prior_correction(table: pd.DataFrame, *, base: npt.ArrayLike | None = None,
                     seed: int = 0) -> PriorCorrection | None:
    """Return the old-cohort correction of a TRAINING corpus, or ``None`` without old rows.

    ``base`` are the rows' weights before the correction (``w_adapt``, ``w_video``; 1 when
    absent): ``w_pos`` solves the effective-rate equation on weighted sums, so the rate
    holds whatever the base weights. Raises :class:`OldRowsRefusedError` when old rows
    are present but set A holds fewer than :data:`MIN_SET_A_OLD_NEGATIVES` judged old
    negatives (ruling (b) item 1(b): old rows train only alongside them).
    """
    old = (table["cohort"] == "old").to_numpy()
    if not old.any():
        return None
    set_a = old & (table["basis"] == SET_A_BASIS).to_numpy()
    y = table["y"].to_numpy().astype(np.int8)
    n_a, k_a = int(set_a.sum()), int(y[set_a].sum())
    if n_a - k_a < MIN_SET_A_OLD_NEGATIVES:
        msg = (f"{int(old.sum())} old-cohort row(s) reached training without set A's judged "
               f"old negatives ({n_a} set-A rows, {n_a - k_a} negative; at least "
               f"{MIN_SET_A_OLD_NEGATIVES} needed); ruling 2026-10-08 (b) item 1(b): old-cohort "
               "rows train only alongside old negatives judged by Andrea - provisional runs "
               "are new-cohort only")
        raise OldRowsRefusedError(msg)
    human = old & table["basis"].isin(sorted(HUMAN_OLD_BASES)).to_numpy()
    marks = old & ~human
    if (y[marks] != 1).any():
        msg = ("an inherited old-cohort row is not a positive; old marks are positives only "
               "(only Andrea's own judgements may be old negatives)")
        raise ValueError(msg)
    bw = np.ones(len(table)) if base is None else np.asarray(base, dtype=np.float64)
    n_pos = float(bw[old & (y == 1)].sum())
    n_neg = float(bw[old & (y == 0)].sum())
    rate = k_a / n_a
    rec = table["recording"].astype(str).to_numpy()[set_a]
    codes, uniq = pd.factorize(pd.Series(rec), sort=True)
    kk = np.bincount(codes, weights=y[set_a].astype(np.float64), minlength=len(uniq))
    nn = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(uniq), size=(PRIOR_CI_RESAMPLES, len(uniq)))
    rates = kk[draw].sum(axis=1) / nn[draw].sum(axis=1)
    lo, hi = (float(v) for v in np.quantile(rates, [0.025, 0.975]))
    return PriorCorrection(n_set_a=n_a, k_set_a=k_a, n_marks=int(marks.sum()), rate=rate,
                           ci=(lo, hi), w_pos=_w_pos(rate, n_pos, n_neg),
                           w_pos_ci=(_w_pos(lo, n_pos, n_neg), _w_pos(hi, n_pos, n_neg)))


def prior_weights(table: pd.DataFrame, *, base: npt.ArrayLike | None = None, seed: int = 0
                  ) -> tuple[F64, PriorCorrection | None]:
    """Per-row multipliers of a training corpus: ``w_pos`` on old positives, 1 elsewhere.

    Multiply them into ``base`` (the same weights passed here) to train.
    """
    corr = prior_correction(table, base=base, seed=seed)
    w = np.ones(len(table), dtype=np.float64)
    if corr is not None:
        pos_old = (table["cohort"] == "old").to_numpy() & (table["y"] == 1).to_numpy()
        w[pos_old] = corr.w_pos
    return w, corr


def fit(x: pd.DataFrame, y: npt.ArrayLike, w: npt.ArrayLike | None = None, *,
        num_threads: int, rounds: int = NUM_BOOST_ROUND,
        init_model: lgb.Booster | None = None,
        params: Mapping[str, Any] = FIXED_PARAMS) -> lgb.Booster:
    """Train (or, with ``init_model``, continue) a binary booster on ``x``.

    ``num_threads`` is required: the caller owns the core budget.
    Raises if ``y`` holds a single class - a one-class model is not a classifier.
    """
    yy = np.asarray(y).astype(np.int8)
    if np.unique(yy).size < 2:  # noqa: PLR2004
        msg = f"training labels hold a single class ({np.unique(yy).tolist()})"
        raise ValueError(msg)
    p = {**params, "num_threads": int(num_threads)}
    ds = lgb.Dataset(x.astype(np.float64), label=yy,
                     weight=None if w is None else np.asarray(w, dtype=np.float64),
                     free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds, init_model=init_model,
                     keep_training_booster=True)


def predict_raw(booster: lgb.Booster, x: pd.DataFrame) -> F64:
    """Uncalibrated ``P(motion)`` from the booster."""
    return np.asarray(booster.predict(x.astype(np.float64)), dtype=np.float64)


def _canon_s(t: float) -> str:
    """Return the one canonical text form of a time in seconds (invariant 22)."""
    if not math.isfinite(t):
        msg = f"event time must be finite, got {t!r}"
        raise ValueError(msg)
    return f"{t:.6f}"


def event_id(recording: str, start_s: float, stop_s: float) -> str:
    """Identity of one judged core: ``recording|start|stop``, times at 1 us."""
    return f"{recording}|{_canon_s(start_s)}|{_canon_s(stop_s)}"


def corpus_hash(table: pd.DataFrame) -> str:
    """SHA-256 of the sorted training event ids (and labels): the model's corpus identity.

    Uniqueness is asserted (invariant 27): two rows with one id raise.
    """
    ids = [event_id(r, a, b) for r, a, b in
           zip(table["recording"].astype(str), table["start_s"], table["stop_s"], strict=True)]
    if len(set(ids)) != len(ids):
        msg = "duplicate event ids in the training corpus; a core was judged twice"
        raise ValueError(msg)
    ys = table["y"].astype(int).tolist() if "y" in table.columns else [0] * len(ids)
    body = "\n".join(f"{i}|{y}" for i, y in sorted(zip(ids, ys, strict=True)))
    return hashlib.sha256(body.encode("ascii")).hexdigest()


def shap_top_features(booster: lgb.Booster, x: pd.DataFrame, top_n: int = 20
                      ) -> pd.DataFrame:
    """Mean |SHAP| per feature, largest first, via GEMSBlanking's ``detector.review``.

    Uses ``detector.review.compute_shap_for_windows`` (LightGBM ``pred_contrib``) by
    import, not a copy. Raises ``FileNotFoundError`` / ``ImportError`` where the private
    checkout is absent.
    """
    review = import_detector_module("review")
    _top, shap, _base = review.compute_shap_for_windows(booster, x.astype(np.float64), 1)
    mean_abs = np.mean(np.abs(np.asarray(shap, dtype=np.float64)), axis=0)
    out = pd.DataFrame({"feature": booster.feature_name(), "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
