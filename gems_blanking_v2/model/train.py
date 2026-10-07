"""Task 12: the binary LightGBM classifier (``y = 1`` for motion) on core features.

* **Native missing values (ruling (b) R2).** Features absent for a cohort (the
  common-mode family on the 5-channel old cohort) stay ``nan``; LightGBM routes them
  natively. Nothing is imputed.
* **Fixed hyperparameters.** :data:`FIXED_PARAMS` is used for every mode, so "the same
  hyperparameter search" (task 12) holds trivially. No search is run in this build
  (``optuna`` is not installed); the values are recorded in every provenance record.
  They differ from GEMSBlanking's window-level ``DEFAULT_HPARAMS`` (``min_data_in_leaf
  = 100`` suits 10^5 windows, not a few hundred cores per animal).
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
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.detect.features import FEATURE_NAMES
from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.model.evaluate import W_ADAPT_GRID

__all__ = [
    "ADAPT_ROUNDS",
    "FIXED_PARAMS",
    "NUM_BOOST_ROUND",
    "W_ADAPT_GRID",
    "LeakyFeatureError",
    "check_leakage",
    "corpus_hash",
    "event_id",
    "feature_columns",
    "fit",
    "predict_raw",
    "sample_weights",
    "shap_top_features",
]

F64 = npt.NDArray[np.float64]

FIXED_PARAMS: Final[Mapping[str, Any]] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "use_missing": True,
    "zero_as_missing": False,
}
"""LightGBM parameters for every mode. ``use_missing`` keeps ``nan`` native (R2);
``zero_as_missing`` is off because a feature value of exactly 0 is a value."""

NUM_BOOST_ROUND: Final = 300
"""Boosting rounds for a model trained from scratch (no early stopping: no inner
validation split is carved out of corpora this small)."""

ADAPT_ROUNDS: Final = 100
"""Extra rounds when mode B continues a pooled model with ``init_model``."""


W_VIDEO: Final = 3.0
"""Up-weight of ``provenance == 'video_assisted'`` positives."""

LEAK_ETA2: Final = 0.999
"""A feature whose between-recording variance share reaches this is an identifier."""

LEAK_MIN_GROUPS: Final = 3
"""Below this many recordings with values, the share is not informative."""

NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset({
    "recording", "animal", "cohort", "start_s", "stop_s", "judgement", "source", "basis",
    "label_set", "label_source", "span_id", "peak_signal", "peak_band", "y", "cluster",
    "animal_key", "provenance", "folder",
})
"""Bookkeeping columns that must never be trained on."""


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
