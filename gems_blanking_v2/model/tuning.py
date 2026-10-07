"""Task 12: hyperparameter tuning, only ever nested inside a training fold.

RULING 2026-10-08 item 1: fixed parameters for the provisional runs; for final training,
tuning (optuna, as in GEMSBlanking's ``hyperopt_worker.py``) is allowed **only nested
inside each LOAO training fold, never on the held-out animal**. So:

* Tuning is **off by default**: :func:`~gems_blanking_v2.model.modes.run_modes` takes a
  ``tuner`` that defaults to ``None`` and then trains with
  :data:`~gems_blanking_v2.model.params.FIXED_PARAMS`.
* A tuner is called with the rows of ONE fold's training corpus and their groups, and
  nothing else - it cannot see the held-out animal because it is never handed it.
  :func:`~gems_blanking_v2.model.modes.run_modes` also asserts, per call, that the
  target's animal key (LOAO) or the held-out recording (LORO) is not among the groups.
* The inner validation (:func:`inner_cv_f1`) is leave-one-group-out over those groups:
  the training fold's animals for A and B, the target's training recordings for C.
* GEMSBlanking's own search space (``w_neg``, ``fp_weight``, ``fn_weight``) is not reused:
  ``w_neg`` weights unlabelled windows as negatives (invariant 9). The space here is
  LightGBM's own parameters (:data:`SEARCH_SPACE`), starting from ``retrain.py``'s
  defaults.

``optuna`` is not installed in this environment; :class:`OptunaInnerCV` raises a clear
``ImportError`` when called without it. The tuner's :meth:`OptunaInnerCV.record` goes
into the run record's ``extra["tuning"]``, which ``run_modes`` checks.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.model.evaluate import DECISION_P, scores
from gems_blanking_v2.model.params import FIXED_PARAMS, NUM_BOOST_ROUND
from gems_blanking_v2.model.train import fit, predict_raw, prior_weights

__all__ = [
    "SEARCH_SPACE",
    "TUNING_OFF",
    "OptunaInnerCV",
    "Tuner",
    "inner_cv_f1",
    "tuning_record",
]

SEARCH_SPACE: Final[Mapping[str, tuple[str, float, float, bool]]] = {
    "num_leaves": ("int", 7, 63, True),
    "min_data_in_leaf": ("int", 10, 200, True),
    "learning_rate": ("float", 0.01, 0.2, True),
    "feature_fraction": ("float", 0.5, 1.0, False),
    "lambda_l2": ("float", 1e-3, 10.0, True),
}
"""``name: (kind, low, high, log)``. Everything else stays at ``FIXED_PARAMS``."""

TUNING_OFF: Final[Mapping[str, Any]] = {"enabled": False}
"""The run-record entry of a run without tuning (an absent entry reads the same)."""


class Tuner(Protocol):
    """Chooses parameters from ONE training fold's rows, by inner validation on them."""

    def __call__(self, x: pd.DataFrame, y: npt.NDArray[np.int8], w: npt.NDArray[np.float64],
                 groups: npt.NDArray[np.str_], *, scorable: npt.NDArray[np.bool_],
                 meta: pd.DataFrame, num_threads: int) -> Mapping[str, Any]:
        """Return the full LightGBM parameter dict to train this fold with.

        ``w`` are the base weights; ``scorable`` marks the rows the outer protocol scores;
        ``meta`` (``cohort``, ``basis``, ``y``, ``recording``) lets the inner folds
        recompute the old-cohort prior correction on their own training rows.
        """
        ...

    def record(self) -> Mapping[str, Any]:
        """Return the run record's ``extra["tuning"]`` entry."""
        ...


def tuning_record(tuner: Tuner | None) -> dict[str, Any]:
    """Return the run-record entry for ``tuner`` (``TUNING_OFF`` for none)."""
    return dict(TUNING_OFF) if tuner is None else dict(tuner.record())


def inner_cv_f1(params: Mapping[str, Any], x: pd.DataFrame, y: npt.ArrayLike,
                w: npt.ArrayLike, groups: npt.ArrayLike, *, rounds: int, num_threads: int,
                scorable: npt.ArrayLike | None = None,
                meta: pd.DataFrame | None = None) -> float:
    """Pooled F1 (at ``DECISION_P``) of leave-one-group-out predictions on these rows only.

    Each group is predicted by a model trained on the other groups; a group whose
    complement holds one class is skipped. Only rows marked ``scorable`` (the outer
    protocol's scorable rows: new-cohort audit spans, every old row) are scored - the
    rest train only, as in the outer folds. With ``meta`` each inner training set's
    weights are ``w`` times the old-cohort prior correction recomputed on that set
    (:func:`~gems_blanking_v2.model.train.prior_weights`), as every outer fit is. ``nan``
    when fewer than two groups can be scored or F1 is undefined.
    """
    yy = np.asarray(y).astype(np.int8)
    ww = np.asarray(w, dtype=np.float64)
    gg = np.asarray(groups).astype(str)
    sc = (np.ones(len(yy), dtype=bool) if scorable is None
          else np.asarray(scorable).astype(bool))
    ys, ps = [], []
    for g in sorted(set(gg.tolist())):
        held = gg == g
        ev = held & sc
        if not ev.any() or np.unique(yy[~held]).size < 2:  # noqa: PLR2004
            continue
        wt = ww[~held]
        if meta is not None:
            wt = wt * prior_weights(meta.loc[~held], base=wt)[0]
        booster = fit(x.loc[~held], yy[~held], wt, num_threads=num_threads,
                      rounds=rounds, params=params)
        ys.append(yy[ev])
        ps.append(predict_raw(booster, x.loc[ev]) >= DECISION_P)
    if len(ys) < 2:  # noqa: PLR2004
        return float("nan")
    return scores(np.concatenate(ys), np.concatenate(ps)).f1


@dataclass(frozen=True)
class OptunaInnerCV:
    """Optuna TPE search over :data:`SEARCH_SPACE`, scored by :func:`inner_cv_f1`."""

    n_trials: int = 30
    seed: int = 0
    rounds: int = NUM_BOOST_ROUND
    space: Mapping[str, tuple[str, float, float, bool]] = field(
        default_factory=lambda: dict(SEARCH_SPACE))

    def record(self) -> dict[str, Any]:
        """Return the run-record entry: backend, budget, seed, space, objective, nesting."""
        return {"enabled": True, "backend": "optuna TPE", "n_trials": self.n_trials,
                "seed": self.seed, "rounds": self.rounds,
                "space": {k: list(v) for k, v in self.space.items()},
                "objective": "inner leave-one-group-out F1 at DECISION_P",
                "nesting": ("inside each training fold only: groups are the fold's training "
                            "animals (A, B) or the target's training recordings (C); the "
                            "held-out animal or recording is never passed")}

    def __call__(self, x: pd.DataFrame, y: npt.NDArray[np.int8], w: npt.NDArray[np.float64],
                 groups: npt.NDArray[np.str_], *, scorable: npt.NDArray[np.bool_],
                 meta: pd.DataFrame, num_threads: int) -> dict[str, Any]:
        """Search, then return ``FIXED_PARAMS`` updated with the best trial's values."""
        try:
            optuna = importlib.import_module("optuna")
        except ImportError as exc:
            msg = ("tuning needs optuna, which is not installed here; tuning is off by "
                   "default (tuner=None trains with FIXED_PARAMS)")
            raise ImportError(msg) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:  # noqa: ANN401 - optuna's Trial, untyped here
            params = dict(FIXED_PARAMS)
            for name, (kind, lo, hi, log) in self.space.items():
                if kind == "int":
                    params[name] = trial.suggest_int(name, int(lo), int(hi), log=log)
                else:
                    params[name] = trial.suggest_float(name, lo, hi, log=log)
            v = inner_cv_f1(params, x, y, w, groups, rounds=self.rounds,
                            num_threads=num_threads, scorable=scorable, meta=meta)
            return v if np.isfinite(v) else 0.0

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return {**FIXED_PARAMS, **study.best_params}
