"""Task 12: the fixed LightGBM parameters and the record of what was reused from GEMSBlanking.

RULING 2026-10-08 item 1: ``retrain.py`` cannot be reused for training - it cannot read
the per-core feature table and it trains unlabelled windows as negatives, which breaks
invariant 9 (``unjudged`` is not ``negative``). What carries no label semantics is
reused, and the deviation from task 12's reuse list is written into every run record
(:data:`TASK12_REUSE`, via :func:`~gems_blanking_v2.model.evaluate.run_protocol`).

The parameters are ``retrain.py``'s LightGBM defaults - ``detector/model.py``'s
``DEFAULT_HPARAMS``, which ``retrain.py`` trains with (``hp = dict(M.DEFAULT_HPARAMS)``) -
held here by value so training runs where the private checkout is absent, and checked
against the checkout by :func:`verify_retrain_defaults` (a test, and the provisional
drivers, call it). Only two keys are added, both about missing values (ruling (b) R2).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from gems_blanking_v2.io.detector_core import import_detector_module

__all__ = [
    "ADAPT_ROUNDS",
    "FIXED_PARAMS",
    "NUM_BOOST_ROUND",
    "RETRAIN_DEFAULTS_SOURCE",
    "RETRAIN_DEFAULT_HPARAMS",
    "TASK12_REUSE",
    "verify_retrain_defaults",
]

RETRAIN_DEFAULTS_SOURCE: Final = (
    "GEMSBlanking detector/model.py DEFAULT_HPARAMS - the parameters detector/retrain.py "
    "trains with (hp = dict(M.DEFAULT_HPARAMS)); read 2026-10-07 from the detector-core "
    "submodule of detector-pyqt")

RETRAIN_DEFAULT_HPARAMS: Final[Mapping[str, Any]] = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "min_data_in_leaf": 100,
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}
"""``retrain.py``'s LightGBM defaults, by value (:func:`verify_retrain_defaults`)."""

FIXED_PARAMS: Final[Mapping[str, Any]] = {
    **RETRAIN_DEFAULT_HPARAMS,
    "use_missing": True,
    "zero_as_missing": False,
}
"""The parameters of every mode in the provisional runs: :data:`RETRAIN_DEFAULT_HPARAMS`
plus ``use_missing`` (``nan`` stays native, R2) and ``zero_as_missing = False`` (a
feature value of exactly 0 is a value). Tuning, when enabled, starts from these
(:mod:`~gems_blanking_v2.model.tuning`)."""

NUM_BOOST_ROUND: Final = 300
"""Boosting rounds for a model trained from scratch. ``retrain.py`` runs up to 2000
with early stopping on a random 10% split of its rows; a random split of cores puts
cores of one recording on both sides, so that stopping point would be chosen on
leaked rows. Fixed rounds instead; a tuned run may tune them inside the fold."""

ADAPT_ROUNDS: Final = 100
"""Extra rounds when mode B continues a pooled model with ``init_model``."""

TASK12_REUSE: Final[Mapping[str, Any]] = {
    "ruling": "RULING 2026-10-08 item 1",
    "task12_reuse_list": [
        "GEMSBlanking detector/retrain.py (LightGBM + hyperopt)",
        "detector-pyqt ui/workers/hyperopt_worker.py",
        "detector/review.py (SHAP HTMLs)",
        "detector/heldout_eval.py",
        "detector/animal_id.py extract_animal_letter",
        "promotion / rollback / current_model pointer / provenance.json machinery",
    ],
    "reused": {
        "detector/model.py DEFAULT_HPARAMS (retrain.py's LightGBM defaults)": (
            "the fixed parameters of every mode: FIXED_PARAMS = DEFAULT_HPARAMS + "
            "use_missing, zero_as_missing; the starting point of any tuning"),
        "detector/review.py": ("SHAP contributions (compute_shap_for_windows) and the "
                               "disagreement review HTML (model/shap_review.py)"),
        "detector/heldout_eval.py": (
            "per-recording confusion metrics and their micro/macro aggregation "
            "(_compute_metrics, _aggregate) on our per-core predictions, reported beside "
            "our scores in the comparison when the checkout is present"),
        "detector/animal_id.py extract_animal_letter": (
            "an independent second reading of the animal letter in every fold guard"),
        "registry / provenance": (
            "the append-only registry log, per-(mode, animal) promotion pointer as a label "
            "only, provenance.json per model (model/registry.py, model/provenance.py)"),
    },
    "not_reused": {
        "detector/retrain.py retrain() training": (
            "cannot read the per-core feature table, and trains unlabelled windows as "
            "negatives (trust_level unlabeled_clean at weight w_neg): invariant 9, unjudged "
            "is not negative. An adapter cannot fix the label semantics without changing "
            "retrain.py, and GEMSBlanking is not modified. The build's trainer (model/train.py) "
            "trains LightGBM on judged cores only; unsure and unjudged are excluded."),
        "detector/hyperopt.py search space (w_neg, fp_weight, fn_weight)": (
            "w_neg is the weight of unlabelled windows as negatives - the same invariant-9 "
            "semantics. Tuning here (model/tuning.py) is optuna over LightGBM parameters, "
            "nested inside each training fold (inner CV on the training animals), never on "
            "the held-out animal, and off by default"),
        "retrain.py early stopping (2000 rounds, random 10% split)": (
            "a random split of cores leaks recordings across the split; fixed rounds"),
    },
    "params_source": RETRAIN_DEFAULTS_SOURCE,
}
"""Task 12's reuse list, what is reused, and what is not and why - written to the run
record (ruling 2026-10-08 item 1: 'Record the deviation from task 12's reuse list in the
run record')."""


def verify_retrain_defaults() -> dict[str, Any]:
    """Read ``DEFAULT_HPARAMS`` from the checkout and compare with the copy held here.

    Returns ``{"verified": True, "values": ...}``; raises ``ValueError`` naming the keys
    that differ, and lets ``FileNotFoundError`` / ``ImportError`` through when the private
    checkout is absent (the caller decides whether that is fatal).
    """
    model = import_detector_module("model")
    theirs = dict(model.DEFAULT_HPARAMS)
    if theirs != dict(RETRAIN_DEFAULT_HPARAMS):
        keys = sorted(set(theirs) ^ set(RETRAIN_DEFAULT_HPARAMS)
                      | {k for k in set(theirs) & set(RETRAIN_DEFAULT_HPARAMS)
                         if theirs[k] != RETRAIN_DEFAULT_HPARAMS[k]})
        msg = (f"retrain.py's DEFAULT_HPARAMS differ from the copy held here at {keys}: "
               f"checkout {theirs}")
        raise ValueError(msg)
    return {"verified": True, "values": theirs}
