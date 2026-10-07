"""Task 12 acceptance 5: the SHAP review HTML of a registered model.

Reuses GEMSBlanking's ``detector/review.py`` by import, unchanged (verified 2026-10-07 at
detector-core ``599d1fe``: ``Disagreement``, ``compute_shap_for_windows`` and
``generate_review_html`` exist with the signatures used here):

* ``compute_shap_for_windows`` - LightGBM ``pred_contrib`` TreeSHAP, top contributions
  per row and the full SHAP matrix;
* ``Disagreement`` + ``generate_review_html`` - the self-contained scoring page (top-K
  model-vs-human disagreements, keyboard scoring, JSON export).

Two adaptations, both outside GEMSBlanking's code: a *disagreement* here is a judged
core the model calls motion (calibrated ``P(motion) >= DECISION_P``) that a human
judged not motion; and the channel plot slot carries a text note, because this review
is built from the core feature table, not from the recording's samples. Sample
positions are 1-based MATLAB-style as ``Disagreement`` documents, from each
recording's own ``fs`` (never hardcoded). A second page, ``top_features.html``, ranks
features by mean |SHAP| over every scored core.

Written into ``models/<id>/shap/`` once (a second write raises), atomically.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.io.store import atomic_write_bytes, atomic_write_text
from gems_blanking_v2.model.evaluate import DECISION_P
from gems_blanking_v2.model.registry import ModelSpec, Registry
from gems_blanking_v2.model.train import feature_columns, predict_raw

__all__ = ["CONTEXT_S", "REVIEW_NAME", "TOP_FEATURES_NAME", "write_shap_review"]

REVIEW_NAME: Final = "review.html"
TOP_FEATURES_NAME: Final = "top_features.html"
CONTEXT_S: Final = 2.0
"""Context either side of a core reported on the review page, seconds."""


def _top_features_html(spec: ModelSpec, names: list[str], mean_abs: np.ndarray,
                       top_n: int, n_rows: int) -> str:
    order = np.argsort(-mean_abs)[:top_n]
    rows = "\n".join(
        f"<tr><td>{i + 1}</td><td><code>{html.escape(names[j])}</code></td>"
        f"<td>{mean_abs[j]:.4f}</td></tr>" for i, j in enumerate(order))
    title = html.escape(f"{spec.mode} {spec.animal or 'all animals'} - {spec.model_id}")
    return (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>Top features</title></head><body><h1>Top features by mean |SHAP|</h1>"
            f"<p>{title}; version <code>{html.escape(spec.version)}</code>; "
            f"{n_rows} scored cores.</p><table><tr><th>#</th><th>feature</th>"
            f"<th>mean |SHAP| (log-odds)</th></tr>\n{rows}\n</table></body></html>\n")


def write_shap_review(registry: Registry, spec: ModelSpec, cores: pd.DataFrame, *,
                      fs: Mapping[str, float], top_k: int = 20, top_n: int = 25) -> Path:
    """Write ``review.html`` and ``top_features.html`` for a registered model.

    ``cores`` holds judged cores (``recording``, ``start_s``, ``stop_s``, ``y`` and the
    model's feature columns); ``fs`` maps each recording to its sample rate. Returns the
    ``shap`` directory. Raises if it exists already, if a recording has no ``fs``, or if
    the GEMSBlanking checkout is unavailable (``FileNotFoundError`` / ``ImportError``).
    """
    review = import_detector_module("review")
    out_dir = registry.store.model_dir(spec.model_id) / "shap"
    if out_dir.exists():
        msg = f"{out_dir} exists; a model's SHAP review is written once"
        raise FileExistsError(msg)
    no_fs = sorted(set(cores["recording"].astype(str)) - set(fs))
    if no_fs:
        msg = f"no fs for recording(s) {no_fs[:5]}; it is read, never assumed"
        raise ValueError(msg)
    booster = registry.booster(spec)
    x = cores[feature_columns(cores, booster.feature_name())].astype(np.float64)
    p = registry.calibrator(spec).apply(predict_raw(booster, x))
    y = cores["y"].to_numpy().astype(int)
    _top, shap_all, _base = review.compute_shap_for_windows(booster, x, top_n=3)
    mean_abs = np.mean(np.abs(np.asarray(shap_all, dtype=np.float64)), axis=0)

    idx = np.flatnonzero((y == 0) & (p >= DECISION_P))
    idx = idx[np.argsort(-p[idx], kind="stable")][:top_k]
    dis, plots = [], []
    for i in idx:
        r = cores.iloc[int(i)]
        f = float(fs[str(r["recording"])])
        a, b = float(r["start_s"]), float(r["stop_s"])
        dis.append(review.Disagreement(
            recording_id=str(r["recording"]),
            position_sample=round(0.5 * (a + b) * f) + 1, model_prob=float(p[i]), label=0,
            fs=f, context_start=max(1, round((a - CONTEXT_S) * f) + 1),
            context_end=round((b + CONTEXT_S) * f), feature_row_index=int(i)))
        plots.append(f"<p>core [{a:.3f}, {b:.3f}) s - review built from the core feature "
                     "table; the signal is not loaded here.</p>")
    shap_top = (review.compute_shap_for_windows(booster, x.iloc[idx], top_n=3)[0]
                if len(idx) else [])
    out_dir.mkdir(parents=True)
    tmp = out_dir / (REVIEW_NAME + ".tmp")
    review.generate_review_html(
        spec.model_id, disagreements=dis, shap_top=shap_top, channel_plots=plots,
        model_version=f"{spec.version} {spec.mode} {spec.animal or 'all'}",
        threshold_used=float(DECISION_P), output_path=tmp)
    # generate_review_html writes in text mode, which is CRLF on Windows; the page itself
    # is built with "\n" only, so undoing the translation restores it exactly (rule 10).
    atomic_write_bytes(out_dir / REVIEW_NAME, tmp.read_bytes().replace(b"\r\n", b"\n"))
    tmp.unlink()
    atomic_write_text(out_dir / TOP_FEATURES_NAME,
                      _top_features_html(spec, booster.feature_name(), mean_abs, top_n,
                                         len(cores)))
    return out_dir
