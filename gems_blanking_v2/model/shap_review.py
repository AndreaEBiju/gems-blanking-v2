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

import hashlib
import html
import os
import platform
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.io.store import atomic_write_bytes, atomic_write_text, safe_component
from gems_blanking_v2.model.evaluate import DECISION_P
from gems_blanking_v2.model.labels import NEGATIVE_JUDGEMENTS
from gems_blanking_v2.model.modes import refuse_test_rows
from gems_blanking_v2.model.registry import ModelSpec, Registry
from gems_blanking_v2.model.train import feature_columns, predict_raw

__all__ = ["CONTEXT_S", "DEEPEST_LEAF", "HOST_CHARS", "REVIEW_NAME", "TOP_FEATURES_NAME",
           "build_dir", "review_samples", "write_shap_review"]

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

    ``cores`` holds JUDGED cores (``recording``, ``cohort``, ``animal``, ``label_set``,
    ``start_s``, ``stop_s``, ``judgement`` and the model's feature columns); ``fs`` maps
    each recording to its sample rate. A disagreement is a core whose judgement is a
    negative (:data:`~gems_blanking_v2.model.labels.NEGATIVE_JUDGEMENTS`) that the model
    calls motion; an ``unjudged`` or ``unsure`` core is refused, never read as a negative
    (invariant 9), and test-set cores are refused (R1). Returns the ``shap`` directory,
    built in a temporary sibling and moved into place, so a failed build leaves nothing
    behind. Raises if it exists already, if a recording has no ``fs``, or if the
    GEMSBlanking checkout is unavailable (``FileNotFoundError`` / ``ImportError``).
    """
    review = import_detector_module("review")
    out_dir = registry.store.model_dir(spec.model_id) / "shap"
    if out_dir.exists():
        msg = f"{out_dir} exists; a model's SHAP review is written once"
        raise FileExistsError(msg)
    if "judgement" not in cores.columns:
        msg = "cores need a judgement column; y alone cannot tell unjudged from negative"
        raise ValueError(msg)
    judged = {"motion", *NEGATIVE_JUDGEMENTS}
    bad = sorted(set(cores["judgement"].astype(str)) - judged)
    if bad:
        msg = f"cores with judgement {bad} are not judged; refused (invariant 9)"
        raise ValueError(msg)
    refuse_test_rows(cores, context="a SHAP review")
    no_fs = sorted(set(cores["recording"].astype(str)) - set(fs))
    if no_fs:
        msg = f"no fs for recording(s) {no_fs[:5]}; it is read, never assumed"
        raise ValueError(msg)
    booster = registry.booster(spec)
    x = cores[feature_columns(cores, booster.feature_name())].astype(np.float64)
    p = registry.calibrator(spec).apply(predict_raw(booster, x))
    negative = cores["judgement"].isin(NEGATIVE_JUDGEMENTS).to_numpy()
    _top, shap_all, _base = review.compute_shap_for_windows(booster, x, top_n=3)
    mean_abs = np.mean(np.abs(np.asarray(shap_all, dtype=np.float64)), axis=0)

    idx = np.flatnonzero(negative & (p >= DECISION_P))
    idx = idx[np.argsort(-p[idx], kind="stable")][:top_k]
    dis, plots = [], []
    for i in idx:
        r = cores.iloc[int(i)]
        f = float(fs[str(r["recording"])])
        a, b = float(r["start_s"]), float(r["stop_s"])
        centre, c0, c1 = review_samples(a, b, f)
        dis.append(review.Disagreement(
            recording_id=str(r["recording"]), position_sample=centre,
            model_prob=float(p[i]), label=0, fs=f, context_start=c0, context_end=c1,
            feature_row_index=int(i)))
        plots.append(f"<p>core [{a:.3f}, {b:.3f}) s - review built from the core feature "
                     "table; the signal is not loaded here.</p>")
    shap_top = (review.compute_shap_for_windows(booster, x.iloc[idx], top_n=3)[0]
                if len(idx) else [])
    def write(build: Path) -> None:
        tmp = build / (REVIEW_NAME + ".raw")
        review.generate_review_html(
            spec.model_id, disagreements=dis, shap_top=shap_top, channel_plots=plots,
            model_version=f"{spec.version} {spec.mode} {spec.animal or 'all'}",
            threshold_used=float(DECISION_P), output_path=tmp)
        # generate_review_html writes in text mode, which is CRLF on Windows; the page is
        # built with "\n" only, so undoing the translation restores it exactly (rule 10).
        atomic_write_bytes(build / REVIEW_NAME, tmp.read_bytes().replace(b"\r\n", b"\n"))
        tmp.unlink()
        atomic_write_text(build / TOP_FEATURES_NAME,
                          _top_features_html(spec, booster.feature_name(), mean_abs, top_n,
                                             len(cores)))

    too_long = registry.store.check_path_length(build_dir(out_dir) / DEEPEST_LEAF)
    if too_long:  # rule 5: fail with a clear message, not an OSError deep in a write
        raise ValueError(too_long)
    _publish(out_dir, write)
    return out_dir


HOST_CHARS: Final = 15
"""Length of the host part of a build directory name (keeps the deepest path short)."""

HOST_PREFIX: Final = 8
"""Readable characters of the host name kept in the host part."""

MKSTEMP_RANDOM_CHARS: Final = 8
"""Length of the random part of a ``tempfile.mkstemp`` name (CPython's
``_RandomNameSequence``); a test creates a real temp file and checks the bound."""

DEEPEST_LEAF: Final = max(
    [REVIEW_NAME + ".raw"]  # written by detector.review itself
    + [f".{n}.{'x' * MKSTEMP_RANDOM_CHARS}.tmp"  # store.atomic_write_* temp files
       for n in (REVIEW_NAME, TOP_FEATURES_NAME)], key=len)
"""The longest file name the build directory ever holds - the one rule 5 is checked on."""


def build_dir(out_dir: Path) -> Path:
    """Return this process's own build directory: ``shap.building-<host>-<pid>``.

    The host part is ``HOST_PREFIX`` readable characters plus a SHA-256 digest of the
    FULL host name, ``HOST_CHARS`` in all, so two hosts sharing a long prefix (two
    ``<Name>-MacBook-Pro.local``) never share a build directory name (invariant 27).
    """
    full = platform.node() or "host"
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:HOST_CHARS - HOST_PREFIX - 1]
    readable = safe_component(full, "host")[:HOST_PREFIX].rstrip(". ") or "host"
    return out_dir.with_name(f"shap.building-{readable}-{digest}-{os.getpid()}")


def _publish(out_dir: Path, write: Callable[[Path], None]) -> None:
    """Build into this process's own sibling directory, check it, move it into place.

    Only OUR build directory is ever removed. Model directories live on the shared drive,
    so another machine may be building the same review right now; deleting its
    ``shap.building-*`` would let it move a gutted directory into the write-once
    ``shap/``. A leftover from a killed process is harmless: it is never read, and every
    build uses its own name (host + PID). The build must hold both pages before the move.
    """
    build = build_dir(out_dir)
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    try:
        write(build)
        incomplete = [n for n in (REVIEW_NAME, TOP_FEATURES_NAME) if not (build / n).is_file()]
        if incomplete:
            msg = f"SHAP review build {build} lacks {incomplete}; not moved into place"
            raise RuntimeError(msg)
        os.replace(build, out_dir)  # noqa: PTH105 - a directory move, atomic or nothing
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise


def review_samples(start_s: float, stop_s: float, fs: float) -> tuple[int, int, int]:
    """Our 0-based half-open ``[start_s, stop_s)`` as ``Disagreement``'s 1-based samples.

    The core is first put on the sample grid, ``s0 = round(start_s * fs)`` and
    ``s1 = round(stop_s * fs)`` (0-based half-open ``[s0, s1)``). Returns
    ``(centre, context_start, context_end)``: the 1-based sample of the core's middle
    sample ``(s0 + s1) // 2`` (the upper middle of an even-length core), and the
    1-based INCLUSIVE context: the core widened by ``CONTEXT_S`` on each side,
    ``[s0 - c, s1 + c)`` with ``c = round(CONTEXT_S * fs)``, whose start is clipped at
    sample 1 (invariant 15: 0-based ``k`` is 1-based ``k + 1``; a half-open stop ``s``
    is the inclusive 1-based ``s``). Integer arithmetic only, so no round-half-to-even
    on a midpoint can shift the centre.
    """
    s0, s1 = round(start_s * fs), round(stop_s * fs)
    c = round(CONTEXT_S * fs)
    return ((s0 + s1) // 2 + 1, max(1, s0 - c + 1), s1 + c)
