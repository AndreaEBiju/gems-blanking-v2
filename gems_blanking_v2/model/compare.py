"""Task 12: the comparison artifact, and the verdicts ruled by R9 and rulings (i)/(j).

Per-animal scores, matched-protocol tables, the learning curve and calibration.

Rules applied here, all fixed before results (they are read from the run record):

* **Per animal, never averaged** (task 12 validation protocol).
* **Matched protocol only** (invariant 12): B vs C and A vs B are paired on the same
  held-out rows of the target animal; A vs C is emitted with ``comparable = False`` and
  never enters a verdict.
* **R9 mode rule:** C beats B only if ``F1(C) - F1(B) >= 0.03`` and the 95% paired
  cluster-bootstrap interval of the difference excludes 0. B is swept over ``w_adapt``
  and the sweep is not tuned on: C must beat B at **every** swept weight - comparing C
  with the best B is the conservative reading (written to the run record as such).
* **R9 small folds:** an animal with fewer than 20 positives is reported in its own
  table and never decides.
* **Rulings (i)/(j) tier steps:** a step is kept unless audit-span F1 falls by at least
  0.02 with the 95% interval of the fall excluding 0, relative to the previous step.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.io.store import atomic_write_text
from gems_blanking_v2.model.evaluate import (
    R9Thresholds,
    cluster_bootstrap_ci,
    ece,
    paired_diff_ci,
    reliability_curve,
    scores,
)

__all__ = [
    "MIN_CURVE_POINTS",
    "BvCVerdict",
    "TierVerdict",
    "b_vs_c_verdict",
    "curve_sizes",
    "markdown_table",
    "matched_protocol_table",
    "summarize",
    "tier_step_verdict",
    "write_comparison",
]

MIN_CURVE_POINTS: Final = 5
"""The learning curve has at least this many log-spaced points (task 12)."""

VERDICT_RULE: Final = (
    "C beats B only if F1(C)-F1(B) >= c_beats_b_min_f1_gain with the paired cluster-"
    "bootstrap interval excluding 0, at EVERY swept w_adapt; animals with fewer than "
    "min_positives_deciding positives never decide; A vs C is never compared."
)
"""How the B-vs-C verdict reads the w_adapt sweep. Written to the run record."""


def _group_keys(preds: pd.DataFrame) -> list[tuple[str, str, float]]:
    keys = preds[["mode", "target", "w_adapt"]].drop_duplicates()
    return [(str(m), str(t), float(w)) for m, t, w in keys.itertuples(index=False)]


def _sel(preds: pd.DataFrame, mode: str, target: str, w: float) -> pd.DataFrame:
    m = (preds["mode"] == mode) & (preds["target"] == target)
    m &= preds["w_adapt"].isna() if math.isnan(w) else (preds["w_adapt"] == w)
    return preds.loc[m]


def summarize(preds: pd.DataFrame, r9: R9Thresholds, *, seed: int = 0) -> pd.DataFrame:
    """One row per (mode, target, w_adapt): scores with cluster-bootstrap CIs, ECE, baselines.

    ``f1`` is the model's own decision (``raw >= 0.5``); ``f1_cal`` uses the cross-fitted
    calibrated probability (rows it could not calibrate are excluded and counted).
    ``deciding`` is False when the animal has fewer than ``r9.min_positives_deciding``
    positives.
    """
    rows = []
    for mode, target, w in _group_keys(preds):
        g = _sel(preds, mode, target, w)
        y = g["y"].to_numpy()
        cl = g["cluster"].to_numpy()
        s = scores(y, g["yhat"].to_numpy())
        lo, hi = cluster_bootstrap_ci(y, g["yhat"].to_numpy(), cl, n_resamples=r9.n_bootstrap,
                                      seed=seed)
        p_lo, p_hi = cluster_bootstrap_ci(y, g["yhat"].to_numpy(), cl, stat="precision",
                                          n_resamples=r9.n_bootstrap, seed=seed)
        r_lo, r_hi = cluster_bootstrap_ci(y, g["yhat"].to_numpy(), cl, stat="recall",
                                          n_resamples=r9.n_bootstrap, seed=seed)
        cal = g["p_cal"].notna().to_numpy() if "p_cal" in g.columns else np.zeros(len(g), bool)
        ece_cal = ece(g["p_cal"].to_numpy()[cal], y[cal]) if cal.any() else math.nan
        f1_cal = (scores(y[cal], g["p_cal"].to_numpy()[cal] >= 0.5).f1  # noqa: PLR2004
                  if cal.any() else math.nan)
        allm = scores(y, np.ones_like(y))
        thr_ok = g["yhat_thr"].notna().to_numpy()
        thr = scores(y[thr_ok], g["yhat_thr"].to_numpy()[thr_ok] > 0) if thr_ok.any() else None
        rows.append({
            "mode": mode, "target": target, "w_adapt": w,
            "protocol": str(g["protocol"].iloc[0]),
            "n": s.n, "n_pos": s.n_pos, "prevalence": s.prevalence,
            "n_recordings": int(g["recording"].nunique()), "n_clusters": int(len(set(cl))),
            "precision": s.precision, "precision_lo": p_lo, "precision_hi": p_hi,
            "recall": s.recall, "recall_lo": r_lo, "recall_hi": r_hi,
            "f1": s.f1, "f1_lo": lo, "f1_hi": hi,
            "ece_raw": ece(g["raw"].to_numpy(), y), "ece_cal": ece_cal,
            "ece_pass": bool(np.isfinite(ece_cal) and ece_cal <= r9.ece_max),
            "f1_cal": f1_cal, "n_uncalibrated": int((~cal).sum()),
            "f1_all_motion": allm.f1,
            "f1_threshold": thr.f1 if thr is not None else math.nan,
            "n_threshold_unscored": int((~thr_ok).sum()),
            "beats_all_motion": bool(s.f1 > allm.f1),
            "beats_threshold": bool(thr is not None and s.f1 > thr.f1),
            "n_train_events_median": float(g["n_train_events"].median()),
            "deciding": s.n_pos >= r9.min_positives_deciding,
        })
    return pd.DataFrame(rows)


def _paired(preds: pd.DataFrame, a: tuple[str, float], b: tuple[str, float], target: str,
            r9: R9Thresholds, seed: int) -> dict[str, object] | None:
    ga = _sel(preds, a[0], target, a[1])
    gb = _sel(preds, b[0], target, b[1])
    if ga.empty or gb.empty:
        return None
    j = ga.merge(gb, on="row", suffixes=("_a", "_b"))
    if j.empty:
        return None
    diff, lo, hi = paired_diff_ci(j["y_a"], j["yhat_a"], j["yhat_b"], j["cluster_a"],
                                  n_resamples=r9.n_bootstrap, seed=seed)
    return {"target": target, "a": a[0], "w_a": a[1], "b": b[0], "w_b": b[1],
            "n_rows": len(j), "n_pos": int(j["y_a"].sum()), "f1_b_minus_a": diff,
            "lo": lo, "hi": hi}


def matched_protocol_table(preds: pd.DataFrame, r9: R9Thresholds, *, seed: int = 0
                           ) -> pd.DataFrame:
    """B vs C, A vs B (per swept ``w_adapt``), and A vs C marked ``comparable=False``."""
    out = []
    ws = sorted({float(w) for w in preds.loc[preds["mode"] == "adapted", "w_adapt"]})
    nan = math.nan
    for target in sorted(set(preds["target"])):
        for w in ws:
            for a, b, comp, q in (((("adapted", w)), ("per_animal", nan), True, "B vs C"),
                                  ((("pooled", nan)), ("adapted", w), True, "A vs B")):
                r = _paired(preds, a, b, target, r9, seed)
                if r is not None:
                    out.append({**r, "comparison": q, "comparable": comp})
        r = _paired(preds, ("pooled", nan), ("per_animal", nan), target, r9, seed)
        if r is not None:
            out.append({**r, "comparison": "A vs C", "comparable": False,
                        "f1_b_minus_a": nan, "lo": nan, "hi": nan})
    return pd.DataFrame(out)


@dataclass(frozen=True)
class BvCVerdict:
    """The B-vs-C verdict for one animal."""

    target: str
    verdict: str
    deciding: bool
    task11_investigation: bool
    detail: str


def b_vs_c_verdict(matched: pd.DataFrame, r9: R9Thresholds) -> list[BvCVerdict]:
    """Apply R9: C beats B only by >= 0.03 F1 with the CI excluding 0, at every swept w.

    In the ``B vs C`` rows of :func:`matched_protocol_table`, ``a`` is B and ``b`` is C,
    so ``f1_b_minus_a`` is ``F1(C) - F1(B)``.
    """
    out: list[BvCVerdict] = []
    if matched.empty:
        return out
    bc = matched[matched["comparison"] == "B vs C"]
    for target in sorted(set(bc["target"])):
        rows = bc[bc["target"] == target]
        n_pos = int(rows["n_pos"].iloc[0])
        deciding = n_pos >= r9.min_positives_deciding
        gains = rows["f1_b_minus_a"].to_numpy(dtype=np.float64)
        los = rows["lo"].to_numpy(dtype=np.float64)
        his = rows["hi"].to_numpy(dtype=np.float64)
        c_wins = bool(len(gains)) and all(
            g >= r9.c_beats_b_min_f1_gain and lo > 0 for g, lo in zip(gains, los, strict=True))
        verdict = "C > B" if c_wins else "B >= C"
        if not deciding:
            verdict = f"not deciding (n_pos={n_pos} < {r9.min_positives_deciding}); {verdict}"
        detail = "; ".join(f"w={w:g}: dF1(C-B)={g:+.3f} [{lo:+.3f},{hi:+.3f}]"
                           for w, g, lo, hi in zip(rows["w_a"], gains, los, his, strict=True))
        out.append(BvCVerdict(target, verdict, deciding, c_wins and deciding, detail))
    return out


@dataclass(frozen=True)
class TierVerdict:
    """Whether one tier step is kept (rulings (i)/(j))."""

    step: str
    previous: str
    f1_previous: float
    f1_step: float
    diff: float
    lo: float
    hi: float
    kept: bool


def tier_step_verdict(y: npt.ArrayLike, yhat_previous: npt.ArrayLike,
                      yhat_step: npt.ArrayLike, clusters: npt.ArrayLike, r9: R9Thresholds,
                      *, step: str, previous: str, seed: int = 0) -> TierVerdict:
    """Drop the step if F1 falls by >= ``r9.tier_drop_f1`` with the 95% CI excluding 0."""
    diff, lo, hi = paired_diff_ci(y, yhat_previous, yhat_step, clusters,
                                  n_resamples=r9.n_bootstrap, seed=seed)
    dropped = diff <= -r9.tier_drop_f1 and np.isfinite(hi) and hi < 0
    return TierVerdict(step=step, previous=previous,
                       f1_previous=scores(y, yhat_previous).f1,
                       f1_step=scores(y, yhat_step).f1, diff=diff, lo=lo, hi=hi,
                       kept=not dropped)


def curve_sizes(n_min: int, n_max: int, n_points: int = 6) -> list[int]:
    """Log-spaced training sizes, at least :data:`MIN_CURVE_POINTS` distinct integers."""
    if n_points < MIN_CURVE_POINTS:
        msg = f"a learning curve needs >= {MIN_CURVE_POINTS} points, asked for {n_points}"
        raise ValueError(msg)
    if not 0 < n_min < n_max:
        msg = f"need 0 < n_min < n_max, got {n_min}, {n_max}"
        raise ValueError(msg)
    k = n_points
    while True:
        sizes = sorted({round(v) for v in np.geomspace(n_min, n_max, k)})
        if len(sizes) >= MIN_CURVE_POINTS or k > 4 * n_points:
            break
        k += 1
    if len(sizes) < MIN_CURVE_POINTS:
        msg = f"cannot place {MIN_CURVE_POINTS} distinct sizes in [{n_min}, {n_max}]"
        raise ValueError(msg)
    return [int(s) for s in sizes]


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render ``df`` as a GitHub-flavoured markdown table (no ``tabulate`` dependency)."""
    if df.empty:
        return "(none)"

    def cell(v: object) -> str:
        if isinstance(v, float | np.floating):
            return "" if not np.isfinite(v) else format(float(v), floatfmt)
        return str(v).replace("|", "/")

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _plot_curves(curve: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib as mpl  # noqa: PLC0415 - headless backend set before pyplot

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    paths = []
    for target, g in curve.groupby("target"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, h_raw in g.groupby("line"):
            h = h_raw.sort_values("n_train_events")
            ax.errorbar(h["n_train_events"], h["f1"],
                        yerr=[h["f1"] - h["f1_lo"], h["f1_hi"] - h["f1"]], marker="o",
                        capsize=3, label=str(label))
        ax.set_xscale("log")
        ax.set_xlabel("training events")
        ax.set_ylabel("F1 (held-out, cluster-bootstrap 95% CI)")
        ax.set_title(f"learning curve - {target}")
        ax.legend(fontsize=7)
        p = out / f"curve_{str(target).replace(':', '_')}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def write_comparison(out_dir: Path, stamp: str, *, summary: pd.DataFrame,
                     matched: pd.DataFrame, verdicts: Sequence[BvCVerdict],
                     corpus: pd.DataFrame, preds: pd.DataFrame,
                     curve: pd.DataFrame | None = None,
                     meta: Mapping[str, object] | None = None,
                     r9: R9Thresholds | None = None) -> Path:
    """Write ``comparison_<stamp>.parquet`` and a readable ``comparison_<stamp>.md``.

    The parquet holds the per-(mode, animal, w) summary; the report adds the matched-
    protocol table (A vs C marked not comparable), the corpus table, reliability curves
    per mode and the verdicts. Readable as a file on its own (task 12).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"comparison_{stamp}.parquet"
    summary.to_parquet(pq, index=False)
    lines = [f"# Mode comparison {stamp}", ""]
    if meta:
        lines += ["```json", json.dumps(dict(meta), indent=1, sort_keys=True, default=str),
                  "```", ""]
    if r9 is not None:
        lines += ["R9 thresholds (from the run record): `"
                  + json.dumps(asdict(r9), sort_keys=True) + "`", "", VERDICT_RULE, ""]
    dec = summary[summary["deciding"]] if len(summary) else summary
    small = summary[~summary["deciding"]] if len(summary) else summary
    lines += ["## Per animal, per mode (deciding: >= 20 positives)", "",
              markdown_table(dec), "",
              "## Animals with fewer than 20 positives (reported, never decide)", "",
              markdown_table(small), "",
              "## Matched-protocol comparisons", "",
              markdown_table(matched),
              "", "A vs C is not comparable (LOAO vs LORO, invariant 12) and is never used.",
              "", "## Verdicts (B vs C)", ""]
    lines += [f"- **{v.target}**: {v.verdict}"
              + (" - task 11 feature-invariance investigation triggered"
                 if v.task11_investigation else "") + f" ({v.detail})" for v in verdicts]
    lines += ["", "## Corpus", "",
              markdown_table(corpus),
              "", "## Calibration (reliability, cross-fitted calibrated p)", ""]
    for (mode, target, w), g in preds.groupby(["mode", "target", "w_adapt"], dropna=False):
        ok = g["p_cal"].notna()
        if not ok.any():
            continue
        rc = reliability_curve(g.loc[ok, "p_cal"], g.loc[ok, "y"])
        rc = rc[rc["count"] > 0]
        lines += [f"### {mode} {target} w={w}", "",
                  markdown_table(rc), ""]
    if curve is not None and len(curve):
        curve.to_parquet(out_dir / f"learning_curve_{stamp}.parquet", index=False)
        pngs = _plot_curves(curve, out_dir)
        lines += ["## Learning curves", ""] + [f"![{p.stem}]({p.name})" for p in pngs] + [
            "", markdown_table(curve), ""]
    atomic_write_text(out_dir / f"comparison_{stamp}.md", "\n".join(lines) + "\n")
    return pq
