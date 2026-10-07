"""Task 12: the comparison artifact, and the verdicts of R9 and rulings (i), (j), 2026-10-08 (b).

Per-animal scores, matched-protocol tables, the learning curve, calibration and the
cohort probe.

Rules applied here. The thresholds come from the run record (``r9`` arguments are the
value :func:`~gems_blanking_v2.model.evaluate.require_run_record` returned), and the
rule texts are written into that record before training
(:func:`~gems_blanking_v2.model.evaluate.run_protocol`):

* **Per animal, never averaged** (task 12 validation protocol).
* **Matched protocol only** (invariant 12): B vs C and A vs B are paired on the same
  held-out rows of the target animal; A vs C is emitted with ``comparable = False`` and
  never enters a verdict.
* **"C beats B" (ruling 2026-10-08 (b) item 2):** C is compared with B at B's best
  adaptation weight, chosen per held-out recording by inner validation on that fold's
  training data (:func:`~gems_blanking_v2.model.modes.run_modes`, the ``inner_selected``
  series), never on the test fold. C beats B only if ``F1(C) - F1(B) >= 0.03`` and the
  95% paired cluster-bootstrap interval of the difference excludes 0 (R9). The per-w
  sweep is reported and never decides.
* **"< 20 positives" (ruling 2026-10-08 (b) item 2):** per fold as the protocol defines
  it - the held-out animal for LOAO (mode A), the held-out recording for B and C. A fold
  below it is reported separately (:func:`small_fold_table`) and never decides: the
  deciding comparisons use the rows of deciding folds only.
* **Cohort probe (ruling 2026-10-08 (b) item 1(d)):** :func:`cohort_probe`.
* **Rulings (i)/(j) tier steps:** a step is kept unless audit-span F1 falls by at least
  0.02 with the 95% interval of the fall excluding 0, relative to the previous kept step
  (:func:`tier_chain`).
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.io.store import atomic_write_bytes, atomic_write_text, validate_component
from gems_blanking_v2.model.evaluate import (
    CALIBRATION_NOTE,
    COHORT_RULE,
    SMALL_FOLD_RULE,
    VERDICT_RULE,
    R9Thresholds,
    cluster_bootstrap_ci,
    ece,
    heldout_eval_metrics,
    paired_diff_ci,
    reliability_curve,
    scores,
)

__all__ = [
    "MIN_CURVE_POINTS",
    "BvCVerdict",
    "TierVerdict",
    "b_vs_c_verdict",
    "cohort_probe",
    "curve_sizes",
    "markdown_table",
    "matched_protocol_table",
    "small_fold_table",
    "summarize",
    "tier_chain",
    "tier_step_verdict",
    "write_comparison",
]

MIN_CURVE_POINTS: Final = 5
"""The learning curve has at least this many log-spaced points (task 12)."""

SELECTED: Final = math.nan
"""The ``w_adapt`` of mode B's inner-selected series (each fold's own chosen w)."""


def _group_keys(preds: pd.DataFrame) -> list[tuple[str, str, float]]:
    keys = preds[["mode", "target", "w_adapt"]].drop_duplicates()
    return [(str(m), str(t), float(w)) for m, t, w in keys.itertuples(index=False)]


def _sel(preds: pd.DataFrame, mode: str, target: str, w: float) -> pd.DataFrame:
    m = (preds["mode"] == mode) & (preds["target"] == target)
    m &= preds["w_adapt"].isna() if math.isnan(w) else (preds["w_adapt"] == w)
    return preds.loc[m]


def _fold_pos(g: pd.DataFrame, fold_col: str = "fold", y_col: str = "y") -> pd.Series:
    return g.groupby(fold_col)[y_col].sum()


def summarize(preds: pd.DataFrame, r9: R9Thresholds, *, seed: int = 0) -> pd.DataFrame:
    """One row per (mode, target, w_adapt): scores with cluster-bootstrap CIs, ECE, baselines.

    ``f1`` is the model's own decision (``raw >= 0.5``); ``f1_cal`` uses the held-out
    calibrated probability (rows it could not calibrate are excluded and counted).
    ``w_adapt`` is ``nan`` for A, C and B's inner-selected series. Folds are as the
    protocol defines them (the animal for A, the held-out recording for B and C): a fold
    with fewer than ``r9.min_positives_deciding`` positives never decides, ``deciding``
    is True when at least one fold decides, and ``f1_deciding_folds`` is the F1 on the
    rows of deciding folds. ``f1_heldout_eval_*`` are GEMSBlanking's ``heldout_eval``
    metrics on the same predictions (absent without the checkout).
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
        per_fold = _fold_pos(g)
        dec_folds = per_fold[per_fold >= r9.min_positives_deciding].index
        in_dec = g["fold"].isin(dec_folds).to_numpy()
        he = heldout_eval_metrics(y, g["yhat"].to_numpy(), g["recording"].to_numpy())
        rows.append({
            "mode": mode, "target": target, "w_adapt": w,
            "w_rule": str(g["w_rule"].iloc[0]) if "w_rule" in g.columns else "",
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
            "n_folds": len(per_fold), "n_deciding_folds": len(dec_folds),
            "deciding": len(dec_folds) > 0,
            "f1_deciding_folds": (scores(y[in_dec], g["yhat"].to_numpy()[in_dec]).f1
                                  if in_dec.any() else math.nan),
            "min_pos_per_fold": int(per_fold.min()) if len(per_fold) else 0,
            "median_pos_per_fold": float(per_fold.median()) if len(per_fold) else 0.0,
            "ece_cal_uses_target_labels": mode != "pooled",
            "f1_heldout_eval_micro": he.get("f1_heldout_eval_micro", math.nan),
            "f1_heldout_eval_macro": he.get("f1_heldout_eval_macro", math.nan),
        })
    return pd.DataFrame(rows)


def _paired(preds: pd.DataFrame, a: tuple[str, float], b: tuple[str, float], target: str,
            r9: R9Thresholds, seed: int) -> dict[str, object] | None:
    """F1(b) - F1(a) on the rows of b's deciding folds (b is B or C: a LORO-type fold)."""
    ga = _sel(preds, a[0], target, a[1])
    gb = _sel(preds, b[0], target, b[1])
    if ga.empty or gb.empty:
        return None
    j = ga.merge(gb, on="row", suffixes=("_a", "_b"))
    if j.empty:
        return None
    per_fold = _fold_pos(j, "fold_b", "y_a")
    dec = per_fold[per_fold >= r9.min_positives_deciding].index
    jd = j[j["fold_b"].isin(dec)]
    out: dict[str, object] = {
        "target": target, "a": a[0], "w_a": a[1], "b": b[0], "w_b": b[1],
        "n_rows": len(j), "n_pos": int(j["y_a"].sum()), "n_folds": len(per_fold),
        "n_deciding_folds": len(dec), "n_rows_deciding": len(jd),
        "n_pos_deciding": int(jd["y_a"].sum())}
    if jd.empty:
        out.update({"f1_b_minus_a": math.nan, "lo": math.nan, "hi": math.nan})
    else:
        diff, lo, hi = paired_diff_ci(jd["y_a"], jd["yhat_a"], jd["yhat_b"], jd["cluster_a"],
                                      n_resamples=r9.n_bootstrap, seed=seed)
        out.update({"f1_b_minus_a": diff, "lo": lo, "hi": hi})
    return out


def _chosen_w(preds: pd.DataFrame, target: str) -> str:
    g = _sel(preds, "adapted", target, SELECTED)
    if g.empty or "w_chosen" not in g.columns:
        return ""
    per = g.groupby("fold")["w_chosen"].first()
    return json.dumps({f"{w:g}": int(n) for w, n in per.value_counts().sort_index().items()})


def matched_protocol_table(preds: pd.DataFrame, r9: R9Thresholds, *, seed: int = 0
                           ) -> pd.DataFrame:
    """B vs C and A vs B - at B's inner-selected w (deciding) and per swept w (reported).

    Every difference is on the rows of the deciding folds (>= ``min_positives_deciding``
    positives in the held-out recording). A vs C is marked ``comparable=False``.
    ``decides`` is True only on the B(inner-selected) vs C row.
    """
    out = []
    ws = sorted({float(w) for w in preds.loc[preds["mode"] == "adapted", "w_adapt"]
                 if np.isfinite(w)})
    nan = math.nan
    for target in sorted(set(preds["target"])):
        chosen = _chosen_w(preds, target)
        for w in [SELECTED, *ws]:
            label = "inner-selected" if math.isnan(w) else f"w={w:g}"
            for a, b, q in ((("adapted", w), ("per_animal", nan), "B vs C"),
                            (("pooled", nan), ("adapted", w), "A vs B")):
                r = _paired(preds, a, b, target, r9, seed)
                if r is not None:
                    out.append({**r, "comparison": q, "b_weight": label, "comparable": True,
                                "decides": q == "B vs C" and math.isnan(w),
                                "w_chosen_per_fold": chosen if math.isnan(w) else ""})
        r = _paired(preds, ("pooled", nan), ("per_animal", nan), target, r9, seed)
        if r is not None:
            out.append({**r, "comparison": "A vs C", "b_weight": "", "comparable": False,
                        "decides": False, "w_chosen_per_fold": "",
                        "f1_b_minus_a": nan, "lo": nan, "hi": nan})
    return pd.DataFrame(out)


def small_fold_table(preds: pd.DataFrame, r9: R9Thresholds) -> pd.DataFrame:
    """Every fold below the positives bar, per mode series: reported, never deciding."""
    rows = []
    for mode, target, w in _group_keys(preds):
        g = _sel(preds, mode, target, w)
        for fold, h in g.groupby("fold"):
            n_pos = int(h["y"].sum())
            if n_pos >= r9.min_positives_deciding:
                continue
            s = scores(h["y"].to_numpy(), h["yhat"].to_numpy())
            rows.append({"mode": mode, "target": target, "w_adapt": w, "fold": fold,
                         "n": s.n, "n_pos": n_pos, "f1": s.f1, "precision": s.precision,
                         "recall": s.recall})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class BvCVerdict:
    """The B-vs-C verdict for one animal."""

    target: str
    verdict: str
    deciding: bool
    task11_investigation: bool
    detail: str


def b_vs_c_verdict(matched: pd.DataFrame, r9: R9Thresholds) -> list[BvCVerdict]:
    """Apply ruling 2026-10-08 (b) item 2 with R9's margin and CI rule.

    Decided on the ``B vs C`` row at B's inner-selected weight (``decides``), over the
    deciding folds; ``f1_b_minus_a`` there is ``F1(C) - F1(B)``. A target with no deciding
    fold is "not deciding"; one with no inner-selected B gets no verdict. The swept
    weights are listed in ``detail`` for the record, never used.
    """
    out: list[BvCVerdict] = []
    if matched.empty:
        return out
    bc = matched[matched["comparison"] == "B vs C"]
    for target in sorted(set(bc["target"])):
        rows = bc[bc["target"] == target]
        sel = rows[rows["decides"].astype(bool)]
        sweep = rows[~rows["decides"].astype(bool)]
        detail = "; ".join(f"{b}: dF1(C-B)={g:+.3f} [{lo:+.3f},{hi:+.3f}]"
                           for b, g, lo, hi in zip(sweep["b_weight"], sweep["f1_b_minus_a"],
                                                   sweep["lo"], sweep["hi"], strict=True))
        if sel.empty:
            out.append(BvCVerdict(target, "no inner-selected B; no verdict", False, False,
                                  detail))
            continue
        r = sel.iloc[0]
        n_dec = int(r["n_deciding_folds"])
        gain, lo, hi = float(r["f1_b_minus_a"]), float(r["lo"]), float(r["hi"])
        c_wins = bool(np.isfinite(gain) and np.isfinite(lo)
                      and gain >= r9.c_beats_b_min_f1_gain and lo > 0)
        head = (f"inner-selected B (w per fold {r['w_chosen_per_fold']}): dF1(C-B)="
                f"{gain:+.3f} [{lo:+.3f},{hi:+.3f}] on {n_dec} deciding fold(s)")
        if n_dec == 0:
            out.append(BvCVerdict(target, f"not deciding (no held-out recording with >= "
                                  f"{r9.min_positives_deciding} positives)", False, False,
                                  f"{head}; sweep: {detail}"))
            continue
        out.append(BvCVerdict(target, "C > B" if c_wins else "B >= C", True, c_wins,
                              f"{head}; sweep (reported only): {detail}"))
    return out


def cohort_probe(preds: pd.DataFrame, table: pd.DataFrame, r9: R9Thresholds
                 ) -> pd.DataFrame:
    """Return the cohort-identification probe, per mode (ruling 2026-10-08 (b) item 1(d)).

    On out-of-fold predictions of cores judged ``physiology``, the median P(motion) of
    old-cohort cores minus that of new-cohort cores must be at most
    ``r9.cohort_probe_max_delta`` in absolute value. P(motion) is the model's raw
    output - the quantity its decisions threshold - with calibrated medians beside it.
    Mode B is its inner-selected series. A mode with no physiology-judged cores of either
    cohort is ``not computable`` with the reason, and ``passes`` is then absent (``nan``)
    - never a pass by default.
    """
    series = {"pooled": math.nan, "adapted": SELECTED, "per_animal": math.nan}
    out = []
    for mode, w in series.items():
        g = preds[(preds["mode"] == mode) & (preds["w_adapt"].isna() if math.isnan(w)
                                             else preds["w_adapt"] == w)]
        rows = table.iloc[g["row"].to_numpy().astype(int)] if len(g) else table.iloc[[]]
        phys = (rows["judgement"] == "physiology").to_numpy()
        coh = rows["cohort"].astype(str).to_numpy()
        rec: dict[str, object] = {"mode": mode, "n_physiology_new": int((phys & (coh == "new"))
                                                                        .sum()),
                                  "n_physiology_old": int((phys & (coh == "old")).sum())}
        missing = [c for c in ("new", "old") if not (phys & (coh == c)).any()]
        if g.empty:
            rec.update({"status": "not computable: no out-of-fold predictions of this mode",
                        "passes": math.nan})
        elif missing:
            rec.update({"status": ("not computable: no "
                                   + " or ".join(f"{c}-cohort" for c in missing)
                                   + " cores judged physiology in this mode's out-of-fold "
                                   "predictions"), "passes": math.nan})
        else:
            raw = g["raw"].to_numpy()
            med = {c: float(np.median(raw[phys & (coh == c)])) for c in ("new", "old")}
            delta = med["old"] - med["new"]
            rec.update({"median_raw_new": med["new"], "median_raw_old": med["old"],
                        "delta": delta, "passes": abs(delta) <= r9.cohort_probe_max_delta,
                        "status": "pass" if abs(delta) <= r9.cohort_probe_max_delta
                        else "fail"})
            if "p_cal" in g.columns:
                pc = g["p_cal"].to_numpy()
                for c in ("new", "old"):
                    v = pc[phys & (coh == c)]
                    v = v[np.isfinite(v)]
                    if v.size:
                        rec[f"median_cal_{c}"] = float(np.median(v))
        rec["max_delta"] = r9.cohort_probe_max_delta
        out.append(rec)
    return pd.DataFrame(out)


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


def tier_chain(pooled: Mapping[str, pd.DataFrame], chain: Sequence[str], r9: R9Thresholds,
               *, seed: int = 0) -> tuple[list[TierVerdict], str]:
    """Rulings (i)/(j): run the nested tier steps; return every verdict and the largest kept.

    ``pooled[step]`` holds mode A's LOAO predictions of the new-cohort audit spans for the
    training corpus of that step (columns ``eid``, ``target``, ``cluster``, ``y``,
    ``yhat``). ``chain`` is the order of the steps (``["1", "1+2a", "1+2a+2b"]``); each is
    compared with the previous KEPT step on the same rows and kept unless F1 falls by at
    least ``r9.tier_drop_f1`` with the 95% interval excluding 0. Raises when a step's rows
    differ from the previous step's (the spans scored must be identical) or a row is not
    a new-cohort audit span.
    """
    for step in chain:
        if not pooled[step]["cluster"].astype(str).str.startswith("span:").all():
            msg = f"tier step {step}: rows outside the new-cohort audit spans are scored"
            raise ValueError(msg)
    prev = chain[0]
    verdicts = []
    for step in chain[1:]:
        a, b = pooled[prev], pooled[step]
        j = a.merge(b, on=["eid", "target"], suffixes=("_a", "_b"))
        if not len(j) == len(a) == len(b):
            msg = f"tier steps {prev} and {step} score different rows ({len(a)}, {len(b)})"
            raise ValueError(msg)
        v = tier_step_verdict(j["y_a"], j["yhat_a"], j["yhat_b"], j["cluster_a"], r9,
                              step=step, previous=prev, seed=seed)
        verdicts.append(v)
        if v.kept:
            prev = step
    return verdicts, prev


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
        p = out / validate_component(f"curve_{str(target).replace(':', '_')}.png")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        atomic_write_bytes(p, buf.getvalue())
        plt.close(fig)
        paths.append(p)
    return paths


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically (temp file in the same directory, then replace)."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_comparison(out_dir: Path, stamp: str, *, summary: pd.DataFrame,
                     matched: pd.DataFrame, verdicts: Sequence[BvCVerdict],
                     corpus: pd.DataFrame, preds: pd.DataFrame,
                     curve: pd.DataFrame | None = None,
                     meta: Mapping[str, object] | None = None,
                     r9: R9Thresholds | None = None,
                     probe: pd.DataFrame | None = None,
                     small: pd.DataFrame | None = None,
                     w_selection: pd.DataFrame | None = None) -> Path:
    """Write ``comparison_<stamp>.parquet`` and a readable ``comparison_<stamp>.md``.

    The parquet holds the per-(mode, animal, w) summary; the report adds the matched-
    protocol table (A vs C marked not comparable), the small folds, mode B's inner w
    selection, the corpus table, reliability curves per mode, the verdicts and the cohort
    probe per mode (:func:`cohort_probe`; absent means NOT COMPUTED, never a pass).
    Readable as a file on its own (task 12).
    """
    validate_component(stamp)  # cross-platform rule 9: the stamp becomes a file name
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"comparison_{stamp}.parquet"
    _write_parquet(summary, pq)
    lines = [f"# Mode comparison {stamp}", ""]
    if meta:
        lines += ["```json", json.dumps(dict(meta), indent=1, sort_keys=True, default=str),
                  "```", ""]
    if r9 is not None:
        lines += ["R9 thresholds (from the run record): `"
                  + json.dumps(asdict(r9), sort_keys=True) + "`", "", VERDICT_RULE, "",
                  SMALL_FOLD_RULE, "", CALIBRATION_NOTE, "", COHORT_RULE, ""]
    dec = summary[summary["deciding"]] if len(summary) else summary
    nodec = summary[~summary["deciding"]] if len(summary) else summary
    lines += ["## Per animal, per mode (at least one deciding fold)", "",
              "w_adapt is empty for A, C and B's inner-selected series (w_rule).", "",
              markdown_table(dec), "",
              "## Series with no deciding fold (reported, never decide)", "",
              markdown_table(nodec), "",
              "## Folds with fewer than 20 positives (reported, never decide)", "",
              markdown_table(small) if small is not None else "(not supplied)", "",
              "## Mode B: inner-validated w_adapt per held-out recording", "",
              markdown_table(w_selection) if w_selection is not None else "(not supplied)",
              "", "## Matched-protocol comparisons (on deciding folds)", "",
              markdown_table(matched),
              "", "A vs C is not comparable (LOAO vs LORO, invariant 12) and is never used. "
              "Only the B(inner-selected) vs C row decides.",
              "", "## Verdicts (B vs C)", ""]
    lines += [f"- **{v.target}**: {v.verdict}"
              + (" - task 11 feature-invariance investigation triggered"
                 if v.task11_investigation else "") + f" ({v.detail})" for v in verdicts]
    lines += ["", "## Cohort probe (ruling 2026-10-08 (b) item 1(d))", ""]
    if probe is None or probe.empty:
        lines += ["NOT COMPUTED - no probe was supplied. This is not a pass.", ""]
    else:
        lines += [markdown_table(probe), ""]
    lines += ["## Corpus", "",
              markdown_table(corpus),
              "", "## Calibration (reliability, held-out calibrated p)", "",
              "Mode A: calibrated on the other targets' out-of-fold predictions (zero target "
              "labels). Modes B and C: cross-fitted over the target's own held-out clusters.",
              ""]
    for (mode, target, w), g in preds.groupby(["mode", "target", "w_adapt"], dropna=False):
        ok = g["p_cal"].notna()
        if not ok.any():
            continue
        rc = reliability_curve(g.loc[ok, "p_cal"], g.loc[ok, "y"])
        rc = rc[rc["count"] > 0]
        wl = ("inner-selected" if mode == "adapted" and isinstance(w, float) and math.isnan(w)
              else f"w={w}")
        lines += [f"### {mode} {target} {wl}", "",
                  markdown_table(rc), ""]
    if curve is not None and len(curve):
        _write_parquet(curve, out_dir / f"learning_curve_{stamp}.parquet")
        pngs = _plot_curves(curve, out_dir)
        lines += ["## Learning curves", ""] + [f"![{p.stem}]({p.name})" for p in pngs] + [
            "", markdown_table(curve), ""]
    atomic_write_text(out_dir / f"comparison_{stamp}.md", "\n".join(lines) + "\n")
    return pq
