"""Task 12: the three training modes, their folds, and one pass that trains them all.

========  ==================================================  ============================
mode      training corpus                                     protocol
========  ==================================================  ============================
A POOLED  every animal except the target                      LOAO
B ADAPTED pooled model, continued (``init_model``) on pooled  LOAO-then-adapt; held-out
          + the target's *other* recordings at ``w_adapt``    recordings of the target
C PER_    the target's other recordings only                  LORO within the animal
  ANIMAL
========  ==================================================  ============================

All three are evaluated on **the same rows** - every scorable judged core of the target
animal, one held-out recording at a time for B and C - so B vs C and A vs B are
matched-protocol comparisons. For a new-cohort animal the scorable rows are its **audit
spans** only (``span_id`` set; R1: "scored on its own audit spans"); a judged core outside
any span may train (B's adaptation, C) but is never scored. A vs C is computed on the
same rows too but is **not comparable** (LOAO vs LORO are different tasks, invariant 12)
and :mod:`~gems_blanking_v2.model.compare` labels it so.

Animals are keyed ``"<cohort>:<letter>"`` (:func:`animal_key`): the old cohort's ``J``
(JEL) is not the new cohort's ``J``, and the old cohort's unconfirmed token ``?`` is its
own unknown group - never a target and never a per-animal model. Its rows may train a
pooled model for a NEW-cohort target, but never for an old-cohort target: a "?" recording
may be that very animal (the count excluded is recorded on the fold).

:func:`prepare_table` refuses the prospective test set outright: any row of new-cohort
I, J or K (``labels.is_test_animal``), or any row whose ``label_set`` is not ``train``.

Leakage guards (task 12 tests): :func:`assert_disjoint_animals` on every fold, by animal
key and - where GEMSBlanking's ``extract_animal_letter`` can read one - by the letter in
the recording name; :func:`assert_no_recording_leak` on every mode-B fold, so no
recording of the target is both adapted on and evaluated.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.io.detector_core import import_detector_module
from gems_blanking_v2.model.evaluate import (
    BASELINE_FEATURE,
    CALIBRATION_KIND,
    DECISION_P,
    W_ADAPT_GRID,
    Calibrator,
    ThresholdBaseline,
    cluster_bootstrap_ci,
    cluster_ids,
    crossfit_calibrate,
    ece,
    require_run_record,
    scores,
)
from gems_blanking_v2.model.labels import OLD_TIERS, animal_key, is_test_animal
from gems_blanking_v2.model.provenance import build_provenance, corpus_composition
from gems_blanking_v2.model.registry import (
    ModelSpec,
    Registry,
    calibrator_relpath,
    model_content_id,
)
from gems_blanking_v2.model.train import (
    ADAPT_ROUNDS,
    NUM_BOOST_ROUND,
    check_leakage,
    corpus_hash,
    feature_columns,
    fit,
    predict_raw,
    sample_weights,
)
from gems_blanking_v2.types import TrainingMode

__all__ = [
    "UNKNOWN_ANIMAL",
    "Fold",
    "LabelOptIns",
    "ModeRefusedError",
    "ModeRun",
    "Refusal",
    "adapted_folds",
    "animal_key",
    "animal_letter",
    "assert_disjoint_animals",
    "assert_no_recording_leak",
    "learning_curve",
    "loao_folds",
    "per_animal_folds",
    "prepare_table",
    "read_renames",
    "refuse_test_rows",
    "run_modes",
]

I64 = npt.NDArray[np.int64]

UNKNOWN_ANIMAL: Final = "?"
"""The old cohort's unconfirmed animal token. Its own group; never a target."""

Protocol = Literal["LOAO", "LOAO_ADAPT", "LORO"]

PROTOCOL: Final[dict[TrainingMode, Protocol]] = {
    TrainingMode.POOLED: "LOAO",
    TrainingMode.ADAPTED: "LOAO_ADAPT",
    TrainingMode.PER_ANIMAL: "LORO",
}
"""The evaluation protocol of each mode; metrics always carry it."""



class ModeRefusedError(ValueError):
    """A mode cannot be fitted for an animal; the message says which and why."""

    def __init__(self, mode: TrainingMode, target: str, reason: str) -> None:
        """Record the mode, the animal and the reason."""
        super().__init__(f"{mode} refused for {target}: {reason}")
        self.mode = mode
        self.target = target
        self.reason = reason


@dataclass(frozen=True)
class LabelOptIns:
    """The label opt-ins a registering pass trained under (recorded in provenance).

    ``allow_model_labels`` must be a real ``bool``; when False every training row must
    have ``label_source == "human"``. ``keep_tiers`` are the old-cohort tiers admitted
    (drawn from :data:`~gems_blanking_v2.model.labels.OLD_TIERS`); every old-cohort
    recording in the table must carry one of them. :meth:`check` enforces both against
    the table actually trained on.
    """

    allow_model_labels: bool
    keep_tiers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Refuse a non-bool flag and any tier outside OLD_TIERS."""
        flag: object = self.allow_model_labels  # typed bool; checked because it is data
        if not isinstance(flag, bool):
            msg = f"allow_model_labels must be a bool, got {flag!r}"
            raise TypeError(msg)
        bad = sorted(set(self.keep_tiers) - set(OLD_TIERS))
        if bad:
            msg = f"keep_tiers must be drawn from {OLD_TIERS}, got {bad}"
            raise ValueError(msg)

    def check(self, table: pd.DataFrame, old_tiers: Mapping[str, str] | None) -> None:
        """Raise unless ``table`` is consistent with these opt-ins."""
        if not self.allow_model_labels:
            src = sorted(set(table["label_source"].astype(str)) - {"human"}) if (
                "label_source" in table.columns) else ["(no label_source column)"]
            if src:
                msg = f"allow_model_labels is False but the table has label_source {src}"
                raise ValueError(msg)
        old = table.loc[table["cohort"] == "old", "recording"].astype(str).unique()
        tiers = {(old_tiers or {}).get(r) for r in old}
        outside = sorted(map(str, tiers - set(self.keep_tiers)))
        if outside:
            msg = (f"old-cohort recordings at tier(s) {outside} are outside keep_tiers "
                   f"{list(self.keep_tiers)}")
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the provenance record."""
        return {"allow_model_labels": self.allow_model_labels,
                "keep_tiers": sorted(self.keep_tiers)}


@dataclass(frozen=True)
class Refusal:
    """A mode (or one fold of it) that was not fitted, and why."""

    mode: str
    target: str
    reason: str
    fold: str = ""



def animal_letter(recording: str) -> str | None:
    """Return the animal letter GEMSBlanking reads from a recording name, or ``None``.

    By import of ``detector.animal_id.extract_animal_letter``; ``None`` when the private
    checkout is absent, in which case only the declared ``animal`` column is checked.
    """
    try:
        mod = import_detector_module("animal_id")
    except (FileNotFoundError, ImportError):
        return None
    letter = mod.extract_animal_letter(recording)
    return None if letter is None else str(letter)


@dataclass(frozen=True)
class Fold:
    """One fit-and-evaluate unit. Index arrays are positional rows of the table.

    ``train`` is the pooled corpus for A and B and the target's other recordings for C;
    ``adapt`` is the target's adaptation rows (B only).
    """

    mode: TrainingMode
    target: str
    train: I64
    evaluate: I64
    adapt: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    held_out: tuple[str, ...] = ()
    n_excluded_unknown: int = 0
    """Old "?" rows left out of this fold's training (old-cohort targets only)."""

    @property
    def protocol(self) -> Protocol:
        """The evaluation protocol of this fold's mode."""
        return PROTOCOL[self.mode]


def prepare_table(table: pd.DataFrame, *,
                  renames: Path | None = None) -> pd.DataFrame:
    """Return a positional copy with ``animal_key``, ``cluster`` and ``name_letter`` added.

    ``table`` is the output of :func:`~gems_blanking_v2.model.labels.training_rows`
    (it must carry ``y``) joined to the feature columns.

    ``name_letter`` is the animal letter GEMSBlanking reads from the recording name - an
    independent second reading of the animal that the fold checks compare across. Where
    it disagrees with the declared ``animal`` this raises, naming the recordings, unless
    the recording is declared in the rename record ``renames`` (:func:`read_renames`, the
    one source of acknowledged renames; the caller copies it into the run record): a
    block whose folder was renamed after acquisition, where the folder name is
    authoritative for meaning (Andrea, 2026-09-26; CLAUDE.md invariant 30). An
    acknowledged recording's
    ``name_letter`` is its declared animal; an unacknowledged mismatch never passes.
    """
    for col in ("recording", "animal", "cohort", "y", "label_set"):
        if col not in table.columns:
            msg = f"table is missing required column {col!r}"
            raise ValueError(msg)
    refuse_test_rows(table)
    per_rec = table.groupby(table["recording"].astype(str))[["animal", "cohort"]].nunique()
    multi = per_rec[(per_rec > 1).any(axis=1)]
    if len(multi):
        msg = (f"recording(s) {list(multi.index[:5])} carry more than one animal or cohort; "
               "one recording is one animal of one cohort")
        raise ValueError(msg)
    out = table.reset_index(drop=True).copy()
    if "span_id" not in out.columns:
        out["span_id"] = None
    out["animal_key"] = [animal_key(c, a) for c, a in
                         zip(out["cohort"].astype(str), out["animal"].astype(str), strict=True)]
    if "cluster" not in out.columns:
        out["cluster"] = cluster_ids(out).to_numpy()
    acknowledged = read_renames(renames) if renames is not None else {}
    cohort_of = dict(zip(out["recording"].astype(str), out["cohort"].astype(str),
                         strict=False))
    wrong_cohort = sorted(r for r, (c, _a) in acknowledged.items()
                          if r in cohort_of and cohort_of[r] != c)
    if wrong_cohort:
        msg = f"rename record names the wrong cohort for {wrong_cohort[:5]}"
        raise ValueError(msg)
    letter_of = {r: animal_letter(r) for r in out["recording"].astype(str).unique()}
    first = out.drop_duplicates("recording")  # one animal per recording, asserted above
    declared = dict(zip(first["recording"].astype(str), first["animal"].astype(str),
                        strict=True))
    bad = []
    for rec, letter in letter_of.items():
        if letter is None or letter == declared[rec]:
            continue
        if acknowledged.get(rec, ("", ""))[1] == declared[rec]:
            letter_of[rec] = declared[rec]
        else:
            bad.append((rec, declared[rec], letter))
    if bad:
        msg = (f"recording name reads a different animal than the declared one: {bad}; "
               "declare a ruled rename in the rename record (read_renames)")
        raise ValueError(msg)
    out["name_letter"] = out["recording"].astype(str).map(letter_of)
    out.attrs["renames"] = renames_record(renames) if renames is not None else None
    return out


def refuse_test_rows(table: pd.DataFrame, *, context: str = "training") -> None:
    """R1: no new-cohort I/J/K row and no row with label_set != 'train' may be used.

    ``context`` names the use in the message ("training", "a SHAP review", ...).
    """
    test = np.array([is_test_animal(c, a) for c, a in
                     zip(table["cohort"].astype(str), table["animal"].astype(str), strict=True)],
                    dtype=bool)
    not_train = (table["label_set"] != "train").to_numpy()
    if test.any() or not_train.any():
        msg = (f"{int(test.sum())} row(s) of the prospective test set (new-cohort I/J/K) and "
               f"{int(not_train.sum())} row(s) with label_set != 'train' reached {context}; "
               "R1: they are never trained on and never scored")
        raise ValueError(msg)


def renames_record(path: Path) -> dict[str, object]:
    """Return what a run record's ``extra["renames"]`` must carry: SHA-256 and content.

    Validated through :func:`read_renames` first. :func:`run_modes` compares this with
    the record, so the renames a table was prepared with are provably the recorded ones.
    """
    read_renames(path)
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(),
            "content": json.loads(data.decode("utf-8"))}


def _check_renames_recorded(table: pd.DataFrame, record_path: Path) -> None:
    raw = json.loads(Path(record_path).read_text(encoding="utf-8"))
    have = (raw.get("extra") or {}).get("renames")
    want = table.attrs.get("renames")
    if have != want:
        msg = ("the run record's extra['renames'] does not match the rename record this table "
               "was prepared with; write renames_record(path) into the record before training")
        raise ValueError(msg)


_LETTER: Final = re.compile(r"^[A-Z]$")


def read_renames(path: Path) -> dict[str, tuple[str, str]]:
    """Read the declared rename record: the ONE source of acknowledged renames.

    A JSON list of ``{"recording", "cohort", "animal", "basis"}``: a recording whose
    folder was renamed after acquisition, filed under ``animal`` although its name reads
    another letter. ``basis`` names the ruling. Raises on a non-object entry, a duplicate
    recording, a missing field, or an animal that is not one upper-case letter. Returns
    ``{recording: (cohort, animal)}`` for :func:`prepare_table`, which checks the cohort
    against the table; the caller writes :func:`renames_record` into the run record.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        msg = f"{path}: a rename record is a JSON list"
        raise ValueError(msg)
    out: dict[str, tuple[str, str]] = {}
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            msg = f"{path}: entry {i} is not an object"
            raise ValueError(msg)
        for key in ("recording", "cohort", "animal", "basis"):
            if not row.get(key):
                msg = f"{path}: entry {i} is missing {key!r}"
                raise ValueError(msg)
        if not _LETTER.match(str(row["animal"])) or row["cohort"] not in ("old", "new"):
            msg = f"{path}: entry {i} has animal {row['animal']!r} / cohort {row['cohort']!r}"
            raise ValueError(msg)
        if row["recording"] in out:
            msg = f"{path}: recording {row['recording']!r} is declared twice"
            raise ValueError(msg)
        out[str(row["recording"])] = (str(row["cohort"]), str(row["animal"]))
    return out


def _rows(mask: npt.NDArray[np.bool_]) -> I64:
    return np.flatnonzero(mask).astype(np.int64)


def _scorable(table: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Rows that may be scored: every old-cohort row; a new-cohort row only in an audit span."""
    return ((table["cohort"] == "old") | table["span_id"].notna()).to_numpy()


def _assert_scored_on_spans(table: pd.DataFrame, fold: Fold) -> None:
    ev = table.iloc[fold.evaluate]
    new = (ev["cohort"] == "new").to_numpy()
    bad = new & ~ev["cluster"].astype(str).str.startswith("span:").to_numpy()
    if bad.any() or ev.loc[new, "span_id"].isna().any():
        msg = (f"{fold.mode} fold for {fold.target}: new-cohort rows scored outside an audit "
               "span (R1: each new-cohort animal is scored on its own audit spans)")
        raise AssertionError(msg)


def loao_folds(table: pd.DataFrame, targets: Sequence[str]) -> list[Fold]:
    """Mode A: one fold per target animal key - train on every other key, score the target."""
    keys = table["animal_key"].to_numpy()
    scorable = _scorable(table)
    folds = []
    for t in targets:
        _check_target(t)
        is_t = keys == t
        ev = is_t & scorable
        if not ev.any():
            msg = f"target {t} has no scorable rows"
            raise ValueError(msg)
        train, n_unknown = _pooled_training(keys, t)
        f = Fold(TrainingMode.POOLED, t, train=_rows(train), evaluate=_rows(ev),
                 held_out=tuple(sorted(set(table.loc[ev, "recording"].astype(str)))),
                 n_excluded_unknown=n_unknown)
        assert_disjoint_animals(table, f)
        _assert_scored_on_spans(table, f)
        folds.append(f)
    return folds


def _pooled_training(keys: npt.NDArray[np.object_], target: str
                     ) -> tuple[npt.NDArray[np.bool_], int]:
    """Return the pooled corpus of a target (modes A and B alike, invariant 33).

    Every other animal key; for an OLD-cohort target also without the old "?" rows,
    which may be that very animal. Returns the mask and how many "?" rows it left out.
    """
    train = keys != target
    if not target.startswith("old:"):
        return train, 0
    unknown_old = keys == f"old:{UNKNOWN_ANIMAL}"
    return train & ~unknown_old, int((train & unknown_old).sum())


def _check_target(t: str) -> None:
    if t.split(":", 1)[-1] == UNKNOWN_ANIMAL:
        msg = (f"{t} is the unconfirmed old-cohort token; it is its own unknown group and "
               "is never a target or a per-animal model")
        raise ValueError(msg)


def _target_recordings(table: pd.DataFrame, target: str, mode: TrainingMode) -> list[str]:
    try:
        _check_target(target)
    except ValueError as exc:
        raise ModeRefusedError(mode, target, str(exc)) from exc
    is_t = table["animal_key"] == target
    recs = sorted(set(table.loc[is_t, "recording"].astype(str)))
    if len(recs) < 2:  # noqa: PLR2004
        raise ModeRefusedError(mode, target, f"needs >= 2 labelled recordings, has {len(recs)}")
    scored = sorted(set(table.loc[is_t & _scorable(table), "recording"].astype(str)))
    if not scored:
        raise ModeRefusedError(mode, target, "no recording with scorable (audit-span) rows")
    return scored


def adapted_folds(table: pd.DataFrame, target: str) -> list[Fold]:
    """Mode B: one fold per held-out recording of the target.

    ``train`` = every other animal (the pooled prior), ``adapt`` = the target's other
    recordings, ``evaluate`` = the held-out recording. Split by recording, never by
    event, and checked by :func:`assert_no_recording_leak`.
    """
    recs = _target_recordings(table, target, TrainingMode.ADAPTED)
    keys = table["animal_key"].to_numpy()
    rec = table["recording"].astype(str).to_numpy()
    is_t = keys == target
    scorable = _scorable(table)
    pooled, n_unknown = _pooled_training(keys, target)
    folds = []
    for r in recs:
        f = Fold(TrainingMode.ADAPTED, target, train=_rows(pooled),
                 adapt=_rows(is_t & (rec != r)),
                 evaluate=_rows(is_t & (rec == r) & scorable), held_out=(r,),
                 n_excluded_unknown=n_unknown)
        assert_disjoint_animals(table, f)
        assert_no_recording_leak(table, f)
        _assert_scored_on_spans(table, f)
        folds.append(f)
    return folds


def per_animal_folds(table: pd.DataFrame, target: str) -> list[Fold]:
    """Mode C: LORO within the target animal. Refuses an animal that cannot be fitted.

    Refused when the key is the unknown token, the animal has fewer than two labelled
    recordings, or its labels hold one class (the old cohort's marks are positives only,
    so no old animal can be fitted alone until negatives are labelled).
    """
    recs = _target_recordings(table, target, TrainingMode.PER_ANIMAL)
    is_t = table["animal_key"].to_numpy() == target
    y = table["y"].to_numpy()[is_t]
    if np.unique(y).size < 2:  # noqa: PLR2004
        raise ModeRefusedError(TrainingMode.PER_ANIMAL, target,
                               f"labels hold one class only ({sorted(set(y.tolist()))})")
    rec = table["recording"].astype(str).to_numpy()
    scorable = _scorable(table)
    folds = []
    for r in recs:
        f = Fold(TrainingMode.PER_ANIMAL, target, train=_rows(is_t & (rec != r)),
                 evaluate=_rows(is_t & (rec == r) & scorable), held_out=(r,))
        _assert_single_animal(table, f)
        _assert_scored_on_spans(table, f)
        folds.append(f)
    return folds


def assert_disjoint_animals(table: pd.DataFrame, fold: Fold) -> None:
    """No animal on both sides of a fold, by key and by the letter in the name.

    For A and B the training side must not contain the target at all; ``adapt`` is the
    target's own rows by design and is checked by :func:`assert_no_recording_leak`.
    """
    if fold.mode is TrainingMode.PER_ANIMAL:
        _assert_single_animal(table, fold)
        return
    keys = table["animal_key"].to_numpy()
    train_keys = set(keys[fold.train].tolist())
    eval_keys = set(keys[fold.evaluate].tolist())
    both = train_keys & eval_keys
    if both:
        msg = f"{fold.mode} fold for {fold.target}: animal(s) {sorted(both)} on both sides"
        raise AssertionError(msg)
    _assert_letters_disjoint(table, fold.train, fold.evaluate, fold)


def _letters(table: pd.DataFrame, rows: I64) -> set[tuple[str, str]]:
    sub = table.iloc[rows]
    out = set()
    for cohort, rec in set(zip(sub["cohort"].astype(str), sub["recording"].astype(str),
                               strict=True)):
        if "name_letter" in sub.columns:
            letter = sub.loc[sub["recording"].astype(str) == rec, "name_letter"].iloc[0]
            letter = None if pd.isna(letter) else str(letter)
        else:
            letter = animal_letter(rec)
        if letter is not None:
            out.add((cohort, letter))
    return out


def _assert_letters_disjoint(table: pd.DataFrame, a: I64, b: I64, fold: Fold) -> None:
    both = _letters(table, a) & _letters(table, b)
    if both:
        msg = (f"{fold.mode} fold for {fold.target}: animal letter(s) {sorted(both)} read from "
               "recording names appear on both sides")
        raise AssertionError(msg)


def _assert_single_animal(table: pd.DataFrame, fold: Fold) -> None:
    keys = set(table["animal_key"].to_numpy()[np.concatenate([fold.train, fold.evaluate])]
               .tolist())
    if keys != {fold.target}:
        msg = f"per-animal fold for {fold.target} holds animal keys {sorted(keys)}"
        raise AssertionError(msg)


def assert_no_recording_leak(table: pd.DataFrame, fold: Fold) -> None:
    """No recording of the target in both the adaptation corpus and the evaluation set.

    The easiest mistake in task 12, and it inflates mode B exactly where the comparison
    matters. Also checks that the pooled part holds no target row.
    """
    rec = table["recording"].astype(str).to_numpy()
    shared = set(rec[fold.adapt].tolist()) & set(rec[fold.evaluate].tolist())
    if shared:
        msg = f"mode B fold for {fold.target}: recording(s) {sorted(shared)} adapted AND scored"
        raise AssertionError(msg)
    keys = table["animal_key"].to_numpy()
    if (keys[fold.train] == fold.target).any():
        msg = f"mode B fold for {fold.target}: target rows in the pooled corpus"
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# one pass over all modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeRun:
    """Everything one :func:`run_modes` pass produced.

    ``predictions`` is long-format, one row per (mode, w_adapt, evaluated core):
    ``mode, target, protocol, w_adapt, fold, row, recording, cluster, y, raw, p_cal,
    yhat, yhat_thr, n_train_events``. ``w_adapt`` is ``nan`` outside mode B.
    """

    predictions: pd.DataFrame
    refusals: tuple[Refusal, ...]
    corpus: pd.DataFrame
    registered: tuple[str, ...] = ()
    """Model ids registered by this pass (empty unless a registry was given)."""


def _subsample(rows: I64, n: int | None, rng: np.random.Generator) -> I64:
    if n is None or n >= rows.size:
        return rows
    return np.sort(rng.choice(rows, size=n, replace=False)).astype(np.int64)


def _baseline_pred(x_train: pd.Series, y_train: npt.NDArray[np.int8],
                   x_eval: pd.Series) -> npt.NDArray[np.float64]:
    try:
        b = ThresholdBaseline.fit(x_train.to_numpy(), y_train, BASELINE_FEATURE)
    except ValueError:
        return np.full(len(x_eval), np.nan)
    return b.predict(x_eval.to_numpy()).astype(np.float64)


def _record(mode: TrainingMode, fold: Fold, t: pd.DataFrame, raw: npt.NDArray[np.float64],
            thr: npt.NDArray[np.float64], n_train: int, w: float, fold_name: str,
            n_adapt: int = 0) -> pd.DataFrame:
    ev = t.iloc[fold.evaluate]
    return pd.DataFrame({
        "mode": str(mode), "target": fold.target, "protocol": PROTOCOL[mode],
        "w_adapt": w, "fold": fold_name, "row": fold.evaluate,
        "recording": ev["recording"].astype(str).to_numpy(),
        "cluster": ev["cluster"].astype(str).to_numpy(),
        "y": ev["y"].to_numpy().astype(np.int8), "raw": raw,
        "yhat": (raw >= DECISION_P).astype(np.int8), "yhat_thr": thr,
        "n_train_events": n_train, "n_adapt_events": n_adapt,
        "n_excluded_unknown": fold.n_excluded_unknown,
    })


def run_modes(table: pd.DataFrame, *, targets: Sequence[str], record_path: Path,  # noqa: PLR0912, PLR0915
              num_threads: int, modes: Sequence[TrainingMode] = tuple(TrainingMode),
              w_adapt_grid: Sequence[float] = W_ADAPT_GRID,
              calibration: Literal["isotonic"] | None = "isotonic",
              train_size: int | None = None, seed: int = 0,
              rounds: int = NUM_BOOST_ROUND, adapt_rounds: int = ADAPT_ROUNDS,
              registry: Registry | None = None, user: str = "",
              old_tiers: Mapping[str, str] | None = None,
              label_opt_ins: LabelOptIns | None = None) -> ModeRun:
    """Train and score every requested mode for every target, in one pass.

    Requires an existing run record (R9 thresholds written before training). ``table``
    comes from :func:`prepare_table`. ``train_size`` subsamples each fold's training
    corpus (the pooled part for A and B, the per-animal part for C) to that many events,
    for the learning curve; ``None`` uses everything. Calibration is cross-fitted per
    (mode, animal, ``w_adapt``) over the target's clusters
    (:func:`~gems_blanking_v2.model.evaluate.crossfit_calibrate`), so ``p_cal`` is
    held-out. Refused modes are returned, never silently skipped.

    With ``registry`` (rooted wherever the caller says) the pass also trains and
    REGISTERS the final model of every evaluated mode (:func:`_register_finals`): one
    POOLED model on every row, and per target a PER_ANIMAL model on all its rows and one
    ADAPTED model per swept ``w_adapt``. Each carries the held-out metrics of its
    protocol, a calibrator fitted on its mode's held-out predictions, and a
    ``provenance.json``. Registered models are never promoted and none is a default;
    ``user`` is required, ``old_tiers`` is required when old-cohort rows train, and a
    learning-curve pass (``train_size``) never registers.
    """
    require_run_record(record_path)
    refuse_test_rows(table)
    _check_renames_recorded(table, record_path)
    if registry is not None:
        if not user:
            msg = "registering models needs the acting user"
            raise ValueError(msg)
        if train_size is not None or calibration is None:
            msg = "only a full, calibrated pass registers models (no train_size, calibration on)"
            raise ValueError(msg)
        if set(modes) != set(TrainingMode) or sorted(map(float, w_adapt_grid)) != sorted(
                W_ADAPT_GRID):
            msg = ("a registering pass trains all three modes over the full recorded w_adapt "
                   f"grid {list(W_ADAPT_GRID)} (task 12: train all three, always)")
            raise ValueError(msg)
        if not isinstance(label_opt_ins, LabelOptIns):
            msg = ("a registering pass records the label opt-ins it trained under: "
                   "label_opt_ins=LabelOptIns(allow_model_labels=..., keep_tiers=(...))")
            raise ValueError(msg)
        label_opt_ins.check(table, old_tiers)
    if calibration not in (None, CALIBRATION_KIND):
        msg = f"calibration {calibration!r} is not the recorded kind {CALIBRATION_KIND!r}"
        raise ValueError(msg)
    off_grid = sorted(set(map(float, w_adapt_grid)) - set(W_ADAPT_GRID))
    if off_grid:
        msg = f"w_adapt {off_grid} is not on the recorded grid {W_ADAPT_GRID}"
        raise ValueError(msg)
    feats = feature_columns(table)
    x = table[feats]
    check_leakage(x, table["recording"].astype(str).to_numpy())
    y = table["y"].to_numpy().astype(np.int8)
    w_all = sample_weights(table)
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    refusals: list[Refusal] = []
    corpus: list[dict[str, object]] = []
    priors: dict[str, tuple[lgb.Booster, I64]] = {}

    def train_on(rows: I64, *, init: lgb.Booster | None = None, w: npt.NDArray[np.float64]
                 | None = None, n_rounds: int = rounds) -> lgb.Booster:
        ww = w_all[rows] if w is None else w
        return fit(x.iloc[rows], y[rows], ww, num_threads=num_threads, rounds=n_rounds,
                   init_model=init)

    for target in targets:
        pooled: lgb.Booster | None = None
        pooled_rows: I64 | None = None
        need_pooled = TrainingMode.POOLED in modes or TrainingMode.ADAPTED in modes
        if need_pooled:
            (fa,) = loao_folds(table, [target])
            pooled_rows = _subsample(fa.train, train_size, rng)
            pooled = train_on(pooled_rows)
            priors[target] = (pooled, pooled_rows)
            corpus.append(_corpus_row(table, TrainingMode.POOLED, target, pooled_rows))
            if TrainingMode.POOLED in modes:
                thr = _baseline_pred(x[BASELINE_FEATURE].iloc[pooled_rows], y[pooled_rows],
                                     x[BASELINE_FEATURE].iloc[fa.evaluate])
                parts.append(_record(TrainingMode.POOLED, fa, table,
                                     predict_raw(pooled, x.iloc[fa.evaluate]), thr,
                                     int(pooled_rows.size), np.nan, "all"))
        if TrainingMode.ADAPTED in modes and pooled is not None and pooled_rows is not None:
            try:
                b_folds = adapted_folds(table, target)
            except ModeRefusedError as exc:
                refusals.append(Refusal(str(exc.mode), target, exc.reason))
                b_folds = []
            for wa in w_adapt_grid:
                for f in b_folds:
                    rows = np.concatenate([pooled_rows, f.adapt])
                    ww = np.concatenate([w_all[pooled_rows], wa * w_all[f.adapt]])
                    booster = train_on(rows, init=pooled, w=ww, n_rounds=adapt_rounds)
                    thr = _baseline_pred(x[BASELINE_FEATURE].iloc[rows], y[rows],
                                         x[BASELINE_FEATURE].iloc[f.evaluate])
                    parts.append(_record(TrainingMode.ADAPTED, f, table,
                                         predict_raw(booster, x.iloc[f.evaluate]), thr,
                                         int(rows.size), float(wa), f.held_out[0],
                                         n_adapt=int(f.adapt.size)))
            if b_folds:
                corpus.append(_corpus_row(table, TrainingMode.ADAPTED, target,
                                          np.concatenate([pooled_rows, b_folds[0].adapt]),
                                          n_folds=len(b_folds)))
        if TrainingMode.PER_ANIMAL in modes:
            try:
                c_folds = per_animal_folds(table, target)
            except ModeRefusedError as exc:
                refusals.append(Refusal(str(exc.mode), target, exc.reason))
                c_folds = []
            for f in c_folds:
                rows = _subsample(f.train, train_size, rng)
                if np.unique(y[rows]).size < 2:  # noqa: PLR2004
                    refusals.append(Refusal(str(TrainingMode.PER_ANIMAL), target,
                                            "training recordings hold one class",
                                            f.held_out[0]))
                    continue
                booster = train_on(rows)
                thr = _baseline_pred(x[BASELINE_FEATURE].iloc[rows], y[rows],
                                     x[BASELINE_FEATURE].iloc[f.evaluate])
                parts.append(_record(TrainingMode.PER_ANIMAL, f, table,
                                     predict_raw(booster, x.iloc[f.evaluate]), thr,
                                     int(rows.size), np.nan, f.held_out[0]))
            if c_folds:
                corpus.append(_corpus_row(table, TrainingMode.PER_ANIMAL, target,
                                          c_folds[0].train, n_folds=len(c_folds)))
    preds = pd.concat(parts, ignore_index=True) if parts else _empty_predictions()
    preds["p_cal"] = np.nan
    if calibration is not None and len(preds):
        for _key, g in preds.groupby(["mode", "target", "w_adapt"], dropna=False):
            preds.loc[g.index, "p_cal"] = crossfit_calibrate(
                g["raw"].to_numpy(), g["y"].to_numpy(), g["cluster"].to_numpy(),
                kind=calibration)
    registered: list[str] = []
    if registry is not None:
        registered = _register_finals(
            table, preds, y=y, w_all=w_all, train_on=train_on, priors=priors,
            modes=modes, targets=targets, w_adapt_grid=w_adapt_grid,
            adapt_rounds=adapt_rounds, record_path=record_path, registry=registry,
            user=user, old_tiers=old_tiers, label_opt_ins=label_opt_ins,
            refusals=refusals)
    return ModeRun(predictions=preds, refusals=tuple(refusals), corpus=pd.DataFrame(corpus),
                   registered=tuple(registered))


def _protocol_metrics(g: pd.DataFrame) -> dict[str, float]:
    """Held-out metrics of one mode's predictions; a metric that is undefined is absent."""
    s = scores(g["y"].to_numpy(), g["yhat"].to_numpy())
    vals = {"f1": s.f1, "precision": s.precision, "recall": s.recall,
            "prevalence": s.prevalence, "n": float(s.n), "n_pos": float(s.n_pos),
            "ece_raw": ece(g["raw"].to_numpy(), g["y"].to_numpy())}
    cal = g["p_cal"].notna().to_numpy()
    if cal.any():
        yc = g["y"].to_numpy()[cal]
        pc = g["p_cal"].to_numpy()[cal]
        sc = scores(yc, pc >= DECISION_P)
        vals.update({"ece_cal": ece(pc, yc), "f1_cal": sc.f1, "precision_cal": sc.precision,
                     "recall_cal": sc.recall, "n_cal": float(sc.n)})
    return {k: float(v) for k, v in vals.items() if np.isfinite(v)}


METRIC_BASIS: Final[dict[str, str]] = {
    "f1/precision/recall": "the model's own decision, raw P(motion) >= DECISION_P",
    "f1_cal/precision_cal/recall_cal": ("cross-fitted calibrated p_cal >= DECISION_P, on the "
                                        "n_cal rows a calibrator could be fitted for"),
    "ece_raw": "ECE of raw P(motion)", "ece_cal": "ECE of cross-fitted calibrated p_cal",
}
"""The decision basis of every metric a registered model carries (written to provenance)."""


def _register_finals(table: pd.DataFrame, preds: pd.DataFrame, *,
                     y: npt.NDArray[np.int8], w_all: npt.NDArray[np.float64],
                     train_on: Callable[..., lgb.Booster],
                     priors: Mapping[str, tuple[lgb.Booster, I64]],
                     modes: Sequence[TrainingMode], targets: Sequence[str],
                     w_adapt_grid: Sequence[float], adapt_rounds: int,
                     record_path: Path, registry: Registry, user: str,
                     old_tiers: Mapping[str, str] | None,
                     label_opt_ins: LabelOptIns | None,
                     refusals: list[Refusal]) -> list[str]:
    """Train, calibrate and register the final model of each evaluated mode."""
    run_id = str(json.loads(Path(record_path).read_text(encoding="utf-8"))["run_id"])
    keys = table["animal_key"].to_numpy()
    out: list[str] = []

    def put(mode: TrainingMode, animal: str | None, booster: lgb.Booster, rows: I64,
            held: pd.DataFrame, protocol: str, w: float | None) -> None:
        if held.empty or held["y"].nunique() < 2:  # noqa: PLR2004
            refusals.append(Refusal(str(mode), animal or "all",
                                    "no two-class held-out predictions to calibrate on",
                                    "final"))
            return
        cal = Calibrator.fit(held["raw"].to_numpy(), held["y"].to_numpy(), CALIBRATION_KIND)
        corpus_rows = table.iloc[rows]
        chash = corpus_hash(corpus_rows)
        mid = model_content_id(booster, cal, mode=mode, animal=animal, corpus_hash=chash)
        spec = ModelSpec(mode=mode, animal=animal, version=run_id, corpus_hash=chash,
                         calibrator=calibrator_relpath(mid), trained_at=datetime.now(UTC),
                         metrics={protocol: _protocol_metrics(held)},
                         n_train_events=int(rows.size))
        held_targets = sorted(set(held["target"].astype(str)))
        fitted_on = (
            "held-out predictions of the per-target pooled LOAO models (one model per "
            "target, each trained without that target); NOT of this final model, which is "
            "trained on every row" if mode is TrainingMode.POOLED else
            f"held-out predictions of this target's {protocol} folds; NOT of this final "
            "model, which is trained on all of the target's rows")
        prov = build_provenance(spec, corpus=corpus_composition(corpus_rows, old_tiers),
                                record_path=record_path, w_adapt=w, calibration={
                                    "kind": CALIBRATION_KIND, "fitted_on": fitted_on,
                                    "protocol": protocol, "targets": held_targets,
                                    "n_predictions": len(held)})
        if label_opt_ins is not None:
            prov["label_opt_ins"] = label_opt_ins.to_dict()
        prov["metric_basis"] = dict(METRIC_BASIS)
        if mode is TrainingMode.POOLED:
            prov["metrics_note"] = (
                f"metrics are {protocol} (each target scored by a model trained without it); "
                "this final model includes every animal key: "
                + ", ".join(sorted(set(corpus_rows["animal_key"].astype(str)))))
        out.append(registry.register(spec, booster, cal, user=user, provenance=prov,
                                     corpus_id=run_id))

    if TrainingMode.POOLED in modes:
        rows = np.arange(len(table), dtype=np.int64)
        held = preds[preds["mode"] == str(TrainingMode.POOLED)]
        put(TrainingMode.POOLED, None, train_on(rows), rows, held, PROTOCOL[
            TrainingMode.POOLED], None)
    for target in targets:
        is_t = keys == target
        t_rows = _rows(is_t)
        if TrainingMode.PER_ANIMAL in modes:
            held = preds[(preds["mode"] == str(TrainingMode.PER_ANIMAL))
                         & (preds["target"] == target)]
            if not held.empty and np.unique(y[t_rows]).size == 2:  # noqa: PLR2004
                put(TrainingMode.PER_ANIMAL, target, train_on(t_rows), t_rows, held,
                    PROTOCOL[TrainingMode.PER_ANIMAL], None)
            else:
                refusals.append(Refusal(str(TrainingMode.PER_ANIMAL), target,
                                        "no held-out predictions or one-class labels",
                                        "final"))
        if TrainingMode.ADAPTED in modes and target in priors:
            prior, prior_rows = priors[target]
            for wa in w_adapt_grid:
                held = preds[(preds["mode"] == str(TrainingMode.ADAPTED))
                             & (preds["target"] == target) & (preds["w_adapt"] == wa)]
                if held.empty:
                    refusals.append(Refusal(str(TrainingMode.ADAPTED), target,
                                            f"no held-out predictions at w_adapt={wa:g}",
                                            "final"))
                    continue
                rows = np.concatenate([prior_rows, t_rows])
                ww = np.concatenate([w_all[prior_rows], wa * w_all[t_rows]])
                booster = train_on(rows, init=prior, w=ww, n_rounds=adapt_rounds)
                put(TrainingMode.ADAPTED, target, booster, rows, held,
                    PROTOCOL[TrainingMode.ADAPTED], float(wa))
    return out


def learning_curve(table: pd.DataFrame, *, targets: Sequence[str], record_path: Path,
                   num_threads: int, sizes: Sequence[int],
                   modes: Sequence[TrainingMode] = tuple(TrainingMode),
                   w_adapt_grid: Sequence[float] = W_ADAPT_GRID, seed: int = 0,
                   n_resamples: int = 1000) -> pd.DataFrame:
    """F1 against training-set size at matched event counts (task 12's required control).

    For each size, every mode's training corpus is subsampled to that many events (the
    pooled part for A and B, the per-animal part for C) and rescored on the same held-out
    rows. A mode-C point is emitted only if **every** fold had at least that many
    training events, so a point never silently stands for a smaller corpus. One line per
    mode (and per ``w_adapt`` for B), labelled with its protocol (invariant 12).
    ``n_train_events`` is the median actual size of the subsampled corpus - for B the
    pooled part only, so B sits on the same x as A; its adaptation rows are reported in
    ``n_adapt_events``.
    """
    rows = []
    for n in sizes:
        run = run_modes(table, targets=targets, record_path=record_path,
                        num_threads=num_threads, modes=modes, w_adapt_grid=w_adapt_grid,
                        calibration=None, train_size=int(n), seed=seed)
        p = run.predictions
        for (mode, target, w), g in p.groupby(["mode", "target", "w_adapt"], dropna=False):
            per_fold = g.groupby("fold")["n_train_events"].first()
            n_adapt = g.groupby("fold")["n_adapt_events"].first()
            if mode == str(TrainingMode.PER_ANIMAL) and int(per_fold.min()) < int(n):
                continue
            s = scores(g["y"].to_numpy(), g["yhat"].to_numpy())
            lo, hi = cluster_bootstrap_ci(g["y"].to_numpy(), g["yhat"].to_numpy(),
                                          g["cluster"].to_numpy(), n_resamples=n_resamples,
                                          seed=seed)
            proto = PROTOCOL[TrainingMode(str(mode))]
            line = (f"{mode} ({proto})" if mode != str(TrainingMode.ADAPTED)
                    else f"adapted w={w:g} ({proto}; x excludes adaptation rows)")
            rows.append({"target": target, "line": line, "mode": mode, "protocol": proto,
                         "w_adapt": w, "requested_size": int(n),
                         "n_train_events": float((per_fold - n_adapt).median()),
                         "n_adapt_events": float(n_adapt.median()), "f1": s.f1,
                         "f1_lo": lo, "f1_hi": hi, "n_pos": s.n_pos, "n": s.n})
    return pd.DataFrame(rows)


def _empty_predictions() -> pd.DataFrame:
    cols = ["mode", "target", "protocol", "w_adapt", "fold", "row", "recording", "cluster",
            "y", "raw", "yhat", "yhat_thr", "n_train_events", "n_adapt_events",
            "n_excluded_unknown"]
    return pd.DataFrame({c: [] for c in cols})


def _corpus_row(table: pd.DataFrame, mode: TrainingMode, target: str, rows: I64, *,
                n_folds: int = 1) -> dict[str, object]:
    """Labelled events, positive fraction and recording count of a mode's TRAINING corpus.

    For B and C, whose corpus changes with the held-out recording, this is the first
    fold's training corpus (held-out rows excluded); ``n_folds`` says how many there are.
    """
    sub = table.iloc[rows]
    return {"mode": str(mode), "target": target, "corpus_of": "fold 1 of "
            f"{n_folds}" if n_folds > 1 else "the only fold", "n_events": len(sub),
            "positive_fraction": float(sub["y"].mean()) if len(sub) else float("nan"),
            "n_recordings": int(sub["recording"].nunique()),
            "n_animals": int(sub["animal_key"].nunique())}
