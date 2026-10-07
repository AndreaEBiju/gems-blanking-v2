"""Task 10: labels as judgments on candidate cores, without inventing negatives.

Rulings applied (2026-10-07 (b)):

* **R3 - the unit is the core.** A core is ``motion`` when at least half of its own
  duration lies under a mark. A core that touches a mark less than that is ``unjudged``.
  A core that touches no mark *inside an exhaustively marked audit span* is a human-
  confirmed negative (``physiology``: Change 1's "not motion" key). In the old cohort,
  whose marks are positives only, every unmatched core is ``unjudged``, never a negative.
* **R1 - I, J and K are the prospective test set.** Their labels never enter training.
* **label_source is required (task 10).** A recording without one is ``unknown`` and is
  excluded, not assumed human; ``model`` labels need an explicit opt-in recorded in
  provenance.

The old cohort's labels are ``removedSegmentIdx`` inside ``*_blankmotion.mat`` (written by
``browseMotionArtifacts.m``). Four things measured about those files, all handled here:
1-based inclusive ``[start stop]`` rows (converted once, at this boundary - invariant 15);
v7.3 files store the array transposed as (2, N), so the format is branched on and the
array transposed, never reshaped; ``removedSegments`` can disagree, so only
``removedSegmentIdx`` is trusted; and a segment longer than 0.9 x the epoch is a protocol
exclusion (the hand-blanked stim epoch), counted as ``excluded_epoch``, never an event.
``blankingApplied`` false or absent means the file was never reviewed - not clean.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.io import loadmat, whosmat

from gems_blanking_v2.io.recording import spans_from_matlab_intervals

__all__ = [
    "ADJUDICATED_BASIS",
    "ADJUDICATION_JUDGEMENTS",
    "EXCLUDED_EPOCH_FRACTION",
    "HUMAN_OLD_BASES",
    "LABEL_COLUMNS",
    "MOTION_OVERLAP",
    "OLD_TIERS",
    "SET_A_BASIS",
    "TEST_ANIMALS",
    "AliasRow",
    "BlankmotionLabels",
    "alias_table",
    "animal_key",
    "apply_adjudications",
    "is_test_animal",
    "judge_core",
    "read_blankmotion_labels",
    "recording_label_source",
    "split_label_spans",
    "training_rows",
    "write_alias_table",
]

Judgement = Literal["motion", "physiology", "unsure", "unjudged", "line_noise"]
Source = Literal["human", "model", "inherited"]
LabelSource = Literal["human", "model", "mixed", "unknown"]

MOTION_OVERLAP: Final = 0.5
"""A core is motion when at least this fraction of its own duration is under a mark (R3)."""

EXCLUDED_EPOCH_FRACTION: Final = 0.9
"""A segment longer than this fraction of its epoch is a protocol exclusion, not an event."""

OLD_TIERS: Final[tuple[str, ...]] = ("1", "2a", "2b")
"""Old-cohort label tiers (rulings 2026-10-07 (i), (j)): 1 manifest-listed; 2a decidable
and not grid-locked; 2b undecidable, admitted on the date route only. Anything else is
excluded. The training check adds them in this order, one step at a time."""

TEST_ANIMALS: Final[frozenset[str]] = frozenset({"I", "J", "K"})
"""The prospective test set (ruling 2026-10-07 (b) R1): NEW-cohort animals I, J and K, never
in training. Letters are reused across cohorts - the old cohort's JEL is also "J" - so the set
applies to ``cohort == "new"`` only; see :func:`is_test_animal` and :func:`animal_key`."""


def is_test_animal(cohort: str, animal: str) -> bool:
    """Whether (cohort, animal) is in R1's prospective test set: new-cohort I, J, K only."""
    return cohort == "new" and animal in TEST_ANIMALS


def animal_key(cohort: str, animal: str) -> str:
    """Return the animal's identity for grouping (LOAO folds, per-animal models): ``cohort:animal``.

    The two cohorts are different rats that reuse letters (old JEL and new J are both "J");
    grouping on the bare letter would put two animals in one fold.
    """
    return f"{cohort}:{animal}"

LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "recording", "animal", "cohort", "start_s", "stop_s", "judgement", "source",
    "basis", "label_set", "label_source",
)
"""The task 10 table: one row per judged core. ``basis`` records why a row has its
judgement (``mark_overlap``, ``exhaustive_span``, ``adjudicated``, ``partial_overlap``,
``unmarked``); ``label_set`` is ``train`` or ``test``."""

NEGATIVE_JUDGEMENTS: Final[frozenset[str]] = frozenset({"physiology", "line_noise"})
"""Not motion. ``line_noise`` (ruling (c) item 3, key 4) counts as a negative."""

SET_A_BASIS: Final = "set_a_random"
"""``basis`` of an old-cohort core that Andrea judged in labelling set A's random old
sample (ruling 2026-10-08 (b) item 1(c)). What the sample is: a random draw of 300
UNMARKED (unjudged) old cores from tier 1/2a/2b recordings, stratified by animal in
proportion to each animal's unmarked cores with a floor of 10 per animal - not a draw of
all old cores. Its motion rate is therefore the rate among unmarked cores; the cohort's
rate combines it with the marked share (``train.old_motion_rate``). Judged by a human, so
neither inherited nor tiered; the old-cohort negative source without which no old-cohort
row may train (item 1(b))."""

ADJUDICATED_BASIS: Final = "adjudicated"
"""``basis`` of a core judged on the adjudication screen outside set A's random old
sample (new-cohort set A cores, the hum add-on). A human label, never a motion-rate
estimate."""

HUMAN_OLD_BASES: Final[frozenset[str]] = frozenset({SET_A_BASIS, ADJUDICATED_BASIS})
"""Old-cohort bases judged by Andrea herself: not inherited marks, so not tiered and
allowed to be negatives."""

ADJUDICATION_JUDGEMENTS: Final[frozenset[str]] = frozenset(
    {"motion", "physiology", "unsure", "line_noise"})
"""What the screen writes: keys 1-4. ``unjudged`` is never written (invariant 9)."""


# ---------------------------------------------------------------------------
# the old cohort's label files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlankmotionLabels:
    """What one ``*_blankmotion.mat`` says, before any interpretation.

    ``intervals`` are MATLAB's 1-based inclusive ``[start, stop]`` sample rows, shape
    (N, 2), already transposed from the v7.3 layout. ``blanking_applied`` is ``None`` when
    the file does not carry the flag.
    """

    path: Path
    intervals: npt.NDArray[np.int64]
    fs: float
    n_samples: int
    blanking_applied: bool | None
    file_format: Literal["v5", "v7.3"]

    @property
    def reviewed(self) -> bool:
        """True only when the file says blanking was applied; absent or false is unreviewed."""
        return self.blanking_applied is True


def _is_hdf5(path: Path) -> bool:
    with path.open("rb") as fh:
        head = fh.read(520)
    return b"\x89HDF\r\n\x1a\n" in head


def _as_rows(a: npt.ArrayLike, *, v73: bool) -> npt.NDArray[np.int64]:
    arr = np.asarray(a)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if v73:
        # HDF5 stores MATLAB's column-major (N, 2) as (2, N) - for every N, including 1
        # and 2. Transpose; never reshape: reshape(-1, 2) pairs start[k] with start[k+1]
        # (measured: 86-93% NaN instead of 100%).
        if arr.ndim != 2 or arr.shape[0] != 2:  # noqa: PLR2004
            msg = f"removedSegmentIdx has shape {arr.shape}; a v7.3 file stores (2, N)"
            raise ValueError(msg)
        arr = arr.T
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 2:  # noqa: PLR2004
        msg = f"removedSegmentIdx must be (N, 2) [start stop], got {arr.shape}"
        raise ValueError(msg)
    return np.asarray(np.rint(arr), dtype=np.int64)


def read_blankmotion_labels(path: Path) -> BlankmotionLabels:
    """Read ``removedSegmentIdx``, ``fs``, ``blankingApplied`` and the length of ``yOut``.

    Never loads ``yOut`` itself (hundreds of MB): its length comes from the header.
    """
    path = Path(path)
    if _is_hdf5(path):
        with h5py.File(path, "r") as f:
            if "removedSegmentIdx" not in f:
                msg = f"{path.name}: no removedSegmentIdx (only that field is trusted)"
                raise ValueError(msg)
            ds = f["removedSegmentIdx"]
            empty = ("MATLAB_empty" in ds.attrs
                     and bool(np.asarray(ds.attrs["MATLAB_empty"]).ravel()[0]))
            idx = np.zeros((0, 2), dtype=np.int64) if empty else _as_rows(ds[()], v73=True)
            fs = float(np.asarray(f["fs"][()]).ravel()[0])
            n = int(max(f["yOut"].shape)) if "yOut" in f else 0
            ba = (bool(np.asarray(f["blankingApplied"][()]).ravel()[0])
                  if "blankingApplied" in f else None)
        return BlankmotionLabels(path, idx, fs, n, ba, "v7.3")
    shapes = {name: shape for name, shape, _cls in whosmat(path)}
    if "removedSegmentIdx" not in shapes:
        msg = f"{path.name}: no removedSegmentIdx (only that field is trusted)"
        raise ValueError(msg)
    want = [v for v in ("removedSegmentIdx", "fs", "blankingApplied") if v in shapes]
    m = loadmat(path, variable_names=want)
    idx = _as_rows(m["removedSegmentIdx"], v73=False)
    fs = float(np.asarray(m["fs"]).ravel()[0])
    n = int(max(shapes["yOut"])) if "yOut" in shapes else 0
    ba = bool(np.asarray(m["blankingApplied"]).ravel()[0]) if "blankingApplied" in m else None
    return BlankmotionLabels(path, idx, fs, n, ba, "v5")


def split_label_spans(labels: BlankmotionLabels, epoch_s: float | None = None
                      ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return ``(event_spans, excluded_epoch_spans)`` in seconds, 0-based half-open.

    ``epoch_s`` defaults to the file's own length. A span longer than
    ``EXCLUDED_EPOCH_FRACTION`` of it is a protocol exclusion, not an event.
    """
    epoch = float(epoch_s) if epoch_s is not None else labels.n_samples / labels.fs
    if not epoch > 0:
        msg = f"{labels.path.name}: epoch length unknown; pass epoch_s"
        raise ValueError(msg)
    spans = spans_from_matlab_intervals(labels.intervals, labels.fs)
    events = [s for s in spans if (s[1] - s[0]) <= EXCLUDED_EPOCH_FRACTION * epoch]
    excluded = [s for s in spans if (s[1] - s[0]) > EXCLUDED_EPOCH_FRACTION * epoch]
    return events, excluded


# ---------------------------------------------------------------------------
# judging cores (R3)
# ---------------------------------------------------------------------------


def _overlap(a: float, b: float, spans: Sequence[tuple[float, float]]) -> float:
    """Seconds of [a, b) covered by the union of ``spans``."""
    ivs = sorted((max(a, s), min(b, e)) for s, e in spans if e > a and s < b)
    merged: list[list[float]] = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return float(sum(e - s for s, e in merged))


def judge_core(start_s: float, stop_s: float, marks: Sequence[tuple[float, float]], *,
               exhaustive_span: tuple[float, float] | None = None) -> tuple[Judgement, str]:
    """Judge one core against marks (R3). Returns ``(judgement, basis)``.

    ``exhaustive_span`` is the audit span the marks were drawn in exhaustively; a core must
    lie wholly inside it to be a negative. Without one (the old cohort), an unmatched core
    is ``unjudged``.
    """
    dur = stop_s - start_s
    if not dur > 0:
        msg = f"a core must have positive duration, got [{start_s}, {stop_s})"
        raise ValueError(msg)
    frac = _overlap(start_s, stop_s, marks) / dur
    if frac >= MOTION_OVERLAP:
        return "motion", "mark_overlap"
    if frac > 0:
        return "unjudged", "partial_overlap"
    inside = (exhaustive_span is not None
              and exhaustive_span[0] <= start_s and stop_s <= exhaustive_span[1])
    if inside:
        return "physiology", "exhaustive_span"
    return "unjudged", "unmarked"


# ---------------------------------------------------------------------------
# label_source and the training loader
# ---------------------------------------------------------------------------


def recording_label_source(value: str | None) -> LabelSource:
    """Normalise a recording's ``label_source``; absent means ``unknown`` (excluded)."""
    if value is None or value == "":
        return "unknown"
    if value not in ("human", "model", "mixed"):
        msg = f"label_source must be human, model or mixed, got {value!r}"
        raise ValueError(msg)
    return value  # type: ignore[return-value]


def training_rows(table: pd.DataFrame, *, allow_model_labels: bool = False,
                  old_tiers: Mapping[str, str] | None = None,
                  keep_tiers: Iterable[str] = ("1",)) -> pd.DataFrame:
    """Return the rows task 12 may train on, with a binary ``y`` (1 = motion).

    Drops ``unjudged`` and ``unsure`` (never negatives), every test-set row and animal
    (R1), and every recording whose ``label_source`` is ``unknown`` - or ``model`` /
    ``mixed`` unless ``allow_model_labels`` (an opt-in the caller records in provenance).

    Old-cohort rows (rulings 2026-10-07 (i), (j)) are kept only for recordings whose tier
    in ``old_tiers`` is one of ``keep_tiers`` (labels from :data:`OLD_TIERS`; default tier 1
    only). A recording absent from the map, or with any other label, is excluded - never
    assumed certain. ``old_tiers=None`` keeps no tiered old-cohort row. Old cores Andrea
    judged herself (``basis`` in :data:`HUMAN_OLD_BASES`) are not inherited marks, so the
    tier filter does not apply to them.
    Raises naming the column if a required column is missing.
    """
    missing = [c for c in LABEL_COLUMNS if c not in table.columns]
    if missing:
        msg = f"label table is missing required columns: {missing}"
        raise ValueError(msg)
    if table["label_source"].isna().any():
        msg = "label_source is null on some rows; write 'unknown' explicitly so it is excluded"
        raise ValueError(msg)
    ok_src = {"human"} | ({"model", "mixed"} if allow_model_labels else set())
    keep = (table["judgement"].isin(["motion", *NEGATIVE_JUDGEMENTS])
            & (table["label_set"] == "train")
            & ~(table["cohort"].eq("new") & table["animal"].isin(TEST_ANIMALS))
            & table["label_source"].isin(ok_src))
    allowed = set(keep_tiers)
    unknown = allowed - set(OLD_TIERS)
    if unknown:
        msg = f"keep_tiers must be drawn from {OLD_TIERS}, got {sorted(unknown)}"
        raise ValueError(msg)
    tiers = old_tiers or {}
    old_ok = table["recording"].map(lambda r: tiers.get(r) in allowed)
    keep &= (table["cohort"] != "old") | old_ok | table["basis"].isin(sorted(HUMAN_OLD_BASES))
    out = table.loc[keep].copy()
    out["y"] = (out["judgement"] == "motion").astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# the old cohort's animal alias table (inferred, then user-confirmed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AliasRow:
    """One animal token from the old cohort's file stems, awaiting confirmation."""

    token: str
    proposed: str | None
    n_files: int
    examples: tuple[str, ...]


def _token(stem: str) -> str | None:
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else None


def alias_table(stems: Iterable[str],
                propose: Mapping[str, str | None] | None = None) -> list[AliasRow]:
    """Group stems by their animal token (the second underscore token) for confirmation.

    The proposal comes from ``propose`` (token -> letter), typically GEMSBlanking's
    ``extract_animal_letter`` applied to a stem; a token it cannot place gets ``None``.
    Nothing here confirms anything: every row is for Andrea to accept or correct, so that
    no silent normaliser decides that ``ft`` is Frenchtoast (task 10 loading note 4).
    """
    by: dict[str, list[str]] = {}
    for s in stems:
        t = _token(s)
        if t is not None:
            by.setdefault(t, []).append(s)
    order = sorted(by.items(), key=lambda kv: kv[0].lower())
    return [AliasRow(token=t, proposed=(propose or {}).get(t), n_files=len(v),
                     examples=tuple(sorted(v)[:3])) for t, v in order]


def write_alias_table(rows: Sequence[AliasRow], path: Path) -> Path:
    """Write the proposal for Andrea: token, proposed letter, count, examples, confirmed=false."""
    body = [{"token": r.token, "proposed": r.proposed, "n_files": r.n_files,
             "examples": list(r.examples), "confirmed": False} for r in rows]
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(body, indent=1, ensure_ascii=True) + "\n"
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# adjudications (task 16 Change 1 screen) onto the core table
# ---------------------------------------------------------------------------


def apply_adjudications(cores: pd.DataFrame, adj: pd.DataFrame, *,
                        rate_sample_keys: Iterable[str]
                        ) -> tuple[pd.DataFrame, dict[str, object]]:
    """Put the adjudication screen's judgements on the cores they judged.

    ``cores`` and ``adj`` both carry ``core_key`` (built by the screen's own key function,
    so the join uses one construction site). Each judged core takes its judgement as
    written: ``motion`` (key 1) is a positive, ``physiology`` (2) and ``line_noise`` (4)
    are negatives, ``unsure`` (3) is kept as ``unsure`` and excluded by
    :func:`training_rows`; a core nobody judged keeps its own judgement (``unjudged``
    stays unjudged - invariant 9). ``source`` and ``label_source`` become ``human``;
    ``basis`` is :data:`SET_A_BASIS` for an OLD-cohort core whose key is in
    ``rate_sample_keys`` - the queue rows drawn as set A's random old sample (``why ==
    "old_random"``), matched by core key, never by queue file - and
    :data:`ADJUDICATED_BASIS` otherwise.

    A core judged twice keeps the later judgement (``at``). A judgement that would
    overwrite a core that already carries a different judgement (an audit-span label, an
    inherited mark) is not applied: the core becomes ``unjudged`` with basis
    ``adjudication_conflict`` and is counted - two disagreeing labels never train.
    Raises when a judgement's key matches no core, or its cohort, animal or label_set
    disagree with the core's (a data error, not a label).
    """
    need = ("core_key", "judgement", "at", "cohort", "animal", "label_set")
    gap = [c for c in need if c not in adj.columns]
    if gap or "core_key" not in cores.columns:
        msg = f"adjudication join needs core_key on both sides and {list(need)}; missing {gap}"
        raise ValueError(msg)
    bad = sorted(set(adj["judgement"].astype(str)) - ADJUDICATION_JUDGEMENTS)
    if bad:
        msg = f"judgements {bad} are not what the screen writes ({sorted(ADJUDICATION_JUDGEMENTS)})"
        raise ValueError(msg)
    a = adj.sort_values("at", kind="stable")
    n_dup = int(a["core_key"].duplicated().sum())
    a = a.drop_duplicates("core_key", keep="last").set_index("core_key")
    out = cores.copy()
    pos = pd.Series(np.arange(len(out)), index=out["core_key"].to_numpy())
    missing = sorted(set(a.index) - set(pos.index))
    if missing:
        msg = f"{len(missing)} judgement(s) match no core, e.g. {missing[:3]}"
        raise ValueError(msg)
    rows = pos.loc[a.index].to_numpy()
    for col in ("cohort", "animal", "label_set"):
        mism = out[col].astype(str).to_numpy()[rows] != a[col].astype(str).to_numpy()
        if mism.any():
            msg = (f"{int(mism.sum())} judgement(s) disagree with their core on {col!r}, e.g. "
                   f"{list(a.index[mism][:3])}")
            raise ValueError(msg)
    prior = out["judgement"].astype(str).to_numpy()[rows]
    new = a["judgement"].astype(str).to_numpy()
    conflict = (prior != "unjudged") & (prior != new)
    is_old = a["cohort"].astype(str).to_numpy() == "old"
    from_random = a.index.isin(list(set(rate_sample_keys)))
    basis = np.where(is_old & from_random, SET_A_BASIS, ADJUDICATED_BASIS)
    ok = rows[~conflict]
    out.loc[out.index[ok], "judgement"] = new[~conflict]
    out.loc[out.index[ok], "basis"] = basis[~conflict]
    out.loc[out.index[ok], ["source", "label_source"]] = "human"
    out.loc[out.index[rows[conflict]], "judgement"] = "unjudged"
    out.loc[out.index[rows[conflict]], "basis"] = "adjudication_conflict"
    counts = (a.assign(basis=basis).groupby(["cohort", "basis", "judgement"]).size()
              .rename("n").reset_index())
    report: dict[str, object] = {
        "n_judgements": len(adj), "n_cores_judged": len(a), "n_duplicate_judgements": n_dup,
        "n_conflicts": int(conflict.sum()),
        "conflicts": [str(k) for k in a.index[conflict][:20]],
        "counts": {f"{c}|{q}|{j}": int(n) for c, q, j, n in counts.itertuples(index=False)},
        "n_set_a_old": int((basis == SET_A_BASIS)[~conflict].sum())}
    return out, report
