"""Task 10: old-cohort label files, core judgments (R3), and the training loader."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.model import labels as lb
from scipy.io import savemat

FS = 1000.0


def _write_v5(path: Path, idx: np.ndarray, n: int, applied: bool | None = True) -> Path:
    body = {"removedSegmentIdx": idx.astype(np.float64), "fs": FS, "yOut": np.zeros((n, 1))}
    if applied is not None:
        body["blankingApplied"] = np.uint8(applied)
    savemat(path, body)
    return path


def _write_v73(path: Path, idx: np.ndarray, n: int, applied: bool | None = True) -> Path:
    # MATLAB v7.3 is HDF5, column-major: a MATLAB (N, 2) array is stored as (2, N), and a
    # MATLAB (n, 1) yOut as (1, n). A real file also has a 512-byte user block.
    with h5py.File(path, "w", userblock_size=512) as f:
        f["removedSegmentIdx"] = idx.T.astype(np.float64)
        f["fs"] = np.array([[FS]])
        f["yOut"] = np.zeros((1, n))
        if applied is not None:
            f["blankingApplied"] = np.array([[np.uint8(applied)]])
    return path


IDX = np.array([[11, 20], [101, 300], [401, 401]])
"""1-based inclusive rows: 10 + 200 + 1 = 211 samples."""


@pytest.mark.parametrize("writer", [_write_v5, _write_v73], ids=["v5", "v7.3"])
def test_both_formats_read_the_same_rows_and_mask_fraction(
        tmp_path: Path, writer: Callable[..., Path]) -> None:
    labels = lb.read_blankmotion_labels(writer(tmp_path / "x_blankmotion.mat", IDX, 1000))
    np.testing.assert_array_equal(labels.intervals, IDX)
    assert labels.n_samples == 1000 and labels.fs == FS and labels.reviewed
    events, excluded = lb.split_label_spans(labels)
    masked = sum(b - a for a, b in events) * FS
    assert masked == pytest.approx(211) and excluded == []


def test_two_intervals_in_v73_are_transposed_not_reshaped(tmp_path: Path) -> None:
    # (2, 2) is the shape where a reshape and a transpose disagree silently
    idx = np.array([[1, 50], [501, 1000]])
    labels = lb.read_blankmotion_labels(_write_v73(tmp_path / "y_blankmotion.mat", idx, 1000))
    np.testing.assert_array_equal(labels.intervals, idx)


def test_one_based_inclusive_converts_to_zero_based_half_open(tmp_path: Path) -> None:
    path = _write_v5(tmp_path / "z_blankmotion.mat", np.array([[5, 5]]), 100)
    labels = lb.read_blankmotion_labels(path)
    events, _ = lb.split_label_spans(labels)
    assert events == [(4 / FS, 5 / FS)]


def test_a_segment_over_090_of_the_epoch_is_a_protocol_exclusion(tmp_path: Path) -> None:
    idx = np.array([[1, 905], [950, 960]])  # 905 of 1000 samples, then a 11-sample event
    labels = lb.read_blankmotion_labels(_write_v5(tmp_path / "s_blankmotion.mat", idx, 1000))
    events, excluded = lb.split_label_spans(labels)
    assert excluded == [(0.0, 0.905)] and events == [(0.949, 0.960)]


def test_a_long_event_under_090_of_the_epoch_is_still_an_event(tmp_path: Path) -> None:
    labels = lb.read_blankmotion_labels(
        _write_v5(tmp_path / "l_blankmotion.mat", np.array([[1, 850]]), 1000))
    events, excluded = lb.split_label_spans(labels)
    assert events == [(0.0, 0.850)] and excluded == []


def test_absent_or_false_blanking_applied_is_unreviewed(tmp_path: Path) -> None:
    a = lb.read_blankmotion_labels(_write_v5(tmp_path / "a_blankmotion.mat", IDX, 1000, None))
    b = lb.read_blankmotion_labels(_write_v73(tmp_path / "b_blankmotion.mat", IDX, 1000, False))
    assert a.blanking_applied is None and not a.reviewed
    assert b.blanking_applied is False and not b.reviewed


def test_core_judgments_follow_r3() -> None:
    marks = [(1.0, 2.0)]
    span = (0.0, 10.0)
    assert lb.judge_core(1.2, 1.8, marks, exhaustive_span=span) == ("motion", "mark_overlap")
    assert lb.judge_core(1.5, 2.5, marks, exhaustive_span=span)[0] == "motion"  # exactly 50%
    assert lb.judge_core(1.6, 2.6, marks, exhaustive_span=span) == ("unjudged", "partial_overlap")
    assert lb.judge_core(5.0, 5.1, marks, exhaustive_span=span) == ("physiology", "exhaustive_span")
    assert lb.judge_core(9.95, 10.05, marks, exhaustive_span=span) == ("unjudged", "unmarked")
    # the old cohort: marks are positives only
    assert lb.judge_core(5.0, 5.1, marks) == ("unjudged", "unmarked")
    assert lb.judge_core(1.2, 1.8, marks)[0] == "motion"


def test_overlap_uses_the_union_of_marks() -> None:
    # two overlapping marks must not be double-counted: 0.4 s of a 1.0 s core
    assert lb.judge_core(0.0, 1.0, [(0.0, 0.3), (0.1, 0.4)])[0] == "unjudged"


def _table(**over: object) -> pd.DataFrame:
    rows = [  # in LABEL_COLUMNS order
        ("r1", "A", "new", 0, 1, "motion", "human", "mark_overlap", "train", "human"),
        ("r1", "A", "new", 2, 3, "physiology", "human", "exhaustive_span", "train", "human"),
        ("r1", "A", "new", 4, 5, "unjudged", "human", "partial_overlap", "train", "human"),
        ("r1", "A", "new", 6, 7, "unsure", "human", "adjudicated", "train", "human"),
        ("r2", "I", "new", 0, 1, "motion", "human", "mark_overlap", "test", "human"),
        ("r3", "L", "old", 0, 1, "motion", "inherited", "mark_overlap", "train", "unknown"),
        ("r4", "J", "new", 0, 1, "line_noise", "human", "adjudicated", "train", "human"),
    ]
    df = pd.DataFrame(rows, columns=list(lb.LABEL_COLUMNS))
    for k, v in over.items():
        df[k] = v
    return df


def test_the_loader_drops_unjudged_unsure_test_animals_and_unknown_sources() -> None:
    out = lb.training_rows(_table())
    assert out[["start_s", "judgement"]].values.tolist() == [[0, "motion"], [2, "physiology"]]
    assert out["y"].tolist() == [1, 0]
    # an I/J/K row is never trained on, even if mislabelled 'train'
    assert "J" not in set(out["animal"])


def _old_rows() -> pd.DataFrame:
    rows = [(f"o{k}", "L", "old", 0, 1, "motion", "inherited", "mark_overlap", "train", "human")
            for k in range(1, 5)]
    return pd.DataFrame(rows, columns=list(lb.LABEL_COLUMNS))


def test_old_cohort_rows_follow_the_tiers_of_ruling_i() -> None:
    t = _old_rows()
    tiers = {"o1": 1, "o2": 2, "o3": 0}  # o4 is absent from the map
    assert lb.training_rows(t)["recording"].tolist() == []  # no map, no old rows
    assert lb.training_rows(t, old_tiers=tiers)["recording"].tolist() == ["o1"]
    both = lb.training_rows(t, old_tiers=tiers, include_tier2=True)
    assert both["recording"].tolist() == ["o1", "o2"]
    # new-cohort rows never depend on the old-cohort tiers
    assert len(lb.training_rows(_table(), old_tiers={})) == 2


def test_model_labels_need_an_explicit_opt_in() -> None:
    t = _table()
    t.loc[0, "label_source"] = "model"
    assert 0 not in lb.training_rows(t)["start_s"].tolist()
    assert 0 in lb.training_rows(t, allow_model_labels=True)["start_s"].tolist()


def test_a_missing_column_or_null_source_raises_naming_it() -> None:
    with pytest.raises(ValueError, match="label_source"):
        lb.training_rows(_table().drop(columns=["label_source"]))
    t = _table()
    t.loc[0, "label_source"] = None
    with pytest.raises(ValueError, match="null"):
        lb.training_rows(t)


def test_label_source_absent_is_unknown_and_garbage_raises() -> None:
    assert lb.recording_label_source(None) == "unknown"
    assert lb.recording_label_source("human") == "human"
    with pytest.raises(ValueError):
        lb.recording_label_source("humans")


def test_the_alias_table_proposes_and_never_confirms(tmp_path: Path) -> None:
    stems = ["E1000_FRE_E1000_bl_1356", "E10_ft_E10_bl_0001", "M100_loli_MS2_bl_1",
             "M100_loll2_MS2_bl_2", "E100_jel_E100_bl_1151", "E100_JEL_E100_bl_1315"]
    rows = lb.alias_table(stems, propose={"FRE": "F", "JEL": "J", "jel": "J"})
    toks = {r.token: r for r in rows}
    assert set(toks) == {"FRE", "ft", "loli", "loll2", "jel", "JEL"}
    assert toks["ft"].proposed is None and toks["loll2"].proposed is None
    path = lb.write_alias_table(rows, tmp_path / "aliases.json")
    assert all(r["confirmed"] is False for r in json.loads(path.read_text(encoding="utf-8")))
