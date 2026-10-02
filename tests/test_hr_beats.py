"""Tests for :mod:`gems_blanking_v2.emit.hr_beats` - the MATLAB boundary (invariant 15)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.emit.hr_beats import (
    read_blank_spans,
    read_hr_beats,
    to_blank_spans,
    to_heartlocs,
    write_hr_beats,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.io import loadmat

FS = 24414.0625


def test_a_beat_at_zero_based_sample_k_is_heartlocs_k_plus_one() -> None:
    h = to_heartlocs([4 / FS, 10 / FS], FS, 0.0, 100)
    assert h.shape == (2, 1) and h.ravel().tolist() == [5.0, 11.0]


def test_the_epoch_offset_is_subtracted_before_indexing() -> None:
    h = to_heartlocs([132.0 + 4 / FS], FS, 132.0, 100)
    assert h.ravel().tolist() == [5.0]


def test_a_beat_outside_the_epoch_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the epoch"):
        to_heartlocs([100 / FS], FS, 0.0, 100)
    with pytest.raises(ValueError, match="outside the epoch"):
        to_heartlocs([-1 / FS], FS, 0.0, 100)


def test_two_beats_on_one_sample_are_refused() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        to_heartlocs([4 / FS, 4.2 / FS], FS, 0.0, 100)


def test_the_file_holds_her_variables(tmp_path: Path) -> None:
    p = write_hr_beats(tmp_path / "b.mat", [4 / FS], fs=FS, epoch_start_s=0.0, n_samples=100,
                       channel="L_T", source="task 05 + transient veto")
    m = loadmat(p)
    assert m["heartlocs"].shape == (1, 1) and float(m["heartlocs"][0, 0]) == 5.0
    assert float(np.asarray(m["fs"]).squeeze()) == FS
    assert str(np.asarray(m["beatChannel"]).squeeze()) == "L_T"


@given(st.lists(st.integers(0, 10**6), min_size=1, max_size=50, unique=True),
       st.floats(0.0, 2000.0, allow_nan=False))
@settings(max_examples=200, deadline=None)
def test_the_round_trip_is_exact_to_the_sample(samples: list[int], start: float) -> None:
    """Sample k -> heartlocs k+1 -> back to the same sample, whatever the epoch start."""
    import tempfile  # noqa: PLC0415

    k = np.sort(np.asarray(samples))
    t = start + k / FS
    with tempfile.TemporaryDirectory() as d:
        p = write_hr_beats(Path(d) / "b.mat", t, fs=FS, epoch_start_s=start, n_samples=10**6 + 1,
                           channel="X", source="test")
        back, fs = read_hr_beats(p)
    assert fs == FS
    assert np.array_equal(np.round((back - start) * FS).astype(int), k)


def test_hr_beats_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.emit.hr_beats" not in chain.generation_modules()


# --- mask-grade beats (addendum to ruling (c) 1) --------------------------------


def test_mask_grade_beats_round_trip_under_their_own_name_and_variable(tmp_path) -> None:  # noqa: ANN001
    from gems_blanking_v2.emit.hr_beats import read_mask_beats, write_mask_beats  # noqa: PLC0415

    b = np.array([0.5, 0.65, 0.8])
    f = write_mask_beats(tmp_path / "r_peri_r_beats.mat", b, fs=1000.0, epoch_start_s=0.0,
                         n_samples=2000, channel="RVN3", source="count gate only")
    m = loadmat(f)
    assert "heartlocs" not in m  # her function loads heartlocs and refuses without it
    assert m["maskBeatlocs"].ravel().tolist() == [501.0, 651.0, 801.0]
    t, fs = read_mask_beats(f)
    assert fs == 1000.0 and np.allclose(t, b)


def test_the_two_grades_cannot_be_written_or_read_as_each_other(tmp_path) -> None:  # noqa: ANN001
    from gems_blanking_v2.emit.hr_beats import (  # noqa: PLC0415
        read_mask_beats,
        write_hr_beats,
        write_mask_beats,
    )

    kw = {"fs": 1000.0, "epoch_start_s": 0.0, "n_samples": 2000, "channel": "c", "source": "s"}
    with pytest.raises(ValueError, match="mask-grade name"):
        write_hr_beats(tmp_path / "r_peri_r_beats.mat", [0.5], **kw)
    with pytest.raises(ValueError, match="must end with"):
        write_mask_beats(tmp_path / "r_beats.mat", [0.5], **kw)
    hrv = write_hr_beats(tmp_path / "r_beats.mat", [0.5], **kw)
    with pytest.raises(ValueError, match="not a mask-grade"):
        read_mask_beats(hrv)


# --- gap tags (ruling (h) 3) -------------------------------------------------------------


def test_gap_tags_ride_along_without_changing_what_her_function_reads(tmp_path) -> None:  # noqa: ANN001
    from gems_blanking_v2.emit.hr_beats import read_gap_after  # noqa: PLC0415

    b = np.array([0.8, 0.5, 0.65])  # unsorted on purpose: tags follow the sorted order
    kw = {"fs": 1000.0, "epoch_start_s": 0.0, "n_samples": 2000, "channel": "c", "source": "s"}
    plain = write_hr_beats(tmp_path / "a_beats.mat", b, **kw)
    tagged = write_hr_beats(tmp_path / "b_beats.mat", b, gap_after=[False, False, True], **kw)
    p, t = loadmat(plain), loadmat(tagged)
    assert np.array_equal(p["heartlocs"], t["heartlocs"])
    assert p["fs"].item() == t["fs"].item()
    assert read_gap_after(plain) is None
    # tags follow the beats into sorted order: 0.5 (F), 0.65 (T), 0.8 (F)
    assert read_gap_after(tagged).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="entries for 3 beats"):
        write_hr_beats(tmp_path / "c_beats.mat", b, gap_after=[True], **kw)


# --- blankSpans (ruling 2026-10-02 (g) 4) ----------------------------------------------


def test_a_one_sample_blank_span_is_her_five_five() -> None:
    sp = to_blank_spans([[4 / FS, 5 / FS]], FS, 0.0, 100)
    assert sp.shape == (1, 2) and sp.tolist() == [[5.0, 5.0]]  # 1-based inclusive, as blankIdx
    off = to_blank_spans([[132.0 + 4 / FS, 132.0 + 6 / FS]], FS, 132.0, 100)
    assert off.tolist() == [[5.0, 6.0]]


def test_blank_spans_are_clipped_to_the_epoch_and_empty_ones_dropped() -> None:
    sp = to_blank_spans([[-1.0, 2 / FS], [3 / FS, 3.2 / FS], [98 / FS, 1.0]], FS, 0.0, 100)
    assert sp.tolist() == [[1.0, 2.0], [99.0, 100.0]]
    with pytest.raises(ValueError, match="must not overlap"):
        to_blank_spans([[0.0, 10 / FS], [5 / FS, 20 / FS]], FS, 0.0, 100)


@given(st.lists(st.integers(1, 40), min_size=2, max_size=12), st.integers(0, 5000))
@settings(max_examples=60, deadline=None)
def test_blank_spans_round_trip_through_the_file(steps: list[int], offset: int) -> None:
    edges = np.cumsum(steps)  # sample boundaries, strictly increasing
    pairs = edges[: 2 * (edges.size // 2)].reshape(-1, 2)  # [k0, k1) half-open, disjoint
    start = offset / FS
    spans = np.column_stack([pairs[:, 0] / FS + start, pairs[:, 1] / FS + start])
    n = int(edges.max()) + 5
    with tempfile.TemporaryDirectory() as d:
        p = write_hr_beats(Path(d) / "b.mat", [start], fs=FS, epoch_start_s=start, n_samples=n,
                           channel="x", source="t", blank_spans_s=spans)
        stored = loadmat(p)["blankSpans"]
        np.testing.assert_array_equal(stored, np.column_stack([pairs[:, 0] + 1, pairs[:, 1]]))
        np.testing.assert_allclose(read_blank_spans(p), spans, rtol=0.0, atol=1e-9)


def test_a_beat_inside_a_blank_span_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside a blank span"):
        write_hr_beats(tmp_path / "b.mat", [10 / FS], fs=FS, epoch_start_s=0.0, n_samples=100,
                       channel="x", source="t", blank_spans_s=[[10 / FS, 11 / FS]])
    # a span's edges are not inside it: [10/fs, 11/fs) is the one sample 11 (1-based)
    write_hr_beats(tmp_path / "b.mat", [9 / FS, 11 / FS], fs=FS, epoch_start_s=0.0,
                   n_samples=100, channel="x", source="t", blank_spans_s=[[10 / FS, 11 / FS]])


def test_without_blank_spans_the_file_has_no_field(tmp_path: Path) -> None:
    p = write_hr_beats(tmp_path / "b.mat", [4 / FS], fs=FS, epoch_start_s=0.0, n_samples=100,
                       channel="x", source="t")
    assert "blankSpans" not in loadmat(p) and read_blank_spans(p) is None
