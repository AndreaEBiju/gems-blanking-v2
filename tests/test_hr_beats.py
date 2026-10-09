"""Tests for :mod:`gems_blanking_v2.emit.hr_beats` - the MATLAB boundary (invariant 15)."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.emit.hr_beats import (
    beats_file_record,
    read_blank_spans,
    read_hr_beats,
    to_blank_spans,
    to_heartlocs,
    train_origin,
    write_hr_beats,
)
from gems_blanking_v2.extent.grid import seconds_to_sample
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.io import loadmat

from tests.conftest import RoutedTrain, make_routed_train

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


# ---------------------------------------------------------------------------
# the origin of a stored train (fix (c), origin_check/REPORT.md, 2026-10-09)
# ---------------------------------------------------------------------------

I0_132 = 3_222_656
"""round(132 x 24414.0625): the stim_rec routing region's first file sample (REPORT.md)."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_routed(tmp: Path, tr: RoutedTrain) -> Path:
    """Write ``tr`` exactly as hr10_pass.py does: region-relative, epoch_start_s = 0."""
    return write_hr_beats(tmp / "x_beats.mat", tr.beats_s, fs=tr.fs, epoch_start_s=0.0,
                          n_samples=tr.n_region, channel="LVN2-RVN2", source="synthetic",
                          gap_after=tr.gap_after, blank_spans_s=tr.blank_spans_s)


def test_a_stim_rec_train_takes_its_origin_from_the_region_not_the_file(tmp_path: Path) -> None:
    """The stored file says 0; the build record's region says round(132 fs). The latter wins."""
    tr = make_routed_train(seed=4)
    assert tr.origin_sample0 == I0_132
    p = _write_routed(tmp_path, tr)
    m = loadmat(p)
    assert float(m["epochStart_s"].squeeze()) == 0.0  # the stored, load-bearing 0
    origin, n_region = train_origin(list(tr.region_s), tr.fs, what="x",
                                    n_region_samples=tr.n_region,
                                    declared_start_s=float(m["epochStart_s"].squeeze()))
    assert (origin, n_region) == (I0_132, tr.n_region)
    h = m["heartlocs"].ravel().astype(np.int64)
    np.testing.assert_array_equal(origin + h - 1, tr.file_samples0)
    assert int(h.max()) <= n_region <= tr.n_file - origin


def test_a_bl_train_has_origin_zero() -> None:
    tr = make_routed_train(region_start_s=0.0, file_s=1189.0, seed=1)
    assert train_origin(list(tr.region_s), tr.fs, what="bl", n_region_samples=tr.n_region,
                        declared_start_s=0.0) == (0, tr.n_region)


@settings(max_examples=300, deadline=None)
@given(region_start=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
       offset=st.integers(0, 50_000), fs=st.sampled_from([FS, 24414.0, 1000.0, 610.3515625]))
def test_a_beat_at_file_time_t_comes_back_as_t(region_start: float, offset: int,
                                               fs: float) -> None:
    """Write a beat region-relative (hr10_pass.py), resolve the origin, read it back.

    The writer side is hr10_pass.py's own slice, ``data[round(lo*fs):round(hi*fs)]`` (a
    ``range`` stands in for the file's rows, so the beat's region row is found, not
    computed), independent of :func:`train_origin`. Includes fractional-sample region
    starts: only round(lo x fs) returns the beat to its own file sample.
    """
    hi = region_start + 300.0
    file_rows = range(round(hi * fs) + 10)
    region = file_rows[round(region_start * fs):round(hi * fs)]  # hr10_pass.py:50-53
    s = region[min(offset, len(region) - 1)]  # the beat's 0-based file sample
    t = s / fs                                # its file time
    with tempfile.TemporaryDirectory() as d:
        p = write_hr_beats(Path(d) / "b.mat", [region.index(s) / fs], fs=fs, epoch_start_s=0.0,
                           n_samples=len(region), channel="x", source="t")
        h = int(loadmat(p)["heartlocs"].squeeze())
    origin, n_region = train_origin([region_start, hi], fs, what="h")
    assert n_region == len(region)
    back = origin + h - 1
    assert back == s and back / fs == t


def test_the_build_record_and_a_declared_origin_must_agree() -> None:
    r = [132.0, 1320.78]
    n = seconds_to_sample(1320.78, FS) - I0_132
    assert train_origin(r, FS, what="a", declared_sample0=I0_132)[0] == I0_132
    assert train_origin(r, FS, what="a", declared_start_s=132.0)[0] == I0_132
    with pytest.raises(ValueError, match="rid_x: the beats file declares epochStartSample0 0"):
        train_origin(r, FS, what="rid_x", declared_sample0=0)
    with pytest.raises(ValueError, match="rid_x: the beats file declares epochStart_s 100"):
        train_origin(r, FS, what="rid_x", declared_start_s=100.0)
    with pytest.raises(ValueError, match="rid_x: build record n_samples"):
        train_origin(r, FS, what="rid_x", n_region_samples=n + 1)
    with pytest.raises(ValueError, match="rid_x: build record region_s"):
        train_origin([132.0], FS, what="rid_x")
    with pytest.raises(ValueError, match="rid_x: build record region_s"):
        train_origin([132.0, 100.0], FS, what="rid_x")
    with pytest.raises(ValueError, match="before the recording's first sample"):
        train_origin([-1.0, 10.0], FS, what="rid_x")


def test_the_beats_file_record_counts_the_beats_in_the_epoch(tmp_path: Path) -> None:
    tr = make_routed_train(seed=2)
    p = _write_routed(tmp_path, tr)
    h = loadmat(p)["heartlocs"].ravel()
    i0, n = tr.origin_sample0, tr.n_file - tr.origin_sample0
    rec = beats_file_record(grade="hrv", store_rel="data/A/x/x_beats.mat", sha256=_sha(p),
                            origin_sample0=tr.origin_sample0, heartlocs=h,
                            epoch_start_sample=i0, n_samples=n, published=False,
                            read_from="scratchpad:pr/b1/hrc_g/x_beats.mat")
    assert rec["hrv_beats"] == "data/A/x/x_beats.mat" and "mask_grade_beats" not in rec
    assert rec["origin_sample0"] == I0_132 and rec["n_beats"] == h.size
    assert rec["n_in_epoch"] == h.size  # the recovery epoch holds the whole train
    # a later sub-epoch: exactly the beats whose file sample lies in [j0, j0 + m)
    j0, m = i0 + 1_000_000, 2_000_000
    sub = beats_file_record(grade="mask", store_rel="data/A/x/x_peri_r_beats.mat",
                            sha256=_sha(p), origin_sample0=tr.origin_sample0, heartlocs=h,
                            epoch_start_sample=j0, n_samples=m, published=True,
                            read_from="gems_root:data/A/x/x_peri_r_beats.mat")
    f = tr.file_samples0
    assert sub["mask_grade_beats"].endswith("_peri_r_beats.mat")
    assert sub["n_in_epoch"] == int(((f >= j0) & (f < j0 + m)).sum()) > 0
    # with the origin taken as the file's 0, the count is wrong: the defect, measured
    wrong = beats_file_record(grade="hrv", store_rel="data/A/x/x_beats.mat", sha256=_sha(p),
                              origin_sample0=0, heartlocs=h, epoch_start_sample=i0,
                              n_samples=n, published=False, read_from="x")
    assert wrong["n_in_epoch"] < rec["n_in_epoch"]
    json.dumps(rec, allow_nan=False)


@pytest.mark.parametrize(("kw", "err"), [
    ({"grade": "x"}, ValueError), ({"store_rel": "/data/a.mat"}, ValueError),
    ({"store_rel": "C:/data/a.mat"}, ValueError), ({"store_rel": "data\\a.mat"}, ValueError),
    ({"sha256": "ABC"}, ValueError), ({"origin_sample0": 1.0}, TypeError),
    ({"origin_sample0": True}, TypeError), ({"epoch_start_sample": -1}, TypeError)])
def test_the_beats_file_record_refuses_malformed_fields(kw: dict[str, object],
                                                         err: type[Exception]) -> None:
    good: dict[str, object] = {"grade": "hrv", "store_rel": "data/A/x/x_beats.mat",
                               "sha256": "0" * 64, "origin_sample0": 0, "heartlocs": [1, 5],
                               "epoch_start_sample": 0, "n_samples": 10, "published": True,
                               "read_from": "x"}
    beats_file_record(**good)  # type: ignore[arg-type]
    with pytest.raises(err):
        beats_file_record(**{**good, **kw})  # type: ignore[arg-type]
