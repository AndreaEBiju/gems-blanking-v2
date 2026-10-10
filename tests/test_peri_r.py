"""RULING 2026-10-08 (k) 1 and 3: peri-R NaN spans in the spike mask; step3b's guard is 0.

(k) 1, Python side (``emit.peri_r`` through ``emit.handoff.write_mask_file``):

* spans ``[R - before, R + after)`` around every beat, sample-exact widths, clipped at the
  epoch edges and merged where they overlap, in ``blank_spikes_*`` and nowhere else;
* every other consumer's file variables are identical with and without a train;
* a recording with no train: the spike spans are exactly the motion and distrust spans;
* the provenance carries the declared window file's sha256;
* the 1-based beat index is converted once: a one-sample shift is caught.
* a routed ``peri_r_ms`` / ``peri_r_narrow_ms`` outside the declared window is named
  (review fix 1: "no file is under-blanked"); a -18 ms entry against 16 ms is refused.

(k) 3, MATLAB side (Andrea's step functions, read only, through ``process_dataset_v2``):
around each R, the samples step3b leaves out of the activity RMS are exactly the peri-R NaN
span plus step2's edge pad - checked per 1 s bin, on the fixture window shaped like the
measured one and on a narrow one, where a 15 ms guard would reach past span + pad.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.emit import handoff as ho
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit import peri_r as pr
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.extent.grid import n_grid_frames, seconds_to_sample
from gems_blanking_v2.extent.tolerance import EDGE_SETTLING
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.io import loadmat, savemat

from tests.conftest import (
    TEST_PERI_R_WINDOW,
    make_line_distrust,
    make_mains_spike_t,
    peri_r_like,
)
from tests.test_matlab_acceptance import _matlab, _processing_new

FS = 24414.0625
N = int(60.0 * FS) + 37
READS: dict[str, tuple[str, ...]] = {"spikes": ("L_T", "R_T"), "slow_wave": ("ANT1",),
                                     "mmc": ("ANT1",), "hrv": ("RVN2",),
                                     "breathing": ("RVN2",), "velocity": ()}
MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/calibrator.json"}
GATE: dict[str, Any] = {"min_retention": 0.5, "animal_median": {}}
NB, NA = round(16.0e-3 * FS), round(9.5e-3 * FS)  # independent of the module: 391, 232
HARNESS = Path(__file__).parent / "matlab"
NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"


def _window_file(tmp: Path, before: float = 16.0, after: float = 9.5) -> Path:
    p = tmp / "perir_window.json"
    p.write_text(json.dumps({"schema": "perir_window/1",
                             "constant_window": {"before_ms": before, "after_ms": after}}),
                 encoding="utf-8", newline="\n")
    return p


def _prov() -> MaskProvenance:
    return MaskProvenance(model=MODEL, thresholds={"tolerances": {"spikes": 4.0}},
                          reference_values={"L_T|300-3000": [1.2, 0.3]}, code_commit="c0ffee",
                          generation_sha="0133349b3ebeff80", routing_hash="64c2e1ea",
                          created_at="2026-10-09T05:00:00+00:00", recording="rec1")


def _masks() -> dict[mk.MaskKey, mk.ConsumerMask]:
    spans = [mk.MaskSpan("spikes", "L_T", 10.003, 10.4, "in_band"),
             mk.MaskSpan("slow_wave", "ANT1", 20.0, 22.0, "in_band"),
             mk.MaskSpan("mmc", "ANT1", 40.0, 41.0, "in_band"),
             mk.MaskSpan("hrv", "RVN2", 30.0, 30.5, "in_band"),
             mk.MaskSpan("breathing", "RVN2", 30.0, 31.5, "in_band")]
    return mk.build_masks(READS, spans, n_frames=n_grid_frames(N, FS), t0_s=0.0)


_LINE: dict[str, Any] = {}


def _line() -> Any:  # noqa: ANN401 - LineDistrustRecord
    if "x" not in _LINE:
        # L_T's first minute is mains-locked, so the record distrusts it: the spike spans
        # are then motion + distrust + peri-R, three sources the writer must all keep
        raw = {s: make_mains_spike_t(FS, N / FS, locked_minutes=(0,) if s == "L_T" else (),
                                     seed=31 + k).signal[:N]
               for k, s in enumerate(READS["spikes"])}
        _LINE["x"] = make_line_distrust(raw, FS, recording="rec1")
        assert _LINE["x"].sample_spans("L_T"), "the fixture must distrust a minute of L_T"
    return _LINE["x"]


# Beats (1-based heartlocs, origin 0): a regular train, one beat whose span is cut by the
# epoch start, a pair 8 ms apart (their spans merge), and one beat past the epoch end whose
# span still reaches into it.
REGULAR = np.arange(5000, N - 5000, 4100, dtype=np.int64)
HEARTLOCS = np.unique(np.concatenate([[101], REGULAR, [700_001, 700_196], [N + 101]]))


def _expected(beats_1based: np.ndarray, n: int, *, origin: int = 0, start: int = 0) -> np.ndarray:
    """Boolean samples blanked by the peri-R spans, built here without emit.peri_r."""
    out = np.zeros(n, dtype=bool)
    for h in beats_1based:
        r = int(h) - 1 + origin - start
        out[max(0, r - NB):max(0, min(n, r + NA))] = True
    return out


def _rows(b: np.ndarray) -> np.ndarray:
    """1-based inclusive MATLAB rows of a boolean sample mask."""
    d = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    k0, k1 = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return np.column_stack([k0 + 1, k1]).astype(np.float64).reshape(-1, 2)


def _bool(rows: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    for a, b in np.asarray(rows, dtype=np.int64).reshape(-1, 2):
        out[a - 1:b] = True
    return out


def _write(tmp: Path, name: str, peri: pr.PeriRRecord | None) -> dict[str, Any]:
    nobeat = [] if peri is not None and peri.train is not None else None  # (g) 1: none here
    path = ho.write_mask_file(tmp / name, _masks(), _prov(), signals=READS, fs=FS, n_samples=N,
                              epoch_start_s=0.0, line_distrust=_line(), peri_r=peri,
                              no_beat_minutes=nobeat, **GATE)
    return loadmat(path)


def _record(tmp: Path, heartlocs: np.ndarray | None, window: Path | None = None,
            origin: int = 0) -> pr.PeriRRecord:
    w = pr.load_peri_r_window(window or _window_file(tmp))
    rec = peri_r_like(_line(), heartlocs=heartlocs, origin_sample=origin, window=w)
    assert rec is not None
    return rec


def _motion_and_distrust(signal: str) -> np.ndarray:
    m = _masks()[("spikes", signal, "300-3000")]
    out = mk.frames_to_samples(m.invalid, FS, N)
    for a, b in _line().sample_spans(signal):
        out[a:b] = True
    return out


# ---------------------------------------------------------------- (k) 1: the spans

def test_spans_surround_every_beat_with_the_exact_widths(tmp_path: Path) -> None:
    """[R - 391, R + 232) per beat at 24414.0625 Hz; clipped, merged; on both spike signals."""
    m = _write(tmp_path, "a.mat", _record(tmp_path, HEARTLOCS))
    peri = _expected(HEARTLOCS, N)
    assert np.array_equal(m["perir_spikes_L_T"], _rows(peri))
    assert np.array_equal(m["perir_spikes_R_T"], _rows(peri))
    for sig in READS["spikes"]:
        assert np.array_equal(m[f"blank_spikes_{sig}"], _rows(peri | _motion_and_distrust(sig)))
    rows = m["perir_spikes_L_T"].astype(np.int64)
    widths = rows[:, 1] - rows[:, 0] + 1
    regular = [(int(h) - NB, int(h) - 1 + NA) for h in REGULAR]  # 1-based inclusive, by hand
    assert set(regular) <= {tuple(r) for r in rows.tolist()}
    assert np.count_nonzero(widths == NB + NA) == len(regular)
    assert rows[0].tolist() == [1, 101 - 1 + NA]                        # cut at the start
    assert [700_001 - NB, 700_196 - 1 + NA] in rows.tolist()           # merged pair
    assert rows[-1].tolist() == [N + 101 - NB, N]                       # beat past the end
    gate = json.loads(str(m["gate_json"][0]))
    assert gate["spike_peri_r_fraction"]["L_T"] == pytest.approx(peri.sum() / N)
    retention = json.loads(str(m["retention_json"][0]))
    assert retention["spikes|L_T|300-3000"] == _masks()[("spikes", "L_T", "300-3000")].retention


def test_other_consumers_are_identical_with_and_without_a_train(tmp_path: Path) -> None:
    """Invariant 2: every non-spike variable is the same array, byte for byte."""
    with_train = _write(tmp_path, "a.mat", _record(tmp_path, HEARTLOCS))
    no_train = _write(tmp_path, "b.mat", _record(tmp_path, None))
    others = [k for k in with_train if k.startswith(("blank_", "notmeasured_"))
              and not k.startswith("blank_spikes_")]
    assert sorted(others) == ["blank_breathing_RVN2", "blank_hrv_RVN2", "blank_mmc_ANT1",
                              "blank_slow_wave_ANT1", "notmeasured_mmc_ANT1"]
    for k in [*others, "retention_json", "events_json", "notcomputed_json", "fs", "nSamples"]:
        a, b = with_train[k], no_train[k]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
    for sig in READS["spikes"]:  # the spike spans did change, and only by adding
        wt, nt = (_bool(m[f"blank_spikes_{sig}"], N) for m in (with_train, no_train))
        assert not (nt & ~wt).any()
    assert (wt & ~nt).sum() > 0  # R_T: not covered by a distrusted minute


def test_a_recording_with_no_train_keeps_its_spike_mask(tmp_path: Path) -> None:
    """No train: blank_spikes is exactly motion + distrust; the record says why."""
    m = _write(tmp_path, "b.mat", _record(tmp_path, None))
    for sig in READS["spikes"]:
        assert np.array_equal(m[f"blank_spikes_{sig}"], _rows(_motion_and_distrust(sig)))
        assert m[f"perir_spikes_{sig}"].size == 0
    prov = json.loads(str(m["provenance_json"][0]))["spike_peri_r"]
    assert prov["train"].startswith("none:")
    assert prov["n_spans"] == 0 and prov["samples_blanked"] == 0


def test_the_provenance_carries_the_window_files_hash(tmp_path: Path) -> None:
    """The window is a declared input: its file's sha256 and name travel into the mask."""
    wf = _window_file(tmp_path)
    m = _write(tmp_path, "a.mat", _record(tmp_path, HEARTLOCS, wf))
    prov = MaskProvenance.from_json(str(m["provenance_json"][0]))
    win = prov.spike_peri_r["window"]
    assert win == {"before_ms": 16.0, "after_ms": 9.5, "file": "perir_window.json",
                   "sha256": hashlib.sha256(wf.read_bytes()).hexdigest()}
    assert prov.spike_peri_r["window_samples"] == {"before": NB, "after": NA}
    assert prov.spike_peri_r["train"] == {"file": "synthetic", "grade": "hrv"}
    assert json.loads(str(m["perir_json"][0]))["window"] == win
    # another file, another hash; and a stale provenance naming another window is refused
    (tmp_path / "x").mkdir()
    rec2 = _record(tmp_path, HEARTLOCS, _window_file(tmp_path / "x", after=9.0))
    assert rec2.window.sha256 != win["sha256"]
    with pytest.raises(ProvenanceError, match="different peri-R"):  # prov names the first
        ho.write_mask_file(tmp_path / "c.mat", _masks(), prov, signals=READS, fs=FS,
                           n_samples=N, epoch_start_s=0.0, line_distrust=_line(),
                           peri_r=rec2, **GATE)


def test_a_one_sample_shift_is_caught(tmp_path: Path) -> None:
    """Beat h (1-based), origin o, epoch at i0: r = h - 1 + o - i0, row [r - NB + 1, r + NA]."""
    assert pr.beat_epoch_samples([1, 2], origin_sample=0, epoch_start_sample=0).tolist() == [0, 1]
    assert pr.beat_epoch_samples([10], origin_sample=7, epoch_start_sample=3).tolist() == [13]
    h = np.array([50_000, 60_000], dtype=np.int64)
    origin = seconds_to_sample(1.0, FS)  # a train detected on a region starting at 1 s
    m = _write(tmp_path, "s.mat", _record(tmp_path, h, origin=origin))
    want = [[int(x) - 1 + origin - NB + 1, int(x) - 1 + origin + NA] for x in h]
    assert m["perir_spikes_L_T"].astype(np.int64).tolist() == want
    assert np.array_equal(_bool(m["perir_spikes_L_T"], N), _expected(h, N, origin=origin))


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(-2000, 12_000), max_size=40), st.integers(-50, 400),
       st.integers(-50, 400), st.integers(1, 10_000))
def test_spans_equal_the_boolean_union(r: list[int], nb: int, na: int, n: int) -> None:
    """Property: merged, clipped spans cover exactly the union of [r - nb, r + na) in [0, n)."""
    if nb + na <= 0:
        return
    w = pr.PeriRWindow(nb / FS * 1000.0, na / FS * 1000.0, "w.json", "0" * 64)
    assert w.samples(FS) == (nb, na)
    spans = pr.peri_r_sample_spans(r, w, fs=FS, n_samples=n)
    got = np.zeros(n, dtype=bool)
    for a, b in spans:
        got[a:b] = True
    want = np.zeros(n, dtype=bool)
    for x in r:
        want[max(0, x - nb):max(0, min(n, x + na))] = True
    assert np.array_equal(got, want)
    assert all(a < b for a, b in spans)
    assert all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))  # disjoint


# ---------------------------------------------------------------- refusals

def test_the_window_file_is_validated(tmp_path: Path) -> None:
    p = tmp_path / "w.json"
    for doc, msg in [({"schema": "x", "constant_window": {"before_ms": 1, "after_ms": 1}},
                      "schema"),
                     ({"schema": "perir_window/1"}, "constant_window"),
                     ({"schema": "perir_window/1", "constant_window": {"before_ms": 1}},
                      "after_ms"),
                     ({"schema": "perir_window/1",
                       "constant_window": {"before_ms": 5, "after_ms": -5}}, "holds no time"),
                     ({"schema": "perir_window/1",
                       "constant_window": {"before_ms": "16", "after_ms": 9.5}}, "finite")]:
        p.write_text(json.dumps(doc), encoding="utf-8", newline="\n")
        with pytest.raises(ValueError, match=msg):
            pr.load_peri_r_window(p)


def test_the_writer_refuses_a_missing_or_mismatched_record(tmp_path: Path) -> None:
    good = _record(tmp_path, HEARTLOCS)
    with pytest.raises(ValueError, match="peri-R record"):
        _write(tmp_path, "x.mat", None)
    one = pr.build_peri_r(recording="rec1", signals=("L_T",), window=good.window, fs=FS,
                          n_samples=N, epoch_start_s=0.0, epoch_start_sample=0, heartlocs=None,
                          origin_sample=None, train=None)
    with pytest.raises(ValueError, match="covers"):
        _write(tmp_path, "x.mat", one)
    other = pr.build_peri_r(recording="rec2", signals=READS["spikes"], window=good.window,
                            fs=FS, n_samples=N, epoch_start_s=0.0, epoch_start_sample=0,
                            heartlocs=None, origin_sample=None, train=None)
    with pytest.raises(ValueError, match="rec2"):
        _write(tmp_path, "x.mat", other)
    short = pr.build_peri_r(recording="rec1", signals=READS["spikes"], window=good.window,
                            fs=FS, n_samples=N - 1, epoch_start_s=0.0, epoch_start_sample=0,
                            heartlocs=None, origin_sample=None, train=None)
    with pytest.raises(ValueError, match="samples"):
        _write(tmp_path, "x.mat", short)
    with pytest.raises(ValueError, match="epoch_start_sample"):
        pr.build_peri_r(recording="rec1", signals=READS["spikes"], window=good.window, fs=FS,
                        n_samples=N, epoch_start_s=1.0, epoch_start_sample=24413,
                        heartlocs=None, origin_sample=None, train=None)
    with pytest.raises(ValueError, match="together"):
        pr.build_peri_r(recording="rec1", signals=READS["spikes"], window=good.window, fs=FS,
                        n_samples=N, epoch_start_s=0.0, epoch_start_sample=0,
                        heartlocs=HEARTLOCS, origin_sample=None, train=None)
    with pytest.raises(ValueError, match="without a routed train"):
        pr.PeriRRecord("rec1", ("L_T",), good.window, FS, N, 0.0, 0, ((0, 5),), None)
    reads_none = {**READS, "spikes": ()}
    masks = {k: v for k, v in _masks().items() if k[0] != "spikes"}
    with pytest.raises(ValueError, match="covers"):
        ho.write_mask_file(tmp_path / "y.mat", masks, _prov(), signals=reads_none, fs=FS,
                           n_samples=N, epoch_start_s=0.0, line_distrust=None, peri_r=good,
                           **GATE)


# ---------------------------------------------------------------- (k) 3: step3b's guard

V2_DUR_S = 20.0
RR_S = 0.17
WINDOWS = {"measured_shape": (16.0, 9.5), "narrow": (2.0, 3.0)}
"""The fixture shaped like the measured window, and a narrow one Andrea could choose: a
+/-15 ms guard reaches past span + 10.5 ms pad only when a side is under 4.5 ms."""


def _guard_case(tmp: Path) -> dict[str, Any]:
    n = int(V2_DUR_S * FS)
    beats = np.round(np.arange(0.11, V2_DUR_S - 0.05, RR_S) * FS).astype(np.int64) + 1  # 1-based
    pad = seconds_to_sample(EDGE_SETTLING["spikes"].pad_s, FS)  # v2's P.edgeBufferMs, (c) 1
    cases: dict[str, Any] = {}
    for name, (before, after) in WINDOWS.items():
        w = pr.PeriRWindow(before, after, f"{name}.json", "0" * 64)
        spans = pr.peri_r_sample_spans(beats - 1, w, fs=FS, n_samples=n)
        # noise only (spike amplitude 0): overlapping spikes trip step3b's own 15-sigma
        # excursion guard (+/-5 ms), an exclusion unrelated to the cardiac guard - with
        # 25 uV spikes one bin in 20 s lost 247 samples to it. Without spikes the only
        # exclusions are the NaN spans, their pad, and the guard under test.
        y = np.column_stack([make_mains_spike_t(FS, V2_DUR_S, amp_uv=0.0,
                                                seed=41 + k).signal[:n] * 1e-6
                             for k in range(2)])
        for a, b in spans:
            y[a:b, :] = np.nan
        f = tmp / f"guard_{name}.mat"
        savemat(f, {"y": y, "fs": FS, "rpeakSamples": beats.astype(np.float64)})
        nb, na = w.samples(FS)
        excluded = np.zeros(n, dtype=bool)
        for r in beats - 1:
            excluded[max(0, r - nb - pad):min(n, r + na + pad)] = True
        cases[name] = {"file": f.as_posix(), "excluded": excluded}
    return cases


def test_step3b_excludes_exactly_the_peri_r_span_plus_the_edge_pad(tmp_path: Path) -> None:
    """(k) 3: with v2's guard of 0, step3b drops exactly span + pad around each R, per 1 s bin."""
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    cases = _guard_case(tmp_path)
    case_file, res_file = tmp_path / "case.json", tmp_path / "result.json"
    case_file.write_text(json.dumps({"labels": ["L_T", "R_T"],
                                     "cases": {k: v["file"] for k, v in cases.items()}}),
                         encoding="utf-8", newline="\n")
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); "
           f"check_perir_guard('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res = json.loads(res_file.read_text(encoding="utf-8"))
    bad: dict[str, Any] = {}
    for name, case in cases.items():
        r = res[name]
        assert r["error"] == "", (name, r["error"])
        assert r["edge_buffer_ms"] == 10.5, name  # RULING 2026-10-09 (c) 1
        n_bin = int(r["bin_n"])
        exc = case["excluded"]
        n = exc.size
        lens = [min(n, i + n_bin) - i for i in range(0, n, n_bin)]
        want = [int(exc[i:i + n_bin].sum()) for i in range(0, n, n_bin)]
        for k, vfrac in enumerate(r["valid_frac"]):
            got = [int(round((1.0 - v) * ln)) for v, ln in zip(np.atleast_1d(vfrac), lens,
                                                                 strict=True)]
            diff = [(i, g, w) for i, (g, w) in enumerate(zip(got, want, strict=True)) if g != w]
            if diff:
                bad[f"{name}/{k}"] = diff[:3]
    # the exclusions first, so a reverted guard fails on what it does, not on its value
    assert not bad, f"step3b excluded other samples than span + pad: {bad}"
    assert {name: res[name]["guard_ms"] for name in cases} == dict.fromkeys(cases, 0)


# ------------------------------------------- review fix 1: routed window inside the declared


def _entry(**cuffs: dict[str, Any]) -> dict[str, Any]:
    return {"spike": cuffs, "hr": {"channel": "LVN2-RVN2", "detector": "x", "n_beats": 1},
            "stomach_ref": {}}


def test_a_routed_window_wider_than_the_declared_one_is_named() -> None:
    """A -18 ms routed peri_r_ms against the declared 16 ms is outside; so is each side."""
    w = TEST_PERI_R_WINDOW  # [-16.0, 9.5] ms
    assert pr.routed_outside_window(_entry(L={"route": "multi", "peri_r_ms": [-18.0, 5.0],
                                               "peri_r_beats": "hrv"}), w) == [
        "L peri_r_ms [-18.0, 5.0]"]
    assert pr.routed_outside_window(_entry(R={"route": "multi", "peri_r_ms": [-4.0, 9.75],
                                               "peri_r_beats": "hrv"}), w) == [
        "R peri_r_ms [-4.0, 9.75]"]
    narrow = {"route": "multi", "peri_r_narrow_ms": [[-3.0, 2.0], [-16.5, -12.0]],
              "peri_r_beats": "mask"}
    assert pr.routed_outside_window(_entry(L=narrow), w) == ["L peri_r_narrow_ms [-16.5, -12.0]"]


def test_routed_windows_inside_the_declared_one_pass_edges_included() -> None:
    """The hull's own setters (-16.0 and 9.5 exactly) are inside; a cuff without one is too."""
    w = TEST_PERI_R_WINDOW
    e = _entry(L={"route": "multi", "peri_r_ms": [-16.0, 9.5], "peri_r_beats": "hrv",
                  "peri_r_narrow_ms": [[-16.0, -10.0], [2.0, 9.5]]},
               R={"route": "scalar"})
    assert pr.routed_outside_window(e, w) == []
    assert pr.routed_outside_window(_entry(), w) == []


# ---------------------------------------------------------------------------
# RULING 2026-10-09 (b) 1: the window per recording class
# ---------------------------------------------------------------------------

EXC = "gems_a_t02_2_3_bl_215610_20260925T015614Z"
OWN = "gems_x_t01_bl_000000_20260901T000000Z"


def _table_file(tmp_path: Path, **over: object) -> Path:
    doc: dict[str, object] = {
        "schema": pr.WINDOW_SCHEMA_V2,
        "default_window": {"before_ms": 11.5, "after_ms": 9.5},
        "recording_windows": {
            EXC: {"before_ms": 16.0, "after_ms": 9.5, "class": "ruled_exception",
                  "why": "RULING 2026-10-09 (b) 1, both cuffs"},
            OWN: {"before_ms": 13.0, "after_ms": 9.5, "class": "own_routed_extent",
                  "why": "routed L peri_r_ms [-13.0, 0.0] exceeds the default"}}}
    doc.update(over)
    f = tmp_path / "perir_window.json"
    f.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return f


def _entry_ab(a: float, b: float) -> dict[str, object]:
    return {"spike": {"L": {"route": "scalar", "peri_r_ms": [a, b]}, "R": {"route": "scalar"}}}


def test_each_recording_gets_the_window_of_its_class(tmp_path: Path) -> None:
    f = _table_file(tmp_path)
    t = pr.load_peri_r_windows(f)
    assert t.sha256 == hashlib.sha256(f.read_bytes()).hexdigest() and t.source == f.name
    d = t.for_recording("gems_b_bl_x")
    assert (d.before_ms, d.after_ms, d.window_class) == (11.5, 9.5, "default")
    e = t.for_recording(EXC)
    assert (e.before_ms, e.after_ms, e.window_class) == (16.0, 9.5, "ruled_exception")
    o = t.for_recording(OWN)
    assert (o.before_ms, o.after_ms, o.window_class) == (13.0, 9.5, "own_routed_extent")
    assert e.provenance()["class"] == "ruled_exception" and e.provenance()["recording"] == EXC
    assert e.provenance()["rule"] == pr.PERI_R_RULE_BY_CLASS
    assert t.provenance()["listed_by_class"] == {"ruled_exception": 1, "own_routed_extent": 1}
    assert t.provenance()["sha256"] == t.sha256
    # the refusal checks a recording against the window that applies to IT
    wide = _entry_ab(-16.0, -2.0)
    assert pr.routed_outside_window(wide, t.for_recording("gems_b_bl_x")) == [
        "L peri_r_ms [-16.0, -2.0]"]
    assert pr.routed_outside_window(wide, t.for_recording(EXC)) == []
    assert pr.routed_outside_window(_entry_ab(-13.0, 0.0), t.for_recording(OWN)) == []
    assert pr.routed_outside_window(_entry_ab(-13.5, 0.0), t.for_recording(OWN)) != []
    assert pr.routed_outside_window(_entry_ab(-11.5, 9.5), d) == []
    # a span record built on a class window carries the class rule
    rec = pr.build_peri_r(recording=EXC, signals=("L_T",), window=e, fs=FS, n_samples=10_000,
                          epoch_start_s=0.0, epoch_start_sample=0, heartlocs=[5000],
                          origin_sample=0, train={"sha256": "x"})
    nb, na = e.samples(FS)
    assert rec.spans == ((4999 - nb, 4999 + na),)
    assert rec.provenance()["rule"] == pr.PERI_R_RULE_BY_CLASS


def test_a_malformed_window_table_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        pr.load_peri_r_windows(_table_file(tmp_path, schema=pr.WINDOW_SCHEMA))
    with pytest.raises(ValueError, match="default_window"):
        pr.load_peri_r_windows(_table_file(tmp_path, default_window={"before_ms": 11.5}))
    with pytest.raises(ValueError, match="recording_windows"):
        pr.load_peri_r_windows(_table_file(tmp_path, recording_windows=None))
    for bad, what in (({"before_ms": 12.0, "after_ms": 9.5, "class": "default", "why": "x"},
                       "class"),
                      ({"before_ms": 12.0, "after_ms": 9.5, "class": "widest", "why": "x"},
                       "class"),
                      ({"before_ms": 12.0, "after_ms": 9.5, "class": "own_routed_extent"},
                       "why")):
        with pytest.raises(ValueError, match=what):
            pr.load_peri_r_windows(_table_file(tmp_path, recording_windows={OWN: bad}))
