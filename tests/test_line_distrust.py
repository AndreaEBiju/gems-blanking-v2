"""RULINGS 2026-10-08 (d) 2 and (e) Q1-Q5: spike-consumer distrust for mains-locked spikes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.acceptance import report as ar
from gems_blanking_v2.constants import GRID_S
from gems_blanking_v2.derive.derivations import tripole
from gems_blanking_v2.detect import chain
from gems_blanking_v2.emit import handoff as ho
from gems_blanking_v2.emit import line_distrust as ld
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit import qc
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.extent.grid import n_grid_frames, to_matlab_inclusive
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.io import loadmat

from tests.conftest import (
    all_valid_spike_mask,
    make_line_distrust,
    make_mains_spike_contacts,
    make_mains_spike_t,
    make_spike_pair,
    make_spike_times,
)

FS = 24414.0625
N60 = int(round(60.0 * FS))
READS: dict[str, tuple[str, ...]] = {"spikes": ("L_T",), "slow_wave": ("ANT1",),
                                     "mmc": ("ANT1",), "hrv": ("RVN2",),
                                     "breathing": ("RVN2",), "velocity": ()}
MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/calibrator.json"}


def _distrusted(rec: ld.LineDistrustRecord, sig: str) -> set[int]:
    return {r.test.minute for r in rec.minutes if r.distrusted and r.test.signal == sig}


def _status(rec: ld.LineDistrustRecord, sig: str) -> dict[int, str]:
    return {r.test.minute: r.test.status for r in rec.minutes if r.test.signal == sig}


def _row(rec: ld.LineDistrustRecord, sig: str, minute: int) -> ld.MinuteTest:
    (t,) = [r.test for r in rec.minutes if r.test.signal == sig and r.test.minute == minute]
    return t


def _motion(sig: str, n: int, spans: tuple[tuple[float, float], ...] = (), *,
            t0: float = 0.0) -> mk.ConsumerMask:
    frames = n_grid_frames(n, FS)
    return mk.ConsumerMask("spikes", sig, "300-3000",
                           mk.mask_frames(spans, frames, t0_s=t0), GRID_S, t0)


def _prov(recording: str = "rec1", **kw: object) -> MaskProvenance:
    base: dict[str, object] = {
        "model": MODEL, "thresholds": {"source": "test"}, "reference_values": {"none": 0},
        "code_commit": "c0ffee", "generation_sha": "0133349b3ebeff80",
        "routing_hash": "64c2e1ea", "created_at": "2026-10-08T05:00:00+00:00",
        "recording": recording}
    base.update(kw)
    return MaskProvenance(**base)  # type: ignore[arg-type]


def _rows(ps: dict[int, float], sig: str = "L_T") -> tuple[ld.MinuteTest, ...]:
    """Hand-made pass-1 rows (the decision API's input), one per minute: p or untested."""
    out = []
    for m, p in sorted(ps.items()):
        if math.isnan(p):
            out.append(ld.MinuteTest(sig, m, 60.0 * m, 60.0 * (m + 1), "untested_few_spikes", 4))
        else:
            out.append(ld.MinuteTest(sig, m, 60.0 * m, 60.0 * (m + 1), "tested", 500, 499, 40,
                                     0.05, p, 60.0))
    return tuple(out)


def _decide(rows: tuple[ld.MinuteTest, ...], *, recording: str = "rec1", n: int = N60,
            t0: float = 0.0, table: ld.AnimalPTable | None = None) -> ld.LineDistrustRecord:
    table = table or ld.AnimalPTable.from_tests("new:A", {recording: rows})
    return ld.decide(rows, recording=recording, fs=FS, epoch_start_s=t0, n_samples=n,
                     family=table)


def test_line_distrust_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.emit.line_distrust" not in chain.generation_modules()


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------


def test_planted_locked_minutes_are_exactly_the_distrusted_ones() -> None:
    left = make_mains_spike_t(FS, 360.0, locked_minutes=(1, 4), seed=1)
    right = make_mains_spike_t(FS, 360.0, locked_minutes=(2,), seed=2)
    rec = make_line_distrust({"L_T": left.signal, "R_T": right.signal}, FS, recording="rec1")
    assert _distrusted(rec, "L_T") == {1, 4} and _distrusted(rec, "R_T") == {2}
    assert set(_status(rec, "L_T").values()) == {"tested"}
    assert rec.family["kind"] == "animal_x_cohort" and rec.family["size"] == 12
    assert rec.distrusted_spans("L_T") == [(60.0, 120.0), (240.0, 300.0)]


def test_the_lock_test_is_calibrated_under_a_poisson_null_and_sees_a_locked_train() -> None:
    null = [ld.lock_test([make_spike_times(60.0, rate_hz=20.0, seed=s)]).p for s in range(200)]
    assert 0.0 < float(np.mean(np.asarray(null) < 0.05)) <= 0.10
    assert float(np.mean(np.asarray(null) < 0.01)) <= 0.03
    locked = ld.lock_test([make_spike_times(60.0, rate_hz=20.0, locked_keep=0.2, seed=7)])
    assert locked.p < 1e-12 and locked.k_locked > 100


def test_fewer_than_ten_spikes_is_untested_and_never_distrusted() -> None:
    nine = np.arange(9) / 60.0 + 1.0
    assert not ld.lock_test([nine]).tested
    ten = ld.lock_test([np.arange(10) / 60.0 + 1.0])
    assert ten.tested and (ten.m_intervals, ten.k_locked) == (9, 9)
    assert ten.p == pytest.approx(ten.p0 ** 9, rel=1e-9)  # P(X >= 9 | 9, p0)
    sig = make_mains_spike_t(FS, 180.0, locked_minutes=(2,), sparse_minutes=(1,), seed=5)
    rec = make_line_distrust({"L_T": sig.signal}, FS, recording="r")
    m1 = _row(rec, "L_T", 1)
    assert m1.status == "untested_few_spikes" and m1.n_spikes is not None
    assert m1.n_spikes < ld.MIN_SPIKES and 1 not in _distrusted(rec, "L_T")
    assert rec.family["size"] == 2
    with pytest.raises(ValueError, match="untested"):
        ld.MinuteResult(m1, distrusted=True)


def test_q3_the_boundary_minute_is_tested_when_its_in_epoch_part_reaches_30_s() -> None:
    full = make_mains_spike_t(FS, 360.0, locked_minutes=(0, 1, 2, 4), seed=4).signal
    for t0, edge, want in ((150.0, "tested", {2, 4}), (155.0, "not_assessable_short", {4})):
        view = full[int(round(t0 * FS)):]
        view.flags.writeable = False  # a loader's read-only view (invariant 17)
        rec = make_line_distrust({"L_T": view}, FS, recording="r", epoch_start_s=t0)
        status = _status(rec, "L_T")
        assert 0 not in status and 1 not in status  # stim minutes: not in the epoch at all
        assert status[2] == edge and _distrusted(rec, "L_T") == want
        assert (_row(rec, "L_T", 2).start_s, _row(rec, "L_T", 2).stop_s) == (t0, 180.0)


def test_the_rule_runs_on_t_not_on_the_contacts() -> None:
    v1, v2, v3 = make_mains_spike_contacts(FS, 180.0, locked_minutes=(1,), seed=5)
    k0, k1 = int(round(60.0 * FS)), int(round(120.0 * FS))
    assert ld.lock_test([ld.detect_spikes(v1[k0:k1], FS)]).p < 1e-12  # fixture is locked
    rec = make_line_distrust({"L_T": tripole(v1, v2, v3)}, FS, recording="r")
    assert _distrusted(rec, "L_T") == set()
    assert set(_status(rec, "L_T").values()) == {"tested"}
    for name in ("L_V1", "RVN1", "T"):
        with pytest.raises(ValueError, match="never on a contact"):
            ld.minute_tests({name: v1}, FS, epoch_start_s=0.0,
                            motion={name: _motion(name, v1.size)})


def test_q4_gaps_flat_and_short_minutes() -> None:
    sig = make_mains_spike_t(FS, 205.0, locked_minutes=(0,), seed=6).signal.copy()
    sig[int(round(20.0 * FS)) + 3:int(round(30.0 * FS)) + 3] = np.nan  # 10 s gap, off-grid
    sig[int(round(62.0 * FS)) + 3:int(round(97.0 * FS)) + 3] = np.nan  # 35 s gap
    sig[int(round(120.0 * FS)) + 1:int(round(180.0 * FS)) - 1] = 5.0  # a raw sample at each edge
    rec = make_line_distrust({"L_T": sig}, FS, recording="r")
    assert _status(rec, "L_T") == {0: "tested", 1: "untested_short_valid",
                                   2: "untested_no_signal", 3: "not_assessable_short"}
    m0, m1 = _row(rec, "L_T", 0), _row(rec, "L_T", 1)
    assert m0.valid_s == pytest.approx(50.0, abs=1e-3) and m1.valid_s == pytest.approx(25.0,
                                                                                      abs=1e-3)
    assert _distrusted(rec, "L_T") == {0}  # 50 s of finite stretches are tested
    # an interval across the gap is not an interval: two stretches, one interval each fewer
    assert m0.m_intervals == m0.n_spikes - 2


def test_q4_intervals_across_a_gap_do_not_count_and_the_rate_pools_stretches() -> None:
    p60 = 1.0 / 60.0
    a = 1.0 + np.array([0.0, 0.05, 0.12, 0.2, 0.31, 0.4])
    b = a[-1] + p60 + np.array([0.0, 0.07, 0.2, 0.26, 0.33, 0.5])  # first b is 1/60 after a
    st_ = ld.lock_test([a, b])
    assert (st_.n_spikes, st_.m_intervals, st_.k_locked) == (12, 10, 0)
    joined = ld.lock_test([np.concatenate([a, b])])
    assert (joined.m_intervals, joined.k_locked) == (11, 1)  # what the gap rule prevents
    lam = 12 / ((a[-1] - a[0]) + (b[-1] - b[0]))
    p0 = sum(math.exp(-lam * (per - 0.001)) - math.exp(-lam * (per + 0.001))
             for per in (p60, 1.0 / 30.0))
    assert st_.p0 == pytest.approx(p0, rel=1e-12)
    single = ld.lock_test([a, b, np.array([5.0])])  # a one-spike stretch: no interval, no rate
    assert single.n_spikes == 13 and single.m_intervals == 10
    assert single.p0 == pytest.approx(p0, rel=1e-12)


def test_q5_spikes_under_the_motion_mask_are_not_read() -> None:
    sig = make_mains_spike_t(FS, 180.0, locked_spans_s=((62.0, 88.0),), seed=15).signal
    plain = make_line_distrust({"L_T": sig}, FS, recording="r")
    assert _distrusted(plain, "L_T") == {1}
    motion = {"L_T": _motion("L_T", sig.size, ((61.0, 89.0),))}
    rec = make_line_distrust({"L_T": sig}, FS, recording="r", motion=motion)
    m1 = _row(rec, "L_T", 1)
    assert m1.status == "tested" and _distrusted(rec, "L_T") == set()
    assert m1.valid_s == pytest.approx(32.0, abs=0.02)
    assert m1.n_spikes is not None and m1.n_spikes < _row(plain, "L_T", 1).n_spikes  # type: ignore[operator]


def test_the_motion_masks_must_be_the_spike_consumers_own_on_this_epoch() -> None:
    x = make_mains_spike_t(FS, 60.0, seed=16).signal
    good = _motion("L_T", x.size)
    for bad in ({}, {"L_T": _motion("R_T", x.size)},
                {"L_T": mk.ConsumerMask("mmc", "L_T", "2-50", good.invalid, GRID_S, 0.0)},
                {"L_T": _motion("L_T", x.size, t0=1.0)},
                {"L_T": _motion("L_T", x.size // 2)}):
        with pytest.raises(ValueError, match="motion mask"):
            ld.minute_tests({"L_T": x}, FS, epoch_start_s=0.0, motion=bad)


def test_holm_steps_down_and_stops() -> None:
    assert ld.holm_reject([0.001, 0.004, 0.02, 0.5], 0.01).tolist() == [True, False, False,
                                                                        False]
    assert ld.holm_reject([0.006, 0.0001, 0.001], 0.01).tolist() == [True, True, True]
    assert ld.holm_reject([0.004, 0.0045, 0.0099], 0.01).tolist() == [False, False, False]
    assert ld.holm_reject([0.25, 0.9, 0.9, 0.9], 1.0)[0]  # p == alpha / M is rejected
    assert not ld.holm_reject([0.26, 0.9, 0.9, 0.9], 1.0)[0]
    with pytest.raises(ValueError, match="finite"):
        ld.holm_reject([0.1, float("nan")], 0.01)


# ---------------------------------------------------------------------------
# Q1: the family is the animal x cohort's
# ---------------------------------------------------------------------------


def test_the_family_is_required_and_is_the_animal_table() -> None:
    with pytest.raises(TypeError, match="family"):
        ld.decide(_rows({0: 0.001}), recording="r", fs=FS, epoch_start_s=0.0,  # type: ignore[call-arg]
                  n_samples=N60)
    for bad in ("recording", "animal", None):
        with pytest.raises(TypeError, match="animal x cohort"):
            ld.decide(_rows({0: 0.001}), recording="r", fs=FS, epoch_start_s=0.0,
                      n_samples=N60, family=bad)  # type: ignore[arg-type]
    for key in ("A", "new:", ":A", "new:A:x"):
        with pytest.raises(ValueError, match="cohort:animal"):
            ld.AnimalPTable(key, {})
    assert ld.AnimalPTable("old:J", {}).animal_key == "old:J"


def test_the_animal_family_decides_the_correction() -> None:
    """A p of 0.005 survives Holm in a family of one, but not in the animal's eleven."""
    mine = _rows({0: 0.005})
    assert _distrusted(_decide(mine, recording="r1"), "L_T") == {0}
    others = {f"r{i}": _rows({0: 0.5}) for i in range(2, 12)}
    table = ld.AnimalPTable.from_tests("new:A", {"r1": mine, **others})
    rec = _decide(mine, recording="r1", table=table)
    assert _distrusted(rec, "L_T") == set()
    assert rec.family == {"kind": "animal_x_cohort", "animal_key": "new:A", "size": 11,
                          "sha256": table.sha256}


def test_decide_animal_runs_every_recording_first() -> None:
    r1 = make_mains_spike_t(FS, 180.0, locked_minutes=(1,), seed=9).signal
    r2 = make_mains_spike_t(FS, 180.0, seed=10).signal
    passes = {rid: ld.pass1({"L_T": x}, FS, epoch_start_s=0.0,
                            motion={"L_T": _motion("L_T", x.size)})
              for rid, x in (("r1", r1), ("r2", r2))}
    table, recs = ld.decide_animal("new:A", passes)
    assert set(recs) == {"r1", "r2"} and len(table.pvalues) == 6
    assert _distrusted(recs["r1"], "L_T") == {1} and _distrusted(recs["r2"], "L_T") == set()
    assert recs["r1"].family == recs["r2"].family and recs["r1"].family["size"] == 6
    assert ld.AnimalPTable.from_json(table.to_json()) == table
    t1 = passes["r1"].tests
    missing = ld.AnimalPTable.from_tests("new:A", {"r2": passes["r2"].tests})
    altered = ld.AnimalPTable("new:A", {**table.pvalues, ("r1", "L_T", 0): 0.123})
    for bad in (missing, altered):
        with pytest.raises(ValueError, match="did not test the same thing"):
            _decide(t1, recording="r1", n=r1.size, table=bad)
    cleaned = ld.AnimalPTable.from_tests("new:A", {"r1": t1}, cleaner="some_cleaner")
    with pytest.raises(ValueError, match="cleaner"):
        _decide(t1, recording="r1", n=r1.size, table=cleaned)


def test_the_two_pass_match_tolerates_float_noise_only() -> None:
    mine = _rows({0: 0.001, 1: 0.5})
    table = ld.AnimalPTable.from_tests("new:A", {"r1": mine, "r2": _rows({0: 0.5})})

    def _with(scale: float) -> ld.AnimalPTable:
        pv = dict(table.pvalues)
        pv[("r1", "L_T", 0)] *= scale
        return ld.AnimalPTable("new:A", pv)

    n = math.ceil(120.0 * FS)
    assert _distrusted(_decide(mine, recording="r1", n=n, table=_with(1 + 1e-14)), "L_T") == {0}
    with pytest.raises(ValueError, match="did not test the same thing"):
        _decide(mine, recording="r1", n=n, table=_with(1 + 1e-9))


def test_decide_refuses_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        ld.decide(_rows({0: 0.5}), recording="rec1", fs=2000.0, epoch_start_s=0.0,
                  n_samples=120000, family=ld.AnimalPTable("new:A", {("rec1", "L_T", 0): 0.5}))
    with pytest.raises(ValueError, match="must hold samples"):
        _decide((), n=0)
    with pytest.raises(ValueError, match="appears twice"):
        _decide(_rows({0: 0.5}) * 2, table=ld.AnimalPTable.from_tests(
            "new:A", {"rec1": _rows({0: 0.5})}))
    with pytest.raises(ValueError, match="not inside the epoch"):
        _decide(_rows({0: 0.5, 1: 0.5}))
    with pytest.raises(ValueError, match="not inside the epoch"):
        _decide(_rows({0: 0.5}), t0=30.0)


# ---------------------------------------------------------------------------
# the cleaner hook
# ---------------------------------------------------------------------------


class _Replace:
    """Test double for a mains cleaner: returns a given signal (no cleaner is adopted)."""

    def __init__(self, name: str, clean: np.ndarray) -> None:
        self.name = name
        self.clean = clean
        self.saw_copy = False

    def __call__(self, x: np.ndarray, fs: float) -> np.ndarray:
        self.saw_copy = bool(x.flags.writeable)
        return self.clean


def test_a_cleaner_re_runs_the_test_and_minutes_regain_trust() -> None:
    raw = make_mains_spike_t(FS, 180.0, locked_minutes=(1,), seed=11).signal
    raw.flags.writeable = False
    clean = make_mains_spike_t(FS, 180.0, seed=12).signal
    before = make_line_distrust({"L_T": raw}, FS, recording="r")
    assert _distrusted(before, "L_T") == {1} and "cleaner" not in before.provenance
    cleaner = _Replace("hum_bug_v5_fixed", clean)
    p = ld.pass1({"L_T": raw}, FS, epoch_start_s=0.0, motion={"L_T": _motion("L_T", raw.size)},
                 cleaner=cleaner)
    _table, recs = ld.decide_animal("new:A", {"r": p})
    assert _distrusted(recs["r"], "L_T") == set() and cleaner.saw_copy
    assert recs["r"].provenance["cleaner"] == "hum_bug_v5_fixed"
    with pytest.raises(ValueError, match="different cleaners"):
        ld.decide_animal("new:A", {"r": p, "s": ld.pass1(
            {"L_T": raw}, FS, epoch_start_s=0.0, motion={"L_T": _motion("L_T", raw.size)})})


# ---------------------------------------------------------------------------
# Q2 / Q2b: NaN in the spike input only; the hold is motion's; the cost table
# ---------------------------------------------------------------------------

N3 = math.ceil(180.0 * FS)  # the epoch holds all three whole minutes
N3_FRAMES = n_grid_frames(N3, FS)


def _masks3(spike_spans: list[tuple[float, float]], slow: tuple[float, float] = (100.0, 101.0)
            ) -> dict[mk.MaskKey, mk.ConsumerMask]:
    reads = {k: v for k, v in READS.items() if k != "velocity"}
    spans = [mk.MaskSpan("spikes", "L_T", a, b, "in_band_eng") for a, b in spike_spans]
    spans += [mk.MaskSpan("slow_wave", "ANT1", *slow, "in_band_x"),
              mk.MaskSpan("mmc", "ANT1", *slow, "in_band_x")]
    return mk.build_masks(reads, spans, n_frames=N3_FRAMES, t0_s=0.0)


def _rec3(recording: str = "rec1", ps: dict[int, float] | None = None
          ) -> ld.LineDistrustRecord:
    return _decide(_rows(ps or {0: 0.5, 1: 1e-9, 2: math.nan}), recording=recording, n=N3)


def _write(tmp_path: Path, masks: dict[mk.MaskKey, mk.ConsumerMask],
           line: ld.LineDistrustRecord | None, prov: MaskProvenance | None = None,
           signals: dict[str, tuple[str, ...]] | None = None, name: str = "r.mat") -> Path:
    return ho.write_mask_file(tmp_path / name, masks, prov or _prov(), signals=signals or READS,
                              fs=FS, n_samples=N3, epoch_start_s=0.0, min_retention=0.5,
                              animal_median={}, line_distrust=line)


def test_qc_reports_spike_time_lost_beside_the_mask() -> None:
    masks = _masks3([(10.0, 20.0), (70.0, 75.0)])
    row = qc.spike_time_lost(masks, _rec3())["L_T"]
    epoch_s = N3_FRAMES * 0.01
    assert row["epoch_s"] == pytest.approx(epoch_s)
    assert row["mask_blank_s"] == pytest.approx(15.0)
    assert row["line_distrust_s"] == pytest.approx(60.0)
    assert row["line_distrust_only_s"] == pytest.approx(55.0)
    assert row["total_lost_s"] == pytest.approx(70.0)  # the overlap [70, 75) counted once
    assert row["line_distrust_frac"] == pytest.approx(60.0 / epoch_s)
    assert (row["minutes_tested"], row["minutes_untested"], row["minutes_distrusted"]) == (
        2, 1, 1)
    assert qc.retention_by_key(masks)["spikes|L_T|300-3000"] == pytest.approx(
        1 - 15.0 / epoch_s)


def test_q2_distrusted_minutes_are_nan_in_the_spike_input_only(tmp_path: Path) -> None:
    masks = _masks3([(10.0, 20.0), (119.5, 125.0)])
    line = _rec3()
    m = loadmat(_write(tmp_path, masks, line))
    without = loadmat(_write(tmp_path, masks, _rec3(ps={0: 0.5, 1: 0.5, 2: math.nan}),
                             name="none.mat"))
    d0, d1 = int(round(60.0 * FS)), int(round(120.0 * FS))
    motion = masks[("spikes", "L_T", "300-3000")]
    runs = mk.mask_sample_spans(motion.invalid, FS, N3)
    # blank_spikes = motion U distrust (the [119.5, 125) blank joins the distrusted minute)
    assert m["blank_spikes_L_T"].tolist() == to_matlab_inclusive(
        [runs[0][0], d0], [runs[0][1], runs[1][1]]).tolist()
    assert without["blank_spikes_L_T"].tolist() == to_matlab_inclusive(
        [a for a, _ in runs], [b for _, b in runs]).tolist()
    assert m["distrust_spikes_L_T"].tolist() == to_matlab_inclusive([d0], [d1]).tolist()
    for name in ("blank_slow_wave_ANT1", "blank_mmc_ANT1", "blank_hrv_RVN2",
                 "notmeasured_mmc_ANT1"):  # no other consumer changes (invariant 2)
        assert np.array_equal(m[name], without[name]), name
    for name, value in m.items():
        if name.startswith("__") or value.dtype.kind == "U":
            continue
        assert_no_zero_runs(np.asarray(value, dtype=np.float64).ravel(), what=name)
    assert json.loads(str(m["retention_json"][0]))["spikes|L_T|300-3000"] == pytest.approx(
        motion.retention)  # accounting stays motion's
    # the Python twin: NaN on motion and on the distrusted minute, finite and nonzero elsewhere
    x = make_mains_spike_t(FS, 180.0 + 1e-4, seed=13).signal[:N3]
    y = ld.spike_input(x, FS, motion, line)
    assert np.isnan(y[d0:d1]).all() and np.isnan(y[runs[0][0]:runs[0][1]]).all()
    lo, hi = runs[0][1] + int(0.01 * FS), d0 - int(0.01 * FS)
    assert np.array_equal(y[lo:hi], x[lo:hi]) and not (y[np.isfinite(y)] == 0).any()
    blank = np.zeros(N3, dtype=bool)
    for a, b in m["blank_spikes_L_T"].astype(int).tolist():
        blank[a - 1:b] = True
    assert np.array_equal(np.isnan(y), blank)  # Python input == what MATLAB is told
    assert line.recording == "rec1" and ld.LineDistrustRecord.from_json(
        str(m["linedistrust_json"][0])) == line
    with pytest.raises(ValueError, match="line-noise span"):
        mk.build_masks({"spikes": ("L_T",)},
                       [mk.MaskSpan("spikes", "L_T", 60.0, 120.0, "line_noise_cuff_minute")],
                       n_frames=N3_FRAMES, t0_s=0.0)


def test_q2b_distrust_never_holds_and_over_half_is_listed(tmp_path: Path) -> None:
    masks = _masks3([(10.0, 11.0)])
    heavy = _rec3(ps={0: 1e-9, 1: 1e-9, 2: 0.5})  # 120 of 180 s distrusted
    g = json.loads(str(loadmat(_write(tmp_path, masks, heavy))["gate_json"][0]))
    assert g["held"] is False and not g["reasons"]
    assert g["line_distrust_listed"] == {"L_T": pytest.approx(120.0 / (N3_FRAMES * 0.01))}
    assert g["spike_time_lost"]["L_T"]["total_lost_s"] == pytest.approx(120.0)
    light = json.loads(str(loadmat(_write(tmp_path, masks, _rec3(), name="l.mat"))
                           ["gate_json"][0]))
    assert light["line_distrust_listed"] == {}  # 60 of 180 s: reported, not listed
    assert qc.line_distrust_listed({"L_T": {"line_distrust_frac": 0.5}}) == {}  # > 50% only


def test_the_handoff_refuses_a_missing_or_mismatched_record(tmp_path: Path) -> None:
    masks = _masks3([])
    with pytest.raises(ValueError, match="must carry its line-distrust record"):
        _write(tmp_path, masks, None)
    with pytest.raises(ValueError, match="covers"):
        _write(tmp_path, masks, _decide(_rows({0: 0.5}, "R_T"), n=N3))
    with pytest.raises(ValueError, match="samples from"):
        _write(tmp_path, masks, _decide(_rows({1: 0.5}), n=N3, t0=60.0))
    with pytest.raises(ValueError, match="is for 'other'"):
        _write(tmp_path, masks, _rec3("other"))
    with pytest.raises(ProvenanceError, match="different spike line-distrust rule"):
        _write(tmp_path, masks, _rec3(), prov=_prov(spike_line_distrust={"rule": "x"}))
    no_spikes = {**READS, "spikes": ()}
    other = {k: v for k, v in masks.items() if k[0] != "spikes"}
    with pytest.raises(ProvenanceError, match="no record is carried"):
        _write(tmp_path, other, None, prov=_prov(spike_line_distrust={"rule": "x"}),
               signals=no_spikes)
    path = _write(tmp_path, other, None, signals=no_spikes, name="ok.mat")
    assert not any(k.startswith("distrust_") for k in loadmat(path))
    assert not [p for p in tmp_path.iterdir() if p.name != "ok.mat"]


def test_provenance_records_the_rules_and_round_trips(tmp_path: Path) -> None:
    line = _rec3()
    prov = MaskProvenance.from_json(str(loadmat(_write(tmp_path, _masks3([]), line))
                                        ["provenance_json"][0]))
    sld = prov.spike_line_distrust
    assert sld["rule"] == "RULING 2026-10-08 (d) item 2; (e) Q1-Q5" and sld["alpha"] == 0.01
    assert sld["test_version"] == "mains_lock_binom_v2" == ld.TEST_VERSION
    assert sld["family"]["kind"] == "animal_x_cohort" and sld["family"]["size"] == 2
    par = sld["parameters"]
    assert par["signal"] == "T" and par["min_valid_s"] == 30.0
    assert par["filter"]["span"] == "stretch" and par["filter"]["min_stretch_samples"] == 28
    assert "across a gap" in par["intervals"] and "motion" in par["motion"]
    assert par["lock_periods_s"] == [1 / 60, 1 / 30] and par["refractory_s"] == 0.001
    assert "cleaner" not in sld
    assert MaskProvenance.from_json(prov.to_json()) == prov
    again = _write(tmp_path, _masks3([]), line, prov=_prov(spike_line_distrust=sld),
                   name="again.mat")
    assert MaskProvenance.from_json(str(loadmat(again)["provenance_json"][0])) == prov


finite_p = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.one_of(st.none(), finite_p), min_size=1, max_size=8),
       st.sampled_from([None, "cleaner_x"]), st.floats(0.0, 1e4, allow_nan=False))
def test_the_record_round_trips_exactly(ps: list[float | None], cleaner: str | None,
                                        t0: float) -> None:
    rows = []
    for m, p in enumerate(ps):
        a, b = t0 + 60.0 * m, t0 + 60.0 * m + 60.0
        if p is None:
            rows.append(ld.MinuteTest("R_T", m, a, b, "untested_short_valid", valid_s=12.5))
        else:
            rows.append(ld.MinuteTest("R_T", m, a, b, "tested", 200, 198, 7, 0.04, p, 59.5))
    rid = "r" + chr(0x2028) + "x"
    table = ld.AnimalPTable.from_tests("old:J", {rid: rows}, cleaner=cleaner)
    rec = ld.decide(tuple(rows), recording=rid, fs=FS, epoch_start_s=t0,
                    n_samples=math.ceil(60.0 * len(ps) * FS) + 1, family=table, cleaner=cleaner)
    text = rec.to_json()
    assert text.isascii() and "NaN" not in text and "null" not in text
    assert ld.LineDistrustRecord.from_json(text) == rec
    assert ld.LineDistrustRecord.from_json(text).to_json() == text
    assert ld.AnimalPTable.from_json(table.to_json()) == table


def test_untested_rows_equal_their_copies_whatever_nan_object_they_hold() -> None:
    a = ld.MinuteTest("L_T", 0, 0.0, 60.0, "untested_few_spikes", 3, 2, None, float("nan"),
                      float("nan"), float("nan"))
    assert a == ld.MinuteTest("L_T", 0, 0.0, 60.0, "untested_few_spikes", 3, 2)


def test_a_record_missing_a_required_field_raises_naming_it() -> None:
    doc = json.loads(_rec3().to_json())
    del doc["provenance"]["family"]
    with pytest.raises(ValueError, match="'family'"):
        ld.LineDistrustRecord.from_json(json.dumps(doc))
    doc = json.loads(_rec3().to_json())
    doc["minutes"][0]["status"] = None
    with pytest.raises(ValueError, match="'status'"):
        ld.LineDistrustRecord.from_json(json.dumps(doc))
    doc = json.loads(_rec3().to_json())
    doc["provenance"]["parameters"]["refractory_s"] = 0.002
    with pytest.raises(ValueError, match="other test parameters"):
        ld.LineDistrustRecord.from_json(json.dumps(doc))
    doc = json.loads(_rec3().to_json())
    doc["provenance"]["family"]["kind"] = "recording"
    with pytest.raises(ValueError, match="animal_x_cohort"):
        ld.LineDistrustRecord.from_json(json.dumps(doc))


# ---------------------------------------------------------------------------
# golden statistic, refractory, raw T, cost table, motion accounting
# ---------------------------------------------------------------------------


def test_the_lock_statistic_matches_a_hand_made_train_exactly() -> None:
    """Golden values for one stretch: k, p0 and p against closed forms computed here.

    1/30 x3 and 1/60 + 0.9 ms x2 are locked; 1/60 + 1.001 ms, 1/60 + 1.5 ms x2 and
    1/60 - 1.2 ms are not; four unlocked fillers. (Exactly 1 ms off is float-fragile after
    a cumulative sum, so the boundary is probed at +/-1 us of it.)
    """
    p60, p30 = 1.0 / 60.0, 1.0 / 30.0
    locked = [p30, p30, p30, p60 + 0.0009, p60 + 0.0009]
    unlocked = [p60 + 0.001001, p60 + 0.0015, p60 + 0.0015, p60 - 0.0012,
                0.05, 0.07, 0.11, 0.2]
    t = 1.0 + np.concatenate([[0.0], np.cumsum(locked + unlocked)])
    st_ = ld.lock_test([t])
    n, m, k = t.size, t.size - 1, len(locked)
    assert (st_.n_spikes, st_.m_intervals, st_.k_locked) == (n, m, k)
    lam = n / (t[-1] - t[0])
    p0 = sum(math.exp(-lam * (per - 0.001)) - math.exp(-lam * (per + 0.001)) for per in (p60, p30))
    assert st_.p0 == pytest.approx(p0, rel=1e-12)
    p = sum(math.comb(m, j) * p0 ** j * (1 - p0) ** (m - j) for j in range(k, m + 1))
    assert st_.p == pytest.approx(p, rel=1e-9)


@pytest.mark.parametrize(("offset", "kept"), [(0, 2), (-1, 1), (1, 2)])
def test_the_refractory_keeps_a_peak_exactly_one_refractory_later(offset: int, kept: int
                                                                  ) -> None:
    gap = int(0.001 * FS)
    x, k1, k2 = make_spike_pair(FS, 60.0, gap_samples=gap + offset, seed=0)
    k = np.round(ld.detect_spikes(x, FS) * FS).astype(int)
    assert k[(k > k1 - 100) & (k < k2 + 100)].tolist() == [k1, k2][:kept]


def test_masked_t_is_refused_and_a_raw_gap_is_not() -> None:
    raw = make_mains_spike_t(FS, 120.0, seed=14).signal
    invalid = mk.mask_frames([(10.0, 10.5), (70.0, 72.0)], n_grid_frames(raw.size, FS),
                             t0_s=0.0)
    motion = {"L_T": all_valid_spike_mask("L_T", FS, raw.size)}
    with pytest.raises(ValueError, match="looks masked"):
        ld.minute_tests({"L_T": mk.apply_mask(raw, FS, invalid)}, FS, epoch_start_s=0.0,
                        motion=motion)
    gap = raw.copy()
    gap[1234567:1240000] = np.nan  # an acquisition gap: not on the 10 ms grid
    tests = ld.minute_tests({"L_T": gap}, FS, epoch_start_s=0.0, motion=motion)
    assert [t.status for t in tests] == ["tested", "tested", "not_assessable_short"]


def test_q2_another_consumer_reading_the_same_signal_keeps_its_own_spans(tmp_path: Path
                                                                         ) -> None:
    """HR on L_T here: only the spike consumer's spans take the distrusted minutes."""
    reads = {**{k: v for k, v in READS.items() if k != "velocity"},
             "hrv": ("L_T",), "breathing": ("L_T",)}
    masks = mk.build_masks(reads, [mk.MaskSpan("spikes", "L_T", 10.0, 11.0, "in_band_eng")],
                           n_frames=N3_FRAMES, t0_s=0.0)
    m = loadmat(_write(tmp_path, masks, _rec3(), signals={**reads, "velocity": ()}))
    assert m["blank_hrv_L_T"].size == 0 and m["blank_breathing_L_T"].size == 0
    assert m["blank_spikes_L_T"].shape == (2, 2)  # the motion span and the distrusted minute


def test_the_cost_table_sums_per_animal_and_cuff() -> None:
    a1 = qc.spike_time_lost(_masks3([(10.0, 20.0), (70.0, 75.0)]), _rec3())
    a2 = qc.spike_time_lost(_masks3([(0.0, 30.0)]), _rec3("rec2", {0: 0.5, 1: 0.5, 2: 0.5}))
    b1 = qc.spike_time_lost(_masks3([]), _rec3())
    table = qc.spike_time_lost_by_animal([("new:A", "rec1", a1), ("new:A", "rec2", a2),
                                          ("new:B", "rec1", b1)])
    a = table["new:A"]["L_T"]
    epoch = 2 * N3_FRAMES * 0.01
    assert a["n_recordings"] == 2 and a["epoch_s"] == pytest.approx(epoch)
    assert a["mask_blank_s"] == pytest.approx(45.0)
    assert a["line_distrust_s"] == pytest.approx(60.0)
    assert a["total_lost_s"] == pytest.approx(100.0)
    assert a["line_distrust_frac"] == pytest.approx(60.0 / epoch)
    assert (a["minutes_tested"], a["minutes_untested"], a["minutes_distrusted"]) == (5, 1, 1)
    assert table["new:B"]["L_T"]["line_distrust_s"] == pytest.approx(60.0)
    with pytest.raises(ValueError, match="given twice"):
        qc.spike_time_lost_by_animal([("new:A", "rec1", a1), ("new:A", "rec1", a1)])


def test_a_line_noise_reason_is_unknown_to_the_motion_accounting() -> None:
    with pytest.raises(ValueError, match="unknown mask reason"):
        ar.masked_motion_seconds(
            [mk.MaskSpan("spikes", "L_T", 0.0, 60.0, "line_noise_cuff_minute")], 600.0)
