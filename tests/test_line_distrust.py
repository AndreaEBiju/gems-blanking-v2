"""RULING 2026-10-08 (d) 2: per-cuff-minute spike-consumer distrust for mains-locked spikes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
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
    make_mains_spike_contacts,
    make_mains_spike_t,
    make_spike_times,
)

FS = 24414.0625
READS: dict[str, tuple[str, ...]] = {"spikes": ("L_T",), "slow_wave": ("ANT1",),
                                     "mmc": ("ANT1",), "hrv": ("RVN2",),
                                     "breathing": ("RVN2",), "velocity": ()}
MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/calibrator.json"}


def _distrusted(rec: ld.LineDistrustRecord, sig: str) -> set[int]:
    return {r.test.minute for r in rec.minutes if r.distrusted and r.test.signal == sig}


def _status(rec: ld.LineDistrustRecord, sig: str) -> dict[int, str]:
    return {r.test.minute: r.test.status for r in rec.minutes if r.test.signal == sig}


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
            out.append(ld.MinuteTest(sig, m, 60.0 * m, 60.0 * (m + 1), "tested", 500, 40, 0.05,
                                     p))
    return tuple(out)


def test_line_distrust_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.emit.line_distrust" not in chain.generation_modules()


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------


def test_planted_locked_minutes_are_exactly_the_distrusted_ones() -> None:
    """Two cuffs, 6 minutes: a 60 Hz-locked train in some minutes, Poisson elsewhere."""
    left = make_mains_spike_t(FS, 360.0, locked_minutes=(1, 4), seed=1)
    right = make_mains_spike_t(FS, 360.0, locked_minutes=(2,), seed=2)
    rec = ld.cuff_minute_distrust({"L_T": left.signal, "R_T": right.signal}, FS,
                                  recording="rec1", epoch_start_s=0.0, family="recording")
    assert _distrusted(rec, "L_T") == {1, 4}
    assert _distrusted(rec, "R_T") == {2}
    assert set(_status(rec, "L_T").values()) == {"tested"}
    assert rec.family == {"kind": "recording", "size": 12}
    assert rec.distrusted_spans("L_T") == [(60.0, 120.0), (240.0, 300.0)]


def test_the_lock_test_is_calibrated_under_a_poisson_null_and_sees_a_locked_train() -> None:
    """Under the null, p is roughly uniform; with a locked train it is vanishing."""
    null = [ld.lock_test(make_spike_times(60.0, rate_hz=20.0, seed=s)).p for s in range(200)]
    assert 0.0 < float(np.mean(np.asarray(null) < 0.05)) <= 0.10
    assert float(np.mean(np.asarray(null) < 0.01)) <= 0.03
    locked = ld.lock_test(make_spike_times(60.0, rate_hz=20.0, locked_keep=0.2, seed=7))
    assert locked.p < 1e-12 and locked.k_locked > 100


def test_fewer_than_ten_spikes_is_untested_and_never_distrusted() -> None:
    nine = np.arange(9) / 60.0 + 1.0  # perfectly locked, but only nine spikes
    assert not ld.lock_test(nine).tested and math.isnan(ld.lock_test(nine).p)
    ten = np.arange(10) / 60.0 + 1.0
    st10 = ld.lock_test(ten)
    assert st10.tested and st10.k_locked == 9 and st10.p < 1e-6
    assert st10.p == pytest.approx(st10.p0 ** 9, rel=1e-9)  # P(X >= 9 | 9, p0): all locked
    sig = make_mains_spike_t(FS, 180.0, locked_minutes=(2,), sparse_minutes=(1,), seed=5)
    rec = ld.cuff_minute_distrust({"L_T": sig.signal}, FS, recording="r", epoch_start_s=0.0,
                                  family="recording")
    (m1,) = [r for r in rec.minutes if r.test.minute == 1]
    assert m1.test.status == "untested_few_spikes" and m1.test.n_spikes is not None
    assert m1.test.n_spikes < ld.MIN_SPIKES and not m1.distrusted
    assert rec.family["size"] == 2  # the untested minute is not in the Holm family
    with pytest.raises(ValueError, match="untested"):
        ld.MinuteResult(m1.test, distrusted=True)


def test_stim_minutes_are_never_tested() -> None:
    """Recovery epoch from 150 s: minutes 0-1 are stim (absent), minute 2 straddles."""
    full = make_mains_spike_t(FS, 360.0, locked_minutes=(0, 1, 2, 4), seed=4)
    k0 = int(round(150.0 * FS))
    view = full.signal[k0:]
    view.flags.writeable = False  # a loader's read-only view (invariant 17)
    rec = ld.cuff_minute_distrust({"L_T": view}, FS, recording="r", epoch_start_s=150.0,
                                  family="recording")
    status = _status(rec, "L_T")
    assert 0 not in status and 1 not in status
    assert status[2] == "not_assessable_epoch_edge"
    assert {m for m, s in status.items() if s == "tested"} == {3, 4, 5}
    assert _distrusted(rec, "L_T") == {4}
    (m2,) = [r.test for r in rec.minutes if r.test.minute == 2]
    assert (m2.start_s, m2.stop_s) == (150.0, 180.0)


def test_the_rule_runs_on_t_not_on_the_contacts() -> None:
    """A common-mode locked train on all three contacts cancels in T: not distrusted."""
    v1, v2, v3 = make_mains_spike_contacts(FS, 180.0, locked_minutes=(1,), seed=5)
    k0, k1 = int(round(60.0 * FS)), int(round(120.0 * FS))
    on_contact = ld.lock_test(ld.detect_spikes(v1[k0:k1], FS))  # the fixture is locked there
    assert on_contact.p < 1e-12
    rec = ld.cuff_minute_distrust({"L_T": tripole(v1, v2, v3)}, FS, recording="r",
                                  epoch_start_s=0.0, family="recording")
    assert _distrusted(rec, "L_T") == set()
    assert set(_status(rec, "L_T").values()) == {"tested"}
    for name in ("L_V1", "RVN1", "T"):
        with pytest.raises(ValueError, match="never on a contact"):
            ld.minute_tests({name: v1}, FS, epoch_start_s=0.0)


def test_non_finite_flat_and_short_minutes_are_untested() -> None:
    sig = make_mains_spike_t(FS, 145.0, seed=6).signal.copy()
    sig[int(10 * FS)] = np.nan
    sig[int(60 * FS):int(120 * FS)] = 5.0
    rec = ld.cuff_minute_distrust({"L_T": sig}, FS, recording="r", epoch_start_s=0.0,
                                  family="recording")
    assert _status(rec, "L_T") == {0: "untested_non_finite", 1: "untested_no_signal",
                                   2: "not_assessable_short"}
    assert rec.family["size"] == 0 and _distrusted(rec, "L_T") == set()


def test_holm_steps_down_and_stops() -> None:
    assert ld.holm_reject([0.001, 0.004, 0.02, 0.5], 0.01).tolist() == [True, False, False,
                                                                        False]
    assert ld.holm_reject([0.006, 0.0001, 0.001], 0.01).tolist() == [True, True, True]
    # the first failure stops it: 0.0045 <= 0.01/2 and 0.0099 <= 0.01/1, but 0.004 > 0.01/3
    assert ld.holm_reject([0.004, 0.0045, 0.0099], 0.01).tolist() == [False, False, False]
    assert ld.holm_reject([0.25, 0.9, 0.9, 0.9], 1.0)[0]  # p == alpha / M is rejected
    assert not ld.holm_reject([0.26, 0.9, 0.9, 0.9], 1.0)[0]
    with pytest.raises(ValueError, match="finite"):
        ld.holm_reject([0.1, float("nan")], 0.01)


# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------


def test_the_family_argument_is_required_and_named() -> None:
    sig = {"L_T": make_mains_spike_t(FS, 60.0, seed=8).signal}
    with pytest.raises(TypeError, match="family"):
        ld.cuff_minute_distrust(sig, FS, recording="r", epoch_start_s=0.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="family"):
        ld.decide(_rows({0: 0.001}), recording="r", fs=FS, epoch_start_s=0.0,  # type: ignore[call-arg]
                  n_samples=int(60 * FS))
    for bad in ("animal", None, {"kind": "recording"}):
        with pytest.raises(ValueError, match="family must be"):
            ld.decide(_rows({0: 0.001}), recording="r", fs=FS, epoch_start_s=0.0,
                      n_samples=int(60 * FS), family=bad)  # type: ignore[arg-type]


def test_the_family_decides_the_correction() -> None:
    """A p of 0.005 survives Holm alone, but not in an animal family of eleven."""
    mine = _rows({0: 0.005})
    rec = ld.decide(mine, recording="r1", fs=FS, epoch_start_s=0.0, n_samples=int(60 * FS),
                    family="recording")
    assert _distrusted(rec, "L_T") == {0}
    others = {f"r{i}": _rows({0: 0.5}) for i in range(2, 12)}
    table = ld.AnimalPTable.from_tests("new:A", {"r1": mine, **others})
    rec = ld.decide(mine, recording="r1", fs=FS, epoch_start_s=0.0, n_samples=int(60 * FS),
                    family=table)
    assert _distrusted(rec, "L_T") == set()
    assert rec.family == {"kind": "animal", "animal": "new:A", "size": 11,
                          "sha256": table.sha256}


def test_the_two_pass_animal_family_on_signals() -> None:
    r1 = make_mains_spike_t(FS, 180.0, locked_minutes=(1,), seed=9).signal
    r2 = make_mains_spike_t(FS, 180.0, seed=10).signal
    t1 = ld.minute_tests({"L_T": r1}, FS, epoch_start_s=0.0)
    t2 = ld.minute_tests({"L_T": r2}, FS, epoch_start_s=0.0)
    table = ld.AnimalPTable.from_tests("new:A", {"r1": t1, "r2": t2})
    assert ld.AnimalPTable.from_json(table.to_json()) == table
    rec = ld.cuff_minute_distrust({"L_T": r1}, FS, recording="r1", epoch_start_s=0.0,
                                  family=ld.AnimalPTable.from_json(table.to_json()))
    assert _distrusted(rec, "L_T") == {1} and rec.family["size"] == 6
    n = r1.size
    # a table that does not hold exactly this recording's tests refuses
    missing = ld.AnimalPTable.from_tests("new:A", {"r2": t2})
    altered = ld.AnimalPTable("new:A", {**table.pvalues, ("r1", "L_T", 0): 0.123})
    for bad in (missing, altered):
        with pytest.raises(ValueError, match="did not test the same thing"):
            ld.decide(t1, recording="r1", fs=FS, epoch_start_s=0.0, n_samples=n, family=bad)
    cleaned = ld.AnimalPTable.from_tests("new:A", {"r1": t1}, cleaner="some_cleaner")
    with pytest.raises(ValueError, match="cleaner"):
        ld.decide(t1, recording="r1", fs=FS, epoch_start_s=0.0, n_samples=n, family=cleaned)
    with pytest.raises(ValueError, match="empty"):
        ld.AnimalPTable("new:A", {})


# ---------------------------------------------------------------------------
# the cleaner hook (ruling (d) 3)
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
    before = ld.cuff_minute_distrust({"L_T": raw}, FS, recording="r", epoch_start_s=0.0,
                                     family="recording")
    assert _distrusted(before, "L_T") == {1} and "cleaner" not in before.provenance
    cleaner = _Replace("hum_bug_v5_fixed", clean)
    after = ld.cuff_minute_distrust({"L_T": raw}, FS, recording="r", epoch_start_s=0.0,
                                    family="recording", cleaner=cleaner)
    assert _distrusted(after, "L_T") == set()
    assert after.provenance["cleaner"] == "hum_bug_v5_fixed" and cleaner.saw_copy


# ---------------------------------------------------------------------------
# the record, the handoff, QC, provenance
# ---------------------------------------------------------------------------

N3 = int(round(180.0 * FS))
N3_FRAMES = n_grid_frames(N3, FS)


def _spike_mask(spans: list[tuple[float, float]]) -> dict[mk.MaskKey, mk.ConsumerMask]:
    reads = {k: v for k, v in READS.items() if k != "velocity"}
    return mk.build_masks(reads, [mk.MaskSpan("spikes", "L_T", a, b, "in_band_eng")
                                  for a, b in spans], n_frames=N3_FRAMES, t0_s=0.0)


def _rec3(recording: str = "rec1") -> ld.LineDistrustRecord:
    return ld.decide(_rows({0: 0.5, 1: 1e-9, 2: math.nan}), recording=recording, fs=FS,
                     epoch_start_s=0.0, n_samples=N3, family="recording")


def test_qc_reports_spike_time_lost_beside_the_mask() -> None:
    masks = _spike_mask([(10.0, 20.0), (70.0, 75.0)])
    lost = qc.spike_time_lost(masks, _rec3())
    row = lost["L_T"]
    epoch_s = N3_FRAMES * 0.01  # whole 10 ms frames of the 180 s epoch
    assert row["epoch_s"] == pytest.approx(epoch_s)
    assert row["mask_blank_s"] == pytest.approx(15.0)  # [10, 20) + [70, 75)
    assert row["line_distrust_s"] == pytest.approx(60.0)  # minute 1
    assert row["line_distrust_only_s"] == pytest.approx(55.0)  # minute 1 less [70, 75)
    assert row["total_lost_s"] == pytest.approx(70.0)  # [10, 20) + [60, 120)
    assert row["mask_blank_frac"] == pytest.approx(15.0 / epoch_s)
    assert row["line_distrust_frac"] == pytest.approx(60.0 / epoch_s)
    assert row["line_distrust_only_frac"] == pytest.approx(55.0 / epoch_s)
    assert row["total_lost_frac"] == pytest.approx(70.0 / epoch_s)
    assert (row["minutes_tested"], row["minutes_untested"], row["minutes_distrusted"]) == (
        2, 1, 1)
    # line distrust is not blank: the gates see the mask alone
    retention = qc.retention_by_key(masks)["spikes|L_T|300-3000"]
    assert retention == pytest.approx(1 - 15.0 / epoch_s)
    report = qc.QcReport("rec1", 0, {}, {}, retention_flagged=False, held=False,
                         spike_time_lost=lost)
    assert json.loads(report.to_json())["spike_time_lost"]["L_T"]["minutes_distrusted"] == 1
    with pytest.raises(ValueError, match="covers"):
        qc.spike_time_lost(masks, ld.decide(_rows({0: 0.5}, "R_T"), recording="rec1", fs=FS,
                                            epoch_start_s=0.0, n_samples=N3,
                                            family="recording"))


def _write(tmp_path: Path, masks: dict[mk.MaskKey, mk.ConsumerMask],
           line: ld.LineDistrustRecord | None, prov: MaskProvenance | None = None,
           signals: dict[str, tuple[str, ...]] | None = None, name: str = "r.mat") -> Path:
    return ho.write_mask_file(tmp_path / name, masks, prov or _prov(), signals=signals or READS,
                              fs=FS, n_samples=N3, epoch_start_s=0.0, min_retention=0.5,
                              animal_median={}, line_distrust=line)


def test_the_handoff_carries_the_distrust_beside_never_inside_the_mask(tmp_path: Path) -> None:
    masks = _spike_mask([(10.0, 20.0)])
    line = _rec3()
    path = _write(tmp_path, masks, line)
    m = loadmat(path)
    blank = m["blank_spikes_L_T"]
    k0, k1 = mk.mask_sample_spans(masks[("spikes", "L_T", "300-3000")].invalid, FS, N3)[0]
    assert blank.tolist() == [[k0 + 1, k1]]  # the motion span only: nothing from the rule
    d0, d1 = int(round(60.0 * FS)), int(round(120.0 * FS))
    assert m["distrust_spikes_L_T"].tolist() == to_matlab_inclusive([d0], [d1]).tolist()
    for name, value in m.items():
        if name.startswith("__") or value.dtype.kind == "U":
            continue
        assert_no_zero_runs(np.asarray(value, dtype=np.float64).ravel(), what=name)
    assert ld.LineDistrustRecord.from_json(str(m["linedistrust_json"][0])) == line
    # NaN lands on the motion frames only: the distrusted minute stays finite in the mask
    x = make_mains_spike_t(FS, 180.0, seed=13).signal
    y = mk.apply_mask(x, FS, masks[("spikes", "L_T", "300-3000")].invalid)
    assert np.isfinite(y[d0:d1]).all() and np.isnan(y[k0:k1]).all()
    assert not np.isnan(y[k1 + int(0.1 * FS):]).any()
    # and the motion mask never accepts a line-noise span
    with pytest.raises(ValueError, match="line-noise span"):
        mk.build_masks({"spikes": ("L_T",)},
                       [mk.MaskSpan("spikes", "L_T", 60.0, 120.0, "line_noise_cuff_minute")],
                       n_frames=N3_FRAMES, t0_s=0.0)


def test_the_handoff_refuses_a_missing_or_mismatched_record(tmp_path: Path) -> None:
    masks = _spike_mask([])
    with pytest.raises(ValueError, match="must carry its line-distrust record"):
        _write(tmp_path, masks, None)
    with pytest.raises(ValueError, match="covers"):
        _write(tmp_path, masks, ld.decide(_rows({0: 0.5}, "R_T"), recording="rec1", fs=FS,
                                          epoch_start_s=0.0, n_samples=N3, family="recording"))
    with pytest.raises(ValueError, match="samples from"):
        _write(tmp_path, masks, ld.decide(_rows({0: 0.5}), recording="rec1", fs=FS,
                                          epoch_start_s=60.0, n_samples=N3, family="recording"))
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


def test_provenance_records_the_rule_and_round_trips(tmp_path: Path) -> None:
    line = _rec3()
    path = _write(tmp_path, _spike_mask([]), line)
    prov = MaskProvenance.from_json(str(loadmat(path)["provenance_json"][0]))
    sld = prov.spike_line_distrust
    assert sld["rule"] == "RULING 2026-10-08 (d) item 2" and sld["alpha"] == 0.01
    assert sld["family"] == {"kind": "recording", "size": 2}
    assert sld["test_version"] == ld.TEST_VERSION and sld["consumer"] == "spikes"
    assert "cleaner" not in sld and "T" in sld["input"]
    assert MaskProvenance.from_json(prov.to_json()) == prov
    # the writer's copy and the record agree, so passing it in explicitly is accepted
    again = _write(tmp_path, _spike_mask([]), line, prov=_prov(spike_line_distrust=sld),
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
        if p is None:
            rows.append(ld.MinuteTest("R_T", m, t0 + 60.0 * m, t0 + 60.0 * m + 60.0,
                                      "untested_few_spikes", 3))
        else:
            rows.append(ld.MinuteTest("R_T", m, t0 + 60.0 * m, t0 + 60.0 * m + 60.0, "tested",
                                      200, 7, 0.04, p))
    rec = ld.decide(tuple(rows), recording="r" + chr(0x2028) + "x", fs=FS, epoch_start_s=t0,
                    n_samples=N3, family="recording", cleaner=cleaner)
    text = rec.to_json()
    assert text.isascii() and "NaN" not in text and "null" not in text
    assert ld.LineDistrustRecord.from_json(text) == rec
    assert ld.LineDistrustRecord.from_json(text).to_json() == text


def test_untested_rows_equal_their_copies_whatever_nan_object_they_hold() -> None:
    a = ld.MinuteTest("L_T", 0, 0.0, 60.0, "untested_few_spikes", 3, None, float("nan"),
                      float("nan"))
    assert a == ld.MinuteTest("L_T", 0, 0.0, 60.0, "untested_few_spikes", 3)


def test_a_record_missing_a_required_field_raises_naming_it() -> None:
    doc = json.loads(_rec3().to_json())
    del doc["provenance"]["family"]
    with pytest.raises(ValueError, match="'family'"):
        ld.LineDistrustRecord.from_json(json.dumps(doc))
    doc = json.loads(_rec3().to_json())
    doc["minutes"][0]["status"] = None
    with pytest.raises(ValueError, match="'status'"):
        ld.LineDistrustRecord.from_json(json.dumps(doc))
