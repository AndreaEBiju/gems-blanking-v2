"""Task 15: per-consumer masks, the MATLAB file, QC gates (provenance: test_emit_provenance)."""

from __future__ import annotations

import functools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hypothesis
import numpy as np
import pytest
from gems_blanking_v2.detect import chain
from gems_blanking_v2.emit import handoff as ho
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit import qc
from gems_blanking_v2.emit.line_distrust import LineDistrustRecord
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.extent import tolerance as tl
from gems_blanking_v2.extent.grid import frame_sample_bounds, n_grid_frames, to_matlab_inclusive
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs, find_zero_runs
from gems_blanking_v2.types import Candidate, Event
from scipy.io import loadmat

from tests.conftest import make_band_z, make_line_distrust, make_mains_spike_t, peri_r_like

FS = 24414.0625
DUR_S = 60.0
N_SAMPLES = int(DUR_S * FS)
N_FRAMES = n_grid_frames(N_SAMPLES, FS)
TOL = tl.ToleranceTable({"spikes": 4.0, "velocity": 4.0, "mmc": 3.0, "slow_wave": 3.0,
                         "breathing": 3.0}, source="synthetic test table")
SIGNALS = {"spikes": ("L_T",), "slow_wave": ("ANT1",), "mmc": ("ANT1",),
           "velocity": ("L_V1", "L_V3")}
MODEL = {"mode": "pooled", "version": "0.3.0", "corpus_hash": "ab" * 16,
         "calibrator": "models/x/calibrator.json",
         "trained_at": datetime(2026, 10, 8, tzinfo=UTC), "n_train_events": 1200}
# The real gate path, with medians that let an ordinary recording through.
GATE: dict[str, Any] = {"min_retention": 0.5, "animal_median": {
    "spikes|L_T|300-3000": 0.02, "slow_wave|ANT1|0-2": 0.02, "mmc|ANT1|2-50": 0.02}}

READS: dict[str, tuple[str, ...]] = {"spikes": ("L_T",), "slow_wave": ("ANT1",),
                                     "mmc": ("ANT1",), "hrv": ("RVN2",),
                                     "breathing": ("RVN2",), "velocity": ()}
"""What the synthetic recording reads - every consumer named, as the writer requires."""


def _reads(**override: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {**READS, **override}



def _prov() -> MaskProvenance:
    return MaskProvenance(model=MODEL, thresholds={"tolerances": {"spikes": 4.0}},
                          reference_values={"L_T|300-3000": [1.2, 0.3]}, code_commit="c0ffee",
                          generation_sha="0133349b3ebeff80", routing_hash="64c2e1ea",
                          created_at="2026-10-08T05:00:00+00:00", recording="rec1")


@functools.cache
def _ld(t0: float = 0.0, n_samples: int = N_SAMPLES) -> LineDistrustRecord:
    """Return the spike consumer's line-distrust record for the synthetic recording."""
    t = make_mains_spike_t(FS, n_samples / FS, seed=21).signal[:n_samples]
    return make_line_distrust({"L_T": t}, FS, recording="rec1", epoch_start_s=t0)


def test_emit_is_outside_the_generation_hash() -> None:
    mods = chain.generation_modules()
    for m in ("masks", "qc", "provenance"):
        assert f"gems_blanking_v2.emit.{m}" not in mods
    assert "gems_blanking_v2.extent.grid" not in mods


def _eng_only(t0: float = 0.0) -> tuple[dict[Any, Any], tl.Extent]:
    """Masks for an event over the ENG tolerance only, in an epoch starting at ``t0``.

    z is on the epoch's grid (frame 0 at ``t0``); the event is on the recording's timeline.
    """
    ev = Event(Candidate(t0 + 19.5, t0 + 21.0, ("L_T",), ("300-3000",), 9.0, "electrical"),
               "motion", float("nan"), "human")
    z = {("L_T", "300-3000"): make_band_z("300-3000", DUR_S, bumps=((20.0, 20.2, 12.0, "L_T"),),
                                          signal="L_T").z_max,
         ("ANT1", "0-2"): make_band_z("0-2", DUR_S, signal="ANT1").z_max,
         ("ANT1", "2-50"): make_band_z("2-50", DUR_S, signal="ANT1").z_max}
    exts, decisions = [], []
    spike_ext = None
    for consumer, sig in (("spikes", "L_T"), ("slow_wave", "ANT1"), ("mmc", "ANT1")):
        e = tl.compute_extent(ev, z, consumer, signal=sig, tolerances=TOL, fs=FS, z_t0_s=t0)
        if e is not None:
            exts.append(("ev1", e))
            decisions.append(RouteDecision("ev1", consumer, "reject", "x", "in_band"))
            if consumer == "spikes":
                spike_ext = e
    assert spike_ext is not None
    spans = mk.spans_from_routing(exts, decisions)
    return mk.build_masks(READS, spans, n_frames=N_FRAMES, t0_s=t0), spike_ext


def test_masks_are_not_merged_across_consumers() -> None:
    masks, _ = _eng_only()
    spikes = masks[("spikes", "L_T", "300-3000")].invalid
    slow = masks[("slow_wave", "ANT1", "0-2")].invalid
    assert spikes.any() and not slow.any()
    assert not np.array_equal(spikes, slow)


def test_velocity_is_absent_unless_task_18_is_present() -> None:
    masks, _ = _eng_only()
    assert not any(k[0] == "velocity" for k in masks)
    with_v = mk.build_masks(SIGNALS, [mk.MaskSpan("velocity", "L_V1", 1.0, 2.0, "x")],
                            n_frames=N_FRAMES, t0_s=0.0, include_velocity=True)
    assert ("velocity", "L_V1", "300-3000") in with_v


def test_velocity_mask_is_the_intersection_of_v1_and_v3_validity() -> None:
    spans = [mk.MaskSpan("velocity", "L_V1", 1.0, 2.0, "x"),
             mk.MaskSpan("velocity", "L_V3", 5.0, 6.0, "x")]
    m = mk.build_masks(SIGNALS, spans, n_frames=N_FRAMES, t0_s=0.0, include_velocity=True)
    v1, v3 = m[("velocity", "L_V1", "300-3000")], m[("velocity", "L_V3", "300-3000")]
    v = mk.velocity_mask(v1, v3)
    assert np.array_equal(~v.invalid, ~v1.invalid & ~v3.invalid)
    with pytest.raises(ValueError, match="velocity"):
        mk.velocity_mask(v1, _eng_only()[0][("spikes", "L_T", "300-3000")])


def test_only_masking_routes_mask_and_a_missing_decision_raises() -> None:
    ext = tl.Extent("spikes", "L_T", "300-3000", 1.0, 2.0, 1.0, 2.0, 0.0, 0.025)
    keep = mk.spans_from_routing([("e", ext)], [RouteDecision("e", "spikes", "correct", "x")])
    assert keep == []
    with pytest.raises(KeyError, match="no routing decision"):
        mk.spans_from_routing([("e", ext)], [])


def test_cuff_distrust_joins_only_that_cuffs_spike_mask() -> None:
    entry = {"spike": {"L": {"route": "multi", "distrusted_spans": [[0.0, 60.0]]},
                       "R": {"route": "distrusted", "why": "x"},
                       "X": {"route": "multi"}}, "hr": {"none": "x"}, "stomach_ref": {}}
    spans = mk.distrusted_spike_spans(entry, region_start_s=100.0, region_stop_s=700.0)
    assert [(s.consumer, s.signal, s.start_s, s.stop_s) for s in spans] == [
        ("spikes", "L_T", 100.0, 160.0), ("spikes", "R_T", 100.0, 700.0)]
    m = mk.build_masks({"spikes": ("R_T",)}, [spans[1]], n_frames=60000, t0_s=100.0)
    assert m[("spikes", "R_T", "300-3000")].invalid.all()  # wholly distrusted
    with pytest.raises(ValueError, match="line-noise span"):  # never a blank (ruling (c))
        mk.build_masks({"spikes": ("R_T",)},
                       [mk.MaskSpan("spikes", "R_T", 0.0, 60.0, "line_noise_cuff_minute")],
                       n_frames=60000, t0_s=0.0)


def test_a_recovery_epoch_starting_at_120_s_is_shifted_exactly_once(tmp_path: Path) -> None:
    """Extents on the recording timeline, masks from t0 = 120 s, spans from the epoch start."""
    masks, ext = _eng_only(t0=120.0)
    m = masks[("spikes", "L_T", "300-3000")]
    (a, b), = mk.masked_spans_s(m)
    assert a <= ext.start_s and ext.stop_s <= b and a >= 120.0
    path = ho.write_mask_file(tmp_path / "r.mat", masks, _prov(),
                              signals=READS, fs=FS, n_samples=N_SAMPLES,
                              line_distrust=_ld(120.0), peri_r=peri_r_like(_ld(120.0)),
                              epoch_start_s=120.0, **GATE)
    spans = loadmat(path)["blank_spikes_L_T"]
    i0 = int(np.floor((ext.start_s - 120.0) / 0.01))
    assert spans[0, 0] == frame_sample_bounds(i0, i0 + 1, FS)[0] + 1
    assert spans[0, 0] < 21 * FS  # epoch-relative: about 20 s in, not 140 s
    with pytest.raises(ValueError, match="grid starts at 120"):
        ho.write_mask_file(tmp_path / "x.mat", masks, _prov(),
                           signals=READS, fs=FS, n_samples=N_SAMPLES,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           epoch_start_s=0.0, **GATE)


def test_the_frame_count_must_match_the_epoch(tmp_path: Path) -> None:
    masks, _ = _eng_only()
    with pytest.raises(ValueError, match="frames for an epoch"):
        ho.write_mask_file(tmp_path / "x.mat", masks, _prov(),
                           signals=READS, fs=FS, n_samples=N_SAMPLES // 2,
                           line_distrust=_ld(0.0, N_SAMPLES // 2),
                           peri_r=peri_r_like(_ld(0.0, N_SAMPLES // 2)),
                           epoch_start_s=0.0, **GATE)


def test_sample_mask_and_matlab_spans_agree_to_the_inclusive_end() -> None:
    """One conversion: frames 1001-1002 -> the same samples in Python and in MATLAB."""
    inv = np.zeros(N_FRAMES, dtype=bool)
    inv[1001:1003] = True
    samples = mk.frames_to_samples(inv, FS, N_SAMPLES)
    k0, k1 = frame_sample_bounds(1001, 1003, FS)
    (span,) = mk.mask_sample_spans(inv, FS, N_SAMPLES)
    assert span == (k0, k1)
    (mat,) = to_matlab_inclusive([k0], [k1])
    first, last = int(mat[0]), int(mat[1])  # 1-based inclusive
    assert np.flatnonzero(samples)[0] == first - 1 and np.flatnonzero(samples)[-1] == last - 1
    assert samples.sum() == last - first + 1
    assert (first, last) == (244386, 244873)  # round(1001 g fs) + 1, round(1003 g fs)
    with pytest.raises(ValueError, match="at least one sample"):
        to_matlab_inclusive([5], [5])


def test_one_sample_span_is_five_five() -> None:
    assert to_matlab_inclusive([4], [5]).tolist() == [[5.0, 5.0]]


def test_apply_mask_writes_nan_tapers_without_zeros_and_leaves_the_input() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(0, 10, N_SAMPLES)
    x.flags.writeable = False  # a loader's read-only view (invariant 17)
    invalid = mk.mask_frames([(10.0, 10.5)], N_FRAMES, t0_s=0.0)
    y = mk.apply_mask(x, FS, invalid)
    i0, i1 = frame_sample_bounds(1000, 1050, FS)
    assert np.isnan(y[i0:i1]).all() and np.isfinite(y[:i0]).all() and np.isfinite(y[i1:]).all()
    n = int(round(mk.TAPER_S * FS))
    w = y[i0 - n:i0] / x[i0 - n:i0]
    assert (w > 0).all() and (w < 1).all() and np.all(np.diff(w) < 0)
    assert np.array_equal(y[: i0 - n], x[: i0 - n])
    assert find_zero_runs(y) == []


def test_apply_mask_refuses_to_emit_a_zero_run() -> None:
    x = np.ones(int(FS))
    x[100:200] = 0.0
    with pytest.raises(ValueError, match="zero"):
        mk.apply_mask(x, FS, np.zeros(100, dtype=bool))


def test_the_mask_file_has_no_zero_runs_one_span_set_per_consumer_and_r6(tmp_path: Path
                                                                       ) -> None:
    masks, _ = _eng_only()
    masks[("mmc", "ANT1", "2-50")] = mk.ConsumerMask(
        "mmc", "ANT1", "2-50", mk.mask_frames([(30.0, 31.0)], N_FRAMES, t0_s=0.0), 0.01, 0.0)
    path = ho.write_mask_file(tmp_path / "rec1_masks.mat", masks, _prov(),
                              signals=READS, fs=FS,
                              line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                              n_samples=N_SAMPLES, epoch_start_s=0.0, **GATE,
                              events=[{"start": 19.5, "stop": 21.0, "judgement": "motion"}])
    m = loadmat(path)
    for name, value in m.items():
        if name.startswith("__") or value.dtype.kind in "U":
            continue
        assert_no_zero_runs(np.asarray(value, dtype=np.float64).ravel(), what=name)
    assert {"blank_spikes_L_T", "blank_slow_wave_ANT1", "blank_mmc_ANT1",
            "notmeasured_mmc_ANT1"} <= set(m)
    assert m["blank_slow_wave_ANT1"].size == 0
    nm = m["notmeasured_mmc_ANT1"]
    assert nm.shape == (1, 2)
    assert nm[0, 0] == frame_sample_bounds(1500, 1501, FS)[0] + 1  # 30 s - 15 s
    assert nm[0, 1] == frame_sample_bounds(4600, 4601, FS)[0]  # 31 s + 15 s, inclusive end
    assert json.loads(str(m["gate_json"][0]))["held"] is False
    prov = MaskProvenance.from_json(str(m["provenance_json"][0]))
    assert prov.model["mode"] == "pooled"


def test_a_mask_without_a_model_is_refused_on_write(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="model"):
        ho.write_mask_file(tmp_path / "x.mat", _eng_only()[0], None,
                           signals=READS, fs=FS,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, **GATE)
    assert not list(tmp_path.iterdir())


def test_the_event_table_carries_routes_judgement_and_model() -> None:
    ev = Event(Candidate(1.0, 2.0, ("L_T",), ("300-3000", "100-300"), 9.0, "electrical"),
               "motion", float("nan"), "human")
    rows = mk.event_rows({"e": ev}, [RouteDecision("e", "spikes", "reject", "x"),
                                     RouteDecision("e", "slow_wave", "correct", "y")], _prov())
    (row,) = rows
    assert row["routes"] == {"spikes": "reject", "slow_wave": "correct"}
    assert row["bands"] == ["300-3000", "100-300"] and "p_motion" not in row
    json.dumps(rows, allow_nan=False)


# ---------------------------------------------------------------------------
# QC and the gate it enforces
# ---------------------------------------------------------------------------


def _masked(frac: float, consumer: str = "spikes", sig: str = "L_T") -> dict[Any, Any]:
    return mk.build_masks(_reads(**{consumer: (sig,)}),
                          [mk.MaskSpan(consumer, sig, 0.0, frac * DUR_S, "x")],
                          n_frames=N_FRAMES, t0_s=0.0)


def test_the_retention_gate_fires_on_an_over_masked_recording() -> None:
    over = _masked(0.7)
    v = qc.retention_gate(over, min_retention=0.5)
    assert v.flagged and list(v.below) == ["spikes|L_T|300-3000"]
    assert not qc.retention_gate(_eng_only()[0], min_retention=0.5).flagged
    with pytest.raises(ValueError, match="min_retention"):
        qc.retention_gate(over, min_retention=0.0)


def test_a_held_recording_is_written_only_with_a_release(tmp_path: Path) -> None:
    over = _masked(0.7)
    med = {"spikes|L_T|300-3000": 0.1}
    gate = qc.emit_gate(over, min_retention=0.5, animal_median=med)
    assert gate.held and gate.retention_flagged and gate.blank_held
    with pytest.raises(ho.RecordingHeldError, match="release"):
        ho.write_mask_file(tmp_path / "x.mat", over, _prov(),
                           signals=READS, fs=FS, n_samples=N_SAMPLES,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           epoch_start_s=0.0, min_retention=0.5, animal_median=med)
    with pytest.raises(ValueError, match="blank string"):
        ho.write_mask_file(tmp_path / "x.mat", over, _prov(),
                           signals=READS, fs=FS, n_samples=N_SAMPLES,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           epoch_start_s=0.0, min_retention=0.5, animal_median=med,
                           release="   ")
    path = ho.write_mask_file(tmp_path / "x.mat", over, _prov(),
                              signals=READS, fs=FS, n_samples=N_SAMPLES,
                              line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                              epoch_start_s=0.0, min_retention=0.5, animal_median=med,
                              release="Andrea 2026-10-08: anaesthesia lightened, keep")
    g = json.loads(str(loadmat(path)["gate_json"][0]))
    assert g["min_retention"] == 0.5 and g["medians_used"] == med
    assert g["held"] and g["release"].startswith("Andrea") and g["reasons"]


def test_a_blank_over_20_percent_is_held() -> None:
    h = qc.blank_fraction_hold(_masked(0.25), {"spikes|L_T|300-3000": 0.2})
    assert h.held and any("20%" in r for r in h.reasons)


def test_a_blank_over_3x_the_animal_median_is_held() -> None:
    h = qc.blank_fraction_hold(_masked(0.10), {"spikes|L_T|300-3000": 0.02},
                               hum_features={"line_ratio_max": 0.4})
    assert h.held and any("3 x" in r for r in h.reasons)
    assert h.hum_features == {"line_ratio_max": 0.4}
    assert not qc.blank_fraction_hold(_masked(0.05), {"spikes|L_T|300-3000": 0.02}).held


def test_a_zero_animal_median_is_floored() -> None:
    """Median 0 would hold any blank at all; floored at 1%, 2% blank passes, 4% holds."""
    assert not qc.blank_fraction_hold(_masked(0.02), {"spikes|L_T|300-3000": 0.0}).held
    assert qc.blank_fraction_hold(_masked(0.04), {"spikes|L_T|300-3000": 0.0}).held


def test_hr_consumers_are_keyed_by_consumer_and_band() -> None:
    assert qc.median_key("hrv", "RVN2", "10-150") == "hrv|10-150"
    assert qc.median_key("breathing", "LVN1", "0.5-3") == "breathing|0.5-3"
    assert qc.median_key("spikes", "L_T", "300-3000") == "spikes|L_T|300-3000"
    h = qc.blank_fraction_hold(_masked(0.10, "hrv", "LVN1"), {"hrv|10-150": 0.02})
    assert h.held  # the median found under consumer|band whatever channel HR chose


def test_an_unknown_median_is_always_reported() -> None:
    h = qc.blank_fraction_hold(_masked(0.10), {"spikes|L_T|300-3000": None})
    assert not h.held and any("median unknown" in n for n in h.notes)
    h = qc.blank_fraction_hold(_masked(0.30), {})
    assert h.held and any("median unknown" in n for n in h.notes)


def test_top_reasons_count_reason_codes() -> None:
    ds = [RouteDecision(f"e{i}", "spikes", "reject", f"peak {i} sigma", "in_band_eng")
          for i in range(3)]
    h = qc.blank_fraction_hold(_masked(0.30), {}, decisions=ds)
    assert h.top_routes == (("in_band_eng", 3),)


def test_the_qc_report_leaves_missing_quantities_absent() -> None:
    masks = _eng_only()[0]
    r = qc.QcReport(recording="rec1", candidate_count=12,
                    blank_fraction_by_band=qc.blank_fraction_by_band(masks),
                    retention=qc.retention_by_key(masks), retention_flagged=False,
                    held=False, rpeak_gap_fraction=float("nan"), best_hr_channel=None,
                    mmc_not_measured_fraction={"ANT1": 0.1},
                    hold_notes=("animal median unknown for x",))
    rec = json.loads(r.to_json())
    for absent in ("rpeak_gap_fraction", "best_hr_channel", "low_confidence_velocity_windows"):
        assert absent not in rec
    assert rec["candidate_count"] == 12 and "spikes|L_T|300-3000" in rec["retention"]
    assert rec["mmc_not_measured_fraction"] == {"ANT1": 0.1} and rec["hold_notes"]


def test_the_longitudinal_table_is_one_animal_append_only(tmp_path: Path) -> None:
    row = qc.LongitudinalRow("J", "s1", "2026-10-01T03:00:00+00:00", {"L": 0.5}, {"L": 0.5},
                             {"RVN1": 18.0}, {"L_T": 2.3})
    path = tmp_path / "J_longitudinal.jsonl"
    qc.append_longitudinal(path, row)
    qc.append_longitudinal(path, row)
    raw = path.read_bytes()
    assert raw.count(b"\n") == 2 and b"\r" not in raw
    with pytest.raises(ValueError, match="another animal"):
        qc.append_longitudinal(path, qc.LongitudinalRow("H", "s1", "x", {}, {}, {}, {}))


def test_the_writer_computes_the_gate_from_the_masks(tmp_path: Path) -> None:
    """No caller-built gate: an over-masked recording is held whatever the caller wants."""
    over = _masked(0.7)
    with pytest.raises(ho.RecordingHeldError, match="retention"):
        ho.write_mask_file(tmp_path / "x.mat", over, _prov(),
                           signals=READS, fs=FS, n_samples=N_SAMPLES,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           epoch_start_s=0.0, min_retention=0.5,
                           animal_median={"spikes|L_T|300-3000": 0.9})
    path = ho.write_mask_file(tmp_path / "ok.mat", _masked(0.01), _prov(),
                              signals=READS, fs=FS,
                              line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                              n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                              animal_median={})
    g = json.loads(str(loadmat(path)["gate_json"][0]))
    assert g["held"] is False and any("median unknown" in n for n in g["notes"])


def test_the_retention_gate_alone_holds_the_recording(tmp_path: Path) -> None:
    """15% blanked: under 20% and under 3x a 10% median, but retention 0.85 < 0.9."""
    m = _masked(0.15)
    gate = qc.emit_gate(m, min_retention=0.9, animal_median={"spikes|L_T|300-3000": 0.1})
    assert gate.held and gate.retention_flagged and not gate.blank_held
    with pytest.raises(ho.RecordingHeldError, match="retention"):
        ho.write_mask_file(tmp_path / "x.mat", m, _prov(),
                           signals=_reads(spikes=('L_T',)), fs=FS, n_samples=N_SAMPLES,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           epoch_start_s=0.0, min_retention=0.9,
                           animal_median={"spikes|L_T|300-3000": 0.1})


def test_the_writer_refuses_masks_that_leave_a_consumer_out(tmp_path: Path) -> None:
    """Omitting an over-blanked consumer (or passing {}) cannot pass the gate."""
    masks = {**_masked(0.01), **_masked(0.7, "slow_wave", "ANT1")}
    reads = READS
    only_spikes = {k: v for k, v in masks.items() if k[0] == "spikes"}
    for given in (only_spikes, {}):
        with pytest.raises(ValueError, match=r"do not cover|no masks"):
            ho.write_mask_file(tmp_path / "x.mat", given, _prov(), signals=reads, fs=FS,
                               line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                               n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                               animal_median={})
    with pytest.raises(ho.RecordingHeldError):
        ho.write_mask_file(tmp_path / "x.mat", masks, _prov(), signals=reads, fs=FS,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5])
def test_a_non_finite_or_impossible_median_raises_naming_its_key(bad: float) -> None:
    with pytest.raises(ValueError, match=r"spikes\|L_T\|300-3000"):
        qc.blank_fraction_hold(_masked(0.10), {"spikes|L_T|300-3000": bad})


def test_held_recordings_carry_top_routes_and_hum_features(tmp_path: Path) -> None:
    over = _masked(0.3)
    ds = [RouteDecision("e", "spikes", "reject", "x", "in_band_eng")]
    path = ho.write_mask_file(tmp_path / "x.mat", over, _prov(), signals=READS,
                              line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                              fs=FS, n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                              animal_median={}, decisions=ds,
                              hum_features={"line_ratio_max": 0.4},
                              release="Andrea: test release")
    g = json.loads(str(loadmat(path)["gate_json"][0]))
    assert g["top_routes"] == [["in_band_eng", 1]] and g["hum_features"] == {"line_ratio_max": 0.4}


def test_the_writer_refuses_an_empty_or_partial_signals_map(tmp_path: Path) -> None:
    """Signals built from the masks is the hole: they must come from the recording."""
    with pytest.raises(ValueError, match="every consumer"):
        ho.write_mask_file(tmp_path / "x.mat", {}, _prov(), signals={}, fs=FS,
                           line_distrust=None, peri_r=peri_r_like(None),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={})
    with pytest.raises(ValueError, match="no masks"):
        ho.write_mask_file(tmp_path / "x.mat", {}, _prov(),
                           signals=dict.fromkeys(READS, ()), fs=FS, n_samples=N_SAMPLES,
                           line_distrust=None, peri_r=peri_r_like(None),
                           epoch_start_s=0.0, min_retention=0.5, animal_median={})
    masks = _masked(0.01)
    without_slow = {k: v for k, v in READS.items() if k != "slow_wave"}
    with pytest.raises(ValueError, match=r"every consumer.*slow_wave"):
        ho.write_mask_file(tmp_path / "x.mat", masks, _prov(), signals=without_slow, fs=FS,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={})


def test_a_mask_on_the_wrong_band_is_refused(tmp_path: Path) -> None:
    masks = dict(_masked(0.01))
    k = ("spikes", "L_T", "300-3000")
    m = masks.pop(k)
    masks[("spikes", "L_T", "100-300")] = mk.ConsumerMask("spikes", "L_T", "100-300", m.invalid,
                                                          m.grid_s, m.t0_s)
    with pytest.raises(ValueError, match="wrong band"):
        ho.write_mask_file(tmp_path / "x.mat", masks, _prov(), signals=READS, fs=FS,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={})


def test_a_non_finite_hum_feature_raises_naming_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="line_plv_max"):
        ho.write_mask_file(tmp_path / "x.mat", _masked(0.3), _prov(), signals=READS, fs=FS,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={}, hum_features={"line_plv_max": float("nan")},
                           release="Andrea: test")
    assert not list(tmp_path.iterdir())


def test_the_test_reads_cover_exactly_the_expected_consumers() -> None:
    assert set(READS) - set(tl.OUT_OF_BUILD_CONSUMERS) == set(tl.expected_consumers())


def test_an_unknown_consumer_is_a_value_error_naming_it(tmp_path: Path) -> None:
    masks = dict(_masked(0.01))
    k = next(iter(masks))
    odd = {**masks, ("not_a_consumer", k[1], k[2]): masks[k]}
    with pytest.raises(ValueError, match=r"unknown consumers.*not_a_consumer"):
        ho.write_mask_file(tmp_path / "x.mat", odd, _prov(), signals=READS, fs=FS,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={})
    with pytest.raises(ValueError, match=r"unknown consumers.*not_a_consumer"):
        ho.write_mask_file(tmp_path / "x.mat", masks, _prov(),
                           signals={**READS, "not_a_consumer": ("L_T",)}, fs=FS,
                           line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                           n_samples=N_SAMPLES, epoch_start_s=0.0, min_retention=0.5,
                           animal_median={})


def test_a_recording_need_not_name_a_consumer_out_of_this_build(tmp_path: Path) -> None:
    """Velocity is out (task 18, R5): signals without it is complete, not a refusal."""
    no_velocity = {k: v for k, v in READS.items() if k not in tl.OUT_OF_BUILD_CONSUMERS}
    assert "velocity" in READS and "velocity" not in no_velocity
    path = ho.write_mask_file(tmp_path / "r.mat", _masked(0.01), _prov(), signals=no_velocity,
                              line_distrust=_ld(), peri_r=peri_r_like(_ld()),
                              fs=FS, n_samples=N_SAMPLES, epoch_start_s=0.0, **GATE)
    assert path.is_file()


# --- pairs-lead HR channels in MATLAB variable names (ruling 2026-10-02 (c); invariant 22) ---

_NAME = hypothesis.strategies.from_regex(r"[A-Z]{1,4}[0-9]?|[LR]_(T|V[123])", fullmatch=True)


@hypothesis.given(_NAME, hypothesis.strategies.one_of(hypothesis.strategies.none(), _NAME))
def test_a_signal_token_round_trips_exactly(a: str, b: str | None) -> None:
    name = a if b is None else f"{a}-{b}"
    tok = ho.matlab_signal_token(name)
    assert "-" not in tok
    assert ho.signal_from_matlab_token(tok) == name


def test_a_name_that_cannot_round_trip_is_refused() -> None:
    # "A_minus-B" -> "A_minus_minus_B", which reads back as "A-minus_B": overlapping tokens
    for bad in ("A_minus_B", "A-B-C", "A_minus-B"):
        with pytest.raises(ValueError, match="unambiguous"):
            ho.matlab_signal_token(bad)


def test_a_pairs_lead_hr_channel_is_written_and_loads(tmp_path: Path) -> None:
    """33 of the 55 routing-table recordings read HR on a pairs lead such as LVN2-RVN2."""
    lead = "LVN2-RVN2"
    reads = _reads(hrv=(lead,), breathing=(lead,))
    masks = mk.build_masks(reads, [mk.MaskSpan("hrv", lead, 1.0, 2.0, "x")],
                           n_frames=N_FRAMES, t0_s=0.0)
    path = ho.write_mask_file(tmp_path / "p.mat", masks, _prov(), signals=reads, fs=FS,
                              line_distrust=_ld(),
                              peri_r=peri_r_like(_ld()), n_samples=N_SAMPLES, epoch_start_s=0.0,
                              **GATE)
    m = loadmat(str(path))
    assert "blank_hrv_LVN2_minus_RVN2" in m and "blank_breathing_LVN2_minus_RVN2" in m
    lo, hi = frame_sample_bounds(100, 200, FS)
    np.testing.assert_array_equal(m["blank_hrv_LVN2_minus_RVN2"],
                                  to_matlab_inclusive([lo], [hi]))
    assert m["blank_breathing_LVN2_minus_RVN2"].size == 0


def test_hr_not_computed_is_marked_and_refused_while_read(tmp_path: Path) -> None:
    """RULING 2026-10-08 (f) 6: no beat train -> hrv and breathing marked, never read."""
    reads = _reads(hrv=(), breathing=())
    why = "no count-gated beat train passed (routing entry hr.none)"
    nc = {"hrv": why, "breathing": why}
    masks = mk.build_masks(reads, [], n_frames=N_FRAMES, t0_s=0.0)
    path = ho.write_mask_file(tmp_path / "n.mat", masks, _prov(), signals=reads, fs=FS,
                              line_distrust=_ld(),
                              peri_r=peri_r_like(_ld()), n_samples=N_SAMPLES, epoch_start_s=0.0,
                              not_computed=nc, **GATE)
    m = loadmat(str(path))
    assert json.loads(str(m["notcomputed_json"][0])) == nc
    assert not any(k.startswith(("blank_hrv", "blank_breathing")) for k in m)
    plain = ho.write_mask_file(tmp_path / "p.mat", _masked(0.01), _prov(), signals=READS,
                               fs=FS, line_distrust=_ld(),
                               peri_r=peri_r_like(_ld()), n_samples=N_SAMPLES,
                               epoch_start_s=0.0, **GATE)
    assert json.loads(str(loadmat(str(plain))["notcomputed_json"][0])) == {}
    with pytest.raises(ValueError, match="reads nothing"):
        ho.write_mask_file(tmp_path / "x.mat", _masked(0.01), _prov(), signals=READS, fs=FS,
                           line_distrust=_ld(),
                           peri_r=peri_r_like(_ld()), n_samples=N_SAMPLES, epoch_start_s=0.0,
                           not_computed={"hrv": why}, **GATE)
    with pytest.raises(ValueError, match="known consumer and a reason"):
        ho.write_mask_file(tmp_path / "y.mat", masks, _prov(), signals=reads, fs=FS,
                           line_distrust=_ld(),
                           peri_r=peri_r_like(_ld()), n_samples=N_SAMPLES, epoch_start_s=0.0,
                           not_computed={"hrv": " "}, **GATE)


# ---------------------------------------------------------------------------
# RULING 2026-10-09 (b) 2: no-beat minutes
# ---------------------------------------------------------------------------


def _beats_except(region: tuple[float, float], gaps: tuple[tuple[float, float], ...],
                  rr: float = 0.15) -> np.ndarray:
    t = np.arange(region[0] + 0.05, region[1], rr)
    keep = np.ones(t.size, dtype=bool)
    for a, b in gaps:
        keep &= ~((t >= a) & (t < b))
    return t[keep]


def test_a_minute_without_beats_is_distrusted_whether_listed_or_not() -> None:
    """Rejected minutes (hr.blank_s) and unlisted empty minutes; nothing else.

    Windows run from the train's origin, the last one cut at the region's end.
    """
    region = (132.0, 400.0)  # windows from 132: 132-192-252-312-372-400
    beats = _beats_except(region, ((192.0, 252.0), (312.0, 372.0), (372.0, 400.0)))
    got = mk.no_beat_minutes(beat_times_s=beats, region_s=region, rejected_s=[[60.0, 120.0]])
    assert [(m.minute, m.start_s, m.stop_s, m.kind) for m in got] == [
        (1, 192.0, 252.0, "rejected"), (3, 312.0, 372.0, "empty"), (4, 372.0, 400.0, "empty")]
    # a minute with a single beat is not a no-beat minute
    one = np.sort(np.append(beats, 330.0))
    assert [m.minute for m in mk.no_beat_minutes(beat_times_s=one, region_s=region,
                                                  rejected_s=[[60.0, 120.0]])] == [1, 4]
    # a train with beats everywhere: none
    assert mk.no_beat_minutes(beat_times_s=_beats_except(region, ()), region_s=region,
                              rejected_s=[]) == []


def test_a_rejected_minute_must_be_one_window_and_hold_no_beat() -> None:
    region = (0.0, 300.0)
    beats = _beats_except(region, ((60.0, 120.0),))
    with pytest.raises(ValueError, match="not a run of per-minute storage windows"):
        mk.no_beat_minutes(beat_times_s=beats, region_s=region, rejected_s=[[90.0, 150.0]])
    with pytest.raises(ValueError, match="not a run of per-minute storage windows"):
        mk.no_beat_minutes(beat_times_s=beats, region_s=region, rejected_s=[[300.0, 360.0]])
    # merged adjacent rejected minutes, as the per-minute storage writes them
    b3 = _beats_except(region, ((60.0, 180.0),))
    got = mk.no_beat_minutes(beat_times_s=b3, region_s=region, rejected_s=[[60.0, 180.0]])
    assert [(m.minute, m.kind) for m in got] == [(1, "rejected"), (2, "rejected")]
    with pytest.raises(ValueError, match="still holds"):
        mk.no_beat_minutes(beat_times_s=beats, region_s=region, rejected_s=[[180.0, 240.0]])
    # the trailing partial window may be rejected as itself, its end stored at 1 us
    short = (0.0, 150.54)
    b2 = _beats_except(short, ((120.0, 150.54),))
    assert [m.kind for m in mk.no_beat_minutes(beat_times_s=b2, region_s=short,
                                               rejected_s=[[120.0, 150.539996]])] == ["rejected"]
    # under one sample apart (n / fs at 1 us against the 10 ms-rounded region): the same edge
    assert [m.kind for m in mk.no_beat_minutes(beat_times_s=b2, region_s=short,
                                               rejected_s=[[120.0, 150.540018]])] == ["rejected"]
    with pytest.raises(ValueError, match="not a run"):  # a frame away is not
        mk.no_beat_minutes(beat_times_s=b2, region_s=short, rejected_s=[[120.0, 150.55]])


def test_no_beat_minutes_reach_the_spike_masks_only() -> None:
    region = (0.0, 180.0)
    beats = _beats_except(region, ((60.0, 120.0),))
    mins = mk.no_beat_minutes(beat_times_s=beats, region_s=region, rejected_s=[[60.0, 120.0]])
    spans = mk.no_beat_spike_spans(mins, ("L_T", "R_T"))
    assert {(s.consumer, s.signal, s.reason) for s in spans} == {
        ("spikes", "L_T", "no_beat_minute_rejected"), ("spikes", "R_T", "no_beat_minute_rejected")}
    sig = {"spikes": ("L_T", "R_T"), "mmc": ("ANT1",), "slow_wave": ("ANT1",),
           "breathing": ("RVN2",), "hrv": ("RVN2",)}
    masks = mk.build_masks(sig, spans, n_frames=18000, t0_s=0.0)
    for key, m in masks.items():
        if key[0] == "spikes":
            assert m.invalid[6000:12000].all() and not m.invalid[:6000].any()
            assert not m.invalid[12000:].any()
        else:
            assert not m.invalid.any(), key  # invariant 2: no other consumer
