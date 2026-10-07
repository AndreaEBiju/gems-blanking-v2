"""Task 11: features per core - invariance, the motion ratio, missing values, inputs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.detect import chain
from gems_blanking_v2.detect import features as ft
from gems_blanking_v2.types import ChannelInfo, Recording

from tests.conftest import inject_artifact, make_eng, make_multichannel

FS = 24414.0625
DUR = 40.0
Z_ENTER = 3.0
HOST_S = 40.0
"""Long enough for every band to leave assessable frames: 0-2 Hz needs > 25 s (a 7.5 s
window plus settling at both ends)."""


def _nine(seed: int = 3) -> Recording:
    rec, _truth = make_multichannel(FS, DUR, seed=seed)
    return rec


def _five(rec: Recording) -> Recording:
    """Return the recording as the old cohort sees it: one hardware T per cuff, 3 stomach."""
    sig, _w = build_derivations(rec)
    stomach = [c for c in rec.channels if c.role == "stomach"]
    cols = [sig["L_T"], sig["R_T"]] + [rec.data[:, c.index] for c in stomach]
    chans = [ChannelInfo(0, "LVN", "nerve", "L", None, None, "hw_tripole"),
             ChannelInfo(1, "RVN", "nerve", "R", None, None, "hw_tripole")]
    chans += [ChannelInfo(2 + i, c.name, "stomach", None, None, None, "hw_tripole")
              for i, c in enumerate(stomach)]
    return Recording(fs=rec.fs, data=np.column_stack(cols), channels=chans, animal="O",
                     session="old", path=Path("old.mat"))


def _cores() -> list[tuple[float, float, tuple]]:
    return [(5.0, 5.2, (("L_T", "300-3000"),)), (12.30, 12.32, ()), (20.0, 23.0, (("L_T", "0-2"),))]


def test_the_inputs_are_exactly_what_detect_region_read() -> None:
    rec = _nine()
    region = (0.0, DUR)
    det = chain.detect_region(rec, region)
    inp = ft.region_inputs(rec, region, z_enter=Z_ENTER)
    assert set(inp.z) == set(det.z)
    for key, zz in det.z.items():
        np.testing.assert_array_equal(inp.z[key], zz)


def test_the_vector_is_the_same_length_for_5_and_9_channels() -> None:
    rec9 = _nine()
    p9 = ft.prepare(ft.region_inputs(rec9, (0.0, DUR), z_enter=Z_ENTER))
    p5 = ft.prepare(ft.region_inputs(_five(rec9), (0.0, DUR), z_enter=Z_ENTER))
    m9, m5 = ft.feature_matrix(p9, _cores()), ft.feature_matrix(p5, _cores())
    assert list(m9.columns) == list(m5.columns) == list(ft.FEATURE_NAMES)
    assert m9.shape == m5.shape == (3, len(ft.FEATURE_NAMES))


def test_scaling_every_channel_10x_changes_no_feature() -> None:
    rec = _nine()
    big = replace(rec, data=rec.data * 10.0)
    a = ft.feature_matrix(ft.prepare(ft.region_inputs(rec, (0.0, DUR), z_enter=Z_ENTER)), _cores())
    b = ft.feature_matrix(ft.prepare(ft.region_inputs(big, (0.0, DUR), z_enter=Z_ENTER)), _cores())
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True)


def test_the_old_cohort_has_its_contact_features_missing_never_zero() -> None:
    p5 = ft.prepare(ft.region_inputs(_five(_nine()), (0.0, DUR), z_enter=Z_ENTER))
    m5 = ft.feature_matrix(p5, _cores())
    for ctx in ft.CONTEXTS_S:
        sfx = f"_c{round(ctx * 1000)}"
        assert m5[f"cm_resid_max{sfx}"].isna().all()
        assert m5[f"within_minus_across{sfx}"].isna().all()
    p9 = ft.prepare(ft.region_inputs(_nine(), (0.0, DUR), z_enter=Z_ENTER))
    m9 = ft.feature_matrix(p9, _cores())
    assert m9["cm_resid_max_c0"].notna().all()
    assert m9["within_minus_across_c0"].notna().all()
    rates = ft.missing_rates(m5)
    assert rates["cm_resid_max_c0"] == 1.0 and rates["band_ratio_max_c0"] == 0.0


def _ratio_at(x: np.ndarray, t0: float, t1: float) -> float:
    inp = ft.inputs_from_signals({"L_T": x}, FS, {}, z_enter=Z_ENTER)
    return ft.core_features(ft.prepare(inp), t0, t1)["band_ratio_max_c0"]


def test_the_motion_ratio_separates_motion_from_spike_bursts_on_synthetic_data() -> None:
    rng = np.random.default_rng(11)
    motion, burst = [], []
    for k in range(16):
        host = make_eng(FS, HOST_S, rate_hz=10.0, seed=100 + k).signal
        dur = float(rng.uniform(0.05, 0.4))
        kind = ("excursion", "step", "tribo", "drift")[k % 4]
        x, span = inject_artifact(host, FS, 3.0, dur, kind, float(rng.uniform(4, 12)), seed=k)
        motion.append(_ratio_at(x, span.start_s, span.stop_s))
        b = make_eng(FS, dur, rate_hz=400.0, spike_uv=float(rng.uniform(60, 160)), noise_uv=0.0,
                     seed=500 + k).signal
        y = host.copy()
        i0 = int(3.0 * FS)
        y[i0:i0 + b.size] += b
        burst.append(_ratio_at(y, 3.0, 3.0 + dur))
    m, s = np.asarray(motion), np.asarray(burst)
    auc = float(np.mean(m[:, None] > s[None, :]) + 0.5 * np.mean(m[:, None] == s[None, :]))
    assert auc > 0.8, auc


def test_clipping_is_seen_by_the_clip_fraction() -> None:
    # A saturated amplifier: a large step driven into a rail that holds for the whole
    # channel (a rail exceeded elsewhere in the same signal would not be a rail).
    host = make_eng(FS, HOST_S, seed=7).signal
    x, span = inject_artifact(host, FS, 3.0, 0.2, "step", 40.0, seed=1)
    rail = 0.8 * span.peak_uv
    x = np.clip(x, -rail, rail)
    p = ft.prepare(ft.inputs_from_signals({"L_T": x}, FS, {}, z_enter=Z_ENTER))
    inside = ft.core_features(p, span.start_s, span.stop_s)["clip_frac_c0"]
    outside = ft.core_features(p, 1.0, 1.2)["clip_frac_c0"]
    assert inside > 0.2 and outside == 0.0


def test_one_sample_at_the_maximum_is_not_clipping() -> None:
    # the largest spike sets the channel's maximum but is not a flat run at a rail
    x = make_eng(FS, HOST_S, seed=7).signal.copy()
    x[int(3.1 * FS)] += 50 * float(np.max(np.abs(x)))
    p = ft.prepare(ft.inputs_from_signals({"L_T": x}, FS, {}, z_enter=Z_ENTER))
    assert ft.core_features(p, 3.0, 3.2)["clip_frac_c0"] == 0.0


def test_mains_lines_raise_the_line_ratio_and_locking() -> None:
    clean = make_eng(FS, HOST_S, seed=9).signal
    t = np.arange(clean.size) / FS
    hum = clean + sum(20.0 / k * np.sin(2 * np.pi * 60.0 * k * t + k) for k in range(5, 30))
    # mains-locked impulses: a 0.15 ms pulse every 1/60 s
    locked = clean.copy()
    locked[(np.round(np.arange(0, HOST_S, 1 / 60.0) * FS)).astype(int)] += 300.0
    feats = {}
    for name, x in (("clean", clean), ("hum", hum), ("locked", locked)):
        sig = {"L_T": x, "R_T": x[::-1].copy()}
        p = ft.prepare(ft.inputs_from_signals(sig, FS, {}, z_enter=Z_ENTER))
        feats[name] = ft.core_features(p, 3.0, 3.1)
    assert feats["hum"]["line_ratio_max"] > 5 * feats["clean"]["line_ratio_max"]
    assert feats["locked"]["line_plv_max"] > 3 * feats["clean"]["line_plv_max"]


def test_a_core_must_have_positive_duration() -> None:
    p = ft.prepare(ft.inputs_from_signals({"L_T": make_eng(FS, HOST_S, seed=1).signal}, FS, {},
                                          z_enter=Z_ENTER))
    with pytest.raises(ValueError, match="positive duration"):
        ft.core_features(p, 1.0, 1.0)


def test_feature_names_are_unique_and_carry_no_retired_band() -> None:
    assert len(set(ft.FEATURE_NAMES)) == len(ft.FEATURE_NAMES)
    assert ft.BANDS[ft.ENG_BAND].hi_hz == 3000.0  # ruling (b) R8: never 300-5000
