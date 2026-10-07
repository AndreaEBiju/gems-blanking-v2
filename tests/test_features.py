"""Task 11: features per core.

The common signal set, channel-count and scale invariance, the motion ratio, missing
values, inputs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.detect import chain
from gems_blanking_v2.detect import features as ft
from gems_blanking_v2.types import ChannelInfo, Recording
from hypothesis import given
from hypothesis import strategies as st

from tests.conftest import inject_artifact, make_eng, make_multichannel, make_signal_set

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


def test_the_inputs_are_detections_own_z_on_the_common_set() -> None:
    rec = _nine()
    region = (0.0, DUR)
    det = chain.detect_region(rec, region)
    inp = ft.region_inputs(rec, region, z_enter=Z_ENTER)
    assert set(inp.signals) == {"L_T", "R_T", "ANT1", "ANT2", "ANT3"}
    assert set(inp.z) == {q for q in det.z if q[0] in inp.signals}
    assert set(inp.z) < set(det.z)  # contacts and stomach_ref are detection inputs only
    for key, zz in inp.z.items():
        np.testing.assert_array_equal(zz, det.z[key])


def test_a_signal_outside_the_common_set_is_refused() -> None:
    x = make_eng(FS, 30.0, seed=1).signal
    for name in ("L_V1", "stomach_ref", "LVN"):
        with pytest.raises(ValueError, match="outside the common signal set"):
            ft.inputs_from_signals({"L_T": x, name: x}, FS, z_enter=Z_ENTER)


def test_the_9_channel_recording_and_its_5_channel_view_give_the_same_row() -> None:
    # P1: the new cohort's raw contacts and stomach_ref never reach a feature, so the
    # recording seen as the old cohort sees it (one tripole per cuff, raw stomach) is
    # featurised identically - same columns, same values.
    rec9 = _nine()
    p9 = ft.prepare(ft.region_inputs(rec9, (0.0, DUR), z_enter=Z_ENTER))
    p5 = ft.prepare(ft.region_inputs(_five(rec9), (0.0, DUR), z_enter=Z_ENTER))
    m9, m5 = ft.feature_matrix(p9, _cores()), ft.feature_matrix(p5, _cores())
    assert list(m9.columns) == list(m5.columns) == list(ft.FEATURE_NAMES)
    np.testing.assert_allclose(m5.to_numpy(), m9.to_numpy(), rtol=1e-9, atol=1e-12,
                               equal_nan=True)


def test_scaling_every_channel_10x_changes_no_feature() -> None:
    rec = _nine()
    big = replace(rec, data=rec.data * 10.0)
    a = ft.feature_matrix(ft.prepare(ft.region_inputs(rec, (0.0, DUR), z_enter=Z_ENTER)), _cores())
    b = ft.feature_matrix(ft.prepare(ft.region_inputs(big, (0.0, DUR), z_enter=Z_ENTER)), _cores())
    np.testing.assert_allclose(b.to_numpy(), a.to_numpy(), rtol=1e-6, atol=1e-9, equal_nan=True)


def test_the_contact_only_families_are_not_features() -> None:
    for fam in ft.DROPPED_FAMILIES:
        assert not [n for n in ft.FEATURE_NAMES if n.startswith(fam)], fam
    m = ft.feature_matrix(ft.prepare(ft.region_inputs(_nine(), (0.0, DUR), z_enter=Z_ENTER)),
                          _cores())
    rates = ft.missing_rates(m)
    assert rates["band_ratio_max_c0"] == 0.0 and rates["cm_fraction"] == 0.0


def test_the_slow_share_counts_only_common_set_pairs() -> None:
    p = ft.prepare(ft.region_inputs(_nine(), (0.0, DUR), z_enter=Z_ENTER))
    pairs = (("L_T", "0-2"), ("L_V1", "300-3000"), ("stomach_ref", "300-3000"),
             ("ANT1", "300-3000"))
    assert ft.core_features(p, 5.0, 5.2, pairs)["slow_pair_share"] == 0.5
    only_detection = (("L_V1", "0-2"), ("stomach_ref", "0-2"))
    assert np.isnan(ft.core_features(p, 5.0, 5.2, only_detection)["slow_pair_share"])


# ---------------------------------------------------------------------------
# channel-count independence (invariant 10; ruling 2026-10-08 (b) item 1(a))
# ---------------------------------------------------------------------------

COUNT_SEEDS = range(6)
"""Seeds of the 2-vs-8 comparison. Measured with these: the largest |mean shift| over
every feature is 0.12 pooled SD (onset_rate_max_c0). With 3 seeds sampling noise alone
reached 0.29, so fewer seeds would make the tolerance below a coin toss."""

COUNT_TOL_SD: Final = 0.20
"""Tolerance, stated: a feature passes when its mean over cores moves by at most 0.20
pooled standard deviations between 2 and 8 nerve signals of identical per-signal
statistics. The mean, not the median, is the location tested: every pair statistic is
mean-unbiased for the 2-signal value at any count, while an aggregate over more signals
is narrower, so its median moves toward its mean on a skewed feature. The Night 1 code
fails this on 58 of 92 features (cm_fraction by 1.94 SD, frac_signals_over by 1.8,
power_rel_spread by 1.2, z_mean by up to 1.4, every max by 0.2-0.8)."""


@pytest.fixture(scope="module")
def count_tables() -> dict[int, pd.DataFrame]:
    out: dict[int, list[pd.DataFrame]] = {2: [], 8: []}
    for seed in COUNT_SEEDS:
        for n in (2, 8):
            ss = make_signal_set(n, FS, HOST_S, seed=seed)
            p = ft.prepare(ft.inputs_from_signals(ss.signals, FS, z_enter=Z_ENTER))
            out[n].append(ft.feature_matrix(p, [(a, b, ()) for a, b in ss.events + ss.quiet]))
    return {n: pd.concat(v, ignore_index=True) for n, v in out.items()}


def test_every_feature_is_independent_of_the_nerve_signal_count(
        count_tables: dict[int, pd.DataFrame]) -> None:
    two, eight = count_tables[2], count_tables[8]
    bad = {}
    for f in ft.FEATURE_NAMES:
        a, b = two[f].to_numpy(float), eight[f].to_numpy(float)
        assert np.isnan(a).mean() == np.isnan(b).mean(), f  # missingness never moves
        both = np.concatenate([a, b])
        sd = float(np.nanstd(both)) if np.isfinite(both).any() else 0.0
        if sd == 0:
            continue  # constant (or never defined) at both counts: nothing can move
        shift = abs(float(np.nanmean(b)) - float(np.nanmean(a))) / sd
        if shift > COUNT_TOL_SD:
            bad[f] = round(shift, 3)
    assert not bad, bad


def test_the_pair_statistics_equal_max_and_min_for_two_signals() -> None:
    assert ft.pair_max([3.0, -1.0]) == 3.0
    assert ft.pair_min([3.0, -1.0]) == -1.0
    assert np.isnan(ft.pair_max([2.0])) and np.isnan(ft.pair_max([2.0, np.nan]))


@given(st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=2, max_size=10))
def test_pair_max_is_the_mean_over_pairs_of_the_pair_max(v: list[float]) -> None:
    brute = float(np.mean([max(a, b) for a, b in itertools.combinations(v, 2)]))
    assert ft.pair_max(v) == pytest.approx(brute, rel=1e-9, abs=1e-6)
    assert ft.pair_max([*v, math.nan]) == pytest.approx(brute, rel=1e-9, abs=1e-6)
    lo = float(np.mean([min(a, b) for a, b in itertools.combinations(v, 2)]))
    assert ft.pair_min(v) == pytest.approx(lo, rel=1e-9, abs=1e-6)


def _both(x: np.ndarray) -> dict[str, np.ndarray]:
    """Put one signal on both cuffs.

    A pair statistic over one repeated value is that value, so a per-signal property is
    tested through the pair aggregates unchanged.
    """
    return {"L_T": x, "R_T": x.copy()}


def _ratio_at(x: np.ndarray, t0: float, t1: float) -> float:
    inp = ft.inputs_from_signals(_both(x), FS, z_enter=Z_ENTER)
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
    p = ft.prepare(ft.inputs_from_signals(_both(x), FS, z_enter=Z_ENTER))
    inside = ft.core_features(p, span.start_s, span.stop_s)["clip_frac_c0"]
    outside = ft.core_features(p, 1.0, 1.2)["clip_frac_c0"]
    assert inside > 0.2 and outside == 0.0


def test_one_sample_at_the_maximum_is_not_clipping() -> None:
    # the largest spike sets the channel's maximum but is not a flat run at a rail
    x = make_eng(FS, HOST_S, seed=7).signal.copy()
    x[int(3.1 * FS)] += 50 * float(np.max(np.abs(x)))
    p = ft.prepare(ft.inputs_from_signals(_both(x), FS, z_enter=Z_ENTER))
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
        p = ft.prepare(ft.inputs_from_signals(sig, FS, z_enter=Z_ENTER))
        feats[name] = ft.core_features(p, 3.0, 3.1)
    assert feats["hum"]["line_ratio_max"] > 5 * feats["clean"]["line_ratio_max"]
    assert feats["locked"]["line_plv_max"] > 3 * feats["clean"]["line_plv_max"]


def test_a_core_must_have_positive_duration() -> None:
    p = ft.prepare(ft.inputs_from_signals({"L_T": make_eng(FS, HOST_S, seed=1).signal}, FS,
                                          z_enter=Z_ENTER))
    with pytest.raises(ValueError, match="positive duration"):
        ft.core_features(p, 1.0, 1.0)


def test_feature_names_are_unique_and_carry_no_retired_band() -> None:
    assert len(set(ft.FEATURE_NAMES)) == len(ft.FEATURE_NAMES)
    assert ft.BANDS[ft.ENG_BAND].hi_hz == 3000.0  # ruling (b) R8: never 300-5000


PER_SIGNAL_MAX: Final[tuple[str, ...]] = (
    "band_ratio_max", "env_slope_max", "slew_max", "kurtosis_max", "line_length_max",
    "onset_rate_max", "peak_z", "clip_frac")
"""Per-context features that are the pair-max of a value of each signal alone."""


def test_each_max_is_the_pair_max_of_its_per_signal_values() -> None:
    # nerve signals only, so every group statistic is the nerve pair statistic; a signal's
    # own value is read from a set holding it twice (pair-max of one repeated value)
    ss = make_signal_set(3, FS, HOST_S, seed=21)
    nerve = {k: v for k, v in ss.signals.items() if k.endswith("_T")}
    cores = [(a, b, ()) for a, b in ss.events[:6] + ss.quiet[:3]]
    whole = ft.feature_matrix(ft.prepare(ft.inputs_from_signals(nerve, FS, z_enter=Z_ENTER)),
                              cores)
    alone = [ft.feature_matrix(ft.prepare(ft.inputs_from_signals(
        {"L_T": x, "R_T": x.copy()}, FS, z_enter=Z_ENTER)), cores) for x in nerve.values()]
    names = [f"{n}_c{round(c * 1000)}" for n in PER_SIGNAL_MAX for c in ft.CONTEXTS_S]
    names += ["line_ratio_max", "line_plv_max"]
    names += [f"offset_rate_min_c{round(c * 1000)}" for c in ft.CONTEXTS_S]
    for f in names:
        per = np.vstack([m[f].to_numpy() for m in alone])
        agg = ft.pair_min if "_min" in f else ft.pair_max
        want = [agg(per[:, i].tolist()) for i in range(per.shape[1])]
        np.testing.assert_allclose(whole[f].to_numpy(), want, rtol=1e-9, atol=1e-12,
                                   equal_nan=True, err_msg=f)
