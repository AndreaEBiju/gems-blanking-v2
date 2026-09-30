"""Tests for :mod:`gems_blanking_v2.derive.common_mode` (ruling 2026-09-29, task 14)."""

from __future__ import annotations

import numpy as np
import pytest
from gems_blanking_v2.derive.common_mode import (
    SETTLE_S,
    differential_survival,
    fit_leak,
    outside_reference,
    subtract_common_mode,
    subtract_multi,
)
from gems_blanking_v2.derive.derivations import tripole
from gems_blanking_v2.detect import chain
from gems_blanking_v2.types import Recording
from scipy.signal import butter, sosfiltfilt

from tests.conftest import make_shared_ground, make_two_source_ground

FS = 24414.0625


@pytest.fixture(scope="module")
def world():  # noqa: ANN201 - a NamedTuple from conftest
    return make_shared_ground(FS, 30.0, seed=3)


def _leak(g: dict[str, float], cuff: str) -> float:
    """Return what T keeps of the ground signal: 0.5 g1 + 0.5 g3 - g2."""
    return 0.5 * g[f"{cuff}VN1"] + 0.5 * g[f"{cuff}VN3"] - g[f"{cuff}VN2"]


def test_the_reference_is_every_channel_outside_the_cuff(world) -> None:  # noqa: ANN001
    _ref, names = outside_reference(world.rec, "L")
    assert names == ("RVN1", "RVN2", "RVN3", "ANT1", "ANT2", "ANT3")
    _ref, names = outside_reference(world.rec, "R")
    assert "RVN2" not in names and "LVN2" in names and len(names) == 6


def _eng(x: np.ndarray) -> np.ndarray:
    """Filter to the spike consumer's band, order 4, zero phase."""
    return sosfiltfilt(butter(4, (300.0, 3000.0), btype="bandpass", fs=FS, output="sos"), x)


@pytest.mark.parametrize("cuff", ["L", "R"])
def test_the_leak_of_the_ground_signal_is_removed(world, cuff: str) -> None:  # noqa: ANN001
    """T's content of the true ground signal falls 10x or more, and k matches the oracle.

    The oracle is ``k*`` = (T.C) / (ref.C) with the TRUE common signal C as the
    instrument - what k would be with a noise-free reference. ``(T.C)/(C.C)`` is
    what T keeps of C; its expected value is the gain leak 0.5 g1 + 0.5 g3 - g2,
    within the chance correlation of T's own content with C (tested loosely).
    """
    corrected, fit = subtract_common_mode(world.rec, cuff)
    idx = {c.contact_index: c.index for c in world.rec.channels if c.cuff_id == cuff}
    t = tripole(*(np.asarray(world.rec.data[:, idx[i]], float) for i in (1, 2, 3)), 0.5, 0.5)
    ref, _names = outside_reference(world.rec, cuff)
    te, re, ce, ke = _eng(t), _eng(ref), _eng(world.common), _eng(corrected)
    oracle = float(te @ ce / (re @ ce))
    assert fit.k == pytest.approx(oracle, rel=0.05)
    before, after = float(te @ ce / (ce @ ce)), float(ke @ ce / (ce @ ce))
    assert before == pytest.approx(_leak(world.gains, cuff), abs=0.003)
    assert abs(after) < 0.1 * abs(before)
    assert fit.rms_after_uv < fit.rms_before_uv


@pytest.mark.parametrize("cuff", ["L", "R"])
def test_a_differential_spike_survives_within_5_percent(world, cuff: str) -> None:  # noqa: ANN001
    ratio = differential_survival(world.rec, cuff, at_s=4.0003, amp_uv=40.0)
    assert ratio == pytest.approx(1.0, abs=0.05)


def test_a_reference_that_included_the_cuff_would_eat_the_spike(world) -> None:  # noqa: ANN001
    """Show why the reference is outside the cuff.

    With the cuff's own middle contact in it, a differential spike on that contact is
    partly subtracted away.
    """
    rec = world.rec
    fs = float(rec.fs)
    idx = {c.name: c.index for c in rec.channels}
    t = tripole(*(np.asarray(rec.data[:, idx[f"LVN{i}"]], float) for i in (1, 2, 3)), 0.5, 0.5)
    own = np.asarray(rec.data[:, idx["LVN2"]], float)
    k_own, *_ = fit_leak(t, own, fs)
    k_out, *_ = fit_leak(t, outside_reference(rec, "L")[0], fs)
    assert abs(k_own) > 5 * abs(k_out)  # T's own -V2 term dominates an in-cuff fit


def test_nan_stays_nan_and_never_enters_the_fit(world) -> None:  # noqa: ANN001
    data = np.array(world.rec.data, dtype=np.float64)
    fs = float(world.rec.fs)
    i0, i1 = int(2.0 * fs), int(2.5 * fs)
    data[i0:i1, 0] = np.nan  # LVN1 masked: L's T is NaN there
    rec = Recording(fs=world.rec.fs, data=data, channels=world.rec.channels,
                    animal=world.rec.animal, session=world.rec.session, path=world.rec.path)
    corrected, fit = subtract_common_mode(rec, "L")
    assert np.isnan(corrected[i0:i1]).all() and np.isfinite(corrected[:i0]).all()
    pad = int(round(SETTLE_S * fs))
    assert fit.n_fit <= data.shape[0] - (i1 - i0) - 2 * pad
    _full, clean = subtract_common_mode(world.rec, "L")
    assert fit.k == pytest.approx(clean.k, rel=0.05)
    assert not np.any(corrected == 0.0)  # never zero-filled (invariant 1)


def test_the_fit_is_one_whole_recording_scalar_per_cuff(world) -> None:  # noqa: ANN001
    _c, fit = subtract_common_mode(world.rec, "L")
    assert isinstance(fit.k, float) and fit.cuff == "L"
    assert fit.n_fit > 0.9 * world.rec.data.shape[0]


def test_a_cuff_without_three_contacts_or_outside_channels_is_refused(world) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="contacts 1-3"):
        subtract_common_mode(world.rec, "X")
    only_l = [c for c in world.rec.channels if c.cuff_id == "L"]
    rec = Recording(fs=world.rec.fs, data=np.asarray(world.rec.data)[:, [c.index for c in only_l]],
                    channels=tuple(type(c)(**{**{f: getattr(c, f) for f in c.__dataclass_fields__},
                                              "index": k}) for k, c in enumerate(only_l)),
                    animal=world.rec.animal, session=world.rec.session, path=world.rec.path)
    with pytest.raises(ValueError, match="no channel outside"):
        subtract_common_mode(rec, "L")


def test_the_correction_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.derive.common_mode" not in chain.generation_modules()


# --- multi-regressor subtraction (ruling 2026-09-30, item 4b) ------------------


@pytest.fixture(scope="module")
def two():  # noqa: ANN201 - a NamedTuple from conftest
    return make_two_source_ground(FS, 60.0, gain_spread=0.10, seed=11)


def _halves(n: int) -> tuple[np.ndarray, np.ndarray]:
    fit = np.zeros(n, dtype=bool)
    fit[: n // 2] = True
    return fit, ~fit


def _content(x: np.ndarray, src: np.ndarray, sel: np.ndarray) -> float:
    """Return what x keeps of a source, in the ENG band, over the samples ``sel``."""
    xe, se = _eng(x)[sel], _eng(src)[sel]
    return abs(float((xe - xe.mean()) @ (se - se.mean()) / ((se - se.mean()) @ (se - se.mean()))))


@pytest.mark.parametrize("cuff", ["L", "R"])
def test_one_scalar_cannot_cancel_two_sources_and_the_regressors_can(two, cuff: str) -> None:  # noqa: ANN001
    """Check both leaks on the held-out half, for scalar, multi, and multi without the heart.

    The multi fit with the cardiac template cuts both the ground's and the heart's
    leak 4x or more (measured on this fixture: 5.6-10x ground, 7-230x heart), and
    beats both the single scalar and the fit without the template on each source.
    """
    rec = two.rec
    fit, held = _halves(rec.data.shape[0])
    idx = {c.contact_index: c.index for c in rec.channels if c.cuff_id == cuff}
    t = tripole(*(np.asarray(rec.data[:, idx[i]], float) for i in (1, 2, 3)), 0.5, 0.5)
    scalar, _f = subtract_common_mode(rec, cuff)
    multi, mf = subtract_multi(rec, cuff, fit, two.beats_s)
    no_tpl, _m0 = subtract_multi(rec, cuff, fit)
    assert mf.names[-1] == "cardiac" and len(mf.names) == 7
    for src in (two.ground.common, two.cardiac):
        before = _content(t, src, held)
        after = _content(multi, src, held)
        assert after < before / 4.0
        assert after < _content(scalar, src, held) and after < _content(no_tpl, src, held)


def test_one_scalar_on_a_blend_can_make_the_ground_leak_worse(two) -> None:  # noqa: ANN001
    """Reproduce A t05 L (2026-09-30): the single scalar overcorrects one source.

    Fitted to the blend of two sources, it adds ground leak rather than removing it.
    """
    rec = two.rec
    _fit, held = _halves(rec.data.shape[0])
    idx = {c.contact_index: c.index for c in rec.channels if c.cuff_id == "L"}
    t = tripole(*(np.asarray(rec.data[:, idx[i]], float) for i in (1, 2, 3)), 0.5, 0.5)
    scalar, _f = subtract_common_mode(rec, "L")
    assert _content(scalar, two.ground.common, held) > 2.0 * _content(t, two.ground.common, held)


def test_without_beats_there_is_no_cardiac_regressor(two) -> None:  # noqa: ANN001
    fit, _held = _halves(two.rec.data.shape[0])
    _c, mf = subtract_multi(two.rec, "L", fit)
    assert "cardiac" not in mf.names and len(mf.names) == 6 and mf.n_beats_template == 0


def test_the_template_is_built_from_the_fit_half_only(two) -> None:  # noqa: ANN001
    n = two.rec.data.shape[0]
    fit, _held = _halves(n)
    _c, mf = subtract_multi(two.rec, "L", fit, two.beats_s)
    h = int(round(0.020 * FS)) + int(round(0.0005 * FS)) + 2
    centres = np.round(two.beats_s * FS).astype(int)
    usable = centres[(centres - h >= 0) & (centres + h < n)]
    assert mf.n_beats_template == int(fit[usable].sum())


@pytest.mark.parametrize("cuff", ["L", "R"])
def test_a_differential_spike_survives_the_multi_fit(two, cuff: str) -> None:  # noqa: ANN001
    rec = two.rec
    fit, _held = _halves(rec.data.shape[0])
    idx = {c.contact_index: c.index for c in rec.channels if c.cuff_id == cuff}
    tt = np.arange(rec.data.shape[0]) / FS
    t0s = (38.0003, 45.5007, 52.1001)  # in the held-out half
    spike = sum(40.0 * np.exp(-0.5 * ((tt - t0) / (0.0006 / 2.3548)) ** 2) for t0 in t0s)
    data = np.array(rec.data, dtype=np.float64)
    data[:, idx[2]] += spike
    inj = Recording(fs=rec.fs, data=data, channels=rec.channels, animal=rec.animal,
                    session=rec.session, path=rec.path)
    base, _a = subtract_multi(rec, cuff, fit, two.beats_s)
    with_s, _b = subtract_multi(inj, cuff, fit, two.beats_s)
    d_raw, d_cor = _eng(-spike), _eng(with_s - base)
    for t0 in t0s:
        a, b = int((t0 - 0.005) * FS), int((t0 + 0.005) * FS)
        ratio = np.max(np.abs(d_cor[a:b])) / np.max(np.abs(d_raw[a:b]))
        assert ratio == pytest.approx(1.0, abs=0.05)


def test_a_fit_mask_of_the_wrong_length_is_refused(two) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="fit mask"):
        subtract_multi(two.rec, "L", np.ones(10, dtype=bool))


def test_nan_stays_nan_through_the_multi_fit(two) -> None:  # noqa: ANN001
    data = np.array(two.rec.data, dtype=np.float64)
    i0, i1 = int(40.0 * FS), int(40.5 * FS)
    data[i0:i1, 6] = np.nan  # ANT1, an outside regressor for both cuffs
    rec = Recording(fs=two.rec.fs, data=data, channels=two.rec.channels,
                    animal=two.rec.animal, session=two.rec.session, path=two.rec.path)
    fit, _held = _halves(data.shape[0])
    out, _mf = subtract_multi(rec, "L", fit, two.beats_s)
    assert np.isnan(out[i0:i1]).all() and np.isfinite(out[:i0]).all()


def test_the_template_is_placed_with_sub_sample_alignment() -> None:
    """Reproduce a noiseless beat train at non-integer times to < 1% RMS.

    Integer alignment - PIPELINE.md 10.2's failure - leaves ~3.4% on this waveform.
    """
    from gems_blanking_v2.derive.common_mode import _cardiac_regressor  # noqa: PLC0415

    n = int(10 * FS)
    tt = np.arange(n) / FS
    beats = np.cumsum(np.random.default_rng(3).uniform(0.150, 0.160, 60)) + 0.1
    beats = beats[beats < 9.8]
    t = np.zeros(n)
    for b in beats:
        u = tt - b
        m = np.abs(u) < 0.02
        t[m] += 150.0 * (-(u[m] / 0.0004) * np.exp(0.5 - 0.5 * (u[m] / 0.0004) ** 2))
    reg, n_beats = _cardiac_regressor(t, beats, FS, np.ones(n, dtype=bool))
    assert n_beats == beats.size
    assert np.sqrt(np.mean((t - reg) ** 2)) < 0.01 * np.sqrt(np.mean(t**2))


def test_the_held_out_half_never_shapes_the_coefficients(two) -> None:  # noqa: ANN001
    n = two.rec.data.shape[0]
    fit, _held = _halves(n)
    _a, before = subtract_multi(two.rec, "L", fit, two.beats_s)
    data = np.array(two.rec.data, dtype=np.float64)
    data[int(0.8 * n):, :] += np.random.default_rng(9).normal(0.0, 500.0, (n - int(0.8 * n), 9))
    moved = Recording(fs=two.rec.fs, data=data, channels=two.rec.channels,
                      animal=two.rec.animal, session=two.rec.session, path=two.rec.path)
    _b, after = subtract_multi(moved, "L", fit, two.beats_s)
    assert np.allclose(before.coefs, after.coefs, rtol=1e-9, atol=1e-12)


def test_the_multi_fit_keeps_the_settling_pad_out_of_the_fit(two) -> None:  # noqa: ANN001
    data = np.array(two.rec.data, dtype=np.float64)
    n = data.shape[0]
    i0, i1 = int(10.0 * FS), int(10.5 * FS)  # inside the fit half
    data[i0:i1, 6] = np.nan
    rec = Recording(fs=two.rec.fs, data=data, channels=two.rec.channels,
                    animal=two.rec.animal, session=two.rec.session, path=two.rec.path)
    fit, _held = _halves(n)
    _o, mf = subtract_multi(rec, "L", fit, two.beats_s)
    assert mf.n_fit <= int(fit.sum()) - (i1 - i0) - 2 * int(round(SETTLE_S * FS))
