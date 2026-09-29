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
)
from gems_blanking_v2.derive.derivations import tripole
from gems_blanking_v2.detect import chain
from gems_blanking_v2.types import Recording
from scipy.signal import butter, sosfiltfilt

from tests.conftest import make_shared_ground

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
