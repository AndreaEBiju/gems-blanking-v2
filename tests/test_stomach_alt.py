"""Tests for :mod:`gems_blanking_v2.derive.stomach_alt` (ruling 2026-09-30 (b), item 6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.derivations import build_stomach_reference
from gems_blanking_v2.derive.stomach_alt import notch, notched_stomach_ref, readable_stomach_ref
from gems_blanking_v2.detect import chain

from tests.conftest import make_cuff_contacts

FS = 2000.0


@pytest.fixture(scope="module")
def rec():  # noqa: ANN201
    return make_cuff_contacts(FS, 20.0, n_stomach=3, seed=4)


def test_with_nothing_excluded_it_is_the_derivations_stomach_ref(rec) -> None:  # noqa: ANN001
    ref, used = readable_stomach_ref(rec, ())
    derived, _how = build_stomach_reference(rec)
    assert used == ("ANT1", "ANT2", "ANT3")
    assert derived is not None and np.allclose(ref, derived)


def test_excluding_ant1_gives_ant2_minus_the_mean_of_ant2_and_ant3(rec) -> None:  # noqa: ANN001
    ref, used = readable_stomach_ref(rec, ("ANT1",))
    col = {c.name: np.asarray(rec.data[:, c.index], float) for c in rec.channels}
    assert used == ("ANT2", "ANT3")
    assert np.allclose(ref, col["ANT2"] - 0.5 * (col["ANT2"] + col["ANT3"]))


def test_a_hum_on_ant1_does_not_reach_the_alternative(rec) -> None:  # noqa: ANN001
    data = np.array(rec.data, dtype=np.float64)
    t = np.arange(data.shape[0]) / FS
    ant1 = next(c.index for c in rec.channels if c.name == "ANT1")
    data[:, ant1] += 500.0 * np.sin(2 * np.pi * 60.0 * t)
    humming = replace(rec, data=data)
    alt, _u = readable_stomach_ref(humming, ("ANT1",))
    base, _u2 = readable_stomach_ref(rec, ("ANT1",))
    assert np.array_equal(alt, base)
    asis, _u3 = readable_stomach_ref(humming, ())
    assert np.std(asis - readable_stomach_ref(rec, ())[0]) > 100.0


def test_fewer_than_two_readable_contacts_is_refused(rec) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="needs two"):
        readable_stomach_ref(rec, ("ANT1", "ANT2"))


def test_stomach_alt_is_outside_the_generation_hash() -> None:
    assert "gems_blanking_v2.derive.stomach_alt" not in chain.generation_modules()


# --- ruling 2026-09-30 (c) 5: stomach_ref from a notched ANT1 --------------------


def _hum(rec, harmonics: tuple[float, ...]):  # noqa: ANN001, ANN202
    data = np.array(rec.data, dtype=np.float64)
    t = np.arange(data.shape[0]) / FS
    ant1 = next(c.index for c in rec.channels if c.name == "ANT1")
    for f0 in harmonics:
        data[:, ant1] += 500.0 * np.sin(2 * np.pi * f0 * t + f0)
    return replace(rec, data=data)


def _line(x: np.ndarray, f0: float) -> float:
    """Amplitude of the ``f0`` component, from the edges-trimmed signal."""
    k = int(2 * FS)
    y = x[k:-k]
    t = np.arange(y.size) / FS
    return float(2 * np.abs(np.mean(y * np.exp(-2j * np.pi * f0 * t))))


def test_the_notch_removes_the_hum_and_keeps_the_derivation(rec) -> None:  # noqa: ANN001
    clean, _how = build_stomach_reference(rec)
    ref = notched_stomach_ref(_hum(rec, (60.0,)), (60.0,))
    assert _line(ref, 60.0) < 0.01 * _line(build_stomach_reference(_hum(rec, (60.0,)))[0], 60.0)
    # away from the notch the reference is the derivation's, unchanged
    k = int(2 * FS)
    assert np.corrcoef(ref[k:-k], clean[k:-k])[0, 1] > 0.99


def test_a_harmonic_survives_a_sixty_hz_only_notch(rec) -> None:  # noqa: ANN001
    humming = _hum(rec, (60.0, 120.0))
    only60 = notched_stomach_ref(humming, (60.0,))
    both = notched_stomach_ref(humming, (60.0, 120.0))
    assert _line(only60, 120.0) > 100.0
    assert _line(both, 120.0) < 5.0


def test_only_ant1_is_filtered_and_the_input_is_not_written(rec) -> None:  # noqa: ANN001
    humming = _hum(rec, (60.0,))
    before = np.array(humming.data, copy=True)
    notched_stomach_ref(humming, (60.0,))
    assert np.array_equal(np.asarray(humming.data), before)


def test_the_notch_keeps_nan_where_it_was() -> None:
    x = np.sin(2 * np.pi * 3.0 * np.arange(4000) / FS)
    x[1000:1100] = np.nan
    y = notch(x, FS, (60.0,))
    assert np.array_equal(np.isnan(y), np.isnan(x))


def test_a_notch_outside_nyquist_or_a_missing_contact_is_refused(rec) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="outside"):
        notch(np.zeros(100), FS, (1500.0,))
    with pytest.raises(ValueError, match="no stomach contact"):
        notched_stomach_ref(rec, (60.0,), contact="LVN1")


def test_the_reference_is_notched_ant1_minus_the_mean_with_notched_ant1(rec) -> None:  # noqa: ANN001
    humming = _hum(rec, (60.0,))
    col = {c.name: np.asarray(humming.data[:, c.index], float) for c in humming.channels}
    a1 = notch(col["ANT1"], FS, (60.0,))
    expected = a1 - (a1 + col["ANT2"] + col["ANT3"]) / 3.0
    assert np.allclose(notched_stomach_ref(humming, (60.0,)), expected)


def test_the_zero_phase_notch_halves_59_and_61_hz() -> None:
    """NOTCH_Q's docstring: one pass is -3 dB at 60 +/- 1 Hz, so forward-backward gives 0.5."""
    t = np.arange(int(40 * FS)) / FS
    for f in (59.0, 61.0):
        y = notch(np.sin(2 * np.pi * f * t), FS, (60.0,))
        k = int(5 * FS)
        amp = float(np.sqrt(2) * np.std(y[k:-k]))
        assert amp == pytest.approx(0.5, abs=0.03)
    y = notch(np.sin(2 * np.pi * 50.0 * t), FS, (60.0,))  # mmc's band edge passes
    assert float(np.sqrt(2) * np.std(y[int(5 * FS):-int(5 * FS)])) > 0.98
