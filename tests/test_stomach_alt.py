"""Tests for :mod:`gems_blanking_v2.derive.stomach_alt` (ruling 2026-09-30 (b), item 6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gems_blanking_v2.derive.derivations import build_stomach_reference
from gems_blanking_v2.derive.stomach_alt import readable_stomach_ref
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
