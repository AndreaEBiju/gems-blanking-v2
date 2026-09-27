"""The new-cohort adapter: column order, tripole, and units the consumers expect."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gems_blanking_v2.io.channel_map import save_geometry
from gems_blanking_v2.io.stim_split import (
    PROTOCOL_FILENAME,
    default_protocol_book,
    write_protocol_book,
)
from gems_blanking_v2.io.store import GemsStore
from new_cohort_host import COLUMNS, five_column

from tests.conftest import write_tdt_block
from tests.test_channel_map import new_cohort_map
from tests.test_recording_io import needs_detector_core, synthetic_samples, write_flat_h5


@needs_detector_core
def test_the_five_columns_are_rvn_lvn_then_stomach_in_volts(tmp_path: Path) -> None:
    """The consumers index by position: nerve 1:2 (R then L), stomach 3:5."""
    store = GemsStore.initialise(tmp_path / "gems")
    write_protocol_book(default_protocol_book(), store.root / PROTOCOL_FILENAME)
    block = write_tdt_block(store.root / "August-September Chronic Recordings",
                            "gems_j_t01_bl_120000", 1_789_000_000.0)
    y = synthetic_samples() / 1e6  # the cohort protocol declares volts
    path = write_flat_h5(block / "gems_j_t01_bl_120000_sig.h5", y)
    # new_cohort_map stores cuffs L (cols 0-2), R (cols 3-5), then ANT1-3 (6-8).
    save_geometry(new_cohort_map("J"), "gems_j_t01_bl_120000_20260910T002640Z",
                  store, mirror_to_profile=False)

    out, fs, prov = five_column(path, store)

    v = y.astype(np.float64)
    r_t = 0.5 * v[:, 3] + 0.5 * v[:, 5] - v[:, 4]
    l_t = 0.5 * v[:, 0] + 0.5 * v[:, 2] - v[:, 1]
    np.testing.assert_allclose(out[:, 0], r_t, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(out[:, 1], l_t, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(out[:, 2:], v[:, 6:9], rtol=1e-9, atol=1e-15)
    assert fs > 0
    assert prov["columns"] == ",".join(COLUMNS)
