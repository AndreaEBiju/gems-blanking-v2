"""Channel order comes from the file; an impossible unit declaration raises."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from gems_blanking_v2.io.chanlabels import (
    SIGMA_PLAUSIBLE_UV,
    assert_labels_match,
    assert_plausible_units,
    channels_from_labels,
    read_channel_labels,
)

NEW_COHORT = (
    "RVN1", "RVN2", "RVN3", "LVN1", "LVN2", "LVN3", "ANT1", "ANT2", "ANT3",
)


# ---------------------------------------------------------------------------
# order is read, never declared
# ---------------------------------------------------------------------------


def test_the_table_is_built_in_the_file_s_order_not_a_convention() -> None:
    """Build in the file's column order, never in a cuff convention.

    The fixture that started this built cuffs ('L', 'R') and read column 0 as
    LVN1 on a file whose labels start RVN1. Every left/right result inverted.
    """
    chans = channels_from_labels(NEW_COHORT, rostral_end=1, config="independent")

    assert [c.name for c in chans] == list(NEW_COHORT)
    assert chans[0].cuff_id == "R", "column 0 is RVN1 in this cohort, not LVN1"
    assert chans[3].cuff_id == "L"
    assert [c.contact_index for c in chans[:3]] == [1, 2, 3]


def test_roles_come_from_the_label_stem() -> None:
    chans = channels_from_labels(NEW_COHORT, rostral_end=1, config="independent")

    assert [c.role for c in chans[:6]] == ["nerve"] * 6
    assert [c.role for c in chans[6:]] == ["stomach"] * 3
    assert all(c.cuff_id is None for c in chans[6:])


def test_an_unknown_stem_raises_rather_than_defaulting_to_a_role() -> None:
    """Refuse a label with no known stem.

    A stomach contact placed in a nerve cuff would be averaged into the
    tripole, which is a wrong number rather than a missing one.
    """
    with pytest.raises(ValueError, match="unknown stem"):
        channels_from_labels(("RVN1", "WEIRD2"), rostral_end=1, config="independent")


def test_a_map_that_transposes_the_cuffs_is_refused() -> None:
    """The exact failure that prompted this: same names, wrong order."""
    swapped = channels_from_labels(
        ("LVN1", "LVN2", "LVN3", "RVN1", "RVN2", "RVN3", "ANT1", "ANT2", "ANT3"),
        rostral_end=1, config="independent",
    )

    with pytest.raises(ValueError, match="disagrees with the file"):
        assert_labels_match(swapped, NEW_COHORT, "rec.mat")


def test_a_matching_map_passes_and_case_does_not_matter() -> None:
    """Cross-platform rule 7: this lab's own data mixes case for one entity."""
    chans = channels_from_labels(NEW_COHORT, rostral_end=1, config="independent")

    assert_labels_match(chans, NEW_COHORT, "rec.mat")
    assert_labels_match(chans, tuple(s.lower() for s in NEW_COHORT), "rec.mat")


def test_a_file_with_no_labels_is_not_an_error() -> None:
    """The old five-channel cohort has none, which is why meta.json exists."""
    assert_labels_match(
        channels_from_labels(("RVN1",), rostral_end=None, config="hw_tripole"),
        None, "old.mat",
    )


def test_a_channel_count_mismatch_names_both_sides(tmp_path: Path) -> None:
    chans = channels_from_labels(NEW_COHORT, rostral_end=1, config="independent")

    with pytest.raises(ValueError, match="chanlabels has 2"):
        assert_labels_match(chans, ("RVN1", "RVN2"), "rec.mat")


def test_missing_labels_read_as_none_not_as_empty(tmp_path: Path) -> None:
    """Return None when the file carries no labels.

    None means 'the file does not say'. An empty tuple would read as 'the file
    says there are no channels', which is a different and false claim.
    """
    p = tmp_path / "nolabels.h5"
    with h5py.File(p, "w") as f:
        f["y"] = np.zeros((10, 2), dtype=np.float32)
        f["fs"] = 1000.0

    assert read_channel_labels(p) is None


# ---------------------------------------------------------------------------
# units stay declared, but an impossible declaration raises
# ---------------------------------------------------------------------------


def _data(sigma_in_file_units: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, sigma_in_file_units, size=(5000, 3))


def test_a_plausible_declaration_passes() -> None:
    """Volts on a file whose samples are ~8e-6 gives 8 uV - a real tripole."""
    assert_plausible_units(_data(8e-6), "V", 1e6, "rec.mat")


def test_declaring_microvolts_on_a_volts_file_raises() -> None:
    """The 10^6 typo. Sigma would be 8e-6 uV = 8 pV, which no electrode gives."""
    with pytest.raises(ValueError, match="outside the plausible"):
        assert_plausible_units(_data(8e-6), "uV", 1.0, "rec.mat")


def test_declaring_volts_on_a_microvolts_file_raises() -> None:
    """The same typo the other way: sigma would be 8 V."""
    with pytest.raises(ValueError, match="outside the plausible"):
        assert_plausible_units(_data(8.0), "V", 1e6, "rec.mat")


def test_the_message_names_the_unit_and_the_implied_sigma() -> None:
    """A bare 'implausible' would send someone to look at the electrode."""
    with pytest.raises(ValueError) as exc:
        assert_plausible_units(_data(8e-6), "uV", 1.0, "rec.mat")

    msg = str(exc.value)
    assert "'uV'" in msg
    assert "uV," in msg or "uV " in msg
    assert "1e6" in msg or "1000000" in msg or "10^6" in msg


def test_one_dead_channel_cannot_veto_a_sound_declaration() -> None:
    """Judge the median across channels.

    A flat contact therefore does not reject the file - and cannot rescue a
    wrong declaration either.
    """
    d = _data(8e-6)
    d[:, 0] = 0.0

    assert_plausible_units(d, "V", 1e6, "rec.mat")


def test_the_window_is_wide_enough_for_both_measured_cohorts() -> None:
    """Accept every sigma measured on the two cohorts in hand.

    Old-cohort contacts measured 10-41 uV, new-cohort 8-10 uV. The check must
    reject none of them, or it would be inferring rather than sanity-checking.
    """
    lo, hi = SIGMA_PLAUSIBLE_UV

    for sigma_uv in (8.0, 10.0, 18.6, 41.2):
        assert lo <= sigma_uv <= hi
        assert_plausible_units(_data(sigma_uv * 1e-6), "V", 1e6, "rec.mat")


def test_an_unknown_electrode_config_is_refused() -> None:
    """Config selects the tripole derivation; a typo must not pass as a third kind."""
    with pytest.raises(ValueError, match="neither"):
        channels_from_labels(NEW_COHORT, rostral_end=None, config="hw-tripole")
