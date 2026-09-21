"""Loading a recording through ``GEMSBlanking``'s loader and adapting it to A.1.

``GEMSBlanking`` is private, so CI cannot see it. The tests that exercise the real
loader are therefore skipped when no checkout is found, and everything that can be
tested without it - the discovery rules, the 1-based interval conversion, the unit
scaling, the failure messages - is tested unconditionally.

Verified on 2026-09-21 against the sibling checkout at ``../GEMSBlanking``
(``fc217d3``): ``detector.recording_io`` imports under numpy 2.5.3 needing only
numpy and h5py, a flat HDF5 and a v5 ``.mat`` both load with arrays identical and
``fs`` exact, a transposed input is auto-oriented, and a ``_blankmotion.mat``
save-output is refused.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.io.channel_map import (
    ChannelMap,
    load_profile,
    save_geometry,
    save_profile,
)
from gems_blanking_v2.io.detector_core import (
    DETECTOR_CORE_ENV,
    detector_core_available,
    find_detector_core,
    import_detector_module,
)
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs, masked_count
from gems_blanking_v2.io.recording import load_recording, spans_from_matlab_intervals
from gems_blanking_v2.io.store import GemsStore
from scipy.io import savemat

from conftest import make_multichannel
from test_channel_map import new_cohort_map, old_cohort_map

F64 = npt.NDArray[np.float64]

FS = 24414.0625

needs_detector_core = pytest.mark.skipif(
    not detector_core_available(),
    reason="GEMSBlanking checkout not found; it is private, so CI cannot see it",
)


def write_flat_h5(path: Path, y: npt.NDArray[np.float32], fs: float = FS) -> Path:
    """Write the flat ``(y, fs)`` HDF5 layout ``m1_ingest.py`` produces."""
    with h5py.File(path, "w") as f:
        f["y"] = y
        f["fs"] = fs
    return path


def synthetic_samples(
    n_channels: int = 9, n_samples: int = 4_000, seed: int = 0
) -> npt.NDArray[np.float32]:
    """Return samples from the shared generator, float32 as the loader returns them."""
    rec, _ = make_multichannel(FS, n_samples / FS, seed=seed)
    return np.asarray(rec.data[:, :n_channels], dtype=np.float32)


# ---------------------------------------------------------------------------
# discovery - no GEMSBlanking required
# ---------------------------------------------------------------------------


def test_discovery_prefers_an_explicit_root(tmp_path: Path) -> None:
    (tmp_path / "detector").mkdir()
    (tmp_path / "detector" / "recording_io.py").write_text("", encoding="utf-8", newline="\n")
    assert find_detector_core(tmp_path) == tmp_path


def test_discovery_reads_the_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "detector").mkdir()
    (tmp_path / "detector" / "recording_io.py").write_text("", encoding="utf-8", newline="\n")
    monkeypatch.setenv(DETECTOR_CORE_ENV, str(tmp_path))
    assert find_detector_core() == tmp_path


def test_discovery_requires_the_module_not_just_the_directory(tmp_path: Path) -> None:
    """An empty ``detector-core`` submodule is the common case and must not match.

    ``detector-pyqt/detector-core`` is exactly that on this machine: a submodule
    directory that exists and is empty because nobody ran ``submodule update``.
    """
    (tmp_path / "detector").mkdir()
    with pytest.raises(FileNotFoundError, match=r"recording_io\.py"):
        find_detector_core(tmp_path)


def test_an_explicit_root_that_is_wrong_does_not_fall_back(tmp_path: Path) -> None:
    """Naming a checkout and silently getting a different one is the failure to avoid.

    A caller who points at a specific tree - an older one, say - has to be told it
    was not used, rather than discover it from the results.
    """
    with pytest.raises(FileNotFoundError, match="Refusing to fall back") as exc:
        find_detector_core(tmp_path / "also-nowhere")
    assert "explicit root" in str(exc.value)


def test_a_wrong_env_var_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DETECTOR_CORE_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="Refusing to fall back") as exc:
        find_detector_core()
    assert DETECTOR_CORE_ENV in str(exc.value)


def test_the_search_error_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing specified and nothing found, the message has to be actionable."""
    monkeypatch.delenv(DETECTOR_CORE_ENV, raising=False)
    monkeypatch.setattr("gems_blanking_v2.io.detector_core._search_roots", lambda: [Path("/nope")])
    with pytest.raises(FileNotFoundError) as exc:
        find_detector_core()
    message = str(exc.value)
    assert DETECTOR_CORE_ENV in message
    assert "private repository" in message
    assert "Clone it beside this repository" in message


def test_importing_from_a_missing_checkout_raises_before_touching_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DETECTOR_CORE_ENV, str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        import_detector_module("recording_io", tmp_path / "nope")


# ---------------------------------------------------------------------------
# 1-based inclusive -> 0-based half-open
# ---------------------------------------------------------------------------


def test_matlab_intervals_convert_to_half_open_seconds() -> None:
    """``[a, b]`` counting from 1 becomes ``[(a-1)/fs, b/fs)`` counting from 0."""
    spans = spans_from_matlab_intervals(np.array([[1, 10], [101, 200]]), fs=1000.0)
    assert spans == [(0.0, 0.010), (0.100, 0.200)]


def test_a_single_sample_matlab_interval_spans_one_sample() -> None:
    """``[5, 5]`` is one sample: ``[4/fs, 5/fs)``."""
    (start, stop), = spans_from_matlab_intervals(np.array([[5, 5]]), fs=1000.0)
    assert (start, stop) == (0.004, 0.005)
    assert round((stop - start) * 1000.0) == 1


def test_no_intervals_is_an_empty_list() -> None:
    assert spans_from_matlab_intervals(None, fs=FS) == []
    assert spans_from_matlab_intervals(np.zeros((0, 2), dtype=np.int64), fs=FS) == []


def test_a_badly_shaped_interval_array_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(k, 2\)"):
        spans_from_matlab_intervals(np.array([[1, 2, 3]]), fs=FS)


# ---------------------------------------------------------------------------
# the real loader
# ---------------------------------------------------------------------------


@needs_detector_core
def test_their_recording_type_is_not_ours() -> None:
    """Same name, different contract - which is why an adapter exists at all."""
    recording_io = import_detector_module("recording_io")
    fields = set(recording_io.Recording.__dataclass_fields__)
    assert "y" in fields
    assert "data" not in fields
    assert "channels" not in fields
    assert {"recording_id", "rec_type", "stim_end_idx", "existing_bad_intervals"} <= fields


@needs_detector_core
def test_a_flat_hdf5_round_trips_with_arrays_and_fs_intact(tmp_path: Path) -> None:
    y = synthetic_samples()
    path = write_flat_h5(tmp_path / "gems_j_t01_bl_120000.h5", y)

    loaded = load_recording(path, animal="J", channel_map=new_cohort_map("J"))

    assert loaded.recording.fs == FS
    assert loaded.recording.data.dtype == np.float64
    assert loaded.recording.data.shape == y.shape
    assert np.array_equal(loaded.recording.data, y.astype(np.float64))
    assert loaded.recording.animal == "J"
    assert loaded.recording.session == "gems_j_t01_bl_120000"
    assert loaded.recording.path == path


@needs_detector_core
def test_a_mat_round_trips_identically_to_the_hdf5(tmp_path: Path) -> None:
    y = synthetic_samples()
    mat = tmp_path / "gems_j_t01_bl_120000.mat"
    savemat(str(mat), {"y": y, "fs": FS})
    h5 = write_flat_h5(tmp_path / "same.h5", y)

    from_mat = load_recording(mat, animal="J", channel_map=new_cohort_map("J"))
    from_h5 = load_recording(h5, animal="J", channel_map=new_cohort_map("J"))

    assert np.array_equal(from_mat.recording.data, from_h5.recording.data)
    assert from_mat.recording.fs == from_h5.recording.fs == FS


@needs_detector_core
def test_fs_comes_from_the_file_and_is_never_hardcoded(tmp_path: Path) -> None:
    odd_fs = 12207.03125
    path = write_flat_h5(tmp_path / "odd.h5", synthetic_samples(), fs=odd_fs)
    loaded = load_recording(path, animal="J", channel_map=new_cohort_map("J"))
    assert loaded.recording.fs == odd_fs
    assert loaded.recording.fs != FS


@needs_detector_core
@pytest.mark.parametrize(("units", "factor"), [("uV", 1.0), ("mV", 1e3), ("V", 1e6)])
def test_the_declared_units_are_applied(tmp_path: Path, units: str, factor: float) -> None:
    y = synthetic_samples()
    path = write_flat_h5(tmp_path / "units.h5", y)
    mapping = ChannelMap(animal="J", channels=new_cohort_map("J").channels, units=units)  # type: ignore[arg-type]

    loaded = load_recording(path, animal="J", channel_map=mapping)

    assert np.allclose(loaded.recording.data, y.astype(np.float64) * factor, rtol=1e-12)
    assert loaded.provenance["declared_units"] == units
    assert loaded.provenance["scale_to_uv"] == factor


@needs_detector_core
def test_both_cohorts_load_and_report_their_config(tmp_path: Path) -> None:
    new_path = write_flat_h5(tmp_path / "new.h5", synthetic_samples(n_channels=9))
    old_path = write_flat_h5(tmp_path / "old.h5", synthetic_samples(n_channels=5))

    new = load_recording(new_path, animal="J", channel_map=new_cohort_map("J"))
    old = load_recording(old_path, animal="F", channel_map=old_cohort_map("F"))

    assert new.recording.n_channels == 9
    assert new.channel_map.config == "independent"
    assert new.provenance["config"] == "independent"

    assert old.recording.n_channels == 5
    assert old.channel_map.config == "hw_tripole"
    assert old.provenance["config"] == "hw_tripole"


@needs_detector_core
def test_the_geometry_comes_from_meta_json_in_the_store(tmp_path: Path) -> None:
    """The single source of truth: ``data/<animal>/<session>/meta.json``."""
    store = GemsStore.initialise(tmp_path / "gems")
    save_geometry(new_cohort_map("J"), "t01", store, mirror_to_profile=False)
    path = write_flat_h5(tmp_path / "gems_j_t01_bl.h5", synthetic_samples())

    loaded = load_recording(path, animal="J", session="t01", store=store)

    assert loaded.channel_map.n_channels == 9
    assert loaded.direction_valid
    assert loaded.provenance["geometry_source"] == "stored"


@needs_detector_core
def test_meta_json_beats_a_stale_profile_mirror(tmp_path: Path) -> None:
    """A UI session that rebuilt the profile must not override our geometry."""
    store = GemsStore.initialise(tmp_path / "gems")
    profiles = tmp_path / "profiles"
    save_profile(new_cohort_map("J", rostral_end=None), profiles)
    save_geometry(new_cohort_map("J", rostral_end=1), "t01", store, mirror_to_profile=False)
    path = write_flat_h5(tmp_path / "gems_j_t01_bl.h5", synthetic_samples())

    loaded = load_recording(
        path, animal="J", session="t01", store=store, profiles_root=profiles
    )
    assert loaded.direction_valid


@needs_detector_core
def test_the_channel_map_can_come_from_the_saved_profile(tmp_path: Path) -> None:
    save_profile(new_cohort_map("J"), tmp_path / "profiles")
    path = write_flat_h5(tmp_path / "gems_j_t01_bl.h5", synthetic_samples())

    loaded = load_recording(path, animal="J", profiles_root=tmp_path / "profiles")

    assert loaded.channel_map.n_channels == 9
    assert [c.name for c in loaded.recording.channels][:3] == ["LVN1", "LVN2", "LVN3"]


@needs_detector_core
def test_without_a_channel_map_it_refuses_rather_than_inventing_one(tmp_path: Path) -> None:
    """The file has no channel table, so there is nothing to infer from."""
    path = write_flat_h5(tmp_path / "x.h5", synthetic_samples())
    with pytest.raises(ValueError, match="no geometry for animal") as exc:
        load_recording(path, animal="J", profiles_root=tmp_path / "empty")
    # The message names the file the geometry belongs in, not just the failure.
    assert "meta.json" in str(exc.value)


@needs_detector_core
def test_a_channel_count_mismatch_is_refused(tmp_path: Path) -> None:
    """A map describing a different recording would mislabel every channel."""
    path = write_flat_h5(tmp_path / "five.h5", synthetic_samples(n_channels=5))
    with pytest.raises(ValueError, match="describes a different recording"):
        load_recording(path, animal="J", channel_map=new_cohort_map("J"))


@needs_detector_core
def test_an_unknown_rostral_end_warns_once_and_flags_direction(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_flat_h5(tmp_path / "gems_j_t01_bl.h5", synthetic_samples())
    mapping = new_cohort_map("J", rostral_end=None)

    with caplog.at_level(logging.WARNING, logger="gems_blanking_v2.io.channel_map"):
        loaded = load_recording(path, animal="J", channel_map=mapping)

    assert not loaded.direction_valid
    assert loaded.provenance["direction_valid"] is False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "gems_j_t01_bl" in warnings[0].getMessage()


@needs_detector_core
def test_a_blankmotion_save_output_is_refused_by_their_loader(tmp_path: Path) -> None:
    """Their guard, preserved by wrapping rather than reimplementing."""
    path = tmp_path / "gems_j_t01_bl_blankmotion.mat"
    savemat(str(path), {"yOut": synthetic_samples(), "fs": FS})
    with pytest.raises(ValueError, match="labeling OUTPUT"):
        load_recording(path, animal="J", channel_map=new_cohort_map("J"))


@needs_detector_core
def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path / "absent.h5", animal="J", channel_map=new_cohort_map("J"))


@needs_detector_core
def test_provenance_records_what_the_adapter_did(tmp_path: Path) -> None:
    path = write_flat_h5(tmp_path / "gems_j_t01_bl.h5", synthetic_samples())
    loaded = load_recording(path, animal="J", channel_map=new_cohort_map("J"))

    provenance = loaded.provenance
    assert provenance["loader"] == "GEMSBlanking:detector/recording_io.load_recording"
    assert provenance["source_format"] == "raw_h5"
    assert provenance["native_dtype"] == "float32"
    assert provenance["recording_id"] == "gems_j_t01_bl"
    assert provenance["rec_type"] == "baseline"


@needs_detector_core
def test_the_profile_survives_a_round_trip_through_their_own_profile_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interop claim, checked against the real class rather than asserted.

    Their ``Profile.load`` reads known fields one at a time (so an extra key does
    not raise) and ``Profile.save`` writes ``asdict(self)`` (so an extra *top-level*
    key is dropped). Geometry and units therefore live inside
    ``channel_assignment``, which they carry through opaquely.
    """
    profiles = import_detector_module("preprocessing.profiles")
    monkeypatch.setattr(profiles.Profile, "profiles_dir", staticmethod(lambda: tmp_path))

    mapping = ChannelMap(animal="J", channels=new_cohort_map("J").channels, units="mV")
    save_profile(mapping, tmp_path)

    theirs = profiles.Profile.load("J")
    assert theirs is not None, "our profile must be readable by their loader"
    assert theirs.animal_id == "J"
    theirs.notch = {"frequencies_filtered": [60, 120], "q_factor": 30}
    theirs.save()

    ours = load_profile("J", tmp_path)
    assert ours is not None
    assert ours.units == "mV"
    assert ours.channels == mapping.channels
    assert ours.direction_valid


@needs_detector_core
def test_nan_masked_samples_survive_the_load(tmp_path: Path) -> None:
    """Hard invariant 1 across the boundary: NaN in, NaN out, never zero."""
    y = synthetic_samples()
    y[1000:1500, :] = np.nan
    path = write_flat_h5(tmp_path / "masked.h5", y)

    loaded = load_recording(path, animal="J", channel_map=new_cohort_map("J"))

    assert masked_count(loaded.recording.data) == 500 * y.shape[1]
    assert_no_zero_runs(loaded.recording.data, what="loaded data", fs=FS)
