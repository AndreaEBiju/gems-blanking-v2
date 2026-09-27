"""The blind audit's pool: excluded recordings never offered, stim epochs removed."""

from __future__ import annotations

import json
from pathlib import Path

from gems_blanking_v2.io.audit_pool import (
    EligibleRecording,
    assessable_regions,
    eligible_recordings,
    planned_regions,
    stim_epoch_s,
)
from gems_blanking_v2.io.stim_split import default_protocol
from gems_blanking_v2.io.store import GemsStore


def _entry(store: GemsStore, folder: str, *, excluded: bool = False,
           source: bool = True, present: bool = True) -> None:
    animal = folder.split("_")[1].upper()
    session = f"{folder}_20260916T030321Z"
    rel = f"cohort/{folder}/{folder}_sig.mat"
    if present:
        (store.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (store.root / rel).write_bytes(b"")
    doc = {"animal": animal, "session": session, "folder_name": folder,
           "acquired_at": "2026-09-16T03:03:21+00:00", "channels": []}
    if source:
        doc["source_path"] = rel
    if excluded:
        doc["excluded"] = {"reason": "quality_flag", "flags": ["BAD"]}
    meta = store.session_dir(animal, session) / "meta.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(doc), encoding="utf-8", newline="\n")


def test_only_eligible_recordings_are_offered(tmp_path: Path) -> None:
    """Excluded, source-less and not-present recordings never appear - not refused later."""
    store = GemsStore(tmp_path)
    _entry(store, "gems_j_t01_ms3_bl_230315")
    _entry(store, "gems_j_t01_ms3_sr_231323")
    _entry(store, "gems_j_t02_ms1_sr_172811_incomplete", excluded=True)
    _entry(store, "gems_d_t01_3_2_bl_213219", source=False)
    _entry(store, "gems_d_t01_3_2_sr_214228", present=False)
    offered = eligible_recordings(store)
    assert [r.folder_name for r in offered] == [
        "gems_j_t01_ms3_bl_230315", "gems_j_t01_ms3_sr_231323"]
    assert [r.epoch for r in offered] == ["baseline", "stim_recovery"]


def test_the_stim_epoch_is_removed_from_the_assessable_regions() -> None:
    assert assessable_regions(1320.0, (5.0, 128.0)) == ((0.0, 5.0), (128.0, 1320.0))
    assert assessable_regions(600.0, None) == ((0.0, 600.0),)
    assert assessable_regions(1320.0, (0.0, 132.0)) == ((132.0, 1320.0),)


def test_a_stim_recovery_file_loses_the_protocol_stim_window_even_with_a_monitor(
    tmp_path: Path,
) -> None:
    """Measured: a vib-only split found a spurious epoch on an ms file. Not used."""
    source = tmp_path / "gems_j_t01_ms1_sr_165543" / "gems_j_t01_ms1_sr_165543_sig.mat"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    (source.parent / "gems_j_t01_ms1_sr_165543_vib.mat").write_bytes(b"")  # present, ignored
    protocol = default_protocol()
    stim, method = stim_epoch_s(None, source, "stim_recovery", protocol)
    assert method == "protocol_window"
    assert stim == (0.0, protocol.stim_duration_s + protocol.stim_tolerance_s)


def test_a_baseline_has_no_stim_epoch(tmp_path: Path) -> None:
    stim, method = stim_epoch_s(None, tmp_path / "x.mat", "baseline", default_protocol())
    assert stim is None
    assert method == "not_stim_recovery"


def test_audit_marks_live_under_the_session_key(tmp_path: Path) -> None:
    store = GemsStore(tmp_path)
    d = store.audit_dir("J", "gems_j_t01_ms3_bl_230315_20260916T030321Z")
    assert store.relpath(d) == "labels/J/blind_audit/gems_j_t01_ms3_bl_230315_20260916T030321Z"


def test_plan_time_regions_come_from_the_store_alone(tmp_path: Path) -> None:
    """No recording is opened to plan: duration from meta.json, stim by the same rule."""
    protocol = default_protocol()
    src = tmp_path / "x_sig.mat"
    sr = EligibleRecording("J", "k", "gems_j_t01_ms3_sr_231323", src, "stim_recovery", "", 1320.0)
    regions, excluded = planned_regions(sr, protocol)  # type: ignore[misc]
    stim_stop = protocol.stim_duration_s + protocol.stim_tolerance_s
    assert excluded == ((0.0, stim_stop),)
    assert regions == ((stim_stop, 1320.0),)
    bl = EligibleRecording("J", "k", "gems_j_t01_ms3_bl_230315", src, "baseline", "", 600.0)
    assert planned_regions(bl, protocol) == (((0.0, 600.0),), ())


def test_unclassified_or_undated_recordings_are_not_planned(tmp_path: Path) -> None:
    src = tmp_path / "x_sig.mat"
    pre = EligibleRecording("A", "k", "gems_a_pre01_161443", src, "unknown", "", 600.0)
    undated = EligibleRecording("J", "k", "gems_j_t01_ms3_bl_230315", src, "baseline", "", None)
    assert planned_regions(pre, default_protocol()) is None
    assert planned_regions(undated, default_protocol()) is None
