"""The acquisition record: start time, block identity, store key, local date."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from gems_blanking_v2.io.tdt_block import (
    SESSION_KEY_SUFFIX,
    AcquisitionRecordError,
    acquired_at,
    block_name,
    directory_disagrees,
    local_date,
    session_key,
)

from tests.conftest import write_tdt_block

EVENING_EDT = 1_789_527_801.0
"""2026-09-16T03:03:21Z - 23:03:21 EDT on the 15th, the invariant-31 example."""

ONE_DAY_S = 86_400.0


def _recording(block_dir: Path, suffix: str = "_sig.mat") -> Path:
    path = block_dir / f"{block_dir.name}{suffix}"
    path.write_bytes(b"")
    return path


def test_the_start_time_is_read_from_the_tsq_in_utc(tmp_path: Path) -> None:
    block = write_tdt_block(tmp_path, "gems_j_t01_ms3_bl_230315", EVENING_EDT)
    assert acquired_at(_recording(block)) == "2026-09-16T03:03:21+00:00"


def test_no_tsq_means_no_start_time_and_no_key(tmp_path: Path) -> None:
    path = tmp_path / "gems_j_t01_ms3_bl_230315_sig.mat"
    path.write_bytes(b"")
    assert acquired_at(path) is None
    assert block_name(path) is None
    assert session_key(path) is None


def test_the_key_is_the_block_plus_its_utc_start(tmp_path: Path) -> None:
    block = write_tdt_block(tmp_path, "gems_j_t01_ms3_bl_230315", EVENING_EDT)
    assert session_key(_recording(block)) == "gems_j_t01_ms3_bl_230315_20260916T030321Z"


def test_every_file_of_one_block_shares_its_key(tmp_path: Path) -> None:
    """_sig, _notched, _stim: one acquisition, one meta.json - not three."""
    block = write_tdt_block(tmp_path, "gems_a_pre01_161443", EVENING_EDT)
    keys = {session_key(_recording(block, s)) for s in ("_sig.mat", "_notched.mat", "_stim.mat")}
    assert len(keys) == 1


def test_a_copied_block_collapses_onto_the_same_key(tmp_path: Path) -> None:
    """The 27 duplicates: same block, same instrument start, filed under two folders."""
    a = write_tdt_block(tmp_path / "09032026", "gems_b_t01_bl_101500", EVENING_EDT)
    b = write_tdt_block(tmp_path / "09042026", "gems_b_t01_bl_101500", EVENING_EDT)
    assert session_key(_recording(a)) == session_key(_recording(b))


def test_same_block_name_on_different_days_gets_different_keys(tmp_path: Path) -> None:
    """The collision the dateless stem key could not prevent."""
    a = write_tdt_block(tmp_path / "d1", "gems_b_t01_bl_101500", EVENING_EDT)
    b = write_tdt_block(tmp_path / "d2", "gems_b_t01_bl_101500", EVENING_EDT + ONE_DAY_S)
    assert session_key(_recording(a)) != session_key(_recording(b))


def test_renaming_the_directory_does_not_move_the_key(tmp_path: Path) -> None:
    """37 real blocks were renamed after acquisition (_INCOMPLETE, _BAD, edits).

    The key is read from the instrument, so annotating a folder cannot orphan the
    recording's meta.json; the rename is reported instead.
    """
    block = write_tdt_block(tmp_path, "gems_d_t02_es1_sr_211947", EVENING_EDT)
    before = session_key(_recording(block))
    renamed = block.rename(block.parent / "gems_d_t02_es1_sr_211947_INCOMPLETE")
    rec = _recording(renamed)
    assert session_key(rec) == before
    assert directory_disagrees(rec) == (
        "gems_d_t02_es1_sr_211947", "gems_d_t02_es1_sr_211947_INCOMPLETE")


def test_an_unrenamed_block_reports_no_disagreement(tmp_path: Path) -> None:
    block = write_tdt_block(tmp_path, "gems_d_t02_es1_sr_211947", EVENING_EDT)
    assert directory_disagrees(_recording(block)) is None


def test_a_tsq_without_a_tank_stamp_is_refused(tmp_path: Path) -> None:
    block = write_tdt_block(tmp_path, "gems_b_t01_bl_101500", EVENING_EDT, tank="notank")
    with pytest.raises(AcquisitionRecordError, match="cannot read a block name"):
        session_key(_recording(block))


def test_two_tsq_files_in_one_directory_are_refused(tmp_path: Path) -> None:
    block = write_tdt_block(tmp_path, "gems_b_t01_bl_101500", EVENING_EDT)
    write_tdt_block(tmp_path, "gems_b_t01_bl_101500", EVENING_EDT, tank="OTHER-260101-000000")
    with pytest.raises(AcquisitionRecordError, match=r"2 \.tsq"):
        session_key(_recording(block))


def test_the_key_contains_no_character_illegal_on_windows(tmp_path: Path) -> None:
    block = write_tdt_block(tmp_path, "gems_j_t01_ms3_bl_230315", EVENING_EDT)
    key = session_key(_recording(block))
    assert key is not None
    assert not set(key) & set('<>:"/\\|?*')


@pytest.mark.parametrize(
    ("utc", "expected"),
    [
        ("2026-09-16T03:03:21+00:00", date(2026, 9, 15)),  # 23:03 EDT (-4): previous day
        ("2026-12-16T04:03:21+00:00", date(2026, 12, 15)),  # 23:03 EST (-5): previous day
        ("2026-12-16T05:03:21+00:00", date(2026, 12, 16)),  # 00:03 EST: same day
    ],
)
def test_local_date_follows_the_zone_across_dst(utc: str, expected: date) -> None:
    """A fixed -4 would put the December 04:03Z recording on the 16th. The zone does not."""
    assert local_date(utc, "America/New_York") == expected


def test_the_suffix_pattern_recognises_exactly_what_session_key_builds(tmp_path: Path) -> None:
    """One format, recognised where it is built - not re-derived by a second regex."""
    block = write_tdt_block(tmp_path, "gems_j_t01_ms3_bl_230315", EVENING_EDT)
    key = session_key(_recording(block))
    assert key is not None
    assert SESSION_KEY_SUFFIX.search(key)
    assert not SESSION_KEY_SUFFIX.search("gems_j_t01_ms3_bl_230315_sig")
