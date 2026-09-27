"""Quality flags, session pairing and exclusion - the rule Andrea ruled on 2026-09-26."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from gems_blanking_v2.io.channel_map import save_geometry
from gems_blanking_v2.io.quality import (
    ExclusionPlan,
    Pairing,
    PairKey,
    duration_verdict,
    pair_key,
    pair_sessions,
    plan_exclusions,
    quality_flags,
    read_exclusion,
)
from gems_blanking_v2.io.store import GemsStore

from tests.test_channel_map import new_cohort_map


@pytest.mark.parametrize(
    ("folder", "flags"),
    [
        ("gems_i_t01_es1_BAD_bl_162155", ("BAD",)),
        ("gems_i_t02_es3_INCOMPLETE_sr_170422", ("INCOMPLETE",)),
        ("gems_b_t02_3_1_sr_221024_incomplete", ("INCOMPLETE",)),
        ("gems_i_t02_ms1_incomplete_sr_234819", ("INCOMPLETE",)),
        ("gems_j_t01_ms3_bl_230315", ()),
        ("gems_j_t01_badger_bl_230315", ()),  # a whole token, not a substring
    ],
)
def test_flags_are_whole_tokens_in_any_case(folder: str, flags: tuple[str, ...]) -> None:
    assert quality_flags(folder) == flags


def test_the_pair_key_ignores_flags_time_and_infix() -> None:
    bl = pair_key("gems_d_t02_3_2_bl_205035_INCOMPLETE")
    sr = pair_key("gems_d_t02_3_2_sr_210046_INCOMPLETE")
    assert bl == (PairKey("D", "t02", "3_2"), "bl")
    assert sr == (PairKey("D", "t02", "3_2"), "sr")


def test_a_name_outside_the_convention_has_no_pair_key() -> None:
    assert pair_key("App_ms10_1_bl_001835") is None


T0 = datetime(2026, 9, 16, 21, 0, tzinfo=UTC)


def _at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _plan(folders: dict[str, str], minutes: dict[str, float]) -> ExclusionPlan:
    """Andrea's window, 2026-09-26: baseline first, 5-20 min."""
    return plan_exclusions(folders, {k: _at(v) for k, v in minutes.items()},
                           gap_lo_min=5.0, gap_hi_min=20.0)


def test_a_flagged_recording_takes_its_confirmed_partner_with_it() -> None:
    plan = _plan(
        {"k_bl": "gems_j_t02_ms1_bl_171800", "k_sr": "gems_j_t02_ms1_sr_172811_incomplete",
         "k_other": "gems_j_t02_ms3_bl_203840"},
        {"k_bl": 0, "k_sr": 10.2, "k_other": 200},
    )
    assert plan.excluded["k_sr"]["flags"] == ["INCOMPLETE"]
    assert plan.excluded["k_bl"]["partner_of"] == "k_sr"
    assert "10.2 min" in plan.excluded["k_bl"]["source"]
    assert "k_other" not in plan.excluded
    assert not (plan.ambiguous or plan.unconfirmed or plan.missing)


def test_the_window_resolves_a_name_ambiguity() -> None:
    """The 11 real cases: two name matches, exactly one 10.1-10.2 min away."""
    plan = _plan(
        {"sr": "gems_d_t02_es1_sr_211947_INCOMPLETE",
         "bl_early": "gems_d_t02_es1_bl_200359", "bl_adjacent": "gems_d_t02_es1_bl_210933"},
        {"sr": 75.9, "bl_early": 0.0, "bl_adjacent": 65.7},
    )
    assert plan.excluded["bl_adjacent"]["partner_of"] == "sr"
    assert "bl_early" not in plan.excluded
    assert not plan.ambiguous


@pytest.mark.parametrize("gap", [105.3, 3.0])
def test_a_unique_name_match_outside_the_window_is_not_a_pair(gap: float) -> None:
    """Andrea on the 105.3 min gap: 'likely wrong'. 3 min is too soon for a baseline."""
    plan = _plan({"bl": "gems_d_t02_3_3_bl_231755_INCOMPLETE", "sr": "gems_d_t02_3_3_sr_010312"},
                 {"bl": 0.0, "sr": gap})
    assert set(plan.excluded) == {"bl"}
    assert plan.unconfirmed == {"bl": [("sr", gap)]}


def test_a_14_7_minute_gap_is_inside_the_window() -> None:
    """The window is 5-20 min, so a lone candidate 14.7 min before the sr pairs.

    Not a statement about gems_a_t01_ms2: the "14.7 min pair is real" answer that
    first motivated the wide window was given without knowing a full baseline sat
    10.1 min before that sr, and is superseded (Andrea, 2026-09-26) - see
    test_a_flagged_start_beside_a_full_baseline_is_superseded for that session.
    """
    plan = _plan({"bl": "gems_c_t01_ms2_bl_120000_incomplete", "sr": "gems_c_t01_ms2_sr_121442"},
                 {"bl": 0.0, "sr": 14.7})
    assert plan.excluded["sr"]["partner_of"] == "bl"
    assert not plan.unconfirmed


@pytest.mark.parametrize("long_name", ["stim_recovery", "stim_rec", "STIM_RECOVERY"])
def test_the_long_condition_name_pairs_like_sr(long_name: str) -> None:
    """One condition, three spellings: older files write the long name."""
    assert pair_key(f"gems_d_t01_3_2_{long_name}_214228") == (PairKey("D", "t01", "3_2"), "sr")
    plan = _plan({"bl": "gems_d_t01_3_2_bl_213219_INCOMPLETE",
                  "sr": f"gems_d_t01_3_2_{long_name}_214228"}, {"bl": 0.0, "sr": 10.1})
    assert plan.excluded["sr"]["partner_of"] == "bl"


def test_baseline_must_come_first() -> None:
    """A stim recording 10 min BEFORE the baseline is not its partner."""
    plan = _plan({"bl": "gems_a_t01_ms2_bl_162459_incomplete", "sr": "gems_a_t01_ms2_sr_163939"},
                 {"bl": 10.0, "sr": 0.0})
    assert "sr" not in plan.excluded
    assert plan.unconfirmed == {"bl": [("sr", -10.0)]}


def test_two_candidates_inside_the_window_are_reported_not_guessed() -> None:
    plan = _plan(
        {"flag": "gems_i_t01_es1_BAD_bl_162155",
         "a": "gems_i_t01_es1_sr_163207", "b": "gems_i_t01_es1_sr_163900"},
        {"flag": 0.0, "a": 10.1, "b": 12.0},
    )
    assert plan.ambiguous == {"flag": ["a", "b"]}
    assert set(plan.excluded) == {"flag"}


def test_no_name_match_is_reported_as_missing() -> None:
    plan = _plan({"flag": "gems_a_t01_ms2_bl_162459_incomplete"}, {"flag": 0.0})
    assert plan.missing == ["flag"]
    assert set(plan.excluded) == {"flag"}


def test_a_partner_that_is_itself_flagged_keeps_its_own_reason() -> None:
    plan = _plan({"bl": "gems_i_t01_es1_BAD_bl_162155", "sr": "gems_i_t01_es1_BAD_sr_163207"},
                 {"bl": 0.0, "sr": 10.1})
    assert plan.excluded["bl"]["reason"] == "quality_flag"
    assert plan.excluded["sr"]["reason"] == "quality_flag"


def test_pairing_reads_the_corrected_folder_name() -> None:
    """Ruling 2: the folder is right. ms3 pairs with ms3, not the block's ms2."""
    plan = _plan(
        {"sr": "gems_i_t01_ms3_sr_231558_incomplete",
         "bl_ms3": "gems_i_t01_ms3_bl_230549", "bl_ms2": "gems_i_t01_ms2_bl_230000"},
        {"sr": 10.2, "bl_ms3": 0.0, "bl_ms2": 5.0},
    )
    assert plan.excluded["bl_ms3"]["partner_of"] == "sr"
    assert "bl_ms2" not in plan.excluded


def test_an_exclusion_is_read_back_from_meta_json(tmp_path: Path) -> None:
    store = GemsStore(tmp_path)
    path = save_geometry(new_cohort_map("J"), "s1", store, mirror_to_profile=False)
    assert read_exclusion(store, "J", "s1") is None
    document = json.loads(path.read_text(encoding="utf-8"))
    document["excluded"] = {"reason": "quality_flag", "flags": ["BAD"]}
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    assert read_exclusion(store, "J", "s1") == {"reason": "quality_flag", "flags": ["BAD"]}


def test_an_unreadable_meta_json_does_not_read_as_admitted(tmp_path: Path) -> None:
    store = GemsStore(tmp_path)
    path = save_geometry(new_cohort_map("J"), "s1", store, mirror_to_profile=False)
    path.write_text("{not json", encoding="utf-8", newline="\n")
    with pytest.raises(json.JSONDecodeError):
        read_exclusion(store, "J", "s1")


# --- completeness by duration (Andrea, 2026-09-26) ------------------------------


@pytest.mark.parametrize(
    ("infix", "minutes", "verdict"),
    [
        ("bl", 10.0, "ok"), ("bl", 9.5, "borderline"), ("bl", 9.0, "borderline"),
        ("bl", 8.9, "short"), ("sr", 22.0, "ok"), ("sr", 20.0, "borderline"),
        ("sr", 19.7, "short"),
    ],
)
def test_the_duration_verdict_has_a_listed_borderline_band(
    infix: str, minutes: float, verdict: str
) -> None:
    """Within 10% below the threshold is listed for Andrea, not decided."""
    assert duration_verdict(infix, minutes, min_baseline_min=10.0, min_sr_min=22.0) == verdict


def test_a_short_recording_is_excluded_with_its_partner() -> None:
    plan = plan_exclusions(
        {"bl": "gems_d_t01_3_2_bl_213219", "sr": "gems_d_t01_3_2_sr_214228"},
        {"bl": _at(0.0), "sr": _at(10.1)}, gap_lo_min=5.0, gap_hi_min=20.0,
        short={"sr": {"duration_min": 12.0, "threshold_min": 22.0}},
    )
    assert plan.excluded["sr"]["reason"] == "duration_below_threshold"
    assert plan.excluded["sr"]["duration_min"] == 12.0
    assert plan.excluded["bl"]["partner_of"] == "sr"


def test_a_flagged_folder_that_meets_the_duration_stays_excluded() -> None:
    """The duration rule adds to the folder flags; it never removes one."""
    plan = plan_exclusions({"bl": "gems_a_t01_ms2_bl_162459_incomplete"}, {"bl": _at(0.0)},
                           gap_lo_min=5.0, gap_hi_min=20.0, short={})
    assert plan.excluded["bl"]["reason"] == "quality_flag"


# --- several baselines: the ~10 min one is the baseline (Andrea, 2026-09-26) ---


def _pair(folders: dict[str, str], minutes: dict[str, float], unusable: set[str]) -> Pairing:
    return pair_sessions(folders, {k: _at(v) for k, v in minutes.items()},
                         gap_lo_min=5.0, gap_hi_min=20.0, unusable=unusable)


def test_the_full_baseline_is_chosen_over_an_aborted_start_in_the_window() -> None:
    """B/t01/es3 as recorded: 4.5 and 0.3 min aborted starts, then the 10 min one."""
    folders = {"a45": "gems_b_t01_es3_bl_233307", "a03": "gems_b_t01_es3_bl_234157",
               "full": "gems_b_t01_es3_bl_234621", "sr": "gems_b_t01_es3_sr_235630"}
    minutes = {"a45": 0.0, "a03": 8.8, "full": 13.2, "sr": 23.4}
    pairing = _pair(folders, minutes, {"a45", "a03"})
    assert pairing.baseline_of == {"sr": "full"}
    assert pairing.in_window == {"sr": ["a03", "full"]}  # a45 is 23.4 min out
    short = {s: {"duration_min": m, "threshold_min": 10.0} for s, m in (("a45", 4.5), ("a03", 0.3))}
    plan = plan_exclusions(folders, {k: _at(v) for k, v in minutes.items()},
                           gap_lo_min=5.0, gap_hi_min=20.0, short=short)
    assert set(plan.excluded) == {"a45", "a03"}  # the sr is NOT taken down
    assert plan.superseded == {"a03": [("sr", "full")]}
    assert plan.unconfirmed == {"a45": [("sr", 23.4)]}


def test_an_sr_whose_only_baseline_is_an_aborted_start_goes_with_it() -> None:
    """D/t01/es3 as recorded: a 0.5 min start 10 min before the sr, nothing else."""
    folders = {"bl": "gems_d_t01_es3_bl_175524", "sr": "gems_d_t01_es3_sr_180543"}
    plan = plan_exclusions(folders, {"bl": _at(0.0), "sr": _at(10.3)}, gap_lo_min=5.0,
                           gap_hi_min=20.0,
                           short={"bl": {"duration_min": 0.5, "threshold_min": 10.0}})
    assert plan.excluded["sr"]["partner_of"] == "bl"
    assert not plan.superseded


def test_a_flagged_start_beside_a_full_baseline_is_superseded() -> None:
    """gems_a_t01_ms2 as recorded: the full baseline is the pair, the start is not.

    Andrea, 2026-09-26: re-admit the sr - it has a full ~10 min baseline 10.1 min
    before it. This supersedes the earlier "14.7 min pair is real" answer, which
    was asked without that information.
    """
    folders = {"inc": "gems_a_t01_ms2_bl_162459_incomplete", "full": "gems_a_t01_ms2_bl_162931",
               "sr": "gems_a_t01_ms2_sr_163939"}
    plan = _plan(folders, {"inc": 0.0, "full": 4.5, "sr": 14.7})
    assert set(plan.excluded) == {"inc"}
    assert plan.superseded == {"inc": [("sr", "full")]}


def test_two_full_baselines_in_the_window_are_ambiguous_not_guessed() -> None:
    folders = {"b1": "gems_i_t01_es1_bl_162155", "b2": "gems_i_t01_es1_bl_163000",
               "sr": "gems_i_t01_es1_BAD_sr_164000"}
    pairing = _pair(folders, {"b1": 0.0, "b2": 8.0, "sr": 18.0}, set())
    assert pairing.ambiguous == {"sr": ["b1", "b2"]} and not pairing.baseline_of
    plan = _plan(folders, {"b1": 0.0, "b2": 8.0, "sr": 18.0})
    assert plan.ambiguous == {"sr": ["b1", "b2"]}
    assert set(plan.excluded) == {"sr"}


def test_trial_tokens_are_not_renamed() -> None:
    """gems_d restarts keep their names: a t03 baseline never pairs with a t01 sr."""
    folders = {"bl": "gems_d_t03_3_2_bl_212333", "sr": "gems_d_t01_3_2_sr_213343"}
    assert not _pair(folders, {"bl": 0.0, "sr": 10.0}, set()).baseline_of


# --- exclusion by a named ruling (Andrea, 2026-09-26: no_stim_monitor) ------

_NO_MONITOR = {"reason": "no_stim_monitor", "source": "all three monitors exactly zero",
               "ruling": "Andrea 2026-09-26: not sure stimulation happened; exclude it "
                         "and its baseline partner"}


def test_a_ruled_exclusion_takes_its_partner_and_names_the_ruling() -> None:
    folders = {"bl": "gems_i_t03_es2_bl_222937", "sr": "gems_i_t03_es2_sr_223947",
               "other": "gems_i_t03_es3_bl_230000"}
    plan = plan_exclusions(folders, {"bl": _at(0.0), "sr": _at(10.2), "other": _at(40.0)},
                           gap_lo_min=5.0, gap_hi_min=20.0, ruled={"sr": dict(_NO_MONITOR)})
    assert plan.excluded["sr"]["reason"] == "no_stim_monitor"
    assert plan.excluded["bl"]["reason"] == "partner_of_flagged"
    assert plan.excluded["bl"]["partner_of"] == "sr"
    assert plan.excluded["bl"]["ruling"] == _NO_MONITOR["ruling"]
    assert "other" not in plan.excluded


def test_a_flag_outranks_a_ruling_but_keeps_it() -> None:
    folders = {"sr": "gems_i_t03_es2_sr_223947_BAD"}
    plan = plan_exclusions(folders, {"sr": _at(0.0)}, gap_lo_min=5.0, gap_hi_min=20.0,
                           ruled={"sr": dict(_NO_MONITOR)})
    assert plan.excluded["sr"]["reason"] == "quality_flag"
    assert plan.excluded["sr"]["also_ruled"]["reason"] == "no_stim_monitor"


def test_a_ruling_that_names_nothing_or_lacks_its_reason_refuses() -> None:
    folders = {"sr": "gems_i_t03_es2_sr_223947"}
    with pytest.raises(KeyError, match="not a known session"):
        plan_exclusions(folders, {"sr": _at(0.0)}, gap_lo_min=5.0, gap_hi_min=20.0,
                        ruled={"ghost": dict(_NO_MONITOR)})
    with pytest.raises(ValueError, match="ruling"):
        plan_exclusions(folders, {"sr": _at(0.0)}, gap_lo_min=5.0, gap_hi_min=20.0,
                        ruled={"sr": {"reason": "no_stim_monitor"}})
