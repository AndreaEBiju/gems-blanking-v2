"""Condition inference: three states, a closed vocabulary, and no silent default.

The measurements in these tests are on **real recording ids**, recovered from the
LORO summaries in ``GEMSBlanking`` (``loro_out``, ``loro_12rec``,
``loro_no_calib``). The 43 recordings themselves live on an unmounted shared
drive, but their names do not have to be invented.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from gems_blanking_v2.io.conditions import (
    DEFAULT_CONDITIONS_YAML,
    ELECTRICAL_AMPLITUDE_UA,
    ELECTRICAL_PULSE_WIDTH_MS,
    ELECTRICAL_WAVEFORM,
    MECHANICAL_DUTY_FRACTION,
    OLD_COHORT_AMPLITUDE_UA,
    Condition,
    ConditionRule,
    Rules,
    default_rules,
    load_rules,
    write_rules,
)

# Real recording ids, 2026-09-21. Four baselines, ten stim-recovery.
REAL_IDS: tuple[str, ...] = (
    "E1000_FRE_E1000_stim_rec_1406",
    "E1000_JEL_E1000_bl_1315",
    "E100_lol_E100_stim_rec_1122",
    "M100E10_LOL_CME2_stim_rec_2253",
    "M100_JEL_MS2_bl_1945",
    "M100_LOL_MS2_stim_rec_2031",
    "M10E10_ORE_CME_stim_rec_1908",
    "einh_fre_ES1_bl_1046",
    "einh_jel_ES1_stim_rec_1309",
    "mec100_jel_MS2_stim_rec_1346",
    "mec100_lol_MS2_stim_rec_1426",
    "mecfreq_fre_MS2_stim_rec_0035",
    "mecfreq_jel_MS2_bl_2302",
    "mecfreq_loll_MS2_stim_rec_2340",
)

REAL_ANIMALS: dict[str, str] = {
    "E1000_FRE_E1000_stim_rec_1406": "F",
    "E1000_JEL_E1000_bl_1315": "J",
    "E100_lol_E100_stim_rec_1122": "L",
    "M100E10_LOL_CME2_stim_rec_2253": "L",
    "M100_JEL_MS2_bl_1945": "J",
    "M100_LOL_MS2_stim_rec_2031": "L",
    "M10E10_ORE_CME_stim_rec_1908": "O",
    "einh_fre_ES1_bl_1046": "F",
    "einh_jel_ES1_stim_rec_1309": "J",
    "mec100_jel_MS2_stim_rec_1346": "J",
    "mec100_lol_MS2_stim_rec_1426": "L",
    "mecfreq_fre_MS2_stim_rec_0035": "F",
    "mecfreq_jel_MS2_bl_2302": "J",
    "mecfreq_loll_MS2_stim_rec_2340": "L",
}
"""What ``extract_animal_letter`` returns: the second underscore token's initial."""


def second_token_initial(stem: str) -> str | None:
    """Return the documented animal letter, without needing the private repo."""
    tokens = stem.split("_")
    if len(tokens) < 2 or not tokens[1] or not tokens[1][0].isalpha():
        return None
    return tokens[1][0].upper()


@pytest.fixture
def rules() -> Rules:
    """Return the default rule set, with animal extraction needing no checkout."""
    base = default_rules()
    return Rules(
        vocabulary=base.vocabulary,
        rules=base.rules,
        estim_hz=base.estim_hz,
        mstim_hz=base.mstim_hz,
        timepoint_pattern=base.timepoint_pattern,
        strip_suffixes=base.strip_suffixes,
        source=base.source,
        animal_from=second_token_initial,
    )


# ---------------------------------------------------------------------------
# the regression guard
# ---------------------------------------------------------------------------


def test_an_unmatched_filename_is_unknown_not_baseline(rules: Rules) -> None:
    """The bug this task removes.

    ``training_window.py:216`` is ``if _stim_rec ... elif _bl_ ... else baseline``,
    where the ``elif`` and the ``else`` produce the same value - so every
    unrecognised name became a control. Condition is an independent variable in
    ``bulk_mixed_models.m``, so that shifts an effect estimate rather than failing.
    """
    for stem in (
        "some_new_protocol_2026",
        "collaborator_file_01",
        "gems_d_t01_ms1_XX_164012",
        "",
    ):
        result = rules.classify(stem)
        assert result.status == "unknown", stem
        assert result.condition.epoch == "unknown", stem
        assert result.condition.epoch != "baseline", stem
        assert result.needs_a_human


def test_both_cohorts_recovery_conventions_resolve(rules: Rules) -> None:
    """``_stim_rec`` is the old cohort's; ``_sr_`` is the new cohort's.

    A rule set with only ``_stim_rec`` - which is what the spec shipped - makes
    every new-cohort recovery file ``unknown``.
    """
    old = rules.classify("E1000_FRE_E1000_stim_rec_1406")
    assert old.condition.epoch == "stim_recovery"
    assert old.matched_rule == "stim_rec_old"

    new = rules.classify("gems_d_t01_es1_sr_204720")
    assert new.condition.epoch == "stim_recovery"
    assert new.matched_rule == "stim_rec_new"


def test_the_longer_recovery_convention_wins_on_priority(rules: Rules) -> None:
    """``_stim_rec`` is tried before ``_sr`` so the short form cannot shadow it.

    A name carrying both resolves through the old rule, by explicit priority rather
    than by file order or by the accident of an ``in`` test.
    """
    both = rules.classify("x_stim_rec_sr_01")
    assert both.condition.epoch == "stim_recovery"
    assert both.matched_rule == "stim_rec_old"


def test_a_bare_stim_token_is_not_an_epoch(rules: Rules) -> None:
    """``stim`` was a level in the single-label vocabulary and is not an epoch now.

    The four-field table has ``baseline`` and ``stim_recovery`` only; a file whose
    name says ``_stim_`` and nothing else does not say which epoch it is.
    """
    assert rules.classify("X_stim_01").condition.epoch == "unknown"
    assert "stim" not in rules.vocabulary


def test_equal_priority_disagreement_is_ambiguous() -> None:
    """A tie is reported, never broken arbitrarily."""
    rules = Rules(
        vocabulary=("baseline", "stim", "unknown"),
        rules=(
            ConditionRule(id="a", pattern="_x", epoch="baseline", priority=10),
            ConditionRule(id="b", pattern="_x", epoch="stim", priority=10),
        ),
    )
    result = rules.classify("rec_x_01")
    assert result.status == "ambiguous"
    assert result.condition.epoch == "unknown"
    assert result.candidates == ("baseline", "stim")
    assert result.needs_a_human


def test_equal_priority_agreement_is_not_ambiguous() -> None:
    """Two rules proposing the *same* condition is not a conflict."""
    rules = Rules(
        vocabulary=("baseline", "unknown"),
        rules=(
            ConditionRule(id="a", pattern="_x", epoch="baseline", priority=10),
            ConditionRule(id="b", pattern="_x_", epoch="baseline", priority=10),
        ),
    )
    result = rules.classify("rec_x_01")
    assert result.status == "matched"
    assert result.condition.epoch == "baseline"


# ---------------------------------------------------------------------------
# the closed vocabulary
# ---------------------------------------------------------------------------


def test_an_epoch_outside_the_vocabulary_is_rejected_at_write_time(rules: Rules) -> None:
    """'stim', 'Stim' and 'stimulation' as three levels would wreck the models."""
    assert rules.validate_epoch("baseline") == "baseline"
    for bad in ("stimulation", "Stim", "STIM", "baseline2", "", "sham", "drug"):
        with pytest.raises(ValueError, match="closed vocabulary"):
            rules.validate_epoch(bad)


def test_a_rule_proposing_an_unknown_condition_fails_to_load() -> None:
    with pytest.raises(ValueError, match="closed vocabulary"):
        Rules(
            vocabulary=("baseline", "unknown"),
            rules=(ConditionRule(id="a", pattern="_x", epoch="stimulation", priority=1),),
        )


def test_the_vocabulary_must_contain_unknown() -> None:
    """``unknown`` is a real level, because it is what an unclassified row carries."""
    with pytest.raises(ValueError, match="must contain 'unknown'"):
        Rules(vocabulary=("baseline",), rules=())


def test_duplicate_rule_ids_are_rejected() -> None:
    """The id is recorded on every match, so it has to identify one rule."""
    with pytest.raises(ValueError, match="unique"):
        Rules(
            vocabulary=("baseline", "unknown"),
            rules=(
                ConditionRule(id="same", pattern="_a", epoch="baseline", priority=1),
                ConditionRule(id="same", pattern="_b", epoch="baseline", priority=2),
            ),
        )


def test_an_invalid_pattern_is_rejected_with_the_rule_id() -> None:
    with pytest.raises(ValueError, match="invalid pattern"):
        Rules(
            vocabulary=("baseline", "unknown"),
            rules=(
                ConditionRule(id="bad", pattern="_(unclosed", epoch="baseline", priority=1),
            ),
        )


# ---------------------------------------------------------------------------
# case, suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem",
    ["X_STIM_REC_01", "X_stim_rec_01", "X_Stim_Rec_01", "E100_lol_E100_stim_rec_1122"],
)
def test_matching_is_case_insensitive(rules: Rules, stem: str) -> None:
    """Cross-platform rule 7, and the real data needs it: ``lol`` beside ``LOL``."""
    assert rules.classify(stem).condition.epoch == "stim_recovery"


@pytest.mark.parametrize(
    "suffix", ["_notched", "_notchblanked", "_blankmotion", "_NOTCHED"]
)
def test_processing_suffixes_are_stripped_before_matching(rules: Rules, suffix: str) -> None:
    assert rules.classify(f"E1000_JEL_E1000_bl_1315{suffix}").condition.epoch == "baseline"
    assert rules.core_of(f"E1000_JEL_E1000_bl_1315{suffix}") == "E1000_JEL_E1000_bl_1315"


def test_only_one_suffix_is_stripped(rules: Rules) -> None:
    """Their loader strips one too; stacking them is not a convention in use."""
    assert rules.core_of("x_bl_notched_blankmotion") == "x_bl_notched"


# ---------------------------------------------------------------------------
# the real recording ids
# ---------------------------------------------------------------------------


def test_every_real_recording_id_classifies(rules: Rules) -> None:
    """All 14 real ids resolve under the shipped rule set. Measured 2026-09-21.

    The rule set in ``IMPLEMENTATION.md`` classified only 10 of 14: its baseline
    rule is ``_base(line)?``, and this lab writes ``_bl_``. The four it missed were
    every baseline in the sample, so the shipped set adds a ``_bl`` rule.
    """
    statuses = {stem: rules.classify(stem) for stem in REAL_IDS}
    unresolved = {s: c.status for s, c in statuses.items() if c.status != "matched"}
    assert unresolved == {}

    conditions = {s: c.condition.epoch for s, c in statuses.items()}
    assert sum(1 for v in conditions.values() if v == "baseline") == 4
    assert sum(1 for v in conditions.values() if v == "stim_recovery") == 10


def test_the_specs_baseline_rule_alone_misses_this_labs_convention() -> None:
    r"""Recorded so the withdrawn rule is not reintroduced.

    ``_base(line)?(\\b|_)`` matches none of ``_bl_``, and every baseline among the
    real ids is ``_bl_``.
    """
    spec_only = Rules(
        vocabulary=("baseline", "stim", "stim_recovery", "unknown"),
        rules=(
            ConditionRule(
                id="stim_rec",
                pattern=r"_stim_rec(\b|_)",
                epoch="stim_recovery",
                priority=10,
            ),
            ConditionRule(id="stim", pattern=r"_stim(\b|_)", epoch="stim", priority=20),
            ConditionRule(
                id="base",
                pattern=r"_base(line)?(\b|_)",
                epoch="baseline",
                priority=30,
            ),
        ),
        strip_suffixes=("_notched", "_notchblanked", "_blankmotion"),
    )
    unknown = [s for s in REAL_IDS if spec_only.classify(s).status == "unknown"]
    assert len(unknown) == 4
    assert all("_bl_" in s for s in unknown)


def test_the_animal_letter_is_the_second_token_not_the_first_capital(rules: Rules) -> None:
    r"""The spec's ``animal_pattern`` disagrees with their extractor on 14 of 14.

    ``(?<![A-Za-z])([A-Z])(?=[_\\d])`` takes the first capital before a digit: ``E``
    in ``E1000_FRE_...``, ``M`` in ``M100_JEL_...``, and nothing at all in the
    lowercase-led ``einh_fre_...``. The animal is the grouping variable for LORO
    and per-animal models, so this is not cosmetic.
    """
    spec_pattern = re.compile(r"(?<![A-Za-z])([A-Z])(?=[_\d])")
    disagreements = 0
    for stem, expected in REAL_ANIMALS.items():
        assert rules.animal(stem) == expected, stem
        match = spec_pattern.search(stem)
        if (match.group(1) if match else None) != expected:
            disagreements += 1
    assert disagreements == len(REAL_ANIMALS)


def test_an_animal_letter_is_none_when_the_name_does_not_carry_one(rules: Rules) -> None:
    """None, never a guess: it decides which animal a recording is grouped under.

    The three cases their docstring lists: fewer than two underscore tokens, an
    empty second token, and a second token that does not start with a letter.
    """
    assert rules.animal("nounderscore") is None  # fewer than two tokens
    assert rules.animal("x__y") is None  # second token empty
    assert rules.animal("x_9digit") is None  # second token starts with a digit


def test_a_leading_underscore_still_yields_the_second_token(rules: Rules) -> None:
    """An oddity of the rule, recorded rather than hidden.

    ``_leading`` splits to ``["", "leading"]``, so the second token is ``leading``
    and the animal is ``L``. That is what their extractor specifies, and a name
    shaped like this would be silently attributed rather than flagged - worth
    knowing before it appears in a filename.
    """
    assert rules.animal("_leading") == "L"


# ---------------------------------------------------------------------------
# the shared file
# ---------------------------------------------------------------------------


def test_the_default_rule_set_round_trips_through_yaml(tmp_path: Path) -> None:
    path = write_rules(default_rules(), tmp_path / "conditions.yaml")
    reloaded = load_rules(path)

    original = default_rules()
    assert reloaded.vocabulary == original.vocabulary
    assert {r.id for r in reloaded.rules} == {r.id for r in original.rules}
    assert reloaded.strip_suffixes == original.strip_suffixes
    for stem in REAL_IDS:
        assert reloaded.classify(stem).condition == original.classify(stem).condition


def test_the_shipped_yaml_text_parses(tmp_path: Path) -> None:
    """The documented default is the one that loads, not an approximation of it."""
    path = tmp_path / "conditions.yaml"
    path.write_text(DEFAULT_CONDITIONS_YAML, encoding="utf-8", newline="\n")
    rules = load_rules(path)
    assert "unknown" in rules.vocabulary
    assert rules.classify("E1000_JEL_E1000_bl_1315").condition.epoch == "baseline"


def test_the_written_file_is_utf8_with_lf(tmp_path: Path) -> None:
    path = write_rules(default_rules(), tmp_path / "conditions.yaml")
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_a_missing_rule_file_is_not_defaulted(tmp_path: Path) -> None:
    """A rule set that silently appeared would classify differently per machine."""
    with pytest.raises(FileNotFoundError, match="no rule set at"):
        load_rules(tmp_path / "absent.yaml")


def test_a_malformed_rule_file_names_the_problem(tmp_path: Path) -> None:
    path = tmp_path / "conditions.yaml"

    path.write_text("vocabulary: []\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="non-empty `epochs`"):
        load_rules(path)

    path.write_text("vocabulary: [unknown]\nrules: 3\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="`epoch_rules` must be a list"):
        load_rules(path)

    path.write_text(
        "vocabulary: [unknown]\nrules:\n  - {condition: unknown}\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="needs both"):
        load_rules(path)

    path.write_text("vocabulary: [unknown]\nrules: [[]]\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_rules(path)


# ---------------------------------------------------------------------------
# the learning loop
# ---------------------------------------------------------------------------


def test_a_proposed_rule_reports_its_blast_radius_before_saving(rules: Rules) -> None:
    """The loop shows how many other scanned rows a new rule would also match."""
    also = rules.would_match(REAL_IDS, r"_mecfreq", "baseline")
    assert also == []

    also = rules.would_match(REAL_IDS, r"^mecfreq", "baseline")
    assert len(also) == 3
    assert all(s.startswith("mecfreq") for s in also)


def test_a_proposed_rule_with_an_invalid_condition_is_refused(rules: Rules) -> None:
    with pytest.raises(ValueError, match="closed vocabulary"):
        rules.would_match(REAL_IDS, r"_x", "stimulation")


# ---------------------------------------------------------------------------
# the four-field record
# ---------------------------------------------------------------------------

NEW_COHORT_IDS: dict[str, Condition] = {
    "gems_d_t01_ms1_bl_164012": Condition("baseline", None, 10.0, "t01"),
    "gems_d_t01_es1_sr_204720": Condition("stim_recovery", 10.0, None, "t01"),
    "gems_d_t02_es3_sr_101500": Condition("stim_recovery", 1000.0, None, "t02"),
    "gems_d_t03_ms3_bl_090000": Condition("baseline", None, 100.0, "t03"),
    "gems_j_t12_ms2_es2_sr_123456": Condition("stim_recovery", 100.0, 50.0, "t12"),
}
"""New-cohort names and the record each must produce.

ES1/2/3 = 10/100/1000 Hz, MS1/2/3 = 10/50/100 Hz.
"""


@pytest.mark.parametrize(("stem", "expected"), list(NEW_COHORT_IDS.items()))
def test_a_new_cohort_name_yields_the_whole_record(
    rules: Rules, stem: str, expected: Condition
) -> None:
    """Epoch, both frequencies and the timepoint, from one filename."""
    result = rules.classify(stem)
    assert result.status == "matched"
    assert result.condition == expected
    assert not result.needs_a_human


def test_both_stimulation_axes_can_be_present_at_once(rules: Rules) -> None:
    """``msX_esY`` means both at once, which a single label cannot express."""
    condition = rules.classify("gems_j_t12_ms2_es2_sr_123456").condition
    assert condition.estim_hz == 100.0
    assert condition.mstim_hz == 50.0
    assert condition.has_stim


def test_the_frequency_axes_are_ordered_numbers(rules: Rules) -> None:
    """Check the axes are ordered numbers rather than unrelated categories.

    Amplitude, pulse width, waveform and duty cycle are held fixed across levels,
    which is what makes ES1->ES2->ES3 one axis.
    """
    estim = [rules.classify(f"gems_d_t01_es{n}_sr_0000").condition.estim_hz for n in (1, 2, 3)]
    assert estim == [10.0, 100.0, 1000.0]
    assert estim == sorted(v for v in estim if v is not None)

    mstim = [rules.classify(f"gems_d_t01_ms{n}_bl_0000").condition.mstim_hz for n in (1, 2, 3)]
    assert mstim == [10.0, 50.0, 100.0]
    assert mstim == sorted(v for v in mstim if v is not None)


def test_a_baseline_with_no_stim_token_has_no_frequency(rules: Rules) -> None:
    """Absent is a fact about the recording, not a gap in the data."""
    condition = rules.classify("gems_d_t01_bl_164012").condition
    assert condition.epoch == "baseline"
    assert condition.estim_hz is None
    assert condition.mstim_hz is None


def test_an_old_cohort_baseline_carries_its_arms_frequency(rules: Rules) -> None:
    """``E1000_JEL_E1000_bl_1315`` is a baseline *and* says 1000 Hz electrical.

    That follows from the ruling that ``E<n>`` is Hz, and it is worth noticing:
    the frequency identifies which protocol arm the baseline belongs to, not that
    stimulation was applied during it. A model treating ``estim_hz`` as an applied
    stimulus would be wrong on these rows.
    """
    condition = rules.classify("E1000_JEL_E1000_bl_1315").condition
    assert condition.epoch == "baseline"
    assert condition.estim_hz == 1000.0


def test_the_fixed_protocol_parameters_are_recorded() -> None:
    """They are what make the two frequencies axes rather than labels."""
    assert ELECTRICAL_AMPLITUDE_UA == 1000.0
    assert ELECTRICAL_PULSE_WIDTH_MS == 0.3
    assert ELECTRICAL_WAVEFORM == "square_bipolar"
    assert MECHANICAL_DUTY_FRACTION == 0.5


def test_a_timepoint_is_normalised_to_two_digits(rules: Rules) -> None:
    assert rules.classify("gems_d_t1_ms1_bl_0000").condition.timepoint == "t01"
    assert rules.classify("gems_d_t01_ms1_bl_0000").condition.timepoint == "t01"
    assert rules.classify("gems_d_t12_ms1_bl_0000").condition.timepoint == "t12"
    assert rules.classify("E1000_JEL_E1000_bl_1315").condition.timepoint is None


def test_the_record_round_trips_through_json(rules: Rules) -> None:
    """Absent, not null: the JSON half of the missing-scalar convention."""
    for stem in NEW_COHORT_IDS:
        condition = rules.classify(stem).condition
        raw = condition.to_json()
        assert "null" not in str(raw)
        assert Condition.from_json(raw) == condition

    sparse = Condition(epoch="baseline")
    assert sparse.to_json() == {"epoch": "baseline"}
    assert Condition.from_json({"epoch": "baseline"}) == sparse
    assert Condition.from_json({"epoch": "baseline", "estim_hz": None}) == sparse


def test_a_frequency_off_the_defined_levels_is_rejected(rules: Rules) -> None:
    """Reject a frequency that is not one of the defined levels.

    The axes are fixed by the protocol, so a value off them is either a mistake or
    a new level to add deliberately.
    """
    rules.validate_condition(Condition("stim_recovery", 100.0, 50.0, "t01"))
    with pytest.raises(ValueError, match="not one of the defined levels"):
        rules.validate_condition(Condition("stim_recovery", 42.0, None, "t01"))
    with pytest.raises(ValueError, match="not one of the defined levels"):
        rules.validate_condition(Condition("baseline", None, 75.0, None))


# ---------------------------------------------------------------------------
# the old cohort's undefined frequency tokens
# ---------------------------------------------------------------------------


def test_the_old_explicit_hz_form_resolves_onto_the_same_axis(rules: Rules) -> None:
    """RESOLVED 2026-09-21: ``E<n>``/``M<n>`` are Hz on the ES/MS axes.

    So the two conventions merge and pooling cohorts on frequency loses nothing.
    These rows used to block; they now parse.
    """
    result = rules.classify("E1000_FRE_E1000_stim_rec_1406")
    assert result.status == "matched"
    assert result.condition.epoch == "stim_recovery"
    assert result.condition.estim_hz == 1000.0
    assert result.unparsed_stim_tokens == ()
    assert not result.needs_a_human


@pytest.mark.parametrize(
    ("stem", "estim", "mstim"),
    [
        ("x_a_E10_bl_1", 10.0, None),
        ("x_a_E100_bl_1", 100.0, None),
        ("x_a_E1000_bl_1", 1000.0, None),
        ("x_a_M10_bl_1", None, 10.0),
        ("x_a_M100_bl_1", None, 100.0),
        ("M100E10_LOL_CME_sr_1", 10.0, 100.0),
    ],
)
def test_the_explicit_hz_table(
    rules: Rules, stem: str, estim: float | None, mstim: float | None
) -> None:
    """E10/E100/E1000 and M10/M100, plus the combined form."""
    condition = rules.classify(stem).condition
    assert condition.estim_hz == estim
    assert condition.mstim_hz == mstim


def test_an_explicit_token_on_no_defined_level_still_blocks(rules: Rules) -> None:
    """``unparsed_stim_tokens`` survives, for tokens genuinely outside both tables."""
    result = rules.classify("X_JEL_E42_bl_01")
    assert result.unparsed_stim_tokens == ("E42",)
    assert result.condition.estim_hz is None
    assert result.needs_a_human


def test_the_old_cohort_amplitude_is_recorded_as_unknown() -> None:
    """The new cohort fixes 1000 uA; the old cohort's filenames encode frequency only.

    None means unknown and is not defaulted to 1000: confirm before pooling the
    cohorts *on amplitude*. Frequency comparisons are unaffected.
    """
    assert OLD_COHORT_AMPLITUDE_UA is None
    assert ELECTRICAL_AMPLITUDE_UA == 1000.0


def test_the_explicit_hz_token_wins_a_conflict_and_the_loser_is_recorded(
    rules: Rules,
) -> None:
    """``M100_JEL_MS2_bl_1945`` says 100 Hz (M100) and 50 Hz (MS2). Explicit wins.

    The precedence rule makes the row parseable; recording both tokens makes the
    choice auditable, and a cluster of these would say the old filenames are less
    trustworthy than they look. It does not block.
    """
    result = rules.classify("M100_JEL_MS2_bl_1945")
    assert result.condition.mstim_hz == 100.0
    assert result.token_conflict == ("M100", "MS2")
    assert result.unparsed_stim_tokens == ()
    assert not result.needs_a_human


def test_agreeing_forms_are_not_a_conflict(rules: Rules) -> None:
    """``M10`` and ``MS1`` both mean 10 Hz, so there is nothing to record."""
    result = rules.classify("M10_JEL_MS1_bl_1945")
    assert result.condition.mstim_hz == 10.0
    assert result.token_conflict == ()


def test_the_combined_form_fills_both_axes(rules: Rules) -> None:
    """``M100E10`` is mechanical 100 Hz **and** electrical 10 Hz."""
    result = rules.classify("M100E10_LOL_CME2_stim_rec_2253")
    assert result.condition.mstim_hz == 100.0
    assert result.condition.estim_hz == 10.0
    assert result.unparsed_stim_tokens == ()
    assert not result.needs_a_human


def test_a_clean_new_cohort_name_has_no_unparsed_tokens(rules: Rules) -> None:
    for stem in NEW_COHORT_IDS:
        assert rules.classify(stem).unparsed_stim_tokens == ()


def test_every_real_old_cohort_id_now_parses(rules: Rules) -> None:
    """Measured after the ruling: 0 of 14 blocked, where 7 blocked before.

    Resolving ``E<n>``/``M<n>`` to Hz merged the two conventions, so the whole
    recovered set is corpus-eligible on its condition.
    """
    blocked = [s for s in REAL_IDS if rules.classify(s).needs_a_human]
    assert blocked == []
    assert all(rules.classify(s).unparsed_stim_tokens == () for s in REAL_IDS)


def test_two_real_ids_carry_a_token_conflict(rules: Rules) -> None:
    """Both are ``M100 … MS2``: 100 Hz explicit against 50 Hz ordinal."""
    conflicted = {s: rules.classify(s).token_conflict for s in REAL_IDS}
    conflicted = {s: c for s, c in conflicted.items() if c}
    assert len(conflicted) == 2
    assert all(c == ("M100", "MS2") for c in conflicted.values())
    assert all(rules.classify(s).condition.mstim_hz == 100.0 for s in conflicted)
