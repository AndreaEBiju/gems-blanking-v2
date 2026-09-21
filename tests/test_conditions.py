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
        assert result.condition == "unknown", stem
        assert result.condition != "baseline", stem
        assert result.needs_a_human


def test_priority_puts_stim_rec_before_stim(rules: Rules) -> None:
    """``X_stim_rec_01`` is stim_recovery, not stim.

    The current code gets this right only by the accident of testing ``_stim_rec``
    first with an ``in``; here it is explicit priority, so file order cannot break it.
    """
    result = rules.classify("X_stim_rec_01")
    assert result.condition == "stim_recovery"
    assert result.matched_rule == "stim_rec"

    assert rules.classify("X_stim_01").condition == "stim"


def test_equal_priority_disagreement_is_ambiguous() -> None:
    """A tie is reported, never broken arbitrarily."""
    rules = Rules(
        vocabulary=("baseline", "stim", "unknown"),
        rules=(
            ConditionRule(id="a", pattern="_x", condition="baseline", priority=10),
            ConditionRule(id="b", pattern="_x", condition="stim", priority=10),
        ),
    )
    result = rules.classify("rec_x_01")
    assert result.status == "ambiguous"
    assert result.condition == "unknown"
    assert result.candidates == ("baseline", "stim")
    assert result.needs_a_human


def test_equal_priority_agreement_is_not_ambiguous() -> None:
    """Two rules proposing the *same* condition is not a conflict."""
    rules = Rules(
        vocabulary=("baseline", "unknown"),
        rules=(
            ConditionRule(id="a", pattern="_x", condition="baseline", priority=10),
            ConditionRule(id="b", pattern="_x_", condition="baseline", priority=10),
        ),
    )
    result = rules.classify("rec_x_01")
    assert result.status == "matched"
    assert result.condition == "baseline"


# ---------------------------------------------------------------------------
# the closed vocabulary
# ---------------------------------------------------------------------------


def test_a_condition_outside_the_vocabulary_is_rejected_at_write_time(rules: Rules) -> None:
    """'stim', 'Stim' and 'stimulation' as three levels would wreck the models."""
    assert rules.validate_condition("stim") == "stim"
    for bad in ("stimulation", "Stim", "STIM", "baseline2", ""):
        with pytest.raises(ValueError, match="closed vocabulary"):
            rules.validate_condition(bad)


def test_a_rule_proposing_an_unknown_condition_fails_to_load() -> None:
    with pytest.raises(ValueError, match="closed vocabulary"):
        Rules(
            vocabulary=("baseline", "unknown"),
            rules=(ConditionRule(id="a", pattern="_x", condition="stimulation", priority=1),),
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
                ConditionRule(id="same", pattern="_a", condition="baseline", priority=1),
                ConditionRule(id="same", pattern="_b", condition="baseline", priority=2),
            ),
        )


def test_an_invalid_pattern_is_rejected_with_the_rule_id() -> None:
    with pytest.raises(ValueError, match="invalid pattern"):
        Rules(
            vocabulary=("baseline", "unknown"),
            rules=(
                ConditionRule(id="bad", pattern="_(unclosed", condition="baseline", priority=1),
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
    assert rules.classify(stem).condition == "stim_recovery"


@pytest.mark.parametrize(
    "suffix", ["_notched", "_notchblanked", "_blankmotion", "_NOTCHED"]
)
def test_processing_suffixes_are_stripped_before_matching(rules: Rules, suffix: str) -> None:
    assert rules.classify(f"E1000_JEL_E1000_bl_1315{suffix}").condition == "baseline"
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

    conditions = {s: c.condition for s, c in statuses.items()}
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
                condition="stim_recovery",
                priority=10,
            ),
            ConditionRule(id="stim", pattern=r"_stim(\b|_)", condition="stim", priority=20),
            ConditionRule(
                id="base",
                pattern=r"_base(line)?(\b|_)",
                condition="baseline",
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
    assert rules.classify("E1000_JEL_E1000_bl_1315").condition == "baseline"


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
    with pytest.raises(ValueError, match="non-empty `vocabulary`"):
        load_rules(path)

    path.write_text("vocabulary: [unknown]\nrules: 3\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="`rules` must be a list"):
        load_rules(path)

    path.write_text(
        "vocabulary: [unknown]\nrules:\n  - {condition: unknown}\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="missing"):
        load_rules(path)

    path.write_text("vocabulary: [unknown]\nrules: [[]]\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_rules(path)


# ---------------------------------------------------------------------------
# the learning loop
# ---------------------------------------------------------------------------


def test_a_proposed_rule_reports_its_blast_radius_before_saving(rules: Rules) -> None:
    """The loop shows how many other scanned rows a new rule would also match."""
    also = rules.would_match(REAL_IDS, r"_mecfreq", "drug")
    assert also == []

    also = rules.would_match(REAL_IDS, r"^mecfreq", "drug")
    assert len(also) == 3
    assert all(s.startswith("mecfreq") for s in also)


def test_a_proposed_rule_with_an_invalid_condition_is_refused(rules: Rules) -> None:
    with pytest.raises(ValueError, match="closed vocabulary"):
        rules.would_match(REAL_IDS, r"_x", "stimulation")
