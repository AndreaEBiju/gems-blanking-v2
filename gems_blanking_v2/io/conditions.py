"""Condition inference from a filename, against a shared versioned rule set.

The rule this task exists to remove is in ``detector-pyqt``
``ui/windows/training_window.py:216``::

    if "_stim_rec" in core:                       rec_type = "stim_rec"
    elif "_bl_" in core or core.endswith("_bl"):  rec_type = "baseline"
    else:                                         rec_type = "baseline"

The ``elif`` and the ``else`` produce the same value, so **every unrecognised
filename becomes a control**. Condition is an independent variable in
``bulk_mixed_models.m``, so that does not fail loudly; it shifts an effect
estimate. Here an unmatched name is ``unknown`` and blocks until a human resolves
it.

Three states, and only one of them proposes anything:

``matched``
    Exactly one rule won on priority. Proposed, and the rule id is recorded so a
    later correction can flag the rule that got it wrong.
``ambiguous``
    Two or more rules of equal priority disagreed. Resolved by hand.
``unknown``
    Nothing matched. Resolved by hand, and **never defaulted**.

Patterns compile with ``re.IGNORECASE`` (cross-platform rule 7), which the real
data requires: ``E100_lol_E100_stim_rec_1122`` and
``M100E10_LOL_CME2_stim_rec_2253`` are the same convention in different case.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import yaml

__all__ = [
    "CONDITIONS_FILENAME",
    "DEFAULT_CONDITIONS_YAML",
    "Classification",
    "ClassificationStatus",
    "ConditionRule",
    "Rules",
    "animal_from_stem",
    "default_rules",
    "load_rules",
    "write_rules",
]

CONDITIONS_FILENAME: Final = "conditions.yaml"
"""Lives at ``<gems_root>/conditions.yaml`` so every lab member parses identically."""

ClassificationStatus = Literal["matched", "ambiguous", "unknown"]
"""What a filename alone could establish. Scan adds ``duplicate`` and ``known``."""

UNKNOWN_CONDITION: Final = "unknown"
"""The condition of a recording nobody has classified. Must be in the vocabulary."""

DEFAULT_CONDITIONS_YAML: Final = """\
# Condition inference rules, shared through the Drive so every lab member parses
# identically. Edited by hand or appended to by the learning loop.
#
# vocabulary is a CLOSED list. Free text is not allowed: `stim`, `Stim` and
# `stimulation` as three levels would quietly wreck the mixed models. Adding a
# level is a deliberate edit here.
vocabulary: [baseline, stim, stim_recovery, sham, drug, unknown]

# Tried in priority order; the first match wins and its id is recorded.
# `_stim_rec` must be tried before `_stim`, which is why priority is explicit
# rather than left to file order.
rules:
  - id: stim_rec
    pattern: '_stim_rec(\\b|_)'
    condition: stim_recovery
    priority: 10
  - id: stim
    pattern: '_stim(\\b|_)'
    condition: stim
    priority: 20
  # MEASURED 2026-09-21: this lab's baseline convention is `_bl_`, not `_base`.
  # All four baseline recordings among the 14 real ids recovered from the LORO
  # summaries are `_bl_`; the rule set in IMPLEMENTATION.md matched none of them.
  - id: baseline_bl
    pattern: '_bl(\\b|_)'
    condition: baseline
    priority: 30
  - id: baseline_word
    pattern: '_base(line)?(\\b|_)'
    condition: baseline
    priority: 40
  - id: sham
    pattern: '_sham(\\b|_)'
    condition: sham
    priority: 50

strip_suffixes: ['_notched', '_notchblanked', '_blankmotion']
"""


def animal_from_stem(stem: str) -> str | None:
    r"""Return the animal letter, or None when the name does not carry one.

    Delegates to ``GEMSBlanking:detector/animal_id.extract_animal_letter``, which
    takes the first character of the second underscore-separated token and
    upper-cases it - ``E1000_FRE_E1000_bl_1315`` is animal ``F``. That module is
    deliberately dependency-free, so importing it costs nothing.

    **The ``animal_pattern`` regex in ``IMPLEMENTATION.md`` is not used**, because
    it disagrees with that extractor on every real recording id: measured
    2026-09-21, 14 of 14. ``(?<![A-Za-z])([A-Z])(?=[_\d])`` takes the first
    capital before a digit, which is ``E`` in ``E1000_FRE_...`` and ``None`` for
    the lowercase-led ``einh_fre_...``. Animal is the grouping variable for LORO
    and for per-animal models, so getting it wrong is not cosmetic.

    Returns None rather than raising when the checkout is unavailable, so a scan
    still classifies conditions on a machine without the private repository.
    """
    # Imported here, not at module scope: the checkout is private and optional,
    # and this module must stay importable on a machine that lacks it.
    from gems_blanking_v2.io.detector_core import import_detector_module  # noqa: PLC0415

    try:
        animal_id = import_detector_module("animal_id")
    except (FileNotFoundError, ImportError):
        return None
    letter: str | None = animal_id.extract_animal_letter(stem)
    return letter


@dataclass(frozen=True, slots=True)
class ConditionRule:
    """One filename rule.

    Attributes
    ----------
    id
        Stable identifier, recorded on every match so a correction can flag the
        rule that proposed the wrong answer.
    pattern
        Regex, matched case-insensitively against the stripped stem.
    condition
        The condition it proposes. Must be in the vocabulary.
    priority
        Lower wins. Equal priority with different conditions is ``ambiguous``.
    """

    id: str
    pattern: str
    condition: str
    priority: int

    @property
    def regex(self) -> re.Pattern[str]:
        """The compiled pattern. Case-insensitive, per cross-platform rule 7."""
        return re.compile(self.pattern, re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Classification:
    """What the rules could establish from a filename alone."""

    condition: str
    status: ClassificationStatus
    matched_rule: str | None = None
    candidates: tuple[str, ...] = ()
    """Conditions that tied, for ``ambiguous``. Empty otherwise."""

    @property
    def needs_a_human(self) -> bool:
        """Whether this recording is blocked from entering a corpus."""
        return self.status != "matched"


@dataclass(frozen=True, slots=True)
class Rules:
    """A parsed, validated rule set.

    Attributes
    ----------
    vocabulary
        The closed list of conditions. Nothing outside it can be proposed or
        written.
    rules
        In no particular order; :meth:`classify` sorts by priority.
    strip_suffixes
        Processing suffixes removed before matching, longest first.
    source
        Where this came from, for provenance.
    """

    vocabulary: tuple[str, ...]
    rules: tuple[ConditionRule, ...]
    strip_suffixes: tuple[str, ...] = ()
    source: str = "<default>"
    animal_from: Callable[[str], str | None] = field(default=animal_from_stem, repr=False)
    """How the animal letter is derived. Injectable so a scan is testable without
    the private checkout."""

    def __post_init__(self) -> None:
        """Validate the rule set before anything relies on it."""
        if UNKNOWN_CONDITION not in self.vocabulary:
            msg = f"vocabulary must contain {UNKNOWN_CONDITION!r}, got {list(self.vocabulary)}"
            raise ValueError(msg)
        ids = [r.id for r in self.rules]
        if len(set(ids)) != len(ids):
            msg = f"rule ids must be unique, got {ids}"
            raise ValueError(msg)
        for rule in self.rules:
            if rule.condition not in self.vocabulary:
                msg = (
                    f"rule {rule.id!r} proposes condition {rule.condition!r}, which is "
                    f"not in the closed vocabulary {list(self.vocabulary)}. Add the level "
                    "to `vocabulary` deliberately, or fix the rule."
                )
                raise ValueError(msg)
            try:
                re.compile(rule.pattern, re.IGNORECASE)
            except re.error as exc:
                msg = f"rule {rule.id!r} has an invalid pattern {rule.pattern!r}: {exc}"
                raise ValueError(msg) from exc

    def core_of(self, stem: str) -> str:
        """Strip processing suffixes from a stem, longest suffix first."""
        core = stem
        for suffix in sorted(self.strip_suffixes, key=len, reverse=True):
            if core.lower().endswith(suffix.lower()):
                core = core[: -len(suffix)]
                break
        return core

    def classify(self, stem: str) -> Classification:
        """Classify one filename stem. Never defaults to a condition.

        Rules are tried in ascending ``priority``; the first priority level with a
        match decides. If that level proposes more than one distinct condition the
        result is ``ambiguous`` - the tie is reported, not broken arbitrarily.
        """
        core = self.core_of(stem)
        hits = [r for r in self.rules if r.regex.search(core)]
        if not hits:
            return Classification(condition=UNKNOWN_CONDITION, status="unknown")

        best = min(r.priority for r in hits)
        level = [r for r in hits if r.priority == best]
        conditions = {r.condition for r in level}
        if len(conditions) > 1:
            return Classification(
                condition=UNKNOWN_CONDITION,
                status="ambiguous",
                candidates=tuple(sorted(conditions)),
            )
        winner = sorted(level, key=lambda r: r.id)[0]
        return Classification(
            condition=winner.condition, status="matched", matched_rule=winner.id
        )

    def animal(self, stem: str) -> str | None:
        """Return the animal letter for a stem, or None if it cannot be derived."""
        return self.animal_from(self.core_of(stem))

    def validate_condition(self, condition: str) -> str:
        """Return ``condition`` if the vocabulary allows it, else raise.

        Called at write time as well as at load time: a correction typed by a user
        is exactly where a free-text level would otherwise get in.
        """
        if condition not in self.vocabulary:
            msg = (
                f"condition {condition!r} is not in the closed vocabulary "
                f"{list(self.vocabulary)}. Free-text conditions are not allowed - "
                "'stim', 'Stim' and 'stimulation' as three levels would wreck the "
                "mixed models. Add the level to conditions.yaml deliberately."
            )
            raise ValueError(msg)
        return condition

    def would_match(self, stems: Iterable[str], pattern: str, condition: str) -> list[str]:
        """Return the stems a proposed rule would also match.

        The learning loop shows this **before** saving a new rule, so the blast
        radius is visible rather than discovered on the next scan.
        """
        self.validate_condition(condition)
        regex = re.compile(pattern, re.IGNORECASE)
        return [s for s in stems if regex.search(self.core_of(s))]


def _rules_from_document(document: object, source: str) -> Rules:
    """Build :class:`Rules` from a parsed YAML document.

    ``document`` is typed ``object`` because ``yaml.safe_load`` returns whatever the
    file held - a list, a string, ``None`` - and the shape has to be checked rather
    than assumed.
    """
    if not isinstance(document, dict):
        msg = f"{source} must contain a mapping at the top level"
        raise ValueError(msg)
    vocabulary = document.get("vocabulary")
    if not isinstance(vocabulary, list) or not vocabulary:
        msg = f"{source} must define a non-empty `vocabulary` list"
        raise ValueError(msg)

    raw_rules = document.get("rules") or []
    if not isinstance(raw_rules, list):
        msg = f"{source}: `rules` must be a list"
        raise ValueError(msg)

    rules: list[ConditionRule] = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            msg = f"{source}: rule {index} must be a mapping, got {type(raw).__name__}"
            raise ValueError(msg)
        missing = {"pattern", "condition"} - set(raw)
        if missing:
            msg = f"{source}: rule {index} is missing {sorted(missing)}"
            raise ValueError(msg)
        rules.append(
            ConditionRule(
                id=str(raw.get("id") or f"rule_{index}"),
                pattern=str(raw["pattern"]),
                condition=str(raw["condition"]),
                priority=int(raw.get("priority", 100)),
            )
        )

    suffixes = document.get("strip_suffixes") or []
    if not isinstance(suffixes, list):
        msg = f"{source}: `strip_suffixes` must be a list"
        raise ValueError(msg)

    return Rules(
        vocabulary=tuple(str(v) for v in vocabulary),
        rules=tuple(rules),
        strip_suffixes=tuple(str(s) for s in suffixes),
        source=source,
    )


def load_rules(path: Path) -> Rules:
    """Load and validate ``conditions.yaml``.

    Raises
    ------
    FileNotFoundError
        If the file is absent. It is not defaulted: a rule set that silently
        appeared would classify differently on two machines, which is the whole
        problem the shared file solves.
    ValueError
        If the document is malformed, a rule is incomplete, a pattern does not
        compile, or a rule proposes a condition outside the vocabulary.
    """
    path = Path(path)
    if not path.is_file():
        msg = (
            f"no rule set at {path}. Write one with write_rules(default_rules(), path) "
            "and commit it to the shared drive so everyone parses identically."
        )
        raise FileNotFoundError(msg)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        msg = f"{path} could not be parsed: {exc}"
        raise ValueError(msg) from exc
    return _rules_from_document(document or {}, str(path))


def default_rules() -> Rules:
    """Return the starting rule set, including the measured ``_bl`` convention."""
    document = yaml.safe_load(DEFAULT_CONDITIONS_YAML)
    return _rules_from_document(document, "<default>")


def write_rules(rules: Rules, path: Path) -> Path:
    r"""Write a rule set atomically as UTF-8 with ``\n`` endings.

    Round-trips through :func:`load_rules`, so a written file is always loadable.
    """
    document = {
        "vocabulary": list(rules.vocabulary),
        "rules": [
            {
                "id": r.id,
                "pattern": r.pattern,
                "condition": r.condition,
                "priority": r.priority,
            }
            for r in sorted(rules.rules, key=lambda r: (r.priority, r.id))
        ],
        "strip_suffixes": list(rules.strip_suffixes),
    }
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path
