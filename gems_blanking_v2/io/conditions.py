r"""Condition inference: a four-field record, from either cohort's naming convention.

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

**Condition is not one label.** ``ES1/2/3`` and ``MS1/2/3`` are ordered frequency
axes with everything else held fixed, so flattening them into labels would produce
a dozen one-off levels in the mixed models where there is really a small factorial
with numeric parameters. A filename maps to :class:`Condition`: an ``epoch``, an
electrical frequency, a mechanical frequency and a timepoint.

**Two naming conventions, one per cohort**, and a rule set written for either
alone fails on the other:

=================  ==============================  ===============================
\\                 old cohort (the 43)             new cohort (2026 chronic)
=================  ==============================  ===============================
example            ``E1000_FRE_…``, ``einh_fre_…`` ``gems_d_t01_ms1_bl_164012``
stim recovery      ``_stim_rec``                   ``_sr_``
baseline           ``_bl_``                        ``_bl_``
animal             2nd token's initial             2nd token
case               mixed, sometimes lowercase      lowercase
=================  ==============================  ===============================

Every pattern therefore compiles with ``re.IGNORECASE`` (cross-platform rule 7),
and ``_stim_rec`` is tried before ``_sr`` so the longer convention cannot be
shadowed by the shorter one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import yaml

__all__ = [
    "CONDITIONS_FILENAME",
    "DEFAULT_CONDITIONS_YAML",
    "ELECTRICAL_AMPLITUDE_UA",
    "ELECTRICAL_PULSE_WIDTH_MS",
    "ELECTRICAL_WAVEFORM",
    "MECHANICAL_DUTY_FRACTION",
    "Classification",
    "ClassificationStatus",
    "Condition",
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

UNKNOWN_EPOCH: Final = "unknown"
"""The epoch of a recording nobody has classified. Must be in the vocabulary."""

# Held fixed across every electrical level, so ES1->ES2->ES3 is one frequency axis
# rather than three unrelated categories.
ELECTRICAL_AMPLITUDE_UA: Final = 1000.0
"""Electrical stimulation amplitude, microamps. Fixed across ES1-ES3."""

ELECTRICAL_PULSE_WIDTH_MS: Final = 0.3
"""Electrical pulse width, milliseconds. Fixed across ES1-ES3."""

ELECTRICAL_WAVEFORM: Final = "square_bipolar"
"""Electrical waveform. Fixed across ES1-ES3."""

MECHANICAL_DUTY_FRACTION: Final = 0.5
"""Mechanical duty cycle. Fixed across MS1-MS3."""

_STIM_LIKE = re.compile(r"(?<![a-z])([em])(\d{1,4})(?![0-9])", re.IGNORECASE)
# The lookbehind excludes letters but NOT digits, on purpose: in `M100E10` the
# `E10` is preceded by `0`, and excluding digits too would hide half of a name
# that encodes two stimulation parameters. Letters are excluded so `CME2` and
# `mec100` do not produce phantom tokens.
"""A token that looks like a stimulation parameter but is not in either map.

The old cohort writes ``E1000``, ``E100``, ``M100``, ``M10`` and ``M100E10``,
which look like frequencies in Hz but are **not defined** by the ES/MS table. They
are reported rather than interpreted: in ``M100_JEL_MS2_bl_1945`` the recognised
``MS2`` means 50 Hz while an unrecognised ``M100`` sits beside it, and if that
encodes 100 Hz the two disagree. Guessing would quietly merge two conditions.
"""

DEFAULT_CONDITIONS_YAML: Final = """\
# Condition inference, shared through the Drive so every lab member parses
# identically. Edited by hand or appended to by the learning loop.
#
# A filename maps to a four-field RECORD, not to a single label: ES1/2/3 and
# MS1/2/3 are ordered frequency axes with amplitude, pulse width, waveform and
# duty cycle held fixed, so they belong in the model as numbers.

# The closed vocabulary, and it applies to `epoch` only. Free text is not
# allowed: `stim`, `Stim` and `stimulation` as three levels would quietly wreck
# the mixed models. Adding a level is a deliberate edit here.
#
# NOTE: `sham` and `drug` were in the previous vocabulary and are not epochs in
# the four-field table. If either protocol exists, add it here on purpose.
epochs: [baseline, stim_recovery, unknown]

# Tried in priority order; the first level with a match decides, and the rule id
# is recorded. `_stim_rec` (old cohort) must be tried before `_sr` (new cohort),
# which is why priority is explicit rather than left to file order.
epoch_rules:
  - id: stim_rec_old
    pattern: '_stim_rec(\\b|_)'
    epoch: stim_recovery
    priority: 10
  # MEASURED 2026-09-21: the new cohort writes `_sr_`, e.g.
  # gems_d_t01_es1_sr_204720. Without this rule every new-cohort recovery file is
  # `unknown`; the rule set in IMPLEMENTATION.md shipped `_stim_rec` only.
  - id: stim_rec_new
    pattern: '_sr(\\b|_)'
    epoch: stim_recovery
    priority: 15
  # MEASURED 2026-09-21: both cohorts write `_bl_`. The spec's `_base(line)?`
  # matched none of the four baselines among the 14 real old-cohort ids.
  - id: baseline_bl
    pattern: '_bl(\\b|_)'
    epoch: baseline
    priority: 30
  - id: baseline_word
    pattern: '_base(line)?(\\b|_)'
    epoch: baseline
    priority: 40

# Ordered frequency axes, in Hz.
estim_hz: {es1: 10, es2: 100, es3: 1000}
mstim_hz: {ms1: 10, ms2: 50, ms3: 100}

# t01, t02, ... in either case.
timepoint_pattern: '(?<![a-z0-9])t(\\d{1,3})(?![0-9])'

strip_suffixes: ['_notched', '_notchblanked', '_blankmotion']
"""


def animal_from_stem(stem: str) -> str | None:
    r"""Return the animal letter, or None when the name does not carry one.

    Delegates to ``GEMSBlanking:detector/animal_id.extract_animal_letter``, which
    takes the first character of the second underscore-separated token and
    upper-cases it. That works on **both** conventions: ``E1000_FRE_E1000_bl_1315``
    is animal ``F`` and ``gems_d_t01_ms1_bl_164012`` is animal ``D``. The module is
    deliberately dependency-free, so importing it costs nothing.

    **The ``animal_pattern`` regex in the spec is withdrawn**, and was wrong on
    100% of real ids: ``(?<![A-Za-z])([A-Z])(?=[_\d])`` takes the first capital
    before a digit, which is ``E`` in ``E1000_FRE_…`` and nothing at all in the
    lowercase-led ``einh_fre_…``. Animal is the grouping variable for LORO and for
    the per-animal mode.

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
class Condition:
    """One recording's experimental condition, as four fields rather than a label.

    Attributes
    ----------
    epoch
        ``baseline`` or ``stim_recovery``, or ``unknown`` when no rule matched.
    estim_hz
        Electrical stimulation frequency in Hz, or ``None`` when there was none.
        ``None`` is meaningful: a baseline recording legitimately has no stim.
    mstim_hz
        Mechanical stimulation frequency in Hz, or ``None``.
    timepoint
        ``t01``, ``t02``, … or ``None`` when the name does not carry one.

    Amplitude, pulse width, waveform and duty cycle are fixed across levels - see
    :data:`ELECTRICAL_AMPLITUDE_UA` and friends - which is what makes the two
    frequencies ordered axes rather than unrelated categories.
    """

    epoch: str = UNKNOWN_EPOCH
    estim_hz: float | None = None
    mstim_hz: float | None = None
    timepoint: str | None = None

    @property
    def is_unknown(self) -> bool:
        """Whether the epoch could not be established."""
        return self.epoch == UNKNOWN_EPOCH

    @property
    def has_stim(self) -> bool:
        """Whether either stimulation frequency is recorded."""
        return self.estim_hz is not None or self.mstim_hz is not None

    def to_json(self) -> dict[str, Any]:
        """Render for ``meta.json``. A missing field is an **absent key**.

        Per ``CLAUDE.md``: a missing scalar serialised to JSON is absent, never
        ``null`` and never a sentinel. ``estim_hz`` absent means "no electrical
        stimulation", which is a fact about the recording, not a gap in the data.
        """
        out: dict[str, Any] = {"epoch": self.epoch}
        if self.estim_hz is not None:
            out["estim_hz"] = self.estim_hz
        if self.mstim_hz is not None:
            out["mstim_hz"] = self.mstim_hz
        if self.timepoint is not None:
            out["timepoint"] = self.timepoint
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Condition:
        """Read the record back, accepting absent and ``null`` identically."""

        def number(key: str) -> float | None:
            value = raw.get(key)
            return None if value is None else float(value)

        timepoint = raw.get("timepoint")
        return cls(
            epoch=str(raw.get("epoch") or UNKNOWN_EPOCH),
            estim_hz=number("estim_hz"),
            mstim_hz=number("mstim_hz"),
            timepoint=None if timepoint is None else str(timepoint),
        )


@dataclass(frozen=True, slots=True)
class ConditionRule:
    """One filename rule, proposing an ``epoch``.

    Attributes
    ----------
    id
        Stable identifier, recorded on every match so a correction can flag the
        rule that proposed the wrong answer.
    pattern
        Regex, matched case-insensitively against the stripped stem.
    epoch
        The epoch it proposes. Must be in the vocabulary.
    priority
        Lower wins. Equal priority with different epochs is ``ambiguous``.
    """

    id: str
    pattern: str
    epoch: str
    priority: int


@dataclass(frozen=True, slots=True)
class Classification:
    """What the rules could establish from a filename alone."""

    condition: Condition
    status: ClassificationStatus
    matched_rule: str | None = None
    candidates: tuple[str, ...] = ()
    """Epochs that tied, for ``ambiguous``. Empty otherwise."""
    unparsed_stim_tokens: tuple[str, ...] = ()
    """Tokens that look like stimulation parameters but are in neither map.

    Non-empty blocks the row. The old cohort's ``E1000``/``M100`` are exactly this:
    dropping one would silently merge two different stimulation conditions, and
    interpreting one would be a guess.
    """

    @property
    def needs_a_human(self) -> bool:
        """Whether this recording is blocked from entering a corpus."""
        return self.status != "matched" or bool(self.unparsed_stim_tokens)


@dataclass(frozen=True, slots=True)
class Rules:
    """A parsed, validated rule set.

    Attributes
    ----------
    vocabulary
        The closed list of epochs. Nothing outside it can be proposed or written.
    rules
        Epoch rules, in no particular order; :meth:`classify` sorts by priority.
    estim_hz, mstim_hz
        Token to frequency, e.g. ``{"es1": 10.0}``. Ordered axes, so the values are
        numbers.
    timepoint_pattern
        Regex with one group capturing the timepoint digits.
    strip_suffixes
        Processing suffixes removed before matching, longest first.
    source
        Where this came from, for provenance.
    """

    vocabulary: tuple[str, ...]
    rules: tuple[ConditionRule, ...]
    estim_hz: Mapping[str, float] = field(default_factory=dict)
    mstim_hz: Mapping[str, float] = field(default_factory=dict)
    timepoint_pattern: str = r"(?<![a-z0-9])t(\d{1,3})(?![0-9])"
    strip_suffixes: tuple[str, ...] = ()
    source: str = "<default>"
    animal_from: Callable[[str], str | None] = field(default=animal_from_stem, repr=False)
    """How the animal letter is derived. Injectable so a scan is testable without
    the private checkout."""

    def __post_init__(self) -> None:
        """Validate the rule set before anything relies on it."""
        if UNKNOWN_EPOCH not in self.vocabulary:
            msg = f"vocabulary must contain {UNKNOWN_EPOCH!r}, got {list(self.vocabulary)}"
            raise ValueError(msg)
        ids = [r.id for r in self.rules]
        if len(set(ids)) != len(ids):
            msg = f"rule ids must be unique, got {ids}"
            raise ValueError(msg)
        for rule in self.rules:
            if rule.epoch not in self.vocabulary:
                msg = (
                    f"rule {rule.id!r} proposes epoch {rule.epoch!r}, which is not in "
                    f"the closed vocabulary {list(self.vocabulary)}. Add the level to "
                    "`epochs` deliberately, or fix the rule."
                )
                raise ValueError(msg)
            try:
                re.compile(rule.pattern, re.IGNORECASE)
            except re.error as exc:
                msg = f"rule {rule.id!r} has an invalid pattern {rule.pattern!r}: {exc}"
                raise ValueError(msg) from exc
        try:
            re.compile(self.timepoint_pattern, re.IGNORECASE)
        except re.error as exc:
            msg = f"invalid timepoint_pattern {self.timepoint_pattern!r}: {exc}"
            raise ValueError(msg) from exc
        for name, mapping in (("estim_hz", self.estim_hz), ("mstim_hz", self.mstim_hz)):
            for token, hz in mapping.items():
                if not isinstance(hz, int | float) or hz <= 0:
                    msg = f"{name}[{token!r}] must be a positive frequency, got {hz!r}"
                    raise ValueError(msg)

    def core_of(self, stem: str) -> str:
        """Strip processing suffixes from a stem, longest suffix first."""
        core = stem
        for suffix in sorted(self.strip_suffixes, key=len, reverse=True):
            if core.lower().endswith(suffix.lower()):
                core = core[: -len(suffix)]
                break
        return core

    def _frequency(self, core: str, mapping: Mapping[str, float]) -> float | None:
        """Return the frequency a recognised token encodes, or None."""
        for token, hz in mapping.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![0-9])", core, re.IGNORECASE):
                return float(hz)
        return None

    def _timepoint(self, core: str) -> str | None:
        """Return the timepoint as ``t01``, or None."""
        match = re.search(self.timepoint_pattern, core, re.IGNORECASE)
        if match is None:
            return None
        return f"t{int(match.group(1)):02d}"

    def _unparsed_stim_tokens(self, core: str) -> tuple[str, ...]:
        """Return stimulation-looking tokens that neither map defines."""
        known = {t.lower() for t in (*self.estim_hz, *self.mstim_hz)}
        found: list[str] = []
        for match in _STIM_LIKE.finditer(core):
            token = match.group(0)
            if token.lower() not in known:
                found.append(token.upper())
        return tuple(dict.fromkeys(found))

    def classify(self, stem: str) -> Classification:
        """Classify one filename stem into a four-field record. Never defaults.

        Rules are tried in ascending ``priority``; the first priority level with a
        match decides. If that level proposes more than one distinct epoch the
        result is ``ambiguous`` - the tie is reported, not broken arbitrarily.
        """
        core = self.core_of(stem)
        estim = self._frequency(core, self.estim_hz)
        mstim = self._frequency(core, self.mstim_hz)
        timepoint = self._timepoint(core)
        unparsed = self._unparsed_stim_tokens(core)

        hits = [r for r in self.rules if re.search(r.pattern, core, re.IGNORECASE)]
        if not hits:
            return Classification(
                condition=Condition(
                    epoch=UNKNOWN_EPOCH,
                    estim_hz=estim,
                    mstim_hz=mstim,
                    timepoint=timepoint,
                ),
                status="unknown",
                unparsed_stim_tokens=unparsed,
            )

        best = min(r.priority for r in hits)
        level = [r for r in hits if r.priority == best]
        epochs = {r.epoch for r in level}
        if len(epochs) > 1:
            return Classification(
                condition=Condition(
                    epoch=UNKNOWN_EPOCH,
                    estim_hz=estim,
                    mstim_hz=mstim,
                    timepoint=timepoint,
                ),
                status="ambiguous",
                candidates=tuple(sorted(epochs)),
                unparsed_stim_tokens=unparsed,
            )

        winner = sorted(level, key=lambda r: r.id)[0]
        return Classification(
            condition=Condition(
                epoch=winner.epoch,
                estim_hz=estim,
                mstim_hz=mstim,
                timepoint=timepoint,
            ),
            status="matched",
            matched_rule=winner.id,
            unparsed_stim_tokens=unparsed,
        )

    def animal(self, stem: str) -> str | None:
        """Return the animal letter for a stem, or None if it cannot be derived."""
        return self.animal_from(self.core_of(stem))

    def validate_epoch(self, epoch: str) -> str:
        """Return ``epoch`` if the vocabulary allows it, else raise.

        Called at write time as well as at load time: a correction typed by a user
        is exactly where a free-text level would otherwise get in.
        """
        if epoch not in self.vocabulary:
            msg = (
                f"epoch {epoch!r} is not in the closed vocabulary "
                f"{list(self.vocabulary)}. Free-text conditions are not allowed - "
                "'stim', 'Stim' and 'stimulation' as three levels would wreck the "
                "mixed models. Add the level to conditions.yaml deliberately."
            )
            raise ValueError(msg)
        return epoch

    def validate_condition(self, condition: Condition) -> Condition:
        """Validate a whole record: the epoch, and both frequencies against their sets."""
        self.validate_epoch(condition.epoch)
        for name, value, mapping in (
            ("estim_hz", condition.estim_hz, self.estim_hz),
            ("mstim_hz", condition.mstim_hz, self.mstim_hz),
        ):
            if value is None:
                continue
            allowed = sorted({float(v) for v in mapping.values()})
            if float(value) not in allowed:
                msg = (
                    f"{name}={value} is not one of the defined levels {allowed}. The "
                    "frequency axes are fixed by the protocol; a value off them is "
                    "either a new level to add deliberately or a mistake."
                )
                raise ValueError(msg)
        return condition

    def would_match(self, stems: Iterable[str], pattern: str, epoch: str) -> list[str]:
        """Return the stems a proposed rule would also match.

        The learning loop shows this **before** saving a new rule, so the blast
        radius is visible rather than discovered on the next scan.
        """
        self.validate_epoch(epoch)
        regex = re.compile(pattern, re.IGNORECASE)
        return [s for s in stems if regex.search(self.core_of(s))]


def _first_present(document: Mapping[str, Any], *keys: str) -> object:
    """Return the first key present, so an older spelling still loads."""
    for key in keys:
        if key in document:
            return document[key]
    return None


def _rules_from_document(document: object, source: str) -> Rules:
    """Build :class:`Rules` from a parsed YAML document.

    ``document`` is typed ``object`` because ``yaml.safe_load`` returns whatever the
    file held - a list, a string, ``None`` - and the shape has to be checked rather
    than assumed. ``epochs``/``epoch_rules`` are the current spellings;
    ``vocabulary``/``rules`` are accepted so a file written against the single-label
    version still loads.
    """
    if not isinstance(document, dict):
        msg = f"{source} must contain a mapping at the top level"
        raise ValueError(msg)

    vocabulary = _first_present(document, "epochs", "vocabulary")
    if not isinstance(vocabulary, list) or not vocabulary:
        msg = f"{source} must define a non-empty `epochs` list"
        raise ValueError(msg)

    raw_rules = _first_present(document, "epoch_rules", "rules") or []
    if not isinstance(raw_rules, list):
        msg = f"{source}: `epoch_rules` must be a list"
        raise ValueError(msg)

    rules: list[ConditionRule] = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, dict):
            msg = f"{source}: rule {index} must be a mapping, got {type(raw).__name__}"
            raise ValueError(msg)
        epoch = _first_present(raw, "epoch", "condition")
        if "pattern" not in raw or epoch is None:
            msg = f"{source}: rule {index} needs both `pattern` and `epoch`"
            raise ValueError(msg)
        rules.append(
            ConditionRule(
                id=str(raw.get("id") or f"rule_{index}"),
                pattern=str(raw["pattern"]),
                epoch=str(epoch),
                priority=int(raw.get("priority", 100)),
            )
        )

    def frequency_map(key: str) -> dict[str, float]:
        raw_map = document.get(key) or {}
        if not isinstance(raw_map, dict):
            msg = f"{source}: `{key}` must be a mapping of token to frequency"
            raise ValueError(msg)
        return {str(k).lower(): float(v) for k, v in raw_map.items()}

    suffixes = document.get("strip_suffixes") or []
    if not isinstance(suffixes, list):
        msg = f"{source}: `strip_suffixes` must be a list"
        raise ValueError(msg)

    timepoint = document.get("timepoint_pattern") or r"(?<![a-z0-9])t(\d{1,3})(?![0-9])"

    return Rules(
        vocabulary=tuple(str(v) for v in vocabulary),
        rules=tuple(rules),
        estim_hz=frequency_map("estim_hz"),
        mstim_hz=frequency_map("mstim_hz"),
        timepoint_pattern=str(timepoint),
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
        compile, or a rule proposes an epoch outside the vocabulary.
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
    """Return the starting rule set: both cohorts' conventions, measured."""
    document = yaml.safe_load(DEFAULT_CONDITIONS_YAML)
    return _rules_from_document(document, "<default>")


def write_rules(rules: Rules, path: Path) -> Path:
    r"""Write a rule set atomically as UTF-8 with ``\n`` endings.

    Round-trips through :func:`load_rules`, so a written file is always loadable.
    """
    document = {
        "epochs": list(rules.vocabulary),
        "epoch_rules": [
            {"id": r.id, "pattern": r.pattern, "epoch": r.epoch, "priority": r.priority}
            for r in sorted(rules.rules, key=lambda r: (r.priority, r.id))
        ],
        "estim_hz": dict(rules.estim_hz),
        "mstim_hz": dict(rules.mstim_hz),
        "timepoint_pattern": rules.timepoint_pattern,
        "strip_suffixes": list(rules.strip_suffixes),
    }
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return path
