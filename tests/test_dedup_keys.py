"""Property tests for everything used as a dedup or identity key.

``CLAUDE.md``: *anything used as a dedup or identity key must round-trip exactly.
If ``parse(serialise(x)) != x`` for any field, deduplication silently fails.*
That rule came out of 00A, where ``RegistryEvent(metrics=None)`` serialised to
``{}`` and parsed back as ``{}``, so an event never equalled its own reparse - and
conflict-copy dedup is built on exactly that comparison. The rule asks for a
property test rather than an example, which is what this module is.

The keys in the codebase:

* ``RegistryEvent.to_json_line`` / ``from_json_line`` - the dedup key for the
  registry union. Two clients logging the same fact must produce byte-identical
  lines, and two *different* facts must never collide onto one line.
* ``GemsStore.relpath`` / ``abspath`` - the identity of a file across machines. A
  stored POSIX path must resolve back to the same local path under any root.
* ``safe_component`` and ``utc_stamp`` - the components of a registry shard's
  filename, which identifies the shard.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from gems_blanking_v2.io.registry_log import RegistryAction, RegistryEvent
from gems_blanking_v2.io.store import (
    GemsStore,
    append_line,
    read_lines,
    safe_component,
    utc_stamp,
    validate_component,
)
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# The autouse config-isolation fixture in conftest is function-scoped, which
# hypothesis flags by default. It only redirects two module-level functions and
# carries no per-example state, so suppressing the check is correct here.
PROPERTY = settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)

# Text that has caused trouble in a filename, a JSON document or a line-based
# reader: empty, whitespace-only, quotes, backslashes, every line separator
# Unicode defines, astral-plane characters, and a long run.
NASTY_TEXT = st.one_of(
    st.just(""),
    st.just(" "),
    st.just('"'),
    st.just("\\"),
    st.just("\n"),
    st.just("\r\n"),
    st.just("\u2028"),
    st.just("\u2029"),
    st.just("\u0085"),
    st.just("\x00"),
    st.just("σ = 2.26 µV"),  # noqa: RUF001 - deliberately ambiguous
    st.just("BIONICs Lab: Enteric Interfaces Team"),
    st.just("动物-J"),
    st.just("🧠" * 4),
    st.just("x" * 500),
    st.text(),
    st.text(alphabet=st.characters(min_codepoint=0x80)),
)

FINITE_FLOATS = st.floats(allow_nan=False, allow_infinity=False, width=64)

METRICS = st.dictionaries(NASTY_TEXT, FINITE_FLOATS, max_size=5)


@st.composite
def events(draw: st.DrawFn) -> RegistryEvent:
    """Build a RegistryEvent over the full space of each field.

    Every optional field can be empty, and ``metrics`` can be ``{}`` - the two
    cases whose confusion with ``None`` is what this module exists to prevent.
    """
    return RegistryEvent(
        ts=draw(NASTY_TEXT),
        user=draw(NASTY_TEXT),
        action=draw(st.sampled_from(list(RegistryAction))),
        model_id=draw(NASTY_TEXT),
        mode=draw(NASTY_TEXT),
        animal=draw(NASTY_TEXT),
        corpus_id=draw(NASTY_TEXT),
        metrics=draw(METRICS),
    )


# ---------------------------------------------------------------------------
# RegistryEvent: the registry dedup key
# ---------------------------------------------------------------------------


@PROPERTY
@given(event=events())
def test_an_event_equals_its_own_reparse(event: RegistryEvent) -> None:
    """``from_json_line(to_json_line(e)) == e`` for every event, field by field."""
    assert RegistryEvent.from_json_line(event.to_json_line()) == event


@PROPERTY
@given(event=events())
def test_the_canonical_line_is_a_fixed_point(event: RegistryEvent) -> None:
    """Serialising a parsed line reproduces the line, so dedup is stable over time."""
    line = event.to_json_line()
    assert RegistryEvent.from_json_line(line).to_json_line() == line


@PROPERTY
@given(event=events())
def test_equal_events_produce_byte_identical_lines(event: RegistryEvent) -> None:
    """Two clients logging the same fact must dedupe: the union keys on the line."""
    twin = RegistryEvent(
        ts=event.ts,
        user=event.user,
        action=event.action,
        model_id=event.model_id,
        mode=event.mode,
        animal=event.animal,
        corpus_id=event.corpus_id,
        metrics=dict(event.metrics),
    )
    assert twin == event
    assert twin.to_json_line() == event.to_json_line()


@PROPERTY
@given(a=events(), b=events())
def test_different_events_never_collide_onto_one_line(a: RegistryEvent, b: RegistryEvent) -> None:
    """The other half of dedup: a collision would silently merge two distinct facts."""
    assume(a != b)
    assert a.to_json_line() != b.to_json_line()


@PROPERTY
@given(event=events())
def test_a_line_is_always_exactly_one_line_to_any_reader(event: RegistryEvent) -> None:
    r"""No line separator any reader recognises may appear literally in the line.

    ``str.splitlines()`` splits on U+2028, U+2029 and U+0085 as well as ``\n``, so a
    corpus_id containing one would turn a single event into two malformed lines for a
    reader that uses it - or for another language's line splitter. This is why the
    canonical form is ASCII-escaped.
    """
    line = event.to_json_line()
    assert "\n" not in line
    assert "\r" not in line
    assert len(line.splitlines()) == 1
    assert line.isascii()


@PROPERTY
@given(event=events())
def test_a_line_is_standard_json_no_bare_nan_or_infinity(event: RegistryEvent) -> None:
    """A strict reader in any language must be able to parse the line.

    ``json.loads`` accepts bare ``NaN`` and ``Infinity``; almost nothing else does.
    ``parse_constant`` is how a strict reader is simulated.
    """

    def reject(constant: str) -> float:
        msg = f"non-standard JSON constant {constant}"
        raise ValueError(msg)

    parsed = json.loads(event.to_json_line(), parse_constant=reject)
    assert set(parsed) == {
        "ts",
        "user",
        "action",
        "model_id",
        "mode",
        "animal",
        "corpus_id",
        "metrics",
    }


@pytest.mark.parametrize("action", list(RegistryAction))
def test_every_action_round_trips(action: RegistryAction) -> None:
    """Enumerated explicitly as well as sampled, so a new action cannot slip through."""
    event = RegistryEvent("20260921T100000Z", "u", action, "m1")
    back = RegistryEvent.from_json_line(event.to_json_line())
    assert back == event
    assert back.action is action
    assert json.loads(event.to_json_line())["action"] == str(action)


@PROPERTY
@given(
    ts=NASTY_TEXT,
    user=NASTY_TEXT,
    model_id=NASTY_TEXT,
    action=st.sampled_from(list(RegistryAction)),
)
def test_absent_null_and_empty_optionals_all_parse_to_the_same_event(
    ts: str, user: str, model_id: str, action: RegistryAction
) -> None:
    """The 00A defect, generalised to every optional field.

    A line that omits an optional field, one that sets it to ``null`` and one that
    sets it to the empty value must parse to the same event and therefore to the
    same canonical line - otherwise the same fact written by two versions of the
    tool would fail to dedupe.
    """
    required = {"ts": ts, "user": user, "action": str(action), "model_id": model_id}
    spellings = [
        required,
        {**required, "mode": None, "animal": None, "corpus_id": None, "metrics": None},
        {**required, "mode": "", "animal": "", "corpus_id": "", "metrics": {}},
    ]
    parsed = [RegistryEvent.from_json_line(json.dumps(s)) for s in spellings]
    assert parsed[0] == parsed[1] == parsed[2]
    assert len({e.to_json_line() for e in parsed}) == 1
    assert parsed[0].metrics == {}


@PROPERTY
@given(event=events())
def test_a_missing_required_field_raises_rather_than_defaulting(event: RegistryEvent) -> None:
    """A required field is required: silently defaulting it would invent a fact."""
    full = json.loads(event.to_json_line())
    for key in ("ts", "model_id", "action"):
        for broken in ({k: v for k, v in full.items() if k != key}, {**full, key: None}):
            with pytest.raises(ValueError, match="registry line"):
                RegistryEvent.from_json_line(json.dumps(broken))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_metric_is_rejected_at_construction(value: float) -> None:
    """JSON has no NaN, and ``nan != nan`` breaks the round trip outright.

    Found by this module's own round-trip property. The project convention that a
    missing scalar is ``np.nan`` stops at the edge of a JSON file: omit the key.
    """
    assert not math.isfinite(value)
    with pytest.raises(ValueError, match="omit the key"):
        RegistryEvent("20260921T100000Z", "u", RegistryAction.TRAINED, "m1", metrics={"auc": value})


def test_a_hand_written_line_carrying_nan_is_rejected_on_read() -> None:
    """The same guard applies to a line this tool did not write."""
    line = json.dumps(
        {
            "ts": "20260921T100000Z",
            "user": "u",
            "action": "trained",
            "model_id": "m1",
            "metrics": {"auc": float("nan")},
        }
    )
    assert "NaN" in line
    with pytest.raises(ValueError, match="omit the key"):
        RegistryEvent.from_json_line(line)


@pytest.mark.parametrize(
    "corpus_id",
    [
        "plain",
        "",
        "σ = 2.26 µV",  # noqa: RUF001 - a real sigma, which is the point
        "动物-J",
        "🧠",
        "a\nb",
        "a\u2028b",
        "a\u0085b",
        'quote"and\\slash',
    ],
)
def test_an_adversarial_field_survives_a_real_file_round_trip(
    tmp_path: Path, corpus_id: str
) -> None:
    """Through the actual writer and reader, not just the string form."""
    event = RegistryEvent(
        "20260921T100000Z", "u", RegistryAction.TRAINED, "m1", corpus_id=corpus_id
    )
    path = tmp_path / "events.jsonl.u.20260921T100000Z"
    append_line(path, event.to_json_line())

    lines = read_lines(path)
    assert len(lines) == 1
    assert RegistryEvent.from_json_line(lines[0]) == event
    assert b"\r\n" not in path.read_bytes()


# ---------------------------------------------------------------------------
# relpath / abspath: the identity of a file across machines
# ---------------------------------------------------------------------------

# A component that is legal in a path on both platforms. Built from a conservative
# alphabet on purpose: validate_component rejects the rest, and that rejection is
# tested in test_store.py rather than here.
PATH_COMPONENT = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_."),
    min_size=1,
    max_size=12,
).filter(
    lambda s: s == s.rstrip(". ") and s.split(".")[0].upper() not in {"CON", "NUL", "AUX", "PRN"}
)

ROOTS = st.sampled_from(
    [
        Path("/srv/gems"),
        Path("/Users/a/Library/CloudStorage/GoogleDrive-x/Shared drives/BIONICs Lab_ Team"),
        Path("G:/Shared drives/BIONICs Lab\u2236 Enteric Interfaces Team"),
        Path("C:/gems"),
    ]
)


@settings(max_examples=200, deadline=None)
@given(root=ROOTS, parts=st.lists(PATH_COMPONENT, min_size=1, max_size=5))
def test_a_stored_relative_path_resolves_back_to_the_same_local_path(
    root: Path, parts: list[str]
) -> None:
    """``abspath(relpath(p)) == p`` under any root, including one with a substituted colon.

    This is the identity a corpus spec depends on: the path is stored once, on one
    machine, and resolved on every other.
    """
    store = GemsStore(root)
    path = store.root.joinpath(*parts)

    rel = store.relpath(path)

    assert not PurePosixPath(rel).is_absolute()
    assert "\\" not in rel
    assert rel == "/".join(parts)
    assert store.abspath(rel) == path


@settings(max_examples=200, deadline=None)
@given(parts=st.lists(PATH_COMPONENT, min_size=1, max_size=5))
def test_a_relative_path_means_the_same_thing_under_every_root(parts: list[str]) -> None:
    """The same stored string must name the corresponding file under any root."""
    rel = "/".join(parts)
    for root in [Path("/srv/gems"), Path("C:/gems"), Path("G:/Shared drives/x\u2236y")]:
        store = GemsStore(root)
        assert store.relpath(store.abspath(rel)) == rel


# ---------------------------------------------------------------------------
# shard filename components: the identity of a shard
# ---------------------------------------------------------------------------


@PROPERTY
@given(text=NASTY_TEXT)
def test_safe_component_always_produces_a_usable_path_component(text: str) -> None:
    """Its output feeds a shard filename, so it must always pass validation.

    Not a round trip - the mapping is deliberately lossy - but the identity property
    that matters: whatever a git ``user.name`` contains, the shard can be created.
    """
    component = safe_component(text)
    assert validate_component(component) == component


@PROPERTY
@given(text=NASTY_TEXT)
def test_safe_component_is_deterministic(text: str) -> None:
    """The same user must map to the same shard on every append, or shards multiply."""
    assert safe_component(text) == safe_component(text)


@settings(max_examples=300, deadline=None)
@given(
    when=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2099, 12, 31),
        timezones=st.just(UTC),
    )
)
def test_a_timestamp_is_always_a_usable_path_component(when: datetime) -> None:
    """An ISO stamp would carry colons; the shard name embeds this one."""
    stamp = utc_stamp(when)
    assert validate_component(stamp) == stamp
    assert ":" not in stamp
    assert stamp.endswith("Z")
    assert len(stamp) == len("20260921T100000Z")


@settings(max_examples=300, deadline=None)
@given(
    a=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2099, 12, 31),
        timezones=st.just(UTC),
    ),
    b=st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2099, 12, 31),
        timezones=st.just(UTC),
    ),
)
def test_timestamps_sort_in_chronological_order_as_strings(a: datetime, b: datetime) -> None:
    """Replay orders by the stamp as a string, so lexical order must equal time order.

    Over arbitrary pairs rather than a fixed offset. An earlier version built the
    second datetime with ``replace(year=year + 1)``, which hypothesis broke with
    2020-02-29: that day does not exist in 2021, so the *test* raised. The bug was
    in the test, which is the argument for property-testing the test's own helper
    arithmetic too.
    """
    stamp_a, stamp_b = utc_stamp(a), utc_stamp(b)
    if stamp_a == stamp_b:
        # The stamp has one-second resolution, so equal stamps must mean the same
        # second - not merely that ordering happened to collapse.
        assert abs((a - b).total_seconds()) < 1.0
    else:
        assert (stamp_a < stamp_b) == (a < b)
