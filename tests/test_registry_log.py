"""The append-only registry: concurrent appends, conflict copies, replay, compaction.

The tests that matter here are the ones about *order*: on a sync layer, two clients'
writes arrive in an arbitrary order and a reader must not care.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gems_blanking_v2.io.registry_log import (
    LOG_NAME,
    ModelState,
    RegistryAction,
    RegistryEvent,
    append_event,
    compact,
    iter_shard_paths,
    read_events,
    replay,
    shard_path,
)
from gems_blanking_v2.io.store import GemsStore, append_line, validate_component


@pytest.fixture
def store(tmp_path: Path) -> GemsStore:
    """Return a fake gems_root with the real layout."""
    return GemsStore.initialise(tmp_path / "gems")


def event(
    model_id: str,
    action: RegistryAction = RegistryAction.TRAINED,
    *,
    ts: str = "20260920T100000Z",
    user: str = "andrea",
    mode: str = "",
    animal: str = "",
    corpus_id: str = "",
    metrics: dict[str, float] | None = None,
) -> RegistryEvent:
    """Build an event with sensible defaults."""
    return RegistryEvent(
        ts=ts,
        user=user,
        action=action,
        model_id=model_id,
        mode=mode,
        animal=animal,
        corpus_id=corpus_id,
        metrics=metrics or {},
    )


# ---------------------------------------------------------------------------
# the line format
# ---------------------------------------------------------------------------


def test_a_line_round_trips(store: GemsStore) -> None:
    original = event("m1", mode="pooled", animal="J", corpus_id="c1")
    assert RegistryEvent.from_json_line(original.to_json_line()) == original


def test_the_line_is_canonical_so_duplicates_dedupe() -> None:
    """Two clients logging the same fact must produce byte-identical lines."""
    a = RegistryEvent("20260920T100000Z", "u", RegistryAction.TRAINED, "m1", metrics={"auc": 0.9})
    b = RegistryEvent("20260920T100000Z", "u", RegistryAction.TRAINED, "m1", metrics={"auc": 0.9})
    assert a.to_json_line() == b.to_json_line()
    assert json.loads(a.to_json_line())["action"] == "trained"


def test_a_line_is_one_line() -> None:
    line = event("m1", corpus_id="a corpus with spaces").to_json_line()
    assert "\n" not in line


def test_a_malformed_line_raises_rather_than_being_skipped() -> None:
    """Silently dropping a line would lose a fact from an append-only log."""
    with pytest.raises(ValueError, match="malformed registry line"):
        RegistryEvent.from_json_line("{not json")
    with pytest.raises(ValueError, match="unknown action"):
        RegistryEvent.from_json_line(
            json.dumps({"ts": "t", "action": "vaporised", "model_id": "m"})
        )


# ---------------------------------------------------------------------------
# shards
# ---------------------------------------------------------------------------


def test_the_shard_name_is_filename_safe(store: GemsStore) -> None:
    """``events.jsonl.<user>.<utc>`` with an ISO stamp would be unwritable on Windows."""
    path = shard_path(store, "Andrea Biju", "20260920T184500Z")
    assert path.name == "events.jsonl.Andrea_Biju.20260920T184500Z"
    assert validate_component(path.name) == path.name


def test_appending_writes_to_the_users_own_shard(store: GemsStore) -> None:
    path = append_event(store, event("m1", user="andrea"), stamp="20260920T100000Z")
    assert path.parent == store.registry_dir
    assert path.read_bytes().endswith(b"\n")
    assert len(iter_shard_paths(store)) == 1


def test_two_clients_appending_concurrently_produce_two_files_not_a_conflict(
    store: GemsStore,
) -> None:
    append_event(store, event("m1", user="andrea"), stamp="20260920T100000Z")
    append_event(store, event("m2", user="sam"), stamp="20260920T100001Z")
    assert len(iter_shard_paths(store)) == 2
    assert {e.model_id for e in read_events(store)} == {"m1", "m2"}


def test_replay_contains_both_entries_in_either_merge_order(store: GemsStore) -> None:
    """Union-of-lines is order-independent; the reader must not depend on file order."""
    a = event("m1", user="andrea", ts="20260920T100000Z")
    b = event("m2", user="sam", ts="20260920T100001Z")
    append_event(store, a, stamp="20260920T100000Z")
    append_event(store, b, stamp="20260920T100001Z")
    forward = read_events(store)

    # Rebuild with the shards written in the opposite order.
    for p in iter_shard_paths(store):
        p.unlink()
    append_event(store, b, stamp="20260920T100001Z")
    append_event(store, a, stamp="20260920T100000Z")
    assert read_events(store) == forward
    assert set(replay(forward)) == {"m1", "m2"}


def test_a_drive_conflict_copy_is_merged_not_lost(store: GemsStore) -> None:
    """Drive resolves a clash by appending ' (1)'. Those lines are real."""
    append_event(store, event("m1"), stamp="20260920T100000Z")
    conflict = store.registry_dir / f"{LOG_NAME}.andrea.20260920T100000Z (1)"
    append_line(conflict, event("m2", ts="20260920T110000Z").to_json_line())

    ids = [e.model_id for e in read_events(store)]
    assert ids == ["m1", "m2"]


def test_a_conflict_copy_of_the_compacted_log_is_also_read(store: GemsStore) -> None:
    append_line(store.registry_dir / LOG_NAME, event("m1").to_json_line())
    append_line(
        store.registry_dir / f"{LOG_NAME} (1)", event("m2", ts="20260920T110000Z").to_json_line()
    )
    assert [e.model_id for e in read_events(store)] == ["m1", "m2"]


def test_a_duplicated_line_counts_once(store: GemsStore) -> None:
    """A conflict copy usually holds the same lines as its original."""
    line = event("m1").to_json_line()
    append_line(store.registry_dir / f"{LOG_NAME}.andrea.20260920T100000Z", line)
    append_line(store.registry_dir / f"{LOG_NAME}.andrea.20260920T100000Z (1)", line)
    assert len(read_events(store)) == 1


def test_unrelated_files_in_the_registry_directory_are_ignored(store: GemsStore) -> None:
    (store.registry_dir / "notes.txt").write_text("hand-written", encoding="utf-8")
    append_event(store, event("m1"), stamp="20260920T100000Z")
    assert [p.name for p in iter_shard_paths(store)] == ["events.jsonl.andrea.20260920T100000Z"]


def test_reading_an_empty_registry_is_not_an_error(store: GemsStore) -> None:
    assert read_events(store) == []
    assert replay([]) == {}


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_applies_the_last_action_by_timestamp() -> None:
    events = [
        event("m1", RegistryAction.TRAINED, ts="20260920T100000Z", mode="pooled", animal="J"),
        event("m1", RegistryAction.PROMOTED, ts="20260920T110000Z"),
        event("m1", RegistryAction.DEMOTED, ts="20260920T120000Z"),
    ]
    state = replay(events)["m1"]
    assert state.status is RegistryAction.DEMOTED
    assert not state.is_promoted
    assert state.mode == "pooled"  # carried from the trained event
    assert state.animal == "J"


def test_replay_is_independent_of_input_order() -> None:
    events = [
        event("m1", RegistryAction.TRAINED, ts="20260920T100000Z", corpus_id="c1"),
        event("m1", RegistryAction.PROMOTED, ts="20260920T110000Z"),
    ]
    assert replay(events) == replay(list(reversed(events)))


def test_replay_is_idempotent() -> None:
    """Replaying the same facts twice - as a duplicated conflict copy would - is a no-op."""
    events = [
        event("m1", RegistryAction.TRAINED, ts="20260920T100000Z"),
        event("m1", RegistryAction.PROMOTED, ts="20260920T110000Z"),
    ]
    assert replay(events) == replay(events + events)


def test_a_promotion_that_arrives_before_its_training_line_is_kept() -> None:
    """Shards sync independently; state must not depend on which arrived first."""
    state = replay([event("m1", RegistryAction.PROMOTED, ts="20260920T110000Z")])
    assert state["m1"] == ModelState(model_id="m1", status=RegistryAction.PROMOTED)
    assert state["m1"].is_promoted


def test_two_models_trained_on_the_same_corpus_both_survive() -> None:
    """Concurrent training is legal; a human picks. Neither line is dropped."""
    events = [
        event("m1", user="andrea", ts="20260920T100000Z", corpus_id="c1"),
        event("m2", user="sam", ts="20260920T100000Z", corpus_id="c1"),
    ]
    state = replay(events)
    assert set(state) == {"m1", "m2"}
    assert {s.trained_by for s in state.values()} == {"andrea", "sam"}


# ---------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------


def test_compaction_merges_shards_and_preserves_every_event(store: GemsStore) -> None:
    append_event(store, event("m1", user="andrea"), stamp="20260920T100000Z")
    append_event(store, event("m2", user="sam", ts="20260920T110000Z"), stamp="20260920T110000Z")
    before = read_events(store)

    n = compact(store)

    assert n == 2
    assert [p.name for p in iter_shard_paths(store)] == [LOG_NAME]
    assert read_events(store) == before


def test_compaction_is_idempotent(store: GemsStore) -> None:
    append_event(store, event("m1"), stamp="20260920T100000Z")
    compact(store)
    first = (store.registry_dir / LOG_NAME).read_bytes()
    compact(store)
    assert (store.registry_dir / LOG_NAME).read_bytes() == first


def test_compaction_writes_lf_endings(store: GemsStore) -> None:
    append_event(store, event("m1"), stamp="20260920T100000Z")
    compact(store)
    assert b"\r\n" not in (store.registry_dir / LOG_NAME).read_bytes()


def test_appending_after_compaction_still_merges(store: GemsStore) -> None:
    append_event(store, event("m1"), stamp="20260920T100000Z")
    compact(store)
    append_event(store, event("m2", ts="20260920T120000Z"), stamp="20260920T120000Z")
    assert [e.model_id for e in read_events(store)] == ["m1", "m2"]
