"""The ``gems`` command line: ``doctor``, ``preflight``, ``compact``, ``init``.

A Python console-script entry point, not a shell script (cross-platform rule 14),
and guarded with ``if __name__ == "__main__"`` because Windows multiprocessing uses
``spawn`` (rule 13).

``gems doctor`` exists to make cross-platform breakage visible on the machine that
causes it: it prints the platform, the resolved root, the longest path the layout
would generate, whether anything stored is absolute, the item count against the
shared-drive cap, and whether the mount is actually writable.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from gems_blanking_v2.io.registry_log import compact, read_events, replay
from gems_blanking_v2.io.store import (
    ITEM_CAP_WARN_FRACTION,
    MARKER_NAME,
    SHARED_DRIVE_ITEM_CAP,
    GemsStore,
    cache_dir,
    config_path,
    find_gems_root,
    resolve_user_id,
    write_config_root,
)

__all__ = ["main"]


def _resolve(args: argparse.Namespace) -> GemsStore:
    """Resolve the store from ``--root`` or discovery, or exit with a clear message."""
    try:
        return GemsStore(find_gems_root(Path(args.root) if args.root else None))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _doctor(args: argparse.Namespace) -> int:
    """Print everything a user needs to diagnose their own setup."""
    store = _resolve(args)
    identity = resolve_user_id()
    report = store.preflight()

    print(report.summary())
    print(f"marker:      {store.marker}")
    print(f"config:      {config_path()}  ({'present' if config_path().is_file() else 'absent'})")
    print(f"local cache: {cache_dir()}")
    print(f"user:        {identity.user_id}  (from {identity.source})")
    if not identity.is_confident:
        print(
            "warning:     no git identity configured - 'who labelled this' is scientific "
            "metadata. Set git user.email, or pass an explicit user."
        )

    try:
        store.assert_cache_is_outside()
    except ValueError as exc:
        print(f"error:       {exc}")

    count, complete = store.count_items(cap=args.item_cap)
    pct = 100.0 * count / SHARED_DRIVE_ITEM_CAP
    bound = "" if complete else " (lower bound; pass a larger --item-cap for an exact count)"
    print(f"items:       {count}{bound} of {SHARED_DRIVE_ITEM_CAP} ({pct:.1f}%)")
    print("note:        trash counts against the cap but is invisible to the filesystem")
    if complete and count >= ITEM_CAP_WARN_FRACTION * SHARED_DRIVE_ITEM_CAP:
        print(f"warning:     over {ITEM_CAP_WARN_FRACTION:.0%} of the shared-drive item cap")

    events = read_events(store)
    state = replay(events)
    shards = (
        len(list(store.registry_dir.glob("events.jsonl*")))
        if store.registry_dir.is_dir()
        else 0
    )
    promoted = sorted(m for m, s in state.items() if s.is_promoted)
    print(f"registry:    {len(events)} events across {shards} file(s), {len(state)} model(s)")
    print(f"promoted:    {', '.join(promoted) if promoted else '(none)'}")
    return 0 if report.ok else 1


def _preflight(args: argparse.Namespace) -> int:
    """Refuse to start when the corpus is incomplete, and say exactly why."""
    store = _resolve(args)
    report = store.preflight()
    print(report.summary())
    return 0 if report.ok else 1


def _compact(args: argparse.Namespace) -> int:
    """Merge registry shards. Manual, by one person, never automatic."""
    store = _resolve(args)
    n = compact(store)
    print(f"compacted {n} unique events into {store.registry_dir / 'events.jsonl'}")
    return 0


def _init(args: argparse.Namespace) -> int:
    """Create the layout and marker at ``--root`` and record it in per-user config."""
    root = Path(args.root).expanduser()
    store = GemsStore.initialise(root)
    written = write_config_root(store.root)
    print(f"initialised {store.root} ({MARKER_NAME} written)")
    print(f"recorded in {written}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``gems`` console script."""
    parser = argparse.ArgumentParser(
        prog="gems", description=f"gems-blanking-v2 on {platform.system()}"
    )
    parser.add_argument("--root", default=None, help="gems_root; default is discovery by marker")

    # --root is accepted on either side of the subcommand. SUPPRESS matters: without
    # it the subparser's default would overwrite a --root given before the
    # subcommand, and 'gems init --root X' is the form the error messages suggest.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS, help="gems_root")

    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor", parents=[common], help="report platform, root, items, registry"
    )
    doctor.add_argument("--item-cap", type=int, default=50_000, help="stop counting items at N")
    doctor.set_defaults(func=_doctor)

    sub.add_parser(
        "preflight", parents=[common], help="verify the run's inputs and refuse if incomplete"
    ).set_defaults(func=_preflight)
    sub.add_parser(
        "compact", parents=[common], help="merge registry shards (manual, one operator)"
    ).set_defaults(func=_compact)

    init = sub.add_parser("init", parents=[common], help="create the layout and marker at --root")
    init.set_defaults(func=_init)

    args = parser.parse_args(argv)
    if args.command == "init" and not args.root:
        parser.error("init requires --root")
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover - rule 13: spawn re-imports the module
    raise SystemExit(main())
