#!/usr/bin/env python3
"""Regenerate tasks/NN-slug.md from IMPLEMENTATION.md.

IMPLEMENTATION.md is the single source of truth. Each task is delimited by
    <!-- TASK:NN slug=... deps=... gate=... -->  ...  <!-- /TASK -->
Part A (shared contracts) is prepended to every generated task so each file is
self-contained and can be run in a fresh Claude Code session.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "IMPLEMENTATION.md"
OUT = ROOT / "tasks"

HDR = re.compile(
    r"<!--\s*TASK:(?P<num>\d+[A-Z]?)\s+slug=(?P<slug>[\w-]+)\s+deps=(?P<deps>[\w,]+)\s+gate=(?P<gate>yes|no)\s*-->"
)

def main() -> int:
    """Regenerate every task file and the index; return a process exit code."""
    # utf-8 / LF: rule 10. The Windows default is cp1252, which cannot read this file.
    text = SRC.read_text(encoding="utf-8")

    a0 = text.index("## Part A — shared contracts")
    a1 = text.index("## Part B — tasks")
    part_a = text[a0:a1].rstrip()

    blocks, pos = [], 0
    while (m := HDR.search(text, pos)) is not None:
        end = text.index("<!-- /TASK -->", m.end())
        blocks.append((m.groupdict(), text[m.end():end].strip()))
        pos = end + len("<!-- /TASK -->")

    if not blocks:
        print("no tasks found — check the delimiters", file=sys.stderr)
        return 1

    nums = [b[0]["num"] for b in blocks]
    if len(set(nums)) != len(nums):
        print(f"duplicate task numbers: {nums}", file=sys.stderr)
        return 1
    known = set(nums)
    graph: dict[str, set[str]] = {}
    for meta, _ in blocks:
        deps = set()
        for d in meta["deps"].split(","):
            if d in {"none", "all"}:
                continue
            if d not in known:
                print(f"task {meta['num']} depends on unknown task {d}", file=sys.stderr)
                return 1
            deps.add(d)
        graph[meta["num"]] = deps

    # invariant 4: the build graph must be acyclic
    white, grey, black = 0, 1, 2
    colour = dict.fromkeys(graph, white)

    def visit(node: str, path: list[str]) -> list[str] | None:
        colour[node] = grey
        for dep in sorted(graph[node]):
            if colour[dep] == grey:
                return [*path, node, dep]
            if colour[dep] == white and (cyc := visit(dep, [*path, node])) is not None:
                return cyc
        colour[node] = black
        return None

    for n in sorted(graph):
        if colour[n] == white and (cyc := visit(n, [])) is not None:
            print(f"DEPENDENCY CYCLE (violates invariant 4): {' -> '.join(cyc)}", file=sys.stderr)
            return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    for meta, body in blocks:
        gate_note = (
            "\n> **THIS IS A GATE.** Work after this task is wasted if it fails. "
            "Do not proceed past it.\n"
        )
        gate = gate_note if meta["gate"] == "yes" else ""
        deps = "none" if meta["deps"] == "none" else ", ".join(meta["deps"].split(","))
        (OUT / f"{meta['num']}-{meta['slug']}.md").write_text(
            f"<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->\n\n"
            f"{body}\n\n"
            f"---\n\n"
            f"**Depends on tasks:** {deps}\n{gate}\n"
            f"Read `CLAUDE.md` before starting; its invariants apply here and are not repeated.\n\n"
            f"---\n\n# Reference — shared contracts\n\n{part_a}\n",
            encoding="utf-8",
            newline="\n",
        )

    index = ["# Task index\n", "Generated. Gates in **bold**.\n",
             "| # | task | depends on | gate |", "|---|---|---|---|"]
    for meta, _ in blocks:
        name = f"**{meta['slug']}**" if meta["gate"] == "yes" else meta["slug"]
        index.append(f"| {meta['num']} | [{name}]({meta['num']}-{meta['slug']}.md) "
                     f"| {meta['deps']} | {'YES' if meta['gate']=='yes' else ''} |")
    (OUT / "README.md").write_text(
        "\n".join(index) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"wrote {len(blocks)} tasks + index to {OUT}/")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
