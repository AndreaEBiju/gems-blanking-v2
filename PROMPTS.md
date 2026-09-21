# Prompts for building this with Claude Code

The documents are designed so the prompt is short. `CLAUDE.md` is loaded
automatically from the repo root, and each `tasks/NN-*.md` is self-contained
(the shared contracts are appended to every one). **Do not paste spec content
into the prompt** — it is already there, and a second copy that drifts from the
first is worse than none.

**One task per session.** These tasks are sized so that a fresh session can do
exactly one and stop. "Build the pipeline" in a single session produces a large
diff nobody has reviewed, against a spec with gates in it.

---

## 1 · First session — bootstrap the repo

Run this in an empty directory with `CLAUDE.md`, `IMPLEMENTATION.md`,
`PIPELINE.md`, `split_tasks.py` and `tasks/` already copied in.

```
Read CLAUDE.md, then tasks/00-repo-setup.md, then do that task and nothing else.

This is a research tool: correctness of each decision matters more than speed,
and the hard invariants in CLAUDE.md are not negotiable.

Before you write code, tell me your plan and anything in the task you think is
wrong or underspecified. Then build it.

Stop when task 00's acceptance criterion is met. Do not start task 00A.
```

## 2 · Every subsequent task — the template

Replace the task filename. This is the whole prompt.

```
Read CLAUDE.md, then tasks/<NN-slug>.md, and implement that task only.

Plan first, flag anything underspecified or wrong, then build.
Finish with the report the "Definition of done" in CLAUDE.md asks for.

Do not start the next task.
```

For a task that touches a gate, add:

```
This is a gate task. If the acceptance criterion is not met, stop and report —
do not relax the criterion or tune until it passes.
```

## 3 · Tasks that need the real data

Tasks 02, 05 and 09 read recordings. Connect the folder first, then:

```
Read CLAUDE.md, then tasks/02-peri-r-measurement.md, and implement it.

The data is in the connected folder <name>. Work on the files where they are;
do not copy recordings into the repo, and do not write anything into the shared
drive without asking me first.

Several constants in the spec are marked MEASURED on animal J, one recording.
If this data disagrees with any of them, report the disagreement — do not change
the code to match the spec.
```

## 4 · After a few tasks — a review pass

Worth doing every 3–4 tasks, in a fresh session:

```
Read CLAUDE.md. Review the repo against the hard invariants and the
cross-platform rules only — not style.

For each violation: file, line, which invariant, and the smallest fix.
If there are none, say so; do not invent findings.
```

## 5 · When you change the spec

`tasks/` is generated. Never let it be hand-edited:

```
Edit IMPLEMENTATION.md to <change>, then run `python split_tasks.py` and
confirm the dependency-cycle check still passes. Do not edit tasks/ directly.
```

---

## What NOT to put in the prompt

| Don't | Why |
|---|---|
| Paste the design rationale | `PIPELINE.md` has it; a second copy drifts |
| "Build the whole pipeline" | There are gates; the whole point is stopping at them |
| "Make the tests pass" | Invites loosening a threshold, which `CLAUDE.md` calls a failed test |
| Restate the invariants | Already loaded; restating a subset implies the rest are optional |
| "Be thorough / take your time" | Vague; the definition of done is already explicit |
| Name more than one task | The dependency order exists for a reason |

## Two things worth saying out loud in any session

- **"Tell me what's wrong with this task before you build it."** Several spec
  errors so far were found this way — a runaway plausibility rule, a z-score
  that flagged 40% of a clean recording, a dependency cycle.
- **"Report contradictions instead of resolving them."** The spec's constants
  came from one animal and one recording. Code quietly tuned to match a wrong
  constant is the expensive failure.
