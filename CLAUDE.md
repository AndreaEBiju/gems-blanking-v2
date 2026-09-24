# gems-blanking-v2 — project instructions

Motion-artifact detection and per-consumer blanking for 9-channel rodent vagus ENG
and stomach EMG recorded at 24.4 kHz on TDT hardware.

Design rationale lives in `PIPELINE.md`. Step-by-step build instructions live in
`IMPLEMENTATION.md`, and `tasks/NN-*.md` are generated from it by
`python split_tasks.py`. **Never edit `tasks/` by hand** — edit `IMPLEMENTATION.md`
and regenerate.

---

## Hard invariants

Violating any of these is a bug even if tests pass. If a task appears to require
violating one, stop and say so rather than working around it.

1. **Masked samples are `NaN`, never `0`.** No emitted array may contain an
   exact-zero run longer than 2 samples. Zeros are indistinguishable from signal to
   `processing_new/step1_bandpass.m`, which tests `isnan` only.
2. **Masks are never merged across consumers.** One boolean mask per
   `(consumer, signal, band)`. The single exception is *within* a consumer reading
   two channels: velocity requires `V1` AND `V3` valid, an intersection.
3. **Every signal is thresholded against its own σ.** Never apply one signal's σ to
   another. Measured cost of borrowing: 1136 true / 19,624 false detections.
4. **No feedback loops.** No step may consume anything produced by a
   higher-numbered step. The one intended feedback (step 12 → step 07 features) is
   a *precomputed* input, not a loop: velocity runs on its own pass first.
5. **Reference values are whole-file scalars, computed on the LOG envelope.**
   `z = (log E − median(log E)) / (1.4826 · MAD(log E))`, one scalar pair per
   (signal, band), from that file alone. No running baseline, no transfer between
   files, no iteration. *Measured 2026-09-19:* the original linear form with a
   10th-percentile reference is mis-centred — median z = 1.00, p90 = 3.05, so ~10%
   of frames exceed z = 3 per pair. Envelopes are positive and right-skewed; the
   log makes the null symmetric (median 0, p90 1.44).
6. **Detection reads all signals, including raw contacts.** Never detect on the
   tripole alone — the tripole is *defined* by removal of the common mode, which is
   the best artifact evidence available.
7. **Thresholds are fixed and stateless.** No detector may carry adaptive state
   across an artifact.
8. **Never fabricate a sample, a fiducial, or a label.** Missing data is tagged
   missing and excluded. Interpolation for filtering is temporary and must be
   reverted to `NaN` immediately after.
9. **`unjudged` is not `negative`.** A span no human looked at is excluded from
   training, never used as a clean example.
10. **Features must be channel-count independent and animal-invariant.** Aggregate
    across channels (max/median/fraction); never concatenate per-channel columns.
    Absolute microvolts enter only via `z`.
10b. **Thresholding across many (signal × band) pairs is a multiple-comparisons
    problem.** Per pair the clean flag rate is 0.2–3.4%; the union of 36 pairs
    reaches 40–55%. Any flag-rate target is **family-wise**, and cross-channel
    agreement (`nmin`) is the lever that reduces the family. Never quote a
    per-pair rate as if it were the candidate rate.
11. **Everything emitted carries provenance**: the full `ModelSpec` (mode, animal,
    version, corpus hash, calibrator), thresholds, reference values, code commit.
12. **Never compare two models evaluated under different protocols.** A per-animal
    model scored leave-one-recording-out and a pooled model scored
    leave-one-animal-out are solving different problems; the LORO one wins whether
    or not it is better. Hold the protocol fixed or do not report the comparison.
13. **Normalisation does not fix corpus-size imbalance.** Feature normalisation
    addresses covariate shift (scale); training-set size is a variance problem.
    Any comparison between corpora of different size needs a learning-curve
    control at matched event count.

14. **Units are declared, never inferred.** No loader may guess whether a file
    holds volts, millivolts or microvolts. `units` is a required argument with no
    default, validated against the conversion table, and declared once per animal
    in the profile. Guessing wrong is a 10⁶ error that looks like a plausible
    signal. Same class as `rostral_end`: irrecoverable if wrong, so it is stated,
    not inferred.
15. **Index conventions convert at the boundary, explicitly and with a test.**
    MATLAB-side intervals are 1-based inclusive; ours are 0-based half-open, so
    `[5,5]` is one sample → `[4/fs, 5/fs)`. An off-by-one here shifts every mask
    by a sample and is invisible in a plot.
16. **Cross-platform by construction.** The tool is developed on macOS and must
    run unchanged on Windows. See the Cross-platform rules below — most of them
    are about what gets *written to the shared drive*, because those files are
    read by other people's machines.

17. **Any array slice handed out of a loader is a read-only view.** Epochs,
    channel selections and windows are returned as `numpy` views (a 20-minute
    9-channel epoch costs ~2 GB to copy), and invariant 1 means consumers write
    NaN into what they are given. A writeable view therefore silently corrupts
    the parent buffer. Set `arr.flags.writeable = False` before returning; a
    consumer that must modify takes its own copy of the span it actually needs.
    Corollary: a view keeps the whole parent alive, so a loop over recordings
    must not accumulate them.

18. **One word, one meaning — `epoch` and `segment` are not interchangeable.**
    `Condition.epoch` is `baseline` / `stim_recovery`; `stim_split.Epoch` is the
    stim-or-recovery slice; a contiguous run of valid (non-NaN) samples is a
    **`segment`** and is never called an epoch. Three meanings for one word was
    caught in task 06 before it reached the code; keep it that way.

19. **A settling time is a maximum over every filter that touches the edge,**
    and a partially-known maximum is `None`, not the part you know. Task 06
    measures the detection bands; task 13 measures the consumers. Reporting the
    detection-side number alone would be too small, and too-small is the
    direction that silently loses coverage rather than the direction that
    complains.

20. **A negative from a delegated search is not evidence.** A subagent sweep
    reported zero `blankmotion` hits in a repository that contained
    `detector/migrate_blankmotion.py` — the literal substring, in a tracked
    file. Two conclusions were drawn from that sweep and both were wrong.
    **Verify a negative directly before acting on it**, especially one that
    removes something from the plan; a false positive announces itself, a false
    negative is silent.

21. **A library resolved by `sys.path` order is not a dependency, it is an
    accident.** Task 16 found `gems_blanking_v2` importing `detector` from
    `Documents/GEMSBlanking` while the PyQt app imported the same package from
    its `detector-core` submodule — two working trees of one repo, selected by
    discovery order, both needing the identical fix. That works until the two
    diverge, and then the failure is invisible and machine-dependent. **Pin it:
    one checkout, declared as a dependency** (an editable install, or the
    submodule for both), never whichever copy the path happens to reach first.
    This is rule 16's territory — what breaks is the machine you are not sitting
    at.

    **And check what the fix exposes.** `pip install -e` on a *flat-layout*
    repository puts its whole root on `sys.path`, not just its declared
    packages. Doing this for `detector` put a second top-level `tests` package
    on the path, and **Python prefers a regular package found later over a
    namespace package found earlier**, so every `from tests.conftest import ...`
    in this repo silently resolved to the other project's tests. Give your own
    `tests/` an `__init__.py`, and prefer `editable_mode=strict` where setuptools
    honours it.

22. **A value that crosses the MATLAB/Python boundary has a declared canonical
    form and a round-trip test — especially when it is used as a key.**
    MATLAB's `jsonencode` writes `5.0` as `5`; Python's `f"{5.0}"` writes
    `5.0`. A dictionary key built by formatting a float on one side and
    matching it on the other therefore missed for the 5 s duration and **only**
    the 5 s duration — `0.05` and `0.5` render identically on both sides. The
    entire long-duration axis produced zero points, silently, in a result set
    that otherwise looked complete. It is the axis that matters most: the only
    one approaching the slow consumers' impulse responses.

    This is the **fourth** cross-boundary format defect in this project —
    1-based inclusive vs 0-based half-open indices, the v7.3 `(2, N)`
    transpose, `hash()` salting, and now numeric formatting. Route every such
    value through **one shared canonicalising function**, never through two
    formatters that happen to agree on the cases you tried.

23. **Cost-model the work before running it, and treat an arithmetic
    disagreement as a defect signal.** The 5-second bug was not caught by a
    test; it was caught because a masked manifest came out at 1195 points
    against a costed 1710, and an unexplained gap in a plan that had just been
    cost-modelled is where a quiet failure hides. A count that does not
    reconcile is evidence, and it is often the *only* evidence a silent
    data-loss bug produces.

24. **A value fixed by the cohort is declared once, not copied into every
    record.** Writing a constant into 967 per-recording files is 967
    opportunities to type `uV`, and the first file that disagrees with the
    other 966 is indistinguishable from a real finding. Cohort-level constants
    — `units`, electrode `config`, `channel_order_source` — live in
    `protocol.yaml`; `meta.json` carries only what actually varies between
    recordings. The test for which is which: *could two recordings in this
    cohort legitimately differ here?* If no, it is not a per-recording field.
    Corollary: do not pair a value with a boolean saying whether the value is
    known. `rostral_end: None` already means unknown; a separate
    `rostral_end_known: false` is a second source of truth that can disagree
    with the first, and eventually will. One field, `None` for unknown, and the
    schema requires the key to be present so absence is never ambiguous.

25. **When a new validity check breaks an existing test, the fixture is the
    prime suspect, not the check.** A test that passes only because the data it
    feeds the code is physically impossible was testing the mechanism against a
    world that cannot happen. The fix is to make the fixture possible and keep
    the check global — never to exempt the test, because the exemption is
    permanent and the next person reads it as "this check does not apply here".
    *Found in the `meta.json` work:* `assert_plausible_units` immediately failed
    `test_the_declared_units_are_applied[mV]` and `[V]`, both of which asserted
    that the loader accepts a recording with a several-volt noise floor. Scaling
    the fixtures preserved exactly what those tests were for — that the declared
    scale factor is applied — and removed a claim nobody meant to make.

---

## Cross-platform rules (macOS + Windows; Linux best-effort)

Development may happen on **either** macOS or Windows — the first build session
ran on Windows — and the shared drive is read from both. Breakage on the platform
you are not sitting at is invisible, so **CI must run the test suite on
`windows-latest` and `macos-latest` from task 00 onward**, before there is much to
test. Python 3.12 is the reference interpreter; `requires-python = ">=3.11,<3.14"`.

**Paths**

1. **`pathlib.Path` everywhere.** Never string-concatenate a path, never hardcode
   `/` or `\`, never `os.path.join` on a string built elsewhere.
2. **Never store an absolute path in anything shared.** Corpus specs, the
   registry, provenance, `meta.json`, label files — all store paths **relative to
   `gems_root`, in POSIX form** (`as_posix()`), and resolve against the local
   root at read time. An absolute path saved on a Mac cannot resolve on Windows,
   which would silently break every corpus and every model's provenance.
3. **`gems_root` is discovered, never assumed**, by locating the `.gems-root`
   marker. The roots genuinely differ:
   `~/Library/CloudStorage/GoogleDrive-<acct>/Shared drives/<name>` on macOS,
   `G:\Shared drives\<name>` on Windows (drive letter varies), rclone on Linux.
4. **This lab's shared drive is named `BIONICs Lab: Enteric Interfaces Team`, and
   `:` is illegal in a Windows path.** Google Drive for desktop substitutes
   illegal characters, so **the folder name is not the same string on Windows**.
   This alone makes rule 2 non-negotiable. Verify what Windows actually produces
   before the first Windows user is onboarded, and never reconstruct the root by
   pasting the drive name.
5. **Windows `MAX_PATH` is 260** unless long paths are enabled. The real data
   already sits at ~193 characters on Windows before this tool adds
   `models/<32-hex>/shap/<feature>.html`. Keep generated path segments short,
   test the deepest path the layout can produce, and fail with a clear message
   rather than an `OSError` if it is exceeded.
6. **Never create symlinks** (Windows needs elevation) and never rely on hard
   links.

**Filenames**

7. **Match filenames case-insensitively.** Every rule in `conditions.yaml`
   compiles with `re.IGNORECASE`. This lab's own data mixes case for the same
   entity — `gems_d_t01_ms1_bl_164012/` beside
   `GEMS_D_t01_MS1_bl_cam1_....mp4` — so case-sensitive matching would work on
   macOS and Windows and fail on Linux, or vice versa.
8. **Never let two paths differ only by case.** macOS and Windows are
   case-insensitive; such a pair silently collides. Assert on collision when
   scanning.
9. **Sanitise anything used to build a filename**: reject `<>:"/\|?*`, trailing
   dots and spaces, and the Windows reserved names (`CON`, `PRN`, `AUX`, `NUL`,
   `COM1`–`COM9`, `LPT1`–`LPT9`).

**File I/O**

10. **Text files open with `encoding="utf-8"` and `newline="\n"`, explicitly**,
    for both read and write. Default encoding is not UTF-8 on all Windows
    installs, and the registry JSONL must not acquire CRLF.
11. **Writes are atomic**: write to a temp file in the same directory, then
    `os.replace()` (atomic on both platforms). Never write a shared file in
    place.
12. **Assume a file may be locked.** Windows holds exclusive locks on open files,
    so a reader can block a writer. The append-only shard design already avoids
    this — do not add a mutable shared file.

**Runtime**

13. **Guard every entry point with `if __name__ == "__main__":`.** Windows
    multiprocessing uses `spawn`, not `fork`; without the guard a parallel job
    re-imports and re-executes the module.
14. **No shell scripts in the tool.** `gems doctor` and every other command is a
    Python console-script entry point, not `.sh`. Do not call `sed`, `awk`,
    `find` or `which` from code.
15. **Per-user config location** via `platformdirs`, not a hardcoded `~/.gems` —
    that resolves to `%LOCALAPPDATA%` on Windows. **Exception, and it is a real
    one: a file owned and read by another tool is written where that tool reads
    it.** The `detector-pyqt` preprocessing profile lives at
    `~/.detector/preprocessing_profiles/<animal>.json`; writing our copy through
    `platformdirs` would make the interop useless. Document the exception at the
    call site. The rule is about config *we* own.
16. **No `matplotlib` GUI backend assumptions** — set `Agg` for any headless
    figure generation.

**This applies to the repo's own tooling too.** `split_tasks.py` read
`IMPLEMENTATION.md` with the platform default encoding and died on the first em
dash under cp1252 — rule 10, broken by the file that enforces the rules. Any
script in this repo obeys the same contract.

**Declared support must equal tested support.** `requires-python` and the CI
python matrix are one decision, not two. If a version is too expensive to test,
narrow the declaration instead of leaving it untested.

**Required tests** (these belong in `tests/test_portability.py`)

- no absolute path appears in any emitted corpus spec, registry line or
  provenance record
- a corpus spec written with POSIX separators resolves correctly when
  `gems_root` is a Windows-style path (monkeypatch it)
- condition rules match the same file whatever the case of its name
- a scan containing two names differing only in case raises
- the deepest path the layout can generate is reported, with a failure if it
  exceeds 260 characters under a Windows-style root
- every emitted text file round-trips with `\n` endings and UTF-8

---

## Conventions

| Thing | Rule |
|---|---|
| Time in public APIs | seconds, `float64` |
| Time internally | sample indices, `int64`, always paired with an explicit `fs` |
| Amplitude | microvolts, `float64` |
| Frequency | Hz |
| Envelope / z grid | **10 ms**, shared by every band. `n_frames = floor(dur/0.010)` |
| Band naming | `"300-3000"`, `"100-300"`, `"10-150"`, `"2-50"`, `"0.5-3"`, `"0-2"` — `CONSUMERS` may only name a key of `BANDS`, and a test asserts it |
| Signal naming | `"V1".."V3"`, `"T"` per cuff, prefixed by cuff: `"L_V1"`, `"R_T"` |
| Missing scalar, **in memory** | `np.nan`, never `0`, never `-1` |
| Missing scalar, **serialised to JSON** | **the key is absent** — never `null`, never `NaN`, never a sentinel. JSON has no NaN: `json.dumps` emits a bare `NaN` that Python reads back and almost nothing else does, and `nan != nan` breaks round-trip equality outright. The `np.nan` convention stops at the edge of a JSON file. A metric that could not be computed is **absent**. |
| Reading such a field | accept **absent and `null` identically**; write only absent. A required field that is absent or null raises **naming the field**, rather than being defaulted into a record nobody wrote. |
| Boolean masks | `True` = **invalid / masked out** |
| Random seeds | every test and every synthetic generator takes an explicit seed |
| Filters | design with `output='sos'`, apply with `sosfiltfilt`. **Never** `butter(...,'ba')` at these ratios — a 1 Hz corner at 24.4 kHz is numerically unstable and silently returns garbage |
| Decimation | `scipy.signal.decimate(..., ftype='fir', zero_phase=True)` before any sub-100 Hz filtering |

---

## Repo layout

```
gems-blanking-v2/
  gems_blanking_v2/          <- everything importable lives under here
    types.py                 A.1 dataclasses (created in task 00)
    constants.py             GRID_S, BANDS, REFERENCE_STATISTIC, CONSUMERS
    io/        recording load, channel map, store, scan
    derive/    derivations (V1,V2,V3,T)
    physio/    rpeaks, cardiac_window
    bands/     envelope, reference, zscore
    detect/    candidates, features
    model/     train, evaluate, registry  (imports GEMSBlanking)
    extent/    tolerance, routing
    emit/      masks, qc, provenance
    video/     motion
    velocity/  xcorr
  tests/       conftest.py + one test module per source module
  .github/workflows/ci.yml   ubuntu + windows + macos matrix
  IMPLEMENTATION.md  CLAUDE.md  PROMPTS.md  split_tasks.py  tasks/
```

**Never a top-level `io/`** — it shadows the stdlib `io` module.

## Reused from `GEMSBlanking` (private, import — do not fork)

`detector/retrain.py`, `detector/review.py` (SHAP), `detector/heldout_eval.py`,
`detector/animal_id.py`, `detector/recording_io.py`, and the Phase 2 synthetic
machinery. Paths came from `detector-pyqt/DEVELOPER_GUIDE.md` — **verify each
before importing** and report any that moved instead of guessing.

## Model modes

Three training modes share every step of the pipeline and differ only in training
corpus and evaluation protocol (task 12):

- **POOLED** — all animals except the target. **Mandatory**: the only mode that can
  process an animal with no labels.
- **ADAPTED** — pooled prior + the target animal's own labelled events. Matches the
  actual deployment scenario, since ~50 events per new animal get labelled anyway.
- **PER_ANIMAL** — the target animal only.

Task 12A does **not** choose between them: the user picks the model for every
inference run, and the registry's job is to present the options with their metrics
and protocols and to record the choice in provenance. There is no default model and
no fallback chain.
`PER_ANIMAL` beating `ADAPTED` is a **task 11 feature-invariance bug**, not a
result to ship — see task 12.

## Testing

- `pytest` per module against **synthetic signals with known ground truth**.
- `hypothesis` for anything serialised, hashed, or compared — property tests, not
  examples. Example-based coverage is what let three separate serialiser defects
  through in 00A; the property suite found all three in one pass.
- **Verify the test, not just the code:** revert each fix in turn and confirm the
  test written for it actually fails. A test that passes against the bug it was
  written for is worse than no test.
- `tests/conftest.py` owns every generator. No test invents its own signal.
- Every numeric claim in a docstring must have a test that would fail if it were
  wrong.
- A test that passes because a threshold was loosened is a failed test. If a
  tolerance must be widened, say so explicitly in the task report.

## Who owns which files

The spec is authored in one place and the code in another, on different
machines. Clobbering is the hazard, so ownership is explicit:

| File | Owner | Rule |
|---|---|---|
| `IMPLEMENTATION.md`, `CLAUDE.md`, `PIPELINE.md`, `PROMPTS.md` | the spec author | Claude Code does **not** edit these. Propose changes in the task report; they come back in the next drop. |
| `tasks/` | generated | never hand-edited by anyone; regenerate with `split_tasks.py` |
| `split_tasks.py`, all code, all tests, CI config | the repo | the spec author does **not** ship copies of these |

A doc drop therefore replaces exactly four files and can never overwrite code.
Once the repo is on a remote, drops become pull requests and this stops being
a manual step.

## Writing into a file another tool owns

Verified the hard way in task 03: `detector-pyqt`'s `Profile.load` reads
field-by-field, so an unknown key does not raise — but `Profile.save` writes
`asdict(self)`, so **any top-level key we add is silently dropped the next time
their UI saves.** Extra fields must go inside a sub-object that tool carries
through opaquely (`channel_assignment`), and a test must write with our code,
round-trip through *their* class, and read back.

**Single source of truth for geometry is ours, not theirs.** `cuff_id`,
`contact_index` and `rostral_end` live in our store
(`data/<animal>/<session>/meta.json`); the copy in their profile is a **mirror**
for their UI's benefit and may be regenerated from ours at any time. Their
`from_review_session` rebuilds the channel list from a dialog that knows nothing
about geometry, so a user re-running channel assignment in their UI would
otherwise destroy `rostral_end` — which cannot be recovered once the animal is
gone.

## Two testing rules that came from real failures

- **Tests must be hermetic.** No test may read or write the developer's real
  per-user config, `GEMS_ROOT`, or anything outside `tmp_path`. Use an autouse
  fixture that isolates both. *Found in 00A:* running `gems init` once made two
  discovery tests pass for the wrong reason — they would have passed on a clean
  machine and failed on a colleague's.
- **Serialised dedup keys use canonical, ASCII-escaped JSON** (`ensure_ascii=True`,
  sorted keys, fixed separators). `ensure_ascii=False` lets U+2028, U+2029 and
  U+0085 into a JSONL line literally: a reader splitting on `\n` is fine, but
  `str.splitlines()` and most other languages' line splitters see one record as
  two malformed ones — and these files are written to a shared drive for other
  people's tools. Unicode still round-trips exactly, as `\uXXXX`.
- **Anything used as a dedup or identity key must round-trip exactly.** If
  `parse(serialise(x)) != x` for any field, deduplication silently fails.
  *Found in 00A:* `RegistryEvent(metrics=None)` serialised to `{}` and parsed
  back as `{}`, so an event never equalled its own reparse — and conflict-copy
  dedup is built on exactly that comparison. Property-test the round trip.

## Definition of done, per task

1. Module implemented with type hints and docstrings stating units.
2. Tests pass, including the task's own acceptance criterion.
3. `ruff check` and `mypy` clean.
4. A short report: what was built, what was measured, anything that contradicted
   `IMPLEMENTATION.md`.

**Contradictions are the most valuable output.** Several constants in this spec
came from simulation, not from Andrea's data. If real data disagrees, report the
disagreement — do not tune the code until it matches the spec.

## Do not re-implement

See `PIPELINE.md` §10. In particular: adaptive-threshold QRS detection
(Pan–Tompkins), cardiac template subtraction, rate targeting, HMM/Viterbi
smoothing, pooled clean-null calibration, per-cohort model splits, detection on the
tripole. Each was tested or ruled out on a stated requirement.
