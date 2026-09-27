<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

## Task 16 — Labelling UI changes

**Module:** `detector-pyqt`
**Depends on:** 07

### Reuse
`signal_viewer.py`, `overview_strip.py`, `region_table.py`, `predictions_panel.py`,
`review_panel.py`, `queue_panel.py`, 20-deep undo, keyboard interval stepping.

### Change 1 — candidate adjudication becomes the primary mode
`review_panel.py` already does candidate review (context plot, SHAP, 1/2/3 scoring,
Shift+drag to widen). Promote it from a secondary panel to the main interaction,
replacing free interval marking.

### BLOCKER FOUND 2026-09-24: no `meta.json` exists for new-cohort data

Audit mode works end to end on `gems_j_t01_ms3_bl_230315_sig.mat` — loads in 4 s,
54 pairs reduced to 6 traces, BLIND clean, reveal populated, diagnosis correct
over 300 windows — but **the loader refuses to open it without geometry, and no
`meta.json` exists for any new-cohort recording.** The refusal is correct
behaviour, not a bug. It means **labelling cannot start until that file is
written**, which makes `meta.json` generation the critical path to the 09 gate,
ahead of anything else in task 16.

It belongs to 00A/03's store, not to 16.

#### Two silent errors the stand-in fixture made, and the rule that prevents both

1. **Channel order.** The fixture built cuffs `("L", "R")` and labelled column 0
   `LVN1`. The file's own `chanlabels` are `RVN1-3, LVN1-3, ANT1-3` — **R
   first**. Every left/right comparison would have been inverted, silently.
2. **Units.** The fixture declared `units="uV"`. Robust σ is 8.3×10⁻⁶ in file
   units, which as µV would be 8.3 **pV**. The file is in **volts**. A 10⁶ error,
   silently.

**The rule: channel order is read from the file's `chanlabels`, never declared
separately.** The labels are in the file and are authoritative; a `meta.json`
that restates them is a second source of truth that can disagree, and this one
did. `meta.json` declares only what the file does *not* state — units, geometry
(pitch, aperture), animal and session identity — and the loader **asserts its
declared order against `chanlabels`** and raises on mismatch.

**Units are VOLTS, confirmed by Andrea 2026-09-24, and they do not vary across
cohorts.** So `units` is **not a per-recording field**. A constant that never
changes, written into 967 files, is 967 opportunities to type `uV`; declared
once it is one. Put it in `protocol.yaml` beside the other cohort-level
constants, and let `meta.json` carry only what genuinely varies per recording.
A per-recording override may exist for the day a rig changes, but it must be
absent by default rather than restated.

**Units stay declared (invariant 14) but an implausible declaration must fail
loudly.** Declaration prevents inference; it does not prevent a typo. Add a
plausibility check at load: a declared unit that puts robust σ outside roughly
1–500 µV for a nerve or stomach contact raises, naming both the declared unit
and the implied σ. That catches a 10⁶ error without inferring anything — the
declaration still decides, it just has to survive contact with the data.

#### Adjudication of the `meta.json` implementation, 2026-09-24

The implementation was built before this section reached the Windows clone, from
the chat message rather than the spec. It landed on the same design anyway. What
follows ratifies it and records the four places it must change.

**Ratified as built.**

- `read_channel_labels()` **returning `None` for the old five-channel cohort is a
  real answer, not a failure.** Those files carry no `chanlabels`; `None` means
  "the file does not state its order", which is true, and the old cohort's order
  comes from the cohort declaration instead. A function that raised there would
  be reporting a defect in data that is merely older.
- `channels_from_labels()` **taking file order and raising on an unrecognised
  stem.** Raising is right: an unrecognised stem means either a new electrode
  type or a corrupt file, and both need a human. Guessing from position is how
  the `LVN1`/`RVN1` inversion would have survived.
- `assert_labels_match()` **wired into `load_recording`, case-insensitive.**
  Case is a formatting difference, not a geometry difference.
- `assert_plausible_units()` **on the median robust σ across contacts, band
  1–500 µV.** The median, not the mean — one saturated contact must not move the
  verdict. The band is only 500× wide, and the errors it exists to catch are
  10³ and 10⁶, so it has margin on both sides without being tight enough to fire
  on a genuinely quiet or genuinely noisy recording.
- **Scaling the two fixtures rather than exempting the check.** See invariant 25.
  `test_the_declared_units_are_applied[mV]` and `[V]` were asserting that the
  loader accepts a recording with a multi-volt noise floor; that was never what
  they were for.
- `write_new_cohort_meta.py` **dry-run by default.**

**Four required changes.**

1. **`units` moves to `protocol.yaml` and `units_provisional` is deleted.** Units
   are volts, confirmed, cohort-wide and unchanging — there is nothing
   provisional left to flag. A field whose only value is `true` in every file is
   not information, it is a note-to-self that outlives the doubt that produced
   it. (Invariant 24.)
2. **`config` and `channel_order_source` move to `protocol.yaml` too.** Same
   argument, same test: *could two recordings in this cohort legitimately
   differ here?* They cannot. Every new-cohort file has three independent ADCs
   per cuff, so `config: "independent"` is a property of the cohort; every
   new-cohort file carries `chanlabels`, so `channel_order_source: "file
   chanlabels"` is too. `config` is genuinely cohort-distinguishing — the old
   cohort is hardware-shorted tripolar — which is exactly why it belongs in the
   per-cohort file rather than in 967 copies.
3. **`rostral_end_known` is deleted; `rostral_end: None` carries it.** The schema
   requires the key to be present, so `None` unambiguously means unknown and the
   boolean is a second source of truth that can disagree with the first.
   (Invariant 24, corollary.)
4. **The dry run prints distinct value-tuples with counts, not 967 lines.** The
   whole point of a dry run over a near-uniform corpus is to surface the file
   that is *not* like the others; a count of 1 beside one tuple does that in a
   glance, and 967 near-identical lines hide it. Reconcile the count of files
   written against the count of recordings enumerated and state both — invariant
   23 exists because a number that did not add up was the only evidence of a
   silent failure.

**One test to add:** a recording whose numbers are µV-scale but whose declaration
says `V` must raise from `assert_plausible_units`, with the message naming both
the declared unit and the implied σ. A check nothing exercises is a check that
silently stops working. The two scaled fixtures prove the check tolerates correct
data; nothing yet proves it rejects wrong data.

**Post-run:** record the observed median robust σ across the new cohort in
`protocol.yaml` as a comment. If it clusters tightly the 1–500 µV band can be
tightened later on evidence rather than on my guess.

#### STOP: 27 store-path collisions, found 2026-09-24

The generator's own dry run printed this and it was not acted on:

```
recordings enumerated (*_sig.mat): 859
ready to write:                    837
distinct store paths:              810
COLLISIONS (stem repeats by date):  27
```

**837 files written into 810 distinct paths means 27 `meta.json` were
overwritten by a second recording's.** Each surviving file now describes one
recording while sitting in the store path of another, and `meta.json` carries
`animal`, `session` and `channels` — so 27 recordings are one load away from
being labelled with a different recording's identity and channel order. This is
the *same* failure the whole `meta.json` exercise exists to prevent: the
`LVN1`/`RVN1` inversion, except distributed across 27 files instead of a
fixture, and not detectable from the signal.

**Required, before any further writing:**

1. **The generator refuses to write when a store path is not unique.** Not a
   warning line in a summary — a raise, naming every colliding group. A path
   that two recordings both claim is not a path, it is a bug in the key.
2. **Enumerate the 27 and diagnose the key.** Stem-plus-date is evidently not
   unique. Print each colliding group with the full source paths, file sizes and
   mtimes. The two likely causes are a genuine same-day repeat session (in which
   case the key needs the acquisition time or the block index) and the same
   recording reachable by two Drive paths (in which case one is a conflict copy
   and the store is fine but the enumeration is double-counting). These need
   opposite fixes, so do not pick one — read the groups.
3. **Re-derive the key from something the acquisition system guarantees unique.**
   The TDT block directory name is the candidate; it already encodes date and
   time. A key assembled by us from parts we chose is a key we have to prove
   unique, and we just found out it is not.
4. **Then re-run and reconcile to zero.** `enumerated = written + skipped +
   collided`, every term printed, every skip reasoned.

**And three counts in this report do not reconcile.** 859 enumerated against 837
ready leaves 22 unexplained; the prose then says 837 rewritten and, later, 841
written this run. A four-file discrepancy inside one report about writing files
is exactly the signal invariant 23 is about — the 5-second bug announced itself
as 1195 against 1710 and nothing else. Do not rationalise the gap; find it.

#### RESOLVED 2026-09-25: the 27 are cause (b), and the re-key was the wrong instruction

The groups were read before the key was touched, and the answer reverses the
first report. **All 27 signal arrays are bit-identical**; only `createdAt` and
`srcBlock` differ, and the two `srcBlock` values are the same block name under
`09032026` and `09042026`. Eight million by nine float32 samples of biological
noise cannot agree bit-for-bit across two acquisitions. This is one recording
reachable by two Drive paths. **No `meta.json` was overwritten by a *different*
recording's** — each pair wrote identical content — so the store was never
corrupt, only double-counted.

**My step 3 was wrong and should not have been written.** I told it to re-key
off the TDT block *before* the groups had been read, having just objected in the
same message to acting on an unread diagnosis. The re-key would have made the
key finer, which for duplicates is the wrong direction: it would have produced
27 duplicate store entries instead of collapsing them, and left the actual
duplication untouched. The refusal, with the evidence attached, was correct, and
declining a spec instruction on evidence is the behaviour this document wants.
**Ratified: no re-key.** The guard is refusal-plus-verified-dedup.

**The comparison method is the durable finding.** `_sig.mat` keeps `createdAt`
in the header and `srcBlock` in trailing variables — exactly the regions a
head-and-tail hash samples — so the intuitive fast check reports "two different
recordings" with total confidence on two conversions of one block. The second
attempt, a fixed mid-file range, failed differently: two groups differ by three
bytes of metadata string, which shifts every later offset, so the same byte range
lands in different places. **Only the decoded `signal` array answers the
question.** Identical byte size is a reason for suspicion, not reassurance. This
is recorded at the top of `verify_duplicates.py` because the wrong method is the
one that comes to mind first.

#### CLOSED 2026-09-25: b1, and no recording is missing

| check | result |
|---|---|
| shared stems under both dates | 27 |
| identical TDT block name (acquisition timestamp) | 27/27 |
| identical `.tsq` and `.tev` sizes | 27/27 |
| blocks only under `09042026` | 36 |
| …converted to `_sig.mat` | 36/36 |
| …present in the store with `meta.json` | 36/36 |

**The raw TDT blocks themselves are duplicated.** The converter did not read
`09032026` twice; it read two copies of one block. `09032026` holds 27 blocks
and is *wholly contained* in `09042026`'s 63. Nothing is missing, nothing is
corrupt, and this is Andrea's Drive to tidy, not the build's.

**Do not delete anything.** The duplicate folder costs space and nothing else,
and the enumeration handles it. Deleting raw acquisition data to tidy a
bookkeeping artefact is a bad trade in the wrong direction.

#### The block name carries no date, and the folder date is now known-unreliable

Every block under both folders carries the same prefix,
`ME_STIM_Andrea-260824-155430` — the **tank** creation stamp (2026-08-24), not a
per-block one. So the block directory is `<tank>_<stem>`, the stem's trailing
digits are a **time of day** (`gems_h_pre01_160301` → 16:03:01), and **nothing
in the name says which day.** The only thing that did say was the containing
folder, and we have just proved a folder date can be wrong: 27 blocks sit under
a date that is not theirs.

Two consequences.

**1. Acquisition datetime comes from the `.tsq` header, not from the folder.**
The block's own start timestamp is in there — it is what the b1/b2 check just
read — so the authoritative value is available for the cost of a header read.
Add `acquired_at` to `meta.json` from the `.tsq`, and **audit the existing 837
for any `session` value derived from a folder name.** A field taken from the
filesystem when the instrument recorded it itself is a second source of truth,
and this one is already known to disagree. (Invariant 30.)

**2. The stem key is safe only because the guard raises.** Name-plus-time-of-day
with no date means two genuine recordings — same animal, same condition label,
same second on different days — would collide, and across a corpus still growing
past 886 that is a question of when, not whether. The dedup guard catches it:
such a pair fails the identical-signal test and raises. **That guard must never
be downgraded to a warning or a skip.** It is not a migration convenience for
the 27; it is the thing that makes a dateless key tolerable, and `acquired_at`
from the `.tsq` is what will eventually let the key stop being dateless.

And a third reason the re-key was wrong, better than the two already recorded:
the block directory name is `<tank>_<stem>` with a tank stamp shared by every
block, so it is **stem-equivalent for uniqueness**. Re-keying to it would have
bought precisely nothing.

#### RE-KEY NOW, on `acquired_at` — and why this one is right where mine was wrong

`acquired_at` exists for all 837, from the `.tsq`. That changes the key decision,
and the reason is the whole point of invariant 30.

**My withdrawn re-key would have used the folder date, which *differs* between
the two copies — it would have split 27 duplicates into 27 spurious store
entries.** A key on `acquired_at` is *identical* across a duplicate pair, because
the instrument wrote it once and the copy carries it. So the same idea, sourced
from the instrument instead of the filesystem, collapses the duplicates
correctly **and** separates two genuine recordings of the same animal and
condition started at the same clock second on different days. Opposite outcomes
from the same-shaped change, decided entirely by where the value came from.

**Do it now, before labelling starts.** This is the cheapest this will ever be:

- nothing references a store path yet — no labels exist, and
  `duplicates.json` / `new_cohort_meta_plan.json` both regenerate;
- the generator is already idempotent and verified to reconcile to zero, so the
  re-key is one run of a thing that works;
- after labelling begins, changing the key means migrating labels, and the
  dateless key becomes permanent by inertia.

The alternative is keeping the guard forever and accepting that it will raise at
some arbitrary future moment — which is safe, but the moment it picks will be in
the middle of labelling, which is the critical path.

**Before running it:** enumerate every reference to a store path anywhere in the
repo and in the Drive store, and state the list. The claim "nothing references
it yet" is the kind of negative that invariant 20 exists about — establish it,
do not assume it. The dedup guard stays exactly as it is afterwards; it is
cheap, and a key being better is not a reason to remove the thing that catches
the case where it is not.

#### Store the zone, not just UTC — November will break a hardcoded offset

UTC with the zone named was the right call, and the example is the argument:
`gems_j_t01_ms3_bl_230315` is 23:03 **EDT on the 15th**, which is 03:03 **UTC on
the 16th**. The stem's trailing digits are **local** time; `acquired_at` is UTC;
the dates disagree by one for every recording after 20:00 EDT.

Two things follow.

1. **Store the IANA zone (`America/New_York`) alongside the UTC timestamp**, not
   a fixed offset. The corpus is Aug–Sep now, but recovery recordings will run
   past the November DST change, and a hardcoded −4 silently becomes wrong for
   every recording after it — while continuing to produce plausible times. With
   the zone stored, local date is derivable forever; with an offset, it is
   derivable until it isn't.
2. **Any grouping by "day" uses the LOCAL date.** Session grouping, baseline /
   recovery pairing, the per-recording motility state from task 08, and the
   LORO splits all mean the day Andrea ran the animal, not the UTC day.
   Grouping on the UTC date pushes every evening recording into the next day's
   group — a quarter of the corpus landing in the wrong session, produced
   silently, and indistinguishable in the output from a real scheduling
   difference. State the convention at the point where the grouping is computed.

#### The `mmc_burst` twin: declare the call-to-fiducials map

The grep found the second instance and it was worse than the first.
`extract_mmc` produces both `mmc` and `mmc_burst`, gated on `want.mmc` alone,
and worked **only** because a string special-case in the mask loop forced
`want.mmc` true whenever `mmc_burst` was masked in. Since `mmc` is
`criterion_degenerate`, real masks contain `mmc_burst` and **never** `mmc` — so
that special-case was load-bearing for every mmc_burst point in the sweep, and
deleting it as dead tidy-up would have produced nothing for all of them.

**A special case that patches up a flag is evidence the flags do not model the
call graph.** Two instances found in one grep is a pattern, and the structural
fix prevents the third: **declare the call→fiducials map once**, and derive both
the gate and the recording from it —

```
HR_BR_HRVAnalysis_new -> {hrv, breathing}
extract_mmc           -> {mmc, mmc_burst}
```

A call runs iff `mask ∩ outputs(call) ≠ ∅`, and records exactly
`mask ∩ outputs(call)`. Hand-written boolean gates are then impossible to get
wrong, and a fiducial added to an existing call later cannot be forgotten —
which is the failure mode that produced both of these.

**One regression check before moving on.** The refactor is structural and the
old special-case meant mmc_burst results were *correct*, so the 1710 rows should
stand — but "should" is not "do". Re-run five mmc_burst points under the new
code and compare against the stored rows. If they match, say so and keep the
sweep. If they do not, the refactor changed behaviour and the pass is suspect.

#### The serial control: paired, stratified, and with the decision rule fixed in advance

The control exists to answer one question — *is the 2× miss contention or a bad
per-point model?* — and it only answers it if it is designed so that the two
answers look different. Three requirements:

1. **Paired, not fresh.** Re-run **the same point indices** whose `wall_s` is
   already stored from the 8-worker pass. A serial run of different points
   compares two distributions and answers nothing; the same points compare two
   conditions.
2. **Stratified across the duration axis**, ~10 points, because per-point cost
   scales with duration and a mean over a lopsided sample is a mean over the
   wrong thing.
3. **On a quiet machine.** If labelling or anything else is running on that box,
   the control is measuring a third condition. State what else was running.

Then `ratio = parallel_per_point_wall / serial_per_point_wall`, and the action is
**decided now**, so the number does not get re-argued once it has a value:

| ratio | reading | action |
|---|---|---|
| ≥ 1.5 | contention dominates — each point is inflated by sharing one I/O path | sweep workers at 2 / 4 / 8 on one fixed small set, take the knee; do **not** assume more is faster |
| ≤ 1.2 | no meaningful contention — the per-point estimate is simply wrong | recalibrate the cost model's coefficients from the measured serial times and retire the blanket 2× |
| 1.2–1.5 | both, in some mix | recalibrate **and** test 4 workers |

Report the serial per-point mean, the ratio, and the implied parallel efficiency
against the 53% already measured. The blanket empirical 2× stays on every
projection until this lands, and is replaced by the recalibrated model rather
than removed.

#### The serial control landed, the decision rule was wrong, and the remedy is upstream of the worker count

**The rule I wrote was defective and the spec is the thing to blame.** It asked
for stratification across the **duration** axis and then reduced the answer to a
single ratio. Duration was not the axis carrying the variance — consumer was —
and the duration-stratified sample drew zero `slow_wave` points and returned a
tidy aggregate **1.28**, landing in a band whose prescribed action ("recalibrate
and test 4 workers") was wrong for every consumer individually. The measurement
was executed correctly against a specification that measured the wrong thing.

| consumer | serial | parallel | ratio |
|---|---|---|---|
| `hrv` / `breathing` | 6.0 s | 6.2–7.1 s | 1.02–1.21 |
| `T_hardware` | 12.0 s | 12.8–14.9 s | 1.06–1.23 |
| `mmc_burst` | 8.5 s | 11.4–13.4 s | 1.29–1.57 |
| **`slow_wave` (180 s)** | **15.9 s** | **98.9 s** | **6.23** |
| **`slow_wave` (420 s)** | **32.2 s** | **237.2 s** | **7.36** |

**And the per-point model was never the problem.** `slow_wave = 15.2 s` in the
model against 15.9 s measured serially is a good coefficient. The blanket 2× was
one consumer behaving differently under parallelism, averaged across a
population where nothing else does.

#### Do NOT run the 2/4/8 sweep yet — three cheap measurements come first

A worker sweep costs hours and is downstream of a question that costs minutes.
A 6–7× per-point inflation at 8 workers on a **32-core** machine has three
distinct explanations with three opposite remedies, and the sweep cannot tell
them apart:

1. **MATLAB is implicitly multithreading and the machine is genuinely
   saturated.** `filtfilt` and the BLAS calls underneath it use every core by
   default. If one `slow_wave` run takes 15.9 s across ~32 threads, that is
   ~500 core-seconds of real work; eight of them is ~4000 core-seconds, which on
   32 cores is ~125 s per point. **The observed 98.9 s is consistent with
   this.** If so, the 6.6 h is near the machine's capacity, 8 workers is already
   right, nothing is being wasted, and the fix is not scheduling at all.
2. **Oversubscription.** 8 processes × 32 threads = 256 threads thrashing 32
   cores, where the same work would finish faster with each worker pinned to one
   thread. Remedy: `maxNumCompThreads(1)` per worker — and then possibly *more*
   workers, not fewer.
3. **I/O bound.** Cores idle, one Drive path saturated. Remedy: fewer workers,
   or stage segments to local disk first.

Measure, in this order, during an 8-worker `slow_wave`-heavy burst:

```
maxNumCompThreads inside a worker        -> is it 32 or 1?
total CPU utilisation across 32 cores    -> saturated or idle?
disk read throughput                     -> pegged or quiet?
```

Saturated + multithreaded is (1). Idle cores with high context-switching is (2).
Idle cores with a pegged disk is (3). **Report all three numbers**, then act.
The 2/4/8 sweep is only worth running under (2) or (3).

#### The likely real win is algorithmic, not scheduling

`slowWaveAnalysis_new` low-passes for a gastric slow wave at **~3 cpm = 0.05 Hz**
in data sampled at **24414 Hz**. Even the burst content of interest is well
under 50 Hz. Filtering at the full rate to extract a 0.05 Hz rhythm does on the
order of a hundred times more arithmetic than the signal requires.

**Check whether it decimates before filtering.** Anti-alias decimate to ~200 Hz
first — a factor of ~120 in filter work — then run the existing low-pass on the
decimated series. If `slow_wave` drops from 15.9 s to a fraction of a second,
the contention question dissolves rather than being scheduled around, and it
takes `slow_wave` off the critical path for every future pass as well.

Do this check before the worker sweep too. An algorithmic factor of 100 makes a
scheduling factor of 2 irrelevant, and the order matters: tuning the worker
count around a filter that should not cost this much bakes the waste into the
plan. If it already decimates, say so and the question is closed.

#### Recalibration, once the above is known

Do not recalibrate the model to the *parallel* per-point times. Those are a
property of a worker count, not of the work, and freezing them into the model
makes every future projection wrong the moment the worker count changes. Model
**serial per-point cost** — which is already close to right — and apply a
separate, named contention factor per consumer, since the measurement just
showed contention is per-consumer and not global. The blanket 2× is retired by
that, not by an aggregate replacement.

#### The fourth case: under-subscription, and the machine is three-quarters idle

None of my three hypotheses was right, and the reason is that I never questioned
the **baseline's** thread budget. MATLAB gives each pool worker **one** thread;
the serial control ran in the client with **24**. The 6–7× "contention" was the
multithreading speedup the workers never had. Thread-matched, every
parallel/serial ratio is **0.91–1.24**. There is no contention to relieve and
the 2/4/8 sweep was correctly not run. (Invariant 36.)

**The finding nobody has acted on is in the same table: 26% CPU, ~8 cores busy,
24 idle, run queue 0, disk 99% idle.** Eight single-threaded workers on a
32-core machine leave three quarters of it unused, and the fix is to add
workers, not threads. The measured numbers decide this:

- MATLAB's intra-call threading gives `slow_wave` **5.95×** on 24 threads
  (89.9 s → 15.1 s) — **25% efficiency**.
- Running independent points in separate processes gives ratios of 0.91–1.24 —
  **~100% efficiency**.

So a core spent on a process is worth about four spent inside one call. **Raise
the worker count toward the core count**, single-threaded, rather than giving
workers more threads. Set it by memory, not by guesswork: measure a worker's
peak RSS, then `workers = min(cores − 4, floor(0.8 × available_RAM / RSS))`, and
report both numbers. Leave the headroom — the machine also has to stay usable
for labelling.

#### Decimation: NO for T. The arithmetic says it is not needed.

Decimating changes `slowWaveAnalysis_new`, and Step 9's premise is that **a
tolerance is a property of the consumer as it is**. Measuring a decimated
consumer yields a tolerance for a pipeline Andrea is not running. That alone
would make it a bad trade — and the cost table says the trade buys nothing:

| configuration | extension (1510 points) |
|---|---|
| as measured, 8 workers | 5.0 h |
| duplicate `smoothdata` removed (−47% of `slow_wave`) | ~3.0 h |
| …and workers raised 8 → 24 | **~1.0 h** |
| decimated, 8 workers (the proposal) | 0.9 h |

**The two changes that do not touch the consumer get to the same place as the
one that does.** Take those and leave the consumer alone.

**Do take the duplicate `smoothdata` removal** — line 186 recomputing line 146
is not a change to the consumer, it is the removal of a recomputation, and it is
~47% of `slow_wave`'s single-threaded cost. Two conditions: **verify
bit-identity empirically** on at least three real segments across conditions
before editing anything (an argument that two calls must be identical is not a
measurement that they are — this document has a long record of that distinction
mattering), and **Andrea decides**, because `processing_new` is her pipeline.
**Andrea approved the removal, 2026-09-26: "make it once only."** The
bit-identity verification is still required before the edit lands — her approval
is of the intent, and the three-segment check is what establishes the edit
matches it. Record the check's result beside the change.

**Decimation itself is a good idea in the wrong place.** Low-passing at 24414 Hz
for a 0.05 Hz rhythm is genuinely wasteful and would pay back on every future
run of her pipeline, not just on T. **File it as a task 08 row** to be validated
on its merits: decimated versus full-rate slow-wave output on several
recordings, agreement within a stated tolerance, then adopt. It does not belong
inside T, where it would silently redefine the thing being measured.

#### The censoring result is a design finding, not a bracket that needs widening

This is the most important thing in the report and the recommendation
under-weights it. Across *every* kind — including the deterministic `step`,
`clip` and `drift` — **71–75% of seed-rows are censored**, curves are monotone,
nulls are zero, a replicate seed agrees with the locate seed only **58%** of the
time, and **62 of 155 rows are censored on both sides**, meaning the placement
spread exceeds the 2× the bracket spans.

**Where the artifact lands dominates the tolerance by more than the quantity
being estimated.** That is not a bracketing problem. A ±2-step extension chases
the tails of a distribution whose width is the actual result, and on these
numbers it will still leave many rows censored — 1510 points spent to learn
that.

**Before the extension, run a seed-depth diagnostic.** Take three or four
(consumer, kind) cells — include one deterministic kind and `tribo` — and run
**20 seeds** instead of 5 across the *existing* amplitude grid. A few dozen
points, and it answers the question the extension cannot: what is the shape and
width of the placement distribution? Then spend on the axis that carries the
variance, which is invariant 34 applied to the thing invariant 34 was learned
on. If placement dominates as the evidence suggests, seeds buy more than steps,
and 5 seeds was never enough to characterise a spread this wide.

**And the tolerance itself has to change shape.** If sensitivity varies by more
than 2× with placement, a scalar tolerance is not a property of the consumer —
it is a property of one arbitrary placement. `consumer_tolerances.json` should
carry a **conservative quantile across placements** (the most-sensitive end,
stated as such) **plus the observed spread**, so the 09 gate errs toward
rejecting blanking that would fail at an unlucky placement rather than passing
on a lucky one. The spread is a result in its own right and Andrea should see
it: it says her consumers' sensitivity depends on where an artifact falls about
as much as on how large it is.

#### Invariant 35's example was wrong — the mechanism is the smoothing window

Correction accepted, and it makes the lesson stronger. The cost is not in
`filtfilt` (0.39 s). It is `smoothdata` with a 5 s Gaussian window = 122,070
samples, run twice, whose cost grows as **span × window**, and both scale with
fs — so the waste goes as **fs², not fs**. Decimating 100× is ~10⁴ less
arithmetic, not 10². The invariant's text is updated; the example named the
wrong mechanism and the corrected one is a better argument for the same rule.

#### Re-key applied 2026-09-26, and what the block names turned out to be

**Applied and idempotent.** 921 enumerated = 870 written + 27 deduplicated + 24
skipped, residual 0; 835 old stem-keyed directories retired after verification,
2 kept for held recordings; second `--apply` wrote 0. Every key derivation —
`recording.py`, `scan.py` ×2, the generator — now routes through one
`session_key()`, and the key format lives beside `SESSION_STAMP` in
`tdt_block.py` with a test tying them together (invariant 33). Two latent bugs
died with the old key: `_notched` files never found their `_sig` metadata, and
`scan.py`'s `core_of` and the loader's `path.stem` produced different keys for
old-cohort files.

**37 block folders were renamed after acquisition**, so the folder name and the
block name the instrument wrote disagree (`App_ms10_1_bl_001835` on disk,
`t01_ms1_bl_001835` inside the block). The renames are human annotations —
quality flags, condition changes — and **two of them name a different animal
from the instrument.** The key now uses the instrument's block name, which is
stable however often a folder is renamed; renames are recorded in the plan, not
refused; the two animal mismatches are **held** until Andrea says which is right.

**Refinement to invariant 30, forced by this.** A block name is *typed by a
person* into the instrument before recording; it is not measured. So the
instrument is authoritative for what it **measures** — start time, sample rate,
the channel labels wired into the rig — and a name typed into it is merely the
*earliest* human annotation, which a later rename may be correcting. For
**identity** (the key) the instrument name is right because it never moves. For
**meaning** (condition, animal, quality) the two annotations are candidates and
a disagreement goes to Andrea, because a rename made to mark a condition change
is exactly the kind of correction the original typed name cannot know about.

- **Preserve every rename's text as metadata** (`folder_name` beside
  `block_name`). A quality flag Andrea typed into a folder name is a label, and
  discarding it because the key does not need it destroys it.
- **Condition authority is an open question for Andrea**: where the folder and
  the block disagree about condition, which one is the correction? Until
  answered, recordings whose two names disagree about *condition* are handled
  like the animal mismatches — held from anything that groups by condition —
  rather than silently taking either.

#### Andrea's rulings on the renames, 2026-09-26

1. **The two `gems_b_t01_2_1_*` blocks are animal A.** The folder rename was the
   correction. Release them from hold as animal A.
2. **Condition renames: the folder is right** (`ms2`→`ms3`, `1_2`→`1_1`).
3. **Exclude every `_BAD` and `_INCOMPLETE` recording, and its paired baseline or
   stim/recovery file if one exists.** Out of labelling, training, T hosts and
   any corpus. Excluded, not deleted: the flag and the reason are recorded in
   the store, and nothing is removed from the Drive.

**So the general rule is settled:** the instrument's block name is the
**identity** (the key, which never moves); the **folder name is authoritative
for meaning** — animal, condition, quality — because Andrea renames folders to
correct them. Apply the folder's meaning automatically, and keep logging every
block/folder disagreement in the plan so a future rename is visible rather than
silently absorbed.

**Pairing needs a rule, stated and tested, not improvised.** A pair is the
baseline and the stim/recovery recording from the same session: same animal,
same trial token, same condition token, differing only in the `bl` / `sr` infix
(e.g. `gems_j_t01_ms3_bl_230315` and `gems_j_t01_ms3_sr_231323`). Derive it from
the **corrected** folder names (ruling 2). Report, do not guess, every excluded
recording whose partner is **ambiguous** (more than one candidate) or
**missing** (none). An exclusion list is only as good as the pairing that
produced it.

**Pairing resolved by acquisition time — Andrea, 2026-09-26.** Baseline is
recorded **~10 min before** stim/recovery. So acquisition-time adjacency is the
pairing rule, and it is a **verification for every pair, not only a tiebreaker**:

- Declare it as a cohort constant in `protocol.yaml` (invariant 24):
  `pair_gap_min: 10`, baseline first, tolerance **±3 min**. The 11 ambiguous
  sessions all resolve at 10.1–10.2 min, well inside it.
- **A name-matched pair outside the window is not a pair.** Andrea: a 105.3 min
  gap is "likely wrong". Treat it as unpaired and list it for her. The 14.7 min
  pair also falls outside ±3 and is listed the same way, not accepted.
- The lesson for pairing: two recordings sharing name tokens are a *candidate*
  pair; the acquisition times are what confirm it. The same shape as invariant
  30 — the name is a human annotation, the timestamp is measured.

**Pairing, second round — Andrea, 2026-09-26.**

- **The 14.7 min pair is real** (`gems_a_t01_ms2_bl_162459_incomplete` ↔
  `…sr_163939`). The ±3 min tolerance was my guess and it was too tight. Replace
  the symmetric tolerance with explicit bounds: **baseline precedes stim/recovery
  by 5–20 min** (`pair_gap_min_lo: 5`, `pair_gap_min_hi: 20`). The 105.3 min
  pair stays unpaired. The ambiguity check is what makes a wider window safe:
  two candidates inside it still refuse. Report every pair the widened window
  adds beyond the 14.7 min one.
- **`stim_recovery` and `sr` are the same condition** — older files use the long
  name, newer ones the short. Each holds **2 min stim + 20 min recovery in one
  file**. So the scan's condition inference and the pairing's name-token match
  must treat the two tokens as equivalent. **Re-check the 3 "no name match"
  baselines with that equivalence** — a partner named `stim_recovery` would
  have been invisible to a matcher looking for `sr`, and that is a likelier
  cause of "missing" than a recording that was never made.

**Check whether any T host is among the excluded.** If one is, its tolerance
rows were measured on a recording Andrea considers bad, and they must be
re-measured on an eligible host or marked.

#### The `smoothdata` duplicate: verified, now apply it

Bit-identical on all three segments (baseline 599 s with 3 blanked spans, stim
120 s, recovery 1205 s with 155 blanked spans and 4.2% NaN) at 1 and 24 threads:
all 17 output fields and all peak locations. Single-threaded saving 23–54% per
call. **Both of Andrea's conditions are met — her approval and the check — so
apply the patch to her working copy.** Leaving a verified, approved patch
unapplied is not caution, it is a second decision nobody asked for. Keep the
patch file and put the check's result in a comment at the edit.

**Found alongside: `run_continuous.m` calls `slowWaveAnalysis_new` with 12
arguments; the function takes 11 with no `varargin`.** That call errors on
reach. **Andrea, 2026-09-26: the function is newer — update the caller.** One
check first: read the 12th (mode) argument at all three call sites (lines 127,
141, 153). If all three pass the same mode and it matches what the function now
does unconditionally, drop it. **If they pass different modes**, the function
lost behaviour the caller relied on, and dropping the argument collapses three
behaviours into one silently — report that to Andrea instead of editing.
**Result:** they did differ — lines 141 and 153 pass `'single'`; line 127 passes
`'stim_rec'` with a struct holding both epochs, relying on a mode removed before
the repo's first commit. **Andrea, 2026-09-26:** `stim_rec` is just the old name
for an `sr` file, which holds 2 min stim then 20 min recovery. Nothing was
shared across the epochs. So line 127 becomes **two `'single'` calls, one per
epoch**, from the struct it already builds; drop the mode argument at 141 and
153; and correct the function header (lines 24–26), which still documents the
removed mode.

#### Workers: memory binds at ~10, and MATLAB had been capping at 8 all along

Peak RSS is **4.2–4.7 GB per worker** on 54.7 GB available, so memory — not
cores — caps this machine at ~10. My ~24-worker projection ignored memory and
was wrong; invariant 37's text (set it from RSS) was right and my table
contradicted it. At 10 workers the extension projects to ~2.4 h, not ~1.0 h.

And the reason every T run so far used exactly 8: **MATLAB's `Processes`
profile caps `NumWorkers` at 8 by default.** Raising it on an in-session cluster
object rather than editing the persistent profile was right — that profile is
Andrea's and governs all her other MATLAB work.

**If throughput ever matters more, the lever is memory per worker, not cores.**
A 420 s segment of 9 channels in double is ~740 MB, so 4.7 GB is ~6 copies
resident. Loading only the channels a call needs (slow_wave reads the stomach
contacts, 3 of 9), or single precision where the consumer tolerates it, would
roughly double or triple the worker count. Not now — noted for when a pass is
long enough to justify it.

#### Seed-depth diagnostic: two conditions on reading it

Bisecting each seed's crossing on the existing grid (~5 evaluations plus 2
monotonicity checks, reusing seeds 1–5) is a better design than the 20-seed full
grid I asked for, at a fifth of the cost.

1. **Bisection is only valid on a monotone curve**, and `breathing` is
   non-monotone in 12% of cases. On a non-monotone seed, bisection finds *a*
   crossing, not the lowest one — and the tolerance is the **sensitive-end**
   quantile, so a missed lower crossing is **anti-conservative**. Any seed that
   fails its monotonicity check is reported as non-monotone and its bisection
   result is **not** used as a crossing; treat it as unresolved, or grid it.
2. **All four cells are on one host (`host1_JEL`).** That is fine for learning
   the shape of the placement distribution; it is not the distribution for the
   cohort. State it in the result, and do not size the full extension from one
   recording's spread without saying that is what was done.

The redesigned `final()` — each seed an interval, conservative sensitive-end
quantile plus spread — is the right shape. Only 7 rows resolve a spread wider
than one √2 step, which is what a 3-point bracket can show; the width is hidden
in the 110 rows reporting a bound at the sensitive end, and the diagnostic is
what will measure it.

#### Seed-depth results: placement moves the threshold 4–16×

434 points in 47 min (costed ~500 / ~40 — the cost model now holds). No seed
failed its monotonicity confirmation, so every bisection result stands.

| cell | kind | resolved | spread across placements |
|---|---|---|---|
| step 0.5 s, slow_wave | deterministic | 20 / 20 | 16× (0.35–5.7σ) |
| step 0.05 s, T_hardware | deterministic | 14 / 20; 6 below the 0.5σ floor | ≥4× |
| clip 0.5 s, mmc_burst | deterministic | 18 / 20 | 8× |
| tribo 0.5 s, hrv | stochastic | 20 / 20 | 8× |

Deterministic kinds spread as widely as tribo, so the variance is in **where**
the artifact lands, not in its randomness. The order-statistics point is right
and must travel with every number: the minimum of 5 placements estimates roughly
the **17th** percentile and reaches the true 10th only 41% of the time
(1 − 0.9⁵); 22 placements are needed for 90%.

#### Test the phase hypothesis BEFORE buying more placements — it is free

The proposed explanation — that the threshold depends on where the artifact
falls relative to physiological events (R-peaks for `hrv`, slow-wave peaks for
`slow_wave`, bursts for `mmc_burst`, spikes for `T_hardware`) — is the most
important open question in T, and the data to test it already exists. Test it
before choosing between (a), (b) and (c), because it changes what (b) and (c)
should be:

- **If phase explains the spread**, the tolerance is a *function* of phase, the
  sensitive end is the worst phase, and it is measurable directly with a few
  placements aimed at that phase. 20 *random* placements per row would be
  buying precision on the wrong axis — invariant 34, for the third time.
- **If it does not**, the spread is irreducible placement variance and (c) is
  the honest spend.

**Check the bunching against the grid first.** Bisection on a discrete grid can
only return grid values, so "8 of 14 at exactly 2.0σ" means 8 of 14 thresholds
fell in the grid interval ending at 2.0σ. Whether that is a real concentration
depends on the interval widths; compare against what a smooth distribution
would put in each bin before reading it as structure.

The test: for each placement, the artifact's position relative to the nearest
relevant physiological event (as a phase, or as a distance in the event's own
time units), against its threshold. Report per cell, with the correlation and a
permutation null.

#### Decision on spend

- **(a) now** — extend only the seeds censored on the sensitive side (550 points,
  ~39 min on 10 workers), after extending the low amplitude grid for nerve rows
  so the 6 `T_hardware` placements below 0.5σ can resolve. Every resulting
  number labelled **"≈17th percentile, 5 placements"**.
- **The phase test in parallel** — no new evaluations.
- **(b) is declined.** 15–25 h to buy precision on every row, most of which the
  gate never reads, and possibly on the wrong axis.
- **(c) waits for both the phase test and task 14.** Task 14 (routing) decides
  which consumer rows the 09 gate reads; the phase test decides whether those
  rows need random placements or phase-aimed ones.

**One cheap efficiency:** each round spent ~200 s reloading all three host spans
from the Drive, including the 420 s span these cells never used. Load only the
spans a round's points need.

**`slow_wave` contention rose 1.10 → 1.26 from 8 to 10 workers.** First evidence
that adding processes costs something. Record the per-worker-count factor in the
cost model rather than a single number, and re-measure if the count changes
again.

#### CORRECTION 2026-09-26: the phase effects are large, the sample is small

With the numbers in hand my "read as null" was wrong in its wording. Per cell:
`slow_wave` r = 0.59 (p = 0.027), `mmc_burst` |ρ| = 0.52 (p = 0.021),
`T_hardware` |ρ| = 0.40 (p = 0.079), `hrv` r = 0.24 (p = 0.61). Correlations of
0.4–0.6 are **large** effects; they miss Holm correction because n = 20 on one
host, not because they are small. Fisher's combination of the four gives
p ≈ 0.007 — valid only if the cells' placements were drawn independently, which
must be checked before quoting it (shared seeds would correlate them).

**So phase plausibly matters for three of four consumers, and (c) must be
designed to settle it** rather than assume either answer: **stratify (c)'s
random placements across phase bins** of each consumer's event. That yields an
unbiased quantile (weight the bins by their true occupancy) *and* a powered
phase test from the same points, with the same pre-declared directions. Pure
random placement answers only the first question; pure phase-aimed placement
only the second.

#### Phase test: null after correction, directions all as predicted (superseded wording)

No cell survives Holm correction; the bunching check is closest at p = 0.057
(`T_hardware`). All four pre-declared directions came out as predicted — under a
sign test that alone is p = 1/16 ≈ 0.06. Read it as **no evidence of a large
phase effect on one host with 20 placements**, not as evidence of no effect.

Consequences:

- **Treat the spread as placement variance for now.** The tolerance stays a
  conservative quantile plus spread; do not model it as a function of phase.
- **(c) is now the honest spend** once task 14 says which rows the gate reads.
- **Re-run the phase test on (c)'s data at no extra cost.** (c) produces ~20
  placements per gate row across hosts — several times this test's sample — so
  a real modest effect would show there if it exists. Pre-declare the same
  directions.

#### The (a) extension landed — four rulings

589 points, none failed, 0.92× the model. Of 275 sensitive-side censored seeds,
141 are now measured, 118 are still extendable, and 19 sit at the grid's absolute
limit (real "at least this sensitive" results).

1. **`breathing` is non-monotone in 19% of seed curves.** A larger artifact
   sometimes *un*-breaks it, which is physically backwards and points at the
   criterion rather than the consumer. **Check what `breathing_changed`
   compares.** If it compares a count or a rate, an artifact that adds one
   spurious breath and deletes one real breath nets to "unchanged", and the
   criterion flips as amplitude rises. That is the `match_s = 0` shape from
   `mmc` again. If so, compare **matched fiducials** within a stated time
   tolerance, as `mmc` now does, and re-run the breathing rows. A criterion that
   can miss damage is anti-conservative, which is the wrong direction for a
   tolerance.
2. **For any non-monotone seed, the sensitive end is the lowest amplitude
   observed to break it**, not a bisection crossing. State it so.
3. **The 109 locate-only rows get `gate_eligible: false` as a field**, not a
   sentence. One centred placement on a quantity that moves 4–16× with placement
   is not a tolerance. A rule the gate must remember is a rule the gate will
   eventually forget (invariant 27's reasoning); a field it reads cannot be
   forgotten. Rows become eligible only when replicated.
4. **The 118 extendable seeds wait for task 14**, like (c): refining bounds on
   rows the gate may never read is spend without a reader.

**`parfor` chunking cost 46% (4,254 s against an ideal 2,910 s).** Worth fixing
before (c). `parforOptions(pool, 'RangePartitionMethod', 'fixed',
'SubrangeSize', 1)` makes each iteration its own chunk, so the longest-first
ordering survives; `parfeval` with a queue is the alternative. Measure it on a
small manifest before relying on it.

#### Where (c) runs: new-cohort hosts, if the consumers can run there

Every tolerance so far is measured on **old-cohort** hosts, and this section
already records that old-cohort tolerances are **provisional** until re-derived
on new-cohort data — different noise statistics, hardware vs software tripole,
fed vs fasted. The loader path that caveat was waiting for now exists. So (c)'s
placements should be spent on **new-cohort hosts**: 20 placements on an old-cohort
host buy precision on a number already marked provisional. Before (c), establish
whether the five consumers can run on a new-cohort recording (the `_new`
versions exist; `extract_mmc` already ran on `gems_j`), and what the adapter
from the 9-channel store to their inputs needs. Pick hosts from the eligible,
non-excluded set by measured blanked fraction and longest clean run, as before.

#### Rulings from the third pairing / breathing / run_continuous pass (Andrea, 2026-09-26)

**1. Breathing: timing matters, so adopt the displacement criterion.** Andrea
uses breath timing downstream, not only rate, so a breath displaced by one R-R
interval (~140 ms) is damage. `breathing_changed` counts a breath as changed if
it is added, lost, **or displaced by more than 0.07 s**. The threshold is not
arbitrary: breaths are sampled at heartbeat locations, so detected breath times
are quantised to R-R (~140–165 ms), and half an R-R interval separates "same
beat" from "moved a beat". It tightens 11 of 48 breathing rows. The residual
non-monotonicity (22 → 25 rows) is intrinsic — a winner-take-all detector
sampled at R-peaks is non-monotone in amplitude by construction — and is
handled by the lowest-failing-amplitude rule, not by the criterion. It is a
change to a pre-registered criterion, made on Andrea's statement of what the
output is used for; record it as such, with the date.

**2. Completeness is also a duration rule.** Andrea: a baseline shorter than
~10 min, or a stim/recovery file shorter than ~2 + 20 min, is **incomplete**.
Declare it in `protocol.yaml` (`min_baseline_min: 10`, `min_sr_min: 22`,
cohort constants). Apply it to every recording from its measured duration.

- It **adds** to the folder flags, never removes one: an `_INCOMPLETE` folder
  that happens to meet the duration stays excluded.
- A short recording is excluded **with its partner**, as before.
- "Around" means borderline cases exist. List every recording within 10% of
  its threshold (9–10 min baselines, ~20–22 min sr files) for Andrea instead of
  deciding them silently.
- **Dry run first.** If the rule would exclude more than ~10% of the corpus,
  stop and report before applying; a rule that removes a large share of the
  data deserves her look at the list first.
- This settles the two `gems_d` restarts: judge the later `t01` sessions by
  duration like everything else.

**3. CORRECTED 2026-09-26 — the "some are wrong" claim is not borne out on the
Drive.** It was my inference from Andrea saying she had used results; she has
since said she is **not sure which script** produced them. Measured:
`run_continuous.m` never completed an iteration on either drive (no
`_HR_BR_HRV_results.mat` or `_slowWave_results.mat` anywhere); the shifted HR
call **errors** when blanked segments exist (nearly always) and runs silently
only when none do; none of the 386 `*_HRBR.mat` files carries the shifted
fingerprint. So the suspect set on the Drive is **empty so far**. Remaining:
attribute each of the 386 files to the script and function version that wrote
it (saved fields, parameter set, file format), list the attribution for Andrea,
and flag any file whose writer cannot be determined. Results stored off the
Drive cannot be checked until Andrea connects that folder. The original
requirement, kept for reference:

**3 (original). `run_continuous.m` HR/BR/HRV results have been used, and some are wrong.**
The calls to `HR_BR_HRVAnalysis_new` passed arguments shifted one position,
without an error, after the function dropped its mode argument. Required:

- Establish what the shift did: which parameter received the mode string, which
  received what, and whether that produced errors, defaults or plausible wrong
  numbers.
- Find when the signature changed (git history of `processing_new`, plus the
  uncommitted working copy). **Outputs produced by `run_continuous.m` after that
  date are suspect; outputs from before it are not.**
- Find every such output file (saved variable names, file names, timestamps),
  list them for Andrea, and list what consumes them downstream
  (`bulk_mixed_models.m` and the like). **Do not overwrite or regenerate
  anything until she has seen the list** — regeneration replaces files she has
  already built analyses on.
- The slow-wave calls would have *errored* (12 arguments into 11), so no
  `run_continuous.m` slow-wave output exists from that period; say so if the
  search confirms it.

**4. `main_mod.m`: fix it (Andrea, 2026-09-26), not retire it.** Apply the same
fix as `run_continuous.m` to its shifted HR/BR calls (lines 232, 242, 252) and
its 12-argument slow-wave calls, and run `checkcode`. For the older defects:
fix the clear-cut ones — crashes, wrong variable names, dead branches — and
**list any fix that would change analysis results** for Andrea before applying
it. It stays uncommitted like the rest of `processing_new`.

**5. Multiple baselines in a session: the ~10 min one is the baseline** (Andrea).
The duration rule already excludes the short aborted starts; pair the ~10 min
baseline with its stim/recovery by time as usual. Do not rename trial tokens
(the `gems_d` `t03` → `t01` restarts keep their folder names).

**Parfor correction.** The extension's 46% overhead was **not** chunking:
`'auto'` and `'fixed1'` are both within 1–3% of ideal at 60 and 240 points and
on the extension's heavy head. The likeliest cause is contention from
concurrent test and generator runs on the same machine. Each point now records
its worker and finish time, so a recurrence will show its cause. Keep `'auto'`
unless a measurement says otherwise; **do not run heavy local work alongside a
timed sweep.**

**New-cohort host selection needs ≥180 s clean runs.** `gems_j_t01_ms3_bl_230315`
runs all five consumers deterministically, but its longest clean run is 136 s.
`T_hardware` found 127 spikes on the right cuff and only 16 on the left, which
fits the polarity and left-cuff-weight concern below; settle that before any
new-cohort tolerance is believed.

#### The new-cohort adapter must reproduce the old input convention

The consumers were validated on the old cohort's `_blankmotion.mat` inputs, so
the adapter's job is to hand them what they saw then, not a better signal:

- **No common-average reference on the stomach channels.** A CAR over `ANT1–3`
  removes what the three contacts share, and the slow wave is largely shared, so
  it attenuates exactly what `slow_wave` measures. Pass `ANT1–3` referenced as
  the old cohort's were; check that reference in the old files rather than
  assuming it.
- **Match the tripole's polarity to the old hardware tripole**, verified from the
  sign of real spike waveforms, not from the formula. Spike detection is often
  single-polarity; an inverted tripole could halve detected spikes and look like
  a physiological difference.
- **Report the left cuff's fitted tripole weights.** If they are far from 0.5 /
  0.5, the software tripole is not what the hardware would have produced, and
  that belongs in the `T_hardware` row's provisional caveat.

#### UPDATE 2026-09-26: the blind audit is launchable

**Tools → Blind recall audit…** opens a separate window. Recordings offered: only
those with a store entry, a reachable source file and no exclusion (844 now);
an excluded one is refused even if requested directly. Each open draws one
120 s span from a recorded seed, after removing stimulation and 20 s edge
guards. Shift+drag marks; out-of-span marks are refused. **Commit writes marks
and a plan record to `labels/<animal>/blind_audit/<session>/` before any
candidate or z-trace is computed.** Six mutations caught; exercised end to end
on one `sr` and one `bl` recording (open 14–34 s, reveal 20–40 s). The bridge now
imports `gems_blanking_v2` as a declared dependency (invariant 21), and
`meta.json` records `source_path` relative to the store root.

**Before the formal audit counts:** the five-span plan (5 contiguous 2-min spans,
~10 min total, stratified across 2–3 new animals and conditions, seed recorded)
exists in the planner but is not wired to the window. **Wire it**, so each open
advances through the plan and the window shows progress (span k of 5). Spans
Andrea marks before that are a UI trial, not audit data, unless they happen to
be drawn by the plan — keep them, marked as trial.

#### Stim epoch location depends on modality — task 03B has the same defect

On `sr` files the channel that carries stimulation depends on modality: `ms`
shows it on **ADC2 only**, `es` on **`vib` / `adc1` only**, combined conditions on
all three. A split reading `vib` alone found a spurious "stim epoch" at
1,163–1,283 s on an `ms` file whose `vib` is flat throughout. The audit now
excludes the protocol window at the start of every `sr` file (**0–132 s**),
which matched all 10 sampled files and agrees with Andrea's description (2 min
stim, then 20 min recovery).

**Task 03B must choose its channel by modality**, parsed from the corrected folder
name, and **cross-check the detected epoch against the protocol window**: stim
should begin within a few seconds of file start and last ~120 s. A detected
epoch outside that is flagged, never silently used — a detector reading the
wrong channel returns a confident, wrong answer rather than no answer, which is
exactly what happened here.

#### Rulings after the five-span wiring (Andrea, 2026-09-26)

- **Re-admit the four `sr` files** (A/t01/ms2, B/t01/es3, I/t01/3_3, I/t01/ms1).
  Each has an aborted short baseline *and* a full ~10 min baseline; the full one
  is the pair, so the `sr` has a good partner. The aborted starts stay excluded.
  This **supersedes** the earlier "14.7 min pair is real" answer, which was given
  without knowing a full baseline sat 10.1 min before that `sr` — my question
  was missing information, not her ruling inconsistent. Apply: excluded 61 → 57.
- **`cme<n>` (animal K) is combined mechanical + electrical.** Read all three
  monitor channels, as for the other combined conditions.
- **`main_mod.m`, the three held fixes — all approved:**
  1. slow wave in the single-recording branch uses the **stomach channels only**
     (3:5), matching everywhere else;
  2. **include `sr` folders** in the folder filter and the condition match, so
     stim/recovery files are preprocessed and artifact-cleaned;
  3. the hard-coded loop ranges become a **parameter defaulting to all files**,
     and **existing outputs are never overwritten unless explicitly asked**
     (skip-if-exists by default, an `overwrite` flag to force).
- **03B edge detection.** Four combined-modality files are wrongly flagged because
  a small noise-floor step reads as continued stimulation. Refused loudly, so
  nothing is corrupted, but fix the edge rule: define the stim-off edge against
  the post-stim noise floor measured on the recovery segment, not the pre-stim
  one, and require the ON level to be a clear multiple of it. Re-run the 259
  and confirm the four pass without any of the 255 changing.
- **`vib` is numerically identical to `adc1`.** Treat it as one channel, not two
  independent confirmations: agreement between identical copies proves nothing.
  For `es`, the check is therefore single-channel — say so in the split report.

#### After the re-admission pass (2026-09-26)

- **`gems_i_t03_es2_sr_223947`: all three monitor channels exactly zero.**
  Andrea is not sure stimulation happened. **Exclude it and its baseline
  partner**, reason `no_stim_monitor`. Refusing a constant monitor in 03B was
  right: the old and new code both reported a fabricated 0–120 s "pass" from an
  argmax over a flat score (invariant 41). This changes the audit pool (527 →
  expected 525), so apply it **before** any audit plan is created. If a plan was
  already created, check whether it draws either recording; if it does, discard
  that plan (it has no committed spans yet) and have Andrea create a new one.
- **03B onset rule extended to the post-/pre-stim floor as well — ratified.** It
  goes beyond the ruling's letter for the right reason: the same floor step
  preceded stim on two files.
- **`main_mod.m` stage 1 on new-cohort folders must refuse** until it takes its
  stream names and channel indices from the file's `chanlabels` (the single
  source, per the channel-order rule) rather than the old cohort's
  `[1:2,17:19]`. Fix (b) admitted `sr` folders, and a wrongly indexed
  `_notched.mat` would be silent. Refuse loudly now; build the per-cohort map
  when Andrea next needs stage 1 on new data.
- **Adapter checks run when Andrea is not labelling.** They read full recordings
  over the Drive. She will say when she stops for the day.

#### The blind audit is not launchable — this is now the critical path (superseded, see update above)

Session, controller, dock and planner exist but are not wired into the app, there
is no way to start audit mode, marks go wherever the caller says rather than into
the store, and the bridge still finds `gems_blanking_v2` by `sys.path` insertion
(invariant 21). Andrea cannot start labelling, and labelling is the human
critical path to the 09 gate. **After the current runs finish, this comes before
more T work.** Done means Andrea can open the app, pick an eligible new-cohort
recording, and do a blind span with marks written to the store under the
session key — excluded recordings never offered.

#### Confirm `rostral_end` landed

The report does not mention it. It was to ride along with the re-key: `rostral_end:
1` in `protocol.yaml`, deleted from `meta.json`, `direction_valid` false for the
old cohort regardless. If the pull predated that commit, it is one more
idempotent run.

#### The median σ: do not run a sweep for it

The offer was a stratified sample of ~30 recordings to characterise the cohort's
robust σ. **Declined.** It costs hours of Drive bandwidth, which is the scarce
resource this week and is already contended by T and by labelling, and it buys a
tighter plausibility band — a band whose job is to catch errors of 10³ and 10⁶,
which the current 1–500 µV already does with margin on both sides.

**Record it as a by-product instead.** `assert_plausible_units` already computes
the median robust σ on every load. Log that value, with the recording id, to
provenance. The distribution then accumulates for free as labelling and
processing proceed, and by the time there is enough of it to justify tightening
the band, it will have been collected by passes that had to happen anyway. A
sweep run to obtain a number an existing pass already computes is a second pass
over the same bytes. (Invariant 26.)

#### The corpus is growing under the build

852 at the first scan, 859 now. Andrea is still collecting, and she will be for
weeks, so every count in every report ages the moment it is printed. Two
consequences, both required:

- **Every count is printed with the enumeration timestamp that produced it.** A
  bare "859 recordings" in a report is not reproducible and cannot be reconciled
  against a later one.
- **The generator is idempotent and incremental.** Re-running it writes only
  what is missing, leaves existing files untouched, and reports new / existing /
  collided separately. A one-shot migration script is wrong for a corpus that is
  still growing; this will be run many times.

#### Cheap checks, both clean

- **Compression is gzip-4 throughout.** Task 10 is unblocked — no re-write pass.
- **`Documents/GEMSBlanking` is clean at `599d1fe`.** No uncommitted local
  divergence behind the import.

### BUILD CHANGE 2 FIRST — it is the only part on the critical path

Change 2 (blind recall audit) unblocks step L, which unblocks the 09 gate.
Change 1 (candidate adjudication) is not needed until **task 10**, which is
*post*-gate. So ship Change 2 on its own, let the labelling start, and build
Change 1 after. Doing them together delays the labelling for work nothing is
waiting on.

### The z-trace panel: 6 traces, not 54

Nine signals × six bands is 54 traces and no human reads that. The diagnostic
question is narrow — *was z high anywhere at this moment?* — so show **one trace
per band, the maximum across all signals**, with the `z_enter` threshold drawn on
each. Six traces answer blind-spot-versus-threshold directly. Which signal
carried the maximum is a click-to-expand detail, not the default view.

The maximum is taken across **all** signals including raw contacts, per hard
invariant 6 — detection reads the contacts, so the audit must show what detection
saw.

### The blind-mode sampling rule

**Contiguous spans, not scattered minutes.** Recall is
`(artifacts the human found that no candidate covers) / (all artifacts the human
found)`, which is only computable over a *complete* span. Scattered ten-second
snippets destroy the context a human needs and make the denominator meaningless.

```
~10 minutes total, as 5 contiguous 2-minute spans
uniform-random start within each recording
stratified across 2-3 new animals and across conditions
seed recorded in provenance
excludes stim epochs (03B) and unassessable edges (task 06 settling)
```

**Explicitly NOT candidate-dense sampling.** Choosing where to look using the
thing under test reintroduces the anchoring the blind mode exists to prevent — at
the level of span selection rather than of marking. The objection that raised
this had it exactly right.

### Change 2 — NEW recall-audit mode

**Blind first, reveal second.** The human marks artifacts on the raw signal with
candidates **hidden**, commits, and only then are candidates and z-traces revealed.
If candidates are visible while marking, the labeller anchors on them and measured
recall is inflated — the gate would be measuring its own output. This is not a
preference; without it the gate is circular and worthless.

After the reveal: a scroll view over the sampled minutes with candidates overlaid
**and the band z-traces shown alongside the raw signal**.

The z-traces are the entire point. When the human finds a missed artifact, they
distinguish:
- **generator blind spot** — z genuinely low → needs a new band or feature
- **threshold too high** — z high but sub-threshold → needs a lower `z_enter`

Different diagnoses, different fixes, and **indistinguishable from the raw trace
alone**. Without this the audit tells you that recall is bad but not why.

### Change 3 — preprocessing defaults
Set the candidate-harmonic list in the Preprocessing tab to `60` only. Do not enable
harmonic detection: a 120 Hz notch rings *inside* the ENG band at 2.41 µV/mV
(threshold at a 7.5 mV excursion) where the 60 Hz notch contributes 0.221 µV/mV.

### Labelling budget — design to this
~500–1000 judged events across ~12 recordings spanning all animals, one keystroke
each: **1–3 hours once**. Then ~50 uncertain events per new animal (~15 min). Zero
per production dataset. If the UI requires materially more than this, it is wrong.

### Acceptance
Audit mode usable on a real recording; a missed artifact can be classified as blind
spot vs threshold in one glance.

---

**Depends on tasks:** 07

Read `CLAUDE.md` before starting; its invariants apply here and are not repeated.

---

# Reference — shared contracts

## Part A — shared contracts

Injected into every generated task file.

### A.1 Core data structures

```python
@dataclass(frozen=True)
class ChannelInfo:
    index:         int                 # column in the raw matrix
    name:          str                 # "RVN", "ANT1", ...
    role:          Literal["nerve", "stomach", "aux"]
    cuff_id:       str | None          # "L" / "R"; None for stomach and aux
    contact_index: int | None          # 1..3 along the cuff; None if not a cuff
    rostral_end:   int | None          # which contact_index is rostral; None = unknown
    config:        Literal["hw_tripole", "independent"]

@dataclass(frozen=True)
class Recording:
    fs:        float                   # 24414.0625 nominal - read it, never hardcode
    data:      np.ndarray              # (n_samples, n_channels), float64, microvolts
    channels:  list[ChannelInfo]
    animal:    str                     # "F", "J", "L", "O", ...
    session:   str
    path:      Path

@dataclass(frozen=True)
class Candidate:
    start_s:    float
    stop_s:     float
    signals:    tuple[str, ...]        # which derived signals crossed
    bands:      tuple[str, ...]        # which bands crossed
    peak_z:     float
    provenance: Literal["electrical", "video_assisted"]

class TrainingMode(StrEnum):
    POOLED     = "pooled"       # all animals except the target; the mandatory mode
    ADAPTED    = "adapted"      # pooled prior + target animal's own labels
    PER_ANIMAL = "per_animal"   # target animal only

@dataclass(frozen=True)
class Event:                           # a Candidate after step 08
    candidate:  Candidate
    judgement:  Literal["motion", "physiology", "unsure", "unjudged"]
    p_motion:   float
    source:     Literal["human", "model", "inherited"]
```

### A.2 The grid

Every envelope, z-trace and mask lives on a shared grid:

```python
GRID_S = 0.010
n_frames = int(np.floor(duration_s / GRID_S))
frame_centre_s(i) = (i + 0.5) * GRID_S
```

A band's analysis window is **centred** on the frame centre and may be much longer
than the grid step. Windows overlap; that is intended.

### A.3 Bands

```python
BANDS: dict[str, BandSpec] = {
    #  name        lo     hi    window_s     2*B*T
    "300-3000": (  300., 3000.,   0.025),   # 135  <- time-resolution choice
    "100-300":  (  100.,  300.,   0.075),   #  30
    "10-150":   (   10.,  150.,   0.100),   #  28   (was 1-100/150 ms: see task 05)
    "2-50":     (    2.,   50.,   0.310),   #  29.8
    "0.5-3":    (   0.5,    3.,   6.000),   #  30
    "0-2":      (   0.0,    2.,   7.500),   #  30
}
REFERENCE_STATISTIC = "median_of_log"   # binding; see hard invariant 5
```

**There is no `ref_pct`.** The percentile reference was measured to be
mis-centred (invariant 5) and is superseded by the median of the log envelope.
The old percentiles (10 / 10 / 10 / 10 / 25 / 25) are recorded in A.5 as history
only — do not put them in code, where a second reference rule would compete with
the binding one.

```
Window lengths come from effective degrees of freedom `2·B·T ≈ 30`. The two slowest
bands use the 25th percentile because a 10-minute file holds only 80–100 independent
frames there, where p10 carries ~13% standard error.

### A.4 Consumers

```python
CONSUMERS = [
    # name              signal              band        tolerance
    ("spikes",          "T",                "300-3000", "4.5 sigma, sample level"),
    # ("slow_c", "T", "100-300", "own sigma"),   STRUCK 2026-09-23: never implemented anywhere; see Step 9
    ("velocity",        ("V1","V3"),        "300-3000", "peak ratio > 1"),
    ("mmc",             "stomach_referenced", "2-50",   "3 x moving MAD"),
    ("slow_wave",       "stomach_referenced", "0-2",    "peak displacement"),
    ("breathing",       "best_hr_channel",  "0.5-3",    "peak inserted or lost"),
    ("hrv",             "best_hr_channel",  "10-150",   "operational: beat train unchanged"),
]
```

No 0–2 or 2–50 Hz mask on nerve signals. No 300–5000 Hz mask on stomach signals.

### A.5 Constants measured already — do not re-derive

| Quantity | Value |
|---|---|
| Cuff v9 contact pitch / aperture | 1.50 mm / 3.00 mm |
| 60 Hz notch ring into ENG band | 0.221 µV per mV excursion |
| Bandpass gain at 60 Hz through `filtfilt` | −36.4 dB |
| QRS energy above 300 Hz | 0.00–0.54% |
| σ reduction, 300–5000 → 300–3000 | **12–14% measured**, not the 24% A.5b implies |
| QRS energy below 3 Hz | 0.00% |
| QRS energy in 100–300 Hz (8–12 ms QRS) | 32.1–65.2% |
| Smooth motion energy below 300 Hz | 98.7–99.7% |
| Saturating step, energy above 300 Hz | 32.2% |
| Velocity artifact tolerance | ~1× raw, ~5× band-limited, ~1.4× broadband |
| Velocity resolution `Δv/v ≈ v/(B·L)` | 4% @ 0.5 m/s, 7% @ 1, 14% @ 2, 35% @ 5 |

---

### A.5b — ENG band upper corner: measured, changed 300–5000 → 300–3000

Tested at Andrea's request. Method: detect events on a wide 300–11000 Hz tripole,
take the **median** power spectrum of ±1.5 ms event windows against random
background windows (median, not mean — a handful of giant artifacts otherwise
dominate by six orders of magnitude), and find where the ratio approaches 1.

On the healthy cuff (R), event energy is concentrated **below ~2 kHz** and reaches
background by ~4 kHz:

| | 0.5k | 1k | 1.5k | 2k | 2.5k | 3k | 4k | 5k | 7k | 9k |
|---|---|---|---|---|---|---|---|---|---|---|
| event/background | 151 | 36 | 9.1 | 2.4 | 1.9 | **1.6** | 1.1 | 1.1 | 1.2 | 1.2 |

Crosses 2× at **2374 Hz** and 1.2× at 3730 Hz. Everything above ~4 kHz in the old
300–5000 band was noise, and it cost real sensitivity:

| corner | σ(T) µV | events/s | median amp/σ | p90 amp/σ | cardiac peak/base |
|---|---|---|---|---|---|
| 300–2000 | 1.83 | 10.5 | 6.09 | 10.88 | 2.93× |
| **300–3000** | **2.26** | **8.2** | **5.84** | **9.77** | **3.56×** |
| 300–5000 | 2.96 | 5.8 | 5.62 | 8.43 | 4.31× |

Narrowing to 3 kHz **lowers σ by 24%, raises the event rate 41%, improves
event-to-threshold separation, and reduces the cardiac peak-to-baseline ratio.**
300–2000 is better still on every count but clips the 2–2.4 kHz shoulder where the
ratio is still above 2×, so **3000 Hz is the defensible choice**; revisit 2000 Hz
if the cross-animal set agrees.

**This breaks comparability with previously processed data.** σ changes, so the
4.5σ threshold changes, so every historical spike count and firing rate changes.
Reprocess rather than mix, and record the band in provenance.

**Confirm on the cross-animal set before pinning** — this is one recording.

### A.5c — Per-cuff health check (new QC metric, free)

The same event/background spectrum is a **cuff diagnostic**, and on this recording
it fails the left cuff:

| | 0.5k | 1k | 2k | 3k | 5k | 9k | peri-R rise |
|---|---|---|---|---|---|---|---|
| cuff R | 151 | 36 | 2.4 | 1.6 | 1.1 | 1.2 | **1.3–2.9%** |
| cuff L | 1.06 | 1.16 | 1.03 | 1.19 | 1.84 | **15.4** | **688–867%** |

Cuff L has no spectral signature of neural events anywhere, its ratio *rises* at
9 kHz (backwards for real spikes), and its peri-R rise is two orders of magnitude
larger than cuff R's. Consistent with LVN3's −2.15 V excursion and LVN3 ranking
last on cardiac template SNR (179 vs 400–600).

**Emit this per cuff per recording**: event/background ratio at 1 kHz, the
frequency where it crosses 2×, and the ENG-band peri-R rise. A cuff that looks
like L above should be flagged before its data reaches any analysis.

---

## Part A.6 — Build order

Revised 2026-09-19 after the first real-data gate run. **The gate is measured on
newly labelled new-cohort data, not on the 43 old recordings.**

Why not the old data: those 4 animals are the hardware-shorted tripole — 5
channels, one derived signal per nerve. The generator reads the **raw contacts**
and uses cross-channel agreement to control the multiple-comparisons problem that
dominated the first run. With no raw contacts, `nmin`, the common-mode features
and the "detect on contacts, threshold against the tripole" logic are all
untestable there. Measuring recall for a degraded generator and generalising to a
different one is not a gate. The old labels were also drawn on notched
hardware-tripole signal with ±100–500 ms boundary precision — the very imprecision
this design exists to remove.

```
 1  00   repo + test harness
 2  00A  Drive store, preflight, registry            ─ everything reads through it
 3  01 ◆ zeros -> NaN                                 ─ one line, do it first
 4  03   loader + channel map
    03A  scan + condition inference                   ─ parallel with 04-07
 5  04   derivations (0.5/0.5 tripole, no fit)
 6  06   band envelopes, log-z, median reference
 7  07   candidate generation
 8  02   peri-R cardiac window                        ─ needs 05; parallel
    05   R-peaks (10-150 Hz, k=6, global plausibility)
 9  T    CONSUMER TOLERANCE DERIVATION                ─ NEW, see below
10  16   labelling UI incl. blind recall-audit mode   ─ needed regardless
11  L    label ~10 min EXHAUSTIVELY, 2-3 new animals  ─ human, ~2-4 h
12  09 ◆ GATE: real recall + precision vs those labels
--------------------------------------------------------------------- gate ------
13  11   features
14  10   convert old labels to events (what survives)
15  12   classifier, three modes      12A registry
16  13   extent      14 routing      15 masks + QC
17  08   MATLAB fixes                 ─ DONE 2026-09-23
18  16A  application shell + UI tree  ─ added: was missing from this list
    16B  training/eval dashboard      ─ added: was missing from this list
19  17   video        18 velocity      03B stim split  ─ 03B DONE
20  19 ◆ end-to-end acceptance
```

*Corrected 2026-09-23: 16A and 16B were never scheduled here even though both
are specified below, and together they are the largest remaining block. 19's
acceptance runs through the shell, so it depends on them.*

Steps 1–12 are the whole pre-gate commitment: a loader, a candidate generator and
a labelling UI. **None of that is wasted if the gate fails** — the U-Net fallback
consumes the same envelopes, the same UI and the same labels.

### Step 9 — consumer tolerance derivation (prerequisite for the gate)

The revised pass condition is "≥98% recall for artifacts **above each consumer's
tolerance**". Those amplitudes are not yet known, and the gate is unanswerable
without them. Derive them without labels:

```
for each consumer, for each artifact kind:
    for amplitude in a log sweep (10 uV .. 50 mV):
        inject into a clean span
        run THAT CONSUMER'S OWN analysis with and without the injection
        record the change in its output
    tolerance = the amplitude at which the output first changes materially
```

"Output changes materially" is consumer-specific and already defined in Part A.4 —
for R-peaks the beat train changes, for spikes the 4.5σ crossing set changes, for
slow wave a peak displaces. Report a tolerance curve per consumer, not a scalar.

**First evidence this matters:** a 200 µV common-mode artifact is only rejected
~2.5× by the tripole (measured), so it still lands at ~27σ on `T`. Intuition about
what is "small" is unreliable here.

#### T is a MATLAB task, in `processing_new`. Decided 2026-09-23.

A tolerance is a property of **the consumer as it actually runs**, and five of
the seven consumers are MATLAB: `spikes` (`step2_noise_sigma` +
`batch_spike_detect`), `mmc` (`extract_mmc`), `slow_wave`
(`slowWaveAnalysis_new`), `breathing` and `hrv` (`HR_BR_HRVAnalysis_new`).
Reimplementing any of them in Python to make T callable from the Python harness
would measure **the reimplementation's** tolerance, and two implementations of a
consumer means two tolerances — one of which would silently be wrong. No bridge,
no subprocess handoff: T is a script in `processing_new` that writes a small
`consumer_tolerances.json`, and the Python side reads that file.

This does not move any other boundary. **Python does detection and blanking;
MATLAB does the science; the handoff is per-consumer masks in `.mat`** — exactly
what detector-pyqt did and what tasks 08 and 15 already assume. Model training
stays in Python.

#### The language split, stated precisely

"T is a MATLAB task" means **the consumer runs are MATLAB**. It does not mean
everything is. The artifact library is Python — `GEMSBlanking:detector/
synthesize.py` with `inject_saturation` / `inject_drift` / `inject_broadband` /
`inject_transplant` — and reimplementing artifact morphology in MATLAB would
repeat the exact error the rule exists to prevent, one level down. So:

```
Python  generate injected signals, write one .mat per (kind, amplitude, duration)
MATLAB  driver runs the five consumers over all of them, writes their outputs
either  diff outputs against the clean baseline, emit consumer_tolerances.json
```

The rule is **never reimplement the thing you are measuring**. Crossing a
language boundary with files is not a violation of it; it is the same handoff
tasks 08 and 15 already use.

#### Use `conftest.inject_artifact`, not `synthesize.py`

`synthesize.py`'s injectors randomise amplitude **and** duration internally from
an `rng`, express amplitude in MAD units, and place the artifact at the end of
the array; `inject_broadband` caps duration at 0.5 s, so the sweep's 5 s point
is unreachable. They were built to generate a training corpus, where variety is
the point. **T needs the opposite: a known amplitude at a known time.**

`tests/conftest.py:inject_artifact(sig, fs, t0_s, dur_s, kind, amp_ratio, seed)`
takes exactly the sweep's axes, implements all four kinds, and CLAUDE.md already
makes `conftest` the owner of every generator. This is consistent with what
task 09 already says — "prefer transplant for the gate; **keep the parametric
kinds for the amplitude/duration sweep, where a known amplitude is the
point**". Do not modify `synthesize.py`.

#### The sweep grid is in σ, with µV reported alongside

Per-channel robust σ on the host is 16–35 µV and **the data are in volts**
(invariant 14 — declared, not inferred). A fixed µV amplitude is therefore a
different multiple of σ on each channel, while every consumer thresholds in σ.
**Grid the sweep in σ** so it is comparable across channels, animals and
cohorts, and report µV as a derived column.

The 50 mV ceiling is ~1400×σ and ~100× the largest sample in the recording
(max |x| = 0.5 mV). Run it, and **mark where the curve leaves the physically
observed range** rather than truncating — a tolerance that only exists above
anything the electrode has ever seen is a finding about the consumer, not a
number to use.

#### Artifact kinds: the four that already exist

`step`, `clip`, `drift`, `tribo` — `conftest.py`'s names, reconciling with
`synthesize.py` as `inject_saturation` spanning `step` + `clip`,
`inject_drift` → `drift`, `inject_broadband` → `tribo`. Task 09 already sweeps
these four; T uses the same four so the two are comparable. `inject_transplant`
is **not** used here: T needs a known amplitude, and a transplanted real chunk
does not have one.

#### T is a surface, not a curve — duration is the second axis

The slow consumers cannot resolve a brief injection: measured impulse responses
are `2-50` 486 ms, `0-2` 1146 ms, `0.5-3` 3988 ms, so a 50 ms artifact reaching
`slow_wave` is smeared across more than a second. Sweep duration
**{50 ms, 500 ms, 5 s}** alongside amplitude, and check that the longest
duration exceeds each consumer's own impulse response — for `0.5-3` at 3988 ms
it barely does, so report that consumer's row as bounded by the sweep rather
than as a resolved tolerance.

#### Three further rules, and the host recording

2. **Inject into a clean span, and prove it was clean before injecting.** State
   the criterion used to select it and record the span in the output. An
   injection on top of existing artifact measures the sum of the two, and a
   tolerance derived that way is silently too high.
3. **"Materially" is written down per consumer BEFORE measuring, not after.**
   A criterion chosen once the curves are in hand is a curve-fitting exercise.
   A.4's column is the starting point; make each row concrete (how many
   crossings, how far a peak displaces) and commit to it in the task's output.
4. **Sweep amplitude logarithmically and report the curve.** A scalar read off
   a coarse grid is a grid artifact — this project has produced three wrong
   constants that way (the σ-reduction factor twice, the cardiac window once).

##### The old-cohort nerve channel IS a tripole — a hardware one

The old cohort's two nerve channels are not single-ended. A.6 records the
cohort as **"the hardware-shorted tripole — 5 channels, one derived signal per
nerve"**: the tripole is formed physically by shorting the outer contacts
before the amplifier, so `RVN` and `LVN` each *are* `T`, with `a = b = 0.5`
forced by the wiring rather than chosen in software. **Common-mode rejection is
therefore exercised**, and a spike tolerance measured here is a tolerance for a
tripole, not for a bare contact.

What does **not** transfer is the same distinction as the stomach reference,
one level up: a hardware tripole sums before a single ADC, the software tripole
of task 04 sums three separately digitised channels, so the software version
carries more converter noise and any inter-channel gain or phase mismatch
leaves residual common mode. Label the row **`T_hardware`** and record the
caveat; treat it as provisional for the new cohort exactly as `mmc` and
`slow_wave` are.

**CONFIRMED 2026-09-23 by Andrea from the wiring record: tripolar cuffs with
the outer contacts shorted together before the amplifier.** A.6 is right. The
old-cohort `RVN` and `LVN` channels each **are** `T`, with `a = b = 0.5` forced
by the wiring. Label the row **`T_hardware`**. Common-mode rejection is
exercised and the spike tolerance measured here is a tripole tolerance.

**The measurement could not have settled this, and one of the arguments used
against A.6 was wrong — mine.** For the record, since the same mistake is easy
to repeat:

| | host 1 | host 2 |
|---|---|---|
| QRS **shape** correlation, nerve ch1↔ch2 | +0.980 | +0.973 |
| same, stomach pair | +0.620 | +0.753 |
| QRS peak on nerve | 4.94σ / 2.87σ | |
| QRS peak on stomach | 0.25–0.48σ | |

I argued the stomach pair was the control and the nerve pair was failing it.
**The stomach is not a valid control**: it sits at a different distance and
orientation to the heart, so it sees a field with more spatial variation. A
tripole's residual is the second spatial derivative of the field along the
cuff, and for a **distant** source that is a scaled copy of the same waveform
at both cuffs — so 0.98 between two genuine tripoles is expected, not anomalous.
Comparing two montages at different distances from the source and treating the
difference as evidence about montage was the error. The correct verdict was the
one reached first: **if the contacts are shorted before the ADC, the file is
byte-identical to a single-ended recording and no analysis of it can
discriminate.** Hardware questions need hardware records.

**Finding worth keeping: a confirmed tripole still passes the QRS at 3–5σ.**
That is consistent with the ~2.5× common-mode rejection in A.5 and it
strengthens the project's premise rather than weakening it — the tripole alone
does not remove cardiac or motion common mode, which is why per-consumer
blanking exists at all. Record it in the T output alongside the tolerances.

What still does **not** transfer to the new cohort: a hardware tripole sums
before a single ADC; task 04's software tripole sums three separately digitised
channels, so it carries more converter noise and any inter-channel gain or
phase mismatch leaves residual common mode. Treat `T_hardware` tolerances as
provisional for the new cohort, exactly as `mmc` and `slow_wave` are.

**Host: an old-cohort `bl` recording, and say why in the output.** The five
consumers run on the 5-channel `_blankmotion.mat` format today, not on the
9-channel new cohort, so the old cohort is where they can be exercised
unmodified. `E1000_JEL_E1000_bl_1315` was proposed on `bad_fraction` 0.046, **and that
number is the wrong quantity** — it is the *model's* bad-window fraction on
that LORO test fold, not the file's manually blanked fraction, which measures
0.0012. Two different things read off the same word. Pick the host on the
measured blanked fraction and on longest clean run, not on the LORO column. Use a second host as a robustness
check. **Caveat to record:** tolerances derived on old-cohort noise statistics
must be re-derived on new-cohort data once the loader path exists; invariant 12
forbids comparing across them, so they are provisional until then.

**Do not diff against the per-recording `_vengmetrics.mat` / `_mmc.mat` /
`_slowWaves.mat` / `_HRVMeasures.mat` sitting beside each recording.** They are
tempting as a free clean baseline and they are the wrong one: they were produced
by the stale configurations below, with older code. Recompute the clean baseline
with the corrected consumer, in the same run.

#### Correct the three stale configurations FIRST — approved

`batch_spike_detect.m` hardcodes 300–5000 Hz, 6σ, order 3 and never reads
`pipeline_params.m`, which already says 300–3000, 4.5σ, order 4,
`sigmaReference='session'`. `detectSortNerveSpikesECAP.m:156` computes its own
whole-file σ and ignores the session reference. `HR_BR_HRVAnalysis_new.m:171`
still has `hrBandHz = [1 100]`, the band task 05 retired on measurement
(10–150 at k=6: 0 false beats in 3160; 1–100 at k=3: 14).

Step 9's premise is that a tolerance is a property of **the consumer as it
actually runs**. Run it today and T is stale on arrival, the 09 gate inherits
the staleness, and the labelling then targets it. **These are task 08 rows that
task 08 missed** — it listed `step1_bandpass` and `step2_noise_sigma` but not
the detector that consumes them, which is an omission in the spec, not in the
work. Fix all three, then derive T once.

Passing the session σ into `detectSortNerveSpikesECAP` also resolves the
sweep's worst confound for free: a σ recomputed with the injection present is
inflated by it, which deletes **real** spikes far from the injection site, so
the tolerance curve would carry a global subtractive term on top of the local
additive one. The session σ is a median over ~240 windows and one injection
barely moves it. Report the curve decomposed into **spurious-added** and
**real-lost** regardless — both are real damage and they have different causes.

Expect the spike curve to be **non-monotonic** and do not "fix" it:
`maxAmpUV = 150` means an injection above that produces candidates the amplitude
gate then discards, so beyond 150 µV the damage stops being spurious spikes and
becomes σ inflation and masking alone. That is the consumer's own crude artifact
rejection showing up in the measurement, and it belongs in the report.

#### `bad_fraction == 0` is ambiguous — check `blankingApplied`

The second host was nearly a trap: `M50E100_lol_CME1_bl_2124` reports
`bad_fraction` 0.0000, and that meant **never reviewed**, not clean. Its
MAD-σ is 0.119 µV against a std of 19.7 µV with **57% of samples below
0.1 µV** — the robust σ is not a noise scale at all, so a σ-gridded sweep there
would be meaningless and incomparable to any other host.

Two rules follow, both general:

1. **`blankingApplied` is in every `_blankmotion.mat`. Read it.** Zero segments
   with `blankingApplied = true` means reviewed and clean; zero with it false or
   absent means unreviewed. Never treat the second as the first — anywhere,
   including when building a training corpus.
2. **Sanity-check σ before gridding on it**: reject a channel whose `std / σ`
   is far from ~1.4, or whose samples pile up near zero. A robust estimator is
   robust, not omniscient.

#### One grid point is one realisation — replicate near the crossing

`inject_artifact` takes a seed, and `tribo` in particular is stochastic. A
threshold read off a single realisation per amplitude is a single-seed
estimate presented as a constant, which is how this project produced three
wrong numbers already. It is also sensitive to **where** the artifact landed:
a 500 ms injection falling on a breath peak is not the same experiment as one
falling between breaths.

Two passes, which costs little:

1. **Locate** — one seed across the full amplitude grid, per (kind, duration,
   channel-set).
2. **Replicate** — **5 seeds** at the five grid points bracketing the crossing,
   with the injection position `t0` varied by seed as well as the waveform.
   Report the threshold as **median and range across seeds**, never a bare
   scalar, and treat a range spanning more than one √2 grid step as a finding
   about the consumer rather than noise to average away.

#### `clip`'s amplitude axis is inverted — do not let the crossing finder assume monotonicity

For `step`, `drift` and `tribo`, `amp_ratio` scales an **additive** artifact, so
larger means more damage. For `clip`, `amp_ratio` **is the rail**: a lower rail
clips more of the signal. Damage therefore **decreases** with amplitude, and at
0.5σ the rail sits below most of the signal and destroys it.

The tolerance for `clip` is the rail **above** which nothing changes, and a
crossing finder that scans upward for "first amplitude at which the output
changes" will return the bottom of the grid for every `clip` row and look
plausible doing it. Detect the crossing per-kind with the direction declared,
and report `clip`'s tolerance as an upper-rail figure with its own units and
sign convention stated.

#### `mmc`: the criterion was mis-specified, and the diagnosis of it was too

**Corrected 2026-09-23 by reading `extract_mmc.m:299`.** `mmc.burst.events` was
never a dense per-sample threshold boolean: `ev_bool` sets **one sample true per
burst peak**, with peaks grouped at the 0.5 s `burstRefractory`. It was already
an event list.

So the `+3/−4` at 0.5σ was **3 bursts added and 4 lost**, and its cause was
`match_s = 0` — a burst peak moving by one sample scoring as one lost plus one
added — **not** hundreds of thousands of borderline per-sample decisions. The
invariant-10b reading was wrong. Recorded because it is the **third** time a
component's behaviour has been asserted from a symptom rather than read from its
source (task 01's `labeled_save.py`, the stomach-as-control argument in the
tripole check, this). The pattern is always the same: the symptom fits a known
failure mode, and the check that would refute it is cheaper than the reasoning
that supports it.

**Consequence: the strict row is not a tolerance.** An exact-match comparison at
sample resolution (41 µs) on a burst peak in a 2–50 Hz band whose impulse
response is 486 ms is a timing criterion four orders of magnitude tighter than
the signal can support. It fires at the bottom of any grid by construction, and
reporting "mmc tolerance < 0.09σ" would be a finding about the comparison, not
about the consumer.

Emit it with a status rather than a number, borrowing task 02's pattern:

```
mmc_exact   status = "criterion_degenerate"   no tolerance value
mmc_burst   status = "measured"               the tolerance task 14 routes on
```

`mmc_burst` matches within `burstRefractory` and uses it as the displacement
floor — **derived from the consumer** (the separation below which `extract_mmc`
itself will not call two bursts distinct), not chosen. Keep the downward grid
extension; it is already generated and it confirms the flatness cheaply.

#### MEASURED 2026-09-23: `mmc.burst` is not detecting MMC

Rayleigh test of burst phase within the enclosing slow-wave cycle, same
channel, from the baselines — no extra runs:

| | bursts/cycle | median R | rows with p < 0.05 |
|---|---|---|---|
| 9 channel×span rows, 2 animals | **5.0–9.3** | **0.076** | **0 of 9** |

Mean phases scatter across the cycle (0.03–0.98) with no consistency between
channels or animals. Temporal structure was tested separately to rule out MMC
phase III: inter-burst CV 0.53–1.16 (Poisson-like to slightly *regular*, not the
CV ≫ 1 of long epochs separated by quiescence), and the largest inter-burst gap
anywhere is 18.1 s across spans of 180–420 s — no quiescent period at all.

**The "90–120 minute" figure used here was wrong — that is human and dog.**
A **rat gastric** MMC cycle is **17.5 ± 5.8 min**: phase I (quiescence)
5.4 ± 1.1, phase II 7.1 ± 2.8, phase III (intense bursting) 3.2 ± 0.8,
phase IV 1.8 ± 0.4 (Zheng *et al.*, rat antrum). Two consequences, both
favourable:

- **A 20-minute recovery recording covers roughly one full MMC cycle**, so MMC
  is observable in the new cohort. The claim that nothing could be resolved on
  this timescale was based on the wrong species' number.
- The T spans are 3–7 min, i.e. **within a single phase**. "No quiescent period
  in 180–420 s" is therefore expected and is not evidence about MMC either way.

**More important: the old cohort was FED.** Animals had ad-lib food, and the fed
pattern *replaces* the MMC with continuous irregular activity — no phase I
quiescence, no phase III bursts. So the continuous low-level 2–50 Hz activity
seen in `E1000_JEL_E1000_bl_1315` (no quiet baseline, no discrete episodes,
envelope hovering at threshold) **is what a fed stomach should look like**, and
`extract_mmc` was developed and tuned on recordings in which MMC does not
exist. That is the cleanest explanation yet for why no grouping window produces
a stable burst count on it. What it establishes is narrower and sufficient: the events are
neither one-per-slow-wave-cycle, nor phase-locked, nor temporally structured.
**A tolerance on them is a tolerance for a generic 2–50 Hz activity-episode
detector, whatever the variable is called.**

##### The decisive cheap follow-up: re-group with a longer refractory

The most likely benign explanation is **over-fragmentation**, not
mis-detection: `group_events` uses a 0.5 s `burstRefractory`, and a single
real gastric spike burst lasting a few seconds can easily contain several
suprathreshold episodes separated by more than 0.5 s — which would produce
exactly 5–9 detections per cycle from one true burst.

Test it in one line: **re-group the existing burst times at 2 s, 3 s and 5 s
and see whether the group count converges on the slow-wave cycle count.** If it
does, the detector is right and `burstRefractory` is too short; if it does not,
the detector is responding to something that is not slow-wave-locked at all.
Either answer is worth more than the tolerance.

##### SUPERSEDED by the fixed-threshold measurement — see the `extract_mmc` row in task 08

The re-grouping analysis below was run on **fed** old-cohort data with the
**moving** threshold in place. Both of those are now known to be wrong for this
question: the fed pattern has no MMC, and the moving threshold suppresses the
episodes. On **fasted** new-cohort data with a **fixed** session reference, the
answer is **1.2 activity episodes per slow-wave cycle** (40 episodes in 600 s
against a 3.3 cpm slow wave) — which is the one-burst-per-slow-wave relationship
physiology predicts, obtained without tuning any grouping parameter toward it.
Keep the text below as the record of how the wrong answer was reached.

##### Re-grouping: confirmed. Phase: a powered null, and it survives both caveats

Groups per slow-wave cycle, re-grouping the existing peak times, median across
9 channel×span rows: **0.5 s → 5.92**, 1 s → 4.88, 2 s → 2.15, **3 s → 0.94
(range 0.81–1.41)**, 5 s → 0.36. **`burstRefractory = 0.5 s` fragments roughly
6:1**, and 3 s converges on one burst per cycle across two animals and three
spans.

Phase locking is absent even so, and pooling makes it a **powered** null:
n = 949 at the original grouping, R = **0.029**, against ~0.056 needed for
p < 0.05. Two objections to that null, both answerable:

- **"Fragmentation dilutes R."** It does, but not nearly enough. Fragments of
  one locked burst occupy at most a few seconds of a ~12.8 s cycle; phases
  spread uniformly over a fraction `w` of a cycle give
  `R ≈ sin(πw)/(πw)`, which at `w = 0.25` is **0.90**. Fragmentation could
  reduce a true R of 0.9 to 0.81, not to 0.029.
- **"The phase reference is itself noisy."** `slowWaveAnalysis_new` warns that
  **23–38% of slow-wave intervals exceed 8 cpm** on these very spans. Cycles
  with a spurious peak contribute near-random phase, so `R_obs ≈ (1−f)·R_true`;
  at `f = 0.3` that bounds `R_true ≲ 0.041`. **Strong locking is excluded.
  Weak locking remains indistinguishable from none**, and that is the honest
  limit.

So the two findings are separable and both stand: the **rate** is wrong through
over-fragmentation, and the events are **not strongly slow-wave locked** even
when correctly grouped.

##### This belongs on task 08, not task T

It is a defect in `processing_new`, found while measuring against it. **Add a
task 08 row**: validate or rename the `mmc` consumer. If the re-grouping test
says over-fragmentation, the fix is `burstRefractory`. If it says otherwise,
the variable is misnamed — `mmc` in a results table implies migrating motor
complex to any reader, and a name is a claim.

**T proceeds regardless.** The caveat rides with the number in
`consumer_tolerances.json`; it does not block the sweep.

#### Superseded — the original sanity-check instruction

The baseline is **90 / 111 / 73 bursts in 180 s** — 30–37 per minute — against a
slow wave measured at **4.7 cpm** on the same span. That is a factor of **7**.
If gastric spike bursts are phase-locked to the slow wave, roughly one per cycle
is expected and 7× suggests the burst detector is firing on something else.

Check it with data already in hand: take the baseline `mmc` burst times and
`slow_wave` peak times and test whether bursts concentrate at a consistent slow-
wave phase. **If they do not, the `mmc` tolerance is a tolerance for a detector
that is not detecting MMC**, and that matters more than its numeric value.

#### The 105 empty bracket points, found 2026-09-25

The replicate sweep reported 1710 rows and zero failures. The per-point consumer
counts did not agree with that:

```
consumers run per point: {0: 105, 1: 795, 2: 675, 3: 85, 4: 45, 5: 5}
```

All 105 zero-consumer points carried the mask `('breathing',)`. `hrv` and
`breathing` come out of a single `HR_BR_HRVAnalysis_new` call, and the sweep
gated that call on `want.hrv` alone — so every breathing-only point computed
nothing, and those are precisely the points that existed to fill breathing's
bracket. 6% of the pass produced nothing while the run reported success.

**The cause is the shape worth remembering.** Before per-consumer masking,
`want.hrv` and `want.breathing` were *always equal*, so reading one as a proxy
for the other was harmless and correct. Masking made them independent, and every
place that had quietly relied on their equality became a bug at that moment
without being edited. (Invariant 28.)

The fix is the right kind: the call now runs if either is wanted and records only
the fiducials actually requested, so the recorded set cannot disagree with the
mask again. That converts a property somebody has to check into one that cannot
be violated.

**Name the count for what it counts.** `n_consumers_run` reaching 4 and 5 for a
three-consumer mask is explained — one `mmc` call yields `mmc` and `mmc_burst`,
one HR_BR call yields `hrv` and `breathing` — but an explanation that has to be
repeated is a naming defect, and this is invariant 18 again: two different
things (consumers invoked, fiducials recorded) sharing one name. Rename it
`n_fiducials_recorded`, and assert separately that **the set of consumers
invoked equals the mask exactly**. A set comparison, not a count — a count can
agree by coincidence, and this bug is what a count agreeing by coincidence looks
like.

#### The cost model is biased 2×, and that is now a measurement

**23,875 s (6.6 h) on 8 workers against a 3.3 h projection.** Same direction and
roughly the same factor as the previous estimate. Two consecutive 2× misses in
the same direction is not variance, it is a calibration error, and by invariant
23 an arithmetic disagreement is a defect signal whether or not anything visibly
broke.

**Until the serial control lands, multiply every projection from this model by
2 and state that the factor is empirical.** A projection quoted without it is
known to be wrong.

The serial control decides where the factor lives, and the two answers imply
different actions:

- Summed per-point `wall_s` is 28.17 h, measured *under* 8-way contention, so it
  is not a single-core figure. Against 6.6 h × 8 = 52.8 h of worker time, that
  is **53% utilisation** — low for work this close to embarrassingly parallel.
- If the serial per-point cost matches the model, the 2× is contention and the
  remedy is fewer workers, not more: this pass reads large files off the Drive,
  and 8 workers contending for one I/O path can be slower than 4 that do not.
- If the serial per-point cost is itself ~2× the model, the per-point estimate
  is wrong and the worker count is innocent.

Report both numbers — serial per-point mean and the implied parallel efficiency
— and set the worker count from them rather than from 8 being a round number.

#### Report the statistic as well as the fiducial, for `hrv`

A.4 declares `hrv`'s criterion as "beat train unchanged", and the
pre-registered 1 ms displacement honours that. But 1 ms on one beat out of
~1250 moves RMSSD by well under a tenth of a percent, so the fiducial
criterion is far stricter than "material to the statistic the consumer
exists to produce". Both are legitimate and they will give different
tolerances.

**Report both curves from the same runs** — the fiducial one as primary, since
that is what A.4 declares, and a statistic-level one (RMSSD, SDNN, pNN5)
alongside. Task 14's routing should be able to see the gap between "the beat
train moved" and "the number a paper would report moved".

#### Two smaller corrections to step B

- **`slow_wave` needs a longer span than 180 s.** Thirteen to fifteen peaks per
  channel at ~4.7 cpm is the thinnest statistics of the five and its curve will
  have the widest error bars. Host 2 has a single **599 s** clean run — run the
  `slow_wave` rows there, where the same span gives ~47 peaks per channel.
- **`breathing`: report displacement, do not threshold on it.** Count-only is
  the right *criterion*, but the stated reason — that half the 3988 ms impulse
  response exceeds the breath interval — conflates settling time with timing
  resolution. A 0.5–3 Hz band passes 1.58 Hz breathing perfectly well and can
  time successive peaks; the long impulse response is an edge effect, not an
  inability to resolve. So report the displacement distribution as data and let
  it show whether a displacement criterion is recoverable later.

#### Injection targets, and the cross-consumer coupling

Inject per-consumer into the channels that consumer actually reads (nerve for
`spikes`, stomach for `mmc` / `slow_wave`, the HR channel for `breathing` /
`hrv`), **plus a common-mode condition** that puts the same waveform on all
five channels at once. The common-mode condition is the motion case and it is
the only one that exercises the tripole's rejection — without it the sweep
measures differential artifact only.

**`mmc` is downstream of `hrv`.** `extract_mmc` needs the `_HRBR` output, so
each sweep point runs `HR_BR_HRVAnalysis_new` first. The consequence is real
and must be reported rather than engineered away: **an injection on a nerve
channel can reach `mmc` indirectly**, by perturbing the R-peaks used for
cardiac removal. A consumer's tolerance is therefore not a property of its own
input alone, and the dependency belongs in the output.

#### `stomach_ref` — defined 2026-09-23, and it means two different things per cohort

It is the **reference electrode for the stomach EMG**, and how it reaches the
data changed between cohorts:

| | referencing | stomach channels in the file |
|---|---|---|
| **old cohort** | in **hardware** — 3 recording channels against a local reference that was never digitised | 3, already referenced |
| **new cohort** | **none in the TDT banks**; the reference electrode is digitised as its own channel and subtracted in software | 3 recorded, of which one **is** the reference → 2 usable signals |

**Consequence for T: nothing is blocked.** The old-cohort host is
hardware-referenced, so `mmc` and `slow_wave` run there unmodified. **Derive
all five consumers on the old cohort.**

**Consequence for later, which is not small.** Software referencing is not
equivalent to hardware referencing and the tolerances do not transfer
unchanged:

- **Noise rises by about √2.** A hardware differential amplifier has one noise
  source; subtracting two separately digitised channels sums two independent
  ones. A tolerance expressed against σ therefore shifts.
- **Gain and phase mismatch between the two ADC paths leaves residual common
  mode**, which is exactly the motion signal the tripole work exists to remove.
  Any mismatch shows up as motion surviving the subtraction.
- **The number of usable stomach signals drops from 3 to 2**, so any statistic
  pooled across stomach channels is not comparable across cohorts — invariant
  12 territory.

**The name is wrong and must change before it causes a bug.** `stomach_ref`
reads as "the reference electrode" and A.4 uses it to mean "the stomach signal
*after* referencing" — two things one letter apart, and the consumer table is
consumed by code. Rename: the electrode is **`STOM_REF`** (a raw contact), the
derived consumer signal is **`stomach_referenced`**. Detection still reads
`STOM_REF` like any other raw contact (invariant 6 — motion appears on it too);
it is only barred from being a *consumer* signal.

**The software re-referencing derivation has no owner.** Task 04 derives
`V1, V2, V3, T` for the nerve and stops there. The stomach derivation belongs
beside it. But **which derivation is a measurement, not a declaration** —
Andrea's instruction, and the right one: there is no tripole here, no geometry
forcing the answer, so test it.

##### The stomach electrodes are ordered, and that predicts the answer

The three contacts run **closest to the pylorus → farthest**, and the gastric
slow wave **propagates** along that axis (proximal to distal, a few mm/s). This
makes the reference choice a trade-off rather than a ranking:

- **Common mode is instantaneous** across all three, so any subtraction rejects
  it.
- **The slow wave is not** — it arrives at each contact with a phase lag, so
  subtraction also *cancels part of the signal*, and it cancels most between
  the contacts that are closest together in propagation time.

Prediction to test: **an end electrode is a better reference than the middle
one**, because the middle gives two short-baseline pairs and cancels most; an
end gives one long-baseline pair that preserves the most slow wave. If the
measurement disagrees with that, the disagreement is the interesting result.

##### Test three families, not three electrodes

Restricting the test to "which of the three as reference" assumes a
common-reference montage. Two alternatives are standard for propagating gastric
signals and must be in the comparison:

```
common reference   STOM_i - STOM_k        for each k   -> 2 signals   (Andrea's proposal)
common average     STOM_i - mean(STOM)                 -> 3 signals   (max common-mode rejection)
sequential bipolar STOM_1-STOM_2, STOM_2-STOM_3        -> 2 signals   (classic for propagation;
                                                                       also yields direction and velocity)
```

##### The criterion is pre-registered, per consumer, and is NOT one number

**"Best output" chosen after seeing the curves is curve-fitting**, and this
project has already produced a channel that scored best by detecting 53% fewer
beats. Fix the metrics first, report all of them per derivation, and let the
trade-off be visible:

| metric | what it protects |
|---|---|
| slow-wave SNR in `0-2` | signal preserved, not cancelled |
| `2-50` SNR | the `mmc` consumer, which is a different band |
| residual common mode | coherence of the derived signal with the common component shared across the **nerve** channels, which is motion |
| cancellation loss | derived amplitude against the best single contact's amplitude |

**`mmc` (2-50 Hz) and `slow_wave` (0-2 Hz) may prefer different derivations,
and that is a legitimate outcome** — they are different consumers and the
consumer table already allows per-consumer signals. Do not force one winner.

Once chosen: **declared in the channel map, recorded in provenance, and never
mixed** (invariant 12). This is a measurement task on **new-cohort** data, where
`STOM_REF` is digitised — `gems_j_t01_ms3_bl_230315` has the three contacts. It
**does not block T**, which runs on the hardware-referenced old cohort.

#### `slow_c` is a stale row — struck

Searched `processing_new` (136 `.m`), `GEMSBlanking`, `detector-pyqt`, the full
git history of all three, both `.docx` reports, and the C-fibre hypothesis
(`compound action`, `CAP`, `unmyelinated`, `c-fib*`, `slowConduction`): zero
hits outside this document. Independently corroborated: of the eight derived
output files beside each of 406 recordings, **none is `slow_c`-shaped**. It was
never implemented because I invented it. **Remove it from A.4.** If a
slow/C-fibre consumer is wanted later, `step5e_multiband_validate.m:47` already
implements the mechanism (tripole → sub-spike band → own-sigma MAD) with band C
at 100–500 Hz; promoting and narrowing it is a small change, and it would be a
new consumer with its own tolerance, not this row revived.

**Scope: the five implemented consumers.** `velocity` is task 18 and does not
exist yet; its tolerance is already measured and recorded in A.5 (~1× raw, ~5×
band-limited, ~1.4× broadband), so T records that value by reference rather than
re-deriving it. **`slow_c` is struck** — see the ruling below; this sentence
previously said "find it or say so" and contradicted it.

### Step 11 — labelling that doubles as the measurement

Label **complete spans of ~10 minutes total**, exhaustively, across 2–3 new
animals — not candidate adjudication. Exhaustive labelling of a short stretch is
what makes recall computable; adjudicating candidates can only measure precision.

---
