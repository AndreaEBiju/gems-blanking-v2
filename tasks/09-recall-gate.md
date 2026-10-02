<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

## Task 09 — Candidate recall gate

**Module:** `detect/recall.py`, `tests/test_recall.py`
**Depends on:** 07
**Gate:** **YES — this can invalidate the whole architecture**

### Purpose
The two-stage design rests on one assumption: **a candidate is generated wherever a
real artifact exists.** The classifier can only improve precision; it cannot recover
an artifact that was never proposed. Measure this before building the classifier.

### Measure on newly labelled NEW-cohort data
The gate runs against exhaustive human labels on 2–3 new animals (Part A.6, step
11), **not** against the 43 old recordings — see Part A.6 for why the old cohort
cannot test this generator. Synthetic injection is demoted to a secondary probe
for artifact classes the labels may not contain.

### What the old labels are still good for
Two numbers, read from `*_segment_indices.mat` alone — no signal files, no cost:
- the **duration cap** for task 07 (99th percentile of labelled segment durations),
  currently a guess
- **artifact prevalence** — roughly what fraction of a recording was marked, which
  says whether a 12–22% clean flag rate is plausible or wild

Grab both opportunistically; do not block on them.

### Use the real artifact library, not hand-made waveforms
**Verified 2026-09-21:** the Phase 2 machinery is `GEMSBlanking:detector/synthesize.py`
(`mark_high_confidence_clean`, `inject_saturation`, `inject_drift`,
`inject_broadband`, **`inject_transplant`**, `synthesize_positives`,
`BadChunkLibrary`, `INJECTION_FUNCTIONS`), plus `detector/phase2.py` and
`scripts/phase2_synthesize.py`.

**`inject_transplant` and `BadChunkLibrary` are better than anything in
`conftest.py`** — they splice *real* artifact chunks into clean spans, so the
morphology is real rather than a guess about what motion looks like. The first
gate run used hand-built waveforms and one of the four (`tribo`) was malformed.
Prefer transplant for the gate; keep the parametric kinds for the
amplitude/duration sweep, where a known amplitude is the point.

Reconcile the naming rather than duplicating it: `inject_saturation` spans what
`conftest` now splits into `step` and `clip`, `inject_drift` → `drift`,
`inject_broadband` → `tribo`.

### Method A — synthetic injection (secondary)
Inject artifacts of known type, amplitude and duration into real clean recordings.
Reuse the Phase 2 synthetic machinery in `GEMSBlanking` — it moves from a validation
figure to a runtime component. Sweep amplitude ratio 0.5–20× and duration 5 ms–5 s
across all four artifact kinds. Report a recall surface, not a scalar.

### Method B — human recall audit
Sample a few minutes from each of ≥6 recordings, have them scrolled in full (task
16's audit mode), and count artifacts a human finds that no candidate covers.

### PRE-DECLARED 2026-09-26, before any audit marks exist: how the gate is scored

Written down now so that nothing about the scoring is chosen after seeing
Andrea's marks.

**Unit and coverage.** One artifact = one interval Andrea commits in a blind span.
An artifact is **covered** if at least one candidate overlaps it by any amount —
the two-stage design needs a candidate to be *proposed* inside the artifact;
task 13's extent sets the boundaries. Report, as secondary numbers only, the
fraction of each artifact's duration that candidates cover, and the recall at
≥50% overlap.

**Statistic.** `recall = covered / found`, with an exact (Clopper–Pearson) 95%
interval. Artifacts within one span are not independent, so also report a
span-level bootstrap interval; if the two disagree, the wider one is the one
quoted.

**What 10 minutes can and cannot show — plan the audit as sequential.** Proving
recall ≥ 98% needs the lower bound above 98%: with **zero** misses that takes
~150 artifacts (0.05^(1/n) ≥ 0.98 → n ≥ 149). Five 2-min spans will likely
yield tens, not ~150. So the first round **can refute** the gate — a single
miss in ~30 artifacts puts the point estimate below 98% — but it **cannot
confirm** it. Therefore:

1. After round 1 (the 5-span plan), score it.
2. **Any miss:** diagnose each one from the z-traces — *generator blind spot*
   (z low → new band or feature) versus *threshold* (z high but under
   `z_enter`) — fix task 07, and only then continue. Labelling more against a
   generator already known to miss wastes Andrea's time.
3. **Zero misses:** draw another round with a fresh recorded seed, same
   stratification, and continue until the lower bound clears 98% or the labelling
   budget Andrea sets runs out. Report the bound reached either way.
4. Report per round: artifacts found, covered, missed, the interval, and
   artifacts found per minute (so the number of rounds needed can be projected
   rather than guessed).

**Ratified 2026-09-27, after the scorer was built (before any marks exist):**

- **The gate uses the one-sided 95% lower bound**, the more conservative of
  Clopper–Pearson and the span bootstrap. The claim is "recall ≥ 98%", a
  one-sided claim, and 149 artifacts at zero misses is the one-sided figure. My
  text above said "95% interval" and projected with the one-sided bound; that
  was inconsistent, and the one-sided bound is the one that governs. Two-sided
  intervals are shown alongside, as description.
- **A new round needs every miss diagnosed *and* its task-07 fix recorded.**
- **Diagnosis classes use the generator's own thresholds**: blind spot (z <
  `z_exit` in every band), threshold (`z_exit` ≤ z < `z_enter`), **gated** (z
  reached `z_enter` but a later generator rule rejected it), undiagnosed (no
  trace covers the artifact).

**Rounds labelled before a generator fix cannot count toward the gate after it.**
Once a miss has been used to change task 07, the marks that revealed it are
tuning data for the generator. Re-scoring them with the fixed generator is
useful — it shows whether the fix recovers the misses without new labelling —
but the gate's lower bound is computed **only from rounds labelled after the
last fix**. Otherwise the gate measures the generator on the very artifacts it
was adjusted to catch. The score file records which rounds are eligible.

**A score records exactly what it scored.** The window does not save the
candidates it revealed, so the scorer recomputes them. Two guards:
- the score file records the generator's parameters (`z_enter`, `z_exit`, gate
  settings) and the package commit (or a hash of the working-tree source when
  uncommitted);
- from now on, the window writes a digest of the revealed candidate list into
  each span record at reveal, and the scorer refuses to score a span whose
  recomputed candidates do not match it. Spans revealed by an app started
  before this change have no digest; score them with a warning saying so.

This supersedes Method B's "≥6 recordings" for the first round: the 5-span plan
across 2–3 animals is round 1; later rounds add recordings.

### ROUND 1 RESULT, 2026-09-28 — recall passes, the generator fails its budget

**Pre-declared score:** 91 found, 91 covered, 0 missed (9.1 per minute). One-sided
95% lower bound **0.968** against 0.98: not cleared. All five spans carried
reveal digests and every recompute matched. Clopper–Pearson set the bound; the
span bootstrap collapsed to 1.000–1.000, as it must with zero misses, and says
nothing about clustering (80 of 91 marks sit in two spans, so 0.968 is if
anything optimistic).

| span | animal, cond | candidates / 120 s | time covered | marks | chance recall |
|---|---|---|---|---|---|
| 1 | B bl | 307 | 20.5% | 1 | 0.25 |
| 2 | J sr | 344 | 73.7% | 48 | 0.85 |
| 3 | A sr | **1,340** | **95.9%** | 1 | 0.98 |
| 4 | B sr | 180 | 93.4% | 32 | 0.96 |
| 5 | I bl | 153 | 93.0% | 9 | 0.96 |

Chance recall (descriptive; defined before the score existed): the probability a
randomly placed interval of each mark's duration overlaps a candidate. In spans
3–5 candidates cover 93–96% of the time, so 100% recall there is **no evidence**
the generator responds to artifacts. Nearly all the evidence is span 2.

**Ruling: do not add a post-hoc chance condition — apply the budget that was
already declared.** This section has required, since before any marks existed,
that the pinned `z_enter` meet **both** recall ≥ 98% **and ≤ 3000 candidates per
recording**. Scaled to a 20-min recording, span 3 is ~13,000 and spans 1–2 are
~3,000–3,400. **The budget condition fails.** So the correct reading of round 1
is not "one more round clears the gate" — it is that the generator, as
configured, flags most of the recording in some spans, and a second round drawn
now would clear the recall bound by coverage alone. Also enforce it in code:
`--check-next` and the Create-plan button refuse while the current generator
exceeds the budget on the eligible pool.

**Order of work before round 2:**

1. **Attribute the over-coverage.** In each high-coverage span, which signal
   and band carried the max z over the covered time? If a few signals dominate,
   it is a contact-quality problem (the adapter checks found collapsed tripole
   fits, a flat J ANT3, K's two cuffs with identical σ and zero spikes); if it
   is spread across signals, it is the threshold or the reference. Also check
   whether stim/recovery spans are high because the reference is computed over
   a window that includes, or excludes, the recovery state. No labels needed.
2. **Fix accordingly** — a per-contact quality screen that removes dead, flat
   or duplicated contacts from the max (invariant 6 says detection reads the
   contacts; it does not say it must read broken ones), and/or the `z_enter`
   sweep 2.0–4.0.
3. **Measure the budget on many eligible new-cohort recordings**, not five
   spans: candidates per recording and time-covered fraction, per animal and
   condition. Unlabelled, so it costs no labelling.
4. **Re-score round 1 at the new setting as a TUNING CHECK**, and record the fix
   with `fixed_at`. Round 1 is then tuning data by the existing rule.
5. **Round 2 at the pinned setting is the first gate evidence.** Chance recall
   and time-covered fraction are reported with every round from now on.

### CORRECTION AND RULINGS, 2026-09-28 (after the budget measurement)

**My "budget fails" ruling was wrong.** I scaled span 3's 1,340 candidates as if
they were per 120 s span; they were counted over the whole assessable region.
Measured properly on 60 pool recordings (5 per animal × condition), the budget
holds with a wide margin: median 335, q90 716, max 1184, none over 3000.
**Ratified:** the budget statistic is the 90th percentile over ≥ 30 sampled
recordings, keyed to the generator's hash.

**The real issue is time coverage, not count** — median 54% of time covered,
pooled chance recall 0.897. That does **not** make the gate uninformative: a
generator blind to artifacts would score ~0.90 and, over 149 artifacts, miss
~15 of them, so clearing a 0.98 lower bound is strong evidence it responds to
artifacts beyond chance. What would make the gate uninformative is coverage high
enough that chance itself approaches the bar. **So, declared now, before round
2:**

- **A round counts as gate evidence only if the pooled chance-recall upper 95%
  bound is below 0.98.** At the current setting it is 0.956. If a future
  generator change pushes chance to the bar, scoring refuses to count the round
  and says why. Report the margin (0.98 − chance upper bound) with every round.

**Keep `z_enter` 3.0 and `z_exit` 1.5.** Every lever that cuts coverage costs
recall on the round-1 tuning marks: `z_exit` 2.25 loses 5/91, 2-of-54 agreement
loses 9/91, `z_exit` 3.0 loses 10/91. Recall is what the gate protects;
precision is the classifier's job. The hotter first ~3 minutes of every region
is left alone: those minutes plausibly carry more real motion (the animal has
just been handled), and invariant 5 fixes the reference.

**Contact screen: ratified**, including both corrections found on real data (the
copy threshold at √(1−r²) < 0.01, and "off the cuff" judged against every peer,
not the mean). Round 1 is tuning data, `fixed_at` 2026-09-28T17:07:01Z.

**Three structural changes before round 2:**

1. **One construction site for detection (invariant 33).** Move the chain
   (derivations → contact screen → z → candidates) out of the audit bridge into
   `gems_blanking_v2`, and have the bridge call it. Production and audit must
   share it. Prove the move changes nothing: candidate lists identical on all
   60 cached budget regions and the 5 round-1 regions. Then re-run the budget
   under the new hash.
2. **The generator hash covers the generation chain only** — derivations,
   screen, z, candidate generation and their constants — not the scorer or
   reports. A reporting edit must not invalidate a budget record or a round.
3. **`pooled_gate` drops any round scored under a generator hash different from
   the current one**, in addition to `fixed_at`, so a forgotten fix record
   cannot leave tuning data in the pool. Fix the projection to count only
   eligible rounds (149 fresh artifacts now, not "58 more").

Then round 2 may be drawn.

### Spike polarity — Andrea's prior, and the check it implies

Andrea: vagal spike rates are usually 1–5 spikes/s, and as she remembers it the
negative-only convention gave reasonable numbers on the old setup. **Keep
negative-only for now**, and test that memory against data rather than against
recollection:

- Report event **rates per second** per cuff under both conventions, next to
  the 1–5/s range. From the numbers already measured: ORE negative-only is
  0.3–0.5/s and absolute 1.0–1.9/s; JEL is 32–46/s negative-only and 72–90/s
  absolute — both far above 1–5/s, so what the JEL events are is itself a
  question (multi-unit activity, a stimulation remnant in an `E1000` host, or
  something else).
- **Compare with what her pipeline actually reported** for the same recordings:
  the old `*_vengmetrics.mat` (or equivalent) outputs hold her historical rates.
  If they sit near the negative-only numbers, her memory and the pipeline
  agree; if not, the record says so.
- **New cohort:** most software-tripole cuffs give 0–7 events in a span — near
  0/s, far below 1–5/s. Either the tripole is not capturing spikes at 6σ, or the
  left-cuff connector problem below is hiding them. Report this as a finding;
  it matters for everything spike-based on the new cohort.

### Polarity result, 2026-09-28 — negative-only stands; mains impulses found

At her pipeline's real settings (4.5σ, 1.0 ms, 100–5000 Hz historical /
300–3000 Hz now — not the detector's defaults, which I had quoted; invariant
39), and excluding mains-contaminated trains, her historical `vengmetrics`
median is **5.0 spikes/s**, about half inside 1–5/s. Negative-only matches her
pipeline's counts; absolute does not. **Keep negative-only.** New-cohort
tripoles give 0.1–11/s at her settings — low, not near zero (my "0–7 events"
came from the 6σ defaults).

**Mains contamination in her historical spike trains.** 47 of 175 trains are
dominated by 30/60/90 Hz structure, and rate tracks the contamination (Spearman
0.65). JEL's "spikes" are mains impulses: event-train peaks at 30/60/90 Hz,
median inter-event 16.6 ms, widths 0.12–0.16 ms — too narrow for a unit — in bl
and sr files alike. Impulses this narrow are broadband, so a 60 Hz notch (with or
without harmonics) does not remove them. Consequences:

- **`T_hardware` tolerance rows measured on the JEL host are invalid** — they
  measured the sensitivity of a mains-impulse count, not of spike detection.
  Mark them `gate_eligible: false`, reason `mains_dominated_host`; ORE (clean in
  every train) is the old-cohort `T_hardware` host from now on. JEL rows for
  other consumers are unaffected by this finding.
- **List every contaminated train for Andrea** — recording, cuff, rate, mains
  ratio — so she can see which of her analyses used them. Change nothing in her
  outputs.
- **A mains-locked-event flag is worth adding to the spike path later**
  (inter-event histogram at 1/60 s, phase locking to mains). Andrea's call.

### Generator hash must include the loader

The hash now derives from the chain's imports (15 modules) and excludes the
scorer and store — right. But the **loader** sits outside it, and the loader
decides which samples reach the chain: a change to channel mapping, units,
NaN handling or slicing changes candidates without moving the hash. **Include
the load path from file to the array handed to `detect_region`** (loader,
channel map, units scaling, NaN interop). The stim split stays outside: it
chooses regions, and each span's region is recorded, so it cannot alter the
scoring of a recorded region. Re-run the budget once under the widened hash.
**Ordering — corrected.** I wrote that this "does not block round 2"; it would:
Create-plan refuses without a budget measured under the current hash, so
shipping the widened hash first would lock Andrea out until a Drive-heavy re-run.
So: **Andrea creates round 2's plan first, under the current hash and its valid
budget. The widening is built in a separate worktree, not the tree the app runs
from** (a lazy import mid-session could otherwise load a mix of versions), and
lands only after round 2's last span is committed. Then re-run the budget under
the widened hash, then score round 2. The loader itself is unchanged, so round
2's reveal digests verify under the new scope.

### Contact quality — a systematic LEFT-cuff problem

Every screen failure in the 60-recording pool is "off the cuff" (r ≈ 0 with its
peers, σ 3.8–31 µV — alive, but sharing nothing), and **every one is on the left
cuff**, intermittently (1–5 of ~10 recordings per animal). Right cuffs pass in
every recording. An intermittent failure on one side in every animal points at
the left lead or connector rather than at individual electrodes — a hardware
question for Andrea, and one worth settling before the next implants. Velocity
(task 18) is possible on the right cuff throughout and on the left cuff in
6–10 of ~10 recordings per animal; restrict velocity to recordings whose cuff
passes the screen.

Extend the screen to stomach contacts later: B's ANT1 is 60 Hz-dominated
(σ 361 µV vs 54–58 µV) and feeds `stomach_ref`.

### ROUND 2 RESULT, 2026-09-28 — 14 misses; the rule says stop

`plan_20260928T200912Z_3dc153e2` (seed 1036080098), labelled after the last fix,
every reveal digest verified. **166 found, 152 covered, 14 missed** (16.6/min);
recall 0.916 (CP 0.863–0.953), one-sided lower bound **0.786** against 0.98.
Time covered 54.8%; chance recall 0.750 (0.687–0.807), margin +0.173 — so the
round is valid gate evidence, and it fails. The generator responds to artifacts
(P(all covered by chance) ≈ 3e-24) but misses too many. Andrea confirms she used
the **same criterion** as round 1; round 2 simply contained more, and briefer,
artifacts.

**Misses, by class — none gated:**

- **Threshold (7):** z 1.54–2.98, **six of seven peak on `stomach_ref`**.
- **Blind spot (7):** z < 1.5 in every band; all **20–60 ms** marks, peaking on
  raw nerve contacts, R_T or `stomach_ref`. Eight of the 14 are in span 1
  (H stim/recovery).

**Diagnosis 1 — the stomach is read only through a common average.** The
detection set carries raw nerve contacts but, for the stomach, only
`stomach_ref`, a common average of ANT1–3. A common average subtracts what the
contacts share, and motion artifact is largely what they share — so the one
stomach signal detection reads is built to cancel the thing it should detect.
That fits six threshold misses peaking on `stomach_ref` just under `z_enter`.
**Proposed fix:** add raw ANT1–3 to the detection signal set — invariant 6
(detection reads the contacts) applied to the stomach as it already is to the
nerves — and extend the contact screen to stomach contacts, adding a mains-hum
rule (B's ANT1 is 60 Hz-dominated). Do **not** lower `z_enter`: at 2.0 chance
recall reaches 0.95 and the chance margin collapses, and pinning it to 2.8 to
catch 2.85 and 2.98 would be fitting the threshold to the misses.

**Diagnosis 2 — brief events are a structural blind spot of envelope z.** A
20–60 ms event is averaged away by an envelope window longer than itself, so no
threshold recovers it. Whether these are target artifacts (brief electrode pops
— "sharp excursions", which Andrea blanks) or something else is **Andrea's
judgement from the traces**, not the generator's. If they are artifacts, the fix
is a fast path: a short-window detector on the broadband raw contacts, sized to
the marks' durations.

**Evaluation rule for any fix (tuning, not gate evidence):** re-score rounds 1
and 2 with the candidate fix and report, together — misses recovered by class;
round-1 marks still covered; time covered and candidate budget on the 60-region
pool; pooled chance recall and its margin against 0.98. A fix that recovers
misses by raising coverage until chance approaches the bar is not a fix.

**Overlapping marks are one artifact — Andrea, 2026-09-28.** Because of the
viewport's width she sometimes marked one artifact as several slightly
overlapping marks. The scoring unit therefore becomes **one artifact = one
connected run of committed marks**: marks in the same span whose intervals
overlap or touch (gap ≤ 0) are merged into their union before scoring. The rule
is mechanical, independent of the candidates, and applied identically to every
round.

- **Marks separated by any positive gap stay separate** — that could be two
  artifacts. List pairs with gaps under 100 ms for Andrea to see, unmerged.
- **Report both** the as-committed score and the merged score for rounds 1 and
  2, clearly labelled. Both rounds become tuning data once a fix is recorded, so
  this changes no gate decision; from round 3 the merged unit is the declared
  unit.
- **Check this first against the 7 brief misses.** A 20–60 ms sliver at a
  viewport edge, overlapping a larger mark that *was* covered, is exactly what
  this would produce. Any brief miss that merges into a covered artifact is not
  a blind spot at all, and does not need a fast path.
- **Committed marks files are not rewritten.** The merge is a scoring step over
  the committed marks, not an edit of them.
- **Fix the cause in the window for round 3:** a mark may extend past the
  visible viewport, and at commit the window shows any overlapping or touching
  marks merged, with a count, before she confirms.

**Committed marks are never edited.** If Andrea judges a brief mark not to be an
artifact, that is recorded as her classification of the miss and reported
alongside the score; the score itself stands as committed. Changing labels after
learning they were missed is the one kind of relabelling the audit cannot
survive.

Rounds 1 and 2 are both tuning data once a fix is recorded. **Round 3 is the
next gate evidence.**

### RULING 2026-09-29 — the contact screen must not hide signals from detection

**My ratification of the contact screen for detection was wrong.** I ratified it
without asking whether the consumers also stop reading a screened contact. They
do not: `processing_new` consumes its inputs whole and knows nothing of the
screen. So removing a contact from detection's max while every consumer still
reads it guarantees that artifacts on that contact damage consumers unflagged.
Round 2 showed the cost directly: the screen removed the left cuff's contact 3 in
4 of 5 spans, and that contact carried rail-scale electrode pops
(24,000–59,000× σ, one at −2 V) exactly inside Andrea's marks — **9 of the 14
misses, 5 of the 7 "brief blind spots"**. Diagnosis 2 is withdrawn for those
five: they were a screen defect, not an envelope-z limitation. (Invariant 43.)

**Rulings:**

1. **Detection reads every signal any consumer reads.** The contact screen is
   removed from detection's max. It stays for what it is right for: whether a
   cuff's `T` is trusted, which cuffs task 18 may use for velocity
   (`velocity_cuffs()`), and the per-recording contact-health report.
2. **Raw ANT1–3 join the detection set**, on the same principle — `slow_wave`
   and `mmc` read them raw, so an artifact the three share (and a common average
   cancels) damages those consumers. Adopted on principle, not on the two
   threshold misses it recovers.
3. **No stomach screen for detection** — it hides `stomach_ref` and cost 5
   round-1 marks. A hum rule may inform the contact-health report only.
4. **`z_enter` 3.0 and `z_exit` 1.5 unchanged.**
5. **Adopt 1 + 2 as one recorded fix** (`fixed_at`), after measuring them on the
   60-region pool: budget q90, time covered, and pooled chance recall with its
   margin. On the round regions the combination covers 246/250 with chance upper
   0.916 (margin +0.064) — adequate, but the smallest margin yet, so it is
   reported with every round from now on. If the pool pushes chance past the bar,
   stop and report rather than adopt.

Rounds 1 and 2 become tuning data under this fix. **Round 3 is the next gate
evidence**, drawn only after the fix is adopted, the budget re-measured under
its hash, and Andrea has classified the misses in the PDF.

**Remaining misses after the fix (tuning view): 2 brief (r2 s1 at 326.44 and
326.91 s) and 2 threshold.** Fix 2 (a fast path) is considered only if Andrea
classifies the two brief ones as artifacts.

### RULING 2026-09-29 — how a miss is closed

The chain fix is **adopted** (`0abda6b`, `fixed_at` 2026-09-28T23:35:43Z, budget
key `0133349b…_4c4337ce…`): q90 595, median time covered 61.5%, pooled chance
upper 0.916 (margin +0.064). Rounds 1 and 2 are tuning data. Four round-2 misses
remain: s1 #22 (60 ms) and s1 #23 (20 ms), brief; s1 #29 (420 ms) and s3 #17
(50 ms), threshold.

The sequential rule required a task-07 fix for every miss before another round.
That rule exists so Andrea does not label against a generator already known to
miss **for a fixable reason**; it cannot mean "no round until every miss is
fixable", which would block forever on a miss nothing should fix. **A miss is
closed by exactly one of:**

1. **Fixed** — a recorded generator change recovers it (the 10 already closed).
2. **Not the target class** — Andrea classifies it as not an artifact she would
   blank. Recorded as her classification; the committed mark and score stand.
3. **Accepted limitation** — diagnosed, no fix without fitting the generator to
   the miss, and accepted with a written reason. The two threshold misses close
   this way: lowering or pinning `z_enter` to reach them is exactly the fitting
   this section forbids.

An accepted limitation is not a pass. **Fresh rounds measure it honestly**: if
that kind of miss recurs, it counts against the gate in rounds that are gate
evidence, and a recurring class is then the signal to design a fix (e.g. the
fast path for brief events). `check_next_round` accepts any of the three
closures and records which.

**The two brief misses wait for Andrea.** If they are artifacts, they close as
accepted limitations (a fast path for two events would be fitting); if not, as
not-target, and a labelling note is declared before round 3 so round 3 marks the
same class of event consistently.

**Watch the margin.** Median time covered rose from 54% to 61.5%; round 1 alone
now sits at +0.013. Any further expansion of the detection set must be measured
against the chance margin before adoption, as this one was.

**Contact-health report scoped to the 68 measured recordings — ratified.**
Streaming ~700 GB to extend it to the cohort is not worth it now (invariant 26).

### ROUND 3 RESULT, 2026-09-29 — clean, not yet enough; planner must stratify conditions

First gate-evidence round under the adopted chain, merged unit as declared:
**101 found, 101 covered**, one-sided lower bound **0.971** against 0.98 — not
cleared, 48 more artifacts needed at zero misses. Chance 0.731, margin +0.168;
time covered 51%. Every digest verified. None of the four accepted limitations
recurred.

**All five spans were stim/recovery.** The spec (Change 2's sampling rule) says
the plan is stratified across animals **and conditions**; the planner only
rotates animals. That is a planner defect against the spec, not a property of
round 3's result, and round 3 stands as valid evidence for what it sampled.
But a gate that clears with no baseline span among its eligible rounds has not
measured the generator on baselines. **Rulings:**

1. **The planner balances conditions across the eligible pool's spans**, using
   only the composition of earlier eligible plans — never their scores. With
   round 3 at 5 sr / 0 bl, round 4 draws baseline spans until the pool is
   balanced (5 bl), then later rounds alternate. Correct the docstring to say
   exactly this.
2. **The gate requires at least 3 eligible spans of each condition**, in
   addition to the lower bound and the chance margin. A lower bound reached on
   one condition only is reported as such and does not clear.
3. Test both, mutation-check both. The planner is outside the generation hash,
   so no budget re-run; the app needs a restart to load it.

### ROUND 4 MISS, 2026-09-29 — a common-mode transient; decide by consumer harm, not by size

Round 4 (5 baseline spans): 154/155 pooled with round 3, lower bound 0.970,
5 bl + 5 sr spans. The one miss is a **1.2 ms spike, ~310 µV, the same size on all
nine raw channels** — both vagus cuffs and the stomach. Andrea would blank it, and
says there are many like it she did not mark because they are hard to see; six
more of the same shape sit unmarked in the same span.

**What it is.** No nerve spike appears at the same instant and amplitude on two
cuffs and three stomach contacts. An identical, simultaneous transient on every
channel is electrical common mode (rig, ground, connector), not physiology.

**Why it probably does not need a mask — to be measured, not assumed:**

- **The spike consumer reads `T`**, and any tripole with a + b = 1 cancels a signal
  that is identical on its three contacts exactly (a·x + b·x − x = 0). Only the
  residual from inter-channel gain mismatch survives.
- **Stomach, HR and breathing consumers work below 100 Hz**, where a 1.2 ms spike
  carries almost no energy.
- **Velocity reads raw contact pairs**, and a common-mode transient there is a
  zero-lag (infinitely fast) event. That is physically impossible for conduction
  at 1.5 mm pitch, so task 18 must reject zero-lag / all-channel-simultaneous
  events — which is Andrea's point: distinguish them from nerve spikes by their
  characteristics, downstream.

**Also: human labels will keep missing this class.** They are hard to see, so the
audit systematically under-counts them, and the audit cannot be the instrument
that validates handling them. Physics (simultaneity across all channels) and
injection of a known 1.2 ms common-mode waveform can.

**Decision rule:**

1. **Measure consumer harm** on the seven transients plus a larger sample found by
   a simple all-channel-simultaneity filter in a few recordings: residual
   amplitude on `T` (300–3000 Hz) against the spike consumer's 4.5σ threshold;
   whether `detectSortNerveSpikesECAP` actually fires on them; their energy below
   100 Hz; and their rate per minute.
2. **If no consumer is damaged:** close the miss as `accepted_limitation` with the
   reason "common-mode transient, harmless to every consumer by construction",
   citing the measurement. Add zero-lag rejection to task 18's requirements. No
   generator change, so rounds 3 and 4 stay eligible.
3. **If the spike consumer is damaged** (residual on `T` crosses 4.5σ): add a
   deterministic common-mode-transient detector — simultaneity across channels,
   not amplitude, as the criterion — validated by injection, and routed to the
   `T` consumer's mask. Because it only **adds** candidates, rounds 3 and 4 remain
   valid, conservative evidence *if* (a) it is proven a strict superset on every
   cached region, (b) their original scores are kept, not re-scored, so the miss
   that motivated it still counts, and (c) the chance margin is recomputed with
   the added coverage. Record this as an explicit additive extension, never a
   silent exception to the hash rule.

### RULING 2026-09-29 — common-mode events: close the miss, correct the class, don't blank it

**Measured** (10 round-3/4 recordings, her real spike settings: 300–3000 Hz,
order 4, negative, 4.5σ, 8–150 µV, 0.2–2.5 ms):

- **The round-4 miss is harmless**: 2.3σ / 1.6σ on T, her detector does not fire.
- **The class is not rare.** Same-instant, same-polarity, equal-amplitude events on
  ≥ 7 of 9 contacts run at **124–2,900 per minute in every recording**, rate-modulated
  at 0.32–0.63 Hz with bursts. Common mode dominates the raw contacts' spike band
  (σ ~10 µV raw vs ~2.2 µV on T). The seven seen were the visible tail.
- **The tripole cancels them to ~1%**, but gain mismatch leaks. Damage is concentrated:
  1,319 of 1,343 firing events are in two cuff-recordings — **H t01 es2 bl left
  (83 false spikes/min)**, a cuff with a distrusted contact (σ(T) 4.9 vs ~2.2 µV,
  LVN1 6.7 vs 10–11 µV), and **B t03 sr right (14/min)**.
- My premises that failed: "electrical, not physiology" is unestablished (the
  0.32–0.63 Hz rhythm could be physiological common mode through the reference);
  "almost no energy below 100 Hz" is wrong (median 11–35%) — `mmc`/`slow_wave`
  are protected because `stomach_ref` cancels it, not by band; and the HR consumer
  reads 10–150 Hz with `best_hr_channel` undefined for the new cohort.

**Rulings:**

1. **Close the round-4 miss** as `accepted_limitation`: "common-mode transient;
   this instance measured harmless to the spike consumer (2.3σ/1.6σ on T;
   detector did not fire)". Round 5 may then be drawn. The class is a routing
   question (task 14), not a candidate-recall one — most of it is invisible to a
   human, so the audit cannot measure it anyway.
2. **Do not blank the class.** Masking T for ±1.5 ms wherever it reaches 4σ at an
   event would also remove real spikes that coincide with events — at up to ~48
   events/s, a real share of the spike consumer's time — and if the event rhythm
   is physiological, it would bias exactly the rhythm-locked spike analyses. Keep
   it only as a fallback.
3. **Distrusted cuff → the spike consumer does not read that cuff's T** for that
   recording (invariant 43 allows distrusting a derived quantity). Confirm the
   contact screen does flag H t01 es2 left; if it does not, say so.
4. **Correct the leak by subtraction (task 14's "subtract" route).** Per cuff per
   recording, regress T (300–3000 Hz) on a common-mode reference built from the
   channels **outside** that cuff (the other cuff and the stomach contacts, so
   none of this nerve's own signal is in it), and subtract k·ref. Verify, as the
   route requires: firing on events drops to the random-time control rate; an
   injected *differential* spike on T (present on this cuff only) survives within
   5% amplitude; the residual is reported per cuff. If it fails verification, fall
   back to 2.
5. **Measure first, before building:** the fraction of spike-consumer time within
   ±1 ms of an event, per recording, and whether her detected spikes coincide
   with events more than chance — that quantifies the concern behind ruling 2.
6. **HR consumer:** define `best_hr_channel` for the new cohort (task 05's ranking),
   then decide harm with the operational test (beat train unchanged), not by
   amplitude.
7. **Task 18:** velocity must not run on raw contacts — with this background, a
   raw-pair cross-correlation is dominated by the zero-lag common mode. Run it on
   the common-mode-corrected contacts from 4 (or on differential pairs), and
   reject any event arriving at zero lag across the pair or on all nine contacts
   within 0.2 ms: conduction at 1.5 mm pitch cannot be simultaneous.

**Andrea, 2026-09-29, on the source.** The new cohort has **no reference
channel**; in the old cohort the nerve reference was on the cuff and ground in the
abdominal wall, and the stomach's reference and ground were both in the fundus. So
new-cohort contacts are single-ended against the shared ground/reference, and
**anything at that site enters all nine channels identically** — the likeliest
source of the common mode (e.g. muscle activity near the ground), which also
explains why it is on both cuffs and the stomach at once. Consequences:

- Differential derivations (the tripole, contact differences) cancel it up to gain
  mismatch; **any consumer that reads a single raw contact carries it in full**
  (the HR channel; velocity if run on raw contacts). Rulings 5–7 stand.
- The subtraction reference in ruling 5 is sound: the common mode is the ground
  site's signal, identical on every channel, so channels outside the cuff measure
  it without this nerve's activity.
- The 20–40/min rhythm has no analytic meaning for Andrea **unless it is
  breathing**. Her breathing range is 70–150/min, so it is probably not, but check
  rather than assume: cross-correlate the event-rate time course with the breathing
  trace and the heart-rate trace from `HR_BR_HRVAnalysis_new` in the same
  recordings.
- **Andrea: the new cohort's ground, for both nerve and stomach, is in the
  abdominal wall.** So the common mode is most likely abdominal-wall muscle
  activity entering every channel through the shared ground. The old cohort did
  not have this problem on the nerves, because each cuff had its own reference on
  the cuff; the change of referencing between cohorts is what exposed it.
  Old-cohort results on this point do not transfer to the new cohort.
- For future implants (Andrea's call, not the build's): a dedicated reference
  electrode placed away from muscle, or recording each cuff differentially, would
  remove most of this at the source.

### Candidates are much wider than the marks — expected, but it must not reach the mask

Andrea, 2026-09-29: candidates cover far more signal than she marks, in almost
every case. That is the generator's design, not a defect in it: candidates run
from `z_enter` down to `z_exit` (1.5), merge across 100 ms gaps, take the max over
~57 signals, and the slow bands' 6–7.5 s windows smear onsets by seconds.
Measured earlier: covered time is 1.75–2.4× the time any signal is over
`z_enter`. Narrowing it at the generator costs recall (`z_exit` 2.25 lost 5/91 on
tuning marks), and recall is what the gate protects. **So width is handled
downstream, and three things must hold:**

1. **The blank is never the candidate interval.** Task 13 already says extent is
   where *that consumer's band* exceeds *that consumer's tolerance*, plus measured
   settling. Add to task 13's acceptance: on the audit marks, report per-consumer
   extent duration against mark duration. A spike-consumer extent that tracks the
   candidate width rather than the mark is a task 13 failure.
2. **Each candidate carries its core** — the sub-intervals where some signal is over
   `z_enter` — so the classifier's features and Andrea's candidate adjudication
   (task 16 Change 1) look at where the evidence is, not at seconds of clean
   signal around it. Compute cores in a module **outside** the generation hash,
   from the same z traces, so the budget and eligible rounds are untouched.
3. **Measure the review-queue load.** Candidates longer than the duration cap (p99
   of labelled durations) go to review and are never auto-masked. If merged
   candidates routinely exceed it, the review queue becomes the bottleneck;
   measure that fraction now.

### Candidate width measured, 2026-09-29 — judge long candidates by their cores

Over all 405 merged marks from rounds 1–4 (cores module `detect/cores.py`, outside
the hash; generation hash unchanged):

- **Under the duration cap, candidates are ~1.9× the marks by time**, and 61% of
  candidate time is the hysteresis tail (`z_enter` → `z_exit`); merging adds 0.5%.
  **The core touching a mark is about the mark's size** (median 0.73× baseline,
  0.90× stim/recovery). Cores localise the evidence as intended.
- **Over the cap: 24 candidates (1.9%) carry 44% of candidate time and contain 49%
  of covered marks.** Baseline over-cap candidates are 88% core — sustained real
  elevation, not hysteresis; the largest is 164 s, driven by L_V3 at 2–50 Hz above
  `z_enter` for 127 s, in a span where Andrea marked 90 artifacts.
- **The duration cap had no source.** The spec's p99 was to come from old
  `*_segment_indices.mat`; none exist. **Provisional cap: 15.3 s**, the p99 of merged
  audit marks — resting on ~4 marks and bounded by the 2-min span. Recompute after
  each round and record its source.

**Rulings:**

1. **A long candidate is judged by its cores.** Downstream of generation —
   classification, extent, Andrea's candidate adjudication — an over-cap candidate
   is split into its cores, and only a core that itself exceeds the cap goes to the
   review queue. Sending half of all artifacts to manual review is not a queue, it
   is the whole job. Generation and the gate are untouched.
2. **Attribute the 24 over-cap candidates** before designing more: for each, the
   signal and band of its core, whether that contact is distrusted by the contact
   screen, and whether any consumer reads that (signal, band) pair. L_V3 at 2–50 Hz
   is a left-cuff contact (the side that fails the screen) in a band no nerve
   consumer reads; if that pattern dominates, it says what the long candidates are.
3. The ~10% of marks wider than any candidate are rough edges on Andrea's marks,
   not misses; no action.

### ROUND 5 AND THE POST-ROUND-5 PASS, 2026-09-30 — score against the pass condition as written; correct, then mask the residual; veto raw contacts for HR

**Measured (Claude Code, 2026-09-30):**

- **Round 5:** 51/53. Pooled rounds 3–5: **205/208, lower bound 0.959** (Clopper-Pearson 0.963, bootstrap 0.959). Not cleared.
  - Eligible spans: 8 baseline, 7 stim/recovery.
  - Chance margin: round 5 −0.001 (candidates covered 65% of labelled time); pooled +0.143.
  - Duration cap: 14.9 s (p99 of 458 marks; rests on 5).
- **All three gate-evidence misses are baseline, and all are below `z_enter` on every signal:**
  - s4#m0: common mode, closed.
  - s3#m7: max z 2.12, on stomach_ref 2–50 Hz and ANT1 300–3000 Hz.
  - s3#m13: max z 2.52, on ANT1 10–150 Hz.
  - Both round-5 misses are in B t03 es2, and both peak on ANT1.
- **Subtraction (ruling 4 of 2026-09-29):**
  - It removes the linear leak everywhere (R² of T on the reference goes to 0.000), and injected differential spikes survive (1.000).
  - But **3 of 20 cuff-recordings pass** the event-firing test. It helps only where the leak was a large gain error.
  - In A t05 L it **raised** firing on events far from beats (0.019 → 0.081): one scalar fits a blend of two sources.
- **Part of the common-mode class is the heartbeat:** events fall within 5 ms of a beat 1.3–5× more often than chance.
- **HR:** 200 injected transients at the recording's own event amplitude cost 4–135 beats on a raw contact, and 0–2 on T.
- **Event rate:** it co-varies with breathing (4/9 recordings) and heart rate (6/9) at r ≈ 0.1–0.35, near zero lag, even after cardiac events are excluded. The modulation (20–40/min) is not the breathing rhythm (76–98/min).
- **Contact screen:**
  - It does **not** flag H t01 es2 L. L1 is low-gain (R² 0.67 against 0.98 for its peers) but correlated (r 0.70), so the left T is 75% common mode.
  - It flags B t02 1_3 L on L1, yet L1 carries the ground signal (R² 0.94) while L2 and L3 do not (0.04 and 0.01).
  - ANT1 is mains-dominated in B t01 3_2, B t02 3_3 and B t03 2_2.

**What the gate arithmetic says.**
- A one-sided Clopper-Pearson bound ≥ 0.98 needs 386 artifacts at 3 misses, 456 at 4, and 523 at 5: about 67 more per miss.
- At the pool's observed miss rate (3/208, 1.4%), a miss arrives about every 70 artifacts. So the bound creeps toward about 0.986 and clears only near ~1,200 artifacts, some 20 more rounds. Baseline alone is 104/107.

That is **not** a reason to relax the gate (see "Do not"). It is a reason to check that the gate is scored against what task 09 says it is. The revised pass condition reads "≥98% recall for artifacts **above each consumer's tolerance**", and the audit scores every mark. Every gate-evidence miss is below `z_enter` on every signal. If those misses are also below every consumer's tolerance, the audit is partly measuring recall of harmless events.

**Rulings:**

1. **The two round-5 misses.**
   - **Andrea classifies both** from the misses PDF: would she blank it?
     - Not-target: close it that way.
     - Target: measure consumer harm exactly as for s4#m0, using each consumer's own input and threshold over the mark: the spike detector's firing, and the HR beat train. If harmless, close it as `accepted_limitation`: "threshold class, recurring, measured harmless". If a consumer is harmed, **stop and report**. That is branch 1 of "If it fails", a diagnosable class.
   - **Diagnostic, not a fix:**
     - Is ANT1 mains-dominated in B t03 es2, as it is in B t03 2_2? A mains line sits inside 10–150 Hz and inflates that signal's baseline σ, which depresses z. That would be one cause behind both misses.
     - Report the misses' z on a notched ANT1.
     - A notch is a generator change: it resets eligible rounds through `fixed_at`, so it is not adopted without a ruling.
   - **Round 6 is drawn only after both misses are closed.**

2. **Measure the audit against the pass condition as written. Do not adopt it yet.**
   - Claude Code writes a per-consumer damage rule **before computing it on any mark**, blind to hit or miss. It must come from each consumer's own input and threshold:
     - spike consumer: the T it reads and the 4.5σ detector;
     - HR: the operational beat-train test;
     - stomach consumers: their own bands.
   - I ratify the rule and Andrea approves it. Only then is it applied to **all** pooled marks, hits and misses alike. The tolerance-filtered recall is reported beside the raw recall, and raw recall stays in every round report whatever is decided.
   - **Whether the gate moves to tolerance-filtered recall is Andrea's decision**, recorded before the round it first applies to is scored.

3. **Artifact or physiology? The lag histogram decides.**
   - For each cuff, histogram her detected spikes against lag from the nearest event: 0.1 ms bins, ±50 ms. Do this separately for near-beat and far-from-beat events, with the random-time control.
   - A peak confined to the event's own width (about ±1 ms) is electrical. No conduction or reflex coupling to a heartbeat or a muscle twitch is sub-millisecond.
   - A broad hump of tens of ms is physiology and is **never** masked.
   - Report per cuff. Rulings 4b and 4c apply only where the peak is narrow.

4. **Correct, then mask the residual.** Everything here lives in `derive/`, outside the hash.
   - **a. Keep the single-scalar subtraction only where it helps.** Keep it on a cuff only if it lowers event-locked firing in **both** the near-beat and the far-from-beat subsets. Otherwise the consumer reads uncorrected T. A correction that adds artifacts is worse than none (A t05 L).
   - **b. One extension, in the same pass: multi-regressor subtraction.**
     - Each outside channel is its own regressor, plus a beat-locked cardiac template.
     - Fit on one half of the recording, verify on the other.
     - Adopt it per cuff only if it passes the pre-declared verification: no significant excess firing on events, and injected spikes surviving within 5%.
   - **c. Cuffs that still fail get the pre-declared fallback, for the spike consumer only:** mask T for ±1.5 ms wherever T reaches 4σ within an event, **with exposure accounting**. Masked time is removed from the denominator of every rate, and from every phase bin of any rhythm-locked analysis.
     - This answers the objection in ruling 2 of 2026-09-29. Real spikes inside the mask are lost only in proportion to masked time, so rates stay unbiased. And a peak confined to ±1 ms is not physiology, so nothing rhythm-locked is removed.
     - Events co-vary with breathing and heart rate, i.e. with state. That is why exposure accounting is mandatory, not optional.
   - **Verify the mask:**
     - in unmasked time, firing on events is not above control;
     - injected differential spikes outside the mask survive within 5%;
     - injected spikes inside the mask are lost at the masked-time fraction.
   - **Report per cuff:** masked time %, spikes removed %, and the excess event-locked spikes as a fraction of the cuff's spikes, before and after.

5. **Trust in T for the spike consumer is decided by ruling 4's verification, not by contact heuristics.**
   - H t01 es2 L shows why: a low-gain contact passes the correlation rule while T is 75% common mode.
   - The contact screen remains for consumers that read raw contacts (HR, velocity), with two fixes for the new cohort:
     - **Missing ground signal:** with a shared ground and no local reference, every healthy single-ended contact must carry the ground-site signal. A contact with R² near zero against the outside reference is the suspect one, whatever the peer correlation says. In B t02 1_3 L that is **L2 and L3**, not L1. Report the L2–L3 correlation: a short between them would make them agree with each other and not with the ground. The whole left cuff of B t02 1_3 is distrusted for every consumer until this is explained.
     - **Low gain:** flag a contact whose R² sits well below its peers' (0.67 against 0.98), with a `PROVISIONAL_` threshold.
   - **ANT1** in B t01 3_2, B t02 3_3 and B t03 2_2: no consumer reads it. Detection keeps reading it. Invariant 43 forbids hiding a signal from detection alone; it does not forbid the reverse.

6. **HR: the operational injection test becomes task 05's transient veto.**
   - A channel is vetoed if 200 transients at the recording's own non-cardiac event amplitude change the beat train (lost beats, or a fiducial shifted by more than 2 ms) in more than 1% of injections: `PROVISIONAL_MAX_TRANSIENT_HARM = 0.01`.
   - Template SNR ranks only the survivors. SNR cannot see this harm, because the far-field ECG is common mode too.
   - **Measure, don't adopt:** does a minimum-QRS-width check in the beat detector remove the harm on raw contacts? A 1.2 ms transient cannot be a rat QRS. Adopting it is a separate ruling.
   - **H t01 3_3, every candidate vetoed:** raising is correct (task 05). Report whether the implausible intervals are short (extra beats) or long (missed beats), per channel. Do not move the threshold on one recording.

7. **The old-cohort QRS premise is withdrawn for the new cohort.**
   - "The QRS carries 0.00–0.54% of its energy above 300 Hz" was measured on an old-cohort tripole.
   - Re-measure it per new-cohort cuff, on T (corrected as in ruling 4) and on raw contacts.
   - Anything in tasks 02 and 07 that relied on the old figure uses the new-cohort number for the new cohort.

8. **Ratified:**
   - The width attribution. The real review load is the 12 over-cap candidates read by a consumer on a trusted contact (42 marks, 409 s).
   - The duration cap of 14.9 s, provisional.
   - The breathing and HR cross-correlation. The events co-vary with state and do not form a rhythm of analytic interest. No further analysis, beyond the exposure accounting in 4c.

### RULING 2026-09-30 (b) — the time mask is withdrawn; judge each spike by what the other channels saw

**Measured (Claude Code, gems `ce3bafa`, `fa5955c`; hash unchanged):**
- **Round-5 misses:** both harmless to the spike and HR consumers.
  - ANT1 in B t03 es2 is mains-dominated (91% of its power). Notching *lowers* the misses' z (2.52 → 0.58), so part of the in-mark elevation is the hum itself.
  - Both still await Andrea's classification.
- **Lag histograms:** 14 of 20 cuffs have a narrow (±1 ms) electrical core, 4–6× their flank on the worst. Humps appear only around near-beat events. Every cuff has a flat floor near events at 1–15× the global rate.
- **Excess event-locked spikes as a fraction of each cuff's spikes:** up to **0.62** (B t03 R), and 0.44 each on H t05 L and H t01 3_3 L. **On the worst new-cohort cuffs, a quarter to more than half of the detected spikes are electrical.**
- **The gated 4c mask fails its own premise:**
  - H t05 L loses 88% of its spikes against 2.1% effective exposure.
  - The gate (T ≥ 4σ at an event) selects exactly the events where T carries a spike-sized deflection, real or not. Her detector then pads every NaN run by 10 ms.
  - So the mask is placed *by the spikes* and removes them wholesale. Its "no firing on events in unmasked time" check cannot fail.
- **4b's check passed a correction that tripled spikes everywhere** (B t02 1_3 L, 3,684 → 9,413), because the control rate rose with them.
- **HR:**
  - The veto leaves A t02, H t01 es2 and H t05 with **no HR channel**.
  - The veto ran on task 05's detector, which is more robust than the consumer's own `findpeaks`. On B t02 3_3 RVN3 the consumer's own injection test lost 4 beats per 200.
  - A QRS-width floor does not help: a transient moves the fiducial within the beat wave.
  - H t01 3_3's implausible intervals are extra beats at about half the RR interval.
- **QRS energy above 300 Hz, new cohort:** median 0.24% on T. Tail up to 4.4% (A t05 R_T), and up to 15% on raw contacts.

**Rulings:**

1. **Withdraw ruling 4c (the gated time mask) and my claim that masked-time loss is proportional.** Both fail for the reason above. Any gate must be independent of the cuff under test, and a NaN mask must never reach her detector (its 10 ms pad multiplies it).

2. **Replace it with a per-spike outside-coincidence veto.** This is Andrea's proposal from 2026-09-29: tell leak from nerve spikes by their characteristics. Physiology cannot put one vagus's spike on the other cuff and the stomach at the same instant. **Decided per spike, after detection, in `derive/`:**
   - Run her detector on the T the consumer reads (4a/4b as ruled). For each detected spike, take the peak |outside reference| (300–3000 Hz, channels outside this cuff) within ±w of the spike time, in units of that reference's robust σ.
   - **w** is the cuff's measured core half-width from the lag histogram, rounded up to 0.1 ms. Veto the spike if the peak exceeds **θ**.
   - **θ is set on the first half** so that at random times the veto fires at most 2% (the chance loss). It is applied to the whole recording, and the chance loss is reported on the second half.
   - Nothing is NaN'd. Exposure is unchanged. Rates are divided by (1 − chance loss), with the chance loss reported per event-rate tertile, because state moves the outside noise.
   - **Verification, pre-declared, and independent of the veto:**
     - (i) **Left–right spike cross-correlogram** per recording, 0.1 ms bins over ±50 ms. Two vagi cannot fire together to sub-millisecond precision, so a narrow zero-lag peak is leak. Pass: no zero-lag core above flank after the veto, where there was one before. This uses neither the events nor the reference.
     - (ii) Injected differential spikes at random times are lost at the chance-loss rate (±1%) on the second half.
     - (iii) The surviving spikes' lag histogram around events has no narrow core.
   - **Report per cuff:** spikes before and after, fraction vetoed, chance loss, and (i)–(iii).
   - **Also report the floor.** After the veto, does the 1–15× floor near events remain?
     - If it drops, it was sub-event leak (muscle activity below the event finder's size rule), not state.
     - If it remains, it is state co-variation and stays.

3. **4b's check is amended:** a correction is adopted only if the spike rate more than 50 ms from any event does not rise by more than 5%. A correction that adds spikes in clean time is adding noise. B t02 1_3 L is not adopted (it is distrusted anyway). A t05 fails survival and is not adopted.

4. **Cardiac humps are not physiology until shown to be.** My ruling 3 ("broad hump = physiology, never mask") was too quick. A rat QRS lasts ~10–20 ms, so QRS leak on T is itself a broad hump. Cardiac-locked vagal afferent firing follows the pressure pulse, tens of ms *after* the R-wave, and is not centred on the QRS.
   - Histogram spikes against lag from the **R-peak** (not the event) on the four hump cuffs, before and after ruling 2.
   - A hump centred within the QRS that ruling 2 removes is leak.
   - A hump within the QRS that survives goes to the task 02/13 peri-R route, measured per new-cohort cuff by crossing rate per lag bin (task 02's method, now needed because of ruling 7).
   - A hump after the QRS is physiology and is left alone.

5. **HR:**
   - **The veto runs on the detector that produces the beats the analysis uses.** If Andrea's HRV analysis uses its own `findpeaks` on the chosen channel, the binding veto uses the validated replica (99.7–100% match). Report task 05's detector beside it.
   - **The three recordings with no channel.** Measure one route: two-pass bridging.
     - Pass 1 detects beats.
     - Events within 5 ms of a pass-1 beat are cardiac and are *not* bridged.
     - The remaining events are bridged ±1.5 ms by linear interpolation on the raw contact.
     - Pass 2 detects the beats.
   - Judge the route by the same injection veto, run through both passes. Adopt it per recording only if it passes. Until then these recordings have no HR, and raising is correct.
   - **The QRS-width floor is rejected.**
   - **H t01 3_3:** are the extra mid-cycle peaks locked to stimulation pulses (present only in stim periods) or to the cardiac cycle (a T-wave, present throughout)? Report only; no threshold moves.

6. **`stomach_ref` in the three ANT1-hum recordings:** measure the harm first, and don't touch `derivations.py`. Compare each stomach consumer's output on `stomach_ref` as-is against an alternative built outside the hash that excludes ANT1. If the outputs agree within the consumer's tolerance, no action. If not, the stomach consumers read the alternative in those recordings (invariant 43 permits distrusting a derived quantity).

7. **Ratified:**
   - The contact-screen fixes (B t02 1_3 L: L2 and L3 shorted or locally driven, whole cuff distrusted; H t01 es2 L1 low-gain).
   - The revised lag-histogram rule, with its clean controls.
   - The sub-sample cardiac template inside 4b: CLAUDE.md's "do not re-implement" line is narrowed to allow it under the three stated conditions.
   - The finder and veto modules outside the hash.
   - The non-cardiac-only gains for injection.

8. **Damage rule (ruling 2 of 2026-09-30): ratified with amendments.** Andrea approves before it is applied.
   - **Spike consumer:** the test is **two-sided**. An artifact that suppresses the detector (σ inflation, clipping) damages as much as one that adds spikes; s3#m7 had 0 spikes where 2.5 were expected. A mark is damaging on a cuff at p < **0.05** (not 0.005), on the consumer's input as routed at ratification. Counting a mark as damaging is the conservative direction: it keeps the mark in the denominator. A strict threshold on a sub-second Poisson count has almost no power and would label nearly everything harmless, which inflates the filtered recall.
   - **HR:** damaging if any RR interval touching the mark deviates more than 20% from the local median. The ±1-beat count rule may add damage but never clear it (one missed beat is a doubled RR, which HRV cannot absorb).
   - **Slow wave:** a 1 s peak shift, *or* a cycle added or lost, is damage.
   - **Unassessable → target:** ratified. A recording with no HR channel is unassessable for HR.
   - **Velocity left out:** ratified. Consequence, written down: the filtered recall says nothing about velocity, so task 18 must handle its own artifacts (zero-lag rejection) or be re-scored when it is built.
   - **The routing table the rule reads is frozen at ratification.** A later routing change means recomputing it for every mark.

**Order.** Andrea classifies the two misses. If she draws round 6, Claude Code stops Drive and MATLAB work and builds rulings 2–5 against synthetic data only. The real-data runs follow her "done", in this order: 2, 3, 4, 5, 6, then the damage rule on all pooled marks once Andrea approves it.

**Andrea, 2026-09-30:**
- **s3#m7 and s3#m13: "can be ignored"** — close both as `not_target` (her classification; committed marks and the round-5 score stand). Both also measured harmless. Round 6 may be drawn once `--check-next` returns ALLOWED.
- **The damage rule is approved as amended.** Apply it after the real-data runs, in the order given.
- **Beats are computed once.** Her HRV analysis has its own `findpeaks`. She does not want double computation if the pipeline's beats are reliable. So:
  - **One detector, and the veto tests it.** The pipeline's beat detector (task 05's, the more robust of the two) is the only beat computation, and ruling 5's transient veto runs on it. The replica is kept for the comparison below, then retired.
  - **Beats are stored in her analysis's own format**: the same variable names, units, and sample or time base her function builds today, per recording, beside her existing outputs. Claude Code reads the format from her code and does not invent one.
  - **A new post-processing-only version of her function** reads the stored beats and skips peak finding. Her original function stays untouched, beside it.
  - **Reliability is shown before the switch.** Feed the new function her own `findpeaks` beats converted to the stored format: its HRV outputs must equal the original's exactly. Then compare pipeline beats with her `findpeaks` beats per recording (match %, and where they disagree, which one the injection test and the RR plausibility favour).
  - **She switches when she has seen that comparison.** For recordings with no vetted channel, nothing is stored and the function says so. It does not fall back to its own peak finding.

### RULING 2026-09-30 (c) — veto ratified; distrust where it fails; HR needs an independent count check; notch mmc's input

**Measured (gems `d346584`, `dd5b062`; hash unchanged):**
- **The veto on 11 cuffs.** It rejects **22–74%** of each cuff's detected spikes. Its chance loss is 0.9–7.2% on the second half, and up to 10.7% in the busiest tertile.
  - (ii) passes on all 11 at 3,000 injections.
  - (iii) fails on H t01 es2 L (event core ×3.0 remains, w = 0.2 ms).
  - (i) The left–right zero-lag core clears in A t01, A t02 and H t01 3_3, but **persists in A t05** (×4.0 → ×4.3).
  - The floor near events mostly drops, so it was sub-event leak. Part of it remains on four cuffs: that is state.
- **Humps.** All four QRS-centred humps survive the veto. H t05 L has a further component peaking at +14.5 ms after the R.
- **HR detectors.** Task 05's detector and her `findpeaks` each win on some recordings.
  - Task 05 misses 12–43% of beats on A t05 and B t02 1_3, locks onto every other beat on A t02, and picks a **one-beat** channel on A t01 that passes every gate.
  - `findpeaks` adds false beats on B t03 and B t02 3_3.
  - Two-pass bridging fails everywhere.
  - H t01 3_3's extra peaks sit at 0.50 RR and are cardiac-locked, not stim-locked.
- **The post-processing-only HRV function** (`HR_BR_HRVAnalysis_beats.m`) is identical to the original on 9/9 recordings when fed her own beats.
- **stomach_ref in the hum recordings.** slow_wave is untouched by the hum (100% of peaks match with ANT1 notched). mmc is changed by it (Jaccard 0.18 and 0.25 in B t01 3_2 and B t02 3_3; 0.82 in B t03).
  - The no-ANT1 alternative changes both consumers far more (Jaccard 0.16–0.20), because it is a different derivation.
- **Damage rule on all 208 marks.** 180 are target.
  - Filtered recall is **179/180, lower bound 0.974**, against 205/208 (0.959 raw: the span bootstrap is the more conservative; Clopper-Pearson alone is 0.963).
  - The only target miss is s4#m0, flagged by a spike deficit (p = 0.033).
  - The round-5 recordings had no routing entry and were judged on uncorrected T without the veto.

**Rulings:**

1. **The veto is ratified as the spike consumer's leak rejection**, with these amendments:
   - **θ stays fixed per cuff**, as calibrated. It is a physical criterion and must not be loosened in busy periods.
   - **Rates are corrected by the per-tertile chance loss.** Measure it at ≥ 10,000 random times over the whole recording. (ii) showed that injected loss tracks it, so the correction is valid.
   - **Report the corrected rate beside the raw count everywhere.** Any analysis resolved by state must use the loss of the tertile it sits in.

2. **Where verification fails, the cuff is distrusted for the spike consumer in that recording** (ruling (b) 5: trust follows verification).
   - **H t01 es2 L** fails (iii). It is also the low-gain cuff. Distrusted.
   - **A t05:** first run (i) again, restricted to spikes outside the peri-R window measured under ruling 3.
     - If the zero-lag core vanishes, the residual synchrony was QRS leak on both cuffs, and the peri-R route handles it.
     - If it persists, distrust both A t05 cuffs for the spike consumer.

3. **The QRS humps go to the peri-R route (tasks 02 and 13), measured per new-cohort cuff.**
   - **Extent:** the crossing rate per lag bin around the R-peak, after the veto, on the routed T. The extent is where it exceeds the flank.
   - **It is a time mask on R-peak lags, so it is independent of this cuff's spikes.** Exposure is removed and rates are corrected. Analyses locked to the cardiac phase report the blanked phase as not measured; they do not interpolate it.
   - **R-peaks come from the recording's vetted beat source (ruling 4)**, never from a detector that halves or misses beats.
   - **H t05 L's +14.5 ms component is unresolved. It is not physiology by default.**
     - +14.5 ms is too early for pressure-driven afferent firing (tens of ms after the R, following the pulse) and is within reach of the QRS tail.
     - Report its lag and width against the QRS template's > 300 Hz envelope on that cuff.
     - If it sits under the envelope, it is leak and the extent covers it. If it is clear of the envelope, it is left alone.

4. **HR: a beat train is stored only if it passes an independent count check.**
   - Task 05 lists beat count as "a sanity print only". That is why a one-beat channel passed. The 2026-09-23 header already asked for beat-count consensus.
   - **The independent reference: a per-minute heart rate from the autocorrelation** of the rectified 10–150 Hz signal, searched only over rat heart rates (250–550 bpm). It uses no peak picking, so it cannot share a detector's errors.
   - **Count gate** (`PROVISIONAL_MAX_COUNT_DEV = 0.05`): a beat train passes if its beats per minute are within 5% of the autocorrelation rate in ≥ 95% of minutes where the autocorrelation peak is clear. It must also pass the transient veto and task 05's plausibility gates.
     - Every-other-beat (50%), 12–43% misses, +7% false beats and a single beat all fail it.
     - If the autocorrelation is not clear in most minutes, the recording is unassessable and nothing is stored.
   - **One computation, two candidates.** Per channel, the pipeline computes both candidate trains (task 05's detector and the `findpeaks` replica), gates both, and stores one:
     - the passing train with the best template SNR;
     - on a tie, the one closest to the autocorrelation count.
     
     Her function only post-processes what is stored. This satisfies Andrea's "compute once": nothing is recomputed downstream.
   - **A t01:** nothing is stored until a train passes.
   - **A t02, H t01 es2 and H t05:** re-run under this gate with both candidates. Bridging is dropped.
   - **H t01 3_3:** measure, without adopting, a refractory relative to the running median RR, rejecting any peak within 0.6 × median RR of the previous one and keeping the one that matches the template. Check it against the autocorrelation count. A heart does not halve its interval in one beat, but it can speed up quickly, so the rule must not remove real beats during rate rises.
   - Andrea switches to `HR_BR_HRVAnalysis_beats.m` only after seeing the gated comparison.

5. **stomach_ref in the three ANT1-hum recordings: stomach consumers read stomach_ref built from notched ANT1**, derived outside the hash. They do not read the no-ANT1 alternative.
   - The notch removes the hum and keeps the derivation, which is the only change warranted.
   - It applies to all three recordings, including B t03 (0.82), so that no tolerance judgement is needed.
   - slow_wave is unchanged by it (100% match), so the rule is uniform at no cost.
   - Report whether a 60 Hz-only notch (Andrea's practice) suffices, or whether harmonics are needed.
   - Detection keeps un-notched ANT1 (invariant 43).

6. **The damage rule's routing must be complete.**
   - Compute the round-5 recordings' routes by the same mechanical rules (4a, 4b, veto, distrust).
   - Recompute only those marks. The rule itself is unchanged.
   - Report the result beside the current one. Both are recorded.

7. **The gate question is put to Andrea.** Raw recall stays in every report whatever she decides.
   - **Filtered recall at 1 target miss needs 236 target marks** for a lower bound ≥ 0.98, about 56 more at zero further target misses: one or two rounds. With one more target miss, 313.
   - **Raw recall at 3 misses needs 386**, and at the observed miss rate about 1,200.

**Andrea, 2026-09-30: the gate is filtered recall.** From round 6 on, task 09's gate is a one-sided 95% lower bound ≥ 0.98 (the more conservative of Clopper-Pearson and span bootstrap) over marks the ratified damage rule calls **target**. This is task 09's written pass condition: "above each consumer's tolerance".
- The other conditions are unchanged: ≥ 3 eligible spans per condition, and a positive pooled chance margin, computed over target marks.
- Raw recall and its bound are reported beside it in every round.
- The rule and its routing are frozen as ratified. A change to either means recomputing every pooled mark, and it is reported as a change.

**Addendum to ruling (c), 2026-09-30 — answers to the build's questions:**

1. **The peri-R mask may use "mask-grade" beats.** A train that passes the count gate but fails the transient veto is allowed for the peri-R time mask only.
   - It is stored under a separate name, and `HR_BR_HRVAnalysis_beats.m` never reads it. HRV still needs a fully vetted train.
   - The veto protects fiducial precision, which HRV needs. The mask needs every beat to be present, which the count gate checks.
   - Fiducial jitter is self-correcting here: the extent is measured from the crossing-rate histogram against that same train, so jitter widens the measured hump and the extent with it.
   - Run ruling 3 on A t02 L, H t01 3_3 L and H t05 L with these trains, and place H t05 L's +14.5 ms component.
   - A recording with no train passing the count gate gets no peri-R route. Its spike-consumer cuffs with a QRS hump are distrusted.
2. **Recompute every pooled mark once routing settles.** Ruling (c) changed routing (A t05 and H t01 es2 L distrusted, new HR trains, notched stomach_ref), so Andrea's clause applies: ruling 6's "only those marks" is superseded.
   - When items 3–6 are complete, freeze the routing table and record its hash.
   - Recompute the damage rule on all pooled marks and report the old and new classes side by side, with the reasons for every mark that changes class.
   - Round 6 is scored on that frozen table.
3. **An input the routing excludes counts as unassessable, therefore target.** This covers a distrusted cuff, a recording with no HR train, and any consumer with no admissible input.
   - It is the convention already ratified for missing HR channels. The gate must not become easier to pass by excluding data.
4. **Test skips.** List the 23 extra skips in the clean worktree with their reasons. A skip other than "needs local real data that is not in the repo" is a defect: fix it.
5. **stomach_ref's 60 Hz residual** (about 1,500× the floor in B t01 and B t02 3_3, after the ANT1 notch): report which inputs carry it. Compare mmc with 60 Hz notched on every stomach_ref input against ANT1 alone. If mmc differs, the notch applies to all inputs in those recordings.

### RULING 2026-09-30 (d) — the frozen table and the recompute: addendum 3 corrected; round 6 appends

**Measured (gems `c53a2ba`; hash unchanged):**
- **Frozen routing table:** `46ce9b0b0081…`, stored at `labels/blind_audit_routing/routing_46ce9b0b00819823.json`.
- **Under addendum 3, all 208 pooled marks are target.** 28 changed class, and 29 are target *only* because an input is excluded. Filtered recall therefore equals raw: 205/208, lower bound 0.959.
- **s3#m7 and s3#m13** had been measured harmless on B t03 es2 R_T. They became target misses only because R was then distrusted by check (iii).
- **Round-5 routes:** no cuff adopts multi (A t01 1_2 L: 147 → 28,736 spikes, rejected by the amended 4b). B t03 es2 R is distrusted by check (iii). Four hump cuffs are distrusted for lack of a count-gated train.
- **Spike consumer, rounds 3–5:** about 11 of 30 cuff-recordings are now distrusted.
- **H t05 L:** against a count-correct train there is no QRS-centred hump. There is a component at +14.25 ms, 0.5 ms wide, outside the > 300 Hz QRS envelope (−5.0 to +2.3 ms).
- **B t01 3_2:** notching all stomach_ref inputs exposes real mmc damage on 3 marks (the hum had inflated the MAD).
- **Skips:** the 23 extra clean-worktree skips were a test predicate defect, now fixed (clean worktree 1137 passed / 7 skipped, the same as local).

**Rulings:**

1. **Addendum 3 was wrong. It is replaced.** It contradicted task 09's written pass condition, "above each consumer's tolerance". An input no consumer reads cannot damage a consumer. Its visible failure: two marks measured harmless became "damaging" because we stopped reading the cuff they were measured on.

   **Andrea decided (2026-09-30): only consumers that run count.**
   - **A consumer's input that the routing excludes contributes nothing** to a mark's class. This covers a distrusted cuff for the spike consumer, and HR in a recording with no HRV-grade train.
   - **Unassessable still means target.** If a consumer does read the input but cannot judge the mark (for example, fewer than 10 local beats), the mark is target. The same holds if no running consumer can judge the mark at all.
   - This also supersedes the clause in ruling (b) 8 that made a recording with no HR channel unassessable for HR.
   - **This correction was made knowing its effect** (about 178/179 instead of 205/208). So both classifications are computed and reported every round: "consumers that run" is the gate, and "excluded = target" is reported beside it with raw recall. Neither report is ever dropped.
   - A later change to either convention needs Andrea's decision, recorded before the next score.

2. **Round 6 and later rounds append to the frozen table.**
   - A new recording's routes are computed by the same mechanical rules and **appended**. Existing entries must stay byte-identical.
   - Store the table as per-recording entries, each with its own hash. The score records the old table hash, the new one, and a check that every old entry hash is unchanged.
   - That counts as "scored on the frozen table". Any change to an existing entry is a routing change: recompute every pooled mark and report it as a change.

3. **Sub-millisecond R-locked components are electrical, whatever their lag.** This is the same reasoning as ruling (a) 3: no neural or reflex chain holds 0.5 ms precision.
   - H t05 L's +14.25 ms component goes into its peri-R extent.
   - The QRS-envelope test missed it because the envelope comes from the averaged template's energy, where a small late high-frequency feature falls below threshold.
   - Rule: an R-locked excess narrower than 1 ms at half height is included in the peri-R extent. Broader components follow ruling (c) 3.

4. **Ratified:**
   - mask-grade beats (`maskBeatlocs`), with their guards;
   - the round-5 routes;
   - the all-inputs notch in B t01 3_2 and B t02 3_3, and ANT1 only in B t03 2_2;
   - `frozen_filtered_gate`;
   - the skip fix;
   - the peri-R extents for A t01 L (−9.5 to −4.5 ms) and B t03 es2 L (−2.0 to 0 ms).

5. **Recompute now under ruling 1 on the same routing.** The routing is unchanged, so the table hash stays `46ce9b0b…`.
   - Report both classifications per round and pooled, the gate numbers under each, and the marks whose class differs between them, with reasons.
   - Store the "consumers that run" classes as the gate classes under that hash.

6. **Report the spike-consumer coverage.** Give a table of every cuff-recording in rounds 3–5 marked trusted or distrusted, with the reason, and the share of spike-consumer time lost to distrust.
   - Most distrust now comes from missing count-gated HR trains (hump cuffs) and from failed check (iii).
   - Better HR trains would recover data. That is the next lever, and it is Andrea's priority call.

### RULING 2026-09-30 (e) — (d) 3 corrected; HR trains are the next lever, diagnosed before designed

**Measured (gems `835cc84`; hash unchanged):**
- **Gate ("consumers that run"), pooled:** 178/179, lower bound **0.974** (Clopper-Pearson; bootstrap 0.980). Chance margin +0.142. 8 baseline and 7 stim/recovery spans. Not cleared; the only target miss is s4#m0.
- **The other two numbers:** "excluded = target" and raw are both 205/208 (0.959). The 29 marks that differ between the classifications are all target only because an input is excluded.
- **Routing:** per-entry hashes, append-only, and three-number reports are in place.
- **H t05 L's "+14.25 ms component" is 4–5 spikes of 281** (p = 0.017 before correction for the search across lags). It fails the p < 1e-3 bar used for every other R-locked test. The "0.5 ms width" was the bin size. Ruling (d) 3 rested on my misreading of the earlier report.
- **Spike-consumer coverage, rounds 3–5:** 11 of 30 cuff-recordings are distrusted, **36.4% of spike-consumer time**. 6 of them (24.9% of time) are hump cuffs with no count-gated HR train. A t01 1_2 and H t03 each miss the count gate by one minute (94.7%).

**Rulings:**

1. **(d) 3 is corrected.** A narrow R-locked excess is electrical only if it is also significant at the bar every R-locked test uses: p < 1e-3 after correction for the search across lags.
   - Width is measured at half height at 0.1 ms bins, never read off the bin size.
   - H t05 L's component does not qualify, so its entry reverts to the `46ce9b0b` content.
   - That is a routing change: append the corrected entry and recompute every pooled mark. No class change is expected, since `ca04a0f9` changed none.
   - The general rule stands for any component that does qualify.

2. **The count-gate thresholds do not move to rescue A t01 1_2 or H t03.** Loosening a threshold after seeing which recordings miss it by one minute is fitting.
   - `PROVISIONAL_MAX_COUNT_DEV` and the 95% bar get their permanent values in build-order step 9, from the cross-animal set, like task 05's other provisional thresholds.

3. **HR trains are the next lever. Diagnose them before designing anything.** For every recording in rounds 3–5 without an HRV-grade train, report per channel, per detector:
   - the minutes that fail the count gate;
   - for each failing minute, why: missed beats, extra beats (and whether they sit at about 0.5 RR), transient hijacks, or an autocorrelation reference that is not clear. A minute whose reference is not clear should have been excluded as unassessable; check that it was;
   - the veto harm, and its mechanism.

4. **Constraints for any detector change that follows. It is proposed to me first, not adopted.**
   - **Andrea's rule for missed beats:** re-search near the expected time with relaxed width and height tolerances, so the fiducial is measured. Never insert a beat at the expected time.
   - **Extra beats from a second mid-cycle peak** (H t01 3_3): resolve within a refractory relative to the running median RR by template match, not by amplitude. The refractory must not remove real beats during rate rises. Check it with the count gate, which uses the independent autocorrelation.
   - **Design out-of-sample:** develop on some animals and judge on held-out animals. Report the held-out result as the evidence.
   - **The same gates apply unchanged:** count gate, transient veto and task 05 plausibility.

5. **Priority is Andrea's call. The default is to run in parallel.** Andrea draws audit rounds. Claude Code does ruling 3's diagnosis on locally cached data and synthetic data while she labels, and real-data runs follow her "done".

### RULING 2026-09-30 (f) — the HR candidate detector: ratified with amendments

**Measured (diagnosis on 9 recordings, 216 trains):**
- The count gate reproduces exactly. 532 unclear minutes were excluded as ruled.
- **Capture, not loss, is the veto harm on raw contacts.** The injected transient itself becomes the beat in 50–94% of harmed injections. This alone blocks A t01, H t05 and H t01 es2.
- After 10–150 Hz filtering, a captured transient's width falls inside that channel's beat-width range on all nine channels (10.6–21.6 ms on four; about 150–190 ms and about 7 ms on the others, matching their beats). So **width cannot separate it** from a beat. A width floor changed veto harm by no more than ±0.005.
- **Most findpeaks extras sit at its 100 ms spacing floor** (RR median 101–116 ms, phase 0.60–0.70 RR). A true 0.4–0.6 RR mid-cycle peak is a minority, except on H t01 3_3, A t02 and H t09.
- Task 05 misses beats on A t02 and H t01 3_3.
- Tripoles have almost no harm but a weak ECG, so most of their minutes are unclear.
- The autocorrelation reference is wrong in 2 minutes, both in H t03.
- One read-only Drive pass was made with no round open. That is within the rule: the constraint is no Drive or MATLAB while a round is open.

**The design in the build's report is ratified: a new module outside the hash, one extra candidate train per channel, the same unchanged gates. These amendments apply:**

1. **The template is the median across clean beats**, where clean means RR within 10% of the running median. Do **not** exclude beats that have an event within 20 ms: on channels where the QRS edge itself triggers the event finder, that would exclude most beats. The median is robust to the occasional transient.

2. **Judge suspect candidates by masked correlation.**
   - A candidate within 2 ms of a common-mode event is scored by template correlation with the event's ±2 ms **excised from both** candidate and template.
   - A real beat hit by a transient keeps most of its QRS and scores well. A captured transient with no QRS under it has little left and scores poorly.
   - The excision is for scoring only. The output signal is untouched (invariant 8).
   - **Fiducial of a kept suspect beat:** taken from masked template alignment, expressed as the template's peak offset, so it means the same thing as a clean beat's peak.
   - Report, on clean beats, how far alignment-based fiducials sit from peak-based ones. Any systematic offset must be under 0.5 ms, or mixed fiducials will add HRV jitter.

3. **The correlation floor is a rule fixed now, not a number tuned on A/B.** Answer to (b): per channel, the floor is the **1st percentile of that channel's own clean-beat template correlations**. H applies the same rule to its own clean beats. The quantity that protects against capture is never tuned across animals.

4. **The refractory fraction is a development parameter, capped at 0.8.**
   - The 0.6 used in the earlier measurement cannot resolve the dominant extras. At RR ≈ 150–165 ms, 0.6 × RR = 90–99 ms, below the 101–116 ms extras.
   - Choose k on A/B from {0.6, 0.65, 0.7, 0.75, 0.8} and fix it before H runs. Real beats during rate rises stay above ~0.8 × median.
   - The rate-rise check is the pass/fail test: the count gate in every minute where the autocorrelation rate rises > 10%.

5. **Re-searched beats must pass the same correlation floor** (rule 3). Only height and width are relaxed.
   - The height and width multipliers are development parameters, fixed on A/B before H.
   - Nothing is inserted and gaps stay tagged (Andrea's rule, invariant 8).

6. **Holding out H whole is accepted.** Answer to (a).
   - Capture is present in the development set (A t01), so the capture mechanism is not first met in the held-out set.
   - Holding out one animal is thin evidence. So every recording from round 6 on, gated by the frozen detector, is reported as further held-out evidence, per animal.

7. **Written down before H runs:** k, the height and width multipliers, the template and floor rules, and the code commit.
   - The H run happens once. If it fails, report it. Do not tune and re-run on H.

8. **Report the regression set (B's five recordings):** stored trains reproduced (≥ 99% of beats within 2 ms) or still passing every gate. Also report the beats changed and why.

### RULING 2026-09-30 (g) — HR candidate detector: synthetic development amendments

**Measured, on synthetic fixtures only (`make_hr_trouble`, matched to the diagnosis; no real data):**
- The fixture reproduces every measured failure of the existing detectors: findpeaks recall 0.77 with 201 extras under capture, extras at the spacing floor, task 05 misses.
- Ruling (f) as written fails in four ways:
  - **10–150 Hz masked scoring:** 0/71 beats hit by a transient were kept.
  - **Excising only the suspect's own event:** real beats 25–80 ms from a transient were lost, and their gaps were searched but not filled.
  - **The 1st-percentile floor on the (f) 1 clean set:** captures make up 1.9% of it, the floor collapses, and 40–70% of captures are kept.
  - **Periodic secondary peaks halve the train's running median**, so the refractory removes nothing at any k.
- Fiducial alignment offset: 0.002–0.035 ms.

**Rulings.** These are design corrections made on synthetic data before any real-data development. The A/B → single-H protocol of ruling (f) 6–7 is unchanged.

1. **Score in 10–900 Hz. Fiducials stay 10–150 Hz peaks.** The template and every correlation are computed in 10–900 Hz, where a 1.2 ms transient stays compact enough to excise (64/71 hit beats kept).

2. **Excise every event in the ±40 ms window, for candidates and clean beats alike**, so the floor is made exactly as the score is.
   - If excision leaves under 50% of the window, the score is undefined. The candidate is not kept on score: it is dropped, its gap is re-searched, and the number of such candidates is reported.

3. **The clean set and floor come from an independent reference.** This replaces (f) 1's running-median clean definition and refines (f) 3.
   - **Clean** means both adjacent RRs are within 10% of the per-minute **autocorrelation RR**, in clear minutes only.
   - **The floor** is the 1st percentile of masked correlations over clean beats that are **not suspect**. A capture is suspect by construction, so impostors are excluded rather than diluted.
   - **With fewer than 50 non-suspect clean beats, this detector yields no train on that channel.** There is no fallback to the suspect-contaminated set.

4. **Periodic secondary peaks: option (b), with (c) as the fallback.**
   - **(b) Seed.** The template is seeded from task 05's beats, provided task 05's train has ≤ 25% short intervals (< 0.75 × the autocorrelation RR). The seed only needs to be pure, not complete, so task 05's missed beats do not matter here. The clean and floor rules of ruling 3 then apply.
   - **(c) Split.** Otherwise, split the candidates into the two interleaved subtrains and keep the one with the higher template SNR (task 05's own criterion, a shape decision).
     - Report both subtrains' SNR and their fiducial jitter (SD of the RR difference between the two).
     - A cardiac-locked secondary wave can also be reproducible, so if the two SNRs are within 20% of each other, the split is ambiguous and the channel yields no train.
   - **The refractory's running median is taken over template-matching beats** (at or above the unmasked floor), so extras cannot halve it. k stays a development parameter in {0.6, …, 0.8}.

5. **Unchanged:**
   - the count gate, the transient veto and task 05's plausibility gates;
   - no insertion (invariant 8);
   - the relaxed height and width multipliers and k are fixed on A/B before H;
   - H runs once;
   - every recording from round 6 on is further held-out evidence.

6. **Order:**
   - Finish the module with amendments 1–4.
   - Synthetic tests, with a fixture case per failure mode, including a mid-cycle wave with SNR comparable to the QRS for the ambiguity rule. Then mutants.
   - Commit, then A/B development (no round open), then the H run.

### RULING 2026-09-30 (h) — HR template detector: interpretations ratified; the floor applies to every candidate

**Measured (gems `e48667e`, synthetic only; hash unchanged; not wired into `gated_selection`):**
- **The template train beats findpeaks and task 05** on capture (0.954 against 0.891 at 0.5/s; 0.797 against 0.650 at 2/s, with 34 extras against 517), on floor extras (1.000/0), on regular mid-cycle peaks (split), and on rate rises (refractory removes 0).
- **It correctly yields no train** where the autocorrelation is never clear (capture 6/s), where the split is ambiguous, or where there are fewer than 50 non-suspect clean beats.
- **Not solved:**
  - task 05's misses (0.948, 60 extras);
  - irregular mid-cycle peaks on 25% of cycles (ambiguous split);
  - missing beats whose slots are taken by noise peaks (17).
  
  All three share one cause: **non-suspect candidates never face the floor.**
- **Fixture bug found and fixed:** transients were packed into the first 45 s.

**Rulings:**

1. **The three interpretations are ratified:**
   - "Template-matching" uses the floor built like the beat's own score (masked if an event is in its window, unmasked otherwise).
   - The running median leaves out intervals ≥ 1.5 × the minute's autocorrelation RR.
   - No clear minute means no train, reported as its own reason.

2. **The floor applies to every candidate, suspect or not.** A candidate below its floor is dropped, and its gap is re-searched under the same floor.
   - **The price** is that about 1% of real beats fall below a 1st-percentile floor by construction. These are the channel's worst-shaped beats, often ectopic or noise-corrupted, which HRV practice (NN intervals) excludes anyway.
   - **The trade:** a false beat splits one RR into two short ones and corrupts HRV. A tagged gap can be excluded.
   - Report the fraction of real beats dropped per fixture, and re-measure every failure mode, the three unsolved ones in particular.

3. **Gaps must reach the analysis.** Every dropped or unfilled beat leaves a tagged gap. The stored beats file must carry those tags, for example a `gapAfter` logical vector the length of `heartlocs`, or a list of excluded intervals. Otherwise Andrea's function reads a gap as one long real RR interval.
   - Report how `HR_BR_HRVAnalysis_new.m` handles RR outliers today (any range or ratio rejection) and whether that already excludes gap-spanning intervals.
   - If it does not, propose an **optional** gap input to `HR_BR_HRVAnalysis_beats.m`. With no tags given, outputs stay identical, and the identity proof still holds.
   - Changing her analysis's behaviour is **Andrea's decision**. Report first; do not build it.

4. **Then A/B development, stopping before H.**
   - After the synthetic re-measure passes, run A/B development while no round is open, and choose k and the height and width multipliers.
   - **Report** the A/B results, the frozen parameter set and the commit, then stop. The single H run follows my go-ahead.

**Andrea, 2026-09-30: add `GapAfter`.**
- `HR_BR_HRVAnalysis_beats.m` gets an optional name-value input, `'GapAfter'`: a logical vector the length of `heartlocs`.
- An RR interval whose start beat is tagged is masked exactly as her [100, 500] ms rule masks an implausible interval, and it is treated as a run boundary.
- With `'GapAfter'` absent, outputs are identical to the original, and the identity proof is re-run to show it.
- Add a test on a stored train with known gaps, checking the masked intervals and that no successive difference crosses a gap.
- `HR_BR_HRVAnalysis_new.m` stays untouched. processing_new stays local and uncommitted until Andrea says otherwise.
- **Found on the way:** her 100–500 ms rule passes both a missed-beat interval (about 310–380 ms) and findpeaks' floor extras (about 101–116 ms). So HRV computed from her own `findpeaks` beats today counts both as real intervals. Report, per recording already analysed, how many intervals fall into each class (for her information only; no re-analysis unless she asks).

### RULING 2026-10-01 — A/B shows no gain: H is not run; diagnose, and measure tagging instead of vetoing

**Measured (gems `a228a4a`; 10 A/B recordings, 30-point grid, pre-registered rule):**
- **The template detector gives none of the four development recordings without an HRV-grade train a train, at any combination.**
  - At the chosen point, 48 of 120 channels have fewer than 50 non-suspect clean beats (41 have zero), 11 split ambiguously, 47 fail the transient veto, and 9 fail only the count gate.
  - The closest misses fail the veto: A t01 RVN1 matches the count in 100% of minutes but has harm 0.18. A t02 RVN2 is at 74% with harm 0.43.
- **The no-floor channels are not starved by suspects** (only 1–16% of beats are within 2 ms of an event). Why the clean set is empty was not recorded.
- **Fiducial alignment on real clean beats:** −0.21 to +0.02 ms.
- **The parameter set** (k 0.7, height 0.25, width 0.5) was chosen by the third tie-break, with zero gains.
- **Regression set:** unchanged except B t01, where a template L_T train (SNR 761, 98.0% of stored beats within 2 ms) would replace the stored one.
- **GapAfter is done** (local in processing_new): identity 9/9, the known-gaps test passes, misspelt names are refused, and the nargin fix now lives in the generator.
- **Her own findpeaks beats:** floor extras are 0.2–14% of the intervals her 100–500 ms rule accepts. Missed-beat intervals are near zero.

**Rulings:**

1. **H is not run.** A held-out test of a detector that gained nothing on its development set measures nothing and would spend the held-out animal.
   - The frozen parameter set is void.
   - The template detector stays out of `gated_selection`. No stored train changes, B t01 included.
   - H remains unspent for a future design.

2. **The diagnostic pass is approved: one Drive pass with no round open, recording everything.**
   - **For every channel that yields no train:** the size of the clean set before and after the non-suspect filter, the seed route ((b) or (c)), and the intervals of the seed subtrains against the autocorrelation RR. This tests the hypothesis that a split seed with irregular extras leaves no interval within 10% of the reference.
   - **For every train that fails the veto:** classify each harmed injection as lost (no beat within 2 ms of the true R), moved (a beat within 2–20 ms), or captured (the transient itself accepted). Report it per channel.

3. **Measure, don't adopt: tagging beats near real transients instead of vetoing the channel.** GapAfter now lets HRV exclude intervals, so a beat near a known transient can be tagged instead of trusted. In that case the measure of harm that matters is wrong intervals that are not tagged.
   - **Tagging rule** (fixed now): every beat within ±20 ms of a non-cardiac common-mode event found by `find_events` gets both adjacent intervals tagged.
   - **The injection must be realistic:** a common-mode transient on all nine channels at the recording's own amplitude, so `find_events` can find it, with injections drawn at random times.
   - **Untagged harm:** the fraction of injections that change the train without the changed interval being tagged.
   - **Tagging cost:** the fraction of intervals tagged by the recording's own real events.
   - **Report per channel**, for task 05, findpeaks and the template trains, with the count gate alongside.
   - Do not adopt anything. A veto redefined as "untagged harm ≤ 1%, tagged fraction ≤ 5%" would be a ruling made after seeing these numbers, so it needs Andrea's decision, with both numbers on the table.

4. **Her findpeaks floor extras are for Andrea to act on.** Up to 14% of accepted intervals are false short intervals, which inflate RMSSD and SDNN in any HRV already computed from those channels.
   - Possible interim fixes in her own code: a spacing floor relative to the median RR instead of 100 ms, or a prominence floor. Either changes her outputs.
   - This is her decision. Nothing is changed.

5. **Ratified:** the GapAfter implementation and its proofs; the generator carrying the nargin fix; explicit option parsing; and the finding that the earlier fix was a hand edit.

### RULING 2026-10-01 (b) — the stomach_ref notch is a mechanical rule, applied to every hum recording

- **The rule.** The notch rulings ((c) 5 and addendum 5) named three recordings because those were the only ones known. The rule is mechanical: **any recording whose stomach screen flags ANT1 as mains-dominated gets the notch** in the stomach consumers' stomach_ref.
  - The inputs are chosen by addendum 5's comparison: all stomach_ref inputs if the all-inputs notch changes mmc against ANT1-only, otherwise ANT1 only.
- **Round 6:** B t03 3_1 and B t03 ms3 are routed with the notch.
- **B t03 es2:** its unnotched frozen entry is an omission (my error).
  - Correct it as a routing change: append the corrected entry and recompute every pooled mark.
  - Decided now, before its effect is known. It holds even though it may reclassify s3#m7 and s3#m13, which are in that recording.
- **Report the gate twice,** on table `2de52b5d` (as built) and on the corrected table, with every mark that changes class and why. The corrected table governs.

**Applied 2026-10-01 (corrected table `628d2a28`):**
- **All three hum recordings take the all-inputs notch.** mmc Jaccard between ANT1-only and all inputs: 0.72 (es2), 0.58 (3_1), 0.31 (ms3).
- **Pooled gate: 242/245, lower bound 0.967. Not cleared.**
  - On `2de52b5d` as built it was 238/239 (0.9803).
  - The difference is s3#m7 and s3#m13, which become target misses through mmc damage on the notched stomach_ref.
  - Their `not_target` closure stands as Andrea's classification. The scores stand (closure rule 2).
- **About 141 more target marks** with no new target miss are needed (386 at 3 misses).
- **slow_wave is not immune to the notch.** In B t03 3_1 it removed slow_wave damage from s4#m6 and s4#m7. Ruling (c) 5's "slow_wave untouched" holds only for the first three recordings; the all-inputs comparison stays the mechanical rule.

### RULING 2026-10-01 (c) — tagging rejected; HR from a cross-site lead that nulls the ground signal (measure first)

**Measured (diagnostic pass, 10 A/B recordings; H unspent):**
- **Template detector, no-train channels:** the non-suspect filter is not the cause (41 of 48 channels have zero clean beats before it).
  - The task 05 seed on route (b) is not a cardiac train: median interval 23× the autocorrelation RR, and at most 10.5% of intervals within 10%.
  - The route (c) halves are alike (1.33–2.11× RR, near-equal SNR), so the split is ambiguous.
- **Veto harm is mostly capture for every detector:** task 05 64%, findpeaks 83%, template 54%. The rest is mostly lost beats; moved beats are 2–5%.
- **Tagging:**
  - It costs 20–79% of intervals (median 34–45%).
  - No A recording gains a train.
  - `find_events` catches only 26–64% of the injected transients.
  - The circular own-beat variant tags less and hides capture, as predicted.

**Rulings:**

1. **Tagging is rejected** and is not offered to Andrea for adoption. Its cost is a third or more of the data, and it rescues nothing.

2. **Why every single-channel approach fails.** On a raw contact the ground-site transient and the far-field ECG arrive together. The transient is captured because nothing on one channel can tell it apart from a beat.
   - The tripole cancels both: its contacts are 1.5 mm apart, so they see the same cardiac potential.
   - The ground signal enters every channel with only modest gain differences (contact R² 0.67–0.99 against the outside reference). The cardiac far-field differs strongly **between sites** (left neck, right neck, stomach), because the heart's dipole projects differently onto each.
   - **So a combination of channels from different sites can cancel the ground signal and keep the heart.** In effect it is a bipolar ECG lead built from distant contacts.

3. **Measure, don't adopt. Two variants, A/B only, H unspent:**
   - **(i) Cross-site pairs:** every difference of two raw contacts from different sites (L cuff − R cuff, cuff − stomach).
   - **(ii) Ground-nulling weights:** per recording, estimate the ground gain vector **g** from non-cardiac events (the `cm_gains` method) and the cardiac gain vector **h** from the beat-locked template on the vetted or best available train. Choose **w** to maximise (w·h)² / (wᵀ Σ w) subject to w·g = 0, where Σ is the noise covariance from beat- and event-free stretches.
     - Report how stable g is over the recording (per-minute estimates). If g drifts, w must be estimated per segment.
   - **Detectors:** task 05 and findpeaks, run on the derived lead. The template detector is not used.
   - **Gates, unchanged:** count gate, task 05 plausibility, and the transient veto. The veto injects **realistic** transients: the measured g pattern on all nine channels at the recording's own amplitude, so the test sees exactly what the lead must cancel.
   - **Report per recording:** the best lead, its gate results, its veto harm against the best single-channel harm, and whether the recording gains an HRV-grade train. Report A t01, A t02, A t01 1_2 and A t04 first.
   - **No adoption.** If it works on A/B, the design is written down and frozen and goes to H once, as in ruling (f) 6–7.

4. **The round-record note is confirmed.** Add the `notes` entry to `change_2de52b5d2537d0b2_to_628d2a28cd1f5ed5.json`, written atomically, as proposed.

**Amendments to ruling (c), 2026-10-01 (applied by the build):**
1. **The binding veto, for every lead, is the real-pattern injection:** each transient carries one real non-cardiac event's gain pattern and amplitude. The median-g injection is circular for the weighted lead (w·g = 0 cancels it by construction), so it is reported only.
2. **Half-split protocol.** Everything is estimated and chosen on the first half: event classes, g, h, Σ, w, the best lead and the best single channel. The second half judges only those choices.
   - A recording gains a train only if the chosen lead passes on the second half and the chosen single channel does not.
   - Report how many candidates passed on the first half.
   - Second-half per-minute weights may use that minute's own g, since labels play no part. Drift is tested out of sample.

### RULING 2026-10-01 (d) — the cross-site lead works on A/B; freeze a pairs-only design, cross-check it, then H once

**Measured (half-split protocol, 10 A/B recordings; H unspent; nothing written to the store):**
- **A t01, A t01 1_2 and A t04, which have no single-channel HRV-grade train, each gain one on the held-out half** from a left−right neck contact pair (A t01 via the per-minute weighted lead). The harm is 0.000 under both injections.
- **A t05's gain is real:** its stored L_T fails the real-pattern veto on the second half (0.04).
- **B t01 and B t03 2_2's "gains" are selection artifacts:** SNR picked a raw contact over an L_T that also passed.
- **A t02 and B t03 es2 fail** the second half's count gate (0.889 and 0.80), not the veto.
- **g is 0.96–1.05 on every contact**, with 99–100% sign agreement, and does not drift: the first-half w leaks ≤ 0.04 on the second half. Every chosen pair is left−right neck; none involves the stomach.
- **The weighted lead is fragile:**
  - It could not be estimated in A t01 1_2 (28 clean beats for h) or in B t03 es2 (no quiet samples for Σ, at about 31 events/s).
  - Σ rests on only 1.2–2.1 quiet seconds in four recordings.
  - Where h came from a non-passing train, it can point along g (A t04 cos −0.99).
- **A contact carrying almost no ground signal:** A t04 LVN3 (g −0.03). A contact-health observation, relevant to task 18. *(Corrected 2026-10-02: B t03 2_2 LVN3, listed here earlier, is healthy: g 0.985, agreement 0.999.)*

**Rulings:**

1. **The design frozen for H is pairs only.**
   - **Candidates:** the 9 left−right neck contact differences in both orientations (18), each with task 05 and findpeaks.
   - **Not in the design:** the weighted lead, cross-site pairs involving the stomach, and the template detector.
   - **Why:** pairs carried the substantive result, need neither h nor Σ, and fewer candidates means less selection inflation. Restricting the candidates is a design decision made on development data, with H unspent.
   - **No Σ floor is set**, since the weighted lead is out.
   - **Report from the existing records (no re-run):** whether pairs-only still gives A t01, A t01 1_2, A t04 and A t05 a train under the same half-split selection.

2. **An independent count cross-check is required before H**, as the build proposed. A lead locked on another periodic source would pass its own count gate. Run it on every recording, against references that do not use the lead:
   - **(a) Rate:** per minute, the lead's beat rate against the autocorrelation rate of each other site's raw channels and of stomach_ref, wherever that minute is clear. The lead passes if it agrees within 5% with the median clear reference in ≥ 95% of minutes that have one.
   - **(b) Timing, where a vetted train exists** (A t05, B recordings): ≥ 99% of the lead's beats within 2 ms of the vetted beats, plus a constant offset, which is reported.
   - **(c) Morphology:** the lead's beat-locked template on each raw contact must show the QRS (template SNR on the raw contacts above that contact's own noise-shuffled template).
   - A recording keeps its gain only if (a) and (c) pass, and (b) where applicable.

3. **Freeze, then H once.**
   - Commit the pairs-only module with tests and mutants, outside the hash. Record the commit, the candidate list, the selection rule, the half-split protocol, the binding real-pattern veto and the cross-check.
   - Run H once, on every H recording in rounds 3–6, H t05 2_1 sr included. Report it per recording with all checks. If it fails, report the failure and do not tune.
   - **Adoption is a separate ruling after H.** If adopted, the lead enters `gated_selection` as an additional candidate source, and each recording that gains a train is appended as a routing change with a full recompute.

4. **A note for task 18:** a contact with g near 0 (A t04 LVN3) is not seeing the shared ground. It is likely open or detached, and must not be used for conduction velocity. *(Corrected 2026-10-02: B t03 2_2 LVN3 is healthy.)*

### RULING 2026-10-02 — the cross-check, corrected on development data before H

**Measured (pairs-only, commit `775a809`, A/B only):** under ruling (d) 2 as written, only A t01 es1 keeps its gain. The breakdown:
- **(a) A t01 1_2 and B t01 fail on reference quantisation.** The stomach references' autocorrelation steps by about 6% at these rates, and the median of two split references lands between steps.
- **(b) B t01 and B t03 2_2 miss 99% narrowly:** 99.2% and 99.1% of beats fall within 3 ms (between-lead fiducial jitter). B t02 1_3 passed with a spread of ±0.7 ms.
- **(b) A t05 is a genuine disagreement:** about 21% of beats on each side are unmatched within 20 ms, though per-minute counts agree. Its "vetted" L_T itself fails the real-pattern veto (0.04).
- **(c) A t04 fails only on the detached LVN3** (SNR 5.4 against a shuffle maximum of 16.4). Every healthy contact shows the QRS at 93–227.
- **B t03 2_2's chosen pair uses the detached LVN3.**

**Changing the check now is allowed.** It is part of the design and H is unspent. But A/B results under the corrected check are no longer evidence; H is.

**The corrections are physical ones, not looser thresholds:**

1. **Detached contacts.** A contact is detached if its first-half |g| < 0.2 or its sign agreement is < 0.75. This rule comes from the ground-signal physics, not from the cross-check.
   - A pair containing a detached contact is **not a candidate**.
   - **(c)** requires the QRS on every non-detached contact, with at least 4 non-detached contacts on at least 2 sites. Fewer than that means unassessable, so no gain.

2. **(a) Measure the reference better rather than widening the tolerance.**
   - Cross-check references use parabolic sub-lag refinement of the autocorrelation peak. The count gate itself is unchanged.
   - A minute whose clear references disagree among themselves by > 5% is unassessable, since there is nothing to judge against.
   - **Pass:** agreement within 5% in ≥ 95% of assessable minutes, where assessable minutes are at least half of the minutes with any clear reference. Otherwise the recording is unassessable, so no gain.

3. **(b) Identity is the question, not fiducial precision.** The veto already judges precision.
   - Match beats within **5 ms** after the constant offset, which is far below half an RR, so a wrong beat cannot match. Require ≥ 99% matched **in both directions**.
   - Report the SD of the matched differences; it is not a gate.
   - **A reference that fails the binding real-pattern veto is not a valid timing reference,** so (b) does not apply.
   - **But two trains that disagree by more than 5% of beats** (unmatched within 20 ms, either direction) mean at least one is wrong. Neither is trusted until a beat-by-beat check resolves which one carries the QRS: for each unmatched beat, the template correlation on the non-detached raw contacts.
   - Until then **A t05 gains nothing.** Report the resolution.

4. **The real-pattern veto is the binding veto for every source.**
   - When the lead is adopted, `gated_selection` re-vets every candidate source (task 05, findpeaks and the lead) under it, and every changed recording is appended as a routing change with a full recompute.
   - Report now, without writing anything, which stored HRV-grade trains fail it (A t05 L_T is one).

5. **Order:**
   - Re-run the A/B cross-check under corrections 1–3, with the candidate list cut by correction 1.
   - Report it as development data only.
   - Commit the corrected module, with tests and mutants, and record the frozen design.
   - **H runs once, after Andrea says "done" on round 7.**

### RULING 2026-10-02 (b) — round 7; a diagnosable miss class; H go-ahead

**Measured:**
- **Round 7:** 32/33 raw. On appended table `218e053d` (no existing entry changed), there are 28 targets and 5 below tolerance. **Pooled gate: 269/273, lower bound 0.966. Not cleared.** With 4 target misses, 456 target marks are needed: about 183 more with no further target miss.
- **The new target miss, s5#m5** (B t01 3_1, a hum recording): max z 2.81 on ANT3. Its only damage is to mmc, on the notched stomach_ref.
- **Corrected cross-check (`78a5b2d`, development data only):**
  - A t01 es1 and A t01 1_2 keep their gains; A t04 keeps its gain with LVN3 excluded.
  - **A t05 is resolved:** 97.4% of the lead's unmatched beats carry the QRS, against 1.3% of the stored L_T findpeaks train's. **The stored A t05 L_T train sits on a non-QRS peak in about a fifth of its beats.**
  - B t03 ms3 R and H t01 1_2 L are distrusted (check (iii)).

**Rulings:**

1. **A diagnosable miss class: hum recordings, mmc-only damage, below threshold.** Three of the four target misses (s3#m7, s3#m13, s5#m5) are in B hum recordings. Each is below `z_enter` on every detection signal, and each is a target only through mmc damage on the **notched** stomach_ref.
   - **Hypothesis.** Detection reads un-notched inputs (invariant 43 keeps ANT1 un-notched for detection). In hum recordings, mains leakage inflates the baseline σ of the stomach signals and depresses their z. The consumer reads the cleaner notched signal, so detection is blind to exactly what mmc is sensitive to.
   - This is branch 1 of task 09's "If it fails": a diagnosable class, which gets a generator feature and a re-measure.
   - **Measure first, writing nothing to the store:**
     - (a) The three misses' z on the notched stomach_ref, in detection's own bands and windows. Would any cross `z_enter`?
     - (b) The same z for every covered target mark in hum recordings, to check the class separates.
     - (c) The coverage and chance-margin cost of adding the notched stomach_ref signals in hum recordings only.
   - **If (a) confirms it, the fix is the pre-declared additive extension** (ruling of 2026-09-29, round-4 miss, branch 3):
     - add the notched stomach_ref signals in hum recordings as extra detection signals;
     - prove it is a strict superset on every cached region;
     - keep every original score (the four misses still count);
     - recompute the chance margin.
     
     The generation hash is unchanged and the extension is recorded explicitly. Propose it to me before building.

2. **H go-ahead.** The frozen design is `78a5b2d`. When the A/B re-run and item 4 finish, run H once on every H recording in rounds 3–7, under that exact commit and protocol, with nothing changed. Report per recording with all checks. If it fails, report the failure and do not tune.

3. **A t05's stored L_T train is wrong in about a fifth of its beats.** It is still the stored HRV-grade train and A t05's peri-R source. At adoption it fails the binding real-pattern veto and is replaced (ruling 2026-10-02, 4).
   - **Report now** which pooled marks' classes depend on A t05's HR or peri-R route, so the size of the change at adoption is known.
   - **Andrea:** any HRV she has already computed from A t05's L_T findpeaks beats is affected.

4. **B t02 1_3 ANT2 "detached" with (a) newly failing:** report the cause. Do not change the rule for it.

### RULING 2026-10-02 (c) — H result; the pairs lead is adopted; detached contacts leave the rate references

**Measured (H once, under `78a5b2d`, byte-identity checked before each recording):**
- **H t05 2_1 bl, H t09 2_2 bl and H t01 1_2 bl gain an HRV-grade train that passes every check.** The second-half harm is 0.000 under both injections, rate and morphology pass, and no chosen single channel passes.
- **H t01 es2 fails** (veto 0.03, rate 0.75).
- **In all three H stim/recovery recordings, no pair passes the count gate and plausibility at all.**
- **Development (A/B) gains:** A t01 es1, A t01 1_2 and A t04. A t05 is resolved in the lead's favour: 97.4% of its unmatched beats carry the QRS, against 1.3% for L_T.
- **All 10 stored HRV-grade trains pass the real-pattern veto** on their full regions. A t05 L_T, B t03 2_2 L_T and B t03 es2 L_T sit exactly at 0.01. The veto cannot see wrong-peak picking.
- **The hum/mmc class is not explained by baseline inflation.** On notched inputs the three misses peak at z 0.70–2.81. The extension would be a strict superset at negligible cost, but recovers nothing.

**Rulings:**

1. **The pairs-only lead (`78a5b2d`) is adopted** as an additional candidate source in `gated_selection`. Every source faces the same gates, with the real-pattern veto binding (ruling 2026-10-02, 4).
   - **Incumbent preference:** a stored HRV-grade train that still passes every gate stays, unless rule 2 applies. The lead fills recordings with no passing train. This keeps adoption to the cases with evidence and avoids churn (B t03 2_2 keeps its L_T).
   - **Recordings that gain:** A t01 es1, A t01 1_2, A t04, H t05 2_1 bl, H t09 2_2 bl, H t01 1_2 bl, and A t05 by replacement (rule 2).
   - Each recording that gains is appended as a routing change. Its peri-R routes are recomputed from the new train, since hump cuffs distrusted for lack of a train may be restored. Then every pooled mark is recomputed, with all three numbers and the class changes listed.
   - Beats are stored in Andrea's format with `gapAfter` (`emit/hr_beats.py`).

2. **Disagreement beats incumbency.** Where a stored train and a passing lead disagree on > 5% of beats (unmatched within 20 ms, in either direction), resolve beat by beat by template correlation on the non-detached contacts. The train that carries the QRS wins.
   - Applied now: A t05 → the lead.
   - Report the same check for every recording where a stored train and a passing lead coexist.

3. **Detached contacts are left out of the rate references in (a),** as they are left out of (c). A contact that does not see the ground signal is not a physiological reference.
   - This is decided after H, but it changes no H outcome (no H contact is detached). On development data it changes only B t02 1_3, which gains nothing.
   - Record it as a post-H amendment to the adopted module, and re-run the A/B and H cross-checks to show that no adoption decision changes.

4. **HR in stim/recovery remains open.** No lead passes there on H, and most stim/recovery recordings without HR stay without it.
   - The likely cause is stimulation itself, but that is a hypothesis.
   - **Measure, don't build:** in the H and A stim/recovery recordings, compare the count gate and veto in the 2-minute stim epoch against the 20-minute recovery. Report whether the failures are confined to stim.
   - Priority is Andrea's call.

5. **The hum/mmc miss class: next hypothesis, measure only.** mmc's damage rule judges a mark against a local ±30 s MAD, while detection uses the whole-region baseline.
   - Report the three misses' z against a ±30 s local baseline on detection's stomach bands, the same for the covered targets in those spans, and the coverage cost.
   - Do not propose an extension unless the misses cross `z_enter` and separate from the covered targets.

**Addendum to ruling 2026-10-02 (c), on storage:**
- **The stored product must pass on what it is used for.** The half-split protocol exists for honest selection. A stored train covers the whole region, so `gated_selection` judges it on the full region with every gate, the cross-check included.
- A ruled gain that fails on the full region is not stored. Report it with the cause.
- If the cause is a defect in how the full-region check is computed, fix it, test it and re-run. If the cause is real, the recording has no gain.
- Do not split storage into "valid minutes" without a ruling.

### RULING 2026-10-02 (d) — the "fixed-rate" stomach references are a mains defect in the check; fix and re-run

**Measured (full-region adoption pass, `c7358cc`):**
- A t01 es1 matches its validation.
- A t05 is replaced by the RVN1−LVN1 pair (rule 2).
- **A t04 and A t01 1_2 store nothing on the full region.** Rate (a) fails because the stomach references sit at fixed rates, 360.2, 399.6, 423.7 and 480.2 bpm, each constant to ±0.2, while the lead varies smoothly. A t01 1_2 also fails morphology on LVN3 (SNR 7.2 against 7.9).
- A t03 3_2 bl (round 7, not on the ruled list) would store a pair.
- B incumbents stay.

**Those rates are 7200/m bpm for integer m** (m = 20, 18, 17, 15 to within 0.01): autocorrelation lags of exactly m/120 s. The references are locking onto **rectified 60 Hz hum** (120 Hz after rectification), not onto any physiological source.
- This also explains the "≈6% steps" seen earlier: neighbouring m differ by 5–7%.
- So it is a **defect in how the cross-check reference is computed** (the addendum's first case), not evidence against the lead.

**Rulings:**

1. **Fix the references.**
   - Notch every rate-reference channel at 60 Hz and its harmonics up to the band edge before rectification and autocorrelation.
   - Then flag any reference minute whose rate lies within ±0.3 bpm of 7200/m as hum-locked, and treat it as not clear.
   - Add a test: a synthetic rhythm with rectified hum must not lock onto the 1/120 s grid.
   - Neither the lead nor the count gate changes. The lead is a left−right difference, so the hum is common mode and cancels.

2. **Re-run the cross-check with the fixed references** on every A/B and H recording, and in the full-region adoption pass.
   - Report every decision that changes, in either direction.
   - **A change to an H result is reported as a post-H correction of a defect in the check, not as new evidence.** H t01 es2 fails the veto, so it cannot become a gain.

3. **Check the count gate for the same defect, report only.** For every stored train and every passing pair, count the minutes where the train's own autocorrelation rate sits within ±0.3 bpm of 7200/m.
   - If the count gate is hum-locked anywhere, report the recording and stop. Do not change the gate without a ruling.

4. **A t01 1_2's LVN3 morphology failure is real under (c).** If it still fails after the fix, A t01 1_2 has no gain. Do not relax (c).

5. **The adopted module applies mechanically to every recording, not only the list known at adoption.**
   - A recording where full-region `gated_selection` stores a passing pair (every gate and the cross-check) is appended, for example A t03 3_2 bl.
   - Report such recordings separately as further held-out evidence, since they were never seen during design.

6. **Then append and recompute** as ruled, and push `c7358cc` or its fix.

### RULING 2026-10-02 (e) — mains in the count gate; the guard becomes a persistence test; veto precision near threshold

**Measured (gems `c7358cc` + `d203469`, held locally; nothing appended):**
- **The notch fixes the references.** Animal A's stomach references track the lead within about 1%.
- **The ±0.3 bpm guard discards true heart readings** that happen to sit on the 7200/m grid (A t01 es1 minute 11 at 360; A t04 minute 1 at 379), and that alone costs both gains.
- **The count gate itself can lock onto mains.** In B t02 3_3 three passing pairs have their own autocorrelation on the grid in 5–13 minutes, while the heart runs at about 311 bpm. Scattered single grid minutes elsewhere are consistent with the heart sitting at a grid rate.
- **H t01 es2:** its pair failed the veto on H's held-out half (0.03) but passes over the full region (≤ 0.01), with 200 injections in each case.
- **Item 4:** stim/recovery HR is lost mainly to the 20-minute count gate. The first 2 minutes of recovery give a passing train in all 8 recordings.
- **Item 5:** no extension. On a ±30 s local baseline only s5#m5 crosses `z_enter`, the misses do not separate from the covered targets, and the pooled chance margin would fall from +0.149 to +0.108.

**Rulings:**

1. **The count gate gets the same mains fix as the references.** Its autocorrelation input is notched at 60 Hz and its harmonics up to the band edge before rectification, for every candidate source.
   - A gate whose reference measures mains does not measure the heart. This is the same defect, so it gets the same fix. The gate's thresholds (0.05, 95%) are unchanged.
   - Re-gate every stored train and candidate. Report every change in either direction. An incumbent that fails after the fix goes through `gated_selection` as usual.

2. **The ±0.3 guard is replaced by a persistence test** for references and for the count gate.
   - A minute is hum-locked only if the rate sits within ±0.3 bpm of the **same** 7200/m value for ≥ 3 consecutive clear minutes. Such minutes are treated as not clear.
   - **Why:** a mains lock is constant to ±0.2 bpm for minutes, as measured, while a real heart rarely holds within 0.6 bpm for 3 minutes. A single-minute coincidence is not a lock.
   - The fixed guard was part of a defect fix, not of the frozen design. It is replaced because of its own measured false positives.
   - Keep reporting single-minute grid coincidences as a diagnostic, not a gate.

3. **The veto's precision near the threshold.** The veto estimates a fraction from 200 injections, which is too few at 0.01: 2 hits against 6 is within noise.
   - Rule, applied to every source, stored or candidate: if the one-sided 95% interval of the harm estimate contains 0.01, add injections in blocks of 200, up to 2,000, until it does not. Then decide on the estimate.
   - The threshold is unchanged. Report every decision this changes, including A t05 L_T, B t03 2_2 L_T, B t03 es2 L_T and H t01 es2.
   - **H t01 es2 is then decided by the full-region estimate**, like any other recording. Its H result stays recorded as "failed on the held-out half", and a later storage is not counted as H evidence.

4. **The hum/mmc miss class is closed as `accepted_limitation`** ("below threshold on every detection signal under global and local baselines; damage only to mmc in hum recordings; no generator feature recovers it without large cost"). Its misses keep counting.

5. **Re-run in this order:**
   - (i) the A/B and H cross-checks;
   - (ii) the full-region adoption pass under rulings 1–3;
   - (iii) item 4 (stim epoch against the recovery windows), because the count-gate fix may change it.
   
   Report the changes. If no other stop condition arises, append, recompute and push. The stim/recovery question goes to Andrea with the re-run item 4 numbers.

### RULING 2026-10-02 (f) — adoption ratified; HR stored per minute so Andrea's windowing applies; mask-grade pairs need the cross-check

**Measured:** the adoption is appended as table `4055cdce`, with 13 entries changed. **Pooled gate 272/276 (0.966), not cleared**; excluded = target 318/322; raw 328/332; chance margin +0.147.
- All seven ruled gains hold. Three held-out recordings were appended mechanically.
- Veto precision resolved every threshold case.
- **Stim/recovery:** the full 20-minute recovery stores a train in 3 of 8 recordings, and the first 2 minutes do in 7 of 8. What fails is the whole-recording count gate, not stimulation.

**Andrea, 2026-10-02:** her analyses (`processing_new`) already work in windows and keep a window when less than 50% of it is blanked. *(She wrote "60 min windows". The build must read `processing_new` and record the exact window length and rule it uses for HR/HRV.)* Nothing is wrong with that approach. It fails here only because our storage rule is whole-recording: one bad stretch means no train at all, so her window rule never gets to act.

**Rulings:**

1. **The adoption is ratified,** as are the lock test reading the refined peak, the persistence limitation (minutes are lost, never wrong), and the s5#m5 closure (s3#m7 and s3#m13 keep their earlier `not_target` closures).

2. **HR storage becomes per minute, so that her windowing does the rest.** This supersedes the addendum's "no valid-minutes storage without a ruling".
   - **A train is eligible** if it passes the recording-level gates: the transient veto with precision, task 05 plausibility, and for pairs the cross-check's morphology (c) and timing (b) where it applies.
   - **Each minute is valid** only if it is clear and the train's beats are within 5% of the minute's autocorrelation rate (the count gate's own test, now applied per minute). For pairs, the minute must also pass (a) where it is assessable.
   - **Minutes that are not valid become tagged gaps.** Their beats are removed, and `gapAfter` marks the boundaries. Wrong beats are never left in place to look valid to the window rule.
   - **Store the train with the most valid minutes** (ties go to template SNR). Incumbent preference and rule 2 still apply among trains that pass the recording-level gates.
   - **Report per recording:** valid minutes and longest valid run, and how many of her analysis windows would then pass her < 50%-blanked rule. This is the number that matters to her.
   - Recordings that already pass end to end keep their trains; their invalid minutes, if any, become gaps.
   - **Development first.** Run on the A/B and H stim/recovery and baseline recordings, report, and then append as routing changes with the full recompute. Nothing is appended before I see the report.

3. **Mask-grade pairs need the cross-check.** A pair establishes that it is cardiac only through (a) and (c). A mask-grade pair (used for peri-R, where the veto is not required) must therefore pass (a) and (c).
   - Re-check A t02 and A t01 3_2 sr. If either fails, its peri-R route reverts. That is a routing change, with the recompute.

### Adapter-check findings, 2026-09-28

- **Tripole polarity.** The old hardware tripole's large events are mostly
  **positive** (JEL top-100: 97% positive); the new software `T` is mostly
  **negative** (B, J, I). The spike consumer detects negative peaks only, so on
  the old cohort it counted the minority polarity. **Andrea: literature usually
  shows negative spikes; compare negative-only against absolute-value detection
  on both the old hardware tripole and the new software tripole** before
  choosing. Report per cuff: event counts, the amplitude distribution of each
  polarity, and waveform averages of each, on old clean windows and on
  artifact-screened new windows. Do not flip the adapter or change the consumer
  until she has seen it.
- **Fitted tripole weights collapse to one outer contact** in most cuffs (left:
  A, B, I; right: A, B, H, K; A's right fit leaves 0–1). Only J and H-left are
  near 0.5. That puts a poor or dead outer contact on most cuffs, which matters
  beyond T: **task 18's conduction velocity needs three good contacts per
  cuff.** Report, per animal and cuff, which contact looks poor and why.
- **Stomach reference.** The old ANT1–3 are a shared single reference, not a
  common average, and the adapter passes new ANT1–3 raw — **matches; no
  change.** The detection derivation's `stomach_ref` is a common average and
  removes most of the shared slow wave; that is a detection question, noted for
  step 1 above.
- **Anomalies for Andrea:** animal I's stomach channels share almost nothing
  (slow-wave r −0.05); J's ANT3 is nearly flat (6.5 µV vs ~64 µV); K's R_T and
  L_T have identical σ (2.71 µV) and zero spikes — check whether K's two cuffs'
  raw channels are duplicates.

### Threshold sweep
Sweep `z_enter` over 2.0–4.0. For each value report recall, candidate count,
fraction of frames flagged, and true-positive fraction on the 43 labelled
recordings. **Pin the value** that meets both: recall ≥98% and ≤3000
candidates/recording.

### MEASURED 2026-09-19 — first run on real data (animal J baseline, 601 s)

Injected 48 artifacts (4 kinds × 4 absolute amplitudes × 3 durations) as common
mode with per-channel gain jitter; z on the **log** envelope, median reference,
6 bands, 9 raw contacts.

| nmin | z | clean flag | cands | 200 µV | 1 mV | 5 mV | 20 mV |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 22.1% | 136 | 75% | 92% | **100%** | **100%** |
| 1 | 6 | 11.9% | 105 | 42% | 75% | **100%** | **100%** |
| 3 | 4 | 8.4% | 41 | 42% | 83% | **100%** | **100%** |
| 3 | 6 | **1.8%** | 6 | 0% | 58% | 92% | **100%** |

**Verdict: conditional. Large artifacts (≥5 mV) are caught reliably; ~1 mV is
marginal; 200 µV is not caught at a usable flag rate.** And 200 µV matters — it
arrives as common mode, the tripole rejects it only ~2.5×, so it still lands at
~27σ on `T`.

**Four defects were found in the test itself before these numbers, and each one
had produced a spurious failure.** Record them so they are not repeated:

1. **z as specified (`ref = p10`, scale = MAD of the linear envelope) is
   mis-centred.** It puts median z at 1.00 and p90 at 3.05, so ~10% of frames
   exceed z=3 *per pair*. **Use a robust z on the log envelope** (median
   reference): median 0, p90 1.44, p99 4.21. Envelopes are positive and
   right-skewed; the log makes the null symmetric. This also matches the
   `onset_rate` feature already defined on the log envelope.
2. **The union across (signal × band) pairs is a multiple-comparisons problem.**
   Per pair the clean flag rate is only 0.2–3.4%, which is on target — but the
   union of 36 pairs reaches 40–55%. Any threshold must be set on the
   **family-wise** rate, or cross-channel agreement (`nmin`) used to reduce the
   family. The spec's "≤3% of frames" target was silently per-pair.
3. **Omitting the slow bands made every long artifact look missed.** A 2 s event
   is 0.25–0.5 Hz and invisible to bands above 2 Hz. Always run all six.
4. **Amplitudes must be absolute, not multiples of the wideband MAD.** MAD here is
   26–62 µV, so "16× MAD" is ~500 µV — below this recording's own p99.9 (370–870
   µV) and far below its real excursions (2 mV to 2 V). Scaling to MAD made
   realistic artifacts look tiny.

### Pass condition — REVISED
The original "≥98% of injected synthetics" over a grid that includes artifacts at
1× the noise floor is **unachievable and wrong**: an artifact at the noise level
is undetectable by construction and also harmless. The criterion must be tied to
what damages a consumer:

> ≥98% recall for artifacts **above each consumer's tolerance**, at a family-wise
> clean flag rate ≤5%.

Deriving the per-consumer damaging amplitude is a prerequisite, not an afterthought.

### Synthetic injection is NOT sufficient as the gate
It cannot distinguish "the generator over-fires" from "the recording is genuinely
contaminated", because there are no labels. On this file the clean flag rate at a
permissive threshold is 12–22%, and **nothing in the experiment can say whether
that is wrong.**

**Use the 43 already-labelled recordings instead.** Replay the generator against
the existing human interval labels and measure real recall and real precision.
That is a direct measurement, needs no injection, and should run before any
further threshold tuning. Keep injection as a *secondary* probe for artifact
classes the labels may not contain.

### If it fails
**Stop. Do not proceed to tasks 10–12.** Two branches:
1. Recall fails only for a diagnosable class (e.g. slow drifts) → add a band or a
   generator feature for that class and re-measure.
2. Recall fails broadly → **abandon the two-stage split** and build the fallback: a
   1D U-Net over the multi-band envelope stack, trained as dense segmentation with
   the same LOAO protocol. That learns the detection function instead of
   thresholding a hand-built statistic.

Record which branch was taken and why.

### Do not
Do not relax the pass condition to proceed. The gate exists precisely because the
rest of the architecture is worthless without it.

---

**Depends on tasks:** 07

> **THIS IS A GATE.** Work after this task is wasted if it fails. Do not proceed past it.

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
