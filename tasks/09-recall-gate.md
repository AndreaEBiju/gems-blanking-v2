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
