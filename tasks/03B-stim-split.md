<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

## Task 03B — Stim / recovery split and epoch exclusion

**Module:** `io/stim_split.py`
**Depends on:** 03, 03A. **Must run before task 06** — see "why the ordering matters".

> **Build order: not now.** This task is fully specified but is **deliberately not
> built during the early phases.** It runs at load time inside the finished suite,
> per file, and nothing in tasks 05 (R-peaks), 02 (cardiac window) or the
> channel-identification work depends on it — those are validated on baseline
> recordings first. Build it when the pipeline is assembled, not before.

### Purpose
A `stim_recovery` recording contains a stimulation epoch followed by a recovery
epoch. The stim epoch is **excluded from all downstream processing and blanking**;
only recovery is analysed. This is the "ignore the stim duration" decision, made
concrete.

**Scope:** this task identifies and excludes the stim epoch. Removing artifact
*within* the stim epoch remains deferred (Part C) — nothing here depends on it.

### What exists today
`processing_new/splitStimRecovery.m`, plus `splitStimRecoveryManual.m` for manual
override and `blank_stim_spikes_nan.m`. Port the mechanism, fix three things.

Current method, on a separate `vib` on/off channel at its own `fs_vib`:

```
vib_sm  = movmean(vib, max(5, round(0.01*fs_vib)))           % 10 ms smooth
vib_env = sqrt(movmean(vib_sm.^2, max(5, round(0.1*fs_vib))))% 100 ms RMS
threshold = (prctile(vib_env,20) + prctile(vib_env,80)) / 2  % if not supplied
ON = vib_env > threshold
   -> keep only the LARGEST ON segment
   -> minDurSec   = 10
   -> searchPadSec = 2.0  (edge refinement against the envelope crossing)
writes <base>_stim.mat  {x_stim, stimMask, dig_aligned, fs_sig, detectInfo}
       <base>_recovery.mat {x_recovery, recoveryMask, ...}
```

The stim portion is **kept**, not deleted — that is the right call and it is what
lets the deferred stim work happen later. Since the split now runs at load time
rather than as a batch pre-pass, "kept" means **returned as an epoch**: nothing
writes `_stim.mat`, and `.mat` sidecars are not part of this task. The MATLAB
files are the thing being ported away from.

Both epochs are returned as **numpy views, not copies** — correct, and necessary:
a 20-minute 9-channel epoch is ~2 GB to copy. Two consequences that must be
handled here rather than discovered downstream:

- **Return the views read-only** (`arr.flags.writeable = False`). A view shares the
  parent's buffer, so any consumer that masks in place — and hard invariant 1 says
  masking writes NaN — would silently mutate the source array. Read-only turns that
  into an immediate exception; a consumer that needs to mask takes its own copy of
  the slice it actually needs.
- **The view keeps the whole parent alive.** Holding a recovery epoch holds the
  full file's array, stim included. That is fine for one file and is not fine for a
  batch loop; whatever iterates over recordings must not accumulate epochs.

### The stim duration is known: **120 s, fixed by protocol**

This is the most useful fact available and it changes the method. Detection stops
being "find the ON region" (two unknowns, onset and offset) and becomes "find the
onset of a known-width window" (one unknown), with the duration left over as a
**free validity check**.

**Protocol: 2 min stim followed by 20 min recovery**, every `stim_recovery` file.
`stim_duration_s: 120`, `recovery_duration_s: 1200`, `stim_tolerance_s: 12` live in
the protocol config, per cohort, overridable per recording and recorded in
provenance. **Do not hardcode them** — protocols change, and a silently wrong 120
would be worse than no prior at all.

**Scope, decided 2026-09-22: the chronic recordings only.** `protocol.yaml` sits
at `<gems_root>/protocol.yaml` and each protocol entry names the `scan_roots` it
governs:

```yaml
protocols:
  - name: chronic_2min_20min
    applies_to: ["August-September Chronic Recordings"]
    stim_duration_s: 120
    recovery_duration_s: 1200
    stim_tolerance_s: 12
```

The balloon trials are a different experiment and **must not inherit the 120 s
prior**. A `stim_recovery` file that no protocol entry covers is a **refusal**,
not a default — same rule as a missing `vib` channel. Silently splitting a
balloon trial against a protocol that does not describe it would produce a
confident, wrong boundary, which is worse than stopping.

The tolerance is 12 s, not a tight few seconds: recording start/stop routinely
consumes several seconds at the edges, so a narrow band would flag normal captures.
The known **recovery** duration is a second, independent check — a file whose
recovery epoch is far from 20 min is suspect regardless of what the stim epoch
measured. This check was stated in the first draft and given no owner; it is now
required. Emit `recovery_duration_flag` when
`|recovery_measured − recovery_duration_s| > stim_tolerance_s`, independently of
the stim status, and record both durations.

It matters most exactly where the stim check is weakest: **a `clipped_start` file
has a censored stim duration but an uncensored recovery duration**, so the
recovery check is the only validation that still works on it. A `clipped_start`
file whose recovery is also off is a different and worse thing than one whose
recovery is 20 min, and the two must be distinguishable.

### Fix 1 — matched-width search, not a threshold

```
env      = RMS envelope of vib          (10 ms smooth -> 100 ms RMS, as today)
W        = round(stim_duration_s * fs_vib)
score(t) = mean(env[t : t+W]) - mean(env outside that window)
t_hat    = argmax score(t)                      # LOCATION only
```

A boxcar of known width slid over the envelope. This is strictly better than a
threshold on three counts:

- **No duty-cycle assumption at all.** The score is a contrast, so it does not care
  what fraction of the record is ON.
- **A mid-stim dropout cannot split the epoch.** A threshold would produce two ON
  segments and "largest segment" would keep one; the matched filter spans the gap
  because the window width is fixed.
- **Lower onset variance**, which matters directly: every recovery analysis is
  expressed as time since stim offset, so onset error propagates into all of them.

**The matched filter locates the epoch. It does not measure it.** This distinction
was missed in the first draft of this task and is the correction that matters most:
a ±2 s refinement around `t_hat` cannot report anything outside
`120 ± 2 s`, so the duration check of Fix 2 — the whole reason the known width is
worth having — would be structurally incapable of failing. A 95 s stim would be
reported as 120.0 s and would pass. **Do not bound the edge search by the window
width.**

Measure the duration instead by crossing, in two stages:

```
1. inside [t_hat, t_hat+W], find the FIRST and LAST samples above the ON threshold
2. follow each of those edges OUTWARD until the envelope falls back below
   threshold and stays below for hold_s (default 0.25 s)
   -> the outward walk is unbounded by W; the epoch may end up shorter or
      longer than the search window, which is exactly the point
3. duration = t_off - t_on, reported as measured, never as the prior
```

Verified on synthetic 95 / 113 / 120 / 140 s stim epochs: all four recovered to
within 110 ms, unchanged by the introduction of the hold.

**`EDGE_HOLD_S` bounds the damage from a mis-set threshold, and that is a second
reason to keep it short.** Measured, varying only the hold, with the
truncation-biased threshold of the next paragraph in place:

| hold | onset error with a biased threshold | ratio to a correctly anchored one |
|---|---|---|
| **0.25 s** (binding) | 0.641 s | 12.8× |
| 1.0 s | 20.955 s | 419× |
| 2.0 s | 220.765 s | 4415× |

A too-low threshold makes OFF noise intermittently cross, and the walk hops from
one excursion to the next; the hold is what stops the hopping. Note what this
means: the hold is now doing **two jobs** — tolerating a brief dip at a real edge,
and capping runaway — and those have no reason to want the same value. Do not
retune it to make a test pass. If edge dips ever demand a longer hold, the runaway
bound must be re-established by the flag below rather than by the hold.

**The outward walk is unbounded by design, so it needs a censoring flag rather
than a clamp.** If either edge walks more than `stim_tolerance_s` beyond the
matched window, set `walk_extended = True` and force status `review`.
**`walk_extended` is evaluated before `clipped_start`**, and the claim first made
here — that the flag "can only fire on files that would reach `review` anyway" —
is wrong in exactly one case, which is why the order matters. A `clipped_start`
file *skips* the tolerance check, so an overrunning walk on one would otherwise
pass unremarked. It must not: the recovery epoch's `t0` rides on the offset
boundary, and a walk that overran by more than the tolerance means that boundary
is uncertain by more than the tolerance. Everywhere else the original claim
holds and the flag is a diagnostic rather than a gate. Report the extension
distance per edge in provenance. The 220 s case above would be reported as a
measured 220 s with `walk_extended` set, never as a silent 220 s duration.

**The ON threshold must be anchored on OFF samples outside the matched window.**
The original specification took `off_level` and `off_σ` from the bottom decile of
the whole envelope, which is truncation-biased by construction: selecting the
lowest 10% of a distribution and taking its spread underestimates σ badly
(measured 0.000321 against a true 0.001119 — 3.5× low). With the threshold that
much too close to the OFF mean, 21% of genuine OFF samples cross it and the
outward walk of step 2 runs away; measured onset error −1.945 s. Anchoring on the
OFF population *outside* `[t_hat, t_hat+W]` — an unselected sample of the same
distribution — gives −50 ms on the same signal. Use
`threshold = off_level + 8·off_σ` with both terms from that population.

### Fix 2 — duration becomes a check, not an output

The status table in the first draft had **overlapping rows** — an 8 s clipped start
(detected 112 s) satisfied both "pass" and "clipped_start", so the outcome depended
on evaluation order. Clipping and the tolerance check are independent and are
evaluated independently:

**Step 1 — clipping flags** (state of the ON threshold at the first and last sample
of the record; both are flags, neither is a status):

| Test | Flag |
|---|---|
| ON at the **first** sample | `clipped_start = True` |
| ON at the **last** sample | `clipped_end = True` |

**Step 2 — status**, evaluated in this order, first match wins:

| Condition | Status | Meaning |
|---|---|---|
| `clipped_end` | **`fail`** | recording stopped during stim — there is no recovery epoch in this file; raise |
| `clipped_start` | `clipped_start` | recording started mid-stim; offset is valid, recovery is intact, duration is a lower bound and the tolerance check is **skipped, not passed** |
| `\|measured − stim_duration_s\| ≤ stim_tolerance_s` | `pass` | clean capture |
| otherwise | `review` | flag for manual review; do not proceed silently |

Clipped-at-start is benign and common; its measured duration is censored, so
reporting it as "within tolerance" would be a false reassurance. Clipped-at-end
means the file has nothing to analyse and must say so rather than emitting a
near-empty recovery epoch.

Expected epoch count is **1**. Report the count; more than one is a protocol
mismatch and goes to manual review. **The count is checked after both clipping
rows**, not before: `fail` and `clipped_start` are statements about whether this
file's recovery epoch is usable, and a second segment elsewhere does not make an
absent recovery epoch present or an intact one unusable. A `clipped_start` file
with two segments therefore stays `clipped_start`. The count is carried in
provenance and surfaced in the audit **regardless of status**, so the escalation
that is being declined here happens in the report instead. This is safe because
the matched filter is immune to the competing-segment failure mode (PIPELINE
§10.10) — a second segment is a protocol-conformity note, not a detection risk.

`epoch_count` is defined as the number of ON
segments anywhere in the record whose duration is at least **10% of
`stim_duration_s`** after the outward walk — i.e. it is a check on stim-like
activity *elsewhere* in the file, not a second detector. The 10% figure is a
reporting threshold, not a physical constant: its only job is to keep envelope
ripple out of the count. Record it in provenance alongside the count.

### Fix 3 — check the historical splits, because the old threshold was out of range

The old auto-threshold `(p20 + p80)/2` is only meaningful when the stim occupies
roughly 20–80% of the record. With a fixed 120 s stim:

| file duration | stim fraction |
|---|---|
| 10 min | 20% — at the very bottom edge |
| 15 min | 13% |
| 20 min | 10% |
| 25 min | **8%** |

**Every stim recording sits at or below the bottom of that range.** At 8%, both the
20th and 80th percentiles fall inside the OFF distribution, so the threshold is set
from OFF-state statistics alone and lands inside the OFF noise.

**This was measured, and the prediction that follows from it was wrong.** At 8%
duty the p20/p80 rule recovers the onset to **74 ms** — not the tens of seconds
predicted above. The rescuer is `keep only the LARGEST ON segment`: the threshold
does land in the OFF noise and does produce a spray of spurious ON segments, but
they are all short, and the one real 120 s segment wins by a wide margin. Low duty
cycle on its own is therefore **not** a failure mode of the old code, and the
historical splits are probably mostly fine.

The rule fails when something else in the record produces an ON segment *longer*
than the stim, however much quieter — sustained handling, a motor left running, a
cable rubbing on the vib monitor. Then "largest segment" selects the contaminant
and the split lands somewhere else entirely (measured: **+400 s**). The
matched-width search is immune because the width is fixed and the score is a
contrast: a long low-amplitude segment scores worse than the real one.

So the audit below is still worth running, but it is looking for a **different and
rarer thing** than originally stated: files with a competing long ON segment, not a
systematic duty-cycle bias. Expect most files to agree to well under a second.

Two residual biases *are* systematic and are what the signed statistics will show:
the truncation-biased OFF σ of Fix 1 (−1.945 s measured onset error, i.e. onset
called early, ~2 s of pre-stim baseline labelled stim) and the ±2 s refinement
clamp. Both sit inside the old edge-refinement window, so an audit that only lists
"disagreements larger than the refinement window" would report nothing while a 2 s
systematic bias was present.

**First subtask: re-split every existing `stim_recovery` recording and diff the
boundaries against the current MATLAB output.** Report **signed onset and signed
offset differences separately**, with median and IQR, not a single unsigned
boundary difference — onset error costs pre-stim baseline, offset error contaminates
the recovery reference, and only the second one damages the science. A file whose
old boundary is off by tens of seconds has had that much stim contaminating its
recovery reference — or that much recovery thrown away.

`minDurSec = 10` is deleted; it is superseded by the known width.

### Why the ordering matters — this is the real reason it must precede task 06

Task 06 computes a **whole-file percentile reference and MAD** per signal per band.
If the stim epoch is still in the array when that runs, the stim artifacts — the
largest excursions anywhere in the recording — inflate both. A raised reference and
an inflated MAD mean artifacts must be *larger* to reach `z > 3` during recovery,
so detection is suppressed exactly in the window where the post-stim dynamics being
measured actually live.

Splitting first makes the recovery reference a recovery-only statistic. This is
also why the running-window baseline was rejected (task 06): same mechanism, same
consequence.

**Measured, and the mechanism is not the one stated above.** The inflation depends
on the stim **fraction**, not on the artifact amplitude — the numbers are identical
whether the stim is 10× or 1000× σ. That follows directly from hard invariant 5:
the reference is the median and MAD of the **log** envelope, and both respond to
*how many frames* are contaminated, not how large the contamination is. A frame is
either in the stim population or it is not; making it larger moves it further out
along a tail the median and MAD already ignore.

| stim fraction | true recovery z = 3.00 reads, on the unsplit reference |
|---|---|
| 8% (the real 2 min / 20 min protocol) | **2.47** |

A frame that should clear a z > 3 gate reads 2.47 and is silently dropped. That is
the quantitative statement of why 03B precedes 06, and it means the harm is a fixed
property of the protocol rather than something that varies file to file with how
violent the stim was.

### Required behaviours
- **No `vib` / stim-monitor channel on a `stim_recovery` file → block.** Do not
  infer stim timing from the neural channels; that is circular and unnecessary
  when a monitor channel exists.
- Write recovery and stim as separate processing units, each with `t0_offset_s`
  into the original recording, so absolute time is never lost. Recovery's local
  `t = 0` is meaningful — the post-stim dynamics start there.
- **Never filter across the split boundary.** Treat it as an epoch edge: mark the
  first and last filter-settling window of each epoch `unassessable` (task 13
  measures the settling time). Until task 13 exists the settling width is
  **`None`, meaning not yet measured** — not 0, not a placeholder guess. A
  consumer that reads `None` must refuse rather than assume zero; this is the same
  rule as conventions (missing scalar = `np.nan` in memory, key absent in JSON),
  applied to a duration that has an owner in a later task.
- **Accounting: `excluded_epoch` is not `masked_motion`.** Keep them as separate
  categories everywhere — QC, retention, and the coverage-confound regression
  (task 19). A deterministic protocol exclusion counted as model-driven blanking
  would corrupt exactly the check that is supposed to catch confounds.
- **Valid-duration denominators use the recovery epoch's duration**, never the
  original file's. Every rate downstream is wrong by the stim fraction otherwise.
- The detected boundary is shown to the user with the `vib` envelope and is
  **confirmable and manually adjustable** — carry over what
  `splitStimRecoveryManual.m` does.
- Record boundaries, `stim_duration_s` used, method (`matched` / `manual`),
  detected duration, clipping flags, duty cycle, epoch count and the
  threshold cross-check in provenance.

### Tests

Two tests in the first draft did not discriminate and are replaced. Both
replacements were derived by measurement, and the originals are kept alongside as
documentation of what the old code actually does.

- a synthetic 120 s stim in a 25-minute file (8% duty cycle): the matched-width
  search recovers the onset to within one envelope window — **and so does the
  p20/p80 rule** (74 ms). Assert both. This test documents that low duty cycle is
  not the failure mode; it must not assert that the old rule fails, because it
  does not.
- **the discriminating case:** the same file plus a 300 s contaminant segment at
  a quarter of the stim amplitude. Matched-width finds the stim; p20/p80 with
  "largest ON segment" selects the contaminant and misses by +400 s. **Assert the
  old behaviour fails here**, so the fix cannot be silently reverted.
- a 95 / 113 / 140 s stim epoch: the **measured** duration is recovered to within
  150 ms in each case. This is the regression test against re-introducing a
  refinement bounded by the search width — without it, all three report 120.0 s.
- OFF statistics taken from the bottom decile of the whole envelope are worse
  than statistics taken outside the matched window **by at least a factor of
  ten**. Assert the ratio, not an absolute error. The −1.945 s originally quoted
  here was measured against the ±2 s clamped refinement and does not survive its
  removal: under the specified algorithm (`EDGE_HOLD_S = 0.25`) the same biased
  threshold costs 0.641 s, a ratio of 12.8×. Assert 0.641 s as the measured value
  and ≥10× as the binding claim.
- a 3 s dropout in the middle of the stim: one epoch, not two — **note this passes
  on the current MATLAB too**, because `removeShortSegments` bridges OFF gaps
  shorter than `minDurSec`. Keep it as a non-regression test, not as evidence of
  an improvement.
- **a 15 s dropout: one epoch.** This is the discriminating length — longer than
  `minDurSec = 10`, so the old code splits it and "largest segment" returns a
  truncated epoch, while the matched filter spans it.
- recording starts 8 s into the stim: `clipped_start = True`, offset still correct,
  recovery epoch intact
- recording stops 20 s before the stim ends: **raises**, no recovery epoch emitted
- a detected duration of 95 s with no clipping flags for review (outside ±12 s)
- a detected duration of 113 s passes (inside ±12 s) — the tolerance is not tighter
  than normal start/stop jitter
- a file with no `vib` channel and condition `stim_recovery` raises
- `stim_duration_s` is read from config, not hardcoded — changing it to 60 changes
  the search width
- the recovery epoch's band reference differs materially from the reference
  computed on the unsplit file: at the real 8% protocol, a recovery frame at true
  z = 3.00 reads **≤ 2.6** on the unsplit reference. Assert the numeric shrinkage,
  not merely that the two differ.
- the same shrinkage is obtained with the stim amplitude at 10× and at 1000× σ —
  it is a function of duty cycle alone (hard invariant 5: the reference is taken
  on the log envelope)
- `epoch_count` is 2 for a file with a second stim-like segment longer than 10% of
  `stim_duration_s`, and 1 when that segment is shorter
- a clipped-start file reports status `clipped_start`, **not** `pass`, even when
  its censored duration lands inside ±12 s
- the returned epoch arrays are not writeable, and an in-place write to one raises
- a signal whose outward walk runs past `stim_tolerance_s` sets `walk_extended`
  and reports status `review`, with the measured duration still reported as
  measured
- `EDGE_HOLD_S` at 1.0 s and 2.0 s reproduces the runaway table above — a
  regression test on the reason the hold is short, so a future widening is a
  deliberate act
- a recovery epoch 300 s short of `recovery_duration_s` sets
  `recovery_duration_flag`, and does so on a `clipped_start` file too
- a `clipped_start` file with `epoch_count = 2` reports status `clipped_start`,
  and the count still appears in the audit row
- `excluded_epoch` duration never appears in the motion-blanking fraction

### Acceptance
Run on every existing `stim_recovery` recording. Report per file: detected
duration against the expected 120 s, clipping flags, epochs found, and **signed
onset and signed offset differences** against the current MATLAB result.

**Plot the distribution of detected durations** — it should be a tight spike at
120 s with a short tail of clipped captures, and anything else is a finding.

**Plot the signed onset and offset differences as separate distributions**, and
report median and IQR for each. Do not filter to "disagreements larger than the
edge-refinement window": the biases this audit is most likely to find (≈−2 s from
the truncated OFF σ, plus the ±2 s clamp) sit *inside* that window, and a filtered
report would show nothing. A non-zero median is a finding in its own right.

Call out separately every file where the two methods disagree by more than the
stim duration — those are the competing-long-segment cases, and they are the ones
whose recovery reference has been contaminated or whose recovery data was
discarded.

**Deferred: no Drive mount on the build machine.** This acceptance run cannot
execute until the archive is reachable. It is not a blocker for the rest of the
build — record it as outstanding and run it when the mount exists.

---

**Depends on tasks:** 03, 03A

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
