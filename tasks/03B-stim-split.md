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

The tolerance is 12 s, not a tight few seconds: recording start/stop routinely
consumes several seconds at the edges, so a narrow band would flag normal captures.
The known **recovery** duration is a second, independent check — a file whose
recovery epoch is far from 20 min is suspect regardless of what the stim epoch
measured.

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
within 110 ms.

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
mismatch and goes to manual review. `epoch_count` is defined as the number of ON
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
- OFF statistics taken from the bottom decile of the whole envelope give an onset
  error larger than 1 s on a signal where statistics taken outside the matched
  window give better than 100 ms. Assert the second; assert the first is worse by
  at least a factor of ten.
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
    ("slow_c",          "T",                "100-300",  "own sigma"),
    ("velocity",        ("V1","V3"),        "300-3000", "peak ratio > 1"),
    ("mmc",             "stomach_ref",      "2-50",     "3 x moving MAD"),
    ("slow_wave",       "stomach_ref",      "0-2",      "peak displacement"),
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
17  08   MATLAB fixes                 ─ any time, independent
18  17   video        18 velocity      03B stim split
19  19 ◆ end-to-end acceptance
```

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

### Step 11 — labelling that doubles as the measurement

Label **complete spans of ~10 minutes total**, exhaustively, across 2–3 new
animals — not candidate adjudication. Exhaustive labelling of a short stretch is
what makes recall computable; adjudicating candidates can only measure precision.

---
