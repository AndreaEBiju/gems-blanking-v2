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

The stim portion is **kept** as a separate file, not deleted. Keep that — it is the
right call, and it is what lets the deferred stim work happen later.

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
onset    = argmax score(t)
then refine both edges locally against the envelope crossing (±2 s, as today)
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

Keep the OFF-anchored threshold (`off_level + 8σ`, where `off_level` and `off_σ`
come from the bottom decile of the envelope) only for **edge refinement** and as a
reported cross-check — not as the primary detector.

### Fix 2 — duration becomes a check, not an output

| Condition | Meaning | Action |
|---|---|---|
| `\|detected − 120\| ≤ 12 s` | clean capture | pass |
| detected < 120 and ON at the **first** sample | recording started mid-stim | `clipped_start = True`; offset is still valid, recovery is fine |
| detected < 120 and ON at the **last** sample | recording stopped during stim | **fail — there is no recovery epoch in this file** |
| `\|detected − 120\| > 12 s`, not clipped | something is wrong | flag for manual review; do not proceed silently |

Clipping is detected by testing the ON state at the first and last sample, which is
exactly the case Andrea describes ("recording start/stop sometimes consumes a few
seconds"). Clipped-at-start is benign and common; clipped-at-end means the file has
nothing to analyse and must say so rather than emitting a near-empty recovery epoch.

Expected epoch count is **1**. Report the count; more than one is a protocol
mismatch and goes to manual review.

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
from OFF-state statistics alone and lands inside the OFF noise. What happens next
depends on how quiet the OFF state is, and `minDurSec = 10` plus the ±2 s edge
refinement may have rescued some files — which is precisely why this needs
measuring rather than assuming.

**First subtask: re-split every existing `stim_recovery` recording and diff the
boundaries against the current MATLAB output.** Report the distribution of
differences. A file whose old boundary is off by tens of seconds has had that much
stim contaminating its recovery reference — or that much recovery thrown away.

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

### Required behaviours
- **No `vib` / stim-monitor channel on a `stim_recovery` file → block.** Do not
  infer stim timing from the neural channels; that is circular and unnecessary
  when a monitor channel exists.
- Write recovery and stim as separate processing units, each with `t0_offset_s`
  into the original recording, so absolute time is never lost. Recovery's local
  `t = 0` is meaningful — the post-stim dynamics start there.
- **Never filter across the split boundary.** Treat it as an epoch edge: mark the
  first and last filter-settling window of each epoch `unassessable` (task 13
  measures the settling time).
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
- a synthetic 120 s stim in a 25-minute file (8% duty cycle): the matched-width
  search recovers the onset to within one envelope window, and the p20/p80 rule
  does **not** (**assert the old behaviour fails**, so the fix cannot be silently
  reverted)
- a 3 s dropout in the middle of the stim: one epoch, not two
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
  computed on the unsplit file — the quantitative statement of "why the ordering
  matters"
- `excluded_epoch` duration never appears in the motion-blanking fraction

### Acceptance
Run on every existing `stim_recovery` recording. Report per file: detected
duration against the expected 120 s, clipping flags, epochs found, and the boundary
difference against the current MATLAB result. **Plot the distribution of detected
durations** — it should be a tight spike at 120 s with a short tail of clipped
captures, and anything else is a finding. Call out every file where the two methods
disagree by more than the edge-refinement window; those are the files whose recovery
reference has been contaminated, or whose recovery data was being discarded.

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
    "1-100":    (    1.,  100.,   0.150),   #  29.7
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
    ("hrv",             "best_hr_channel",  "1-100",    "operational: beat train unchanged"),
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
