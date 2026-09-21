<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

## Task 16A — Application shell and UI tree

**Module:** `detector-pyqt/ui/`
**Depends on:** 16

### Principle
Every irreversible or interpretive decision is the user's, and is **explicit**.
Every mechanical step is automatic. Anywhere the tool would otherwise pick for the
user — which model, whether recall is good enough, whether to accept a mask — it
stops and asks, and records the answer.

### The tree

```
GEMS Blanking
│
├── 1  DATASET                                    [setup, once per animal]
│     ├─ Import                     → pick TDT blocks / HDF5
│     ├─ Channel map                ▲ USER: role, cuff_id, contact_index,
│     │                                    rostral_end, cohort config
│     │                               saved per animal, reused thereafter
│     └─ Preprocessing              ▲ USER: notch list (default 60, locked
│                                          warning if harmonics added)
│
├── 2  PHYSIOLOGY                                 [automatic, user reviews]
│     ├─ R-peaks                    → per-channel beat trains
│     │    └─ Best-channel table    ▲ USER: accept ranked pick or override
│     │                               shows template SNR, implausible %, rescue %
│     ├─ Cardiac windows            → measured per channel per band
│     └─ Derivations (a, b)         → fitted; flagged if unstable vs history
│
├── 3  CANDIDATES                                 [automatic, user tunes]
│     ├─ Generate                   → candidate list at current z_enter
│     ├─ Threshold sweep            ▲ USER: pick z_enter from the sweep plot
│     │                               (recall vs count vs TP-fraction)
│     └─ Recall audit         ◆GATE ▲ USER: scroll sampled minutes, mark misses
│          └─ diagnosis             → per miss: blind spot (z low) or
│                                     threshold (z high, sub-threshold)
│
├── 4  LABEL                                      [user work — the only real cost]
│     ├─ Adjudication queue         ▲ USER: motion / physiology / unsure  (1/2/3)
│     │                               context plot + SHAP + Shift-drag to widen
│     ├─ Progress                   → judged / unjudged / by animal
│     └─ Import prior labels        → 43 recordings → events (task 10)
│
├── 5  TRAIN                                      [automatic, user configures]
│     ├─ Configure                  ▲ USER: animals to include, w_adapt sweep,
│     │                                    hyperopt budget, seed
│     ├─ Run                        → trains POOLED + ADAPTED + PER_ANIMAL
│     └─ ▶ RESULTS DASHBOARD        → task 16B
│
├── 6  INFER                                      [user chooses the model]
│     ├─ Select recordings          ▲ USER
│     ├─ Select model               ▲ USER — per animal, no default, no fallback
│     │                               picker lists every applicable model with
│     │                               its metrics AND the protocol they came from
│     ├─ Confirm assignment table   ▲ USER: animal → model, shown before running
│     └─ Run                        → events, extents, routes
│
├── 7  MASKS                                      [automatic, user accepts]
│     ├─ Per-consumer preview       → mask overlay per consumer, retention %
│     ├─ Sustained-event queue      ▲ USER: events over the duration cap
│     ├─ Recording-level gate       ▲ USER: accept or reject low-retention files
│     └─ Export                     → NaN masks + provenance + QC report
│
└── 8  MONITOR                                    [longitudinal, read-only]
      ├─ Per-animal drift           → (a,b), peri-R amplitude, noise floor
      ├─ Best HR channel history    → changes flag electrode issues
      └─ Blanking fraction by       → the coverage-confound check
         condition and by mode
```

`▲ USER` = input required, the run blocks. `◆GATE` = the step can fail and stop
the project. Everything else runs unattended.

### Screen by screen

Every screen states its **job** in one line, and every screen has a persistent
**Drive status strip** (root, sync state, unmerged registry shards, checksum
failures). A red strip blocks training and inference — see task 00A.

---

**1 · DATASET** — *job: make the recording machine-readable and say where it came
from.*

Two tabs.

**Scan & review** — point at a parent folder; it walks it, finds every recording,
and proposes **animal** and **condition** for each from the shared
`conditions.yaml` rules (task 03A). The review table shows path, animal,
condition, the rule that matched, and status, with `unknown`, `ambiguous` and
`duplicate` rows sorted to the top. Nothing is written until confirmed.

**Stim split** — for every `stim_recovery` recording: the `vib` envelope with the
detected stim epoch shaded, the measured duty cycle, the epoch count, and
draggable boundaries. The stim epoch is **kept on disk** and excluded from all
downstream processing; recovery proceeds alone.

**Channel map** — per-channel role, `cuff_id`, `contact_index`, `rostral_end`,
cohort and notch list, with a sparkline preview. Saved per animal and reused; a
later recording only asks if the channel count changed.

| You must | Notes |
|---|---|
| resolve every `unknown` / `ambiguous` condition | **nothing defaults to `baseline`** — unresolved rows cannot enter a corpus |
| confirm or drag the stim/recovery boundary on every `stim_recovery` file | a file with no detected epoch **fails loudly** rather than passing through as pure recovery |
| correct any wrong proposal, and optionally save it as a rule | the rule preview shows how many other recordings it would also match |
| assign role, `cuff_id`, `contact_index` per channel | inferred from names, always confirmed |
| enter `rostral_end` | **no default** — absent means unsigned velocity forever |
| confirm cohort (`hw_tripole` / `independent`) | inferred from channel count |
| confirm notch list | `60` only; adding 120 shows the in-band ringing warning |

---

**2 · PHYSIOLOGY** — *job: establish the beat train and measure the cardiac window
before anything else touches the data.*

Contains: per-channel beat-detection table (template SNR, implausible %, rescue %,
beat count vs median), the ranked best-HR-channel pick, the peri-R band profile
per channel per band, and the fitted `(a, b)` with its history for that animal.

| You must | Notes |
|---|---|
| accept or override the ranked HR channel | override is recorded and shown in QC |
| glance at the peri-R profiles | a window appearing above 300 Hz contradicts the model and should stop you |

Everything else here is automatic.

---

**3 · CANDIDATES** — *job: get recall high enough that the classifier is worth
training. This screen can fail the project.*

Contains: the threshold sweep (recall / count / TP-fraction vs `z_enter`), the
candidate list, and **recall-audit mode** — a scroll over randomly sampled minutes
with candidates overlaid *and the band z-traces beside the raw trace*.

| You must | Notes |
|---|---|
| pick `z_enter` from the sweep | default 3.0, defensible 2–4 |
| scroll the audit samples and mark missed artifacts | the screen will not report a recall number from zero reviewed samples |
| classify each miss: blind spot or threshold | z low → new feature needed; z high but sub-threshold → lower the threshold. **The z-traces exist to make this distinguishable** |

◆ **Gate.** Below the recall target, stop and read task 09 before continuing.

---

**4 · LABEL** — *job: produce the training signal. The only screen that costs real
time.*

Contains: the adjudication queue (context plot, band z-traces, SHAP once a model
exists), progress by animal, and an importer that replays candidates over the 43
previously-marked recordings.

| You must | Notes |
|---|---|
| judge each candidate: motion / physiology / unsure | one keystroke — 1 / 2 / 3 |
| widen a boundary where the extent is visibly wrong | Shift+drag, as today |

Budget: ~500–1000 events across ~12 recordings, **1–3 hours once**; then ~50 per
new animal. `unsure` is excluded from training, and a span nobody judged stays
`unjudged` — never a negative.

---

**5 · TRAIN** — *job: choose exactly what the model learns from, and make that
choice reproducible.*

This screen inherits the best part of the existing `training_window.py` — the
per-animal `held_out` checkboxes and "Only animals" filters — and makes it
explicit and shareable.

Contains, as tabs:

- **Corpus builder** — one row per recording: animal, session, duration, judged
  events, positive fraction, **condition**, last trained on, and a three-way
  `train / held_out / excluded` control. Filters by animal, **condition**, cohort
  and date. Recordings whose condition is `unknown` or `ambiguous` are shown but
  **not selectable** — resolve them on screen 1 first.
  Bulk actions ("hold out all of animal O"). Saves as a **named corpus spec**
  (`corpora/<corpus_id>.json`) that anyone in the lab can load and reuse.
- **Configure** — modes to train (all three by default), `w_adapt` sweep set,
  hyperopt budget, seed.
- **Run** — progress and log, as today.
- **Versions** — the registry table, carried over from the existing Versions tab:
  `model_id`, mode, animal, corpus, created, created_by, and its held-out metrics
  **with the protocol printed**. Provenance JSON viewer. Promote / demote writes a
  line to the append-only log (task 00A).

| You must | Notes |
|---|---|
| set `train / held_out / excluded` per recording | **`excluded` requires a reason string** — an exclusion without a recorded reason is how a corpus becomes indefensible |
| name and save the corpus spec | immutable once used; editing creates a new id |
| choose which modes to train | default all three |
| promote a model, or not | never automatic |

---

**6 · INFER** — *job: apply a model you consciously chose.*

Contains: recording selector, the **model picker**, and the assignment table.

The picker lists every applicable model with `mode`, `corpus_id`, `created_by`,
its metrics **and the protocol those metrics came from**, and an `unvalidated`
flag where there are no held-out numbers. It has **no default selection and no
fallback**.

| You must | Notes |
|---|---|
| pick recordings | |
| pick a model **per animal** | the run will not start otherwise |
| confirm the animal → model assignment table | shown before anything executes |

A batch spanning animals with different modes is allowed and recorded, and the QC
report flags it — mode then varies across the dataset and becomes a covariate.

---

**7 · MASKS** — *job: decide whether the output is good enough to keep.*

Contains: per-consumer mask overlay with retention %, the sustained-event queue
(events over the duration cap, never auto-masked), the recording-level retention
gate, and the export panel.

| You must | Notes |
|---|---|
| adjudicate sustained events | these are level shifts, not events |
| accept or reject low-retention recordings | a heavily-masked recording should not leave silently |
| choose which consumers to export | all, by default |

Export writes NaN masks, the event table, the QC report and full provenance. It
never modifies a source file.

---

**8 · MONITOR** — *job: notice the electrode degrading before it ruins a cohort.*

Read-only, longitudinal, per animal: fitted `(a, b)` over sessions, peri-R template
amplitude, noise floor, best-HR-channel changes, and blanking fraction **by
condition and by mode**.

| You must | Notes |
|---|---|
| nothing — but check it weekly | a changing best-HR channel or drifting `(a,b)` is an electrode telling you something |

The blanking-fraction-by-condition plot is the coverage-confound check (task 19)
and is the single most important panel here scientifically.

---

### Where the user can adjust things — the complete list

| Screen | Control | Default | Consequence of changing it |
|---|---|---|---|
| Dataset | channel roles, `cuff_id`, `contact_index` | inferred from names | wrong → derivations and velocity are wrong |
| Dataset | `rostral_end` | **none — must be entered** | absent → unsigned velocity + warning |
| Dataset | notch list | `60` | adding 120 rings *inside* the ENG band at 2.41 µV/mV |
| Physiology | HR channel | ranked pick | override recorded and shown in QC |
| Candidates | `z_enter` | 3.0 (range 2–4) | ↓ recall ↑, count ↑; ↑ the reverse |
| Candidates | duration cap | p99 of prior labels | above it → review queue, never auto-masked |
| Label | judgement | — | the training signal |
| Train | **`train` / `held_out` / `excluded` per recording** | all `train` | the corpus spec; exclusions need a reason |
| Train | modes to train | all three | |
| Train | `w_adapt` sweep | {1,3,10,30} | |
| Train | hyperopt budget, seed | | reproducibility |
| Train | promote / demote | — | append-only, reversible |
| Infer | **model per animal** | **none** | the run will not start without it |
| Masks | accept / reject recording | — | |
| Masks | consumer subset to export | all | |

### Hard UI rules
1. **No silent defaults on anything interpretive.** Model choice, recall
   sufficiency, and recording acceptance all block on the user.
2. **Every screen states what it will do before doing it**, and shows the inputs it
   will use.
3. **Any override is recorded in provenance** with who/when, and surfaces in QC.
4. **Nothing is destructive.** Masks are new files; the source is never modified.
5. **The gate screens cannot be skipped by clicking through.** The recall audit
   requires marked-or-confirmed samples before it will report a recall number.

### Tests
- launching inference without a model choice is impossible (asserted at the
  controller, not just disabled in the widget)
- an override of the HR channel appears in the exported provenance
- the recall-audit screen refuses to emit a number from zero reviewed samples

### Acceptance
A new user can go from raw TDT block to exported masks following the tree, with
every blocking decision visible and explained on the screen where it is made.

---

**Depends on tasks:** 16

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
