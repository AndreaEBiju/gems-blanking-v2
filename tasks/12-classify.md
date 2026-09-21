<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

## Task 12 — Classifier: three training modes

**Module:** `model/train.py`, `model/evaluate.py`, `model/modes.py`
**Depends on:** 10, 11

### Reuse wholesale (import, do not fork)
`GEMSBlanking:detector/retrain.py` (LightGBM + hyperopt),
`detector-pyqt/ui/workers/hyperopt_worker.py`, `detector/review.py` (SHAP HTMLs),
`detector/heldout_eval.py`, `detector/animal_id.py:extract_animal_letter`, and the
existing promotion / rollback / `current_model` pointer / `provenance.json`
machinery. Per-animal model support already exists in `GEMSBlanking` — extend it,
do not rebuild it.

### Target and weighting (all modes)
- **Target**: one judgment per candidate event — `motion` / `physiology` /
  `unsure`. Not a per-window label. `unsure` and `unjudged` are excluded from
  training *and* from scoring.
- **Weight `provenance='video_assisted'` positives up.** Boundary examples by
  construction — the electrical threshold missed them, so their signature is weak —
  and they are the route by which video improves performance on **video-less**
  recordings. **Measure the cost**: precision on video-less recordings with and
  without the upweighting. Report both.
- **No temporal smoothing** (`PIPELINE.md` §10.4).

### Baseline first
Before training anything, run a **fixed-threshold baseline**: classify every
candidate as motion. Then a single-feature threshold on the 100–300 ÷ 300–5000
ratio. Report both. **If no learned mode beats them, ship the threshold.** This is
the cheapest possible outcome and must be ruled out explicitly.

---

### The three modes

The same features, the same candidate definition, the same hyperparameter search.
**Only the training corpus and the evaluation protocol change.**

```python
class TrainingMode(StrEnum):
    POOLED    = "pooled"      # all animals, target animal fully excluded
    ADAPTED   = "adapted"     # pooled prior + target animal's own labelled events
    PER_ANIMAL= "per_animal"  # target animal only
```

| Mode | Training corpus | Applies to | Protocol |
|---|---|---|---|
| **A · POOLED** | every animal except the target | a brand-new animal with **zero** labels | LOAO |
| **B · ADAPTED** | pooled + the target animal's labelled events, upweighted | a new animal after ~50 labels | LOAO-then-adapt, held-out recordings of the target animal |
| **C · PER_ANIMAL** | the target animal only | that animal only | LORO within animal |

**Mode A is mandatory.** It is the only mode that can process an animal with no
labels at all, so it must exist regardless of what the comparison shows. Modes B
and C are optional and are justified only by beating it.

### Mode B is the one that matches deployment, and it is missing from the two-mode framing

The labelling budget already assumes **~50 judged events per new animal**. So the
real deployment scenario is **few-shot**, not zero-shot: by the time a new animal's
data is processed, some of its labels exist. Mode A measures a harder problem than
the one actually faced, and mode C throws away the other animals entirely. Mode B
is the one that uses everything available.

Implement B as: train the pooled model, then continue training (LightGBM
`init_model=`) on a corpus of pooled events plus the target animal's events at
weight `w_adapt`. Sweep `w_adapt` over `{1, 3, 10, 30}` and report the curve —
do not pick a value by intuition.

**Strict separation:** the target animal's events used for adaptation must never
appear in that animal's evaluation set. Split the target animal's labelled events
by *recording*, not by event, so no recording contributes to both.

---

### How to compare the modes — read this before reporting any number

**Comparing modes A and C directly is invalid**, and this is the main risk in the
proposal. They are evaluated on different tasks:

- mode A (LOAO) predicts on an animal it has **never seen**
- mode C (LORO) predicts on a **different recording of an animal it knows**

LORO is the easier task, so mode C will look better whether or not it is better.
Any comparison must hold the protocol fixed.

**The only valid comparisons:**

| Question | Compare | Protocol held fixed |
|---|---|---|
| Is there animal-specific structure worth capturing? | B vs C | both on held-out recordings of the target animal |
| What does a new animal cost us? | A vs B | both on held-out recordings of the target animal |
| Is pooling actively harmful? | A vs C | **not directly comparable — do not report this pair** |

**The decisive comparison is B vs C.**
- If **B ≥ C**, the pooled prior is worth keeping, mode C ships nothing, and you
  maintain one model plus an adaptation step instead of N models.
- If **C > B** materially, that is *not* a reason to ship per-animal models. It
  means the pooled data is actively hurting, which can only happen if the features
  are **not animal-invariant** — i.e. a task 11 bug. Investigate task 11 first, via
  SHAP on the features that differ most between the two.

### Corpus-size imbalance — normalisation does not solve this

Feature normalisation (z-scoring, ratio features, animal-invariant construction)
addresses **covariate shift**: features sitting at different scales across animals.
It does nothing about **training set size**, which is a *variance* problem, not a
*scale* problem. A mode-C model trained on one animal's ~10 recordings sees roughly
1/N the events of the pooled model and will sit at a different point on its
learning curve. Any A-vs-C or B-vs-C difference is then confounded by corpus size.

**Required control: the learning curve.** Subsample the pooled/adapted training set
to the *same event count* as the per-animal set and retrain. Report:

```
performance vs training events, per mode, per animal
  x: n_training_events, log-spaced, >=5 points
  y: LOAO / within-animal F1 with bootstrap CI
```

If the per-animal curve lies on the pooled curve at equal event count, the
difference was corpus size and there is no animal-specific structure. If it lies
above, there is. **This plot is the deliverable that answers the colleague's
question**; the headline F1 numbers do not.

Also report, per animal: labelled event count, positive fraction, and recording
count — modes cannot be interpreted without them.

### Calibration
Probabilities from models trained on different corpora are **not comparable**.
Task 13 thresholds on `P(motion)`, so every mode must be calibrated on held-out
data (isotonic or Platt) before its probabilities are used, and the calibration
must be fitted per mode per animal. An uncalibrated mode-C model will mask a
different amount of data than mode A at the same nominal threshold, and that
difference will look like a detection difference.

### Prior evidence — state it in the report
Andrea has already run per-animal training: it **performed worse than pooling all
animals together**. That is direct evidence against mode C, with one caveat — it is
not known whether the protocols were matched at the time, so it may have been the
invalid A-vs-C comparison above. The task is to redo it correctly, not to assume
either answer.

### Validation protocol
**Leave-one-animal-out, reported per animal, never averaged.** LORO measures
within-animal generalisation; the stated goal is cross-animal, so LORO reads
optimistic. Report both where a mode requires it, but the shipping decision for
mode A is LOAO.

The 4 new unlabelled animals are a **one-shot prospective test set**. They must be
labelled **blind, before any model sees them**. Do not iterate against them, and do
not use them to choose between modes.

### Tests
- the training loader drops `unjudged` and `unsure` (regression guard on task 10)
- LOAO folds contain no animal on both sides — assert by animal letter
- **mode B leakage guard**: assert no recording of the target animal appears in
  both the adaptation corpus and the evaluation set. This is the easiest mistake in
  the whole task and it inflates mode B exactly where the comparison matters
- mode C training corpus contains exactly one animal letter
- a deliberately leaky feature (recording index) is rejected by a leakage check
- calibration: post-calibration reliability curve within tolerance on held-out data

### The corpus spec is an explicit, named, immutable object
Training never takes "all the data". It takes a **corpus spec** built by the user
on the Train screen (task 16A) and written to `corpora/<corpus_id>.json`:

```json
{ "corpus_id": "c_2026-09-18_baseline43",
  "created_by": "andrea", "created_at": "...",
  "recordings": [ {"animal":"F","session":"...","role":"train"},
                  {"animal":"O","session":"...","role":"held_out"},
                  {"animal":"L","session":"...","role":"excluded",
                   "reason":"electrode failure wk3"} ],
  "notes": "..." }
```

`role` ∈ `{train, held_out, excluded}`, per **recording**, not per animal — this
preserves the flexibility of the existing per-animal tab's `held_out` checkboxes
while making the choice reproducible and shareable. An `excluded` recording
**requires a reason string**; exclusions without recorded reasons are how a corpus
quietly becomes indefensible.

The spec is immutable once used by a training run; editing produces a new
`corpus_id`. Every model's provenance names the `corpus_id` it was trained on, so
two lab members can tell whether they trained on the same data.

### Train all three, always
A training run trains **all three modes** for the selected animals in one pass and
emits the comparison artifact below. Training one mode in isolation is allowed for
iteration but does not produce a shippable model — the comparison is part of the
deliverable, not an optional follow-up.

### The comparison artifact
`model/compare.py` writes `comparison_<timestamp>.parquet` + a rendered report
containing, per animal:

| Output | Content |
|---|---|
| **Learning curves** | F1 vs `n_training_events`, log-spaced, ≥5 points, one line per mode, bootstrap CI. **The deliverable that answers the per-animal-vs-pooled question.** |
| Matched-protocol table | B vs C on held-out recordings of the target animal; A vs B likewise. A-vs-C present but explicitly marked `not comparable` |
| Corpus table | labelled events, positive fraction, recording count, per animal per mode |
| Calibration | reliability curve per mode |
| Verdict | B ≥ C, or C > B with the task 11 investigation flagged |

This feeds the dashboard in task 16B; it must be readable as a file on its own too.

### Acceptance
1. Per-animal, per-mode precision / recall / F1 against both baselines.
2. The learning-curve plot, per mode per animal.
3. The B-vs-C verdict, with the task 11 investigation triggered if C > B.
4. Calibration curves per mode.
5. SHAP review HTML for the top features of the pooled model.
6. `provenance.json` recording mode, corpus composition, and `w_adapt`.

### Do not
Do not report an A-vs-C comparison as if it were meaningful. Do not ship mode C
without the learning-curve control. Do not tune `w_adapt` against the prospective
test animals.

---

**Depends on tasks:** 10, 11

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
