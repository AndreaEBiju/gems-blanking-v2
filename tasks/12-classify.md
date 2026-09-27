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
