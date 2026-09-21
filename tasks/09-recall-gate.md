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

### Method A — synthetic injection (secondary)
Inject artifacts of known type, amplitude and duration into real clean recordings.
Reuse the Phase 2 synthetic machinery in `GEMSBlanking` — it moves from a validation
figure to a runtime component. Sweep amplitude ratio 0.5–20× and duration 5 ms–5 s
across all four artifact kinds. Report a recall surface, not a scalar.

### Method B — human recall audit
Sample a few minutes from each of ≥6 recordings, have them scrolled in full (task
16's audit mode), and count artifacts a human finds that no candidate covers.

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
