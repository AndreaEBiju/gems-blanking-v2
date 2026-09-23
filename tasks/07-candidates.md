<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

## Task 07 — Candidate generation

**Module:** `detect/candidates.py`
**Depends on:** 05, 06 for behaviour; **02 for types only**
**Gate:** no (but task 09 measures whether it worked)

> **The old header said "Depends on: 02, 05, 06" and contradicted A.6**, which
> puts 02 after 07. A.6 is right and the header was sloppy: 02 needs real
> recordings to measure a peri-R window, 07 needs only the *shape* of what 02
> returns. Build 07 against task 02's declared return type and accept
> `cardiac_windows=None`.
>
> **`None` means "not yet measured" and must be recorded as such.** It does not
> mean "no cardiac contamination" — those are different claims and only the
> first is true today. The report says which.

**Downstream consumers take the `CandidateReport`, not the bare list.** The
over-cap routing and the `cardiac_windows=None` state live on the report, so a
stage handed only `list[Candidate]` has silently lost both. Task 15 in
particular must never auto-mask an over-cap candidate, and it cannot know which
those are from the list alone.

### Signature
```python
def generate_candidates(
    z: dict[tuple[str,str], np.ndarray], beats: BeatTrain,
    cardiac_windows: dict, video: VideoMotion | None = None,
    z_enter: float = 3.0, z_exit: float = 1.5,
    min_dur_s: float = 0.020, merge_gap_s: float = 0.100,
    duration_cap_s: float | None = None,
) -> list[Candidate]
```

### Algorithm
```
enter on  z > z_enter
exit  on  z < z_exit                       # hysteresis
discard   duration < min_dur_s
merge     gaps < merge_gap_s
if duration > duration_cap_s:  route to review, never auto-mask
suppress inside cardiac_window — 100–300 Hz ONLY
where video motion exceeds its own threshold:
    also enter on z > 2.0, tag provenance='video_assisted'
```

`duration_cap_s` = the **99th percentile of the labelled segment durations in the
existing 43 recordings**. Compute it once from `*_segment_indices.mat` and pin it;
do not guess. Anything longer is a sustained level shift, not an event.

**The labels are not on this drive.** A full walk of all 3442 directories under
`<gems_root>` on 2026-09-22 found **zero** `*_segment_indices.mat`. This is no
longer "the archive is unreachable" — the archive is mounted and the files are
not in it. Before concluding they are lost, search for the companion
`*_segments.mat` that `browseMotionArtifacts` writes, and the sibling folders on
the shared drive (`GEMS-Lyna`, `Louise`, `Arjun`).

Until they are found it is `None`, meaning **no cap was applied and
that fact is recorded** — provenance key absent, per the conventions table,
never null and never a stand-in infinity. An empty `over_cap` under `None` means
no cap ran; it must not be read as "nothing exceeded the cap".

### Threshold
`z_enter = 3.0` is a starting value; the defensible range is 2–4, bounded below by
class balance (≥5% true positives — with ~150 real artifacts per recording that
means ≤3000 candidates, i.e. flag ≲3% of frames) and above by recall. **Sweep it in
task 09 and pin it there**, do not tune it here.

### How (signal, band) pairs combine — state it explicitly

The signature takes z for **every** pair and the algorithm above does not say how
they are combined. Left implicit it becomes a union over ~36 pairs, which hard
invariant 10b names as a multiple-comparisons problem: at a per-pair rate of
0.2–3.4%, the union runs 40–55%. Make the rule an explicit parameter, because
task 09 has to sweep it alongside `z_enter`:

```
combine: "any" | "k_of_n"      # default "any"
k: int = 1                     # ignored unless combine == "k_of_n"
```

`"any"` is the recall-maximising default and this step's only job is recall — so
it is the right starting point, not the right answer. Record the per-pair and
union flag rates in the candidate report so task 09 sweeps against measured
numbers rather than the 0.2–3.4% estimate.

Measured on 36 independent clean pairs at `z_enter = 3.0`: worst pair
**0.13%**, union **4.6%** — a factor of **35**. Note what that measurement is
and is not. Synthetic pairs are *independent*; real `(signal, band)` pairs are
correlated, because one artifact lands in several bands at once, so for a given
per-pair rate the real union is **lower** than the independence prediction.
Real per-pair rates are also higher than 0.13%. The two effects push opposite
ways and neither is small, so **task 09 re-measures both on real recordings**
and the 35× is the independent-case bound, not a forecast.

Cross-band coincidence is **not** a suppression rule here. It is a classifier
feature (task 11); using it to gate candidates would destroy the evidence task
12 needs.

### Rules the algorithm left open, now closed

**NaN frames are no evidence, not evidence of quiet.** Task 06 emits `nan` for
unassessable frames (settling edges, short segments). A NaN can neither open nor
sustain a candidate. But `merge_gap_s` **does** bridge across one: a gap is not
evidence of a second event whether it is quiet or unmeasured, and the two error
directions are not symmetric — bridging over-masks a span nobody assessed, while
refusing to bridge leaves unassessed samples unmasked next to a known artifact.
For a gap under 100 ms the first is clearly the right way to be wrong.

**`video_assisted` marks a span that exists *because* the bar was lowered** —
no pair reached the unassisted `z_enter` anywhere in it. A span that would have
been found with the camera off is `electrical` whether or not the animal moved,
which keeps the tag a record of evidence rather than of coincidence. Caveat to
carry into task 13: a span can be `electrical` and still have its **extent**
influenced by the lowered threshold. Existence and extent have different
provenance; 13 owns extent and should not read `electrical` as "no video input".

**`VideoMotion` is a structural `Protocol` declared here**, because task 17
describes ROIs, sync and drift but never names a type. Minimum shape: a motion
trace on the shared 10 ms grid, plus its own threshold. **Task 17 must satisfy
this protocol**, not redefine it.

### This step's only job is recall
Precision is task 12's job. Candidate count is **not** review burden — the
classifier judges every candidate and humans label a sample.

### Tests
- injected artifacts at known times: every one produces a candidate whose span
  contains the injected span
- hysteresis: a z-trace dipping to 2.0 mid-event yields one candidate, not two
- a cardiac-only synthetic yields no candidates in `100-300` and **does** yield
  them in `300-3000` if a real artifact is present there (proves suppression is
  band-scoped). **The artifact must sit entirely inside one peri-R window** or
  the test does not discriminate: a 400 ms artifact against a 40 ms window
  survives band-wide suppression too. This matters at the real rate — at animal
  J's 364 bpm a beat arrives every 165 ms, so 40 ms windows cover **24%** of the
  recording, and band-wide suppression would silently discard a quarter of all
  short ENG artifacts in a band where A.5 measures the QRS carrying
  0.00–0.54% of its energy. *The band name here read `300–5000` until 2026-09-22; A.5b moved
  the ENG band and this line did not follow. Band names are the exact strings in
  A.3, everywhere, always.*
- duration cap routes to review rather than dropping
- `video_assisted` provenance is set only where video exceeded threshold

### Acceptance
Candidate count per recording and the fraction of frames flagged, reported per
animal, at the pinned threshold.

### Do not
Do not add rate targeting (`PIPELINE.md` §10.3).

---

**Depends on tasks:** 05, 06, 02

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
