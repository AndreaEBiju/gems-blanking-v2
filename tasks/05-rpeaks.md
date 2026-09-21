<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

## Task 05 — R-peaks, gap rescue, best-channel ranking

**Module:** `physio/rpeaks.py`
**Depends on:** 03
**Gate:** no

### Purpose
Produce a beat train good enough for HRV, and choose which of the 9 channels to use
for it — replacing the manual `hrChanIdx`.

Current method (`HR_BR_HRVAnalysis_new.m:160-179`) runs `findpeaks` on the
**detrended raw** signal with the low-pass commented out (`yFilt = xFill`) and only
`MinPeakDistance`. Measured at severe motion: 361/399 beats, 57 false, RR error 1.3%.

### Signature
```python
@dataclass(frozen=True)
class BeatTrain:
    t_s:        np.ndarray                     # beat times
    tag:        np.ndarray                     # 'detected'|'rescued'
    gaps:       list[tuple[float, float, int]] # (t0, t1, m) unrecovered, m beats missing
    rescue_rate: float
    implausible_frac: float

def detect_rpeaks(x: np.ndarray, fs: float) -> BeatTrain
def rank_hr_channels(rec, trains: dict[str, BeatTrain]) -> tuple[str, pd.DataFrame]
```

### Algorithm — pass 1
1. Decimate to ~2 kHz (`ftype='fir'`, `zero_phase=True`), then band-limit
   **10–150 Hz** with `sos`.
2. `findpeaks`, prominence **`6 × MAD-σ`**, refractory `R_MIN`.

> **RULED 2026-09-21 — 10–150 Hz at k=6 is binding.** This section previously
> said 1–100 Hz at k=3; that was the earlier measurement and my correction never
> landed here. Both animal J (template SNR 611 vs 446, long-interval rate 2.0%
> vs 3.6%) and the synthetic (0 false beats vs 25 over 3160 true beats) favour
> 10–150 / k=6. In 1–100 the in-band noise floor collapses to ~0.6 µV while a
> weak beat still carries ~18 µV of prominence, so noise and signal sit on the
> same side of 3σ.
>
> **A.4's `1-100` for the hrv consumer was the same stale number, not a separate
> decision, and it moves too.** The contamination band for that consumer must be
> the band its detector actually reads. `BANDS` now carries `10-150` (100 ms
> window, 2·B·T = 28) in place of `1-100`, which is otherwise an orphan with no
> consumer. **Task 02's 1–100 peri-R row needs re-measuring at 10–150.**
3. Drop any peak closer than `0.55 × local median RR` (median over ±10 intervals).

**`R_MIN` must be measured, not inherited.** The 90 ms value is from the human ECG
literature where RR is 800–1000 ms. At rat rates it collides with the plausibility
rule and can suppress real beats:

| HR | RR | 0.55 × RR | vs 90 ms |
|---|---|---|---|
| 300 bpm | 200 ms | 110 ms | rule active |
| 400 bpm | 150 ms | 82 ms | **rule vacuous** |
| 500 bpm | 120 ms | 66 ms | refractory is 75% of RR |
| 600 bpm | 100 ms | 55 ms | refractory is **90% of RR** — deletes real beats |

**MEASURED (animal J, 10 min baseline):** HR **363–364 bpm**, RR median
**164.7 ms**, identical across all 9 channels and every band tested. The RR
histogram is a tight unimodal spike at 155–180 ms with essentially nothing real
below 145 ms. So `R_MIN = 60 ms` (0.36 × RR) is safe, and the inherited **90 ms is
not** — it sits at 0.55 × RR, on top of the plausibility threshold itself. Repeat
the histogram on ≥5 files before pinning.

### The plausibility rule must use a GLOBAL RR, not the accepted sequence

The local-median form specified above **runs away**, measured on animal J: dropping
a beat raises the local median, which raises the threshold, which drops more beats.

| frac | local median (self-referential) | global median |
|---|---|---|
| 0.60 | short 6.35%, long 2.02% | short 2.11%, long 1.52% |
| 0.70 | short 0.75%, long **10.96%** | short 0.09%, long 1.85% |
| 0.75 | **collapses** — RRmed 330 ms, 54% dropped | short 0.00%, long 2.05% |

The global version is monotone and stable to 0.85.

> **The runaway does not reproduce on the current synthetic, and that is a
> generator gap, not evidence against it.** It needs false peaks to seed the
> feedback, and `make_ecg` produces almost none. On animal J the short-interval
> population clustered at **~90 ms ≈ 0.55 × RR** — the signature of detecting the
> **T wave** as well as the R peak.
>
> **Add `make_ecg(t_wave=True)`**: a second deflection at 0.5–0.6 × RR, ~30–50% of
> R amplitude. That reproduces both the short-interval cluster *and* the runaway,
> and makes this claim testable in CI rather than only on real data.
>
> Also add a real-data regression test against
> **`gems_j_t01_ms3_bl_230315`** (animal J baseline, 601 s), where the collapse
> was observed: local form at frac 0.75 → RR median 330 ms and 54% of peaks
> dropped; global form → 3511 beats, RR 166.1 ms, short 0.00%, long 2.05%. It
> skips when the file is unreachable. Compute the threshold from
`median(RR)` over the whole file, restricted to 80–500 ms. **Operating point:
`0.75 × global RR` (≈124 ms here).** This is the same failure mode as the adaptive
QRS threshold in §10.1 — adaptive state contaminated by the artifact it is meant
to reject.

### Motion is the dominant error source — measured

Excluding the noisiest fraction of the record (by 10–150 Hz envelope) on LVN1:

| beats kept | short-interval rate | long-interval rate |
|---|---|---|
| quietest 100% | 8.16% | 2.04% |
| quietest 95% | 5.17% | 2.09% |
| quietest 90% | 3.80% | 2.05% |
| quietest 80% | **2.31%** | 2.21% |
| quietest 50% | 0.50% | 2.43% |

**False beats are motion-driven and collapse with noise; missed beats are flat at
~2% regardless.** So the residual FP population is exactly what blanking removes,
and the ~2% FN population is what the pass-2 rescue targets. This is the first
quantitative evidence on real data that the blanking pipeline improves HRV.

### Algorithm — pass 2, gap-targeted re-detection
```
for each interval d[j]:
    lo = local median RR ; m = round(d[j]/lo)
    if m < 2 or |d[j] − m·lo| > 0.20·lo:  continue      # 0.20 MEASURED TOO STRICT:
                                                        # only 18-38 of ~350 long
                                                        # intervals qualified on
                                                        # animal J. Widen, and
                                                        # re-measure once the
                                                        # global-RR plausibility
                                                        # fix lowers RR CV.
    for q in 1 .. m−1:
        win  = det[j] + q·d[j]/m  ±  0.25·lo
        cand = findpeaks(x[win], prominence = 1.2·MAD-σ,
                                 width in [0.5, 2.0] × median QRS width)
        if cand empty:  record gap as unrecovered (store m)   # insert NOTHING
        else:           take argmax correlation with this channel's QRS template
                        append, tag 'rescued'
```

Relaxing the threshold is legitimate **because the search space shrank**: pass 1 has
~2180 independent opportunities over 120 s (Bonferroni z = 4.08), pass 2 has ~30
windows (z = 2.95) — a ~28% lower threshold at the same family-wise false-positive
rate, before the width and template priors are counted.

**Never insert a fabricated beat.** Measured at 9% missed beats, 300 runs:

| handling | RMSSD | SDNN | SD1 | placement error |
|---|---|---|---|---|
| gap left in (`diff` across) | +771% | +766% | +771% | — |
| midpoint insertion | −6.3% | −4.2% | −6.3% | 2.82 ms mean |
| re-detected in window | +0.0% | +0.0% | +0.0% | 0.24 ms mean |

Midpoint insertion forces the flanking intervals equal, so their successive
difference is exactly zero — RMSSD/SD1/pNN are all successive-difference
statistics, so the bias is systematic and **downward**, the same direction as a
vagal-tone effect.

Downstream contract: rate may use `m` to count missing beats; HRV excludes every
interval touching an unrecovered gap.

### Algorithm — best-channel ranking
```
template  = mean of ±40 ms windows about the beats   (non-time-locked content
                                                      averages down as 1/sqrt(N))
SE[j]     = std across beats at sample j / sqrt(n_beats)
SNR[ch]   = peak_to_peak(template) / mean(SE)
choose argmax SNR subject to implausible_frac and rescue_rate gates
```

SNR measures **reproducibility**, which predicts HRV reliability, and it is
self-policing: a channel detecting artifacts averages to a near-flat template. Worked
values from simulation — clean 234, noisy-with-amplitude-wander 27.6, no-cardiac 4.7.

**Veto thresholds are PROVISIONAL and must be set in build-order step 9**, from
the cross-animal set, not from one animal or a synthetic. Use
`PROVISIONAL_MAX_IMPLAUSIBLE_FRAC = 0.10` and
`PROVISIONAL_MAX_RESCUE_RATE = 0.25` until then — the prefix is the point, so a
provisional number cannot quietly become a constant. For calibration: animal J
sits near 2% implausible at the good operating point and 14–20% at k=3; the
synthetic's wander channel is 28%. Raising when **every** channel is vetoed is
correct — silently returning the least-bad channel is how a bad recording enters
an analysis.

Gates (veto, not ranking): `implausible_frac` and `rescue_rate` above threshold
disqualify a channel outright. Beat count vs the median across channels is a sanity
print only.

**Re-pick per recording and log it.** A channel that stops being best is a drift
signal.

### Why no motion mask is needed first
Beats lost inside artifacts sit in spans that get masked anyway, so the information
pass 1 cannot recover is information the pipeline was going to discard. This is what
breaks the R-peak / motion circularity — do not add a dependency on step 08.

### Tests
- `make_ecg(weak_frac=0.09)`: pass 1 misses the attenuated beats, pass 2 recovers
  ≥90% of them with fiducial error <1 ms
- RMSSD/SD1 computed from the rescued train are within 2% of ground truth; the
  midpoint-insertion variant is **not** (assert the −6% bias reproduces, so the
  test fails if someone reintroduces insertion)
- an empty rescue window yields a gap record and **no new beat**
- `rank_hr_channels` on three synthetic channels reproduces the ordering above
- a pure-noise channel with artifact-driven "beats" ranks last

### Acceptance
On real data, the chosen channel matches Andrea's manual `hrChanIdx` on a majority
of recordings. **Where it disagrees, plot both and report** — do not assume either
is right.

### Do not
No adaptive threshold (`PIPELINE.md` §10.1). No insertion of fabricated beats.

---

**Depends on tasks:** 03

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
