# gems-blanking-v2 — granular implementation instructions

Authoritative build document. `tasks/NN-*.md` are **generated** from this file by
`python split_tasks.py`; edit here and regenerate.

**`split_tasks.py` is not shipped with this document — the copy in the repo is
canonical.** It was reverted once by a doc drop that carried an older copy, and
now that the repo's own tooling is bound by the cross-platform contract, that
would land as a red CI row rather than a silent regression. Doc updates replace
`IMPLEMENTATION.md`, `CLAUDE.md`, `PIPELINE.md` and `PROMPTS.md` only.

Read `CLAUDE.md` first — its invariants apply to every task and are not repeated
per task. Read `PIPELINE.md` for why each decision was made; this document says
only what to build.

**Ordering is a gate structure, not a preference.** Tasks 01, 02 and 09 are gates:
work after them is wasted if they fail. Task 09 can invalidate the entire
architecture, and the correct response is to stop and build the U-Net fallback, not
to relax the gate.

---

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

## Part B — tasks

<!-- TASK:00 slug=repo-setup deps=none gate=no -->
## Task 00 — Repository skeleton and test harness

**Module:** package root, `tests/conftest.py`
**Depends on:** nothing
**Gate:** no

### Purpose
Create the package and the synthetic-signal generators every later test depends on.
Doing this first means no task has to invent its own test data.

### Build
- `pyproject.toml`: python ≥3.11, `numpy scipy pandas h5py pyarrow lightgbm shap
  matplotlib opencv-python-headless pytest ruff mypy`
- package dirs per `CLAUDE.md`, each with `__init__.py`
- `ruff` and `mypy` config, both strict enough to fail CI
- `gems_blanking_v2/constants.py` holding `GRID_S`, `BANDS`, `REFERENCE_STATISTIC`,
  `CONSUMERS` and the Part A.5 table as module-level constants
- **`gems_blanking_v2/types.py` holding the Part A.1 dataclasses verbatim.**
  Create them here, not in task 03: `make_multichannel` must return a `Recording`,
  so the types are needed before the loader exists. Task 03 imports them and must
  not redefine them.
- **All packages live under `gems_blanking_v2/`** — `gems_blanking_v2/io/`, not a
  top-level `io/`, which would shadow the stdlib module of that name.

### `tests/conftest.py` — required fixtures

```python
def make_beats(fs, dur_s, rr_s=0.150, sd_rr_s=0.005, seed=0) -> np.ndarray
    """Ground-truth R times, seconds. Realistic rat HRV.
    The drawn intervals are affine-corrected so the REALIZED mean and SD equal
    rr_s and sd_rr_s exactly -- 2% accuracy on an SD estimate needs ~1250 beats
    (relative SE = 1/sqrt(2n)) and a 120 s file holds ~800. Consequence: the
    intervals are not an i.i.d. draw, so do not use this fixture to test an
    estimator's own sampling behaviour."""

def make_qrs(fs, width_ms=10.0) -> np.ndarray
    """Triphasic kernel: central positive lobe with two small symmetric negatives.
    IMPORTANT: the flanking lobes must be <=0.2 of the peak, else the kernel's own
    rebound is detected as a second peak and every beat is counted twice."""

def make_ecg(fs, dur_s, amp_uv=60.0, weak_frac=0.0, noise_uv=5.0, seed=0) -> (sig, beats, weak_idx)
    """Beat train convolved with the kernel, on a noise floor.
    noise_uv is REQUIRED for the weak-beat claim to mean anything: with no noise
    MAD -> 0 and nothing is below any multiple of it. At noise_uv=5, a 3x MAD-sigma
    prominence is ~15 uV, above the 8-13 uV attenuated beats and well below the
    60 uV normal ones. Test that relationship; do not assert it in prose.
    NOTE the real-data threshold is 6x MAD-sigma in 10-150 Hz (task 05), not 3x."""

def make_eng(fs, dur_s, rate_hz=20.0, spike_uv=80.0, noise_uv=6.0, seed=0)
    """Poisson spike train, biphasic 0.6 ms waveform, white noise."""

def make_slow(fs, dur_s, freq_hz=0.05, amp_uv=200.0, seed=0)
    """Gastric slow wave."""

def inject_artifact(sig, fs, t0_s, dur_s, kind, amp_ratio, seed=0)
    """kind in {'excursion','drift','step','tribo','clip'}.

    amp_ratio is DEFINED as  max|out - in| over the span, divided by the host's
    MAD-sigma measured BEFORE injection. The first four kinds are purely additive
    so that definition is exact and the ground truth is recoverable.

    'clip' is the exception and is deliberately NOT additive: it hard-limits the
    host at +/- amp_ratio * MAD-sigma. Task 14 routes clipping around the
    classifier precisely because saturation is non-linear and the envelope can
    UNDERSTATE the damage -- an additive flat-top would not exercise that path.
    For 'clip' the ground truth is the span and the rail value, not amp_ratio.

    Returns (sig, truth_span). 'step' is the additive flat-topped version that
    used to be called 'saturation'; the rename exists so the two are not
    confused."""

def make_multichannel(fs, dur_s, n_cuff=2, n_stomach=3, common_mode=True, seed=0)
    # returns a Recording from gems_blanking_v2/types.py (created by THIS task,
    # imported by task 03 -- do not redefine the A.1 dataclasses there)
    #
    # Injects ADDITIVE kinds only. A clip rail belongs to one amplifier channel,
    # so it cannot live in a common-mode trace that is then scaled per channel --
    # clipping the shared component and multiplying it by 0.8 and 1.25 models
    # nothing physical. A test needing a saturated channel clips one column of
    # rec.data directly.
    """Full synthetic recording with a known common-mode artifact component,
    per-contact gains, and per-contact independent neural content.
    Returns (Recording, ground_truth_dict)."""
```

### Measured properties to preserve

- **Pass-2 false-positive rate on genuinely empty windows: ~2.8% (1 in 36).** The
  rescue returns only peaks the trace actually contains — it never fabricates a
  sample — but a relaxed 1.2σ threshold plus a width prior plus a template argmax
  will occasionally all pass on noise. This is the price of the relaxed
  threshold and is reported, not engineered away with a correlation floor the
  design does not specify.
- **`PASS2_MULTIPLE_TOLERANCE = 0.40` is a judgement, not a measurement.** At the
  synthetic's 3.3% RR CV it is indistinguishable from 0.20 (161/161 admitted
  either way); it only bites at realistic spread — 10% CV: 78% → 98%, 20% CV:
  52% → 85%. Test the *rule's* behaviour across CVs; do not assert the constant.
- **`weak_frac` in `make_ecg` is calibrated against the BROADBAND MAD, and
  detection band-limits.** Band-limiting moves σ by 4–9× (0.62 µV in 1–100,
  2.89 µV in 10–150, 5.46 µV broadband), so an "8–13 µV weak beat" lands ~30×
  above the in-band threshold and cannot be missed. Use `weak_amp_uv` to
  calibrate against the band actually detected in. The trap is that
  band-limiting is *precisely what makes weak beats detectable*.

### Tests
`tests/test_constants.py` — **every band named in `CONSUMERS` is a key of
`BANDS`.** This one line would have caught the 300-5000/300-3000 contradiction
that survived in this document; add it before anything else.

`tests/test_conftest.py` — each generator reproduces its own spec: `make_beats`
returns the requested mean RR and SD to within 2%; `make_qrs` has exactly one local
maximum above 0.2 of peak; `inject_artifact` returns a span whose measured
amplitude ratio matches `amp_ratio` to within 10%.

### Acceptance
`pytest` green, `ruff` and `mypy` clean on an empty package.

### Do not
Do not skip testing the generators. A generator bug produces confident wrong
numbers in every downstream test, which is exactly how the QRS double-peak and the
unstable `butter` bugs were introduced during design.
<!-- /TASK -->

<!-- TASK:00A slug=drive-store deps=00 gate=no -->
## Task 00A — Shared storage on Google Drive

**Module:** `io/store.py`, `io/registry_log.py`
**Depends on:** 00

### Purpose
Data, per-trial metadata and trained models all live in a **synced Google Drive
folder** so anyone in the lab can clone the repo and immediately have both the
data and the current models. The repo holds **code only** — no data, no models, no
labels.

```
git clone <repo>            →  code
Google Drive (synced)       →  data + labels + models + registry
config: gems_root = <path>  →  the one thing each user sets locally
```

### The root, decided 2026-09-22

**`<gems_root>` = the `GEMS-Andrea` folder on the shared drive.** Measured
spellings, which differ and must never be reconstructed from each other:

```
Windows  G:\Shared drives\BIONICs Lab  Enteric Interfaces Team\GEMS-Andrea
macOS    /Users/<user>/Library/CloudStorage/GoogleDrive-<acct>/Shared drives/BIONICs Lab: Enteric Interfaces Team/GEMS-Andrea
```

Google Drive substitutes **U+0020 SPACE** for the illegal colon on Windows,
giving **two consecutive spaces**. Not U+2236, not U+F03A, not an underscore —
four plausible guesses, all wrong, which is the argument *for* rule 2 rather
than against it: the difference is confined to the root, the root is found by
marker, and a stored path such as
`data/J/t01_bl_230315/gems_j_t01_ms3_bl_230315_sig.mat` round-trips
`rel → abs → rel` byte-identically on both machines.

### Root is not scan scope — keep them separate

`GEMS-Andrea` contains `processing_new`, `TDTMatlabSDK`, `nerve-processing` and
`IACUC Inspection 092026`: code and admin, not data. **The root is a path
anchor; it is not a licence to walk everything under it.** Config carries an
explicit list:

```yaml
gems_root: <the path above>          # the anchor for every stored path
scan_roots:                          # the ONLY places recordings are looked for
  - "August-September Chronic Recordings"
  - "081526BalloonTrial"
  # ... listed explicitly, POSIX-relative to gems_root
```

A recording outside every `scan_root` is not in the corpus. Adding a folder is
an edit to this list, never an automatic consequence of someone dropping files
on the drive. This also keeps the `SHARED_DRIVE_ITEM_CAP` accounting honest,
since the code trees stop counting toward it.

### Measured cost of the drive, and what it forbids

Streaming (not mirrored), one 983 MB `_sig.mat`:

| operation | time |
|---|---|
| `stat` | 0 ms |
| first 10 MB, cold | 1.12 s |
| **last** 10 MB, cold (seek) | 1.06 s |
| full 983 MB | 77.6 s (12.7 MB/s) |
| either end, after a full read | ~0.02 s |

**Random access into a streamed placeholder is not the hazard it looked like** —
seeking to the tail costs the same as reading the head, because Drive fetches
ranges rather than materialising the file. Header-only and metadata-only passes
are cheap and should be preferred everywhere they suffice.

Directory enumeration is the slow part **on a cold cache**: 4.7 directories per
second, 3442 directories, a **12-minute** walk. Measured again after that walk:
**8 seconds**, ~90× faster, because Drive caches directory metadata locally once
enumerated. So 12 minutes is a first-run-on-a-new-machine cost, not a recurring
one, and an interactive re-scan is viable after the first. Two consequences,
both binding:

- **Task 03A's scan must still be cached**, but for the cold case only. A
  12-minute first walk cannot sit in front of a user pressing "scan"; an 8 s
  warm one can. Persist the index under
  `cache/` (local, never inside `gems_root`), key entries by path plus mtime
  plus size, and re-walk only what changed. The UI shows the cached tree
  immediately and refreshes behind it.
- **A full-sample pass over the corpus is an overnight job**, not a step in a
  task. 967 `_sig.mat` files at ~78 s each is on the order of **21 hours**, and
  it also pulls the whole corpus onto local disk, which streaming mode exists to
  avoid. Any task whose acceptance says "run on every recording" — 03B above all
  — must say whether it needs samples or only headers, and be runnable in
  resumable batches.

### The corpus is 967 recordings, not 43

Several tasks below say "the 43 old recordings". That number is the **old
labelled cohort**, and it is now the minority: the drive holds **967 `_sig.mat`
files across 3442 directories**, each with a `_vib.mat` beside it. Where a task
says 43 it means the old labelled cohort specifically; where it means the corpus
it says corpus. **No `*_segment_indices.mat` exists anywhere under the root**, so
the old cohort's labels are not on this drive — see task 07's `duration_cap_s`.

### Why this needs care
Google Drive is a **sync layer, not a database**. It has no atomic rename, no
locking across clients, and when two users write the same path it silently
produces `file (1).json`. A registry that is edited in place **will** be corrupted
the first time two people train on the same day. The layout below is designed so
that never happens.

### Layout

```
<gems_root>/
  data/
    <animal>/<session>/            raw.h5, meta.json, video.mp4
                                   (meta.json, NOT channel_map.json: it carries
                                    geometry from task 03 AND condition/who/when
                                    from 03A, so both writers read-modify-write)
  trials/
    <animal>/<session>/trials.jsonl         one line per trial, appended by the
                                            acquisition machine (single writer).
                                            NOT one file per trial -- see the
                                            shared-drive item cap below.
  labels/
    <animal>/events_<user>_<utc>.parquet    per-user, per-session, write-once
  corpora/
    <corpus_id>.json                        named corpus spec, immutable
  models/
    <model_id>/                             model_id = content hash, immutable
      model.txt  calibrator.pkl  provenance.json  metrics.json  shap/
  registry/
    events.jsonl                            APPEND-ONLY log (see below)
    events.jsonl.<user>.<utc>               conflict-safe shards, merged on read
  cache/                                    LOCAL ONLY — never inside gems_root
```

### The three rules that make this safe

1. **Everything is write-once and content-addressed.** `model_id` is the sha256 of
   the model file plus its corpus hash. A model directory, once written, is never
   modified. Retraining produces a new `model_id`, never a new version of an old
   one.
2. **The registry is an append-only log, never a mutable pointer file.**
   `registry/events.jsonl` holds one JSON object per line:
   `{ts, user, action, model_id, mode, animal, corpus_id, metrics}` where `action`
   ∈ `{trained, promoted, demoted, retired}`. Current state is computed by
   **replaying the log**, not by reading a field.
   - Each client appends to its **own shard** `events.jsonl.<user>.<compact-utc>`, and
     readers take the **union of all shards**. Two users writing at once produce
     two files, not a conflict.
   - A periodic `compact` command merges shards into `events.jsonl` — run manually,
     by one person, never automatically. This is **the one mutable shared file**
     the design allows, and it is the exception to rule 12: write atomically, read
     the merged file back and compare it against the shard union, and only then
     unlink the shards.
   - Union-of-lines is order-independent and idempotent, so a Drive conflict copy
     merges correctly by construction.
3. **Every file carries a sha256, and readers verify it.** Drive can present a
   partially synced file as complete. A checksum mismatch must raise, not warn —
   silently training on a truncated parquet is the failure mode this prevents.

   **The checksum lives in the *referencing* document, not in a sidecar.** A
   corpus spec and a `provenance.json` each carry a manifest of
   `FileRef{rel_path, sha256, size_bytes}`. Sidecar `.sha256` files would double
   the item count against the 500,000 cap this whole layout is designed around.
   Consequence, accepted: a file no document references has no checksum and is
   unverifiable — that is correct, because nothing reads it either.

### Required behaviours
- **`gems doctor` and the preflight also report the platform, the resolved root,
  the longest path the current layout would generate, and whether any stored path
  is absolute.** Cross-platform breakage is invisible on the machine that caused
  it.
- **Preflight check** before any training or inference run: resolve `gems_root`,
  assert a `.gems-root` marker file exists, verify every file the run needs is
  present and checksum-clean, and report anything still syncing. **Refuse to start
  on an incomplete corpus** — a model trained on a half-synced dataset is
  indistinguishable from a bad model.
- **Never write scratch or intermediate files into `gems_root`.** Envelopes,
  feature matrices and temp artifacts go to a local cache keyed by content hash.
  Syncing 16 GB of envelopes to every lab member is a real risk here.
- **Large-file hygiene:** mark the `data/` folders the run needs as available
  offline before a long job; streaming reads from Drive File Stream will otherwise
  dominate runtime.
- **Per-user identity** from git config or an explicit setting, recorded in every
  label file and registry line. "Who labelled this" is scientific metadata.
- **Role check in preflight is a WRITE PROBE, not a role query.** Drive roles are
  not visible from the filesystem. Attempt a write into the root and, on
  `PermissionError`, emit: "you are a Contributor; Drive for desktop makes that
  read-only — ask for Content manager". Name the function for what it does.
- **Item-count check in `gems doctor`:** report items against the 500,000 cap and
  warn above 80% — but report it as a **bound, not a number**. Walking 500,000
  items over a streamed Drive is not something a `doctor` run can do, so early-exit
  at an `--item-cap` and make the full count opt-in. Trash counts toward the cap
  and is **invisible to the filesystem**, so the figure is always a lower bound;
  say so in the output.
- **`gems_root` is per-user config via `platformdirs`** (rule 15 — `%LOCALAPPDATA%`
  on Windows), never committed. Resolution order: explicit argument → `GEMS_ROOT`
  env → platformdirs config → legacy `~/.gems/config.toml` → bounded scan for the
  marker. **Never reconstruct the root from the drive name.**
  Provide `gems doctor` to print the resolved root, sync status, shard count and
  any checksum failures.

### Shared-drive specifics (this lab uses a Google **shared drive**, not a personal Drive)

**Roles — and the trap.** Shared-drive roles are Manager / Content manager /
Contributor / Commenter / Viewer. "Contributor" looks like the right least-privilege
choice for lab members, and it is wrong here: **in Google Drive for desktop,
Contributors have read-only access.** Anyone who labels, trains or exports through
the synced folder must be **Content manager**.

| Who | Role | Why |
|---|---|---|
| everyone who labels / trains / exports | **Content manager** | can add, edit, move and trash; **cannot permanently delete** |
| one or two owners | Manager | membership, permanent delete, trash purge |
| collaborators who only read results | Viewer / Commenter | |

Content manager is a real safety net: deletions go to the shared-drive trash and
only a Manager can purge, so an accidental `rm -rf` inside `gems_root` is
recoverable. Do **not** grant Manager broadly just to avoid a permissions error.

**Why a shared drive is the right call:** files are owned by the drive, not by a
person. When a student leaves the lab, nothing disappears and nothing needs
transferring — which is the failure mode of a personal Drive shared out.

**Item cap: 500,000 items per shared drive**, counting files, folders, shortcuts
and trash. This is the one limit this design can actually hit, because of
per-trial metadata.

> **Do not write one file per trial.** Write **one JSONL per session**, appended
> per trial: `trials/<animal>/<session>/trials.jsonl`. Acquisition is a single
> writer, so append is safe there, and this keeps the item count in the thousands
> instead of the hundreds of thousands. It is also far faster to sync — many tiny
> files is the worst case for Drive for desktop.

Budget the item count before committing to a layout, and report it in
`gems doctor`. Trash counts toward the cap, so a Manager should purge periodically.

**Other shared-drive limits worth knowing:** 750 GB uploaded per user per 24 h
(a 25-minute 9-channel float32 recording is ~1.3 GB, so ~570 recordings/day — not
a practical constraint, but relevant to a bulk initial migration); 5 TB max file
size; 100 levels of folder nesting; a file lives in exactly one folder (use
shortcuts, not copies).

**Paths differ per machine**, so `gems_root` stays per-user config with
auto-discovery by scanning for the `.gems-root` marker. **Nothing written into
`gems_root` may contain an absolute path** — store POSIX paths relative to the
root and resolve locally (see the Cross-platform rules in `CLAUDE.md`).

> **This drive's name contains a character Windows cannot use.** It is
> `BIONICs Lab: Enteric Interfaces Team`, and `:` is illegal in a Windows path,
> so Drive for desktop substitutes it and **the folder is not the same string on
> Windows**. Never reconstruct the root from the drive name; always discover it
> via the marker file. Confirm what Windows actually produces before onboarding
> the first Windows user.
>
> Windows `MAX_PATH` is 260 unless long paths are enabled, and this data already
> sits at ~193 characters on Windows before the tool appends anything. Keep
> generated segments short and check the deepest path the layout can produce.

| OS | Typical root |
|---|---|
| macOS | `~/Library/CloudStorage/GoogleDrive-<account>/Shared drives/<DriveName>` |
| Windows | `G:\Shared drives\<DriveName>` |
| Linux | **no official Drive for desktop client** — rclone or equivalent; confirm before assuming a lab Linux box can participate |

**Streaming vs offline.** Drive for desktop streams shared-drive files by default.
Random-access reads into a streamed HDF5 are slow enough to dominate training
runtime, so either mark the needed `data/` folders available offline, or have the
preflight **copy the run's inputs into the local content-addressed cache first**.
Do the copy; it is more predictable than relying on pinning.

### Concurrency is still not free — state the limits
This design tolerates **concurrent appends** and **concurrent reads**. It does not
make Drive transactional. Two users training the same corpus simultaneously will
produce two valid models and two log lines, and a human decides which is promoted.
That is the correct behaviour for a research tool, but say it in the UI rather
than pretending the conflict cannot happen.

### Tests
- two simulated clients appending concurrently produce a log whose replay contains
  both entries, in either merge order
- a Drive-style conflict copy (`events.jsonl (1)`) is merged, not lost or
  duplicated
- a truncated parquet fails the checksum and raises before training starts
- preflight refuses a corpus with a missing file and names it
- no code path writes to `gems_root/cache`
- replaying the log twice is idempotent

### Acceptance
A second lab member clones the repo, sets `gems_root`, runs `gems doctor`, and can
immediately list the same models and corpora as the first — with no manual copying
and no shared-write conflicts.
<!-- /TASK -->

<!-- TASK:01 slug=nan-interop deps=00 gate=yes -->
## Task 01 — Zeros → NaN interop fix

**Module:** `GEMSBlanking:detector/labeled_save.py` (or `processing_new/step0_load_data.m`)
**Depends on:** 00
**Gate:** met 2026-09-21 — **by verification, not by a fix**

> **THE PREMISE OF THIS TASK WAS WRONG.** The fix already landed upstream in
> GEMSBlanking commit **`a95d1ff` "Blanking: fill bad samples with NaN, not 0"**:
> `detector/labeled_save.py:205` is `BLANK_FILL_VALUE = float("nan")`, it is the
> only blank-fill site, the writer upcasts to float64 so NaN survives `savemat`,
> it records `blank_fill = "nan"` in the sidecar, and a regression test already
> exists at `tests/test_phase8.py:412`. `step0_load_data.m` converts neither way
> and `step1_bandpass.m` tests `isnan` only, so the chain is NaN-correct end to
> end.
>
> **I asserted the zeroing behaviour without ever reading `labeled_save.py`** —
> GEMSBlanking is private and §3 says its paths were taken from
> `DEVELOPER_GUIDE.md`. That caveat covered the *paths*; I stated the *behaviour*
> as fact in §0, in invariant 1's justification, and here. Invariant 1 stands on
> its own merits — zero is a legal signal value, so inferring invalidity from it
> is a bug regardless — but its justification was stale.
>
> What remained, and is now done: the measurement, enforcement so it cannot
> regress, and an audit for legacy files.
>
> **The legacy audit is now concrete and cheap.** It was blocked on not knowing
> where pre-`a95d1ff` outputs lived; task 07's label hunt answered it. Every
> `*_blankmotion.mat` holds `yOut` **and** `removedSegmentIdx`, so the audit is
> a self-contained test on one file: read `yOut` at the indices
> `removedSegmentIdx` names, and see whether those samples are `0` or `NaN`.
> No reference file, no guesswork, one indexed read per recording. A file whose
> masked samples are exactly `0` was written by the old writer and every
> statistic derived from it carries the ringing.
>
> **Measured 2026-09-23: clean, and a full sweep is not needed.** Six files
> spanning v5 and v7.3, raw and notched, including a partially-blanked case
> (6 segments, 20% masked): masked samples **100% NaN**, unmasked **0% NaN**,
> exact-zero fraction 0.00%, longest exact-zero run 0 samples. The outside-mask
> column is the one that makes this non-vacuous — all-NaN would have scored
> 100% inside too.
>
> **More importantly the population is wrong for this audit.** These 406 files
> are written by `browseMotionArtifacts.m:516`
> (`yBlanked(idxRanges(k,1):idxRanges(k,2),:) = NaN`), which has always written
> NaN. The zeroing defect was in **GEMSBlanking's Python `labeled_save.py`**
> before `a95d1ff`, which is a different writer producing different files. So
> the remaining audit target is any output of *that* tool from before
> `a95d1ff` — not these. Do not spend 272 GB of streaming confirming a
> property already established from the writer's source.

### Purpose
`_blankmotion.mat` currently writes `yOut` with masked ranges **zeroed**.
`processing_new/step1_bandpass.m:51` recognises only `NaN`:

```matlab
invalid = isnan(x);
if any(invalid); xfill = fillmissing(x,'linear','EndValues','nearest'); end
xf = filtfilt(b, a, xfill);
xf(invalid) = NaN;
```

*Historical — this described the pre-`a95d1ff` writer.* A mask boundary written
as a hard step to zero rings through an order-4 Butterworth applied by `filtfilt`
(effective order 8) into adjacent *valid* samples.

**"The damage is worst for short blanks" is withdrawn.** Measured per boundary,
the zeroed damage is flat-to-rising with gap length and *lowest* at 2 ms. Short
blanks matter because they produce more boundaries per second, not because each
boundary does more damage.

### Build
No change to make in either sibling repo — the writer is already correct. What
this task delivers instead:

- `gems_blanking_v2/io/nan_interop.py` — `find_zero_runs` / `assert_no_zero_runs`
  enforcing invariant 1 on everything **we** emit (task 15 consumes it), plus a
  **read-only** audit that flags legacy files: long zero runs and no NaN means
  `predates_the_nan_fix`, and the file should be re-exported from its labels. The
  audit **never converts zeros to NaN** — that is the `zeros_are_invalid`
  inference this task's "Do not" forbids.
- the ringing measurement above
- a round-trip acceptance run through real MATLAB

**The residual risk is files exported before `a95d1ff`**, which still contain
zeroed gaps that nothing detects. Those need an audit pass over the real corpus
on the shared drive before any of them is trained on — not doable from the build
machine, which has no Drive mount.

### Tests
- `test_no_zero_runs`: round-trip a masked recording; assert no exact-zero run
  longer than 2 samples in any channel of the emitted `yOut`
- `test_nan_preserved`: assert masked sample count out == masked sample count in
- `test_ringing` — **MEASURED 2026-09-21 in real MATLAB R2026a**, the actual
  chain (order-4 Butterworth 100–5000 Hz through `filtfilt`,
  `fillmissing(…,'linear','EndValues','nearest')`), median peak deviation in the
  50 ms either side of a boundary over **100 gap positions**:

  | out-of-band content | gap | zeroed | NaN | ratio |
  |---|---|---|---|---|
  | 0 µV | 10 ms | 6.6 | 12.5 | **0.53** |
  | 60 µV (ECG-scale) | 10 ms | 28.1 | 12.3 | **2.29** |
  | 60 µV | 50 ms | 30.9 | 12.7 | 2.43 |
  | 500 µV (drift-scale) | 10 ms | 227.2 | 13.4 | **16.98** |
  | 500 µV | 100 ms | 260.5 | 18.4 | 14.13 |

  **"The NaN path must be materially lower" is false as stated.** It holds on
  realistic hosts (2×–21×) and reverses to ~2× *worse* when the host carries
  nothing outside the passband. Mechanism: zeroing's step is set by the host's
  instantaneous **raw** value at the boundary, which is dominated by out-of-band
  ECG, drift and motion; interpolation is continuous with the host and its error
  does not scale with that content at all (NaN stays ~10–18 µV across the sweep).
  The decision is unchanged — real hosts are never content-free — but the claim
  needed qualifying. Same behaviour at this project's 300–3000 Hz band:
  0.49 / ~2.1 / 16–19.

  *Method note:* a single gap position gave ratios of 0.42–2.58 with no pattern.
  One draw would have produced a confident wrong number; the sweep is the test.

### Acceptance
No exact-zero runs in emitted output; MATLAB side loads the file and produces the
same valid-sample count Python wrote.

### Do not
Do not add a `zeros_are_invalid` compatibility flag. Zero is a legal signal value;
inferring invalidity from it is the bug.
<!-- /TASK -->

<!-- TASK:02 slug=peri-r-measurement deps=00,01 gate=yes -->
## Task 02 — Peri-R cardiac window measurement

**Module:** `physio/cardiac_window.py`
**Depends on:** 00, 01
**Gate:** YES — this is the largest deterministic win available and it needs no model

### Purpose
The current pipeline blanks ±15 ms around every R-peak on **all** channels
(`step1a_blank_cardiac.m:41-43`, `D.y(blank,:) = NaN`). At ~400 bpm that is ~20% of
every recording. Predicted from QRS energy distribution, almost none of it is needed
above 300 Hz or below 3 Hz. Measure it rather than assume it.

A working prototype already exists: `periR_window_check.m` (171 lines, delivered
separately). Port it or call it.

### Signature
```python
def measure_cardiac_window(
    rec: Recording, rpeaks_s: np.ndarray, bands: dict = BANDS,
    half_window_frac_rr: float = 0.40,   # NOT a fixed 100 ms: must be < RR/2
    baseline_pct: float = 50.0,
) -> dict[tuple[str, str], tuple[float, float] | None]:
    """(channel, band) -> (t_start_s, t_stop_s) relative to R, or None if flat."""
```

### Algorithm
1. For each channel and band, compute the band envelope — **task 06's**
   `band_envelope`, not task 07's (07 has no envelope function; this line was
   wrong). Use the current six bands of A.3, with `padtype="constant"` and the
   settling trim already in place. **The 2026-09-19 numbers below were taken at
   the old band edges** (`300-5000`, `1-100`) with neither fix, so they are a
   prior to compare against, not a result to reproduce.
2. Extract ±`halfwidth_s` about every R-peak; average across beats.
3. Baseline = the `baseline_pct` percentile of the profile's outer thirds.
4. Window = the **contiguous** span around lag 0 exceeding
   `baseline + 3 × MAD(outer thirds)`. If no sample exceeds it, return `None`.
5. Report recovered duty cycle: `1 − (window_duration × beat_rate)` vs the current
   uniform 30 ms.

### The envelope is the wrong instrument — average the SIGNAL, not its envelope

The algorithm above averages **envelopes** across beats, and an envelope over
`window_s` cannot resolve anything shorter than `window_s`. At animal J's
RR = 165 ms the extraction profile is ±66 ms = 132 ms wide, against these
windows:

| band | `window_s` | profile / window | envelope method |
|---|---|---|---|
| 300-3000 | 25 ms | 5.3× | usable |
| 100-300 | 75 ms | 1.8× | marginal |
| 10-150 | 100 ms | 1.3× | **blind** |
| 2-50 | 310 ms | 0.43× | **structurally impossible** |
| 0.5-3 | 6 s | 0.022× | **structurally impossible** |
| 0-2 | 7.5 s | 0.018× | **structurally impossible** |

For the bottom three, every frame in the profile is computed from a window that
contains lag 0, so the profile **cannot vary with lag** — known before any data
is read. Worse, the baseline in step 3 comes from the profile's outer thirds,
which sit inside that same window: baseline equals peak, nothing exceeds
`baseline + 3·MAD`, and step 4 returns `None`. **`None` there means "cannot
see", and it is indistinguishable from "no cardiac window in this band", which
is the opposite conclusion.**

**Average the band-passed signal, not its envelope.** Coherent averaging across
beats is limited only by the sample rate, and it is the physically right
operation: the QRS is time-locked to R and the neural signal is not, so the
average isolates the cardiac contribution and suppresses everything else by
√n_beats. Then take the extent on the averaged waveform.

```
1. band-pass the channel (task 06's filter design; padtype="constant")
2. coherent average of the SIGNAL over ±min(halfwidth_s, 0.45·RR) about each R
3. extent = contiguous span about lag 0 where |average| exceeds the null (below)
```

**The null is an anti-phase control, not the profile's own outer thirds.**
Repeat step 2 triggered on the midpoints between successive R-peaks. That
control has the same beat periodicity, the same beat count and the same filter,
and differs only in phase — so anything the R-triggered average shows above it
is genuinely locked to the QRS. Add a jittered-trigger surrogate (triggers
displaced by a random offset per beat) as a second null; where the anti-phase
and jittered nulls disagree, the band is picking up the cardiac *rhythm* rather
than the QRS, which is the next point.

### Three of the six bands need a different question, not a better measurement

At 364 bpm the heart-rate fundamental is **~6 Hz**, and where that sits relative
to a band changes what "cardiac contamination" even means:

- **`0-2` and `0.5-3` are below the fundamental.** Predicted QRS energy there is
  0.00%. The answer is **`no_window`** on spectral grounds and it needs no
  measurement to defend — run the coherent average anyway as a check, but a null
  result there is a confirmation, not a failure to see.
- **`2-50` contains the fundamental and its first harmonics, and its
  `window_s` (310 ms) exceeds RR (165 ms).** Peri-R blanking is the **wrong
  instrument** here: the contamination is a continuous narrowband component, not
  an event, and blanking ±anything around a 6 Hz trigger removes the whole
  record. Return **`not_applicable`** with that reason. Handling it belongs to
  task 13/14 as regression or a notch at the beat rate and its harmonics — or as
  an accepted, modelled confound — never as a blank.
- **`10-150` contains harmonics 2–25 and its window is 100 ms against RR 165 ms.**
  Genuinely marginal; the coherent average is what decides it.

**The return value must carry which of these it is.** `tuple | None` cannot, and
`cardiacRemoveWinMs` and task 13 consume it:

```
status: "measured" | "no_window" | "unresolvable" | "not_applicable"
```

`no_window` means resolved and nothing above the null. `unresolvable` means the
method could not see. `not_applicable` means peri-R blanking is not the right
operation for this band at this heart rate. **Only `measured` may produce a
blanking extent**; the other three produce zero blanking and a recorded reason.

### GATE VERDICT — measured 2026-09-23, `gems_j_t01_ms3_bl_230315`, 3543 beats, RR 165.6 ms

**Passed, and the win is real but smaller and less uniform than this task
claimed.** Current MATLAB blanks ±15 ms on all channels: 30 ms at RR 165.6 ms is
**18.1% duty**.

| band | status | extent | duty now | vs the uniform 30 ms |
|---|---|---|---|---|
| 300-3000 | measured ×9 | −8.8 .. +5.2 ms | **8.4%** | **−9.7 points — more than half the cardiac blanking in the ENG band is unnecessary** |
| 100-300 | measured ×9 | −32.0 .. +26.1 ms | **35%** | **+17 points — the current blank is too NARROW here** |
| 10-150 | unresolvable ×9 | — | 0 | impz 141 ms > the 133 ms profile |
| 2-50 | not_applicable ×9 | — | 0 | contains the 6 Hz fundamental |
| 0.5-3, 0-2 | unresolvable ×18 | — | 0 | impz 4.0 s / 1.1 s |

Threshold-crossing rise, the spike consumer's own measure: **1.41–1.57×** at lag 0
on the six channels that resolve cleanly.

**The asymmetry is the finding.** A single uniform window is simultaneously too
wide for the ENG band and too narrow for `100-300`. "Per-band windows recover
data" is true only where it is true; stated flatly it would have been half
wrong.

**`100-300` at 35% duty is a task 13/14 problem, not a win.** Its own impulse
response is 34.4 ms, so each blank expands by roughly that much at each edge
once settling is honoured: 58 + 69 = **127 ms out of every 165.6 ms, ~76%**.
Blanking is very likely the wrong instrument for `100-300` at this heart rate
for the same reason it is wrong for `2-50` — a notch at the beat rate and
harmonics, or an accepted modelled confound, should be evaluated against it
before 127 ms per beat is thrown away. **Do not treat the `100-300` extent as an
instruction to blank until task 13 has compared the two.**

### The measured extent is convolved with the trigger's own timing error

Coherent averaging removes the envelope floor but introduces a different one:
the average is the true waveform convolved with the distribution of R-peak
timing errors. Jitter **widens** the measured extent, so unlike the envelope
floor this biases upward — toward over-blanking. Report a fourth number,
`trigger_jitter_s`, and state the extent as `extent_signal_s ⊕ jitter` rather
than as a bare width. A measurement whose jitter is comparable to its extent is
`unresolvable`, not `measured`.

This matters here specifically: **no channel in this recording passes task 05's
provisional vetoes**, so the trigger train is imperfect by the project's own
standard. The result still stands — false and missed triggers both *reduce*
`peak_over_null`, and 72–87 in `100-300` cannot be produced by a bad train — but
the extents should be read as upper bounds until a clean recording confirms
them.

### The coherent mean must be robust, and the trigger channel must be vetted

`LVN3` carries a ~400× transient (`max|x|` 2.148 against 0.005–0.02 elsewhere)
with a normal MAD-σ, and a plain mean is not robust to it — its coherent average
and null are both ~100× the other channels'. Use a **10% trimmed mean across
beats**, and report the plain mean alongside it so the difference stays visible.

`LVN3` was also selected as the trigger. Beat-count consensus is the right
selector and is a clear improvement on lowest `implausible_frac` — which chose
`ANT1`, a channel that scored well *because* it missed 53% of the beats, the
purest form of a good metric earned for a bad reason. But consensus alone does
not exclude a channel that is electrically broken. **Add a transient veto to
trigger selection**: `max|x| / MAD-σ` far outside the cohort disqualifies a
channel from being the trigger regardless of its beat count.

### `unresolvable` beats `no_window` — the objection is upheld

This task told you to expect `no_window` for `0-2` and `0.5-3` on spectral
grounds. **That was wrong, and the correction is upheld.** Both bands ring for
1.1–4.0 s inside a 133 ms profile, so the method cannot see there, and
`no_window` would be claiming a negative that was never measured. The spectral
argument is still true and belongs in the `reason` string, where it is a note
rather than a result. Operationally the two are identical — only `measured`
produces a blanking extent — so the only thing at stake is honesty about what
was established, which is the whole reason the enum exists.

### Real-data tests need an explicit opt-in

`test_rpeaks.py` and `test_stim_split.py` still skip as "not reachable" with a
configured root, because the hermetic autouse fixture isolates the per-user
config and they can never see it. That is the fixture working as designed — the
hermetic rule stays. Add an explicit opt-in instead: a test marked
`@pytest.mark.real_data` runs only when `GEMS_REAL_DATA=1`, and that marker is
the only thing permitted to read the real per-user config. Default CI stays
hermetic; a local run against the drive is one environment variable.

### The ENG band gets a third measure, matched to its consumer

For `300-3000` the consumer is spike detection, so measure what the consumer
does: **threshold-crossing rate per lag bin**, which is sample-resolution and
needs no envelope at all. This is the measurement that already falsified the
original prediction — 1.53×2.12× rise at lag 0 on three channels, where the
prediction said there would be nothing. Keep it, and report it alongside the
coherent average rather than instead of it.

### The measured extent is an envelope extent, not a signal extent

An envelope computed over `window_s` cannot resolve anything shorter than
`window_s`: a perfect impulse at lag 0 produces a measured extent of about
±`window_s/2`. The 2026-09-19 result shows this plainly — `300-5000` measured
−15 to +14 ms with a 25 ms envelope window, which is the window and essentially
nothing else. **Taken at face value it would justify blanking 30 ms of a band
where the QRS carries 0.00–0.54% of its energy.**

So report three numbers per `(channel, band)`, never one:

```
extent_env_s      the measured span                       (what step 4 returns)
window_s          the band's envelope window              (the resolution floor)
extent_signal_s   max(0, extent_env_s - window_s)         (the blanking input)
```

**`extent_signal_s` is what task 13 and the MATLAB `cardiacRemoveWinMs` consume.**
An `extent_env_s` at or below `window_s` means *unresolved*, which for blanking
purposes is `0` plus a note — not `window_s`. Flag any band where the two are
within 20% of each other: that is the regime where the answer is "shorter than we
can see", and reporting it as a measurement would bake a resolution limit into
every recording as if it were physiology.

### Expected result — falsifies the plan if wrong
Predicted from QRS energy (superseded by the measurement below — kept for the
reasoning):

| band | 8 ms QRS | 12 ms | 20 ms |
|---|---|---|---|
| 0–2 Hz | 0.00% | 0.00% | 0.00% |
| 0.5–3 Hz | 0.00% | 0.00% | 0.02% |
| 2–50 Hz | 4.0% | 12.0% | 38.8% |
| 1–100 Hz | 23.9% | 52.8% | 89.9% |
| 100–300 Hz | **65.2%** | **32.1%** | 2.5% |
| 300–5000 Hz | 0.54% | **0.00%** | **0.00%** |

**MEASURED 2026-09-19 on `gems_j_t01_ms3_bl_230315` (animal J, 10 min, 3511 beats,
RR 164.7 ms), channels LVN1 / RVN3 / ANT2 — the prediction above is PARTLY WRONG:**

| band | env win | measured rise | measured extent | duty at RR 165 ms |
|---|---|---|---|---|
| 300–5000 | 25 ms | **10.4–15.7%** | −15 to +14 ms | 17.8% |
| 100–300 | 75 ms | 43–65% | −49 to +44 ms | **56%** |
| 1–100 | 150 ms | 0.4–2.0% | none (ANT2 excepted) | 0% |
| 2–50 | 310 ms | 0.0–0.6% | none | 0% |

Two corrections follow:

1. **There IS a cardiac window above 300 Hz.** The operationally correct test for
   the spike consumer — peri-R rate of `|x| > 4.5σ` crossings — gives a
   **1.53–2.12× rise at lag 0** on all three channels. The current ±15 ms blank in
   that band is therefore roughly right, **not** the waste this document claimed.
2. **A measured extent includes the envelope window's own smearing.** The 25 ms
   envelope contributes ±12.5 ms of the ±15 ms measured at 300–5000 Hz, so true
   QRS content there is ~±2 ms — consistent with the energy model. For *masking*
   the smeared extent is the correct one, because it is what the consumer sees.
3. **Bands whose envelope window exceeds RR cannot show a peri-R modulation at
   all.** At RR 165 ms the 1–100 Hz (150 ms) and 2–50 Hz (310 ms) windows are
   0.9× and 1.9× RR, so "no window" there is partly by construction. State the
   `window/RR` ratio next to every result.

**The peri-R half-window must be < RR/2.** ±100 ms at RR 165 ms overlaps the
neighbouring beat and contaminates the baseline estimate; use `0.40 × RR`.

### Tests
- synthetic ECG + ENG: the measured 300–5000 Hz window is `None`
- synthetic with a deliberately wideband QRS: a window **is** found above 300 Hz
  (proves the measurement can detect one, so `None` means absence not failure)
- recovered duty cycle matches an independent hand calculation

### Acceptance
Run on ≥5 real recordings across ≥3 animals. Report the window per channel per
band and the recovered duty cycle. **If a substantial 300–5000 Hz window appears on
real data, stop and report** — it contradicts the QRS energy model and the cardiac
plan needs revisiting.

### Do not
Do not revive template subtraction (`step1b_remove_cardiac.m`); see `PIPELINE.md`
§10.2.
<!-- /TASK -->

<!-- TASK:03 slug=io-channel-map deps=00 gate=no -->
## Task 03 — Recording IO and channel map

**Module:** `io/recording.py`, `io/channel_map.py`
**Depends on:** 00
**Gate:** no

### Purpose
Load TDT blocks and attach the geometry the new cohort needs.

### Build
- Wrap `GEMSBlanking:detector/recording_io.py` (`Recording`, `load_recording`;
  handles `.mat`, chunked HDF5, flat HDF5). **Verify the path before importing.**
- Prefer flat HDF5 via `detector-pyqt/scripts/m1_ingest.py` — the viewer is 5× faster
  on it.
- Extend the channel table with `cuff_id`, `contact_index`, `rostral_end`, `config`.
- Persist to `~/.detector/preprocessing_profiles/<animal>.json`, the existing
  location used by `detector-pyqt/ui/widgets/channel_assignment.py`.
- Read `fs` from the file. Never hardcode 24414.

### Known defect carried from 00A — fix in a follow-up
`store.find_gems_root` resolves explicit → env → config → scan, and **a bad
explicit root falls through to the scan**. If you name a root and silently get a
different one, that is the silent-wrong-root failure. Explicit and
`GEMS_ROOT` must be authoritative and raise. (Found in task 03, in the same
pattern in `detector_core`, where it was fixed.)

### `rostral_end` handling

**RESOLVED 2026-09-26 — Andrea: "it's always channel 1 rostral."** Contact 1 of
every nerve cuff faces the head, in every animal. That is a surgical convention,
so it is a **cohort constant** and belongs in `protocol.yaml`, not in `meta.json`
(invariant 24):

```yaml
rostral_end: 1          # contact_index facing rostrally, every nerve cuff
                        # source: Andrea, 2026-09-26 — surgical convention
```

- **Delete `rostral_end` from `meta.json`.** It is `null` in all 837 files; a
  per-recording field that is null everywhere and now known everywhere is a
  constant written 837 times.
- **Applies to the nerve cuffs only** (`RVN`, `LVN`). `ANT1–3` are stomach
  electrodes; the conduction-velocity step does not run on them.
- **Old cohort: moot.** Its hardware quasi-tripole shorts the end contacts, so
  each nerve gives one derived channel and there is no inter-contact lag to take
  a velocity from. Velocity is new-cohort only.
- **This is a declaration, not a guess.** The rule below forbids *inferring*
  orientation from channel order or names; a stated surgical convention entered
  once, with its source, is exactly the declaration that rule asks for.

**Sign convention, stated once:** a spike reaching contact 1 before contact 3 is
travelling **caudally, away from the brain → efferent**. Reaching 3 before 1 is
travelling **rostrally → afferent**. `direction_valid` becomes `True` for
new-cohort nerve channels.

**One consistency check, as a flag and not a substitute.** The abdominal vagus is
predominantly afferent by fibre count (~80%), so across a recording
afferent-signed events should usually dominate. Report the afferent fraction per
animal once velocities exist. An animal where efferent-signed events clearly
dominate is a candidate for a cuff placed the other way round that one time —
**flag it to Andrea; never flip it automatically.** Fibre counts are not spike
counts, so this is a prompt to check the surgery record, not evidence on its
own.

The original handling, which still applies to any recording outside the
convention:

It cannot be reconstructed once the animal is gone. If absent:
- emit **unsigned** velocities
- attach `direction_valid = False` to every velocity record
- log a warning once per recording, at WARNING level

**Never guess it** — from channel order, name, or anything else.

### Tests
- round-trip a synthetic `.mat` and a flat HDF5, assert identical arrays and `fs`
- a channel table missing `rostral_end` loads, sets `direction_valid=False`, warns
- old-cohort (`hw_tripole`, 5 channels) and new-cohort (`independent`, 9 channels)
  files both load and report the right `config`

### Acceptance
Both cohorts load; `config` is inferred correctly from the channel count and
`contact_index` presence; the profile JSON round-trips.
<!-- /TASK -->

<!-- TASK:03A slug=scan-conditions deps=03,00A gate=no -->
## Task 03A — Folder scan and condition inference

**Module:** `io/scan.py`, `io/conditions.py`
**Depends on:** 03, 00A

### Purpose
Point the tool at a parent folder, have it find every recording and **propose** the
animal and experimental condition for each, let the user correct anything wrong,
and only then let those recordings become selectable for training. This exists in
`detector-pyqt` today and is the right interaction — this task keeps it and fixes
one thing.

### What exists today
`training_window.py::_on_per_animal_add_folder` calls
`find_blankmotion_files(Path(folder), recursive=True)`, then per file:

```python
source_stem = source.stem
if source_stem.endswith("_notched"):      core = source_stem[:-len("_notched")]
elif source_stem.endswith("_notchblanked"): core = source_stem[:-len("_notchblanked")]
else:                                      core = source_stem
animal   = extract_animal_letter(core)
rec_type = "stim_rec" if "_stim_rec" in core else "baseline"
```

### The one thing to fix: **never silently default a condition**

`rec_type = "stim_rec" if ... else "baseline"` means **every unrecognised filename
becomes `baseline`.** A recording whose name does not match the expected pattern —
a typo, a new protocol, a file from a collaborator — is silently relabelled as a
control. Condition is an independent variable in `bulk_mixed_models.m`, so this
does not produce a visible error; it produces a quiet mislabelling that shifts an
effect estimate.

Replace with three states:

| State | Meaning | Effect |
|---|---|---|
| matched | exactly one rule matched | proposed, user confirms |
| ambiguous | two or more rules matched with different conditions | **must be resolved by hand** |
| `unknown` | no rule matched | **must be resolved by hand** — never defaults |

`unknown` and `ambiguous` recordings are listed but **cannot enter a corpus**.
Blocking is the point: a recording nobody has classified should not silently
become a control.

### Rules live in a shared, versioned file — not in code

`<gems_root>/conditions.yaml`, so every lab member parses identically and the rule
set improves over time.

### The two real naming conventions — measured, not assumed

**The lab has two, one per cohort, and a rule set written for either alone fails
on the other.** Verified 2026-09-21: 14 old-cohort ids recovered from
`GEMSBlanking`'s LORO summaries, and new-cohort directory names read directly
off the shared drive.

| | old cohort (the 43) | new cohort (2026 chronic) |
|---|---|---|
| example | `E1000_FRE_…`, `M100_JEL_…`, `einh_fre_…` | `gems_d_t01_ms1_bl_164012`, `gems_d_t01_es1_sr_204720` |
| animal | 2nd underscore token's initial (`FRE`→F) | 2nd token (`d`→D) |
| baseline | `_bl_` | `_bl_` |
| stim recovery | `_stim_rec` | **`_sr_`** |
| case | mixed, sometimes all-lowercase | lowercase |

Three consequences:

1. **`_sr` must be a rule.** The spec shipped `_stim_rec` only; every new-cohort
   recovery file would be `unknown`. Order matters — try `_stim_rec` before
   `_sr`, and both before any baseline rule.
2. **The spec's `animal_pattern` regex is withdrawn.** `(?<![A-Za-z])([A-Z])(?=[_\d])`
   was wrong on 100% of real ids — it takes `E` from `E1000_FRE_…` and nothing at
   all from a lowercase-led name. Use
   `GEMSBlanking:detector/animal_id.py:extract_animal_letter` (second underscore
   token's initial), which is dependency-free and works on **both** conventions.
   Animal is the grouping variable for LORO and for the per-animal mode, so this
   is not cosmetic.
3. **Everything matches case-insensitively** — the same cohort writes
   `gems_d_t01_ms1_bl_164012/` and `GEMS_D_t01_MS1_bl_cam1_….mp4`.

### Condition is not one string — it is four fields

`ES` is electrical stim and `MS` is mechanical, and `msX_esY` means both at once.
Flattening these into condition labels would produce a dozen one-off levels in
the mixed models, when they are in fact a small factorial with **ordered
numeric parameters**:

| field | values |
|---|---|
| `epoch` | `baseline` \| `stim_recovery` |
| `estim_hz` | `None`, or 10 (ES1) / 100 (ES2) / 1000 (ES3) |
| `mstim_hz` | `None`, or 10 (MS1) / 50 (MS2) / 100 (MS3) |
| `timepoint` | `t01`, `t02`, … |

**`estim_hz` / `mstim_hz` name the protocol arm, not an applied stimulus.**
`E1000_JEL_E1000_bl_1315` is a *baseline* carrying `estim_hz = 1000`: it is the
baseline recorded for the 1000 Hz arm, with nothing being delivered during it.
`epoch` is what says whether stimulation was on. Any model treating `estim_hz`
as an applied stimulus will be wrong on every baseline row — use the
interaction with `epoch`, or carry an explicit `stim_on = (epoch ==
"stim_recovery")`.

Fixed across all electrical levels: **1000 µA, 0.3 ms pulse width, square,
bipolar**. Fixed across all mechanical levels: **50% duty cycle**. So ES1→ES2→ES3
is one frequency axis (10/100/1000 Hz) and MS1→MS2→MS3 is another
(10/50/100 Hz) — model them as ordered numbers, not as unrelated categories.

`conditions.yaml` therefore maps a filename to this **record**, not to a single
label. The closed vocabulary applies to `epoch`; the two frequency fields are
validated against their allowed sets.

### RESOLVED 2026-09-21 — the old cohort's tokens are the same axes in Hz

Confirmed by Andrea: **`E<n>` and `M<n>` are frequencies in Hz**, the same two
axes the ES/MS ordinals name. So the conventions merge directly and nothing is
lost by pooling cohorts on frequency:

| old form | Hz | new ordinal |
|---|---|---|
| `E10` | 10 | ES1 |
| `E100` | 100 | ES2 |
| `E1000` | 1000 | ES3 |
| `M10` | 10 | MS1 |
| — | 50 | MS2 |
| `M100` | 100 | MS3 |

`M100E10` is mechanical 100 Hz **and** electrical 10 Hz — the combined case the
record already handles.

**Precedence when a name carries both forms and they disagree: the explicit-Hz
token wins.** Ruled on `M100_JEL_MS2_bl_1945` — `mstim_hz = 100` from `M100`,
not 50 from `MS2`. Generalise it: an explicit-Hz token beats an ordinal, for
both axes.

**Record the conflict, do not silently drop it.** A row resolved this way
carries `token_conflict: ["M100", "MS2"]` in its `meta.json` and in provenance.
The precedence rule makes it parseable; the record makes it auditable, and a
cluster of these would say the old filenames are less trustworthy than they
look.

`unparsed_stim_tokens` stays, for tokens genuinely outside both tables — it just
no longer fires on these seven.

**One thing still unknown, and it matters only if you pool cohorts on
amplitude:** the new cohort fixes electrical stimulation at 1000 µA, and the old
cohort's filenames encode frequency only. Confirm the old amplitude was also
1000 µA before treating the two as one dataset. Frequency comparisons are safe
either way.

### The learning loop
When the user corrects a condition, offer: *"Add a rule so this is automatic next
time?"* with the proposed regex pre-filled, and show how many other currently-scanned
recordings that rule would also match, **before** saving. The rule is appended to
`conditions.yaml` (write to a per-user shard, merged like the registry — task 00A).

A correction that contradicts a rule that *did* match flags the rule for review.
That is how a bad rule gets caught rather than propagating.

### Signature
```python
@dataclass(frozen=True)
class Condition:
    epoch:     Literal["baseline", "stim_recovery"]
    estim_hz:  int | None      # 10 / 100 / 1000; absent when no electrical stim
    mstim_hz:  int | None      # 10 / 50 / 100;   absent when no mechanical stim
    timepoint: str | None      # "t01", ...
    # serialises with ABSENT keys, never null: estim_hz absent on a baseline means
    # "no electrical stimulation", which is a fact, not a gap. (CLAUDE.md, the
    # JSON missing-scalar rule.)

@dataclass(frozen=True)
class ScanResult:
    path:         Path
    content_hash: str | None       # None = "not needed", never "unknown"
    animal:       str | None       # extract_animal_letter; works on both cohorts
    session:      str | None
    condition:    Condition | None            # None when unresolved
    status:       Literal["matched","ambiguous","unknown","duplicate","known"]
    matched_rule: str | None
    candidates:   list[str]
    unparsed_stim_tokens: list[str]  # e.g. ["E1000"] -- see below. NON-EMPTY
                                     # BLOCKS the row from a corpus even when
                                     # the epoch matched.

def scan(root: Path, rules: Rules, known: Registry) -> list[ScanResult]
def apply_corrections(results, corrections, user: str) -> None   # merges into meta.json
```

### Required behaviours
- **`gems doctor` and the preflight also report the platform, the resolved root,
  the longest path the current layout would generate, and whether any stored path
  is absolute.** Cross-platform breakage is invisible on the machine that caused
  it.
- **Preflight check** before any training or inference run: resolve `gems_root`,
  assert a `.gems-root` marker file exists, verify every file the run needs is
  present and checksum-clean, and report anything still syncing. **Refuse to start
  on an incomplete corpus** — a model trained on a half-synced dataset is
  indistinguishable from a bad model.
- **Never write scratch or intermediate files into `gems_root`.** Envelopes,
  feature matrices and temp artifacts go to a local cache keyed by content hash.
  Syncing 16 GB of envelopes to every lab member is a real risk here.
- **Large-file hygiene:** mark the `data/` folders the run needs as available
  offline before a long job; streaming reads from Drive File Stream will otherwise
  dominate runtime.
- **Per-user identity** from git config or an explicit setting, recorded in every
  label file and registry line. "Who labelled this" is scientific metadata.
- **Role check in preflight is a WRITE PROBE, not a role query.** Drive roles are
  not visible from the filesystem. Attempt a write into the root and, on
  `PermissionError`, emit: "you are a Contributor; Drive for desktop makes that
  read-only — ask for Content manager". Name the function for what it does.
- **Item-count check in `gems doctor`:** report items against the 500,000 cap and
  warn above 80% — but report it as a **bound, not a number**. Walking 500,000
  items over a streamed Drive is not something a `doctor` run can do, so early-exit
  at an `--item-cap` and make the full count opt-in. Trash counts toward the cap
  and is **invisible to the filesystem**, so the figure is always a lower bound;
  say so in the output.
- **`gems_root` is per-user config via `platformdirs`** (rule 15 — `%LOCALAPPDATA%`
  on Windows), never committed. Resolution order: explicit argument → `GEMS_ROOT`
  env → platformdirs config → legacy `~/.gems/config.toml` → bounded scan for the
  marker. **Never reconstruct the root from the drive name.**
  Provide `gems doctor` to print the resolved root, sync status, shard count and
  any checksum failures.

### Shared-drive specifics (this lab uses a Google **shared drive**, not a personal Drive)

**Roles — and the trap.** Shared-drive roles are Manager / Content manager /
Contributor / Commenter / Viewer. "Contributor" looks like the right least-privilege
choice for lab members, and it is wrong here: **in Google Drive for desktop,
Contributors have read-only access.** Anyone who labels, trains or exports through
the synced folder must be **Content manager**.

| Who | Role | Why |
|---|---|---|
| everyone who labels / trains / exports | **Content manager** | can add, edit, move and trash; **cannot permanently delete** |
| one or two owners | Manager | membership, permanent delete, trash purge |
| collaborators who only read results | Viewer / Commenter | |

Content manager is a real safety net: deletions go to the shared-drive trash and
only a Manager can purge, so an accidental `rm -rf` inside `gems_root` is
recoverable. Do **not** grant Manager broadly just to avoid a permissions error.

**Why a shared drive is the right call:** files are owned by the drive, not by a
person. When a student leaves the lab, nothing disappears and nothing needs
transferring — which is the failure mode of a personal Drive shared out.

**Item cap: 500,000 items per shared drive**, counting files, folders, shortcuts
and trash. This is the one limit this design can actually hit, because of
per-trial metadata.

> **Do not write one file per trial.** Write **one JSONL per session**, appended
> per trial: `trials/<animal>/<session>/trials.jsonl`. Acquisition is a single
> writer, so append is safe there, and this keeps the item count in the thousands
> instead of the hundreds of thousands. It is also far faster to sync — many tiny
> files is the worst case for Drive for desktop.

Budget the item count before committing to a layout, and report it in
`gems doctor`. Trash counts toward the cap, so a Manager should purge periodically.

**Other shared-drive limits worth knowing:** 750 GB uploaded per user per 24 h
(a 25-minute 9-channel float32 recording is ~1.3 GB, so ~570 recordings/day — not
a practical constraint, but relevant to a bulk initial migration); 5 TB max file
size; 100 levels of folder nesting; a file lives in exactly one folder (use
shortcuts, not copies).

**Paths differ per machine**, so `gems_root` stays per-user config with
auto-discovery by scanning for the `.gems-root` marker. **Nothing written into
`gems_root` may contain an absolute path** — store POSIX paths relative to the
root and resolve locally (see the Cross-platform rules in `CLAUDE.md`).

> **This drive's name contains a character Windows cannot use.** It is
> `BIONICs Lab: Enteric Interfaces Team`, and `:` is illegal in a Windows path,
> so Drive for desktop substitutes it and **the folder is not the same string on
> Windows**. Never reconstruct the root from the drive name; always discover it
> via the marker file. Confirm what Windows actually produces before onboarding
> the first Windows user.
>
> Windows `MAX_PATH` is 260 unless long paths are enabled, and this data already
> sits at ~193 characters on Windows before the tool appends anything. Keep
> generated segments short and check the deepest path the layout can produce.

| OS | Typical root |
|---|---|
| macOS | `~/Library/CloudStorage/GoogleDrive-<account>/Shared drives/<DriveName>` |
| Windows | `G:\Shared drives\<DriveName>` |
| Linux | **no official Drive for desktop client** — rclone or equivalent; confirm before assuming a lab Linux box can participate |

**Streaming vs offline.** Drive for desktop streams shared-drive files by default.
Random-access reads into a streamed HDF5 are slow enough to dominate training
runtime, so either mark the needed `data/` folders available offline, or have the
preflight **copy the run's inputs into the local content-addressed cache first**.
Do the copy; it is more predictable than relying on pinning.

### Concurrency is still not free — state the limits
This design tolerates **concurrent appends** and **concurrent reads**. It does not
make Drive transactional. Two users training the same corpus simultaneously will
produce two valid models and two log lines, and a human decides which is promoted.
That is the correct behaviour for a research tool, but say it in the UI rather
than pretending the conflict cannot happen.

### Tests
- two simulated clients appending concurrently produce a log whose replay contains
  both entries, in either merge order
- a Drive-style conflict copy (`events.jsonl (1)`) is merged, not lost or
  duplicated
- a truncated parquet fails the checksum and raises before training starts
- preflight refuses a corpus with a missing file and names it
- no code path writes to `gems_root/cache`
- replaying the log twice is idempotent

### Acceptance
A second lab member clones the repo, sets `gems_root`, runs `gems doctor`, and can
immediately list the same models and corpora as the first — with no manual copying
and no shared-write conflicts.
<!-- /TASK -->

<!-- TASK:03B slug=stim-split deps=03,03A gate=no -->
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
<!-- /TASK -->

<!-- TASK:04 slug=derivations deps=03 gate=no -->
## Task 04 — Derivations (V1, V2, V3, T)

**Module:** `derive/derivations.py`
**Depends on:** 03
**Gate:** no

### Purpose
Produce the signals every later step consumes. Both representations are kept and
they do different jobs: raw contacts carry the artifact evidence and the
inter-contact delay; the tripole has a lower σ and is what spike detection reads.

**Do not quote a σ-reduction factor as a constant.** This document has carried
both "~6×" and "2.5–2.8×" and **both are wrong as constants.** The ratio is not a
property of the tripole; it is a property of how much common mode a recording
happens to contain. Holding contact gains fixed and sweeping only the
common-mode amplitude moves it from under 2 to nearly 10 (measured in task 04,
and tested as a monotone sweep rather than asserted). 2.5–2.8× is what animal J's
baseline contained. Report the measured ratio per recording as a **QC number**;
never let anything downstream depend on a fixed value.

### Signature
```python
def build_derivations(rec: Recording) -> tuple[dict[str, np.ndarray], dict[str, CuffWeights]]:
    """Returns ({signal_name: trace}, {cuff_id: CuffWeights}).
    Names are cuff-prefixed: 'L_V1', 'L_T', 'R_V2', ... plus 'stomach_ref'."""

@dataclass(frozen=True)
class CuffWeights:
    a: float; b: float                  # APPLIED -- always 0.5 / 0.5
    fitted_a: float; fitted_b: float    # DIAGNOSTIC, never applied; nan if unfittable
    degenerate: bool                    # fitted a -> 0 or 1: the fit collapsed
```

**The fit is computed as a diagnostic and never applied.** The Acceptance below
asks that `(a, b)` be stable across sessions and reads drift as electrode
degradation — vacuous if the reported pair is the applied one, which is 0.5/0.5
by construction and cannot drift. A 2-tuple cannot say which was applied, and
hiding that is how a later bug gets written. `degenerate` exists so a drift log
does not read the measured `a → 1.0` collapse as drift.

**Why fitting cannot help much, mechanistically:** 0.5/0.5 cancels a common mode
**exactly** whenever the outer gains are symmetric about the middle one —
`(1.2, 1.0, 0.8)` and even `(1.4, 1.0, 0.6)` cancel perfectly. The fit can only
beat naive on the *asymmetric* part of a mismatch. That is a plausible mechanism
for the measurement below, and it is now a test.

### Algorithm
```
T = a·V1 + b·V3 − V2,    subject to a + b = 1
```
**MEASURED 2026-09-19 (animal J): do not fit — use the naive 0.5/0.5.** On cuff L
both variance-minimising fits went degenerate (`a → 1.0`, i.e. the tripole
collapsed to a bipolar) and gave σ(T) = 3.35 µV against 2.92 µV for 0.5/0.5. On
cuff R the fitted and naive weights differ by <1% in σ(T). A bounded search over
`a ∈ [0.2, 0.8]` lands on 0.50 and 0.59 — no better than naive. Minimising 20–300
Hz variance reduces that band 32–36× but **does not improve, and can degrade, the
300–5000 Hz noise floor**, which is the band that matters.

**Strong empirical support for invariant 6 (detect on raw contacts, not `T`):**
the fraction of samples above 4.5σ in 300–5000 Hz is **5.8–6.4% on single contacts
but 0.044% on the tripole** — a ~130× reduction. Those single-contact "events" are
overwhelmingly common mode, which is exactly the artifact evidence the tripole is
defined to remove. Conversely, the clean-file z>3 flag rate is *higher* on the
tripole (5–9%) than on single contacts (0.2–2.6%), so the tripole is also the
wrong place to threshold.

Historical note — the original instruction was to fit `(a, b)` per cuff by
minimising `var(T)` restricted to **20–300 Hz** — the band where motion dominates and neural content is minimal. One
free parameter, so solve in closed form or with a 1-D scalar minimiser; do not use
a general optimiser.

This corrects contact-impedance mismatch, which a hardware short cannot. For
`config == "hw_tripole"` the tripole arrives pre-formed: pass it through and record
`(a, b) = (nan, nan)`.

Stomach: old cohort was hardware-referenced in TDT and passes through. **The new
cohort's reference was unspecified in this design and is now decided: common
average across the stomach contacts, recorded and warned.**

**It is not a neutral choice and the warning is load-bearing.** The gastric slow
wave is largely *common* across the array, so a common average attenuates part
of the very signal the `slow_wave` and `mmc` consumers read. It is the
defensible default when no reference electrode was designated, but if one
actually was, that is a different signal and this must change. Expose
`build_stomach_reference` publicly and record the path in provenance (task 15) —
the 2-tuple return has no slot for it.

**Open question for Andrea:** was a specific stomach contact intended as the
reference in the new cohort, or is common-average correct?

### Tests
- inject a known common-mode component with unequal per-contact gains; assert the
  fitted `(a,b)` recovers the gain ratio to within 5% and that `T` suppresses the
  common mode by >20 dB
- assert `a + b == 1` exactly (to floating point)
- assert σ(`T`) < σ(`V1`) on synthetic data with common-mode present, and that the
  ratio is ≈1 when it is absent
- `hw_tripole` input passes through unchanged with NaN weights

### Acceptance
On real recordings, `(a, b)` lands near `(0.5, 0.5)` and is **stable across
sessions for the same animal**. Log it — drift in `(a,b)` across weeks is an
electrode-degradation signal (see task 15).

### Do not
Do not run detection on `T` alone (invariant 6).
<!-- /TASK -->

<!-- TASK:05 slug=rpeaks deps=03 gate=no -->
## Task 05 — R-peaks, gap rescue, best-channel ranking

> **Measured 2026-09-23 on `gems_j_t01_ms3_bl_230315`: no channel passes the
> provisional vetoes.** All nine exceed `PROVISIONAL_MAX_IMPLAUSIBLE_FRAC = 0.10`
> except `ANT1`, which fails `PROVISIONAL_MAX_RESCUE_RATE` at 0.44 — and `ANT1`
> scores well on the first veto **because it missed 53% of the beats**. A veto
> that a channel passes by detecting less is not a veto; rank on beat-count
> consensus, not on lowest `implausible_frac`, and add the transient veto
> described in task 02. Task 09's gate numbers must account for a runaway
> recording disqualifying every channel rather than assuming at least one
> survives.

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

**A sharper property than "monotone and stable", and a better assertion:
tightening the fraction walks the global form's retained RR *toward* the truth
and it stops there — it never overshoots.** Measured on the two-population
generator (true RR 150.4 ms):

| fraction | kept | retained RR | vs true |
|---|---|---|---|
| 0.60–0.75 | 765 → 694 | 75.7 → 76.7 ms | 0.50× |
| 0.85 | 472 | 146.0 ms | 0.97× |
| 0.95 | 396 | **150.4 ms** | **1.00×** |

The local form goes to 2× and locks. Assert the non-overshoot, not the
monotonicity.

**And a limit on the global rule, worth knowing:** it is only as good as the
provisional peak set. When a false-peak population approaches the size of the
real one, the whole-file median lands on the **false** interval — measured at
85 ms against a true 150.4 ms, i.e. the R-to-T interval. That is not an argument
for the local form, whose error at the same point is 303–364 ms and unbounded;
the global rule's error is *bounded by the contamination*. But the operating
point carries the rule.

**QC consequence:** a `global_rr_s` far from the other channels' means that
channel's peak set is contaminated — not that the heart rate changed. Emit it
per channel and compare across channels.

> **The runaway needs TWO populations, not one** — corrected 2026-09-22 after
> building it. A false-peak population **alone is self-correcting** under the
> local rule: dropping every T wave leaves exactly the true RR, and the local
> form recovers 395/395 beats. The runaway needs T waves **plus** a
> long-interval population (≈50% of beats attenuated below threshold). Then it
> reproduces across seeds — local 303–364 ms retained RR and 59–63% dropped,
> against animal J's 330 ms and 54%; global 84.5–87.0 ms and 9–13%.
>
> The mechanism is **bistability**: once a run of long intervals lifts the
> accepted median above one RR, every real interval reads as "too short", which
> leaves alternate beats, which makes the median 2×RR, which locks it in.
>
> It needs false peaks to seed the feedback, and `make_ecg` produces almost
> none. On animal J the short-interval
> population clustered at **~90 ms ≈ 0.55 × RR** — the signature of detecting the
> **T wave** as well as the R peak.
>
> **Add `make_ecg(t_wave=True)`**: a second deflection at 0.5–0.6 × RR.
>
> **WIDTH, not amplitude, decides whether it is detected** — measured, and not
> what I expected. The band pass is a *shape* filter, so a narrow T wave reads as
> a QRS however small it is, and sweeping amplitude 0.10→0.50 at fixed width
> barely moves the peak count because σ moves with it. Sweeping width against
> 395 true beats: 40 ms → 768 peaks (every beat doubled), 50 ms → 553,
> **60 ms → 395 (correct)**, 80 ms → 395. Set the width at the crossover
> (`T_WAVE_WIDTH_FACTOR = 6.0`, 60 ms) so one generator serves both tests: the
> binding k=6 detector ignores it, the superseded k=3 doubles every beat. That reproduces both the short-interval cluster *and* the runaway,
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
<!-- /TASK -->

<!-- TASK:06 slug=envelopes deps=04,03B gate=no -->
## Task 06 — Band envelopes, reference, z

**Module:** `bands/envelope.py`, `bands/reference.py`, `bands/zscore.py`

> **This step consumes an EPOCH, not a file.** The whole-file reference is
> whole-*epoch*. It must **refuse** a recording whose condition is
> `stim_recovery` and which has not been split — silently accepting one puts the
> stim artifacts into the reference and the MAD, which is the failure this
> ordering exists to prevent (see below). Make that a checked precondition, not
> an assumption: then the dependency on 03B is a contract rather than a
> build-order coupling, and a baseline recording needs no split at all.

**Depends on:** 04, **03B** — the stim epoch must already be split off, or its
artifacts inflate the reference and MAD and suppress detection during recovery
**Gate:** no

### Signature
```python
def band_envelope(x, fs, lo, hi, window_s, grid_s=GRID_S) -> np.ndarray   # (n_frames,)
def log_envelope(env) -> np.ndarray                  # log(max(env, eps)), one place
def epoch_reference(log_env) -> Reference            # (median, scale), NOT one float
def zscore(log_env, reference) -> np.ndarray         # refuses a foreign (signal, band)
```

The first draft of this block contradicted the Algorithm below it in three ways
and the Algorithm is the binding one. It carried a `pct` argument, which is the
superseded linear/percentile rule that A.3 deleted so that it could not compete
with invariant 5 — there is no percentile. It returned one float, but the rule
produces **two** scalars and `zscore` had nowhere to take the second. And both
took `env` while steps 3–4 operate on `log(env)`, so either the log happened
twice or in neither place. `log_envelope` is a separate step precisely so there
is exactly one site where it happens. `file_reference` is renamed
`epoch_reference` because the banner above redefines the scope to whole-epoch;
a function named for a file while operating on an epoch is the same drift that
left A.4 naming a band `1-100` after detection had moved off it.

### Algorithm
1. Band-limit with `sos`. For bands below ~100 Hz, **decimate first** — a 1 Hz
   corner at 24.4 kHz is a normalised frequency of 8×10⁻⁵ and `butter(...,'ba')`
   returns numerical garbage there. This bug produced a MAD-σ of 10²⁰⁸ during
   design; guard against it with an explicit assertion on the filter output range.
2. RMS envelope over the band's own window, sampled on the shared 10 ms grid.
3. Take logs: `l = log(max(env, eps))`. Envelopes are positive and right-skewed;
   the log is what makes the null symmetric.
4. `ref = median(l)` and `scale = 1.4826 × MAD(l)`, **one scalar pair per
   (signal, band), over the whole file**. Then `z = (l − ref) / max(scale, floor)`.

   *Measured 2026-09-19:* the earlier linear form (`ref = p10` of the linear
   envelope, scale = its MAD) puts median z at 1.00 and p90 at 3.05 — about 10% of
   frames over z = 3 **per pair**, which the union across 36 pairs turns into
   40–55% of the file. The log form gives median 0, p90 1.44, p99 4.21. See
   invariant 5 and task 09.
5. Gate dead/saturated channels **before** dividing: a flat channel gives MAD → 0
   and z → ∞.

### Why a whole-file scalar and not a running window
A running baseline adapts to slow change — and in a post-stim recovery file the
slow change *is the measurement*. A rising baseline would require artifacts to be
larger to cross threshold during recovery than at baseline, manufacturing a
condition confound. It also has no edge problem, which matters because the post-stim
dynamics live in the first window.

### Memory
9 channels × 6 bands × 37 M samples is ~16 GB if materialised. **Stream per band**;
only the 100 Hz envelope is retained. "100 Hz" here is the **frame rate** of the
10 ms grid, not a band — no band is named `1-100` any more. That reading is the
one that makes the arithmetic above work.

**Ceiling: 1 GB peak RSS.** Streaming one band of one channel at a time gives a
working set of one channel plus its decimated copy, about 350 MB at
9 × 25 min × 24.4 kHz, with **64.8 MB** of retained frame-rate envelope
(9 channels × 6 bands × 150,000 frames × 8 bytes — the 8 MB first written here
was one channel, not nine) — so 1 GB is
roughly 3× headroom, loose enough to survive the difference between how macOS
and Windows account for resident memory. Measure it with `psutil` as a
**test-only** dependency; `resource` is absent on Windows and `tracemalloc`
undercounts numpy's buffers. Keep the algorithmic assertions (nothing larger
than the input is materialised; the retained output is frame-rate) as the
primary test — they are the property that matters and they hold on every
platform.

### Filter settling is not negligible in the slow bands — measured

`sosfiltfilt` does not remove the edge transient, it only halves it, and the
default `padlen` is set from the filter *order* rather than from the length of
its impulse response. At a 0.5 Hz corner the impulse response runs for seconds
while the default pad is a fraction of one. The consequence is not cosmetic:

| band | dof, untrimmed | dof, one window trimmed per end | spec |
|---|---|---|---|
| 300-3000 | 135.3 | 135.3 | 135.0 |
| 100-300 | 30.5 | 30.6 | 30.0 |
| 10-150 | 28.5 | 28.5 | 28.0 |
| 2-50 | 27.8 | 29.9 | 29.8 |
| **0.5-3** | **22.9** | **32.0** | 30.0 |
| **0-2** | **26.3** | **31.7** | 30.0 |

Individual seeds in `0.5-3` reached an effective dof of **0.7** untrimmed — the
transient carrying more envelope variance than the signal. Trim one analysis
window at each end of every segment before computing the reference. The fast
bands are unaffected, which is why this stayed hidden.

**Two quantities are being conflated and they must be separated**, because they
coincide numerically by accident: the **filter** settling time is a property of
`(lo, hi, fs, order)`, and the **analysis window** is a property of the dof
budget. `window_s` was chosen as roughly `dof / (2B)`. Measure the filter term
from the designed `sos`'s impulse response, at **1% of peak** (task 13's own
criterion, so the two are comparable), and define

```
settling_s(band) = max(impulse_response_length_s, window_s)     # report BOTH
```

Measured, and the separation was not cosmetic — **two bands are filter-limited**:

| band | impulse response | `window_s` | binding term |
|---|---|---|---|
| 300-3000 | 0.005 s | 0.025 | window |
| 100-300 | 0.034 s | 0.075 | window |
| **10-150** | **0.141 s** | 0.100 | **filter** |
| **2-50** | **0.486 s** | 0.310 | **filter** |
| 0.5-3 | 3.988 s | 6.000 | window |
| 0-2 | 1.146 s | 7.500 | window |

The one-window trim first specified here was **71%** and **64%** of what those
two bands actually need. A test moves `window_s` by 10× and asserts the
impulse-response term does not move.

### `padtype`, not `padlen` — the edge transient is an artifact of odd extension

Raising `padlen` to the impulse-response length makes the slow bands
**substantially worse**, which is the opposite of the prediction that motivated
measuring it. Effective dof against a spec of 30, pooled over seeds of
10-minute white noise:

| band | no trim, default pad | no trim, raised pad | trim (shipped) |
|---|---|---|---|
| 0.5-3 | 22.9 | **9.9** | 32.0 |
| 0-2 | 26.3 | **19.5** | 31.7 |
| 2-50 | 27.8 | 28.9 | 30.0 |

Worst single seed at `0.5-3`: **0.9** raised, 29.2 trimmed.

The cause is scipy's default `padtype="odd"`. Odd extension reflects
antisymmetrically about the endpoint, which for a band-pass with a low corner
injects a large artificial **low-frequency** excursion directly into the band
being measured — so a longer pad injects more of it, and the "fix" makes the
defect worse.

**`padtype="constant"` at the default `padlen` is binding.** All six bands, no
trim, 8 seeds × 10-minute white noise, worst single seed in brackets:

| band | type | spec | `odd` | `even` | **`constant`** |
|---|---|---|---|---|---|
| 300-3000 | bandpass | 135.0 | 135.3 (134.3) | 135.3 (134.3) | 135.3 (134.3) |
| 100-300 | bandpass | 30.0 | 30.5 (30.1) | 30.5 (30.1) | 30.5 (30.1) |
| 10-150 | bandpass | 28.0 | 28.5 (28.1) | 28.5 (28.1) | 28.5 (28.1) |
| 2-50 | bandpass | 29.8 | 27.8 (24.3) | 29.9 (28.9) | **29.9 (28.9)** |
| 0.5-3 | bandpass | 30.0 | 22.9 (0.7) | 25.0 (7.5) | **30.6 (24.6)** |
| 0-2 | **lowpass** | 30.0 | 26.3 (3.1) | 24.8 (5.8) | **31.6 (27.4)** |

`constant` is best or tied-best in all six and never worse than `odd`, so this
is a correction, not a trade.

**`even` is not an alternative and must not be substituted.** It ties on the
band-passes above 2 Hz and is **worse than `odd`** on the `0-2` low-pass
(24.8 against 26.3). It looked viable in a three-band spot check run at a
*raised* pad, where it does work; at the default pad it does not. Two mechanisms
are at work and only one of them is the DC argument:

- **Band-passes (five of six):** a constant pad is pure DC, which the
  high-pass side rejects in-band by construction.
- **The `0-2` low-pass:** DC is *inside* the passband, so it is not rejected —
  yet `constant` still wins by the largest margin of any band. The reason is
  continuity, not rejection: a flat extension matches the endpoint in value and
  has zero slope, so there is no step for the filter to ring on, and a DC offset
  moves the envelope's mean while leaving the relative variance that dof
  measures untouched.

**The trim stays, and its justification is now different from the one it was
introduced with.** Under `odd` the trim was what made the slow bands work
(`0.5-3` was 22.9 untrimmed). Under `constant` the untrimmed figure is **30.6**,
so the padding alone carries the statistic and that argument is spent. The
remaining argument was always the stronger one: **padding fabricates samples**,
so an edge frame's filter input is partly invented whatever the `padtype`, and
a frame whose input cannot be validated is `unassessable` for detection however
healthy the aggregate dof looks. The padtype defends the *statistic*; the trim
withholds the *frames*.

Note the direction of the bias, which is what makes withholding the right
response rather than a conservative one: the pad is quiet, so edge frames are
biased **toward looking clean**. An untrimmed edge is a systematic
false-negative region — the same failure shape as a stim-inflated reference,
detection suppressed silently in a fixed part of every recording.

Note also that trimming now *raises* measured dof slightly away from spec
(`0.5-3`: 30.6 untrimmed, 32.0 trimmed) because it shortens `T`. **That is not
a reason to remove it.** Anyone optimising this number later will find that
argument and it is answered here.

The cost is ~1% of a 1200 s recovery epoch: at this stage NaNs come from
acquisition dropouts, not from masking (which happens later), so segments are
not in fact fragmented.

Enforce the separation from 03B as an **import-graph** assertion — `bands` must
not import `stim_split` — rather than by grepping module source for a call name.
The import edge is the invariant; a source-text test breaks on reformatting and
passes on a re-exported alias.

**This does not resolve 03B's `None`.** The epoch-boundary `unassessable` width is
the maximum over *every* filter that touches the boundary, and the consumer
filters (task 13) are a different chain from these detection bands. Task 06
supplies the detection-side term only. `Epoch` settling stays `None` until 13
lands; a partial maximum reported as the answer would be too small, which is the
one direction that silently loses coverage.

### Vocabulary: three different things are called "epoch"

`Condition.epoch` is `baseline` / `stim_recovery`. `stim_split.Epoch` is the
stim-or-recovery slice. The NaN rule below means a contiguous run of valid
samples. **The third one is called a `segment`** and the word "epoch" is never
used for it anywhere in the codebase.

### NaN handling
Interpolate-then-restore is fine where gaps are short relative to the band's period
(30 ms at 300–5000 Hz). For **0–2 and 0.5–3 Hz**, a 1 s gap is half a cycle of the
signal being measured — process **segment-wise** with a minimum segment length and
mark short segments `unassessable`.

### Known limitation — do not fix here
In the ENG band the envelope contains neural activity, so `z` conflates signal and
noise by construction. Measured: clean-frame p99 rises 2.33 → 22.14 between a quiet
and an active animal. The discriminators are **classifier features** (onset rate,
cross-channel commonality, band ratio), not generator parameters. **Do not tune the
generator to stop over-firing during activity changes.**

### Tests
- white noise of known σ through each band: measured envelope matches the
  analytic expectation within 5%
- effective DOF check: the variance of the envelope matches **that band's own
  spec dof** within 20% — not a global 30, which is only correct for four of the
  six. There is **no ENG exemption**; `constants.py` claiming one is wrong and
  should be corrected. All six pass against their own value.
- the same check at the full sample rate **fails** for `2-50` (effective dof 11
  against 29.8): decimate-first is load-bearing for *accuracy*, not only for the
  numerical stability that step 1 gives as its reason
- the envelope of white noise matches the analytic expectation only when the
  filter's **equivalent noise bandwidth** is used (`filtfilt` passes ~90% of
  nominal — twice the 5% tolerance) and when the expectation uses the **input's**
  Nyquist. White noise of fixed σ is not the same signal at two sample rates, so
  the fs-independence test uses a sine.
- settling: `0.5-3` untrimmed gives an effective dof below 25 and trimmed gives
  30 ± 3; assert both, so the trim cannot be removed silently
- `settling_s` reports the impulse-response term and the window term separately,
  and returns their maximum
- filter-stability assertion fires on a deliberately ill-conditioned `ba` design
- a flat (dead) channel is gated, not divided by zero
- slow-band segment handling: a 1 s gap in a 0.05 Hz signal produces two
  segments, not one interpolated trace
- memory: a 25-minute 9-channel synthetic completes under a stated RSS ceiling

### Acceptance
All six bands produce finite z on real data; peak memory stays within the ceiling;
the reference scalars are logged per (signal, band).
<!-- /TASK -->

<!-- TASK:07 slug=candidates deps=05,06,02 gate=no -->
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
not in it. **Both of those are the wrong filename.** Reading `browseMotionArtifacts.m`:
it is called as `browseMotionArtifacts(y, fs, 10, mode, fullfile(folderPath,
condition))` and writes **three** files beside the recording —

```
<condition>_segments.mat           removedSegments
<condition>_segment_indices.mat    removedSegmentIdx
<condition>_blankmotion.mat        yOut, fs, t, removedSegments,
                                   removedSegmentIdx, blankingApplied, hrChanIdx
```

(with `_stim_` / `_recovery_` infixed in `stim_rec` mode). **The third file
contains the labels as well**, it is what `batch_process.m` consumes downstream,
and Andrea confirms it is the one that matters — the two sidecars may never have
been kept. **Search for `*_blankmotion.mat`, and read
`removedSegmentIdx` out of it.**

It also carries `hrChanIdx`, which closes the separate open question about a
manual heart-rate channel for the old cohort.

The extended search that found nothing (**zero** across 8721 directories:
GEMS-Andrea 3442, GEMS-Lyna 2012, Louise 99, Arjun 3168) therefore proves less
than it appeared to. Andrea confirms the old cohort lives on **a different Drive
or account**, not on the team drive — consistent with its naming convention
(`E10_FRE_E10_stim_rec_0030_...`) matching nothing in the new corpus.

`duration_cap_s` is therefore computed from `removedSegmentIdx` **inside the
`*_blankmotion.mat` files**, not from a sidecar.

**Found 2026-09-23, and not by walking:**
`processing_new/investigate_implausible_alpha.m:38` hardcodes the path.

```
G:\Shared drives\BIONICs Lab Workspace\Project Folders\GEMS\Survivals
```

A second shared drive the account already had. **406 `*_blankmotion.mat`,
272.7 GB**, animals Lollipop 128 / Jelly 120 / Oreo 78 / Frenchtoast 54 /
Nutella 9 / Mochi 9 / Twix 1, 7 unparsed. Stem form is
`<cond>_<animal>_<site>_<bl|stim_rec>_<nnnn>[_notched_v0.2.2][_stim|_recovery]_blankmotion.mat`
— **the animal is the second token, the first is the condition.** Reading the
code for the path rather than enumerating the drive is the lesson worth keeping:
the walk found one stray `.fig`; the source found the cohort.

### The `_stim_` files are protocol exclusions, not motion labels — EXCLUDE THEM

152 of the 406 are `_stim_`, and they are blanked **91–99.6% in a single segment
spanning `[0, 119.5] s`**. That is not artifact labelling. That is someone
excluding the stim epoch by hand — exactly what task 03B now does
deterministically, done manually and years earlier.

**Ingesting them as artifact labels would teach task 12 that two minutes of
every stim recording is one artifact.** It would also destroy `duration_cap_s`,
whose entire purpose is the 99th percentile of *event* durations: 152 segments
of 119.5 s would set the cap above any real event and the cap would never fire.

Rule: **a segment whose duration exceeds `0.9 × epoch_duration` is a protocol
exclusion, not an event.** Count it as `excluded_epoch`, never as
`masked_motion` (task 03B's accounting rule, arriving from the other
direction), and drop it before computing `duration_cap_s` or building any
training corpus. Only the **125 `bl` and 122 `recovery`** files carry partial
coverage (20–82%) and only those are labels.

### Loading these files — four things measured, all of which bite

1. **`removedSegmentIdx` is N×2 `[start stop]`, 1-based inclusive.** Confirmed
   from `timeToIndices` at `browseMotionArtifacts.m:502`, not inferred from
   shape. This is hard invariant 15's boundary and it is documented in their
   source, so convert once, at the boundary, with a test.
2. **v7.3 files store that array transposed as (2, N).** The cohort mixes v5 and
   v7.3, and `reshape(-1, 2)` on the v7.3 layout pairs `start[k]` with
   `start[k+1]`, producing overlapping ranges and reading **86–93% NaN instead
   of 100%** — a plausible-looking wrong answer that reads as partial
   corruption rather than as a bug. Branch on format, transpose rather than
   reshape, and make the regression test a v7.3 file with a known mask
   fraction.
3. **`removedSegments` and `removedSegmentIdx` can disagree** — one file had
   `removedSegments` as an empty `uint8 (0,0)` beside a valid `Idx` pair.
   **Trust `removedSegmentIdx`.**
4. **Animal tokens collide on more than case**: `JEL`/`jel`/`Jel`/`jelly`,
   `LOL`/`lol`/`Lol`/`loll`/`loll2`/`loli`, `FRE`/`fre`/`ft`. Case-insensitive
   matching (rules 7 and 8) handles the first kind and **not** the second —
   `loll2` and `loli` are not case variants of anything. This needs an explicit
   **animal alias table, inferred and then user-confirmed**, the same pattern as
   03A's condition-name inference. Do not let a silent normaliser decide that
   `ft` is Frenchtoast.

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
<!-- /TASK -->

<!-- TASK:08 slug=matlab-fixes deps=01 gate=no -->
## Task 08 — MATLAB correctness fixes in `processing_new`

**Module:** `processing_new/*` (independent of the detector; run in parallel)
**Depends on:** 01
**Gate:** no

### Purpose
**Most of these currently make a better detector look worse**, because better
detection produces more, shorter, better-placed gaps. Fixing them first means the
detector is evaluated on instruments that can register its improvement.

*Revised 2026-09-22 — the original text said "each of these", and measurement
does not support the uniform claim.* Two rows are decisive, one is not:

- **Slow wave: decisive.** A **one-second** blank anywhere in a 60 s window
  currently returns `NaN` for the whole window. Pooling across clean runs
  recovers 14.24 / 15.00 / 13.85 cpm for 1 s / 4 s / 8 s gaps where the current
  code returns nothing. This is the instrument problem in its purest form.
- **Band corners: decisive by construction**, since a per-consumer extent
  computed for 100–5000 when the consumer analyses 300–3000 is measuring the
  wrong thing.
- **HRV: not an instrument problem.** See that row. The fix is kept as hygiene,
  not as a recovery of lost sensitivity, and **no historical HRV number
  changes**.

| File | Change |
|---|---|
| `step0_load_data.m` | read per-consumer masks; NaN the `removedSegmentIdx` regions if task 01 landed on the MATLAB side |
| `step1_bandpass.m` | **NEW, and load-bearing: the corners are 100–5000 Hz, but A.5b moved this project's ENG band to 300–3000.** If they disagree, every per-consumer extent is computed for a band the consumer does not actually analyse. Change the MATLAB corners to match, and accept that σ and therefore every historical spike count changes with it (A.5b already says reprocess rather than mix) |
| `pipeline_params.m` | `edgeBufferMs` from measured `impz`; `cardiacRemoveWinMs` **and** `envCardiacGuardMs` from task 02 — both, or censoring is inconsistent across stages |
| `step1a_blank_cardiac.m:41-43` | per-channel, per-band windows instead of `D.y(blank,:) = NaN` |
| `step2_noise_sigma.m:82-97` | **keep** the Quian Quiroga estimator — it is correct (`std` inflates 35% at 20 spk/s where Quiroga inflates 1.9%) — but take σ from a **fixed session reference**, not a 5 s running window. **Done and measured:** across a 5 → 100 spk/s ramp at a constant true 5.0 µV, the running threshold drifts **+5.5%** and the session threshold 0.0%. The session reference is the **median of the valid window estimates** per channel, not a global MAD — a contiguous contaminated stretch inflates only the windows it lands in, and the median over ~240 of them discards those. The damage is **attenuation of the contrast**, which is the number to quote: true rate ratio 7.08, session-σ measures 7.14, running-σ measures 6.97. The running window over-detects where the rate is low and under-detects where it is high, so it shrinks exactly the post-stim rate rise the experiment exists to measure. `'running'` is retained only so the two can be compared on a real file |
| `HR_BR_HRVAnalysis_new.m:160-165` | restore a deliberate 1–100 Hz band before `findpeaks` |
| `HR_BR_HRVAnalysis_new.m:276-287` | runs-aware successive differences for RMSSD, pNN5, SD1, SD2, SampEn, ApEn. The runs list already exists in `dfaRR_gapAware.m:24-25` and was never propagated. **Measured: the fix does not move the number**, by at most 0.2 percentage points across blanking fractions 1.9–13.8% and RR drifts 0–30%; worst realistic case (a 360-beat gap across a steep rate transition) old +1%, new +0.0%. The splice bug is not being hit because the pipeline **drops** the gap-spanning interval rather than emitting it — `dfaRR_gapAware`'s gap formula only makes sense if it does — so the residual error is a difference between two ordinary intervals flanking a hole, not a spliced one. The +771% measured in task 05 was for a train that *retains* the spanning interval, which is what `diff()` on beat times gives you and is not what MATLAB does. **Keep the fix** — it makes the guarantee structural rather than incidental, and `nSplicedDiffs` now reports how many differences were refused — but expect no change to any historical HRV number |
| `HR_BR_HRVAnalysis_new.m:679` | `heartCountSeries` needs a valid-duration denominator. Done: joined by `heartCountValidSec` and `heartCountRateSeries`, NaN below a **named** minimum-valid fraction (currently 0.5 — put it in `pipeline_params.m`, do not leave a bare 0.5 in the code). **`heartCountSeries` was kept for continuity, so audit every call site and redirect it**: leaving a correct and an incorrect series side by side, with the incorrect one keeping the older and more familiar name, is how the wrong one gets used |
| `movingCardiacMetrics` | Found during row 4: the **windowed** HRV path had the same `diff(RR_win)` splice as the whole-record one. Now routed through `hrvRunsAware`. Expect no numeric change, for the reason given in the row above |
| `HR_BR_HRVAnalysis_new.m` | promote `RR_implausibleFraction` / `br_implausibleFraction` from warnings to masks — for a periodic always-present signal, an implausible rate *is* evidence of contamination. Done: RR outside [100, 500] ms is excluded and `rrRuns` treats the hole as a run boundary, so no successive difference crosses it; beats are **not** invented back, consistent with the gap-rescue decision in task 05. **Two asymmetries to resolve:** the breath path *drops a peak* where the cardiac path *drops an interval*, and it drops **the later peak of each implausible pair** — an ordering rule, not a principled one. Drop by **lower prominence** instead, and make the breath hole a run boundary too, so the two signals are censored the same way |
| `slowWaveAnalysis_new.m:159-161` | pool peaks across clean runs instead of taking only the longest — two clean 28 s halves in a 60 s window currently return NaN |
| `bulk_mixed_models.m` | coverage weights + covariate + minimum-coverage exclusion. `nRR_used`, `fr_validFrac`, `validDur_s` are all computed and none is used. Done as three separate things, correctly: **exclusion** (`minCoverage = 0.5`), **weight** (linear in coverage), and **covariate in both the full and reduced models**, so the interaction test is at matched coverage. Unverified end to end — `normRows` is built at runtime from the unreachable archive. **Task 19 must REFUSE when `hasCoverage` is false, not warn.** A confound check that silently runs without its confound covariate reports "no confound" for the wrong reason, which is worse than not running; this is hard invariant 19's rule applied to a covariate rather than a settling time |
| `extract_mmc.m` — **the `mmc` consumer may not be detecting MMC** | Measured 2026-09-23 while deriving T: 5.0–9.3 burst events per slow-wave cycle, Rayleigh R median 0.076, **0 of 9 channel×span rows significant**, no temporal clustering (CV 0.53–1.16, largest gap 18.1 s). Most likely over-fragmentation — `group_events`' 0.5 s `burstRefractory` splitting one real spike burst into several. **Re-group the existing burst times at 2 / 3 / 5 s and see whether the count converges on the slow-wave cycle count.** If yes, fix `burstRefractory`; if no, the variable is misnamed and `mmc` in a results table is a claim the data does not support |
| `slowWaveAnalysis_new.m` — **the slow-wave peak train is unreliable** | The function's own warning on the T host spans: **23–38% of slow-wave intervals exceed 8 cpm**, i.e. shorter than 7.5 s, against a rat gastric slow wave near 5 cpm. Spurious extra peaks. This matters twice: it is the phase reference that limits the `mmc` locking result above, **and** it is the `slow_wave` consumer's own fiducial, so its tolerance inherits the same noise. Reproducible (the determinism control passes), so differencing still works — but a consumer whose fiducial train is a third implausible needs fixing before its tolerance is quoted as physiology |
| **NEW: derive MMC phase fractions per recording, and use them as a covariate** | The old cohort was **fed** (ad-lib food); the new cohort is **fasted 4–6 h with a ~1.5 kcal treat hourly**. The fed pattern replaces the MMC with continuous irregular activity, so the two cohorts are in different motility states and a model that averages over them is averaging two different experiments. Time since the last treat is not logged for data already collected. **Recover the state from the signal instead** — see the block below — and add `phase_frac_I/II/III` and `motility_state` to `bulk_mixed_models` as covariates, the same way coverage was added |
| `browseMotionArtifacts.m:34` | `validateattributes(..., 'finite')` throws on NaN, so an already-blanked file cannot be re-browsed. Remove if the browser is kept |
| ~~every `filtfilt` call at a low corner~~ | **Audited and withdrawn as a padtype problem — measured, no change needed.** The row predicted that MATLAB's mandatory odd extension would reproduce task 06's transient. It does not, and the reason is worth keeping: MATLAB pads `3·2·n_sections` = **6 samples**, which at 24.4 kHz is **0.246 ms**. Odd and constant padding differ by 11.8% of sd at 0.5 s from the edge, 1.1% at 2 s and **0.00% by 15 s**; peaks surviving the existing edge buffer, **13 either way**. scipy's damage came from odd-extending across a pad long enough for the signal to move; a quarter-millisecond pad of a 0.15 Hz signal cannot. **The settling itself is real and is already handled**: measured `impz` 8.17 s against a 15 s buffer, and the code's `order/cutoff` heuristic (13.33 s) over-estimates it, which is the safe direction. Generalising a scipy result to MATLAB without measuring was the error here |
| `extract_mmc.m` — **`xf(~isfinite(xf)) = 0`** | **Hard invariant 1, violated, live.** If `fillmissing` leaves anything non-finite it becomes **zero**, and a zero is indistinguishable from signal to everything downstream. This is the exact defect task 01 was written for — which turned out to have been fixed upstream in `a95d1ff` before this project began — found here for real, in a different file. **Highest priority row in this task.** Fix to NaN and let the consumer decide; audit the rest of `processing_new` for the same construct |
| **`extract_mmc.m` — `burstRefractory` confirmed at 0.5–1 s on fasted data** | Measured on `gems_j_t01_ms3_sr_231323` recovery (19.8 min, fasted): with a fixed threshold at 2× the quiescent floor, episodes are **1.4 s median (IQR 0.8–2.6)** separated by a **4.0 s median gap**, at **0.73 per slow-wave cycle** (4.30 cpm). Burst duration matches the literature; the refractory must sit well below the 4 s gap, so **0.5–1 s is correct and 3 s would merge adjacent episodes**. The baseline recording gave 1.2 per cycle by the same method — both near the one-burst-per-slow-wave relationship, without tuning toward it |
| **`extract_mmc.m` — the 30 s MOVING MAD is the bug, not `burstRefractory`** | **Measured 2026-09-23 on `gems_j_t01_ms3_bl_230315` (new cohort, fasted), ANT1−ANT3.** The gastric activity episodes last 2–6 s and recur every few seconds to tens of seconds, so a **30 s moving window contains the episode in its own baseline** and the threshold rises with the signal it is meant to detect. Measured, same recording, same envelope: **3× moving MAD over 30 s finds 3 episodes totalling 5 s — 1% of the record. A fixed session reference at 2× the quiescent floor finds 40 episodes, median 2.6 s, 23% duty.** This is the `step2_noise_sigma` defect a third time: an adaptive baseline that adapts to the thing being measured. **Fix the threshold, not the grouping.** With a fixed reference the episodes are already well separated (median inter-episode gap 5.0 s), so `burstRefractory` should stay **small — 0.5–1 s** and the earlier 3 s proposal is withdrawn: at a median gap of 5 s, 3 s would merge roughly a quarter of adjacent pairs |
| `extract_mmc.m` — `detect_crossings` 30 s moving MAD | An independent instance of the `step2_noise_sigma` row above: a moving noise estimate whose threshold rises with activity. Same failure, same direction, same fix — a fixed session reference |
| `extract_mmc.m` — blank restore is sample-exact | The blank-before-filter-restore pattern is otherwise done correctly (`bl` → `fillmissing` → `filtfilt` → `y(bl) = NaN`), but **only the blanked samples are restored**. With a 0.486 s impulse response, roughly half a second either side of every blank is filter output computed partly from interpolated data, and it is kept. **This is task 13's question arriving in MATLAB**: the restore must extend by the consumer's settling time, not by the blank. Audit every blank-restore in `processing_new` for the same pattern — it is a shape, not a one-off |
| `extract_mmc.m` 2–50 Hz bandpass | **The finding the padtype audit actually produced.** Measured `impz` **0.486 s** and **no edge buffer anywhere**, feeding the MMC statistic. Every other low-corner site is covered: `HR_BR` 1–100 Hz 0.159 s against 0.75 s (4.7×), slow wave 8.17 s against 15 s (1.8×), `step1_bandpass` 0.0051 s against 5 ms (marginal — raise to 10 ms). Read the MMC chain's cardiac-interpolation logic before touching it; the buffer interacts with it |
| `slowWaveAnalysis_new.m` — duplicate `smoothdata` removed | **Approved by Andrea 2026-09-26; verified bit-identical** on baseline (599 s, 3 blanked spans), stim (120 s) and recovery (1205 s, 155 blanked spans, 4.2% NaN) at 1 and 24 threads — all 17 output fields and all peak locations. Line 186 recomputed line 146 per channel. Saves 23–54% per call single-threaded. Output unchanged, so T's `slow_wave` tolerances are unaffected. |
| `slowWaveAnalysis_new.m` — decimate before filtering (**proposal, validate before adopting**) | Filters and smooths a ~0.05 Hz rhythm at the full 24414 Hz. The cost is the 5 s Gaussian `smoothdata` window (122,070 samples), which scales with span × window — **fs²** — so decimating 100× is ~10⁴ less work and 100× less saved data. **This changes the consumer**, so: compare decimated vs full-rate output (slow-wave peak times, rates, dominant frequency) on several recordings across conditions; adopt only if agreement is within a tolerance stated in advance; then re-run T's `slow_wave` rows on the adopted version (minutes, once decimated). Never adopt inside T. |
| `run_continuous.m:127,141,153` — 12 arguments to an 11-argument function | Calls `slowWaveAnalysis_new` with a 12th mode-string argument; the function takes 11 and has no `varargin`, so the call errors on reach. One side is stale. **Andrea decides which**; do not fix by guessing. |

**Reuse rather than reinvent:** `dfaGapAware.m` (pooled runs), `step5f_fano_slope.m`
(epochs + rate-matched surrogates carrying identical censoring — extend the same
pattern to CV2 and LV), `step5e_multiband_validate.m` (peri-R histogram validation).

### Deriving MMC phase fractions from the stomach EMG

**MEASURED 2026-09-23 on `gems_j_t01_ms3_sr_231323` (new cohort, fasted,
19.8 min recovery): there is no phase I, so a 3-state classifier has no
quiescent state to find.** Longest run below 1.5× the quiescent floor is **47 s**
against an expected phase I of **5.4 min**, and the 30 s epoch levels span only
8.6–21.8 µV (2.5×). In a 19.8 min window against a 17.5 min cycle, a 5.4 min
phase I should have been captured with near certainty. This animal is in
continuous phase-II-like activity.

**So report continuous covariates first, and the state classification only if
quiescence appears:**

```
duty_at_2x_floor        fraction of the epoch above 2x the quiescent floor
longest_quiescent_run_s the statistic that actually discriminates here
epoch_level_range       max/min of the 30 s epoch medians
```

These are robust whether or not the animal is cycling, and they are what the
model covariate should be. Forcing a three-state fit on a recording with one
state produces three numbers that describe the fitter, not the stomach.

##### Let the data choose the number of states, and settle it on a SESSION timeline

**Do not force three states.** Fit 1-, 2- and 3-state models per recording and
select by **BIC**. "How many states does this recording support" then becomes an
output rather than an assumption, and a recording with no quiescence reports one
state instead of inventing three.

**A 20-minute recording cannot show a 17.5-minute rhythm** — one cycle is not
periodicity. The recordings within a session are consecutive and timestamped
(`09162026` spans 16:18 to 01:15, about nine hours), so **stitch each session's
30 s epoch activity into one series in timestamp order, gaps included, against
clock time.** Cycling then becomes visible rather than inferred.

**The two competing explanations predict different periods, which is what makes
this decisive without any independent measure of motility:**

| explanation | expected period over a 6 h session |
|---|---|
| the animal is cycling (MMC) | **~17.5 min** — ~20 cycles |
| the hourly treat is resetting it | **~60 min** — ~6 cycles |
| neither: continuous phase II | no periodicity; drop the phase model, keep the continuous covariates |

An autocorrelation or periodogram of the session timeline separates twenty
cycles from six trivially.

**Three figures, in descending order of what they settle** — specified here
before the numbers exist, for the same reason the tolerance criteria were
pre-registered:

1. **Session timeline**, one panel per session, treat times marked if they can be
   reconstructed even to the nearest ten minutes. This is the one that answers
   the question.
2. **Histogram of `longest_quiescent_run_s`** across all recordings, with 5.4 min
   marked. Every recording capping below a minute is conclusive across animals;
   a few reaching minutes means those animals cycle and the rest need
   explaining.
3. **Pooled distribution of 30 s epoch levels**, per-recording normalised.
   Distinct phases make it multimodal, one state makes it unimodal. Caveat to
   state: normalising by each recording's own floor assumes the floor *is*
   quiescence, so report raw µV alongside.

**Why there is no phase I is an open question and it matters to the experimental
design**: 4–6 h fasting with a ~1.5 kcal treat hourly may not be enough to
establish interdigestive cycling, or the stim may reset it, or the treat may
keep the animal in the fed pattern throughout. Worth resolving before the
covariate is used, because "no phase I in any recording" and "no phase I in this
recording" have different consequences.

**Method.** 2–50 Hz envelope → 30 s epochs → three states from event rate and
envelope amplitude: **quiescent** (phase I), **intermittent** (phase II),
**intense** (phase III). Phase IV is ~1.8 min and will not separate reliably;
fold it into the II–III boundary and say so.

**Use a 3-state HMM on the epoch series, and note that this does not contradict
PIPELINE §10.4.** That section rejects an HMM for smoothing *candidate*
decisions, on the structural grounds that the classifier emits one decision per
event and there is no per-frame probability trace to smooth. Here there **is** a
per-frame trace — the envelope — and the states genuinely persist for minutes,
which is the situation an HMM is for. Different problem, opposite conclusion,
both correct.

**Set thresholds from the pooled distribution across recordings, not per
recording.** Per-recording normalisation would force every recording to contain
all three states by construction, which is exactly the outcome a fed recording
must be able to contradict.

**Independent validation, and it is not circular:** if the classifier is right,
the interval between successive phase III onsets should come out near the
literature value of **17.5 ± 5.8 min** for rat antrum, and the fractions near
**I 31%, II 41%, III 18%, IV 10%**. Neither number is used in fitting, so
agreement is evidence. A fed recording should show **no** phase III and no
quiescence — that is the positive control the old cohort provides for free.

**Three caveats that decide how the baseline-vs-recovery comparison is read:**

1. **One recording is about one cycle.** Baseline is ~10 min and recovery
   ~20 min against a 17.5 min cycle, so a single recording's phase fractions
   carry sampling error of roughly the phase-duration SDs (I ±1.1, III ±0.8
   min) — 20–25% relative. **Pool within condition**; do not read a single
   recording's fractions as a measurement.
2. **Baseline systematically precedes recovery by ~12 min, which is 0.7 of a
   cycle.** So recovery samples a *later cycle phase* than baseline **by
   construction**, with no stim involved. This averages out only if the cycle
   phase at recording start is random across sessions — which it plausibly is,
   since treat timing was not controlled, but it is an assumption and should be
   tested by checking whether baseline phase fractions are flat across
   sessions.
3. **Unequal durations → weight.** A 20 min recovery estimates its fractions
   more precisely than a 10 min baseline; weight by duration, as with coverage.

**Also check whether condition correlates with recording order.** Three ~22 min
recordings fit in one hourly treat interval, so if conditions run in a fixed
order within a session, time-since-treat correlates with condition and the
confound is systematic rather than random. One look at the session logs settles
it.

### Tests
MATLAB-side: for each fix, a before/after on one recording with the delta reported.
The RMSSD fix in particular should move the number materially — if it does not, the
splice bug was not being hit and that is worth knowing.

### Acceptance
Every row done or explicitly deferred with a reason. Report the numeric before/after
for the HRV and slow-wave fixes.
<!-- /TASK -->

<!-- TASK:09 slug=recall-gate deps=07 gate=yes -->
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
- **Andrea: the new cohort's ground, for both nerve and stomach, is in the
  abdominal wall.** So the common mode is most likely abdominal-wall muscle
  activity entering every channel through the shared ground. The old cohort did
  not have this problem on the nerves, because each cuff had its own reference on
  the cuff; the change of referencing between cohorts is what exposed it.
  Old-cohort results on this point do not transfer to the new cohort.
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

### Candidate width measured, 2026-09-29 — judge long candidates by their cores

Over all 405 merged marks from rounds 1–4 (cores module `detect/cores.py`, outside
the hash; generation hash unchanged):

- **Under the duration cap, candidates are ~1.9× the marks by time**, and 61% of
  candidate time is the hysteresis tail (`z_enter` → `z_exit`); merging adds 0.5%.
  **The core touching a mark is about the mark's size** (median 0.73× baseline,
  0.90× stim/recovery). Cores localise the evidence as intended.
- **Over the cap: 24 candidates (1.9%) carry 44% of candidate time and contain 49%
  of covered marks.** Baseline over-cap candidates are 88% core — sustained real
  elevation, not hysteresis; the largest is 164 s, driven by L_V3 at 2–50 Hz above
  `z_enter` for 127 s, in a span where Andrea marked 90 artifacts.
- **The duration cap had no source.** The spec's p99 was to come from old
  `*_segment_indices.mat`; none exist. **Provisional cap: 15.3 s**, the p99 of merged
  audit marks — resting on ~4 marks and bounded by the 2-min span. Recompute after
  each round and record its source.

**Rulings:**

1. **A long candidate is judged by its cores.** Downstream of generation —
   classification, extent, Andrea's candidate adjudication — an over-cap candidate
   is split into its cores, and only a core that itself exceeds the cap goes to the
   review queue. Sending half of all artifacts to manual review is not a queue, it
   is the whole job. Generation and the gate are untouched.
2. **Attribute the 24 over-cap candidates** before designing more: for each, the
   signal and band of its core, whether that contact is distrusted by the contact
   screen, and whether any consumer reads that (signal, band) pair. L_V3 at 2–50 Hz
   is a left-cuff contact (the side that fails the screen) in a band no nerve
   consumer reads; if that pattern dominates, it says what the long candidates are.
3. The ~10% of marks wider than any candidate are rough edges on Andrea's marks,
   not misses; no action.

### ROUND 5 AND THE POST-ROUND-5 PASS, 2026-09-30 — score against the pass condition as written; correct, then mask the residual; veto raw contacts for HR

**Measured (Claude Code, 2026-09-30):**

- **Round 5:** 51/53. Pooled rounds 3–5: **205/208, lower bound 0.959** (Clopper-Pearson 0.963, bootstrap 0.959). Not cleared.
  - Eligible spans: 8 baseline, 7 stim/recovery.
  - Chance margin: round 5 −0.001 (candidates covered 65% of labelled time); pooled +0.143.
  - Duration cap: 14.9 s (p99 of 458 marks; rests on 5).
- **All three gate-evidence misses are baseline, and all are below `z_enter` on every signal:**
  - s4#m0: common mode, closed.
  - s3#m7: max z 2.12, on stomach_ref 2–50 Hz and ANT1 300–3000 Hz.
  - s3#m13: max z 2.52, on ANT1 10–150 Hz.
  - Both round-5 misses are in B t03 es2, and both peak on ANT1.
- **Subtraction (ruling 4 of 2026-09-29):**
  - It removes the linear leak everywhere (R² of T on the reference goes to 0.000), and injected differential spikes survive (1.000).
  - But **3 of 20 cuff-recordings pass** the event-firing test. It helps only where the leak was a large gain error.
  - In A t05 L it **raised** firing on events far from beats (0.019 → 0.081): one scalar fits a blend of two sources.
- **Part of the common-mode class is the heartbeat:** events fall within 5 ms of a beat 1.3–5× more often than chance.
- **HR:** 200 injected transients at the recording's own event amplitude cost 4–135 beats on a raw contact, and 0–2 on T.
- **Event rate:** it co-varies with breathing (4/9 recordings) and heart rate (6/9) at r ≈ 0.1–0.35, near zero lag, even after cardiac events are excluded. The modulation (20–40/min) is not the breathing rhythm (76–98/min).
- **Contact screen:**
  - It does **not** flag H t01 es2 L. L1 is low-gain (R² 0.67 against 0.98 for its peers) but correlated (r 0.70), so the left T is 75% common mode.
  - It flags B t02 1_3 L on L1, yet L1 carries the ground signal (R² 0.94) while L2 and L3 do not (0.04 and 0.01).
  - ANT1 is mains-dominated in B t01 3_2, B t02 3_3 and B t03 2_2.

**What the gate arithmetic says.**
- A one-sided Clopper-Pearson bound ≥ 0.98 needs 386 artifacts at 3 misses, 456 at 4, and 523 at 5: about 67 more per miss.
- At the pool's observed miss rate (3/208, 1.4%), a miss arrives about every 70 artifacts. So the bound creeps toward about 0.986 and clears only near ~1,200 artifacts, some 20 more rounds. Baseline alone is 104/107.

That is **not** a reason to relax the gate (see "Do not"). It is a reason to check that the gate is scored against what task 09 says it is. The revised pass condition reads "≥98% recall for artifacts **above each consumer's tolerance**", and the audit scores every mark. Every gate-evidence miss is below `z_enter` on every signal. If those misses are also below every consumer's tolerance, the audit is partly measuring recall of harmless events.

**Rulings:**

1. **The two round-5 misses.**
   - **Andrea classifies both** from the misses PDF: would she blank it?
     - Not-target: close it that way.
     - Target: measure consumer harm exactly as for s4#m0, using each consumer's own input and threshold over the mark: the spike detector's firing, and the HR beat train. If harmless, close it as `accepted_limitation`: "threshold class, recurring, measured harmless". If a consumer is harmed, **stop and report**. That is branch 1 of "If it fails", a diagnosable class.
   - **Diagnostic, not a fix:**
     - Is ANT1 mains-dominated in B t03 es2, as it is in B t03 2_2? A mains line sits inside 10–150 Hz and inflates that signal's baseline σ, which depresses z. That would be one cause behind both misses.
     - Report the misses' z on a notched ANT1.
     - A notch is a generator change: it resets eligible rounds through `fixed_at`, so it is not adopted without a ruling.
   - **Round 6 is drawn only after both misses are closed.**

2. **Measure the audit against the pass condition as written. Do not adopt it yet.**
   - Claude Code writes a per-consumer damage rule **before computing it on any mark**, blind to hit or miss. It must come from each consumer's own input and threshold:
     - spike consumer: the T it reads and the 4.5σ detector;
     - HR: the operational beat-train test;
     - stomach consumers: their own bands.
   - I ratify the rule and Andrea approves it. Only then is it applied to **all** pooled marks, hits and misses alike. The tolerance-filtered recall is reported beside the raw recall, and raw recall stays in every round report whatever is decided.
   - **Whether the gate moves to tolerance-filtered recall is Andrea's decision**, recorded before the round it first applies to is scored.

3. **Artifact or physiology? The lag histogram decides.**
   - For each cuff, histogram her detected spikes against lag from the nearest event: 0.1 ms bins, ±50 ms. Do this separately for near-beat and far-from-beat events, with the random-time control.
   - A peak confined to the event's own width (about ±1 ms) is electrical. No conduction or reflex coupling to a heartbeat or a muscle twitch is sub-millisecond.
   - A broad hump of tens of ms is physiology and is **never** masked.
   - Report per cuff. Rulings 4b and 4c apply only where the peak is narrow.

4. **Correct, then mask the residual.** Everything here lives in `derive/`, outside the hash.
   - **a. Keep the single-scalar subtraction only where it helps.** Keep it on a cuff only if it lowers event-locked firing in **both** the near-beat and the far-from-beat subsets. Otherwise the consumer reads uncorrected T. A correction that adds artifacts is worse than none (A t05 L).
   - **b. One extension, in the same pass: multi-regressor subtraction.**
     - Each outside channel is its own regressor, plus a beat-locked cardiac template.
     - Fit on one half of the recording, verify on the other.
     - Adopt it per cuff only if it passes the pre-declared verification: no significant excess firing on events, and injected spikes surviving within 5%.
   - **c. Cuffs that still fail get the pre-declared fallback, for the spike consumer only:** mask T for ±1.5 ms wherever T reaches 4σ within an event, **with exposure accounting**. Masked time is removed from the denominator of every rate, and from every phase bin of any rhythm-locked analysis.
     - This answers the objection in ruling 2 of 2026-09-29. Real spikes inside the mask are lost only in proportion to masked time, so rates stay unbiased. And a peak confined to ±1 ms is not physiology, so nothing rhythm-locked is removed.
     - Events co-vary with breathing and heart rate, i.e. with state. That is why exposure accounting is mandatory, not optional.
   - **Verify the mask:**
     - in unmasked time, firing on events is not above control;
     - injected differential spikes outside the mask survive within 5%;
     - injected spikes inside the mask are lost at the masked-time fraction.
   - **Report per cuff:** masked time %, spikes removed %, and the excess event-locked spikes as a fraction of the cuff's spikes, before and after.

5. **Trust in T for the spike consumer is decided by ruling 4's verification, not by contact heuristics.**
   - H t01 es2 L shows why: a low-gain contact passes the correlation rule while T is 75% common mode.
   - The contact screen remains for consumers that read raw contacts (HR, velocity), with two fixes for the new cohort:
     - **Missing ground signal:** with a shared ground and no local reference, every healthy single-ended contact must carry the ground-site signal. A contact with R² near zero against the outside reference is the suspect one, whatever the peer correlation says. In B t02 1_3 L that is **L2 and L3**, not L1. Report the L2–L3 correlation: a short between them would make them agree with each other and not with the ground. The whole left cuff of B t02 1_3 is distrusted for every consumer until this is explained.
     - **Low gain:** flag a contact whose R² sits well below its peers' (0.67 against 0.98), with a `PROVISIONAL_` threshold.
   - **ANT1** in B t01 3_2, B t02 3_3 and B t03 2_2: no consumer reads it. Detection keeps reading it. Invariant 43 forbids hiding a signal from detection alone; it does not forbid the reverse.

6. **HR: the operational injection test becomes task 05's transient veto.**
   - A channel is vetoed if 200 transients at the recording's own non-cardiac event amplitude change the beat train (lost beats, or a fiducial shifted by more than 2 ms) in more than 1% of injections: `PROVISIONAL_MAX_TRANSIENT_HARM = 0.01`.
   - Template SNR ranks only the survivors. SNR cannot see this harm, because the far-field ECG is common mode too.
   - **Measure, don't adopt:** does a minimum-QRS-width check in the beat detector remove the harm on raw contacts? A 1.2 ms transient cannot be a rat QRS. Adopting it is a separate ruling.
   - **H t01 3_3, every candidate vetoed:** raising is correct (task 05). Report whether the implausible intervals are short (extra beats) or long (missed beats), per channel. Do not move the threshold on one recording.

7. **The old-cohort QRS premise is withdrawn for the new cohort.**
   - "The QRS carries 0.00–0.54% of its energy above 300 Hz" was measured on an old-cohort tripole.
   - Re-measure it per new-cohort cuff, on T (corrected as in ruling 4) and on raw contacts.
   - Anything in tasks 02 and 07 that relied on the old figure uses the new-cohort number for the new cohort.

8. **Ratified:**
   - The width attribution. The real review load is the 12 over-cap candidates read by a consumer on a trusted contact (42 marks, 409 s).
   - The duration cap of 14.9 s, provisional.
   - The breathing and HR cross-correlation. The events co-vary with state and do not form a rhythm of analytic interest. No further analysis, beyond the exposure accounting in 4c.

### RULING 2026-09-30 (b) — the time mask is withdrawn; judge each spike by what the other channels saw

**Measured (Claude Code, gems `ce3bafa`, `fa5955c`; hash unchanged):**
- **Round-5 misses:** both harmless to the spike and HR consumers.
  - ANT1 in B t03 es2 is mains-dominated (91% of its power). Notching *lowers* the misses' z (2.52 → 0.58), so part of the in-mark elevation is the hum itself.
  - Both still await Andrea's classification.
- **Lag histograms:** 14 of 20 cuffs have a narrow (±1 ms) electrical core, 4–6× their flank on the worst. Humps appear only around near-beat events. Every cuff has a flat floor near events at 1–15× the global rate.
- **Excess event-locked spikes as a fraction of each cuff's spikes:** up to **0.62** (B t03 R), and 0.44 each on H t05 L and H t01 3_3 L. **On the worst new-cohort cuffs, a quarter to more than half of the detected spikes are electrical.**
- **The gated 4c mask fails its own premise:**
  - H t05 L loses 88% of its spikes against 2.1% effective exposure.
  - The gate (T ≥ 4σ at an event) selects exactly the events where T carries a spike-sized deflection, real or not. Her detector then pads every NaN run by 10 ms.
  - So the mask is placed *by the spikes* and removes them wholesale. Its "no firing on events in unmasked time" check cannot fail.
- **4b's check passed a correction that tripled spikes everywhere** (B t02 1_3 L, 3,684 → 9,413), because the control rate rose with them.
- **HR:**
  - The veto leaves A t02, H t01 es2 and H t05 with **no HR channel**.
  - The veto ran on task 05's detector, which is more robust than the consumer's own `findpeaks`. On B t02 3_3 RVN3 the consumer's own injection test lost 4 beats per 200.
  - A QRS-width floor does not help: a transient moves the fiducial within the beat wave.
  - H t01 3_3's implausible intervals are extra beats at about half the RR interval.
- **QRS energy above 300 Hz, new cohort:** median 0.24% on T. Tail up to 4.4% (A t05 R_T), and up to 15% on raw contacts.

**Rulings:**

1. **Withdraw ruling 4c (the gated time mask) and my claim that masked-time loss is proportional.** Both fail for the reason above. Any gate must be independent of the cuff under test, and a NaN mask must never reach her detector (its 10 ms pad multiplies it).

2. **Replace it with a per-spike outside-coincidence veto.** This is Andrea's proposal from 2026-09-29: tell leak from nerve spikes by their characteristics. Physiology cannot put one vagus's spike on the other cuff and the stomach at the same instant. **Decided per spike, after detection, in `derive/`:**
   - Run her detector on the T the consumer reads (4a/4b as ruled). For each detected spike, take the peak |outside reference| (300–3000 Hz, channels outside this cuff) within ±w of the spike time, in units of that reference's robust σ.
   - **w** is the cuff's measured core half-width from the lag histogram, rounded up to 0.1 ms. Veto the spike if the peak exceeds **θ**.
   - **θ is set on the first half** so that at random times the veto fires at most 2% (the chance loss). It is applied to the whole recording, and the chance loss is reported on the second half.
   - Nothing is NaN'd. Exposure is unchanged. Rates are divided by (1 − chance loss), with the chance loss reported per event-rate tertile, because state moves the outside noise.
   - **Verification, pre-declared, and independent of the veto:**
     - (i) **Left–right spike cross-correlogram** per recording, 0.1 ms bins over ±50 ms. Two vagi cannot fire together to sub-millisecond precision, so a narrow zero-lag peak is leak. Pass: no zero-lag core above flank after the veto, where there was one before. This uses neither the events nor the reference.
     - (ii) Injected differential spikes at random times are lost at the chance-loss rate (±1%) on the second half.
     - (iii) The surviving spikes' lag histogram around events has no narrow core.
   - **Report per cuff:** spikes before and after, fraction vetoed, chance loss, and (i)–(iii).
   - **Also report the floor.** After the veto, does the 1–15× floor near events remain?
     - If it drops, it was sub-event leak (muscle activity below the event finder's size rule), not state.
     - If it remains, it is state co-variation and stays.

3. **4b's check is amended:** a correction is adopted only if the spike rate more than 50 ms from any event does not rise by more than 5%. A correction that adds spikes in clean time is adding noise. B t02 1_3 L is not adopted (it is distrusted anyway). A t05 fails survival and is not adopted.

4. **Cardiac humps are not physiology until shown to be.** My ruling 3 ("broad hump = physiology, never mask") was too quick. A rat QRS lasts ~10–20 ms, so QRS leak on T is itself a broad hump. Cardiac-locked vagal afferent firing follows the pressure pulse, tens of ms *after* the R-wave, and is not centred on the QRS.
   - Histogram spikes against lag from the **R-peak** (not the event) on the four hump cuffs, before and after ruling 2.
   - A hump centred within the QRS that ruling 2 removes is leak.
   - A hump within the QRS that survives goes to the task 02/13 peri-R route, measured per new-cohort cuff by crossing rate per lag bin (task 02's method, now needed because of ruling 7).
   - A hump after the QRS is physiology and is left alone.

5. **HR:**
   - **The veto runs on the detector that produces the beats the analysis uses.** If Andrea's HRV analysis uses its own `findpeaks` on the chosen channel, the binding veto uses the validated replica (99.7–100% match). Report task 05's detector beside it.
   - **The three recordings with no channel.** Measure one route: two-pass bridging.
     - Pass 1 detects beats.
     - Events within 5 ms of a pass-1 beat are cardiac and are *not* bridged.
     - The remaining events are bridged ±1.5 ms by linear interpolation on the raw contact.
     - Pass 2 detects the beats.
   - Judge the route by the same injection veto, run through both passes. Adopt it per recording only if it passes. Until then these recordings have no HR, and raising is correct.
   - **The QRS-width floor is rejected.**
   - **H t01 3_3:** are the extra mid-cycle peaks locked to stimulation pulses (present only in stim periods) or to the cardiac cycle (a T-wave, present throughout)? Report only; no threshold moves.

6. **`stomach_ref` in the three ANT1-hum recordings:** measure the harm first, and don't touch `derivations.py`. Compare each stomach consumer's output on `stomach_ref` as-is against an alternative built outside the hash that excludes ANT1. If the outputs agree within the consumer's tolerance, no action. If not, the stomach consumers read the alternative in those recordings (invariant 43 permits distrusting a derived quantity).

7. **Ratified:**
   - The contact-screen fixes (B t02 1_3 L: L2 and L3 shorted or locally driven, whole cuff distrusted; H t01 es2 L1 low-gain).
   - The revised lag-histogram rule, with its clean controls.
   - The sub-sample cardiac template inside 4b: CLAUDE.md's "do not re-implement" line is narrowed to allow it under the three stated conditions.
   - The finder and veto modules outside the hash.
   - The non-cardiac-only gains for injection.

8. **Damage rule (ruling 2 of 2026-09-30): ratified with amendments.** Andrea approves before it is applied.
   - **Spike consumer:** the test is **two-sided**. An artifact that suppresses the detector (σ inflation, clipping) damages as much as one that adds spikes; s3#m7 had 0 spikes where 2.5 were expected. A mark is damaging on a cuff at p < **0.05** (not 0.005), on the consumer's input as routed at ratification. Counting a mark as damaging is the conservative direction: it keeps the mark in the denominator. A strict threshold on a sub-second Poisson count has almost no power and would label nearly everything harmless, which inflates the filtered recall.
   - **HR:** damaging if any RR interval touching the mark deviates more than 20% from the local median. The ±1-beat count rule may add damage but never clear it (one missed beat is a doubled RR, which HRV cannot absorb).
   - **Slow wave:** a 1 s peak shift, *or* a cycle added or lost, is damage.
   - **Unassessable → target:** ratified. A recording with no HR channel is unassessable for HR.
   - **Velocity left out:** ratified. Consequence, written down: the filtered recall says nothing about velocity, so task 18 must handle its own artifacts (zero-lag rejection) or be re-scored when it is built.
   - **The routing table the rule reads is frozen at ratification.** A later routing change means recomputing it for every mark.

**Order.** Andrea classifies the two misses. If she draws round 6, Claude Code stops Drive and MATLAB work and builds rulings 2–5 against synthetic data only. The real-data runs follow her "done", in this order: 2, 3, 4, 5, 6, then the damage rule on all pooled marks once Andrea approves it.

**Andrea, 2026-09-30:**
- **s3#m7 and s3#m13: "can be ignored"** — close both as `not_target` (her classification; committed marks and the round-5 score stand). Both also measured harmless. Round 6 may be drawn once `--check-next` returns ALLOWED.
- **The damage rule is approved as amended.** Apply it after the real-data runs, in the order given.
- **Beats are computed once.** Her HRV analysis has its own `findpeaks`. She does not want double computation if the pipeline's beats are reliable. So:
  - **One detector, and the veto tests it.** The pipeline's beat detector (task 05's, the more robust of the two) is the only beat computation, and ruling 5's transient veto runs on it. The replica is kept for the comparison below, then retired.
  - **Beats are stored in her analysis's own format**: the same variable names, units, and sample or time base her function builds today, per recording, beside her existing outputs. Claude Code reads the format from her code and does not invent one.
  - **A new post-processing-only version of her function** reads the stored beats and skips peak finding. Her original function stays untouched, beside it.
  - **Reliability is shown before the switch.** Feed the new function her own `findpeaks` beats converted to the stored format: its HRV outputs must equal the original's exactly. Then compare pipeline beats with her `findpeaks` beats per recording (match %, and where they disagree, which one the injection test and the RR plausibility favour).
  - **She switches when she has seen that comparison.** For recordings with no vetted channel, nothing is stored and the function says so. It does not fall back to its own peak finding.

### RULING 2026-09-30 (c) — veto ratified; distrust where it fails; HR needs an independent count check; notch mmc's input

**Measured (gems `d346584`, `dd5b062`; hash unchanged):**
- **The veto on 11 cuffs.** It rejects **22–74%** of each cuff's detected spikes. Its chance loss is 0.9–7.2% on the second half, and up to 10.7% in the busiest tertile.
  - (ii) passes on all 11 at 3,000 injections.
  - (iii) fails on H t01 es2 L (event core ×3.0 remains, w = 0.2 ms).
  - (i) The left–right zero-lag core clears in A t01, A t02 and H t01 3_3, but **persists in A t05** (×4.0 → ×4.3).
  - The floor near events mostly drops, so it was sub-event leak. Part of it remains on four cuffs: that is state.
- **Humps.** All four QRS-centred humps survive the veto. H t05 L has a further component peaking at +14.5 ms after the R.
- **HR detectors.** Task 05's detector and her `findpeaks` each win on some recordings.
  - Task 05 misses 12–43% of beats on A t05 and B t02 1_3, locks onto every other beat on A t02, and picks a **one-beat** channel on A t01 that passes every gate.
  - `findpeaks` adds false beats on B t03 and B t02 3_3.
  - Two-pass bridging fails everywhere.
  - H t01 3_3's extra peaks sit at 0.50 RR and are cardiac-locked, not stim-locked.
- **The post-processing-only HRV function** (`HR_BR_HRVAnalysis_beats.m`) is identical to the original on 9/9 recordings when fed her own beats.
- **stomach_ref in the hum recordings.** slow_wave is untouched by the hum (100% of peaks match with ANT1 notched). mmc is changed by it (Jaccard 0.18 and 0.25 in B t01 3_2 and B t02 3_3; 0.82 in B t03).
  - The no-ANT1 alternative changes both consumers far more (Jaccard 0.16–0.20), because it is a different derivation.
- **Damage rule on all 208 marks.** 180 are target.
  - Filtered recall is **179/180, lower bound 0.974**, against 205/208 (0.959 raw: the span bootstrap is the more conservative; Clopper-Pearson alone is 0.963).
  - The only target miss is s4#m0, flagged by a spike deficit (p = 0.033).
  - The round-5 recordings had no routing entry and were judged on uncorrected T without the veto.

**Rulings:**

1. **The veto is ratified as the spike consumer's leak rejection**, with these amendments:
   - **θ stays fixed per cuff**, as calibrated. It is a physical criterion and must not be loosened in busy periods.
   - **Rates are corrected by the per-tertile chance loss.** Measure it at ≥ 10,000 random times over the whole recording. (ii) showed that injected loss tracks it, so the correction is valid.
   - **Report the corrected rate beside the raw count everywhere.** Any analysis resolved by state must use the loss of the tertile it sits in.

2. **Where verification fails, the cuff is distrusted for the spike consumer in that recording** (ruling (b) 5: trust follows verification).
   - **H t01 es2 L** fails (iii). It is also the low-gain cuff. Distrusted.
   - **A t05:** first run (i) again, restricted to spikes outside the peri-R window measured under ruling 3.
     - If the zero-lag core vanishes, the residual synchrony was QRS leak on both cuffs, and the peri-R route handles it.
     - If it persists, distrust both A t05 cuffs for the spike consumer.

3. **The QRS humps go to the peri-R route (tasks 02 and 13), measured per new-cohort cuff.**
   - **Extent:** the crossing rate per lag bin around the R-peak, after the veto, on the routed T. The extent is where it exceeds the flank.
   - **It is a time mask on R-peak lags, so it is independent of this cuff's spikes.** Exposure is removed and rates are corrected. Analyses locked to the cardiac phase report the blanked phase as not measured; they do not interpolate it.
   - **R-peaks come from the recording's vetted beat source (ruling 4)**, never from a detector that halves or misses beats.
   - **H t05 L's +14.5 ms component is unresolved. It is not physiology by default.**
     - +14.5 ms is too early for pressure-driven afferent firing (tens of ms after the R, following the pulse) and is within reach of the QRS tail.
     - Report its lag and width against the QRS template's > 300 Hz envelope on that cuff.
     - If it sits under the envelope, it is leak and the extent covers it. If it is clear of the envelope, it is left alone.

4. **HR: a beat train is stored only if it passes an independent count check.**
   - Task 05 lists beat count as "a sanity print only". That is why a one-beat channel passed. The 2026-09-23 header already asked for beat-count consensus.
   - **The independent reference: a per-minute heart rate from the autocorrelation** of the rectified 10–150 Hz signal, searched only over rat heart rates (250–550 bpm). It uses no peak picking, so it cannot share a detector's errors.
   - **Count gate** (`PROVISIONAL_MAX_COUNT_DEV = 0.05`): a beat train passes if its beats per minute are within 5% of the autocorrelation rate in ≥ 95% of minutes where the autocorrelation peak is clear. It must also pass the transient veto and task 05's plausibility gates.
     - Every-other-beat (50%), 12–43% misses, +7% false beats and a single beat all fail it.
     - If the autocorrelation is not clear in most minutes, the recording is unassessable and nothing is stored.
   - **One computation, two candidates.** Per channel, the pipeline computes both candidate trains (task 05's detector and the `findpeaks` replica), gates both, and stores one:
     - the passing train with the best template SNR;
     - on a tie, the one closest to the autocorrelation count.
     
     Her function only post-processes what is stored. This satisfies Andrea's "compute once": nothing is recomputed downstream.
   - **A t01:** nothing is stored until a train passes.
   - **A t02, H t01 es2 and H t05:** re-run under this gate with both candidates. Bridging is dropped.
   - **H t01 3_3:** measure, without adopting, a refractory relative to the running median RR, rejecting any peak within 0.6 × median RR of the previous one and keeping the one that matches the template. Check it against the autocorrelation count. A heart does not halve its interval in one beat, but it can speed up quickly, so the rule must not remove real beats during rate rises.
   - Andrea switches to `HR_BR_HRVAnalysis_beats.m` only after seeing the gated comparison.

5. **stomach_ref in the three ANT1-hum recordings: stomach consumers read stomach_ref built from notched ANT1**, derived outside the hash. They do not read the no-ANT1 alternative.
   - The notch removes the hum and keeps the derivation, which is the only change warranted.
   - It applies to all three recordings, including B t03 (0.82), so that no tolerance judgement is needed.
   - slow_wave is unchanged by it (100% match), so the rule is uniform at no cost.
   - Report whether a 60 Hz-only notch (Andrea's practice) suffices, or whether harmonics are needed.
   - Detection keeps un-notched ANT1 (invariant 43).

6. **The damage rule's routing must be complete.**
   - Compute the round-5 recordings' routes by the same mechanical rules (4a, 4b, veto, distrust).
   - Recompute only those marks. The rule itself is unchanged.
   - Report the result beside the current one. Both are recorded.

7. **The gate question is put to Andrea.** Raw recall stays in every report whatever she decides.
   - **Filtered recall at 1 target miss needs 236 target marks** for a lower bound ≥ 0.98, about 56 more at zero further target misses: one or two rounds. With one more target miss, 313.
   - **Raw recall at 3 misses needs 386**, and at the observed miss rate about 1,200.

**Andrea, 2026-09-30: the gate is filtered recall.** From round 6 on, task 09's gate is a one-sided 95% lower bound ≥ 0.98 (the more conservative of Clopper-Pearson and span bootstrap) over marks the ratified damage rule calls **target**. This is task 09's written pass condition: "above each consumer's tolerance".
- The other conditions are unchanged: ≥ 3 eligible spans per condition, and a positive pooled chance margin, computed over target marks.
- Raw recall and its bound are reported beside it in every round.
- The rule and its routing are frozen as ratified. A change to either means recomputing every pooled mark, and it is reported as a change.

**Addendum to ruling (c), 2026-09-30 — answers to the build's questions:**

1. **The peri-R mask may use "mask-grade" beats.** A train that passes the count gate but fails the transient veto is allowed for the peri-R time mask only.
   - It is stored under a separate name, and `HR_BR_HRVAnalysis_beats.m` never reads it. HRV still needs a fully vetted train.
   - The veto protects fiducial precision, which HRV needs. The mask needs every beat to be present, which the count gate checks.
   - Fiducial jitter is self-correcting here: the extent is measured from the crossing-rate histogram against that same train, so jitter widens the measured hump and the extent with it.
   - Run ruling 3 on A t02 L, H t01 3_3 L and H t05 L with these trains, and place H t05 L's +14.5 ms component.
   - A recording with no train passing the count gate gets no peri-R route. Its spike-consumer cuffs with a QRS hump are distrusted.
2. **Recompute every pooled mark once routing settles.** Ruling (c) changed routing (A t05 and H t01 es2 L distrusted, new HR trains, notched stomach_ref), so Andrea's clause applies: ruling 6's "only those marks" is superseded.
   - When items 3–6 are complete, freeze the routing table and record its hash.
   - Recompute the damage rule on all pooled marks and report the old and new classes side by side, with the reasons for every mark that changes class.
   - Round 6 is scored on that frozen table.
3. **An input the routing excludes counts as unassessable, therefore target.** This covers a distrusted cuff, a recording with no HR train, and any consumer with no admissible input.
   - It is the convention already ratified for missing HR channels. The gate must not become easier to pass by excluding data.
4. **Test skips.** List the 23 extra skips in the clean worktree with their reasons. A skip other than "needs local real data that is not in the repo" is a defect: fix it.
5. **stomach_ref's 60 Hz residual** (about 1,500× the floor in B t01 and B t02 3_3, after the ANT1 notch): report which inputs carry it. Compare mmc with 60 Hz notched on every stomach_ref input against ANT1 alone. If mmc differs, the notch applies to all inputs in those recordings.

### RULING 2026-09-30 (d) — the frozen table and the recompute: addendum 3 corrected; round 6 appends

**Measured (gems `c53a2ba`; hash unchanged):**
- **Frozen routing table:** `46ce9b0b0081…`, stored at `labels/blind_audit_routing/routing_46ce9b0b00819823.json`.
- **Under addendum 3, all 208 pooled marks are target.** 28 changed class, and 29 are target *only* because an input is excluded. Filtered recall therefore equals raw: 205/208, lower bound 0.959.
- **s3#m7 and s3#m13** had been measured harmless on B t03 es2 R_T. They became target misses only because R was then distrusted by check (iii).
- **Round-5 routes:** no cuff adopts multi (A t01 1_2 L: 147 → 28,736 spikes, rejected by the amended 4b). B t03 es2 R is distrusted by check (iii). Four hump cuffs are distrusted for lack of a count-gated train.
- **Spike consumer, rounds 3–5:** about 11 of 30 cuff-recordings are now distrusted.
- **H t05 L:** against a count-correct train there is no QRS-centred hump. There is a component at +14.25 ms, 0.5 ms wide, outside the > 300 Hz QRS envelope (−5.0 to +2.3 ms).
- **B t01 3_2:** notching all stomach_ref inputs exposes real mmc damage on 3 marks (the hum had inflated the MAD).
- **Skips:** the 23 extra clean-worktree skips were a test predicate defect, now fixed (clean worktree 1137 passed / 7 skipped, the same as local).

**Rulings:**

1. **Addendum 3 was wrong. It is replaced.** It contradicted task 09's written pass condition, "above each consumer's tolerance". An input no consumer reads cannot damage a consumer. Its visible failure: two marks measured harmless became "damaging" because we stopped reading the cuff they were measured on.

   **Andrea decided (2026-09-30): only consumers that run count.**
   - **A consumer's input that the routing excludes contributes nothing** to a mark's class. This covers a distrusted cuff for the spike consumer, and HR in a recording with no HRV-grade train.
   - **Unassessable still means target.** If a consumer does read the input but cannot judge the mark (for example, fewer than 10 local beats), the mark is target. The same holds if no running consumer can judge the mark at all.
   - This also supersedes the clause in ruling (b) 8 that made a recording with no HR channel unassessable for HR.
   - **This correction was made knowing its effect** (about 178/179 instead of 205/208). So both classifications are computed and reported every round: "consumers that run" is the gate, and "excluded = target" is reported beside it with raw recall. Neither report is ever dropped.
   - A later change to either convention needs Andrea's decision, recorded before the next score.

2. **Round 6 and later rounds append to the frozen table.**
   - A new recording's routes are computed by the same mechanical rules and **appended**. Existing entries must stay byte-identical.
   - Store the table as per-recording entries, each with its own hash. The score records the old table hash, the new one, and a check that every old entry hash is unchanged.
   - That counts as "scored on the frozen table". Any change to an existing entry is a routing change: recompute every pooled mark and report it as a change.

3. **Sub-millisecond R-locked components are electrical, whatever their lag.** This is the same reasoning as ruling (a) 3: no neural or reflex chain holds 0.5 ms precision.
   - H t05 L's +14.25 ms component goes into its peri-R extent.
   - The QRS-envelope test missed it because the envelope comes from the averaged template's energy, where a small late high-frequency feature falls below threshold.
   - Rule: an R-locked excess narrower than 1 ms at half height is included in the peri-R extent. Broader components follow ruling (c) 3.

4. **Ratified:**
   - mask-grade beats (`maskBeatlocs`), with their guards;
   - the round-5 routes;
   - the all-inputs notch in B t01 3_2 and B t02 3_3, and ANT1 only in B t03 2_2;
   - `frozen_filtered_gate`;
   - the skip fix;
   - the peri-R extents for A t01 L (−9.5 to −4.5 ms) and B t03 es2 L (−2.0 to 0 ms).

5. **Recompute now under ruling 1 on the same routing.** The routing is unchanged, so the table hash stays `46ce9b0b…`.
   - Report both classifications per round and pooled, the gate numbers under each, and the marks whose class differs between them, with reasons.
   - Store the "consumers that run" classes as the gate classes under that hash.

6. **Report the spike-consumer coverage.** Give a table of every cuff-recording in rounds 3–5 marked trusted or distrusted, with the reason, and the share of spike-consumer time lost to distrust.
   - Most distrust now comes from missing count-gated HR trains (hump cuffs) and from failed check (iii).
   - Better HR trains would recover data. That is the next lever, and it is Andrea's priority call.

### RULING 2026-09-30 (e) — (d) 3 corrected; HR trains are the next lever, diagnosed before designed

**Measured (gems `835cc84`; hash unchanged):**
- **Gate ("consumers that run"), pooled:** 178/179, lower bound **0.974** (Clopper-Pearson; bootstrap 0.980). Chance margin +0.142. 8 baseline and 7 stim/recovery spans. Not cleared; the only target miss is s4#m0.
- **The other two numbers:** "excluded = target" and raw are both 205/208 (0.959). The 29 marks that differ between the classifications are all target only because an input is excluded.
- **Routing:** per-entry hashes, append-only, and three-number reports are in place.
- **H t05 L's "+14.25 ms component" is 4–5 spikes of 281** (p = 0.017 before correction for the search across lags). It fails the p < 1e-3 bar used for every other R-locked test. The "0.5 ms width" was the bin size. Ruling (d) 3 rested on my misreading of the earlier report.
- **Spike-consumer coverage, rounds 3–5:** 11 of 30 cuff-recordings are distrusted, **36.4% of spike-consumer time**. 6 of them (24.9% of time) are hump cuffs with no count-gated HR train. A t01 1_2 and H t03 each miss the count gate by one minute (94.7%).

**Rulings:**

1. **(d) 3 is corrected.** A narrow R-locked excess is electrical only if it is also significant at the bar every R-locked test uses: p < 1e-3 after correction for the search across lags.
   - Width is measured at half height at 0.1 ms bins, never read off the bin size.
   - H t05 L's component does not qualify, so its entry reverts to the `46ce9b0b` content.
   - That is a routing change: append the corrected entry and recompute every pooled mark. No class change is expected, since `ca04a0f9` changed none.
   - The general rule stands for any component that does qualify.

2. **The count-gate thresholds do not move to rescue A t01 1_2 or H t03.** Loosening a threshold after seeing which recordings miss it by one minute is fitting.
   - `PROVISIONAL_MAX_COUNT_DEV` and the 95% bar get their permanent values in build-order step 9, from the cross-animal set, like task 05's other provisional thresholds.

3. **HR trains are the next lever. Diagnose them before designing anything.** For every recording in rounds 3–5 without an HRV-grade train, report per channel, per detector:
   - the minutes that fail the count gate;
   - for each failing minute, why: missed beats, extra beats (and whether they sit at about 0.5 RR), transient hijacks, or an autocorrelation reference that is not clear. A minute whose reference is not clear should have been excluded as unassessable; check that it was;
   - the veto harm, and its mechanism.

4. **Constraints for any detector change that follows. It is proposed to me first, not adopted.**
   - **Andrea's rule for missed beats:** re-search near the expected time with relaxed width and height tolerances, so the fiducial is measured. Never insert a beat at the expected time.
   - **Extra beats from a second mid-cycle peak** (H t01 3_3): resolve within a refractory relative to the running median RR by template match, not by amplitude. The refractory must not remove real beats during rate rises. Check it with the count gate, which uses the independent autocorrelation.
   - **Design out-of-sample:** develop on some animals and judge on held-out animals. Report the held-out result as the evidence.
   - **The same gates apply unchanged:** count gate, transient veto and task 05 plausibility.

5. **Priority is Andrea's call. The default is to run in parallel.** Andrea draws audit rounds. Claude Code does ruling 3's diagnosis on locally cached data and synthetic data while she labels, and real-data runs follow her "done".

### RULING 2026-09-30 (f) — the HR candidate detector: ratified with amendments

**Measured (diagnosis on 9 recordings, 216 trains):**
- The count gate reproduces exactly. 532 unclear minutes were excluded as ruled.
- **Capture, not loss, is the veto harm on raw contacts.** The injected transient itself becomes the beat in 50–94% of harmed injections. This alone blocks A t01, H t05 and H t01 es2.
- After 10–150 Hz filtering, a captured transient's width falls inside that channel's beat-width range on all nine channels (10.6–21.6 ms on four; about 150–190 ms and about 7 ms on the others, matching their beats). So **width cannot separate it** from a beat. A width floor changed veto harm by no more than ±0.005.
- **Most findpeaks extras sit at its 100 ms spacing floor** (RR median 101–116 ms, phase 0.60–0.70 RR). A true 0.4–0.6 RR mid-cycle peak is a minority, except on H t01 3_3, A t02 and H t09.
- Task 05 misses beats on A t02 and H t01 3_3.
- Tripoles have almost no harm but a weak ECG, so most of their minutes are unclear.
- The autocorrelation reference is wrong in 2 minutes, both in H t03.
- One read-only Drive pass was made with no round open. That is within the rule: the constraint is no Drive or MATLAB while a round is open.

**The design in the build's report is ratified: a new module outside the hash, one extra candidate train per channel, the same unchanged gates. These amendments apply:**

1. **The template is the median across clean beats**, where clean means RR within 10% of the running median. Do **not** exclude beats that have an event within 20 ms: on channels where the QRS edge itself triggers the event finder, that would exclude most beats. The median is robust to the occasional transient.

2. **Judge suspect candidates by masked correlation.**
   - A candidate within 2 ms of a common-mode event is scored by template correlation with the event's ±2 ms **excised from both** candidate and template.
   - A real beat hit by a transient keeps most of its QRS and scores well. A captured transient with no QRS under it has little left and scores poorly.
   - The excision is for scoring only. The output signal is untouched (invariant 8).
   - **Fiducial of a kept suspect beat:** taken from masked template alignment, expressed as the template's peak offset, so it means the same thing as a clean beat's peak.
   - Report, on clean beats, how far alignment-based fiducials sit from peak-based ones. Any systematic offset must be under 0.5 ms, or mixed fiducials will add HRV jitter.

3. **The correlation floor is a rule fixed now, not a number tuned on A/B.** Answer to (b): per channel, the floor is the **1st percentile of that channel's own clean-beat template correlations**. H applies the same rule to its own clean beats. The quantity that protects against capture is never tuned across animals.

4. **The refractory fraction is a development parameter, capped at 0.8.**
   - The 0.6 used in the earlier measurement cannot resolve the dominant extras. At RR ≈ 150–165 ms, 0.6 × RR = 90–99 ms, below the 101–116 ms extras.
   - Choose k on A/B from {0.6, 0.65, 0.7, 0.75, 0.8} and fix it before H runs. Real beats during rate rises stay above ~0.8 × median.
   - The rate-rise check is the pass/fail test: the count gate in every minute where the autocorrelation rate rises > 10%.

5. **Re-searched beats must pass the same correlation floor** (rule 3). Only height and width are relaxed.
   - The height and width multipliers are development parameters, fixed on A/B before H.
   - Nothing is inserted and gaps stay tagged (Andrea's rule, invariant 8).

6. **Holding out H whole is accepted.** Answer to (a).
   - Capture is present in the development set (A t01), so the capture mechanism is not first met in the held-out set.
   - Holding out one animal is thin evidence. So every recording from round 6 on, gated by the frozen detector, is reported as further held-out evidence, per animal.

7. **Written down before H runs:** k, the height and width multipliers, the template and floor rules, and the code commit.
   - The H run happens once. If it fails, report it. Do not tune and re-run on H.

8. **Report the regression set (B's five recordings):** stored trains reproduced (≥ 99% of beats within 2 ms) or still passing every gate. Also report the beats changed and why.

### RULING 2026-09-30 (g) — HR candidate detector: synthetic development amendments

**Measured, on synthetic fixtures only (`make_hr_trouble`, matched to the diagnosis; no real data):**
- The fixture reproduces every measured failure of the existing detectors: findpeaks recall 0.77 with 201 extras under capture, extras at the spacing floor, task 05 misses.
- Ruling (f) as written fails in four ways:
  - **10–150 Hz masked scoring:** 0/71 beats hit by a transient were kept.
  - **Excising only the suspect's own event:** real beats 25–80 ms from a transient were lost, and their gaps were searched but not filled.
  - **The 1st-percentile floor on the (f) 1 clean set:** captures make up 1.9% of it, the floor collapses, and 40–70% of captures are kept.
  - **Periodic secondary peaks halve the train's running median**, so the refractory removes nothing at any k.
- Fiducial alignment offset: 0.002–0.035 ms.

**Rulings.** These are design corrections made on synthetic data before any real-data development. The A/B → single-H protocol of ruling (f) 6–7 is unchanged.

1. **Score in 10–900 Hz. Fiducials stay 10–150 Hz peaks.** The template and every correlation are computed in 10–900 Hz, where a 1.2 ms transient stays compact enough to excise (64/71 hit beats kept).

2. **Excise every event in the ±40 ms window, for candidates and clean beats alike**, so the floor is made exactly as the score is.
   - If excision leaves under 50% of the window, the score is undefined. The candidate is not kept on score: it is dropped, its gap is re-searched, and the number of such candidates is reported.

3. **The clean set and floor come from an independent reference.** This replaces (f) 1's running-median clean definition and refines (f) 3.
   - **Clean** means both adjacent RRs are within 10% of the per-minute **autocorrelation RR**, in clear minutes only.
   - **The floor** is the 1st percentile of masked correlations over clean beats that are **not suspect**. A capture is suspect by construction, so impostors are excluded rather than diluted.
   - **With fewer than 50 non-suspect clean beats, this detector yields no train on that channel.** There is no fallback to the suspect-contaminated set.

4. **Periodic secondary peaks: option (b), with (c) as the fallback.**
   - **(b) Seed.** The template is seeded from task 05's beats, provided task 05's train has ≤ 25% short intervals (< 0.75 × the autocorrelation RR). The seed only needs to be pure, not complete, so task 05's missed beats do not matter here. The clean and floor rules of ruling 3 then apply.
   - **(c) Split.** Otherwise, split the candidates into the two interleaved subtrains and keep the one with the higher template SNR (task 05's own criterion, a shape decision).
     - Report both subtrains' SNR and their fiducial jitter (SD of the RR difference between the two).
     - A cardiac-locked secondary wave can also be reproducible, so if the two SNRs are within 20% of each other, the split is ambiguous and the channel yields no train.
   - **The refractory's running median is taken over template-matching beats** (at or above the unmasked floor), so extras cannot halve it. k stays a development parameter in {0.6, …, 0.8}.

5. **Unchanged:**
   - the count gate, the transient veto and task 05's plausibility gates;
   - no insertion (invariant 8);
   - the relaxed height and width multipliers and k are fixed on A/B before H;
   - H runs once;
   - every recording from round 6 on is further held-out evidence.

6. **Order:**
   - Finish the module with amendments 1–4.
   - Synthetic tests, with a fixture case per failure mode, including a mid-cycle wave with SNR comparable to the QRS for the ambiguity rule. Then mutants.
   - Commit, then A/B development (no round open), then the H run.

### RULING 2026-09-30 (h) — HR template detector: interpretations ratified; the floor applies to every candidate

**Measured (gems `e48667e`, synthetic only; hash unchanged; not wired into `gated_selection`):**
- **The template train beats findpeaks and task 05** on capture (0.954 against 0.891 at 0.5/s; 0.797 against 0.650 at 2/s, with 34 extras against 517), on floor extras (1.000/0), on regular mid-cycle peaks (split), and on rate rises (refractory removes 0).
- **It correctly yields no train** where the autocorrelation is never clear (capture 6/s), where the split is ambiguous, or where there are fewer than 50 non-suspect clean beats.
- **Not solved:**
  - task 05's misses (0.948, 60 extras);
  - irregular mid-cycle peaks on 25% of cycles (ambiguous split);
  - missing beats whose slots are taken by noise peaks (17).
  
  All three share one cause: **non-suspect candidates never face the floor.**
- **Fixture bug found and fixed:** transients were packed into the first 45 s.

**Rulings:**

1. **The three interpretations are ratified:**
   - "Template-matching" uses the floor built like the beat's own score (masked if an event is in its window, unmasked otherwise).
   - The running median leaves out intervals ≥ 1.5 × the minute's autocorrelation RR.
   - No clear minute means no train, reported as its own reason.

2. **The floor applies to every candidate, suspect or not.** A candidate below its floor is dropped, and its gap is re-searched under the same floor.
   - **The price** is that about 1% of real beats fall below a 1st-percentile floor by construction. These are the channel's worst-shaped beats, often ectopic or noise-corrupted, which HRV practice (NN intervals) excludes anyway.
   - **The trade:** a false beat splits one RR into two short ones and corrupts HRV. A tagged gap can be excluded.
   - Report the fraction of real beats dropped per fixture, and re-measure every failure mode, the three unsolved ones in particular.

3. **Gaps must reach the analysis.** Every dropped or unfilled beat leaves a tagged gap. The stored beats file must carry those tags, for example a `gapAfter` logical vector the length of `heartlocs`, or a list of excluded intervals. Otherwise Andrea's function reads a gap as one long real RR interval.
   - Report how `HR_BR_HRVAnalysis_new.m` handles RR outliers today (any range or ratio rejection) and whether that already excludes gap-spanning intervals.
   - If it does not, propose an **optional** gap input to `HR_BR_HRVAnalysis_beats.m`. With no tags given, outputs stay identical, and the identity proof still holds.
   - Changing her analysis's behaviour is **Andrea's decision**. Report first; do not build it.

4. **Then A/B development, stopping before H.**
   - After the synthetic re-measure passes, run A/B development while no round is open, and choose k and the height and width multipliers.
   - **Report** the A/B results, the frozen parameter set and the commit, then stop. The single H run follows my go-ahead.

**Andrea, 2026-09-30: add `GapAfter`.**
- `HR_BR_HRVAnalysis_beats.m` gets an optional name-value input, `'GapAfter'`: a logical vector the length of `heartlocs`.
- An RR interval whose start beat is tagged is masked exactly as her [100, 500] ms rule masks an implausible interval, and it is treated as a run boundary.
- With `'GapAfter'` absent, outputs are identical to the original, and the identity proof is re-run to show it.
- Add a test on a stored train with known gaps, checking the masked intervals and that no successive difference crosses a gap.
- `HR_BR_HRVAnalysis_new.m` stays untouched. processing_new stays local and uncommitted until Andrea says otherwise.
- **Found on the way:** her 100–500 ms rule passes both a missed-beat interval (about 310–380 ms) and findpeaks' floor extras (about 101–116 ms). So HRV computed from her own `findpeaks` beats today counts both as real intervals. Report, per recording already analysed, how many intervals fall into each class (for her information only; no re-analysis unless she asks).

### RULING 2026-10-01 — A/B shows no gain: H is not run; diagnose, and measure tagging instead of vetoing

**Measured (gems `a228a4a`; 10 A/B recordings, 30-point grid, pre-registered rule):**
- **The template detector gives none of the four development recordings without an HRV-grade train a train, at any combination.**
  - At the chosen point, 48 of 120 channels have fewer than 50 non-suspect clean beats (41 have zero), 11 split ambiguously, 47 fail the transient veto, and 9 fail only the count gate.
  - The closest misses fail the veto: A t01 RVN1 matches the count in 100% of minutes but has harm 0.18. A t02 RVN2 is at 74% with harm 0.43.
- **The no-floor channels are not starved by suspects** (only 1–16% of beats are within 2 ms of an event). Why the clean set is empty was not recorded.
- **Fiducial alignment on real clean beats:** −0.21 to +0.02 ms.
- **The parameter set** (k 0.7, height 0.25, width 0.5) was chosen by the third tie-break, with zero gains.
- **Regression set:** unchanged except B t01, where a template L_T train (SNR 761, 98.0% of stored beats within 2 ms) would replace the stored one.
- **GapAfter is done** (local in processing_new): identity 9/9, the known-gaps test passes, misspelt names are refused, and the nargin fix now lives in the generator.
- **Her own findpeaks beats:** floor extras are 0.2–14% of the intervals her 100–500 ms rule accepts. Missed-beat intervals are near zero.

**Rulings:**

1. **H is not run.** A held-out test of a detector that gained nothing on its development set measures nothing and would spend the held-out animal.
   - The frozen parameter set is void.
   - The template detector stays out of `gated_selection`. No stored train changes, B t01 included.
   - H remains unspent for a future design.

2. **The diagnostic pass is approved: one Drive pass with no round open, recording everything.**
   - **For every channel that yields no train:** the size of the clean set before and after the non-suspect filter, the seed route ((b) or (c)), and the intervals of the seed subtrains against the autocorrelation RR. This tests the hypothesis that a split seed with irregular extras leaves no interval within 10% of the reference.
   - **For every train that fails the veto:** classify each harmed injection as lost (no beat within 2 ms of the true R), moved (a beat within 2–20 ms), or captured (the transient itself accepted). Report it per channel.

3. **Measure, don't adopt: tagging beats near real transients instead of vetoing the channel.** GapAfter now lets HRV exclude intervals, so a beat near a known transient can be tagged instead of trusted. In that case the measure of harm that matters is wrong intervals that are not tagged.
   - **Tagging rule** (fixed now): every beat within ±20 ms of a non-cardiac common-mode event found by `find_events` gets both adjacent intervals tagged.
   - **The injection must be realistic:** a common-mode transient on all nine channels at the recording's own amplitude, so `find_events` can find it, with injections drawn at random times.
   - **Untagged harm:** the fraction of injections that change the train without the changed interval being tagged.
   - **Tagging cost:** the fraction of intervals tagged by the recording's own real events.
   - **Report per channel**, for task 05, findpeaks and the template trains, with the count gate alongside.
   - Do not adopt anything. A veto redefined as "untagged harm ≤ 1%, tagged fraction ≤ 5%" would be a ruling made after seeing these numbers, so it needs Andrea's decision, with both numbers on the table.

4. **Her findpeaks floor extras are for Andrea to act on.** Up to 14% of accepted intervals are false short intervals, which inflate RMSSD and SDNN in any HRV already computed from those channels.
   - Possible interim fixes in her own code: a spacing floor relative to the median RR instead of 100 ms, or a prominence floor. Either changes her outputs.
   - This is her decision. Nothing is changed.

5. **Ratified:** the GapAfter implementation and its proofs; the generator carrying the nargin fix; explicit option parsing; and the finding that the earlier fix was a hand edit.

### RULING 2026-10-01 (b) — the stomach_ref notch is a mechanical rule, applied to every hum recording

- **The rule.** The notch rulings ((c) 5 and addendum 5) named three recordings because those were the only ones known. The rule is mechanical: **any recording whose stomach screen flags ANT1 as mains-dominated gets the notch** in the stomach consumers' stomach_ref.
  - The inputs are chosen by addendum 5's comparison: all stomach_ref inputs if the all-inputs notch changes mmc against ANT1-only, otherwise ANT1 only.
- **Round 6:** B t03 3_1 and B t03 ms3 are routed with the notch.
- **B t03 es2:** its unnotched frozen entry is an omission (my error).
  - Correct it as a routing change: append the corrected entry and recompute every pooled mark.
  - Decided now, before its effect is known. It holds even though it may reclassify s3#m7 and s3#m13, which are in that recording.
- **Report the gate twice,** on table `2de52b5d` (as built) and on the corrected table, with every mark that changes class and why. The corrected table governs.

**Applied 2026-10-01 (corrected table `628d2a28`):**
- **All three hum recordings take the all-inputs notch.** mmc Jaccard between ANT1-only and all inputs: 0.72 (es2), 0.58 (3_1), 0.31 (ms3).
- **Pooled gate: 242/245, lower bound 0.967. Not cleared.**
  - On `2de52b5d` as built it was 238/239 (0.9803).
  - The difference is s3#m7 and s3#m13, which become target misses through mmc damage on the notched stomach_ref.
  - Their `not_target` closure stands as Andrea's classification. The scores stand (closure rule 2).
- **About 141 more target marks** with no new target miss are needed (386 at 3 misses).
- **slow_wave is not immune to the notch.** In B t03 3_1 it removed slow_wave damage from s4#m6 and s4#m7. Ruling (c) 5's "slow_wave untouched" holds only for the first three recordings; the all-inputs comparison stays the mechanical rule.

### RULING 2026-10-01 (c) — tagging rejected; HR from a cross-site lead that nulls the ground signal (measure first)

**Measured (diagnostic pass, 10 A/B recordings; H unspent):**
- **Template detector, no-train channels:** the non-suspect filter is not the cause (41 of 48 channels have zero clean beats before it).
  - The task 05 seed on route (b) is not a cardiac train: median interval 23× the autocorrelation RR, and at most 10.5% of intervals within 10%.
  - The route (c) halves are alike (1.33–2.11× RR, near-equal SNR), so the split is ambiguous.
- **Veto harm is mostly capture for every detector:** task 05 64%, findpeaks 83%, template 54%. The rest is mostly lost beats; moved beats are 2–5%.
- **Tagging:**
  - It costs 20–79% of intervals (median 34–45%).
  - No A recording gains a train.
  - `find_events` catches only 26–64% of the injected transients.
  - The circular own-beat variant tags less and hides capture, as predicted.

**Rulings:**

1. **Tagging is rejected** and is not offered to Andrea for adoption. Its cost is a third or more of the data, and it rescues nothing.

2. **Why every single-channel approach fails.** On a raw contact the ground-site transient and the far-field ECG arrive together. The transient is captured because nothing on one channel can tell it apart from a beat.
   - The tripole cancels both: its contacts are 1.5 mm apart, so they see the same cardiac potential.
   - The ground signal enters every channel with only modest gain differences (contact R² 0.67–0.99 against the outside reference). The cardiac far-field differs strongly **between sites** (left neck, right neck, stomach), because the heart's dipole projects differently onto each.
   - **So a combination of channels from different sites can cancel the ground signal and keep the heart.** In effect it is a bipolar ECG lead built from distant contacts.

3. **Measure, don't adopt. Two variants, A/B only, H unspent:**
   - **(i) Cross-site pairs:** every difference of two raw contacts from different sites (L cuff − R cuff, cuff − stomach).
   - **(ii) Ground-nulling weights:** per recording, estimate the ground gain vector **g** from non-cardiac events (the `cm_gains` method) and the cardiac gain vector **h** from the beat-locked template on the vetted or best available train. Choose **w** to maximise (w·h)² / (wᵀ Σ w) subject to w·g = 0, where Σ is the noise covariance from beat- and event-free stretches.
     - Report how stable g is over the recording (per-minute estimates). If g drifts, w must be estimated per segment.
   - **Detectors:** task 05 and findpeaks, run on the derived lead. The template detector is not used.
   - **Gates, unchanged:** count gate, task 05 plausibility, and the transient veto. The veto injects **realistic** transients: the measured g pattern on all nine channels at the recording's own amplitude, so the test sees exactly what the lead must cancel.
   - **Report per recording:** the best lead, its gate results, its veto harm against the best single-channel harm, and whether the recording gains an HRV-grade train. Report A t01, A t02, A t01 1_2 and A t04 first.
   - **No adoption.** If it works on A/B, the design is written down and frozen and goes to H once, as in ruling (f) 6–7.

4. **The round-record note is confirmed.** Add the `notes` entry to `change_2de52b5d2537d0b2_to_628d2a28cd1f5ed5.json`, written atomically, as proposed.

**Amendments to ruling (c), 2026-10-01 (applied by the build):**
1. **The binding veto, for every lead, is the real-pattern injection:** each transient carries one real non-cardiac event's gain pattern and amplitude. The median-g injection is circular for the weighted lead (w·g = 0 cancels it by construction), so it is reported only.
2. **Half-split protocol.** Everything is estimated and chosen on the first half: event classes, g, h, Σ, w, the best lead and the best single channel. The second half judges only those choices.
   - A recording gains a train only if the chosen lead passes on the second half and the chosen single channel does not.
   - Report how many candidates passed on the first half.
   - Second-half per-minute weights may use that minute's own g, since labels play no part. Drift is tested out of sample.

### RULING 2026-10-01 (d) — the cross-site lead works on A/B; freeze a pairs-only design, cross-check it, then H once

**Measured (half-split protocol, 10 A/B recordings; H unspent; nothing written to the store):**
- **A t01, A t01 1_2 and A t04, which have no single-channel HRV-grade train, each gain one on the held-out half** from a left−right neck contact pair (A t01 via the per-minute weighted lead). The harm is 0.000 under both injections.
- **A t05's gain is real:** its stored L_T fails the real-pattern veto on the second half (0.04).
- **B t01 and B t03 2_2's "gains" are selection artifacts:** SNR picked a raw contact over an L_T that also passed.
- **A t02 and B t03 es2 fail** the second half's count gate (0.889 and 0.80), not the veto.
- **g is 0.96–1.05 on every contact**, with 99–100% sign agreement, and does not drift: the first-half w leaks ≤ 0.04 on the second half. Every chosen pair is left−right neck; none involves the stomach.
- **The weighted lead is fragile:**
  - It could not be estimated in A t01 1_2 (28 clean beats for h) or in B t03 es2 (no quiet samples for Σ, at about 31 events/s).
  - Σ rests on only 1.2–2.1 quiet seconds in four recordings.
  - Where h came from a non-passing train, it can point along g (A t04 cos −0.99).
- **A contact carrying almost no ground signal:** A t04 LVN3 (g −0.03). A contact-health observation, relevant to task 18. *(Corrected 2026-10-02: B t03 2_2 LVN3, listed here earlier, is healthy: g 0.985, agreement 0.999.)*

**Rulings:**

1. **The design frozen for H is pairs only.**
   - **Candidates:** the 9 left−right neck contact differences in both orientations (18), each with task 05 and findpeaks.
   - **Not in the design:** the weighted lead, cross-site pairs involving the stomach, and the template detector.
   - **Why:** pairs carried the substantive result, need neither h nor Σ, and fewer candidates means less selection inflation. Restricting the candidates is a design decision made on development data, with H unspent.
   - **No Σ floor is set**, since the weighted lead is out.
   - **Report from the existing records (no re-run):** whether pairs-only still gives A t01, A t01 1_2, A t04 and A t05 a train under the same half-split selection.

2. **An independent count cross-check is required before H**, as the build proposed. A lead locked on another periodic source would pass its own count gate. Run it on every recording, against references that do not use the lead:
   - **(a) Rate:** per minute, the lead's beat rate against the autocorrelation rate of each other site's raw channels and of stomach_ref, wherever that minute is clear. The lead passes if it agrees within 5% with the median clear reference in ≥ 95% of minutes that have one.
   - **(b) Timing, where a vetted train exists** (A t05, B recordings): ≥ 99% of the lead's beats within 2 ms of the vetted beats, plus a constant offset, which is reported.
   - **(c) Morphology:** the lead's beat-locked template on each raw contact must show the QRS (template SNR on the raw contacts above that contact's own noise-shuffled template).
   - A recording keeps its gain only if (a) and (c) pass, and (b) where applicable.

3. **Freeze, then H once.**
   - Commit the pairs-only module with tests and mutants, outside the hash. Record the commit, the candidate list, the selection rule, the half-split protocol, the binding real-pattern veto and the cross-check.
   - Run H once, on every H recording in rounds 3–6, H t05 2_1 sr included. Report it per recording with all checks. If it fails, report the failure and do not tune.
   - **Adoption is a separate ruling after H.** If adopted, the lead enters `gated_selection` as an additional candidate source, and each recording that gains a train is appended as a routing change with a full recompute.

4. **A note for task 18:** a contact with g near 0 (A t04 LVN3) is not seeing the shared ground. It is likely open or detached, and must not be used for conduction velocity. *(Corrected 2026-10-02: B t03 2_2 LVN3 is healthy.)*

### RULING 2026-10-02 — the cross-check, corrected on development data before H

**Measured (pairs-only, commit `775a809`, A/B only):** under ruling (d) 2 as written, only A t01 es1 keeps its gain. The breakdown:
- **(a) A t01 1_2 and B t01 fail on reference quantisation.** The stomach references' autocorrelation steps by about 6% at these rates, and the median of two split references lands between steps.
- **(b) B t01 and B t03 2_2 miss 99% narrowly:** 99.2% and 99.1% of beats fall within 3 ms (between-lead fiducial jitter). B t02 1_3 passed with a spread of ±0.7 ms.
- **(b) A t05 is a genuine disagreement:** about 21% of beats on each side are unmatched within 20 ms, though per-minute counts agree. Its "vetted" L_T itself fails the real-pattern veto (0.04).
- **(c) A t04 fails only on the detached LVN3** (SNR 5.4 against a shuffle maximum of 16.4). Every healthy contact shows the QRS at 93–227.
- **B t03 2_2's chosen pair uses the detached LVN3.**

**Changing the check now is allowed.** It is part of the design and H is unspent. But A/B results under the corrected check are no longer evidence; H is.

**The corrections are physical ones, not looser thresholds:**

1. **Detached contacts.** A contact is detached if its first-half |g| < 0.2 or its sign agreement is < 0.75. This rule comes from the ground-signal physics, not from the cross-check.
   - A pair containing a detached contact is **not a candidate**.
   - **(c)** requires the QRS on every non-detached contact, with at least 4 non-detached contacts on at least 2 sites. Fewer than that means unassessable, so no gain.

2. **(a) Measure the reference better rather than widening the tolerance.**
   - Cross-check references use parabolic sub-lag refinement of the autocorrelation peak. The count gate itself is unchanged.
   - A minute whose clear references disagree among themselves by > 5% is unassessable, since there is nothing to judge against.
   - **Pass:** agreement within 5% in ≥ 95% of assessable minutes, where assessable minutes are at least half of the minutes with any clear reference. Otherwise the recording is unassessable, so no gain.

3. **(b) Identity is the question, not fiducial precision.** The veto already judges precision.
   - Match beats within **5 ms** after the constant offset, which is far below half an RR, so a wrong beat cannot match. Require ≥ 99% matched **in both directions**.
   - Report the SD of the matched differences; it is not a gate.
   - **A reference that fails the binding real-pattern veto is not a valid timing reference,** so (b) does not apply.
   - **But two trains that disagree by more than 5% of beats** (unmatched within 20 ms, either direction) mean at least one is wrong. Neither is trusted until a beat-by-beat check resolves which one carries the QRS: for each unmatched beat, the template correlation on the non-detached raw contacts.
   - Until then **A t05 gains nothing.** Report the resolution.

4. **The real-pattern veto is the binding veto for every source.**
   - When the lead is adopted, `gated_selection` re-vets every candidate source (task 05, findpeaks and the lead) under it, and every changed recording is appended as a routing change with a full recompute.
   - Report now, without writing anything, which stored HRV-grade trains fail it (A t05 L_T is one).

5. **Order:**
   - Re-run the A/B cross-check under corrections 1–3, with the candidate list cut by correction 1.
   - Report it as development data only.
   - Commit the corrected module, with tests and mutants, and record the frozen design.
   - **H runs once, after Andrea says "done" on round 7.**

### RULING 2026-10-02 (b) — round 7; a diagnosable miss class; H go-ahead

**Measured:**
- **Round 7:** 32/33 raw. On appended table `218e053d` (no existing entry changed), there are 28 targets and 5 below tolerance. **Pooled gate: 269/273, lower bound 0.966. Not cleared.** With 4 target misses, 456 target marks are needed: about 183 more with no further target miss.
- **The new target miss, s5#m5** (B t01 3_1, a hum recording): max z 2.81 on ANT3. Its only damage is to mmc, on the notched stomach_ref.
- **Corrected cross-check (`78a5b2d`, development data only):**
  - A t01 es1 and A t01 1_2 keep their gains; A t04 keeps its gain with LVN3 excluded.
  - **A t05 is resolved:** 97.4% of the lead's unmatched beats carry the QRS, against 1.3% of the stored L_T findpeaks train's. **The stored A t05 L_T train sits on a non-QRS peak in about a fifth of its beats.**
  - B t03 ms3 R and H t01 1_2 L are distrusted (check (iii)).

**Rulings:**

1. **A diagnosable miss class: hum recordings, mmc-only damage, below threshold.** Three of the four target misses (s3#m7, s3#m13, s5#m5) are in B hum recordings. Each is below `z_enter` on every detection signal, and each is a target only through mmc damage on the **notched** stomach_ref.
   - **Hypothesis.** Detection reads un-notched inputs (invariant 43 keeps ANT1 un-notched for detection). In hum recordings, mains leakage inflates the baseline σ of the stomach signals and depresses their z. The consumer reads the cleaner notched signal, so detection is blind to exactly what mmc is sensitive to.
   - This is branch 1 of task 09's "If it fails": a diagnosable class, which gets a generator feature and a re-measure.
   - **Measure first, writing nothing to the store:**
     - (a) The three misses' z on the notched stomach_ref, in detection's own bands and windows. Would any cross `z_enter`?
     - (b) The same z for every covered target mark in hum recordings, to check the class separates.
     - (c) The coverage and chance-margin cost of adding the notched stomach_ref signals in hum recordings only.
   - **If (a) confirms it, the fix is the pre-declared additive extension** (ruling of 2026-09-29, round-4 miss, branch 3):
     - add the notched stomach_ref signals in hum recordings as extra detection signals;
     - prove it is a strict superset on every cached region;
     - keep every original score (the four misses still count);
     - recompute the chance margin.
     
     The generation hash is unchanged and the extension is recorded explicitly. Propose it to me before building.

2. **H go-ahead.** The frozen design is `78a5b2d`. When the A/B re-run and item 4 finish, run H once on every H recording in rounds 3–7, under that exact commit and protocol, with nothing changed. Report per recording with all checks. If it fails, report the failure and do not tune.

3. **A t05's stored L_T train is wrong in about a fifth of its beats.** It is still the stored HRV-grade train and A t05's peri-R source. At adoption it fails the binding real-pattern veto and is replaced (ruling 2026-10-02, 4).
   - **Report now** which pooled marks' classes depend on A t05's HR or peri-R route, so the size of the change at adoption is known.
   - **Andrea:** any HRV she has already computed from A t05's L_T findpeaks beats is affected.

4. **B t02 1_3 ANT2 "detached" with (a) newly failing:** report the cause. Do not change the rule for it.

### RULING 2026-10-02 (c) — H result; the pairs lead is adopted; detached contacts leave the rate references

**Measured (H once, under `78a5b2d`, byte-identity checked before each recording):**
- **H t05 2_1 bl, H t09 2_2 bl and H t01 1_2 bl gain an HRV-grade train that passes every check.** The second-half harm is 0.000 under both injections, rate and morphology pass, and no chosen single channel passes.
- **H t01 es2 fails** (veto 0.03, rate 0.75).
- **In all three H stim/recovery recordings, no pair passes the count gate and plausibility at all.**
- **Development (A/B) gains:** A t01 es1, A t01 1_2 and A t04. A t05 is resolved in the lead's favour: 97.4% of its unmatched beats carry the QRS, against 1.3% for L_T.
- **All 10 stored HRV-grade trains pass the real-pattern veto** on their full regions. A t05 L_T, B t03 2_2 L_T and B t03 es2 L_T sit exactly at 0.01. The veto cannot see wrong-peak picking.
- **The hum/mmc class is not explained by baseline inflation.** On notched inputs the three misses peak at z 0.70–2.81. The extension would be a strict superset at negligible cost, but recovers nothing.

**Rulings:**

1. **The pairs-only lead (`78a5b2d`) is adopted** as an additional candidate source in `gated_selection`. Every source faces the same gates, with the real-pattern veto binding (ruling 2026-10-02, 4).
   - **Incumbent preference:** a stored HRV-grade train that still passes every gate stays, unless rule 2 applies. The lead fills recordings with no passing train. This keeps adoption to the cases with evidence and avoids churn (B t03 2_2 keeps its L_T).
   - **Recordings that gain:** A t01 es1, A t01 1_2, A t04, H t05 2_1 bl, H t09 2_2 bl, H t01 1_2 bl, and A t05 by replacement (rule 2).
   - Each recording that gains is appended as a routing change. Its peri-R routes are recomputed from the new train, since hump cuffs distrusted for lack of a train may be restored. Then every pooled mark is recomputed, with all three numbers and the class changes listed.
   - Beats are stored in Andrea's format with `gapAfter` (`emit/hr_beats.py`).

2. **Disagreement beats incumbency.** Where a stored train and a passing lead disagree on > 5% of beats (unmatched within 20 ms, in either direction), resolve beat by beat by template correlation on the non-detached contacts. The train that carries the QRS wins.
   - Applied now: A t05 → the lead.
   - Report the same check for every recording where a stored train and a passing lead coexist.

3. **Detached contacts are left out of the rate references in (a),** as they are left out of (c). A contact that does not see the ground signal is not a physiological reference.
   - This is decided after H, but it changes no H outcome (no H contact is detached). On development data it changes only B t02 1_3, which gains nothing.
   - Record it as a post-H amendment to the adopted module, and re-run the A/B and H cross-checks to show that no adoption decision changes.

4. **HR in stim/recovery remains open.** No lead passes there on H, and most stim/recovery recordings without HR stay without it.
   - The likely cause is stimulation itself, but that is a hypothesis.
   - **Measure, don't build:** in the H and A stim/recovery recordings, compare the count gate and veto in the 2-minute stim epoch against the 20-minute recovery. Report whether the failures are confined to stim.
   - Priority is Andrea's call.

5. **The hum/mmc miss class: next hypothesis, measure only.** mmc's damage rule judges a mark against a local ±30 s MAD, while detection uses the whole-region baseline.
   - Report the three misses' z against a ±30 s local baseline on detection's stomach bands, the same for the covered targets in those spans, and the coverage cost.
   - Do not propose an extension unless the misses cross `z_enter` and separate from the covered targets.

**Addendum to ruling 2026-10-02 (c), on storage:**
- **The stored product must pass on what it is used for.** The half-split protocol exists for honest selection. A stored train covers the whole region, so `gated_selection` judges it on the full region with every gate, the cross-check included.
- A ruled gain that fails on the full region is not stored. Report it with the cause.
- If the cause is a defect in how the full-region check is computed, fix it, test it and re-run. If the cause is real, the recording has no gain.
- Do not split storage into "valid minutes" without a ruling.

### RULING 2026-10-02 (d) — the "fixed-rate" stomach references are a mains defect in the check; fix and re-run

**Measured (full-region adoption pass, `c7358cc`):**
- A t01 es1 matches its validation.
- A t05 is replaced by the RVN1−LVN1 pair (rule 2).
- **A t04 and A t01 1_2 store nothing on the full region.** Rate (a) fails because the stomach references sit at fixed rates, 360.2, 399.6, 423.7 and 480.2 bpm, each constant to ±0.2, while the lead varies smoothly. A t01 1_2 also fails morphology on LVN3 (SNR 7.2 against 7.9).
- A t03 3_2 bl (round 7, not on the ruled list) would store a pair.
- B incumbents stay.

**Those rates are 7200/m bpm for integer m** (m = 20, 18, 17, 15 to within 0.01): autocorrelation lags of exactly m/120 s. The references are locking onto **rectified 60 Hz hum** (120 Hz after rectification), not onto any physiological source.
- This also explains the "≈6% steps" seen earlier: neighbouring m differ by 5–7%.
- So it is a **defect in how the cross-check reference is computed** (the addendum's first case), not evidence against the lead.

**Rulings:**

1. **Fix the references.**
   - Notch every rate-reference channel at 60 Hz and its harmonics up to the band edge before rectification and autocorrelation.
   - Then flag any reference minute whose rate lies within ±0.3 bpm of 7200/m as hum-locked, and treat it as not clear.
   - Add a test: a synthetic rhythm with rectified hum must not lock onto the 1/120 s grid.
   - Neither the lead nor the count gate changes. The lead is a left−right difference, so the hum is common mode and cancels.

2. **Re-run the cross-check with the fixed references** on every A/B and H recording, and in the full-region adoption pass.
   - Report every decision that changes, in either direction.
   - **A change to an H result is reported as a post-H correction of a defect in the check, not as new evidence.** H t01 es2 fails the veto, so it cannot become a gain.

3. **Check the count gate for the same defect, report only.** For every stored train and every passing pair, count the minutes where the train's own autocorrelation rate sits within ±0.3 bpm of 7200/m.
   - If the count gate is hum-locked anywhere, report the recording and stop. Do not change the gate without a ruling.

4. **A t01 1_2's LVN3 morphology failure is real under (c).** If it still fails after the fix, A t01 1_2 has no gain. Do not relax (c).

5. **The adopted module applies mechanically to every recording, not only the list known at adoption.**
   - A recording where full-region `gated_selection` stores a passing pair (every gate and the cross-check) is appended, for example A t03 3_2 bl.
   - Report such recordings separately as further held-out evidence, since they were never seen during design.

6. **Then append and recompute** as ruled, and push `c7358cc` or its fix.

### RULING 2026-10-02 (e) — mains in the count gate; the guard becomes a persistence test; veto precision near threshold

**Measured (gems `c7358cc` + `d203469`, held locally; nothing appended):**
- **The notch fixes the references.** Animal A's stomach references track the lead within about 1%.
- **The ±0.3 bpm guard discards true heart readings** that happen to sit on the 7200/m grid (A t01 es1 minute 11 at 360; A t04 minute 1 at 379), and that alone costs both gains.
- **The count gate itself can lock onto mains.** In B t02 3_3 three passing pairs have their own autocorrelation on the grid in 5–13 minutes, while the heart runs at about 311 bpm. Scattered single grid minutes elsewhere are consistent with the heart sitting at a grid rate.
- **H t01 es2:** its pair failed the veto on H's held-out half (0.03) but passes over the full region (≤ 0.01), with 200 injections in each case.
- **Item 4:** stim/recovery HR is lost mainly to the 20-minute count gate. The first 2 minutes of recovery give a passing train in all 8 recordings.
- **Item 5:** no extension. On a ±30 s local baseline only s5#m5 crosses `z_enter`, the misses do not separate from the covered targets, and the pooled chance margin would fall from +0.149 to +0.108.

**Rulings:**

1. **The count gate gets the same mains fix as the references.** Its autocorrelation input is notched at 60 Hz and its harmonics up to the band edge before rectification, for every candidate source.
   - A gate whose reference measures mains does not measure the heart. This is the same defect, so it gets the same fix. The gate's thresholds (0.05, 95%) are unchanged.
   - Re-gate every stored train and candidate. Report every change in either direction. An incumbent that fails after the fix goes through `gated_selection` as usual.

2. **The ±0.3 guard is replaced by a persistence test** for references and for the count gate.
   - A minute is hum-locked only if the rate sits within ±0.3 bpm of the **same** 7200/m value for ≥ 3 consecutive clear minutes. Such minutes are treated as not clear.
   - **Why:** a mains lock is constant to ±0.2 bpm for minutes, as measured, while a real heart rarely holds within 0.6 bpm for 3 minutes. A single-minute coincidence is not a lock.
   - The fixed guard was part of a defect fix, not of the frozen design. It is replaced because of its own measured false positives.
   - Keep reporting single-minute grid coincidences as a diagnostic, not a gate.

3. **The veto's precision near the threshold.** The veto estimates a fraction from 200 injections, which is too few at 0.01: 2 hits against 6 is within noise.
   - Rule, applied to every source, stored or candidate: if the one-sided 95% interval of the harm estimate contains 0.01, add injections in blocks of 200, up to 2,000, until it does not. Then decide on the estimate.
   - The threshold is unchanged. Report every decision this changes, including A t05 L_T, B t03 2_2 L_T, B t03 es2 L_T and H t01 es2.
   - **H t01 es2 is then decided by the full-region estimate**, like any other recording. Its H result stays recorded as "failed on the held-out half", and a later storage is not counted as H evidence.

4. **The hum/mmc miss class is closed as `accepted_limitation`** ("below threshold on every detection signal under global and local baselines; damage only to mmc in hum recordings; no generator feature recovers it without large cost"). Its misses keep counting.

5. **Re-run in this order:**
   - (i) the A/B and H cross-checks;
   - (ii) the full-region adoption pass under rulings 1–3;
   - (iii) item 4 (stim epoch against the recovery windows), because the count-gate fix may change it.
   
   Report the changes. If no other stop condition arises, append, recompute and push. The stim/recovery question goes to Andrea with the re-run item 4 numbers.

### RULING 2026-10-02 (f) — adoption ratified; HR stored per minute so Andrea's windowing applies; mask-grade pairs need the cross-check

**Measured:** the adoption is appended as table `4055cdce`, with 13 entries changed. **Pooled gate 272/276 (0.966), not cleared**; excluded = target 318/322; raw 328/332; chance margin +0.147.
- All seven ruled gains hold. Three held-out recordings were appended mechanically.
- Veto precision resolved every threshold case.
- **Stim/recovery:** the full 20-minute recovery stores a train in 3 of 8 recordings, and the first 2 minutes do in 7 of 8. What fails is the whole-recording count gate, not stimulation.

**Andrea, 2026-10-02:** her analyses (`processing_new`) already work in windows and keep a window when less than 50% of it is blanked. *(She wrote "60 min windows". The build must read `processing_new` and record the exact window length and rule it uses for HR/HRV.)* Nothing is wrong with that approach. It fails here only because our storage rule is whole-recording: one bad stretch means no train at all, so her window rule never gets to act.

**Rulings:**

1. **The adoption is ratified,** as are the lock test reading the refined peak, the persistence limitation (minutes are lost, never wrong), and the s5#m5 closure (s3#m7 and s3#m13 keep their earlier `not_target` closures).

2. **HR storage becomes per minute, so that her windowing does the rest.** This supersedes the addendum's "no valid-minutes storage without a ruling".
   - **A train is eligible** if it passes the recording-level gates: the transient veto with precision, task 05 plausibility, and for pairs the cross-check's morphology (c) and timing (b) where it applies.
   - **Each minute is valid** only if it is clear and the train's beats are within 5% of the minute's autocorrelation rate (the count gate's own test, now applied per minute). For pairs, the minute must also pass (a) where it is assessable.
   - **Minutes that are not valid become tagged gaps.** Their beats are removed, and `gapAfter` marks the boundaries. Wrong beats are never left in place to look valid to the window rule.
   - **Store the train with the most valid minutes** (ties go to template SNR). Incumbent preference and rule 2 still apply among trains that pass the recording-level gates.
   - **Report per recording:** valid minutes and longest valid run, and how many of her analysis windows would then pass her < 50%-blanked rule. This is the number that matters to her.
   - Recordings that already pass end to end keep their trains; their invalid minutes, if any, become gaps.
   - **Development first.** Run on the A/B and H stim/recovery and baseline recordings, report, and then append as routing changes with the full recompute. Nothing is appended before I see the report.

3. **Mask-grade pairs need the cross-check.** A pair establishes that it is cardiac only through (a) and (c). A mask-grade pair (used for peri-R, where the veto is not required) must therefore pass (a) and (c).
   - Re-check A t02 and A t01 3_2 sr. If either fails, its peri-R route reverts. That is a routing change, with the recompute.

### RULING 2026-10-02 (g) — per-minute storage: reading A with a majority rule 2; the trailing minute; rejected minutes reach Andrea as blanked time

**Measured (gems `175a76d`, development only, nothing appended):**
- **Andrea's windowing, read from `processing_new`:** HR uses 60 s windows (the longest clean stretch); beat counts and HRV use 20 s windows (count rate kept if ≥ 50% of the window is valid); all are centred with a 1 s step. There is no 60-minute HR window.
- **Under reading A** (the literal reading of (f) 2): 18 of 19 incumbents keep their trains, and five stim/recovery recordings gain pair trains (11–18 of 19 minutes valid). B t01 3_1 loses its incumbent: rule 2 against a pair failing (a) is unresolved, since both trains' unmatched beats carry the QRS (0.72 vs 0.85).
- **Under reading B** (pairs must also pass (a) over the recording), those five recordings store nothing.
- **The trailing partial minute** (about 49 s) is never assessed, so its beats are removed even from trains that pass end to end.
- **`HR_BR_HRVAnalysis_beats.m` uses `gapAfter` only for RR intervals.** A removed minute reads as clean time with no beats, so HR and counts across it read low.
- **Both mask-grade pairs fail (a) by one minute each** (A t01 3_2 sr minute 14: +6.1%; A t02 minute 8: −5.1%). Their peri-R routes revert.

**Rulings:**

1. **Reading A stands.** In per-minute storage a pair is eligible on the recording-level gates (veto with precision, plausibility, morphology (c), timing (b) where it applies). Per-minute (a) removes the minutes it fails.
   - **Why:** every minute that is kept has its rate confirmed against independent references, and (c) confirms the QRS across the recording. Identity is established for exactly the minutes stored.
   - Item 3's requirement that mask-grade pairs pass (a) over the recording stays. A mask covers the whole recording and has no per-minute gaps.

2. **Rule 2 needs a clear winner.** Train X wins over train Y only if X's unmatched beats carry the QRS in a majority (> 0.5) and Y's in a minority (< 0.5).
   - **Otherwise the comparison is ambiguous.** An incumbent stays. With no incumbent, neither train is stored, and the recording is reported.
   - **Effect:** A t05 (0.974 vs 0.002) and A t01 es1 (0.867 vs 0.022) are unchanged. B t01 3_1 (0.72 vs 0.85) keeps its incumbent. B t03 3_1 (pair 0.88 vs L_T 0.32) goes to the pair.
   - This tightens the rule. The ruled cases already met it.

3. **The trailing partial minute** is assessed with the same tests if it is ≥ 30 s long, with the count test scaled to its duration. Otherwise it is a tagged gap.

4. **Andrea, 2026-10-02: add `BlankSpans`.** `HR_BR_HRVAnalysis_beats.m` gets an optional input `'BlankSpans'`: an N×2 list of [start stop] times read from the beats file's new `blankSpans` field.
   - Those spans are merged into her blank mask, so the 60 s HR and 20 s count/HRV windows treat rejected minutes exactly as artifact blanking.
   - `gapAfter` stays for RR.
   - With `'BlankSpans'` absent, outputs are identical (re-run the 9/9 identity proof).
   - Add a test: a stored train with a removed minute gives the same HR in windows away from the span, and treats windows across it by her ≥ 50%-valid rule.
   - `HR_BR_HRVAnalysis_new.m` stays untouched, and processing_new stays local and uncommitted.

5. **Then append as one routing change from `4055cdce`:**
   - per-minute storage under rulings 1–3 for every recording;
   - the two mask-grade reverts (A t01 3_2 sr, A t02);
   - peri-R recomputed from any changed train;
   - the full recompute with all three numbers.
   
   Report: per recording, valid minutes, longest run, and Andrea's passing 60 s HR and 20 s count windows with `BlankSpans` applied. Push after verification.

**Applied 2026-10-02 (table `6fac53ee`, gems `21d82a9`):**
- **Pooled gate 277/281 (lower bound 0.966), not cleared.** Excluded = target 307/311; raw 328/332; chance margin +0.151.
- Per-minute storage gives HR to five stim/recovery recordings that had none (12–19 of 20 minutes valid). Andrea's MATLAB matches the emulated window counts on four files.
- **Ratified interpretations:**
  - `blankSpans` is stored as 1-based inclusive sample indices (her `blankIdx` convention).
  - Peri-R stays placed by whole-recording trains only.
  - A mark within 0.2 s of a rejected HR span is excluded for HR.
  - Rule 2's majority test applies in both choosers.
- **Seed `perir.py`'s chance-loss sampler,** so that its exposure numbers reproduce. No routing field depends on them.
- **For Andrea (her code, her decision):** `processing_new` drops beats within 0.75 s of any blank but counts that time as valid, so count rates read slightly low next to every blank. Fixing it means excluding the edge-buffer time from valid seconds too.

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
<!-- /TASK -->

<!-- TASK:10 slug=label-conversion deps=07,09 gate=no -->
## Task 10 — Convert the prior labelled corpus to event judgments

> ### The "43 recordings" figure is wrong. The previous model was trained on **12**.
>
> Source: `My Drive/data from ML PC 8 13 26/detector-pyqt`, the working copy from
> the ML machine — `detector-core/loro_out/loro_summary.json`, 12 LORO folds.
> Every number below is from that file, not from reconstruction.
>
> **Not every `*_blankmotion.mat` is a human label.** Some are the previous
> model's *inference output*. Training on those would be circular — the new
> detector learning the old detector's decisions, including its errors, with no
> way to tell from the file. The old manifest already models this correctly:
> `detector/manifest.py:55` defines `LABEL_SOURCES = {"human", "model",
> "mixed"}` per recording. **Carry `label_source` into our registry as a
> required field, and never train on `model` without an explicit opt-in that is
> recorded in provenance.** A recording with no `label_source` is `unknown` and
> is excluded, not assumed human.
>
> **The 12, with the previous model's own LORO results:**
>
> | recording | real pos | recall | bad frac |
> |---|---|---|---|
> | `E1000_FRE_E1000_stim_rec_1406` | 496 | 0.992 | 0.294 |
> | `E1000_JEL_E1000_bl_1315` | **4** | 0.500 | 0.046 |
> | `E100_lol_E100_stim_rec_1122` | 1981 | **0.406** | 0.234 |
> | `M100E10_LOL_CME2_stim_rec_2253` | 1239 | 0.852 | 0.431 |
> | `M100_JEL_MS2_bl_1945` | **18** | **0.056** | 0.001 |
> | `M100_LOL_MS2_stim_rec_2031` | 1665 | 0.811 | 0.325 |
> | `M10E10_ORE_CME_stim_rec_1908` | 1104 | 0.971 | 0.617 |
> | `mec100_jel_MS2_stim_rec_1346` | 482 | 0.925 | 0.250 |
> | `mec100_lol_MS2_stim_rec_1426` | 325 | 1.000 | **0.984** |
> | `mecfreq_fre_MS2_stim_rec_0035` | 308 | 0.620 | 0.216 |
> | `mecfreq_jel_MS2_bl_2302` | **15** | 0.933 | 0.178 |
> | `mecfreq_loll_MS2_stim_rec_2340` | 3524 | 0.692 | 0.429 |
>
> Pooled: **`recall_real` 0.734**, `recall_syn` **1.000**, mean bad fraction
> 0.334, gates passed 2 / 7 / 10 of 12.
>
> **Four things to take from this table.**
>
> 1. **0.734 is the number to beat.** Task 09's gate has had no empirical target;
>    it has one now, measured by the same LORO protocol on the same data.
> 2. **`recall_syn = 1.000` against `recall_real = 0.734` is the synthetic-transfer
>    gap, measured.** A detector that finds every injected artifact and 73% of
>    real ones is being scored on the wrong thing. Task 09's gate must be carried
>    by real labels; synthetic recall is a smoke test, not evidence.
> 3. **Three folds are not measurements.** 4, 15 and 18 real positives — a recall
>    of 0.056 on 18 positives is one beat either way. Exclude folds below a stated
>    positive count from the pooled figure and report them separately, or the
>    headline is an average over three coin flips.
> 4. **`bad_fraction = 0.984` with `recall = 1.000`** is perfect recall on a
>    recording that is 98% positive. Report prevalence beside recall everywhere,
>    always.
>
> **The animals are four, not seven: FRE ×2, JEL ×4, LOL ×5, ORE ×1.** This is a
> hard constraint on the three training modes (task 12): **a per-animal model for
> ORE cannot exist** at n=1, and FRE at n=2 has no held-out fold worth reporting.
> The per-animal versus pooled comparison must state which animals it could
> actually fit, and the learning-curve control matters more here than the
> normalisation argument ever did.
>
> **The previous formulation was positive-unlabelled. Ours is not, and must not
> become so by imitation.** `phase_01` §1.6 sets
> `trust_level ∈ {"real_positive", "unlabeled_clean"}` and `phase_03` weights
> the second at `w_neg = 0.3`. That was the right hedge **for that design**: it
> labelled *spans* and then swept a stride across the entire recording, so every
> window outside a marked span really was unlabelled, and calling it clean would
> have been a lie.
>
> **This design does not have that problem.** Task 10 replays candidate
> generation and a human adjudicates *candidates*; a candidate that was examined
> and not marked is a **human-confirmed negative**, not an unlabelled one. The
> classifier is ordinary binary and there is no `w_neg` to sweep. Importing PU
> here would be inheriting the shape of someone else's constraint.
>
> The unlabelled region is real but lives elsewhere: **frames the generator
> never proposed**. That is a recall question, owned by task 09's gate and its
> exhaustively-labelled ten minutes, not a loss weight. Separating those two
> concerns is precisely what lets the classifier stay binary.
>
> **Two consequences that do need stating.**
>
> 1. **An unadjudicated candidate is excluded, never a negative.** Task 07 is
>    explicit that humans label a *sample* of candidates. Candidates outside that
>    sample carry no label and must be dropped from training — the training set
>    is smaller, not dirtier. A silent `fillna(0)` anywhere near the label column
>    would reintroduce PU's problem without PU's hedge.
> 2. **How the sample is drawn is a modelling decision and must be recorded.**
>    Uncertainty sampling produces a training set enriched for hard cases, which
>    is efficient for learning and biased for *evaluation*. **The held-out
>    evaluation sample must be drawn at random**, separately from whatever
>    strategy selects training candidates, and the two draws recorded distinctly
>    in provenance.

**Module:** `model/labels.py`
**Depends on:** 07, 09

### Purpose
The existing labels are human-dragged intervals. The new target is one judgment per
candidate **event**. Convert without inventing negatives.

### Algorithm
1. Replay candidate generation over all 43 recordings.
2. A candidate overlapping a human-marked segment inherits `motion`
   (`source='inherited'`).
3. **Everything else is `unjudged`, not `negative`.**
4. Sample the unjudged and have them reviewed to estimate how much was previously
   being mislabelled as clean.

### Why this matters
The old design assigned unmarked spans `w_neg = 0.3`, so real artifacts the human
skipped taught the model that artifacts are clean. That is the most likely reason
recall resisted five separate reweighting knobs, and it is not fixable by
reweighting — only by not asserting the label.

### Outputs
A parquet table: `recording, animal, candidate fields, judgement, source`. Plus a
report: how many candidates inherited `motion`, how many are `unjudged`, and the
estimated true-positive rate among the unjudged sample.

### Tests
- a candidate overlapping a labelled segment by ≥50% inherits `motion`
- a candidate in an unmarked span is `unjudged` and is **excluded** by the training
  loader (assert the loader drops it — this is the guard against the old bug)

### Acceptance
Report the `unjudged` true-positive estimate. If it is high, say so plainly: it
quantifies how much the previous model was being actively mistrained.
<!-- /TASK -->

<!-- TASK:11 slug=features deps=07 gate=no -->
## Task 11 — Features per candidate

**Module:** `detect/features.py` (replaces the Phase 1 parquet builder)
**Depends on:** 07. Tasks 17 (video) and 18 (velocity) contribute **optional
columns**; build and ship the feature matrix without them, then add the columns
when those tasks land. They are not build dependencies — treating them as such
creates a cycle through 12→13→14→15.

### Hard constraints
- **Channel-count independent**: aggregate per-channel statistics (max, median,
  fraction above threshold). **Never concatenate per-channel columns** — one model
  must serve both the 5-channel and 9-channel cohorts.
- **Animal-invariant**: ratios and cross-channel relations. Absolute microvolts
  enter only as `z`.

### Feature families
- **band-power ratio 100–300 ÷ 300–5000** — motion carries low-band energy a nerve
  burst does not; close to a free discriminator
- **spatial**: fraction of channels above threshold; mean pairwise envelope
  correlation; each channel's power ÷ median across channels; common-mode ÷
  residual power; within-cuff vs across-cuff agreement (contacts 1.5 mm apart see
  near-identical artifact and different neural signal)
- **onset rate** = derivative of the **log** envelope (scale-free fractional rate).
  This is where sustained-vs-transient discrimination lives: level and derivative
  are correlated for brief events and decouple for sustained ones
- **shape**: envelope derivative, slew rate, kurtosis, line length, spectral
  entropy, spectral edge
- **clipping fraction** — also a hard mask criterion in task 14
- all of the above recomputed at ±100, 250, 500 ms of context
- **video** (task 17): motion energy and its derivative, max over ±100/250/500 ms,
  time since last motion peak, **per ROI: headstage, tether, commutator**
- **non-physiological-velocity energy** (task 18), new cohort only

### Tests
- feature vector length is identical for a 5-channel and a 9-channel recording
- scaling every channel by 10× leaves all features unchanged except the explicit
  amplitude ones (the animal-invariance test)
- the 100–300 ÷ 300–5000 ratio separates injected motion from injected spike bursts
  with AUC > 0.8 **on synthetic data alone**, before any training

### Acceptance
Feature matrix builds for both cohorts; the invariance tests pass; per-feature
missing-value rates reported.
<!-- /TASK -->

<!-- TASK:12 slug=classify deps=10,11 gate=no -->
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
<!-- /TASK -->

<!-- TASK:12A slug=model-registry deps=12 gate=no -->
## Task 12A — Model registry and **user-selected** inference

**Module:** `model/registry.py`
**Depends on:** 12

### Purpose
With three modes there is more than one model, so something must record which
model scored which recording. **That choice belongs to the user, not to an
automatic rule** — this is a research tool, and a pipeline that silently swaps
models between runs makes results unreproducible and unexplainable.

The registry's job is therefore to **present the options honestly and record the
choice**, never to decide.

### Signature
```python
@dataclass(frozen=True)
class ModelSpec:
    mode:        TrainingMode
    animal:      str | None        # None for POOLED
    version:     str
    corpus_hash: str               # hash of the training event ids
    calibrator:  Path
    trained_at:  datetime
    metrics:     dict              # the numbers it shipped on, by protocol
    n_train_events: int

def list_applicable(rec: Recording, registry: Registry) -> list[ModelSpec]
    """Every model that MAY be applied to this recording, with its metrics
    attached so the user can choose on evidence. Never returns a default."""

def run_inference(rec: Recording, chosen: ModelSpec, registry: Registry) -> ...
    """Fails loudly if `chosen` is not in list_applicable(rec)."""
```

### Rules
- **No automatic selection anywhere.** There is no "default model", no fallback
  chain, no `current_model` auto-resolution at inference time. If the user has not
  chosen, the run does not start.
- `list_applicable` **excludes** models that cannot legally apply: a `PER_ANIMAL`
  or `ADAPTED` model for a different animal. Assert on `animal` mismatch and fail
  loudly rather than filtering silently.
- Each returned `ModelSpec` carries the metrics it shipped on **and the protocol
  those metrics came from**, so the picker can show "F1 0.88 (LOAO)" next to
  "F1 0.94 (LORO)" without inviting the invalid comparison (invariant 12).
- Mark a model **`unvalidated`** if it has no held-out metrics. It stays
  selectable — a research tool should not block experimentation — but the choice is
  flagged in provenance and in the UI.
- Registry is append-only. Promotion and rollback reuse the existing
  `current_model` pointer machinery, extended to a pointer **per (mode, animal)**;
  that pointer is a *label*, not an auto-selector.
- The chosen `ModelSpec` goes into the mask provenance (task 15). **A mask whose
  provenance does not name a model is invalid.**

### Batch runs
For a batch, the user chooses **once per animal**, not once per recording, and the
UI shows the resolved assignment table for confirmation before anything runs. A
batch spanning animals with different chosen modes is allowed and is recorded — but
the QC report must surface it, because mode then varies across the dataset and
becomes a covariate (see task 19).

### Tests
- `list_applicable` never returns a `PER_ANIMAL` model for another animal
- `run_inference` with a model not in `list_applicable` raises
- no code path produces a `ModelSpec` without an explicit user choice — assert by
  searching for a default argument in the inference entry point
- provenance round-trips the full `ModelSpec`
- an `unvalidated` model is selectable and is flagged in provenance

### Acceptance
Given an animal with all three modes trained, the picker lists all three with
their metrics and protocols, and running without a choice is impossible.
<!-- /TASK -->

<!-- TASK:13 slug=extent deps=12 gate=no -->
## Task 13 — Extent per consumer

**Module:** `extent/tolerance.py`
**Depends on:** 12

### Purpose
A confirmed motion event does not have one duration — it has one duration **per
consumer**, because a 200 ms excursion destroys spike detection and is invisible to
the slow-wave analysis.

### Signature
```python
def compute_extent(event: Event, z: dict, consumer: str) -> tuple[float, float]
```

**`P(motion)` must come from a calibrated model** (task 12). Thresholding on raw
LightGBM output is mode-dependent: the same nominal threshold masks different
amounts of data under a pooled vs a per-animal model, and that difference reads as
a detection difference when it is a calibration artifact.

### Algorithm
For each confirmed motion event × consumer, find the interval where **that band**
exceeds **that consumer's tolerance** (table in Part A.4), padded by the measured
filter settling time.

**Measure the settling time**: run `impz` on the actual bandpass and take where it
falls below 1% of peak. `P.edgeBufferMs` is currently 5; expect 30–50 ms. Report the
measured value per band.

### The cardiac tolerance is operational, not an amplitude
Suppress the span, re-run the peak detector, ask whether the beat train changed. A
1–2 ms fiducial shift moves RMSSD and SD1, so an amplitude threshold is only an
approximation to this test. Implement the operational version.

### Slow bands
Extents there are inherently coarse — the 0.5–3 Hz mask has ~6 s resolution and no
padding logic changes that. Do not pretend to finer resolution than the window
allows; report the resolution alongside the mask.

### Tests
- an event that crosses the ENG tolerance but not the slow-wave tolerance produces
  an extent for `spikes` and **none** for `slow_wave`
- measured settling time from `impz` matches an independent step-response test
- the operational cardiac test: an event that shifts a fiducial by 2 ms is caught;
  one that shifts it by 0.05 ms is not

### Acceptance
Extent per consumer emitted for every event; measured settling times logged per
band; slow-band resolution stated.
<!-- /TASK -->

<!-- TASK:14 slug=routing deps=13 gate=no -->
## Task 14 — Routing

**Module:** `extent/routing.py`
**Depends on:** 13

### Purpose
Rejecting data is the most expensive response. Try the cheaper ones first.

### Order — strictly
1. **correct** — where contaminant and signal are separable in frequency *within
   that band*. A baseline drift is already gone from the ENG band after the 100 Hz
   high-pass, so rejecting that span destroys good data for nothing. **Note the
   qualifier**: a 0.3 Hz drift *is* in the slow-wave band and this route does not
   apply there.
2. **subtract** — stereotyped, timing known, residual verified. Verification is
   mandatory: emit the residual and assert it is below the band's noise floor.
3. **reject** — only where the disturbance overlaps the signal in **both** time and
   frequency.

### Clipping bypasses the classifier
If the amplifier saturates, the signal is non-linear and the envelope can
*understate* the damage. Mask directly on the fraction of samples at the rail,
before and independent of any model decision.

### Tests
- a pure drift event in the 300–5000 Hz consumer routes to `correct`, not `reject`
- the same drift in the 0–2 Hz consumer routes to `reject`
- a clipped span is masked with no model call (assert the classifier is not invoked)
- a `subtract` route whose residual exceeds the noise floor falls back to `reject`

### Acceptance
Routing decision recorded per event per consumer, with counts by route.
<!-- /TASK -->

<!-- TASK:15 slug=emit-qc deps=14 gate=no -->
## Task 15 — Mask emission, QC, provenance

**Module:** `emit/masks.py`, `emit/qc.py`, `emit/provenance.py`
**Depends on:** 14

### Emit
- one boolean mask **per consumer** over the common grid (`True` = invalid)
- **cosine taper 5–10 ms** at each boundary
- event table: `start`, `stop`, `P(motion)`, judgement, bands affected, routing
  decision, provenance
- **mask provenance**: the full `ModelSpec` from task 12A (mode, animal, version,
  corpus hash, calibrator), plus thresholds, reference values and code commit.
  Masks get regenerated as the model improves and every downstream analysis must
  know which one it used. **A mask whose provenance does not name a model is
  invalid** — assert this on write, not on read.
- **recording-level gate**: if retention falls below a threshold, flag the whole
  recording rather than silently emitting a heavily-masked one

### QC report per recording
Candidate count, blanking fraction per band, retention, R-peak gap fraction, best
HR channel, tripole weights `(a,b)`, sustained-event review queue, low-confidence
velocity windows, rescue rate per channel.

### Drift monitoring across weeks
Log `(a, b)`, peri-R template amplitude, and the noise floor per session. Three
numbers that say when an electrode is degrading — free once the pipeline computes
them, and worthless if not persisted. Write them to a per-animal longitudinal table.

### Tests
- no exact-zero runs in any emitted array (invariant 1, asserted again here)
- masks are **not** merged across consumers (invariant 2): assert the `spikes` and
  `slow_wave` masks differ on a synthetic where only one tolerance is crossed
- velocity mask is the intersection of `V1` and `V3` validity — the one allowed
  merge
- provenance round-trips and is non-empty
- the retention gate fires on a deliberately over-masked recording

### Acceptance
MATLAB loads the emitted file and `step1_bandpass.m` produces the expected valid
sample count. Retention reported per band per channel.
<!-- /TASK -->

<!-- TASK:16 slug=ui-recall-audit deps=07 gate=no -->
## Task 16 — Labelling UI changes

**Module:** `detector-pyqt`
**Depends on:** 07

### Reuse
`signal_viewer.py`, `overview_strip.py`, `region_table.py`, `predictions_panel.py`,
`review_panel.py`, `queue_panel.py`, 20-deep undo, keyboard interval stepping.

### Change 1 — candidate adjudication becomes the primary mode
`review_panel.py` already does candidate review (context plot, SHAP, 1/2/3 scoring,
Shift+drag to widen). Promote it from a secondary panel to the main interaction,
replacing free interval marking.

### BLOCKER FOUND 2026-09-24: no `meta.json` exists for new-cohort data

Audit mode works end to end on `gems_j_t01_ms3_bl_230315_sig.mat` — loads in 4 s,
54 pairs reduced to 6 traces, BLIND clean, reveal populated, diagnosis correct
over 300 windows — but **the loader refuses to open it without geometry, and no
`meta.json` exists for any new-cohort recording.** The refusal is correct
behaviour, not a bug. It means **labelling cannot start until that file is
written**, which makes `meta.json` generation the critical path to the 09 gate,
ahead of anything else in task 16.

It belongs to 00A/03's store, not to 16.

#### Two silent errors the stand-in fixture made, and the rule that prevents both

1. **Channel order.** The fixture built cuffs `("L", "R")` and labelled column 0
   `LVN1`. The file's own `chanlabels` are `RVN1-3, LVN1-3, ANT1-3` — **R
   first**. Every left/right comparison would have been inverted, silently.
2. **Units.** The fixture declared `units="uV"`. Robust σ is 8.3×10⁻⁶ in file
   units, which as µV would be 8.3 **pV**. The file is in **volts**. A 10⁶ error,
   silently.

**The rule: channel order is read from the file's `chanlabels`, never declared
separately.** The labels are in the file and are authoritative; a `meta.json`
that restates them is a second source of truth that can disagree, and this one
did. `meta.json` declares only what the file does *not* state — units, geometry
(pitch, aperture), animal and session identity — and the loader **asserts its
declared order against `chanlabels`** and raises on mismatch.

**Units are VOLTS, confirmed by Andrea 2026-09-24, and they do not vary across
cohorts.** So `units` is **not a per-recording field**. A constant that never
changes, written into 967 files, is 967 opportunities to type `uV`; declared
once it is one. Put it in `protocol.yaml` beside the other cohort-level
constants, and let `meta.json` carry only what genuinely varies per recording.
A per-recording override may exist for the day a rig changes, but it must be
absent by default rather than restated.

**Units stay declared (invariant 14) but an implausible declaration must fail
loudly.** Declaration prevents inference; it does not prevent a typo. Add a
plausibility check at load: a declared unit that puts robust σ outside roughly
1–500 µV for a nerve or stomach contact raises, naming both the declared unit
and the implied σ. That catches a 10⁶ error without inferring anything — the
declaration still decides, it just has to survive contact with the data.

#### Adjudication of the `meta.json` implementation, 2026-09-24

The implementation was built before this section reached the Windows clone, from
the chat message rather than the spec. It landed on the same design anyway. What
follows ratifies it and records the four places it must change.

**Ratified as built.**

- `read_channel_labels()` **returning `None` for the old five-channel cohort is a
  real answer, not a failure.** Those files carry no `chanlabels`; `None` means
  "the file does not state its order", which is true, and the old cohort's order
  comes from the cohort declaration instead. A function that raised there would
  be reporting a defect in data that is merely older.
- `channels_from_labels()` **taking file order and raising on an unrecognised
  stem.** Raising is right: an unrecognised stem means either a new electrode
  type or a corrupt file, and both need a human. Guessing from position is how
  the `LVN1`/`RVN1` inversion would have survived.
- `assert_labels_match()` **wired into `load_recording`, case-insensitive.**
  Case is a formatting difference, not a geometry difference.
- `assert_plausible_units()` **on the median robust σ across contacts, band
  1–500 µV.** The median, not the mean — one saturated contact must not move the
  verdict. The band is only 500× wide, and the errors it exists to catch are
  10³ and 10⁶, so it has margin on both sides without being tight enough to fire
  on a genuinely quiet or genuinely noisy recording.
- **Scaling the two fixtures rather than exempting the check.** See invariant 25.
  `test_the_declared_units_are_applied[mV]` and `[V]` were asserting that the
  loader accepts a recording with a multi-volt noise floor; that was never what
  they were for.
- `write_new_cohort_meta.py` **dry-run by default.**

**Four required changes.**

1. **`units` moves to `protocol.yaml` and `units_provisional` is deleted.** Units
   are volts, confirmed, cohort-wide and unchanging — there is nothing
   provisional left to flag. A field whose only value is `true` in every file is
   not information, it is a note-to-self that outlives the doubt that produced
   it. (Invariant 24.)
2. **`config` and `channel_order_source` move to `protocol.yaml` too.** Same
   argument, same test: *could two recordings in this cohort legitimately
   differ here?* They cannot. Every new-cohort file has three independent ADCs
   per cuff, so `config: "independent"` is a property of the cohort; every
   new-cohort file carries `chanlabels`, so `channel_order_source: "file
   chanlabels"` is too. `config` is genuinely cohort-distinguishing — the old
   cohort is hardware-shorted tripolar — which is exactly why it belongs in the
   per-cohort file rather than in 967 copies.
3. **`rostral_end_known` is deleted; `rostral_end: None` carries it.** The schema
   requires the key to be present, so `None` unambiguously means unknown and the
   boolean is a second source of truth that can disagree with the first.
   (Invariant 24, corollary.)
4. **The dry run prints distinct value-tuples with counts, not 967 lines.** The
   whole point of a dry run over a near-uniform corpus is to surface the file
   that is *not* like the others; a count of 1 beside one tuple does that in a
   glance, and 967 near-identical lines hide it. Reconcile the count of files
   written against the count of recordings enumerated and state both — invariant
   23 exists because a number that did not add up was the only evidence of a
   silent failure.

**One test to add:** a recording whose numbers are µV-scale but whose declaration
says `V` must raise from `assert_plausible_units`, with the message naming both
the declared unit and the implied σ. A check nothing exercises is a check that
silently stops working. The two scaled fixtures prove the check tolerates correct
data; nothing yet proves it rejects wrong data.

**Post-run:** record the observed median robust σ across the new cohort in
`protocol.yaml` as a comment. If it clusters tightly the 1–500 µV band can be
tightened later on evidence rather than on my guess.

#### STOP: 27 store-path collisions, found 2026-09-24

The generator's own dry run printed this and it was not acted on:

```
recordings enumerated (*_sig.mat): 859
ready to write:                    837
distinct store paths:              810
COLLISIONS (stem repeats by date):  27
```

**837 files written into 810 distinct paths means 27 `meta.json` were
overwritten by a second recording's.** Each surviving file now describes one
recording while sitting in the store path of another, and `meta.json` carries
`animal`, `session` and `channels` — so 27 recordings are one load away from
being labelled with a different recording's identity and channel order. This is
the *same* failure the whole `meta.json` exercise exists to prevent: the
`LVN1`/`RVN1` inversion, except distributed across 27 files instead of a
fixture, and not detectable from the signal.

**Required, before any further writing:**

1. **The generator refuses to write when a store path is not unique.** Not a
   warning line in a summary — a raise, naming every colliding group. A path
   that two recordings both claim is not a path, it is a bug in the key.
2. **Enumerate the 27 and diagnose the key.** Stem-plus-date is evidently not
   unique. Print each colliding group with the full source paths, file sizes and
   mtimes. The two likely causes are a genuine same-day repeat session (in which
   case the key needs the acquisition time or the block index) and the same
   recording reachable by two Drive paths (in which case one is a conflict copy
   and the store is fine but the enumeration is double-counting). These need
   opposite fixes, so do not pick one — read the groups.
3. **Re-derive the key from something the acquisition system guarantees unique.**
   The TDT block directory name is the candidate; it already encodes date and
   time. A key assembled by us from parts we chose is a key we have to prove
   unique, and we just found out it is not.
4. **Then re-run and reconcile to zero.** `enumerated = written + skipped +
   collided`, every term printed, every skip reasoned.

**And three counts in this report do not reconcile.** 859 enumerated against 837
ready leaves 22 unexplained; the prose then says 837 rewritten and, later, 841
written this run. A four-file discrepancy inside one report about writing files
is exactly the signal invariant 23 is about — the 5-second bug announced itself
as 1195 against 1710 and nothing else. Do not rationalise the gap; find it.

#### RESOLVED 2026-09-25: the 27 are cause (b), and the re-key was the wrong instruction

The groups were read before the key was touched, and the answer reverses the
first report. **All 27 signal arrays are bit-identical**; only `createdAt` and
`srcBlock` differ, and the two `srcBlock` values are the same block name under
`09032026` and `09042026`. Eight million by nine float32 samples of biological
noise cannot agree bit-for-bit across two acquisitions. This is one recording
reachable by two Drive paths. **No `meta.json` was overwritten by a *different*
recording's** — each pair wrote identical content — so the store was never
corrupt, only double-counted.

**My step 3 was wrong and should not have been written.** I told it to re-key
off the TDT block *before* the groups had been read, having just objected in the
same message to acting on an unread diagnosis. The re-key would have made the
key finer, which for duplicates is the wrong direction: it would have produced
27 duplicate store entries instead of collapsing them, and left the actual
duplication untouched. The refusal, with the evidence attached, was correct, and
declining a spec instruction on evidence is the behaviour this document wants.
**Ratified: no re-key.** The guard is refusal-plus-verified-dedup.

**The comparison method is the durable finding.** `_sig.mat` keeps `createdAt`
in the header and `srcBlock` in trailing variables — exactly the regions a
head-and-tail hash samples — so the intuitive fast check reports "two different
recordings" with total confidence on two conversions of one block. The second
attempt, a fixed mid-file range, failed differently: two groups differ by three
bytes of metadata string, which shifts every later offset, so the same byte range
lands in different places. **Only the decoded `signal` array answers the
question.** Identical byte size is a reason for suspicion, not reassurance. This
is recorded at the top of `verify_duplicates.py` because the wrong method is the
one that comes to mind first.

#### CLOSED 2026-09-25: b1, and no recording is missing

| check | result |
|---|---|
| shared stems under both dates | 27 |
| identical TDT block name (acquisition timestamp) | 27/27 |
| identical `.tsq` and `.tev` sizes | 27/27 |
| blocks only under `09042026` | 36 |
| …converted to `_sig.mat` | 36/36 |
| …present in the store with `meta.json` | 36/36 |

**The raw TDT blocks themselves are duplicated.** The converter did not read
`09032026` twice; it read two copies of one block. `09032026` holds 27 blocks
and is *wholly contained* in `09042026`'s 63. Nothing is missing, nothing is
corrupt, and this is Andrea's Drive to tidy, not the build's.

**Do not delete anything.** The duplicate folder costs space and nothing else,
and the enumeration handles it. Deleting raw acquisition data to tidy a
bookkeeping artefact is a bad trade in the wrong direction.

#### The block name carries no date, and the folder date is now known-unreliable

Every block under both folders carries the same prefix,
`ME_STIM_Andrea-260824-155430` — the **tank** creation stamp (2026-08-24), not a
per-block one. So the block directory is `<tank>_<stem>`, the stem's trailing
digits are a **time of day** (`gems_h_pre01_160301` → 16:03:01), and **nothing
in the name says which day.** The only thing that did say was the containing
folder, and we have just proved a folder date can be wrong: 27 blocks sit under
a date that is not theirs.

Two consequences.

**1. Acquisition datetime comes from the `.tsq` header, not from the folder.**
The block's own start timestamp is in there — it is what the b1/b2 check just
read — so the authoritative value is available for the cost of a header read.
Add `acquired_at` to `meta.json` from the `.tsq`, and **audit the existing 837
for any `session` value derived from a folder name.** A field taken from the
filesystem when the instrument recorded it itself is a second source of truth,
and this one is already known to disagree. (Invariant 30.)

**2. The stem key is safe only because the guard raises.** Name-plus-time-of-day
with no date means two genuine recordings — same animal, same condition label,
same second on different days — would collide, and across a corpus still growing
past 886 that is a question of when, not whether. The dedup guard catches it:
such a pair fails the identical-signal test and raises. **That guard must never
be downgraded to a warning or a skip.** It is not a migration convenience for
the 27; it is the thing that makes a dateless key tolerable, and `acquired_at`
from the `.tsq` is what will eventually let the key stop being dateless.

And a third reason the re-key was wrong, better than the two already recorded:
the block directory name is `<tank>_<stem>` with a tank stamp shared by every
block, so it is **stem-equivalent for uniqueness**. Re-keying to it would have
bought precisely nothing.

#### RE-KEY NOW, on `acquired_at` — and why this one is right where mine was wrong

`acquired_at` exists for all 837, from the `.tsq`. That changes the key decision,
and the reason is the whole point of invariant 30.

**My withdrawn re-key would have used the folder date, which *differs* between
the two copies — it would have split 27 duplicates into 27 spurious store
entries.** A key on `acquired_at` is *identical* across a duplicate pair, because
the instrument wrote it once and the copy carries it. So the same idea, sourced
from the instrument instead of the filesystem, collapses the duplicates
correctly **and** separates two genuine recordings of the same animal and
condition started at the same clock second on different days. Opposite outcomes
from the same-shaped change, decided entirely by where the value came from.

**Do it now, before labelling starts.** This is the cheapest this will ever be:

- nothing references a store path yet — no labels exist, and
  `duplicates.json` / `new_cohort_meta_plan.json` both regenerate;
- the generator is already idempotent and verified to reconcile to zero, so the
  re-key is one run of a thing that works;
- after labelling begins, changing the key means migrating labels, and the
  dateless key becomes permanent by inertia.

The alternative is keeping the guard forever and accepting that it will raise at
some arbitrary future moment — which is safe, but the moment it picks will be in
the middle of labelling, which is the critical path.

**Before running it:** enumerate every reference to a store path anywhere in the
repo and in the Drive store, and state the list. The claim "nothing references
it yet" is the kind of negative that invariant 20 exists about — establish it,
do not assume it. The dedup guard stays exactly as it is afterwards; it is
cheap, and a key being better is not a reason to remove the thing that catches
the case where it is not.

#### Store the zone, not just UTC — November will break a hardcoded offset

UTC with the zone named was the right call, and the example is the argument:
`gems_j_t01_ms3_bl_230315` is 23:03 **EDT on the 15th**, which is 03:03 **UTC on
the 16th**. The stem's trailing digits are **local** time; `acquired_at` is UTC;
the dates disagree by one for every recording after 20:00 EDT.

Two things follow.

1. **Store the IANA zone (`America/New_York`) alongside the UTC timestamp**, not
   a fixed offset. The corpus is Aug–Sep now, but recovery recordings will run
   past the November DST change, and a hardcoded −4 silently becomes wrong for
   every recording after it — while continuing to produce plausible times. With
   the zone stored, local date is derivable forever; with an offset, it is
   derivable until it isn't.
2. **Any grouping by "day" uses the LOCAL date.** Session grouping, baseline /
   recovery pairing, the per-recording motility state from task 08, and the
   LORO splits all mean the day Andrea ran the animal, not the UTC day.
   Grouping on the UTC date pushes every evening recording into the next day's
   group — a quarter of the corpus landing in the wrong session, produced
   silently, and indistinguishable in the output from a real scheduling
   difference. State the convention at the point where the grouping is computed.

#### The `mmc_burst` twin: declare the call-to-fiducials map

The grep found the second instance and it was worse than the first.
`extract_mmc` produces both `mmc` and `mmc_burst`, gated on `want.mmc` alone,
and worked **only** because a string special-case in the mask loop forced
`want.mmc` true whenever `mmc_burst` was masked in. Since `mmc` is
`criterion_degenerate`, real masks contain `mmc_burst` and **never** `mmc` — so
that special-case was load-bearing for every mmc_burst point in the sweep, and
deleting it as dead tidy-up would have produced nothing for all of them.

**A special case that patches up a flag is evidence the flags do not model the
call graph.** Two instances found in one grep is a pattern, and the structural
fix prevents the third: **declare the call→fiducials map once**, and derive both
the gate and the recording from it —

```
HR_BR_HRVAnalysis_new -> {hrv, breathing}
extract_mmc           -> {mmc, mmc_burst}
```

A call runs iff `mask ∩ outputs(call) ≠ ∅`, and records exactly
`mask ∩ outputs(call)`. Hand-written boolean gates are then impossible to get
wrong, and a fiducial added to an existing call later cannot be forgotten —
which is the failure mode that produced both of these.

**One regression check before moving on.** The refactor is structural and the
old special-case meant mmc_burst results were *correct*, so the 1710 rows should
stand — but "should" is not "do". Re-run five mmc_burst points under the new
code and compare against the stored rows. If they match, say so and keep the
sweep. If they do not, the refactor changed behaviour and the pass is suspect.

#### The serial control: paired, stratified, and with the decision rule fixed in advance

The control exists to answer one question — *is the 2× miss contention or a bad
per-point model?* — and it only answers it if it is designed so that the two
answers look different. Three requirements:

1. **Paired, not fresh.** Re-run **the same point indices** whose `wall_s` is
   already stored from the 8-worker pass. A serial run of different points
   compares two distributions and answers nothing; the same points compare two
   conditions.
2. **Stratified across the duration axis**, ~10 points, because per-point cost
   scales with duration and a mean over a lopsided sample is a mean over the
   wrong thing.
3. **On a quiet machine.** If labelling or anything else is running on that box,
   the control is measuring a third condition. State what else was running.

Then `ratio = parallel_per_point_wall / serial_per_point_wall`, and the action is
**decided now**, so the number does not get re-argued once it has a value:

| ratio | reading | action |
|---|---|---|
| ≥ 1.5 | contention dominates — each point is inflated by sharing one I/O path | sweep workers at 2 / 4 / 8 on one fixed small set, take the knee; do **not** assume more is faster |
| ≤ 1.2 | no meaningful contention — the per-point estimate is simply wrong | recalibrate the cost model's coefficients from the measured serial times and retire the blanket 2× |
| 1.2–1.5 | both, in some mix | recalibrate **and** test 4 workers |

Report the serial per-point mean, the ratio, and the implied parallel efficiency
against the 53% already measured. The blanket empirical 2× stays on every
projection until this lands, and is replaced by the recalibrated model rather
than removed.

#### The serial control landed, the decision rule was wrong, and the remedy is upstream of the worker count

**The rule I wrote was defective and the spec is the thing to blame.** It asked
for stratification across the **duration** axis and then reduced the answer to a
single ratio. Duration was not the axis carrying the variance — consumer was —
and the duration-stratified sample drew zero `slow_wave` points and returned a
tidy aggregate **1.28**, landing in a band whose prescribed action ("recalibrate
and test 4 workers") was wrong for every consumer individually. The measurement
was executed correctly against a specification that measured the wrong thing.

| consumer | serial | parallel | ratio |
|---|---|---|---|
| `hrv` / `breathing` | 6.0 s | 6.2–7.1 s | 1.02–1.21 |
| `T_hardware` | 12.0 s | 12.8–14.9 s | 1.06–1.23 |
| `mmc_burst` | 8.5 s | 11.4–13.4 s | 1.29–1.57 |
| **`slow_wave` (180 s)** | **15.9 s** | **98.9 s** | **6.23** |
| **`slow_wave` (420 s)** | **32.2 s** | **237.2 s** | **7.36** |

**And the per-point model was never the problem.** `slow_wave = 15.2 s` in the
model against 15.9 s measured serially is a good coefficient. The blanket 2× was
one consumer behaving differently under parallelism, averaged across a
population where nothing else does.

#### Do NOT run the 2/4/8 sweep yet — three cheap measurements come first

A worker sweep costs hours and is downstream of a question that costs minutes.
A 6–7× per-point inflation at 8 workers on a **32-core** machine has three
distinct explanations with three opposite remedies, and the sweep cannot tell
them apart:

1. **MATLAB is implicitly multithreading and the machine is genuinely
   saturated.** `filtfilt` and the BLAS calls underneath it use every core by
   default. If one `slow_wave` run takes 15.9 s across ~32 threads, that is
   ~500 core-seconds of real work; eight of them is ~4000 core-seconds, which on
   32 cores is ~125 s per point. **The observed 98.9 s is consistent with
   this.** If so, the 6.6 h is near the machine's capacity, 8 workers is already
   right, nothing is being wasted, and the fix is not scheduling at all.
2. **Oversubscription.** 8 processes × 32 threads = 256 threads thrashing 32
   cores, where the same work would finish faster with each worker pinned to one
   thread. Remedy: `maxNumCompThreads(1)` per worker — and then possibly *more*
   workers, not fewer.
3. **I/O bound.** Cores idle, one Drive path saturated. Remedy: fewer workers,
   or stage segments to local disk first.

Measure, in this order, during an 8-worker `slow_wave`-heavy burst:

```
maxNumCompThreads inside a worker        -> is it 32 or 1?
total CPU utilisation across 32 cores    -> saturated or idle?
disk read throughput                     -> pegged or quiet?
```

Saturated + multithreaded is (1). Idle cores with high context-switching is (2).
Idle cores with a pegged disk is (3). **Report all three numbers**, then act.
The 2/4/8 sweep is only worth running under (2) or (3).

#### The likely real win is algorithmic, not scheduling

`slowWaveAnalysis_new` low-passes for a gastric slow wave at **~3 cpm = 0.05 Hz**
in data sampled at **24414 Hz**. Even the burst content of interest is well
under 50 Hz. Filtering at the full rate to extract a 0.05 Hz rhythm does on the
order of a hundred times more arithmetic than the signal requires.

**Check whether it decimates before filtering.** Anti-alias decimate to ~200 Hz
first — a factor of ~120 in filter work — then run the existing low-pass on the
decimated series. If `slow_wave` drops from 15.9 s to a fraction of a second,
the contention question dissolves rather than being scheduled around, and it
takes `slow_wave` off the critical path for every future pass as well.

Do this check before the worker sweep too. An algorithmic factor of 100 makes a
scheduling factor of 2 irrelevant, and the order matters: tuning the worker
count around a filter that should not cost this much bakes the waste into the
plan. If it already decimates, say so and the question is closed.

#### Recalibration, once the above is known

Do not recalibrate the model to the *parallel* per-point times. Those are a
property of a worker count, not of the work, and freezing them into the model
makes every future projection wrong the moment the worker count changes. Model
**serial per-point cost** — which is already close to right — and apply a
separate, named contention factor per consumer, since the measurement just
showed contention is per-consumer and not global. The blanket 2× is retired by
that, not by an aggregate replacement.

#### The fourth case: under-subscription, and the machine is three-quarters idle

None of my three hypotheses was right, and the reason is that I never questioned
the **baseline's** thread budget. MATLAB gives each pool worker **one** thread;
the serial control ran in the client with **24**. The 6–7× "contention" was the
multithreading speedup the workers never had. Thread-matched, every
parallel/serial ratio is **0.91–1.24**. There is no contention to relieve and
the 2/4/8 sweep was correctly not run. (Invariant 36.)

**The finding nobody has acted on is in the same table: 26% CPU, ~8 cores busy,
24 idle, run queue 0, disk 99% idle.** Eight single-threaded workers on a
32-core machine leave three quarters of it unused, and the fix is to add
workers, not threads. The measured numbers decide this:

- MATLAB's intra-call threading gives `slow_wave` **5.95×** on 24 threads
  (89.9 s → 15.1 s) — **25% efficiency**.
- Running independent points in separate processes gives ratios of 0.91–1.24 —
  **~100% efficiency**.

So a core spent on a process is worth about four spent inside one call. **Raise
the worker count toward the core count**, single-threaded, rather than giving
workers more threads. Set it by memory, not by guesswork: measure a worker's
peak RSS, then `workers = min(cores − 4, floor(0.8 × available_RAM / RSS))`, and
report both numbers. Leave the headroom — the machine also has to stay usable
for labelling.

#### Decimation: NO for T. The arithmetic says it is not needed.

Decimating changes `slowWaveAnalysis_new`, and Step 9's premise is that **a
tolerance is a property of the consumer as it is**. Measuring a decimated
consumer yields a tolerance for a pipeline Andrea is not running. That alone
would make it a bad trade — and the cost table says the trade buys nothing:

| configuration | extension (1510 points) |
|---|---|
| as measured, 8 workers | 5.0 h |
| duplicate `smoothdata` removed (−47% of `slow_wave`) | ~3.0 h |
| …and workers raised 8 → 24 | **~1.0 h** |
| decimated, 8 workers (the proposal) | 0.9 h |

**The two changes that do not touch the consumer get to the same place as the
one that does.** Take those and leave the consumer alone.

**Do take the duplicate `smoothdata` removal** — line 186 recomputing line 146
is not a change to the consumer, it is the removal of a recomputation, and it is
~47% of `slow_wave`'s single-threaded cost. Two conditions: **verify
bit-identity empirically** on at least three real segments across conditions
before editing anything (an argument that two calls must be identical is not a
measurement that they are — this document has a long record of that distinction
mattering), and **Andrea decides**, because `processing_new` is her pipeline.
**Andrea approved the removal, 2026-09-26: "make it once only."** The
bit-identity verification is still required before the edit lands — her approval
is of the intent, and the three-segment check is what establishes the edit
matches it. Record the check's result beside the change.

**Decimation itself is a good idea in the wrong place.** Low-passing at 24414 Hz
for a 0.05 Hz rhythm is genuinely wasteful and would pay back on every future
run of her pipeline, not just on T. **File it as a task 08 row** to be validated
on its merits: decimated versus full-rate slow-wave output on several
recordings, agreement within a stated tolerance, then adopt. It does not belong
inside T, where it would silently redefine the thing being measured.

#### The censoring result is a design finding, not a bracket that needs widening

This is the most important thing in the report and the recommendation
under-weights it. Across *every* kind — including the deterministic `step`,
`clip` and `drift` — **71–75% of seed-rows are censored**, curves are monotone,
nulls are zero, a replicate seed agrees with the locate seed only **58%** of the
time, and **62 of 155 rows are censored on both sides**, meaning the placement
spread exceeds the 2× the bracket spans.

**Where the artifact lands dominates the tolerance by more than the quantity
being estimated.** That is not a bracketing problem. A ±2-step extension chases
the tails of a distribution whose width is the actual result, and on these
numbers it will still leave many rows censored — 1510 points spent to learn
that.

**Before the extension, run a seed-depth diagnostic.** Take three or four
(consumer, kind) cells — include one deterministic kind and `tribo` — and run
**20 seeds** instead of 5 across the *existing* amplitude grid. A few dozen
points, and it answers the question the extension cannot: what is the shape and
width of the placement distribution? Then spend on the axis that carries the
variance, which is invariant 34 applied to the thing invariant 34 was learned
on. If placement dominates as the evidence suggests, seeds buy more than steps,
and 5 seeds was never enough to characterise a spread this wide.

**And the tolerance itself has to change shape.** If sensitivity varies by more
than 2× with placement, a scalar tolerance is not a property of the consumer —
it is a property of one arbitrary placement. `consumer_tolerances.json` should
carry a **conservative quantile across placements** (the most-sensitive end,
stated as such) **plus the observed spread**, so the 09 gate errs toward
rejecting blanking that would fail at an unlucky placement rather than passing
on a lucky one. The spread is a result in its own right and Andrea should see
it: it says her consumers' sensitivity depends on where an artifact falls about
as much as on how large it is.

#### Invariant 35's example was wrong — the mechanism is the smoothing window

Correction accepted, and it makes the lesson stronger. The cost is not in
`filtfilt` (0.39 s). It is `smoothdata` with a 5 s Gaussian window = 122,070
samples, run twice, whose cost grows as **span × window**, and both scale with
fs — so the waste goes as **fs², not fs**. Decimating 100× is ~10⁴ less
arithmetic, not 10². The invariant's text is updated; the example named the
wrong mechanism and the corrected one is a better argument for the same rule.

#### Re-key applied 2026-09-26, and what the block names turned out to be

**Applied and idempotent.** 921 enumerated = 870 written + 27 deduplicated + 24
skipped, residual 0; 835 old stem-keyed directories retired after verification,
2 kept for held recordings; second `--apply` wrote 0. Every key derivation —
`recording.py`, `scan.py` ×2, the generator — now routes through one
`session_key()`, and the key format lives beside `SESSION_STAMP` in
`tdt_block.py` with a test tying them together (invariant 33). Two latent bugs
died with the old key: `_notched` files never found their `_sig` metadata, and
`scan.py`'s `core_of` and the loader's `path.stem` produced different keys for
old-cohort files.

**37 block folders were renamed after acquisition**, so the folder name and the
block name the instrument wrote disagree (`App_ms10_1_bl_001835` on disk,
`t01_ms1_bl_001835` inside the block). The renames are human annotations —
quality flags, condition changes — and **two of them name a different animal
from the instrument.** The key now uses the instrument's block name, which is
stable however often a folder is renamed; renames are recorded in the plan, not
refused; the two animal mismatches are **held** until Andrea says which is right.

**Refinement to invariant 30, forced by this.** A block name is *typed by a
person* into the instrument before recording; it is not measured. So the
instrument is authoritative for what it **measures** — start time, sample rate,
the channel labels wired into the rig — and a name typed into it is merely the
*earliest* human annotation, which a later rename may be correcting. For
**identity** (the key) the instrument name is right because it never moves. For
**meaning** (condition, animal, quality) the two annotations are candidates and
a disagreement goes to Andrea, because a rename made to mark a condition change
is exactly the kind of correction the original typed name cannot know about.

- **Preserve every rename's text as metadata** (`folder_name` beside
  `block_name`). A quality flag Andrea typed into a folder name is a label, and
  discarding it because the key does not need it destroys it.
- **Condition authority is an open question for Andrea**: where the folder and
  the block disagree about condition, which one is the correction? Until
  answered, recordings whose two names disagree about *condition* are handled
  like the animal mismatches — held from anything that groups by condition —
  rather than silently taking either.

#### Andrea's rulings on the renames, 2026-09-26

1. **The two `gems_b_t01_2_1_*` blocks are animal A.** The folder rename was the
   correction. Release them from hold as animal A.
2. **Condition renames: the folder is right** (`ms2`→`ms3`, `1_2`→`1_1`).
3. **Exclude every `_BAD` and `_INCOMPLETE` recording, and its paired baseline or
   stim/recovery file if one exists.** Out of labelling, training, T hosts and
   any corpus. Excluded, not deleted: the flag and the reason are recorded in
   the store, and nothing is removed from the Drive.

**So the general rule is settled:** the instrument's block name is the
**identity** (the key, which never moves); the **folder name is authoritative
for meaning** — animal, condition, quality — because Andrea renames folders to
correct them. Apply the folder's meaning automatically, and keep logging every
block/folder disagreement in the plan so a future rename is visible rather than
silently absorbed.

**Pairing needs a rule, stated and tested, not improvised.** A pair is the
baseline and the stim/recovery recording from the same session: same animal,
same trial token, same condition token, differing only in the `bl` / `sr` infix
(e.g. `gems_j_t01_ms3_bl_230315` and `gems_j_t01_ms3_sr_231323`). Derive it from
the **corrected** folder names (ruling 2). Report, do not guess, every excluded
recording whose partner is **ambiguous** (more than one candidate) or
**missing** (none). An exclusion list is only as good as the pairing that
produced it.

**Pairing resolved by acquisition time — Andrea, 2026-09-26.** Baseline is
recorded **~10 min before** stim/recovery. So acquisition-time adjacency is the
pairing rule, and it is a **verification for every pair, not only a tiebreaker**:

- Declare it as a cohort constant in `protocol.yaml` (invariant 24):
  `pair_gap_min: 10`, baseline first, tolerance **±3 min**. The 11 ambiguous
  sessions all resolve at 10.1–10.2 min, well inside it.
- **A name-matched pair outside the window is not a pair.** Andrea: a 105.3 min
  gap is "likely wrong". Treat it as unpaired and list it for her. The 14.7 min
  pair also falls outside ±3 and is listed the same way, not accepted.
- The lesson for pairing: two recordings sharing name tokens are a *candidate*
  pair; the acquisition times are what confirm it. The same shape as invariant
  30 — the name is a human annotation, the timestamp is measured.

**Pairing, second round — Andrea, 2026-09-26.**

- **The 14.7 min pair is real** (`gems_a_t01_ms2_bl_162459_incomplete` ↔
  `…sr_163939`). The ±3 min tolerance was my guess and it was too tight. Replace
  the symmetric tolerance with explicit bounds: **baseline precedes stim/recovery
  by 5–20 min** (`pair_gap_min_lo: 5`, `pair_gap_min_hi: 20`). The 105.3 min
  pair stays unpaired. The ambiguity check is what makes a wider window safe:
  two candidates inside it still refuse. Report every pair the widened window
  adds beyond the 14.7 min one.
- **`stim_recovery` and `sr` are the same condition** — older files use the long
  name, newer ones the short. Each holds **2 min stim + 20 min recovery in one
  file**. So the scan's condition inference and the pairing's name-token match
  must treat the two tokens as equivalent. **Re-check the 3 "no name match"
  baselines with that equivalence** — a partner named `stim_recovery` would
  have been invisible to a matcher looking for `sr`, and that is a likelier
  cause of "missing" than a recording that was never made.

**Check whether any T host is among the excluded.** If one is, its tolerance
rows were measured on a recording Andrea considers bad, and they must be
re-measured on an eligible host or marked.

#### The `smoothdata` duplicate: verified, now apply it

Bit-identical on all three segments (baseline 599 s with 3 blanked spans, stim
120 s, recovery 1205 s with 155 blanked spans and 4.2% NaN) at 1 and 24 threads:
all 17 output fields and all peak locations. Single-threaded saving 23–54% per
call. **Both of Andrea's conditions are met — her approval and the check — so
apply the patch to her working copy.** Leaving a verified, approved patch
unapplied is not caution, it is a second decision nobody asked for. Keep the
patch file and put the check's result in a comment at the edit.

**Found alongside: `run_continuous.m` calls `slowWaveAnalysis_new` with 12
arguments; the function takes 11 with no `varargin`.** That call errors on
reach. **Andrea, 2026-09-26: the function is newer — update the caller.** One
check first: read the 12th (mode) argument at all three call sites (lines 127,
141, 153). If all three pass the same mode and it matches what the function now
does unconditionally, drop it. **If they pass different modes**, the function
lost behaviour the caller relied on, and dropping the argument collapses three
behaviours into one silently — report that to Andrea instead of editing.
**Result:** they did differ — lines 141 and 153 pass `'single'`; line 127 passes
`'stim_rec'` with a struct holding both epochs, relying on a mode removed before
the repo's first commit. **Andrea, 2026-09-26:** `stim_rec` is just the old name
for an `sr` file, which holds 2 min stim then 20 min recovery. Nothing was
shared across the epochs. So line 127 becomes **two `'single'` calls, one per
epoch**, from the struct it already builds; drop the mode argument at 141 and
153; and correct the function header (lines 24–26), which still documents the
removed mode.

#### Workers: memory binds at ~10, and MATLAB had been capping at 8 all along

Peak RSS is **4.2–4.7 GB per worker** on 54.7 GB available, so memory — not
cores — caps this machine at ~10. My ~24-worker projection ignored memory and
was wrong; invariant 37's text (set it from RSS) was right and my table
contradicted it. At 10 workers the extension projects to ~2.4 h, not ~1.0 h.

And the reason every T run so far used exactly 8: **MATLAB's `Processes`
profile caps `NumWorkers` at 8 by default.** Raising it on an in-session cluster
object rather than editing the persistent profile was right — that profile is
Andrea's and governs all her other MATLAB work.

**If throughput ever matters more, the lever is memory per worker, not cores.**
A 420 s segment of 9 channels in double is ~740 MB, so 4.7 GB is ~6 copies
resident. Loading only the channels a call needs (slow_wave reads the stomach
contacts, 3 of 9), or single precision where the consumer tolerates it, would
roughly double or triple the worker count. Not now — noted for when a pass is
long enough to justify it.

#### Seed-depth diagnostic: two conditions on reading it

Bisecting each seed's crossing on the existing grid (~5 evaluations plus 2
monotonicity checks, reusing seeds 1–5) is a better design than the 20-seed full
grid I asked for, at a fifth of the cost.

1. **Bisection is only valid on a monotone curve**, and `breathing` is
   non-monotone in 12% of cases. On a non-monotone seed, bisection finds *a*
   crossing, not the lowest one — and the tolerance is the **sensitive-end**
   quantile, so a missed lower crossing is **anti-conservative**. Any seed that
   fails its monotonicity check is reported as non-monotone and its bisection
   result is **not** used as a crossing; treat it as unresolved, or grid it.
2. **All four cells are on one host (`host1_JEL`).** That is fine for learning
   the shape of the placement distribution; it is not the distribution for the
   cohort. State it in the result, and do not size the full extension from one
   recording's spread without saying that is what was done.

The redesigned `final()` — each seed an interval, conservative sensitive-end
quantile plus spread — is the right shape. Only 7 rows resolve a spread wider
than one √2 step, which is what a 3-point bracket can show; the width is hidden
in the 110 rows reporting a bound at the sensitive end, and the diagnostic is
what will measure it.

#### Seed-depth results: placement moves the threshold 4–16×

434 points in 47 min (costed ~500 / ~40 — the cost model now holds). No seed
failed its monotonicity confirmation, so every bisection result stands.

| cell | kind | resolved | spread across placements |
|---|---|---|---|
| step 0.5 s, slow_wave | deterministic | 20 / 20 | 16× (0.35–5.7σ) |
| step 0.05 s, T_hardware | deterministic | 14 / 20; 6 below the 0.5σ floor | ≥4× |
| clip 0.5 s, mmc_burst | deterministic | 18 / 20 | 8× |
| tribo 0.5 s, hrv | stochastic | 20 / 20 | 8× |

Deterministic kinds spread as widely as tribo, so the variance is in **where**
the artifact lands, not in its randomness. The order-statistics point is right
and must travel with every number: the minimum of 5 placements estimates roughly
the **17th** percentile and reaches the true 10th only 41% of the time
(1 − 0.9⁵); 22 placements are needed for 90%.

#### Test the phase hypothesis BEFORE buying more placements — it is free

The proposed explanation — that the threshold depends on where the artifact
falls relative to physiological events (R-peaks for `hrv`, slow-wave peaks for
`slow_wave`, bursts for `mmc_burst`, spikes for `T_hardware`) — is the most
important open question in T, and the data to test it already exists. Test it
before choosing between (a), (b) and (c), because it changes what (b) and (c)
should be:

- **If phase explains the spread**, the tolerance is a *function* of phase, the
  sensitive end is the worst phase, and it is measurable directly with a few
  placements aimed at that phase. 20 *random* placements per row would be
  buying precision on the wrong axis — invariant 34, for the third time.
- **If it does not**, the spread is irreducible placement variance and (c) is
  the honest spend.

**Check the bunching against the grid first.** Bisection on a discrete grid can
only return grid values, so "8 of 14 at exactly 2.0σ" means 8 of 14 thresholds
fell in the grid interval ending at 2.0σ. Whether that is a real concentration
depends on the interval widths; compare against what a smooth distribution
would put in each bin before reading it as structure.

The test: for each placement, the artifact's position relative to the nearest
relevant physiological event (as a phase, or as a distance in the event's own
time units), against its threshold. Report per cell, with the correlation and a
permutation null.

#### Decision on spend

- **(a) now** — extend only the seeds censored on the sensitive side (550 points,
  ~39 min on 10 workers), after extending the low amplitude grid for nerve rows
  so the 6 `T_hardware` placements below 0.5σ can resolve. Every resulting
  number labelled **"≈17th percentile, 5 placements"**.
- **The phase test in parallel** — no new evaluations.
- **(b) is declined.** 15–25 h to buy precision on every row, most of which the
  gate never reads, and possibly on the wrong axis.
- **(c) waits for both the phase test and task 14.** Task 14 (routing) decides
  which consumer rows the 09 gate reads; the phase test decides whether those
  rows need random placements or phase-aimed ones.

**One cheap efficiency:** each round spent ~200 s reloading all three host spans
from the Drive, including the 420 s span these cells never used. Load only the
spans a round's points need.

**`slow_wave` contention rose 1.10 → 1.26 from 8 to 10 workers.** First evidence
that adding processes costs something. Record the per-worker-count factor in the
cost model rather than a single number, and re-measure if the count changes
again.

#### CORRECTION 2026-09-26: the phase effects are large, the sample is small

With the numbers in hand my "read as null" was wrong in its wording. Per cell:
`slow_wave` r = 0.59 (p = 0.027), `mmc_burst` |ρ| = 0.52 (p = 0.021),
`T_hardware` |ρ| = 0.40 (p = 0.079), `hrv` r = 0.24 (p = 0.61). Correlations of
0.4–0.6 are **large** effects; they miss Holm correction because n = 20 on one
host, not because they are small. Fisher's combination of the four gives
p ≈ 0.007 — valid only if the cells' placements were drawn independently, which
must be checked before quoting it (shared seeds would correlate them).

**So phase plausibly matters for three of four consumers, and (c) must be
designed to settle it** rather than assume either answer: **stratify (c)'s
random placements across phase bins** of each consumer's event. That yields an
unbiased quantile (weight the bins by their true occupancy) *and* a powered
phase test from the same points, with the same pre-declared directions. Pure
random placement answers only the first question; pure phase-aimed placement
only the second.

#### Phase test: null after correction, directions all as predicted (superseded wording)

No cell survives Holm correction; the bunching check is closest at p = 0.057
(`T_hardware`). All four pre-declared directions came out as predicted — under a
sign test that alone is p = 1/16 ≈ 0.06. Read it as **no evidence of a large
phase effect on one host with 20 placements**, not as evidence of no effect.

Consequences:

- **Treat the spread as placement variance for now.** The tolerance stays a
  conservative quantile plus spread; do not model it as a function of phase.
- **(c) is now the honest spend** once task 14 says which rows the gate reads.
- **Re-run the phase test on (c)'s data at no extra cost.** (c) produces ~20
  placements per gate row across hosts — several times this test's sample — so
  a real modest effect would show there if it exists. Pre-declare the same
  directions.

#### The (a) extension landed — four rulings

589 points, none failed, 0.92× the model. Of 275 sensitive-side censored seeds,
141 are now measured, 118 are still extendable, and 19 sit at the grid's absolute
limit (real "at least this sensitive" results).

1. **`breathing` is non-monotone in 19% of seed curves.** A larger artifact
   sometimes *un*-breaks it, which is physically backwards and points at the
   criterion rather than the consumer. **Check what `breathing_changed`
   compares.** If it compares a count or a rate, an artifact that adds one
   spurious breath and deletes one real breath nets to "unchanged", and the
   criterion flips as amplitude rises. That is the `match_s = 0` shape from
   `mmc` again. If so, compare **matched fiducials** within a stated time
   tolerance, as `mmc` now does, and re-run the breathing rows. A criterion that
   can miss damage is anti-conservative, which is the wrong direction for a
   tolerance.
2. **For any non-monotone seed, the sensitive end is the lowest amplitude
   observed to break it**, not a bisection crossing. State it so.
3. **The 109 locate-only rows get `gate_eligible: false` as a field**, not a
   sentence. One centred placement on a quantity that moves 4–16× with placement
   is not a tolerance. A rule the gate must remember is a rule the gate will
   eventually forget (invariant 27's reasoning); a field it reads cannot be
   forgotten. Rows become eligible only when replicated.
4. **The 118 extendable seeds wait for task 14**, like (c): refining bounds on
   rows the gate may never read is spend without a reader.

**`parfor` chunking cost 46% (4,254 s against an ideal 2,910 s).** Worth fixing
before (c). `parforOptions(pool, 'RangePartitionMethod', 'fixed',
'SubrangeSize', 1)` makes each iteration its own chunk, so the longest-first
ordering survives; `parfeval` with a queue is the alternative. Measure it on a
small manifest before relying on it.

#### Where (c) runs: new-cohort hosts, if the consumers can run there

Every tolerance so far is measured on **old-cohort** hosts, and this section
already records that old-cohort tolerances are **provisional** until re-derived
on new-cohort data — different noise statistics, hardware vs software tripole,
fed vs fasted. The loader path that caveat was waiting for now exists. So (c)'s
placements should be spent on **new-cohort hosts**: 20 placements on an old-cohort
host buy precision on a number already marked provisional. Before (c), establish
whether the five consumers can run on a new-cohort recording (the `_new`
versions exist; `extract_mmc` already ran on `gems_j`), and what the adapter
from the 9-channel store to their inputs needs. Pick hosts from the eligible,
non-excluded set by measured blanked fraction and longest clean run, as before.

#### Rulings from the third pairing / breathing / run_continuous pass (Andrea, 2026-09-26)

**1. Breathing: timing matters, so adopt the displacement criterion.** Andrea
uses breath timing downstream, not only rate, so a breath displaced by one R-R
interval (~140 ms) is damage. `breathing_changed` counts a breath as changed if
it is added, lost, **or displaced by more than 0.07 s**. The threshold is not
arbitrary: breaths are sampled at heartbeat locations, so detected breath times
are quantised to R-R (~140–165 ms), and half an R-R interval separates "same
beat" from "moved a beat". It tightens 11 of 48 breathing rows. The residual
non-monotonicity (22 → 25 rows) is intrinsic — a winner-take-all detector
sampled at R-peaks is non-monotone in amplitude by construction — and is
handled by the lowest-failing-amplitude rule, not by the criterion. It is a
change to a pre-registered criterion, made on Andrea's statement of what the
output is used for; record it as such, with the date.

**2. Completeness is also a duration rule.** Andrea: a baseline shorter than
~10 min, or a stim/recovery file shorter than ~2 + 20 min, is **incomplete**.
Declare it in `protocol.yaml` (`min_baseline_min: 10`, `min_sr_min: 22`,
cohort constants). Apply it to every recording from its measured duration.

- It **adds** to the folder flags, never removes one: an `_INCOMPLETE` folder
  that happens to meet the duration stays excluded.
- A short recording is excluded **with its partner**, as before.
- "Around" means borderline cases exist. List every recording within 10% of
  its threshold (9–10 min baselines, ~20–22 min sr files) for Andrea instead of
  deciding them silently.
- **Dry run first.** If the rule would exclude more than ~10% of the corpus,
  stop and report before applying; a rule that removes a large share of the
  data deserves her look at the list first.
- This settles the two `gems_d` restarts: judge the later `t01` sessions by
  duration like everything else.

**3. CORRECTED 2026-09-26 — the "some are wrong" claim is not borne out on the
Drive.** It was my inference from Andrea saying she had used results; she has
since said she is **not sure which script** produced them. Measured:
`run_continuous.m` never completed an iteration on either drive (no
`_HR_BR_HRV_results.mat` or `_slowWave_results.mat` anywhere); the shifted HR
call **errors** when blanked segments exist (nearly always) and runs silently
only when none do; none of the 386 `*_HRBR.mat` files carries the shifted
fingerprint. So the suspect set on the Drive is **empty so far**. Remaining:
attribute each of the 386 files to the script and function version that wrote
it (saved fields, parameter set, file format), list the attribution for Andrea,
and flag any file whose writer cannot be determined. Results stored off the
Drive cannot be checked until Andrea connects that folder. The original
requirement, kept for reference:

**3 (original). `run_continuous.m` HR/BR/HRV results have been used, and some are wrong.**
The calls to `HR_BR_HRVAnalysis_new` passed arguments shifted one position,
without an error, after the function dropped its mode argument. Required:

- Establish what the shift did: which parameter received the mode string, which
  received what, and whether that produced errors, defaults or plausible wrong
  numbers.
- Find when the signature changed (git history of `processing_new`, plus the
  uncommitted working copy). **Outputs produced by `run_continuous.m` after that
  date are suspect; outputs from before it are not.**
- Find every such output file (saved variable names, file names, timestamps),
  list them for Andrea, and list what consumes them downstream
  (`bulk_mixed_models.m` and the like). **Do not overwrite or regenerate
  anything until she has seen the list** — regeneration replaces files she has
  already built analyses on.
- The slow-wave calls would have *errored* (12 arguments into 11), so no
  `run_continuous.m` slow-wave output exists from that period; say so if the
  search confirms it.

**4. `main_mod.m`: fix it (Andrea, 2026-09-26), not retire it.** Apply the same
fix as `run_continuous.m` to its shifted HR/BR calls (lines 232, 242, 252) and
its 12-argument slow-wave calls, and run `checkcode`. For the older defects:
fix the clear-cut ones — crashes, wrong variable names, dead branches — and
**list any fix that would change analysis results** for Andrea before applying
it. It stays uncommitted like the rest of `processing_new`.

**5. Multiple baselines in a session: the ~10 min one is the baseline** (Andrea).
The duration rule already excludes the short aborted starts; pair the ~10 min
baseline with its stim/recovery by time as usual. Do not rename trial tokens
(the `gems_d` `t03` → `t01` restarts keep their folder names).

**Parfor correction.** The extension's 46% overhead was **not** chunking:
`'auto'` and `'fixed1'` are both within 1–3% of ideal at 60 and 240 points and
on the extension's heavy head. The likeliest cause is contention from
concurrent test and generator runs on the same machine. Each point now records
its worker and finish time, so a recurrence will show its cause. Keep `'auto'`
unless a measurement says otherwise; **do not run heavy local work alongside a
timed sweep.**

**New-cohort host selection needs ≥180 s clean runs.** `gems_j_t01_ms3_bl_230315`
runs all five consumers deterministically, but its longest clean run is 136 s.
`T_hardware` found 127 spikes on the right cuff and only 16 on the left, which
fits the polarity and left-cuff-weight concern below; settle that before any
new-cohort tolerance is believed.

#### The new-cohort adapter must reproduce the old input convention

The consumers were validated on the old cohort's `_blankmotion.mat` inputs, so
the adapter's job is to hand them what they saw then, not a better signal:

- **No common-average reference on the stomach channels.** A CAR over `ANT1–3`
  removes what the three contacts share, and the slow wave is largely shared, so
  it attenuates exactly what `slow_wave` measures. Pass `ANT1–3` referenced as
  the old cohort's were; check that reference in the old files rather than
  assuming it.
- **Match the tripole's polarity to the old hardware tripole**, verified from the
  sign of real spike waveforms, not from the formula. Spike detection is often
  single-polarity; an inverted tripole could halve detected spikes and look like
  a physiological difference.
- **Report the left cuff's fitted tripole weights.** If they are far from 0.5 /
  0.5, the software tripole is not what the hardware would have produced, and
  that belongs in the `T_hardware` row's provisional caveat.

#### UPDATE 2026-09-26: the blind audit is launchable

**Tools → Blind recall audit…** opens a separate window. Recordings offered: only
those with a store entry, a reachable source file and no exclusion (844 now);
an excluded one is refused even if requested directly. Each open draws one
120 s span from a recorded seed, after removing stimulation and 20 s edge
guards. Shift+drag marks; out-of-span marks are refused. **Commit writes marks
and a plan record to `labels/<animal>/blind_audit/<session>/` before any
candidate or z-trace is computed.** Six mutations caught; exercised end to end
on one `sr` and one `bl` recording (open 14–34 s, reveal 20–40 s). The bridge now
imports `gems_blanking_v2` as a declared dependency (invariant 21), and
`meta.json` records `source_path` relative to the store root.

**Before the formal audit counts:** the five-span plan (5 contiguous 2-min spans,
~10 min total, stratified across 2–3 new animals and conditions, seed recorded)
exists in the planner but is not wired to the window. **Wire it**, so each open
advances through the plan and the window shows progress (span k of 5). Spans
Andrea marks before that are a UI trial, not audit data, unless they happen to
be drawn by the plan — keep them, marked as trial.

#### Stim epoch location depends on modality — task 03B has the same defect

On `sr` files the channel that carries stimulation depends on modality: `ms`
shows it on **ADC2 only**, `es` on **`vib` / `adc1` only**, combined conditions on
all three. A split reading `vib` alone found a spurious "stim epoch" at
1,163–1,283 s on an `ms` file whose `vib` is flat throughout. The audit now
excludes the protocol window at the start of every `sr` file (**0–132 s**),
which matched all 10 sampled files and agrees with Andrea's description (2 min
stim, then 20 min recovery).

**Task 03B must choose its channel by modality**, parsed from the corrected folder
name, and **cross-check the detected epoch against the protocol window**: stim
should begin within a few seconds of file start and last ~120 s. A detected
epoch outside that is flagged, never silently used — a detector reading the
wrong channel returns a confident, wrong answer rather than no answer, which is
exactly what happened here.

#### Rulings after the five-span wiring (Andrea, 2026-09-26)

- **Re-admit the four `sr` files** (A/t01/ms2, B/t01/es3, I/t01/3_3, I/t01/ms1).
  Each has an aborted short baseline *and* a full ~10 min baseline; the full one
  is the pair, so the `sr` has a good partner. The aborted starts stay excluded.
  This **supersedes** the earlier "14.7 min pair is real" answer, which was given
  without knowing a full baseline sat 10.1 min before that `sr` — my question
  was missing information, not her ruling inconsistent. Apply: excluded 61 → 57.
- **`cme<n>` (animal K) is combined mechanical + electrical.** Read all three
  monitor channels, as for the other combined conditions.
- **`main_mod.m`, the three held fixes — all approved:**
  1. slow wave in the single-recording branch uses the **stomach channels only**
     (3:5), matching everywhere else;
  2. **include `sr` folders** in the folder filter and the condition match, so
     stim/recovery files are preprocessed and artifact-cleaned;
  3. the hard-coded loop ranges become a **parameter defaulting to all files**,
     and **existing outputs are never overwritten unless explicitly asked**
     (skip-if-exists by default, an `overwrite` flag to force).
- **03B edge detection.** Four combined-modality files are wrongly flagged because
  a small noise-floor step reads as continued stimulation. Refused loudly, so
  nothing is corrupted, but fix the edge rule: define the stim-off edge against
  the post-stim noise floor measured on the recovery segment, not the pre-stim
  one, and require the ON level to be a clear multiple of it. Re-run the 259
  and confirm the four pass without any of the 255 changing.
- **`vib` is numerically identical to `adc1`.** Treat it as one channel, not two
  independent confirmations: agreement between identical copies proves nothing.
  For `es`, the check is therefore single-channel — say so in the split report.

#### After the re-admission pass (2026-09-26)

- **`gems_i_t03_es2_sr_223947`: all three monitor channels exactly zero.**
  Andrea is not sure stimulation happened. **Exclude it and its baseline
  partner**, reason `no_stim_monitor`. Refusing a constant monitor in 03B was
  right: the old and new code both reported a fabricated 0–120 s "pass" from an
  argmax over a flat score (invariant 41). This changes the audit pool (527 →
  expected 525), so apply it **before** any audit plan is created. If a plan was
  already created, check whether it draws either recording; if it does, discard
  that plan (it has no committed spans yet) and have Andrea create a new one.
- **03B onset rule extended to the post-/pre-stim floor as well — ratified.** It
  goes beyond the ruling's letter for the right reason: the same floor step
  preceded stim on two files.
- **`main_mod.m` stage 1 on new-cohort folders must refuse** until it takes its
  stream names and channel indices from the file's `chanlabels` (the single
  source, per the channel-order rule) rather than the old cohort's
  `[1:2,17:19]`. Fix (b) admitted `sr` folders, and a wrongly indexed
  `_notched.mat` would be silent. Refuse loudly now; build the per-cohort map
  when Andrea next needs stage 1 on new data.
- **Adapter checks run when Andrea is not labelling.** They read full recordings
  over the Drive. She will say when she stops for the day.

#### The blind audit is not launchable — this is now the critical path (superseded, see update above)

Session, controller, dock and planner exist but are not wired into the app, there
is no way to start audit mode, marks go wherever the caller says rather than into
the store, and the bridge still finds `gems_blanking_v2` by `sys.path` insertion
(invariant 21). Andrea cannot start labelling, and labelling is the human
critical path to the 09 gate. **After the current runs finish, this comes before
more T work.** Done means Andrea can open the app, pick an eligible new-cohort
recording, and do a blind span with marks written to the store under the
session key — excluded recordings never offered.

#### Confirm `rostral_end` landed

The report does not mention it. It was to ride along with the re-key: `rostral_end:
1` in `protocol.yaml`, deleted from `meta.json`, `direction_valid` false for the
old cohort regardless. If the pull predated that commit, it is one more
idempotent run.

#### The median σ: do not run a sweep for it

The offer was a stratified sample of ~30 recordings to characterise the cohort's
robust σ. **Declined.** It costs hours of Drive bandwidth, which is the scarce
resource this week and is already contended by T and by labelling, and it buys a
tighter plausibility band — a band whose job is to catch errors of 10³ and 10⁶,
which the current 1–500 µV already does with margin on both sides.

**Record it as a by-product instead.** `assert_plausible_units` already computes
the median robust σ on every load. Log that value, with the recording id, to
provenance. The distribution then accumulates for free as labelling and
processing proceed, and by the time there is enough of it to justify tightening
the band, it will have been collected by passes that had to happen anyway. A
sweep run to obtain a number an existing pass already computes is a second pass
over the same bytes. (Invariant 26.)

#### The corpus is growing under the build

852 at the first scan, 859 now. Andrea is still collecting, and she will be for
weeks, so every count in every report ages the moment it is printed. Two
consequences, both required:

- **Every count is printed with the enumeration timestamp that produced it.** A
  bare "859 recordings" in a report is not reproducible and cannot be reconciled
  against a later one.
- **The generator is idempotent and incremental.** Re-running it writes only
  what is missing, leaves existing files untouched, and reports new / existing /
  collided separately. A one-shot migration script is wrong for a corpus that is
  still growing; this will be run many times.

#### Cheap checks, both clean

- **Compression is gzip-4 throughout.** Task 10 is unblocked — no re-write pass.
- **`Documents/GEMSBlanking` is clean at `599d1fe`.** No uncommitted local
  divergence behind the import.

### BUILD CHANGE 2 FIRST — it is the only part on the critical path

Change 2 (blind recall audit) unblocks step L, which unblocks the 09 gate.
Change 1 (candidate adjudication) is not needed until **task 10**, which is
*post*-gate. So ship Change 2 on its own, let the labelling start, and build
Change 1 after. Doing them together delays the labelling for work nothing is
waiting on.

### The z-trace panel: 6 traces, not 54

Nine signals × six bands is 54 traces and no human reads that. The diagnostic
question is narrow — *was z high anywhere at this moment?* — so show **one trace
per band, the maximum across all signals**, with the `z_enter` threshold drawn on
each. Six traces answer blind-spot-versus-threshold directly. Which signal
carried the maximum is a click-to-expand detail, not the default view.

The maximum is taken across **all** signals including raw contacts, per hard
invariant 6 — detection reads the contacts, so the audit must show what detection
saw.

### The blind-mode sampling rule

**Contiguous spans, not scattered minutes.** Recall is
`(artifacts the human found that no candidate covers) / (all artifacts the human
found)`, which is only computable over a *complete* span. Scattered ten-second
snippets destroy the context a human needs and make the denominator meaningless.

```
~10 minutes total, as 5 contiguous 2-minute spans
uniform-random start within each recording
stratified across 2-3 new animals and across conditions
seed recorded in provenance
excludes stim epochs (03B) and unassessable edges (task 06 settling)
```

**Explicitly NOT candidate-dense sampling.** Choosing where to look using the
thing under test reintroduces the anchoring the blind mode exists to prevent — at
the level of span selection rather than of marking. The objection that raised
this had it exactly right.

### Change 2 — NEW recall-audit mode

**Blind first, reveal second.** The human marks artifacts on the raw signal with
candidates **hidden**, commits, and only then are candidates and z-traces revealed.
If candidates are visible while marking, the labeller anchors on them and measured
recall is inflated — the gate would be measuring its own output. This is not a
preference; without it the gate is circular and worthless.

After the reveal: a scroll view over the sampled minutes with candidates overlaid
**and the band z-traces shown alongside the raw signal**.

The z-traces are the entire point. When the human finds a missed artifact, they
distinguish:
- **generator blind spot** — z genuinely low → needs a new band or feature
- **threshold too high** — z high but sub-threshold → needs a lower `z_enter`

Different diagnoses, different fixes, and **indistinguishable from the raw trace
alone**. Without this the audit tells you that recall is bad but not why.

### Change 3 — preprocessing defaults
Set the candidate-harmonic list in the Preprocessing tab to `60` only. Do not enable
harmonic detection: a 120 Hz notch rings *inside* the ENG band at 2.41 µV/mV
(threshold at a 7.5 mV excursion) where the 60 Hz notch contributes 0.221 µV/mV.

### Labelling budget — design to this
~500–1000 judged events across ~12 recordings spanning all animals, one keystroke
each: **1–3 hours once**. Then ~50 uncertain events per new animal (~15 min). Zero
per production dataset. If the UI requires materially more than this, it is wrong.

### Acceptance
Audit mode usable on a real recording; a missed artifact can be classified as blind
spot vs threshold in one glance.
<!-- /TASK -->

<!-- TASK:16A slug=app-shell deps=16 gate=no -->
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
<!-- /TASK -->

<!-- TASK:16B slug=results-dashboard deps=12,16A gate=no -->
## Task 16B — Training and evaluation dashboard

**Module:** `ui/dashboard/`, rendered as a self-contained HTML report **and**
embedded in the app
**Depends on:** 12, 16A

### Purpose
This is a research instrument, so the dashboard's job is **interpretation, not
reassurance**. It must make the underlying data legible before it shows any model
score, and it must make it easy to see *why* a number is what it is. A dashboard
that shows F1 and nothing else is worse than no dashboard, because it invites
shipping a model nobody understands.

**Reliable, not overcomplicated:** six panels, in this order. Anything else is a
drill-down, not a panel.

---

### Panel 1 — Corpus (before any model score)

What the model was actually trained on. Per animal: recordings, candidates,
judged events, **positive fraction**, unjudged fraction.

- Form: horizontal bars, animals on the y-axis, sorted by event count
- A stat-tile row above: total judged events, animals, overall positive fraction,
  % of candidates still unjudged
- **Why first:** every downstream number is conditioned on this, and corpus size is
  the confounder in the mode comparison (invariant 13)

### Panel 2 — Candidate recall (the gate)

- Form: **heatmap**, injected amplitude ratio × duration, cell = recall. Sequential
  single hue, light→dark. Never a rainbow.
- Beside it: the threshold sweep — recall, candidate count and TP-fraction vs
  `z_enter`, with the pinned value marked
- Two measured lines, labelled: synthetic-injection recall and human-audit recall.
  They answer different questions and must not be averaged

### Panel 3 — Learning curves **(the most important panel)**

F1 vs `n_training_events`, log x, one line per mode, small multiples per animal,
bootstrap CI bands.

- Series: POOLED / ADAPTED / PER_ANIMAL → categorical slots 1, 2, 3
- **This is what answers "is per-animal better, or does it just have different
  data".** If the per-animal point lies on the pooled curve at matched event count,
  the difference was corpus size
- Annotate each animal's actual event count with a vertical rule, so the reader can
  see where that animal really sits

### Panel 4 — Per-animal performance, never averaged

Grouped bars: precision / recall / F1, grouped by animal, one bar per mode.

- The two baselines (all-motion, single-feature threshold) drawn as **reference
  rules**, not as extra series
- Each bar labelled with its **protocol** (LOAO / within-animal). Bars of different
  protocol are visually separated and carry a "not comparable" marker between
  groups — the UI must make invariant 12 hard to violate by eye
- No mean-across-animals row anywhere on this panel

### Panel 5 — Calibration and error inspection

- Reliability curve per mode (predicted vs observed, diagonal reference)
- Confusion counts at the operating threshold
- **Click any cell → the actual waveform.** False positives and false negatives open
  the candidate in the signal viewer with its band z-traces. This is the single
  most valuable feature on the dashboard for a research tool: a number you cannot
  trace back to a trace is not evidence

### Panel 6 — Feature behaviour

- SHAP summary for the pooled model (reuse `GEMSBlanking:detector/review.py`)
- **Per-animal feature distributions** for the top 10 features, as small-multiple
  ridgelines. A feature whose distribution separates animals is **not
  animal-invariant** — this is the diagnostic that explains a PER_ANIMAL win and
  points at the task 11 bug

---

### Chart construction rules
- Categorical hues assigned in **fixed order**, never cycled: mode 1 = slot 1,
  mode 2 = slot 2, mode 3 = slot 3, regardless of which modes are present
- **Never a dual-axis chart.** Recall and candidate count on one plot → two plots
  or index to a common base
- Sequential = one hue light→dark (recall heatmap). Diverging = two hues + neutral
  midpoint (only for signed quantities, e.g. Δ vs baseline)
- Legend present for ≥2 series; ≤4 series also directly labelled
- Hover tooltip on every mark; crosshair on the line charts
- A **table view** for every panel — this is a research tool and the numbers get
  copied into papers
- Light and dark both explicitly designed, not an automatic flip
- Run the palette validator before shipping; do not eyeball CVD safety

### Export
One button: the whole dashboard as a **self-contained HTML file** with the data
embedded, plus the underlying parquet. It goes in the lab notebook and into
supplementary material, so it must open with no server and no network.

### What must NOT be on the dashboard
- A single headline "accuracy" number
- Any average across animals
- An A-vs-C comparison presented as a comparison
- Any score without its protocol label
- Green/red pass badges on anything the user has not been shown the evidence for

### Tests
- panel 4 refuses to render a cross-animal mean
- a model without held-out metrics renders as `unvalidated`, not as a blank
- the exported HTML opens offline with all data present
- palette validator passes for both light and dark

### Acceptance
Andrea can look at the dashboard after a training run and answer, without asking
anyone: what was it trained on, did candidate generation work, is per-animal
actually better or just differently-sized, which animal is worst and why, and what
does a typical error look like.
<!-- /TASK -->

<!-- TASK:17 slug=video deps=03 gate=no -->
## Task 17 — Video motion features

**Module:** `video/motion.py`
**Depends on:** 03

A prototype already exists: `video_artifact_coincidence.py` (270 lines) — it probes
the container via `ffprobe`, computes ROI motion energy via OpenCV, loads segment
labels (v5 and v7.3 `.mat`), fits the sync lag by cross-correlation, and estimates
drift from the first and last thirds. Start from it.


> **Task 07 declared `VideoMotion` as a structural `Protocol`** because this task
> names ROIs, sync and drift but never a type. **Satisfy that protocol; do not
> redefine it.** Minimum shape: a motion trace on the shared 10 ms grid plus its
> own threshold. If 17 needs a richer object, widen the protocol in task 07 and
> say so — do not let two shapes exist.

### Sync
Camera and TDT share a start trigger, so the **offset** is solved. **Drift is not**:
100 ppm over 20 minutes is 120 ms, longer than the feature window.
1. Check for per-frame PTS first.
2. Failing that, fit **one linear warp per recording** by cross-correlating motion
   energy against existing labels.
3. Report the fitted drift in ppm per recording.

### ROIs
The **tether and commutator**, not just the animal. The cable causes the artifact,
and cable / connector / headstage sources enter **downstream of the electrode**,
where no montage can reject them — which is exactly why video adds information the
electrical channels cannot.

### Accuracy budget
Feature use needs ~100 ms. **Labelling adjudication needs only ~1 s**, so that use
works today with no drift correction. Ship adjudication first.

### Unanswered — resolve before building
Container, fps, and whether per-frame timestamps exist are all unknown. Probe first
and report, rather than assuming.

### Tests
- a synthetic video with a known injected lag recovers that lag to within one frame
- drift estimation recovers a deliberately warped timebase to within 20 ppm
- ROI motion energy responds to motion in the ROI and not outside it

### Acceptance
Lag and drift reported per recording; motion-energy traces aligned to the neural
timebase; coincidence with existing labels reported.
<!-- /TASK -->

<!-- TASK:18 slug=velocity deps=04 gate=no -->
## Task 18 — Conduction velocity and direction

**Module:** `velocity/xcorr.py`
**Depends on:** 04 (new-cohort recordings only)

Runs in **two passes**. Pass 1 needs no mask: it produces the
non-physiological-energy trace that task 11 consumes as a feature. Pass 2 runs after
task 15 and uses the velocity mask to produce the reported velocity estimates.
Keeping these separate is what stops the feature feedback from becoming a loop
(invariant 4).

### Algorithm
Cross-correlate contact pairs **band-limited to 300–5000 Hz first**, on segments the
velocity mask passes. Peak lag gives velocity; its sign against `rostral_end` gives
direction.

### Measured tolerances
Raw correlation survives artifact only to ~1× the neural amplitude; band-limited
first, to ~5×; against a saturating broadband artifact, ~1.4×. Band-limiting is not
optional.

### Do not restrict the lag search
Restricting to physiological velocities **does not help and is harmful** — it
converts an obviously broken estimate into a plausible-looking wrong one. Leave the
search unrestricted and reject on confidence instead.

### Confidence
Emit **peak ratio** with every estimate: neural peak height ÷ zero-lag peak height,
from the same correlation. Below ~1, discard that window. This is stronger than any
mask because it is derived from the quantity being estimated.

### Feed back to task 11
Energy above ~50 m/s cannot be neural, so it can only be common-mode contamination.
That is the strongest artifact feature available on the new configuration. Emit it
as a per-frame trace for the feature builder. **This is a precomputed input, not a
loop** — velocity runs its own pass first (invariant 4).

### Geometry — cuff v9
Pitch **1.50 mm**, aperture **3.00 mm** (measured from the STL: grooves at
z = 1.00 / 2.50 / 4.00 mm). Resolution `Δv/v ≈ v/(B·L)`: 4% at 0.5 m/s, 7% at 1,
14% at 2, 35% at 5. **A C-fibre instrument, not an A-fibre one** — say so in the
output metadata. Delays are never the limit: 3 ms at 1 m/s on the outer pair, 73
samples at 24.4 kHz.

### Tests
- a synthetic propagating volley at a known velocity is recovered within the
  predicted resolution
- a zero-delay common-mode artifact produces a peak ratio < 1 and is discarded
- missing `rostral_end` yields unsigned output with `direction_valid=False`

### Acceptance
Velocity, direction, and peak-ratio confidence emitted per window; the
non-physiological-energy trace available to task 11.
<!-- /TASK -->

<!-- TASK:19 slug=acceptance deps=all gate=yes -->
## Task 19 — End-to-end acceptance

**Depends on:** everything
**Gate:** YES — this is the ship decision

| Test | Pass condition |
|---|---|
| Candidate recall | ≥98% of injected synthetics produce a candidate, at a threshold yielding ≤3000 candidates/recording |
| Class balance | ≥5% of candidates are true positives on labelled recordings |
| No zeros | no exact-zero runs in any emitted `yOut`; every mask boundary tapered |
| Cardiac scope | peri-R profile flat in 300–5000 Hz and below 3 Hz after masking |
| Retention | reported per band per channel against the current all-channel full-band baseline; **ENG-band retention substantially higher than the slow bands** |
| Cross-animal | blind-test event recall on the four never-seen animals, reported **individually** |
| **Coverage confound** | blanking fraction must **not** depend on condition. If stimulation drives movement and this goes unchecked, the pipeline has automated a confound rather than fixed one. Compute it on **`masked_motion` only** — deterministic `excluded_epoch` (stim) and cardiac windows are excluded from the numerator, or the regression measures the protocol rather than the artifact |
| Downstream | `fracISIclean`, slow-wave non-NaN fraction, DFA `alpha2` availability, `nRR_used`, median `step5f` epoch length — all up; between-animal endpoint variance down |
| Velocity | peak-ratio confidence emitted with every estimate; unsigned output with a warning where `rostral_end` is missing |
| **Mode comparison** | learning-curve plot produced per mode per animal; A-vs-C never reported as a comparison; B-vs-C verdict stated with the task 11 investigation triggered if C > B |
| **Model routing** | every emitted mask names its `ModelSpec`; routing rule logged per recording; re-running a recording selects the same model |
| **Calibration** | every shipped model calibrated on held-out data; reliability curve within tolerance |

The **coverage confound** row is the one that matters most scientifically and is the
easiest to skip. Test it explicitly: regress blanking fraction on condition and
report the coefficient.

**With multiple modes there is a second confound of the same shape:** if different
animals are scored by different modes, and mode correlates with anything (labelling
effort, cohort, recording date), then mode becomes a hidden covariate on every
downstream endpoint. Report blanking fraction **by mode** as well as by condition,
and carry the mode into `bulk_mixed_models.m` as a factor if it varies across the
dataset.

### Deliverable
A single report: every row, pass/fail, with the number. Plus a list of every place
real data disagreed with a constant in this document.
<!-- /TASK -->

---

## Part C — deliberately out of scope

- **Artifact removal *within* the stim epoch.** The stim epoch itself is
  identified and excluded by task 03B, which is what "ignore the stim duration"
  means operationally — that part **is** in scope and must be built. What remains
  deferred is *recovering usable signal from inside* the stim epoch: stim artifacts
  are large, periodic and have known timing from the `vib` channel, so they belong
  in the same category as cardiac (deterministic, known timing, per-band). Deferred
  at Andrea's request pending her own validation. The stim epoch is **kept on
  disk**, not discarded, so this can be revisited without re-acquiring anything.
- **Stomach referencing change.** Old animals were hardware-referenced in TDT; new
  ones are raw and referenced in postprocessing. Good for detection (preserves the
  common mode) but `extract_mmc`'s `k = 3 × MAD` threshold and the slow-wave
  prominence criteria were tuned on a different signal. **MMC and slow-wave
  endpoints may not be comparable across cohorts** — flag, do not silently pool.
- **`Raww` anti-alias setting** is unknown, so whether the ENG low-pass can rise
  above 5 kHz is undetermined. Ask before changing it.

## Part D — open questions to resolve with Andrea

1. `Raww` anti-alias filter setting.
2. Video container, fps, and whether per-frame PTS exist.
2b. **What Windows Drive substitutes for the `:` in `BIONICs Lab: Enteric
   Interfaces Team`.** Still open as of 2026-09-21 — Drive for desktop is
   installed on the Windows build machine but has never been signed in, so
   nothing is mounted and there is no folder name to read. **Not guessed**:
   `∶` (U+2236), `` (U+F03A) and `_` are all plausible and are three different
   strings. Mitigated rather than answered — discovery is by marker only, every
   stored path is relative, and `tests/test_portability.py` parametrises the root
   name over all four candidates. `gems doctor` prints the resolved root, so the
   first signed-in machine answers it in one command.
3. What the old TDT stomach reference actually was.
4. Current `recall_real` from the previous model, and the target.
5. The RR histogram needed to set `R_MIN` (task 05).
6. ~~What `E1000` / `M100` etc. mean~~ — **answered 2026-09-21: Hz, same axes as
   ES/MS; explicit-Hz beats the ordinal on conflict.** See task 03A.
7. **Was a specific stomach contact intended as the reference for the new
   cohort**, or is common-average correct? Common-average attenuates the
   slow wave, which is largely common across the array (task 04).
8. **Was the old cohort's electrical stimulation also 1000 µA?** The new cohort
   fixes it there; old filenames encode frequency only. Only matters for pooling
   cohorts on amplitude — frequency comparisons are unaffected.
