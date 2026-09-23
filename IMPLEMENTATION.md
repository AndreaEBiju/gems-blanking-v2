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

**Stated limit of the test:** a rat MMC cycle is 90–120 minutes and these spans
are 3–7 minutes, so nothing here can confirm or refute MMC on its own
timescale. What it establishes is narrower and sufficient: the events are
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
| `browseMotionArtifacts.m:34` | `validateattributes(..., 'finite')` throws on NaN, so an already-blanked file cannot be re-browsed. Remove if the browser is kept |
| ~~every `filtfilt` call at a low corner~~ | **Audited and withdrawn as a padtype problem — measured, no change needed.** The row predicted that MATLAB's mandatory odd extension would reproduce task 06's transient. It does not, and the reason is worth keeping: MATLAB pads `3·2·n_sections` = **6 samples**, which at 24.4 kHz is **0.246 ms**. Odd and constant padding differ by 11.8% of sd at 0.5 s from the edge, 1.1% at 2 s and **0.00% by 15 s**; peaks surviving the existing edge buffer, **13 either way**. scipy's damage came from odd-extending across a pad long enough for the signal to move; a quarter-millisecond pad of a 0.15 Hz signal cannot. **The settling itself is real and is already handled**: measured `impz` 8.17 s against a 15 s buffer, and the code's `order/cutoff` heuristic (13.33 s) over-estimates it, which is the safe direction. Generalising a scipy result to MATLAB without measuring was the error here |
| `extract_mmc.m` — **`xf(~isfinite(xf)) = 0`** | **Hard invariant 1, violated, live.** If `fillmissing` leaves anything non-finite it becomes **zero**, and a zero is indistinguishable from signal to everything downstream. This is the exact defect task 01 was written for — which turned out to have been fixed upstream in `a95d1ff` before this project began — found here for real, in a different file. **Highest priority row in this task.** Fix to NaN and let the consumer decide; audit the rest of `processing_new` for the same construct |
| `extract_mmc.m` — `detect_crossings` 30 s moving MAD | An independent instance of the `step2_noise_sigma` row above: a moving noise estimate whose threshold rises with activity. Same failure, same direction, same fix — a fixed session reference |
| `extract_mmc.m` — blank restore is sample-exact | The blank-before-filter-restore pattern is otherwise done correctly (`bl` → `fillmissing` → `filtfilt` → `y(bl) = NaN`), but **only the blanked samples are restored**. With a 0.486 s impulse response, roughly half a second either side of every blank is filter output computed partly from interpolated data, and it is kept. **This is task 13's question arriving in MATLAB**: the restore must extend by the consumer's settling time, not by the blank. Audit every blank-restore in `processing_new` for the same pattern — it is a shape, not a one-off |
| `extract_mmc.m` 2–50 Hz bandpass | **The finding the padtype audit actually produced.** Measured `impz` **0.486 s** and **no edge buffer anywhere**, feeding the MMC statistic. Every other low-corner site is covered: `HR_BR` 1–100 Hz 0.159 s against 0.75 s (4.7×), slow wave 8.17 s against 15 s (1.8×), `step1_bandpass` 0.0051 s against 5 ms (marginal — raise to 10 ms). Read the MMC chain's cardiac-interpolation logic before touching it; the buffer interacts with it |

**Reuse rather than reinvent:** `dfaGapAware.m` (pooled runs), `step5f_fano_slope.m`
(epochs + rate-matched surrogates carrying identical censoring — extend the same
pattern to CV2 and LV), `step5e_multiband_validate.m` (peri-R histogram validation).

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
