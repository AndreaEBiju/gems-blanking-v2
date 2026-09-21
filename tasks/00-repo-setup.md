<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

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

---

**Depends on tasks:** none

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
