<!-- GENERATED from IMPLEMENTATION.md by split_tasks.py — DO NOT EDIT -->

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
set improves over time:

```yaml
vocabulary: [baseline, stim, stim_recovery, sham, drug, unknown]   # CLOSED list
rules:
  - { pattern: '_stim_rec(\b|_)', condition: stim_recovery, priority: 10 }
  - { pattern: '_stim(\b|_)',     condition: stim,          priority: 20 }
  - { pattern: '_base(line)?(\b|_)', condition: baseline,   priority: 30 }
strip_suffixes: ['_notched', '_notchblanked', '_blankmotion']
animal_pattern: '(?<![A-Za-z])([A-Z])(?=[_\d])'
```

- **The vocabulary is closed.** Free-text conditions are not allowed — `stim`,
  `Stim` and `stimulation` as three distinct levels would quietly wreck the mixed
  models. Adding a level is an explicit edit to `vocabulary`.
- Rules are tried in `priority` order; the **first** match wins, and the rule id is
  recorded. Order matters: `_stim_rec` must be tried before `_stim`, which the
  current code gets right only by accident of the `in` test.
- If two rules of equal priority match different conditions → `ambiguous`.

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
class ScanResult:
    path:         Path
    content_hash: str
    animal:       str | None
    session:      str | None
    condition:    str                  # includes "unknown"
    status:       Literal["matched","ambiguous","unknown","duplicate","known"]
    matched_rule: str | None
    candidates:   list[str]            # for ambiguous

def scan(root: Path, rules: Rules, known: Registry) -> list[ScanResult]
def apply_corrections(results, corrections, user: str) -> None   # writes meta.json
```

### Required behaviours
- **Recursive scan** of the given folder for the configured extensions; default to
  the shared-drive `data/` tree.
- **Deduplicate by content hash**, not by path. Folder reorganisations and Drive
  conflict copies produce the same recording at two paths; flag as `duplicate` and
  show both paths rather than ingesting twice.
- **Skip already-known recordings** (status `known`), but re-verify their checksum.
- **Sort `unknown` and `ambiguous` to the top** of the review table — the rows
  needing attention should not be buried among hundreds of correct ones.
- **Bulk edit**: multi-select rows → set animal or condition in one action.
- Every correction records `who` and `when` in `meta.json`; a user-set condition
  is marked `source: human` and is never overwritten by a later rescan.
- Scanning is **read-only** until the user confirms. Nothing is written to
  `gems_root` during a scan.
- Scans can be slow over a streamed shared drive — walk metadata only, hash lazily,
  and show progress.

### Tests
- a filename matching no rule yields `unknown`, **not** `baseline` (the regression
  guard on the current behaviour)
- `X_stim_rec_01` resolves to `stim_recovery`, not `stim` — priority ordering
- two equal-priority rules matching different conditions yield `ambiguous`
- the same content at two paths yields one `duplicate` row naming both
- a human-set condition survives a rescan
- a condition outside `vocabulary` is rejected at write time
- corrections are written with user and timestamp
- an `unknown` recording cannot be added to a corpus spec

### Acceptance
Point the tool at the shared drive's `data/` root: it lists every recording with a
proposed animal and condition, flags what it could not determine instead of
guessing, and after corrections every row is either confirmed or explicitly
excluded. Report how many of the 43 existing recordings parse cleanly under the
initial rule set — **and how many the current code would have silently called
`baseline`.**

---

**Depends on tasks:** 03, 00A

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
