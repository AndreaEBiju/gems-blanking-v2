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
