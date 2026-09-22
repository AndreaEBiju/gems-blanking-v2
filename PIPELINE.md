# GEMS motion-artifact pipeline — implementation brief

Target audience: Claude Code, implementing this against the existing repos.

**Repos referenced**

| Repo | Language | Status |
|---|---|---|
| `AndreaEBiju/processing_new` | MATLAB | public — downstream physiology pipeline |
| `AndreaEBiju/detector-pyqt` | Python | public — labeling / inference / training UI |
| `AndreaEBiju/GEMSBlanking` | Python | **private** — detector backend, consumed as the `detector-core/` submodule |

Paths in `processing_new` and `detector-pyqt` were verified by reading the source.
Paths in `GEMSBlanking` are taken from `detector-pyqt/DEVELOPER_GUIDE.md`'s module
table and from import statements; **verify them before editing.**

---

## 0. What this replaces and why

The current detector scores 100 ms windows densely and learns artifact-vs-clean
from human-dragged interval labels. Three consequences, all measured or read from
the code:

1. Unmarked spans become weighted negatives (`w_neg = 0.3`), so real artifacts the
   human skipped teach the model that artifacts are clean. This is the most likely
   reason recall resisted five separate reweighting knobs.
2. Extent is learned when it is computable. Boundary precision from Shift+drag is
   ±100–500 ms against a 100 ms window, which caps window-level recall in a way
   that mimics underfitting.
3. One mask, all channels, full band. A QRS carries **0.00%** of its energy below
   3 Hz and **0.00–0.54%** above 300 Hz, so the uniform ±15 ms cardiac blank
   destroys ~20% of every recording to remove a contaminant that mostly is not
   there.

The new design keeps the platform and replaces the detector core:

- **deterministic candidate generation** → **one learned judgment per candidate**
  → **deterministic extent against each consumer's own tolerance**
- masks indexed by `(signal, band, time)`, never merged across consumers
- everything not learned is arithmetic or a measurement

**The load-bearing assumption:** candidate recall must be near-complete. It is
measurable (synthetic injection + sampled human audit) before the classifier is
built. If it cannot be met at a workable threshold, abandon the two-stage split and
build a dense 1D U-Net on the multi-band spectrogram instead.

---

## 1. Architecture

```
00 load + channel map ─────────────────── EXISTS (extend)
01 notch 60 Hz ────────────────────────── EXISTS
02 derivations (V1,V2,V3,T) ───────────── NEW
03 R-peaks + best-channel ranking ─────── NEW (replaces current method)
04 cardiac window measurement ─────────── NEW
05 band envelopes + reference + z ─────── NEW
06 candidate generation ───────────────── NEW
07 features per candidate ─────────────── NEW (replaces Phase 1)
08 classify (the only learned step) ───── MODIFY (retarget existing LightGBM)
09 extent per consumer ────────────────── NEW
10 routing (correct / subtract / reject)─ NEW
11 mask emission + QC report ──────────── MODIFY (labeled_save)
12 velocity + direction ───────────────── NEW (new-configuration animals only)
    video motion features ─────────────── NEW (feeds 06 and 07)
```

No loops. Step 03 depends on nothing downstream; step 05's reference is a
percentile of the file itself; step 09's tolerance uses a robust statistic over the
same file. Verify this stays true if you reorder anything.

---

## 2. Signals, bands, consumers

### Derived signals (step 02)

Per 3-contact nerve cuff: `V1`, `V2`, `V3` (raw contacts) and `T` (tripole).
Per stomach set: the 3 contacts plus the postprocessing-referenced derivation.

Tripole: `T = a*V1 + b*V3 - V2` with `a + b = 1`. Fit `a, b` per cuff per recording
by minimising variance of `T` in **20–300 Hz**, where motion dominates and neural
content is minimal. This corrects contact-impedance mismatch, which a hardware
short cannot. Old-configuration animals arrive with `T` pre-formed in hardware and
skip the fit — record `a, b = NaN` for them.

### Bands

Compute **all bands on all signals** (envelopes are cheap 100 Hz traces). Emit masks
only for the consumer pairs below.

| Band | Window for ~30 DOF | Reference percentile |
|---|---|---|
| 300–3000 Hz | 25 ms | 10th |  *(was 300–5000 — measured, see IMPLEMENTATION.md A.5b)*
| 100–300 Hz | 75 ms | 10th |
| 1–100 Hz | 150 ms | 10th |
| 2–50 Hz | 310 ms | 10th |
| 0.5–3 Hz | 6 s | 25th |
| 0–2 Hz | 7.5 s | 25th |

Window length is set by effective degrees of freedom `2*B*T`. A 25 ms window holds
0.05 cycles of a 1 Hz signal — it samples an instantaneous value, not an envelope.
Keep a **common 10 ms output grid** for all bands so features align; centre a
band-appropriate window on each grid point.

Percentile choice trades contamination robustness (lower is safer) against
estimator noise (higher is tighter). High-frame bands can afford the 10th; the two
slowest bands have only 80–100 frames in a 10-minute file, where the 10th
percentile carries ~13% standard error, so use the 25th.

### Consumers — these and only these get masks

| Consumer | Signal | Band | Tolerance |
|---|---|---|---|
| Spike detection, envelope, ECAP | `T` | 300–3000 | 4.5σ (sample level) |
| Slow-C | `T` | 100–300 | its own σ |
| Velocity + direction | pair `(V1,V3)` | 300–3000 | peak ratio > 1 |
| MMC | stomach derivation | 2–50 | 3 × moving MAD |
| Slow wave | stomach derivation | 0–2 | peak displacement |
| Breathing rate | best HR channel | 0.5–3 | peak inserted or lost |
| R-peaks / HRV | best HR channel | 1–100 | **operational**: beat train unchanged |

No 0–2 or 2–50 Hz mask on nerve signals. No 300–5000 Hz mask on stomach signals.

**Masks are never merged across consumers.** The single exception is *within* a
consumer that reads two channels: velocity requires `V1` AND `V3` valid, so that
one is an intersection.

---

## 3. Module by module

### 00 · Load and channel map — EXISTS, extend

- Exists: `GEMSBlanking:detector/recording_io.py` (`Recording`, `load_recording`,
  handles `.mat`, chunked HDF5, flat HDF5).
- Exists: `detector-pyqt/ui/widgets/channel_assignment.py` — per-channel role
  (nerve / stomach / other), label, sparkline preview, saved per animal to
  `~/.detector/preprocessing_profiles/<animal>.json`.
- Exists: `detector-pyqt/scripts/m1_ingest.py` — `.mat` → flat HDF5 with a
  pre-downsampled overview. Use this; the viewer is 5× faster on it.

**NEW — extend the channel table with:**
- `cuff_id` and `contact_index` (1–3) per nerve channel
- `rostral_end` — which contact index is rostral

`rostral_end` cannot be reconstructed after the animal is gone, and without it the
sign of every conduction velocity is meaningless. **If absent, the pipeline must
emit unsigned velocities with a loud warning, never guess.**

### 01 · Notch — EXISTS, do not extend

- Exists: `detector-pyqt/ui/widgets/notch_review.py` +
  `GEMSBlanking:detector` preprocessing; MATLAB equivalent at `main.m:64-73`.
- **60 Hz only. Do not enable harmonic detection.** Measured: the 60 Hz notch adds
  0.221 µV of in-band ring per mV of excursion (would need an ~82 mV artifact to
  reach 4.5σ), because the ring sits at 60 Hz and the 100 Hz bandpass rejects it by
  −36.4 dB through `filtfilt`. A 120 Hz notch rings *inside* the ENG band at
  2.41 µV/mV — threshold at a 7.5 mV excursion.
- **Change the default candidate-harmonic list in the Preprocessing tab to `60`.**

### 02 · Derivations — NEW

`derive/derivations.py`

```
in :  notched channel matrix + channel table
out:  dict of named signals per cuff {V1,V2,V3,T}; fitted (a,b) per cuff
```

Detection consumes **all** signals as recorded. Do **not** detect on the tripole
alone: the tripole is defined as the operation that removes the common-mode
component, and the common-mode component is the best artifact evidence available.

### 03 · R-peaks and best-channel ranking — NEW, replaces current method

`physio/rpeaks.py`

Current method is `HR_BR_HRVAnalysis_new.m:160-179`: `findpeaks` on the **detrended
raw** signal with the low-pass commented out (`yFilt = xFill`) and only
`MinPeakDistance`, no amplitude or prominence threshold. Measured at severe motion:
361/399 beats, 57 false, **RR error 1.3%**.

**Replace with:**
1. band-limit **1–100 Hz** (this alone takes RR error to 0.2%)
2. `findpeaks` with a **fixed** robust prominence (3 × MAD-σ), 90 ms refractory
3. plausibility rejection — drop any beat closer than `0.55 ×` the local median RR
4. **pass 2 — gap-targeted re-detection.** Where an RR is a clean multiple `m` of
   the local median (within 20%), re-run `findpeaks` inside `expected ± 0.25 × RR`
   with **relaxed** tolerances — prominence `1.2 × MAD-σ` (vs `3.0` in pass 1),
   width in `[0.5, 2.0] ×` median QRS width — and take the candidate with the
   highest correlation to that channel's QRS template from the ranking step. Tag
   `rescued`.
   *Justification for relaxing:* pass 1 has ~2180 independent opportunities over
   120 s (Bonferroni z = 4.08); pass 2 has ~30 windows (z = 2.95) — a ~28% lower
   threshold at the same family-wise false-positive rate, bought by the location
   prior.
   **Never insert a fabricated beat.** If the window is empty, tag the gap
   `unrecovered`: rate may use `m` for the count, HRV excludes every interval
   touching it. Measured at 9% missed beats — midpoint insertion biases RMSSD and
   SD1 by **−6.3%**, systematically *downward* (equal flanking intervals give zero
   successive difference), which is the same direction as a vagal-tone effect;
   leaving the gap in biases them **+771%**.
   Log rescue rate per channel per recording; it feeds the channel gate.

Thresholds are **fixed and stateless** — see §10.1 before substituting an
adaptive detector.

**No motion mask is needed here** — the beats lost inside artifacts sit in spans
that get masked anyway, so the information pass 1 cannot get is information the
pipeline was going to discard. This is what breaks the R-peak / motion circularity.

**NEW — best-channel ranking.** Run detection on all channels, then rank by:
- **cardiac template SNR** = peri-R averaged amplitude ÷ standard error across
  beats (the ranking metric — measures reproducibility, which predicts reliability)
- RR implausibility fraction (a gate, not a ranking)
- beat count relative to the median across channels (sanity)

Pick the highest template SNR subject to the implausibility gate. **Re-pick per
recording and log it** — replaces the manual `hrChanIdx`, and a channel that stops
being best is a drift signal.

### 04 · Cardiac window measurement — NEW

`physio/cardiac_window.py`. Prototype already written and delivered as
`periR_window_check.m` — port or call it.

Measure the peri-R averaged band envelope **per channel per band**; mask only where
it exceeds baseline. Predicted from QRS energy distribution:

| band | 8 ms QRS | 12 ms | 20 ms |
|---|---|---|---|
| 0–2 Hz | 0.00% | 0.00% | 0.00% |
| 0.5–3 Hz | 0.00% | 0.00% | 0.02% |
| 2–50 Hz | 4.0% | 12.0% | 38.8% |
| 1–100 Hz | 23.9% | 52.8% | 89.9% |
| 100–300 Hz | **65.2%** | **32.1%** | 2.5% |
| 300–5000 Hz | 0.54% | **0.00%** | **0.00%** |

So expect: **no cardiac mask above 300 Hz, a real one in 100–300, none below 3 Hz.**
Current `step1a_blank_cardiac.m:41-43` does `D.y(blank,:) = NaN` uniformly — that is
the ~20% duty cycle being spent where it buys nothing.

Validate with the existing peri-R histogram machinery in
`processing_new/step5e_multiband_validate.m` — a flat post-mask profile passes.

Cardiac contamination is **masked, not subtracted** — see §10.2.

### 05 · Band envelopes, reference, z — NEW

`bands/envelope.py`, `bands/reference.py`, `bands/zscore.py`

```
in :  derived signals
out:  z[signal][band][t] on the common 10 ms grid; reference values
```

- RMS envelope per band with the band's own window (table in §2), on a shared grid
- **Reference = the percentile of that band's envelope over that file**, one scalar
  per (signal, band). No running window, no transfer between files, no iteration.
- `z = (E - ref) / (1.4826 * MAD(E))`, MAD taken over the same file
- **Floor the denominator** and gate dead/saturated channels before dividing —
  a flat channel gives MAD → 0 and z → ∞
- **Stream per band.** 9 channels × 6 bands × 37 M samples is ~16 GB if materialised

**Why not a running window:** at t = 0 of a recovery file there is no data on one
side, and the post-stim dynamics live in that first half-window. Worse, a running
baseline *adapts to slow change*, and in recovery the slow change is the
measurement — a rising baseline means artifacts must be larger to cross threshold
during recovery than during baseline, so the normalisation would manufacture a
condition confound.

**Known limitation, do not try to fix:** in the ENG band the envelope contains
neural activity, so the z conflates signal and noise by construction. Measured, the
clean-frame p99 rises from 2.33 to 22.14 between a quiet and an active animal. The
discriminators are classifier features (onset rate, cross-channel commonality, band
ratio), not generator parameters. **Do not tune the generator to stop over-firing
during activity changes.**

**Do not filter across NaN in the slow bands.** Interpolate-then-restore is fine
where gaps are short relative to the band's period (30 ms gaps at 300–5000 Hz).
For 0–2 and 0.5–3 Hz, a 1 s motion gap is half a cycle of the signal being
measured — process **epoch-wise** with a minimum epoch length and mark short epochs
unassessable.

### 06 · Candidate generation — NEW

`detect/candidates.py`

```
in :  z[signal][band][t], R-peaks, cardiac windows, video motion (optional)
out:  candidate list {start, stop, signals, bands, peak_z, provenance}
```

- threshold `z > 3`, hysteresis exit at `z > 1.5`
- minimum duration 20 ms; merge gaps < 100 ms
- **duration cap** = the 99th percentile of the durations in the existing 43
  recordings' labelled segments. Compute it once from
  `*_segment_indices.mat`. Anything longer is a sustained level shift, **routed to
  review, never auto-masked**
- cardiac exclusion applies to **100–300 Hz only**
- **video-assisted second threshold:** `z > 2` during windows where video motion
  exceeds its own threshold. Tag those candidates `provenance = video_assisted`

`z = 3` is a starting value. The defensible range is 2–4, bounded below by class
balance (keep the candidate set at ≥5% true positives — with ~150 real artifacts
per recording that means ≤3000 candidates, so flag ≲3% of frames) and above by
recall. Sweep it on the labelled recordings and pin it.

**Candidate count is not review burden.** The classifier judges every candidate
automatically; humans label a sample — see §10.3.

### 07 · Features — NEW, replaces Phase 1

`detect/features.py`. Replaces the Phase 1 parquet builder in `GEMSBlanking`.

Must be **channel-count independent** so one model serves the 5-channel and
9-channel cohorts: aggregate per-channel statistics (max, median, fraction above
threshold), never concatenate per-channel columns.

Restrict to **animal-invariant** quantities — ratios and cross-channel relations.
Absolute microvolts enter only as z.

- band-power ratios, especially **100–300 ÷ 300–5000** (motion carries low-band
  energy a nerve burst does not — close to a free discriminator)
- fraction of channels above threshold; mean pairwise envelope correlation; each
  channel's power ÷ the median across channels; **common-mode ÷ residual power**
- within-cuff vs across-cuff agreement — contacts 1.5 mm apart see near-identical
  artifact and different neural signal
- **onset rate**: derivative of the **log** envelope (scale-free fractional rate).
  This is where F3's sustained-vs-transient discrimination lives — level and
  derivative are correlated for brief events and decouple for sustained ones,
  which is the case that matters
- envelope derivative, slew rate, kurtosis, line length, spectral entropy and edge
- **clipping fraction** — also a hard mask criterion, see step 10 (Routing)
- the same summaries at ±100, 250, 500 ms of context
- **video** motion energy and its derivative, max over ±100/250/500 ms, time since
  last motion peak, **per ROI: headstage, tether, commutator**
- **non-physiological-velocity energy** from step 12 (new-configuration only)

### 08 · Classify — MODIFY

Reuse wholesale:
- LightGBM training, hyperopt: `GEMSBlanking:detector/retrain.py`,
  `detector-pyqt/ui/workers/hyperopt_worker.py`
- promotion, rollback, shared `current_model` pointer, `provenance.json`
- SHAP review HTMLs: `GEMSBlanking:detector/review.py`
- manifest, `held_out` flag, `detector/animal_id.py:extract_animal_letter`
- held-out evaluation: `GEMSBlanking:detector/heldout_eval.py`

**Change: the target.** One judgment per candidate event — motion / physiology /
unsure — not a per-window label. `unsure` is excluded from training and from
scoring.

**Weight `provenance = video_assisted` positives up.** They are boundary examples
by construction — the electrical threshold missed them, so their signature is weak
— and they are the route by which video improves performance on video-less
recordings. Check the cost: precision on video-less recordings with and without the
upweighting.

**No temporal smoothing** — see §10.4.

**Three training modes**, sharing every other step and differing only in training
corpus and evaluation protocol: `POOLED` (all animals but the target — mandatory,
the only mode that works with zero labels), `ADAPTED` (pooled prior + the target
animal's own labels — matches deployment, since ~50 events per new animal are
labelled anyway), `PER_ANIMAL` (target animal only).

**Validation is leave-one-animal-out**, reported per animal, never averaged.

**Comparing pooled-LOAO against per-animal-LORO is invalid** — they are different
tasks and the LORO one wins regardless. The decisive comparison is `ADAPTED` vs
`PER_ANIMAL` on matched protocol, with a learning-curve control at equal event
count, because feature normalisation addresses covariate shift and does nothing
about corpus size. `PER_ANIMAL` winning means the features are not animal-invariant
— a step 07 bug, not a result.

### 09 · Extent — NEW

`extent/tolerance.py`

For each confirmed motion event × consumer, find the interval where that band
exceeds that consumer's tolerance, padded by the **measured** filter settling time.

Run `impz` on the actual bandpass and take where it falls below 1% of peak.
`P.edgeBufferMs` is currently 5; expect 30–50 ms.

The cardiac tolerance is **operational**, not an amplitude: suppress the span,
re-run the peak detector, ask whether the beat train changed. A 1–2 ms fiducial
shift moves RMSSD and SD1, so amplitude thresholds are an approximation to this.

Slow-band extents are inherently coarse — the 0.5–3 Hz mask has ~6 s resolution and
no amount of padding logic changes that.

### 10 · Routing — NEW

`extent/routing.py`. Reject is the most expensive response and must be last.

- **correct** where contaminant and signal are separable in frequency *within that
  band*: drift is already gone from the ENG band after the 100 Hz high-pass, so
  rejecting that span destroys good data for nothing. Note the qualifier — a 0.3 Hz
  drift is *in* the slow-wave band and this route does not apply there
- **subtract** where stereotyped with known timing and residual is verified
- **reject** only where the disturbance overlaps the signal in time and frequency

**Clipping bypasses the classifier.** If the amplifier saturates the signal is
non-linear and the envelope can *understate* the damage — mask directly on the
fraction of samples at the rail.

### 11 · Mask emission and QC — MODIFY

`GEMSBlanking:detector/labeled_save.py`

**One-line fix, do this first:** `_blankmotion.mat` currently writes `yOut` with bad
ranges **zeroed**, while `processing_new/step1_bandpass.m:51` recognises only NaN
(`invalid = isnan(x)` → `fillmissing` → `filtfilt` → restore NaN). So today every
mask boundary is a hard step to zero that rings through an order-8 zero-phase filter
into adjacent valid samples. Worst for short blanks, which is what good detection
produces. Fix in `labeled_save` or in `processing_new/step0_load_data.m`.

Also emit:
- one boolean mask per consumer over the common grid
- cosine taper 5–10 ms at each boundary
- event table: `start`, `stop`, `P(motion)`, judgement, bands affected, routing
  decision, provenance
- **mask provenance**: model version, thresholds, reference values. Masks get
  regenerated as the model improves and every analysis must know which one it used
- **recording-level gate**: if retention falls below a threshold, flag the whole
  recording rather than silently emitting a heavily-masked one

QC report per recording: candidate count, blanking fraction per band, retention,
R-peak gap fraction, best HR channel, tripole weights `(a,b)`, sustained-event
queue, low-confidence velocity windows.

**Drift monitoring across weeks:** log `(a,b)`, peri-R amplitude and the noise floor
per session. Three numbers that say when an electrode is degrading, free once the
pipeline computes them.

### 12 · Velocity and direction — NEW, new-configuration animals only

`velocity/xcorr.py`

Cross-correlate contact pairs **band-limited to 300–5000 Hz first** on segments the
mask passes.

Measured tolerances: raw correlation survives artifact only to ~1× the neural
amplitude; band-limited first, to ~5×; against a saturating broadband artifact,
~1.4×. **Restricting the lag search to physiological velocities does not help and
is harmful** — it converts an obviously broken estimate into a plausible-looking
wrong one. Leave the search unrestricted.

**Emit a peak-ratio confidence with every estimate**: neural peak height ÷ zero-lag
peak height, from the same correlation. Below ~1, discard that window. Stronger
than any mask because it is derived from the quantity being estimated.

**Feed back to step 07:** energy above ~50 m/s cannot be neural, so it can only be
common-mode contamination. That is the strongest artifact feature available on the
new configuration.

Geometry from cuff v9 (measured from the STL: grooves at z = 1.00 / 2.50 / 4.00 mm):
pitch **1.50 mm**, aperture **3.00 mm**. Velocity resolution `Δv/v ≈ v/(B·L)`:
4% at 0.5 m/s, 7% at 1, 14% at 2, 35% at 5. **A C-fibre instrument, not an A-fibre
one.** Delays are never the limit — 3 ms at 1 m/s on the outer pair, 73 samples.

### Video — NEW

`video/motion.py`. Prototype delivered as `video_artifact_coincidence.py` — it
already probes the container, computes ROI motion energy, fits the sync lag and
drift by cross-correlation against existing labels, and reports coincidence.

- shared start trigger solves offset; **drift does not solve itself** — 100 ppm over
  20 minutes is 120 ms, longer than the feature window
- check for per-frame PTS first; failing that fit one linear warp per recording
- ROI the **tether and commutator**, not just the animal — the cable causes the
  artifact, and cable/connector/headstage sources enter *downstream of the
  electrode* where no montage can reject them
- feature use needs 100 ms accuracy; **labelling adjudication needs only ~1 s**, so
  that use works today with no drift correction

---

## 4. MATLAB-side changes in `processing_new`

Independent of the detector. **Each of these currently makes a better detector look
worse**, because better detection produces more, shorter, better-placed gaps.

| File | Change |
|---|---|
| `step0_load_data.m` | read per-consumer masks; NaN the `removedSegmentIdx` regions if `labeled_save` is not fixed |
| `pipeline_params.m` | `edgeBufferMs` from measured `impz`; `cardiacRemoveWinMs` and `envCardiacGuardMs` from the peri-R measurement — **both, or the censoring is inconsistent across stages** |
| `step1a_blank_cardiac.m:41-43` | per-channel, per-band windows instead of `D.y(blank,:) = NaN` |
| `step2_noise_sigma.m:82-97` | keep the Quian Quiroga estimator (correct — `std` inflates 35% at 20 spk/s where Quiroga inflates 1.9%), but take σ from a **fixed session reference**, not a 5 s running window. Measured: Quiroga inflates 12% at 100 spk/s, so a post-stim rate rise raises the 4.5σ threshold and suppresses detection of the effect being measured |
| `HR_BR_HRVAnalysis_new.m:160-165` | restore a deliberate 1–100 Hz band before `findpeaks` |
| `HR_BR_HRVAnalysis_new.m:276-287` | runs-aware successive differences for RMSSD, pNN5, SD1, SD2, SampEn, ApEn. The runs list already exists in `dfaRR_gapAware.m:24-25` and was never propagated |
| `HR_BR_HRVAnalysis_new.m:679` | `heartCountSeries` needs a valid-duration denominator |
| `HR_BR_HRVAnalysis_new.m` | promote `RR_implausibleFraction` / `br_implausibleFraction` from warnings to masks — for periodic always-present signals, an implausible rate *is* evidence of contamination |
| `slowWaveAnalysis_new.m:159-161` | pool peaks across clean runs instead of taking only the longest — two clean 28 s halves inside a 60 s window currently return NaN |
| `bulk_mixed_models.m` | coverage weights + covariate + minimum-coverage exclusion. `nRR_used`, `fr_validFrac`, `validDur_s` are all computed and none is used |

**Reuse these patterns rather than reinventing:** `dfaGapAware.m` (pooled runs),
`step5f_fano_slope.m` (epochs + rate-matched surrogates carrying identical
censoring — extend to CV2 and LV), `step5e_multiband_validate.m` (peri-R histogram
validation, and the band-C check that decides whether 100–300 Hz holds a distinct
population).

---

## 5. Labelling UI — MODIFY `detector-pyqt`

Reuse: `signal_viewer.py`, `overview_strip.py`, `region_table.py`,
`predictions_panel.py`, `review_panel.py`, `queue_panel.py`, 20-deep undo,
keyboard interval stepping.

**Change the primary interaction from free marking to candidate adjudication.**
`review_panel.py` already does candidate review (context plot, SHAP, 1/2/3 scoring,
Shift+drag to widen) — promote that to the main mode.

**NEW — recall-audit mode.** A scroll view over a randomly sampled few minutes with
candidates overlaid **and the band z-traces shown alongside the raw signal**. The
z-traces are the point: when a missed artifact is found, they distinguish
*generator blind spot* (z genuinely low — needs a new feature) from *threshold too
high* (z high but sub-threshold — needs a lower threshold). Different diagnoses,
different fixes, indistinguishable from the raw trace alone.

Also: `validateattributes(data, ..., 'finite')` in
`processing_new/browseMotionArtifacts.m:34` throws on NaN, so an already-blanked
file cannot be re-browsed. Remove if that browser is kept.

**Labelling budget:** ~500–1000 judged events across ~12 recordings spanning all
animals, one keystroke each — 1–3 hours **once**. Then ~50 uncertain events per new
animal (~15 min). Zero per production dataset.

---

## 6. Build order

**Revised 2026-09-19 after the first real-data gate run.** The recall gate is
measured on **newly labelled new-cohort data**, not the 43 old recordings: those
animals are the hardware-shorted tripole, so there are no raw contacts, and the
cross-channel agreement and common-mode logic that the generator depends on cannot
be tested there at all. The old labels also carry ±100–500 ms boundary precision.

1. **`labeled_save` zeros → NaN.** One line, and everything downstream is
   contaminated until it lands.
2. **Peri-R measurement.** Measured on animal J: a real window in 100–300 Hz
   (±45–50 ms) and a real one in 300–5000 Hz (spike-crossing rate 1.5–2.1× at
   lag 0). Band-specific, not removable.
3. **MATLAB correctness fixes** (§4). In parallel, independent of everything else.
4. **Loader, derivations, envelopes, candidates** (00A, 03, 04, 06, 07). No model.
   Tripole is naive 0.5/0.5 — the weight fit is measured to be unnecessary.
5. **Consumer tolerance derivation.** For each consumer, sweep injected amplitude
   and find where that consumer's own output first changes. The gate's pass
   condition is stated in these units and cannot be evaluated without them.
6. **Labelling UI with blind recall-audit mode.** Needed regardless of the gate
   outcome. Candidates stay hidden until the human commits, or the measurement is
   circular.
7. **Label ~10 minutes exhaustively across 2–3 new animals.** Complete spans, not
   candidate adjudication — exhaustive labelling is what makes recall computable.
8. **GATE: real recall and precision against those labels.** If recall cannot be
   met at a workable flag rate, stop and build the 1D U-Net over the same band
   envelopes — it reuses the loader, the envelopes, the UI and the labels, so
   nothing built so far is lost.
9. **Convert the 43 old recordings** to whatever event judgments survive the
   cohort difference; use them for the duration cap and a prevalence prior.
10. **Features, then train (step 08), LOAO**, three modes.
11. **Steps 09–11** — extent, routing, masks, QC.
12. **Video** (step 07 features + the step 06 second threshold).
13. **Velocity** (step 12), **stim split** (03B, at load time).

---

## 7. Acceptance tests

| Test | Pass condition |
|---|---|
| Candidate recall | ≥98% of injected synthetics produce a candidate, at a threshold yielding ≤3000 candidates/recording |
| Class balance | ≥5% of candidates are true positives on labelled recordings |
| No zeros | no exact-zero runs in any emitted `yOut`; every mask boundary tapered |
| Cardiac scope | peri-R profile flat in 300–5000 Hz and below 3 Hz after masking |
| Retention | reported per band per channel against the current all-channel full-band baseline; **ENG-band retention substantially higher than the slow bands** |
| Cross-animal | blind-test event recall on four never-seen animals, reported individually |
| **Coverage confound** | blanking fraction must **not** depend on condition. If stimulation drives movement and this is unchecked, the pipeline has automated a confound rather than fixed one |
| Downstream | `fracISIclean`, slow-wave non-NaN fraction, DFA `alpha2` availability, `nRR_used`, median `step5f` epoch length — all up; between-animal endpoint variance down |
| Velocity | peak-ratio confidence emitted with every estimate; unsigned output with a warning where `rostral_end` is missing |

---

## 8. Known gaps — deliberately out of scope

- **Stim artifact.** The `stim_recovery` file contains the stimulation period; stim
  artifacts are large, periodic, and have known timing from the `vib` / stim-monitor
  channel. They belong in the same category as cardiac — deterministic, known
  timing, per-band, excluded before candidate generation. Deferred pending
  validation.
- **Stomach referencing change.** Old animals were hardware-referenced in TDT; new
  ones are raw and referenced in postprocessing. This is good for detection
  (preserves common mode) but means `extract_mmc`'s `k = 3 × MAD` threshold and the
  slow-wave prominence criteria were tuned on a different signal. **MMC and
  slow-wave endpoints may not be comparable across cohorts unless the derivation is
  matched.** Determine what the TDT reference actually was.
- **`Raww` anti-alias setting** — unverified. Decides whether raising the low-pass
  from 5 kHz toward 8–10 kHz is possible at all. Two independent reasons to want
  it: A-fibre energy above 5 kHz, and `Δv/v ∝ 1/B`.
- **Event-level sequence prior** — only if classifier errors cluster in bouts.

---

## 9. Measured constants — do not re-derive

| Quantity | Value | Source |
|---|---|---|
| 60 Hz notch ring, in ENG band | 0.221 µV per mV of excursion | simulation of the exact filter chain |
| Bandpass gain at 60 Hz | −36.4 dB through `filtfilt` | `butter(4,[100 5000]/nyq)` |
| Smooth motion energy below 300 Hz | 98.7–99.7% | half-cosine excursions, 5–500 ms |
| Saturating step, energy above 300 Hz | 32.2% | DC step |
| QRS energy above 300 Hz | 0.00–0.54% | biphasic pulse, 8–20 ms |
| Cuff v9 pitch / aperture | 1.50 mm / 3.00 mm | STL, grooves at z = 1.00/2.50/4.00 |
| Cuff spatial filter peak | `f = v/(2d)` — C fibres 167–667 Hz | geometry |
| Moving 100→300 Hz costs | 4.9–6.1% of C-fibre power | cuff transfer function at d = 1.5 mm |
| Velocity artifact tolerance | ~1× raw, ~5× band-limited, ~1.4× broadband | Monte Carlo |
| Quiroga vs `std` at 20 spk/s | 1.9% vs 35% inflation | simulation |
| Clean-frame p99, quiet → active animal | 2.33 → 22.14 | simulation |
| R-peak RR error, raw → band-limited | 1.3% → 0.2% | simulation, severe motion |


---

## 10. Rejected approaches — do not re-implement

Each of these is a standard or previously-attempted solution that was tested and
failed, or was ruled out on a stated requirement. They are recorded **only** so
that nobody spends a day rediscovering them. Nothing in this section is to be
built.

### 10.1 Adaptive-threshold QRS detection (Pan–Tompkins)

Tested, rat-scaled (band 25–150 Hz, 20 ms integration, 90 ms refractory). Collapsed
to **8–77 of 399 beats** under heavy motion. Cause: the algorithm squares the
signal before updating its running signal-peak estimate, so a 15× excursion becomes
225× and a single artifact captures the adaptive state for many beats afterwards.
A stateless threshold degrades gracefully instead. Applies to any detector carrying
adaptive state through an artifact.

### 10.2 Cardiac template subtraction

`processing_new/step1b_remove_cardiac.m`, already out of the flow. Two diagnosed
failures: integer-sample alignment leaves ~25 µV residual at 24.4 kHz, and
`max(|seg|)` over a ±5 ms search window hijacks the fiducial onto real spikes and
deletes them. Superseded by measuring the cardiac window per band (§04), which
shows there is nothing to subtract above 300 Hz anyway.

### 10.3 Rate targeting on the candidate generator

Ruled out on requirement. It reintroduces a feedback loop between detection and
its own output, and it was motivated by a false premise — that candidate count
equals human review burden. It does not: the classifier judges every candidate and
humans label a sample.

### 10.4 HMM / Viterbi temporal smoothing

Ruled out structurally, not empirically. The classifier emits one decision per
candidate **event**, so there is no per-frame probability trace for a state model
to smooth. If errors turn out to cluster in bouts, the correct addition is an
event-level sequence prior — after that is demonstrated, not before.

### 10.5 Pooled clean-null threshold calibration

Ruled out on generalisation. Measured: the clean-frame p99 moves 2.33 → 6.36 →
22.14 between quiet and active animals, so a null calibrated on one cohort does not
transfer to an unseen animal — which is the stated goal.

### 10.6 Two-model / two-track split by cohort

Ruled out on complexity and on data. Channel-count-agnostic features (§07) let one
model serve both the 5-channel and 9-channel cohorts, and splitting would halve the
training set for each.

### 10.7 Detection on the tripole

Ruled out on principle. The tripole is *defined* by removal of the common-mode
component, which is the single best evidence for motion. Detecting there
guarantees missed artifacts. Detection uses all signals; the tripole is a consumer,
thresholded against its own σ.

### 10.8 Bottom-decile OFF statistics for the stim threshold (03B)

Ruled out on measurement. Taking `off_level` and `off_σ` from the lowest decile of
the vib envelope is truncation-biased by construction: σ came out 0.000321 against
a true 0.001119 (3.5× low), 21% of genuine OFF samples then crossed the threshold,
and onset error was −1.945 s. Anchor on the OFF population *outside* the matched
window instead — an unselected sample of the same distribution — which gave
−50 ms. The general form: never estimate a spread from a set you selected by
magnitude.

### 10.9 Bounding the stim edge refinement by the search width (03B)

Ruled out because it makes the check it exists to serve impossible. A ±2 s
refinement around a 120 s matched window can only ever report 120 ± 2 s, so a 95 s
stim passes the duration check at "120.0 s". The matched filter **locates**; a
first/last crossing followed outward, unbounded, **measures**. Any future
"tighten the refinement window" proposal is this mistake again.

### 10.10 Low duty cycle as the reason the old split failed (03B)

Recorded because it was *asserted here and then falsified*, which is the same
class of error as 10.2's premise. At the real 8% duty the old `(p20+p80)/2` rule
recovers the onset to 74 ms — `keep the largest ON segment` rescues it, because
the spurious ON segments the low threshold produces are all short. The old rule
fails on a *competing long segment* (measured +400 s), not on duty cycle. Don't
re-motivate the historical audit on duty-cycle grounds.
