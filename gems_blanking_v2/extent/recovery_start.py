"""RULING 2026-10-08 (k) 2: where each analysis's recovery starts, and the file Night 6 reads.

Each analysis starts at (RULING 2026-10-09 (i) 3 (c): stim-off is TWO times)

    max(electrical end + electrical settling + its own settling,
        mechanical end                       + its own settling)
  = max(electrical end + electrical settling, mechanical end) + its own settling.

The electrical end is the stimulator's ``AmA`` record; the electrical settling is measured
from it; recovery never starts before the mechanical end (the gate/MotorOn offset), and
motion from the mechanical stimulus is the motion masks' job. There is no shared maximum
over analyses. The first terms are per file, measured by the recovery-start script
(``electrical_settle_s`` is the ABSOLUTE time electrical end + settling); their maximum is
the file's INPUT MASK point (:func:`input_mask`, the one place it is derived), and every
start and cut is that point plus a reach. The last term is per analysis and is declared
here, once (invariant 33): :data:`ANALYSES` is the one table, and
:func:`analysis_settling` is the one place a figure is derived from it.

The table
---------
An analysis is one Night 6 consumer: one call input, one mask, one start (the analyses
are keyed by consumer name, so the wrapper trims exactly the mask it already applies).
Its outputs are listed separately because their reach differs: heart rate is an output
of the ``hrv`` run, and ``mmc_burst`` (grouped bursts) one of the ``mmc`` run - one call
with one input, so one start.

Each output is a CASCADE of stages - a filter feeding a window feeding a window - and
the part of the cascade that reaches back before ``t`` is the SUM of its stages' reach.
The analysis's settling is the MAXIMUM over its outputs (invariant 19: every output
whose edge the start must cover). A stage's reach, by its kind:

* ``filter impulse response`` - where the one-way impulse response falls below 1% of
  peak, measured on the declared design with task 13's method
  (:func:`~gems_blanking_v2.extent.tolerance.impulse_settling_s`); or, for an FIR, its
  half-length (a centred, finite response);
* ``centred window`` - half the window (RULING (k) 2);
* ``trailing window`` - the whole window (RULING (k) 2);
* ``fixed bin, reported at its centre`` - half the bin: a value stamped at the bin
  centre reaches back half a bin, exactly as a centred window does;
* ``window, part before t`` - an asymmetric window: the part before ``t``;
* ``filter at a NaN edge, measured`` - the chain's filter zero phase over her fill, as
  measured at a NaN edge (``tolerance.EDGE_SETTLING``, RULING 2026-10-09 (c));
* ``look-back (peak spacing or event grouping)`` - the whole spacing or gap ((c) 3 (d));
* ``blank around an event, extending a NaN edge`` - the whole blank: an event just past
  the edge extends the NaN by up to the blank's full width.

Whether each window is centred or trailing is read from Andrea's code and cited with
``file:line``. Her working tree is not a commit (``slowWaveAnalysis_new.m`` is modified
and ``HR_BR_HRVAnalysis_beats.m`` untracked at f93e250), so :data:`SOURCE_FILES` records
the SHA-256 of every cited processing_new file as read, and a test checks both the hashes
and that every cited line still says what the table says it does.

What counts, and what does not (RULING 2026-10-09 (c) 3, :data:`RULING_SETTLING`)
-------------------------------------------------------------------------------------
* **(b)** chained stages add up only where they run on filled or filtered data. A stage
  that skips NaN (``movmedian(..., 'omitnan')``, a window over valid samples or events
  only) contributes NOTHING at a blanked edge: it is declared ``skips_nan`` and counted
  as 0, its nominal reach still reported. Filters are the MEASURED zero-phase settling at
  a NaN edge where it was measured (spikes 7.78 ms, mmc 1.16 s; ``tolerance.EDGE_SETTLING``),
  task 13's impulse response elsewhere (slow wave's low-pass 8.17 s, the HR band).
* **(c)** the CV2 bins and the mmc delay window are kept (counted).
* **(d)** peak spacing and event grouping count where they look back in time (the beat
  detector's plausibility gate, breath troughs, slow-wave peaks, mmc grouping); the
  refractory period is negligible. RULING 2026-10-09 (d) 3: the mmc rate window looks back
  5 s and is counted, so the mmc rate's cut is its input's settling + 5 s.
* **edge guards** (step2's 10.5 ms pad, HR's 0.75 s and slow wave's 15 s edge buffers) act
  AFTER an edge and do not reach back. Slow wave's acts at every masked span, the
  electrical lead-in included, because Night 6 passes each channel's masked spans as
  ``blankIdx`` (RULING 2026-10-09 (c) 6); HR's acts only at the array ends (``blankIdx = []``).
* **epoch-wide statistics** (session sigma, detrend, averages) have no time to trim by;
  the input is masked through the input mask point (max of the electrical settling and
  the mechanical end), which keeps them free of unsettled data, and they are computed over
  [input mask point, epoch end].

Trim mode (:data:`TRIM_MODES`, a REQUIRED declaration of every Night 6 batch)
------------------------------------------------------------------------------
RULING 2026-10-09 item 6: Night 6 runs mode (B), ``mask_to_electrical_drop_outputs``,
defined PER OUTPUT VARIABLE:

* every analysis's input is masked (NaN) only through the INPUT MASK point,
  max(electrical end + electrical settling, mechanical end) (RULING 2026-10-09 (i) 3 (c)),
  one point for all;
* every time-stamped output variable is then cut at ITS OWN cut point, by its class
  (:data:`TRIM_CLASSES`, declared per variable in :data:`OUTPUT_VARS` with ``file:line``):

  - ``valid_only`` (i): computed over valid input samples or events only, so her own
    rule decides the edge windows (the rule is cited per variable, :class:`EdgeRule`):
    cut at the input mask point + the settling of its own INPUT (its cascade without
    its own window), never half its window;
  - ``filled_or_filtered`` (ii): computed on filled-in or filtered data: cut at the
    input mask point + its FULL reach (the whole cascade);

  (the input mask point + a reach is max(electrical end + electrical settling + reach,
  mechanical end + reach): RULING 2026-10-09 (i) 3 (c), exactly);
  - a trimmed variable with no class, or an unknown one, is refused by name;

* every windowed output value carries its valid fraction (:class:`ValidFraction`): her
  own variable where she already saves one, otherwise a sibling ``<name>_validFraction``
  computed from the masked input her function saw, over her own window;
* whole-epoch averages and counts of trimmed series are recomputed from the kept values
  (:class:`Recompute`); her original values are kept in the marker (:data:`TRIM_MARKER`);
* not-computed mmc events are NaN, never "no event" (action ``nan_events``: the logical
  series becomes double);
* windows stay centred: nothing is re-stamped.

Mode (A), ``mask_to_own_start`` (each analysis's input masked to its own start), is
withdrawn by the same ruling and refused by name (:data:`WITHDRAWN_TRIM_MODES`), so a
stale batch list cannot run it. Epoch-wide scalars that are not recomputed stay, recorded
as computed over [input mask point, epoch end]; an output whose time convention is
not known is left untrimmed and listed by name, never dropped silently.

An unknown stage makes the analysis's settling ``None`` with the missing stage named -
never the part that is known (invariant 19). That analysis then keeps 132 s for the
file, labelled ``fixed_132s_settling_unknown`` (the user's rule, 2026-10-08).

The file Night 6 reads
----------------------
:func:`write_recovery_starts` writes one JSON document - per file: the two stim times
(electrical and mechanical end, with their source), the electrical settling and the
mechanical end each as an exact 0-based FILE sample, and the input mask point - their
maximum - as one too (which of the two binds is recorded), per analysis the start in
seconds (for the
record) and as an exact 0-based FILE sample (MATLAB never converts seconds to samples,
invariants 15 and 22), its basis and its source, and per CUT (one per owner, output and
class the map uses) the cut point the same way - plus the table it was computed with,
the output time map, the cited files' hashes and the held files. Night 6 records the
file's path and SHA-256 in every stim_rec epoch's record.

Electrical settling only: no term here, and nothing that feeds one, uses heart rate,
firing rate or slow-wave rate (RULING (j) 6).

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from gems_blanking_v2.extent.grid import first_sample_at_or_after, seconds_to_sample
from gems_blanking_v2.extent.tolerance import (
    CONSUMER_FILTERS,
    EDGE_SETTLING,
    SPIKE_WAVEFORM_BEFORE_PEAK_S,
    ConsumerFilter,
    EdgeSettling,
    edge_settling_record,
    edge_settling_sha256,
    expected_consumers,
    impulse_settling_s,
)
from gems_blanking_v2.io.store import atomic_write_text
from gems_blanking_v2.physio.rpeaks import (
    DECIMATE_TARGET_HZ,
    GLOBAL_RR_FRACTION,
    R_MIN_S,
    RR_PLAUSIBLE_RANGE_S,
)

__all__ = [
    "ACTIONS",
    "ANALYSES",
    "BASIS_CUT",
    "BASIS_FIXED",
    "BASIS_MEASURED",
    "BEAT_SPACING_S",
    "BREATH_TROUGH_SPACING_S",
    "CONVENTIONS",
    "FIXED_START_S",
    "FIXED_START_SOURCE",
    "HELD_NO_SETTLING",
    "HELD_UNDETECTED",
    "OUTPUT_FILES",
    "OUTPUT_VARS",
    "RECOMPUTE_KINDS",
    "RULING",
    "RULING_SETTLING",
    "RULING_TIMES",
    "RULING_TRIM",
    "SCHEMA",
    "SOURCE_FILES",
    "SOURCE_LABEL",
    "TRIM_CLASSES",
    "TRIM_MARKER",
    "TRIM_MODES",
    "VALID_FRACTION_KINDS",
    "VALID_FRACTION_SUFFIX",
    "WITHDRAWN_TRIM_MODES",
    "Analysis",
    "AnalysisSettling",
    "EdgeRule",
    "Excluded",
    "Output",
    "OutputFile",
    "OutputVar",
    "Recompute",
    "Stage",
    "ValidFraction",
    "analysis_settling",
    "beat_decimation_factor",
    "cut_id",
    "file_starts",
    "input_mask",
    "output_reach_s",
    "output_times_record",
    "read_recovery_starts",
    "recovery_starts_document",
    "source_files_record",
    "stage_counted_s",
    "stage_settling_s",
    "table_record",
    "trim_cuts",
    "write_recovery_starts",
]

RULING: Final = "RULING 2026-10-08 (k) 2"
RULING_TRIM: Final = "RULING 2026-10-09 item 6"
SCHEMA: Final = "gems-blanking-v2 recovery starts v5"
"""v5 (RULING 2026-10-09 (i) 3 (c)): stim-off is two times. Every measured file carries
``electrical_end_s``, ``mechanical_end_s`` and ``mechanical_end_sample0`` beside the
electrical settling, and ``input_mask_s`` / ``input_mask_sample0`` = their maximum, with
``input_mask_binding``; ``stim_off_s`` is gone (one word, one meaning). Every start and cut
is the input mask point plus a reach. A v4 file (one stim time) is refused here and by
``night6_recovery_start``. The three samples are each the first sample at or after their
seconds field (``extent.grid.first_sample_at_or_after``; review 8 finding 5 - a v5 file
written with the mechanical end ROUNDED fails that check and is refused as stale).
v4: ``edge_settling_sha256`` (``tolerance.edge_settling_sha256``) beside
``edge_settling``; a file whose edge settlings are not this build's is stale and refused
here and by ``night6_recovery_start`` (review 2026-10-09). v3: per-file ``cuts`` (one cut
point per owner, output and trim class, RULING
2026-10-09 item 6) and the per-variable classes, valid fractions and recomputed averages
in the output time map. v2 (no cuts) is refused: its owner-start cut is not mode (B).
v2 added the electrical settling sample, the output time map and the cited files'
hashes (``source_files``: ``[{file, sha256}]`` rows)."""
SOURCE_FILES: Final[Mapping[str, str]] = {
    "HR_BR_HRVAnalysis_beats.m":
        "14f85d66de4097f5385965f5f6df31e3ce00b51bb86d9295adf19d715a9b9670",
    "extract_mmc.m": "cdc46f5aa68deb60897b7a7bdb2bac8344adf625b0d16fbdcb607825cfc9546d",
    "pipeline_params.m": "3e9b8e0a63fdaf1a0405f89f51a3cc54078c58098d268e6dab2220e1ab9c30a9",
    "slowWaveAnalysis_new.m":
        "7512caf0cf7212c368101b805a40f97dfb300ea8e33c0e34e6b05adeb472f320",
    "step1_bandpass.m": "6df4633b29c39bdf3d8831a7213af058bd83f3ed5017da4005c00373f2299159",
    "step2_noise_sigma.m": "484b14982a04ac10bf1f3b5460d90085b2bc5e9da19720ac42acae506596776d",
    "step3_detect.m": "9a097bee5c5e0fcf0ad6eb1c40150f1fa0b7a814f1ee76a59f3d8255eb3fe750",
    "step3b_envelope.m": "8e5bf31708056ea37d6abfdbb80f69a8de9b4c1b78d90e040fc949b19c9528aa",
    "step4_waveforms.m": "742220b61bd22109c782d9f8763cb3306f455b17bf5d16985409f7fcc4d79dfb",
    "step5c_modality_test.m":
        "c6333309d160fb6a3753447bd75a062ea945d4a3f1b1485382bb6fbea35b8a94",
    "step6_spike_report.m": "e0c6a1641d26195a87a36674231656da62f40e9c7e2be220cc2f720b93e368fd",
}
"""SHA-256 of the raw bytes of every processing_new file a ``file:line`` below cites, as
read on 2026-10-09 (her working tree: not a commit). The same hash Night 6 records for
each function it calls (``night6_sha256_file``)."""
SOURCE_LABEL: Final = "processing_new files by SHA-256 (extent.recovery_start.SOURCE_FILES)"


def source_files_record() -> list[dict[str, str]]:
    """Return :data:`SOURCE_FILES` as ``[{file, sha256}]`` rows, sorted by file.

    A list, not an object: MATLAB's ``jsondecode`` mangles keys such as ``extract_mmc.m``
    into field names (invariant 22), and Night 6 checks every row against the file it
    resolves at run time (``night6_recovery_start``, ``night6:recoveryStartSource``).
    """
    return [{"file": k, "sha256": v} for k, v in sorted(SOURCE_FILES.items())]

TRIM_MODES: Final = ("mask_to_electrical_drop_outputs",)
"""The Night 6 trim mode: (B), per output variable (RULING 2026-10-09 item 6). The batch
list must still name it (no default). The same name is ``matlab/night6/night6_trim_modes.m``
(a test holds them equal)."""

WITHDRAWN_TRIM_MODES: Final[Mapping[str, str]] = {
    "mask_to_own_start": (
        "trim mode (A), withdrawn by RULING 2026-10-09 item 6: Night 6 runs mode (B), "
        "mask_to_electrical_drop_outputs (input masked through the input mask point only, "
        "outputs trimmed per variable)"),
}
"""Modes refused BY NAME, with the ruling, so a stale batch list cannot run one. (A) is
removed rather than kept unselectable: code no batch can reach is untested surface, and
the commit history keeps it (``night6_trim_modes.m`` lists the same names)."""

TrimClass = Literal["valid_only", "filled_or_filtered"]
TRIM_CLASSES: Final[Mapping[str, str]] = {
    "valid_only": (
        "(i) computed over valid input samples or events only, so her own rule decides the "
        "edge windows: cut at the input mask point + the settling of its own INPUT (its "
        "cascade without its own window), never half its window"),
    "filled_or_filtered": (
        "(ii) computed on filled-in or filtered data: cut at the input mask point + its "
        "full reach (the whole cascade)"),
}
"""RULING 2026-10-09 item 6. Every trimmed variable declares one; none is guessed."""

VALID_FRACTION_SUFFIX: Final = "_validFraction"
"""A computed valid fraction is saved as the sibling ``<name>_validFraction`` (her names,
types and shapes are kept; the sibling is listed in the marker)."""

FIXED_START_S: Final = 132.0
FIXED_START_SOURCE: Final = (
    "io.stim_split protocol book chronic_2min_20min: stim_duration_s 120 + "
    "stim_tolerance_s 12 (a tolerance on the detected stim duration, task 03B, a3e129d); "
    "io.audit_pool.stim_epoch_s (b4d5479) excludes 0-132 s, so routing and the masks start "
    "every stim_rec recovery epoch there")
"""Where the current recovery epoch starts. Not a settling measurement (RULING (k))."""

BASIS_MEASURED: Final = "input_mask_plus_own_settling"
BASIS_CUT: Final = "input_mask_plus_class_reach"
"""The input mask point is max(electrical end + electrical settling, mechanical end)."""
RULING_TIMES: Final = "RULING 2026-10-09 (i) 3 (c)"
BASIS_FIXED: Final = "fixed_132s_settling_unknown"
HELD_UNDETECTED: Final = "held_stim_edges_undetected"
HELD_NO_SETTLING: Final = "held_no_electrical_settling"

StageKind = Literal["filter impulse response", "filter at a NaN edge, measured",
                    "centred window", "trailing window", "fixed bin, reported at its centre",
                    "window, part before t", "look-back (peak spacing or event grouping)",
                    "blank around an event, extending a NaN edge"]
How = Literal["impz", "impz_beat_rate", "fir_half_beat", "edge", "half", "whole", "before_t",
              "unknown"]

RULING_SETTLING: Final = "RULING 2026-10-09 (c) 3"
"""The settling rules for trim mode (B) and this table: (b) chained stages add up only where
they run on filled or filtered data - a stage that skips NaN contributes nothing at a
blanked edge; (c) the CV2 bins and the mmc delay window are kept; (d) peak spacing and
event grouping count where they look back in time, the refractory period is negligible.
RULING 2026-10-09 (d) 3: the mmc rate window looks back 5 s and is counted ((c) 3 (d))."""

BEAT_SPACING_S: Final = max(R_MIN_S, GLOBAL_RR_FRACTION * RR_PLAUSIBLE_RANGE_S[1])
"""How far the beat detector's peak spacing looks back ((c) 3 (d)): its plausibility gate
drops a beat closer than ``GLOBAL_RR_FRACTION`` (0.75) x the whole-file RR to the previous
kept one, the RR bounded by ``RR_PLAUSIBLE_RANGE_S`` (at most 0.5 s): 0.375 s, which holds
the 60 ms pass-1 distance (``R_MIN_S``) inside it. A bound: the file's own RR is shorter."""

BREATH_TROUGH_SPACING_S: Final = 1.0
"""How far her breath-trough selection looks back ((c) 3 (d)): ``findpeaks`` MinPeakDistance
``minBreathSepBeats = max(2, round((60 / maxBreathRate_bpm) fs / meanRR))`` beats
(``HR_BR_HRVAnalysis_beats.m:384-387``, maxBreathRate_bpm 170 at :221). In time that is
``max(2, round(0.3529 / RR)) * RR``, whose maximum over her plausible RR range [0.1, 0.5] s
(:314-315) is 1.0 s, at RR = 0.5 s (two beats). A test recomputes it."""

FIR_HALF_TAPS_PER_Q: Final = 10
"""``scipy.signal.decimate(ftype="fir")`` designs ``firwin(20 q + 1)`` (``half_len = 10 q``)
and, with ``zero_phase=True``, applies it centred through ``resample_poly``: a beat
fiducial reaches ``10 q`` input samples before itself (scipy 1.18, ``signal/_signaltools``)."""


def beat_decimation_factor(fs: float) -> int:
    """Return the beat detector's decimation factor at ``fs`` (``physio.rpeaks._prepare``)."""
    return max(int(fs // DECIMATE_TARGET_HZ), 1)


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage of an output's cascade, with its source.

    ``how`` says how its reach is obtained: ``impz`` (task 13's measured impulse
    settling of ``filter`` at the recording's rate), ``impz_beat_rate`` (the same
    method at the beat detector's decimated rate), ``fir_half_beat`` (the beat
    detector's decimating FIR, half-length), ``half`` / ``whole`` / ``before_t`` of
    ``window_s``, ``edge`` (the chain's filter MEASURED at a NaN edge as v2 runs it,
    ``EdgeSettling.filter_s``; RULING 2026-10-09 (c)), or ``unknown`` (a figure nobody has
    measured: the analysis is then unknown, invariant 19).

    ``skips_nan``: the stage computes over valid samples or events only, so it contributes
    NOTHING at a blanked edge (:data:`RULING_SETTLING` (b)); its nominal reach is still
    reported. ``note`` says why a stage counts or not where the rule needs saying.
    """

    what: str
    kind: StageKind
    how: How
    source: str
    anchor: str
    """A fragment of the cited source line, checked by a test (the line still says it)."""
    basis: Literal["measured", "code"]
    window_s: float | None = None
    filter: ConsumerFilter | None = None
    edge: EdgeSettling | None = None
    skips_nan: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class Output:
    """One output of an analysis's call, and the cascade of stages it is made through.

    ``key`` names it for the per-variable cuts (an identifier, unique in its analysis);
    ``own_window`` says the LAST stage is the output's own window - the one a class (i)
    variable's cut leaves out (RULING 2026-10-09 item 6: "not by half its window").
    """

    name: str
    stages: tuple[Stage, ...]
    source: str
    key: str = ""
    own_window: bool = False


@dataclass(frozen=True, slots=True)
class Excluded:
    """A term listed but NOT counted, with the reason (selection rule, guard, global)."""

    what: str
    category: Literal["selection rule", "edge guard", "epoch-wide statistic", "display only"]
    source: str
    why: str


@dataclass(frozen=True, slots=True)
class Analysis:
    """One Night 6 analysis: one consumer, one call input, one start."""

    name: str
    call: str
    outputs: tuple[Output, ...]
    excluded: tuple[Excluded, ...] = field(default=())
    note: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisSettling:
    """An analysis's own settling at ``fs``: ``None`` if any stage is unknown."""

    analysis: str
    fs: float
    settling_s: float | None
    binding_output: str | None
    per_output_s: Mapping[str, float | None]
    missing: tuple[str, ...]


# ---------------------------------------------------------------------------
# the stages, as read from the code
# ---------------------------------------------------------------------------

_HR_BR = "HR_BR_HRVAnalysis_beats.m"
_SW = "slowWaveAnalysis_new.m"
_MMC = "extract_mmc.m"
_RUN = "matlab/night6/night6_run_recording.m"
_SKIPS = "skips NaN: contributes nothing at a blanked edge (RULING 2026-10-09 (c) 3 (b))"

_BEAT_FIR = Stage(
    "beat fiducials: the beat detector's decimating FIR (to ~2 kHz), zero phase",
    "filter impulse response", "fir_half_beat",
    "gems_blanking_v2/physio/rpeaks.py:329-330 (scipy decimate ftype='fir', zero_phase)",
    'decimate(y, factor, ftype="fir", zero_phase=True)', "code")
_BEAT_BAND = Stage(
    "beat fiducials: the beat detector's 10-150 Hz order-4 band, at the decimated rate, "
    "zero phase over the linear fill: measured at a NaN edge",
    "filter at a NaN edge, measured", "edge",
    "gems_blanking_v2/physio/rpeaks.py:335-336 (butter 4, DETECT_BAND_HZ = HR_BAND; fill :319-325)",
    'sos = butter(4, [lo / nyq, min(hi, nyq * 0.9) / nyq], btype="bandpass"', "measured",
    edge=EDGE_SETTLING["hr_band"],
    note="0.2374 s at fs/12 (edgepad/hr_edge_settling_fs12.json); the one-way impz was 0.141 s "
         "(Andrea's item 2b answers: wherever the HR band enters a reach)")
_BEAT_SPACING = Stage(
    "beat peak spacing: the plausibility gate keeps one of two beats closer than 0.75 x the "
    "global RR (<= 0.75 x 0.5 s; the 60 ms pass-1 distance lies inside it)",
    "look-back (peak spacing or event grouping)", "whole",
    "gems_blanking_v2/physio/rpeaks.py:577 (gate :353-379; GLOBAL_RR_FRACTION :117, "
    "RR_PLAUSIBLE_RANGE_S :124, R_MIN distance :567)", "GLOBAL_RR_FRACTION * global_rr,",
    "code", window_s=BEAT_SPACING_S,
    note="peak spacing on filtered data counts where it looks back (RULING 2026-10-09 (c) 3 "
         "(d)); a bound - the file's own RR is shorter")
_BEATS = (_BEAT_FIR, _BEAT_BAND, _BEAT_SPACING)
"""The stored beat train (routing, frozen) is located by these stages; hrv, breathing and
mmc read it. Its other steps are selection rules and epoch-wide statistics (listed under
each analysis that reads it)."""

_SPIKE_BAND = Stage(
    "300-3000 Hz order-4 bandpass, filtfilt over NaN filled for the filter and restored: "
    "measured at a NaN edge as v2 runs it",
    "filter at a NaN edge, measured", "edge",
    "step1_bandpass.m:57 (filtfilt; NaN filled at :53 and restored at :58)",
    "xf = filtfilt(b, a, xfill);", "measured", edge=EDGE_SETTLING["spikes"],
    note="7.782 ms zero phase, either side (edgepad/settling.json); task 13's one-way impz "
         "was 5.1 ms (RULING 2026-10-09 (c) 1)")


def _skip(what: str, kind: StageKind, source: str, anchor: str, window_s: float,
          why: str) -> Stage:
    """Declare a window over valid samples or events only: reach reported, nothing counted."""
    return Stage(what, kind, "half", source, anchor, "code", window_s=window_s,
                 skips_nan=True, note=f"{_SKIPS}: {why}")


_SPIKES = Analysis(
    "spikes", "process_dataset_v2 (her process_dataset steps, matlab/night6)",
    (
        Output("spike times", (_SPIKE_BAND,), "step3_detect.m (on D.filtered)",
               key="spike_times"),
        Output("spike waveforms", (
            _SPIKE_BAND,
            Stage("waveform window before the aligned peak: wfPreMs 1 ms + wfAlignSearchMs "
                  "0.5 ms", "window, part before t", "before_t", "step4_waveforms.m:44-46",
                  "npre   = round(P.wfPreMs  * 1e-3 * fs);", "code",
                  window_s=SPIKE_WAVEFORM_BEFORE_PEAK_S,
                  note="the recovery start is the AFTER-gap side: 1.5 ms reaches back; the "
                       "2.5 ms after the peak reaches into a gap ahead, which the 10.5 ms "
                       "edge pad covers"),
        ), "step4_waveforms.m", key="spike_waveforms"),
        Output("firing rate (frBinSec 1 s bins)", (
            _SPIKE_BAND,
            _skip("firing-rate bins, frBinSec = 1 s, value at the bin centre",
                  "fixed bin, reported at its centre",
                  "step6_spike_report.m:62 (firing_rate, :140-152, t = bin centre)",
                  "firing_rate(cen, valid, N, fs, P.frBinSec)", 1.0,
                  "spikes over the bin's valid seconds (:149)"),
        ), "step6_spike_report.m:62", key="firing_rate", own_window=True),
        Output("activity envelope (envBinSec 1 s RMS bins)", (
            _SPIKE_BAND,
            Stage("artifact-excursion pad, +/- 5 ms", "centred window", "half",
                  "step3b_envelope.m:72", "artPad = round(0.005 * fs);", "code",
                  window_s=0.010, note="dilates excursions found on the filtered trace"),
            _skip("RMS bins, envBinSec = 1 s, value at the bin centre",
                  "fixed bin, reported at its centre", "step3b_envelope.m:107",
                  "t_c(b) = ((i0 + i1) / 2 - 1) / fs;", 1.0,
                  "an RMS over the bin's valid samples (:99)"),
        ), "step3b_envelope.m", key="envelope", own_window=True),
        Output("rolling CV2 (cv2WinSec 30 s bins)", (
            _SPIKE_BAND,
            Stage("CV2 bins, cv2WinSec = 30 s, value at the bin centre",
                  "fixed bin, reported at its centre", "half",
                  "step6_spike_report.m:94 (rolling_cv2, :262-271, t = edges + winSec/2)",
                  "rolling_cv2(st, isims, isClean, Tend, P.cv2WinSec)", "code",
                  window_s=30.0,
                  note="KEPT (RULING 2026-10-09 (c) 3 (c)) although it reads gap-clean ISIs "
                       "only. Andrea, 2026-10-09: cv2_roll is cut at this FULL kept reach "
                       "(electrical + input + half the 30 s bin), the analysis's binding "
                       "figure - so the bin is not an own window left out of the cut"),
        ), "step6_spike_report.m:94", key="cv2", own_window=False),
        Output("noise sigma windows (sigmaWindowSec 5 s, step 2.5 s, at the window centre)", (
            _SPIKE_BAND,
            _skip("sigma window, sigmaWindowSec = 5 s, value at its centre", "centred window",
                  "step2_noise_sigma.m:98 (window :90-91; win :42)",
                  "cWin(w) = (i0 + i1) / 2;", 5.0,
                  "a MAD over the window's valid samples (:95)"),
        ), "step2_noise_sigma.m:89-99", key="sigma_windows", own_window=True),
    ),
    (
        Excluded("step2 edge pad, edgeBufferMs 10.5 ms (night6_v2_params, RULING 2026-10-09 "
                 "(c) 1) around every invalid sample", "edge guard",
                 "step2_noise_sigma.m:41 and :75-77; matlab/night6/night6_v2_params.m",
                 "acts after an edge (it dilates every invalid sample, so the masked start "
                 "too) and covers the measured 10.28 ms (step1 7.78 + step4 2.5); it does not "
                 "reach back"),
        Excluded("wrapper check: no spike within 5 ms (NanPadMs) of a NaN", "edge guard",
                 "matlab/night6/process_dataset_v2.m:143-151", "a refusal check, not a stage"),
        Excluded("refractory 1 ms (detection dead time, re-enforced after alignment)",
                 "selection rule", "step4_waveforms.m:47 and :78",
                 "negligible (RULING 2026-10-09 (c) 3 (d))"),
        Excluded("session sigma (median of 5 s window MADs)", "epoch-wide statistic",
                 "step2_noise_sigma.m:114", "untimed: masking the input keeps it settled"),
        Excluded("Fano curve, autocorrelogram, rate PSD, bursts, modality test",
                 "epoch-wide statistic", "step6_spike_report.m:79, :85, :88, :91; "
                 "step5c_modality_test.m", "untimed: masking the input keeps them settled"),
        Excluded("Vpp over time (vppBinSec 5 s)", "display only", "step6_spike_report.m:392",
                 "binstat is called by plot_report only"),
    ),
    "The 30 s CV2 bins bind (kept by (c) 3 (c)); without them the spike start would be "
    "the input mask point + 12.8 ms (the envelope's band + excursion pad).")

_HRV = Analysis(
    "hrv", "HR_BR_HRVAnalysis_beats (stored beats), hrv run",
    (
        Output("heart rate (hrBrWinSec 60 s, centred)", (
            *_BEATS,
            _skip("heart-rate window, hrBrWinSec = 60 s (night6 params), centred",
                  "centred window", f"{_HR_BR}:835 (t0 = tc - halfHrBr; :816)",
                  "t0   = tc - halfHrBr;", 60.0,
                  "beats in the longest clean stretch, NaN at an invalid centre (:829, :844)"),
        ), f"{_HR_BR}:833-849", key="heart_rate", own_window=True),
        Output("beat count and HRV metrics (winSec 20 s, centred)", (
            *_BEATS,
            _skip("count/HRV window, winSec = 20 s (night6 params), centred",
                  "centred window", f"{_HR_BR}:873 (t0w = tc - halfWin; :817)",
                  "t0w   = tc - halfWin;", 20.0,
                  "beats over the window's valid samples and valid RR intervals (:892, :998)"),
        ), f"{_HR_BR}:871-917", key="count_hrv", own_window=True),
        Output("sample entropy (fixed 60 s, centred)", (
            *_BEATS,
            _skip("sample-entropy window, fixed 60 s, centred", "centred window",
                  f"{_HR_BR}:924 (t0se = tc - halfSampEn; :818-819)",
                  "t0se  = tc - halfSampEn;", 60.0, "valid RR intervals only (:928, :998)"),
        ), f"{_HR_BR}:920-934", key="sampen", own_window=True),
        Output("beat fiducials as read (heartlocs, RR intervals)", _BEATS,
               f"{_HR_BR}:289-299, :310", key="beats"),
        Output("heart-band trace (heartBeatSeries): her 10-150 Hz order-4 filtfilt over the "
               "linear fill, detrended over the epoch", (
                   Stage("her 10-150 Hz order-4 bandpass (night6 order 4), filtfilt over the "
                         "linear fill: measured at a NaN edge", "filter at a NaN edge, measured",
                         "edge", f"{_HR_BR}:282 (fill :262; butter :280)",
                         "yFilt    = filtfilt(sos, g, xFill);", "measured",
                         edge=EDGE_SETTLING["hr_band"],
                         note="0.2374 s, the maximum of the full-rate measurement of this filter "
                              "(0.1999 s; Night 6 passes the full fs, night6_run_recording.m:429) "
                              "and the beat detector's fs/12 one (0.2374 s), invariant 19; the "
                              "one-way impz was 0.140 s"),
               ), f"{_HR_BR}:286, :302-303", key="heart_band_trace"),
    ),
    (
        Excluded("edgeBufferSec 0.75 s (night6 params) at blank and signal edges", "edge guard",
                 f"{_HR_BR}:249-256", "acts after an edge; does not reach back. Built from "
                 "blankIdx and the array ends only, and night6 passes blankIdx = [] to HR: no "
                 "buffer at a masked start"),
        Excluded("global HRV, average heart rate", "epoch-wide statistic",
                 f"{_HR_BR}:436, :453-466", "untimed: masking the input keeps them settled"),
        Excluded("beat detector: global sigma, global RR, QRS template; the rescue search "
                 "between two accepted beats", "selection rule",
                 "gems_blanking_v2/physio/rpeaks.py:558-585 (rescue :480-538)",
                 "upstream (routing, frozen); the rescue places a beat only between two kept "
                 "beats, so it adds no look-back before the first"),
    ),
    "Heart rate is an output of this run (one call, one input), so it shares hrv's start. "
    "Every window here reads valid beats only, so the beats bind (RULING 2026-10-09 (c) 3 (b)).")

_TROUGH_SPACING = Stage(
    "breath troughs: findpeaks MinPeakDistance minBreathSepBeats (>= 2 beats), at most 1.0 s "
    "over her plausible RR range", "look-back (peak spacing or event grouping)", "whole",
    f"{_HR_BR}:387 (minBreathSepBeats :385; maxBreathRate_bpm :221; RR range :314-315)",
    "findpeaks(-candidateVals, 'MinPeakDistance', minBreathSepBeats)", "code",
    window_s=BREATH_TROUGH_SPACING_S,
    note="peak spacing counts where it looks back (RULING 2026-10-09 (c) 3 (d)); a bound")

_BREATHING = Analysis(
    "breathing", "HR_BR_HRVAnalysis_beats (stored beats), breathing run",
    (
        Output("breath rate (hrBrWinSec 60 s, centred)", (
            *_BEATS, _TROUGH_SPACING,
            _skip("breath-rate window, hrBrWinSec = 60 s (night6 params), centred",
                  "centred window", f"{_HR_BR}:835 (t0 = tc - halfHrBr; :816)",
                  "t0   = tc - halfHrBr;", 60.0,
                  "troughs in the longest clean stretch, NaN at an invalid centre (:860)"),
        ), f"{_HR_BR}:851-868", key="breath_rate", own_window=True),
        Output("breath troughs (br_locs_true): stored beats, read on the raw signal",
               (*_BEATS, _TROUGH_SPACING), f"{_HR_BR}:378-391", key="breath_troughs"),
    ),
    (
        Excluded("edgeBufferSec 0.75 s (night6 params)", "edge guard", f"{_HR_BR}:249-256",
                 "acts after an edge; does not reach back. Built from blankIdx and the "
                 "array ends only, and night6 passes blankIdx = [] to HR: no buffer at a "
                 "masked start"),
        Excluded("average breath rate", "epoch-wide statistic", f"{_HR_BR}:437",
                 "untimed: masking the input keeps it settled"),
    ))

_SW_LP = Stage("low-pass 0.15 Hz order 2 (batch_process settings, night6 params), filtfilt "
               "over NaN filled for the filter", "filter impulse response", "impz",
               f"{_SW}:133 (fill :126; butter :131)", "filtSignal = filtfilt(sos, g, xFill);",
               "measured", filter=CONSUMER_FILTERS["slow_wave"],
               note="task 13's measured 8.17 s (extent.tolerance, impz at 24414 Hz); her 15 s "
                    "edgeBufferSec covers it at every masked span, which Night 6 now passes "
                    "as blankIdx (RULING 2026-10-09 (c) 6)")
_SW_SMOOTH = Stage("gaussian smoothdata, window 5 s (night6 params), centred", "centred window",
                   "half", f"{_SW}:156", "smoothdata(filteredSignal, 1, 'gaussian', windowlen)",
                   "code", window_s=5.0, note="runs on the filtered data")
_SW_SPACING = Stage("findpeaks MinPeakDistance 6 s on the smoothed trace",
                    "look-back (peak spacing or event grouping)", "whole",
                    f"{_SW}:200 (minPeakDist_samp :165)",
                    "'MinPeakDistance', minPeakDist_samp);", "code", window_s=6.0,
                    note="peak spacing on filtered data counts where it looks back "
                         "(RULING 2026-10-09 (c) 3 (d))")

_SLOW_WAVE = Analysis(
    "slow_wave", "slowWaveAnalysis_new, one ANT channel at a time (each channel the same chain)",
    (
        Output("slow-wave trace", (_SW_LP, _SW_SMOOTH), f"{_SW}:156, :160", key="sw_trace"),
        Output("slow-wave peak times", (_SW_LP, _SW_SMOOTH, _SW_SPACING), f"{_SW}:200-205",
               key="sw_peaks"),
        Output("slow-wave rate (rateWinSec 60 s, centred)", (
            _SW_LP, _SW_SMOOTH, _SW_SPACING,
            _skip("rate window, rateWinSec = 60 s, centred", "centred window",
                  f"{_SW}:246 (winStartSamp = ctrSamp - halfWinSamp; :169, :229)",
                  "winStartSamp = max(1, ctrSamp - halfWinSamp);", 60.0,
                  "peaks over the window's pooled clean runs only (:262-273)"),
        ), f"{_SW}:225-280", key="sw_rate", own_window=True),
    ),
    (
        Excluded("edgeBufferSec 15 s at every masked span (Night 6 passes each channel's "
                 "masked spans as blankIdx, RULING 2026-10-09 (c) 6) and at the signal ends",
                 "edge guard", f"{_SW}:108-118",
                 "acts after an edge; does not reach back. It covers the ~8 s low-pass "
                 "settling at a masked edge, the electrical lead-in included"),
        Excluded("detrend over the whole epoch; avgSlowWave = mean of the rate series",
                 "epoch-wide statistic", f"{_SW}:139, :283",
                 "untimed: masking the input keeps them settled"),
    ),
    "The same chain for each ANT channel, so one start serves all three channels' masks "
    "and the shared-call shortcut ((j) 5 (a)) stays exact.")

_MMC_BLANK = Stage("cardiac blank around each stored beat, cardiacBlankMs 25 ms each side: a "
                   "beat up to 25 ms past a NaN edge extends it by up to 50 ms",
                   "blank around an event, extending a NaN edge", "whole",
                   f"{_MMC}:85 (blank :99-102)", "half = max(1, round(cardMs/1000*fs));", "code",
                   window_s=0.050,
                   note="the whole blank, not half: the band then settles from the extended "
                        "edge (re-derived under RULING 2026-10-09 (c) 3; was 25 ms)")
_MMC_BAND = Stage("2-50 Hz order-4 bandpass, filtfilt over NaN filled for the filter: "
                  "measured at a NaN edge", "filter at a NaN edge, measured", "edge",
                  f"{_MMC}:106 (fill :104)", "y = filtfilt(sos,gd,xf);", "measured",
                  edge=EDGE_SETTLING["mmc"],
                  note="1.1594 s zero phase (edgepad/mmc_edge_settling.json); task 13's impz "
                       "was 0.486 s")
_MMC_MED = Stage("moving median, sigmaWin 30 s (default), centred", "centred window", "half",
                 f"{_MMC}:231 (win :230; sigmaWin :45)",
                 "med = movmedian(y, win, 'omitnan');", "code", window_s=30.0,
                 skips_nan=True, note=f"{_SKIPS}: 'omitnan'")
_MMC_MAD = Stage("moving MAD of (y - moving median), 30 s, centred: nested in the median",
                 "centred window", "half", f"{_MMC}:232",
                 "sig = movmedian(abs(y-med), win, 'omitnan') / 0.6745;", "code",
                 window_s=30.0, skips_nan=True, note=f"{_SKIPS}: 'omitnan'")
_MMC_GROUP = Stage("event grouping by valid-time gap: burstRefractory 0.5 s (bursts; the "
                   "firing level's 0.05 s lies inside it - one key serves both levels)",
                   "look-back (peak spacing or event grouping)", "whole",
                   f"{_MMC}:245-246 (refractories :46-47)",
                   "vgap = [Inf; (cumValid(idx(2:end)) - cumValid(idx(1:end-1)))/fs];", "code",
                   window_s=0.5,
                   note="grouping counts where it looks back (RULING 2026-10-09 (c) 3 (d)): "
                        "the gap is in VALID time, so at the blanked edge nothing lies behind "
                        "it, but from the cut it looks back over valid, unsettled samples")
_MMC_EVENTS = (*_BEATS, _MMC_BLANK, _MMC_BAND, _MMC_MED, _MMC_MAD, _MMC_GROUP)
_MMC_RATE_WIN = Stage("rate window W = 10 s (default), centred: looks back 5 s",
                      "centred window", "half", f"{_MMC}:287 (centers :115)",
                      "lo = max(1, floor((centers(w)-W/2)*fs)+1);", "code", window_s=10.0,
                      note="COUNTED although it reads valid samples only: its rate window looks "
                           "back 5 s (RULING 2026-10-09 (d) 3, under (c) 3 (d))")

_MMC_A = Analysis(
    "mmc", "extract_mmc (ANT1-3 raw, stored beats)",
    (
        Output("firing and burst (mmc_burst) event times", _MMC_EVENTS,
               f"{_MMC}:109-111 (detect_crossings :220-235)", key="mmc_events"),
        # RULING 2026-10-09 (d) 3: the rate cut is the events' settling + 5 s, so the rate
        # window is NOT the class (i) own window left out of the cut (own_window False)
        Output("firing and burst (mmc_burst) rate and peak amplitude (W 10 s, centred)",
               (*_MMC_EVENTS, _MMC_RATE_WIN), f"{_MMC}:115-117 (event_rate :282-296)",
               key="mmc_rate"),
        Output("cross-channel delay (delayW 30 s on the firing rate, centred)", (
            *_MMC_EVENTS, _MMC_RATE_WIN,
            Stage("delay window delayW = 30 s (default), labelled at its centre",
                  "centred window", "half", f"{_MMC}:265 (wlen :259)",
                  "delay_t(s) = (lo+hi)/2 * S;", "code", window_s=30.0,
                  note="runs on rate rows mean-filled where invalid (:270): filled data; "
                       "KEPT (RULING 2026-10-09 (c) 3 (c))"),
        ), f"{_MMC}:119 (xchan_delay :255-272)", key="mmc_delay", own_window=True),
        Output("conditioned signal (mmc.signal): 2-50 Hz band over the cardiac-blanked fill",
               (*_BEATS, _MMC_BLANK, _MMC_BAND), f"{_MMC}:103-108, :153", key="mmc_signal"),
    ),
    (
        Excluded("avgRate, pctBlanked, peri-R and PSD QC; the global sigma fill",
                 "epoch-wide statistic", f"{_MMC}:295, :233, :123-150",
                 "untimed: masking the input keeps them settled"),
    ),
    "The moving median and MAD skip NaN and add nothing at a blanked edge (RULING 2026-10-09 "
    "(c) 3 (b)); task 13's 30 s and 15 s are withdrawn. The 30 s delay window binds.")

ANALYSES: Final[Mapping[str, Analysis]] = {
    a.name: a for a in (_SPIKES, _HRV, _BREATHING, _SLOW_WAVE, _MMC_A)}
"""THE table (invariant 33). Keyed by consumer: the Night 6 consumers of this build
(``tolerance.expected_consumers()``), a test holds the two equal."""


# ---------------------------------------------------------------------------
# the output time map (mask_to_electrical_drop_outputs)
# ---------------------------------------------------------------------------

Role = Literal["trim", "time_axis", "epoch_scalar", "recomputed", "input", "parameter",
               "container", "unknown"]
Convention = Literal["row1", "sec0", "sec_row1", "sec_xchan_delay"]
Action = Literal["nan", "drop", "nan_events"]

CONVENTIONS: Final[Mapping[str, str]] = {
    "row1": "a 1-based epoch row (a window centre may be fractional): position p = v - 1",
    "sec0": "seconds from epoch row 1, t = (row - 1) / fs: position p = v * fs",
    "sec_row1": "seconds = 1-based row / fs, one sample later than sec0 "
                "(computeValidRRIntervals RR_times = s1 / fs): position p = v * fs - 1",
    "sec_xchan_delay": "extract_mmc delay_t = (lo + hi) / 2 * S counts RATE ROWS from 1, not "
                       "rate_t: the delay window's true centre is delay_t + W/2 - S (W, S "
                       "from mmc.params; 4 s later than the stamp at W 10, S 1): position "
                       "p = (v + W/2 - S) * fs",
}
"""How a stamp ``v`` becomes a 0-based epoch position ``p`` (in samples). An entry is
BEFORE its analysis's start iff ``p < L``, ``L = start_sample0 - epoch_start_sample0``
(the number of epoch rows before the start). Comparisons are in samples (invariant 15)."""

ACTIONS: Final[Mapping[str, str]] = {
    "nan": "the value at that stamp is set NaN, her own 'not computed' marker; the stamp "
           "(the time axis) is kept, so kept values keep their index",
    "drop": "the entry is removed from its event list (spike, beat, peak, RR interval, "
            "burst), together with every list co-indexed with the same stamps",
    "nan_events": "a full-rate logical event series becomes DOUBLE (1 event, 0 no event) and "
                  "its rows before the cut are NaN: not computed, never 'no event' (RULING "
                  "2026-10-09 item 6). Readers that call logical() on it fail loudly; find() "
                  "would count a NaN row as an event, so a reader must test == 1",
}
"""Every action is recorded in the trimmed file itself (:data:`TRIM_MARKER`)."""

TRIM_MARKER: Final = "night6_recovery_trim"
"""The top-level variable Night 6 adds to every output file it trims: per trimmed variable
path, the owner, its trim class and cut, the governing edge rule, the action, the stamp
and its convention, the cut point (seconds and 0-based file sample), the first computed
epoch row and sample, the mode, per leaf how many entries are not computed and where,
and its valid-fraction variable; the sibling variables Night 6 added; and every
recomputed average with her original value. It travels with the file, so any reader can
tell 'not computed' from a value without the run record. Her variables keep their names
(and, but for ``nan_events``, their types and shapes)."""


@dataclass(frozen=True, slots=True)
class EdgeRule:
    """The rule in her code that decides a class (i) variable's edge windows.

    ``half_valid`` is True only for her >= 50 %-valid window rule; the other rules are
    cited as they are (RULING 2026-10-09 item 6 asks for every output it governs).
    ``anchor`` is checked on the cited line by a test.
    """

    rule: str
    source: str
    anchor: str
    half_valid: bool = False


VFKind = Literal["her", "hr_window", "sw_window", "mmc_rate_window", "mmc_delay_window",
                 "step2_window", "cv2_bins"]
VALID_FRACTION_KINDS: Final[Mapping[str, str]] = {
    "her": "her own variable already holds it (no sibling added)",
    "hr_window": "HR_BR_HRVAnalysis_beats: rows idx0 = max(1, round((tc - W/2) fs) + 1) .. "
                 "idx1 = min(N, round((tc + W/2) fs) + 1) (:835-838, :873-876); valid = "
                 "~invalidMask (:226, saved)",
    "sw_window": "slowWaveAnalysis_new: rows max(1, c - round(W fs / 2)) .. min(N, c + "
                 "round(W fs / 2)), c = the centre row (:229, :246-247); valid = ~invalidMask "
                 "(:97, saved); fs = the rate she ran at, her saved fs (``rate``): the epoch's, "
                 "or fs / 78 when Night 6 runs her decimated (RULING 2026-10-09 (c) 6)",
    "mmc_rate_window": "extract_mmc event_rate: rows max(1, floor((c - W/2) fs) + 1) .. min(N, "
                       "floor((c + W/2) fs)) (:287-288); valid = ~isnan(mmc.signal(:, ch)) "
                       "(:284, as the call saved it), per channel",
    "mmc_delay_window": "extract_mmc xchan_delay: firing-rate rows lo .. min(M, lo + wlen - 1), "
                        "lo = 1 : step : max(1, M - wlen + 1) (:259-264); valid = both rates "
                        "of the pair finite (:268), per pair",
    "step2_window": "step2_noise_sigma: rows (w - 1) step + 1 .. min(N, i0 + win - 1) "
                    "(:42-43, :90-91); valid = the channel's validMask (saved as invalidRuns)",
    "cv2_bins": "step6 rolling_cv2: rows whose time (row - 1) / fs is in [edges(b), "
                "edges(b + 1)) (:263, :267); valid = the channel's validMask (invalidRuns)",
}
"""How each valid fraction is computed: the fraction of the window's rows whose input
sample was valid, over the window exactly as her code defines it, from the masked input
her function saw (as saved in the output file). Every row of the stamp gets one, computed
or not."""


@dataclass(frozen=True, slots=True)
class ValidFraction:
    """Where a windowed variable's valid fraction comes from (:data:`VALID_FRACTION_KINDS`).

    ``variable`` is her own variable (kind ``her``); ``width`` the variable holding the
    window length in seconds (or ``width_s``, a constant of her code); ``validity`` the
    variable the input validity is read from; ``rate`` the variable holding the sample
    rate of the validity rows when it may differ from the epoch's (her saved ``fs``: slow
    wave can run decimated, RULING 2026-10-09 (c) 6). Paths are dotted, in the same file.
    """

    kind: VFKind
    source: str
    variable: str = ""
    width: str = ""
    width_s: float | None = None
    validity: str = ""
    rate: str = ""


RecomputeKind = Literal["mean_omitnan", "mean_well_sampled", "count", "events_per_valid_s"]
RECOMPUTE_KINDS: Final[Mapping[str, str]] = {
    "mean_omitnan": "mean(of, 1, 'omitnan') over the trimmed series (not-computed rows are "
                    "NaN): her own expression, on the kept values",
    "mean_well_sampled": "mean(of(where >= 0.5 & isfinite(finite)), 'omitnan') over the "
                         "trimmed bins: her step3b_envelope.m:110-112 rule, on the kept bins",
    "count": "numel of the trimmed list (the same struct element)",
    "events_per_valid_s": "kept events (== 1) / (kept rows whose validity sample is not NaN "
                          "/ fs), per column: her extract_mmc.m:295 definition, on the kept span",
}


@dataclass(frozen=True, slots=True)
class Recompute:
    """How a whole-epoch average or count is recomputed from the kept values."""

    kind: RecomputeKind
    of: str
    where: str = ""
    finite: str = ""
    validity: str = ""


@dataclass(frozen=True, slots=True)
class OutputFile:
    """One output file kind of a Night 6 call, matched by a regular expression on its name."""

    kind: str
    pattern: str
    call: str
    owner: str
    source: str


@dataclass(frozen=True, slots=True)
class OutputVar:
    """One variable of an output file (a dotted path into the loaded ``.mat``), classified.

    ``role``: ``trim`` (time-stamped: dropped or flagged before its cut, by ``stamp`` /
    ``convention`` / ``action``; its cut is ``trim_class`` of the owner's output keyed
    ``reach``, with the ``edge`` rule of her code for class (i) and the ``valid`` fraction
    of a windowed one), ``time_axis`` (a stamp vector, kept), ``epoch_scalar`` (no time:
    computed over [electrical settling, epoch end] and kept), ``recomputed`` (a whole-epoch
    average or count, recomputed from the kept values by ``recompute``), ``input`` (a
    description of what the call was given), ``parameter``, ``container`` (a struct whose
    fields are classified one by one), or ``unknown`` (time convention not determinable:
    left UNTRIMMED and listed by name). A struct-array container is walked element by
    element; a sibling ``stamp`` is read from the same element.
    """

    file: str
    path: str
    role: Role
    source: str
    owner: str = ""
    anchor: str = ""
    stamp: str = ""
    convention: Convention | None = None
    action: Action | None = None
    why: str = ""
    trim_class: TrimClass | None = None
    reach: str = ""
    edge: EdgeRule | None = None
    valid: ValidFraction | None = None
    recompute: Recompute | None = None


_STEP2, _STEP3, _STEP3B = "step2_noise_sigma.m", "step3_detect.m", "step3b_envelope.m"
_STEP4, _STEP6 = "step4_waveforms.m", "step6_spike_report.m"

OUTPUT_FILES: Final[Mapping[str, OutputFile]] = {f.kind: f for f in (
    OutputFile("spikes_v2", r"_spikes_v2\.mat$", "process_dataset_v2", "spikes",
               f"{_RUN} save_spikes_v2 (her D fields, -v7.3)"),
    OutputFile("HRBR", r"_HRBR\.mat$", "HR_BR_HRVAnalysis_beats", "hrv", f"{_HR_BR}:660, :663"),
    OutputFile("HRVMeasures", r"_HRVMeasures\.mat$", "HR_BR_HRVAnalysis_beats", "hrv",
               f"{_HR_BR}:661, :685"),
    OutputFile("slowWaves", r"_slowWaves_.+\.mat$", "slowWaveAnalysis_new", "slow_wave",
               f"{_SW}:340, one file per kept channel (matlab/night6/night6_keep_slow_wave.m)"),
    OutputFile("mmc", r"_mmc\.mat$", "extract_mmc", "mmc", f"{_MMC}:170"),
)}
"""Every output file Night 6 keeps from a call. A new file matching none of these is
listed as untrimmed by name (figures included); none is dropped silently."""


def _trim(file: str, path: str, source: str, anchor: str, stamp: str, convention: Convention,
          action: Action, owner: str = "", *, cls: TrimClass, reach: str,
          edge: EdgeRule | None = None, valid: ValidFraction | None = None) -> OutputVar:
    return OutputVar(file, path, "trim", source, owner, anchor, stamp, convention, action,
                     trim_class=cls, reach=reach, edge=edge, valid=valid)


def _axis(file: str, path: str, source: str, anchor: str, owner: str = "") -> OutputVar:
    return OutputVar(file, path, "time_axis", source, owner, anchor)


def _many(file: str, role: Role, paths: str, source: str, why: str = "",
          owner: str = "") -> tuple[OutputVar, ...]:
    return tuple(OutputVar(file, p, role, source, owner, why=why) for p in paths.split())


_EPOCH_WHY = "no time: computed over [input mask point, epoch end] (mode (B))"
_RECOMPUTED_WHY = ("a whole-epoch average or count of a trimmed series: recomputed from the "
                   "kept values (RULING 2026-10-09 item 6); her value is kept in the marker")


def _recomputed(file: str, path: str, source: str, rc: Recompute,
                owner: str = "") -> OutputVar:
    return OutputVar(file, path, "recomputed", source, owner, why=_RECOMPUTED_WHY, recompute=rc)


_VI: Final = "valid_only"
_FF: Final = "filled_or_filtered"

# --- the rules in her code that decide each class (i) variable's edge windows ----------
_E_SIGMA = EdgeRule("at least max(100, sigmaMinValidFrac 0.2 x window) valid samples "
                    "(pipeline_params.m:74)", f"{_STEP2}:95 (minValid :88)",
                    "if numel(s) >= minValid")
_E_ENV = EdgeRule("none per bin: an RMS whenever any sample of the bin is valid; her >= 0.5 "
                  "rule (step3b_envelope.m:110) governs only meanRMS_uv and meanExcess_uv",
                  f"{_STEP3B}:99 (vfrac :98)", "if any(sel)")
_E_FR = EdgeRule("none: a rate whenever any sample of the bin is valid (vsec > 0), on the "
                 "valid-seconds denominator", f"{_STEP6}:149 (validFrac :148)",
                 "if vsec > 0; fr(b) = cnt(b)/vsec; end")
_E_CV2 = EdgeRule("at least 3 gap-clean ISIs in the bin (an ISI is clean when no invalid "
                  "sample lies between its spikes, :49)", f"{_STEP6}:269", "if numel(dd) >= 3;")
_E_BURST = EdgeRule("bursts are built from gap-clean ISIs only", f"{_STEP6}:49 (detect_bursts "
                    ":240-252)", "isClean = vbet == span;")
_E_BEATS = EdgeRule("a stored beat in an invalid or edge-buffered sample is rejected",
                    f"{_HR_BR}:298", "validHeartPeakMask = ~invalidMask(heartlocsRaw)")
_E_RR = EdgeRule("an RR interval is kept only when no invalid sample lies between its beats",
                 f"{_HR_BR}:998 (computeValidRRIntervals)",
                 "if csumInvalid(s2) - csumInvalid(s1+1) == 0")
_E_HR = EdgeRule("the beat rate over the LONGEST clean stretch of the window, at least "
                 "hrMinStretchSec 1 s (:425) and hrMinPeaksInStretch 3 beats; NaN when the "
                 "centre sample is invalid (:829)", f"{_HR_BR}:844, :848",
                 "if stretchLen >= hrMinStretchSamp")
_E_BR = EdgeRule("the trough rate over the longest clean stretch of the window, at least "
                 "brMinStretchSec 5 s (:419) and 3 troughs; NaN when the centre sample is "
                 "invalid (:829)", f"{_HR_BR}:860, :865", "stretchLen >= brMinStretchSamp")
_E_BRLOCS = EdgeRule("a trough in an invalid or edge-buffered sample is rejected",
                     f"{_HR_BR}:391", "br_locs_true = br_locs_true(~invalidMask(br_locs_true)")
_E_COUNT = EdgeRule("none: the raw beat count over the window's valid samples (heartPeakTrain "
                    "is 0 at an invalid sample, :307), biased low in a partly invalid window; "
                    "heartCountRateSeries is her corrected form", f"{_HR_BR}:893",
                    "heartCountSeries(i) = sum(heartPeakTrain")
_E_COUNT_SEC = EdgeRule("none: the window's valid seconds", f"{_HR_BR}:894",
                        "heartCountValidSec(i) = winValidSec;")
_E_COUNT_RATE = EdgeRule("at least minValidFracForCount = 0.5 of the window valid (:802)",
                         f"{_HR_BR}:895", "if winValidSec >= minValidFracForCount",
                         half_valid=True)
_E_HRV = EdgeRule("at least minRR valid RR intervals in the window (max(3, round(0.1 x 400 x "
                  "winSec / 60)) = 13 at winSec 20, :217)", f"{_HR_BR}:902",
                  "if nRR_used(i) >= minRR")
_E_NRR = EdgeRule("none: the count of valid RR intervals in the window", f"{_HR_BR}:901",
                  "nRR_used(i) = sum(keep);")
_E_SAMPEN = EdgeRule("at least minRR_sampEn = 40 valid RR intervals in the window (:822)",
                     f"{_HR_BR}:928", "if nSE >= minRR_sampEn")
_E_MMC_RATE = EdgeRule("at least minValidFrac = 0.5 of the window valid, of W seconds (:49)",
                       f"{_MMC}:290", "if vd < minVF*W; continue; end", half_valid=True)
_E_SW_RATE = EdgeRule("pooled clean stretches of at least minStretchSec 30 s of the 60 s window "
                      "(:170) - a 50 % rule, but counted over peaks of filled, low-passed and "
                      "smoothed data (:126-156), so class (ii)", f"{_SW}:273",
                      "if pooledLen < minStretchSamp", half_valid=True)
_E_DELAY = EdgeRule("at least 4 rows with both rates finite; the other rows are set to the "
                    "mean (0 once it is removed) before the cross-correlation - a fill, so "
                    "class (ii)", f"{_MMC}:269-270", "a(~ok)=0")

# --- where each windowed variable's valid fraction comes from ------------------------
_VF_SIGMA = ValidFraction("step2_window", f"{_STEP2}:42-43, :90-91", width="sigmaWin.windowSec",
                          validity="invalidRuns")
_VF_ENV = ValidFraction("her", f"{_STEP3B}:98, :118", variable="envelope.validFrac")
_VF_FR = ValidFraction("her", f"{_STEP6}:148 (:62)", variable="metrics.fr_validFrac")
_VF_CV2 = ValidFraction("cv2_bins", f"{_STEP6}:263, :267", width="info.P.cv2WinSec",
                        validity="invalidRuns")
_VF_HRBR = ValidFraction("hr_window", f"{_HR_BR}:835-838", width="hrBrWinSec",
                         validity="invalidMask")
_VF_WIN = ValidFraction("hr_window", f"{_HR_BR}:873-876", width="winSec",
                        validity="invalidMask")
_VF_SE = ValidFraction("hr_window", f"{_HR_BR}:923-925 (seconds on RR_times; rows as :875-876)",
                       width="sampEnWinSec", validity="invalidMask")
_VF_SW = ValidFraction("sw_window", f"{_SW}:229, :246-247", width="rateWinSec",
                       validity="invalidMask", rate="fs")
_VF_MMC_RATE = ValidFraction("mmc_rate_window", f"{_MMC}:284-290", width="mmc.params.W",
                             validity="mmc.signal")
_VF_DELAY = ValidFraction("mmc_delay_window", f"{_MMC}:259-268", width="mmc.params.delayW",
                          validity="mmc.firing.rate")

_SPIKE_VARS = (
    *_many("spikes_v2", "container", "spikes envelope metrics metrics.burst sigmaWin",
           f"{_RUN} save_spikes_v2"),
    *_many("spikes_v2", "parameter",
           "fs neuralChannels channelLabels condition cardiacBlankWinMs bandInfo detectInfo "
           "signals info nSamples spikes.channel spikes.label spikes.wf_t_ms envelope.guardMs "
           "metrics.label metrics.condition metrics.wf_t_ms sigmaWin.windowSec "
           "sigmaWin.stepFrac", f"{_RUN} save_spikes_v2; her steps"),
    *_many("spikes_v2", "input", "rpeakSamples rpeakTimes invalidRuns",
           f"{_RUN} save_spikes_v2 (the beats and masked runs it was given)"),
    *_many("spikes_v2", "epoch_scalar",
           "noiseInfo modality spikes.rate_hz spikes.validSec "
           "spikes.meanWaveform spikes.stdWaveform spikes.nDetected spikes.screen "
           "metrics.validDur_s "
           "metrics.meanRate_hz metrics.medianVpp_uv metrics.medianFWHM_ms metrics.CV "
           "metrics.CV2 metrics.LV metrics.refracViolFrac metrics.nISItotal metrics.nISIclean "
           "metrics.fracISIclean metrics.fanoCanon metrics.fanoSlope metrics.fano_T "
           "metrics.fano_F metrics.acg_lag metrics.acg metrics.psd_f metrics.psd_p "
           "metrics.meanWaveform metrics.bandLo metrics.bandHi metrics.burst.hasBursts "
           "metrics.burst.thrMs metrics.burst.void metrics.burst.nBursts "
           "metrics.burst.rate_per_min metrics.burst.meanDur_s metrics.burst.meanSpikes "
           "metrics.burst.intraRate_hz metrics.burst.fracInBurst",
           f"{_STEP2}:147; {_STEP3}:91-93; {_STEP4}:119-124; {_STEP3B}:119-120; "
           f"{_STEP6}:52-91; step5c_modality_test.m:60", _EPOCH_WHY),
    _recomputed("spikes_v2", "spikes.nSpikes", f"{_STEP4}:122",
                Recompute("count", "spikes.alignedCenters")),
    _recomputed("spikes_v2", "metrics.nSpikes", f"{_STEP6}:55",
                Recompute("count", "spikes.alignedTimes")),
    _recomputed("spikes_v2", "envelope.meanRMS_uv", f"{_STEP3B}:110-111",
                Recompute("mean_well_sampled", "envelope.rms_uv", where="envelope.validFrac",
                          finite="envelope.rms_uv")),
    _recomputed("spikes_v2", "envelope.meanExcess_uv", f"{_STEP3B}:110, :112",
                Recompute("mean_well_sampled", "envelope.excess_uv",
                          where="envelope.validFrac", finite="envelope.rms_uv")),
    _axis("spikes_v2", "sigmaWin.centers", f"{_STEP2}:98 (cell per channel)",
          "cWin(w) = (i0 + i1) / 2;"),
    _trim("spikes_v2", "sigmaWin.sigma", f"{_STEP2}:96", "sigWin(w) = median(abs(s)) / 0.6745;",
          "sigmaWin.centers", "row1", "nan", cls=_VI, reach="sigma_windows", edge=_E_SIGMA,
          valid=_VF_SIGMA),
    *(_trim("spikes_v2", f"spikes.{v}", f"{_STEP3}:{ln}", a, "spikes.centers", "row1", "drop",
            cls=_FF, reach="spike_times")
      for v, ln, a in (("centers", 86, "spikes(k).centers          = locs;"),
                       ("times", 87, "spikes(k).times            = (locs - 1) / fs;"),
                       ("peakAmp_uv", 88, "spikes(k).peakAmp_uv       = peakAmp_uv;"),
                       ("threshAtSpike_uv", 89, "spikes(k).threshAtSpike_uv = threshAtSpike_uv;"),
                       ("artifactMask", 90, "spikes(k).artifactMask     = artifactMask;"))),
    *(_trim("spikes_v2", f"spikes.{v}", f"{_STEP4}:{ln}", a, "spikes.alignedCenters", "row1",
            "drop", cls=_FF, reach="spike_waveforms")
      for v, ln, a in (("waveforms", 114, "D.spikes(k).waveforms      = W;"),
                       ("alignedCenters", 115, "D.spikes(k).alignedCenters = aligned;"),
                       ("alignedTimes", 116, "D.spikes(k).alignedTimes   = (aligned - 1) / fs;"),
                       ("Vpp_uv", 117, "D.spikes(k).Vpp_uv         = Vpp;"),
                       ("width_ms", 118, "D.spikes(k).width_ms       = wid;"))),
    _axis("spikes_v2", "envelope.t", f"{_STEP3B}:107 (bin centre)",
          "t_c(b) = ((i0 + i1) / 2 - 1) / fs;"),
    *(_trim("spikes_v2", f"envelope.{v}", f"{_STEP3B}:{ln}", a, "envelope.t", "sec0", "nan",
            cls=_VI, reach="envelope", edge=_E_ENV, valid=_VF_ENV)
      for v, ln, a in (("rms_uv", 115, "D.envelope(k).rms_uv       = rms;"),
                       ("sigmaFloor_uv", 116, "D.envelope(k).sigmaFloor_uv = sigFloor;"),
                       ("excess_uv", 117, "D.envelope(k).excess_uv    = excess;"),
                       ("validFrac", 118, "D.envelope(k).validFrac    = vfrac;"))),
    _axis("spikes_v2", "metrics.fr_t", f"{_STEP6}:150 (bin centre)",
          "t(b) = ((i0+i1)/2-1)/fs;"),
    *(_trim("spikes_v2", f"metrics.{v}", f"{_STEP6}:62", "[M.fr_t, M.fr_hz, M.fr_validFrac]",
            "metrics.fr_t", "sec0", "nan", cls=_VI, reach="firing_rate", edge=_E_FR,
            valid=_VF_FR) for v in ("fr_hz", "fr_validFrac")),
    _axis("spikes_v2", "metrics.cv2_t", f"{_STEP6}:264 (bin start + winSec/2)",
          "t = edges(1:end-1)+winSec/2;"),
    _trim("spikes_v2", "metrics.cv2_roll", f"{_STEP6}:94", "[M.cv2_t, M.cv2_roll]",
          "metrics.cv2_t", "sec0", "nan", cls=_VI, reach="cv2", edge=_E_CV2, valid=_VF_CV2),
    *(_trim("spikes_v2", f"metrics.burst.{v}", f"{_STEP6}:249 (st = alignedTimes, sec0)",
            "onsets(end+1)=st(i0); offsets(end+1)=st(i1);", "metrics.burst.onsets", "sec0",
            "drop", cls=_VI, reach="spike_waveforms", edge=_E_BURST)
      for v in ("onsets", "offsets")),
)

_HR_SERIES = (("heartRateSeries", 849, "heartRateSeries(i) = hrPeaksInStretch", "hrv",
               "heart_rate", _E_HR, _VF_HRBR),
              ("heartCountSeries", 893, "heartCountSeries(i) = sum(heartPeakTrain", "hrv",
               "count_hrv", _E_COUNT, _VF_WIN),
              ("heartCountValidSec", 894, "heartCountValidSec(i) = winValidSec;", "hrv",
               "count_hrv", _E_COUNT_SEC, _VF_WIN),
              ("heartCountRateSeries", 896, "heartCountRateSeries(i) = heartCountSeries(i)",
               "hrv", "count_hrv", _E_COUNT_RATE, _VF_WIN),
              ("breathRateSeries", 866, "breathRateSeries(i) = numel(peaksInStretch)",
               "breathing", "breath_rate", _E_BR, _VF_HRBR))
_HRV_SERIES = (("hrv_series", 910, "hrv_series(i)   = hrv_val;", "count_hrv", _E_HRV, _VF_WIN),
               ("rmssd_series", 911, "rmssd_series(i) = hv.rmssd;", "count_hrv", _E_HRV,
                _VF_WIN),
               ("pnn5_series", 912, "pnn5_series(i)  = hv.pnn5;", "count_hrv", _E_HRV, _VF_WIN),
               ("sd1_series", 915, "sd1_series(i)  = sd1_val;", "count_hrv", _E_HRV, _VF_WIN),
               ("sd2_series", 916, "sd2_series(i)  = hv.sd2;", "count_hrv", _E_HRV, _VF_WIN),
               ("sampEn_series", 929, "sampEn_series(i) = sampleEntropyFast(", "sampen",
                _E_SAMPEN, _VF_SE),
               ("nRR_used", 901, "nRR_used(i) = sum(keep);", "count_hrv", _E_NRR, _VF_WIN))


def _heartlocs(file: str) -> OutputVar:
    return _trim(file, "heartlocs", f"{_HR_BR}:299",
                 "heartlocs          = heartlocsRaw(validHeartPeakMask);", "heartlocs", "row1",
                 "drop", cls=_VI, reach="beats", edge=_E_BEATS)


_HR_VARS = (
    *_many("HRBR", "parameter",
           "chanidx blankIdx edgeBufferSec winSec hrBrWinSec stepSec hrMinStretchSec "
           "hrMinPeaksInStretch", f"{_HR_BR}:663-674"),
    *_many("HRBR", "parameter",
           "brMinStretchSec brMinPeaksInStretch minBreathRate_bpm maxBreathRate_bpm",
           f"{_HR_BR}:663-674", owner="breathing"),
    *_many("HRBR", "input", "invalidMask edgeMask", f"{_HR_BR}:226, :249-256"),
    _axis("HRBR", "t", f"{_HR_BR}:213", "t = (0:N-1)' / fs;"),
    _axis("HRBR", "metrics_t", f"{_HR_BR}:792 (window centres, :834-836)",
          "metrics_t = (0 : stepSec : sigDurSec)';"),
    *(_recomputed("HRBR", v, f"{_HR_BR}:{ln}", Recompute("mean_omitnan", of), owner=o)
      for v, ln, of, o in (("avgHeartRate", 436, "heartRateSeries", "hrv"),
                           ("avgBreathRate", 437, "breathRateSeries", "breathing"),
                           ("avgHeartCount", 438, "heartCountSeries", "hrv"),
                           ("avgHeartCountRate", 442, "heartCountRateSeries", "hrv"))),
    *_many("HRBR", "epoch_scalar", "br_implausibleFraction", f"{_HR_BR}:402", _EPOCH_WHY,
           owner="breathing"),
    OutputVar("HRBR", "RR_implausibleMask", "unknown", f"{_HR_BR}:318", why=(
        "one flag per RR interval BEFORE the implausible ones are removed; the times of "
        "that list are not saved, so it has no stamp to trim by: untrimmed, listed")),
    _trim("HRBR", "heartBeatSeries", f"{_HR_BR}:303 (t :213)",
          "heartBeatSeries(invalidMask) = NaN;", "t", "sec0", "nan", cls=_FF,
          reach="heart_band_trace"),
    _heartlocs("HRBR"),
    *(_trim("HRBR", v, f"{_HR_BR}:{ln}", a, "metrics_t", "sec0", "nan", owner=o, cls=_VI,
            reach=r, edge=e, valid=vf)
      for v, ln, a, o, r, e, vf in _HR_SERIES),
    _trim("HRBR", "br_locs_true", f"{_HR_BR}:388 (filtered :391)",
          "br_locs_true = heartlocs(br_locs);", "br_locs_true", "row1", "drop",
          owner="breathing", cls=_VI, reach="breath_troughs", edge=_E_BRLOCS),
    *_many("HRVMeasures", "epoch_scalar",
           "hrv rmssd pnn5 sd1 sd2 sampEn appxEn RR_implausibleFraction dfa_alpha1 dfa_alpha2 "
           "dfa_alphaFull dfa_R2_1 dfa_R2_2 dfa_nCross dfa_nWindows dfa_excludedScales",
           f"{_HR_BR}:464-486, :317", _EPOCH_WHY),
    *_many("HRVMeasures", "input", "invalidMask edgeMask", f"{_HR_BR}:226, :249-256"),
    *_many("HRVMeasures", "parameter",
           "blankIdx edgeBufferSec sampEnWinSec winSec hrBrWinSec stepSec minRR",
           f"{_HR_BR}:685-694"),
    _axis("HRVMeasures", "t", f"{_HR_BR}:213", "t = (0:N-1)' / fs;"),
    _axis("HRVMeasures", "metrics_t", f"{_HR_BR}:792", "metrics_t = (0 : stepSec : sigDurSec)';"),
    _heartlocs("HRVMeasures"),
    *(_trim("HRVMeasures", v, f"{_HR_BR}:995, :1000 (RR_times = s1 / fs, s1 the start beat's "
            "1-based row)", "RR_times(end+1,1)     = s1 / fs;", "RR_times", "sec_row1", "drop",
            cls=_VI, reach="beats", edge=_E_RR)
      for v in ("RR_intervals", "RR_times")),
    *(_trim("HRVMeasures", v, f"{_HR_BR}:{ln}", a, "metrics_t", "sec0", "nan", cls=_VI,
            reach=r, edge=e, valid=vf)
      for v, ln, a, r, e, vf in _HRV_SERIES),
)

_SW_VARS = (
    *_many("slowWaves", "parameter",
           "blankIdx edgeBufferSec rateWinSec minStretchSec minPeaksInStretch fs window "
           "windowlen channel channelColumn maskSignal keptFrom decimation",
           f"{_SW}:340-345; matlab/night6/night6_keep_slow_wave.m (decimation: only when "
           "Night 6 ran her decimated - the factor, both rates, her own peak rows, the "
           "source spans; slowWavePeakLocs is then mapped back to epoch rows)"),
    *_many("slowWaves", "input", "invalidMask edgeMask", f"{_SW}:97, :111-118"),
    _recomputed("slowWaves", "avgSlowWave", f"{_SW}:283",
                Recompute("mean_omitnan", "slowWaveRateSeries")),
    *_many("slowWaves", "epoch_scalar", "sw_implausibleFraction", f"{_SW}:212", _EPOCH_WHY),
    _axis("slowWaves", "t", f"{_SW}:89", "t          = (0:N-1)' / fs;"),
    _axis("slowWaves", "slowWaveRateTime", f"{_SW}:178 (the window's centre sample, :246)",
          "slowWaveRateTime = t(rateT_idx);"),
    _trim("slowWaves", "slowWaveTimeSeries", f"{_SW}:160",
          "slowWaveTimeSeries(invalidMask, :) = NaN;", "t", "sec0", "nan", cls=_FF,
          reach="sw_trace"),
    _trim("slowWaves", "slowWaveRateSeries", f"{_SW}:279",
          "slowWaveRateSeries(ti, ci) = pooledPeaks", "slowWaveRateTime", "sec0", "nan",
          cls=_FF, reach="sw_rate", edge=_E_SW_RATE, valid=_VF_SW),
    _trim("slowWaves", "slowWavePeakLocs", f"{_SW}:205", "slowWavePeakLocs{ci} = locs;",
          "slowWavePeakLocs", "row1", "drop", cls=_FF, reach="sw_peaks"),
)

_MMC_VARS = (
    *_many("mmc", "container", "mmc mmc.firing mmc.burst mmc.qc", f"{_MMC}:151-166"),
    *_many("mmc", "parameter",
           "mmc.fs mmc.pairs mmc.params mmc.firing.refractory mmc.burst.refractory "
           "mmc.qc.srcFile mmc.qc.dataVar mmc.qc.gastricCols mmc.qc.periR_t", f"{_MMC}:124-166"),
    OutputVar("mmc", "mmc.qc.rpeakT", "input", f"{_MMC}:125", why="the beats it was given"),
    *(_recomputed("mmc", f"mmc.{lvl}.avgRate", f"{_MMC}:295",
                  Recompute("events_per_valid_s", f"mmc.{lvl}.events", validity="mmc.signal"))
      for lvl in ("firing", "burst")),
    *_many("mmc", "epoch_scalar",
           "mmc.qc.meanHR mmc.qc.pctBlanked "
           "mmc.qc.rateNanFrac mmc.qc.nFirings mmc.qc.nBursts mmc.qc.periR_raw "
           "mmc.qc.periR_cond mmc.qc.psd_f mmc.qc.psd_raw mmc.qc.psd_cond",
           f"{_MMC}:126-146", _EPOCH_WHY),
    _axis("mmc", "mmc.t", f"{_MMC}:152 (t :65)", "mmc.fs = fs; mmc.t = t;"),
    _axis("mmc", "mmc.rate_t", f"{_MMC}:154 (centers :115, windows :287-288)",
          "mmc.rate_t = centers;"),
    _axis("mmc", "mmc.delay_t", f"{_MMC}:159 (delay_t :265: RATE ROWS, not rate_t)",
          "mmc.delay_t = delay_t; mmc.delay = delay;"),
    _trim("mmc", "mmc.signal", f"{_MMC}:153", "mmc.signal = single(cond);", "mmc.t", "sec0",
          "nan", cls=_FF, reach="mmc_signal"),
    *(_trim("mmc", f"mmc.{lvl}.events", f"{_MMC}:302 (ev_bool, :155-157)",
            "for ch = 1:3; ev(evIdx{ch},ch) = true; end", "mmc.t", "sec0", "nan_events",
            cls=_FF, reach="mmc_events")
      for lvl in ("firing", "burst")),
    *(_trim("mmc", f"mmc.{lvl}.{v}", f"{_MMC}:{ln}", a, "mmc.rate_t", "sec0", "nan", cls=_VI,
            reach="mmc_rate", edge=_E_MMC_RATE, valid=_VF_MMC_RATE)
      for lvl in ("firing", "burst")
      for v, ln, a in (("rate", 292, "rate(w,ch) = sum(inw)/vd;"),
                       ("peakAmp", 293, "peakAmp(w,ch) = mean(pka(inw));"))),
    _trim("mmc", "mmc.delay", f"{_MMC}:272 (window :264-265)",
          "delay(s,p) = lags(mi)*S;", "mmc.delay_t", "sec_xchan_delay", "nan", cls=_FF,
          reach="mmc_delay", edge=_E_DELAY, valid=_VF_DELAY),
)

OUTPUT_VARS: Final[tuple[OutputVar, ...]] = (*_SPIKE_VARS, *_HR_VARS, *_SW_VARS, *_MMC_VARS)
"""THE per-output time map (invariant 33): every variable of every kept output file,
classified, and every trimmed one with its class, its cut and (if windowed) its valid
fraction. An ``owner`` left empty is the file's owner (:data:`OUTPUT_FILES`)."""


XCHAN_DELAY_PARAMS: Final = ("mmc.params.W", "mmc.params.S")
"""Where ``sec_xchan_delay`` reads W and S (``extract_mmc.m:161``)."""


def _owner(v: OutputVar) -> str:
    return v.owner or OUTPUT_FILES[v.file].owner


def cut_id(owner: str, key: str, trim_class: str) -> str:
    """Return the one name of a cut, ``<owner>.<output key>.<class>`` (never a JSON key)."""
    return f"{owner}.{key}.{trim_class}"


def _root_declared(decl: Mapping[str, OutputVar], path: str) -> bool:
    return path.partition(".")[0] in decl or path in decl


def _check_trim_var(kind: str, path: str, v: OutputVar, decl: Mapping[str, OutputVar],
                    owner: str) -> dict[str, Any]:
    """Validate a trimmed variable's class, cut, edge rule and valid fraction; its row."""
    where = f"{kind}/{path}"
    if v.trim_class is None:
        msg = f"{where} has no trim class: declare one of {sorted(TRIM_CLASSES)} (never guessed)"
        raise ValueError(msg)
    if v.trim_class not in TRIM_CLASSES:
        msg = f"{where}: unknown trim class {v.trim_class!r}"
        raise ValueError(msg)
    keys = [o.key for o in ANALYSES[owner].outputs]
    if not v.reach or v.reach not in keys:
        msg = f"{where}: reach {v.reach!r} is not an output key of {owner} ({keys})"
        raise ValueError(msg)
    if v.trim_class == "valid_only" and v.edge is None:
        msg = f"{where}: a valid_only variable must cite the rule that decides its edge windows"
        raise ValueError(msg)
    row: dict[str, Any] = {"trim_class": v.trim_class, "output_key": v.reach,
                           "cut": cut_id(owner, v.reach, v.trim_class)}
    if v.edge is not None:
        row["edge"] = {"rule": v.edge.rule, "source": v.edge.source,
                       "half_valid": v.edge.half_valid}
    if v.valid is not None:
        row["valid_fraction"] = _check_valid_fraction(kind, path, v.valid, decl)
    return row


def _check_valid_fraction(kind: str, path: str, vf: ValidFraction,
                          decl: Mapping[str, OutputVar]) -> dict[str, Any]:
    """Validate where a windowed variable's valid fraction comes from; its record."""
    where = f"{kind}/{path}"
    if vf.kind not in VALID_FRACTION_KINDS:
        msg = f"{where}: unknown valid-fraction kind {vf.kind!r}"
        raise ValueError(msg)
    d: dict[str, Any] = {"kind": vf.kind, "source": vf.source}
    if vf.kind == "her":
        if vf.variable not in decl:
            msg = f"{where}: its valid fraction {vf.variable!r} is not declared in {kind}"
            raise ValueError(msg)
        return d | {"variable": vf.variable}
    sib = path + VALID_FRACTION_SUFFIX
    if sib in decl:
        msg = f"{where}: the sibling {sib!r} would overwrite a declared variable"
        raise ValueError(msg)
    for ref in (vf.width, vf.validity, vf.rate):
        if ref and not _root_declared(decl, ref):
            msg = f"{where}: valid fraction reads {ref!r}, not declared in {kind}"
            raise ValueError(msg)
    if not vf.validity or not (vf.width or vf.width_s is not None):
        msg = f"{where}: a computed valid fraction needs its window and its validity"
        raise ValueError(msg)
    d |= {"sibling": sib, "validity": vf.validity}
    if vf.width:
        d["width"] = vf.width
    if vf.width_s is not None:
        d["width_s"] = vf.width_s
    if vf.rate:
        d["rate"] = vf.rate
    return d


def _check_recomputed(kind: str, path: str, v: OutputVar,
                      decl: Mapping[str, OutputVar]) -> dict[str, Any]:
    rc = v.recompute
    if rc is None or rc.kind not in RECOMPUTE_KINDS:
        msg = f"{kind}/{path}: a recomputed variable needs a known recompute kind"
        raise ValueError(msg)
    of = decl.get(rc.of)
    if of is None or of.role != "trim":
        msg = f"{kind}/{path}: it is recomputed from {rc.of!r}, not a trimmed variable of {kind}"
        raise ValueError(msg)
    d: dict[str, Any] = {"kind": rc.kind, "of": rc.of}
    for name, ref in (("where", rc.where), ("finite", rc.finite), ("validity", rc.validity)):
        if ref:
            if ref not in decl:
                msg = f"{kind}/{path}: {name} {ref!r} is not declared in {kind}"
                raise ValueError(msg)
            d[name] = ref
    if rc.kind == "mean_well_sampled" and not (rc.where and rc.finite):
        msg = f"{kind}/{path}: mean_well_sampled needs where and finite"
        raise ValueError(msg)
    if rc.kind == "events_per_valid_s" and not rc.validity:
        msg = f"{kind}/{path}: events_per_valid_s needs its validity"
        raise ValueError(msg)
    return {"recompute": d}


def _role_fields(kind: str, path: str, v: OutputVar,
                 decl: Mapping[str, OutputVar]) -> dict[str, Any]:
    """Validate a variable's role-specific declaration; the fields its row adds."""
    if v.role == "trim":
        st = decl.get(v.stamp)
        if st is None or st.role not in ("time_axis", "trim"):
            msg = f"{kind}/{path}: stamp {v.stamp!r} is not a declared time axis"
            raise ValueError(msg)
        if v.convention not in CONVENTIONS or v.action not in ACTIONS:
            msg = f"{kind}/{path}: convention or action missing"
            raise ValueError(msg)
        return ({"stamp": v.stamp, "convention": v.convention, "action": v.action}
                | _check_trim_var(kind, path, v, decl, _owner(v)))
    if v.role == "recomputed":
        return _check_recomputed(kind, path, v, decl)
    if v.trim_class is not None or v.valid is not None or v.recompute is not None:
        msg = f"{kind}/{path}: a {v.role} variable carries a trim class or a fraction"
        raise ValueError(msg)
    return {}


def output_times_record() -> dict[str, Any]:
    """Return the output time map as data, for the starts document (Night 6 applies it).

    Checked here (a malformed map is refused, never half applied): every file and owner
    is known, every path is unique within its file and its parent is a declared
    container, and every ``trim`` names a stamp that is itself a declared time axis or a
    trimmed list, a known convention and a known action, a trim class (RULING 2026-10-09
    item 6: refused by name when absent or unknown), an output of its owner to be cut by,
    the edge rule of a class (i) variable, and a valid fraction whose inputs are declared;
    every ``recomputed`` names a trimmed variable of its file.
    """
    analyses = set(ANALYSES)
    by_file: dict[str, dict[str, OutputVar]] = {k: {} for k in OUTPUT_FILES}
    for v in OUTPUT_VARS:
        if v.file not in OUTPUT_FILES:
            msg = f"{v.path}: unknown output file {v.file!r}"
            raise ValueError(msg)
        if _owner(v) not in analyses:
            msg = f"{v.file}/{v.path}: owner {_owner(v)!r} is not an analysis"
            raise ValueError(msg)
        if v.path in by_file[v.file]:
            msg = f"{v.file}/{v.path} is declared twice"
            raise ValueError(msg)
        by_file[v.file][v.path] = v
    rows: list[dict[str, Any]] = []
    for kind, decl in by_file.items():
        for path, v in decl.items():
            parent = path.rpartition(".")[0]
            if parent and (parent not in decl or decl[parent].role != "container"):
                msg = f"{kind}/{path}: its parent {parent!r} is not a declared container"
                raise ValueError(msg)
            row: dict[str, Any] = {"file": kind, "path": path, "role": v.role,
                                   "owner": _owner(v), "source": v.source}
            if v.why:
                row["why"] = v.why
            row |= _role_fields(kind, path, v, decl)
            rows.append(row)
            if path.split(".")[0] == TRIM_MARKER:
                msg = f"{kind}/{path}: {TRIM_MARKER!r} is the trim marker, not her variable"
                raise ValueError(msg)
    return {"conventions": dict(CONVENTIONS), "actions": dict(ACTIONS),
            "trim_classes": dict(TRIM_CLASSES), "ruling": RULING_TRIM,
            "valid_fraction_kinds": dict(VALID_FRACTION_KINDS),
            "valid_fraction_suffix": VALID_FRACTION_SUFFIX,
            "recompute_kinds": dict(RECOMPUTE_KINDS),
            "marker_variable": TRIM_MARKER,
            "xchan_delay_params": list(XCHAN_DELAY_PARAMS),
            "files": [{"kind": f.kind, "pattern": f.pattern, "call": f.call, "owner": f.owner,
                       "source": f.source} for f in OUTPUT_FILES.values()],
            "vars": rows}


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


def stage_settling_s(stage: Stage, fs: float) -> float | None:
    """Return the part of ``stage`` reaching back before ``t``, s; ``None`` if unknown."""
    if stage.how == "unknown":
        return None
    if stage.how in ("impz", "impz_beat_rate"):
        if stage.filter is None:
            msg = f"{stage.what}: a filter stage names no filter"
            raise ValueError(msg)
        rate = fs if stage.how == "impz" else fs / beat_decimation_factor(fs)
        return float(impulse_settling_s(stage.filter, float(rate)))
    if stage.how == "fir_half_beat":
        return FIR_HALF_TAPS_PER_Q * beat_decimation_factor(fs) / float(fs)
    if stage.how == "edge":
        if stage.edge is None:
            msg = f"{stage.what}: a measured edge stage names no measurement"
            raise ValueError(msg)
        return float(stage.edge.filter_s)
    if stage.window_s is None or not (math.isfinite(stage.window_s) and stage.window_s > 0):
        msg = f"{stage.what}: a window stage needs a finite positive window_s"
        raise ValueError(msg)
    if stage.how == "half":
        return stage.window_s / 2.0
    return float(stage.window_s)  # "whole" and "before_t": the stated part, all of it


def stage_counted_s(stage: Stage, fs: float) -> float | None:
    """Return what ``stage`` adds to a cascade at a blanked edge, s; ``None`` if unknown.

    :data:`RULING_SETTLING` (b): a stage that skips NaN adds nothing (0.0) - its nominal
    reach (:func:`stage_settling_s`) is still reported, never silently dropped. Every
    other stage adds its reach.
    """
    if stage.skips_nan:
        return 0.0
    return stage_settling_s(stage, fs)


def analysis_settling(name: str, fs: float,
                      table: Mapping[str, Analysis] | None = None) -> AnalysisSettling:
    """``max`` over outputs of the ``sum`` over each cascade; ``None`` if any stage is unknown.

    An analysis absent from the table is unknown too (named in ``missing``), never 0.
    """
    tab = ANALYSES if table is None else table
    a = tab.get(name)
    if a is None:
        return AnalysisSettling(name, float(fs), None, None, {}, (f"{name}: not in the table",))
    per: dict[str, float | None] = {}
    missing: list[str] = []
    for out in a.outputs:
        vals = [stage_counted_s(s, fs) for s in out.stages]
        gaps = [f"{name}/{out.name}: {s.what}" for s, v in zip(out.stages, vals, strict=True)
                if v is None]
        missing += gaps
        per[out.name] = None if gaps else float(sum(v for v in vals if v is not None))
    if not a.outputs:
        missing.append(f"{name}: no output declared")
    if missing:
        return AnalysisSettling(name, float(fs), None, None, per, tuple(missing))
    bind = max(per, key=lambda k: float(per[k] or 0.0))
    return AnalysisSettling(name, float(fs), per[bind], bind, per, ())


def output_reach_s(owner: str, key: str, trim_class: str, fs: float,
                   table: Mapping[str, Analysis] | None = None
                   ) -> tuple[float | None, tuple[str, ...]]:
    """How far back a variable of class ``trim_class`` on output ``key`` reaches, s.

    RULING 2026-10-09 item 6: class (i) ``valid_only`` counts the settling of its own
    INPUT - the output's cascade without its own window (``Output.own_window``), so never
    half that window; class (ii) ``filled_or_filtered`` counts the full cascade. Returns
    ``(None, missing)`` when a counted stage is unknown, or the owner or the output is not
    in the table (invariant 19: never the known part, never 0).
    """
    if trim_class not in TRIM_CLASSES:
        msg = f"{owner}/{key}: unknown trim class {trim_class!r}"
        raise ValueError(msg)
    tab = ANALYSES if table is None else table
    a = tab.get(owner)
    if a is None:
        return None, (f"{owner}: not in the table",)
    hit = [o for o in a.outputs if o.key == key]
    if len(hit) != 1:
        return None, (f"{owner}: {len(hit)} outputs keyed {key!r}",)
    out = hit[0]
    stages = out.stages[:-1] if trim_class == "valid_only" and out.own_window else out.stages
    vals = [stage_counted_s(s, fs) for s in stages]
    gaps = tuple(f"{owner}/{out.name}: {s.what}" for s, v in zip(stages, vals, strict=True)
                 if v is None)
    if gaps:
        return None, gaps
    return float(sum(v for v in vals if v is not None)), ()


def trim_cuts() -> list[tuple[str, str, str]]:
    """Every ``(owner, output key, class)`` the map cuts by, sorted (one cut point each)."""
    out = {(_owner(v), v.reach, str(v.trim_class)) for v in OUTPUT_VARS if v.role == "trim"}
    return sorted(out)


def input_mask(*, fs: float, electrical_settle_s: float,
               mechanical_end_s: float) -> dict[str, Any]:
    """Return the file's input mask point: max(electrical end + settling, mechanical end).

    The ONE place it is derived (invariant 33). Returned in seconds and as exact 0-based
    FILE samples, each the FIRST SAMPLE AT OR AFTER its time (``ceil(t fs)``, the rule's
    convention for an event time: :func:`~gems_blanking_v2.extent.grid.first_sample_at_or_after`,
    the one conversion - review 8 finding 5; ``round`` could put the mechanical end half a
    sample before the gate closed). The conversion is monotone, so the sample of the
    maximum is the maximum of the two samples - all three are written, and the document
    asserts the identity and each conversion - with which time binds: ``electrical``,
    ``mechanical``, or ``both`` on the same sample.
    """
    ke = first_sample_at_or_after(float(electrical_settle_s), fs)
    km = first_sample_at_or_after(float(mechanical_end_s), fs)
    t = max(float(electrical_settle_s), float(mechanical_end_s))
    k = first_sample_at_or_after(t, fs)
    if k != max(ke, km):
        msg = f"input mask sample {k} is not max({ke}, {km})"
        raise AssertionError(msg)
    bind = "both" if ke == km else ("electrical" if ke > km else "mechanical")
    return {"input_mask_s": t, "input_mask_sample0": k, "input_mask_binding": bind,
            "electrical_settle_sample0": ke, "mechanical_end_sample0": km}


def _held_record(*, session: str, fs: float | None, electrical_end_s: float | None,
                 mechanical_end_s: float | None, electrical_settle_s: float | None,
                 times_source: str) -> dict[str, Any] | None:
    """Return the held record of a file that gets no starts, or ``None`` if it gets them."""
    held: dict[str, Any] = {"session": session, "times_source": times_source}
    if fs is not None:
        if not (math.isfinite(fs) and fs > 0):
            msg = f"{session}: fs must be finite and positive, got {fs}"
            raise ValueError(msg)
        held["fs"] = float(fs)
    for key, v in (("electrical_end_s", electrical_end_s),
                   ("mechanical_end_s", mechanical_end_s)):
        if v is not None:
            held[key] = float(v)
    if electrical_end_s is None or mechanical_end_s is None:
        gone = [k for k, v in (("electrical", electrical_end_s),
                               ("mechanical", mechanical_end_s)) if v is None]
        return {**held, "basis": HELD_UNDETECTED,
                "why": f"stim end not determined ({' and '.join(gone)}): the start is never "
                       f"assumed (invariant 41, {RULING_TIMES})"}
    if electrical_settle_s is None:
        return {**held, "basis": HELD_NO_SETTLING,
                "why": "the electrical-only rule found no settling (RULING 2026-10-09 (i) 3)"}
    return None


def file_starts(*, session: str, fs: float | None, electrical_end_s: float | None,
                mechanical_end_s: float | None, electrical_settle_s: float | None,
                times_source: str, electrical_source: str,
                analyses: Sequence[str] | None = None,
                table: Mapping[str, Analysis] | None = None,
                fixed_start_s: float = FIXED_START_S) -> dict[str, Any]:
    """Return one file's record: per-analysis starts, or the reason the file is held.

    RULING 2026-10-09 (i) 3 (c): ``electrical_end_s`` (the stimulator's ``AmA`` record) and
    ``mechanical_end_s`` (the gate/MotorOn offset) are BOTH required - from file start, s.
    ``electrical_settle_s`` is the ABSOLUTE time at which the electrical-only rule found the
    file settled, measured from the electrical end (so never before it). The input mask
    point is ``max(electrical_settle_s, mechanical_end_s)`` (:func:`input_mask`); each
    analysis starts there + its own settling, each cut there + its class reach.
    A file whose stim times were not determined, or whose signal never settles, is HELD as
    a whole (invariant 41): no start rows, so Night 6 refuses it by name. Within a measured
    file, an analysis whose own settling is unknown keeps ``fixed_start_s``, labelled, with
    the missing figure named; the others get their own starts. Missing values are absent
    keys. ``fs`` (the file's own rate) may be ``None`` only for a file held before it was
    read.
    """
    names = list(ANALYSES if analyses is None else analyses)
    held = _held_record(session=session, fs=fs, electrical_end_s=electrical_end_s,
                        mechanical_end_s=mechanical_end_s,
                        electrical_settle_s=electrical_settle_s, times_source=times_source)
    if held is not None:
        return held
    if fs is None or electrical_end_s is None or mechanical_end_s is None \
            or electrical_settle_s is None:  # for the type checker: _held_record covers these
        msg = f"{session}: a measured file needs its fs"
        raise ValueError(msg)
    if electrical_settle_s < electrical_end_s:
        msg = (f"{session}: electrical settling {electrical_settle_s} s precedes the "
               f"electrical end {electrical_end_s} s")
        raise ValueError(msg)
    im = input_mask(fs=float(fs), electrical_settle_s=float(electrical_settle_s),
                    mechanical_end_s=float(mechanical_end_s))
    t_mask = float(im["input_mask_s"])
    basis_text = (f"max(electrical end + electrical settling ({electrical_source}), "
                  f"mechanical end) ({times_source}; {RULING_TIMES})")
    fixed0 = seconds_to_sample(fixed_start_s, fs)
    rows: list[dict[str, Any]] = []
    for name in names:
        s = analysis_settling(name, fs, table)
        if s.settling_s is None:
            rows.append({"analysis": name, "start_s": float(fixed_start_s),
                         "start_sample0": fixed0, "basis": BASIS_FIXED,
                         "missing_settling": list(s.missing),
                         "source": f"{RULING}: own settling unknown, keeps {fixed_start_s:g} s "
                                   "(user rule 2026-10-08; never a partial maximum)"})
            continue
        start = t_mask + s.settling_s
        k0 = seconds_to_sample(start, fs)
        row: dict[str, Any] = {
            "analysis": name, "start_s": start, "start_sample0": k0, "basis": BASIS_MEASURED,
            "own_settling_s": s.settling_s, "binding_output": s.binding_output,
            "source": (f"{RULING}: {basis_text} + own settling "
                       f"(extent.recovery_start.ANALYSES[{name!r}], {SOURCE_LABEL})")}
        if k0 < fixed0:
            row["relation"] = "earlier_than_fixed: runs from the fixed start; early part " \
                              "deferred to the add-on"
        elif k0 == fixed0:
            row["relation"] = "at_fixed_start"
        else:
            row["relation"] = "later_than_fixed: trimmed at Night 6"
        rows.append(row)
    cuts = _file_cuts(float(fs), t_mask, float(fixed_start_s), basis_text, table)
    return {"session": session, "fs": float(fs), "times_source": times_source,
            "electrical_end_s": float(electrical_end_s),
            "mechanical_end_s": float(mechanical_end_s),
            "electrical_settle_s": float(electrical_settle_s),
            "electrical_s": float(electrical_settle_s) - float(electrical_end_s),
            **im, "analyses": rows, "cuts": cuts}


def _file_cuts(fs: float, input_mask_s: float, fixed_start_s: float, basis_text: str,
               table: Mapping[str, Analysis] | None) -> list[dict[str, Any]]:
    """One row per cut the map uses: the input mask point + the class reach, or 132 s."""
    fixed0 = seconds_to_sample(fixed_start_s, fs)
    cuts: list[dict[str, Any]] = []
    for owner, key, cls in trim_cuts():
        reach, missing = output_reach_s(owner, key, cls, fs, table)
        c: dict[str, Any] = {"cut": cut_id(owner, key, cls), "owner": owner,
                             "output_key": key, "trim_class": cls}
        if reach is None:
            c |= {"start_s": fixed_start_s, "start_sample0": fixed0,
                  "basis": BASIS_FIXED, "missing_settling": list(missing),
                  "source": f"{RULING_TRIM}: reach unknown, keeps {fixed_start_s:g} s "
                            "(user rule 2026-10-08; never a partial reach)"}
        else:
            start = input_mask_s + reach
            c |= {"start_s": start, "start_sample0": seconds_to_sample(start, fs),
                  "basis": BASIS_CUT, "reach_s": reach,
                  "source": (f"{RULING_TRIM}: {basis_text} + the {cls} reach of "
                             f"{owner}/{key} (extent.recovery_start, {SOURCE_LABEL})")}
        cuts.append(c)
    return cuts


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------


def table_record(fs: float, table: Mapping[str, Analysis] | None = None) -> list[dict[str, Any]]:
    """Return the table as data, with every stage's reach at ``fs`` (absent when unknown)."""
    tab = ANALYSES if table is None else table
    out: list[dict[str, Any]] = []
    for name in sorted(tab):
        a = tab[name]
        s = analysis_settling(name, fs, tab)
        rec: dict[str, Any] = {"analysis": name, "call": a.call, "note": a.note,
                               "missing": list(s.missing), "outputs": [],
                               "excluded": [{"what": e.what, "category": e.category,
                                             "source": e.source, "why": e.why}
                                            for e in a.excluded]}
        if s.settling_s is not None:
            rec["settling_s"] = s.settling_s
            rec["binding_output"] = s.binding_output
        for o in a.outputs:
            stages = []
            for st in o.stages:
                d: dict[str, Any] = {"what": st.what, "kind": st.kind, "source": st.source,
                                     "basis": st.basis}
                v = stage_settling_s(st, fs)
                if v is not None:
                    d["reach_s"] = v
                c = stage_counted_s(st, fs)
                if c is not None:
                    d["counted_s"] = c  # (c) 3 (b): 0 where the stage skips NaN
                d["skips_nan"] = st.skips_nan
                if st.note:
                    d["note"] = st.note
                if st.edge is not None:
                    d["measurement"] = {"ruling": st.edge.ruling,
                                        "files": [{"file": f, "sha256": h}
                                                  for f, h in st.edge.files]}
                stages.append(d)
            orec: dict[str, Any] = {"output": o.name, "source": o.source, "stages": stages}
            if s.per_output_s.get(o.name) is not None:
                orec["settling_s"] = s.per_output_s[o.name]
            rec["outputs"].append(orec)
        out.append(rec)
    return out


def _check_cuts(sess: str, cuts: object) -> None:
    """Every cut the map uses, once, as an exact sample (invariant 27: asserted at write)."""
    if not isinstance(cuts, list):
        msg = f"{sess}: cuts (one per owner, output and class) are absent"
        raise TypeError(msg)
    ids = [str(c.get("cut")) for c in cuts]
    if len(set(ids)) != len(ids):
        msg = f"{sess}: a cut appears twice ({sorted(ids)})"
        raise ValueError(msg)
    want = {cut_id(*c) for c in trim_cuts()}
    if set(ids) != want:
        msg = (f"{sess}: cuts {sorted(set(ids) ^ want)} differ from the map's "
               f"(missing {sorted(want - set(ids))})")
        raise ValueError(msg)
    for c in cuts:
        k = c.get("start_sample0")
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            msg = f"{sess}/{c.get('cut')}: start_sample0 must be an int >= 0"
            raise TypeError(msg)


def _is_sample(k: object) -> bool:
    return isinstance(k, int) and not isinstance(k, bool) and k >= 0


def _check_times(sess: str, f: Mapping[str, Any]) -> None:
    """Both stim times and the input mask point, consistent to the sample (RULING (i) 3 (c)).

    ``input_mask_sample0`` must equal max(``electrical_settle_sample0``,
    ``mechanical_end_sample0``); each of the three samples must be the first sample at or
    after its seconds field at the file's ``fs`` (``first_sample_at_or_after``, the one
    conversion); the electrical settling may not precede the electrical end; and no
    measured start or cut may precede the input mask point: a file that disagrees with
    itself is refused at write time and at read time (invariant 27).
    """
    for key in ("electrical_settle_sample0", "mechanical_end_sample0", "input_mask_sample0"):
        if not _is_sample(f.get(key)):
            msg = f"{sess}: {key} must be an int >= 0 (mode (B), {RULING_TIMES})"
            raise TypeError(msg)
    for key in ("fs", "electrical_end_s", "mechanical_end_s", "electrical_settle_s",
                "input_mask_s"):
        v = f.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            msg = f"{sess}: {key} is absent or not a finite number ({RULING_TIMES})"
            raise TypeError(msg)
    fs = float(f["fs"])
    if fs <= 0:
        msg = f"{sess}: fs must be positive, got {fs}"
        raise ValueError(msg)
    ke, km, k = (int(f[x]) for x in ("electrical_settle_sample0", "mechanical_end_sample0",
                                     "input_mask_sample0"))
    if k != max(ke, km):
        msg = (f"{sess}: input_mask_sample0 {k} is not max(electrical_settle_sample0 {ke}, "
               f"mechanical_end_sample0 {km}) ({RULING_TIMES})")
        raise ValueError(msg)
    for ks, ts in (("electrical_settle_sample0", "electrical_settle_s"),
                   ("mechanical_end_sample0", "mechanical_end_s"),
                   ("input_mask_sample0", "input_mask_s")):
        want = first_sample_at_or_after(float(f[ts]), fs)
        if int(f[ks]) != want:
            msg = (f"{sess}: {ks} {f[ks]} is not the first sample at or after {ts} "
                   f"{f[ts]!r} s at fs {fs!r} ({want}; extent.grid.first_sample_at_or_after, "
                   f"{RULING_TIMES})")
            raise ValueError(msg)
    if float(f["electrical_settle_s"]) < float(f["electrical_end_s"]):
        msg = (f"{sess}: electrical_settle_s {f['electrical_settle_s']!r} precedes "
               f"electrical_end_s {f['electrical_end_s']!r}: the settling is measured from the "
               f"electrical end ({RULING_TIMES})")
        raise ValueError(msg)
    late = [str(r.get("analysis", r.get("cut"))) for r in (*f["analyses"], *f.get("cuts", []))
            if r.get("basis") in (BASIS_MEASURED, BASIS_CUT) and int(r["start_sample0"]) < k]
    if late:
        msg = f"{sess}: measured start(s) {late} precede the input mask sample {k}"
        raise ValueError(msg)


def recovery_starts_document(files: Sequence[Mapping[str, Any]], *, fs: float,
                             fixed_start_s: float = FIXED_START_S,
                             table: Mapping[str, Analysis] | None = None,
                             extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the document; uniqueness of every key is asserted here (invariant 27)."""
    measured: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in files:
        sess = str(f["session"])
        if sess.casefold() in seen:
            msg = f"session {sess!r} appears twice: one start per file"
            raise ValueError(msg)
        seen.add(sess.casefold())
        if "analyses" in f:
            names = [str(r["analysis"]) for r in f["analyses"]]
            if len(set(names)) != len(names):
                msg = f"{sess}: an analysis appears twice ({names})"
                raise ValueError(msg)
            for r in f["analyses"]:
                k = r["start_sample0"]
                if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                    msg = f"{sess}/{r['analysis']}: start_sample0 must be an int >= 0"
                    raise TypeError(msg)
            _check_cuts(sess, f.get("cuts"))
            _check_times(sess, f)
            measured.append(dict(f))
        else:
            held.append(dict(f))
    doc: dict[str, Any] = {
        "schema": SCHEMA, "ruling": RULING, "fs": float(fs),
        "fixed_start_s": float(fixed_start_s), "fixed_start_source": FIXED_START_SOURCE,
        "source_files": source_files_record(), "trim_modes": list(TRIM_MODES),
        "withdrawn_trim_modes": dict(WITHDRAWN_TRIM_MODES),
        "output_times": output_times_record(), "table": table_record(fs, table),
        "settling_rules": RULING_SETTLING, "edge_settling": edge_settling_record(),
        "edge_settling_sha256": edge_settling_sha256(),
        "files": sorted(measured, key=lambda d: str(d["session"])),
        "held": sorted(held, key=lambda d: str(d["session"])),
    }
    if extra:
        doc["extra"] = dict(extra)
    return doc


def write_recovery_starts(path: Path, doc: Mapping[str, Any]) -> str:
    """Write the document canonically and atomically; return the text written.

    Sorted keys, ASCII-escaped, no NaN (a missing value is an absent key), LF endings.
    """
    text = json.dumps(doc, ensure_ascii=True, sort_keys=True, indent=1, allow_nan=False) + "\n"
    atomic_write_text(Path(path), text)
    return text


def read_recovery_starts(path: Path) -> dict[str, Any]:
    """Read and validate a starts file; refuses a wrong schema or a malformed row by name."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        msg = f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA!r}"
        raise ValueError(msg)
    for key in ("fs", "fixed_start_s", "files", "held", "table", "output_times",
                "source_files", "trim_modes", "edge_settling", "edge_settling_sha256"):
        if doc.get(key) is None:
            msg = f"{path}: required field {key!r} is absent"
            raise ValueError(msg)
    if (doc["edge_settling"] != edge_settling_record()
            or doc["edge_settling_sha256"] != edge_settling_sha256()):
        msg = (f"{path}: its edge_settling (sha256 {str(doc['edge_settling_sha256'])[:16]}) is "
               f"not this build's (sha256 {edge_settling_sha256()[:16]}): the starts were "
               "derived under other edge settlings - regenerate the file (schema v5)")
        raise ValueError(msg)
    recovery_starts_document([*doc["files"], *doc["held"]], fs=float(doc["fs"]),
                             fixed_start_s=float(doc["fixed_start_s"]))
    known = set(expected_consumers())
    for f in doc["files"]:
        for r in f["analyses"]:
            if r["analysis"] not in known:
                msg = f"{path}: {f['session']} names analysis {r['analysis']!r}, not a consumer"
                raise ValueError(msg)
    return dict(doc)
