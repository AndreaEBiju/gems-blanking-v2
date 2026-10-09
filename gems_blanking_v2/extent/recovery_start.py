"""RULING 2026-10-08 (k) 2: where each analysis's recovery starts, and the file Night 6 reads.

Each analysis starts at

    the file's detected stim-off  +  the measured electrical settling ((j) 6 (i))
                                  +  THAT analysis's own filter or window settling.

There is no shared maximum over analyses. The first two terms are per file and are
measured by the recovery-start script (``electrical_settle_s`` returns their sum as one
absolute time). The third is per analysis and is declared here, once (invariant 33):
:data:`ANALYSES` is the one table, and :func:`analysis_settling` is the one place a
figure is derived from it.

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
* ``window, part before t`` - an asymmetric window: the part before ``t``.

Whether each window is centred or trailing is read from Andrea's code and cited with
``file:line``. Her working tree is not a commit (``slowWaveAnalysis_new.m`` is modified
and ``HR_BR_HRVAnalysis_beats.m`` untracked at f93e250), so :data:`SOURCE_FILES` records
the SHA-256 of every cited processing_new file as read, and a test checks both the hashes
and that every cited line still says what the table says it does.

What is NOT counted, and why (each listed per analysis, never silently dropped):

* **selection rules** (``findpeaks`` MinPeakDistance, event grouping, refractory
  periods). They are neither a filter nor a window, which is what (k) 2 measures, and
  their reach is a chain (a peak removed by a taller one removed by a taller one), so
  it has no fixed bound. Flagged for a ruling, not counted.
* **edge guards** (step2's 10 ms pad, HR's 0.75 s and slow wave's 15 s edge buffers).
  They act AFTER an edge, so they do not reach back. They do NOT act at a masked start
  in Night 6: step2's pad dilates every invalid sample, but HR's and slow wave's edge
  masks are built from ``blankIdx`` and the array ends only
  (``HR_BR_HRVAnalysis_beats.m:249-256``, ``slowWaveAnalysis_new.m:111-118``) and Night 6
  passes ``blankIdx = []`` (``night6_run_recording.m`` call_one), so a NaN lead-in gets
  no edge buffer from either. Their windows that straddle a masked start are computed
  over her ``'nearest'`` fill of the masked span (review of be402a1, finding 1 (b), (c)).
* **epoch-wide statistics** (session sigma, detrend, averages). They have no time to
  trim by. Both trim modes mask the input before at least the electrical settling, which
  keeps them free of unsettled data; under ``mask_to_electrical_drop_outputs`` they are
  computed over [electrical settling, epoch end], and the record says so.

Trim modes (:data:`TRIM_MODES`, a REQUIRED declaration of every Night 6 batch)
-------------------------------------------------------------------------------
* ``mask_to_own_start`` - each analysis's input is masked (NaN) up to its OWN start.
  Costs, per analysis, the window settling after the electrical settling (spikes 15 s,
  hrv and breathing 30 s, slow wave 41 s, mmc 51 s at the cohort rate), and its outputs
  in the first window after the start are partial windows over the fill of the mask.
* ``mask_to_electrical_drop_outputs`` - every analysis's input is masked up to the
  electrical settling, one point for all; every output stamped before the analysis's
  own start is then dropped or flagged as not computed, by :data:`OUTPUT_VARS` (the
  per-output time map, read from her code with ``file:line``). Epoch-wide scalars stay,
  recorded as computed over [electrical settling, epoch end]; an output whose time
  convention is not known is left untrimmed and listed by name, never dropped silently.

An unknown stage makes the analysis's settling ``None`` with the missing stage named -
never the part that is known (invariant 19). That analysis then keeps 132 s for the
file, labelled ``fixed_132s_settling_unknown`` (the user's rule, 2026-10-08).

The file Night 6 reads
----------------------
:func:`write_recovery_starts` writes one JSON document - per file: the electrical
settling as an exact 0-based FILE sample, and per analysis the start in seconds (for the
record) and as an exact 0-based FILE sample (MATLAB never converts seconds to samples,
invariants 15 and 22), its basis and its source - plus the table it was computed with,
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

from gems_blanking_v2.extent.grid import seconds_to_sample
from gems_blanking_v2.extent.tolerance import (
    CONSUMER_FILTERS,
    ConsumerFilter,
    expected_consumers,
    impulse_settling_s,
)
from gems_blanking_v2.io.store import atomic_write_text
from gems_blanking_v2.physio.rpeaks import DECIMATE_TARGET_HZ

__all__ = [
    "ACTIONS",
    "ANALYSES",
    "BASIS_FIXED",
    "BASIS_MEASURED",
    "CONVENTIONS",
    "FIXED_START_S",
    "FIXED_START_SOURCE",
    "HELD_NO_SETTLING",
    "HELD_UNDETECTED",
    "OUTPUT_FILES",
    "OUTPUT_VARS",
    "RULING",
    "SCHEMA",
    "SOURCE_FILES",
    "SOURCE_LABEL",
    "TRIM_MARKER",
    "TRIM_MODES",
    "Analysis",
    "AnalysisSettling",
    "Excluded",
    "Output",
    "OutputFile",
    "OutputVar",
    "Stage",
    "analysis_settling",
    "beat_decimation_factor",
    "file_starts",
    "output_times_record",
    "read_recovery_starts",
    "recovery_starts_document",
    "source_files_record",
    "stage_settling_s",
    "table_record",
    "write_recovery_starts",
]

RULING: Final = "RULING 2026-10-08 (k) 2"
SCHEMA: Final = "gems-blanking-v2 recovery starts v2"
"""v2: the electrical settling sample, the output time map and the cited files' hashes
(``source_files``: ``[{file, sha256}]`` rows, checked by Night 6 against the files it
resolves; no v2 file had been written for a run when the rows replaced an object)."""
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

TRIM_MODES: Final = ("mask_to_own_start", "mask_to_electrical_drop_outputs")
"""The two Night 6 trim semantics; the batch list must name one (no default). The same
two names are ``matlab/night6/night6_trim_modes.m`` (a test holds them equal)."""

FIXED_START_S: Final = 132.0
FIXED_START_SOURCE: Final = (
    "io.stim_split protocol book chronic_2min_20min: stim_duration_s 120 + "
    "stim_tolerance_s 12 (a tolerance on the detected stim duration, task 03B, a3e129d); "
    "io.audit_pool.stim_epoch_s (b4d5479) excludes 0-132 s, so routing and the masks start "
    "every stim_rec recovery epoch there")
"""Where the current recovery epoch starts. Not a settling measurement (RULING (k))."""

BASIS_MEASURED: Final = "stim_off_plus_electrical_plus_own_settling"
BASIS_FIXED: Final = "fixed_132s_settling_unknown"
HELD_UNDETECTED: Final = "held_stim_edges_undetected"
HELD_NO_SETTLING: Final = "held_no_electrical_settling"

StageKind = Literal["filter impulse response", "centred window", "trailing window",
                    "fixed bin, reported at its centre", "window, part before t"]
How = Literal["impz", "impz_beat_rate", "fir_half_beat", "half", "whole", "before_t",
              "unknown"]

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
    ``window_s``, or ``unknown`` (a figure nobody has measured: the analysis is then
    unknown, invariant 19).
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


@dataclass(frozen=True, slots=True)
class Output:
    """One output of an analysis's call, and the cascade of stages it is made through."""

    name: str
    stages: tuple[Stage, ...]
    source: str


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

_BEAT_FIR = Stage(
    "beat fiducials: the beat detector's decimating FIR (to ~2 kHz), zero phase",
    "filter impulse response", "fir_half_beat",
    "gems_blanking_v2/physio/rpeaks.py:329-330 (scipy decimate ftype='fir', zero_phase)",
    'decimate(y, factor, ftype="fir", zero_phase=True)', "code")
_BEAT_BAND = Stage(
    "beat fiducials: the beat detector's 10-150 Hz order-4 band, at the decimated rate",
    "filter impulse response", "impz_beat_rate",
    "gems_blanking_v2/physio/rpeaks.py:335-336 (butter 4, DETECT_BAND_HZ = HR_BAND)",
    'sos = butter(4, [lo / nyq, min(hi, nyq * 0.9) / nyq], btype="bandpass"', "measured",
    filter=CONSUMER_FILTERS["hrv"])
_BEATS = (_BEAT_FIR, _BEAT_BAND)
"""The stored beat train (routing, frozen) is located by these two stages; hrv,
breathing and mmc read it. Its other steps are selection rules and epoch-wide statistics
(listed under each analysis that reads it)."""

_SPIKE_BAND = Stage(
    "300-3000 Hz order-4 bandpass, filtfilt over NaN filled for the filter and restored",
    "filter impulse response", "impz",
    "step1_bandpass.m:57 (filtfilt; NaN filled at :53 and restored at :58)",
    "xf = filtfilt(b, a, xfill);", "measured", filter=CONSUMER_FILTERS["spikes"])

_SPIKES = Analysis(
    "spikes", "process_dataset_v2 (her process_dataset steps, matlab/night6)",
    (
        Output("spike times", (_SPIKE_BAND,), "step3_detect.m (on D.filtered)"),
        Output("spike waveforms", (
            _SPIKE_BAND,
            Stage("waveform window before the aligned peak: wfPreMs 1 ms + wfAlignSearchMs "
                  "0.5 ms", "window, part before t", "before_t", "step4_waveforms.m:44-46",
                  "npre   = round(P.wfPreMs  * 1e-3 * fs);", "code", window_s=0.0015),
        ), "step4_waveforms.m"),
        Output("firing rate (frBinSec 1 s bins)", (
            _SPIKE_BAND,
            Stage("firing-rate bins, frBinSec = 1 s, value at the bin centre",
                  "fixed bin, reported at its centre", "half",
                  "step6_spike_report.m:62 (firing_rate, :140-152, t = bin centre)",
                  "firing_rate(cen, valid, N, fs, P.frBinSec)", "code", window_s=1.0),
        ), "step6_spike_report.m:62"),
        Output("activity envelope (envBinSec 1 s RMS bins)", (
            _SPIKE_BAND,
            Stage("artifact-excursion pad, +/- 5 ms", "centred window", "half",
                  "step3b_envelope.m:72", "artPad = round(0.005 * fs);", "code",
                  window_s=0.010),
            Stage("RMS bins, envBinSec = 1 s, value at the bin centre",
                  "fixed bin, reported at its centre", "half", "step3b_envelope.m:107",
                  "t_c(b) = ((i0 + i1) / 2 - 1) / fs;", "code", window_s=1.0),
        ), "step3b_envelope.m"),
        Output("rolling CV2 (cv2WinSec 30 s bins)", (
            _SPIKE_BAND,
            Stage("CV2 bins, cv2WinSec = 30 s, value at the bin centre",
                  "fixed bin, reported at its centre", "half",
                  "step6_spike_report.m:94 (rolling_cv2, :262-271, t = edges + winSec/2)",
                  "rolling_cv2(st, isims, isClean, Tend, P.cv2WinSec)", "code",
                  window_s=30.0),
        ), "step6_spike_report.m:94"),
    ),
    (
        Excluded("step2 edge pad, edgeBufferMs 10 ms around every invalid sample", "edge guard",
                 "step2_noise_sigma.m:41 and :75-77; pipeline_params.m:69",
                 "acts after an edge (it dilates every invalid sample, so the masked start "
                 "too) and covers the 5.1 ms ringing "
                 "the bandpass row counts; it does not reach back"),
        Excluded("wrapper check: no spike within 5 ms (NanPadMs) of a NaN", "edge guard",
                 "matlab/night6/process_dataset_v2.m:143-151", "a refusal check, not a stage"),
        Excluded("refractory 1 ms (detection dead time, re-enforced after alignment)",
                 "selection rule", "step4_waveforms.m:47 and :78",
                 "a selection chain, not a filter or window"),
        Excluded("session sigma (median of 5 s window MADs)", "epoch-wide statistic",
                 "step2_noise_sigma.m:114", "untimed: masking the input keeps it settled"),
        Excluded("Fano curve, autocorrelogram, rate PSD, bursts, modality test",
                 "epoch-wide statistic", "step6_spike_report.m:79, :85, :88, :91; "
                 "step5c_modality_test.m", "untimed: masking the input keeps them settled"),
        Excluded("Vpp over time (vppBinSec 5 s)", "display only", "step6_spike_report.m:392",
                 "binstat is called by plot_report only"),
    ),
    "The 30 s CV2 bins bind; without them the spike start would be stim-off + "
    "electrical + 0.5 s (the 1 s rate and envelope bins).")

_HRV = Analysis(
    "hrv", "HR_BR_HRVAnalysis_beats (stored beats), hrv run",
    (
        Output("heart rate (hrBrWinSec 60 s, centred)", (
            *_BEATS,
            Stage("heart-rate window, hrBrWinSec = 60 s (night6 params), centred",
                  "centred window", "half", f"{_HR_BR}:835 (t0 = tc - halfHrBr; :816)",
                  "t0   = tc - halfHrBr;", "code", window_s=60.0),
        ), f"{_HR_BR}:833-849"),
        Output("beat count and HRV metrics (winSec 20 s, centred)", (
            *_BEATS,
            Stage("count/HRV window, winSec = 20 s (night6 params), centred",
                  "centred window", "half", f"{_HR_BR}:873 (t0w = tc - halfWin; :817)",
                  "t0w   = tc - halfWin;", "code", window_s=20.0),
        ), f"{_HR_BR}:871-917"),
        Output("sample entropy (fixed 60 s, centred)", (
            *_BEATS,
            Stage("sample-entropy window, fixed 60 s, centred", "centred window", "half",
                  f"{_HR_BR}:924 (t0se = tc - halfSampEn; :818-819)",
                  "t0se  = tc - halfSampEn;", "code", window_s=60.0),
        ), f"{_HR_BR}:920-934"),
    ),
    (
        Excluded("the call's own 10-150 Hz filter (heartBeatSeriesClean)", "display only",
                 f"{_HR_BR}:278-286", "with stored beats it feeds only the detrended trace; "
                 "the beats are READ (:289-299), not found on it"),
        Excluded("edgeBufferSec 0.75 s (night6 params) at blank and signal edges", "edge guard",
                 f"{_HR_BR}:249-256", "acts after an edge; does not reach back. Built from "
                 "blankIdx and the array ends only, and night6 passes blankIdx = []: no buffer at "
                 "a masked start"),
        Excluded("global HRV, average heart rate", "epoch-wide statistic",
                 f"{_HR_BR}:436, :453-466", "untimed: masking the input keeps them settled"),
        Excluded("beat detector: global sigma, global RR, QRS template, pass-1 distance",
                 "selection rule", "gems_blanking_v2/physio/rpeaks.py:558-585",
                 "upstream (routing, frozen); computed on the routed region from 132 s"),
    ),
    "Heart rate is an output of this run (one call, one input), so it shares hrv's start.")

_BREATHING = Analysis(
    "breathing", "HR_BR_HRVAnalysis_beats (stored beats), breathing run",
    (
        Output("breath rate (hrBrWinSec 60 s, centred)", (
            *_BEATS,
            Stage("breath-rate window, hrBrWinSec = 60 s (night6 params), centred",
                  "centred window", "half", f"{_HR_BR}:835 (t0 = tc - halfHrBr; :816)",
                  "t0   = tc - halfHrBr;", "code", window_s=60.0),
        ), f"{_HR_BR}:851-868"),
    ),
    (
        Excluded("breath troughs: findpeaks MinPeakDistance (in beats)", "selection rule",
                 f"{_HR_BR}:387", "a selection chain, not a filter or window; the troughs "
                 "are read from the raw signal at the beats (:378), no filter"),
        Excluded("edgeBufferSec 0.75 s (night6 params)", "edge guard", f"{_HR_BR}:249-256",
                 "acts after an edge; does not reach back. Built from blankIdx and the "
                 "array ends only, and night6 passes blankIdx = []: no buffer at a masked "
                 "start"),
        Excluded("average breath rate", "epoch-wide statistic", f"{_HR_BR}:437",
                 "untimed: masking the input keeps it settled"),
    ))

_SLOW_WAVE = Analysis(
    "slow_wave", "slowWaveAnalysis_new, one ANT channel at a time (each channel the same chain)",
    (
        Output("slow-wave trace and peak times", (
            Stage("low-pass 0.15 Hz order 2 (batch_process settings, night6 params), "
                  "filtfilt over NaN filled for the filter", "filter impulse response",
                  "impz", f"{_SW}:133 (fill :126; butter :131)",
                  "filtSignal = filtfilt(sos, g, xFill);", "measured",
                  filter=CONSUMER_FILTERS["slow_wave"]),
            Stage("gaussian smoothdata, window 5 s (night6 params), centred",
                  "centred window", "half", f"{_SW}:156",
                  "smoothdata(filteredSignal, 1, 'gaussian', windowlen)", "code",
                  window_s=5.0),
        ), f"{_SW}:156, :200"),
        Output("slow-wave rate (rateWinSec 60 s, centred)", (
            Stage("low-pass 0.15 Hz order 2", "filter impulse response", "impz",
                  f"{_SW}:133", "filtSignal = filtfilt(sos, g, xFill);", "measured",
                  filter=CONSUMER_FILTERS["slow_wave"]),
            Stage("gaussian smoothdata, 5 s, centred", "centred window", "half", f"{_SW}:156",
                  "smoothdata(filteredSignal, 1, 'gaussian', windowlen)", "code",
                  window_s=5.0),
            Stage("rate window, rateWinSec = 60 s, centred", "centred window", "half",
                  f"{_SW}:246 (winStartSamp = ctrSamp - halfWinSamp; :169, :229)",
                  "winStartSamp = max(1, ctrSamp - halfWinSamp);", "code", window_s=60.0),
        ), f"{_SW}:225-280"),
    ),
    (
        Excluded("findpeaks MinPeakDistance 6 s", "selection rule", f"{_SW}:200 (:165)",
                 "a selection chain, not a filter or window"),
        Excluded("edgeBufferSec 15 s at the signal edges (and blankIdx, which night6 leaves "
                 "empty: NaN spans get no edge buffer)", "edge guard", f"{_SW}:108-118",
                 "acts after an edge; does not reach back. Not at a masked start: "
                 "blankIdx is empty in night6"),
        Excluded("detrend over the whole epoch; avgSlowWave = mean of the rate series",
                 "epoch-wide statistic", f"{_SW}:139, :283",
                 "untimed: masking the input keeps them settled"),
    ),
    "The same chain for each ANT channel, so one start serves all three channels' masks "
    "and the shared-call shortcut ((j) 5 (a)) stays exact.")

_MMC_BLANK = Stage("cardiac blank around each stored beat, cardiacBlankMs 25 ms, centred",
                   "centred window", "half", f"{_MMC}:85 (blank :99-102)",
                   "half = max(1, round(cardMs/1000*fs));", "code", window_s=0.050)
_MMC_BAND = Stage("2-50 Hz order-4 bandpass, filtfilt over NaN filled for the filter",
                  "filter impulse response", "impz", f"{_MMC}:106 (fill :104)",
                  "y = filtfilt(sos,gd,xf);", "measured", filter=CONSUMER_FILTERS["mmc"])
_MMC_MED = Stage("moving median, sigmaWin 30 s (default), centred", "centred window", "half",
                 f"{_MMC}:231 (win :230; sigmaWin :45)",
                 "med = movmedian(y, win, 'omitnan');", "code", window_s=30.0)
_MMC_MAD = Stage("moving MAD of (y - moving median), 30 s, centred: nested in the median",
                 "centred window", "half", f"{_MMC}:232",
                 "sig = movmedian(abs(y-med), win, 'omitnan') / 0.6745;", "code",
                 window_s=30.0)
_MMC_EVENTS = (*_BEATS, _MMC_BLANK, _MMC_BAND, _MMC_MED, _MMC_MAD)

_MMC_A = Analysis(
    "mmc", "extract_mmc (ANT1-3 raw, stored beats)",
    (
        Output("firing and burst (mmc_burst) event times", _MMC_EVENTS,
               f"{_MMC}:109-111 (detect_crossings :220-235)"),
        Output("firing and burst (mmc_burst) rate and peak amplitude (W 10 s, centred)", (
            *_MMC_EVENTS,
            Stage("rate window W = 10 s (default), centred", "centred window", "half",
                  f"{_MMC}:287 (centers :115)",
                  "lo = max(1, floor((centers(w)-W/2)*fs)+1);", "code", window_s=10.0),
        ), f"{_MMC}:115-117 (event_rate :282-296)"),
        Output("cross-channel delay (delayW 30 s on the firing rate, centred)", (
            *_MMC_EVENTS,
            Stage("rate window W = 10 s, centred", "centred window", "half", f"{_MMC}:287",
                  "lo = max(1, floor((centers(w)-W/2)*fs)+1);", "code", window_s=10.0),
            Stage("delay window delayW = 30 s (default), labelled at its centre",
                  "centred window", "half", f"{_MMC}:265 (wlen :259)",
                  "delay_t(s) = (lo+hi)/2 * S;", "code", window_s=30.0),
        ), f"{_MMC}:119 (xchan_delay :255-272)"),
    ),
    (
        Excluded("event grouping by valid-time gap: 0.05 s (firings), 0.5 s (bursts)",
                 "selection rule", f"{_MMC}:245-246 (refractories :46-47)",
                 "a grouping chain, not a filter or window"),
        Excluded("avgRate, pctBlanked, peri-R and PSD QC; the global sigma fill",
                 "epoch-wide statistic", f"{_MMC}:295, :233, :123-150",
                 "untimed: masking the input keeps them settled"),
    ),
    "The two moving medians are NESTED (the MAD is taken around the moving median), so a "
    "detection reaches back 15 + 15 = 30 s, not the 15 s half-window task 13 pads mmc "
    "extents with. The 30 s delay window binds; without the delay output the mmc start "
    "would be 15 s earlier.")

ANALYSES: Final[Mapping[str, Analysis]] = {
    a.name: a for a in (_SPIKES, _HRV, _BREATHING, _SLOW_WAVE, _MMC_A)}
"""THE table (invariant 33). Keyed by consumer: the Night 6 consumers of this build
(``tolerance.expected_consumers()``), a test holds the two equal."""


# ---------------------------------------------------------------------------
# the output time map (mask_to_electrical_drop_outputs)
# ---------------------------------------------------------------------------

Role = Literal["trim", "time_axis", "epoch_scalar", "input", "parameter", "container",
               "unknown"]
Convention = Literal["row1", "sec0", "sec_row1", "sec_xchan_delay"]
Action = Literal["nan", "drop", "false"]

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
    "false": "a full-rate logical event series: the rows are set false so her variable "
             "keeps its type and her readers still load it (a logical has no NaN); a false "
             "row before the start means NOT COMPUTED, never 'no event', and that is "
             "recorded in the file itself (TRIM_MARKER). How events are represented in the "
             "not-computed region is still Andrea's ruling to make",
}
"""Every action is recorded in the trimmed file itself (:data:`TRIM_MARKER`)."""

TRIM_MARKER: Final = "night6_recovery_trim"
"""The top-level variable Night 6 adds to every output file it trims (drop mode): per
trimmed variable path, the owner, the action, the stamp and its convention, the owner's
start (seconds and 0-based file sample), the first computed epoch row and sample, the
mode, and per leaf how many entries are not computed and where. It travels with the
file, so any reader can tell 'not computed' from a value without the run record. Her
variables keep their names and types."""


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

    ``role``: ``trim`` (time-stamped: dropped or flagged before the owner's start, by
    ``stamp`` / ``convention`` / ``action``), ``time_axis`` (a stamp vector, kept),
    ``epoch_scalar`` (no time: computed over [electrical settling, epoch end] and kept),
    ``input`` (a description of what the call was given), ``parameter``, ``container``
    (a struct whose fields are classified one by one), or ``unknown`` (time convention
    not determinable: left UNTRIMMED and listed by name). A struct-array container is
    walked element by element; a sibling ``stamp`` is read from the same element.
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
          action: Action, owner: str = "") -> OutputVar:
    return OutputVar(file, path, "trim", source, owner, anchor, stamp, convention, action)


def _axis(file: str, path: str, source: str, anchor: str, owner: str = "") -> OutputVar:
    return OutputVar(file, path, "time_axis", source, owner, anchor)


def _many(file: str, role: Role, paths: str, source: str, why: str = "",
          owner: str = "") -> tuple[OutputVar, ...]:
    return tuple(OutputVar(file, p, role, source, owner, why=why) for p in paths.split())


_EPOCH_WHY = "no time: computed over [electrical settling, epoch end] in the drop mode"

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
           "noiseInfo modality spikes.nSpikes spikes.rate_hz spikes.validSec "
           "spikes.meanWaveform spikes.stdWaveform spikes.nDetected spikes.screen "
           "envelope.meanRMS_uv envelope.meanExcess_uv metrics.nSpikes metrics.validDur_s "
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
    _axis("spikes_v2", "sigmaWin.centers", f"{_STEP2}:98 (cell per channel)",
          "cWin(w) = (i0 + i1) / 2;"),
    _trim("spikes_v2", "sigmaWin.sigma", f"{_STEP2}:96", "sigWin(w) = median(abs(s)) / 0.6745;",
          "sigmaWin.centers", "row1", "nan"),
    *(_trim("spikes_v2", f"spikes.{v}", f"{_STEP3}:{ln}", a, "spikes.centers", "row1", "drop")
      for v, ln, a in (("centers", 86, "spikes(k).centers          = locs;"),
                       ("times", 87, "spikes(k).times            = (locs - 1) / fs;"),
                       ("peakAmp_uv", 88, "spikes(k).peakAmp_uv       = peakAmp_uv;"),
                       ("threshAtSpike_uv", 89, "spikes(k).threshAtSpike_uv = threshAtSpike_uv;"),
                       ("artifactMask", 90, "spikes(k).artifactMask     = artifactMask;"))),
    *(_trim("spikes_v2", f"spikes.{v}", f"{_STEP4}:{ln}", a, "spikes.alignedCenters", "row1",
            "drop")
      for v, ln, a in (("waveforms", 114, "D.spikes(k).waveforms      = W;"),
                       ("alignedCenters", 115, "D.spikes(k).alignedCenters = aligned;"),
                       ("alignedTimes", 116, "D.spikes(k).alignedTimes   = (aligned - 1) / fs;"),
                       ("Vpp_uv", 117, "D.spikes(k).Vpp_uv         = Vpp;"),
                       ("width_ms", 118, "D.spikes(k).width_ms       = wid;"))),
    _axis("spikes_v2", "envelope.t", f"{_STEP3B}:107 (bin centre)",
          "t_c(b) = ((i0 + i1) / 2 - 1) / fs;"),
    *(_trim("spikes_v2", f"envelope.{v}", f"{_STEP3B}:{ln}", a, "envelope.t", "sec0", "nan")
      for v, ln, a in (("rms_uv", 115, "D.envelope(k).rms_uv       = rms;"),
                       ("sigmaFloor_uv", 116, "D.envelope(k).sigmaFloor_uv = sigFloor;"),
                       ("excess_uv", 117, "D.envelope(k).excess_uv    = excess;"),
                       ("validFrac", 118, "D.envelope(k).validFrac    = vfrac;"))),
    _axis("spikes_v2", "metrics.fr_t", f"{_STEP6}:150 (bin centre)",
          "t(b) = ((i0+i1)/2-1)/fs;"),
    *(_trim("spikes_v2", f"metrics.{v}", f"{_STEP6}:62", "[M.fr_t, M.fr_hz, M.fr_validFrac]",
            "metrics.fr_t", "sec0", "nan") for v in ("fr_hz", "fr_validFrac")),
    _axis("spikes_v2", "metrics.cv2_t", f"{_STEP6}:264 (bin start + winSec/2)",
          "t = edges(1:end-1)+winSec/2;"),
    _trim("spikes_v2", "metrics.cv2_roll", f"{_STEP6}:94", "[M.cv2_t, M.cv2_roll]",
          "metrics.cv2_t", "sec0", "nan"),
    *(_trim("spikes_v2", f"metrics.burst.{v}", f"{_STEP6}:249 (st = alignedTimes, sec0)",
            "onsets(end+1)=st(i0); offsets(end+1)=st(i1);", "metrics.burst.onsets", "sec0",
            "drop") for v in ("onsets", "offsets")),
)

_HR_SERIES = (("heartRateSeries", 849, "heartRateSeries(i) = hrPeaksInStretch", "hrv"),
              ("heartCountSeries", 893, "heartCountSeries(i) = sum(heartPeakTrain", "hrv"),
              ("heartCountValidSec", 894, "heartCountValidSec(i) = winValidSec;", "hrv"),
              ("heartCountRateSeries", 896, "heartCountRateSeries(i) = heartCountSeries(i)",
               "hrv"),
              ("breathRateSeries", 866, "breathRateSeries(i) = numel(peaksInStretch)",
               "breathing"))
_HRV_SERIES = (("hrv_series", 910, "hrv_series(i)   = hrv_val;"),
               ("rmssd_series", 911, "rmssd_series(i) = hv.rmssd;"),
               ("pnn5_series", 912, "pnn5_series(i)  = hv.pnn5;"),
               ("sd1_series", 915, "sd1_series(i)  = sd1_val;"),
               ("sd2_series", 916, "sd2_series(i)  = hv.sd2;"),
               ("sampEn_series", 929, "sampEn_series(i) = sampleEntropyFast("),
               ("nRR_used", 901, "nRR_used(i) = sum(keep);"))


def _heartlocs(file: str) -> OutputVar:
    return _trim(file, "heartlocs", f"{_HR_BR}:299",
                 "heartlocs          = heartlocsRaw(validHeartPeakMask);", "heartlocs", "row1",
                 "drop")



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
    *_many("HRBR", "epoch_scalar", "avgHeartRate avgHeartCount avgHeartCountRate",
           f"{_HR_BR}:436-442", _EPOCH_WHY),
    *_many("HRBR", "epoch_scalar", "avgBreathRate br_implausibleFraction",
           f"{_HR_BR}:437, :402", _EPOCH_WHY, owner="breathing"),
    OutputVar("HRBR", "RR_implausibleMask", "unknown", f"{_HR_BR}:318", why=(
        "one flag per RR interval BEFORE the implausible ones are removed; the times of "
        "that list are not saved, so it has no stamp to trim by: untrimmed, listed")),
    _trim("HRBR", "heartBeatSeries", f"{_HR_BR}:303 (t :213)",
          "heartBeatSeries(invalidMask) = NaN;", "t", "sec0", "nan"),
    _heartlocs("HRBR"),
    *(_trim("HRBR", v, f"{_HR_BR}:{ln}", a, "metrics_t", "sec0", "nan", owner=o)
      for v, ln, a, o in _HR_SERIES),
    _trim("HRBR", "br_locs_true", f"{_HR_BR}:388 (filtered :391)",
          "br_locs_true = heartlocs(br_locs);", "br_locs_true", "row1", "drop",
          owner="breathing"),
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
            "1-based row)", "RR_times(end+1,1)     = s1 / fs;", "RR_times", "sec_row1", "drop")
      for v in ("RR_intervals", "RR_times")),
    *(_trim("HRVMeasures", v, f"{_HR_BR}:{ln}", a, "metrics_t", "sec0", "nan")
      for v, ln, a in _HRV_SERIES),
)

_SW_VARS = (
    *_many("slowWaves", "parameter",
           "blankIdx edgeBufferSec rateWinSec minStretchSec minPeaksInStretch fs window "
           "windowlen channel channelColumn maskSignal keptFrom",
           f"{_SW}:340-345; matlab/night6/night6_keep_slow_wave.m"),
    *_many("slowWaves", "input", "invalidMask edgeMask", f"{_SW}:97, :111-118"),
    *_many("slowWaves", "epoch_scalar", "avgSlowWave sw_implausibleFraction",
           f"{_SW}:283, :212", _EPOCH_WHY),
    _axis("slowWaves", "t", f"{_SW}:89", "t          = (0:N-1)' / fs;"),
    _axis("slowWaves", "slowWaveRateTime", f"{_SW}:178 (the window's centre sample, :246)",
          "slowWaveRateTime = t(rateT_idx);"),
    _trim("slowWaves", "slowWaveTimeSeries", f"{_SW}:160",
          "slowWaveTimeSeries(invalidMask, :) = NaN;", "t", "sec0", "nan"),
    _trim("slowWaves", "slowWaveRateSeries", f"{_SW}:279",
          "slowWaveRateSeries(ti, ci) = pooledPeaks", "slowWaveRateTime", "sec0", "nan"),
    _trim("slowWaves", "slowWavePeakLocs", f"{_SW}:205", "slowWavePeakLocs{ci} = locs;",
          "slowWavePeakLocs", "row1", "drop"),
)

_MMC_VARS = (
    *_many("mmc", "container", "mmc mmc.firing mmc.burst mmc.qc", f"{_MMC}:151-166"),
    *_many("mmc", "parameter",
           "mmc.fs mmc.pairs mmc.params mmc.firing.refractory mmc.burst.refractory "
           "mmc.qc.srcFile mmc.qc.dataVar mmc.qc.gastricCols mmc.qc.periR_t", f"{_MMC}:124-166"),
    OutputVar("mmc", "mmc.qc.rpeakT", "input", f"{_MMC}:125", why="the beats it was given"),
    *_many("mmc", "epoch_scalar",
           "mmc.firing.avgRate mmc.burst.avgRate mmc.qc.meanHR mmc.qc.pctBlanked "
           "mmc.qc.rateNanFrac mmc.qc.nFirings mmc.qc.nBursts mmc.qc.periR_raw "
           "mmc.qc.periR_cond mmc.qc.psd_f mmc.qc.psd_raw mmc.qc.psd_cond",
           f"{_MMC}:126-146, :295", _EPOCH_WHY),
    _axis("mmc", "mmc.t", f"{_MMC}:152 (t :65)", "mmc.fs = fs; mmc.t = t;"),
    _axis("mmc", "mmc.rate_t", f"{_MMC}:154 (centers :115, windows :287-288)",
          "mmc.rate_t = centers;"),
    _axis("mmc", "mmc.delay_t", f"{_MMC}:159 (delay_t :265: RATE ROWS, not rate_t)",
          "mmc.delay_t = delay_t; mmc.delay = delay;"),
    _trim("mmc", "mmc.signal", f"{_MMC}:153", "mmc.signal = single(cond);", "mmc.t", "sec0",
          "nan"),
    *(_trim("mmc", f"mmc.{lvl}.events", f"{_MMC}:302 (ev_bool, :155-157)",
            "for ch = 1:3; ev(evIdx{ch},ch) = true; end", "mmc.t", "sec0", "false")
      for lvl in ("firing", "burst")),
    *(_trim("mmc", f"mmc.{lvl}.{v}", f"{_MMC}:{ln}", a, "mmc.rate_t", "sec0", "nan")
      for lvl in ("firing", "burst")
      for v, ln, a in (("rate", 292, "rate(w,ch) = sum(inw)/vd;"),
                       ("peakAmp", 293, "peakAmp(w,ch) = mean(pka(inw));"))),
    _trim("mmc", "mmc.delay", f"{_MMC}:272 (window :264-265)",
          "delay(s,p) = lags(mi)*S;", "mmc.delay_t", "sec_xchan_delay", "nan"),
)

OUTPUT_VARS: Final[tuple[OutputVar, ...]] = (*_SPIKE_VARS, *_HR_VARS, *_SW_VARS, *_MMC_VARS)
"""THE per-output time map (invariant 33): every variable of every kept output file,
classified. An ``owner`` left empty is the file's owner (:data:`OUTPUT_FILES`)."""

XCHAN_DELAY_PARAMS: Final = ("mmc.params.W", "mmc.params.S")
"""Where ``sec_xchan_delay`` reads W and S (``extract_mmc.m:161``)."""


def _owner(v: OutputVar) -> str:
    return v.owner or OUTPUT_FILES[v.file].owner


def output_times_record() -> dict[str, Any]:
    """Return the output time map as data, for the starts document (Night 6 applies it).

    Checked here (a malformed map is refused, never half applied): every file and owner
    is known, every path is unique within its file and its parent is a declared
    container, and every ``trim`` names a stamp that is itself a declared time axis or a
    trimmed list, a known convention and a known action.
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
            if v.role == "trim":
                st = decl.get(v.stamp)
                if st is None or st.role not in ("time_axis", "trim"):
                    msg = f"{kind}/{path}: stamp {v.stamp!r} is not a declared time axis"
                    raise ValueError(msg)
                if v.convention not in CONVENTIONS or v.action not in ACTIONS:
                    msg = f"{kind}/{path}: convention or action missing"
                    raise ValueError(msg)
                row |= {"stamp": v.stamp, "convention": v.convention, "action": v.action}
            rows.append(row)
            if path.split(".")[0] == TRIM_MARKER:
                msg = f"{kind}/{path}: {TRIM_MARKER!r} is the trim marker, not her variable"
                raise ValueError(msg)
    return {"conventions": dict(CONVENTIONS), "actions": dict(ACTIONS),
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
    if stage.window_s is None or not (math.isfinite(stage.window_s) and stage.window_s > 0):
        msg = f"{stage.what}: a window stage needs a finite positive window_s"
        raise ValueError(msg)
    if stage.how == "half":
        return stage.window_s / 2.0
    return float(stage.window_s)  # "whole" and "before_t": the stated part, all of it


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
        vals = [stage_settling_s(s, fs) for s in out.stages]
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


def file_starts(*, session: str, fs: float | None, stim_off_s: float | None,
                electrical_settle_s: float | None, stim_off_source: str,
                electrical_source: str, analyses: Sequence[str] | None = None,
                table: Mapping[str, Analysis] | None = None,
                fixed_start_s: float = FIXED_START_S) -> dict[str, Any]:
    """Return one file's record: per-analysis starts, or the reason the file is held.

    ``electrical_settle_s`` is the ABSOLUTE time from the file start at which rule (j) 6 (i)
    found every channel settled (stim-off + electrical settling); it is also written as
    ``electrical_settle_sample0``, the one rounding, for the drop mode's input mask.
    A file whose stim edges
    were not detected, or whose signal never settles, is HELD as a whole (invariant 41):
    no start rows, so Night 6 refuses it by name. Within a measured file, an analysis
    whose own settling is unknown keeps ``fixed_start_s``, labelled, with the missing
    figure named; the others get their own starts. Missing values are absent keys.
    ``fs`` (the file's own rate) may be ``None`` only for a file held before it was read.
    """
    names = list(ANALYSES if analyses is None else analyses)
    held: dict[str, Any] = {"session": session}
    if fs is not None:
        if not (math.isfinite(fs) and fs > 0):
            msg = f"{session}: fs must be finite and positive, got {fs}"
            raise ValueError(msg)
        held["fs"] = float(fs)
    if stim_off_s is None:
        return {**held, "basis": HELD_UNDETECTED,
                "why": "stim edges not detected (03B): the start is never assumed (invariant 41)"}
    if electrical_settle_s is None:
        return {**held, "basis": HELD_NO_SETTLING, "stim_off_s": float(stim_off_s),
                "why": "no channel set settled for 1 s inside its 140-200 s range ((j) 6 (i))"}
    if fs is None:
        msg = f"{session}: a measured file needs its fs"
        raise ValueError(msg)
    if electrical_settle_s < stim_off_s:
        msg = (f"{session}: electrical settling {electrical_settle_s} s precedes stim-off "
               f"{stim_off_s} s")
        raise ValueError(msg)
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
        start = float(electrical_settle_s) + s.settling_s
        k0 = seconds_to_sample(start, fs)
        row: dict[str, Any] = {
            "analysis": name, "start_s": start, "start_sample0": k0, "basis": BASIS_MEASURED,
            "own_settling_s": s.settling_s, "binding_output": s.binding_output,
            "source": (f"{RULING}: stim-off ({stim_off_source}) + electrical settling "
                       f"({electrical_source}) + own settling "
                       f"(extent.recovery_start.ANALYSES[{name!r}], {SOURCE_LABEL})")}
        if k0 < fixed0:
            row["relation"] = "earlier_than_fixed: runs from the fixed start; early part " \
                              "deferred to the add-on"
        elif k0 == fixed0:
            row["relation"] = "at_fixed_start"
        else:
            row["relation"] = "later_than_fixed: trimmed at Night 6"
        rows.append(row)
    return {"session": session, "fs": float(fs), "stim_off_s": float(stim_off_s),
            "electrical_settle_s": float(electrical_settle_s),
            "electrical_settle_sample0": seconds_to_sample(float(electrical_settle_s), fs),
            "electrical_s": float(electrical_settle_s) - float(stim_off_s),
            "analyses": rows}


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
                stages.append(d)
            orec: dict[str, Any] = {"output": o.name, "source": o.source, "stages": stages}
            if s.per_output_s.get(o.name) is not None:
                orec["settling_s"] = s.per_output_s[o.name]
            rec["outputs"].append(orec)
        out.append(rec)
    return out


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
            ke = f.get("electrical_settle_sample0")
            if not isinstance(ke, int) or isinstance(ke, bool) or ke < 0:
                msg = f"{sess}: electrical_settle_sample0 must be an int >= 0 (both trim modes)"
                raise TypeError(msg)
            measured.append(dict(f))
        else:
            held.append(dict(f))
    doc: dict[str, Any] = {
        "schema": SCHEMA, "ruling": RULING, "fs": float(fs),
        "fixed_start_s": float(fixed_start_s), "fixed_start_source": FIXED_START_SOURCE,
        "source_files": source_files_record(), "trim_modes": list(TRIM_MODES),
        "output_times": output_times_record(), "table": table_record(fs, table),
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
                "source_files", "trim_modes"):
        if doc.get(key) is None:
            msg = f"{path}: required field {key!r} is absent"
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
