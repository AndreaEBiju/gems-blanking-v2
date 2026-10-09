"""Night 6 wrapper (matlab/night6): the MATLAB side of the mask handoff, checked from Python.

Builds a synthetic store - a 9-contact recording whose every sample encodes its own file
index, its ``meta.json``, a whole-file beats file and mask folders written by the real
``emit.handoff.write_mask_file`` - and runs ``tests/matlab/night6_check.m`` in ONE MATLAB
process. ``night6_run_recording`` runs with ``DryRun`` (the real loading, epoch slicing,
masking and skip logic; the consumers' calls are not made). Checks:

* the ``_minus_`` token round trip across the boundary, both directions, including the
  refusals (invariant 22) - ``night6_signal_from_token`` is the exact inverse of
  ``matlab_signal_token`` on fixed edge cases and 400 seeded random names;
* the epoch start is the mask file's ``epochStartSample0`` (Andrea, 2026-10-09): a mask
  file whose ``epochStart_s`` is moved by 3.4 samples still slices at the stored sample,
  so MATLAB never converts seconds to samples (invariants 15, 22);
* every consumer input is NaN on exactly the samples its OWN mask names - 1-based
  inclusive spans into the epoch - and nowhere else (invariants 1, 2, 15): hrv and
  breathing on one channel get two different inputs;
* slow wave runs one ANT channel at a time (Andrea, 2026-10-09): every column of the run
  for channel i is NaN on exactly mask i, so the function's joint mask is mask i; channels
  with identical masks share a run; every channel is kept by exactly one run; and after
  Andrea's real ``slowWaveAnalysis_new`` only the kept channel's outputs survive, equal to
  that channel's column of a direct call;
* epoch slicing: the first valid sample of every input decodes to file sample
  ``i0 + row``, through the raw, pairs-lead and tripole recipes, in volts from declared
  microvolts;
* the whole-file beats are re-based to each epoch (heartlocs, gapAfter, blankSpans);
* the spike consumer is ``process_dataset_v2`` (RULING 2026-10-08 (i)): the ruled step
  list (her process_dataset steps in her order, without step1a or step1b), resolved from
  processing_new, P = 300-3000 Hz and 4.5 sigma, step3b's R-peak guard kept; masked
  samples stay NaN, no spike lies in a NaN span or its 5 ms pad, a pad wider than her
  own fires the check, and an all-NaN, a constant or a non-integer R-peak input
  is refused by name (invariant 41); a constant spike channel is dropped by the wrapper;
* a recording whose hrv/breathing are "not computed" skips them with the reason; mmc is
  "not computed" on a "pre" recording with no beat train, and skipped for want of R-peaks
  on any other recording without beats.

Skips, with the reason, when MATLAB or ``processing_new`` is absent (as
``test_matlab_acceptance``).
"""

from __future__ import annotations

import json
import re
import string
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.emit import line_distrust as ld
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit.handoff import (
    EPOCH_START_SAMPLE_KEY,
    epoch_start_fields,
    matlab_signal_token,
    signal_from_matlab_token,
    write_mask_file,
)
from gems_blanking_v2.emit.hr_beats import write_hr_beats
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.extent.grid import n_grid_frames, seconds_to_sample
from gems_blanking_v2.extent.tolerance import extent_consumers
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.io import loadmat, savemat

from tests.conftest import make_line_distrust, make_mains_spike_t, make_slow
from tests.test_matlab_acceptance import _matlab, _processing_new

FS = 24414.0625
N_FILE = int(10.0 * FS) + 11
LABELS = ("RVN1", "RVN2", "RVN3", "LVN1", "LVN2", "LVN3", "ANT1", "ANT2", "ANT3")
PRIMES = np.array([2.0, 3.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0])
"""Column c holds (1-based file sample) x PRIMES[c] microvolts, so every derived signal
is (file sample) x a nonzero coefficient - the tripole's included."""
COEF = {"R_T": 0.5 * 2 + 0.5 * 7 - 3, "L_T": 0.5 * 11 + 0.5 * 17 - 13,
        "LVN2-RVN2": 13.0 - 3.0, "ANT1": 19.0, "ANT2": 23.0, "ANT3": 29.0}
MODEL = "0123456789abcdef0123456789abcdef"
NIGHT6 = Path(__file__).resolve().parents[1] / "matlab" / "night6"
HARNESS = Path(__file__).parent / "matlab"
GASTRIC = ("ANT1", "ANT2", "ANT3")
HR = "LVN2-RVN2"
SESSION_A = "syn_a_t01_bl_100000_20260101T150000Z"
SESSION_B = "syn_b_pre01_110000_20260101T160000Z"
SESSION_C = "syn_c_t01_bl_120000_20260101T170000Z"
SESSION_D = "syn_d_t01_bl_130000_20260101T180000Z"
SESSION_E = "syn_e_t01_bl_140000_20260101T190000Z"
SESSIONS = (SESSION_A, SESSION_B, SESSION_C, SESSION_D, SESSION_E)
READS_A: dict[str, tuple[str, ...]] = {"spikes": ("L_T", "R_T"), "slow_wave": GASTRIC,
                                       "mmc": GASTRIC, "hrv": (HR,), "breathing": (HR,),
                                       "velocity": ()}
READS_B: dict[str, tuple[str, ...]] = {"spikes": ("L_T",), "slow_wave": GASTRIC,
                                       "mmc": GASTRIC, "hrv": (), "breathing": (),
                                       "velocity": ()}
READS_E: dict[str, tuple[str, ...]] = {**READS_B, "spikes": ("L_T", "R_T")}
SPANS_A = [mk.MaskSpan("spikes", "L_T", 1.003, 1.4, "in_band"),
           mk.MaskSpan("spikes", "L_T", 6.0, 6.5, "in_band"),
           mk.MaskSpan("spikes", "R_T", 0.0, 10.1, "in_band"),  # dead: dropped, L_T runs
           mk.MaskSpan("slow_wave", "ANT3", 0.5, 1.5, "in_band"),
           mk.MaskSpan("slow_wave", "ANT1", 7.0, 8.0, "in_band"),
           mk.MaskSpan("mmc", "ANT2", 3.0, 3.9, "in_band"),
           mk.MaskSpan("mmc", "ANT1", 3.95, 5.2, "in_band"),  # crosses both epoch edges
           mk.MaskSpan("hrv", HR, 1.0, 1.2, "in_band"),
           mk.MaskSpan("hrv", HR, 8.1, 8.3, "in_band"),
           mk.MaskSpan("breathing", HR, 2.5, 3.0, "in_band")]  # differs from hrv: 2 runs
SPANS_B = [mk.MaskSpan("spikes", "L_T", 4.0, 4.5, "in_band"),
           mk.MaskSpan("slow_wave", "ANT2", 0.0, 10.1, "in_band"),  # dead: ITS run refused
           mk.MaskSpan("mmc", "ANT3", 9.0, 9.9, "in_band")]
SPANS_E = [mk.MaskSpan("spikes", "R_T", 2.0, 2.5, "in_band"),
           mk.MaskSpan("slow_wave", "ANT1", 3.0, 4.0, "in_band")]
NOT_COMPUTED = {"hrv": "no count-gated beat train (synthetic)",
                "breathing": "no count-gated beat train (synthetic)"}
EPOCHS_A = ((0.0, int(round(4.0 * FS))), (5.0, N_FILE - round(5.0 * FS)))
BEATS_S = np.array([t for t in np.arange(0.05, 9.99, 0.113) if not 3.5 <= t < 5.5])
BEAT_BLANK_S = [[3.5, 5.5]]
SHIFT_SAMPLES = 3.4
"""Session D's mask file has epochStart_s moved by this many samples AFTER it was written:
a reader that rounded seconds x fs would slice 3 samples late."""
CONSTANT_UV = {"LVN1": 5.0, "LVN2": 7.0, "LVN3": 11.0}
"""Session E's left contacts are constant, so its L_T is constant (1 uV)."""
V2_STEPS_HERS = re.compile(r"^\s*D\s*=\s*(step\w+)\(D,\s*P,\s*plotMode\);", re.MULTILINE)


def _channels() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, lab in enumerate(LABELS):
        ch: dict[str, Any] = {"label": lab, "signal_index": i, "config": "independent"}
        if lab.startswith("ANT"):
            ch["role"] = "stomach"
        else:
            ch.update(role="nerve", cuff_id=lab[0], contact_index=int(lab[-1]))
        out.append(ch)
    return out


def _provenance(session: str, extra: dict[str, Any]) -> MaskProvenance:
    return MaskProvenance(model={"mode": "pooled", "version": "synthetic", "corpus_hash": "0" * 32,
                                 "calibrator": "models/synthetic/cal.json"},
                          thresholds={"source": "synthetic"}, reference_values={"none": 0},
                          code_commit="test", generation_sha="0133349b3ebeff80",
                          routing_hash="test", created_at="2026-10-08T05:00:00+00:00",
                          recording=session, extra=extra)


def _write_epoch(folder: Path, session: str, reads: dict[str, tuple[str, ...]],
                 spans: list[mk.MaskSpan], start: float, n: int, extra: dict[str, Any],
                 not_computed: dict[str, str] | None) -> dict[mk.MaskKey, mk.ConsumerMask]:
    masks = mk.build_masks(reads, spans, n_frames=n_grid_frames(n, FS), t0_s=start)
    rng = np.random.default_rng(int(start) + 7)
    spike_sigs = reads["spikes"]
    line = make_line_distrust({s: rng.normal(0.0, 5.0, n) for s in spike_sigs}, FS,
                              recording=session, epoch_start_s=start,
                              motion={s: masks[("spikes", s, "300-3000")] for s in spike_sigs})
    write_mask_file(folder / f"e{round(start)}_masks.mat", masks, _provenance(session, extra),
                    signals=reads, fs=FS, n_samples=n, epoch_start_s=start, min_retention=0.5,
                    animal_median={f"{c}|{s}|{b}": 0.3 for c, s, b in masks},
                    line_distrust=line, not_computed=not_computed,
                    epoch_start_sample=seconds_to_sample(start, FS),
                    release="synthetic night6 test: spans are large on purpose")
    return masks


def _shift_start_seconds(path: Path) -> None:
    """Move the file's epochStart_s by SHIFT_SAMPLES samples, leaving the stored sample."""
    m = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    m["epochStart_s"] = float(np.asarray(m["epochStart_s"]).squeeze()) + SHIFT_SAMPLES / FS
    savemat(path, m, do_compression=True)


def _signal(session: str) -> np.ndarray:
    idx = np.arange(1, N_FILE + 1, dtype=np.float64)
    y: np.ndarray = np.asarray(idx[:, None] * PRIMES[None, :], dtype=np.float64)
    if session == SESSION_E:
        for lab, v in CONSTANT_UV.items():
            y[:, LABELS.index(lab)] = v
    return y


def _store(root: Path) -> dict[str, Any]:
    """Write the synthetic store; return what each epoch's input should look like."""
    (root / "rec").mkdir(parents=True)
    expect: dict[str, Any] = {}
    for session, reads, spans, epochs, nc, beats, cond in (
            (SESSION_A, READS_A, SPANS_A, EPOCHS_A, None, BEATS_S, "baseline"),
            (SESSION_B, READS_B, SPANS_B, ((0.0, N_FILE),), NOT_COMPUTED, None, "pre"),
            # a beats file whose beats all lie before this epoch: HR refused (inv. 41)
            (SESSION_C, READS_A, SPANS_A, ((6.0, N_FILE - round(6.0 * FS)),), None,
             BEATS_S[BEATS_S < 3.5], "baseline"),
            # epochStart_s moved after writing: slicing must follow the stored sample
            (SESSION_D, READS_A, SPANS_A, (EPOCHS_A[1],), None, BEATS_S, "baseline"),
            # a baseline with no beat train and a constant left cuff
            (SESSION_E, READS_E, SPANS_E, ((0.0, N_FILE),), NOT_COMPUTED, None, "baseline")):
        sdir = root / "data" / "T" / session
        mdir = sdir / "masks" / MODEL
        mdir.mkdir(parents=True)
        savemat(root / "rec" / f"{session}_sig.mat",
                {"signal": _signal(session), "fs": FS,
                 "chanlabels": np.array(LABELS, dtype=object).reshape(1, -1)})
        meta = {"session": session, "animal": "T", "channels": _channels(),
                "source_path": f"rec/{session}_sig.mat"}
        (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")
        extra: dict[str, Any] = {"condition": cond}
        if beats is not None:
            write_hr_beats(sdir / f"{session}_beats.mat", beats, fs=FS, epoch_start_s=0.0,
                           n_samples=N_FILE, channel=HR, source="synthetic",
                           gap_after=np.arange(beats.size) % 7 == 3,
                           blank_spans_s=BEAT_BLANK_S)
            extra["beats_file"] = {"hrv_beats": f"data/T/{session}/{session}_beats.mat"}
        elif nc is not None:
            extra["hr_channel"] = {"not_computed": nc["hrv"]}
        for start, n in epochs:
            masks = _write_epoch(mdir, session, reads, spans, start, n, extra, nc)
            if session == SESSION_D:
                _shift_start_seconds(mdir / f"e{round(start)}_masks.mat")
            expect[f"{session}/e{round(start)}"] = {"start": start, "n": n, "masks": masks,
                                                    "reads": reads}
    return expect


def _runs(mask: np.ndarray) -> list[list[int]]:
    """Return the True runs as 1-based inclusive [first, last] rows."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [[int(a) + 1, int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1),
                                                 strict=True)]


def _expected_nan(consumer: str, sig: str, e: dict[str, Any], session: str) -> np.ndarray:
    m = e["masks"][(consumer, sig, extent_consumers()[consumer].band)]
    if consumer != "spikes":
        return mk.frames_to_samples(m.invalid, FS, e["n"])
    # the spike input: motion union distrusted minutes, as the Python twin builds it
    rng = np.random.default_rng(int(e["start"]) + 7)
    reads = e["reads"]["spikes"]
    line = make_line_distrust({s: rng.normal(0.0, 5.0, e["n"]) for s in reads}, FS,
                              recording=session, epoch_start_s=e["start"],
                              motion={s: e["masks"][("spikes", s, "300-3000")] for s in reads})
    return np.isnan(ld.spike_input(np.ones(e["n"]), FS, m, line))


def _slow_wave_groups(e: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """``(mask channel, kept channels)`` per slow-wave run: identical masks share a run."""
    band = extent_consumers()["slow_wave"].band
    todo, out = list(GASTRIC), []
    while todo:
        lead = e["masks"][("slow_wave", todo[0], band)].invalid
        same = [s for s in todo
                if np.array_equal(e["masks"][("slow_wave", s, band)].invalid, lead)]
        out.append((todo[0], same))
        todo = [s for s in todo if s not in same]
    return out


def _names(seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = list(string.ascii_letters[:6]) + list("0129_-") + ["_minus_", "minus", "_mi"]
    return ["".join(rng.choice(alphabet, size=rng.integers(1, 7))) for _ in range(400)]


EDGE_NAMES = ["LVN2-RVN2", "L_T", "ANT1", "-", "a--b", "a-b-c", "_minus_", "x_minus_y",
              "a_minus_minus_b", "_minus_minus_", "a-_minus_b", "LVN2_minus_RVN2-x", "", "a-"]


def _py(f: Callable[[str], str], x: str) -> str:
    try:
        return str(f(x))
    except ValueError:
        return "!"


# ---------------------------------------------------------------- item 1: the start sample

@settings(max_examples=300, deadline=None)
@given(start=st.floats(min_value=0.0, max_value=2.0e5, allow_nan=False),
       fs=st.sampled_from([FS, 24414.0, 1000.0, 2000.0, 610.3515625]))
def test_epoch_start_sample_round_trips_through_the_mask_file_format(
        start: float, fs: float, tmp_path_factory: pytest.TempPathFactory) -> None:
    """The stored sample is seconds_to_sample of the start, exactly, after savemat/loadmat."""
    fields = epoch_start_fields(start, fs)
    f = tmp_path_factory.mktemp("start") / "s.mat"
    savemat(f, fields)
    back = float(loadmat(f)[EPOCH_START_SAMPLE_KEY].squeeze())
    assert back == int(back) == seconds_to_sample(start, fs)
    assert float(loadmat(f)["epochStart_s"].squeeze()) == start
    assert epoch_start_fields(start, fs, seconds_to_sample(start, fs)) == fields


def test_epoch_start_sample_must_agree_with_the_seconds() -> None:
    """A caller's sample that disagrees with round(start x fs), or is not an int, raises."""
    i0 = seconds_to_sample(132.0, FS)
    assert epoch_start_fields(132.0, FS, i0)[EPOCH_START_SAMPLE_KEY] == float(i0)
    with pytest.raises(ValueError, match="disagrees"):
        epoch_start_fields(132.0, FS, i0 + 1)
    with pytest.raises(TypeError):
        epoch_start_fields(132.0, FS, float(i0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        epoch_start_fields(0.0, FS, False)
    with pytest.raises(ValueError, match="before"):
        epoch_start_fields(-1.0, FS)


def test_the_mask_file_carries_the_start_sample(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    _write_epoch(folder, SESSION_A, READS_A, SPANS_A, 5.0, EPOCHS_A[1][1], {}, None)
    m = loadmat(folder / "e5_masks.mat")
    assert float(m[EPOCH_START_SAMPLE_KEY].squeeze()) == round(5.0 * FS)


def test_matlab_never_converts_seconds_to_samples() -> None:
    """No seconds x fs rounding for indexing anywhere in the wrapper (invariant 15)."""
    for f in sorted(NIGHT6.glob("*.m")):
        code = "\n".join(line.split("%", 1)[0]
                         for line in f.read_text(encoding="utf-8").splitlines())
        assert "round_half_even" not in code, f.name
        assert not re.search(r"epochStart_s\)?\s*\*|\*\s*[\w.()]*epochStart_s", code), f.name
    assert "epochStartSample0" in (NIGHT6 / "night6_prepare_epoch.m").read_text(encoding="utf-8")


# ---------------------------------------------------------------- item 3: her step list

RULING_I_STEPS = ["step1_bandpass", "step2_noise_sigma", "step3_detect", "step3b_envelope",
                  "step4_waveforms", "step5c_modality_test", "step6_spike_report"]
"""RULING 2026-10-08 (i) 1, verbatim order."""
HERS_NOT_IN_RULING = {"step1a_blank_cardiac", "step1b_remove_cardiac"}
"""Her process_dataset.m calls step1a (her cardiac blank); step1b is never called. The
ruling's list has neither: reported to Andrea, built to the ruling."""


def test_process_dataset_v2_calls_the_ruled_steps_in_her_order() -> None:
    """night6_v2_steps is ruling (i)'s list, which is her process_dataset.m minus step1a/1b."""
    ours = re.findall(r"'(step\w+)'", (NIGHT6 / "night6_v2_steps.m").read_text(encoding="utf-8"))
    assert ours == RULING_I_STEPS
    pnew = _processing_new()
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    hers = V2_STEPS_HERS.findall((pnew / "process_dataset.m").read_text(encoding="utf-8"))
    assert hers, "no step calls found in her process_dataset.m"
    assert ours == [s for s in hers if s not in HERS_NOT_IN_RULING]  # same order
    assert set(hers) - set(ours) == {"step1a_blank_cardiac"}  # the reported difference
    for s in ours:
        assert (pnew / f"{s}.m").is_file(), s
        assert not (NIGHT6 / f"{s}.m").exists(), f"{s} must not be copied into the wrapper"


# ---------------------------------------------------------------- the MATLAB run

def test_night6_wrapper_slices_masks_and_skips(tmp_path: Path) -> None:
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    root = tmp_path / "store"
    expect = _store(root)
    names = [n for n in EDGE_NAMES + _names(5) if n]
    out_root = tmp_path / "out"
    mmc_units = _mmc_units_case(tmp_path)
    sw = _slow_wave_case(tmp_path)
    v2 = _v2_case(tmp_path)
    case: dict[str, Any] = {"tokens": names, "signals": names,
            "gems_root": root.as_posix(), "units": "uV", "out_root": out_root.as_posix(),
            "mask_folders": [(root / "data" / "T" / s / "masks" / MODEL).as_posix()
                             for s in SESSIONS],
            "resume": {"mask_folder": (root / "data" / "T" / SESSION_A / "masks" / MODEL)
                       .as_posix(),
                       "out_dir": (out_root / "T" / SESSION_A / MODEL).as_posix(),
                       "keep_tag": "e0", "stale_tag": "e5"},
            "mmc_units": mmc_units, "slow_wave": sw["case"], "v2": v2["case"],
            "fallback": _fallback_case(tmp_path, root)}
    case_file, res_file = tmp_path / "case.json", tmp_path / "result.json"
    case_file.write_text(json.dumps(case, ensure_ascii=True), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); "
           f"night6_check('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=1500, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res = json.loads(res_file.read_text(encoding="utf-8"))
    assert res["errors"] == [""] * len(SESSIONS), res["errors"]
    assert res["resume_error"] == ""

    _check_tokens(names, res)
    for key, e in expect.items():
        if key.startswith(SESSION_E):
            _check_constant_cuff(out_root / "T" / SESSION_E / MODEL / "e0")
        else:
            _check_epoch(key, e, root, out_root)
    _check_resume(out_root / "T" / SESSION_A / MODEL)
    _check_mmc_units(mmc_units, res)
    _check_slow_wave_keep(sw, res["slow_wave"])
    _check_v2(v2, res["v2"], pnew)
    _check_fallback(res, out_root, tmp_path / "out_fb")


FALLBACK_OK = {"schema": 1, "ruling": "test", "entries": [
    {"animal": "T", "cuff": "L", "reason": "synthetic: peri-R excess classified as leak"}]}
FALLBACK_BAD = {
    "top_key": {**FALLBACK_OK, "extra": 1},
    "entry_key": {"schema": 1, "entries": [{"animal": "T", "cuff": "L", "reason": "x",
                                             "channel": "L_T"}]},
    "cuff": {"schema": 1, "entries": [{"animal": "T", "cuff": "X", "reason": "x"}]},
    "no_reason": {"schema": 1, "entries": [{"animal": "T", "cuff": "L"}]},
    "duplicate": {"schema": 1, "entries": [{"animal": "T", "cuff": "L", "reason": "x"},
                                           {"animal": "T", "cuff": "L", "reason": "y"}]},
    "schema": {"schema": 2, "entries": []},
}
FALLBACK_BAD_IDS = {"top_key": "night6:fallbackKey", "entry_key": "night6:fallbackKey",
                    "cuff": "night6:fallbackEntry", "no_reason": "night6:fallbackEntry",
                    "duplicate": "night6:fallbackDuplicate", "schema": "night6:fallbackSchema"}


def _fallback_case(tmp: Path, root: Path) -> dict[str, Any]:
    """A fallback list naming T x L, and one malformed list per refusal."""
    d = tmp / "fb"
    d.mkdir()
    (d / "ok.json").write_text(json.dumps(FALLBACK_OK), encoding="utf-8", newline="\n")
    bad = []
    for name, doc in FALLBACK_BAD.items():
        (d / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8", newline="\n")
        bad.append((d / f"{name}.json").as_posix())
    return {"file_ok": (d / "ok.json").as_posix(), "bad_files": bad,
            "mask_folder": (root / "data" / "T" / SESSION_A / "masks" / MODEL).as_posix(),
            "out_root": (tmp / "out_fb").as_posix()}


def _check_fallback(res: dict[str, Any], out_root: Path, out_fb: Path) -> None:
    """Default list empty and recorded; T x L listed -> L_T falls back; malformed refused."""
    assert np.atleast_1d(res["fallback_bad"]).tolist() == list(FALLBACK_BAD_IDS.values())
    assert res["fallback_error"] == "", res["fallback_error"]
    default = json.loads((out_root / "T" / SESSION_A / MODEL / "e5" / "night6_record.json")
                         .read_text(encoding="utf-8"))
    assert default["step1a_fallback"]["keys"] == []
    assert Path(default["step1a_fallback"]["file"]).name == "step1a_fallback.json"
    assert default["runs"][0]["step1a_fallback"] == []
    listed = json.loads((out_fb / "T" / SESSION_A / MODEL / "e5" / "night6_record.json")
                        .read_text(encoding="utf-8"))
    assert np.atleast_1d(listed["step1a_fallback"]["keys"]).tolist() == ["T|L"]
    assert len(listed["step1a_fallback"]["sha256"]) == 64
    assert listed["runs"][0]["call"] == "process_dataset_v2"
    assert np.atleast_1d(listed["runs"][0]["step1a_fallback"]).tolist() == ["L_T"]


MMC_FS = 2000.0
MMC_N = 80_000  # 40 s
MMC_HALF = round(25 / 1000 * MMC_FS)  # extract_mmc's default cardiacBlankMs = 25


def _mmc_units_case(tmp: Path) -> dict[str, Any]:
    """Write a small gastric input and its beats file for the mmc R-peak unit check."""
    rng = np.random.default_rng(23)
    beats_s = np.cumsum(rng.uniform(0.15, 0.19, 300))
    beats_s = beats_s[beats_s < (MMC_N - 2 * MMC_HALF) / MMC_FS]
    beats = write_hr_beats(tmp / "mmc_beats.mat", beats_s, fs=MMC_FS, epoch_start_s=0.0,
                           n_samples=MMC_N, channel=HR, source="synthetic")
    savemat(tmp / "mmc_in.mat", {"yOut": rng.normal(0.0, 20e-6, (MMC_N, 3)), "fs": MMC_FS})
    return {"beats_file": beats.as_posix(), "input_file": (tmp / "mmc_in.mat").as_posix(),
            "fs": MMC_FS, "n": MMC_N}


def _check_mmc_units(u: dict[str, Any], res: dict[str, Any]) -> None:
    """Check extract_mmc blanked around the stored beats: samples declared as samples.

    With the unit wrong, extract_mmc reads sample indices as seconds and every R-peak
    moves by a factor of fs - to the last sample - so both checks fail.
    """
    assert res["mmc_error"] == "", res["mmc_error"]
    heartlocs = loadmat(u["beats_file"])["heartlocs"].ravel().astype(np.int64)
    assert np.atleast_1d(res["mmc_rpeak_samples"]).tolist() == heartlocs.tolist()
    blanked = np.zeros(MMC_N, dtype=bool)
    for h in heartlocs:  # 1-based: rows h-half .. h+half
        blanked[max(h - MMC_HALF, 1) - 1:min(h + MMC_HALF, MMC_N)] = True
    want = 100.0 * blanked.mean()
    assert res["mmc_pct_blanked"] == pytest.approx([want] * 3, rel=1e-12)


SW_FS = 200.0
SW_DUR_S = 360.0
SW_FREQ_HZ = (0.05, 0.075, 0.1)   # 3, 4.5 and 6 cpm: three distinguishable columns
SW_MASK_S = ((100.0, 112.0), (230.0, 236.0))


def _slow_wave_case(tmp: Path) -> dict[str, Any]:
    """Three different slow waves with ANT2's mask on all three columns (ANT2 kept)."""
    cols = [make_slow(SW_FS, SW_DUR_S, freq_hz=f, seed=40 + k, noise_uv=5.0).signal * 1e-6
            for k, f in enumerate(SW_FREQ_HZ)]
    x = np.column_stack(cols)
    for a, b in SW_MASK_S:
        x[round(a * SW_FS):round(b * SW_FS), :] = np.nan
    savemat(tmp / "sw_in.mat", {"X": x, "fs": SW_FS})
    case = {"input_file": (tmp / "sw_in.mat").as_posix(), "signals": list(GASTRIC),
            "keep": ["ANT2"], "mask_signal": "ANT2", "base": "e0", "low_pass_on": True,
            "cutoff": 0.15, "order": 2, "window": 5.0, "edge_s": 15.0,
            "direct_dir": (tmp / "sw_direct").as_posix(), "run_dir": (tmp / "sw_run").as_posix()}
    return {"case": case}


def _check_slow_wave_keep(sw: dict[str, Any], r: dict[str, Any]) -> None:
    """Only ANT2's outputs survive, equal to column 2 of a direct call, not column 1 or 3."""
    assert r["error"] == "", r["error"]
    assert r["files"] == ["e0_slowWaves_ANT2.mat"]  # the 3-column file is gone
    direct = [np.atleast_1d(p).tolist() for p in r["direct_peaks"]]
    assert len({tuple(p) for p in direct}) == 3  # the columns really differ
    assert np.atleast_1d(r["kept_peaks"]["ANT2"]).tolist() == direct[1]
    assert r["kept_avg"]["ANT2"] == pytest.approx(np.atleast_1d(r["direct_avg"])[1], rel=0, abs=0)
    assert r["kept_cols"]["ANT2"] == 1
    assert r["kept_channel"]["ANT2"] == "ANT2"
    assert np.atleast_1d(r["kept"]["deleted"]).tolist() == ["e0_swm_ANT2_slowWaves.mat"]


V2_DUR_S = 40.0
V2_NAN_S = {0: ((10.0, 10.5), (25.0, 25.2)), 1: ((5.0, 5.1),)}
V2_PAD_MS = 5.0


def _v2_case(tmp: Path) -> dict[str, Any]:
    """Two synthetic tripole channels (volts) with NaN spans, and integer R-peaks."""
    chans = [make_mains_spike_t(FS, V2_DUR_S, seed=11 + k).signal * 1e-6 for k in range(2)]
    y = np.column_stack(chans)
    for k, spans in V2_NAN_S.items():
        for a, b in spans:
            y[round(a * FS):round(b * FS), k] = np.nan
    rpeaks = np.round(np.arange(0.1, V2_DUR_S - 0.1, 0.17) * FS).astype(np.float64) + 1.0
    savemat(tmp / "v2_in.mat", {"y": y, "fs": FS, "rpeakSamples": rpeaks})
    case = {"input_file": (tmp / "v2_in.mat").as_posix(), "labels": ["L_T", "R_T"],
            "wide_pad_ms": 200.0}
    return {"case": case, "y": y, "n_rpeaks": rpeaks.size, "rpeaks": rpeaks}


def _check_v2(v2: dict[str, Any], r: dict[str, Any], pnew: Path) -> None:
    """process_dataset_v2: her steps, her parameters, NaN kept, no spike near a NaN."""
    assert r["error"] == "", r["error"]
    want = re.findall(r"'(step\w+)'", (NIGHT6 / "night6_v2_steps.m").read_text(encoding="utf-8"))
    assert np.atleast_1d(r["steps"]).tolist() == want
    assert {Path(d).resolve() for d in np.atleast_1d(r["step_dirs"]).tolist()} == {pnew.resolve()}
    assert np.atleast_1d(r["bandpass"]).tolist() == [300, 3000, 4.5]
    assert r["n_rpeaks"] == v2["n_rpeaks"]
    assert r["input_nan_still_nan"] is True
    assert r["step1a_ran"] is False  # ruling (i) 1: heartbeats only as the mask's NaN
    assert r["rpeak_guard_ms"] == 15  # step3b's guard (her P.envCardiacGuardMs)
    pad = int(np.ceil(V2_PAD_MS * 1e-3 * FS))
    checks = r["spike_check"] if isinstance(r["spike_check"], list) else [r["spike_check"]]
    centers = r["centers"] if isinstance(r["centers"], list) else [r["centers"]]
    for k, (chk, cen) in enumerate(zip(checks, centers, strict=True)):
        assert chk["n_in_pad"] == 0
        assert chk["pad_samples"] == pad
        assert chk["min_gap_samples"] >= pad
        c = np.atleast_1d(cen).astype(np.int64)
        assert c.size > 200, (k, c.size)  # it detects: 20 spikes/s of 40 uV in 5 uV noise
        nan = np.isnan(v2["y"][:, k])
        near = np.convolve(nan.astype(np.int64), np.ones(2 * pad + 1, dtype=np.int64),
                           mode="same") > 0
        assert not near[c - 1].any(), k  # independently of the MATLAB check
    # ruling (j) 1 fallback on channel 1 only: her step1a's +/-15 ms windows, exactly
    assert r["fb_error"] == "", r["fb_error"]
    assert np.atleast_1d(r["fb_channels"]).tolist() == ["L_T"]
    n = v2["y"].shape[0]
    w = round(15e-3 * FS)
    blank = np.zeros(n, dtype=bool)
    for rs in v2["rpeaks"].astype(np.int64):
        blank[max(1, rs - w) - 1:min(n, rs + w)] = True
    want0 = int((blank & ~np.isnan(v2["y"][:, 0])).sum())
    assert np.atleast_1d(r["fb_nan_added"]).tolist() == [want0, 0]
    fb_checks = r["fb_spike_check"] if isinstance(r["fb_spike_check"], list) \
        else [r["fb_spike_check"]]
    assert all(c["n_in_pad"] == 0 for c in fb_checks)
    assert r["fb_no_rpeaks"] == "process_dataset_v2:step1aNoRpeaks"
    assert r["wide_pad"] == "process_dataset_v2:spikeInMask"
    assert r["all_nan"] == "process_dataset_v2:noValidSamples"
    assert r["constant"] == "process_dataset_v2:constantInput"
    assert r["bad_rpeaks"] == "process_dataset_v2:rpeaks"


def _check_resume(base: Path) -> None:
    """Check the resume pass: same mask file skipped; another mask file's record rerun aside."""
    kept = base / "e0"  # complete for THIS mask file: skipped, untouched
    assert json.loads((kept / "night6_record.json").read_text(encoding="utf-8"))["status"] \
        == "complete"
    assert (kept / "stale_marker.txt").is_file()
    assert not (base / "e0.old1").exists()
    old, new = base / "e5.old1", base / "e5"  # complete for ANOTHER mask file: rerun
    assert (old / "stale_marker.txt").is_file()
    assert json.loads((old / "night6_record.json").read_text(encoding="utf-8"))[
        "mask_file_sha256"] == "deadbeef"
    assert not (new / "stale_marker.txt").exists()  # nothing stale beside the new outputs
    assert json.loads((new / "night6_record.json").read_text(encoding="utf-8"))["status"] \
        == "dry_run"


def _check_tokens(names: list[str], res: dict[str, Any]) -> None:
    """Check that MATLAB inverts Python's tokens and refuses exactly what Python refuses."""
    for name, back, fwd in zip(names, res["from_token"], res["to_token"], strict=True):
        py_inv = _py(signal_from_matlab_token, name)
        assert back == (py_inv if py_inv != "!" else "!night6:badToken"), name
        py_fwd = _py(matlab_signal_token, name)
        assert fwd == (py_fwd if py_fwd != "!" else "!night6:badSignal"), name
        if py_fwd != "!":
            assert signal_from_matlab_token(py_fwd) == name  # the Python pair is exact too
    assert sum(_py(matlab_signal_token, n) != "!" for n in names) > 100


def _check_constant_cuff(d: Path) -> None:
    """Session E: constant L_T dropped by name; no beat train on a baseline -> no_rpeaks."""
    rec = json.loads((d / "night6_record.json").read_text(encoding="utf-8"))
    cons = rec["consumers"]
    assert rec["runs"][0]["call"] == "process_dataset_v2"
    assert np.atleast_1d(rec["runs"][0]["dropped_no_valid_samples"]).tolist() == ["L_T"]
    assert "constant input in [L_T]" in cons["spikes"]["reason"]
    assert cons["spikes"]["status"] == "planned"
    assert cons["mmc"]["status"] == "skipped_no_rpeaks"  # not 'pre': Andrea's rule is pre only
    for c in ("hrv", "breathing"):
        assert cons[c]["status"] == "skipped_not_computed"


def _check_epoch(key: str, e: dict[str, Any], root: Path,  # noqa: PLR0915 - one check per rule
                 out_root: Path) -> None:
    """Check one epoch's record: plan, skips, NaN placement, slicing, re-based beats."""
    session, tag = key.split("/")
    rec = json.loads((out_root / "T" / session / MODEL / tag / "night6_record.json")
                     .read_text(encoding="utf-8"))
    i0 = round(e["start"] * FS)  # what the writer stored; D's seconds were moved after
    # A/e0 is marked complete by the harness's resume pass (and then skipped)
    assert rec["status"] == ("complete" if key == f"{SESSION_A}/e0" else "dry_run")
    assert rec["epoch"]["start_sample_0based"] == i0
    assert rec["epoch"]["n_samples"] == e["n"]
    if session == SESSION_D:
        assert rec["epoch"]["start_s"] == pytest.approx(e["start"] + SHIFT_SAMPLES / FS,
                                                        rel=0, abs=1e-12)
    cons = rec["consumers"]
    assert cons["velocity"]["status"] == "reads_nothing"
    groups = _slow_wave_groups(e)
    want_runs = [("process_dataset_v2", ["spikes"])]
    if session == SESSION_B:
        for c in ("hrv", "breathing"):
            assert cons[c]["status"] == "skipped_not_computed"
            assert cons[c]["reason"] == NOT_COMPUTED[c]
        # 'pre' with no beat train: mmc not computed (Andrea, 2026-10-09)
        assert cons["mmc"]["status"] == "skipped_not_computed"
        assert "pre" in cons["mmc"]["reason"]
        assert cons["spikes"]["status"] == "planned"
    else:  # R_T dead: dropped, L_T still planned
        assert np.atleast_1d(rec["runs"][0]["dropped_no_valid_samples"]).tolist() == ["R_T"]
        assert np.atleast_1d(cons["spikes"]["dropped_no_valid_samples"]).tolist() == ["R_T"]
        assert cons["spikes"]["status"] == "planned"
    if session == SESSION_C:  # beats file, none in the epoch: HR and mmc refused
        for c in ("hrv", "breathing"):
            assert cons[c]["status"] == "skipped_no_beats_in_epoch"
            assert "none inside this epoch" in cons[c]["reason"]
        assert cons["mmc"]["status"] == "skipped_no_rpeaks"
        assert "beats" not in rec or "epoch_file" not in rec["beats"]
    if session in (SESSION_A, SESSION_D):  # hrv/breathing: two masks, two runs
        want_runs += [("HR_BR_HRVAnalysis_beats", ["hrv"]),
                      ("HR_BR_HRVAnalysis_beats", ["breathing"])]
    want_runs += [("slowWaveAnalysis_new", ["slow_wave"])] * len(groups)
    if session in (SESSION_A, SESSION_D):
        want_runs.append(("extract_mmc", ["mmc"]))
    runs = rec["runs"]
    assert [(r["call"], np.atleast_1d(r["consumers"]).tolist()) for r in runs] == want_runs
    _check_slow_wave_runs(e, session, [r for r in runs if r["call"] == "slowWaveAnalysis_new"],
                          groups, cons["slow_wave"])
    for r in runs:
        consumer = np.atleast_1d(r["consumers"]).tolist()[0]
        inputs = r["inputs"] if isinstance(r["inputs"], list) else [r["inputs"]]
        assert [i["signal"] for i in inputs] == list(
            GASTRIC if consumer in ("slow_wave", "mmc") else e["reads"][consumer])
        for inp in inputs:
            sig = inp["signal"]
            mask_sig = r.get("mask_signal", sig)  # slow wave: one channel's mask on all
            want = _expected_nan(consumer, mask_sig, e, session)
            got = np.asarray(inp.get("nan_runs", []), dtype=np.int64).reshape(-1, 2)
            assert got.tolist() == _runs(want), (key, consumer, sig)
            assert inp["n_nan"] == int(want.sum())
            if want.all():
                assert "first_valid_row" not in inp  # absent, never null
                continue
            row = inp["first_valid_row"]
            assert row == int(np.flatnonzero(~want)[0]) + 1
            file_sample = inp["first_valid_value_V"] / (COEF[sig] * 1e-6)
            assert file_sample == pytest.approx(i0 + row, rel=1e-12), (key, sig)
    if session in (SESSION_A, SESSION_D):
        _check_beats(root / "data" / "T" / session / f"{session}_beats.mat",
                     Path(rec["beats"]["epoch_file"]), i0, e["n"])


def _check_slow_wave_runs(e: dict[str, Any], session: str, runs: list[dict[str, Any]],
                          groups: list[tuple[str, list[str]]], cons: dict[str, Any]) -> None:
    """One run per distinct mask; each run's joint mask is its channel's; every channel once."""
    assert [(r["mask_signal"], np.atleast_1d(r["keep"]).tolist()) for r in runs] == groups
    kept = [s for r in runs for s in np.atleast_1d(r["keep"]).tolist()]
    assert sorted(kept) == sorted(GASTRIC)  # every channel kept by exactly one run
    for r in runs:
        joint = np.zeros(e["n"], dtype=bool)
        inputs = r["inputs"]
        for inp in inputs:
            got = np.asarray(inp.get("nan_runs", []), dtype=np.int64).reshape(-1, 2)
            col = np.zeros(e["n"], dtype=bool)
            for a, b in got:
                col[a - 1:b] = True
            joint |= col
        mask_i = _expected_nan("slow_wave", r["mask_signal"], e, session)
        assert np.array_equal(joint, mask_i)  # the function's any(isnan) IS mask i
    for mask_sig, keep in groups:
        dead = _expected_nan("slow_wave", mask_sig, e, session).all()
        for s in keep:
            st = cons["channels"][s]
            assert st["mask_signal"] == mask_sig
            assert st["status"] == ("skipped_no_valid_samples" if dead else "planned")
    assert cons["status"] == "planned"


def _check_beats(whole_file: Path, epoch_file: Path, i0: int, n: int) -> None:
    """Check the whole-file beats re-based to the epoch: heartlocs, gapAfter, blankSpans."""
    whole, got = loadmat(whole_file), loadmat(epoch_file)
    h = whole["heartlocs"].ravel().astype(np.int64)
    keep = (h > i0) & (h <= i0 + n)
    assert got["heartlocs"].ravel().tolist() == (h[keep] - i0).tolist()
    assert (got["gapAfter"].ravel().astype(bool).tolist()
            == whole["gapAfter"].ravel().astype(bool)[keep].tolist())
    bs = whole["blankSpans"].reshape(-1, 2).astype(np.int64) - i0
    bs = bs[(bs[:, 1] >= 1) & (bs[:, 0] <= n)]
    bs = np.column_stack([np.maximum(bs[:, 0], 1), np.minimum(bs[:, 1], n)])
    assert got["blankSpans"].reshape(-1, 2).tolist() == bs.tolist()
    assert len(bs) == 1  # the rejected span crosses both epochs' edges
    assert float(got["fs"].squeeze()) == FS
