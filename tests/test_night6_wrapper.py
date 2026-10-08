"""Night 6 wrapper (matlab/night6): the MATLAB side of the mask handoff, checked from Python.

Builds a synthetic store - a 9-contact recording whose every sample encodes its own file
index, its ``meta.json``, a whole-file beats file and mask folders written by the real
``emit.handoff.write_mask_file`` - and runs ``tests/matlab/night6_check.m`` in ONE MATLAB
process. ``night6_run_recording`` runs with ``DryRun`` (the real loading, epoch slicing,
masking and skip logic; Andrea's functions are not called). Checks:

* the ``_minus_`` token round trip across the boundary, both directions, including the
  refusals (invariant 22) - ``night6_signal_from_token`` is the exact inverse of
  ``matlab_signal_token`` on fixed edge cases and 400 seeded random names;
* ``night6_round_half_even`` equals Python's ``round`` (the epoch start sample rule);
* every consumer input is NaN on exactly the samples its OWN mask names - 1-based
  inclusive spans into the epoch - and nowhere else (invariants 1, 2, 15): hrv and
  breathing on one channel get two different inputs;
* epoch slicing: the first valid sample of every input decodes to file sample
  ``i0 + row`` with ``i0 = round(epochStart_s * fs)``, through the raw, pairs-lead and
  tripole recipes, in volts from declared microvolts;
* the whole-file beats are re-based to each epoch (heartlocs, gapAfter, blankSpans);
* a recording whose hrv/breathing are "not computed" skips them with the reason, and
  mmc, which needs R-peaks, is skipped when there is no beats file.

Skips, with the reason, when MATLAB or ``processing_new`` is absent (as
``test_matlab_acceptance``).
"""

from __future__ import annotations

import json
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
    matlab_signal_token,
    signal_from_matlab_token,
    write_mask_file,
)
from gems_blanking_v2.emit.hr_beats import write_hr_beats
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.extent.grid import n_grid_frames
from gems_blanking_v2.extent.tolerance import extent_consumers
from scipy.io import loadmat, savemat

from tests.conftest import make_line_distrust
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
READS_A: dict[str, tuple[str, ...]] = {"spikes": ("L_T", "R_T"), "slow_wave": GASTRIC,
                                       "mmc": GASTRIC, "hrv": (HR,), "breathing": (HR,),
                                       "velocity": ()}
READS_B: dict[str, tuple[str, ...]] = {"spikes": ("L_T",), "slow_wave": GASTRIC,
                                       "mmc": GASTRIC, "hrv": (), "breathing": (),
                                       "velocity": ()}
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
           mk.MaskSpan("slow_wave", "ANT2", 0.0, 10.1, "in_band"),  # dead: run refused (inv. 41)
           mk.MaskSpan("mmc", "ANT3", 9.0, 9.9, "in_band")]
NOT_COMPUTED = {"hrv": "no count-gated beat train (synthetic)",
                "breathing": "no count-gated beat train (synthetic)"}
EPOCHS_A = ((0.0, int(round(4.0 * FS))), (5.0, N_FILE - round(5.0 * FS)))
BEATS_S = np.array([t for t in np.arange(0.05, 9.99, 0.113) if not 3.5 <= t < 5.5])
BEAT_BLANK_S = [[3.5, 5.5]]


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
                    release="synthetic night6 test: spans are large on purpose")
    return masks


def _store(root: Path) -> dict[str, Any]:
    """Write the synthetic store; return what each epoch's input should look like."""
    idx = np.arange(1, N_FILE + 1, dtype=np.float64)
    (root / "rec").mkdir(parents=True)
    expect: dict[str, Any] = {}
    for session, reads, spans, epochs, nc in (
            (SESSION_A, READS_A, SPANS_A, EPOCHS_A, None),
            (SESSION_B, READS_B, SPANS_B, ((0.0, N_FILE),), NOT_COMPUTED)):
        sdir = root / "data" / "T" / session
        mdir = sdir / "masks" / MODEL
        mdir.mkdir(parents=True)
        savemat(root / "rec" / f"{session}_sig.mat",
                {"signal": idx[:, None] * PRIMES[None, :], "fs": FS,
                 "chanlabels": np.array(LABELS, dtype=object).reshape(1, -1)})
        meta = {"session": session, "animal": "T", "channels": _channels(),
                "source_path": f"rec/{session}_sig.mat"}
        (sdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")
        extra: dict[str, Any] = {"condition": "baseline" if nc is None else "pre"}
        if nc is None:
            write_hr_beats(sdir / f"{session}_beats.mat", BEATS_S, fs=FS, epoch_start_s=0.0,
                           n_samples=N_FILE, channel=HR, source="synthetic",
                           gap_after=np.arange(BEATS_S.size) % 7 == 3,
                           blank_spans_s=BEAT_BLANK_S)
            extra["beats_file"] = {"hrv_beats": f"data/T/{session}/{session}_beats.mat"}
        else:
            extra["hr_channel"] = {"not_computed": nc["hrv"]}
        for start, n in epochs:
            masks = _write_epoch(mdir, session, reads, spans, start, n, extra, nc)
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


def test_night6_wrapper_slices_masks_and_skips(tmp_path: Path) -> None:
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    root = tmp_path / "store"
    expect = _store(root)
    names = [n for n in EDGE_NAMES + _names(5) if n]
    rounds = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.4999, 2.5000000001, 1e15 + 0.5,
              5.0 * FS, 132.0 * FS, 4.0 * FS, 122070.5, 122071.5]
    out_root = tmp_path / "out"
    case = {"tokens": names, "signals": names, "round_in": rounds,
            "gems_root": root.as_posix(), "units": "uV", "out_root": out_root.as_posix(),
            "mask_folders": [(root / "data" / "T" / s / "masks" / MODEL).as_posix()
                             for s in (SESSION_A, SESSION_B)]}
    case_file, res_file = tmp_path / "case.json", tmp_path / "result.json"
    case_file.write_text(json.dumps(case, ensure_ascii=True), encoding="utf-8", newline="\n")
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{NIGHT6.as_posix()}'); "
           f"addpath('{HARNESS.as_posix()}'); "
           f"night6_check('{case_file.as_posix()}', '{res_file.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    res = json.loads(res_file.read_text(encoding="utf-8"))
    assert res["errors"] == ["", ""], res["errors"]

    _check_tokens(names, res)
    assert res["rounded"] == [round(v) for v in rounds]
    for key, e in expect.items():
        _check_epoch(key, e, root, out_root)


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


def _check_epoch(key: str, e: dict[str, Any], root: Path, out_root: Path) -> None:
    """Check one epoch's record: plan, skips, NaN placement, slicing, re-based beats."""
    session, tag = key.split("/")
    rec = json.loads((out_root / "T" / session / MODEL / tag / "night6_record.json")
                     .read_text(encoding="utf-8"))
    i0 = round(e["start"] * FS)
    assert rec["status"] == "dry_run"
    assert rec["epoch"]["start_sample_0based"] == i0
    assert rec["epoch"]["n_samples"] == e["n"]
    cons = rec["consumers"]
    assert cons["velocity"]["status"] == "reads_nothing"
    want_runs = [("detectSortNerveSpikesECAP", ["spikes"]), ("slowWaveAnalysis_new", ["slow_wave"])]
    if session == SESSION_B:
        for c in ("hrv", "breathing"):
            assert cons[c]["status"] == "skipped_not_computed"
            assert cons[c]["reason"] == NOT_COMPUTED[c]
        assert cons["mmc"]["status"] == "skipped_no_rpeaks"
        assert cons["slow_wave"]["status"] == "skipped_no_valid_samples"
        assert "ANT2" in cons["slow_wave"]["reason"]
        assert cons["spikes"]["status"] == "planned"
    else:  # R_T dead: dropped, L_T still planned; hrv/breathing: two masks, two runs
        assert np.atleast_1d(rec["runs"][0]["dropped_no_valid_samples"]).tolist() == ["R_T"]
        assert cons["spikes"]["status"] == "planned"
        want_runs[1:1] = [("HR_BR_HRVAnalysis_beats", ["hrv"]),
                          ("HR_BR_HRVAnalysis_beats", ["breathing"])]
        want_runs.append(("extract_mmc", ["mmc"]))
    runs = rec["runs"]
    assert [(r["call"], np.atleast_1d(r["consumers"]).tolist()) for r in runs] == want_runs
    for r in runs:
        consumer = np.atleast_1d(r["consumers"]).tolist()[0]
        inputs = r["inputs"] if isinstance(r["inputs"], list) else [r["inputs"]]
        assert [i["signal"] for i in inputs] == list(
            GASTRIC if consumer in ("slow_wave", "mmc") else e["reads"][consumer])
        for inp in inputs:
            sig = inp["signal"]
            want = _expected_nan(consumer, sig, e, session)
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
    if session == SESSION_A:
        _check_beats(root / "data" / "T" / session / f"{session}_beats.mat",
                     Path(rec["beats"]["epoch_file"]), i0, e["n"])


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
