"""Task 15 acceptance: MATLAB loads the mask file and step1_bandpass.m honours it.

Emits masks for a synthetic multi-consumer recording with ``emit.masks``, writes the
``.mat`` handoff, and runs ``tests/matlab/check_step1_masks.m`` in ONE MATLAB process
against Andrea's ``processing_new`` (read-only: it is only put on the path). Checks:

* every ``blank_*`` / ``notmeasured_*`` span set lands on exactly the samples Python's
  sample mask says - first, last and count (invariant 15, 1-based inclusive);
* ``step1_bandpass`` (which tests ``isnan`` only) returns NaN on exactly the blanked
  samples and its valid-sample count equals the count expected from the masks
  (invariant 1: the mask is honoured because it is NaN);
* no exact-zero run longer than 2 samples in what MATLAB filtered or was given.

Skips, with the reason, when MATLAB or ``processing_new`` is not on this machine:
``GEMS_MATLAB`` (else the R2026a default path, else ``matlab`` on PATH) and
``GEMS_PROCESSING_NEW`` (else ``processing_new`` beside this checkout).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.extent.grid import n_grid_frames
from scipy.io import savemat

from tests.conftest import make_eng

FS = 24414.0625
N_SAMPLES = int(60.0 * FS) + 37  # a trailing partial frame, on purpose
HARNESS = Path(__file__).parent / "matlab"
DEFAULT_MATLAB = Path("C:/Program Files/MATLAB/R2026a/bin/matlab.exe")


def _matlab() -> Path | None:
    env = os.environ.get("GEMS_MATLAB")
    if env:
        return Path(env) if Path(env).is_file() else None
    if DEFAULT_MATLAB.is_file():
        return DEFAULT_MATLAB
    found = shutil.which("matlab")
    return Path(found) if found else None


def _processing_new() -> Path | None:
    env = os.environ.get("GEMS_PROCESSING_NEW")
    cand = Path(env) if env else Path(__file__).resolve().parents[2] / "processing_new"
    return cand if (cand / "step1_bandpass.m").is_file() else None


def _masks() -> dict[mk.MaskKey, mk.ConsumerMask]:
    n = n_grid_frames(N_SAMPLES, FS)
    spans = [
        mk.MaskSpan("spikes", "L_T", 10.003, 10.4, "in_band"),
        mk.MaskSpan("spikes", "L_T", 59.95, 61.0, "in_band"),  # runs into the partial frame
        mk.MaskSpan("slow_wave", "ANT1", 20.0, 35.0, "in_band"),
        mk.MaskSpan("mmc", "ANT1", 40.0, 41.0, "in_band"),
    ]
    sig = {"spikes": ("L_T",), "slow_wave": ("ANT1",), "mmc": ("ANT1",)}
    return mk.build_masks(sig, spans, n_frames=n, t0_s=0.0)


def test_matlab_step1_bandpass_honours_the_emitted_masks(tmp_path: Path) -> None:
    matlab, pnew = _matlab(), _processing_new()
    if matlab is None:
        pytest.skip("MATLAB is not on this machine (set GEMS_MATLAB)")
    if pnew is None:
        pytest.skip("processing_new is not on this machine (set GEMS_PROCESSING_NEW)")
    masks = _masks()
    prov = MaskProvenance(model={"mode": "pooled", "version": "synthetic", "corpus_hash": "0" * 32,
                                 "calibrator": "models/synthetic/cal.json"},
                          thresholds={"source": "synthetic"}, reference_values={"none": 0},
                          code_commit="test", generation_sha="0133349b3ebeff80",
                          routing_hash="test", created_at="2026-10-08T05:00:00+00:00",
                          recording="synthetic")
    gate = mk.EmitGate(held=False, reasons=(), retention_flagged=False, blank_held=False)
    mask_file = mk.write_mask_file(tmp_path / "synthetic_masks.mat", masks, prov, fs=FS,
                                   n_samples=N_SAMPLES, epoch_start_s=0.0, gate=gate)
    rng = np.random.default_rng(11)
    eng = make_eng(FS, N_SAMPLES / FS + 0.1, seed=12).signal[:N_SAMPLES]
    y = np.column_stack([eng, rng.normal(0.0, 20.0, N_SAMPLES)])
    sig_file = tmp_path / "synthetic_signal.mat"
    savemat(sig_file, {"y": y, "fs": FS, "channelLabels": np.array(["L_T", "ANT1"], dtype=object),
                       "spikeLabel": "L_T"})
    out = tmp_path / "result.json"
    cmd = (f"addpath('{pnew.as_posix()}'); addpath('{HARNESS.as_posix()}'); "
           f"check_step1_masks('{mask_file.as_posix()}', '{sig_file.as_posix()}', "
           f"'{out.as_posix()}');")
    done = subprocess.run([str(matlab), "-batch", cmd], capture_output=True, text=True,
                          timeout=900, check=False)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    res = json.loads(out.read_text(encoding="utf-8"))

    assert res["fs_matches"] and res["n_samples"] == N_SAMPLES
    by = {m["name"]: m for m in res["masks"]}
    expect = {f"blank_{c}_{s}": m.invalid for (c, s, _b), m in masks.items()}
    expect.update({f"notmeasured_mmc_{s}": f for s, f in mk.mmc_not_measured(masks).items()})
    assert set(by) == set(expect)
    for name, frames in expect.items():
        samples = mk.frames_to_samples(frames, FS, N_SAMPLES)
        idx = np.flatnonzero(samples)
        assert by[name]["n_nan"] == idx.size, name
        if idx.size:  # MATLAB indices are 1-based
            assert (by[name]["first"], by[name]["last"]) == (idx[0] + 1, idx[-1] + 1), name
    spikes = mk.frames_to_samples(masks[("spikes", "L_T", "300-3000")].invalid, FS, N_SAMPLES)
    assert spikes[-1]  # the partial frame took the last frame's state
    assert res["step1_valid"] == N_SAMPLES - int(spikes.sum())
    assert res["step1_nan_equals_blank"] is True
    assert res["max_zero_run_filtered"] <= 2 and res["max_zero_run_input"] <= 2
    assert res["band"] == [300, 3000]
