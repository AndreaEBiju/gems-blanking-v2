"""Hard invariant 1, and the ringing measurement task 01 asks for.

The writer side was already fixed upstream (``GEMSBlanking`` commit ``a95d1ff``,
``labeled_save.BLANK_FILL_VALUE = NaN``), so these tests are about keeping it that
way and about the quantity the spec asserts without ever having measured it.

The filter chain here is a **validated replica** of
``processing_new/step1_bandpass.m``: order-4 Butterworth, 100-5000 Hz at
fs = 24414.0625, zero-phase, with NaN gaps filled by linear interpolation
(``fillmissing(x,'linear','EndValues','nearest')``) and restored afterwards.
Measured against MATLAB R2026a on 2026-09-21:

* filter coefficients agree to 5.3e-15 (max absolute difference over b and a)
* the medians it produces match MATLAB's to within Monte-Carlo error, e.g. at
  500 uV of out-of-band content and a 10 ms gap: MATLAB 227.20 / 13.38 uV,
  replica 226.82 / 13.12 uV
* ``sosfiltfilt`` agrees with the ``filtfilt(b, a, ...)`` chain MATLAB actually
  runs to 1.6e-10 relative, so the replica uses ``sos`` per ``CLAUDE.md`` without
  drifting from what MATLAB does
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.io.nan_interop import (
    MAX_LEGAL_ZERO_RUN,
    ZeroRun,
    assert_no_zero_runs,
    audit_array,
    find_zero_runs,
    masked_count,
)
from scipy.io import loadmat, savemat
from scipy.signal import butter, sosfiltfilt, tf2sos

F64 = npt.NDArray[np.float64]

FS = 24414.0625
"""Nominal TDT rate, the rate the MATLAB measurement was taken at."""

MATLAB_BAND = (100.0, 5000.0)
"""``pipeline_params.m``: bandpassLow 100, bandpassHigh 5000, filterOrder 4."""

ENG_BAND = (300.0, 3000.0)
"""This project's band (A.5b). Measured too, because it is the one we will use."""

MATLAB_ORDER = 4
"""``P.filterOrder``. The spec's "order-8" is the *effective* order after filtfilt
applies it twice; the design order in the code is 4. Do not "fix" this to 8."""


# ---------------------------------------------------------------------------
# find_zero_runs / assert_no_zero_runs
# ---------------------------------------------------------------------------


def test_a_clean_signal_has_no_zero_runs() -> None:
    rng = np.random.default_rng(0)
    assert find_zero_runs(rng.standard_normal(10_000)) == []


def test_two_zeros_are_legal_and_three_are_not() -> None:
    """The invariant's line: a pair happens in real signal, a triple is a fill."""
    x = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    runs = find_zero_runs(x)
    assert len(runs) == 1
    assert (runs[0].start, runs[0].stop, runs[0].length) == (4, 7, 3)
    assert MAX_LEGAL_ZERO_RUN == 2


def test_nan_is_not_zero_and_breaks_a_run() -> None:
    """The whole point: NaN is masked, zero is signal."""
    x = np.array([0.0, 0.0, np.nan, 0.0, 0.0])
    assert find_zero_runs(x) == []
    assert find_zero_runs(np.full(100, np.nan)) == []


def test_a_masked_span_written_as_nan_passes_and_as_zero_fails() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5_000)

    nan_masked = x.copy()
    nan_masked[1000:1500] = np.nan
    assert_no_zero_runs(nan_masked, what="NaN-masked")

    zero_masked = x.copy()
    zero_masked[1000:1500] = 0.0
    with pytest.raises(ValueError, match="hard invariant 1"):
        assert_no_zero_runs(zero_masked, what="zero-masked")


def test_the_error_names_the_channel_the_span_and_the_duration() -> None:
    """"There is a zero run somewhere" is not actionable."""
    x = np.ones((5_000, 3))
    x[2000:2500, :] = 0.0
    with pytest.raises(ValueError, match="hard invariant 1") as exc:
        assert_no_zero_runs(x, what="yOut", fs=FS)
    message = str(exc.value)
    assert "channel 0" in message
    assert "[2000, 2500)" in message
    assert "20.5 ms" in message
    assert "isnan" in message


def test_runs_are_found_per_channel() -> None:
    x = np.ones((100, 3))
    x[10:20, 0] = 0.0
    x[50:80, 2] = 0.0
    runs = find_zero_runs(x)
    assert [(r.channel, r.start, r.stop) for r in runs] == [(0, 10, 20), (2, 50, 80)]


def test_a_run_at_either_edge_is_found() -> None:
    x = np.ones(50)
    x[:5] = 0.0
    x[-7:] = 0.0
    assert [(r.start, r.stop) for r in find_zero_runs(x)] == [(0, 5), (43, 50)]


def test_an_all_zero_channel_is_one_run() -> None:
    runs = find_zero_runs(np.zeros((1_000, 2)))
    assert len(runs) == 2
    assert all(r.length == 1_000 for r in runs)


def test_min_len_is_honoured_and_validated() -> None:
    x = np.array([1.0, 0.0, 0.0, 1.0])
    assert find_zero_runs(x, min_len=2) == [ZeroRun(channel=0, start=1, stop=3)]
    assert find_zero_runs(x, min_len=3) == []
    with pytest.raises(ValueError, match="min_len"):
        find_zero_runs(x, min_len=0)


def test_zero_run_reports_its_duration() -> None:
    # 10 ms is 244.14 samples, so the realised run is one sample short of 10 ms.
    run = ZeroRun(channel=0, start=0, stop=int(round(FS * 0.010)))
    assert run.duration_s(FS) == pytest.approx(0.010, abs=1.0 / FS)
    assert run.length == 244


def test_a_three_dimensional_array_is_rejected() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        find_zero_runs(np.zeros((2, 3, 4)))


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


def test_the_audit_recognises_a_pre_fix_file() -> None:
    """Zero-filled and NaN-free is the signature of the old writer."""
    x = np.ones((1_000, 2))
    x[100:200, :] = 0.0
    report = audit_array(x, source="old_blankmotion.mat")
    assert not report.is_clean
    assert report.predates_the_nan_fix
    assert report.nan_count == 0
    assert "Re-export it" in report.summary(fs=FS)
    assert "do not convert the zeros in place" in report.summary()


def test_the_audit_does_not_claim_a_mixed_file_predates_the_fix() -> None:
    """Zeros alongside NaN could be genuine zero-valued signal. A human decides."""
    x = np.ones((1_000, 2))
    x[100:200, :] = 0.0
    x[400:500, :] = np.nan
    report = audit_array(x)
    assert not report.predates_the_nan_fix
    assert "inspect by hand" in report.summary()


def test_the_audit_passes_a_correctly_written_file() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1_000, 2))
    x[100:200, :] = np.nan
    report = audit_array(x, source="new_blankmotion.mat")
    assert report.is_clean
    assert report.nan_count == 200
    assert "clean" in report.summary()


def test_the_audit_never_modifies_its_input() -> None:
    """Diagnostic only - inferring invalidity from zero is the bug, not the fix."""
    x = np.ones((100, 2))
    x[10:50, :] = 0.0
    before = x.copy()
    audit_array(x)
    assert np.array_equal(x, before)


# ---------------------------------------------------------------------------
# the .mat round trip - the acceptance criterion
# ---------------------------------------------------------------------------


def make_masked_recording(
    n_samples: int = 20_000, n_channels: int = 9, seed: int = 3
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """Build a synthetic recording with NaN-masked spans, as labeled_save writes it."""
    rng = np.random.default_rng(seed)
    y = 20.0 * rng.standard_normal((n_samples, n_channels))
    mask = np.zeros(n_samples, dtype=bool)
    for start, stop in ((1_000, 1_050), (5_000, 5_500), (12_345, 12_346)):
        mask[start:stop] = True
    y[mask, :] = np.nan
    return y, mask


def test_no_zero_runs_survives_a_mat_round_trip(tmp_path: Path) -> None:
    """``scipy.io`` is what labeled_save writes through, so round-trip through it."""
    y, _ = make_masked_recording()
    path = tmp_path / "recording_blankmotion.mat"

    savemat(str(path), {"yOut": y, "fs": FS, "blank_fill": "nan"})
    loaded = np.asarray(loadmat(str(path))["yOut"], dtype=np.float64)

    assert_no_zero_runs(loaded, what="yOut after round trip", fs=FS)
    assert audit_array(loaded).is_clean


def test_nan_preserved_across_a_mat_round_trip(tmp_path: Path) -> None:
    """Masked sample count out == masked sample count in, summed over channels."""
    y, mask = make_masked_recording()
    path = tmp_path / "recording_blankmotion.mat"

    savemat(str(path), {"yOut": y})
    loaded = np.asarray(loadmat(str(path))["yOut"], dtype=np.float64)

    expected = int(mask.sum()) * y.shape[1]
    assert masked_count(y) == expected
    assert masked_count(loaded) == expected
    assert np.array_equal(np.isnan(loaded), np.isnan(y))


def test_a_zero_filled_export_is_caught_before_it_reaches_matlab(tmp_path: Path) -> None:
    """What the old writer produced, and what the gate exists to stop."""
    y, _ = make_masked_recording()
    zeroed = np.nan_to_num(y, nan=0.0)
    path = tmp_path / "old_blankmotion.mat"

    savemat(str(path), {"yOut": zeroed})
    loaded = np.asarray(loadmat(str(path))["yOut"], dtype=np.float64)

    assert masked_count(loaded) == 0  # MATLAB's isnan() would find nothing masked
    with pytest.raises(ValueError, match="hard invariant 1"):
        assert_no_zero_runs(loaded, what="yOut", fs=FS)
    assert audit_array(loaded, source=path).predates_the_nan_fix


# ---------------------------------------------------------------------------
# the ringing measurement
# ---------------------------------------------------------------------------


def fill_linear(x: F64) -> F64:
    """Replicate MATLAB ``fillmissing(x,'linear','EndValues','nearest')``.

    ``np.interp`` is linear between valid samples and clamps to the nearest valid
    value beyond them, which is the same rule.
    """
    bad = np.isnan(x)
    if not bad.any():
        return x
    idx = np.arange(x.size)
    out = x.copy()
    out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return out


def step1_bandpass(x: F64, band: tuple[float, float] = MATLAB_BAND) -> F64:
    """Run the ``step1_bandpass.m`` chain: fill, zero-phase filter, restore NaN."""
    lo, hi = band
    nyq = FS / 2
    sos = tf2sos(*butter(MATLAB_ORDER, [lo / nyq, hi / nyq], btype="bandpass", output="ba"))
    invalid = np.isnan(x)
    out = np.asarray(sosfiltfilt(sos, fill_linear(x)), dtype=np.float64).copy()
    out[invalid] = np.nan
    return out


def ringing(
    band: tuple[float, float], lf_uv: float, gap_ms: float, n_pos: int = 25, seed: int = 0
) -> tuple[float, float]:
    """Median peak deviation in the 50 ms either side of a boundary.

    Returns ``(zeroed, nan)`` in microvolts, against the same host filtered with no
    gap at all. Swept over gap position because the extra damage from zeroing is set
    by the host's instantaneous value at the boundary, which is a random draw - a
    single position measures that draw, not the method.
    """
    rng = np.random.default_rng(seed)
    n = int(round(FS * 2.0))
    t = np.arange(n) / FS
    host = 20.0 * rng.standard_normal(n) + lf_uv * np.sin(2 * np.pi * 8 * t)
    ref = step1_bandpass(host, band)

    gap_n = int(round(FS * gap_ms / 1000.0))
    w = int(round(FS * 0.050))
    zeroed, nanned = [], []
    for g0 in np.linspace(w + 2, n - gap_n - w - 2, n_pos).astype(int):
        g1 = g0 + gap_n
        idx = np.r_[g0 - w : g0, g1 : g1 + w]

        xz = host.copy()
        xz[g0:g1] = 0.0
        zeroed.append(float(np.max(np.abs(step1_bandpass(xz, band)[idx] - ref[idx]))))

        xn = host.copy()
        xn[g0:g1] = np.nan
        nanned.append(float(np.max(np.abs(step1_bandpass(xn, band)[idx] - ref[idx]))))
    return float(np.median(zeroed)), float(np.median(nanned))


def test_the_replica_matches_matlabs_filter_coefficients() -> None:
    """Printed by MATLAB R2026a to 17 significant figures; agreement is ~5e-15."""
    matlab_b = np.array([
        0.047102856562278604, 0, -0.18841142624911442, 0, 0.28261713937367161,
        0, -0.18841142624911442, 0, 0.047102856562278604,
    ])
    matlab_a = np.array([
        1, -4.6798577293765193, 9.4184336978374983, -10.992860945373906,
        8.4720343406538525, -4.4732693236969503, 1.5186457154293045,
        -0.29287739435475968, 0.029751993282094161,
    ])
    nyq = FS / 2
    b, a = butter(MATLAB_ORDER, [100.0 / nyq, 5000.0 / nyq], btype="bandpass", output="ba")
    assert np.max(np.abs(b - matlab_b)) < 1e-14
    assert np.max(np.abs(a - matlab_a)) < 1e-13


@pytest.mark.parametrize("band", [MATLAB_BAND, ENG_BAND], ids=["100-5000", "300-3000"])
@pytest.mark.parametrize("gap_ms", [2.0, 10.0, 50.0])
def test_zeroing_is_far_worse_when_the_host_carries_out_of_band_content(
    band: tuple[float, float], gap_ms: float
) -> None:
    """Measured 2026-09-21. This is the claim task 01 makes and never measured.

    A real recording always carries ECG, respiration and drift at tens to hundreds
    of microvolts - A.5 quotes millivolt excursions - and *that* is what sets the
    step size when a gap is zeroed. Interpolation is continuous with the host at
    both boundaries, so its error does not scale with that content at all.
    """
    z_60, n_60 = ringing(band, lf_uv=60.0, gap_ms=gap_ms)
    z_500, n_500 = ringing(band, lf_uv=500.0, gap_ms=gap_ms)

    assert z_60 / n_60 > 1.8
    assert z_500 / n_500 > 8.0
    # The mechanism: zeroing scales with the out-of-band amplitude, NaN does not.
    assert z_500 > 5.0 * z_60
    assert n_500 < 2.0 * n_60


@pytest.mark.parametrize("band", [MATLAB_BAND, ENG_BAND], ids=["100-5000", "300-3000"])
def test_with_no_out_of_band_content_zeroing_is_the_milder_of_the_two(
    band: tuple[float, float],
) -> None:
    """The measured counter-example to the spec's unconditional claim.

    On a host with nothing outside the passband, a zeroed gap steps only by the
    in-band amplitude, while linear interpolation draws a chord between two noisy
    edge samples and gets it wrong by about as much. Zeroing measures ~2x *better*
    here. It does not change the decision - real hosts are not like this - but the
    spec says the NaN path "must be materially lower", and unconditionally it is not.
    """
    zeroed, nanned = ringing(band, lf_uv=0.0, gap_ms=10.0)
    assert zeroed < nanned
    assert 0.3 < zeroed / nanned < 0.9


def test_per_boundary_damage_does_not_fall_with_gap_length() -> None:
    """The spec's "damage is worst for short blanks" does not hold per boundary.

    Measured: the zeroed path's peak deviation is flat-to-rising from 2 ms to
    250 ms, lowest at the short end. Short blanks matter because there are more of
    them per second, not because each one does more damage.
    """
    short, _ = ringing(MATLAB_BAND, lf_uv=500.0, gap_ms=2.0)
    long, _ = ringing(MATLAB_BAND, lf_uv=500.0, gap_ms=50.0)
    assert short < long


def test_the_nan_path_leaves_the_gap_itself_marked() -> None:
    """The harm zeroing does that no ringing metric captures.

    With zeros the gap's own samples come back as small valid numbers, so every
    NaN-aware estimator downstream silently includes them - which is what pulls a
    MAD noise floor toward zero and lowers spike thresholds.
    """
    rng = np.random.default_rng(4)
    n = int(round(FS * 0.5))
    host = 20.0 * rng.standard_normal(n)
    g0, g1 = n // 3, n // 3 + 1_000

    xn = host.copy()
    xn[g0:g1] = np.nan
    xz = host.copy()
    xz[g0:g1] = 0.0

    out_nan = step1_bandpass(xn)
    out_zero = step1_bandpass(xz)

    assert np.all(np.isnan(out_nan[g0:g1]))
    assert not np.any(np.isnan(out_zero))
    # The zeroed gap depresses a robust noise estimate; the NaN gap cannot.
    sigma_nan = 1.4826 * np.nanmedian(np.abs(out_nan - np.nanmedian(out_nan)))
    sigma_zero = 1.4826 * np.median(np.abs(out_zero - np.median(out_zero)))
    assert sigma_zero < sigma_nan
