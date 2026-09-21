"""Derivations: the tripole, the diagnostic fit, and the stomach reference.

The applied weights are fixed at 0.5/0.5 by measurement, so the tests here divide
in two: what the *applied* tripole does to a common mode, and whether the
*diagnostic* fit recovers the contact gains that would cancel one.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.derive.derivations import (
    FIT_BAND_HZ,
    NAIVE_WEIGHTS,
    CuffWeights,
    build_derivations,
    build_stomach_reference,
    fit_tripole_weight,
    robust_sigma,
    sigma_reduction,
    tripole,
)
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs
from gems_blanking_v2.types import ChannelInfo, Recording

from conftest import make_common_mode, make_eng, make_multichannel
from conftest import robust_sigma as generator_sigma

F64 = npt.NDArray[np.float64]
FS = 24414.0625


def cuff_recording(
    v1: F64,
    v2: F64,
    v3: F64,
    *,
    animal: str = "J",
    stomach: list[F64] | None = None,
) -> Recording:
    """Wrap three contacts, and optionally stomach channels, as a Recording."""
    columns = [v1, v2, v3, *(stomach or [])]
    channels = [
        ChannelInfo(0, "LVN1", "nerve", "L", 1, 1, "independent"),
        ChannelInfo(1, "LVN2", "nerve", "L", 2, 1, "independent"),
        ChannelInfo(2, "LVN3", "nerve", "L", 3, 1, "independent"),
    ]
    channels += [
        ChannelInfo(3 + i, f"ANT{i + 1}", "stomach", None, None, None, "independent")
        for i in range(len(stomach or []))
    ]
    return Recording(
        fs=FS,
        data=np.column_stack(columns).astype(np.float64),
        channels=channels,
        animal=animal,
        session="t01",
        path=__import__("pathlib").Path("synthetic.h5"),
    )


def gained_cuff(
    gains: tuple[float, float, float],
    *,
    cm_sigma_uv: float = 50.0,
    neural_uv: float = 1.0,
    dur_s: float = 3.0,
    seed: int = 0,
) -> tuple[Recording, F64]:
    """Build a cuff whose three contacts share one common mode at unequal gains."""
    common = make_common_mode(FS, dur_s, *FIT_BAND_HZ, sigma_uv=cm_sigma_uv, seed=seed)
    contacts = []
    for i, gain in enumerate(gains):
        neural, _ = make_eng(FS, dur_s, spike_uv=0.0, noise_uv=neural_uv, seed=seed + 10 + i)
        contacts.append(gain * common + neural)
    return cuff_recording(*contacts), common


# ---------------------------------------------------------------------------
# the applied weights
# ---------------------------------------------------------------------------


def test_the_applied_weights_are_the_naive_half_half() -> None:
    """Measured: fitting is not an improvement, and can be worse."""
    assert NAIVE_WEIGHTS == (0.5, 0.5)


def test_a_plus_b_is_exactly_one() -> None:
    """To floating point, not approximately."""
    a, b = NAIVE_WEIGHTS
    assert (a + b) == 1.0

    rec, _ = gained_cuff((1.0, 1.0, 1.0))
    _, weights = build_derivations(rec)
    assert (weights["L"].a + weights["L"].b) == 1.0


def test_the_tripole_is_the_stated_expression() -> None:
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([0.5, 0.5, 0.5])
    v3 = np.array([3.0, 2.0, 1.0])
    assert np.allclose(tripole(v1, v2, v3), 0.5 * v1 + 0.5 * v3 - v2)


def test_weights_that_do_not_sum_to_one_are_refused() -> None:
    v = np.zeros(4)
    with pytest.raises(ValueError, match="must be exactly 1"):
        tripole(v, v, v, a=0.6, b=0.6)
    with pytest.raises(ValueError, match="must be exactly 1"):
        CuffWeights(a=0.6, b=0.6, fitted_a=float("nan"), fitted_b=float("nan"), source="naive")


def test_sigma_of_t_is_below_sigma_of_v1_when_a_common_mode_is_present() -> None:
    rec, _ = gained_cuff((1.05, 1.0, 0.95))
    signals, _ = build_derivations(rec)
    assert robust_sigma(signals["L_T"]) < robust_sigma(signals["L_V1"])


def test_the_ratio_is_about_one_when_no_common_mode_is_present() -> None:
    """With independent content only, the tripole sums three independent traces."""
    contacts = [make_eng(FS, 3.0, spike_uv=0.0, noise_uv=20.0, seed=s)[0] for s in (1, 2, 3)]
    rec = cuff_recording(*contacts)
    signals, _ = build_derivations(rec)

    ratio = robust_sigma(signals["L_T"]) / robust_sigma(signals["L_V1"])
    # 0.5^2 + 1 + 0.5^2 = 1.5 in variance, so sqrt(1.5) = 1.22 with no common mode
    # to remove: the tripole is not a noise reduction on independent channels.
    assert 1.0 < ratio < 1.4


def test_nan_in_any_contact_masks_the_tripole() -> None:
    """A masked contact masks T. Inventing a value there would fabricate a sample."""
    rec, _ = gained_cuff((1.0, 1.0, 1.0))
    rec.data[1000:1500, 1] = np.nan

    signals, _ = build_derivations(rec)
    assert np.all(np.isnan(signals["L_T"][1000:1500]))
    assert not np.any(np.isnan(signals["L_T"][:1000]))


def test_the_tripole_never_emits_a_zero_run() -> None:
    """Hard invariant 1 across a derivation."""
    rec, _ = gained_cuff((1.0, 1.0, 1.0))
    signals, _ = build_derivations(rec)
    for name, trace in signals.items():
        assert_no_zero_runs(trace, what=name, fs=FS)


# ---------------------------------------------------------------------------
# the raw contacts are kept
# ---------------------------------------------------------------------------


def test_the_raw_contacts_are_emitted_alongside_the_tripole() -> None:
    """Keep the contacts, not just the tripole.

    Invariant 6: detection reads the contacts, because T is defined by removing the
    common mode - which is the best artifact evidence there is.
    """
    rec, _ = gained_cuff((1.0, 1.0, 1.0))
    signals, _ = build_derivations(rec)
    assert set(signals) == {"L_V1", "L_V2", "L_V3", "L_T"}


def test_names_are_cuff_prefixed_for_both_cuffs() -> None:
    rec, _ = make_multichannel(FS, 2.0, seed=0)
    signals, weights = build_derivations(rec)
    assert {"L_V1", "L_V2", "L_V3", "L_T", "R_V1", "R_V2", "R_V3", "R_T"} <= set(signals)
    assert "stomach_ref" in signals
    assert set(weights) == {"L", "R"}


def test_a_cuff_missing_a_contact_is_refused() -> None:
    """A partial cuff cannot form a tripole, and skipping it would drop a nerve."""
    contacts = [make_eng(FS, 1.0, seed=s)[0] for s in (1, 2)]
    channels = [
        ChannelInfo(0, "LVN1", "nerve", "L", 1, 1, "independent"),
        ChannelInfo(1, "LVN2", "nerve", "L", 2, 1, "independent"),
    ]
    rec = Recording(
        fs=FS,
        data=np.column_stack(contacts).astype(np.float64),
        channels=channels,
        animal="J",
        session="t01",
        path=__import__("pathlib").Path("x.h5"),
    )
    with pytest.raises(ValueError, match="missing"):
        build_derivations(rec)


# ---------------------------------------------------------------------------
# the diagnostic fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gains",
    [(1.0, 1.0, 1.0), (1.2, 1.0, 0.8), (0.8, 1.0, 1.2), (1.1, 0.9, 1.0)],
)
def test_the_fit_recovers_the_gain_ratio(gains: tuple[float, float, float]) -> None:
    """The weight that cancels a common mode is ``a = (g2 - g3) / (g1 - g3)``.

    From ``a*g1 + (1-a)*g3 - g2 = 0``. This is what "recovers the gain ratio"
    means, and it is why the fit is worth computing even though it is not applied.
    """
    g1, g2, g3 = gains
    rec, _ = gained_cuff(gains, cm_sigma_uv=200.0, neural_uv=0.5)

    fitted = fit_tripole_weight(
        rec.data[:, 0], rec.data[:, 1], rec.data[:, 2], rec.fs
    )
    expected = 0.5 if g1 == g3 else (g2 - g3) / (g1 - g3)
    assert fitted == pytest.approx(expected, abs=0.05)


def test_the_fitted_weights_suppress_the_common_mode_by_over_20_db() -> None:
    """With the fit applied - which the pipeline does not do - rejection is large."""
    gains = (1.2, 1.0, 0.8)
    rec, common = gained_cuff(gains, cm_sigma_uv=200.0, neural_uv=0.5)
    v1, v2, v3 = rec.data[:, 0], rec.data[:, 1], rec.data[:, 2]

    fitted = fit_tripole_weight(v1, v2, v3, rec.fs)
    t_fitted = tripole(v1, v2, v3, fitted)

    # Referenced to the common mode as it appears on V2, the middle contact.
    before = robust_sigma(gains[1] * common)
    after = robust_sigma(t_fitted)
    suppression_db = 20.0 * math.log10(before / after)
    assert suppression_db > 20.0


def test_the_applied_naive_weights_suppress_less_than_the_fit_on_unequal_gains() -> None:
    """The measured trade-off, in one assertion.

    Naive weights leave a residual proportional to ``0.5*g1 + 0.5*g3 - g2``. The fit
    removes it in this band - and the measurement says doing so costs you in
    300-3000 Hz, which is the band spike detection reads.
    """
    rec, _ = gained_cuff((1.2, 1.0, 0.8), cm_sigma_uv=200.0, neural_uv=0.5)
    v1, v2, v3 = rec.data[:, 0], rec.data[:, 1], rec.data[:, 2]

    fitted = fit_tripole_weight(v1, v2, v3, rec.fs)
    assert robust_sigma(tripole(v1, v2, v3, fitted)) < robust_sigma(tripole(v1, v2, v3))


def test_the_fit_is_reported_but_never_applied() -> None:
    """The distinction the acceptance criterion depends on."""
    # (1.3, 1.0, 0.9): the cancelling weight is (1.0-0.9)/(1.3-0.9) = 0.25, so
    # the fit is visibly different from what is applied. Gains symmetric about
    # the middle contact would give exactly 0.5 and prove nothing.
    rec, _ = gained_cuff((1.3, 1.0, 0.9), cm_sigma_uv=200.0, neural_uv=0.5)
    signals, weights = build_derivations(rec)
    record = weights["L"]

    assert (record.a, record.b) == NAIVE_WEIGHTS
    assert record.fitted_a == pytest.approx(0.25, abs=0.05)
    assert record.fitted_a + record.fitted_b == pytest.approx(1.0)
    assert np.allclose(signals["L_T"], tripole(rec.data[:, 0], rec.data[:, 1], rec.data[:, 2]))


def test_a_degenerate_fit_is_flagged_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """``a -> 1.0`` is the measured failure on cuff L: the tripole becomes a bipolar.

    Flagged so a drift log does not read degeneracy as drift.
    """
    # g1 == g2 makes the cancelling weight a = (g2-g3)/(g1-g3) = 1.
    rec, _ = gained_cuff((1.0, 1.0, 0.6), cm_sigma_uv=200.0, neural_uv=0.5)

    with caplog.at_level(logging.WARNING, logger="gems_blanking_v2.derive.derivations"):
        _, weights = build_derivations(rec)

    record = weights["L"]
    assert record.fitted_a == pytest.approx(1.0, abs=0.05)
    assert record.degenerate
    assert (record.a, record.b) == NAIVE_WEIGHTS, "the applied weights are unaffected"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("collapse toward a bipolar" in r.getMessage() for r in warnings)


def test_a_fit_on_too_little_finite_signal_is_nan() -> None:
    """Reported as unknown rather than computed from a fragment."""
    a, b, c = (make_eng(FS, 0.5, seed=s)[0] for s in (1, 2, 3))
    assert math.isnan(fit_tripole_weight(a, b, c, FS))

    nan_trace = np.full(int(FS * 3), np.nan)
    assert math.isnan(fit_tripole_weight(nan_trace, nan_trace.copy(), nan_trace.copy(), FS))


def test_a_fit_with_no_contrast_between_v1_and_v3_is_nan() -> None:
    """``d = V1 - V3`` has no variance, so ``a`` is not identifiable."""
    trace, _ = make_eng(FS, 3.0, seed=1)
    other, _ = make_eng(FS, 3.0, seed=2)
    assert math.isnan(fit_tripole_weight(trace, other, trace, FS))


def test_a_degenerate_fit_does_not_report_degenerate_when_it_is_nan() -> None:
    record = CuffWeights(0.5, 0.5, float("nan"), float("nan"), "naive")
    assert not record.degenerate


# ---------------------------------------------------------------------------
# the old cohort
# ---------------------------------------------------------------------------


def hw_tripole_recording() -> tuple[Recording, F64, list[F64]]:
    """Build the five-channel old cohort: a pre-formed tripole per nerve."""
    left, _ = make_eng(FS, 2.0, seed=1)
    right, _ = make_eng(FS, 2.0, seed=2)
    stomach = [make_eng(FS, 2.0, spike_uv=0.0, noise_uv=30.0, seed=s)[0] for s in (3, 4, 5)]
    channels = [
        ChannelInfo(0, "LVN", "nerve", "L", None, None, "hw_tripole"),
        ChannelInfo(1, "RVN", "nerve", "R", None, None, "hw_tripole"),
        *[
            ChannelInfo(2 + i, f"ANT{i + 1}", "stomach", None, None, None, "hw_tripole")
            for i in range(3)
        ],
    ]
    rec = Recording(
        fs=FS,
        data=np.column_stack([left, right, *stomach]).astype(np.float64),
        channels=channels,
        animal="F",
        session="t01",
        path=__import__("pathlib").Path("old.h5"),
    )
    return rec, left, stomach


def test_a_hardware_tripole_passes_through_unchanged_with_nan_weights() -> None:
    """The contacts are unrecoverable, so there is nothing to derive."""
    rec, left, _ = hw_tripole_recording()
    signals, weights = build_derivations(rec)

    assert np.array_equal(signals["L_T"], left)
    assert "L_V1" not in signals
    assert weights["L"].source == "hw_tripole"
    assert math.isnan(weights["L"].a)
    assert math.isnan(weights["L"].b)
    assert math.isnan(weights["L"].fitted_a)


def test_both_cohorts_produce_a_stomach_ref() -> None:
    old, _, _ = hw_tripole_recording()
    new, _ = make_multichannel(FS, 2.0, seed=0)

    for rec in (old, new):
        signals, _ = build_derivations(rec)
        assert "stomach_ref" in signals
        assert signals["stomach_ref"].shape == (rec.n_samples,)


def test_the_old_cohort_stomach_is_passed_through_as_hardware_referenced() -> None:
    rec, _, stomach = hw_tripole_recording()
    reference, how = build_stomach_reference(rec)
    assert how == "hardware"
    assert reference is not None
    assert np.array_equal(reference, stomach[0])


def test_the_new_cohort_stomach_is_common_averaged_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The design does not state what the new cohort is referenced against.

    A common average is used and recorded. It is not neutral: the gastric slow wave
    is largely common across the array, so this attenuates part of what the
    ``slow_wave`` and ``mmc`` consumers read - hence the warning.
    """
    rec, _ = make_multichannel(FS, 2.0, seed=0)
    with caplog.at_level(logging.WARNING, logger="gems_blanking_v2.derive.derivations"):
        reference, how = build_stomach_reference(rec)

    assert how == "common_average"
    assert reference is not None
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("common average" in m and "confirm the intended reference" in m for m in messages)


def test_a_single_stomach_channel_is_passed_through() -> None:
    """There is nothing to average against."""
    contacts = [make_eng(FS, 1.0, seed=s)[0] for s in (1, 2, 3)]
    stomach, _ = make_eng(FS, 1.0, spike_uv=0.0, noise_uv=30.0, seed=9)
    rec = cuff_recording(*contacts, stomach=[stomach])

    reference, how = build_stomach_reference(rec)
    assert how == "single_channel"
    assert reference is not None
    assert np.array_equal(reference, stomach)


def test_a_recording_with_no_stomach_has_no_stomach_ref() -> None:
    rec, _ = gained_cuff((1.0, 1.0, 1.0))
    signals, _ = build_derivations(rec)
    assert "stomach_ref" not in signals
    assert build_stomach_reference(rec) == (None, None)


# ---------------------------------------------------------------------------
# the measured sigma reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gains", [(1.05, 1.0, 0.95), (1.2, 1.0, 0.8), (1.4, 1.0, 0.6)])
def test_the_naive_weights_cancel_a_symmetric_gain_mismatch_exactly(
    gains: tuple[float, float, float],
) -> None:
    """0.5/0.5 is *optimal* whenever the outer gains are symmetric about the middle.

    ``0.5*g1 + 0.5*g3 - g2 = 0`` holds for every one of these, including a +/-40%
    mismatch. This is the likely reason the measurement found fitting no better
    than naive on real cuffs: the fit can only beat 0.5/0.5 on the *asymmetric*
    part of the mismatch, and there may not be much of it.
    """
    g1, g2, g3 = gains
    assert (0.5 * g1 + 0.5 * g3 - g2) == pytest.approx(0.0, abs=1e-12)

    rec, common = gained_cuff(gains, cm_sigma_uv=200.0, neural_uv=1.0)
    signals, _ = build_derivations(rec)

    # What is left of a 200 uV common mode is the independent content only.
    assert robust_sigma(signals["L_T"]) < 0.05 * robust_sigma(g2 * common)


def test_the_sigma_reduction_is_a_property_of_the_recording_not_the_tripole() -> None:
    """Why ~6x and 2.5-2.8x are both true, and neither is a constant.

    The reduction is set by two things the derivation does not control: how
    *asymmetric* the contact gains are, which fixes the residual coefficient, and
    how large the common mode is next to the independent content. Holding the gains
    fixed and sweeping only the common-mode amplitude moves the ratio from under 2
    to nearly 10, so quoting a single figure for "the tripole" is a category error.

    The measured 2.5-2.8x is reproduced at the low end of that sweep; the ~6x in
    this task's Purpose sits at the high end. Neither should be used as a constant
    downstream.
    """
    gains = (1.2, 1.0, 1.0)  # asymmetric: residual coefficient 0.1
    assert (0.5 * gains[0] + 0.5 * gains[2] - gains[1]) == pytest.approx(0.1)

    ratios: dict[float, float] = {}
    for cm_sigma in (10.0, 20.0, 50.0, 100.0):
        rec, _ = gained_cuff(gains, cm_sigma_uv=cm_sigma, neural_uv=6.0, seed=3)
        signals, _ = build_derivations(rec)
        ratios[cm_sigma] = robust_sigma(signals["L_V1"]) / robust_sigma(signals["L_T"])

    assert sorted(ratios.values()) == list(ratios.values()), "monotone in the common mode"
    assert ratios[10.0] < 2.5, "a modest common mode gives almost no reduction"
    assert 2.5 <= ratios[20.0] <= 4.0, "the measured 2.5-2.8x band is in reach here"
    assert ratios[100.0] > 6.0, "a dominant common mode exceeds the ~6x figure too"


# ---------------------------------------------------------------------------
# the QC number
# ---------------------------------------------------------------------------


def test_the_sigma_ratio_is_reported_per_cuff() -> None:
    """Measured per recording and reported, never assumed."""
    rec, _ = make_multichannel(FS, 2.0, seed=0)
    signals, _ = build_derivations(rec)

    for cuff in ("L", "R"):
        ratio = sigma_reduction(signals, cuff)
        assert math.isfinite(ratio)
        assert ratio > 0.0
        expected = robust_sigma(signals[f"{cuff}_V1"]) / robust_sigma(signals[f"{cuff}_T"])
        assert ratio == pytest.approx(expected)


def test_the_sigma_ratio_is_logged_as_qc(caplog: pytest.LogCaptureFixture) -> None:
    """So it lands in the run log even if nobody calls the function."""
    rec, _ = gained_cuff((1.2, 1.0, 1.0), cm_sigma_uv=40.0, neural_uv=6.0)
    with caplog.at_level(logging.INFO, logger="gems_blanking_v2.derive.derivations"):
        build_derivations(rec)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("sigma(V1)/sigma(T)" in m and "QC only" in m for m in messages)
    assert any("nothing downstream may depend on it" in m for m in messages)


def test_the_sigma_ratio_is_nan_for_a_hardware_tripole() -> None:
    """There are no contacts to compare against."""
    rec, _, _ = hw_tripole_recording()
    signals, _ = build_derivations(rec)
    assert math.isnan(sigma_reduction(signals, "L"))
    assert math.isnan(sigma_reduction(signals, "nosuchcuff"))


def test_robust_sigma_matches_the_generators_estimator() -> None:
    """The package's own copy must agree with the one the generators use."""
    trace, _ = make_eng(FS, 1.0, seed=7)
    assert robust_sigma(trace) == pytest.approx(generator_sigma(trace))

    trace[100:200] = np.nan
    assert robust_sigma(trace) == pytest.approx(generator_sigma(trace))
    assert math.isnan(robust_sigma(np.full(10, np.nan)))
