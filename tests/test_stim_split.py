"""Tests for :mod:`gems_blanking_v2.io.stim_split`.

Every numeric claim in the module's docstrings has a test here that would fail if it
were wrong.

Two of the first draft's required tests did not discriminate and have been replaced,
with the originals kept alongside as documentation of what the old MATLAB actually
does: :func:`test_low_duty_cycle_alone_does_not_defeat_either_method` and
:func:`test_a_three_second_dropout_does_not_split_the_epoch`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from gems_blanking_v2.io.stim_split import (
    AUDIT_COLUMNS,
    EDGE_HOLD_S,
    EDGE_REFINE_PAD_S,
    ENVELOPE_RMS_S,
    PROTOCOL_FILENAME,
    Epoch,
    ProtocolSpec,
    audit_stim_splits,
    audit_summary,
    blanking_fraction,
    default_protocol,
    find_stim_window,
    load_protocol,
    protocol_path,
    split_stim_recovery,
    vib_envelope,
    write_protocol,
)
from gems_blanking_v2.io.stim_split import _measure_extent as rp_measure_extent
from gems_blanking_v2.io.stim_split import _status as rp_status
from gems_blanking_v2.types import ChannelInfo, Recording
from scipy.signal import butter, sosfiltfilt

from conftest import make_eng, make_vib

F64 = npt.NDArray[np.float64]

FS_VIB = 1000.0
"""Monitor sample rate for these tests, Hz. Deliberately not the signal's."""

FS_SIG = 2000.0
"""Signal sample rate, Hz. Low enough to keep a 25-minute synthetic cheap, and
different from :data:`FS_VIB` so the two time bases cannot be conflated."""

FILE_S = 1500.0
"""25 minutes: the worst case in the spec's duty-cycle table, where the 120 s stim
is 8% of the record and both the 20th and 80th percentiles fall inside OFF."""

STIM_START_S = 300.0
STIM_S = 120.0


@pytest.fixture
def protocol() -> ProtocolSpec:
    """Return the lab protocol: 120 s stim, 1200 s recovery, 12 s tolerance."""
    return default_protocol()


def _recording(
    vib: F64,
    *,
    fs: float = FS_SIG,
    dur_s: float = FILE_S,
    with_vib_channel: bool = True,
    vib_name: str = "VIB",
    seed: int = 0,
) -> Recording:
    """Build a two-nerve recording, optionally carrying the monitor as an aux channel.

    ``vib`` must already be at ``fs`` when it is placed in the matrix, because
    :class:`Recording` carries one rate for the whole array.
    """
    n = int(round(dur_s * fs))
    columns = [make_eng(fs, dur_s, seed=seed)[0], make_eng(fs, dur_s, seed=seed + 1)[0]]
    channels = [
        ChannelInfo(0, "LVN1", "nerve", "L", 1, 1, "independent"),
        ChannelInfo(1, "LVN2", "nerve", "L", 2, 1, "independent"),
    ]
    if with_vib_channel:
        columns.append(np.asarray(vib[:n], dtype=np.float64))
        channels.append(ChannelInfo(2, vib_name, "aux", None, None, None, "independent"))

    return Recording(
        fs=fs,
        data=np.column_stack([c[:n] for c in columns]),
        channels=channels,
        animal="J",
        session="t01_sr",
        path=Path("gems_j_t01_es2_sr_120000.mat"),
    )


def _old_percentile_rule(
    env: F64, fs: float, min_dur_s: float = 10.0
) -> tuple[float, float] | None:
    """Run the superseded ``splitStimRecovery.m`` detector, for comparison only.

    ``(p20 + p80)/2`` threshold, short ON runs dropped and short OFF gaps bridged at
    ``minDurSec``, then the largest remaining ON segment. Implemented here and never
    in the module, so its behaviour can be measured rather than asserted.
    """
    threshold = (float(np.percentile(env, 20)) + float(np.percentile(env, 80))) / 2.0
    on = env > threshold
    minimum = max(int(round(min_dur_s * fs)), 1)

    def runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
        padded = np.concatenate([[False], mask, [False]])
        edges = np.diff(padded.astype(np.int8))
        return list(
            zip(
                np.flatnonzero(edges == 1).tolist(),
                np.flatnonzero(edges == -1).tolist(),
                strict=True,
            )
        )

    for start, stop in runs(on):
        if stop - start < minimum:
            on[start:stop] = False
    for start, stop in runs(~on):
        if stop - start < minimum:
            on[start:stop] = True

    segments = runs(on)
    if not segments:
        return None
    start, stop = max(segments, key=lambda run: run[1] - run[0])
    return start / fs, stop / fs


def _log_envelope_reference(x: F64, fs: float, band: tuple[float, float]) -> tuple[float, float]:
    """Return ``(median, 1.4826 * MAD)`` of the log band envelope - invariant 5.

    A preview of what task 06 will own, written with the binding formula rather than
    a lookalike so the two cannot silently diverge. Only used to state "why the
    ordering matters" quantitatively.
    """
    lo, hi = band
    nyquist = fs / 2.0
    sos = butter(
        4, [lo / nyquist, min(hi, 0.9 * nyquist) / nyquist], btype="bandpass", output="sos"
    )
    envelope = np.abs(np.asarray(sosfiltfilt(sos, x), dtype=np.float64))
    frames = int(envelope.size // int(round(0.010 * fs)))
    step = int(round(0.010 * fs))
    binned = envelope[: frames * step].reshape(frames, step).mean(axis=1)
    log_envelope = np.log(binned[binned > 0])
    median = float(np.median(log_envelope))
    return median, 1.4826 * float(np.median(np.abs(log_envelope - median)))


# ---------------------------------------------------------------------------
# the protocol config
# ---------------------------------------------------------------------------


def test_the_protocol_is_never_defaulted(tmp_path: Path) -> None:
    """An absent protocol raises. The stim duration is a prior, not a convenience.

    The same reasoning as ``conditions.yaml``: a value that silently appeared would
    split differently on two machines, and here it would move every boundary in the
    corpus rather than mislabel one file.
    """
    with pytest.raises(FileNotFoundError, match="not defaulted"):
        load_protocol(protocol_path(tmp_path))


def test_a_written_protocol_round_trips(tmp_path: Path, protocol: ProtocolSpec) -> None:
    """Write atomically as UTF-8 with LF endings; always loadable afterwards."""
    path = write_protocol(protocol, protocol_path(tmp_path))

    assert path.name == PROTOCOL_FILENAME
    assert load_protocol(path) == protocol
    assert b"\r\n" not in path.read_bytes()
    assert path.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("missing", ["stim_duration_s", "recovery_duration_s", "stim_tolerance_s"])
def test_a_missing_protocol_field_raises_naming_it(tmp_path: Path, missing: str) -> None:
    """A required field that is absent raises **naming the field**, never defaults."""
    document = default_protocol().to_json()
    del document[missing]
    path = tmp_path / PROTOCOL_FILENAME
    path.write_text(
        "\n".join(f"{k}: {v}" for k, v in document.items()), encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match=missing):
        load_protocol(path)


def test_a_non_physical_duration_raises() -> None:
    """Zero or negative durations are rejected at construction."""
    with pytest.raises(ValueError, match="stim_duration_s"):
        ProtocolSpec(stim_duration_s=0.0, recovery_duration_s=1200.0, stim_tolerance_s=12.0)


# ---------------------------------------------------------------------------
# the matched-width search
# ---------------------------------------------------------------------------


def test_the_matched_search_recovers_the_onset_at_an_eight_percent_duty_cycle() -> None:
    """Recover a 120 s stim in a 25 minute file, to within half an envelope window.

    Measured onset error **-50 ms** and offset error **+53 ms** across onsets at 60 s,
    300 s and 900 s, so a detected 120.10 s against a true 120 s. 50 ms is half of
    :data:`ENVELOPE_RMS_S`: the 100 ms moving RMS turns each edge into a ramp and the
    crossing is found at its near side, symmetrically at both ends. That is the floor
    for this envelope, not a tuning result, which is why the assertion is on the
    measured value rather than on a bound it happens to sit under.
    """
    protocol = default_protocol()
    for onset_s in (60.0, 300.0, 900.0):
        vib, true_start, true_stop = make_vib(FS_VIB, FILE_S, onset_s, STIM_S, seed=0)
        window = find_stim_window(vib_envelope(vib, FS_VIB), FS_VIB, protocol)

        assert window.onset / FS_VIB - float(true_start) == pytest.approx(-0.050, abs=0.005)
        assert window.offset / FS_VIB - float(true_stop) == pytest.approx(0.053, abs=0.005)
        assert window.epoch_count == 1


def test_low_duty_cycle_alone_does_not_defeat_either_method() -> None:
    """**Both** methods recover the onset at 8% duty. Low duty cycle is not the failure.

    The prediction was that ``(p20 + p80)/2`` would break at an 8% duty cycle because
    both percentiles fall inside the OFF distribution. The first half is true - the
    threshold does land in the OFF noise, flagging **46% of the record** where the
    real duty cycle is 8% - but the conclusion does not follow. ``keep only the
    LARGEST ON segment`` rescues it: the spurious segments are all short and the one
    real 120 s segment wins by a wide margin.

    Measured: matched-width **-50 ms**, p20/p80 **-74 ms**. So the historical splits
    are probably mostly fine, and this test asserts that rather than the reverse. The
    discriminating case is
    :func:`test_the_old_percentile_rule_locks_onto_a_longer_quieter_burst`.
    """
    vib, true_start, _stop = make_vib(FS_VIB, FILE_S, STIM_START_S, STIM_S, seed=0)
    env = vib_envelope(vib, FS_VIB)

    threshold = (float(np.percentile(env, 20)) + float(np.percentile(env, 80))) / 2.0
    flagged = float((env > threshold).mean())
    assert flagged > 0.40, "the old threshold stopped landing inside the OFF noise"
    assert flagged > 4.0 * (STIM_S / FILE_S), "it should flag far more than the duty cycle"

    old = _old_percentile_rule(env, FS_VIB)
    assert old is not None
    assert old[0] - float(true_start) == pytest.approx(-0.074, abs=0.02)

    matched = find_stim_window(env, FS_VIB, default_protocol())
    assert matched.onset / FS_VIB - float(true_start) == pytest.approx(-0.050, abs=0.005)


def test_the_old_percentile_rule_locks_onto_a_longer_quieter_burst() -> None:
    """**The discriminating case.** Where the old rule breaks and the matched one does not.

    A 300 s burst - sustained handling, a motor left running, a cable rubbing on the
    monitor - at a quarter of the stim carrier wins on *length* while losing on
    amplitude, so "keep the largest ON segment" locks onto it: measured **+400 s** of
    onset error against the matched search's unchanged -50 ms. The matched filter is
    immune because the width is fixed and the score is a contrast, so a long
    low-amplitude segment scores worse than the real one.

    The split reports two epochs and sends the file to review, which is right - a
    second stim-like burst is a protocol mismatch, not something to resolve silently.
    """
    vib, true_start, _stop = make_vib(
        FS_VIB, FILE_S, STIM_START_S, STIM_S, contaminant=(700.0, 300.0, 0.25), seed=0
    )
    env = vib_envelope(vib, FS_VIB)

    old = _old_percentile_rule(env, FS_VIB)
    assert old is not None
    assert abs(old[0] - float(true_start)) > 300.0, "the old rule stopped failing here"

    window = find_stim_window(env, FS_VIB, default_protocol())
    assert abs(window.onset / FS_VIB - float(true_start)) <= ENVELOPE_RMS_S
    assert window.epoch_count == 2


def test_a_three_second_dropout_does_not_split_the_epoch() -> None:
    """The spec's dropout test - which **today's MATLAB also passes**, and that matters.

    ``removeShortSegments`` in ``splitStimRecovery.m`` bridges OFF gaps shorter than
    ``minDurSec = 10`` as well as dropping short ON runs, so a 3 s dropout is already
    handled there. This test therefore does not discriminate against the old code;
    :func:`test_a_fifteen_second_dropout_does_not_split_the_epoch` is the one that
    does.
    """
    vib, true_start, true_stop = make_vib(
        FS_VIB, FILE_S, STIM_START_S, STIM_S, dropout=(360.0, 3.0), seed=0
    )
    env = vib_envelope(vib, FS_VIB)
    window = find_stim_window(env, FS_VIB, default_protocol())

    assert window.epoch_count == 1
    assert abs(window.onset / FS_VIB - float(true_start)) <= ENVELOPE_RMS_S
    assert abs(window.offset / FS_VIB - float(true_stop)) <= ENVELOPE_RMS_S

    old = _old_percentile_rule(env, FS_VIB)
    assert old is not None
    assert abs(old[0] - float(true_start)) < 0.10, "today's MATLAB bridges a 3 s gap too"


def test_a_fifteen_second_dropout_does_not_split_the_epoch() -> None:
    """A gap longer than the deleted ``minDurSec``, which is where the two differ.

    The matched window spans a dropout of **any** length because the width is fixed;
    the old rule bridges only gaps under 10 s and then keeps whichever side of a
    longer one happens to be bigger, losing the other half of the stim.
    """
    vib, true_start, true_stop = make_vib(
        FS_VIB, FILE_S, STIM_START_S, STIM_S, dropout=(345.0, 15.0), seed=0
    )
    env = vib_envelope(vib, FS_VIB)
    window = find_stim_window(env, FS_VIB, default_protocol())

    assert window.epoch_count == 1
    assert abs(window.onset / FS_VIB - float(true_start)) <= ENVELOPE_RMS_S
    assert abs(window.offset / FS_VIB - float(true_stop)) <= ENVELOPE_RMS_S

    old = _old_percentile_rule(env, FS_VIB)
    assert old is not None
    assert (old[1] - old[0]) < 0.75 * STIM_S, (
        "the old rule kept the whole epoch across a 15 s gap - re-measure"
    )


def test_the_protocol_decides_whether_a_file_passes(protocol: ProtocolSpec) -> None:
    """Changing ``stim_duration_s`` to 60 turns the same file from pass into review.

    The prior is what "detected against expected" is measured against, so a protocol
    edit visibly changes the verdict rather than silently agreeing with a hardcoded
    120. That is the point of keeping it in config.

    The **measured extent deliberately does not move**: after the refinement fix the
    matched filter only locates the epoch and the threshold measures it, so a wrong
    prior misjudges the file without also mis-cutting it. That is the failure mode
    worth having - a flagged file rather than a quietly truncated recovery epoch.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib)

    _stim, _recovery, correct = split_stim_recovery(rec, protocol=protocol)
    _stim2, _recovery2, wrong = split_stim_recovery(
        rec,
        protocol=ProtocolSpec(
            stim_duration_s=60.0, recovery_duration_s=1200.0, stim_tolerance_s=12.0
        ),
    )

    assert correct.status == "pass"
    assert wrong.status == "review"
    assert wrong.protocol.stim_duration_s == 60.0
    assert wrong.detected_duration_s == pytest.approx(correct.detected_duration_s, abs=0.01)
    assert wrong.onset_s == pytest.approx(correct.onset_s, abs=0.01)


def test_a_record_shorter_than_the_stim_window_raises() -> None:
    """A protocol mismatch no boundary can resolve, so it is reported rather than fitted."""
    vib, _start, _stop = make_vib(FS_VIB, 90.0, 10.0, 60.0, seed=0)

    with pytest.raises(ValueError, match="cannot hold one"):
        find_stim_window(vib_envelope(vib, FS_VIB), FS_VIB, default_protocol())


def test_the_split_is_invariant_to_the_monitors_scale() -> None:
    """The monitor carries no declared units, and this is why that is safe.

    Every rule is a contrast on the monitor's own envelope, so scaling it by 1000
    moves no boundary by a single sample. Invariant 14 governs what gets interpreted
    as microvolts; the monitor never is.
    """
    protocol = default_protocol()
    vib, _start, _stop = make_vib(FS_VIB, FILE_S, STIM_START_S, STIM_S, seed=0)

    unit = find_stim_window(vib_envelope(vib, FS_VIB), FS_VIB, protocol)
    scaled = find_stim_window(vib_envelope(vib * 1000.0, FS_VIB), FS_VIB, protocol)

    assert (unit.onset, unit.offset) == (scaled.onset, scaled.offset)


# ---------------------------------------------------------------------------
# the status table
# ---------------------------------------------------------------------------


def test_a_clean_capture_passes(protocol: ProtocolSpec) -> None:
    """120 s detected against 120 s expected, no clipping, one epoch."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    stim, recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.status == "pass"
    assert not report.clipped_start
    assert not report.clipped_end
    assert report.detected_duration_s == pytest.approx(STIM_S, abs=0.2)
    assert stim is not None
    assert report.epoch_count == 1
    assert report.duty_cycle == pytest.approx(STIM_S / FILE_S, rel=0.01)
    assert recovery.duration_s == pytest.approx(FILE_S - STIM_START_S - STIM_S, abs=0.2)


def test_a_recording_that_starts_mid_stim_is_clipped_but_usable(protocol: ProtocolSpec) -> None:
    """Starting 8 s into the stim: benign, offset still valid, recovery intact.

    The censored-duration case: 112 s is **inside** the 12 s tolerance, and reporting
    that as ``pass`` would be a false reassurance, because the true stim was longer
    than what is in the file. The tolerance check is skipped, not passed, and the
    status says so.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, -8.0, STIM_S, seed=0)
    stim, recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.clipped_start
    assert not report.clipped_end
    assert report.status == "clipped_start"
    assert report.detected_duration_s == pytest.approx(112.0, abs=0.5)
    assert abs(report.detected_duration_s - protocol.stim_duration_s) < protocol.stim_tolerance_s
    assert "skipped rather than passed" in report.reason
    assert report.offset_s == pytest.approx(112.0, abs=0.5)
    assert recovery.t0_offset_s == pytest.approx(report.offset_s, abs=0.01)
    assert stim is not None
    assert stim.t0_offset_s == pytest.approx(0.0, abs=0.01)


def test_a_badly_clipped_start_reports_the_same_status(protocol: ProtocolSpec) -> None:
    """Starting 40 s in: far outside the tolerance, and the status is unchanged.

    Both clipped cases report ``clipped_start`` whether the censored duration happens
    to land inside the tolerance or not, because in neither case was the tolerance
    check meaningful. The file is still usable - the offset is valid and the recovery
    epoch is intact.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, -40.0, STIM_S, seed=0)
    _stim, recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.clipped_start
    assert report.detected_duration_s == pytest.approx(80.0, abs=0.5)
    assert abs(report.detected_duration_s - protocol.stim_duration_s) > protocol.stim_tolerance_s
    assert report.status == "clipped_start"
    assert "started mid-stim" in report.reason
    assert recovery.duration_s > 0.9 * protocol.recovery_duration_s


def test_a_recording_that_stops_during_stim_raises(protocol: ProtocolSpec) -> None:
    """No recovery epoch in the file, so it says so rather than emitting a stub.

    The stim runs past the last sample, so the monitor is ON at the end. Emitting a
    near-empty recovery epoch here would put a few seconds of post-stim data into an
    analysis that believes it has 20 minutes.

    Matched on the **clipped-at-end** wording specifically. Slicing would refuse this
    file anyway, for the different reason that the stim epoch reaches the last sample,
    so a looser match would pass with the clipping check deleted - which a mutation
    run found. The two are defence in depth and the test has to name which one fired.
    """
    short_s = 400.0
    vib, _s, _e = make_vib(FS_SIG, short_s, 300.0, STIM_S, seed=0)

    with pytest.raises(ValueError, match="stopped during stimulation"):
        split_stim_recovery(_recording(vib, dur_s=short_s), protocol=protocol)


def test_the_status_table_fails_only_on_a_clipped_end(protocol: ProtocolSpec) -> None:
    """The status rules directly, so no other guard can stand in for them.

    Clipped at the end is the only clipping that is a status; clipped at the start is
    a flag on an otherwise passing file, and both can coexist with a duration inside
    the tolerance.
    """
    clean, _reason = rp_status(
        120.0, protocol, clipped_start=False, clipped_end=False, epoch_count=1
    )
    at_start, _r2 = rp_status(112.0, protocol, clipped_start=True, clipped_end=False, epoch_count=1)
    short, _r3 = rp_status(80.0, protocol, clipped_start=True, clipped_end=False, epoch_count=1)
    at_end, reason = rp_status(95.0, protocol, clipped_start=False, clipped_end=True, epoch_count=1)
    two, _r5 = rp_status(120.0, protocol, clipped_start=False, clipped_end=False, epoch_count=2)
    both, _r6 = rp_status(95.0, protocol, clipped_start=True, clipped_end=True, epoch_count=1)

    assert clean == "pass"
    assert at_start == "clipped_start", "a censored duration inside tolerance is not a pass"
    assert short == "clipped_start", "and neither is one outside it"
    assert at_end == "fail"
    assert "stopped during stimulation" in reason
    assert two == "review"
    assert both == "fail", "clipped_end wins: there is no recovery epoch to salvage"


def test_a_detected_duration_of_95_seconds_goes_to_review(protocol: ProtocolSpec) -> None:
    """Outside +/-12 s and not clipped at either edge: flagged, never silent.

    This case is why the edge refinement had to change. The spec's version searches
    +/-2 s around ``onset + 120``, and a 95 s stim's true offset is 25 s inside that -
    more than twelve pads away - so it would report 120.0 s and pass. The duration
    check would have been structurally unable to catch the deviation it exists for.
    See :data:`~gems_blanking_v2.io.stim_split.EDGE_REFINE_PAD_S`.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, 95.0, seed=0)
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.detected_duration_s == pytest.approx(95.0, abs=0.5)
    assert report.status == "review"
    assert not report.clipped_start
    assert "do not proceed silently" in report.reason


@pytest.mark.parametrize("stim_s", [95.0, 113.0, 140.0])
def test_the_detected_duration_is_measured_not_assumed(
    stim_s: float, protocol: ProtocolSpec
) -> None:
    """Any deviation from the prior is reported, in either direction.

    Measured at 95 s, 113 s and 140 s against a 120 s prior: every one comes back
    within 110 ms of the truth. The matched filter *locates* the epoch and the
    OFF-anchored threshold *measures* it, which is the division of labour that makes
    "duration becomes a check" mean anything.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, stim_s, seed=0)
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.detected_duration_s == pytest.approx(stim_s, abs=0.15)


def test_a_detected_duration_of_113_seconds_passes(protocol: ProtocolSpec) -> None:
    """Inside +/-12 s. The tolerance is not tighter than normal start/stop jitter."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, 113.0, seed=0)
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.detected_duration_s == pytest.approx(113.0, abs=0.5)
    assert report.status == "pass"


def test_a_stim_recovery_file_with_no_monitor_channel_raises(protocol: ProtocolSpec) -> None:
    """Blocked, not inferred. Reading stim timing off the nerve channels is circular."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib, with_vib_channel=False)

    with pytest.raises(ValueError, match="no stimulation-monitor channel"):
        split_stim_recovery(rec, protocol=protocol)


def test_two_candidate_monitor_channels_raise(protocol: ProtocolSpec) -> None:
    """Ambiguity is reported, never resolved by picking the first match."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib)
    rec = Recording(
        fs=rec.fs,
        data=np.column_stack([rec.data, rec.data[:, 2]]),
        channels=[
            *rec.channels,
            ChannelInfo(3, "StimTrig", "aux", None, None, None, "independent"),
        ],
        animal=rec.animal,
        session=rec.session,
        path=rec.path,
    )

    with pytest.raises(ValueError, match="candidate stimulation-monitor channels"):
        split_stim_recovery(rec, protocol=protocol)


@pytest.mark.parametrize("name", ["VIB", "vib", "Vib_1", "DIG", "stim_mon"])
def test_the_monitor_channel_is_matched_case_insensitively(
    name: str, protocol: ProtocolSpec
) -> None:
    """Cross-platform rule 7: this lab's own data mixes case for the same entity."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    _stim, _recovery, report = split_stim_recovery(
        _recording(vib, vib_name=name), protocol=protocol
    )

    assert report.vib_channel == name
    assert report.status == "pass"


# ---------------------------------------------------------------------------
# epochs, time bases and accounting
# ---------------------------------------------------------------------------


def test_the_monitor_may_have_its_own_sample_rate(protocol: ProtocolSpec) -> None:
    """``Recording`` carries one ``fs``, so a monitor on its own clock comes in separately.

    The boundary is found on the monitor's time base, converted to seconds **once**,
    and only then mapped onto the signal grid - going straight from monitor samples to
    signal samples would bake the rate ratio into an integer division (invariant 15).
    Two monitors at 1000 Hz and 4000 Hz must give the same boundary in seconds.
    """
    signal_only = _recording(np.zeros(1), with_vib_channel=False)

    boundaries = []
    for rate in (FS_VIB, 4.0 * FS_VIB):
        vib, _s, _e = make_vib(rate, FILE_S, STIM_START_S, STIM_S, seed=0)
        _stim, recovery, report = split_stim_recovery(
            signal_only, protocol=protocol, vib=vib, fs_vib=rate
        )
        boundaries.append((report.onset_s, report.offset_s, recovery.t0_offset_s))

    assert boundaries[0] == pytest.approx(boundaries[1], abs=0.01)
    assert boundaries[0][0] == pytest.approx(STIM_START_S, abs=ENVELOPE_RMS_S)


def test_an_explicit_monitor_without_its_rate_raises(protocol: ProtocolSpec) -> None:
    """An array with no rate is not a time series. Invariant 15's boundary."""
    vib, _s, _e = make_vib(FS_VIB, FILE_S, STIM_START_S, STIM_S, seed=0)

    with pytest.raises(ValueError, match="needs its own fs_vib"):
        split_stim_recovery(
            _recording(vib, with_vib_channel=False), protocol=protocol, vib=vib
        )


def test_both_epochs_keep_their_place_in_the_original_recording(protocol: ProtocolSpec) -> None:
    """Absolute time is never lost, and the two epochs tile the split exactly."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    stim, recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert stim is not None
    assert stim.t0_offset_s == pytest.approx(report.onset_s, abs=0.001)
    assert stim.t1_offset_s == pytest.approx(recovery.t0_offset_s, abs=0.001)
    assert recovery.t1_offset_s == pytest.approx(FILE_S, abs=0.001)
    assert stim.duration_s == pytest.approx(report.detected_duration_s, abs=0.001)


def test_an_epoch_is_a_view_not_a_copy(protocol: ProtocolSpec) -> None:
    """A 20-minute 9-channel epoch would be gigabytes to copy, and there is no need."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib)
    stim, recovery, _report = split_stim_recovery(rec, protocol=protocol)

    assert stim is not None
    assert stim.recording.data.base is rec.data
    assert recovery.recording.data.base is rec.data
    assert recovery.recording.fs == rec.fs


def test_the_stim_epoch_is_kept_and_marked_excluded(protocol: ProtocolSpec) -> None:
    """Kept, not deleted - that is what lets the deferred within-stim work happen."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    stim, recovery, _report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert stim is not None
    assert stim.excluded
    assert stim.category == "excluded_epoch"
    assert stim.recording.data.shape[0] > 0
    assert not recovery.excluded
    assert recovery.category is None


def test_excluded_epoch_is_not_masked_motion(protocol: ProtocolSpec) -> None:
    """The accounting contract: the two categories can never be summed by accident.

    A protocol exclusion counted as model-driven blanking would corrupt exactly the
    coverage-confound regression that exists to catch confounds, so the excluded epoch
    has no blanking fraction at all rather than a zero one.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    stim, recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert stim is not None
    with pytest.raises(ValueError, match="not model-driven blanking"):
        blanking_fraction(stim, 10.0)

    assert report.to_provenance()["exclusion_category"] == "excluded_epoch"
    assert blanking_fraction(recovery, 10.0) == pytest.approx(10.0 / recovery.duration_s)


def test_the_blanking_denominator_is_the_recovery_epoch_not_the_file(
    protocol: ProtocolSpec,
) -> None:
    """Using the file's duration understates every rate by the stim fraction.

    Here the recovery epoch is 1080 s of a 1500 s file, so a file-length denominator
    would report a blanking fraction 28% too low.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    _stim, recovery, _report = split_stim_recovery(_recording(vib), protocol=protocol)

    measured = blanking_fraction(recovery, 108.0)
    against_file = 108.0 / FILE_S

    assert measured == pytest.approx(0.10, abs=0.002)
    assert against_file == pytest.approx(0.072, abs=0.002)
    assert measured / against_file == pytest.approx(FILE_S / recovery.duration_s, rel=0.01)


def test_an_epoch_carries_no_settling_window_until_task_13_measures_one(
    protocol: ProtocolSpec,
) -> None:
    """Never fabricate. The settling time is unknown here, so it is absent, not guessed."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    _stim, recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert recovery.unassessable_head_s is None
    assert recovery.unassessable_tail_s is None
    assert report.to_provenance()["settling_measured"] is False

    settled = recovery.with_settling(0.5)
    assert settled.unassessable_head_s == 0.5
    assert settled.unassessable_tail_s == 0.5
    assert settled.duration_s == recovery.duration_s


def test_a_settling_window_longer_than_the_epoch_raises(protocol: ProtocolSpec) -> None:
    """An epoch that is entirely settling carries no assessable data, and says so."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    _stim, recovery, _report = split_stim_recovery(_recording(vib), protocol=protocol)

    with pytest.raises(ValueError, match="nothing assessable"):
        recovery.with_settling(recovery.duration_s)


def test_an_epoch_cannot_be_excluded_without_a_category() -> None:
    """Two ways to build an epoch whose accounting would be ambiguous."""
    rec = _recording(np.zeros(10), dur_s=1.0, with_vib_channel=False)

    with pytest.raises(ValueError, match="carries no category"):
        Epoch(name="stim", recording=rec, t0_offset_s=0.0, duration_s=1.0, excluded=True)

    with pytest.raises(ValueError, match="not excluded but carries"):
        Epoch(
            name="recovery",
            recording=rec,
            t0_offset_s=0.0,
            duration_s=1.0,
            excluded=False,
            category="excluded_epoch",
        )


# ---------------------------------------------------------------------------
# the manual path, provenance and logging
# ---------------------------------------------------------------------------


def test_a_declared_stim_end_skips_the_search(protocol: ProtocolSpec) -> None:
    """The ``splitStimRecoveryManual.m`` path, which the loader's ``stim_end_s`` feeds.

    The manual form assumes the stim starts at sample 0, which is ``clipped_start`` by
    construction - stated here rather than left implicit as it is in the MATLAB.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    stim, recovery, report = split_stim_recovery(
        _recording(vib), protocol=protocol, stim_end_s=118.0
    )

    assert report.method == "manual"
    assert report.clipped_start
    assert report.onset_s == 0.0
    assert report.offset_s == pytest.approx(118.0)
    assert report.threshold_crosscheck_s is None
    assert stim is not None
    assert recovery.t0_offset_s == pytest.approx(118.0, abs=0.001)


def test_a_declared_stim_end_outside_the_recording_raises(protocol: ProtocolSpec) -> None:
    """A boundary that is not inside the file is a typo, not a split."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)

    with pytest.raises(ValueError, match="outside this"):
        split_stim_recovery(_recording(vib), protocol=protocol, stim_end_s=FILE_S + 1.0)


def test_provenance_carries_the_protocol_and_omits_what_is_absent(
    protocol: ProtocolSpec,
) -> None:
    """Everything emitted carries provenance, and a missing value is an **absent key**."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)
    record = report.to_provenance()

    assert record["protocol"] == protocol.to_json()
    assert record["method"] == "matched"
    assert set(record) >= {
        "onset_s",
        "offset_s",
        "detected_duration_s",
        "status",
        "clipped_start",
        "clipped_end",
        "duty_cycle",
        "epoch_count",
        "threshold_crosscheck_s",
        "vib_channel",
        "recovery_duration_s",
    }
    assert all(value is not None for value in record.values())

    manual = split_stim_recovery(_recording(vib), protocol=protocol, stim_end_s=118.0)[2]
    assert "threshold_crosscheck_s" not in manual.to_provenance()
    assert "vib_channel" not in manual.to_provenance()


def test_the_threshold_crosscheck_is_reported_but_does_not_decide(
    protocol: ProtocolSpec,
) -> None:
    """The OFF-anchored threshold is a cross-check, never the primary detector.

    On a clean file the two agree closely; the point is that the reported value comes
    from a rule that had no say in the boundary.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.threshold_crosscheck_s is not None
    assert report.threshold_crosscheck_s == pytest.approx(STIM_START_S, abs=EDGE_REFINE_PAD_S)


def test_the_split_is_logged(protocol: ProtocolSpec, caplog: pytest.LogCaptureFixture) -> None:
    """The boundary, the status and the exclusion category, for a human reading QC."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    with caplog.at_level(logging.INFO, logger="gems_blanking_v2.io.stim_split"):
        split_stim_recovery(_recording(vib), protocol=protocol)

    messages = [r.getMessage() for r in caplog.records]
    assert any("excluded_epoch not masked_motion" in m for m in messages)
    assert any("recovery" in m for m in messages)


# ---------------------------------------------------------------------------
# why the ordering matters, quantitatively
# ---------------------------------------------------------------------------


def test_the_stim_epoch_inflates_the_band_reference_it_is_split_out_of() -> None:
    """**Why 03B precedes 06**, as a number, at the real 8% protocol.

    The reference is the median and MAD of the *log* envelope (invariant 5). Leaving
    the stim epoch in inflates both, so a frame that should clear a ``z > 3`` gate
    does not. Expressed as what a genuine recovery frame at ``z = 3.00`` reads when
    the reference came from the unsplit file:

    ==============  ==========  ===============  ==================
    stim fraction   file        MAD inflation    a recovery z=3.00
    ==============  ==========  ===============  ==================
    **8%**          25 min      **1.17x**        **2.47**
    10%             20 min      1.23x            2.33
    20%             10 min      1.61x            1.66
    40%             -           3.61x            0.40
    ==============  ==========  ===============  ==================

    The **MAD inflation** does the damage; the median shift is a tenth of a MAD at 8%.
    A frame that should clear the gate reads 2.47 and is silently dropped, which is
    the whole argument, and it is a fixed property of the protocol rather than
    something that varies with how violent a particular stim was - see
    :func:`test_the_reference_inflation_does_not_depend_on_the_stim_amplitude`.
    """
    fs = FS_SIG
    dur_s = 1500.0
    stim_fraction = STIM_S / dur_s
    assert stim_fraction == pytest.approx(0.08, abs=0.001), "this is the real protocol"

    signal, _spikes = make_eng(fs, dur_s, seed=0)
    stim_end = int(round(STIM_S * fs))
    contaminated = signal.copy()
    rng = np.random.default_rng(0)
    contaminated[:stim_end] += rng.normal(0.0, 100.0 * float(np.std(signal)), size=stim_end)

    band = (100.0, 300.0)
    unsplit_median, unsplit_mad = _log_envelope_reference(contaminated, fs, band)
    recovery_median, recovery_mad = _log_envelope_reference(contaminated[stim_end:], fs, band)

    a_real_z3_frame = recovery_median + 3.0 * recovery_mad
    reads_as = (a_real_z3_frame - unsplit_median) / unsplit_mad

    assert reads_as <= 2.6, f"a true z=3.00 recovery frame reads {reads_as:.2f}, expected <= 2.6"
    assert reads_as == pytest.approx(2.47, abs=0.10)
    assert reads_as < 3.0, "the ordering argument depends on this being a suppression"
    assert unsplit_mad / recovery_mad == pytest.approx(1.17, abs=0.05)


def test_the_reference_inflation_does_not_depend_on_the_stim_amplitude() -> None:
    """The surprising half of the result above, at the same 8% protocol.

    A 10x and a 1000x stim contaminant give the same inflated reference to within
    1e-4 - a relative difference of 1.3e-4 - and therefore the same shrinkage of a
    ``z = 3.00`` frame. It follows from invariant 5: the median and MAD of a log
    envelope respond to *how many* frames are contaminated, not by how much. A frame
    is either in the stim population or it is not, and making it larger moves it
    further out along a tail both statistics already ignore.
    """
    fs = FS_SIG
    dur_s = 1500.0
    signal, _spikes = make_eng(fs, dur_s, seed=0)
    stim_end = int(round(STIM_S * fs))
    sigma = float(np.std(signal))
    recovery = _log_envelope_reference(signal[stim_end:], fs, (100.0, 300.0))

    shrunk_to = []
    for multiple in (10.0, 1000.0):
        contaminated = signal.copy()
        rng = np.random.default_rng(0)
        contaminated[:stim_end] += rng.normal(0.0, multiple * sigma, size=stim_end)
        median, mad = _log_envelope_reference(contaminated, fs, (100.0, 300.0))
        shrunk_to.append(((recovery[0] + 3.0 * recovery[1]) - median) / mad)

    assert shrunk_to[0] == pytest.approx(shrunk_to[1], abs=1e-3)
    assert all(value <= 2.6 for value in shrunk_to)


def test_off_statistics_from_the_bottom_decile_are_truncation_biased() -> None:
    """The measurement behind anchoring the ON threshold outside the matched window.

    The bottom decile is the *lower tail* of the OFF distribution, so selecting it and
    taking its spread underestimates sigma badly - measured **0.000321 against
    0.001119**, 3.5x low. With the threshold that close to the OFF mean, **21% of
    genuine OFF samples cross it** against 0.0075%, and the outward walk runs away:
    onset error **-1.945 s** where anchoring outside the window gives **-50 ms**.

    **The -1.945 s figure does not reproduce under the ratified algorithm, and the
    reason is worth keeping.** It was measured against the old +/-2 s clamped
    refinement. The clamp is gone, and the outward walk that replaced it ends after
    :data:`EDGE_HOLD_S` below threshold, which caps how far a too-low threshold can
    hop from one OFF excursion to the next. Measured on the same signal, varying only
    the hold:

    ========  ==================  ================
    hold      biased onset error  ratio to anchored
    ========  ==================  ================
    0.25 s    **0.641 s**         12.8x
    1.0 s     20.955 s            419x
    2.0 s     220.765 s           4415x
    ========  ==================  ================

    So the two fixes interact, and a short hold is worth having for a second reason:
    it bounds the damage from a mis-set threshold. The sigma figures themselves
    reproduce exactly - 0.000321 against 0.001119. The assertion is the factor of ten
    the task asks for, which holds comfortably at 12.8x; the absolute "> 1 s" does not
    survive the algorithm it was measured against.
    """
    vib, true_start, _stop = make_vib(FS_VIB, FILE_S, STIM_START_S, STIM_S, seed=0)
    env = vib_envelope(vib, FS_VIB)
    protocol = default_protocol()

    width = int(round(protocol.stim_duration_s * FS_VIB))
    onset = int(round(STIM_START_S * FS_VIB))
    outside = np.ones(env.size, dtype=bool)
    outside[onset : onset + width] = False

    decile = env[env <= float(np.quantile(env, 0.10))]
    decile_sigma = 1.4826 * float(np.median(np.abs(decile - np.median(decile))))
    off = env[outside]
    off_sigma = 1.4826 * float(np.median(np.abs(off - np.median(off))))

    assert off_sigma / decile_sigma > 3.0, "the truncation bias stopped reproducing"

    decile_threshold = float(np.median(decile)) + 8.0 * decile_sigma
    off_threshold = float(np.median(off)) + 8.0 * off_sigma
    assert float((off > decile_threshold).mean()) > 0.10
    assert float((off > off_threshold).mean()) < 0.001

    hold = max(int(round(EDGE_HOLD_S * FS_VIB)), 1)
    biased = rp_measure_extent(env, onset, width, hold, decile_threshold)
    anchored = rp_measure_extent(env, onset, width, hold, off_threshold)

    biased_error = abs(biased[0] / FS_VIB - float(true_start))
    anchored_error = abs(anchored[0] / FS_VIB - float(true_start))
    assert anchored_error < 0.100
    assert anchored_error == pytest.approx(0.050, abs=0.005)
    assert biased_error == pytest.approx(0.641, abs=0.05)
    assert biased_error > 10.0 * anchored_error

    # And the interaction: the same bias at the old 2 s pad runs away entirely.
    at_old_pad = rp_measure_extent(
        env, onset, width, max(int(round(EDGE_REFINE_PAD_S * FS_VIB)), 1), decile_threshold
    )
    assert abs(at_old_pad[0] / FS_VIB - float(true_start)) > 100.0


def test_an_overrunning_walk_is_flagged_and_forced_to_review(protocol: ProtocolSpec) -> None:
    """``walk_extended`` is a **censoring flag, not a clamp**.

    A 140 s stim against a 120 s prior: the matched window lands inside the ON region
    and one edge has to walk 19.9 s to reach the truth, past the 12 s tolerance. The
    duration still comes back as the measured 140.1 s - clamping it to 120 would be
    the exact failure the unbounded walk exists to prevent - and the status says the
    number needs looking at.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, 140.0, seed=0)
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.walk_extended
    assert report.status == "review"
    assert report.detected_duration_s == pytest.approx(140.1, abs=0.15)
    assert max(report.head_extension_s, report.tail_extension_s) > protocol.stim_tolerance_s
    assert "reported as measured, not clamped" in report.reason


@pytest.mark.parametrize(("stim_s", "overrun_s"), [(140.0, 20.0), (160.0, 40.0)])
def test_the_extension_is_reported_per_edge(
    stim_s: float, overrun_s: float, protocol: ProtocolSpec
) -> None:
    """Two different meanings, so two numbers - and their **sum** is the invariant.

    A head extension moves the recovery epoch's ``t0``; a tail extension changes how
    much stim ends up inside it, so reporting one combined number would hide which
    happened. But *which* edge does the walking is an argmax tie-break inside the ON
    region - measured at 1 kHz the head walks 19.9 s and the tail 0.2 s, at 2 kHz it
    is 1.2 s and 18.9 s - so only the total is a property of the signal. It equals the
    amount by which the stim exceeds the prior, to within the envelope window.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, stim_s, seed=0)
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)
    record = report.to_provenance()

    head = float(record["head_extension_s"])
    tail = float(record["tail_extension_s"])
    assert head >= 0.0
    assert tail >= 0.0
    assert head + tail == pytest.approx(overrun_s, abs=ENVELOPE_RMS_S * 1.5)
    assert max(head, tail) > protocol.stim_tolerance_s
    assert record["walk_extended"] is True


def test_a_clean_capture_does_not_set_walk_extended(protocol: ProtocolSpec) -> None:
    """The flag must not fire on files it has nothing to say about."""
    for stim_s in (95.0, 113.0, 120.0):
        vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, stim_s, seed=0)
        _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

        assert not report.walk_extended, f"{stim_s} s fired the flag"
        assert max(report.head_extension_s, report.tail_extension_s) < 1.0


def test_the_hold_is_short_because_it_caps_the_runaway() -> None:
    """**Regression test on the reason ``EDGE_HOLD_S`` is 0.25 s**, so widening is deliberate.

    The hold does two jobs - tolerating a brief dip at a real edge, and stopping a
    too-low threshold from hopping between OFF excursions - and they have no reason
    to want the same value. Measured with the truncation-biased threshold in place,
    varying only the hold:

    ========  ====================  ==================
    hold      biased onset error    ratio to anchored
    ========  ====================  ==================
    **0.25**  **0.641 s**           **12.8x**
    1.0       20.955 s              419x
    2.0       220.765 s             4415x
    ========  ====================  ==================

    If a real edge ever demands a longer hold, the runaway bound has to come back
    through ``walk_extended``, not by quietly widening this.
    """
    assert EDGE_HOLD_S == 0.25

    vib, true_start, _stop = make_vib(FS_VIB, FILE_S, STIM_START_S, STIM_S, seed=0)
    env = vib_envelope(vib, FS_VIB)
    protocol = default_protocol()
    width = int(round(protocol.stim_duration_s * FS_VIB))
    onset = int(round(STIM_START_S * FS_VIB))

    decile = env[env <= float(np.quantile(env, 0.10))]
    biased = float(np.median(decile)) + 8.0 * 1.4826 * float(
        np.median(np.abs(decile - np.median(decile)))
    )

    errors = {}
    for hold_s in (0.25, 1.0, 2.0):
        hold = max(int(round(hold_s * FS_VIB)), 1)
        start, _stop_idx = rp_measure_extent(env, onset, width, hold, biased)
        errors[hold_s] = abs(start / FS_VIB - float(true_start))

    assert errors[0.25] == pytest.approx(0.641, abs=0.05)
    assert errors[1.0] == pytest.approx(20.955, abs=1.0)
    assert errors[2.0] == pytest.approx(220.765, abs=5.0)
    assert errors[0.25] < errors[1.0] < errors[2.0]


def test_the_recovery_duration_is_checked_independently_of_the_stim(
    protocol: ProtocolSpec,
) -> None:
    """The second check, and **the only one that still works on a clipped file**.

    Clipping censors the stim duration but leaves the recovery duration intact, so a
    ``clipped_start`` file with a full 20 minutes of recovery and one with 15 minutes
    are different and worse things - and the stim status alone cannot tell them apart,
    because it reports ``clipped_start`` for both.
    """
    intact_s = 8.0 + protocol.recovery_duration_s + 112.0 - 8.0
    short_s = intact_s - 300.0

    flags = {}
    for label, total_s in (("intact", intact_s), ("short", short_s)):
        vib, _s, _e = make_vib(FS_SIG, total_s, -8.0, STIM_S, seed=0)
        _stim, recovery, report = split_stim_recovery(
            _recording(vib, dur_s=total_s), protocol=protocol
        )
        assert report.status == "clipped_start", label
        flags[label] = (report.recovery_duration_flag, recovery.duration_s)

    assert flags["intact"][1] == pytest.approx(protocol.recovery_duration_s, abs=1.0)
    assert not flags["intact"][0]
    assert flags["short"][1] == pytest.approx(protocol.recovery_duration_s - 300.0, abs=1.0)
    assert flags["short"][0], "a 300 s short recovery must be flagged"


def test_the_recovery_flag_fires_on_a_passing_file_too(protocol: ProtocolSpec) -> None:
    """Independent of the stim status means independent, not "only when clipped"."""
    total_s = STIM_START_S + STIM_S + protocol.recovery_duration_s - 300.0
    vib, _s, _e = make_vib(FS_SIG, total_s, STIM_START_S, STIM_S, seed=0)
    _stim, _recovery, report = split_stim_recovery(
        _recording(vib, dur_s=total_s), protocol=protocol
    )

    assert report.status == "pass", "the stim epoch itself is a clean capture"
    assert report.recovery_duration_flag
    assert report.to_provenance()["recovery_duration_flag"] is True


def test_a_clipped_start_file_with_two_segments_stays_clipped_start(
    protocol: ProtocolSpec,
) -> None:
    """The ratified ordering: the count is checked after both clipping rows.

    ``clipped_start`` is a statement about whether *this file's* recovery epoch is
    usable, and a second segment elsewhere does not make an intact one unusable. The
    count is not lost - it is carried in provenance and surfaced in the audit whatever
    the status, so the escalation declined here happens in the report instead.
    """
    vib, _s, _e = make_vib(
        FS_SIG, FILE_S, -8.0, STIM_S, contaminant=(900.0, 30.0, 0.25), seed=0
    )
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.clipped_start
    assert report.epoch_count == 2
    assert report.status == "clipped_start"
    assert report.to_provenance()["epoch_count"] == 2


def test_the_returned_epoch_arrays_are_read_only(protocol: ProtocolSpec) -> None:
    """**Hard invariant 17.** A writeable view would silently corrupt the parent.

    Epochs are views because a 20-minute 9-channel epoch costs about 2 GB to copy, and
    invariant 1 has consumers writing NaN into what they are given. The two together
    mean a writeable epoch lets a consumer masking the recovery epoch overwrite the
    stim epoch's samples, and the source array's, without any error. Read-only turns
    that into an immediate exception at the point of the mistake.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib)
    stim, recovery, _report = split_stim_recovery(rec, protocol=protocol)

    assert stim is not None
    for epoch in (stim, recovery):
        assert not epoch.recording.data.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            epoch.recording.data[0, 0] = np.nan

    assert recovery.recording.data.base is rec.data, "still a view, not a copy"
    assert rec.data.flags.writeable, "the parent itself is untouched"


def test_a_consumer_that_needs_to_mask_takes_its_own_copy(protocol: ProtocolSpec) -> None:
    """The escape hatch invariant 17 names, and that it does not reach the parent."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib)
    _stim, recovery, _report = split_stim_recovery(rec, protocol=protocol)

    before = float(rec.data[int(recovery.t0_offset_s * rec.fs), 0])
    span = recovery.recording.data[:100].copy()
    span[0, 0] = np.nan

    assert np.isnan(span[0, 0])
    assert float(rec.data[int(recovery.t0_offset_s * rec.fs), 0]) == before


def test_a_consumer_reading_an_unmeasured_settling_width_must_refuse(
    protocol: ProtocolSpec,
) -> None:
    """``None`` means not yet measured, and a consumer may not read it as zero.

    Task 13 owns the settling time. A consumer that defaulted it to zero would filter
    straight across the split boundary, which is the one thing the epoch edge exists
    to prevent, and it would do so silently.
    """
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    _stim, recovery, _report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert recovery.unassessable_head_s is None
    with pytest.raises(ValueError, match="not yet measured"):
        recovery.assessable_bounds_s()

    settled = recovery.with_settling(0.5)
    assert settled.assessable_bounds_s() == pytest.approx((0.5, settled.duration_s - 0.5))


@pytest.mark.parametrize(
    ("burst_s", "expected"),
    [(30.0, 2), (6.0, 1)],
)
def test_epoch_count_uses_the_ten_percent_reporting_threshold(
    burst_s: float, expected: int, protocol: ProtocolSpec
) -> None:
    """A second stim-like segment counts when it lasts at least 10% of the stim.

    12 s at the 120 s protocol. The threshold is a reporting rule whose only job is to
    keep envelope ripple out of the count, not a physical constant - so it travels
    with the count in provenance rather than being left implicit.
    """
    assert protocol.secondary_epoch_min_s == pytest.approx(12.0)

    vib, _s, _e = make_vib(
        FS_SIG,
        FILE_S,
        STIM_START_S,
        STIM_S,
        contaminant=(900.0, burst_s, 0.25),
        seed=0,
    )
    _stim, _recovery, report = split_stim_recovery(_recording(vib), protocol=protocol)

    assert report.epoch_count == expected
    assert report.to_provenance()["epoch_count_min_s"] == pytest.approx(12.0)
    assert report.to_provenance()["epoch_count"] == expected


# ---------------------------------------------------------------------------
# the acceptance runner
# ---------------------------------------------------------------------------


def test_the_audit_reports_one_row_per_recording(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """Fixed columns in a fixed order, so the acceptance report is diffable."""
    recordings = []
    for i, stim_s in enumerate((120.0, 95.0, 113.0)):
        vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, stim_s, seed=i)
        recordings.append((tmp_path / f"rec_{i}.mat", _recording(vib, seed=i)))

    table = audit_stim_splits(recordings, protocol)

    assert list(table.columns) == list(AUDIT_COLUMNS)
    assert len(table) == 3
    assert table["detected_duration_s"].tolist() == pytest.approx([120.1, 95.1, 113.1], abs=0.2)
    assert table["status"].tolist() == ["pass", "review", "pass"]
    assert table["epoch_count"].tolist() == [1, 1, 1]
    assert table["error"].isna().all()
    assert not table["path"].str.contains("\\\\").any(), "a path escaped as a Windows string"


def test_the_epoch_count_appears_in_every_audit_row(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """Carried whatever the status, which is what makes the ordering safe.

    The status order declines to escalate a clipped or failing file on the strength of
    a second segment elsewhere. That is only defensible if the count still reaches a
    human, so it is in the audit row for every status, and ``audit_summary`` counts the
    files above one.
    """
    rows = []
    for name, start_s, contaminant in (
        ("clean", STIM_START_S, None),
        ("clipped", -8.0, (900.0, 30.0, 0.25)),
        ("review", STIM_START_S, (900.0, 30.0, 0.25)),
    ):
        vib, _s, _e = make_vib(
            FS_SIG, FILE_S, start_s, STIM_S, contaminant=contaminant, seed=0
        )
        rows.append((tmp_path / f"{name}.mat", _recording(vib)))

    table = audit_stim_splits(rows, protocol)
    summary = audit_summary(table)

    assert table["status"].tolist() == ["pass", "clipped_start", "review"]
    assert table["epoch_count"].tolist() == [1, 2, 2]
    assert table["epoch_count"].notna().all(), "a status must never suppress the count"
    assert summary["epoch_count_above_one_n"] == 2.0


def test_a_split_that_fails_becomes_a_row_not_a_skip(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """A file the split refuses is the most interesting row in the table, not a gap."""
    good, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    bad, _s2, _e2 = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=1)

    table = audit_stim_splits(
        [
            (tmp_path / "good.mat", _recording(good)),
            (tmp_path / "no_monitor.mat", _recording(bad, with_vib_channel=False)),
        ],
        protocol,
    )

    assert len(table) == 2
    failed = table[table["error"].notna()]
    assert len(failed) == 1
    assert "no stimulation-monitor channel" in str(failed.iloc[0]["error"])
    assert failed.iloc[0]["path"].endswith("no_monitor.mat")


def test_the_audit_diffs_against_the_matlab_boundary(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """The first subtask: where the two methods disagree, and by how much.

    A ``_recovery.mat`` whose ``recoveryMask`` puts the boundary 30 s early leaves
    30 s of stim inside the recovery epoch, and the audit says so in seconds rather
    than as a boolean disagreement.
    """
    from scipy.io import savemat  # noqa: PLC0415

    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib)
    path = tmp_path / "rec.mat"

    n = rec.data.shape[0]
    recovery_mask = np.ones(n, dtype=bool)
    recovery_mask[: int(round(390.0 * FS_SIG))] = False  # old boundary 30 s early
    savemat(tmp_path / "rec_recovery.mat", {"recoveryMask": recovery_mask})

    table = audit_stim_splits([(path, rec)], protocol)
    row = table.iloc[0]

    assert float(row["matlab_offset_s"]) == pytest.approx(390.0, abs=0.01)
    assert float(row["offset_difference_s"]) == pytest.approx(30.05, abs=0.2)
    assert float(row["contaminated_recovery_s"]) == pytest.approx(30.05, abs=0.2)


def test_a_recording_the_old_pipeline_never_split_has_no_diff(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """Absent is absent. A missing ``_recovery.mat`` is not agreement."""
    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    table = audit_stim_splits([(tmp_path / "rec.mat", _recording(vib))], protocol)

    assert table.iloc[0]["matlab_onset_s"] is None or np.isnan(
        float(table.iloc[0]["matlab_onset_s"])
    )
    assert table.iloc[0]["status"] == "pass"


def test_the_acceptance_figures_are_written_headlessly(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """Three plots on the ``Agg`` backend: durations, signed onsets, signed offsets.

    The two difference distributions are separate because onset error costs pre-stim
    baseline while offset error contaminates the recovery reference, and only the
    second damages the science.
    """
    import matplotlib  # noqa: PLC0415

    recordings = [
        (
            tmp_path / f"rec_{i}.mat",
            _recording(make_vib(FS_SIG, FILE_S, STIM_START_S, d, seed=i)[0], seed=i),
        )
        for i, d in enumerate((120.0, 118.0, 95.0))
    ]
    figure_path = tmp_path / "figures" / "stim.png"

    audit_stim_splits(recordings, protocol, figure_path=figure_path)

    for expected in ("stim.png", "stim_onset.png", "stim_offset.png"):
        written = figure_path.with_name(expected)
        assert written.is_file(), expected
        assert written.stat().st_size > 0
    assert matplotlib.get_backend().lower() == "agg"


def test_the_audit_summary_reports_signed_medians_and_iqrs(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """A non-zero median is a finding in its own right, so it is reported directly.

    The biases most likely to show up - the truncated OFF sigma and the old +/-2 s
    clamp - sit inside the refinement window, so a report filtered on disagreement
    size would show nothing while a systematic onset bias was present.
    """
    from scipy.io import savemat  # noqa: PLC0415

    recordings = []
    for i in range(3):
        vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=i)
        rec = _recording(vib, seed=i)
        mask = np.ones(rec.data.shape[0], dtype=bool)
        # Old boundary 2 s early at both ends: the systematic bias the audit hunts.
        mask[int(round(298.0 * FS_SIG)) : int(round(418.0 * FS_SIG))] = False
        savemat(tmp_path / f"rec_{i}_recovery.mat", {"recoveryMask": mask})
        recordings.append((tmp_path / f"rec_{i}.mat", rec))

    summary = audit_summary(audit_stim_splits(recordings, protocol))

    assert summary["n"] == 3.0
    assert summary["onset_difference_s_median"] == pytest.approx(1.95, abs=0.2)
    assert summary["offset_difference_s_median"] == pytest.approx(2.05, abs=0.2)
    assert summary["onset_difference_s_iqr"] == pytest.approx(0.0, abs=0.01)
    assert summary["competing_segment_n"] == 0.0
    assert not any(np.isnan(v) for v in summary.values()), "a missing value must be absent"


def test_a_competing_long_segment_is_called_out(
    tmp_path: Path, protocol: ProtocolSpec
) -> None:
    """The files the audit is actually hunting: the old split landed somewhere else.

    Flagged when the two methods disagree by more than the stim duration, not by more
    than the refinement window - that is the threshold the acceptance now specifies,
    because the residual biases sit inside the old window.
    """
    from scipy.io import savemat  # noqa: PLC0415

    vib, _s, _e = make_vib(FS_SIG, FILE_S, STIM_START_S, STIM_S, seed=0)
    rec = _recording(vib)
    mask = np.ones(rec.data.shape[0], dtype=bool)
    mask[int(round(700.0 * FS_SIG)) : int(round(1000.0 * FS_SIG))] = False
    savemat(tmp_path / "rec_recovery.mat", {"recoveryMask": mask})

    table = audit_stim_splits([(tmp_path / "rec.mat", rec)], protocol)

    assert bool(table.iloc[0]["competing_segment"])
    assert float(table.iloc[0]["onset_difference_s"]) == pytest.approx(-400.0, abs=1.0)
    assert audit_summary(table)["competing_segment_n"] == 1.0


def test_the_acceptance_run_against_the_archive(real_recording: Path | None) -> None:
    """Re-split every existing ``stim_recovery`` recording - **Drive only**.

    Skips when the shared drive is unreachable, which is everywhere so far, so this
    task's acceptance is unmeasured: the report per file - detected duration against
    120 s, clipping flags, epochs found, and the boundary difference against the
    current MATLAB result - waits for the Drive, like task 02's.

    What it checks when it does run is only that the runner completes and produces
    the fixed columns. The findings themselves are for a human to read: the
    distribution should be a tight spike at 120 s with a short tail of clipped
    captures, and every file where the two methods disagree by more than the
    edge-refinement window needs calling out.
    """
    if real_recording is None:
        pytest.skip("no gems_root on this machine, so the archive is unreachable")

    root = real_recording.parent
    protocol_file = protocol_path(root)
    if not protocol_file.is_file():
        pytest.skip(f"no {PROTOCOL_FILENAME} on the shared drive yet")

    from gems_blanking_v2.io.recording import load_recording  # noqa: PLC0415

    loaded = [
        (path, load_recording(path, animal="J").recording)
        for path in sorted(root.glob("*_sr_*.mat"))[:5]
    ]
    if not loaded:
        pytest.skip("no stim_recovery recordings found next to the reference file")

    table = audit_stim_splits(loaded, load_protocol(protocol_file))
    assert list(table.columns) == list(AUDIT_COLUMNS)
