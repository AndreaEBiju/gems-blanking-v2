"""Task 15: the MATLAB handoff - write a recording's masks, refusing what QC holds.

:func:`write_mask_file` computes the QC gate FROM THE MASKS ITSELF (``emit.qc.emit_gate``:
the retention gate and the 20% / 3x blank hold), so a caller cannot hand it a gate that
says "pass" for masks that do not; ``min_retention`` and the animal medians are required
arguments. A held recording is written only with a non-blank ``release`` naming who
released it and why. ``gate_json`` records the verdict, ``min_retention``, the medians
used, the unknown-median notes and the release.

The file holds one ``blank_<consumer>_<signal>`` (N x 2, 1-based inclusive samples into
the epoch, via ``extent.grid`` - invariant 15) per mask key, never a merged one, plus
``notmeasured_mmc_<signal>`` (R6), ``provenance_json``, ``events_json``,
``retention_json`` and ``gate_json``, and the epoch's ``fs``, ``nSamples``,
``epochStart_s`` and exact first sample ``epochStartSample0`` (MATLAB slices by the
sample, never by the seconds). Provenance that does not name a model is refused.
``<signal>`` is :func:`matlab_signal_token` of the signal name: a pairs-lead HR channel
``LVN2-RVN2`` is written ``LVN2_minus_RVN2`` (``-`` is illegal in a MATLAB name).

**Spike-consumer line distrust (RULINGS 2026-10-08 (d) 2, (e) Q2).** Distrusted minutes
are NaN in the spike consumer's input only: ``blank_spikes_<cuff>_T`` is the union of the
spike consumer's motion mask and its distrusted minutes (one consumer, two reasons - no
other consumer's spans change, invariant 2), so ``step1_bandpass`` (``isnan`` only) drops
them. The accounting keeps them apart: ``distrust_spikes_<cuff>_T`` carries the
distrusted minutes alone, ``linedistrust_json`` the whole
:class:`~gems_blanking_v2.emit.line_distrust.LineDistrustRecord`, ``retention_json`` and
the 20% / 3x hold read the motion masks only ((e) Q2b), and ``gate_json`` reports
``spike_time_lost`` per cuff (motion blank, distrust, overlap counted once) plus
``line_distrust_listed``: cuffs whose distrusted time exceeds 50%, listed for Andrea,
never held. ``line_distrust`` is a required argument: a recording whose spike consumer
reads anything must carry a record covering exactly those signals, and one that reads
nothing passes ``None``. The record's provenance is copied into ``provenance_json``
(``spike_line_distrust``) from the record itself.

**Spike-consumer peri-R spans (RULING 2026-10-08 (k) 1).** The same shape: the
:class:`~gems_blanking_v2.emit.peri_r.PeriRRecord`'s sample-exact spans around every beat of
the routed train join ``blank_spikes_<cuff>_T`` and no other consumer's spans (invariant 2);
``perir_spikes_<cuff>_T`` carries them alone and ``perir_json`` the record. ``retention_json``
and the hold read the motion masks only; ``gate_json`` reports ``spike_peri_r_fraction``.
``peri_r`` is a required argument like ``line_distrust``: a recording whose spike consumer
reads anything carries a record (one with no train when the routing gives it none - then
its spike spans are exactly the motion and distrust spans), and one that reads nothing
passes ``None``. The record's provenance - the window, its file's sha256, the train - is
copied into ``provenance_json`` (``spike_peri_r``) from the record itself.

**No heartbeat reference (RULING 2026-10-09 (g) 1, replacing (b) 2).** Minutes of the
routed train's region with no beat are KEPT for the spike consumer and flagged, per minute:
``noheartref_json`` (always present: the rule, the flag, the minutes, or why there are none),
``noheartref_spikes_<cuff>_T`` (their 1-based inclusive epoch rows - flags, never blank
spans) and the provenance's ``spike_no_heartbeat_reference``. ``gate_json`` reports
``spike_no_heartbeat_reference_s``; the gate and the hold never see them.

OUTSIDE THE GENERATION HASH.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.io import savemat

from gems_blanking_v2.emit.line_distrust import LineDistrustRecord, minute_sample_bounds
from gems_blanking_v2.emit.masks import (
    NO_BEAT_RULE,
    NO_HEARTBEAT_REFERENCE,
    ConsumerMask,
    MaskKey,
    NoBeatMinute,
    mask_sample_spans,
    mmc_not_measured,
)
from gems_blanking_v2.emit.peri_r import PeriRRecord
from gems_blanking_v2.emit.provenance import MaskProvenance, ProvenanceError
from gems_blanking_v2.emit.qc import EmitGate, emit_gate, line_distrust_listed, spike_time_lost
from gems_blanking_v2.extent.grid import (
    T0_TOLERANCE_S,
    n_grid_frames,
    seconds_to_sample,
    to_matlab_inclusive,
)
from gems_blanking_v2.extent.routing import RouteDecision
from gems_blanking_v2.extent.tolerance import (
    OUT_OF_BUILD_CONSUMERS,
    expected_consumers,
    extent_consumers,
)
from gems_blanking_v2.io.nan_interop import assert_no_zero_runs

__all__ = ["EPOCH_START_SAMPLE_KEY", "PAIR_LEAD_TOKEN", "RecordingHeldError",
           "epoch_start_fields", "matlab_signal_token", "no_heartbeat_reference_record",
           "signal_from_matlab_token", "write_mask_file"]

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
MATLAB_NAME_MAX: Final = 63
"""MATLAB's ``namelengthmax``."""
PAIR_LEAD_TOKEN: Final = "_minus_"
"""How a cross-site pairs lead ``"<plus>-<minus>"`` (``physio.hr_pairs.PairLead.name``;
adopted for HR by ruling 2026-10-02 (c)) appears inside a MATLAB variable name, where
``-`` is illegal: ``LVN2-RVN2`` -> ``LVN2_minus_RVN2``."""


class RecordingHeldError(RuntimeError):
    """QC holds the recording; it is emitted only with an explicit release."""


EPOCH_START_SAMPLE_KEY: Final = "epochStartSample0"
"""The mask file variable holding the epoch's exact first sample (Andrea, 2026-10-09).

0-based: the epoch is file samples ``[i0, i0 + nSamples)`` here, which are MATLAB rows
``i0 + 1 .. i0 + nSamples``. MATLAB slices by this value and never converts
``epochStart_s`` to samples (invariant 15). Stored as an integer-valued double, like
``nSamples``, which is exact below 2**53."""


def epoch_start_fields(epoch_start_s: float, fs: float,
                       epoch_start_sample: int | None = None) -> dict[str, float]:
    """Return the epoch-start variables of a mask file: seconds and the exact sample.

    The ONE place the start sample is formed (invariants 22, 33). It is
    :func:`~gems_blanking_v2.extent.grid.seconds_to_sample` of the start, the rule the
    Night 4/5 runner slices with (``round(lo * fs)``). A caller that passes the sample
    it actually sliced with must agree with that rule; two sources of truth that disagree
    raise rather than letting MATLAB and Python index different samples.
    """
    canonical = seconds_to_sample(epoch_start_s, fs)
    if canonical < 0:
        msg = f"epoch start {epoch_start_s} s is before the recording's first sample"
        raise ValueError(msg)
    if epoch_start_sample is not None:
        if isinstance(epoch_start_sample, bool) or not isinstance(epoch_start_sample,
                                                                  int | np.integer):
            msg = f"epoch_start_sample must be an integer sample index, got {epoch_start_sample!r}"
            raise TypeError(msg)
        if int(epoch_start_sample) != canonical:
            msg = (f"epoch_start_sample {int(epoch_start_sample)} disagrees with "
                   f"round({epoch_start_s} s x {fs} Hz) = {canonical}")
            raise ValueError(msg)
    if canonical >= 2**53:
        msg = f"epoch start sample {canonical} is not exact as a double"
        raise ValueError(msg)
    return {"epochStart_s": float(epoch_start_s), EPOCH_START_SAMPLE_KEY: float(canonical)}


def matlab_signal_token(signal: str) -> str:
    """Return the one canonical form of a signal name in a MATLAB variable name (inv. 22).

    A pairs lead ``"<plus>-<minus>"`` becomes ``"<plus>_minus_<minus>"``; every other name
    is unchanged. Exactly reversible by :func:`signal_from_matlab_token`: a name that
    already contains :data:`PAIR_LEAD_TOKEN`, or more than one ``-``, is refused rather
    than written ambiguously - and so is any other name whose token would not read back
    (``d_minus-fa`` -> ``d_minus_minus_fa``, which reads back as ``d-minus_fa``: the
    replacement overlaps the text before it). Found by the Night 6 cross-boundary
    round trip; the MATLAB twin is ``matlab/night6/night6_token_from_signal.m``.
    """
    token = signal.replace("-", PAIR_LEAD_TOKEN)
    if (PAIR_LEAD_TOKEN in signal or signal.count("-") > 1
            or token.count(PAIR_LEAD_TOKEN) > 1 or token.replace(PAIR_LEAD_TOKEN, "-") != signal):
        msg = f"signal name {signal!r} has no unambiguous MATLAB form"
        raise ValueError(msg)
    return token


def signal_from_matlab_token(token: str) -> str:
    """Inverse of :func:`matlab_signal_token`."""
    if token.count(PAIR_LEAD_TOKEN) > 1 or "-" in token:
        msg = f"{token!r} is not a MATLAB signal token"
        raise ValueError(msg)
    return token.replace(PAIR_LEAD_TOKEN, "-")


def _matlab_name(prefix: str, consumer: str, signal: str) -> str:
    name = f"{prefix}_{consumer}_{matlab_signal_token(signal)}"
    if (not name.replace("_", "").isalnum() or len(name) > MATLAB_NAME_MAX
            or not name[0].isalpha()):
        msg = f"{name!r} is not a MATLAB variable name"
        raise ValueError(msg)
    return name


def _union(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, disjoint union of half-open sample spans (touching spans join)."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _matlab_spans(invalid: Bool, fs: float, n_samples: int, grid_s: float, what: str,
                  extra: Sequence[tuple[int, int]] = ()) -> F64:
    """Return the invalid runs (plus ``extra`` spans, same consumer) as MATLAB spans."""
    spans = _union([*mask_sample_spans(invalid, fs, n_samples, grid_s), *extra])
    if not spans:
        return np.zeros((0, 2), dtype=np.float64)
    out = to_matlab_inclusive([a for a, _ in spans], [b for _, b in spans])
    assert_no_zero_runs(out.ravel(), what=what)
    return out


def _check_coverage(masks: Mapping[MaskKey, ConsumerMask],
                    signals: Mapping[str, Sequence[str]]) -> None:
    """Raise unless ``masks`` cover exactly what the recording reads, on the right bands."""
    consumers = extent_consumers()
    required = expected_consumers()
    absent = [c for c in required if c not in signals]
    if absent:
        msg = (f"signals must name every consumer the recording could read (an empty tuple "
               f"where it reads none); missing {absent}")
        raise ValueError(msg)
    if not masks:
        msg = "refusing to write a mask file with no masks"
        raise ValueError(msg)
    unknown = sorted(({c for c, _s, _b in masks} - set(consumers))
                     | (set(signals) - set(consumers)))
    if unknown:
        msg = f"unknown consumers {unknown}; the consumers are {sorted(consumers)}"
        raise ValueError(msg)
    wrong_band = sorted(f"{c}/{s}: {b}" for c, s, b in masks if b != consumers[c].band)
    if wrong_band:
        msg = f"masks on the wrong band for their consumer: {wrong_band}"
        raise ValueError(msg)
    want_keys = {(c, s) for c, names in signals.items() if c not in OUT_OF_BUILD_CONSUMERS
                 for s in names}
    have_keys = {(c, s) for c, s, _b in masks if c not in OUT_OF_BUILD_CONSUMERS}
    if want_keys != have_keys:
        msg = (f"masks do not cover what this recording reads: missing "
               f"{sorted(want_keys - have_keys)}, unexpected {sorted(have_keys - want_keys)}")
        raise ValueError(msg)


def _check_not_computed(not_computed: Mapping[str, str],
                        signals: Mapping[str, Sequence[str]]) -> None:
    """Raise unless each not-computed consumer is known, has a reason and reads nothing."""
    for consumer, why in not_computed.items():
        if consumer not in expected_consumers() or not str(why).strip():
            msg = f"not_computed needs a known consumer and a reason, got {consumer!r}: {why!r}"
            raise ValueError(msg)
        if signals.get(consumer):
            msg = (f"{consumer} is marked not computed but reads {list(signals[consumer])}; "
                   "a consumer that is not computed reads nothing")
            raise ValueError(msg)


def _with_line_distrust(provenance: MaskProvenance, line_distrust: LineDistrustRecord | None,
                        *, signals: Mapping[str, Sequence[str]], fs: float, n_samples: int,
                        epoch_start_s: float) -> MaskProvenance:
    """Check the record against this file and return the provenance that names it."""
    spike_signals = tuple(sorted(signals.get("spikes", ())))
    if line_distrust is None:
        if spike_signals:
            msg = (f"the spike consumer reads {list(spike_signals)}: the file must carry its "
                   "line-distrust record (RULING 2026-10-08 (d) 2); none given")
            raise ValueError(msg)
        if provenance.spike_line_distrust:
            msg = "provenance names a spike line-distrust rule but no record is carried"
            raise ProvenanceError(msg)
        return provenance
    if tuple(sorted(line_distrust.signals)) != spike_signals:
        msg = (f"the line-distrust record covers {list(line_distrust.signals)} but the spike "
               f"consumer reads {list(spike_signals)}")
        raise ValueError(msg)
    if (line_distrust.fs != float(fs) or line_distrust.n_samples != int(n_samples)
            or abs(line_distrust.epoch_start_s - epoch_start_s) > T0_TOLERANCE_S):
        msg = (f"the line-distrust record is for fs {line_distrust.fs}, {line_distrust.n_samples}"
               f" samples from {line_distrust.epoch_start_s} s; this file is fs {fs}, "
               f"{n_samples} samples from {epoch_start_s} s")
        raise ValueError(msg)
    if line_distrust.recording != provenance.recording:
        msg = (f"the line-distrust record is for {line_distrust.recording!r}, the provenance "
               f"for {provenance.recording!r}")
        raise ValueError(msg)
    want = json.loads(json.dumps(line_distrust.provenance, allow_nan=False))
    if provenance.spike_line_distrust and dict(provenance.spike_line_distrust) != want:
        msg = "provenance names a different spike line-distrust rule than the record carried"
        raise ProvenanceError(msg)
    return dataclasses.replace(provenance, spike_line_distrust=want)


def _with_peri_r(provenance: MaskProvenance, peri_r: PeriRRecord | None, *,
                 signals: Mapping[str, Sequence[str]], fs: float, n_samples: int,
                 epoch_start_s: float, epoch_start_sample: float) -> MaskProvenance:
    """Check the peri-R record against this file and return the provenance that names it."""
    spike_signals = tuple(sorted(signals.get("spikes", ())))
    if peri_r is None:
        if spike_signals:
            msg = (f"the spike consumer reads {list(spike_signals)}: the file must carry its "
                   "peri-R record (RULING 2026-10-08 (k) 1; one with no train when the "
                   "routing gives none); none given")
            raise ValueError(msg)
        if provenance.spike_peri_r:
            msg = "provenance names a peri-R rule but no peri-R record is carried"
            raise ProvenanceError(msg)
        return provenance
    if tuple(sorted(peri_r.signals)) != spike_signals:
        msg = (f"the peri-R record covers {list(peri_r.signals)} but the spike consumer reads "
               f"{list(spike_signals)}")
        raise ValueError(msg)
    if (peri_r.fs != float(fs) or peri_r.n_samples != int(n_samples)
            or abs(peri_r.epoch_start_s - epoch_start_s) > T0_TOLERANCE_S
            or float(peri_r.epoch_start_sample) != float(epoch_start_sample)):
        msg = (f"the peri-R record is for fs {peri_r.fs}, {peri_r.n_samples} samples from "
               f"sample {peri_r.epoch_start_sample}; this file is fs {fs}, {n_samples} samples "
               f"from sample {int(epoch_start_sample)}")
        raise ValueError(msg)
    if peri_r.recording != provenance.recording:
        msg = (f"the peri-R record is for {peri_r.recording!r}, the provenance for "
               f"{provenance.recording!r}")
        raise ValueError(msg)
    want = json.loads(json.dumps(peri_r.provenance(), allow_nan=False))
    if provenance.spike_peri_r and dict(provenance.spike_peri_r) != want:
        msg = "provenance names a different peri-R record than the one carried"
        raise ProvenanceError(msg)
    return dataclasses.replace(provenance, spike_peri_r=want)


def _write_line_distrust(doc: dict[str, Any], masks: Mapping[MaskKey, ConsumerMask],
                         line_distrust: LineDistrustRecord | None
                         ) -> dict[str, dict[str, float | int]] | None:
    """Put the record's spans and JSON in ``doc`` (beside the masks); return its cost."""
    if line_distrust is None:
        return None
    for sig in line_distrust.signals:
        spans = line_distrust.matlab_spans(sig)
        assert_no_zero_runs(spans.ravel(), what=f"line-distrust spans spikes/{sig}")
        doc[_matlab_name("distrust", "spikes", sig)] = spans
    doc["linedistrust_json"] = line_distrust.to_json()
    return spike_time_lost(masks, line_distrust)


def _spike_extra(consumer: str, signal: str, line_distrust: LineDistrustRecord | None,
                 peri_r: PeriRRecord | None) -> list[tuple[int, int]]:
    """Return the sample spans joining ``consumer``'s motion runs: the spike consumer's only."""
    if consumer != "spikes":
        return []
    distrusted = line_distrust.sample_spans(signal) if line_distrust is not None else []
    heart = peri_r.sample_spans(signal) if peri_r is not None else []
    return [*distrusted, *heart]


def _write_peri_r(doc: dict[str, Any], peri_r: PeriRRecord | None,
                  n_samples: int) -> dict[str, float] | None:
    """Put the peri-R spans and JSON in ``doc``; return the blanked fraction per signal."""
    if peri_r is None:
        return None
    spans = peri_r.matlab_spans()
    assert_no_zero_runs(spans.ravel(), what="peri-R spans spikes")
    for sig in peri_r.signals:
        doc[_matlab_name("perir", "spikes", sig)] = spans
    doc["perir_json"] = peri_r.to_json()
    blanked = sum(b - a for a, b in peri_r.spans)
    return dict.fromkeys(peri_r.signals, blanked / n_samples)


def no_heartbeat_reference_record(no_beat_minutes: Sequence[NoBeatMinute] | None,
                                  peri_r: PeriRRecord | None, *,
                                  signals: Mapping[str, Sequence[str]], fs: float,
                                  n_samples: int, epoch_start_s: float) -> dict[str, Any]:
    """Return the epoch's "no heartbeat reference" flags (RULING 2026-10-09 (g) 1).

    No-beat minutes are KEPT for the spike consumer: this record flags them, per minute,
    and never blanks them. Each flagged minute carries its recording-timeline bounds, its
    kind, the flag :data:`~gems_blanking_v2.emit.masks.NO_HEARTBEAT_REFERENCE`, and its
    in-epoch part as seconds and as 0-based half-open epoch samples (the line-distrust
    minute conversion, ``line_distrust.minute_sample_bounds``). A minute with no in-epoch
    sample is not listed.

    Refuses, by name: ``None`` while the spike consumer reads a signal and the peri-R
    record has a routed train (the caller must say which minutes - ``[]`` when none); a
    non-empty list when the spike consumer reads nothing or the recording has no train
    (no-beat minutes are minutes of a train's region); and a minute number given twice.
    """
    spike = tuple(sorted(signals.get("spikes", ())))
    has_train = peri_r is not None and peri_r.train is not None
    rec: dict[str, Any] = {"rule": NO_BEAT_RULE, "flag": NO_HEARTBEAT_REFERENCE,
                           "signals": list(spike), "blanked": False,
                           "counts_toward_retention_hold": False}
    if not spike or not has_train:
        if no_beat_minutes:
            msg = ("no-beat minutes given for an epoch whose spike consumer reads nothing or "
                   "whose recording has no routed train: they are minutes of a train's region")
            raise ValueError(msg)
        rec["minutes"] = []
        rec["none_because"] = ("the spike consumer reads no signal" if not spike else
                               "no routed train ((k) 1): no train region, no no-beat minute")
        return rec
    if no_beat_minutes is None:
        msg = ("the spike consumer reads a signal and the peri-R record has a routed train: "
               "pass the train's no-beat minutes ([] when there are none) so each is flagged "
               "'no heartbeat reference' (RULING 2026-10-09 (g) 1)")
        raise ValueError(msg)
    seen = [m.minute for m in no_beat_minutes]
    if len(set(seen)) != len(seen):
        msg = f"no-beat minute numbers given twice: {sorted(seen)}"
        raise ValueError(msg)
    stop = epoch_start_s + n_samples / fs
    rows: list[dict[str, Any]] = []
    for m in sorted(no_beat_minutes, key=lambda x: x.start_s):
        a, b = max(m.start_s, epoch_start_s), min(m.stop_s, stop)
        if not b > a:
            continue
        k0, k1 = minute_sample_bounds(a, b, epoch_start_s=epoch_start_s, fs=fs,
                                      n_samples=n_samples)
        if k1 <= k0:
            continue
        rows.append({"minute": int(m.minute), "start_s": float(m.start_s),
                     "stop_s": float(m.stop_s), "kind": m.kind, "flag": NO_HEARTBEAT_REFERENCE,
                     "in_epoch_s": [float(a), float(b)], "samples": [int(k0), int(k1)]})
    rec["minutes"] = rows
    rec["flagged_s"] = float(sum(r["in_epoch_s"][1] - r["in_epoch_s"][0] for r in rows))
    return rec


def _with_no_heartbeat_reference(provenance: MaskProvenance, record: Mapping[str, Any]
                                 ) -> MaskProvenance:
    """Copy the flag record into the provenance; refuse one that disagrees with it."""
    want = json.loads(json.dumps(dict(record), allow_nan=False))
    have = provenance.spike_no_heartbeat_reference
    if have and dict(have) != want:
        msg = ("the provenance's spike_no_heartbeat_reference disagrees with the no-beat "
               "minutes this file carries")
        raise ValueError(msg)
    return dataclasses.replace(provenance, spike_no_heartbeat_reference=want)


def _write_no_heartbeat_reference(doc: dict[str, Any], record: Mapping[str, Any]) -> None:
    """``noheartref_json`` (always) and ``noheartref_spikes_<cuff>_T`` per spike signal.

    The spans are 1-based inclusive epoch rows of the flagged minutes - FLAGS, never part
    of ``blank_spikes_*`` (RULING 2026-10-09 (g) 1).
    """
    rows = record["minutes"]
    for sig in record["signals"]:
        if rows:
            out = to_matlab_inclusive([r["samples"][0] for r in rows],
                                      [r["samples"][1] for r in rows])
            assert_no_zero_runs(out.ravel(), what=f"no-heartbeat-reference spans {sig}")
        else:
            out = np.zeros((0, 2), dtype=np.float64)
        doc[_matlab_name("noheartref", "spikes", sig)] = out
    doc["noheartref_json"] = json.dumps(dict(record), sort_keys=True, ensure_ascii=True,
                                        allow_nan=False)


def _gate_record(gate: EmitGate, *, heart_fraction: Mapping[str, float] | None,
                 heartref: Mapping[str, Any], lost: Mapping[str, Any] | None,
                 release: str | None) -> dict[str, Any]:
    """``gate_json``: the verdict, and beside it what the gate never reads."""
    out: dict[str, Any] = {"held": gate.held, "reasons": list(gate.reasons),
                           "retention_flagged": gate.retention_flagged,
                           "blank_held": gate.blank_held,
                           "min_retention": gate.min_retention,
                           "medians_used": dict(gate.medians_used),
                           "notes": list(gate.notes),
                           "top_routes": [list(t) for t in gate.top_routes],
                           "hum_features": dict(gate.hum_features),
                           **({"spike_peri_r_fraction": heart_fraction}
                              if heart_fraction is not None else {})}
    if heartref["signals"]:  # flagged, kept: reported beside the gate, never in it
        out["spike_no_heartbeat_reference_s"] = heartref.get("flagged_s", 0.0)
    if lost is not None:
        out["spike_time_lost"] = lost
        out["line_distrust_listed"] = line_distrust_listed(lost)
    if release:
        out["release"] = release
    return out


def write_mask_file(path: Path, masks: Mapping[MaskKey, ConsumerMask],
                    provenance: MaskProvenance | None, *, signals: Mapping[str, Sequence[str]],
                    fs: float, n_samples: int,
                    epoch_start_s: float, min_retention: float,
                    animal_median: Mapping[str, float | None],
                    line_distrust: LineDistrustRecord | None,
                    peri_r: PeriRRecord | None,
                    decisions: Iterable[RouteDecision] = (),
                    hum_features: Mapping[str, float] | None = None,
                    release: str | None = None,
                    events: Sequence[Mapping[str, Any]] = (),
                    not_computed: Mapping[str, str] | None = None,
                    epoch_start_sample: int | None = None,
                    no_beat_minutes: Sequence[NoBeatMinute] | None = None) -> Path:
    """Write the masks as MATLAB blank spans with their provenance and QC gate.

    The file carries the epoch's exact first sample as :data:`EPOCH_START_SAMPLE_KEY`
    (Andrea, 2026-10-09), formed by :func:`epoch_start_fields`; ``epoch_start_sample``,
    when given, is the sample the caller sliced with and must agree with that rule.

    ``no_beat_minutes`` are the routed train's minutes with no beat (RULING 2026-10-09
    (g) 1): KEPT for the spike consumer and flagged "no heartbeat reference" per minute -
    ``noheartref_json`` (always present), ``noheartref_spikes_<cuff>_T`` (the flagged
    minutes' 1-based inclusive epoch rows; never part of ``blank_spikes_*``), the
    provenance's ``spike_no_heartbeat_reference`` and ``gate_json``'s
    ``spike_no_heartbeat_reference_s``. They are not blanked, so they never reach the
    retention gate or the hold. Required (``[]`` when none) whenever the spike consumer
    reads a signal and ``peri_r`` has a train; see :func:`no_heartbeat_reference_record`.

    ``not_computed`` maps a consumer the recording cannot run to the reason (RULING
    2026-10-08 (f) 6: no beat train passed, so no beats file - ``hrv`` and ``breathing``,
    the two outputs of one HR_BR call, are not computed). Such a consumer must read no
    signal here. It is written as ``notcomputed_json`` (always present; ``{}`` when every
    consumer runs) so the MATLAB side skips it rather than raising.

    ``signals`` maps each consumer to the signals it reads in this recording (as for
    ``build_masks``); the masks must cover exactly those consumer x signal pairs (velocity
    excepted while task 18 is out), so the gate cannot be passed by leaving an
    over-blanked consumer out.

    Refuses: provenance that does not name a model; ``signals`` that does not name every
    consumer of :func:`~gems_blanking_v2.extent.tolerance.expected_consumers`; an empty
    ``masks``; a consumer (in ``masks`` or ``signals``) that is not in the tolerance
    table; a mask on a band other than its consumer's; masks that do not cover
    ``signals``; a held recording (gate computed here, from ``masks``) without a non-blank
    ``release``; a mask whose grid does not start at
    ``epoch_start_s`` or does not have ``floor(n_samples / fs / grid)`` frames; a
    ``line_distrust`` record missing while the spike consumer reads a signal, covering
    other signals than it reads, made for another epoch or recording, or whose rule
    disagrees with one the provenance already names; a ``peri_r`` record missing while the
    spike consumer reads a signal, given while it reads none, or covering other signals,
    another epoch or another recording. Every numeric array is checked for exact-zero runs
    (invariant 1).
    """
    if provenance is None:
        msg = "a mask file must carry provenance naming its model (task 15); none given"
        raise ProvenanceError(msg)
    provenance.validate()
    start_fields = epoch_start_fields(epoch_start_s, fs, epoch_start_sample)
    _check_coverage(masks, signals)
    _check_not_computed(not_computed or {}, signals)
    provenance = _with_peri_r(
        _with_line_distrust(provenance, line_distrust, signals=signals, fs=fs,
                            n_samples=n_samples, epoch_start_s=epoch_start_s),
        peri_r, signals=signals, fs=fs, n_samples=n_samples, epoch_start_s=epoch_start_s,
        epoch_start_sample=start_fields[EPOCH_START_SAMPLE_KEY])
    heartref = no_heartbeat_reference_record(no_beat_minutes, peri_r, signals=signals, fs=fs,
                                             n_samples=n_samples, epoch_start_s=epoch_start_s)
    if heartref["signals"]:
        provenance = _with_no_heartbeat_reference(provenance, heartref)
    gate = emit_gate(masks, min_retention=min_retention, animal_median=animal_median,
                     decisions=decisions, hum_features=hum_features)
    if release is not None and not release.strip():
        msg = "a release must say who released the recording and why; got a blank string"
        raise ValueError(msg)
    if gate.held and not release:
        msg = ("QC holds this recording (" + "; ".join(gate.reasons) + "); emit it only "
               "with an explicit release naming who released it and why")
        raise RecordingHeldError(msg)
    doc: dict[str, Any] = {}
    retention: dict[str, float] = {}
    for (consumer, signal, band), m in sorted(masks.items()):
        if abs(m.t0_s - epoch_start_s) > T0_TOLERANCE_S:
            msg = (f"{consumer}/{signal}: mask grid starts at {m.t0_s} s but the file indexes "
                   f"the epoch from {epoch_start_s} s")
            raise ValueError(msg)
        want = n_grid_frames(n_samples, fs, m.grid_s)
        if m.invalid.size != want:
            msg = f"{consumer}/{signal}: {m.invalid.size} frames for an epoch of {want}"
            raise ValueError(msg)
        doc[_matlab_name("blank", consumer, signal)] = _matlab_spans(
            m.invalid, fs, n_samples, m.grid_s, f"blank spans {consumer}/{signal}",
            extra=_spike_extra(consumer, signal, line_distrust, peri_r))
        retention[f"{consumer}|{signal}|{band}"] = m.retention  # motion only ((e) Q2b)
    for sig, frames in sorted(mmc_not_measured(masks).items()):
        doc[_matlab_name("notmeasured", "mmc", sig)] = _matlab_spans(
            frames, fs, n_samples, masks[("mmc", sig, extent_consumers()["mmc"].band)].grid_s,
            f"not-measured spans mmc/{sig}")
    lost = _write_line_distrust(doc, masks, line_distrust)
    heart_fraction = _write_peri_r(doc, peri_r, n_samples)
    _write_no_heartbeat_reference(doc, heartref)
    gate_doc = _gate_record(gate, heart_fraction=heart_fraction, heartref=heartref, lost=lost,
                            release=release)
    doc["provenance_json"] = provenance.to_json()
    doc["events_json"] = json.dumps(list(events), sort_keys=True, ensure_ascii=True,
                                    allow_nan=False)
    doc["retention_json"] = json.dumps(retention, sort_keys=True, ensure_ascii=True,
                                       allow_nan=False)
    doc["gate_json"] = json.dumps(gate_doc, sort_keys=True, ensure_ascii=True,
                                  allow_nan=False)
    doc.update({"notcomputed_json": json.dumps({k: str(v) for k, v in
                                                (not_computed or {}).items()},
                                               sort_keys=True, ensure_ascii=True,
                                               allow_nan=False),
                "fs": float(fs), **start_fields,
                "nSamples": float(n_samples)})
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with tmp.open("wb") as fh:  # a handle, so savemat cannot append ".mat" to the name
            savemat(fh, doc, do_compression=True)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
