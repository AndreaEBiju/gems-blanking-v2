"""``best_hr_channel`` for the new cohort (ruling 2026-09-29, item 6).

``CONSUMERS`` names ``best_hr_channel`` as the signal ``hrv`` and ``breathing`` read,
and the old cohort took it from Andrea's manual ``hrChanIdx``. The new cohort has
none, so it is defined here by task 05's ranking - template SNR with the
``implausible_frac`` and ``rescue_rate`` vetoes (:func:`rank_hr_channels`) - over
every signal an HR consumer could read:

- the nine raw channels, which carry the shared-ground common mode in full;
- each cuff's tripole ``T`` (the applied weights), which cancels it to the gain
  mismatch;
- ``stomach_ref``, the stomach common average.

Nothing is excluded in advance: the ranking is self-policing (a channel whose
"beats" are artifacts averages to a flat template), and whether the chosen channel
is harmed by the common mode is decided afterwards by the operational test - the
beat train with and without the events - not by where the channel sits. Re-picked
per recording, as task 05 requires.

OUTSIDE THE GENERATION HASH: the detection chain detects beats on ``R_T``
(``chain.BEAT_SIGNAL``) and does not import this module.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.derive.cm_events import Events
from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.physio.rpeaks import BeatTrain, detect_rpeaks, rank_hr_channels
from gems_blanking_v2.types import ChannelInfo, Recording

__all__ = [
    "DERIVED_CANDIDATES",
    "PROVISIONAL_MAX_TRANSIENT_HARM",
    "best_hr_channel",
    "cm_gains",
    "hr_candidates",
    "transient_harm",
]

F64 = npt.NDArray[np.float64]

DERIVED_CANDIDATES: tuple[str, ...] = ("L_T", "R_T", "stomach_ref")
"""The derived signals ranked beside the raw channels, when the recording has them."""


def hr_candidates(rec: Recording) -> Recording:
    """Return ``rec`` with the derived candidates appended as extra columns.

    The raw columns are unchanged and keep their indices; each derived signal that
    this recording can form (:data:`DERIVED_CANDIDATES`) is appended with role
    ``"aux"``. A copy: the input is never written to.
    """
    signals, _weights = build_derivations(rec)
    extra = [(n, signals[n]) for n in DERIVED_CANDIDATES if n in signals]
    clash = {n for n, _ in extra} & {c.name for c in rec.channels}
    if clash:
        msg = f"derived candidate names collide with raw channels: {sorted(clash)}"
        raise ValueError(msg)
    base = np.asarray(rec.data, dtype=np.float64)
    data = np.column_stack([base, *[np.asarray(x, dtype=np.float64) for _n, x in extra]])
    n0 = base.shape[1]
    channels = list(rec.channels) + [
        ChannelInfo(n0 + k, name, "aux", None, None, None, rec.channels[0].config)
        for k, (name, _x) in enumerate(extra)]
    return replace(rec, data=data, channels=channels)


PROVISIONAL_MAX_TRANSIENT_HARM: Final = 0.01
"""PROVISIONAL (ruling 2026-09-30, item 6): a candidate is vetoed when more than this
fraction of :data:`N_INJECTIONS` injected common-mode transients cost a beat or move a
fiducial by more than :data:`FIDUCIAL_SHIFT_S`. Measured with the HR consumer's own
detector (2026-09-30): 200 transients cost 4-135 beats on a raw contact, 0-2 on T."""

N_INJECTIONS: Final = 200
TRANSIENT_FWHM_S: Final = 0.0012
FIDUCIAL_SHIFT_S: Final = 0.002
HARM_WINDOW_S: Final = 0.1
NONCARDIAC_S: Final = 0.020
"""Events this far from every beat are non-cardiac, and only their amplitudes are
injected: part of the common-mode class is the heartbeat's own sharp edge."""


def cm_gains(events: Events, keep: npt.ArrayLike | None = None) -> dict[str, float]:
    """Return each channel's common-mode gain from the events (``keep``: which ones).

    The median over the kept events of the channel's |amplitude| over that event's
    median |amplitude| across channels. Pass the non-cardiac selection: the heart is a
    second common-mode source with its own gain pattern, and the heartbeat's sharp
    edge is part of the event class.
    """
    a = np.abs(np.asarray(events.channel_amp_uv, dtype=np.float64))
    if keep is not None:
        a = a[np.asarray(keep, dtype=bool)]
    if a.size == 0:
        return dict.fromkeys(events.channels, 1.0)
    rel = a / np.median(a, axis=1, keepdims=True)
    return {name: float(np.median(rel[:, k])) for k, name in enumerate(events.channels)}


def _trains(cand: Recording) -> dict[str, BeatTrain]:
    return {c.name: detect_rpeaks(np.asarray(cand.data[:, c.index], dtype=np.float64),
                                  float(cand.fs)) for c in cand.channels}


def _noncardiac(events: Events, beats_s: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Which events are more than :data:`NONCARDIAC_S` from every beat."""
    b = np.sort(np.asarray(beats_s, dtype=np.float64))
    t_ev = np.asarray(events.t_s, dtype=np.float64)
    if b.size == 0 or t_ev.size == 0:
        return np.ones(t_ev.size, dtype=bool)
    i = np.searchsorted(b, t_ev)
    dmin = np.minimum(np.abs(t_ev - b[np.clip(i - 1, 0, b.size - 1)]),
                      np.abs(b[np.clip(i, 0, b.size - 1)] - t_ev))
    return np.asarray(dmin > NONCARDIAC_S, dtype=bool)


def _harmed(b0: F64, b1: F64, times: F64) -> int:
    """How many injections have a beat within the window that is lost or moved."""
    b1 = np.sort(b1)
    count = 0
    for t0 in times:
        near = b0[np.abs(b0 - t0) <= HARM_WINDOW_S]
        if near.size == 0:
            continue
        if b1.size == 0:
            count += 1
            continue
        j = np.searchsorted(b1, near)
        d = np.minimum(np.abs(near - b1[np.clip(j - 1, 0, b1.size - 1)]),
                       np.abs(b1[np.clip(j, 0, b1.size - 1)] - near))
        count += int(np.any(d > FIDUCIAL_SHIFT_S))
    return count


def transient_harm(
    rec: Recording, events: Events, beats_s: npt.ArrayLike, *, seed: int = 0,
    before: dict[str, BeatTrain] | None = None,
) -> dict[str, float]:
    """Return ``{candidate: fraction of injections that cost a beat or moved a fiducial}``.

    :data:`N_INJECTIONS` Gaussian transients of :data:`TRANSIENT_FWHM_S` at uniform
    random times, amplitudes drawn from the recording's own non-cardiac events
    (common-mode amplitude, more than :data:`NONCARDIAC_S` from every beat in
    ``beats_s``), added to every raw channel times its :func:`cm_gains` gain over
    those same non-cardiac events - so a
    tripole or ``stomach_ref`` carries exactly the leak its contacts' gains produce.
    An injection harms a candidate when a beat within :data:`HARM_WINDOW_S` of it has
    no beat within :data:`FIDUCIAL_SHIFT_S` after injection (lost, or moved further).
    """
    fs = float(rec.fs)
    n = rec.data.shape[0]
    keep = _noncardiac(events, beats_s)
    amps = np.asarray(events.cm_amp_uv, dtype=np.float64)[keep]
    if amps.size == 0:
        msg = "no non-cardiac common-mode event to take an amplitude from"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(1.0, n / fs - 1.0, N_INJECTIONS))
    a = rng.choice(amps, N_INJECTIONS)
    g = cm_gains(events, keep)
    width = TRANSIENT_FWHM_S / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    pulse = np.zeros(n)
    half = int(round(10 * width * fs))
    for t0, amp in zip(times, a, strict=True):
        c = int(round(t0 * fs))
        k = np.arange(max(c - half, 0), min(c + half + 1, n))
        pulse[k] += amp * np.exp(-0.5 * ((k / fs - t0) / width) ** 2)
    data = np.array(rec.data, dtype=np.float64)
    for ch in rec.channels:
        data[:, ch.index] += g.get(ch.name, 1.0) * pulse
    before = before if before is not None else _trains(hr_candidates(rec))
    after = _trains(hr_candidates(replace(rec, data=data)))
    return {name: _harmed(np.asarray(tr.t_s), np.asarray(after[name].t_s), times) / N_INJECTIONS
            for name, tr in before.items()}


def best_hr_channel(
    rec: Recording, *, events: Events | None = None, seed: int = 0
) -> tuple[str, pd.DataFrame]:
    """Return ``(name, table)``: task 05's pick over :func:`hr_candidates`.

    With ``events`` (ruling 2026-09-30, item 6), candidates whose
    :func:`transient_harm` exceeds :data:`PROVISIONAL_MAX_TRANSIENT_HARM` are vetoed
    first and template SNR ranks only the survivors - SNR cannot see this harm,
    because the far-field ECG is common mode too. The beats that decide which events
    are non-cardiac come from the channel SNR ranks first (or, when task 05 vetoes
    every channel, the one with the most beats). Raises ``ValueError`` when every
    candidate is vetoed - never a silent least-bad channel.
    """
    cand = hr_candidates(rec)
    trains = _trains(cand)
    if events is None:
        return rank_hr_channels(cand, trains)
    try:
        first, _t = rank_hr_channels(cand, trains)
    except ValueError:
        first = max(trains, key=lambda k: trains[k].n_beats)
    harm = transient_harm(rec, events, trains[first].t_s, seed=seed, before=trains)
    survivors = {k: v for k, v in trains.items() if harm[k] <= PROVISIONAL_MAX_TRANSIENT_HARM}
    if not survivors:
        msg = ("every channel was vetoed for HRV by the transient test: "
               + ", ".join(f"{k}({v:.3f})" for k, v in sorted(harm.items(), key=lambda kv: kv[1])))
        raise ValueError(msg)
    best, table = rank_hr_channels(cand, survivors)
    table["transient_harm"] = table["channel"].map(harm)
    vetoed = pd.DataFrame([{"channel": k, "transient_harm": v, "eligible": False,
                            "vetoed_by": "transient_harm"}
                           for k, v in harm.items() if k not in survivors])
    return best, pd.concat([table, vetoed], ignore_index=True)
