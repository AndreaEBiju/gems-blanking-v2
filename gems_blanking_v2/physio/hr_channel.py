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

import numpy as np
import numpy.typing as npt
import pandas as pd

from gems_blanking_v2.derive.derivations import build_derivations
from gems_blanking_v2.physio.rpeaks import detect_rpeaks, rank_hr_channels
from gems_blanking_v2.types import ChannelInfo, Recording

__all__ = ["DERIVED_CANDIDATES", "best_hr_channel", "hr_candidates"]

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


def best_hr_channel(rec: Recording) -> tuple[str, pd.DataFrame]:
    """Return ``(name, table)``: task 05's pick over :func:`hr_candidates`.

    Raises ``ValueError`` when every candidate is vetoed, as
    :func:`rank_hr_channels` does - never a silent least-bad channel.
    """
    cand = hr_candidates(rec)
    trains = {c.name: detect_rpeaks(np.asarray(cand.data[:, c.index], dtype=np.float64),
                                    float(cand.fs))
              for c in cand.channels}
    return rank_hr_channels(cand, trains)
