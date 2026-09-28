"""Per-contact quality screen: which contacts carry independent, trustworthy signal.

What the screen is FOR (ruling 2026-09-29, invariant 43): whether a cuff's ``T`` is
trusted (a tripole formed from a failed contact keeps the common mode it exists to
remove), which cuffs velocity may use (three good contacts), and the per-recording
contact-health report. What it is NOT for: hiding a contact from detection. Every
consumer still reads a screened contact, so detection must too - round 2's broken
left contact 3 "shared nothing with its peers" because it was popping at rail
scale, and removing it from detection hid 9 of 14 missed artifacts. The stomach
contacts get a health check (:func:`assess_stomach_contacts`, with a mains-hum
rule) that feeds the report only.

The rule, over ``SCREEN_WINDOW_S`` of the recording, in ``SCREEN_BAND_HZ``:

``flat``
    robust sigma below ``FLAT_SIGMA_UV`` microvolts, or more than
    ``FLAT_FRACTION`` of consecutive raw samples exactly equal;
``duplicate``
    some other contact explains it to within a residual of ``DUPLICATE_RESIDUAL``:
    ``sqrt(1 - r**2) < 0.01``, i.e. ``|r| >= 0.99995`` in the ENG band. Stated as
    the residual - the fraction of the contact a scaled, possibly sign-flipped
    copy of the other leaves unexplained - because the margin is only visible
    there: healthy neighbours on this rig reach r = 0.99895 (63 recordings,
    2026-09-28), residual 0.046, so the first draft's ``r >= 0.999`` (residual
    0.045) had no margin at all, and 0.01 has a factor 4.6. Two amplifier
    channels wired to ONE electrode differ by amplifier noise only, which on this
    rig is indistinguishable from healthy neighbours; this screen cannot detect
    that and does not claim to;
``uncorrelated``
    ``r < UNCORRELATED_R`` with EACH of its same-cuff peers. The maximum, not the
    mean: a mean lets one failed contact condemn its two good neighbours, which is
    what the first draft of this rule did to animal A's L1 and L2 (2026-09-28).

On this rig the contacts of one cuff share a common mode that dominates the ENG
band (best same-cuff r 0.81-0.999 on every unscreened contact of 63 recordings,
2026-09-28), which is what makes ``uncorrelated`` a fault rather than a property:
a contact that shares none of it is not electrically on the cuff. The failures
found there sit at r = -0.09 to 0.04 - nothing in between.

Thresholds are fixed and stateless (invariant 7) and each has a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt, welch

from gems_blanking_v2.types import Recording

F64 = npt.NDArray[np.float64]
Reason = Literal["flat", "duplicate", "uncorrelated"]

SCREEN_BAND_HZ: Final = (300.0, 3000.0)
"""The ENG band. Correlation and sigma are measured here, where the nerve is."""

SCREEN_WINDOW_S: Final = 120.0
"""How much of the recording (from ``start_s``) the screen reads."""

FLAT_SIGMA_UV: Final = 0.5
"""Below this ENG-band robust sigma a contact is dead. Every contact of the 63
recordings sat at >= 3.8 uV; 0.5 is a factor of 7 under it."""

FLAT_FRACTION: Final = 0.05
"""Above this fraction of exactly-repeated consecutive samples a contact is flat.
Every contact of the 63 recordings repeated <= 0.4% (quantisation)."""

DUPLICATE_RESIDUAL: Final = 0.01
"""Below this ``sqrt(1 - r**2)`` against another contact, a contact is a copy."""

UNCORRELATED_R: Final = 0.5
"""Below this ENG-band ``r`` with every same-cuff peer, a contact is off the cuff."""


@dataclass(frozen=True, slots=True)
class ContactQuality:
    """One nerve contact's screen result. ``label`` is ``<cuff><contact>``, e.g. ``L3``."""

    label: str
    cuff_id: str
    contact_index: int
    sigma_uv: float
    """ENG-band robust sigma, microvolts."""
    flat_fraction: float
    r_peers: tuple[float, ...]
    """ENG-band r with each same-cuff peer, in contact order."""
    best_other: tuple[str, float]
    """The other contact with the largest ``|r|``, and that r."""
    identical_to: tuple[str, ...]
    reasons: tuple[Reason, ...]

    @property
    def copy_residual(self) -> float:
        """``sqrt(1 - r**2)`` against :attr:`best_other`: what a copy leaves unexplained."""
        return float(np.sqrt(max(0.0, 1.0 - self.best_other[1] ** 2)))

    @property
    def screened(self) -> bool:
        """True when any rule fired."""
        return bool(self.reasons)


def _robust_sigma(x: F64) -> float:
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def assess_contacts(
    rec: Recording, *, start_s: float = 0.0, window_s: float = SCREEN_WINDOW_S
) -> dict[str, ContactQuality]:
    """Screen every cuff contact of ``rec`` over ``[start_s, start_s + window_s)``.

    Raises
    ------
    ValueError
        If the window holds fewer than one second of samples - a screen that has
        seen nothing must not report a contact as good.
    """
    fs = float(rec.fs)
    nerve = [c for c in rec.channels if c.cuff_id is not None and c.contact_index is not None]
    if not nerve:
        return {}
    i0 = max(round(start_s * fs), 0)
    i1 = min(i0 + round(window_s * fs), rec.data.shape[0])
    if i1 - i0 < fs:
        msg = f"contact screen window [{start_s}, {start_s + window_s}) s holds < 1 s of data"
        raise ValueError(msg)
    raw = np.column_stack([np.asarray(rec.data[i0:i1, c.index], dtype=np.float64) for c in nerve])
    raw = raw[np.isfinite(raw).all(axis=1)]
    sos = butter(4, SCREEN_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    eng = sosfiltfilt(sos, raw, axis=0)
    labels = [f"{c.cuff_id}{c.contact_index}" for c in nerve]
    sig = np.array([_robust_sigma(eng[:, k]) for k in range(len(nerve))])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(eng.T)
    r = np.where(np.isfinite(r), r, 0.0)  # a zero-variance contact correlates with nothing

    out: dict[str, ContactQuality] = {}
    for k, c in enumerate(nerve):
        assert c.cuff_id is not None and c.contact_index is not None
        others = [j for j in range(len(nerve)) if j != k]
        peers = [j for j in others if nerve[j].cuff_id == c.cuff_id]
        peers.sort(key=lambda j: nerve[j].contact_index or 0)
        flat = float(np.mean(np.diff(raw[:, k]) == 0)) if raw.shape[0] > 1 else 1.0
        identical = tuple(labels[j] for j in others if np.array_equal(raw[:, k], raw[:, j]))
        j_best = max(others, key=lambda j: abs(r[k, j])) if others else k
        best = (labels[j_best], float(r[k, j_best])) if others else ("", 0.0)
        r_peers = tuple(float(r[k, j]) for j in peers)
        reasons: list[Reason] = []
        if sig[k] < FLAT_SIGMA_UV or flat > FLAT_FRACTION:
            reasons.append("flat")
        if np.sqrt(max(0.0, 1.0 - best[1] ** 2)) < DUPLICATE_RESIDUAL:
            reasons.append("duplicate")
        if r_peers and max(r_peers) < UNCORRELATED_R:
            reasons.append("uncorrelated")
        out[labels[k]] = ContactQuality(
            label=labels[k], cuff_id=c.cuff_id, contact_index=int(c.contact_index),
            sigma_uv=float(sig[k]), flat_fraction=flat, r_peers=r_peers, best_other=best,
            identical_to=identical, reasons=tuple(reasons),
        )
    return out


def screened_signals(quality: dict[str, ContactQuality]) -> frozenset[str]:
    """Return the derived signals a failed contact makes untrustworthy: its ``V``, its ``T``.

    ``T`` goes with it because a tripole formed from a failed contact is not a
    tripole - it keeps the common mode the tripole exists to remove. Untrusted is
    not unread: these stay in detection's max (invariant 43).
    """
    drop: set[str] = set()
    for q in quality.values():
        if q.screened:
            drop |= {f"{q.cuff_id}_V{q.contact_index}", f"{q.cuff_id}_T"}
    return frozenset(drop)


def velocity_cuffs(quality: dict[str, ContactQuality]) -> tuple[str, ...]:
    """Cuffs whose contacts 1, 2 and 3 all pass - the ones velocity can use."""
    cuffs = sorted({q.cuff_id for q in quality.values()})
    return tuple(
        cuff for cuff in cuffs
        if {q.contact_index for q in quality.values() if q.cuff_id == cuff and not q.screened}
        >= {1, 2, 3}
    )


# ---------------------------------------------------------------------------
# stomach contacts: report only (ruling 2026-09-29)
# ---------------------------------------------------------------------------

MAINS_HZ: Final = (60.0, 120.0, 180.0)
"""The mains fundamental and the harmonics the hum rule sums, +/- 1 Hz each."""

MAINS_SHARE_BAND_HZ: Final = (1.0, 1000.0)
"""The band ``mains_share`` is a fraction of."""

HUM_RATIO: Final = 10.0
"""A stomach contact whose mains power exceeds this many times the median of its
peers' is hum-dominated. On the 70-region pool the ratio's p90 was 2.4 and the
values ran continuously to 7.6, then jumped to 13.7-88 (B's ANT1, I's ANT2)."""

StomachReason = Literal["flat", "duplicate", "mains_hum"]


@dataclass(frozen=True, slots=True)
class StomachQuality:
    """One stomach contact's health, for the contact-health report only."""

    label: str
    sigma_uv: float
    """ENG-band robust sigma, microvolts."""
    flat_fraction: float
    copy_residual: float
    """``sqrt(1 - r**2)`` against the most correlated other stomach contact."""
    mains_share: float
    """Mains power as a fraction of 1-1000 Hz power."""
    mains_over_peers: float
    """Mains power over the median of the other stomach contacts' mains power."""
    reasons: tuple[StomachReason, ...]


def assess_stomach_contacts(
    rec: Recording, *, start_s: float = 0.0, window_s: float = SCREEN_WINDOW_S
) -> dict[str, StomachQuality]:
    """Health of every stomach contact over ``[start_s, start_s + window_s)``.

    ``flat`` and ``duplicate`` are the nerve rules; ``mains_hum`` fires when a
    contact's mains power exceeds :data:`HUM_RATIO` times its peers' median. Never
    used by detection (ruling 2026-09-29): it hid ``stomach_ref`` and cost five
    round-1 marks when tried there.
    """
    fs = float(rec.fs)
    stom = [c for c in rec.channels if c.role == "stomach"]
    if not stom:
        return {}
    i0 = max(round(start_s * fs), 0)
    i1 = min(i0 + round(window_s * fs), rec.data.shape[0])
    if i1 - i0 < fs:
        msg = f"stomach health window [{start_s}, {start_s + window_s}) s holds < 1 s of data"
        raise ValueError(msg)
    raw = np.column_stack([np.asarray(rec.data[i0:i1, c.index], dtype=np.float64) for c in stom])
    raw = raw[np.isfinite(raw).all(axis=1)]
    eng = sosfiltfilt(butter(4, SCREEN_BAND_HZ, btype="bandpass", fs=fs, output="sos"),
                      raw, axis=0)
    f, p = welch(raw - raw.mean(axis=0), fs=fs, nperseg=min(int(4 * fs), raw.shape[0]), axis=0)
    mains = np.array([sum(float(p[(f > h - 1) & (f < h + 1), k].sum()) for h in MAINS_HZ)
                      for k in range(raw.shape[1])])
    lo, hi = MAINS_SHARE_BAND_HZ
    total = np.array([float(p[(f > lo) & (f < hi), k].sum()) for k in range(raw.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(eng.T)
    r = np.where(np.isfinite(r), r, 0.0)
    out: dict[str, StomachQuality] = {}
    for k, c in enumerate(stom):
        others = [j for j in range(len(stom)) if j != k]
        peer = float(np.median(mains[others])) if others else float("nan")
        best = max((abs(float(r[k, j])) for j in others), default=0.0)
        sigma = _robust_sigma(eng[:, k])
        flat = float(np.mean(np.diff(raw[:, k]) == 0)) if raw.shape[0] > 1 else 1.0
        residual = float(np.sqrt(max(0.0, 1.0 - best ** 2)))
        over = float(mains[k] / peer) if peer > 0 else float("inf")
        reasons: list[StomachReason] = []
        if sigma < FLAT_SIGMA_UV or flat > FLAT_FRACTION:
            reasons.append("flat")
        if others and residual < DUPLICATE_RESIDUAL:
            reasons.append("duplicate")
        if others and over > HUM_RATIO:
            reasons.append("mains_hum")
        out[c.name] = StomachQuality(
            label=c.name, sigma_uv=sigma, flat_fraction=flat, copy_residual=residual,
            mains_share=float(mains[k] / total[k]) if total[k] > 0 else float("nan"),
            mains_over_peers=over, reasons=tuple(reasons))
    return out
