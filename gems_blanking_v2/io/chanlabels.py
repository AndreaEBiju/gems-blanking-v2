"""Channel order comes from the file. ``meta.json`` never restates it.

The recording carries ``chanlabels`` and they are authoritative. A ``meta.json``
that also lists the order is a second source of truth that can disagree with the
first, and when it disagrees nothing complains - it simply mislabels every
channel. That is not hypothetical: a nine-channel fixture built its cuffs in the
order ``("L", "R")`` and was used against a file whose labels run
``RVN1..RVN3, LVN1..LVN3``, so column 0 was read as ``LVN1`` when the file says
``RVN1``. Every left/right comparison downstream would have been inverted, and
nothing in the pipeline would have noticed.

So ``meta.json`` declares only what the file does **not** state - units, geometry
(``rostral_end``, ``config``), identity - and the order is read from the file
each time. :func:`assert_labels_match` is the guard that keeps the two from
drifting apart again.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np

from gems_blanking_v2.types import ChannelInfo

__all__ = [
    "LABEL_PATTERN",
    "SIGMA_PLAUSIBLE_UV",
    "assert_labels_match",
    "assert_plausible_units",
    "channels_from_labels",
    "read_channel_labels",
]

LABEL_PATTERN: Final = re.compile(r"^(?P<stem>[A-Za-z]+)(?P<idx>\d*)$")
"""``RVN1`` -> stem ``RVN``, index 1. ``ANT2`` -> ``ANT``, 2."""

_NERVE_STEMS: Final = {"LVN": "L", "RVN": "R"}
"""Vagus cuff stems and the cuff they belong to."""

_STOMACH_STEMS: Final = {"ANT", "STOM"}
"""Stomach contact stems. ``ANT`` is what this cohort's export writes."""


def read_channel_labels(path: Path) -> tuple[str, ...] | None:
    """Return the file's ``chanlabels``, or ``None`` when it has none.

    ``None`` is a real answer, not a failure: the old five-channel cohort's files
    carry no label table, which is exactly why ``meta.json`` exists. The caller
    decides whether an absent table is acceptable; this function does not guess.
    """
    path = Path(path)
    if path.suffix.lower() != ".mat":
        return _read_h5_labels(path)
    try:
        from scipy.io import loadmat  # noqa: PLC0415

        raw = loadmat(path, variable_names=["chanlabels"])
    except (NotImplementedError, ValueError, OSError):
        return _read_h5_labels(path)
    if "chanlabels" not in raw:
        return None
    flat = np.asarray(raw["chanlabels"]).ravel()
    out = [str(np.asarray(c).ravel()[0]).strip() for c in flat]
    return tuple(out) if out else None


def _read_h5_labels(path: Path) -> tuple[str, ...] | None:
    """Read labels from the v7.3 / flat-HDF5 form, a dataset of byte strings."""
    try:
        import h5py  # noqa: PLC0415

        with h5py.File(path, "r") as f:
            if "chanlabels" not in f:
                return None
            arr = f["chanlabels"][...]
    except (OSError, KeyError):
        return None
    out: list[str] = []
    for item in np.asarray(arr).ravel():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", "replace").strip())
        else:
            out.append(str(item).strip())
    return tuple(out) if out else None


def channels_from_labels(
    labels: tuple[str, ...],
    *,
    rostral_end: int | None,
    config: str,
) -> list[ChannelInfo]:
    """Build the channel table from the file's own labels, in file order.

    Roles and cuff membership are parsed from the label text because the label is
    what the person wiring the animal wrote down. ``rostral_end`` and ``config``
    cannot be parsed from a label and are passed in - they are precisely the
    things ``meta.json`` exists to declare.

    Raises
    ------
    ValueError
        On a label this does not recognise, or a ``config`` outside
        ``{"hw_tripole", "independent"}``. Guessing a role would put a stomach
        contact in a nerve cuff, and the tripole derivation would then average
        the stomach into the nerve.
    """
    if config not in ("hw_tripole", "independent"):
        msg = f"config {config!r} is neither 'hw_tripole' nor 'independent'"
        raise ValueError(msg)
    cfg = cast(Literal["hw_tripole", "independent"], config)
    out: list[ChannelInfo] = []
    for index, raw in enumerate(labels):
        m = LABEL_PATTERN.match(raw)
        if m is None:
            msg = f"channel label {raw!r} (column {index}) is not of the form NAME[n]"
            raise ValueError(msg)
        stem = m.group("stem").upper()
        idx = int(m.group("idx")) if m.group("idx") else None
        if stem in _NERVE_STEMS:
            out.append(ChannelInfo(
                index=index, name=raw, role="nerve", cuff_id=_NERVE_STEMS[stem],
                contact_index=idx, rostral_end=rostral_end, config=cfg,
            ))
        elif stem in _STOMACH_STEMS:
            out.append(ChannelInfo(
                index=index, name=raw, role="stomach", cuff_id=None,
                contact_index=None, rostral_end=None, config=cfg,
            ))
        else:
            msg = (
                f"channel label {raw!r} (column {index}) has unknown stem {stem!r}; "
                f"known: {sorted(_NERVE_STEMS) + sorted(_STOMACH_STEMS)}. Add it "
                "deliberately rather than letting it default to a role - a stomach "
                "contact placed in a nerve cuff would be averaged into the tripole."
            )
            raise ValueError(msg)
    return out


def assert_labels_match(
    declared: list[ChannelInfo], labels: tuple[str, ...] | None, where: str
) -> None:
    """Raise if a declared table disagrees with the file's labels.

    Compared **in order and by name**, case-insensitively (cross-platform rule 7).
    A mismatch means the map describes a different recording, or the same
    recording with the cuffs transposed, and both mislabel every downstream
    result silently.
    """
    if labels is None:
        return
    got = [c.name for c in declared]
    if len(got) != len(labels):
        msg = (
            f"{where}: map has {len(got)} channels but the file's chanlabels has "
            f"{len(labels)}: {got} vs {list(labels)}"
        )
        raise ValueError(msg)
    bad = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(got, labels, strict=True))
        if a.casefold() != b.casefold()
    ]
    if bad:
        detail = "; ".join(f"column {i}: map {a!r} vs file {b!r}" for i, a, b in bad)
        msg = (
            f"{where}: the channel map disagrees with the file's chanlabels. "
            f"{detail}. The file is authoritative - it is what the person wiring "
            "the animal recorded. A map that transposes two cuffs inverts every "
            "left/right result and nothing downstream would notice."
        )
        raise ValueError(msg)


MIN_SAMPLES_FOR_SIGMA: Final = 2
"""Fewer than two finite samples gives no spread to judge."""

SIGMA_PLAUSIBLE_UV: Final = (1.0, 500.0)
"""Robust sigma a nerve or stomach contact can credibly have, microvolts.

Not an inference - the unit stays declared (invariant 14). This only catches a
declaration that cannot be true. The window is wide on purpose: a quiet tripole
sits near 3-20 uV and a noisy stomach contact near 40 uV, so 1-500 rejects
nothing real while catching the failure that matters. A volts-vs-microvolts
mix-up is a factor of 10^6, which lands six orders outside either end.

Measured on the two cohorts in hand: old-cohort contacts 10-41 uV, new-cohort
8-10 uV. A declaration putting them at 8 pV or 8 V is a typo, not a recording.
"""


def assert_plausible_units(
    data: np.ndarray, units: str, scale_uv: float, where: str
) -> float:
    """Raise if the declared unit implies an impossible signal amplitude.

    Returns the median robust sigma in microvolts, so the caller can record it
    (invariant 26). This function already computes it on every load; a separate
    sweep to obtain the same number would be a second pass over the same bytes,
    and the distribution accumulates for free from passes that had to happen.

    Checks the MEDIAN across channels rather than any single one, so a dead or
    saturated contact cannot veto an otherwise sound declaration - and cannot
    rescue a wrong one either.
    """
    if data.size == 0:
        return float("nan")
    sigmas = []
    for col in range(data.shape[1]):
        x = np.asarray(data[:, col], dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < MIN_SAMPLES_FOR_SIGMA:
            continue
        sigmas.append(1.4826 * float(np.median(np.abs(x - np.median(x)))) * scale_uv)
    if not sigmas:
        return float("nan")
    med = float(np.median(sigmas))
    lo, hi = SIGMA_PLAUSIBLE_UV
    if lo <= med <= hi:
        return med
    factor = med / hi if med > hi else lo / med
    msg = (
        f"{where}: units declared as {units!r} put the median robust sigma at "
        f"{med:.4g} uV, outside the plausible {lo:g}-{hi:g} uV for a nerve or "
        f"stomach contact - off by about {factor:.3g}x. This does not infer the "
        "unit; it refuses a declaration that cannot be true. Check the "
        "declaration: a volts-vs-microvolts mix-up is a factor of 1e6 and looks "
        "like a plausible signal at every later step."
    )
    raise ValueError(msg)
