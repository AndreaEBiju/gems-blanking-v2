"""How ``stomach_ref`` was built, as a record a reader in another language can apply.

Andrea, 2026-10-12: Night 6 builds ``stomach_ref`` from the recipe the mask provenance
records, never from its own idea of the derivation, and never moves a stomach_ref mask onto
a nerve signal. This module makes that record - the ``stomach_ref_recipe`` block of a mask
file's provenance - and is its executable specification:

* :func:`stomach_ref_recipe` reads the recipe OFF the code that makes the signal.
  ``how`` is what :func:`~gems_blanking_v2.derive.derivations.build_stomach_reference`
  returns; the weights are that function's response to a unit impulse on each stomach
  contact (it is linear, so its impulse responses ARE its weights); the notch is the
  routing entry's ``stomach_ref`` field, applied with
  :func:`~gems_blanking_v2.derive.stomach_alt.notch`. Before it returns, the recipe is
  applied to the recording itself (:func:`apply_stomach_ref_recipe`) and must equal the
  derivation's output **bit for bit**, or it raises. So a recipe that disagrees with the
  computation cannot be written: the two cannot diverge silently (invariants 28, 33).
* :func:`proven_stomach_ref_recipe` is what a finalise driver calls (it loads no
  recording): it returns the recipe ``stomach_ref_proof.py`` proved and stored for the
  session (:func:`proven_recipe_path`), after re-checking its proof, and refuses by name
  when the record is absent or its proof does not check.
* :func:`apply_stomach_ref_recipe` applies a recorded recipe in the derivation's own
  order of operations - for ``positive_minus_nanmean``, the positive contact minus the
  NaN-omitting mean of the contacts, the sum taken left to right ``(c1 + c2) + c3`` and
  divided by the count of finite contacts, which is what ``numpy.nanmean`` does over the
  stacked columns. ``matlab/night6/night6_apply_recipe.m`` is its MATLAB twin; the
  explicit ``weights`` are recorded for the reader and checked, but a dot product with
  them is NOT bit-identical to the derivation (it rounds differently, within ~2 ulp of the
  largest contact), so neither side applies them as a dot product.

The recipe is in the recording's own units: the construction is linear and homogeneous,
so it commutes with the declared unit scale up to rounding.

OUTSIDE THE GENERATION HASH (nothing in the candidate chain imports this module).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from gems_blanking_v2.derive.derivations import build_stomach_reference
from gems_blanking_v2.derive.stomach_alt import NOTCH_Q, notch, notched_stomach_ref
from gems_blanking_v2.types import Recording

__all__ = [
    "FORMS",
    "NO_NOTCH",
    "PROVEN_NAME",
    "PROVEN_SCHEMA",
    "RECIPE_KEYS",
    "RECIPE_SCHEMA",
    "STOMACH_REF",
    "StomachRecipeError",
    "apply_stomach_ref_recipe",
    "implied_weights",
    "proven_recipe_path",
    "proven_recipe_record",
    "proven_stomach_ref_recipe",
    "stomach_ref_recipe",
    "validate_stomach_ref_recipe",
]

F64 = npt.NDArray[np.float64]

STOMACH_REF: Final = "stomach_ref"
"""The derived signal this recipe builds (``constants.STOMACH_SIGNALS``)."""

RECIPE_SCHEMA: Final = "stomach_ref_recipe/1"
"""The block's format; a reader refuses any other."""

NO_NOTCH: Final = "none"
"""``notch`` when the routing entry names no notch (``stomach_ref: {}``, the derivation
as-is). Explicit, so an absent notch is never read as "none" (invariant 8)."""

FORMS: Final[Mapping[str, str]] = {"common_average": "positive_minus_nanmean",
                                   "hardware": "passthrough",
                                   "single_channel": "passthrough"}
"""``how`` (``derivations.StomachReference``) -> the construction applied."""

RECIPE_KEYS: Final[tuple[str, ...]] = ("schema", "how", "config", "contacts", "positive",
                                       "form", "weights", "notch", "routing_stomach_ref",
                                       "source")
"""Every key of the block; each is required."""

_ROUTING_KEYS: Final = frozenset({"notch_hz", "contacts"})
_SOURCE: Final = ("gems_blanking_v2.derive.derivations.build_stomach_reference (weights = its "
                  "unit-impulse response per contact; recipe verified bit for bit on the "
                  "recording by derive.stomach_recipe.stomach_ref_recipe)")


class StomachRecipeError(ValueError):
    """A recipe that is incomplete, inconsistent, or disagrees with the derivation."""


def implied_weights(form: str, n_contacts: int, positive_index: int) -> list[float]:
    """Return the weights ``form`` implies, computed as the derivation computes them.

    ``positive_minus_nanmean``: ``(1 if k == positive else 0) - 1 / n`` - exactly the
    derivation's output for a unit impulse on contact ``k`` (``x_pos - (sum) / n``).
    ``passthrough``: ``[1.0]`` (one contact).
    """
    if form == "positive_minus_nanmean":
        return [(1.0 if k == positive_index else 0.0) - 1.0 / n_contacts
                for k in range(n_contacts)]
    if form == "passthrough":
        return [1.0]
    msg = f"stomach_ref_recipe form {form!r} is not one of {sorted(set(FORMS.values()))}"
    raise StomachRecipeError(msg)


def _notch_record(routing: Mapping[str, Any], contacts: Sequence[str]) -> str | dict[str, Any]:
    """Return the notch the routing entry's ``stomach_ref`` field names, or :data:`NO_NOTCH`."""
    extra = sorted(set(routing) - _ROUTING_KEYS)
    if extra:
        msg = f"routing stomach_ref has unknown field(s) {extra}; its variant is unknown"
        raise StomachRecipeError(msg)
    if not routing:
        return NO_NOTCH
    if set(routing) != _ROUTING_KEYS:
        msg = (f"routing stomach_ref {dict(routing)} must name both notch_hz and contacts, "
               "or neither ({} = the derivation as-is)")
        raise StomachRecipeError(msg)
    freqs = [float(f) for f in routing["notch_hz"]]
    which = [str(c) for c in routing["contacts"]]
    if not freqs or not which:
        msg = f"routing stomach_ref {dict(routing)} names an empty notch_hz or contacts"
        raise StomachRecipeError(msg)
    unknown = sorted(set(which) - set(contacts))
    if unknown:
        msg = (f"routing stomach_ref notches {unknown}, which are not stomach contacts "
               f"{list(contacts)}")
        raise StomachRecipeError(msg)
    return {"freqs_hz": freqs, "q": float(NOTCH_Q), "contacts": which,
            "filter": "derive.stomach_alt.notch: iirnotch per frequency, tf2sos, sosfiltfilt, "
                      "on the whole loaded recording before the reference; NaN interpolated "
                      "for the filter and NaN again after"}


def stomach_ref_recipe(rec: Recording, routing_stomach_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``stomach_ref_recipe`` block for ``rec``, verified on ``rec`` itself.

    ``routing_stomach_ref`` is the recording's routing entry's ``stomach_ref`` field
    (``emit.routing``: ``{}`` = the derivation as-is, or ``{notch_hz, contacts}``).
    ``rec`` is the recording exactly as the masks' detection loaded it (any slice of it:
    the construction is per sample, but a notch is not, so pass what was notched).

    Raises :class:`StomachRecipeError` when ``rec`` has no stomach, when the routing field
    is not a known variant, when the derivation is not the construction its ``how`` names
    (its impulse responses are not :func:`implied_weights`), or when the recipe applied to
    ``rec`` differs from the derivation's output in any sample.
    """
    config = rec.channels[0].config if rec.channels else "independent"  # build_derivations'
    stomach = [c for c in rec.channels if c.role == "stomach"]
    if not stomach:
        msg = f"{rec.session}: no stomach channel, so there is no stomach_ref and no recipe"
        raise StomachRecipeError(msg)
    names = [c.name for c in stomach]
    # the derivation's response to a unit impulse on each stomach contact, plus a zero row
    probe = np.zeros((len(stomach) + 1, rec.data.shape[1]), dtype=np.float64)
    for k, c in enumerate(stomach):
        probe[k, c.index] = 1.0
    response, how = build_stomach_reference(replace(rec, data=probe), config)
    if response is None or how is None:
        msg = f"{rec.session}: build_stomach_reference made no stomach_ref"
        raise StomachRecipeError(msg)
    if how not in FORMS:
        msg = f"{rec.session}: build_stomach_reference says how={how!r}, which has no recipe form"
        raise StomachRecipeError(msg)
    form = FORMS[how]
    contacts = names if form == "positive_minus_nanmean" else names[:1]
    weights = implied_weights(form, len(contacts), 0)
    measured = [float(v) for v in response[:len(contacts)]]
    rest = [float(v) for v in response[len(contacts):]]
    if measured != weights or any(v != 0.0 for v in rest):
        msg = (f"{rec.session}: build_stomach_reference's impulse responses {measured} + {rest} "
               f"are not the {form} weights {weights}: the derivation changed, so this recipe "
               "would not describe it")
        raise StomachRecipeError(msg)
    recipe: dict[str, Any] = {
        "schema": RECIPE_SCHEMA, "how": how, "config": config, "contacts": contacts,
        "positive": contacts[0], "form": form, "weights": weights,
        "notch": _notch_record(routing_stomach_ref, names),
        "routing_stomach_ref": {k: list(v) for k, v in routing_stomach_ref.items()},
        "source": _SOURCE,
    }
    validate_stomach_ref_recipe(recipe)
    if recipe["notch"] == NO_NOTCH:
        want, _how = build_stomach_reference(rec, config)
    else:
        want = notched_stomach_ref(rec, recipe["notch"]["freqs_hz"],
                                   recipe["notch"]["contacts"])
    got = apply_stomach_ref_recipe(recipe, {c.name: rec.data[:, c.index] for c in stomach},
                                   float(rec.fs))
    if want is None or not np.array_equal(got, want, equal_nan=True):
        bad = -1 if want is None else int(np.count_nonzero(~((got == want) | (
            np.isnan(got) & np.isnan(want)))))
        msg = (f"{rec.session}: the recipe applied to the recording differs from the "
               f"derivation at {bad} sample(s); the recipe is not written")
        raise StomachRecipeError(msg)
    return recipe


def validate_stomach_ref_recipe(recipe: Mapping[str, Any]) -> None:
    """Raise :class:`StomachRecipeError` naming the field unless ``recipe`` is complete.

    Every key of :data:`RECIPE_KEYS` must be present (absent and ``null`` alike are
    refused); ``form`` must be the one ``how`` maps to; ``positive`` one of ``contacts``
    (the first, as the derivation takes it); ``weights`` exactly :func:`implied_weights`;
    ``notch`` either :data:`NO_NOTCH` or a record of frequencies, Q and contacts.
    """
    missing = [k for k in RECIPE_KEYS if recipe.get(k) is None]
    if missing:
        msg = f"stomach_ref_recipe lacks {', '.join(missing)}"
        raise StomachRecipeError(msg)
    if recipe["schema"] != RECIPE_SCHEMA:
        msg = f"stomach_ref_recipe schema {recipe['schema']!r} is not {RECIPE_SCHEMA!r}"
        raise StomachRecipeError(msg)
    how = recipe["how"]
    if FORMS.get(how) != recipe["form"]:
        msg = f"stomach_ref_recipe how {how!r} does not make form {recipe['form']!r}"
        raise StomachRecipeError(msg)
    contacts = recipe["contacts"]
    if (not isinstance(contacts, list) or not contacts
            or not all(isinstance(c, str) and c for c in contacts)
            or len(set(contacts)) != len(contacts)):
        msg = f"stomach_ref_recipe contacts {contacts!r} must be distinct non-empty names"
        raise StomachRecipeError(msg)
    if recipe["form"] == "passthrough" and len(contacts) != 1:
        msg = f"a passthrough stomach_ref reads one contact, not {contacts}"
        raise StomachRecipeError(msg)
    if recipe["positive"] != contacts[0]:
        msg = (f"stomach_ref_recipe positive {recipe['positive']!r} is not the first contact "
               f"{contacts[0]!r}, which is the one the derivation takes")
        raise StomachRecipeError(msg)
    weights = recipe["weights"]
    want = implied_weights(recipe["form"], len(contacts), 0)
    if (not isinstance(weights, list) or len(weights) != len(want)
            or any(not isinstance(w, int | float) or float(w) != v
                   for w, v in zip(weights, want, strict=True))):
        msg = f"stomach_ref_recipe weights {weights!r} are not the {recipe['form']} weights {want}"
        raise StomachRecipeError(msg)
    if recipe["config"] not in ("independent", "hw_tripole"):
        msg = f"stomach_ref_recipe config {recipe['config']!r} is not a known electrode config"
        raise StomachRecipeError(msg)
    _validate_notch(recipe["notch"], contacts)
    if not isinstance(recipe["routing_stomach_ref"], Mapping):
        msg = "stomach_ref_recipe routing_stomach_ref must be the routing entry's field"
        raise StomachRecipeError(msg)


def _validate_notch(nt: object, contacts: Sequence[str]) -> None:
    if nt == NO_NOTCH:
        return
    if not isinstance(nt, Mapping):
        msg = f"stomach_ref_recipe notch {nt!r} is neither {NO_NOTCH!r} nor a notch record"
        raise StomachRecipeError(msg)
    lacking = [k for k in ("freqs_hz", "q", "contacts") if nt.get(k) is None]
    if lacking:
        msg = f"stomach_ref_recipe notch lacks {', '.join(lacking)}"
        raise StomachRecipeError(msg)
    freqs, q, which = nt["freqs_hz"], nt["q"], nt["contacts"]
    if (not isinstance(freqs, list) or not freqs
            or not all(isinstance(f, int | float) and math.isfinite(f) and f > 0 for f in freqs)):
        msg = f"stomach_ref_recipe notch freqs_hz {freqs!r} must be positive frequencies"
        raise StomachRecipeError(msg)
    if not isinstance(q, int | float) or not math.isfinite(q) or q <= 0:
        msg = f"stomach_ref_recipe notch q {q!r} must be a positive number"
        raise StomachRecipeError(msg)
    if not isinstance(which, list) or not which or not set(which) <= set(contacts):
        msg = f"stomach_ref_recipe notch contacts {which!r} must be recipe contacts {contacts}"
        raise StomachRecipeError(msg)


def apply_stomach_ref_recipe(recipe: Mapping[str, Any], columns: Mapping[str, npt.ArrayLike],
                             fs: float) -> F64:
    """Apply a recorded recipe to ``columns`` (contact name -> samples, recording units).

    The derivation's order of operations, so the result equals it bit for bit (see the
    module docstring); a notch is applied to the named contacts first, over the samples
    given. Raises naming a contact that ``columns`` lacks.
    """
    validate_stomach_ref_recipe(recipe)
    contacts = list(recipe["contacts"])
    lacking = [c for c in contacts if c not in columns]
    if lacking:
        msg = f"stomach_ref_recipe reads {lacking}, which the recording does not have"
        raise StomachRecipeError(msg)
    cols = {c: np.asarray(columns[c], dtype=np.float64) for c in contacts}
    nt = recipe["notch"]
    if nt != NO_NOTCH:
        for c in nt["contacts"]:
            cols[c] = notch(cols[c], float(fs), [float(f) for f in nt["freqs_hz"]], float(nt["q"]))
    stack = np.column_stack([cols[c] for c in contacts])
    if recipe["form"] == "passthrough":
        return np.asarray(stack[:, 0], dtype=np.float64)
    pos = contacts.index(recipe["positive"])
    return np.asarray(stack[:, pos] - np.nanmean(stack, axis=1), dtype=np.float64)

# ---------------------------------------------------------------------------
# the proven recipe: written once per recording by stomach_ref_proof.py, read at finalise
# ---------------------------------------------------------------------------

PROVEN_SCHEMA: Final = "stomach_ref_recipe_proven/1"
"""The proven-recipe record's format; :func:`proven_stomach_ref_recipe` refuses any other."""

PROVEN_NAME: Final = "stomach_ref_recipe_proven.json"
"""The record's file name, in ``data/<animal>/<session>/`` beside ``meta.json``."""

_REF_FIELDS: Final = ("median", "scale", "n_frames", "statistic")


def proven_recipe_path(root: Path, animal: str, session: str) -> Path:
    """Return where one session's proven recipe lives under ``root`` (the one site)."""
    return Path(root) / "data" / animal / session / PROVEN_NAME


def proven_recipe_record(session: str, recipe: Mapping[str, Any],
                         proofs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the record ``stomach_ref_proof.py`` writes once a recording is proved.

    ``proofs`` maps each epoch file to the proof block that established ``recipe`` on it
    (expected vs recomputed reference values, routing entry hash, generation sha). The
    record is checked as :func:`proven_stomach_ref_recipe` will check it, so a record that
    would be refused is never written.
    """
    rec = {"schema": PROVEN_SCHEMA, "recording": session, "recipe": dict(recipe),
           "proofs": {k: dict(v) for k, v in proofs.items()}}
    _check_proven(rec, session, entry_hash_want=None, generation_sha=None)
    return rec


def proven_stomach_ref_recipe(root: Path, animal: str, session: str, *,
                              signals: Mapping[str, Sequence[str]],
                              routing_entry: Mapping[str, Any],
                              generation_sha: str) -> dict[str, Any]:
    """Return ``MaskProvenance.stomach_ref_recipe`` for one recording at finalise.

    Finalise loads no recording, so it never BUILDS the recipe: it reads the one
    ``stomach_ref_proof.py`` proved and stored (:func:`proven_recipe_path` under ``root`` -
    the proof run's ``--out``, or the gems root once the store writer has placed it) and
    re-checks its proof. Returns ``{}`` - no file is read - when no consumer in ``signals``
    reads ``stomach_ref``. Otherwise raises :class:`StomachRecipeError`, naming what failed,
    when the record is absent or unreadable, is for another recording, its recipe is
    incomplete, or its proof does not check: every epoch's recomputed reference values must
    equal the recorded ones exactly (median, scale, frame count, statistic), its routing
    entry hash must be the one recorded AND the hash of ``routing_entry`` (the entry this
    finalise runs under), its generation sha must be the masks' and the code's AND
    ``generation_sha``, and the recipe's routing variant must be ``routing_entry``'s.

    The finalise driver::

        recipe = proven_stomach_ref_recipe(root, animal, session, signals=signals,
                                           routing_entry=entry, generation_sha=GENERATION)
        prov = MaskProvenance(..., stomach_ref_recipe=recipe)   # every copy carries it
        write_mask_file(path, masks, prov, signals=signals, ...)
    """
    if not any(STOMACH_REF in v for v in signals.values()):
        return {}
    path = proven_recipe_path(root, animal, session)
    try:
        rec = json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError as exc:
        msg = (f"{session}: a consumer reads {STOMACH_REF} but there is no proven recipe at "
               f"{path} (stomach_ref_proof.py writes it once the recording is proved)")
        raise StomachRecipeError(msg) from exc
    except (OSError, ValueError) as exc:
        msg = f"{session}: the proven recipe at {path} cannot be read: {exc}"
        raise StomachRecipeError(msg) from exc
    if "stomach_ref" not in routing_entry:
        msg = f"{session}: the routing entry has no stomach_ref field, so its variant is unknown"
        raise StomachRecipeError(msg)
    from gems_blanking_v2.emit.routing import entry_hash  # noqa: PLC0415 - layer: emit
    _check_proven(rec, session, entry_hash_want=entry_hash(routing_entry),
                  generation_sha=str(generation_sha))
    if rec["recipe"]["routing_stomach_ref"] != dict(routing_entry["stomach_ref"]):
        msg = (f"{session}: the proven recipe is for routing stomach_ref "
               f"{rec['recipe']['routing_stomach_ref']}, the entry names "
               f"{dict(routing_entry['stomach_ref'])}")
        raise StomachRecipeError(msg)
    return dict(rec["recipe"])


def _check_proven(rec: Mapping[str, Any], session: str, *, entry_hash_want: str | None,
                  generation_sha: str | None) -> None:
    """Raise naming the first thing about a proven-recipe record that does not check."""
    if not isinstance(rec, Mapping) or rec.get("schema") != PROVEN_SCHEMA:
        msg = f"{session}: the proven recipe's schema is not {PROVEN_SCHEMA!r}"
        raise StomachRecipeError(msg)
    if rec.get("recording") != session:
        msg = f"{session}: the proven recipe is for {rec.get('recording')!r}"
        raise StomachRecipeError(msg)
    recipe = rec.get("recipe")
    if not isinstance(recipe, Mapping):
        msg = f"{session}: the proven recipe record carries no recipe"
        raise StomachRecipeError(msg)
    validate_stomach_ref_recipe(recipe)
    proofs = rec.get("proofs")
    if not isinstance(proofs, Mapping) or not proofs:
        msg = f"{session}: the proven recipe carries no proof"
        raise StomachRecipeError(msg)
    for epoch, proof in proofs.items():
        _check_proof(f"{session} {epoch}", proof, entry_hash_want, generation_sha)


def _check_proof(where: str, proof: Mapping[str, Any], entry_hash_want: str | None,
                 generation_sha: str | None) -> None:
    expected, got = proof.get("expected"), proof.get("recomputed")
    if not isinstance(expected, Mapping) or not expected or not isinstance(got, Mapping):
        msg = f"{where}: the proof has no expected/recomputed reference values"
        raise StomachRecipeError(msg)
    if sorted(expected) != sorted(got):
        msg = f"{where}: the proof recomputed {sorted(got)}, not {sorted(expected)}"
        raise StomachRecipeError(msg)
    for key, e in expected.items():
        if not key.startswith(f"{STOMACH_REF}|"):
            msg = f"{where}: the proof checks {key}, not a {STOMACH_REF} reference"
            raise StomachRecipeError(msg)
        for f in _REF_FIELDS:
            if e.get(f) is None or e.get(f) != got[key].get(f):
                msg = (f"{where}: proof {key} {f}: recorded {e.get(f)!r}, recomputed "
                       f"{got[key].get(f)!r}")
                raise StomachRecipeError(msg)
    if proof.get("match") is not True or proof.get("mismatches"):
        msg = f"{where}: the proof does not say match (mismatches {proof.get('mismatches')})"
        raise StomachRecipeError(msg)
    eh = proof.get("routing_entry_hash") or {}
    if not eh.get("provenance") or eh.get("provenance") != eh.get("recomputed"):
        msg = (f"{where}: the proof's routing entry hash {eh.get('recomputed')!r} is not the "
               f"masks' {eh.get('provenance')!r}")
        raise StomachRecipeError(msg)
    if entry_hash_want is not None and eh["provenance"] != entry_hash_want:
        msg = (f"{where}: the proof is for routing entry {eh['provenance']}, this finalise "
               f"runs under {entry_hash_want}")
        raise StomachRecipeError(msg)
    gs = proof.get("generation_sha") or {}
    if not gs.get("provenance") or gs.get("provenance") != gs.get("code"):
        msg = (f"{where}: the proof's generation sha {gs.get('code')!r} is not the masks' "
               f"{gs.get('provenance')!r}")
        raise StomachRecipeError(msg)
    if generation_sha is not None and gs["provenance"] != generation_sha:
        msg = (f"{where}: the proof is for generation {gs['provenance']}, this finalise "
               f"runs {generation_sha}")
        raise StomachRecipeError(msg)
