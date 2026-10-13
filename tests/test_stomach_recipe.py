"""The stomach_ref recipe (Andrea, 2026-10-12): read off the derivation, recorded, applied.

``derive.stomach_recipe.stomach_ref_recipe`` is the one builder of the ``stomach_ref_recipe``
block a mask file's provenance carries; Night 6 builds ``stomach_ref`` from that block alone
(``tests/test_night6_stomach_ref.py`` holds the MATLAB build to this one). Checked here:

* the recipe applied to the raw channels equals ``build_stomach_reference`` bit for bit -
  NaN samples included - and the notched variant equals ``notched_stomach_ref``;
* the explicit weights are the derivation's unit-impulse responses, and a dot product with
  them is NOT bit-identical (it rounds differently): it agrees to 4 ulp of the largest
  contact, which is why no reader applies the recipe as a dot product;
* the builder refuses a derivation that is not the construction it records, and every
  incomplete or inconsistent block is refused by name;
* the recipe is a field of ``MaskProvenance`` only (review 2026-10-12): ``write_mask_file``
  requires it there when a consumer reads ``stomach_ref``, refuses an incomplete one, and a
  sidecar written from the same provenance object (the driver's ``{stem}_provenance.json``)
  carries the identical recipe. The finalise driver's reader of the PROVEN recipe
  (``proven_stomach_ref_recipe``) is tested with the proof script that writes it
  (``tests/test_stomach_ref_proof.py``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.derive import stomach_recipe as sr
from gems_blanking_v2.derive.derivations import build_derivations, build_stomach_reference
from gems_blanking_v2.derive.stomach_alt import NOTCH_Q, notched_stomach_ref
from gems_blanking_v2.emit import handoff as ho
from gems_blanking_v2.emit import masks as mk
from gems_blanking_v2.emit.provenance import MaskProvenance
from gems_blanking_v2.extent.grid import n_grid_frames
from gems_blanking_v2.types import Recording
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.io import loadmat

from tests.conftest import make_multichannel

FS = 24414.0625
DUR_S = 4.0
AS_IS: dict[str, Any] = {}
NOTCHED = {"notch_hz": [60.0], "contacts": ["ANT1"]}


def _rec(seed: int = 3, n_stomach: int = 3) -> Recording:
    rec, _truth = make_multichannel(FS, DUR_S, n_stomach=n_stomach, seed=seed)
    return rec


def _columns(rec: Recording) -> dict[str, np.ndarray]:
    return {c.name: rec.data[:, c.index] for c in rec.channels if c.role == "stomach"}


def test_the_recipe_records_how_contacts_positive_weights_notch_and_config() -> None:
    recipe = sr.stomach_ref_recipe(_rec(), AS_IS)
    assert recipe == {
        "schema": "stomach_ref_recipe/1", "how": "common_average", "config": "independent",
        "contacts": ["ANT1", "ANT2", "ANT3"], "positive": "ANT1",
        "form": "positive_minus_nanmean", "weights": [1.0 - 1.0 / 3, -1.0 / 3, -1.0 / 3],
        "notch": "none", "routing_stomach_ref": {}, "source": recipe["source"]}
    assert "build_stomach_reference" in recipe["source"]


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_the_recorded_recipe_applied_to_the_raw_channels_is_the_derivation_exactly(
        seed: int) -> None:
    rec = _rec(seed)
    recipe = sr.stomach_ref_recipe(rec, AS_IS)
    want, _how = build_stomach_reference(rec)
    got = sr.apply_stomach_ref_recipe(recipe, _columns(rec), FS)
    assert want is not None and np.array_equal(got, want)       # bit for bit
    signals, _w = build_derivations(rec)                        # what detection read
    assert np.array_equal(got, signals["stomach_ref"])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_a_nan_sample_is_averaged_over_the_finite_contacts_as_the_derivation_does() -> None:
    rec = _rec()
    data = rec.data.copy()
    data[100, 7] = np.nan            # ANT2 at one sample: nanmean over ANT1, ANT3
    data[200, 6] = np.nan            # the positive contact: NaN out
    data[300, 6:9] = np.nan          # every contact
    bad = replace(rec, data=data)
    recipe = sr.stomach_ref_recipe(bad, AS_IS)
    want, _how = build_stomach_reference(bad)
    got = sr.apply_stomach_ref_recipe(recipe, _columns(bad), FS)
    assert want is not None and np.array_equal(got, want, equal_nan=True)
    assert np.isfinite(got[100]) and np.isnan(got[200]) and np.isnan(got[300])


def test_the_weights_are_the_derivation_but_a_dot_product_rounds_differently() -> None:
    """The explicit weights are recorded for the reader; nobody applies them as a dot product.

    Measured: the dot product differs from the derivation in the last bit on most samples,
    always within 4 ulp of the largest contact. This tolerance is the dot product's, stated
    on purpose - the recipe's own test above is exact.
    """
    rec = _rec()
    recipe = sr.stomach_ref_recipe(rec, AS_IS)
    cols = np.column_stack([_columns(rec)[c] for c in recipe["contacts"]])
    dot = cols @ np.asarray(recipe["weights"])
    want, _how = build_stomach_reference(rec)
    assert want is not None
    assert not np.array_equal(dot, want)
    ulp = np.spacing(np.max(np.abs(cols), axis=1))
    assert np.all(np.abs(dot - want) <= 4 * ulp)


def test_the_notched_variant_is_recorded_and_applied_as_notched_stomach_ref() -> None:
    rec = _rec()
    recipe = sr.stomach_ref_recipe(rec, NOTCHED)
    assert recipe["notch"]["freqs_hz"] == [60.0] and recipe["notch"]["q"] == NOTCH_Q
    assert recipe["notch"]["contacts"] == ["ANT1"]
    assert recipe["routing_stomach_ref"] == NOTCHED
    got = sr.apply_stomach_ref_recipe(recipe, _columns(rec), FS)
    assert np.array_equal(got, notched_stomach_ref(rec, [60.0], ["ANT1"]))
    plain, _how = build_stomach_reference(rec)
    assert plain is not None and not np.array_equal(got, plain)


def test_a_single_stomach_channel_and_a_hardware_stomach_are_passed_through() -> None:
    one = sr.stomach_ref_recipe(_rec(n_stomach=1), AS_IS)
    assert (one["how"], one["form"], one["contacts"], one["weights"]) == (
        "single_channel", "passthrough", ["ANT1"], [1.0])
    rec = _rec()
    hw = replace(rec, channels=[replace(c, config="hw_tripole") for c in rec.channels])
    rec_hw = sr.stomach_ref_recipe(hw, AS_IS)
    assert (rec_hw["how"], rec_hw["config"], rec_hw["contacts"]) == (
        "hardware", "hw_tripole", ["ANT1"])
    want, _how = build_stomach_reference(hw)
    assert want is not None
    assert np.array_equal(sr.apply_stomach_ref_recipe(rec_hw, _columns(hw), FS), want)


def test_the_builder_refuses_a_derivation_that_is_not_the_construction_it_records(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The recipe is read off the code; if the code changes, the recipe is not written."""
    def plain_mean(rec: Recording, config: str | None = None) -> tuple[Any, Any]:
        cols = np.column_stack([rec.data[:, c.index] for c in rec.channels
                                if c.role == "stomach"])
        return np.mean(cols, axis=1), "common_average"
    monkeypatch.setattr(sr, "build_stomach_reference", plain_mean)
    with pytest.raises(sr.StomachRecipeError, match="impulse responses"):
        sr.stomach_ref_recipe(_rec(), AS_IS)


def test_the_builder_refuses_a_recipe_that_disagrees_with_the_derivation_on_the_data(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Same weights, different rounding: a dot product is caught on the recording itself."""
    def dot_apply(recipe: dict[str, Any], columns: dict[str, np.ndarray],
                  fs: float) -> np.ndarray:
        cols = np.column_stack([columns[c] for c in recipe["contacts"]])
        return np.asarray(cols @ np.asarray(recipe["weights"]), dtype=np.float64)
    monkeypatch.setattr(sr, "apply_stomach_ref_recipe", dot_apply)
    with pytest.raises(sr.StomachRecipeError, match="differs from the derivation"):
        sr.stomach_ref_recipe(_rec(), AS_IS)


def test_a_recording_without_a_stomach_has_no_recipe() -> None:
    with pytest.raises(sr.StomachRecipeError, match="no stomach"):
        sr.stomach_ref_recipe(_rec(n_stomach=0), AS_IS)


@pytest.mark.parametrize(("routing", "match"), [
    ({"notch_hz": [60.0]}, "both notch_hz and contacts"),
    ({"notch_hz": [60.0], "contacts": ["ANT1"], "q": 10}, "unknown field"),
    ({"notch_hz": [], "contacts": ["ANT1"]}, "empty"),
    ({"notch_hz": [60.0], "contacts": ["ANT9"]}, "not stomach contacts"),
])
def test_an_unknown_routing_variant_is_refused(routing: dict[str, Any], match: str) -> None:
    with pytest.raises(sr.StomachRecipeError, match=match):
        sr.stomach_ref_recipe(_rec(), routing)


def _good() -> dict[str, Any]:
    return sr.stomach_ref_recipe(_rec(), AS_IS)


@pytest.mark.parametrize("key", sr.RECIPE_KEYS)
def test_a_recipe_missing_any_field_is_refused_by_name(key: str) -> None:
    for broken in ({k: v for k, v in _good().items() if k != key}, {**_good(), key: None}):
        with pytest.raises(sr.StomachRecipeError, match=f"lacks {key}"):
            sr.validate_stomach_ref_recipe(broken)


@pytest.mark.parametrize(("change", "match"), [
    ({"schema": "stomach_ref_recipe/0"}, "schema"),
    ({"how": "bipolar"}, "does not make form"),
    ({"form": "passthrough"}, "does not make form"),
    ({"weights": [0.5, -0.25, -0.25]}, "weights"),
    ({"weights": [1.0 - 1.0 / 3, -1.0 / 3]}, "weights"),
    ({"positive": "ANT2"}, "positive"),
    ({"contacts": ["ANT1", "ANT1", "ANT3"]}, "distinct"),
    ({"config": "bipolar"}, "config"),
    ({"notch": "off"}, "neither"),
    ({"notch": {"freqs_hz": [60.0], "contacts": ["ANT1"]}}, "notch lacks q"),
    ({"notch": {"freqs_hz": [-60.0], "q": 30.0, "contacts": ["ANT1"]}}, "freqs_hz"),
    ({"notch": {"freqs_hz": [60.0], "q": 30.0, "contacts": ["ANT9"]}}, "notch contacts"),
])
def test_an_inconsistent_recipe_is_refused(change: dict[str, Any], match: str) -> None:
    with pytest.raises(sr.StomachRecipeError, match=match):
        sr.validate_stomach_ref_recipe({**_good(), **change})


def test_apply_refuses_a_contact_the_recording_lacks() -> None:
    cols = _columns(_rec())
    del cols["ANT3"]
    with pytest.raises(sr.StomachRecipeError, match="ANT3"):
        sr.apply_stomach_ref_recipe(_good(), cols, FS)


# ---------------------------------------------------------------------------
# the mask file carries it
# ---------------------------------------------------------------------------

N = int(30.0 * FS)
READS: dict[str, tuple[str, ...]] = {"spikes": (), "slow_wave": (), "mmc": ("ANT1",),
                                     "hrv": ("stomach_ref",), "breathing": ("stomach_ref",),
                                     "velocity": ()}
NOT_COMPUTED = {"slow_wave": "synthetic: not computed"}
ELSEWHERE = {"spikes": "synthetic: finished per animal in masks/m_spikes"}


def _prov(recipe: dict[str, Any] | None = None) -> MaskProvenance:
    return MaskProvenance(
        model={"mode": "pooled", "version": "t", "corpus_hash": "0" * 32,
               "calibrator": "models/t/cal.json"},
        thresholds={"source": "synthetic"}, reference_values={"stomach_ref|0.5-3": {"median": 1}},
        code_commit="test", generation_sha="0133349b3ebeff80", routing_hash="test",
        created_at="2026-10-12T05:00:00+00:00", recording="syn",
        stomach_ref_recipe=recipe or {})


def _write(path: Path, reads: dict[str, tuple[str, ...]], prov: MaskProvenance) -> Path:
    masks = mk.build_masks(reads, [mk.MaskSpan("hrv", "stomach_ref", 1.0, 2.0, "x")]
                           if reads["hrv"] else [], n_frames=n_grid_frames(N, FS), t0_s=0.0)
    nc = NOT_COMPUTED if reads["hrv"] else {**NOT_COMPUTED, "hrv": "no train",
                                             "breathing": "no train"}
    return ho.write_mask_file(path, masks, prov, signals=reads, fs=FS, n_samples=N,
                              epoch_start_s=0.0, min_retention=0.5, animal_median={},
                              line_distrust=None, peri_r=None, not_computed=nc,
                              elsewhere=ELSEWHERE)


def test_the_mask_file_carries_the_recipe_when_a_consumer_reads_stomach_ref(
        tmp_path: Path) -> None:
    recipe = _good()
    path = _write(tmp_path / "a.mat", READS, _prov(recipe))
    doc = json.loads(str(loadmat(path)["provenance_json"][0]))
    assert doc["stomach_ref_recipe"] == recipe
    again = MaskProvenance.from_json(str(loadmat(path)["provenance_json"][0]))
    assert again.to_json() == str(loadmat(path)["provenance_json"][0])


def test_the_json_sidecar_and_the_mat_carry_an_identical_recipe(tmp_path: Path) -> None:
    """The driver's sidecar and the masks file carry one recipe (review 2026-10-12).

    The driver writes ``{stem}_provenance.json`` from ``prov`` BEFORE the writer (night4.py:
    ``write_json(..., json.loads(prov.to_json()))``). The recipe is in ``prov`` itself, so
    the two copies cannot disagree.
    """
    rec = _rec()
    prov = _prov(sr.stomach_ref_recipe(rec, {}))
    sidecar = tmp_path / "e0_provenance.json"
    sidecar.write_text(json.dumps(json.loads(prov.to_json()), indent=1, sort_keys=True) + "\n",
                       encoding="utf-8", newline="\n")
    mat = _write(tmp_path / "e0_masks.mat", READS, prov)
    in_mat = json.loads(str(loadmat(mat)["provenance_json"][0]))["stomach_ref_recipe"]
    in_json = json.loads(sidecar.read_text(encoding="utf-8"))["stomach_ref_recipe"]
    assert in_mat == in_json == sr.stomach_ref_recipe(rec, {})


def test_the_writer_takes_no_recipe_argument() -> None:
    import inspect  # noqa: PLC0415
    assert "stomach_ref_recipe" not in inspect.signature(ho.write_mask_file).parameters


def test_reading_stomach_ref_without_its_recipe_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^syn: .*must carry its recipe"):  # names it
        _write(tmp_path / "a.mat", READS, _prov())


def test_a_recipe_is_checked_even_when_nobody_reads_stomach_ref(tmp_path: Path) -> None:
    reads = {**READS, "hrv": (), "breathing": ()}
    path = _write(tmp_path / "a.mat", reads, _prov(_good()))       # allowed, and carried
    assert json.loads(str(loadmat(path)["provenance_json"][0]))["stomach_ref_recipe"] == _good()
    broken = {k: v for k, v in _good().items() if k != "weights"}
    with pytest.raises(ValueError, match="lacks weights"):
        _write(tmp_path / "b.mat", reads, _prov(broken))


def test_an_incomplete_recipe_is_refused(tmp_path: Path) -> None:
    broken = {k: v for k, v in _good().items() if k != "notch"}
    with pytest.raises(ValueError, match=r"^syn: .*lacks notch"):
        _write(tmp_path / "a.mat", READS, _prov(broken))


def test_the_finalise_reader_reads_nothing_when_no_consumer_reads_stomach_ref(
        tmp_path: Path) -> None:
    none_read = {**READS, "hrv": (), "breathing": ()}
    assert sr.proven_stomach_ref_recipe(tmp_path, "A", "syn", signals=none_read,
                                        routing_entry={}, generation_sha="x",
                                        reference_values={}) == {}
    with pytest.raises(sr.StomachRecipeError, match="no proven recipe"):
        sr.proven_stomach_ref_recipe(tmp_path, "A", "syn", signals=READS,
                                     routing_entry={"stomach_ref": {}}, generation_sha="x",
                                     reference_values={"e0": {}})


@settings(max_examples=60, deadline=None)
@given(n=st.integers(1, 5), form=st.sampled_from(["positive_minus_nanmean", "passthrough"]),
       freqs=st.lists(st.floats(1.0, 1000.0, allow_nan=False), max_size=3),
       q=st.floats(0.5, 100.0, allow_nan=False))
def test_a_recipe_round_trips_exactly_through_the_provenance(
        n: int, form: str, freqs: list[float], q: float) -> None:
    contacts = [f"ANT{k + 1}" for k in range(1 if form == "passthrough" else n)]
    how = "common_average" if form == "positive_minus_nanmean" else "single_channel"
    recipe = {"schema": sr.RECIPE_SCHEMA, "how": how, "config": "independent",
              "contacts": contacts, "positive": contacts[0], "form": form,
              "weights": sr.implied_weights(form, len(contacts), 0),
              "notch": ({"freqs_hz": freqs, "q": q, "contacts": contacts[:1]} if freqs
                        else sr.NO_NOTCH),
              "routing_stomach_ref": ({"notch_hz": freqs, "contacts": contacts[:1]} if freqs
                                      else {}),
              "source": "hypothesis"}
    sr.validate_stomach_ref_recipe(recipe)
    prov = _prov(recipe)
    back = MaskProvenance.from_json(prov.to_json())
    assert dict(back.stomach_ref_recipe) == recipe
    assert back.to_json() == prov.to_json()
    sr.validate_stomach_ref_recipe(back.stomach_ref_recipe)
