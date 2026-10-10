"""Night 5 stage 1 split (RULING 2026-10-10 item 4): inference cached once, extents per set.

* the stage-1a cache answers every threshold set and scope exactly as the full arrays do
  (property test over thresholds >= 1.44, floors, scopes and confirmation sets);
* stage 1b refuses what the cache cannot answer exactly, and a stage-1a record from another
  wave, model, rule, routing entry or window;
* nothing an extent choice declares can reach stage 1a: the night config enters only
  through the inference allowlist, and the freshness keys take no extent input;
* line-noise pass 1 depends on the spike consumer's motion mask (why it moved to stage 1b).
"""

from __future__ import annotations

import functools
import inspect
import json
import math

import numpy as np
import pytest
from gems_blanking_v2.constants import ENG_BAND
from gems_blanking_v2.emit import line_distrust as ld
from gems_blanking_v2.emit.masks import ConsumerMask
from gems_blanking_v2.extent import routing as rt
from gems_blanking_v2.extent import stage1 as st1
from gems_blanking_v2.extent import tolerance as tl
from gems_blanking_v2.types import Candidate, Event
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hs

from tests.conftest import (
    STAGE1_SIGNALS,
    Stage1Synth,
    make_mains_spike_t,
    make_stage1_epoch,
)

Z_CONSUMERS = ("spikes", "mmc", "slow_wave", "breathing")


@functools.cache
def _synth(seed: int) -> Stage1Synth:
    return make_stage1_epoch(seed=seed)


@functools.cache
def _full(seed: int) -> st1.EpochCache:
    syn = _synth(seed)
    specs = tl.extent_consumers()
    z = {(s, specs[c].band): syn.zdet[(s, specs[c].band)]
         for c, v in syn.signals.items() if c != "hrv" for s in v}
    eng = {s: rt.eng_trace(s, syn.eng_raw[s], syn.fs, x_t0_s=syn.t0_s)
           for s in syn.signals["spikes"]}
    own = {c: tl.own_detection_signals(v, STAGE1_SIGNALS, {})
           for c, v in syn.signals.items() if c != "hrv" and v}
    return st1.EpochCache(events=syn.events, detecting=syn.detecting,  # type: ignore[arg-type]
                          cached_events=frozenset(syn.events), z=z, zdet=syn.zdet, eng=eng,
                          hrv=syn.hrv, signals=syn.signals, own_signals=own,  # type: ignore[arg-type]
                          fs=syn.fs, z_t0_s=syn.t0_s, n_frames=syn.n_frames)


def _sparse(seed: int, cached: frozenset[str]) -> st1.EpochCache:
    """Return the stage-1a cache of ``_full(seed)`` built for the events ``cached``."""
    full = _full(seed)
    wins = {eid: st1.core_frames(full.events[eid].candidate.start_s,
                                 full.events[eid].candidate.stop_s, full.z_t0_s)
            for eid in sorted(cached)}
    by_pair: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for eid, w in wins.items():
        for q in full.detecting[eid]:
            by_pair.setdefault(q, []).append(w)
    zdet = st1.sparse_detection_z(full.zdet, by_pair)
    cores = [(full.events[e].candidate.start_s, full.events[e].candidate.stop_s)
             for e in sorted(cached)]
    eng: dict[str, rt.EngTrace | rt.CachedEngTrace] = {}
    for s, tr in full.eng.items():
        assert isinstance(tr, rt.EngTrace)
        spans = st1.eng_cache_spans(cores, trace_t0_s=tr.t0_s, rate=tr.rate, size=tr.size)
        eng[s] = rt.CachedEngTrace.from_trace(tr, spans)
    return st1.EpochCache(events=full.events, detecting=full.detecting, cached_events=cached,
                          z=full.z, zdet=zdet, eng=eng,
                          hrv={k: full.hrv[k] for k in cached}, signals=full.signals,
                          own_signals=full.own_signals, fs=full.fs, z_t0_s=full.z_t0_s,
                          n_frames=full.n_frames)


def _table(thr: dict[str, float]) -> tl.ToleranceTable:
    return tl.ToleranceTable(thr, "test thresholds")


def _same(a: tuple, b: tuple) -> None:  # type: ignore[type-arg]
    conf_a, ext_a, dec_a, m_a = a
    conf_b, ext_b, dec_b, m_b = b
    assert list(conf_a) == list(conf_b)
    assert ext_a == ext_b
    assert rt.decisions_table(dec_a) == rt.decisions_table(dec_b)
    assert sorted(m_a) == sorted(m_b)
    for k in m_a:
        assert np.array_equal(m_a[k].invalid, m_b[k].invalid), k


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=hs.sampled_from([1, 2, 3]), data=hs.data())
def test_the_cache_answers_every_threshold_set_and_scope_as_the_full_arrays(
        seed: int, data: hs.DataObject) -> None:
    full = _full(seed)
    ids = sorted(full.events)
    cached = frozenset(e for e in ids if data.draw(hs.booleans(), label=f"cache {e}"))
    confirmed = [e for e in sorted(cached) if data.draw(hs.booleans(), label=f"conf {e}")]
    thr = {c: data.draw(hs.floats(st1.CACHE_FLOOR_Z, 12.0), label=c) for c in Z_CONSUMERS}
    scopes = data.draw(hs.one_of(hs.none(), hs.fixed_dictionaries(
        {c: hs.sampled_from(tl.CROSS_BAND_SCOPES) for c in Z_CONSUMERS})), label="scopes")
    floor = data.draw(hs.floats(st1.CACHE_FLOOR_Z, 5.0), label="floor")
    want = st1.extents_routing_masks(full, confirmed, tolerances=_table(thr), scopes=scopes,
                                     floor_z=floor)
    got = st1.extents_routing_masks(_sparse(seed, cached), confirmed, tolerances=_table(thr),
                                    scopes=scopes, floor_z=floor)
    _same(want, got)


def test_the_generator_makes_every_route_and_extent_basis() -> None:
    """The property test above exercises routing both ways and both extent bases."""
    full = _full(1)
    thr = dict.fromkeys(Z_CONSUMERS, 3.0)
    scopes = dict.fromkeys(Z_CONSUMERS, tl.CROSS_BAND_SCOPE_ALL)
    _c, ext, dec, masks = st1.extents_routing_masks(full, sorted(full.events),
                                                    tolerances=_table(thr), scopes=scopes)
    bases = {r.basis for *_x, r in ext if isinstance(r, tl.Extent)}
    assert bases == {tl.EXTENT_BASIS_OWN_BAND, tl.EXTENT_BASIS_DETECTED}
    spike_routes = {d.route for d in dec if d.consumer == "spikes"
                    and d.reason_code != rt.CROSS_BAND_REASON_CODE}
    assert spike_routes == {"correct", "reject"}
    assert any(r is None for *_x, r in ext)
    assert all(m.invalid.any() for m in masks.values())


def test_a_cache_built_with_core_frames_only_is_caught() -> None:
    """Revert check of the sparse rule, in-suite: keeping only the core frames is not exact."""
    seed = 2
    full = _full(seed)
    cached = frozenset(full.events)
    sp = _sparse(seed, cached)
    core_only = {}
    for q, zz in full.zdet.items():
        keep = np.zeros(zz.size, dtype=bool)
        for eid in cached:
            if q in full.detecting[eid]:
                c = full.events[eid].candidate
                i0, i1 = st1.core_frames(c.start_s, c.stop_s, full.z_t0_s)
                keep[max(0, i0):i1] = True
        core_only[q] = np.where(keep, zz, np.nan)
    bad = st1.EpochCache(**{**sp.__dict__, "zdet": core_only})
    thr = dict.fromkeys(Z_CONSUMERS, 20.0)  # own bands never cross: cross-band decides
    scopes = dict.fromkeys(Z_CONSUMERS, tl.CROSS_BAND_SCOPE_ALL)
    args = {"tolerances": _table(thr), "scopes": scopes}
    want = st1.extents_routing_masks(full, sorted(cached), **args)  # type: ignore[arg-type]
    assert st1.extents_routing_masks(sp, sorted(cached), **args)[1] == want[1]  # type: ignore[arg-type]
    assert st1.extents_routing_masks(bad, sorted(cached), **args)[1] != want[1]  # type: ignore[arg-type]


def test_stage1b_refuses_what_the_cache_cannot_answer_exactly() -> None:
    full = _full(1)
    ids = sorted(full.events)
    sp = _sparse(1, frozenset(ids[:5]))
    ok = dict.fromkeys(Z_CONSUMERS, 2.0)
    scopes = dict.fromkeys(Z_CONSUMERS, tl.CROSS_BAND_SCOPE_ALL)
    with pytest.raises(ValueError, match="not in the stage-1a cache"):
        st1.extents_routing_masks(sp, ids[:6], tolerances=_table(ok), scopes=scopes)
    with pytest.raises(ValueError, match="below the cache floor"):
        st1.extents_routing_masks(sp, ids[:5], tolerances=_table(ok), scopes=scopes,
                                  floor_z=1.0)
    with pytest.raises(ValueError, match=r"spikes: threshold 1.2 is below the cache floor"):
        st1.extents_routing_masks(sp, ids[:5], tolerances=_table({**ok, "spikes": 1.2}),
                                  scopes=scopes)


def test_a_cached_eng_trace_refuses_a_span_it_does_not_hold() -> None:
    syn = _synth(1)
    tr = rt.eng_trace("L_T", syn.eng_raw["L_T"], syn.fs, x_t0_s=0.0)
    c = rt.CachedEngTrace.from_trace(tr, [(100, 200), (150, 400), (1000, 1100)])
    assert [(a, v.size) for a, v in c.segments] == [(100, 300), (1000, 100)]
    assert np.array_equal(c.window(slice(120, 390)), tr.window(slice(120, 390)))
    assert (c.size, c.n_finite) == (tr.size, tr.n_finite)
    with pytest.raises(rt.EngCacheMissError):
        c.window(slice(350, 1050))
    assert not issubclass(rt.EngCacheMissError, ValueError)  # never read as "not assessable"


def _keys(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "generation": "0133349b3ebeff80", "feature_version": "p1",
        "choice_file_sha256": "c" * 64, "model_id": "m" * 32, "evaluation_model_id": None,
        "decision_rule": tl.DECISION_RULES["raw_p_ge_0.5"].provenance(), "p_threshold": 0.5,
        "sensitivity_p": [0.7, 0.3], "hrv_window_s": 10.0, "wave_id": "w1",
        "production_routing_sha256": "r" * 64, "routing_entry_hash": "e" * 64,
        "routing_table": "t" * 64, "peri_r_window_sha256": "p" * 64}
    base.update(over)
    return st1.stage1a_keys(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(("key", "value"), [
    ("wave_id", "w2"), ("production_routing_sha256", "s" * 64), ("model_id", "n" * 32),
    ("evaluation_model_id", "v" * 32), ("choice_file_sha256", "d" * 64),
    ("decision_rule", tl.DECISION_RULES["calibrated_p_ge_0.5"].provenance()),
    ("sensitivity_p", [0.3]), ("hrv_window_s", 5.0), ("routing_entry_hash", "f" * 64),
    ("peri_r_window_sha256", "q" * 64), ("generation", "x" * 16)])
def test_stage1b_refuses_a_stage1a_record_from_another_wave_or_model(key: str,
                                                                     value: object) -> None:
    st1.check_stage1a_fresh(_keys(), _keys(), what="rec e0")
    with pytest.raises(st1.Stage1aMismatchError, match=f"{key}: record"):
        st1.check_stage1a_fresh(_keys(**{key: value}), _keys(), what="rec e0")


def test_a_record_of_another_schema_is_refused() -> None:
    old = {**_keys(), "schema": "night5-stage1a/0"}
    with pytest.raises(st1.Stage1aMismatchError, match="schema"):
        st1.check_stage1a_fresh(old, _keys(), what="rec e0")


EXTENT_FIELDS = ("tolerances", "cross_band_scope", "extent_thresholds", "min_retention",
                 "hold", "extent_percentile", "spike_tolerance_method", "pilot_root")


@settings(max_examples=60, deadline=None)
@given(extra=hs.dictionaries(hs.sampled_from(EXTENT_FIELDS),
                             hs.recursive(hs.none() | hs.booleans() | hs.floats(allow_nan=False)
                                          | hs.text(max_size=8),
                                          lambda c: hs.lists(c, max_size=3)
                                          | hs.dictionaries(hs.text(max_size=4), c, max_size=3),
                                          max_leaves=8)))
def test_no_extent_declaration_reaches_stage1a(extra: dict[str, object]) -> None:
    """Stage 1a's view of the night config is byte-identical whatever the extent choice."""
    base = {"decision_rules": {"rules": {"new:J": "raw_p_ge_0.5"}}, "p_threshold": {"value": 0.5},
            "sensitivity_p": {"values": [0.3, 0.7]}, "hrv_window_s": {"value": 10.0},
            "wave": {"id": "w1"}, "stand_in": False, "model_choice": {"sha256": "c" * 64}}
    view = json.dumps(st1.inference_config(base), sort_keys=True)
    assert json.dumps(st1.inference_config({**base, **extra}), sort_keys=True) == view
    assert not set(EXTENT_FIELDS) & set(st1.INFERENCE_CONFIG_KEYS)


def test_the_freshness_keys_take_no_extent_input() -> None:
    params = set(inspect.signature(st1.stage1a_keys).parameters)
    words = ("toler", "scope", "thresh", "extent", "percentile", "hold", "retention", "floor")
    assert not [p for p in params if any(w in p for w in words) and p != "p_threshold"]
    assert set(_keys()) <= set(st1.STAGE1A_KEYS)
    assert _keys()["cache_floor_z"] == st1.CACHE_FLOOR_Z == 1.44


def test_content_hash_covers_every_array_byte_and_the_json_content() -> None:
    a = {"e0|z|L_T|300-3000": np.arange(5, dtype=np.float64)}
    h = st1.content_sha256({"x": 1, "y": [1.5]}, a)
    assert h == st1.content_sha256({"y": [1.5], "x": 1}, dict(a))
    b = {k: v.copy() for k, v in a.items()}
    b["e0|z|L_T|300-3000"][3] = np.nextafter(3.0, 4.0)
    assert st1.content_sha256({"x": 1, "y": [1.5]}, b) != h
    assert st1.content_sha256({"x": 2, "y": [1.5]}, a) != h
    assert st1.content_sha256({"x": 1, "y": [1.5]},
                              {k: v.astype(np.float32) for k, v in a.items()}) != h


@given(start=hs.floats(0, 1e4), dur=hs.floats(1e-3, 30), p=hs.floats(allow_nan=True),
       raw=hs.floats(allow_nan=True), peak=hs.floats(0, 100))
def test_an_event_round_trips_through_its_row(start: float, dur: float, p: float, raw: float,
                                              peak: float) -> None:
    c = Candidate(start, start + dur, ("L_T", "R_V2"), ("0-2", "300-3000"), peak, "electrical")
    ev = Event(c, "unjudged", p, "model")
    row = json.loads(json.dumps(st1.event_row("e0_00001", ev, [("L_T", "0-2")], raw=raw),
                                allow_nan=False))
    back, pairs, raw2 = st1.event_from_row(row)
    assert back.candidate == c and pairs == (("L_T", "0-2"),)
    assert (back.p_motion == p) or (math.isnan(back.p_motion) and not math.isfinite(p))
    assert (raw2 == raw) or (math.isnan(raw2) and not math.isfinite(raw))


def test_an_extent_round_trips_and_none_stays_none() -> None:
    e = tl.Extent("hrv", "R_T", "10-150", 1.1, 2.2, 1.24, 2.06, 0.14, 0.1,
                  tl.EXTENT_BASIS_OWN_BAND)
    assert st1.extent_from_json(json.loads(json.dumps(st1.extent_json(e)))) == e
    assert st1.extent_from_json(st1.extent_json(None)) is None


def test_line_noise_pass1_reads_the_spike_motion_mask() -> None:
    """EXTENT_DEPENDENT's reason pass 1 runs in stage 1b: a motion blank changes its tests."""
    fs, dur = 24414.0625, 120.0
    t = make_mains_spike_t(fs, dur, locked_minutes=(1,), seed=4).signal
    n = t.size
    nf = int(math.floor(dur / 0.010))
    clean = ConsumerMask("spikes", "L_T", ENG_BAND, np.zeros(nf, dtype=bool), 0.010, 0.0)
    inv = np.zeros(nf, dtype=bool)
    inv[7000:9500] = True  # 25 s of motion blank in minute 1
    blanked = ConsumerMask("spikes", "L_T", ENG_BAND, inv, 0.010, 0.0)
    a = ld.pass1({"L_T": t}, fs, epoch_start_s=0.0, motion={"L_T": clean})
    b = ld.pass1({"L_T": t}, fs, epoch_start_s=0.0, motion={"L_T": blanked})
    assert a.n_samples == b.n_samples == n
    assert a.tests[1].valid_s > b.tests[1].valid_s + 24.0
    assert a.tests[0] == b.tests[0]
