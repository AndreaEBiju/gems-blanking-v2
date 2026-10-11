"""RULING 2026-10-10 item 6: released cuffs are kept for the spike consumer and flagged.

* the readings: every cuff review 6 listed (``tests/data/cuff_release_readings.json``, from
  ``review6_fix/cuff_release_readings.txt``) is classified exactly as the file did, from the
  routing entry's ``why`` and the peri-R pass row; R2, R3 and R4 are released, R1 stays;
* any other clause set - another reason, or both clauses with a train - stays distrusted;
* at emission the released cuff's whole-region distrust span is dropped from the spike mask
  and its spike signal is flagged "no heartbeat reference" every minute (``cuffrelease_*``,
  flags, never blank spans); per-minute distrust and every other consumer are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gems_blanking_v2.emit import cuff_trust as ct
from gems_blanking_v2.emit import handoff as ho
from gems_blanking_v2.emit import masks as mk
from scipy.io import loadmat

DATA = Path(__file__).parent / "data" / "cuff_release_readings.json"
FS = 24414.0625


def _fixture() -> list[dict[str, Any]]:
    doc: dict[str, Any] = json.loads(DATA.read_text(encoding="utf-8"))
    return list(doc["cuffs"])


def test_every_listed_cuff_is_classified_as_review_6_did() -> None:
    rows = _fixture()
    assert len(rows) == 72
    got = {(r["recording"], r["cuff"]): ct.reading(r["why"], no_train=ct.no_count_gated_train(
        r.get("perir_row"))) for r in rows}
    want = {(r["recording"], r["cuff"]): r["expected"] for r in rows}
    assert got == want
    counts = {k: sum(v == k for v in want.values()) for k in ct.READINGS}
    assert counts == {"R1": 11, "R2": 20, "R3": 37, "R4": 4}


def test_readings_2_3_4_are_released_and_reading_1_stays() -> None:
    assert {"R2", "R3", "R4"} == ct.RELEASED_READINGS
    for r in _fixture():
        entry = {"spike": {r["cuff"]: {"route": "distrusted", "why": r["why"]}}}
        t = ct.cuff_trust(entry, {r["cuff"]: r.get("perir_row")})[r["cuff"]]
        assert t.released == (r["expected"] != "R1"), r
        assert t.reading == r["expected"]


@pytest.mark.parametrize(("why", "no_train", "want"), [
    (ct.QRS_CLAUSE, False, "R1"),
    (ct.QRS_CLAUSE, True, "R1"),
    (ct.NO_WINDOW_CLAUSE, True, "R2"),
    (ct.NO_WINDOW_CLAUSE, False, "R3"),
    (f"{ct.NO_WINDOW_CLAUSE}; {ct.QRS_CLAUSE}", True, "R4"),
    (f"{ct.QRS_CLAUSE}; {ct.NO_WINDOW_CLAUSE}", True, "R4"),
    (f"{ct.NO_WINDOW_CLAUSE}; {ct.QRS_CLAUSE}", False, None),  # both clauses, a train
    ("veto fails on contact 3", True, None),                    # another reason stays
    (f"{ct.NO_WINDOW_CLAUSE}; veto fails on contact 3", True, None),
    ("", True, None),
])
def test_only_the_ruled_clause_sets_are_readings(why: str, no_train: bool,
                                                 want: str | None) -> None:
    assert ct.reading(why, no_train=no_train) == want


def test_no_count_gated_train_is_review_6s_rule() -> None:
    assert ct.no_count_gated_train({"result": "no count-gated train: no peri-R route"})
    assert ct.no_count_gated_train({"beat_grade": "none", "result": "x"})
    assert not ct.no_count_gated_train({"beat_grade": "hrv", "result": "peri-R route"})
    assert not ct.no_count_gated_train(None)  # no row: a train by that rule (R3, not R2)


ENTRY = {"spike": {"L": {"route": "distrusted", "why": ct.NO_WINDOW_CLAUSE},
                   "R": {"route": "kept", "distrusted_spans": [[10.0, 20.0]]}}}


def test_a_released_cuff_loses_its_whole_region_span_only() -> None:
    before = mk.distrusted_spike_spans(ENTRY, region_start_s=0.0, region_stop_s=60.0)
    after = mk.distrusted_spike_spans(ENTRY, region_start_s=0.0, region_stop_s=60.0,
                                      released=("L",))
    assert [(s.signal, s.reason) for s in before] == [("L_T", "cuff_distrusted"),
                                                      ("R_T", "per_minute_cuff_distrust")]
    assert [(s.signal, s.reason) for s in after] == [("R_T", "per_minute_cuff_distrust")]
    with pytest.raises(ValueError, match="does not distrust them outright"):
        mk.distrusted_spike_spans(ENTRY, region_start_s=0.0, region_stop_s=60.0,
                                  released=("R",))


def test_the_release_is_flagged_every_minute_never_blanked(tmp_path: Path) -> None:
    from tests.conftest import peri_r_like  # noqa: PLC0415
    from tests.test_emit_masks import N_SAMPLES, READS, _ld, _masked, _prov  # noqa: PLC0415
    rec = ct.cuff_trust(ENTRY, {"L": None})["L"].record()
    masks = _masked(0.01)
    kw: dict[str, Any] = {"signals": READS, "fs": FS, "n_samples": N_SAMPLES,
                          "line_distrust": _ld(), "peri_r": peri_r_like(_ld()),
                          "epoch_start_s": 0.0, "min_retention": 0.5, "animal_median": {}}
    plain = loadmat(ho.write_mask_file(tmp_path / "a.mat", masks, _prov(), **kw))
    flagged = loadmat(ho.write_mask_file(tmp_path / "b.mat", masks, _prov(),
                                         cuff_release={"L_T": rec}, **kw))
    assert np.array_equal(plain["blank_spikes_L_T"], flagged["blank_spikes_L_T"])
    doc = json.loads(str(flagged["cuffrelease_json"][0]))
    assert doc["signals"] == ["L_T"] and doc["flag"] == "no heartbeat reference"
    assert doc["blanked"] is False and doc["cuffs"]["L_T"]["reading"] == "R3"
    assert [m["minute"] for m in doc["minutes"]] == [0]  # a 60 s epoch: one minute
    assert flagged["cuffrelease_spikes_L_T"].tolist() == [[1.0, float(N_SAMPLES)]]
    assert json.loads(str(plain["cuffrelease_json"][0]))["signals"] == []
    with pytest.raises(ValueError, match="not read by the spike consumer"):
        ho.write_mask_file(tmp_path / "c.mat", masks, _prov(), cuff_release={"R_T": rec}, **kw)


# ---------------------------------------------------------------------------
# Andrea 2026-10-11 item 4: released cuff-epochs that failed the peri-R test
# ---------------------------------------------------------------------------

SHA = "a" * 64


def _list(entries: list[dict[str, str]]) -> dict[str, Any]:
    return {"schema": "gems-blanking-v2 perir-failed cuffs v1",
            "ruling": "Andrea 2026-10-11 item 4: test", "source": {"file": "x.json", "sha256": SHA},
            "entries": entries}


def test_a_failing_release_is_distrusted_in_primary_and_released_in_sensitivity() -> None:
    lst = ct.perir_failed_list(_list([{"session": "s", "cuff": "L", "epoch": "e0"}]),
                               sha256="b" * 64)
    assert lst.failing("s", "e0") == {"L"} and lst.failing("s", "e132") == frozenset()
    assert lst.record()["sha256"] == "b" * 64 and lst.record()["n_cuff_epochs"] == 1
    entry = {"spike": {"L": {"route": "distrusted", "why": ct.NO_WINDOW_CLAUSE},  # R3: failed
                       "R": {"route": "distrusted", "why": ct.NO_WINDOW_CLAUSE}}}  # untestable
    rows = {"L": None, "R": None}
    prim = ct.cuff_trust(entry, rows, perir_failed=lst.failing("s", "e0"))
    sens = ct.cuff_trust(entry, rows, perir_failed=lst.failing("s", "e0"), scope="sensitivity")
    assert not prim["L"].released and prim["L"].perir_failed
    assert prim["L"].record()["perir_failed"] == ct.PERIR_FAILED_RULE
    assert "flag" not in prim["L"].record()
    assert sens["L"].released and sens["L"].perir_failed
    assert prim["R"].released and sens["R"].released and not prim["R"].perir_failed  # unchanged
    assert prim["R"].record()["flag"] == ct.RELEASE_FLAG
    assert ct.cuff_trust(entry, rows)["L"].released  # a passing / unlisted release stays
    spans = mk.distrusted_spike_spans(entry, region_start_s=0.0, region_stop_s=60.0,
                                      released=("R",), perir_failed=("L",))
    assert [(s.signal, s.reason, s.start_s, s.stop_s) for s in spans] == [
        ("L_T", mk.PERIR_FAILED_REASON, 0.0, 60.0)]
    assert mk.distrusted_spike_spans(entry, region_start_s=0.0, region_stop_s=60.0,
                                     released=("L", "R")) == []  # sensitivity: kept
    with pytest.raises(ValueError, match="both released and distrusted"):
        mk.distrusted_spike_spans(entry, region_start_s=0.0, region_stop_s=60.0,
                                  released=("L",), perir_failed=("L",))
    r1 = {"spike": {"L": {"route": "distrusted", "why": ct.QRS_CLAUSE}}}
    with pytest.raises(ValueError, match="does not release it"):
        ct.cuff_trust(r1, {"L": None}, perir_failed={"L"})


def test_the_peri_r_failed_list_is_declared_unique_and_never_null() -> None:
    e = {"session": "s", "cuff": "L", "epoch": "e0"}
    with pytest.raises(ValueError, match="appears twice"):
        ct.perir_failed_list(_list([e, dict(e)]), sha256=SHA)
    with pytest.raises(ValueError, match="exactly"):
        ct.perir_failed_list(_list([{**e, "p": None}]), sha256=SHA)
    with pytest.raises(ValueError, match="must name"):
        ct.perir_failed_list({**_list([e]), "ruling": "someone"}, sha256=SHA)
    with pytest.raises(ValueError, match="scope"):
        ct.cuff_trust({}, {}, scope="both")
