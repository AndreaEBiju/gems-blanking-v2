"""Task 18's contact exclusion (ruling 2026-10-05, item 2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from gems_blanking_v2.physio import hr_pairs as hp
from gems_blanking_v2.velocity.contacts import excluded_contacts

from tests.conftest import make_cross_site_heart

FS = 24414.0625
STRONG_STOMACH = {"R": 1.0, "L": -0.6, "stomach": 1.0}


def test_an_open_contact_is_excluded_and_no_healthy_one_is() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, open_contact="LVN3",
                              site_gain=STRONG_STOMACH, seed=7)
    assert excluded_contacts(w.rec) == frozenset({"LVN3"})
    ok = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, site_gain=STRONG_STOMACH, seed=7)
    assert excluded_contacts(ok.rec) == frozenset()


def test_the_exclusion_reads_the_first_half_only() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, open_contact="LVN3",
                              site_gain=STRONG_STOMACH, seed=7)
    ok = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, site_gain=STRONG_STOMACH, seed=7)
    n = w.rec.data.shape[0] // 2
    data = np.array(ok.rec.data)
    data[n:] = w.rec.data[n:]  # the contact opens only in the second half
    assert excluded_contacts(replace(ok.rec, data=data)) == frozenset()
    data = np.array(w.rec.data)
    data[n:] = ok.rec.data[n:]  # open in the first half only
    assert excluded_contacts(replace(w.rec, data=data)) == frozenset({"LVN3"})


def test_it_is_the_half_split_protocol_s_detached_set() -> None:
    w = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, open_contact="LVN3",
                              site_gain=STRONG_STOMACH, seed=7)
    first, _second = hp._halves(w.rec)
    h = hp._prepare_half(first, seed=0)
    assert excluded_contacts(w.rec) == hp.detached_contacts(h.events, h.keep)


def test_the_ground_gains_are_measured_on_events_clear_of_the_heartbeat() -> None:
    # a dominant heart, so the top-ranked candidate's beats are the true beats
    w = make_cross_site_heart(FS, 65.0, transients_per_s=2.0, cardiac_uv=600.0, seed=7)
    ev, keep = hp.noncardiac_events(w.rec)
    beats = np.sort(np.asarray(w.beats_s, float))
    t = np.asarray(ev.t_s, float)
    j = np.searchsorted(beats, t)
    d = np.minimum(np.abs(t - beats[np.clip(j - 1, 0, beats.size - 1)]),
                   np.abs(beats[np.clip(j, 0, beats.size - 1)] - t))
    assert (~keep).any()  # some events fall on beats and are dropped
    assert keep[d > 0.025].all()  # clear of every true beat: kept
    assert not keep[d < 0.015].any()  # on a beat: dropped
