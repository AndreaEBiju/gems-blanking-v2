"""Task 18 requirements from the 2026-09-29 common-mode ruling - a STUB until task 18.

The new cohort has no reference channel: every contact is single-ended against one
ground in the abdominal wall, so the ground site's signal (most likely
abdominal-wall muscle) enters all nine channels identically. Measured on the 10
round-3/4 recordings: same-instant, same-polarity, equal-amplitude events on >= 7
of 9 contacts at 124-2,900 per minute in every recording, and common mode dominating
the raw contacts' spike band (sigma ~10 uV raw against ~2.2 uV on T).

What task 18 must satisfy, and what this test must check once ``velocity/xcorr.py``
exists:

1. **Velocity does not run on raw contacts.** A raw-pair cross-correlation under
   this background is dominated by the zero-lag common mode. It runs on the
   common-mode-corrected contacts (``derive/common_mode.py``, the subtraction route)
   or on differential pairs. Test: a raw pair carrying a shared common-mode
   background gives its peak at zero lag; the corrected or differential pair
   recovers the injected conduction lag.
2. **Zero-lag and all-channel events are rejected.** Any event arriving at zero lag
   across the pair, or on all nine contacts within 0.2 ms, is not conduction:
   conduction at 1.5 mm pitch cannot be simultaneous. Test: an identical 1.2 ms
   transient on all nine channels (``make_shared_ground``) yields no velocity
   estimate, while a transient with a physiological lag on one cuff's contacts does.
3. The peak-ratio confidence (neural peak / zero-lag peak) is computed after 1, so
   the common mode does not set the zero-lag peak it is divided by.
"""

from __future__ import annotations

import pytest


def test_velocity_rejects_the_shared_ground_common_mode() -> None:
    """Task 18 acceptance: see the module docstring. Skipped until task 18 lands."""
    pytest.importorskip("gems_blanking_v2.velocity.xcorr",
                        reason="task 18 (conduction velocity) is not built yet")
    pytest.skip("task 18 exists: replace this stub with the requirements in the module "
                "docstring - velocity on raw contacts, or accepting a zero-lag event, fails")
