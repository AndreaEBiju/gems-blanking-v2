"""Motion-artifact detection and per-consumer blanking for rodent vagus ENG and stomach EMG.

Subpackages follow the pipeline order: :mod:`io` loads a recording, :mod:`derive`
builds the per-cuff signals, :mod:`physio` and :mod:`bands` describe them,
:mod:`detect` proposes candidates, :mod:`model` judges them, and :mod:`extent`,
:mod:`emit` turn judgements into per-consumer masks.

All public time is in seconds (``float64``), all amplitude in microvolts
(``float64``), all frequency in Hz. See ``CLAUDE.md`` for the hard invariants.
"""

from gems_blanking_v2.constants import BANDS, CONSUMERS, GRID_S

__all__ = ["BANDS", "CONSUMERS", "GRID_S", "__version__"]

__version__ = "0.1.0"
