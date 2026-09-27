"""Test package for gems_blanking_v2.

This file is load-bearing, not boilerplate. ``detector-core`` (GEMSBlanking) is
installed editable and its flat layout puts its repository root on ``sys.path``,
and it also has a top-level ``tests`` package. Python's import system prefers a
REGULAR package found later on the path over a NAMESPACE package found earlier,
so without this file every ``from tests.conftest import ...`` in this repository
resolved to *their* tests package and failed.

That is the same shadowing hazard ``detector-pyqt``'s own sys.path setup warns
about for its ``ui`` package. Making this a regular package is the fix that does
not depend on path ordering.
"""
