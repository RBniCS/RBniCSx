# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pytest configuration file for tutorials tests."""

import nbvalx.pytest_hooks_notebooks
import pytest

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_sessionstart = nbvalx.pytest_hooks_notebooks.sessionstart
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown


def pytest_runtest_setup(item: nbvalx.pytest_hooks_notebooks.IPyNbFile) -> None:
    """Skip tests if dolfinx is not available."""
    # Do the setup as in nbvalx
    nbvalx.pytest_hooks_notebooks.runtest_setup(item)
    # Check dolfinx availability
    pytest.importorskip("dolfinx")
