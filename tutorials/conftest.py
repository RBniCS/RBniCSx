# Copyright (C) 2021-2025 by the RBniCSx authors
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


def pytest_runtest_setup(item: nbvalx.pytest_hooks_notebooks.IPyNbFile) -> None:
    """Skip tests if dolfinx is not available."""
    # Check dolfinx availability
    pytest.importorskip("dolfinx")
