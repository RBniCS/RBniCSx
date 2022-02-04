# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
pytest configuration file for unit tests.

This file assigns pytest hooks and declares common fixtures used across several files.
"""

import typing

import _pytest.config
import _pytest.main
import nbvalx.pytest_hooks_unit_tests
import numpy as np
import petsc4py
import py
import pytest
import scipy.sparse


@pytest.fixture(scope="module")
def to_dense_matrix() -> typing.Callable:
    """Fixture that returns a function to convert the local part of a sparse PETSc Mat into a dense numpy ndarray."""
    def _(mat: petsc4py.PETSc.Mat) -> np.typing.NDArray[petsc4py.PETSc.ScalarType]:
        """Convert the local part of a sparse PETSc Mat into a dense numpy ndarray."""
        ai, aj, av = mat.getValuesCSR()
        return scipy.sparse.csr_matrix((av, aj, ai), shape=(mat.getLocalSize()[0], mat.getSize()[1])).toarray()
    return _


def pytest_addoption(parser: _pytest.main.Parser) -> None:
    """Add an option to skip testing backends."""
    parser.addoption("--skip-backends", action="store_true", help="Skip tests which require backends to be installed")


def pytest_ignore_collect(path: py.path.local, config: _pytest.config.Config) -> bool:
    """Honor the --skip-backends option to skip tests which require backends to be installed."""
    skip_backends = config.option.skip_backends
    if skip_backends:
        if any([blacklist in str(path) for blacklist in ["tests/unit/backends/", "tests/unit/cpp/backends/"]]):
            return True
        else:
            return False
    else:
        return False


pytest_runtest_setup = nbvalx.pytest_hooks_unit_tests.runtest_setup
pytest_runtest_teardown = nbvalx.pytest_hooks_unit_tests.runtest_teardown
