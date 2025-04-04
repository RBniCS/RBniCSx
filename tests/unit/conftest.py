# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
pytest configuration file for unit tests.

This file assigns pytest hooks and declares common fixtures used across several files.
"""

import pathlib
import typing

import _pytest.compat
import nbvalx.pytest_hooks_unit_tests
import numpy.typing as npt
import petsc4py.PETSc
import pytest
import scipy.sparse


@pytest.fixture(scope="module")
def to_dense_matrix() -> typing.Callable[  # type: ignore[no-any-unimported]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]:
    """Fixture that returns a function to convert the local part of a sparse PETSc Mat into a dense numpy ndarray."""
    def _(mat: petsc4py.PETSc.Mat) -> npt.NDArray[petsc4py.PETSc.ScalarType]:  # type: ignore[no-any-unimported]
        """Convert the local part of a sparse PETSc Mat into a dense numpy ndarray."""
        ai, aj, av = mat.getValuesCSR()
        return scipy.sparse.csr_matrix(  # type: ignore[no-any-return]
            (av, aj, ai), shape=(mat.getLocalSize()[0], mat.getSize()[1])).toarray()
    return _


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add an option to skip testing backends."""
    parser.addoption("--skip-backends", action="store_true", help="Skip tests which require backends to be installed")


def pytest_ignore_collect(  # type: ignore[no-any-unimported]
    collection_path: pathlib.Path, path: _pytest.compat.LEGACY_PATH, config: pytest.Config
) -> bool:
    """Honor the --skip-backends option to skip tests which require backends to be installed."""
    skip_backends = config.option.skip_backends
    if skip_backends:
        if any([
            blacklist in str(collection_path) for blacklist in ["tests/unit/backends/", "tests/unit/cpp/backends/"]
        ]):
            return True
        else:
            return False
    else:
        return False


pytest_runtest_setup = nbvalx.pytest_hooks_unit_tests.runtest_setup
pytest_runtest_teardown = nbvalx.pytest_hooks_unit_tests.runtest_teardown
