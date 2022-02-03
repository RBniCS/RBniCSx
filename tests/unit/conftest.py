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

import nbvalx.pytest_hooks_unit_tests
import numpy as np
import petsc4py
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


pytest_runtest_setup = nbvalx.pytest_hooks_unit_tests.runtest_setup
pytest_runtest_teardown = nbvalx.pytest_hooks_unit_tests.runtest_teardown
