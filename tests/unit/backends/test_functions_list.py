# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox.backends.functions_list module."""

import dolfinx.mesh
import dolfinx_utils.test.fixtures
import mpi4py
import numpy as np
import petsc4py
import pytest

import minirox.backends

tempdir = dolfinx_utils.test.fixtures.tempdir


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    return dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)


@pytest.fixture
def functions_list(mesh: dolfinx.mesh.Mesh) -> minirox.backends.FunctionsList:
    """Generate a minirox.backends.FunctionsList with two entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    functions_list = minirox.backends.FunctionsList(V)
    for i in range(2):
        function = dolfinx.fem.Function(V)
        with function.vector.localForm() as function_local:
            function_local.set(i + 1)
        functions_list.append(function)
    return functions_list


def test_functions_list_function_space(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.function_space."""
    assert functions_list.function_space == functions_list[0].function_space


def test_functions_list_extend(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.extend."""
    functions_list2 = minirox.backends.FunctionsList(functions_list.function_space)
    functions_list2.extend(functions_list)
    assert len(functions_list2) == 2
    for i in range(2):
        assert functions_list2[i] == functions_list[i]


def test_functions_list_len(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__len__."""
    assert len(functions_list) == 2


def test_functions_list_clear(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.clear."""
    functions_list.clear()
    assert len(functions_list) == 0


def test_functions_list_iter(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__iter__."""
    for (index, function) in enumerate(functions_list):
        assert np.allclose(function.vector.array, index + 1)


def test_functions_list_getitem_int(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__getitem__ with integer input."""
    assert np.allclose(functions_list[0].vector.array, 1)
    assert np.allclose(functions_list[1].vector.array, 2)


def test_functions_list_getitem_slice(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__getitem__ with slice input."""
    functions_list2 = functions_list[0:2]
    assert len(functions_list2) == 2
    assert np.allclose(functions_list2[0].vector.array, 1)
    assert np.allclose(functions_list2[1].vector.array, 2)


def test_functions_list_getitem_wrong_type(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        functions_list[0, 0]


def test_functions_list_setitem(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__setitem__."""
    V = functions_list.function_space
    new_function = dolfinx.fem.Function(V)
    new_function.vector.set(3)
    functions_list[0] = new_function

    assert np.allclose(functions_list[0].vector.array, 3)
    assert np.allclose(functions_list[1].vector.array, 2)


def test_functions_list_save_load(functions_list: minirox.backends.FunctionsList, tempdir: str) -> None:
    """Check I/O for a minirox.backends.FunctionsList."""
    functions_list.save(tempdir, "functions_list")

    V = functions_list.function_space
    functions_list2 = minirox.backends.FunctionsList(V)
    functions_list2.load(tempdir, "functions_list")

    assert len(functions_list2) == 2
    for (function, function2) in zip(functions_list, functions_list2):
        assert np.allclose(function2.vector.array, function.vector.array)


def test_functions_list_mul(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__mul__."""
    online_vec = petsc4py.PETSc.Vec().createSeq(2, comm=mpi4py.MPI.COMM_SELF)
    online_vec[0] = 3
    online_vec[1] = 5

    function = functions_list * online_vec
    assert np.allclose(function.vector.array, 13)


def test_functions_list_mul_empty(mesh: dolfinx.mesh.Mesh) -> None:
    """Check minirox.backends.FunctionsList.__mul__ with empty list."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    empty_functions_list = minirox.backends.FunctionsList(V)

    online_vec = petsc4py.PETSc.Vec().createSeq(0, comm=mpi4py.MPI.COMM_SELF)

    should_be_none = empty_functions_list * online_vec
    assert should_be_none is None


def test_functions_list_mul_not_implemented(functions_list: minirox.backends.FunctionsList) -> None:
    """Check minirox.backends.FunctionsList.__mul__ with an incorrect type."""
    with pytest.raises(TypeError):
        functions_list * None
