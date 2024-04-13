# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.functions_list module."""

import pathlib

import dolfinx.fem
import dolfinx.mesh
import mpi4py.MPI
import nbvalx.tempfile
import numpy as np
import pytest

import rbnicsx.backends
import rbnicsx.online


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.fixture
def functions_list(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.FunctionsList:
    """Generate a rbnicsx.backends.FunctionsList with two entries."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    functions_list = rbnicsx.backends.FunctionsList(V)
    for i in range(2):
        function = dolfinx.fem.Function(V)
        with function.x.petsc_vec.localForm() as function_local:
            function_local.set(i + 1)
        functions_list.append(function)
    return functions_list


def test_backends_functions_list_function_space(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.function_space."""
    assert functions_list.function_space == functions_list[0].function_space


def test_backends_functions_list_comm(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.comm."""
    assert functions_list.comm == functions_list[0].function_space.mesh.comm


def test_backends_functions_list_duplicate(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.duplicate."""
    functions_list2 = functions_list.duplicate()
    assert len(functions_list2) == 0


def test_backends_functions_list_extend(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.extend."""
    functions_list2 = functions_list.duplicate()
    functions_list2.extend(functions_list)
    assert len(functions_list2) == 2
    for i in range(2):
        assert functions_list2[i] == functions_list[i]


def test_backends_functions_list_len(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__len__."""
    assert len(functions_list) == 2


def test_backends_functions_list_clear(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.clear."""
    functions_list.clear()
    assert len(functions_list) == 0


def test_backends_functions_list_iter(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__iter__."""
    for (index, function) in enumerate(functions_list):
        assert np.allclose(function.x.array, index + 1)


def test_backends_functions_list_getitem_int(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__getitem__ with integer input."""
    assert np.allclose(functions_list[0].x.array, 1)
    assert np.allclose(functions_list[1].x.array, 2)


def test_backends_functions_list_getitem_slice(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__getitem__ with slice input."""
    functions_list2 = functions_list[0:2]
    assert len(functions_list2) == 2
    assert np.allclose(functions_list2[0].x.array, 1)
    assert np.allclose(functions_list2[1].x.array, 2)


def test_backends_functions_list_getitem_wrong_type(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        functions_list[0, 0]  # type: ignore[call-overload]


def test_backends_functions_list_setitem(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__setitem__."""
    V = functions_list.function_space
    new_function = dolfinx.fem.Function(V)
    with new_function.x.petsc_vec.localForm() as new_function_local:
        new_function_local.set(3)
    functions_list[0] = new_function

    assert np.allclose(functions_list[0].x.array, 3)
    assert np.allclose(functions_list[1].x.array, 2)


def test_backends_functions_list_save_load(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check I/O for a rbnicsx.backends.FunctionsList."""
    with nbvalx.tempfile.TemporaryDirectory(functions_list.comm) as tempdir:
        functions_list.save(pathlib.Path(tempdir), "functions_list")

        functions_list2 = functions_list.duplicate()
        functions_list2.load(pathlib.Path(tempdir), "functions_list")

        assert len(functions_list2) == 2
        for (function, function2) in zip(functions_list, functions_list2):
            assert np.allclose(function2.x.array, function.x.array)


def test_backends_functions_list_mul(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__mul__."""
    online_vec = rbnicsx.online.create_vector(2)
    online_vec[0] = 3
    online_vec[1] = 5

    function = functions_list * online_vec
    assert np.allclose(function.x.array, 13)


def test_backends_functions_list_mul_empty(mesh: dolfinx.mesh.Mesh) -> None:
    """Check rbnicsx.backends.FunctionsList.__mul__ with empty list."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    empty_functions_list = rbnicsx.backends.FunctionsList(V)

    online_vec = rbnicsx.online.create_vector(0)

    should_be_zero_function = empty_functions_list * online_vec
    assert np.allclose(should_be_zero_function.x.array, 0)


def test_backends_functions_list_mul_not_implemented(functions_list: rbnicsx.backends.FunctionsList) -> None:
    """Check rbnicsx.backends.FunctionsList.__mul__ with an incorrect type."""
    with pytest.raises(TypeError):
        functions_list * None
