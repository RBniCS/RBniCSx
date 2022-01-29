# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.gram_schmidt module."""

import typing

import dolfinx.fem
import dolfinx.mesh
import mpi4py
import numpy as np
import pytest
import ufl

import rbnicsx.backends


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.fixture
def functions(mesh: dolfinx.mesh.Mesh) -> typing.List[dolfinx.fem.Function]:
    """Generate a list of two linearly independent functions."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    function0 = dolfinx.fem.Function(V)
    with function0.vector.localForm() as function0_local:
        function0_local.set(1)
    function1 = dolfinx.fem.Function(V)
    function1.interpolate(lambda x: 2 * x[0] + x[1])
    function2 = dolfinx.fem.Function(V)
    with function2.vector.localForm() as function2_local:
        function2_local.set(2)
    function3 = dolfinx.fem.Function(V)
    function3.interpolate(lambda x: x[0] + 2 * x[1])
    return [function0, function1, function2, function3]


@pytest.fixture
def inner_product(mesh: dolfinx.mesh.Mesh) -> ufl.Form:
    """Generate a UFL form storing the L^2 inner product."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    return ufl.inner(u, v) * ufl.dx


def compute_inner_product(
    inner_product: ufl.Form, function_i: dolfinx.fem.Function, function_j: dolfinx.fem.Function
) -> float:
    """Evaluate the inner product between two functions."""
    comm = function_i.function_space.mesh.comm
    test, trial = inner_product.arguments()
    return comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.replace(inner_product, {test: function_i, trial: function_j}))),
        op=mpi4py.MPI.SUM)


def test_gram_schmidt(functions: typing.List[dolfinx.fem.Function], inner_product: ufl.Form) -> None:
    """Check rbnicsx.backends.gram_schmidt."""
    V = functions[0].function_space
    functions_list = rbnicsx.backends.FunctionsList(V)
    assert len(functions_list) == 0

    rbnicsx.backends.gram_schmidt(functions_list, functions[0], inner_product)
    assert len(functions_list) == 1
    assert np.isclose(compute_inner_product(inner_product, functions_list[0], functions_list[0]), 1)
    assert np.allclose(functions_list[0].vector.array, 1)

    rbnicsx.backends.gram_schmidt(functions_list, functions[1], inner_product)
    assert len(functions_list) == 2
    assert np.isclose(compute_inner_product(inner_product, functions_list[0], functions_list[0]), 1)
    assert np.isclose(compute_inner_product(inner_product, functions_list[1], functions_list[1]), 1)
    assert np.isclose(compute_inner_product(inner_product, functions_list[0], functions_list[1]), 0)
    assert np.allclose(functions_list[0].vector.array, 1)
    expected1 = dolfinx.fem.Function(V)
    expected1.interpolate(lambda x: 2 * x[0] + x[1] - 1.5)
    expected1.vector.scale(1 / np.sqrt(compute_inner_product(inner_product, expected1, expected1)))
    assert np.allclose(functions_list[1].vector.array, expected1.vector.array)


def test_gram_schmidt_zero(mesh: dolfinx.mesh.Mesh, inner_product: ufl.Form) -> None:
    """Check rbnicsx.backends.gram_schmidt when adding a linearly dependent function (e.g., zero)."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))

    functions_list = rbnicsx.backends.FunctionsList(V)
    assert len(functions_list) == 0

    zero = dolfinx.fem.Function(V)
    rbnicsx.backends.gram_schmidt(functions_list, zero, inner_product)
    assert len(functions_list) == 0


def test_gram_schmidt_block(functions: typing.List[dolfinx.fem.Function], inner_product: ufl.Form) -> None:
    """Check rbnicsx.backends.gram_schmidt_block."""
    V = functions[0].function_space
    functions_lists = [rbnicsx.backends.FunctionsList(V) for _ in range(2)]
    for functions_list in functions_lists:
        assert len(functions_list) == 0

    rbnicsx.backends.gram_schmidt_block(
        functions_lists, [functions[0], functions[2]], [inner_product, 2 * inner_product])
    for (functions_list, factor) in zip(functions_lists, [1, 2]):
        assert len(functions_list) == 1
        assert np.isclose(compute_inner_product(factor * inner_product, functions_list[0], functions_list[0]), 1)
        assert np.allclose(functions_list[0].vector.array, 1 / np.sqrt(factor))

    rbnicsx.backends.gram_schmidt_block(
        functions_lists, [functions[1], functions[3]], [inner_product, 2 * inner_product])
    for (functions_list, factor, expected1_expr) in zip(
            functions_lists, [1, 2], [lambda x: 2 * x[0] + x[1] - 1.5, lambda x: x[0] + 2 * x[1] - 1.5]):
        assert len(functions_list) == 2
        assert np.isclose(compute_inner_product(factor * inner_product, functions_list[0], functions_list[0]), 1)
        assert np.allclose(functions_list[0].vector.array, 1 / np.sqrt(factor))
        assert np.isclose(compute_inner_product(factor * inner_product, functions_list[1], functions_list[1]), 1)
        assert np.isclose(compute_inner_product(factor * inner_product, functions_list[0], functions_list[1]), 0)
        assert np.allclose(functions_list[0].vector.array, 1 / np.sqrt(factor))
        expected1 = dolfinx.fem.Function(V)
        expected1.interpolate(expected1_expr)
        expected1.vector.scale(1 / np.sqrt(compute_inner_product(factor * inner_product, expected1, expected1)))
        assert np.allclose(functions_list[1].vector.array, expected1.vector.array)
