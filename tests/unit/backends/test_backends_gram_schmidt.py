# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.gram_schmidt module."""

import typing

import dolfinx.fem
import dolfinx.mesh
import mpi4py.MPI
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
    """Generate a list of pairwise linearly independent functions."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
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
def inner_product(mesh: dolfinx.mesh.Mesh) -> ufl.Form:  # type: ignore[no-any-unimported]
    """Generate a UFL form storing the L^2 inner product."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    return ufl.inner(u, v) * ufl.dx


def test_backends_gram_schmidt(  # type: ignore[no-any-unimported]
    functions: typing.List[dolfinx.fem.Function], inner_product: ufl.Form
) -> None:
    """Check rbnicsx.backends.gram_schmidt."""
    V = functions[0].function_space
    functions_list = rbnicsx.backends.FunctionsList(V)
    assert len(functions_list) == 0

    compute_inner_product = rbnicsx.backends.bilinear_form_action(inner_product)

    rbnicsx.backends.gram_schmidt(functions_list, functions[0], compute_inner_product)
    assert len(functions_list) == 1
    assert np.isclose(compute_inner_product(functions_list[0])(functions_list[0]), 1)
    assert np.allclose(functions_list[0].vector.array, 1)

    rbnicsx.backends.gram_schmidt(functions_list, functions[1], compute_inner_product)
    assert len(functions_list) == 2
    assert np.isclose(compute_inner_product(functions_list[0])(functions_list[0]), 1)
    assert np.isclose(compute_inner_product(functions_list[1])(functions_list[1]), 1)
    assert np.isclose(compute_inner_product(functions_list[0])(functions_list[1]), 0)
    assert np.allclose(functions_list[0].vector.array, 1)
    expected1 = dolfinx.fem.Function(V)
    expected1.interpolate(lambda x: 2 * x[0] + x[1] - 1.5)
    expected1.vector.scale(1 / np.sqrt(compute_inner_product(expected1)(expected1)))
    assert np.allclose(functions_list[1].vector.array, expected1.vector.array)


def test_backends_gram_schmidt_zero(  # type: ignore[no-any-unimported]
    mesh: dolfinx.mesh.Mesh, inner_product: ufl.Form
) -> None:
    """Check rbnicsx.backends.gram_schmidt when adding a linearly dependent function (e.g., zero)."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    functions_list = rbnicsx.backends.FunctionsList(V)
    assert len(functions_list) == 0

    compute_inner_product = rbnicsx.backends.bilinear_form_action(inner_product)

    zero = dolfinx.fem.Function(V)
    rbnicsx.backends.gram_schmidt(functions_list, zero, compute_inner_product)
    assert len(functions_list) == 0


def test_backends_gram_schmidt_block(  # type: ignore[no-any-unimported]
    functions: typing.List[dolfinx.fem.Function], inner_product: ufl.Form
) -> None:
    """Check rbnicsx.backends.gram_schmidt_block."""
    V = functions[0].function_space
    functions_lists = [rbnicsx.backends.FunctionsList(V) for _ in range(2)]
    for functions_list in functions_lists:
        assert len(functions_list) == 0

    compute_inner_product = rbnicsx.backends.block_diagonal_bilinear_form_action([inner_product, 2 * inner_product])

    rbnicsx.backends.gram_schmidt_block(
        functions_lists, [functions[0], functions[2]], compute_inner_product)
    for (functions_list, compute_inner_product_, factor) in zip(functions_lists, compute_inner_product, [1, 2]):
        assert len(functions_list) == 1
        assert np.isclose(compute_inner_product_(functions_list[0])(functions_list[0]), 1)
        assert np.allclose(functions_list[0].vector.array, 1 / np.sqrt(factor))

    rbnicsx.backends.gram_schmidt_block(
        functions_lists, [functions[1], functions[3]], compute_inner_product)
    for (functions_list, compute_inner_product_, factor, expected1_expr) in zip(
            functions_lists, compute_inner_product, [1, 2],
            [lambda x: 2 * x[0] + x[1] - 1.5, lambda x: x[0] + 2 * x[1] - 1.5]):
        assert len(functions_list) == 2
        assert np.isclose(compute_inner_product_(functions_list[0])(functions_list[0]), 1)
        assert np.allclose(functions_list[0].vector.array, 1 / np.sqrt(factor))
        assert np.isclose(compute_inner_product_(functions_list[1])(functions_list[1]), 1)
        assert np.isclose(compute_inner_product_(functions_list[0])(functions_list[1]), 0)
        expected1 = dolfinx.fem.Function(V)
        expected1.interpolate(expected1_expr)
        expected1.vector.scale(1 / np.sqrt(compute_inner_product_(expected1)(expected1)))
        assert np.allclose(functions_list[1].vector.array, expected1.vector.array)
