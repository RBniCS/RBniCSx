# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.proper_orthogonal_decomposition module."""

import typing

import dolfinx.fem
import dolfinx.mesh
import mpi4py
import numpy as np
import petsc4py
import pytest
import ufl

import rbnicsx.backends


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.fixture
def functions_list(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.FunctionsList:
    """Generate a rbnicsx.backends.FunctionsList with four linearly dependent entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    functions_list = rbnicsx.backends.FunctionsList(V)
    for i in range(4):
        function = dolfinx.fem.Function(V)
        with function.vector.localForm() as function_local:
            function_local.set(i + 1)
        functions_list.append(function)
    return functions_list


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


@pytest.fixture
def tensors_list_vec(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsList:
    """Generate a rbnicsx.backends.TensorsList with two linearly dependent petsc4py.PETSc.Vec entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(i + 1, v) * ufl.dx for i in range(2)]
    linear_forms_cpp = dolfinx.fem.form(linear_forms)
    vectors = [dolfinx.fem.assemble_vector(linear_form_cpp) for linear_form_cpp in linear_forms_cpp]
    [vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) for vector in vectors]
    tensors_list = rbnicsx.backends.TensorsList(linear_forms_cpp[0], mesh.comm)
    [tensors_list.append(vector) for vector in vectors]
    return tensors_list


@pytest.fixture
def tensors_list_mat(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsList:
    """Generate a rbnicsx.backends.TensorsList with two linearly dependent petsc4py.PETSc.Mat entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [(i + 1) * ufl.inner(u, v) * ufl.dx for i in range(2)]
    bilinear_forms_cpp = dolfinx.fem.form(bilinear_forms)
    matrices = [dolfinx.fem.assemble_matrix(bilinear_form_cpp) for bilinear_form_cpp in bilinear_forms_cpp]
    [matrix.assemble() for matrix in matrices]
    tensors_list = rbnicsx.backends.TensorsList(bilinear_forms_cpp[0], mesh.comm)
    [tensors_list.append(matrix) for matrix in matrices]
    return tensors_list


@pytest.mark.parametrize("normalize", [True, False])
def test_proper_orthogonal_decomposition_functions(
    functions_list: rbnicsx.backends.FunctionsList, inner_product: ufl.Form, normalize: bool
) -> None:
    """Check rbnicsx.backends.proper_orthogonal_decomposition for the case of dolfinx.fem.Function snapshots."""
    eigenvalues, modes, eigenvectors = rbnicsx.backends.proper_orthogonal_decomposition(
        functions_list[:2], inner_product, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    assert np.isclose(eigenvalues[0], 5)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.isclose(compute_inner_product(inner_product, modes[0], modes[0]), 1 if normalize else 5)
    if normalize:
        assert np.allclose(modes[0].vector.array, 1)
    # np.allclose(modes[2], 0) may not be true in arithmetic precision when scaling with a very small eigenvalue
    assert len(eigenvectors) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_proper_orthogonal_decomposition_functions_tol(
    functions_list: rbnicsx.backends.FunctionsList, inner_product: ufl.Form, normalize: bool
) -> None:
    """
    Check rbnicsx.backends.proper_orthogonal_decomposition for the case of dolfinx.fem.Function snapshots.

    The case of non zero tolerance is tested here.
    """
    eigenvalues, modes, eigenvectors = rbnicsx.backends.proper_orthogonal_decomposition(
        functions_list[:2], inner_product, N=2, tol=1e-8, normalize=normalize)
    assert len(eigenvalues) == 2
    assert np.isclose(eigenvalues[0], 5)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 1
    assert np.isclose(compute_inner_product(inner_product, modes[0], modes[0]), 1 if normalize else 5)
    if normalize:
        assert np.allclose(modes[0].vector.array, 1)
    assert len(eigenvectors) == 1


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "stopping_criterion_generator",
    [lambda arg: arg, lambda arg: [arg, arg]])
def test_proper_orthogonal_decomposition_block(
    functions_list: rbnicsx.backends.FunctionsList, inner_product: ufl.Form, normalize: bool,
    stopping_criterion_generator: typing.Callable
) -> None:
    """Check rbnicsx.backends.proper_orthogonal_decomposition_block."""
    eigenvalues, modes, eigenvectors = rbnicsx.backends.proper_orthogonal_decomposition_block(
        [functions_list[:2], functions_list[2:4]], [inner_product, 2 * inner_product],
        N=stopping_criterion_generator(2), tol=stopping_criterion_generator(0.0), normalize=normalize)
    assert len(eigenvalues) == 2
    for (component, eigenvalue_factor) in enumerate([1, 10]):
        assert len(eigenvalues[component]) == 2
        assert np.isclose(eigenvalues[component][0], 5 * eigenvalue_factor)
        assert np.isclose(eigenvalues[component][1], 0)
    assert len(modes) == 2
    for (component, (inner_product_factor, eigenvalue_factor, mode_factor)) in enumerate(zip(
            [1, 2], [1, 10], [1, 1 / np.sqrt(2)])):
        assert len(modes[component]) == 2
        assert np.isclose(
            compute_inner_product(inner_product_factor * inner_product, modes[component][0], modes[component][0]),
            1 if normalize else 5 * eigenvalue_factor)
        if normalize:
            assert np.allclose(modes[component][0].vector.array, mode_factor)
    assert len(eigenvectors) == 2
    for component in range(2):
        assert len(eigenvectors[component]) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_proper_orthogonal_decomposition_vectors(
    tensors_list_vec: rbnicsx.backends.TensorsList, normalize: bool
) -> None:
    """Check rbnicsx.backends.proper_orthogonal_decomposition for the case of petsc4py.PETSc.Vec snapshots."""
    eigenvalues, modes, eigenvectors = rbnicsx.backends.proper_orthogonal_decomposition(
        tensors_list_vec, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    assert eigenvalues[0] > 0
    assert not np.isclose(eigenvalues[0], 0)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.isclose(modes[0].norm(petsc4py.PETSc.NormType.NORM_2), 1 if normalize else np.sqrt(eigenvalues[0]))
    assert len(eigenvectors) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_proper_orthogonal_decomposition_matrices(
    tensors_list_mat: rbnicsx.backends.TensorsList, normalize: bool
) -> None:
    """Check rbnicsx.backends.proper_orthogonal_decomposition for the case of petsc4py.PETSc.Mat snapshots."""
    eigenvalues, modes, eigenvectors = rbnicsx.backends.proper_orthogonal_decomposition(
        tensors_list_mat, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    assert eigenvalues[0] > 0
    assert not np.isclose(eigenvalues[0], 0)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.isclose(modes[0].norm(petsc4py.PETSc.NormType.FROBENIUS), 1 if normalize else np.sqrt(eigenvalues[0]))
    assert len(eigenvectors) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_proper_orthogonal_decomposition_zero(
    mesh: dolfinx.mesh.Mesh, inner_product: ufl.Form, normalize: bool
) -> None:
    """Check rbnicsx.backends.proper_orthogonal_decomposition for the case of all zero snapshots."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    functions_list = rbnicsx.backends.FunctionsList(V)
    functions_list.extend([dolfinx.fem.Function(V) for _ in range(2)])
    eigenvalues, modes, eigenvectors = rbnicsx.backends.proper_orthogonal_decomposition(
        functions_list[:2], inner_product, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    assert np.isclose(eigenvalues[0], 0)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.allclose(modes[0].vector.array, 0)
    assert np.allclose(modes[1].vector.array, 0)
    assert len(eigenvectors) == 2


def test_proper_orthogonal_decomposition_wrong_iterable() -> None:
    """Check rbnicsx.backends.proper_orthogonal_decomposition raises when providing a plain list."""
    with pytest.raises(RuntimeError):
        rbnicsx.backends.proper_orthogonal_decomposition(list(), N=0, tol=0.0)
