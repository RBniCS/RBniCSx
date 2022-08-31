# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.proper_orthogonal_decomposition module."""

import typing

import _pytest.fixtures
import numpy as np
import petsc4py.PETSc
import pytest

import rbnicsx.online


@pytest.fixture
def functions_list_plain() -> rbnicsx.online.FunctionsList:
    """Generate a rbnicsx.online.FunctionsList with two petsc4py.PETSc.Vec entries."""
    vectors = [rbnicsx.online.create_vector(3) for _ in range(4)]
    for (v, vector) in enumerate(vectors):
        for i in range(3):
            vector.setValue(i, (v + 1) * (i + 1))
    functions_list = rbnicsx.online.FunctionsList(3)
    for vector in vectors:
        functions_list.append(vector)
    return functions_list


@pytest.fixture
def functions_list_block() -> rbnicsx.online.FunctionsList:
    """Generate a rbnicsx.online.FunctionsList with two petsc4py.PETSc.Vec entries (block version)."""
    vectors = [rbnicsx.online.create_vector_block([3, 4]) for _ in range(4)]
    for (v, vector) in enumerate(vectors):
        for i in range(7):
            vector.setValue(i, (v + 1) * (i + 1))
    functions_list = rbnicsx.online.FunctionsList([3, 4])
    for vector in vectors:
        functions_list.append(vector)
    return functions_list


@pytest.fixture(params=["functions_list_plain", "functions_list_block"])
def functions_list(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.FunctionsList:
    """Parameterize rbnicsx.online.FunctionsList considering either non-block or block content."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture
def inner_product() -> typing.Callable[[int], petsc4py.PETSc.Mat]:  # type: ignore[no-any-unimported]
    """Return a callable that computes the identity matrix."""
    def _(N: int) -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
        """Return the identity matrix."""
        identity = rbnicsx.online.create_matrix(N, N)
        for i in range(N):
            identity.setValue(i, i, 1)
        identity.assemble()
        return identity

    return _


def compute_inner_product(  # type: ignore[no-any-unimported]
    inner_product: typing.Callable[[int], petsc4py.PETSc.Mat],
    function_i: petsc4py.PETSc.Vec, function_j: petsc4py.PETSc.Vec
) -> petsc4py.PETSc.ScalarType:
    """Evaluate the inner product between two functions."""
    inner_product_action = rbnicsx.online.matrix_action(inner_product)
    return inner_product_action(function_i)(function_j)


@pytest.fixture
def tensors_list_vec_plain() -> rbnicsx.online.TensorsList:
    """Generate a rbnicsx.online.TensorsList with two petsc4py.PETSc.Vec entries."""
    vectors = [rbnicsx.online.create_vector(3) for _ in range(2)]
    for (v, vector) in enumerate(vectors):
        for i in range(3):
            vector.setValue(i, (v + 1) * (i + 1))
    tensors_list = rbnicsx.online.TensorsList(3)
    for vector in vectors:
        tensors_list.append(vector)
    return tensors_list


@pytest.fixture
def tensors_list_vec_block() -> rbnicsx.online.TensorsList:
    """Generate a rbnicsx.online.TensorsList with two petsc4py.PETSc.Vec entries (block version)."""
    vectors = [rbnicsx.online.create_vector_block([3, 4]) for _ in range(2)]
    for (v, vector) in enumerate(vectors):
        for i in range(7):
            vector.setValue(i, (v + 1) * (i + 1))
    tensors_list = rbnicsx.online.TensorsList([3, 4])
    for vector in vectors:
        tensors_list.append(vector)
    return tensors_list


@pytest.fixture(params=["tensors_list_vec_plain", "tensors_list_vec_block"])
def tensors_list_vec(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsList:
    """Parameterize rbnicsx.online.TensorsList on petsc4py.PETSc.Vec (block version or not)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture
def tensors_list_mat_plain() -> rbnicsx.online.TensorsList:
    """Generate a rbnicsx.online.TensorsList with two petsc4py.PETSc.Mat entries."""
    matrices = [rbnicsx.online.create_matrix(4, 3) for _ in range(2)]
    for (m, matrix) in enumerate(matrices):
        for i in range(4):
            for j in range(3):
                matrix.setValue(i, j, (m + 1) * (i * 3 + j + 1))
        matrix.assemble()
    tensors_list = rbnicsx.online.TensorsList((4, 3))
    for matrix in matrices:
        tensors_list.append(matrix)
    return tensors_list


@pytest.fixture
def tensors_list_mat_block() -> rbnicsx.online.TensorsList:
    """Generate a rbnicsx.online.TensorsList with two petsc4py.PETSc.Mat entries (block version)."""
    matrices = [rbnicsx.online.create_matrix_block([7, 3], [4, 5]) for _ in range(2)]
    for (m, matrix) in enumerate(matrices):
        for i in range(10):
            for j in range(9):
                matrix.setValue(i, j, (m + 1) * (i * 9 + j + 1))
        matrix.assemble()
    tensors_list = rbnicsx.online.TensorsList(([7, 3], [4, 5]))
    for matrix in matrices:
        tensors_list.append(matrix)
    return tensors_list


@pytest.fixture(params=["tensors_list_mat_plain", "tensors_list_mat_block"])
def tensors_list_mat(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsList:
    """Parameterize rbnicsx.online.TensorsList on petsc4py.PETSc.Mat (block version or not)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.mark.parametrize("normalize", [True, False])
def test_online_proper_orthogonal_decomposition_functions(  # type: ignore[no-any-unimported]
    functions_list: rbnicsx.online.FunctionsList, inner_product: typing.Callable[[int], petsc4py.PETSc.Mat],
    normalize: bool
) -> None:
    """Check rbnicsx.online.proper_orthogonal_decomposition for the case of snapshots stored in a FunctionsList."""
    size = functions_list[0].size
    inner_product_matrix = inner_product(size)
    eigenvalues, modes, eigenvectors = rbnicsx.online.proper_orthogonal_decomposition(
        functions_list[:2], inner_product_matrix, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    sum_squares_first_size_numbers = size * (size + 1) * (2 * size + 1) / 6
    assert np.isclose(eigenvalues[0], 5 * sum_squares_first_size_numbers)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.isclose(
        compute_inner_product(inner_product_matrix, modes[0], modes[0]),
        1 if normalize else 5 * sum_squares_first_size_numbers)
    if normalize:
        assert np.allclose(modes[0].array, 1 / np.sqrt(sum_squares_first_size_numbers) * np.arange(1, size + 1))
    # np.allclose(modes[2], 0) may not be true in arithmetic precision when scaling with a very small eigenvalue
    assert len(eigenvectors) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_online_proper_orthogonal_decomposition_functions_tol(  # type: ignore[no-any-unimported]
    functions_list: rbnicsx.online.FunctionsList, inner_product: typing.Callable[[int], petsc4py.PETSc.Mat],
    normalize: bool
) -> None:
    """
    Check rbnicsx.online.proper_orthogonal_decomposition for the case of snapshots stored in a FunctionsList.

    The case of non zero tolerance is tested here.
    """
    size = functions_list[0].size
    inner_product_matrix = inner_product(size)
    eigenvalues, modes, eigenvectors = rbnicsx.online.proper_orthogonal_decomposition(
        functions_list[:2], inner_product_matrix, N=2, tol=1e-8, normalize=normalize)
    assert len(eigenvalues) == 2
    sum_squares_first_size_numbers = size * (size + 1) * (2 * size + 1) / 6
    assert np.isclose(eigenvalues[0], 5 * sum_squares_first_size_numbers)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 1
    assert np.isclose(
        compute_inner_product(inner_product_matrix, modes[0], modes[0]),
        1 if normalize else 5 * sum_squares_first_size_numbers)
    if normalize:
        assert np.allclose(modes[0].array, 1 / np.sqrt(sum_squares_first_size_numbers) * np.arange(1, size + 1))
    assert len(eigenvectors) == 1


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "stopping_criterion_generator",
    [lambda arg: arg, lambda arg: [arg, arg]])
def test_online_proper_orthogonal_decomposition_block(  # type: ignore[no-any-unimported]
    functions_list: rbnicsx.online.FunctionsList, inner_product: typing.Callable[[int], petsc4py.PETSc.Mat],
    normalize: bool, stopping_criterion_generator: typing.Callable[
        [typing.Any], typing.Union[typing.Any, typing.Tuple[typing.Any, typing.Any]]]
) -> None:
    """Check rbnicsx.online.proper_orthogonal_decomposition_block."""
    size = functions_list[0].size
    inner_product_matrix = inner_product(size)
    eigenvalues, modes, eigenvectors = rbnicsx.online.proper_orthogonal_decomposition_block(
        [functions_list[:2], functions_list[2:4]], [inner_product_matrix, 2 * inner_product_matrix],
        N=stopping_criterion_generator(2), tol=stopping_criterion_generator(0.0),  # type: ignore[arg-type]
        normalize=normalize)
    assert len(eigenvalues) == 2
    sum_squares_first_size_numbers = size * (size + 1) * (2 * size + 1) / 6
    for (component, eigenvalue_factor) in enumerate([1, 10]):
        assert len(eigenvalues[component]) == 2
        assert np.isclose(eigenvalues[component][0], 5 * sum_squares_first_size_numbers * eigenvalue_factor)
        assert np.isclose(eigenvalues[component][1], 0)
    assert len(modes) == 2
    for (component, (inner_product_factor, eigenvalue_factor, mode_factor)) in enumerate(zip(
            [1, 2], [1, 10], [1, 1 / np.sqrt(2)])):
        assert len(modes[component]) == 2
        assert np.isclose(
            compute_inner_product(
                inner_product_factor * inner_product_matrix, modes[component][0], modes[component][0]),
            1 if normalize else 5 * sum_squares_first_size_numbers * eigenvalue_factor)
        if normalize:
            assert np.allclose(
                modes[component][0].array,
                mode_factor / np.sqrt(sum_squares_first_size_numbers) * np.arange(1, size + 1))
    assert len(eigenvectors) == 2
    for component in range(2):
        assert len(eigenvectors[component]) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_online_proper_orthogonal_decomposition_vectors(
    tensors_list_vec: rbnicsx.online.TensorsList, normalize: bool
) -> None:
    """Check rbnicsx.online.proper_orthogonal_decomposition for the case of petsc4py.PETSc.Vec snapshots."""
    eigenvalues, modes, eigenvectors = rbnicsx.online.proper_orthogonal_decomposition(
        tensors_list_vec, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    assert eigenvalues[0] > 0
    assert not np.isclose(eigenvalues[0], 0)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.isclose(modes[0].norm(petsc4py.PETSc.NormType.NORM_2), 1 if normalize else np.sqrt(eigenvalues[0]))
    assert len(eigenvectors) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_online_proper_orthogonal_decomposition_matrices(
    tensors_list_mat: rbnicsx.online.TensorsList, normalize: bool
) -> None:
    """Check rbnicsx.online.proper_orthogonal_decomposition for the case of petsc4py.PETSc.Mat snapshots."""
    eigenvalues, modes, eigenvectors = rbnicsx.online.proper_orthogonal_decomposition(
        tensors_list_mat, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    assert eigenvalues[0] > 0
    assert not np.isclose(eigenvalues[0], 0)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.isclose(modes[0].norm(petsc4py.PETSc.NormType.FROBENIUS), 1 if normalize else np.sqrt(eigenvalues[0]))
    assert len(eigenvectors) == 2


@pytest.mark.parametrize("normalize", [True, False])
def test_online_proper_orthogonal_decomposition_zero(  # type: ignore[no-any-unimported]
    inner_product: typing.Callable[[int], petsc4py.PETSc.Mat], normalize: bool
) -> None:
    """Check rbnicsx.online.proper_orthogonal_decomposition for the case of all zero snapshots."""
    functions_list = rbnicsx.online.FunctionsList(3)
    functions_list.extend([rbnicsx.online.create_vector(3) for _ in range(2)])
    inner_product_matrix = inner_product(3)
    eigenvalues, modes, eigenvectors = rbnicsx.online.proper_orthogonal_decomposition(
        functions_list[:2], inner_product_matrix, N=2, tol=0.0, normalize=normalize)
    assert len(eigenvalues) == 2
    assert np.isclose(eigenvalues[0], 0)
    assert np.isclose(eigenvalues[1], 0)
    assert len(modes) == 2
    assert np.allclose(modes[0].array, 0)
    assert np.allclose(modes[1].array, 0)
    assert len(eigenvectors) == 2


def test_online_proper_orthogonal_decomposition_wrong_iterable() -> None:
    """Check rbnicsx.online.proper_orthogonal_decomposition raises when providing a plain list."""
    with pytest.raises(RuntimeError):
        rbnicsx.online.proper_orthogonal_decomposition(list(), N=0, tol=0.0)  # type: ignore[call-overload]
