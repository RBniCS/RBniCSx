# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.projection module."""

import typing

import numpy as np
import petsc4py.PETSc
import pytest

import rbnicsx.online


@pytest.fixture
def functions_list() -> rbnicsx.online.FunctionsList:
    """Generate a rbnicsx.online.FunctionsList with several petsc4py.PETSc.Vec entries of dimension 30."""
    vectors = [rbnicsx.online.create_vector(30) for _ in range(14)]
    for (v, vector) in enumerate(vectors):
        for i in range(30):
            vector.setValue(i, (v + 1) * (i + 1))
    functions_list = rbnicsx.online.FunctionsList(3)
    for vector in vectors:
        functions_list.append(vector)
    return functions_list


@pytest.fixture
def linear_form() -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
    """Generate a petsc4py.PETSc.Vec representing a linear form in the 30-dimensional reduced basis space."""
    vector = rbnicsx.online.create_vector(30)
    for i in range(30):
        vector.setValue(i, i + 1)
    return vector


@pytest.fixture
def linear_form_block() -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
    """Generate a petsc4py.PETSc.Vec representing a linear block form in a 2x30-dimensional reduced basis space."""
    vector = rbnicsx.online.create_vector_block([30, 30])
    for i in range(30):
        vector.setValue(i, i + 1)
    for i in range(30, 60):
        vector.setValue(i, 10 * (i - 30 + 1))
    return vector


@pytest.fixture
def bilinear_form() -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
    """Generate a petsc4py.PETSc.Mat representing a bilinear form in the 30-dimensional reduced basis space."""
    matrix = rbnicsx.online.create_matrix(30, 30)
    for i in range(30):
        matrix.setValue(i, i, i + 1)
    matrix.assemble()
    return matrix


@pytest.fixture
def bilinear_form_block() -> petsc4py.PETSc.Mat:  # type: ignore[no-any-unimported]
    """Generate a petsc4py.PETSc.Mat representing a bilinear block form in the 2x30-dimensional reduced basis space."""
    matrix = rbnicsx.online.create_matrix_block([30, 30], [30, 30])
    for I in range(2):  # noqa: E741
        for J in range(2):
            for i in range(30):
                matrix.setValue(I * 30 + i, J * 30 + i, 10**I * (-1)**J * (i + 1))
    matrix.assemble()
    return matrix


def test_online_projection_vector(  # type: ignore[no-any-unimported]
    linear_form: petsc4py.PETSc.Vec, functions_list: rbnicsx.online.FunctionsList
) -> None:
    """Test projection of a linear form onto the reduced basis."""
    N = linear_form.size
    basis_vectors = functions_list[:2]

    online_vec = rbnicsx.online.project_vector(linear_form, basis_vectors)
    assert online_vec.size == 2
    sum_squares_first_N_numbers = N * (N + 1) * (2 * N + 1) / 6
    assert np.allclose(online_vec.array, np.array([1, 2]) * sum_squares_first_N_numbers)

    online_vec2 = rbnicsx.online.project_vector(0.4 * linear_form, basis_vectors)
    rbnicsx.online.project_vector(online_vec2, 0.6 * linear_form, basis_vectors)
    assert online_vec2.size == 2
    assert np.allclose(online_vec2.array, online_vec.array)


def test_online_projection_vector_block(  # type: ignore[no-any-unimported]
    linear_form_block: petsc4py.PETSc.Vec, functions_list: rbnicsx.online.FunctionsList
) -> None:
    """Test projection of a list of linear forms onto the reduced basis."""
    N = linear_form_block.size / 2
    basis_vectors = [functions_list[:2], functions_list[2:5]]

    online_vec = rbnicsx.online.project_vector_block(linear_form_block, basis_vectors)
    assert online_vec.size == 5
    sum_squares_first_N_numbers = N * (N + 1) * (2 * N + 1) / 6
    assert np.allclose(online_vec[0:2], np.array([1, 2]) * sum_squares_first_N_numbers)
    assert np.allclose(online_vec[2:5], np.array([3, 4, 5]) * sum_squares_first_N_numbers * 10)

    online_vec2 = rbnicsx.online.project_vector_block(0.4 * linear_form_block, basis_vectors)
    rbnicsx.online.project_vector_block(online_vec2, 0.6 * linear_form_block, basis_vectors)
    assert online_vec2.size == 5
    assert np.allclose(online_vec2.array, online_vec.array)


def test_online_projection_matrix_galerkin(  # type: ignore[no-any-unimported]
    bilinear_form: petsc4py.PETSc.Mat, functions_list: rbnicsx.online.FunctionsList,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Test projection of a bilinear form onto the reduced basis (for use in Galerkin methods)."""
    N = bilinear_form.size[0]
    assert bilinear_form.size[1] == N
    basis_vectors = functions_list[:2]

    online_mat = rbnicsx.online.project_matrix(bilinear_form, basis_vectors)
    assert online_mat.size == (2, 2)
    sum_cubes_first_N_numbers = N**2 * (N + 1)**2 / 4
    assert np.allclose(online_mat[0, :], np.array([1, 2]) * sum_cubes_first_N_numbers)
    assert np.allclose(online_mat[1, :], np.array([1, 2]) * sum_cubes_first_N_numbers * 2)

    online_mat2 = rbnicsx.online.project_matrix(0.4 * bilinear_form, basis_vectors)
    rbnicsx.online.project_matrix(online_mat2, 0.6 * bilinear_form, basis_vectors)
    assert online_mat2.size == (2, 2)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))


def test_online_projection_matrix_petrov_galerkin(  # type: ignore[no-any-unimported]
    bilinear_form: petsc4py.PETSc.Mat, functions_list: rbnicsx.online.FunctionsList,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Test projection of a bilinear form onto the reduced basis (for use in Petrov-Galerkin methods)."""
    N = bilinear_form.size[0]
    assert bilinear_form.size[1] == N
    basis_vectors = (functions_list[:2], functions_list[2:5])

    online_mat = rbnicsx.online.project_matrix(bilinear_form, basis_vectors)
    assert online_mat.size == (2, 3)
    sum_cubes_first_N_numbers = N**2 * (N + 1)**2 / 4
    assert np.allclose(online_mat[0, :], np.array([3, 4, 5]) * sum_cubes_first_N_numbers)
    assert np.allclose(online_mat[1, :], np.array([3, 4, 5]) * sum_cubes_first_N_numbers * 2)

    online_mat2 = rbnicsx.online.project_matrix(0.4 * bilinear_form, basis_vectors)
    rbnicsx.online.project_matrix(online_mat2, 0.6 * bilinear_form, basis_vectors)
    assert online_mat2.size == (2, 3)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))


def test_online_projection_matrix_block_galerkin(  # type: ignore[no-any-unimported]
    bilinear_form_block: petsc4py.PETSc.Mat, functions_list: rbnicsx.online.FunctionsList,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Test projection of a matrix of bilinear forms onto the reduced basis (for use in Galerkin methods)."""
    N = bilinear_form_block.size[0] / 2
    assert bilinear_form_block.size[1] / 2 == N
    basis_vectors = [functions_list[:2], functions_list[2:5]]

    online_mat = rbnicsx.online.project_matrix_block(bilinear_form_block, basis_vectors)
    assert online_mat.size == (5, 5)
    sum_cubes_first_N_numbers = N**2 * (N + 1)**2 / 4
    assert np.allclose(online_mat[0, 0:2], np.array([1, 2]) * sum_cubes_first_N_numbers)
    assert np.allclose(online_mat[0, 2:5], np.array([3, 4, 5]) * sum_cubes_first_N_numbers * -1)
    assert np.allclose(online_mat[1, 0:2], np.array([1, 2]) * sum_cubes_first_N_numbers * 2)
    assert np.allclose(online_mat[1, 2:5], np.array([3, 4, 5]) * sum_cubes_first_N_numbers * -2)
    assert np.allclose(online_mat[2, 0:2], np.array([1, 2]) * sum_cubes_first_N_numbers * 30)
    assert np.allclose(online_mat[2, 2:5], np.array([3, 4, 5]) * sum_cubes_first_N_numbers * -30)
    assert np.allclose(online_mat[3, 0:2], np.array([1, 2]) * sum_cubes_first_N_numbers * 40)
    assert np.allclose(online_mat[3, 2:5], np.array([3, 4, 5]) * sum_cubes_first_N_numbers * -40)
    assert np.allclose(online_mat[4, 0:2], np.array([1, 2]) * sum_cubes_first_N_numbers * 50)
    assert np.allclose(online_mat[4, 2:5], np.array([3, 4, 5]) * sum_cubes_first_N_numbers * -50)

    online_mat2 = rbnicsx.online.project_matrix_block(0.4 * bilinear_form_block, basis_vectors)
    rbnicsx.online.project_matrix_block(online_mat2, 0.6 * bilinear_form_block, basis_vectors)
    assert online_mat2.size == (5, 5)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))


def test_online_projection_matrix_block_petrov_galerkin(  # type: ignore[no-any-unimported]
    bilinear_form_block: petsc4py.PETSc.Mat, functions_list: rbnicsx.online.FunctionsList,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Test projection of a matrix of bilinear forms onto the reduced basis (for use in Petrov-Galerkin methods)."""
    N = bilinear_form_block.size[0] / 2
    assert bilinear_form_block.size[1] / 2 == N
    basis_vectors = ([functions_list[:2], functions_list[2:5]], [functions_list[5:9], functions_list[9:14]])

    online_mat = rbnicsx.online.project_matrix_block(bilinear_form_block, basis_vectors)
    assert online_mat.size == (5, 9)
    sum_cubes_first_N_numbers = N**2 * (N + 1)**2 / 4
    assert np.allclose(online_mat[0, 0:4], np.array([6, 7, 8, 9]) * sum_cubes_first_N_numbers)
    assert np.allclose(online_mat[0, 4:9], np.array([10, 11, 12, 13, 14]) * sum_cubes_first_N_numbers * -1)
    assert np.allclose(online_mat[1, 0:4], np.array([6, 7, 8, 9]) * sum_cubes_first_N_numbers * 2)
    assert np.allclose(online_mat[1, 4:9], np.array([10, 11, 12, 13, 14]) * sum_cubes_first_N_numbers * -2)
    assert np.allclose(online_mat[2, 0:4], np.array([6, 7, 8, 9]) * sum_cubes_first_N_numbers * 30)
    assert np.allclose(online_mat[2, 4:9], np.array([10, 11, 12, 13, 14]) * sum_cubes_first_N_numbers * -30)
    assert np.allclose(online_mat[3, 0:4], np.array([6, 7, 8, 9]) * sum_cubes_first_N_numbers * 40)
    assert np.allclose(online_mat[3, 4:9], np.array([10, 11, 12, 13, 14]) * sum_cubes_first_N_numbers * -40)
    assert np.allclose(online_mat[4, 0:4], np.array([6, 7, 8, 9]) * sum_cubes_first_N_numbers * 50)
    assert np.allclose(online_mat[4, 4:9], np.array([10, 11, 12, 13, 14]) * sum_cubes_first_N_numbers * -50)

    online_mat2 = rbnicsx.online.project_matrix_block(0.4 * bilinear_form_block, basis_vectors)
    rbnicsx.online.project_matrix_block(online_mat2, 0.6 * bilinear_form_block, basis_vectors)
    assert online_mat2.size == (5, 9)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))
