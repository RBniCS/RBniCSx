# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.tensors_list module."""

import typing

import _pytest.fixtures
import numpy as np
import pytest

import rbnicsx.online


@pytest.fixture
def tensors_list_vec_plain() -> rbnicsx.online.TensorsList:
    """Generate a rbnicsx.online.TensorsList with two petsc4py.PETSc.Vec entries."""
    vectors = [rbnicsx.online.create_vector(3) for _ in range(2)]
    for (v, vector) in enumerate(vectors):
        for i in range(3):
            vector.setValue(i, (v + 1) * (i + 1))
    tensors_list = rbnicsx.online.TensorsList(3)
    [tensors_list.append(vector) for vector in vectors]
    tensors_list.first_vector = vectors[0]
    return tensors_list


@pytest.fixture
def tensors_list_vec_block() -> rbnicsx.online.TensorsList:
    """Generate a rbnicsx.online.TensorsList with two petsc4py.PETSc.Vec entries (block version)."""
    vectors = [rbnicsx.online.create_vector_block([3, 4]) for _ in range(2)]
    for (v, vector) in enumerate(vectors):
        for i in range(7):
            vector.setValue(i, (v + 1) * (i + 1))
    tensors_list = rbnicsx.online.TensorsList([3, 4])
    [tensors_list.append(vector) for vector in vectors]
    tensors_list.first_vector = vectors[0]
    return tensors_list


@pytest.fixture(params=["tensors_list_vec_plain", "tensors_list_vec_block"])
def tensors_list_vec(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsList:
    """Parameterize rbnicsx.online.TensorsList on petsc4py.PETSc.Vec (block version or not)."""
    return request.getfixturevalue(request.param)


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
    [tensors_list.append(matrix) for matrix in matrices]
    tensors_list.first_matrix = matrices[0]
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
    [tensors_list.append(matrix) for matrix in matrices]
    tensors_list.first_matrix = matrices[0]
    return tensors_list


@pytest.fixture(params=["tensors_list_vec_plain", "tensors_list_mat_plain"])
def tensors_list_plain(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsList:
    """Parameterize rbnicsx.online.TensorsList on petsc4py.PETSc.Mat and petsc4py.PETSc.Vec (not block version)."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["tensors_list_vec_block", "tensors_list_mat_block"])
def tensors_list_block(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsList:
    """Parameterize rbnicsx.online.TensorsList on petsc4py.PETSc.Mat and petsc4py.PETSc.Vec (block version)."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["tensors_list_mat_plain", "tensors_list_mat_block"])
def tensors_list_mat(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsList:
    """Parameterize rbnicsx.online.TensorsList on petsc4py.PETSc.Mat (block version or not)."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[
    "tensors_list_vec_plain", "tensors_list_vec_block", "tensors_list_mat_plain", "tensors_list_mat_block"
])
def tensors_list(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsList:
    """Parameterize rbnicsx.online.TensorsList on petsc4py.PETSc.Mat and petsc4py.PETSc.Vec."""
    return request.getfixturevalue(request.param)


def test_online_tensors_list_shape_vec_plain(tensors_list_vec_plain: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.shape in the case of petsc4py.PETSc.Vec content (non-block version)."""
    assert isinstance(tensors_list_vec_plain.shape, int)
    assert tensors_list_vec_plain.shape == 3


def test_online_tensors_list_shape_vec_block(tensors_list_vec_block: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.shape in the case of petsc4py.PETSc.Vec content (block version)."""
    assert isinstance(tensors_list_vec_block.shape, list)
    assert all([isinstance(shape_, int) for shape_ in tensors_list_vec_block.shape])
    assert len(tensors_list_vec_block.shape) == 2
    assert tensors_list_vec_block.shape[0] == 3
    assert tensors_list_vec_block.shape[1] == 4


def test_online_tensors_list_shape_mat_plain(tensors_list_mat_plain: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.shape in the case of petsc4py.PETSc.Mat content (non-block version)."""
    assert isinstance(tensors_list_mat_plain.shape, tuple)
    assert all([isinstance(shape_, int) for shape_ in tensors_list_mat_plain.shape])
    assert len(tensors_list_mat_plain.shape) == 2
    assert tensors_list_mat_plain.shape[0] == 4
    assert tensors_list_mat_plain.shape[1] == 3


def test_online_tensors_list_shape_mat_block(tensors_list_mat_block: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.shape in the case of petsc4py.PETSc.Mat content (block version)."""
    assert isinstance(tensors_list_mat_block.shape, tuple)
    assert all([isinstance(shape_, list) for shape_ in tensors_list_mat_block.shape])
    assert all([isinstance(shape__, int) for shape_ in tensors_list_mat_block.shape for shape__ in shape_])
    assert len(tensors_list_mat_block.shape) == 2
    assert all([len(shape_) == 2 for shape_ in tensors_list_mat_block.shape])
    assert tensors_list_mat_block.shape[0][0] == 7
    assert tensors_list_mat_block.shape[0][1] == 3
    assert tensors_list_mat_block.shape[1][0] == 4
    assert tensors_list_mat_block.shape[1][1] == 5


def test_online_tensors_list_type_vec(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.type in the case of petsc4py.PETSc.Vec content."""
    tensors_list_vec.type == "Vec"


def test_online_tensors_list_type_mat(tensors_list_mat: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.type in the case of petsc4py.PETSc.Mat content."""
    tensors_list_mat.type == "Mat"


def test_online_tensors_list_type_none(tensors_list: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.type at initialization."""
    empty_tensors_list = rbnicsx.online.TensorsList(0)
    assert empty_tensors_list.type is None


def test_online_tensors_list_is_block_plain(tensors_list_plain: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.is_block in the case of non-block tensors."""
    tensors_list_plain.is_block is False


def test_online_tensors_list_is_block_block(tensors_list_block: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.is_block in the case of block tensors."""
    tensors_list_block.is_block is True


def test_online_tensors_list_append_mixed_types(
        tensors_list_vec: rbnicsx.online.TensorsList, tensors_list_mat: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.append mixing up Mat and Vec objects."""
    first_vector = rbnicsx.online.create_vector(1)
    first_matrix = rbnicsx.online.create_matrix(1, 1)

    with pytest.raises(AssertionError):
        tensors_list_vec.append(first_matrix)

    with pytest.raises(AssertionError):
        tensors_list_mat.append(first_vector)


def test_online_tensors_list_append_wrong_type(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.append when providing neither a Mat nor a Vec object."""
    with pytest.raises(RuntimeError):
        tensors_list_vec.append(None)


def test_online_tensors_list_duplicate(tensors_list: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.duplicate."""
    tensors_list2 = tensors_list.duplicate()
    assert len(tensors_list2) == 0


def test_online_tensors_list_extend(tensors_list: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.extend."""
    tensors_list2 = tensors_list.duplicate()
    tensors_list2.extend(tensors_list)
    assert len(tensors_list2) == 2
    for i in range(2):
        assert tensors_list2[i] == tensors_list[i]


def test_online_tensors_list_len(tensors_list: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.__len__."""
    assert len(tensors_list) == 2


def test_online_tensors_list_clear(tensors_list: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.clear."""
    tensors_list.clear()
    assert len(tensors_list) == 0


def test_online_tensors_list_iter_vec(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.__iter__ in the case of petsc4py.PETSc.Vec content."""
    for (index, vector) in enumerate(tensors_list_vec):
        assert np.allclose(vector.array, (index + 1) * tensors_list_vec.first_vector.array)


def test_online_tensors_list_iter_mat(
    tensors_list_mat: rbnicsx.online.TensorsList, to_dense_matrix: typing.Callable
) -> None:
    """Check rbnicsx.online.TensorsList.__iter__ in the case of petsc4py.PETSc.Mat content."""
    for (index, matrix) in enumerate(tensors_list_mat):
        assert np.allclose(to_dense_matrix(matrix), (index + 1) * to_dense_matrix(tensors_list_mat.first_matrix))


def test_online_tensors_list_getitem_int_vec(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.__getitem__ with integer input in the case of petsc4py.PETSc.Vec content."""
    assert np.allclose(tensors_list_vec[0].array, tensors_list_vec.first_vector.array)
    assert np.allclose(tensors_list_vec[1].array, 2 * tensors_list_vec.first_vector.array)


def test_online_tensors_list_getitem_int_mat(
    tensors_list_mat: rbnicsx.online.TensorsList, to_dense_matrix: typing.Callable
) -> None:
    """Check rbnicsx.online.TensorsList.__getitem__ with integer input in the case of petsc4py.PETSc.Mat content."""
    assert np.allclose(to_dense_matrix(tensors_list_mat[0]), to_dense_matrix(tensors_list_mat.first_matrix))
    assert np.allclose(to_dense_matrix(tensors_list_mat[1]), 2 * to_dense_matrix(tensors_list_mat.first_matrix))


def test_online_tensors_list_getitem_slice_vec(
    tensors_list_vec: rbnicsx.online.TensorsList, to_dense_matrix: typing.Callable
) -> None:
    """Check rbnicsx.online.TensorsList.__getitem__ with slice input in the case of petsc4py.PETSc.Vec content."""
    tensors_list_vec2 = tensors_list_vec[0:2]
    assert len(tensors_list_vec2) == 2
    assert np.allclose(tensors_list_vec2[0].array, tensors_list_vec.first_vector.array)
    assert np.allclose(tensors_list_vec2[1].array, 2 * tensors_list_vec.first_vector.array)


def test_online_tensors_list_getitem_slice_mat(
    tensors_list_mat: rbnicsx.online.TensorsList, to_dense_matrix: typing.Callable
) -> None:
    """Check rbnicsx.online.TensorsList.__getitem__ with slice input in the case of petsc4py.PETSc.Mat content."""
    tensors_list_mat2 = tensors_list_mat[0:2]
    assert len(tensors_list_mat2) == 2
    assert np.allclose(to_dense_matrix(tensors_list_mat2[0]), to_dense_matrix(tensors_list_mat.first_matrix))
    assert np.allclose(to_dense_matrix(tensors_list_mat2[1]), 2 * to_dense_matrix(tensors_list_mat.first_matrix))


def test_online_tensors_list_getitem_wrong_type(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        tensors_list_vec[0, 0]


def test_online_tensors_list_setitem_vec(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.__setitem__ in the case of petsc4py.PETSc.Vec content."""
    tensors_list_vec[0] = 3 * tensors_list_vec.first_vector
    assert np.allclose(tensors_list_vec[0].array, 3 * tensors_list_vec.first_vector.array)
    assert np.allclose(tensors_list_vec[1].array, 2 * tensors_list_vec.first_vector.array)


def test_online_tensors_list_setitem_mat(
    tensors_list_mat: rbnicsx.online.TensorsList, to_dense_matrix: typing.Callable
) -> None:
    """Check rbnicsx.online.TensorsList.__setitem__ in the case of petsc4py.PETSc.Mat content."""
    tensors_list_mat[0] = 3 * tensors_list_mat.first_matrix
    assert np.allclose(to_dense_matrix(tensors_list_mat[0]), 3 * to_dense_matrix(tensors_list_mat.first_matrix))
    assert np.allclose(to_dense_matrix(tensors_list_mat[1]), 2 * to_dense_matrix(tensors_list_mat.first_matrix))


def test_online_tensors_list_save_load_vec(tensors_list_vec: rbnicsx.online.TensorsList, tempdir: str) -> None:
    """Check I/O for a rbnicsx.online.TensorsList in the case of petsc4py.PETSc.Vec content."""
    tensors_list_vec.save(tempdir, "tensors_list_vec")

    tensors_list_vec2 = tensors_list_vec.duplicate()
    tensors_list_vec2.load(tempdir, "tensors_list_vec")

    assert len(tensors_list_vec2) == 2
    for (vector, vector2) in zip(tensors_list_vec, tensors_list_vec2):
        assert np.allclose(vector2.array, vector.array)


def test_online_tensors_list_save_load_mat(
    tensors_list_mat: rbnicsx.online.TensorsList, tempdir: str, to_dense_matrix: typing.Callable
) -> None:
    """Check I/O for a rbnicsx.online.TensorsList in the case of petsc4py.PETSc.Mat content."""
    tensors_list_mat.save(tempdir, "tensors_list_mat")

    tensors_list_mat2 = tensors_list_mat.duplicate()
    tensors_list_mat2.load(tempdir, "tensors_list_mat")

    assert len(tensors_list_mat2) == 2
    for (matrix, matrix2) in zip(tensors_list_mat, tensors_list_mat2):
        assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))


def test_online_tensors_list_save_load_empty(tempdir: str) -> None:
    """Check I/O for rbnicsx.online.TensorsList when providing neither a Mat nor a Vec object."""
    empty_tensors_list = rbnicsx.online.TensorsList(0)

    with pytest.raises(RuntimeError):
        empty_tensors_list.save(tempdir, "empty_tensors_list")

    with pytest.raises(RuntimeError):
        empty_tensors_list.load(tempdir, "empty_tensors_list")


def test_online_tensors_list_mul_vec(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.__mul__ in the case of petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(2)
    online_vec[0] = 3
    online_vec[1] = 5

    vector = tensors_list_vec * online_vec
    assert np.allclose(vector.array, 13 * tensors_list_vec.first_vector.array)


def test_online_tensors_list_mul_mat(
    tensors_list_mat: rbnicsx.online.TensorsList, to_dense_matrix: typing.Callable
) -> None:
    """Check rbnicsx.online.TensorsList.__mul__ in the case of petsc4py.PETSc.Mat content."""
    online_vec = rbnicsx.online.create_vector(2)
    online_vec[0] = 3
    online_vec[1] = 5

    matrix = tensors_list_mat * online_vec
    assert np.allclose(to_dense_matrix(matrix), 13 * to_dense_matrix(tensors_list_mat.first_matrix))


def test_online_tensors_list_mul_empty() -> None:
    """Check rbnicsx.online.TensorsList.__mul__ with empty list."""
    empty_tensors_list = rbnicsx.online.TensorsList(0)

    online_vec = rbnicsx.online.create_vector(0)
    should_be_none = empty_tensors_list * online_vec
    assert should_be_none is None


def test_online_tensors_list_mul_not_implemented(tensors_list_vec: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.__mul__ with an incorrect type."""
    with pytest.raises(TypeError):
        tensors_list_vec * None
