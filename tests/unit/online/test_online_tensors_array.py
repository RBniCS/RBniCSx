# Copyright (C) 2021-2026 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.tensors_array module."""

import pathlib
import typing

import _pytest.fixtures
import nbvalx.tempfile
import numpy as np
import numpy.typing as npt
import petsc4py.PETSc
import pytest

import rbnicsx.online


@pytest.fixture
def tensors_1d_array_vec_plain() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with six petsc4py.PETSc.Vec entries."""
    vectors = [rbnicsx.online.create_vector(7) for _ in range(6)]
    for (v, vector) in enumerate(vectors):
        for d in range(7):
            vector.setValue(d, (v + 1) * (d + 1))
    tensors_1d_array = rbnicsx.online.TensorsArray(7, 6)
    for (i, vector) in enumerate(vectors):
        tensors_1d_array[i] = vector
    setattr(tensors_1d_array, "first_vector", vectors[0])
    return tensors_1d_array


@pytest.fixture
def tensors_1d_array_vec_block() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with six petsc4py.PETSc.Vec entries (block version)."""
    vectors = [rbnicsx.online.create_vector_block([3, 4]) for _ in range(6)]
    for (v, vector) in enumerate(vectors):
        for d in range(7):
            vector.setValue(d, (v + 1) * (d + 1))
    tensors_1d_array = rbnicsx.online.TensorsArray([3, 4], 6)
    for (i, vector) in enumerate(vectors):
        tensors_1d_array[i] = vector
    setattr(tensors_1d_array, "first_vector", vectors[0])
    return tensors_1d_array


@pytest.fixture
def tensors_2d_array_vec_plain() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with two-by-three petsc4py.PETSc.Vec entries."""
    vectors = [rbnicsx.online.create_vector(7) for _ in range(6)]
    for (v, vector) in enumerate(vectors):
        for d in range(7):
            vector.setValue(d, (v + 1) * (d + 1))
    tensors_2d_array = rbnicsx.online.TensorsArray(7, (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = vectors[i * 3 + j]
    setattr(tensors_2d_array, "first_vector", vectors[0])
    return tensors_2d_array


@pytest.fixture
def tensors_2d_array_vec_block() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with two-by-three petsc4py.PETSc.Vec entries (block version)."""
    vectors = [rbnicsx.online.create_vector_block([3, 4]) for _ in range(6)]
    for (v, vector) in enumerate(vectors):
        for d in range(7):
            vector.setValue(d, (v + 1) * (d + 1))
    tensors_2d_array = rbnicsx.online.TensorsArray([3, 4], (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = vectors[i * 3 + j]
    setattr(tensors_2d_array, "first_vector", vectors[0])
    return tensors_2d_array


@pytest.fixture
def tensors_1d_array_mat_plain() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with six petsc4py.PETSc.Mat entries."""
    matrices = [rbnicsx.online.create_matrix(10, 9) for _ in range(6)]
    for (m, matrix) in enumerate(matrices):
        for d in range(10):
            for e in range(9):
                matrix.setValue(d, e, (m + 1) * (d * 9 + e + 1))
        matrix.assemble()
    tensors_1d_array = rbnicsx.online.TensorsArray((10, 9), 6)
    for (i, matrix) in enumerate(matrices):
        tensors_1d_array[i] = matrix
    setattr(tensors_1d_array, "first_matrix", matrices[0])
    return tensors_1d_array


@pytest.fixture
def tensors_1d_array_mat_block() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with six petsc4py.PETSc.Mat entries (block version)."""
    matrices = [rbnicsx.online.create_matrix_block([7, 3], [4, 5]) for _ in range(6)]
    for (m, matrix) in enumerate(matrices):
        for d in range(10):
            for e in range(9):
                matrix.setValue(d, e, (m + 1) * (d * 9 + e + 1))
        matrix.assemble()
    tensors_1d_array = rbnicsx.online.TensorsArray(([7, 3], [4, 5]), 6)
    for (i, matrix) in enumerate(matrices):
        tensors_1d_array[i] = matrix
    setattr(tensors_1d_array, "first_matrix", matrices[0])
    return tensors_1d_array


@pytest.fixture
def tensors_2d_array_mat_plain() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with two-by-three petsc4py.PETSc.Mat entries."""
    matrices = [rbnicsx.online.create_matrix(10, 9) for _ in range(6)]
    for (m, matrix) in enumerate(matrices):
        for d in range(10):
            for e in range(9):
                matrix.setValue(d, e, (m + 1) * (d * 9 + e + 1))
        matrix.assemble()
    tensors_2d_array = rbnicsx.online.TensorsArray((10, 9), (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = matrices[i * 3 + j]
    setattr(tensors_2d_array, "first_matrix", matrices[0])
    return tensors_2d_array


@pytest.fixture
def tensors_2d_array_mat_block() -> rbnicsx.online.TensorsArray:
    """Generate a rbnicsx.online.TensorsArray with two-by-three petsc4py.PETSc.Mat entries (block version)."""
    matrices = [rbnicsx.online.create_matrix_block([7, 3], [4, 5]) for _ in range(6)]
    for (m, matrix) in enumerate(matrices):
        for d in range(10):
            for e in range(9):
                matrix.setValue(d, e, (m + 1) * (d * 9 + e + 1))
        matrix.assemble()
    tensors_2d_array = rbnicsx.online.TensorsArray(([7, 3], [4, 5]), (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = matrices[i * 3 + j]
    setattr(tensors_2d_array, "first_matrix", matrices[0])
    return tensors_2d_array


@pytest.fixture(params=["tensors_1d_array_vec_plain", "tensors_1d_array_vec_block"])
def tensors_1d_array_vec(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray with six petsc4py.PETSc.Vec entries (block version or not)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=["tensors_2d_array_vec_plain", "tensors_2d_array_vec_block"])
def tensors_2d_array_vec(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray with two-by-three petsc4py.PETSc.Vec entries (block version or not)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=["tensors_1d_array_mat_plain", "tensors_1d_array_mat_block"])
def tensors_1d_array_mat(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray with six petsc4py.PETSc.Mat entries (block version or not)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=["tensors_2d_array_mat_plain", "tensors_2d_array_mat_block"])
def tensors_2d_array_mat(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray with two-by-three petsc4py.PETSc.mat entries (block version or not)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=[
    "tensors_1d_array_vec_plain", "tensors_1d_array_vec_block", "tensors_2d_array_vec_plain",
    "tensors_2d_array_vec_block"
])
def tensors_array_vec(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray on array shape, with petsc4py.PETSc.Vec content."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=[
    "tensors_1d_array_mat_plain", "tensors_1d_array_mat_block", "tensors_2d_array_mat_plain",
    "tensors_2d_array_mat_block"
])
def tensors_array_mat(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray on array shape, with petsc4py.PETSc.Mat content."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=[
    "tensors_1d_array_vec_plain", "tensors_2d_array_vec_plain", "tensors_1d_array_mat_plain",
    "tensors_2d_array_mat_plain"
])
def tensors_array_plain(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray on petsc4py.PETSc.Mat and petsc4py.PETSc.Vec (not block version)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=[
    "tensors_1d_array_vec_block", "tensors_2d_array_vec_block", "tensors_1d_array_mat_block",
    "tensors_2d_array_mat_block"
])
def tensors_array_block(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray on petsc4py.PETSc.Mat and petsc4py.PETSc.Vec (block version)."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=[
    "tensors_1d_array_vec_plain", "tensors_1d_array_vec_block", "tensors_2d_array_vec_plain",
    "tensors_2d_array_vec_block", "tensors_1d_array_mat_plain", "tensors_1d_array_mat_block",
    "tensors_2d_array_mat_plain", "tensors_2d_array_mat_block"
])
def tensors_array(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.TensorsArray:
    """Parameterize rbnicsx.online.TensorsArray on array shape and petsc4py.PETSc.Mat and petsc4py.PETSc.Vec."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


def test_online_tensors_array_shape_1d_vec_plain(tensors_1d_array_vec_plain: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 1d array, petsc4py.PETSc.Vec non-block content."""
    assert isinstance(tensors_1d_array_vec_plain.shape, tuple)
    assert tensors_1d_array_vec_plain.shape == (6, )
    assert isinstance(tensors_1d_array_vec_plain.content_shape, int)
    assert tensors_1d_array_vec_plain.content_shape == 7
    assert isinstance(tensors_1d_array_vec_plain.flattened_shape, tuple)
    assert tensors_1d_array_vec_plain.flattened_shape == (6, 7)


def test_online_tensors_array_shape_1d_vec_block(tensors_1d_array_vec_block: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 1d array, petsc4py.PETSc.Vec block content."""
    assert isinstance(tensors_1d_array_vec_block.shape, tuple)
    assert tensors_1d_array_vec_block.shape == (6, )
    assert isinstance(tensors_1d_array_vec_block.content_shape, list)
    assert tensors_1d_array_vec_block.content_shape == [3, 4]
    assert isinstance(tensors_1d_array_vec_block.flattened_shape, tuple)
    assert tensors_1d_array_vec_block.flattened_shape == (6, [3, 4])


def test_online_tensors_array_shape_2d_vec_plain(tensors_2d_array_vec_plain: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 2d array, petsc4py.PETSc.Vec non-block content."""
    assert isinstance(tensors_2d_array_vec_plain.shape, tuple)
    assert tensors_2d_array_vec_plain.shape == (2, 3)
    assert isinstance(tensors_2d_array_vec_plain.content_shape, int)
    assert tensors_2d_array_vec_plain.content_shape == 7
    assert isinstance(tensors_2d_array_vec_plain.flattened_shape, tuple)
    assert tensors_2d_array_vec_plain.flattened_shape == (2, 3, 7)


def test_online_tensors_array_shape_2d_vec_block(tensors_2d_array_vec_block: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 2d array, petsc4py.PETSc.Vec block content."""
    assert isinstance(tensors_2d_array_vec_block.shape, tuple)
    assert tensors_2d_array_vec_block.shape == (2, 3)
    assert isinstance(tensors_2d_array_vec_block.content_shape, list)
    assert tensors_2d_array_vec_block.content_shape == [3, 4]
    assert isinstance(tensors_2d_array_vec_block.flattened_shape, tuple)
    assert tensors_2d_array_vec_block.flattened_shape == (2, 3, [3, 4])


def test_online_tensors_array_shape_1d_mat_plain(tensors_1d_array_mat_plain: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 1d array, petsc4py.PETSc.Mat non-block content."""
    assert isinstance(tensors_1d_array_mat_plain.shape, tuple)
    assert tensors_1d_array_mat_plain.shape == (6, )
    assert isinstance(tensors_1d_array_mat_plain.content_shape, tuple)
    assert tensors_1d_array_mat_plain.content_shape == (10, 9)
    assert isinstance(tensors_1d_array_mat_plain.flattened_shape, tuple)
    assert tensors_1d_array_mat_plain.flattened_shape == (6, 10, 9)


def test_online_tensors_array_shape_1d_mat_block(tensors_1d_array_mat_block: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 1d array, petsc4py.PETSc.Mat block content."""
    assert isinstance(tensors_1d_array_mat_block.shape, tuple)
    assert tensors_1d_array_mat_block.shape == (6, )
    assert isinstance(tensors_1d_array_mat_block.content_shape, tuple)
    assert tensors_1d_array_mat_block.content_shape == ([7, 3], [4, 5])
    assert isinstance(tensors_1d_array_mat_block.flattened_shape, tuple)
    assert tensors_1d_array_mat_block.flattened_shape == (6, [7, 3], [4, 5])


def test_online_tensors_array_shape_2d_mat_plain(tensors_2d_array_mat_plain: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 2d array, petsc4py.PETSc.Mat non-block content."""
    assert isinstance(tensors_2d_array_mat_plain.shape, tuple)
    assert tensors_2d_array_mat_plain.shape == (2, 3)
    assert isinstance(tensors_2d_array_mat_plain.content_shape, tuple)
    assert tensors_2d_array_mat_plain.content_shape == (10, 9)
    assert isinstance(tensors_2d_array_mat_plain.flattened_shape, tuple)
    assert tensors_2d_array_mat_plain.flattened_shape == (2, 3, 10, 9)


def test_online_tensors_array_shape_2d_mat_block(tensors_2d_array_mat_block: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.*shape in the case of 2d array, petsc4py.PETSc.Mat block content."""
    assert isinstance(tensors_2d_array_mat_block.shape, tuple)
    assert tensors_2d_array_mat_block.shape == (2, 3)
    assert isinstance(tensors_2d_array_mat_block.content_shape, tuple)
    assert tensors_2d_array_mat_block.content_shape == ([7, 3], [4, 5])
    assert isinstance(tensors_2d_array_mat_block.flattened_shape, tuple)
    assert tensors_2d_array_mat_block.flattened_shape == (2, 3, [7, 3], [4, 5])


def test_online_tensors_array_type_vec(tensors_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.type in the case of petsc4py.PETSc.Vec content."""
    tensors_array_vec.type == "Vec"


def test_online_tensors_array_type_mat(tensors_array_mat: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.type in the case of petsc4py.PETSc.Mat content."""
    tensors_array_mat.type == "Mat"


def test_online_tensors_array_is_block_plain(tensors_array_plain: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.is_block in the case of non-block tensors."""
    tensors_array_plain.is_block is False


def test_online_tensors_array_is_block_block(tensors_array_block: rbnicsx.online.TensorsList) -> None:
    """Check rbnicsx.online.TensorsList.is_block in the case of block tensors."""
    tensors_array_block.is_block is True


def test_online_tensors_array_type_none() -> None:
    """Check rbnicsx.online.TensorsArray.type at initialization."""
    empty_tensors_array = rbnicsx.online.TensorsArray(0, 0)
    assert empty_tensors_array.type is None


def test_online_tensors_array_duplicate(tensors_array: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.duplicate."""
    tensors_array2 = tensors_array.duplicate()
    assert tensors_array2.shape == tensors_array.shape
    assert tensors_array2.content_shape == tensors_array.content_shape
    assert all([tensor is not None for tensor in tensors_array._array.flat])
    assert all([tensor is None for tensor in tensors_array2._array.flat])


def test_online_tensors_array_getitem_1d_int_vec(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with integer input, 1d array and petsc4py.PETSc.Vec content."""
    first_vector = getattr(tensors_1d_array_vec, "first_vector")
    for i in range(6):
        assert np.allclose(tensors_1d_array_vec[i].array, (i + 1) * first_vector.array)


def test_online_tensors_array_getitem_1d_tuple_int_vec(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with wrong index, 1d array and petsc4py.PETSc.Vec content."""
    with pytest.raises(IndexError) as excinfo:
        tensors_1d_array_vec[0, 0].array
    assert str(excinfo.value) == "too many indices for array: array is 1-dimensional, but 2 were indexed"


def test_online_tensors_array_getitem_2d_tuple_int_vec(tensors_2d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with integer input, 2d array and petsc4py.PETSc.Vec content."""
    first_vector = getattr(tensors_2d_array_vec, "first_vector")
    for i in range(2):
        for j in range(3):
            assert np.allclose(
                tensors_2d_array_vec[i, j].array, (i * 3 + j + 1) * first_vector.array)


def test_online_tensors_array_getitem_1d_int_mat(
    tensors_1d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with integer input, 1d array and petsc4py.PETSc.Mat content."""
    first_matrix = getattr(tensors_1d_array_mat, "first_matrix")
    for i in range(6):
        assert np.allclose(
            to_dense_matrix(tensors_1d_array_mat[i]), (i + 1) * to_dense_matrix(first_matrix))


def test_online_tensors_array_getitem_2d_tuple_int_mat(
    tensors_2d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with integer input, 2d array and petsc4py.PETSc.Mat content."""
    first_matrix = getattr(tensors_2d_array_mat, "first_matrix")
    for i in range(2):
        for j in range(3):
            assert np.allclose(
                to_dense_matrix(tensors_2d_array_mat[i, j]),
                (i * 3 + j + 1) * to_dense_matrix(first_matrix))


def test_online_tensors_array_getitem_1d_slice_vec(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with slice input, 1d array and petsc4py.PETSc.Vec content."""
    tensors_1d_array_vec2 = tensors_1d_array_vec[0:2]
    assert tensors_1d_array_vec2.shape == (2, )
    first_vector = getattr(tensors_1d_array_vec, "first_vector")
    assert np.allclose(tensors_1d_array_vec2[0].array, first_vector.array)
    assert np.allclose(tensors_1d_array_vec2[1].array, 2 * first_vector.array)


def test_online_tensors_array_getitem_2d_slice_vec(tensors_2d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with slice input, 2d array and petsc4py.PETSc.Vec content."""
    tensors_2d_array_vec2 = tensors_2d_array_vec[0:1, 0:2]
    assert tensors_2d_array_vec2.shape == (1, 2)
    first_vector = getattr(tensors_2d_array_vec, "first_vector")
    assert np.allclose(tensors_2d_array_vec2[0, 0].array, first_vector.array)
    assert np.allclose(tensors_2d_array_vec2[0, 1].array, 2 * first_vector.array)


def test_online_tensors_array_getitem_1d_slice_mat(
    tensors_1d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with slice input, 1d array and petsc4py.PETSc.Mat content."""
    tensors_1d_array_mat2 = tensors_1d_array_mat[0:2]
    assert tensors_1d_array_mat2.shape == (2, )
    first_matrix = getattr(tensors_1d_array_mat, "first_matrix")
    assert np.allclose(
        to_dense_matrix(tensors_1d_array_mat2[0]), to_dense_matrix(first_matrix))
    assert np.allclose(
        to_dense_matrix(tensors_1d_array_mat2[1]), 2 * to_dense_matrix(first_matrix))


def test_online_tensors_array_getitem_2d_slice_mat(
    tensors_2d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with slice input, 1d array and petsc4py.PETSc.Mat content."""
    tensors_2d_array_mat2 = tensors_2d_array_mat[0:1, 0:2]
    assert tensors_2d_array_mat2.shape == (1, 2)
    first_matrix = getattr(tensors_2d_array_mat, "first_matrix")
    assert np.allclose(
        to_dense_matrix(tensors_2d_array_mat2[0, 0]), to_dense_matrix(first_matrix))
    assert np.allclose(
        to_dense_matrix(tensors_2d_array_mat2[0, 1]), 2 * to_dense_matrix(first_matrix))


def test_online_tensors_array_getitem_wrong_type(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        tensors_1d_array_vec[""]  # type: ignore[call-overload]


def test_online_tensors_array_setitem_1d_vec(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__setitem__ in the case of 1d array and petsc4py.PETSc.Vec content."""
    first_vector = getattr(tensors_1d_array_vec, "first_vector")
    tensors_1d_array_vec[0] = 7 * first_vector
    for i in range(6):
        if i == 0:
            coeff = 7
        else:
            coeff = i + 1
        assert np.allclose(tensors_1d_array_vec[i].array, coeff * first_vector.array)


def test_online_tensors_array_setitem_2d_vec(tensors_2d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__setitem__ in the case of 2d array and petsc4py.PETSc.Vec content."""
    first_vector = getattr(tensors_2d_array_vec, "first_vector")
    tensors_2d_array_vec[0, 0] = 7 * first_vector
    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0:
                coeff = 7
            else:
                coeff = i * 3 + j + 1
            assert np.allclose(tensors_2d_array_vec[i, j].array, coeff * first_vector.array)


def test_online_tensors_array_setitem_1d_mat(
    tensors_1d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.__setitem__ in the case of 1d array and petsc4py.PETSc.Mat content."""
    first_matrix = getattr(tensors_1d_array_mat, "first_matrix")
    tensors_1d_array_mat[0] = 7 * first_matrix
    for i in range(6):
        if i == 0:
            coeff = 7
        else:
            coeff = i + 1
        assert np.allclose(
            to_dense_matrix(tensors_1d_array_mat[i]), coeff * to_dense_matrix(first_matrix))


def test_online_tensors_array_setitem_2d_mat(
    tensors_2d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.__setitem__ in the case of 1d array and petsc4py.PETSc.Mat content."""
    first_matrix = getattr(tensors_2d_array_mat, "first_matrix")
    tensors_2d_array_mat[0, 0] = 7 * first_matrix
    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0:
                coeff = 7
            else:
                coeff = i * 3 + j + 1
            assert np.allclose(
                to_dense_matrix(tensors_2d_array_mat[i, j]), coeff * to_dense_matrix(first_matrix))


def test_online_tensors_array_setitem_mixed_types(
    tensors_1d_array_vec: rbnicsx.online.TensorsArray, tensors_1d_array_mat: rbnicsx.online.TensorsArray
) -> None:
    """Check rbnicsx.online.TensorsArray.__setitem__ mixing up Mat and Vec objects."""
    with pytest.raises(AssertionError):
        tensors_1d_array_vec[0] = getattr(tensors_1d_array_mat, "first_matrix")

    with pytest.raises(AssertionError):
        tensors_1d_array_mat[0] = getattr(tensors_1d_array_vec, "first_vector")


def test_online_tensors_array_setitem_wrong_type(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.__setitem__ when providing neither a Mat nor a Vec object."""
    with pytest.raises(RuntimeError):
        tensors_1d_array_vec[0] = None


def test_online_tensors_array_save_load_1d_vec(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check I/O for a rbnicsx.online.TensorsArray in the case of 1d array and petsc4py.PETSc.Vec content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_1d_array_vec.comm) as tempdir:
        tensors_1d_array_vec.save(pathlib.Path(tempdir), "tensors_1d_array_vec")

        tensors_1d_array_vec2 = tensors_1d_array_vec.duplicate()
        tensors_1d_array_vec2.load(pathlib.Path(tempdir), "tensors_1d_array_vec")

        assert tensors_1d_array_vec2.shape == (6, )
        for i in range(6):
            assert np.allclose(tensors_1d_array_vec2[i].array, tensors_1d_array_vec[i].array)


def test_online_tensors_array_save_load_2d_vec(tensors_2d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check I/O for a rbnicsx.online.TensorsArray in the case of 2d array and petsc4py.PETSc.Vec content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_2d_array_vec.comm) as tempdir:
        tensors_2d_array_vec.save(pathlib.Path(tempdir), "tensors_2d_array_vec")

        tensors_2d_array_vec2 = tensors_2d_array_vec.duplicate()
        tensors_2d_array_vec2.load(pathlib.Path(tempdir), "tensors_2d_array_vec")

        assert tensors_2d_array_vec2.shape == (2, 3)
        for i in range(2):
            for j in range(3):
                assert np.allclose(tensors_2d_array_vec2[i, j].array, tensors_2d_array_vec[i, j].array)


def test_online_tensors_array_save_load_1d_mat(
    tensors_1d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check I/O for a rbnicsx.online.TensorsArray in the case of 1d array and petsc4py.PETSc.Mat content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_1d_array_mat.comm) as tempdir:
        tensors_1d_array_mat.save(pathlib.Path(tempdir), "tensors_1d_array_mat")

        tensors_1d_array_mat2 = tensors_1d_array_mat.duplicate()
        tensors_1d_array_mat2.load(pathlib.Path(tempdir), "tensors_1d_array_mat")

        assert tensors_1d_array_mat2.shape == (6, )
        for i in range(6):
            assert np.allclose(to_dense_matrix(tensors_1d_array_mat2[i]), to_dense_matrix(tensors_1d_array_mat[i]))


def test_online_tensors_array_save_load_2d_mat(
    tensors_2d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check I/O for a rbnicsx.online.TensorsArray in the case of 2d array and petsc4py.PETSc.Mat content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_2d_array_mat.comm) as tempdir:
        tensors_2d_array_mat.save(pathlib.Path(tempdir), "tensors_2d_array_mat")

        tensors_2d_array_mat2 = tensors_2d_array_mat.duplicate()
        tensors_2d_array_mat2.load(pathlib.Path(tempdir), "tensors_2d_array_mat")

        assert tensors_2d_array_mat2.shape == (2, 3)
        for i in range(2):
            for j in range(3):
                assert np.allclose(
                    to_dense_matrix(tensors_2d_array_mat2[i, j]), to_dense_matrix(tensors_2d_array_mat[i, j]))


def test_online_tensors_array_save_load_empty() -> None:
    """Check I/O for rbnicsx.online.TensorsArray when providing neither a Mat nor a Vec object."""
    empty_tensors_array = rbnicsx.online.TensorsArray(0, 0)

    with nbvalx.tempfile.TemporaryDirectory(empty_tensors_array.comm) as tempdir:
        with pytest.raises(RuntimeError):
            empty_tensors_array.save(pathlib.Path(tempdir), "empty_tensors_array")

        with pytest.raises(RuntimeError):
            empty_tensors_array.load(pathlib.Path(tempdir), "empty_tensors_array")


def test_online_tensors_array_contraction_1d_vec(tensors_1d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 1d array and petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    first_vector = getattr(tensors_1d_array_vec, "first_vector")
    contraction = tensors_1d_array_vec.contraction(online_vec, first_vector)
    assert np.isclose(
        contraction, 91 * first_vector.norm(petsc4py.PETSc.NormType.NORM_2)**2)  # type: ignore[attr-defined]


def test_online_tensors_array_contraction_2d_vec(tensors_2d_array_vec: rbnicsx.online.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 2d array and petsc4py.PETSc.Vec content."""
    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]

    first_vector = getattr(tensors_2d_array_vec, "first_vector")
    contraction = tensors_2d_array_vec.contraction(online_vec0, online_vec1, first_vector)
    assert np.isclose(
        contraction, 150 * first_vector.norm(petsc4py.PETSc.NormType.NORM_2)**2)  # type: ignore[attr-defined]


def test_online_tensors_array_contraction_1d_mat(
    tensors_1d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 1d array and petsc4py.PETSc.Mat content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)
    online_solution0 = rbnicsx.online.create_vector(10)
    online_solution0[:] = np.arange(1, 11)
    online_solution1 = rbnicsx.online.create_vector(9)
    online_solution1[:] = np.arange(1, 10)

    first_matrix = getattr(tensors_1d_array_mat, "first_matrix")
    first_matrix_action = rbnicsx.online.matrix_action(first_matrix)
    contraction = tensors_1d_array_mat.contraction(online_vec, online_solution0, online_solution1)
    assert np.isclose(contraction, 91 * first_matrix_action(online_solution1)(online_solution0))


def test_online_tensors_array_contraction_2d_mat(
    tensors_2d_array_mat: rbnicsx.online.TensorsArray,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 2d array and petsc4py.PETSc.Mat content."""
    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]
    online_solution0 = rbnicsx.online.create_vector(10)
    online_solution0[:] = np.arange(1, 11)
    online_solution1 = rbnicsx.online.create_vector(9)
    online_solution1[:] = np.arange(1, 10)

    first_matrix = getattr(tensors_2d_array_mat, "first_matrix")
    first_matrix_action = rbnicsx.online.matrix_action(first_matrix)
    contraction = tensors_2d_array_mat.contraction(
        online_vec0, online_vec1, online_solution0, online_solution1)
    assert np.isclose(contraction, 150 * first_matrix_action(online_solution1)(online_solution0))


def test_online_tensors_array_contraction_1d_vec_too_many_args(
    tensors_1d_array_vec: rbnicsx.online.TensorsArray
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction with wrong inputs, 1d array and petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    first_vector = getattr(tensors_1d_array_vec, "first_vector")
    with pytest.raises(AssertionError):
        tensors_1d_array_vec.contraction(online_vec, online_vec, first_vector)


def test_online_tensors_array_contraction_1d_vec_wrong_vec_dimension(
    tensors_1d_array_vec: rbnicsx.online.TensorsArray
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction with wrong inputs, 1d array and petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    with pytest.raises(AssertionError):
        tensors_1d_array_vec.contraction(online_vec, online_vec)


def test_online_tensors_array_contraction_2d_mat_too_few_args(
    tensors_2d_array_mat: rbnicsx.online.TensorsArray
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction with wrong inputs, 2d array and petsc4py.PETSc.Mat content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)
    online_solution0 = rbnicsx.online.create_vector(10)
    online_solution0[:] = np.arange(1, 11)
    online_solution1 = rbnicsx.online.create_vector(9)
    online_solution1[:] = np.arange(1, 10)

    with pytest.raises(AssertionError):
        tensors_2d_array_mat.contraction(online_vec, online_solution0, online_solution1)


def test_online_tensors_array_contraction_2d_mat_wrong_vec_dimensions(
    tensors_2d_array_mat: rbnicsx.online.TensorsArray
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction with wrong inputs, 2d array and petsc4py.PETSc.Mat content."""
    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]

    with pytest.raises(AssertionError):
        tensors_2d_array_mat.contraction(online_vec0, online_vec1, online_vec0, online_vec1)


def test_online_tensors_contraction_empty() -> None:
    """Check rbnicsx.online.TensorsArray.contraction on an empty array."""
    empty_tensors_array = rbnicsx.online.TensorsArray(0, 0)
    with pytest.raises(RuntimeError):
        empty_tensors_array.contraction()


def test_online_tensors_array_contraction_1d_vec_implicit_args() -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 1d array and trivial petsc4py.PETSc.Vec content."""
    vectors = [rbnicsx.online.create_vector(1) for _ in range(6)]
    for (v, vector) in enumerate(vectors):
        vector.setValue(0, v + 1)
    tensors_1d_array = rbnicsx.online.TensorsArray(1, 6)
    for (i, vector) in enumerate(vectors):
        tensors_1d_array[i] = vector

    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    contraction = tensors_1d_array.contraction(online_vec, vectors[0])
    assert np.isclose(contraction, 91)
    contraction_implicit_trailing_args = tensors_1d_array.contraction(online_vec)
    assert np.isclose(contraction_implicit_trailing_args, contraction)


def test_online_tensors_array_contraction_2d_vec_implicit_args() -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 2d array and trivial petsc4py.PETSc.Vec content."""
    vectors = [rbnicsx.online.create_vector(1) for _ in range(6)]
    for (v, vector) in enumerate(vectors):
        vector.setValue(0, v + 1)
    tensors_2d_array = rbnicsx.online.TensorsArray(1, (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = vectors[i * 3 + j]

    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]

    contraction = tensors_2d_array.contraction(online_vec0, online_vec1, vectors[0])
    assert np.isclose(contraction, 150)
    contraction_implicit_trailing_args = tensors_2d_array.contraction(online_vec0, online_vec1)
    assert np.isclose(contraction_implicit_trailing_args, contraction)


@pytest.mark.parametrize("num_rows", [1, 10])
def test_online_tensors_array_contraction_1d_mat_implicit_args(
    num_rows: int, to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 1d array and trivial petsc4py.PETSc.Mat content."""
    matrices = [rbnicsx.online.create_matrix(num_rows, 1) for _ in range(6)]
    for (m, matrix) in enumerate(matrices):
        for d in range(num_rows):
            matrix.setValue(d, 0, (m + 1) * (d + 1))
        matrix.assemble()
    tensors_1d_array = rbnicsx.online.TensorsArray((num_rows, 1), 6)
    for (i, matrix) in enumerate(matrices):
        tensors_1d_array[i] = matrix

    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)
    online_solution0 = rbnicsx.online.create_vector(num_rows)
    online_solution0[:] = np.arange(1, num_rows + 1)
    online_solution1 = rbnicsx.online.create_vector(1)
    online_solution1[0] = 1

    contraction = tensors_1d_array.contraction(online_vec, online_solution0, online_solution1)
    assert np.isclose(
        contraction, 91 * online_solution0.norm(petsc4py.PETSc.NormType.NORM_2)**2)  # type: ignore[attr-defined]
    contraction_implicit_trailing_args1 = tensors_1d_array.contraction(online_vec, online_solution0)
    assert np.isclose(contraction_implicit_trailing_args1, contraction)
    if num_rows == 1:
        contraction_implicit_trailing_args2 = tensors_1d_array.contraction(online_vec)
        assert np.isclose(contraction_implicit_trailing_args2, contraction)


@pytest.mark.parametrize("num_rows", [1, 10])
def test_online_tensors_array_contraction_2d_mat_implicit_args(
    num_rows: int, to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.online.TensorsArray.contraction in the case of 2d array and trivial petsc4py.PETSc.Mat content."""
    matrices = [rbnicsx.online.create_matrix(num_rows, 1) for _ in range(6)]
    for (m, matrix) in enumerate(matrices):
        for d in range(num_rows):
            matrix.setValue(d, 0, (m + 1) * (d + 1))
        matrix.assemble()
    tensors_2d_array = rbnicsx.online.TensorsArray((num_rows, 1), (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = matrices[i * 3 + j]

    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]
    online_solution0 = rbnicsx.online.create_vector(num_rows)
    online_solution0[:] = np.arange(1, num_rows + 1)
    online_solution1 = rbnicsx.online.create_vector(1)
    online_solution1[0] = 1

    contraction = tensors_2d_array.contraction(online_vec0, online_vec1, online_solution0, online_solution1)
    assert np.isclose(
        contraction, 150 * online_solution0.norm(petsc4py.PETSc.NormType.NORM_2)**2)  # type: ignore[attr-defined]
    contraction_implicit_trailing_args1 = tensors_2d_array.contraction(online_vec0, online_vec1, online_solution0)
    assert np.isclose(contraction_implicit_trailing_args1, contraction)
    if num_rows == 1:
        contraction_implicit_trailing_args2 = tensors_2d_array.contraction(online_vec0, online_vec1)
        assert np.isclose(contraction_implicit_trailing_args2, contraction)
