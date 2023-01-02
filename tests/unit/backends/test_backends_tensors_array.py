# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.tensors_array module."""

import typing

import _pytest.fixtures
import dolfinx.fem
import dolfinx.mesh
import mpi4py.MPI
import nbvalx.tempfile
import numpy as np
import petsc4py.PETSc
import pytest
import ufl

import rbnicsx.backends
import rbnicsx.online


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.fixture
def tensors_1d_array_vec(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsArray:
    """Generate a rbnicsx.backends.TensorsArray with six petsc4py.PETSc.Vec entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(i + 1, v) * ufl.dx for i in range(6)]
    linear_forms_cpp = dolfinx.fem.form(linear_forms)
    vectors = [dolfinx.fem.petsc.assemble_vector(linear_form_cpp) for linear_form_cpp in linear_forms_cpp]
    for vector in vectors:
        vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_1d_array = rbnicsx.backends.TensorsArray(linear_forms_cpp[0], mesh.comm, 6)
    for (i, vector) in enumerate(vectors):
        tensors_1d_array[i] = vector
    setattr(tensors_1d_array, "form_ufl", linear_forms[0])  # for test_tensors_array_setitem_*d_vec
    return tensors_1d_array


@pytest.fixture
def tensors_2d_array_vec(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsArray:
    """Generate a rbnicsx.backends.TensorsArray with two-by-three petsc4py.PETSc.Vec entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(i + 1, v) * ufl.dx for i in range(6)]
    linear_forms_cpp = dolfinx.fem.form(linear_forms)
    vectors = [dolfinx.fem.petsc.assemble_vector(linear_form_cpp) for linear_form_cpp in linear_forms_cpp]
    for vector in vectors:
        vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_2d_array = rbnicsx.backends.TensorsArray(linear_forms_cpp[0], mesh.comm, (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = vectors[i * 3 + j]
    setattr(tensors_2d_array, "form_ufl", linear_forms[0])  # for test_tensors_array_setitem_*d_vec
    return tensors_2d_array


@pytest.fixture
def tensors_1d_array_mat(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsArray:
    """Generate a rbnicsx.backends.TensorsArray with six petsc4py.PETSc.Mat entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [(i + 1) * ufl.inner(u, v) * ufl.dx for i in range(6)]
    bilinear_forms_cpp = dolfinx.fem.form(bilinear_forms)
    matrices = [dolfinx.fem.petsc.assemble_matrix(bilinear_form_cpp) for bilinear_form_cpp in bilinear_forms_cpp]
    for matrix in matrices:
        matrix.assemble()
    tensors_1d_array = rbnicsx.backends.TensorsArray(bilinear_forms_cpp[0], mesh.comm, 6)
    for (i, matrix) in enumerate(matrices):
        tensors_1d_array[i] = matrix
    setattr(tensors_1d_array, "form_ufl", bilinear_forms[0])  # for test_tensors_array_setitem_mat
    return tensors_1d_array


@pytest.fixture
def tensors_2d_array_mat(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsArray:
    """Generate a rbnicsx.backends.TensorsArray with two-by-three petsc4py.PETSc.Mat entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [(i + 1) * ufl.inner(u, v) * ufl.dx for i in range(6)]
    bilinear_forms_cpp = dolfinx.fem.form(bilinear_forms)
    matrices = [dolfinx.fem.petsc.assemble_matrix(bilinear_form_cpp) for bilinear_form_cpp in bilinear_forms_cpp]
    for matrix in matrices:
        matrix.assemble()
    tensors_2d_array = rbnicsx.backends.TensorsArray(bilinear_forms_cpp[0], mesh.comm, (2, 3))
    for i in range(2):
        for j in range(3):
            tensors_2d_array[i, j] = matrices[i * 3 + j]
    setattr(tensors_2d_array, "form_ufl", bilinear_forms[0])  # for test_tensors_array_setitem_mat
    return tensors_2d_array


@pytest.fixture(params=["tensors_1d_array_vec", "tensors_2d_array_vec"])
def tensors_array_vec(request: _pytest.fixtures.SubRequest) -> rbnicsx.backends.TensorsArray:
    """Parameterize rbnicsx.backends.TensorsArray on shape, with petsc4py.PETSc.Vec content."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=["tensors_1d_array_mat", "tensors_2d_array_mat"])
def tensors_array_mat(request: _pytest.fixtures.SubRequest) -> rbnicsx.backends.TensorsArray:
    """Parameterize rbnicsx.backends.TensorsArray on shape, with petsc4py.PETSc.Mat content."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


@pytest.fixture(params=["tensors_1d_array_vec", "tensors_2d_array_vec", "tensors_1d_array_mat", "tensors_2d_array_mat"])
def tensors_array(request: _pytest.fixtures.SubRequest) -> rbnicsx.backends.TensorsArray:
    """Parameterize rbnicsx.backends.TensorsArray on shape and petsc4py.PETSc.Mat and petsc4py.PETSc.Vec."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


def test_backends_tensors_array_type_vec(tensors_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.type in the case of petsc4py.PETSc.Vec content."""
    tensors_array_vec.type == "Vec"


def test_backends_tensors_array_type_mat(tensors_array_mat: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.type in the case of petsc4py.PETSc.Mat content."""
    tensors_array_mat.type == "Mat"


def test_backends_tensors_array_type_none() -> None:
    """Check rbnicsx.backends.TensorsArray.type at initialization."""
    fake_form = None
    empty_tensors_array = rbnicsx.backends.TensorsArray(fake_form, mpi4py.MPI.COMM_WORLD, 0)  # type: ignore[arg-type]
    assert empty_tensors_array.type is None


def test_backends_tensors_array_duplicate(tensors_array: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.duplicate."""
    tensors_array2 = tensors_array.duplicate()
    assert tensors_array2.shape == tensors_array.shape
    assert all([tensor is not None for tensor in tensors_array._array.flat])
    assert all([tensor is None for tensor in tensors_array2._array.flat])


def test_backends_tensors_array_getitem_1d_int_vec(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with integer input, 1d array and petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_1d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    for i in range(6):
        assert np.allclose(tensors_1d_array_vec[i].array, (i + 1) * first_vector.array)


def test_backends_tensors_array_getitem_1d_tuple_int_vec(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with wrong index, 1d array and petsc4py.PETSc.Vec content."""
    with pytest.raises(IndexError) as excinfo:
        tensors_1d_array_vec[0, 0].array
    assert str(excinfo.value) == "too many indices for array: array is 1-dimensional, but 2 were indexed"


def test_backends_tensors_array_getitem_2d_tuple_int_vec(tensors_2d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with integer input, 2d array and petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_2d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    for i in range(2):
        for j in range(3):
            assert np.allclose(tensors_2d_array_vec[i, j].array, (i * 3 + j + 1) * first_vector.array)


def test_backends_tensors_array_getitem_1d_int_mat(  # type: ignore[no-any-unimported]
    tensors_1d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with integer input, 1d array and petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_1d_array_mat.form)
    first_matrix.assemble()
    for i in range(6):
        assert np.allclose(to_dense_matrix(tensors_1d_array_mat[i]), (i + 1) * to_dense_matrix(first_matrix))


def test_backends_tensors_array_getitem_2d_tuple_int_mat(  # type: ignore[no-any-unimported]
    tensors_2d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with integer input, 2d array and petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_2d_array_mat.form)
    first_matrix.assemble()
    for i in range(2):
        for j in range(3):
            assert np.allclose(
                to_dense_matrix(tensors_2d_array_mat[i, j]), (i * 3 + j + 1) * to_dense_matrix(first_matrix))


def test_backends_tensors_array_getitem_1d_slice_vec(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with slice input, 1d array and petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_1d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_1d_array_vec2 = tensors_1d_array_vec[0:2]
    assert tensors_1d_array_vec2.shape == (2, )
    assert np.allclose(tensors_1d_array_vec2[0].array, first_vector.array)
    assert np.allclose(tensors_1d_array_vec2[1].array, 2 * first_vector.array)


def test_backends_tensors_array_getitem_2d_slice_vec(tensors_2d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with slice input, 2d array and petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_2d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_2d_array_vec2 = tensors_2d_array_vec[0:1, 0:2]
    assert tensors_2d_array_vec2.shape == (1, 2)
    assert np.allclose(tensors_2d_array_vec2[0, 0].array, first_vector.array)
    assert np.allclose(tensors_2d_array_vec2[0, 1].array, 2 * first_vector.array)


def test_backends_tensors_array_getitem_1d_slice_mat(  # type: ignore[no-any-unimported]
    tensors_1d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with slice input, 1d array and petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_1d_array_mat.form)
    first_matrix.assemble()
    tensors_1d_array_mat2 = tensors_1d_array_mat[0:2]
    assert tensors_1d_array_mat2.shape == (2, )
    assert np.allclose(to_dense_matrix(tensors_1d_array_mat2[0]), to_dense_matrix(first_matrix))
    assert np.allclose(to_dense_matrix(tensors_1d_array_mat2[1]), 2 * to_dense_matrix(first_matrix))


def test_backends_tensors_array_getitem_2d_slice_mat(  # type: ignore[no-any-unimported]
    tensors_2d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.__getitem__ with slice input, 1d array and petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_2d_array_mat.form)
    first_matrix.assemble()
    tensors_2d_array_mat2 = tensors_2d_array_mat[0:1, 0:2]
    assert tensors_2d_array_mat2.shape == (1, 2)
    assert np.allclose(to_dense_matrix(tensors_2d_array_mat2[0, 0]), to_dense_matrix(first_matrix))
    assert np.allclose(to_dense_matrix(tensors_2d_array_mat2[0, 1]), 2 * to_dense_matrix(first_matrix))


def test_backends_tensors_array_getitem_wrong_type(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.online.TensorsList.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        tensors_1d_array_vec[""]  # type: ignore[call-overload]


def test_backends_tensors_array_setitem_1d_vec(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__setitem__ in the case of 1d array and petsc4py.PETSc.Vec content."""
    form_ufl = getattr(tensors_1d_array_vec, "form_ufl")
    new_vector = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(7 * form_ufl))
    new_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_1d_array_vec[0] = new_vector

    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_1d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    for i in range(6):
        if i == 0:
            coeff = 7
        else:
            coeff = i + 1
        assert np.allclose(tensors_1d_array_vec[i].array, coeff * first_vector.array)


def test_backends_tensors_array_setitem_2d_vec(tensors_2d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__setitem__ in the case of 2d array and petsc4py.PETSc.Vec content."""
    form_ufl = getattr(tensors_2d_array_vec, "form_ufl")
    new_vector = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(7 * form_ufl))
    new_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_2d_array_vec[0, 0] = new_vector

    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_2d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0:
                coeff = 7
            else:
                coeff = i * 3 + j + 1
            assert np.allclose(tensors_2d_array_vec[i, j].array, coeff * first_vector.array)


def test_backends_tensors_array_setitem_1d_mat(  # type: ignore[no-any-unimported]
    tensors_1d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.__setitem__ in the case of 1d array and petsc4py.PETSc.Mat content."""
    form_ufl = getattr(tensors_1d_array_mat, "form_ufl")
    new_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(7 * form_ufl))
    new_matrix.assemble()
    tensors_1d_array_mat[0] = new_matrix

    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_1d_array_mat.form)
    first_matrix.assemble()
    for i in range(6):
        if i == 0:
            coeff = 7
        else:
            coeff = i + 1
        assert np.allclose(to_dense_matrix(tensors_1d_array_mat[i]), coeff * to_dense_matrix(first_matrix))


def test_backends_tensors_array_setitem_2d_mat(  # type: ignore[no-any-unimported]
    tensors_2d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.__setitem__ in the case of 1d array and petsc4py.PETSc.Mat content."""
    form_ufl = getattr(tensors_2d_array_mat, "form_ufl")
    new_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(7 * form_ufl))
    new_matrix.assemble()
    tensors_2d_array_mat[0, 0] = new_matrix

    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_2d_array_mat.form)
    first_matrix.assemble()
    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0:
                coeff = 7
            else:
                coeff = i * 3 + j + 1
            assert np.allclose(to_dense_matrix(tensors_2d_array_mat[i, j]), coeff * to_dense_matrix(first_matrix))


def test_backends_tensors_array_setitem_mixed_types(
    tensors_1d_array_vec: rbnicsx.backends.TensorsArray, tensors_1d_array_mat: rbnicsx.backends.TensorsArray
) -> None:
    """Check rbnicsx.backends.TensorsArray.__setitem__ mixing up Mat and Vec objects."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_1d_array_vec.form)
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_1d_array_mat.form)

    with pytest.raises(AssertionError):
        tensors_1d_array_vec[0] = first_matrix

    with pytest.raises(AssertionError):
        tensors_1d_array_mat[0] = first_vector


def test_backends_tensors_array_setitem_wrong_type(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.__setitem__ when providing neither a Mat nor a Vec object."""
    with pytest.raises(RuntimeError):
        tensors_1d_array_vec[0] = None


def test_backends_tensors_array_save_load_1d_vec(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check I/O for a rbnicsx.backends.TensorsArray in the case of 1d array and petsc4py.PETSc.Vec content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_1d_array_vec.comm) as tempdir:
        tensors_1d_array_vec.save(tempdir, "tensors_1d_array_vec")

        tensors_1d_array_vec2 = tensors_1d_array_vec.duplicate()
        tensors_1d_array_vec2.load(tempdir, "tensors_1d_array_vec")

        assert tensors_1d_array_vec2.shape == (6, )
        for i in range(6):
            assert np.allclose(tensors_1d_array_vec2[i].array, tensors_1d_array_vec[i].array)


def test_backends_tensors_array_save_load_2d_vec(tensors_2d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check I/O for a rbnicsx.backends.TensorsArray in the case of 2d array and petsc4py.PETSc.Vec content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_2d_array_vec.comm) as tempdir:
        tensors_2d_array_vec.save(tempdir, "tensors_2d_array_vec")

        tensors_2d_array_vec2 = tensors_2d_array_vec.duplicate()
        tensors_2d_array_vec2.load(tempdir, "tensors_2d_array_vec")

        assert tensors_2d_array_vec2.shape == (2, 3)
        for i in range(2):
            for j in range(3):
                assert np.allclose(tensors_2d_array_vec2[i, j].array, tensors_2d_array_vec[i, j].array)


def test_backends_tensors_array_save_load_1d_mat(  # type: ignore[no-any-unimported]
    tensors_1d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check I/O for a rbnicsx.backends.TensorsArray in the case of 1d array and petsc4py.PETSc.Mat content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_1d_array_mat.comm) as tempdir:
        tensors_1d_array_mat.save(tempdir, "tensors_1d_array_mat")

        tensors_1d_array_mat2 = tensors_1d_array_mat.duplicate()
        tensors_1d_array_mat2.load(tempdir, "tensors_1d_array_mat")

        assert tensors_1d_array_mat2.shape == (6, )
        for i in range(6):
            assert np.allclose(to_dense_matrix(tensors_1d_array_mat2[i]), to_dense_matrix(tensors_1d_array_mat[i]))


def test_backends_tensors_array_save_load_2d_mat(  # type: ignore[no-any-unimported]
    tensors_2d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check I/O for a rbnicsx.backends.TensorsArray in the case of 2d array and petsc4py.PETSc.Mat content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_2d_array_mat.comm) as tempdir:
        tensors_2d_array_mat.save(tempdir, "tensors_2d_array_mat")

        tensors_2d_array_mat2 = tensors_2d_array_mat.duplicate()
        tensors_2d_array_mat2.load(tempdir, "tensors_2d_array_mat")

        assert tensors_2d_array_mat2.shape == (2, 3)
        for i in range(2):
            for j in range(3):
                assert np.allclose(
                    to_dense_matrix(tensors_2d_array_mat2[i, j]), to_dense_matrix(tensors_2d_array_mat[i, j]))


def test_backends_tensors_array_save_load_empty() -> None:
    """Check I/O for rbnicsx.backends.TensorsArray when providing neither a Mat nor a Vec object."""
    fake_form = None
    empty_tensors_list = rbnicsx.backends.TensorsArray(fake_form, mpi4py.MPI.COMM_WORLD, 0)  # type: ignore[arg-type]

    with nbvalx.tempfile.TemporaryDirectory(empty_tensors_list.comm) as tempdir:
        with pytest.raises(RuntimeError):
            empty_tensors_list.save(tempdir, "empty_tensors_list")

        with pytest.raises(RuntimeError):
            empty_tensors_list.load(tempdir, "empty_tensors_list")


def test_backends_tensors_array_contraction_1d_vec(tensors_1d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction in the case of 1d array and petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_1d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    contraction = tensors_1d_array_vec.contraction(online_vec, first_vector)
    assert np.isclose(contraction, 91 * first_vector.norm(petsc4py.PETSc.NormType.NORM_2)**2)


def test_backends_tensors_array_contraction_2d_vec(tensors_2d_array_vec: rbnicsx.backends.TensorsArray) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction in the case of 2d array and petsc4py.PETSc.Vec content."""
    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]

    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_2d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    contraction = tensors_2d_array_vec.contraction(online_vec0, online_vec1, first_vector)
    assert np.isclose(contraction, 150 * first_vector.norm(petsc4py.PETSc.NormType.NORM_2)**2)


def test_backends_tensors_array_contraction_1d_mat(  # type: ignore[no-any-unimported]
    tensors_1d_array_vec: rbnicsx.backends.TensorsArray, tensors_1d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction in the case of 1d array and petsc4py.PETSc.Mat content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_1d_array_mat.form)
    first_matrix.assemble()
    first_matrix_action = rbnicsx.online.matrix_action(first_matrix)
    contraction = tensors_1d_array_mat.contraction(online_vec, tensors_1d_array_vec[0], tensors_1d_array_vec[1])
    assert np.isclose(contraction, 91 * first_matrix_action(tensors_1d_array_vec[1])(tensors_1d_array_vec[0]))


def test_backends_tensors_array_contraction_2d_mat(  # type: ignore[no-any-unimported]
    tensors_1d_array_vec: rbnicsx.backends.TensorsArray, tensors_2d_array_mat: rbnicsx.backends.TensorsArray,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction in the case of 2d array and petsc4py.PETSc.Mat content."""
    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]

    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_2d_array_mat.form)
    first_matrix.assemble()
    first_matrix_action = rbnicsx.online.matrix_action(first_matrix)
    contraction = tensors_2d_array_mat.contraction(
        online_vec0, online_vec1, tensors_1d_array_vec[0], tensors_1d_array_vec[1])
    assert np.isclose(contraction, 150 * first_matrix_action(tensors_1d_array_vec[1])(tensors_1d_array_vec[0]))


def test_backends_tensors_array_contraction_1d_vec_too_many_args(
    tensors_1d_array_vec: rbnicsx.backends.TensorsArray
) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction with wrong inputs, 1d array and petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_1d_array_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    with pytest.raises(AssertionError):
        tensors_1d_array_vec.contraction(online_vec, online_vec, first_vector)


def test_backends_tensors_array_contraction_1d_vec_wrong_vec_dimension(
    tensors_1d_array_vec: rbnicsx.backends.TensorsArray
) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction with wrong inputs, 1d array and petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    with pytest.raises(AssertionError):
        tensors_1d_array_vec.contraction(online_vec, online_vec)


def test_backends_tensors_array_contraction_2d_mat_too_few_args(
    tensors_1d_array_vec: rbnicsx.backends.TensorsArray, tensors_2d_array_mat: rbnicsx.backends.TensorsArray
) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction with wrong inputs, 2d array and petsc4py.PETSc.Mat content."""
    online_vec = rbnicsx.online.create_vector(6)
    online_vec[:] = np.arange(1, 7)

    with pytest.raises(AssertionError):
        tensors_2d_array_mat.contraction(online_vec, tensors_1d_array_vec[0], tensors_1d_array_vec[1])


def test_backends_tensors_array_contraction_2d_mat_wrong_vec_dimensions(
    tensors_2d_array_mat: rbnicsx.backends.TensorsArray
) -> None:
    """Check rbnicsx.backends.TensorsArray.contraction with wrong inputs, 2d array and petsc4py.PETSc.Mat content."""
    online_vec0 = rbnicsx.online.create_vector(2)
    online_vec0[:] = [1, 2]
    online_vec1 = rbnicsx.online.create_vector(3)
    online_vec1[:] = [3, 4, 5]

    with pytest.raises(AssertionError):
        tensors_2d_array_mat.contraction(online_vec0, online_vec1, online_vec0, online_vec1)


def test_backends_tensors_contraction_empty() -> None:
    """Check rbnicsx.backends.TensorsArray.contraction on an empty array."""
    fake_form = None
    empty_tensors_array = rbnicsx.backends.TensorsArray(fake_form, mpi4py.MPI.COMM_WORLD, 0)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        empty_tensors_array.contraction()
