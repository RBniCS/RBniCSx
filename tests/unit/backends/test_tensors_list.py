# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox.backends.tensors_list module."""

import dolfinx.mesh
import dolfinx_utils.test.fixtures
import mpi4py
import numpy as np
import petsc4py
import pytest
import ufl

import minirox.backends
import utils  # noqa: I001

tempdir = dolfinx_utils.test.fixtures.tempdir


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    return dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)


@pytest.fixture
def tensors_list_vec(mesh: dolfinx.mesh.Mesh) -> minirox.backends.TensorsList:
    """Generate a minirox.backends.TensorsList with two petsc4py.PETSc.Vec entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(i + 1, v) * ufl.dx for i in range(2)]
    vectors = [dolfinx.fem.assemble_vector(linear_form) for linear_form in linear_forms]
    [vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) for vector in vectors]
    tensors_list = minirox.backends.TensorsList(linear_forms[0], mesh.comm)
    [tensors_list.append(vector) for vector in vectors]
    return tensors_list


@pytest.fixture
def tensors_list_mat(mesh: dolfinx.mesh.Mesh) -> minirox.backends.TensorsList:
    """Generate a minirox.backends.TensorsList with two petsc4py.PETSc.Mat entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [(i + 1) * ufl.inner(u, v) * ufl.dx for i in range(2)]
    matrices = [dolfinx.fem.assemble_matrix(bilinear_form) for bilinear_form in bilinear_forms]
    [matrix.assemble() for matrix in matrices]
    tensors_list = minirox.backends.TensorsList(bilinear_forms[0], mesh.comm)
    [tensors_list.append(matrix) for matrix in matrices]
    return tensors_list


@pytest.fixture(params=["tensors_list_vec", "tensors_list_mat"])
def tensors_list(request: object) -> minirox.backends.TensorsList:
    """Parameterize minirox.backends.TensorsList on petsc4py.PETSc.Mat and petsc4py.PETSc.Vec."""
    return request.getfixturevalue(request.param)


def test_tensors_list_type_vec(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.type in the case of petsc4py.PETSc.Vec content."""
    tensors_list_vec.type == "Vec"


def test_tensors_list_type_mat(tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.type in the case of petsc4py.PETSc.Mat content."""
    tensors_list_mat.type == "Mat"


def test_tensors_list_type_none(tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.type at initialization."""
    tensors_list = minirox.backends.TensorsList("fake_form", mpi4py.MPI.COMM_WORLD)
    assert tensors_list.type is None


def test_tensors_list_append_mixed_types(
        tensors_list_vec: minirox.backends.TensorsList, tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.append mixing up Mat and Vec objects."""
    first_vector = dolfinx.fem.assemble_vector(tensors_list_vec.form)
    first_matrix = dolfinx.fem.assemble_matrix(tensors_list_mat.form)

    with pytest.raises(AssertionError):
        tensors_list_vec.append(first_matrix)

    with pytest.raises(AssertionError):
        tensors_list_mat.append(first_vector)


def test_tensors_list_append_wrong_type(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.append when providing neither a Mat nor a Vec object."""
    with pytest.raises(RuntimeError):
        tensors_list_vec.append(None)


def test_tensors_list_extend(tensors_list: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.extend."""
    tensors_list2 = minirox.backends.TensorsList(tensors_list.form, tensors_list.comm)
    tensors_list2.extend(tensors_list)
    assert len(tensors_list2) == 2
    for i in range(2):
        assert tensors_list2[i] == tensors_list[i]


def test_tensors_list_len(tensors_list: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__len__."""
    assert len(tensors_list) == 2


def test_tensors_list_clear(tensors_list: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.clear."""
    tensors_list.clear()
    assert len(tensors_list) == 0


def test_tensors_list_iter_vec(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__iter__ in the case of petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    for (index, vector) in enumerate(tensors_list_vec):
        assert np.allclose(vector.array, (index + 1) * first_vector.array)


def test_tensors_list_iter_mat(tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__iter__ in the case of petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    for (index, matrix) in enumerate(tensors_list_mat):
        assert np.allclose(utils.to_dense_matrix(matrix), (index + 1) * utils.to_dense_matrix(first_matrix))


def test_tensors_list_getitem_int_vec(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__getitem__ with integer input in the case of petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    assert np.allclose(tensors_list_vec[0].array, first_vector.array)
    assert np.allclose(tensors_list_vec[1].array, 2 * first_vector.array)


def test_tensors_list_getitem_int_mat(tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__getitem__ with integer input in the case of petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    assert np.allclose(utils.to_dense_matrix(tensors_list_mat[0]), utils.to_dense_matrix(first_matrix))
    assert np.allclose(utils.to_dense_matrix(tensors_list_mat[1]), 2 * utils.to_dense_matrix(first_matrix))


def test_tensors_list_getitem_slice_vec(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__getitem__ with slice input in the case of petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_list_vec2 = tensors_list_vec[0:2]
    assert len(tensors_list_vec2) == 2
    assert np.allclose(tensors_list_vec2[0].array, first_vector.array)
    assert np.allclose(tensors_list_vec2[1].array, 2 * first_vector.array)


def test_tensors_list_getitem_slice_mat(tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__getitem__ with slice input in the case of petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    tensors_list_mat2 = tensors_list_mat[0:2]
    assert len(tensors_list_mat2) == 2
    assert np.allclose(utils.to_dense_matrix(tensors_list_mat2[0]), utils.to_dense_matrix(first_matrix))
    assert np.allclose(utils.to_dense_matrix(tensors_list_mat2[1]), 2 * utils.to_dense_matrix(first_matrix))


def test_tensors_list_getitem_wrong_type(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        tensors_list_vec[0, 0]


def test_tensors_list_setitem_vec(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__setitem__ in the case of petsc4py.PETSc.Vec content."""
    new_vector = dolfinx.fem.assemble_vector(3 * tensors_list_vec.form)
    new_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    tensors_list_vec[0] = new_vector

    first_vector = dolfinx.fem.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    assert np.allclose(tensors_list_vec[0].array, 3 * first_vector.array)
    assert np.allclose(tensors_list_vec[1].array, 2 * first_vector.array)


def test_tensors_list_setitem_mat(tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__setitem__ in the case of petsc4py.PETSc.Mat content."""
    new_matrix = dolfinx.fem.assemble_matrix(3 * tensors_list_mat.form)
    new_matrix.assemble()
    tensors_list_mat[0] = new_matrix

    first_matrix = dolfinx.fem.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    assert np.allclose(utils.to_dense_matrix(tensors_list_mat[0]), 3 * utils.to_dense_matrix(first_matrix))
    assert np.allclose(utils.to_dense_matrix(tensors_list_mat[1]), 2 * utils.to_dense_matrix(first_matrix))


def test_tensors_list_save_load_vec(tensors_list_vec: minirox.backends.TensorsList, tempdir: str) -> None:
    """Check I/O for a minirox.backends.TensorsList in the case of petsc4py.PETSc.Vec content."""
    tensors_list_vec.save(tempdir, "tensors_list_vec")

    form, comm = tensors_list_vec.form, tensors_list_vec.comm
    tensors_list_vec2 = minirox.backends.TensorsList(form, comm)
    tensors_list_vec2.load(tempdir, "tensors_list_vec")

    assert len(tensors_list_vec2) == 2
    for (vector, vector2) in zip(tensors_list_vec, tensors_list_vec2):
        assert np.allclose(vector2.array, vector.array)


def test_tensors_list_save_load_mat(tensors_list_mat: minirox.backends.TensorsList, tempdir: str) -> None:
    """Check I/O for a minirox.backends.TensorsList in the case of petsc4py.PETSc.Mat content."""
    tensors_list_mat.save(tempdir, "tensors_list_mat")

    form, comm = tensors_list_mat.form, tensors_list_mat.comm
    tensors_list_mat2 = minirox.backends.TensorsList(form, comm)
    tensors_list_mat2.load(tempdir, "tensors_list_mat")

    assert len(tensors_list_mat2) == 2
    for (matrix, matrix2) in zip(tensors_list_mat, tensors_list_mat2):
        assert np.allclose(utils.to_dense_matrix(matrix2), utils.to_dense_matrix(matrix))


def test_tensors_list_mul_vec(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__mul__ in the case of petsc4py.PETSc.Vec content."""
    online_vec = petsc4py.PETSc.Vec().createSeq(2, comm=mpi4py.MPI.COMM_SELF)
    online_vec[0] = 3
    online_vec[1] = 5

    vector = tensors_list_vec * online_vec
    first_vector = dolfinx.fem.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    assert np.allclose(vector.array, 13 * first_vector.array)


def test_tensors_list_mul_mat(tensors_list_mat: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__mul__ in the case of petsc4py.PETSc.Mat content."""
    online_vec = petsc4py.PETSc.Vec().createSeq(2, comm=mpi4py.MPI.COMM_SELF)
    online_vec[0] = 3
    online_vec[1] = 5

    matrix = tensors_list_mat * online_vec
    first_matrix = dolfinx.fem.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    assert np.allclose(utils.to_dense_matrix(matrix), 13 * utils.to_dense_matrix(first_matrix))


def test_tensors_list_mul_empty(mesh: dolfinx.mesh.Mesh) -> None:
    """Check minirox.backends.TensorsList.__mul__ with empty list."""
    fake_form = None
    empty_tensors_list = minirox.backends.TensorsList(fake_form, mesh.comm)

    online_vec = petsc4py.PETSc.Vec().createSeq(0, comm=mpi4py.MPI.COMM_SELF)
    should_be_none = empty_tensors_list * online_vec
    assert should_be_none is None


def test_tensors_list_mul_not_implemented(tensors_list_vec: minirox.backends.TensorsList) -> None:
    """Check minirox.backends.TensorsList.__mul__ with an incorrect type."""
    with pytest.raises(TypeError):
        tensors_list_vec * None
