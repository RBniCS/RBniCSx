# Copyright (C) 2021-2026 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.tensors_list module."""

import pathlib
import typing

import _pytest.fixtures
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import mpi4py.MPI
import nbvalx.tempfile
import numpy as np
import numpy.typing as npt
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
def tensors_list_vec(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsList:
    """Generate a rbnicsx.backends.TensorsList with two petsc4py.PETSc.Vec entries."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(i + 1, v) * ufl.dx for i in range(2)]
    linear_forms_cpp = dolfinx.fem.form(linear_forms)
    vectors = [dolfinx.fem.petsc.assemble_vector(linear_form_cpp) for linear_form_cpp in linear_forms_cpp]
    for vector in vectors:
        vector.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    tensors_list = rbnicsx.backends.TensorsList(linear_forms_cpp[0], mesh.comm)
    for vector in vectors:
        tensors_list.append(vector)
    setattr(tensors_list, "form_ufl", linear_forms[0])  # for test_tensors_list_setitem_vec
    return tensors_list


@pytest.fixture
def tensors_list_mat(mesh: dolfinx.mesh.Mesh) -> rbnicsx.backends.TensorsList:
    """Generate a rbnicsx.backends.TensorsList with two petsc4py.PETSc.Mat entries."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [(i + 1) * ufl.inner(u, v) * ufl.dx for i in range(2)]
    bilinear_forms_cpp = dolfinx.fem.form(bilinear_forms)
    matrices = [dolfinx.fem.petsc.assemble_matrix(bilinear_form_cpp) for bilinear_form_cpp in bilinear_forms_cpp]
    for matrix in matrices:
        matrix.assemble()
    tensors_list = rbnicsx.backends.TensorsList(bilinear_forms_cpp[0], mesh.comm)
    for matrix in matrices:
        tensors_list.append(matrix)
    setattr(tensors_list, "form_ufl", bilinear_forms[0])  # for test_tensors_list_setitem_mat
    return tensors_list


@pytest.fixture(params=["tensors_list_vec", "tensors_list_mat"])
def tensors_list(request: _pytest.fixtures.SubRequest) -> rbnicsx.backends.TensorsList:
    """Parameterize rbnicsx.backends.TensorsList on petsc4py.PETSc.Mat and petsc4py.PETSc.Vec."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


def test_backends_tensors_list_type_vec(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.type in the case of petsc4py.PETSc.Vec content."""
    tensors_list_vec.type == "Vec"


def test_backends_tensors_list_type_mat(tensors_list_mat: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.type in the case of petsc4py.PETSc.Mat content."""
    tensors_list_mat.type == "Mat"


def test_backends_tensors_list_type_none() -> None:
    """Check rbnicsx.backends.TensorsList.type at initialization."""
    fake_form = None
    empty_tensors_list = rbnicsx.backends.TensorsList(fake_form, mpi4py.MPI.COMM_WORLD)  # type: ignore[arg-type]
    assert empty_tensors_list.type is None


def test_backends_tensors_list_append_mixed_types(
        tensors_list_vec: rbnicsx.backends.TensorsList, tensors_list_mat: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.append mixing up Mat and Vec objects."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_list_vec.form)
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_list_mat.form)

    with pytest.raises(AssertionError):
        tensors_list_vec.append(first_matrix)

    with pytest.raises(AssertionError):
        tensors_list_mat.append(first_vector)


def test_backends_tensors_list_append_wrong_type(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.append when providing neither a Mat nor a Vec object."""
    with pytest.raises(RuntimeError):
        tensors_list_vec.append(None)


def test_backends_tensors_list_duplicate(tensors_list: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.duplicate."""
    tensors_list2 = tensors_list.duplicate()
    assert len(tensors_list2) == 0


def test_backends_tensors_list_extend(tensors_list: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.extend."""
    tensors_list2 = tensors_list.duplicate()
    tensors_list2.extend(tensors_list)
    assert len(tensors_list2) == 2
    for i in range(2):
        assert tensors_list2[i] == tensors_list[i]


def test_backends_tensors_list_len(tensors_list: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__len__."""
    assert len(tensors_list) == 2


def test_backends_tensors_list_clear(tensors_list: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.clear."""
    tensors_list.clear()
    assert len(tensors_list) == 0


def test_backends_tensors_list_iter_vec(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__iter__ in the case of petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    for (index, vector) in enumerate(tensors_list_vec):
        assert np.allclose(vector.array, (index + 1) * first_vector.array)


def test_backends_tensors_list_iter_mat(
    tensors_list_mat: rbnicsx.backends.TensorsList,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsList.__iter__ in the case of petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    for (index, matrix) in enumerate(tensors_list_mat):
        assert np.allclose(to_dense_matrix(matrix), (index + 1) * to_dense_matrix(first_matrix))


def test_backends_tensors_list_getitem_int_vec(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__getitem__ with integer input in the case of petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert np.allclose(tensors_list_vec[0].array, first_vector.array)
    assert np.allclose(tensors_list_vec[1].array, 2 * first_vector.array)


def test_backends_tensors_list_getitem_int_mat(
    tensors_list_mat: rbnicsx.backends.TensorsList,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsList.__getitem__ with integer input in the case of petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    assert np.allclose(to_dense_matrix(tensors_list_mat[0]), to_dense_matrix(first_matrix))
    assert np.allclose(to_dense_matrix(tensors_list_mat[1]), 2 * to_dense_matrix(first_matrix))


def test_backends_tensors_list_getitem_slice_vec(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__getitem__ with slice input in the case of petsc4py.PETSc.Vec content."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    tensors_list_vec2 = tensors_list_vec[0:2]
    assert len(tensors_list_vec2) == 2
    assert np.allclose(tensors_list_vec2[0].array, first_vector.array)
    assert np.allclose(tensors_list_vec2[1].array, 2 * first_vector.array)


def test_backends_tensors_list_getitem_slice_mat(
    tensors_list_mat: rbnicsx.backends.TensorsList,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsList.__getitem__ with slice input in the case of petsc4py.PETSc.Mat content."""
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    tensors_list_mat2 = tensors_list_mat[0:2]
    assert len(tensors_list_mat2) == 2
    assert np.allclose(to_dense_matrix(tensors_list_mat2[0]), to_dense_matrix(first_matrix))
    assert np.allclose(to_dense_matrix(tensors_list_mat2[1]), 2 * to_dense_matrix(first_matrix))


def test_backends_tensors_list_getitem_wrong_type(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        tensors_list_vec[0, 0]  # type: ignore[call-overload]


def test_backends_tensors_list_setitem_vec(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__setitem__ in the case of petsc4py.PETSc.Vec content."""
    form_ufl = getattr(tensors_list_vec, "form_ufl")
    new_vector = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(3 * form_ufl))
    new_vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    tensors_list_vec[0] = new_vector

    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert np.allclose(tensors_list_vec[0].array, 3 * first_vector.array)
    assert np.allclose(tensors_list_vec[1].array, 2 * first_vector.array)


def test_backends_tensors_list_setitem_mat(
    tensors_list_mat: rbnicsx.backends.TensorsList,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsList.__setitem__ in the case of petsc4py.PETSc.Mat content."""
    form_ufl = getattr(tensors_list_mat, "form_ufl")
    new_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(3 * form_ufl))
    new_matrix.assemble()
    tensors_list_mat[0] = new_matrix

    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    assert np.allclose(to_dense_matrix(tensors_list_mat[0]), 3 * to_dense_matrix(first_matrix))
    assert np.allclose(to_dense_matrix(tensors_list_mat[1]), 2 * to_dense_matrix(first_matrix))


def test_backends_tensors_list_setitem_mixed_types(
    tensors_list_vec: rbnicsx.backends.TensorsList, tensors_list_mat: rbnicsx.backends.TensorsList
) -> None:
    """Check rbnicsx.backends.TensorsList.__setitem__ mixing up Mat and Vec objects."""
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_list_vec.form)
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_list_mat.form)

    with pytest.raises(AssertionError):
        tensors_list_vec[0] = first_matrix

    with pytest.raises(AssertionError):
        tensors_list_mat[0] = first_vector


def test_backends_tensors_list_setitem_wrong_type(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__setitem__ when providing neither a Mat nor a Vec object."""
    with pytest.raises(RuntimeError):
        tensors_list_vec[0] = None


def test_backends_tensors_list_save_load_vec(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check I/O for a rbnicsx.backends.TensorsList in the case of petsc4py.PETSc.Vec content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_list_vec.comm) as tempdir:
        tensors_list_vec.save(pathlib.Path(tempdir), "tensors_list_vec")

        tensors_list_vec2 = tensors_list_vec.duplicate()
        tensors_list_vec2.load(pathlib.Path(tempdir), "tensors_list_vec")

        assert len(tensors_list_vec2) == 2
        for (vector, vector2) in zip(tensors_list_vec, tensors_list_vec2):
            assert np.allclose(vector2.array, vector.array)


def test_backends_tensors_list_save_load_mat(
    tensors_list_mat: rbnicsx.backends.TensorsList,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check I/O for a rbnicsx.backends.TensorsList in the case of petsc4py.PETSc.Mat content."""
    with nbvalx.tempfile.TemporaryDirectory(tensors_list_mat.comm) as tempdir:
        tensors_list_mat.save(pathlib.Path(tempdir), "tensors_list_mat")

        tensors_list_mat2 = tensors_list_mat.duplicate()
        tensors_list_mat2.load(pathlib.Path(tempdir), "tensors_list_mat")

        assert len(tensors_list_mat2) == 2
        for (matrix, matrix2) in zip(tensors_list_mat, tensors_list_mat2):
            assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))


def test_backends_tensors_list_save_load_empty() -> None:
    """Check I/O for rbnicsx.backends.TensorsList when providing neither a Mat nor a Vec object."""
    fake_form = None
    empty_tensors_list = rbnicsx.backends.TensorsList(fake_form, mpi4py.MPI.COMM_WORLD)  # type: ignore[arg-type]

    with nbvalx.tempfile.TemporaryDirectory(empty_tensors_list.comm) as tempdir:
        with pytest.raises(RuntimeError):
            empty_tensors_list.save(pathlib.Path(tempdir), "empty_tensors_list")

        with pytest.raises(RuntimeError):
            empty_tensors_list.load(pathlib.Path(tempdir), "empty_tensors_list")


def test_backends_tensors_list_mul_vec(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__mul__ in the case of petsc4py.PETSc.Vec content."""
    online_vec = rbnicsx.online.create_vector(2)
    online_vec[0] = 3
    online_vec[1] = 5

    vector = tensors_list_vec * online_vec
    first_vector = dolfinx.fem.petsc.assemble_vector(tensors_list_vec.form)
    first_vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)  # type: ignore[attr-defined]
    assert np.allclose(vector.array, 13 * first_vector.array)


def test_backends_tensors_list_mul_mat(
    tensors_list_mat: rbnicsx.backends.TensorsList,
    to_dense_matrix: typing.Callable[  # type: ignore[name-defined]
        [petsc4py.PETSc.Mat], npt.NDArray[petsc4py.PETSc.ScalarType]]
) -> None:
    """Check rbnicsx.backends.TensorsList.__mul__ in the case of petsc4py.PETSc.Mat content."""
    online_vec = rbnicsx.online.create_vector(2)
    online_vec[0] = 3
    online_vec[1] = 5

    matrix = tensors_list_mat * online_vec
    first_matrix = dolfinx.fem.petsc.assemble_matrix(tensors_list_mat.form)
    first_matrix.assemble()
    assert np.allclose(to_dense_matrix(matrix), 13 * to_dense_matrix(first_matrix))


def test_backends_tensors_list_mul_empty() -> None:
    """Check rbnicsx.backends.TensorsList.__mul__ with empty list."""
    fake_form = None
    empty_tensors_list = rbnicsx.backends.TensorsList(fake_form, mpi4py.MPI.COMM_WORLD)  # type: ignore[arg-type]

    online_vec = rbnicsx.online.create_vector(0)
    should_be_none = empty_tensors_list * online_vec
    assert should_be_none is None


def test_backends_tensors_list_mul_not_implemented(tensors_list_vec: rbnicsx.backends.TensorsList) -> None:
    """Check rbnicsx.backends.TensorsList.__mul__ with an incorrect type."""
    with pytest.raises(TypeError):
        tensors_list_vec * None
