# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox.backends.export and minirox.backends.import_ modules."""

import typing

import dolfinx.fem
import dolfinx.mesh
import dolfinx_utils.test.fixtures
import mpi4py
import numpy as np
import petsc4py
import pytest
import ufl

import minirox.backends

tempdir = dolfinx_utils.test.fixtures.tempdir


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


def test_export_import_function(mesh: dolfinx.mesh.Mesh, tempdir: str) -> None:
    """Check I/O for a dolfinx.fem.Function."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    function = dolfinx.fem.Function(V)
    function.vector.set(1.0)
    minirox.backends.export_function(function, tempdir, "function")

    function2 = minirox.backends.import_function(V, tempdir, "function")
    assert np.allclose(function2.vector.array, function.vector.array)


def test_export_import_functions(mesh: dolfinx.mesh.Mesh, tempdir: str) -> None:
    """Check I/O for a list of dolfinx.fem.Function."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    functions = list()
    indices = list()
    for i in range(2):
        function = dolfinx.fem.Function(V)
        function.vector.set(i + 1)
        functions.append(function)
        indices.append(i)
    minirox.backends.export_functions(functions, np.array(indices, dtype=float), tempdir, "functions")

    functions2 = minirox.backends.import_functions(V, tempdir, "functions")
    assert len(functions2) == 2
    for (function, function2) in zip(functions, functions2):
        assert np.allclose(function2.vector.array, function.vector.array)


def test_export_import_vector(mesh: dolfinx.mesh.Mesh, tempdir: str) -> None:
    """Check I/O for a petsc4py.PETSc.Vec."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_form = ufl.inner(1, v) * ufl.dx
    linear_form_cpp = dolfinx.fem.form(linear_form)
    vector = dolfinx.fem.assemble_vector(linear_form_cpp)
    vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    minirox.backends.export_vector(vector, tempdir, "vector")

    vector2 = minirox.backends.import_vector(linear_form_cpp, mesh.comm, tempdir, "vector")
    assert np.allclose(vector2.array, vector.array)


def test_export_import_vectors(mesh: dolfinx.mesh.Mesh, tempdir: str) -> None:
    """Check I/O for a list of petsc4py.PETSc.Vec."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(i + 1, v) * ufl.dx for i in range(2)]
    linear_forms_cpp = dolfinx.fem.form(linear_forms)
    vectors = [dolfinx.fem.assemble_vector(linear_form_cpp) for linear_form_cpp in linear_forms_cpp]
    [vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) for vector in vectors]
    minirox.backends.export_vectors(vectors, tempdir, "vectors")

    vectors2 = minirox.backends.import_vectors(linear_forms_cpp[0], mesh.comm, tempdir, "vectors")
    assert len(vectors2) == 2
    for (vector, vector2) in zip(vectors, vectors2):
        assert np.allclose(vector2.array, vector.array)


def test_export_import_matrix(mesh: dolfinx.mesh.Mesh, tempdir: str, to_dense_matrix: typing.Callable) -> None:
    """Check I/O for a petsc4py.PETSc.Mat."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_form = ufl.inner(u, v) * ufl.dx
    bilinear_form_cpp = dolfinx.fem.form(bilinear_form)
    matrix = dolfinx.fem.assemble_matrix(bilinear_form_cpp)
    matrix.assemble()
    minirox.backends.export_matrix(matrix, tempdir, "matrix")

    matrix2 = minirox.backends.import_matrix(bilinear_form_cpp, mesh.comm, tempdir, "matrix")
    assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))


def test_export_import_matrices(mesh: dolfinx.mesh.Mesh, tempdir: str, to_dense_matrix: typing.Callable) -> None:
    """Check I/O for a list of petsc4py.PETSc.Mat."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [(i + 1) * ufl.inner(u, v) * ufl.dx for i in range(2)]
    bilinear_forms_cpp = dolfinx.fem.form(bilinear_forms)
    matrices = [dolfinx.fem.assemble_matrix(bilinear_form_cpp) for bilinear_form_cpp in bilinear_forms_cpp]
    [matrix.assemble() for matrix in matrices]
    minirox.backends.export_matrices(matrices, tempdir, "matrices")

    matrices2 = minirox.backends.import_matrices(bilinear_forms_cpp[0], mesh.comm, tempdir, "matrices")
    for (matrix, matrix2) in zip(matrices, matrices2):
        assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))
