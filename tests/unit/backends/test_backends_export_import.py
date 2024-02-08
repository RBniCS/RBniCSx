# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.export and rbnicsx.backends.import_ modules."""

import pathlib
import typing

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import mpi4py.MPI
import nbvalx.tempfile
import numpy as np
import petsc4py.PETSc
import pytest
import ufl

import rbnicsx.backends

all_families = ["Lagrange", "Discontinuous Lagrange"]
all_degrees = [1, 2]


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
def test_backends_export_import_function(mesh: dolfinx.mesh.Mesh, family: str, degree: str) -> None:
    """Check I/O for a dolfinx.fem.Function."""
    V = dolfinx.fem.functionspace(mesh, (family, degree))
    function = dolfinx.fem.Function(V)
    function.vector.set(1.0)

    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        rbnicsx.backends.export_function(function, pathlib.Path(tempdir), "function")

        function2 = rbnicsx.backends.import_function(V, pathlib.Path(tempdir), "function")
        assert np.allclose(function2.vector.array, function.vector.array)


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
def test_backends_export_import_functions(mesh: dolfinx.mesh.Mesh, family: str, degree: str) -> None:
    """Check I/O for a list of dolfinx.fem.Function."""
    V = dolfinx.fem.functionspace(mesh, (family, degree))
    functions = list()
    indices = list()
    for i in range(2):
        function = dolfinx.fem.Function(V)
        function.vector.set(i + 1)
        functions.append(function)
        indices.append(i)

    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        rbnicsx.backends.export_functions(
            functions, np.array(indices, dtype=float), pathlib.Path(tempdir), "functions")

        functions2 = rbnicsx.backends.import_functions(V, pathlib.Path(tempdir), "functions")
        assert len(functions2) == 2
        for (function, function2) in zip(functions, functions2):
            assert np.allclose(function2.vector.array, function.vector.array)


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
def test_backends_export_import_vector(mesh: dolfinx.mesh.Mesh, family: str, degree: str) -> None:
    """Check I/O for a petsc4py.PETSc.Vec."""
    V = dolfinx.fem.functionspace(mesh, (family, degree))
    v = ufl.TestFunction(V)
    linear_form = ufl.inner(1, v) * ufl.dx
    linear_form_cpp = dolfinx.fem.form(linear_form)
    vector = dolfinx.fem.petsc.assemble_vector(linear_form_cpp)
    vector.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)

    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        rbnicsx.backends.export_vector(vector, pathlib.Path(tempdir), "vector")

        vector2 = rbnicsx.backends.import_vector(linear_form_cpp, mesh.comm, pathlib.Path(tempdir), "vector")
        assert np.allclose(vector2.array, vector.array)


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
def test_backends_export_import_vectors(mesh: dolfinx.mesh.Mesh, family: str, degree: str) -> None:
    """Check I/O for a list of petsc4py.PETSc.Vec."""
    V = dolfinx.fem.functionspace(mesh, (family, degree))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(i + 1, v) * ufl.dx for i in range(2)]
    linear_forms_cpp = dolfinx.fem.form(linear_forms)
    vectors = [dolfinx.fem.petsc.assemble_vector(linear_form_cpp) for linear_form_cpp in linear_forms_cpp]
    [vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) for vector in vectors]

    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        rbnicsx.backends.export_vectors(vectors, pathlib.Path(tempdir), "vectors")

        vectors2 = rbnicsx.backends.import_vectors(linear_forms_cpp[0], mesh.comm, pathlib.Path(tempdir), "vectors")
        assert len(vectors2) == 2
        for (vector, vector2) in zip(vectors, vectors2):
            assert np.allclose(vector2.array, vector.array)


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
def test_backends_export_import_matrix(  # type: ignore[no-any-unimported]
    mesh: dolfinx.mesh.Mesh,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]],
    family: str, degree: str
) -> None:
    """Check I/O for a petsc4py.PETSc.Mat."""
    V = dolfinx.fem.functionspace(mesh, (family, degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_form = ufl.inner(u, v) * ufl.dx
    bilinear_form_cpp = dolfinx.fem.form(bilinear_form)
    matrix = dolfinx.fem.petsc.assemble_matrix(bilinear_form_cpp)
    matrix.assemble()

    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        rbnicsx.backends.export_matrix(matrix, pathlib.Path(tempdir), "matrix")

        matrix2 = rbnicsx.backends.import_matrix(bilinear_form_cpp, mesh.comm, pathlib.Path(tempdir), "matrix")
        assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
def test_backends_export_import_matrices(  # type: ignore[no-any-unimported]
    mesh: dolfinx.mesh.Mesh,
    to_dense_matrix: typing.Callable[[petsc4py.PETSc.Mat], np.typing.NDArray[petsc4py.PETSc.ScalarType]],
    family: str, degree: str
) -> None:
    """Check I/O for a list of petsc4py.PETSc.Mat."""
    V = dolfinx.fem.functionspace(mesh, (family, degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [(i + 1) * ufl.inner(u, v) * ufl.dx for i in range(2)]
    bilinear_forms_cpp = dolfinx.fem.form(bilinear_forms)
    matrices = [dolfinx.fem.petsc.assemble_matrix(bilinear_form_cpp) for bilinear_form_cpp in bilinear_forms_cpp]
    [matrix.assemble() for matrix in matrices]

    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        rbnicsx.backends.export_matrices(matrices, pathlib.Path(tempdir), "matrices")

        matrices2 = rbnicsx.backends.import_matrices(
            bilinear_forms_cpp[0], mesh.comm, pathlib.Path(tempdir), "matrices")
        for (matrix, matrix2) in zip(matrices, matrices2):
            assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))
