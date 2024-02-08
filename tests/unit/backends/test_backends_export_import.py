# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.export and rbnicsx.backends.import_ modules."""

import pathlib
import typing

import adios4dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import mpi4py.MPI
import nbvalx.tempfile
import numpy as np
import numpy.typing
import petsc4py.PETSc
import pytest
import ufl

import rbnicsx.backends

all_families = ["Lagrange", "Discontinuous Lagrange"]
all_degrees = [1, 2]
all_repeat = [1, 2]


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(
        comm, 4 * comm.size, 4 * comm.size, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)


def mesh_generator_do_nothing(mesh: dolfinx.mesh.Mesh, path: pathlib.Path) -> dolfinx.mesh.Mesh:
    """Return the provided mesh."""
    return mesh


def mesh_generator_save_to_file(mesh: dolfinx.mesh.Mesh, path: pathlib.Path) -> dolfinx.mesh.Mesh:
    """Save the mesh to file with adios4dolfinx, and return the provided mesh."""
    adios4dolfinx.write_mesh(mesh, path, "bp4")
    return mesh


def mesh_generator_load_from_file(mesh: dolfinx.mesh.Mesh, path: pathlib.Path) -> dolfinx.mesh.Mesh:
    """Load the mesh from file with adios4dolfinx, and return the loaded mesh."""
    return adios4dolfinx.read_mesh(mesh.comm, path, engine="bp4", ghost_mode=dolfinx.mesh.GhostMode.shared_facet)


def mesh_generator_save_to_and_load_from_file(mesh: dolfinx.mesh.Mesh, path: pathlib.Path) -> dolfinx.mesh.Mesh:
    """Save the mesh to file with adios4dolfinx, load it back in, and return the loaded mesh."""
    adios4dolfinx.write_mesh(mesh, path, "bp4")
    return adios4dolfinx.read_mesh(mesh.comm, path, engine="bp4", ghost_mode=dolfinx.mesh.GhostMode.shared_facet)


all_mesh_generators = [
    (
        # use mesh from the mesh fixture when preparing output files
        lambda mesh, tempdir: mesh_generator_do_nothing(mesh, pathlib.Path(tempdir)),
        # use mesh from the mesh fixture when loading input files
        lambda mesh, tempdir: mesh_generator_do_nothing(mesh, pathlib.Path(tempdir)),
        # this case is expected to fail because the mesh must be read back in using adios4dolfinx
        False
    ),
    (
        # use mesh from the mesh fixture when preparing output files, but also save it to file with adios4dolfinx
        lambda mesh, tempdir: mesh_generator_save_to_file(mesh, pathlib.Path(tempdir) / "mesh.bp"),
        # and also use mesh from a standalone mesh checkpoint when loading input files
        lambda mesh, tempdir: mesh_generator_load_from_file(mesh, pathlib.Path(tempdir) / "mesh.bp"),
        # this case is expected to pass
        True
    ),
    (
        # save mesh from the mesh fixture to file with adios4dolfinx and load it back in for output file preparation
        lambda mesh, tempdir: mesh_generator_save_to_and_load_from_file(mesh, pathlib.Path(tempdir) / "mesh.bp"),
        # and also use mesh from a standalone mesh checkpoint when loading input files
        lambda mesh, tempdir: mesh_generator_load_from_file(mesh, pathlib.Path(tempdir) / "mesh.bp"),
        # this case is expected to fail in parallel due to https://github.com/jorgensd/adios4dolfinx/issues/62
        mpi4py.MPI.COMM_WORLD.size == 1
    ),
    (
        # use mesh from the mesh fixture when preparing output files
        lambda mesh, tempdir: mesh_generator_do_nothing(mesh, pathlib.Path(tempdir)),
        # but use mesh from the first function checkpoint when loading input files
        lambda mesh, tempdir: mesh_generator_load_from_file(mesh, pathlib.Path(tempdir) / "adios_0_checkpoint.bp"),
        # this case is expected to pass
        True
    ),
    (
        # use mesh from the mesh fixture when preparing output files, but also save it to file with adios4dolfinx
        lambda mesh, tempdir: mesh_generator_save_to_file(mesh, pathlib.Path(tempdir) / "mesh.bp"),
        # but use mesh from the first function checkpoint when loading input files
        lambda mesh, tempdir: mesh_generator_load_from_file(mesh, pathlib.Path(tempdir) / "adios_0_checkpoint.bp"),
        # this case is expected to pass
        True
    ),
    (
        # save mesh from the mesh fixture to file with adios4dolfinx and load it back in for output file preparation
        lambda mesh, tempdir: mesh_generator_save_to_and_load_from_file(mesh, pathlib.Path(tempdir) / "mesh.bp"),
        # but use mesh from the first function checkpoint when loading input files
        lambda mesh, tempdir: mesh_generator_load_from_file(mesh, pathlib.Path(tempdir) / "adios_0_checkpoint.bp"),
        # this case is expected to pass
        True
    )
]
all_mesh_generators_ids = [
    "in: do_nothing, out: do_nothing, expected_success: False",
    'in: save_to_file("mesh.bp"), out: load_from_file("mesh.bp"), expected_success: True',
    'in: save_to_and_load_from_file("mesh.bp"), out: load_from_file("mesh.bp"), expected_success: only in serial',
    'in: do_nothing, out: load_from_file("adios_0_checkpoint.bp"), expected_success: True',
    'in: save_to_file("mesh.bp"), out: load_from_file("adios_0_checkpoint.bp"), expected_success: True',
    'in: save_to_and_load_from_file("mesh.bp"), out: load_from_file("adios_0_checkpoint.bp"), expected_success: True'
]


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
@pytest.mark.parametrize("repeat", all_repeat)
@pytest.mark.parametrize(
    "mesh_out_generator,mesh_in_generator,expected_success", all_mesh_generators, ids=all_mesh_generators_ids)
def test_backends_export_import_function(
    mesh: dolfinx.mesh.Mesh, family: str, degree: str, repeat: int,
    mesh_out_generator: typing.Callable[[dolfinx.mesh.Mesh, str], dolfinx.mesh.Mesh],
    mesh_in_generator: typing.Callable[[dolfinx.mesh.Mesh, str], dolfinx.mesh.Mesh],
    expected_success: bool
) -> None:
    """Check I/O for a dolfinx.fem.Function."""
    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        def function_space_generator(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:
            """Create a function space on the provided mesh."""
            return dolfinx.fem.functionspace(mesh, (family, degree))

        def expression_generator(r: int) -> typing.Callable[
                [np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]:
            """Return the expression to be interpolated."""
            return lambda x: (r + 1) * (x[0]**2 + x[1]**3)

        mesh_out = mesh_out_generator(mesh, tempdir)
        V_out = function_space_generator(mesh_out)
        for r in range(repeat):
            function_out = dolfinx.fem.Function(V_out)
            function_out.interpolate(expression_generator(r))
            rbnicsx.backends.export_function(function_out, pathlib.Path(tempdir), f"adios_{r}")

        mesh_in = mesh_in_generator(mesh, tempdir)
        V_in = function_space_generator(mesh_in)
        for r in range(repeat):
            function_ex = dolfinx.fem.Function(V_in)
            function_ex.interpolate(expression_generator(r))
            function_in = rbnicsx.backends.import_function(V_in, pathlib.Path(tempdir), f"adios_{r}")
            if expected_success:
                assert np.allclose(function_in.vector.array, function_ex.vector.array)
            else:
                assert not np.allclose(function_in.vector.array, function_ex.vector.array)


@pytest.mark.parametrize("family", all_families)
@pytest.mark.parametrize("degree", all_degrees)
@pytest.mark.parametrize("repeat", all_repeat)
@pytest.mark.parametrize(
    "mesh_out_generator,mesh_in_generator,expected_success", all_mesh_generators, ids=all_mesh_generators_ids)
def test_backends_export_import_functions(
    mesh: dolfinx.mesh.Mesh, family: str, degree: str, repeat: int,
    mesh_out_generator: typing.Callable[[dolfinx.mesh.Mesh, str], dolfinx.mesh.Mesh],
    mesh_in_generator: typing.Callable[[dolfinx.mesh.Mesh, str], dolfinx.mesh.Mesh],
    expected_success: bool
) -> None:
    """Check I/O for a list of dolfinx.fem.Function."""
    with nbvalx.tempfile.TemporaryDirectory(mesh.comm) as tempdir:
        def function_space_generator(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:
            """Create a function space on the provided mesh."""
            return dolfinx.fem.functionspace(mesh, (family, degree))

        T = 4

        def expression_generator(r: int, t: int) -> typing.Callable[
                [np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]:
            """Return the expression to be interpolated."""
            return lambda x: (r + 1) * (x[0]**(t + 2) + x[1]**(t + 3))

        mesh_out = mesh_out_generator(mesh, tempdir)
        V_out = function_space_generator(mesh_out)
        for r in range(repeat):
            functions_out_r = list()
            indices_out_r = list()
            for t in range(T):
                function_out = dolfinx.fem.Function(V_out)
                function_out.interpolate(expression_generator(r, t))
                functions_out_r.append(function_out)
                indices_out_r.append(t)
            rbnicsx.backends.export_functions(
                functions_out_r, np.array(indices_out_r, dtype=float), pathlib.Path(tempdir), f"adios_{r}")

        mesh_in = mesh_in_generator(mesh, tempdir)
        V_in = function_space_generator(mesh_in)
        for r in range(repeat):
            functions_in = rbnicsx.backends.import_functions(V_in, pathlib.Path(tempdir), f"adios_{r}")
            assert len(functions_in) == T
            for t in range(T):
                function_ex_t = dolfinx.fem.Function(V_in)
                function_ex_t.interpolate(expression_generator(r, t))
                if expected_success:
                    assert np.allclose(functions_in[t].vector.array, function_ex_t.vector.array)
                else:
                    assert not np.allclose(functions_in[t].vector.array, function_ex_t.vector.array)


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
