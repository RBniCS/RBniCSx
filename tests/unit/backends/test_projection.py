# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox.backends.projection module."""

import typing

import dolfinx.fem
import dolfinx.mesh
import mpi4py
import numpy as np
import petsc4py
import pytest
import slepc4py
import ufl

import minirox.backends


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.fixture
def functions_list(mesh: dolfinx.mesh.Mesh) -> minirox.backends.FunctionsList:
    """Generate a minirox.backends.FunctionsList with several entries."""
    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    functions_list = minirox.backends.FunctionsList(V)
    for i in range(14):
        function = dolfinx.fem.Function(V)
        with function.vector.localForm() as function_local:
            function_local.set(i + 1)
        functions_list.append(function)
    return functions_list


def test_projection_online_vector_size() -> None:
    """Check that the created online vector has the correct dimension."""
    online_vec = minirox.backends.create_online_vector(2)
    local_size, global_size = online_vec.getSizes()
    assert local_size == global_size
    assert global_size == 2


def test_projection_online_vector_block_size() -> None:
    """Check that the created online vector has the correct dimension (block initialization)."""
    online_vec = minirox.backends.create_online_vector_block([2, 3])
    local_size, global_size = online_vec.getSizes()
    assert local_size == global_size
    assert global_size == 5


def test_projection_online_matrix_size() -> None:
    """Check that the created online matrix has the correct dimension."""
    online_mat = minirox.backends.create_online_matrix(2, 3)
    dimension0, dimension1 = online_mat.getSizes()
    local_size0, global_size0 = dimension0
    local_size1, global_size1 = dimension1
    assert local_size0 == global_size0
    assert local_size1 == global_size1
    assert global_size0 == 2
    assert global_size1 == 3


def test_projection_online_matrix_block_size() -> None:
    """Check that the created online matrix has the correct dimension (block initialization)."""
    online_mat = minirox.backends.create_online_matrix_block([2, 3], [4, 5])
    dimension0, dimension1 = online_mat.getSizes()
    local_size0, global_size0 = dimension0
    local_size1, global_size1 = dimension1
    assert local_size0 == global_size0
    assert local_size1 == global_size1
    assert global_size0 == 5
    assert global_size1 == 9


def test_projection_online_vector_set() -> None:
    """Set some entries in the created online vector."""
    online_vec = minirox.backends.create_online_vector(2)
    for i in range(2):
        online_vec.setValue(i, i + 1)
    online_vec.view()
    for i in range(2):
        assert online_vec[i] == i + 1


def test_projection_online_vector_set_local() -> None:
    """Set some entries in the created online vector using the local setter."""
    online_vec = minirox.backends.create_online_vector(2)
    for i in range(2):
        online_vec.setValueLocal(i, i + 1)
    online_vec.view()
    for i in range(2):
        assert online_vec[i] == i + 1


def test_projection_online_vector_block_set() -> None:
    """Set some entries in the created online vector (block initialization and fill)."""
    N = [2, 3]
    online_vec = minirox.backends.create_online_vector_block(N)
    blocks = np.hstack((0, np.cumsum(N)))
    for I in range(2):  # noqa: E741
        is_I = petsc4py.PETSc.IS().createGeneral(
            np.arange(*blocks[I:I + 2], dtype=np.int32), comm=online_vec.comm)
        is_I.view()
        online_vec_I = online_vec.getSubVector(is_I)
        for i in range(N[I]):
            # online_vec_I.setValueLocal(i, ...) raises an error
            online_vec_I.setValue(i, (I + 1) * 10 + (i + 1))
        online_vec.restoreSubVector(is_I, online_vec_I)
        is_I.destroy()
    online_vec.view()
    for I in range(2):  # noqa: E741
        for i in range(N[I]):
            assert online_vec[blocks[I] + i] == (I + 1) * 10 + (i + 1)


def test_projection_online_matrix_set() -> None:
    """Set some entries in the created online matrix."""
    online_mat = minirox.backends.create_online_matrix(2, 3)
    for i in range(2):
        for j in range(3):
            online_mat.setValue(i, j, i * 2 + j + 1)
    online_mat.assemble()
    online_mat.view()
    for i in range(2):
        for j in range(3):
            assert online_mat[i, j] == i * 2 + j + 1


def test_projection_online_matrix_set_local() -> None:
    """Set some entries in the created online matrix using the local setter."""
    online_mat = minirox.backends.create_online_matrix(2, 3)
    for i in range(2):
        for j in range(3):
            online_mat.setValueLocal(i, j, i * 2 + j + 1)
    online_mat.assemble()
    online_mat.view()
    for i in range(2):
        for j in range(3):
            assert online_mat[i, j] == i * 2 + j + 1


def test_projection_online_matrix_block_set() -> None:
    """Set some entries in the created online matrix (block initialization and fill)."""
    M = [2, 3]
    N = [4, 5]
    online_mat = minirox.backends.create_online_matrix_block(M, N)
    row_blocks = np.hstack((0, np.cumsum(M)))
    col_blocks = np.hstack((0, np.cumsum(N)))
    for I in range(2):  # noqa: E741
        is_I = petsc4py.PETSc.IS().createGeneral(
            np.arange(*row_blocks[I:I + 2], dtype=np.int32), comm=online_mat.comm)
        is_I.view()
        for J in range(2):
            is_J = petsc4py.PETSc.IS().createGeneral(
                np.arange(*col_blocks[J:J + 2], dtype=np.int32), comm=online_mat.comm)
            is_J.view()
            online_mat_IJ = online_mat.getLocalSubMatrix(is_I, is_J)
            for i in range(M[I]):
                for j in range(N[J]):
                    # online_mat_IJ.setValue(i, j, ...) causes a segmentation fault
                    online_mat_IJ.setValueLocal(i, j, (I + 1) * 1000 + (J + 1) * 100 + (i + 1) * 10 + (j + 1))
            online_mat.restoreLocalSubMatrix(is_I, is_J, online_mat_IJ)
            is_J.destroy()
        is_I.destroy()
    online_mat.assemble()
    online_mat.view()
    for I in range(2):  # noqa: E741
        for J in range(2):
            for i in range(M[I]):
                for j in range(N[J]):
                    assert (
                        online_mat[row_blocks[I] + i, col_blocks[J] + j]
                        == (I + 1) * 1000 + (J + 1) * 100 + (i + 1) * 10 + (j + 1))


def test_projection_online_linear_solve() -> None:
    """Solve a linear problem with online data structures."""
    online_vec = minirox.backends.create_online_vector(2)
    online_solution = minirox.backends.create_online_vector(2)
    online_mat = minirox.backends.create_online_matrix(2, 2)
    for i in range(2):
        online_vec.setValue(i, i + 1)
        online_mat.setValue(i, i, 1 / (i + 1))
    online_mat.assemble()

    ksp = petsc4py.PETSc.KSP().create(online_solution.comm)
    ksp.setType(petsc4py.PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(petsc4py.PETSc.PC.Type.LU)
    ksp.setFromOptions()
    ksp.setOperators(online_mat)
    ksp.solve(online_vec, online_solution)
    online_solution.view()

    for i in range(2):
        assert np.isclose(online_solution[i], (i + 1)**2)


def test_projection_online_eigenvalue_solve() -> None:
    """Solve an eigenvalue problem with online data structures."""
    online_mat_left = minirox.backends.create_online_matrix(2, 2)
    online_mat_right = minirox.backends.create_online_matrix(2, 2)
    for i in range(2):
        online_mat_left.setValue(i, i, i + 1)
        online_mat_right.setValue(i, i, 1 / (i + 1))
    online_mat_left.assemble()
    online_mat_right.assemble()

    eps = slepc4py.SLEPc.EPS().create(online_mat_left.comm)
    eps.setType(slepc4py.SLEPc.EPS.Type.LAPACK)
    eps.setProblemType(slepc4py.SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setFromOptions()
    eps.setOperators(online_mat_left, online_mat_right)
    eps.solve()

    assert eps.getConverged() == 2
    for i in range(2):
        assert np.isclose(eps.getEigenvalue(i), (i + 1)**2)
        eigv_i_real = minirox.backends.create_online_vector(2)
        eigv_i_imag = minirox.backends.create_online_vector(2)
        eps.getEigenvector(i, eigv_i_real, eigv_i_imag)
        eigv_i_real.view()
        eigv_i_imag.view()
        assert not np.isclose(eigv_i_real[i], 0)
        assert np.isclose(eigv_i_real[1 - i], 0)
        assert np.isclose(eigv_i_real.norm(), np.sqrt(i + 1))
        assert np.isclose(eigv_i_imag.norm(), 0)


def test_projection_online_nonlinear_solve() -> None:
    """Solve a nonlinear problem with online data structures."""
    class NonlinearProblem(object):
        """Define a nonlinear problem."""

        def F(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, F_vec: petsc4py.PETSc.Vec) -> None:
            """Assemble the residual of the problem."""
            F_vec[0] = (x[0] - 1)**2
            F_vec[1] = (x[1] - 2)**2

        def J(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, J_mat: petsc4py.PETSc.Mat,
              P_mat: petsc4py.PETSc.Mat) -> None:
            """Assemble the jacobian of the problem."""
            J_mat[0, 0] = 2 * x[0] - 2
            J_mat[1, 1] = 2 * x[1] - 4
            J_mat.assemble()

    snes = petsc4py.PETSc.SNES().create(mpi4py.MPI.COMM_SELF)
    snes.getKSP().setType(petsc4py.PETSc.KSP.Type.PREONLY)
    snes.getKSP().getPC().setType(petsc4py.PETSc.PC.Type.LU)

    problem = NonlinearProblem()
    online_residual = minirox.backends.create_online_vector(2)
    snes.setFunction(problem.F, online_residual)
    online_jacobian = minirox.backends.create_online_matrix(2, 2)
    snes.setJacobian(problem.J, J=online_jacobian, P=None)

    snes.setTolerances(atol=1e-14, rtol=1e-14)
    snes.setMonitor(lambda _, it, residual: print(it, residual))

    online_solution = minirox.backends.create_online_vector(2)
    snes.solve(None, online_solution)
    online_solution.view()

    for i in range(2):
        assert np.isclose(online_solution[i], i + 1)


def test_projection_vector(mesh: dolfinx.mesh.Mesh, functions_list: minirox.backends.FunctionsList) -> None:
    """Test projection of a linear form onto the reduced basis."""
    basis_functions = functions_list[:2]

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_form = ufl.inner(1, v) * ufl.dx

    online_vec = minirox.backends.project_vector(linear_form, basis_functions)
    assert online_vec.size == 2
    assert np.allclose(online_vec.array, [1, 2])

    online_vec2 = minirox.backends.project_vector(0.4 * linear_form, basis_functions)
    minirox.backends.project_vector(online_vec2, 0.6 * linear_form, basis_functions)
    assert online_vec2.size == 2
    assert np.allclose(online_vec2.array, online_vec.array)


def test_projection_vector_block(mesh: dolfinx.mesh.Mesh, functions_list: minirox.backends.FunctionsList) -> None:
    """Test projection of a list of linear forms onto the reduced basis."""
    basis_functions = [functions_list[:2], functions_list[2:5]]

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    linear_forms = [ufl.inner(10**i, v) * ufl.dx for i in range(2)]

    online_vec = minirox.backends.project_vector_block(linear_forms, basis_functions)
    assert online_vec.size == 5
    assert np.allclose(online_vec[0:2], [1, 2])
    assert np.allclose(online_vec[2:5], np.array([3, 4, 5]) * 10)

    online_vec2 = minirox.backends.project_vector_block(
        [0.4 * linear_form for linear_form in linear_forms], basis_functions)
    minirox.backends.project_vector_block(
        online_vec2, [0.6 * linear_form for linear_form in linear_forms], basis_functions)
    assert online_vec2.size == 5
    assert np.allclose(online_vec2.array, online_vec.array)


def test_projection_matrix_galerkin(
    mesh: dolfinx.mesh.Mesh, functions_list: minirox.backends.FunctionsList, to_dense_matrix: typing.Callable
) -> None:
    """Test projection of a bilinear form onto the reduced basis (for use in Galerkin methods)."""
    basis_functions = functions_list[:2]

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_form = ufl.inner(u, v) * ufl.dx

    online_mat = minirox.backends.project_matrix(bilinear_form, basis_functions)
    assert online_mat.size == (2, 2)
    assert np.allclose(online_mat[0, :], [1, 2])
    assert np.allclose(online_mat[1, :], np.array([1, 2]) * 2)

    online_mat2 = minirox.backends.project_matrix(0.4 * bilinear_form, basis_functions)
    minirox.backends.project_matrix(online_mat2, 0.6 * bilinear_form, basis_functions)
    assert online_mat2.size == (2, 2)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))


def test_projection_matrix_petrov_galerkin(
    mesh: dolfinx.mesh.Mesh, functions_list: minirox.backends.FunctionsList, to_dense_matrix: typing.Callable
) -> None:
    """Test projection of a bilinear form onto the reduced basis (for use in Petrov-Galerkin methods)."""
    basis_functions = (functions_list[:2], functions_list[2:5])

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_form = ufl.inner(u, v) * ufl.dx

    online_mat = minirox.backends.project_matrix(bilinear_form, basis_functions)
    assert online_mat.size == (2, 3)
    assert np.allclose(online_mat[0, :], [3, 4, 5])
    assert np.allclose(online_mat[1, :], np.array([3, 4, 5]) * 2)

    online_mat2 = minirox.backends.project_matrix(0.4 * bilinear_form, basis_functions)
    minirox.backends.project_matrix(online_mat2, 0.6 * bilinear_form, basis_functions)
    assert online_mat2.size == (2, 3)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))


def test_projection_matrix_block_galerkin(
    mesh: dolfinx.mesh.Mesh, functions_list: minirox.backends.FunctionsList, to_dense_matrix: typing.Callable
) -> None:
    """Test projection of a matrix of bilinear forms onto the reduced basis (for use in Galerkin methods)."""
    basis_functions = [functions_list[:2], functions_list[2:5]]

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [[10**i * (-1)**j * ufl.inner(u, v) * ufl.dx for j in range(2)] for i in range(2)]

    online_mat = minirox.backends.project_matrix_block(bilinear_forms, basis_functions)
    assert online_mat.size == (5, 5)
    assert np.allclose(online_mat[0, 0:2], [1, 2])
    assert np.allclose(online_mat[0, 2:5], np.array([3, 4, 5]) * -1)
    assert np.allclose(online_mat[1, 0:2], np.array([1, 2]) * 2)
    assert np.allclose(online_mat[1, 2:5], np.array([3, 4, 5]) * -2)
    assert np.allclose(online_mat[2, 0:2], np.array([1, 2]) * 30)
    assert np.allclose(online_mat[2, 2:5], np.array([3, 4, 5]) * -30)
    assert np.allclose(online_mat[3, 0:2], np.array([1, 2]) * 40)
    assert np.allclose(online_mat[3, 2:5], np.array([3, 4, 5]) * -40)
    assert np.allclose(online_mat[4, 0:2], np.array([1, 2]) * 50)
    assert np.allclose(online_mat[4, 2:5], np.array([3, 4, 5]) * -50)

    online_mat2 = minirox.backends.project_matrix_block(
        [[0.4 * bilinear_form for bilinear_form in bilinear_forms_] for bilinear_forms_ in bilinear_forms],
        basis_functions)
    minirox.backends.project_matrix_block(
        online_mat2,
        [[0.6 * bilinear_form for bilinear_form in bilinear_forms_] for bilinear_forms_ in bilinear_forms],
        basis_functions)
    assert online_mat2.size == (5, 5)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))


def test_projection_matrix_block_petrov_galerkin(
    mesh: dolfinx.mesh.Mesh, functions_list: minirox.backends.FunctionsList, to_dense_matrix: typing.Callable
) -> None:
    """Test projection of a matrix of bilinear forms onto the reduced basis (for use in Petrov-Galerkin methods)."""
    basis_functions = ([functions_list[:2], functions_list[2:5]], [functions_list[5:9], functions_list[9:14]])

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    bilinear_forms = [[10**i * (-1)**j * ufl.inner(u, v) * ufl.dx for j in range(2)] for i in range(2)]

    online_mat = minirox.backends.project_matrix_block(bilinear_forms, basis_functions)
    assert online_mat.size == (5, 9)
    assert np.allclose(online_mat[0, 0:4], [6, 7, 8, 9])
    assert np.allclose(online_mat[0, 4:9], np.array([10, 11, 12, 13, 14]) * -1)
    assert np.allclose(online_mat[1, 0:4], np.array([6, 7, 8, 9]) * 2)
    assert np.allclose(online_mat[1, 4:9], np.array([10, 11, 12, 13, 14]) * -2)
    assert np.allclose(online_mat[2, 0:4], np.array([6, 7, 8, 9]) * 30)
    assert np.allclose(online_mat[2, 4:9], np.array([10, 11, 12, 13, 14]) * -30)
    assert np.allclose(online_mat[3, 0:4], np.array([6, 7, 8, 9]) * 40)
    assert np.allclose(online_mat[3, 4:9], np.array([10, 11, 12, 13, 14]) * -40)
    assert np.allclose(online_mat[4, 0:4], np.array([6, 7, 8, 9]) * 50)
    assert np.allclose(online_mat[4, 4:9], np.array([10, 11, 12, 13, 14]) * -50)

    online_mat2 = minirox.backends.project_matrix_block(
        [[0.4 * bilinear_form for bilinear_form in bilinear_forms_] for bilinear_forms_ in bilinear_forms],
        basis_functions)
    minirox.backends.project_matrix_block(
        online_mat2,
        [[0.6 * bilinear_form for bilinear_form in bilinear_forms_] for bilinear_forms_ in bilinear_forms],
        basis_functions)
    assert online_mat2.size == (5, 9)
    assert np.allclose(to_dense_matrix(online_mat2), to_dense_matrix(online_mat))
