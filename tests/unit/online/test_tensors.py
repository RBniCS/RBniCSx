# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.tensors module."""

import mpi4py
import numpy as np
import petsc4py
import slepc4py

import rbnicsx.online


def test_online_vector_size() -> None:
    """Check that the created online vector has the correct dimension."""
    online_vec = rbnicsx.online.create_vector(2)
    local_size, global_size = online_vec.getSizes()
    assert local_size == global_size
    assert global_size == 2


def test_online_vector_block_size() -> None:
    """Check that the created online vector has the correct dimension (block initialization)."""
    online_vec = rbnicsx.online.create_vector_block([2, 3])
    local_size, global_size = online_vec.getSizes()
    assert local_size == global_size
    assert global_size == 5


def test_online_matrix_size() -> None:
    """Check that the created online matrix has the correct dimension."""
    online_mat = rbnicsx.online.create_matrix(2, 3)
    dimension0, dimension1 = online_mat.getSizes()
    local_size0, global_size0 = dimension0
    local_size1, global_size1 = dimension1
    assert local_size0 == global_size0
    assert local_size1 == global_size1
    assert global_size0 == 2
    assert global_size1 == 3


def test_online_matrix_block_size() -> None:
    """Check that the created online matrix has the correct dimension (block initialization)."""
    online_mat = rbnicsx.online.create_matrix_block([2, 3], [4, 5])
    dimension0, dimension1 = online_mat.getSizes()
    local_size0, global_size0 = dimension0
    local_size1, global_size1 = dimension1
    assert local_size0 == global_size0
    assert local_size1 == global_size1
    assert global_size0 == 5
    assert global_size1 == 9


def test_online_vector_set() -> None:
    """Set some entries in the created online vector."""
    online_vec = rbnicsx.online.create_vector(2)
    for i in range(2):
        online_vec.setValue(i, i + 1)
    online_vec.view()
    for i in range(2):
        assert online_vec[i] == i + 1


def test_online_vector_set_local() -> None:
    """Set some entries in the created online vector using the local setter."""
    online_vec = rbnicsx.online.create_vector(2)
    for i in range(2):
        online_vec.setValueLocal(i, i + 1)
    online_vec.view()
    for i in range(2):
        assert online_vec[i] == i + 1


def test_online_vector_block_set() -> None:
    """Set some entries in the created online vector (block initialization and fill)."""
    N = [2, 3]
    online_vec = rbnicsx.online.create_vector_block(N)
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


def test_online_matrix_set() -> None:
    """Set some entries in the created online matrix."""
    online_mat = rbnicsx.online.create_matrix(2, 3)
    for i in range(2):
        for j in range(3):
            online_mat.setValue(i, j, i * 2 + j + 1)
    online_mat.assemble()
    online_mat.view()
    for i in range(2):
        for j in range(3):
            assert online_mat[i, j] == i * 2 + j + 1


def test_online_matrix_set_local() -> None:
    """Set some entries in the created online matrix using the local setter."""
    online_mat = rbnicsx.online.create_matrix(2, 3)
    for i in range(2):
        for j in range(3):
            online_mat.setValueLocal(i, j, i * 2 + j + 1)
    online_mat.assemble()
    online_mat.view()
    for i in range(2):
        for j in range(3):
            assert online_mat[i, j] == i * 2 + j + 1


def test_online_matrix_block_set() -> None:
    """Set some entries in the created online matrix (block initialization and fill)."""
    M = [2, 3]
    N = [4, 5]
    online_mat = rbnicsx.online.create_matrix_block(M, N)
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


def test_online_linear_solve() -> None:
    """Solve a linear problem with online data structures."""
    online_vec = rbnicsx.online.create_vector(2)
    online_solution = rbnicsx.online.create_vector(2)
    online_mat = rbnicsx.online.create_matrix(2, 2)
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


def test_online_eigenvalue_solve() -> None:
    """Solve an eigenvalue problem with online data structures."""
    online_mat_left = rbnicsx.online.create_matrix(2, 2)
    online_mat_right = rbnicsx.online.create_matrix(2, 2)
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
        eigv_i_real = rbnicsx.online.create_vector(2)
        eigv_i_imag = rbnicsx.online.create_vector(2)
        eps.getEigenvector(i, eigv_i_real, eigv_i_imag)
        eigv_i_real.view()
        eigv_i_imag.view()
        assert not np.isclose(eigv_i_real[i], 0)
        assert np.isclose(eigv_i_real[1 - i], 0)
        assert np.isclose(eigv_i_real.norm(), np.sqrt(i + 1))
        assert np.isclose(eigv_i_imag.norm(), 0)


def test_online_nonlinear_solve() -> None:
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
    online_residual = rbnicsx.online.create_vector(2)
    snes.setFunction(problem.F, online_residual)
    online_jacobian = rbnicsx.online.create_matrix(2, 2)
    snes.setJacobian(problem.J, J=online_jacobian, P=None)

    snes.setTolerances(atol=1e-14, rtol=1e-14)
    snes.setMonitor(lambda _, it, residual: print(it, residual))

    online_solution = rbnicsx.online.create_vector(2)
    snes.solve(None, online_solution)
    online_solution.view()

    for i in range(2):
        assert np.isclose(online_solution[i], i + 1)
