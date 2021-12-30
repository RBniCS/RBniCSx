# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for minirox.backends.projection module."""

import mpi4py
import numpy as np
import petsc4py
import slepc4py

import minirox.backends


def test_projection_online_vector_size() -> None:
    """Check that the created online vector has the correct dimension."""
    online_vec = minirox.backends.create_online_vector(2)
    local_size, global_size = online_vec.getSizes()
    assert local_size == global_size
    assert global_size == 2


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


def test_projection_online_vector_set() -> None:
    """Set some entries in the created online vector."""
    online_vec = minirox.backends.create_online_vector(2)
    for i in range(2):
        online_vec.setValue(i, i + 1)
    online_vec.view()
    for i in range(2):
        assert online_vec[i] == i + 1


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
        def F(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, F_vec: petsc4py.PETSc.Vec) -> None:
            """Assemble the residual of the problem."""
            F_vec[0] = (x[0] - 1)**2
            F_vec[1] = (x[1] - 2)**2

        def J(self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, J_mat: petsc4py.PETSc.Mat,
              P_mat: petsc4py.PETSc.Mat) -> None:
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
