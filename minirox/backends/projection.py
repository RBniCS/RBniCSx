# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to project matrices and vectors on the reduced basis."""

import mpi4py
import petsc4py


def create_online_vector(N: int) -> petsc4py.PETSc.Vec:
    """
    Create an online vector of the given dimension.

    Parameters
    ----------
    N : int
        Dimension of the vector.

    Returns
    -------
    petsc4py.PETSc.Vec
        Allocated online vector.
    """
    return petsc4py.PETSc.Vec().createSeq(N, comm=mpi4py.MPI.COMM_SELF)


def create_online_matrix(M: int, N: int) -> petsc4py.PETSc.Mat:
    """
    Create an online matrix of the given dimension.

    Parameters
    ----------
    M, N : int
        Dimension of the matrix.

    Returns
    -------
    petsc4py.PETSc.Mat
        Allocated online matrix.
    """
    mat = petsc4py.PETSc.Mat().create(comm=mpi4py.MPI.COMM_SELF)
    mat.setType(petsc4py.PETSc.Mat.Type.DENSE)
    mat.setSizes((M, N))
    mat.setUp()
    return mat
