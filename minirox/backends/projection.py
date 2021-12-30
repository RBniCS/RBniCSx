# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to project matrices and vectors on the reduced basis."""

import typing

import mpi4py
import numpy as np
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
    vec = petsc4py.PETSc.Vec().createSeq(N, comm=mpi4py.MPI.COMM_SELF)
    # Attach the identity local-to-global map
    lgmap = petsc4py.PETSc.LGMap().create(np.arange(N, dtype=np.int32), comm=vec.comm)
    vec.setLGMap(lgmap)
    lgmap.destroy()
    # Setup and return
    vec.setUp()
    return vec


def create_online_vector_block(N: typing.List[int]) -> petsc4py.PETSc.Vec:
    """
    Create an online vector of the given block dimensions.

    Parameters
    ----------
    N : typing.List[int]
        Dimension of the blocks of the vector.

    Returns
    -------
    petsc4py.PETSc.Vec
        Allocated online vector.
    """
    return create_online_vector(sum(N))


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
    mat = petsc4py.PETSc.Mat().createDense((M, N), comm=mpi4py.MPI.COMM_SELF)
    # Attach the identity local-to-global map
    row_lgmap = petsc4py.PETSc.LGMap().create(np.arange(M, dtype=np.int32), comm=mat.comm)
    col_lgmap = petsc4py.PETSc.LGMap().create(np.arange(N, dtype=np.int32), comm=mat.comm)
    mat.setLGMap(row_lgmap, col_lgmap)
    row_lgmap.destroy()
    col_lgmap.destroy()
    # Setup and return
    mat.setUp()
    return mat


def create_online_matrix_block(M: typing.List[int], N: typing.List[int]) -> petsc4py.PETSc.Mat:
    """
    Create an online matrix of the given block dimensions.

    Parameters
    ----------
    M, N : typing.List[int]
        Dimension of the blocks of the matrix.

    Returns
    -------
    petsc4py.PETSc.Mat
        Allocated online matrix.
    """
    return create_online_matrix(sum(M), sum(N))
