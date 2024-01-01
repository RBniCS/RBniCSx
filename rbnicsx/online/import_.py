# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to import matrices and vectors."""

import typing

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx._backends.import_ import (
    import_matrices as import_matrices_super, import_matrix as import_matrix_super,
    import_vector as import_vector_super, import_vectors as import_vectors_super)
from rbnicsx._backends.online_tensors import (
    create_online_matrix as create_matrix, create_online_matrix_block as create_matrix_block,
    create_online_vector as create_vector, create_online_vector_block as create_vector_block)


def import_matrix(  # type: ignore[no-any-unimported]
    M: int, N: int, directory: str, filename: str
) -> petsc4py.PETSc.Mat:
    """
    Import a dense petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    M, N
        Dimension of the online matrix.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrix imported from file.
    """
    return import_matrix_super(lambda: create_matrix(M, N), mpi4py.MPI.COMM_WORLD, directory, filename)


def import_matrix_block(  # type: ignore[no-any-unimported]
    M: typing.List[int], N: typing.List[int], directory: str, filename: str
) -> petsc4py.PETSc.Mat:
    """
    Import a dense petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    M, N
        Dimension of the blocks of the matrix.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrix imported from file.
    """
    return import_matrix_super(lambda: create_matrix_block(M, N), mpi4py.MPI.COMM_WORLD, directory, filename)


def import_matrices(  # type: ignore[no-any-unimported]
    M: int, N: int, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Mat]:
    """
    Import a list of dense petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    M, N
        Dimension of each online matrix.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrices imported from file.
    """
    return import_matrices_super(lambda: create_matrix(M, N), mpi4py.MPI.COMM_WORLD, directory, filename)


def import_matrices_block(  # type: ignore[no-any-unimported]
    M: typing.List[int], N: typing.List[int], directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Mat]:
    """
    Import a list of dense petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    M, N
        Dimension of the blocks of the matrix.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrices imported from file.
    """
    return import_matrices_super(lambda: create_matrix_block(M, N), mpi4py.MPI.COMM_WORLD, directory, filename)


def import_vector(  # type: ignore[no-any-unimported]
    N: int, directory: str, filename: str
) -> petsc4py.PETSc.Vec:
    """
    Import a sequential petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    N
        Dimension of the online vector.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vector imported from file.
    """
    return import_vector_super(lambda: create_vector(N), mpi4py.MPI.COMM_WORLD, directory, filename)


def import_vector_block(  # type: ignore[no-any-unimported]
    N: typing.List[int], directory: str, filename: str
) -> petsc4py.PETSc.Vec:
    """
    Import a sequential petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    N
        Dimension of the blocks of the vector.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vector imported from file.
    """
    return import_vector_super(lambda: create_vector_block(N), mpi4py.MPI.COMM_WORLD, directory, filename)


def import_vectors(  # type: ignore[no-any-unimported]
    N: int, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Vec]:
    """
    Import a list of sequential petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    N
        Dimension of the online vector.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vectors imported from file.
    """
    return import_vectors_super(lambda: create_vector(N), mpi4py.MPI.COMM_WORLD, directory, filename)


def import_vectors_block(  # type: ignore[no-any-unimported]
    N: typing.List[int], directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Vec]:
    """
    Import a list of sequential petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    N
        Dimension of the blocks of the vector.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vectors imported from file.
    """
    return import_vectors_super(lambda: create_vector_block(N), mpi4py.MPI.COMM_WORLD, directory, filename)
