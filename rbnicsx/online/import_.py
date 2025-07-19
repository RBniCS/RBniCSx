# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to import matrices and vectors."""

import pathlib

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx._backends.import_ import (
    import_matrices as import_matrices_super, import_matrix as import_matrix_super,
    import_vector as import_vector_super, import_vectors as import_vectors_super)
from rbnicsx._backends.online_tensors import (
    create_online_matrix as create_matrix, create_online_matrix_block as create_matrix_block,
    create_online_vector as create_vector, create_online_vector_block as create_vector_block)


def import_matrix(
    M: int, N: int, directory: pathlib.Path, filename: str
) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
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


def import_matrix_block(
    M: list[int], N: list[int], directory: pathlib.Path, filename: str
) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
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


def import_matrices(
    M: int, N: int, directory: pathlib.Path, filename: str
) -> list[petsc4py.PETSc.Mat]:  # type: ignore[name-defined]
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


def import_matrices_block(
    M: list[int], N: list[int], directory: pathlib.Path, filename: str
) -> list[petsc4py.PETSc.Mat]:  # type: ignore[name-defined]
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


def import_vector(
    N: int, directory: pathlib.Path, filename: str
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
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


def import_vector_block(
    N: list[int], directory: pathlib.Path, filename: str
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
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


def import_vectors(
    N: int, directory: pathlib.Path, filename: str
) -> list[petsc4py.PETSc.Vec]:  # type: ignore[name-defined]
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


def import_vectors_block(
    N: list[int], directory: pathlib.Path, filename: str
) -> list[petsc4py.PETSc.Vec]:  # type: ignore[name-defined]
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
