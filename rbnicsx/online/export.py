# Copyright (C) 2021-2026 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to export matrices and vectors."""

import pathlib

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx._backends.export import (
    export_matrices as export_matrices_super, export_matrix as export_matrix_super,
    export_vector as export_vector_super, export_vectors as export_vectors_super)


def export_matrix(
    mat: petsc4py.PETSc.Mat, directory: pathlib.Path, filename: str  # type: ignore[name-defined]
) -> None:
    """
    Export a dense petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mat
        Online matrix to be exported.
    directory
        Directory where to export the matrix.
    filename
        Name of the file where to export the matrix.
    """
    assert mat.getType() == petsc4py.PETSc.Mat.Type.SEQDENSE  # type: ignore[attr-defined]
    export_matrix_super(mat, mpi4py.MPI.COMM_WORLD, directory, filename)


export_matrix_block = export_matrix


def export_matrices(
    mats: list[petsc4py.PETSc.Mat], directory: pathlib.Path, filename: str  # type: ignore[name-defined]
) -> None:
    """
    Export a list of dense petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mats
        Online matrices to be exported.
    directory
        Directory where to export the matrix.
    filename
        Name of the file where to export the matrix.
    """
    assert all([mat.getType() == petsc4py.PETSc.Mat.Type.SEQDENSE for mat in mats])  # type: ignore[attr-defined]
    export_matrices_super(mats, mpi4py.MPI.COMM_WORLD, directory, filename)


export_matrices_block = export_matrices


def export_vector(
    vec: petsc4py.PETSc.Vec, directory: pathlib.Path, filename: str  # type: ignore[name-defined]
) -> None:
    """
    Export a sequential petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vec
        Online vector to be exported.
    directory
        Directory where to export the vector.
    filename
        Name of the file where to export the vector.
    """
    assert vec.getType() == petsc4py.PETSc.Vec.Type.SEQ  # type: ignore[attr-defined]
    export_vector_super(vec, mpi4py.MPI.COMM_WORLD, directory, filename)


export_vector_block = export_vector


def export_vectors(
    vecs: list[petsc4py.PETSc.Vec], directory: pathlib.Path, filename: str  # type: ignore[name-defined]
) -> None:
    """
    Export a list of sequential petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vecs
        Online vectors to be exported.
    directory
        Directory where to export the vector.
    filename
        Name of the file where to export the vector.
    """
    assert all([vec.getType() == petsc4py.PETSc.Vec.Type.SEQ for vec in vecs])  # type: ignore[attr-defined]
    export_vectors_super(vecs, mpi4py.MPI.COMM_WORLD, directory, filename)


export_vectors_block = export_vectors
