# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to export PETSc matrices and vectors."""

import pathlib

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx.io import on_rank_zero


def export_matrix(
    mat: petsc4py.PETSc.Mat, comm: mpi4py.MPI.Intracomm,  # type: ignore[name-defined]
    directory: pathlib.Path, filename: str
) -> None:
    """
    Export a petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mat
        Matrix to be exported.
    comm
        Communicator to be used while creating the matrix viewer.
    directory
        Directory where to export the matrix.
    filename
        Name of the file where to export the matrix.
    """
    directory.mkdir(parents=True, exist_ok=True)
    viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
        str(directory / (filename + ".dat")), "w", comm)
    viewer.view(mat)
    viewer.destroy()


def export_matrices(
    mats: list[petsc4py.PETSc.Mat], comm: mpi4py.MPI.Intracomm,  # type: ignore[name-defined]
    directory: pathlib.Path, filename: str
) -> None:
    """
    Export a list of petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mats
        Matrices to be exported.
    comm
        Communicator to be used while creating the matrix viewer.
    directory
        Directory where to export the matrix.
    filename
        Name of the file where to export the matrix.
    """
    (directory / filename).mkdir(parents=True, exist_ok=True)

    # Write out length of the list
    def write_length() -> None:
        with open(directory / filename / "length.dat", "w") as length_file:
            length_file.write(str(len(mats)))
    on_rank_zero(comm, write_length)

    # Write out the list
    for (index, mat) in enumerate(mats):
        viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
            str(directory / filename / (str(index) + ".dat")), "w", comm)
        viewer.view(mat)
        viewer.destroy()


def export_vector(
    vec: petsc4py.PETSc.Vec, comm: mpi4py.MPI.Intracomm,  # type: ignore[name-defined]
    directory: pathlib.Path, filename: str
) -> None:
    """
    Export a petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vec
        Vector to be exported.
    comm
        Communicator to be used while creating the vector viewer.
    directory
        Directory where to export the vector.
    filename
        Name of the file where to export the vector.
    """
    directory.mkdir(parents=True, exist_ok=True)
    viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
        str(directory / (filename + ".dat")), "w", comm)
    viewer.view(vec)
    viewer.destroy()


def export_vectors(
    vecs: list[petsc4py.PETSc.Vec], comm: mpi4py.MPI.Intracomm,  # type: ignore[name-defined]
    directory: pathlib.Path, filename: str
) -> None:
    """
    Export a list of petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vecs
        Vectors to be exported.
    comm
        Communicator to be used while creating the vector viewer.
    directory
        Directory where to export the vector.
    filename
        Name of the file where to export the vector.
    """
    (directory / filename).mkdir(parents=True, exist_ok=True)

    # Write out length of the list
    def write_length() -> None:
        with open(directory / filename / "length.dat", "w") as length_file:
            length_file.write(str(len(vecs)))
    on_rank_zero(comm, write_length)

    # Write out the list
    for (index, vec) in enumerate(vecs):
        viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
            str(directory / filename / (str(index) + ".dat")), "w", comm)
        viewer.view(vec)
        viewer.destroy()
