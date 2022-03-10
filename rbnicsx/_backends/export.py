# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to export PETSc matrices and vectors."""

import os
import typing

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx.io import on_rank_zero


def export_matrix(mat: petsc4py.PETSc.Mat, comm: mpi4py.MPI.Intracomm, directory: str, filename: str) -> None:
    """
    Export a petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mat : petsc4py.PETSc.Mat
        Matrix to be exported.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the matrix viewer.
    directory : str
        Directory where to export the matrix.
    filename : str
        Name of the file where to export the matrix.
    """
    os.makedirs(directory, exist_ok=True)
    viewer = petsc4py.PETSc.Viewer().createBinary(os.path.join(directory, filename + ".dat"), "w", comm)
    viewer.view(mat)
    viewer.destroy()


def export_matrices(
    mats: typing.List[petsc4py.PETSc.Mat], comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> None:
    """
    Export a list of petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mats : typing.List[petsc4py.PETSc.Mat]
        Matrices to be exported.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the matrix viewer.
    directory : str
        Directory where to export the matrix.
    filename : str
        Name of the file where to export the matrix.
    """
    os.makedirs(os.path.join(directory, filename), exist_ok=True)

    # Write out length of the list
    def write_length() -> None:
        with open(os.path.join(directory, filename, "length.dat"), "w") as length_file:
            length_file.write(str(len(mats)))
    on_rank_zero(comm, write_length)

    # Write out the list
    for (index, mat) in enumerate(mats):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "w", comm)
        viewer.view(mat)
        viewer.destroy()


def export_vector(vec: petsc4py.PETSc.Vec, comm: mpi4py.MPI.Intracomm, directory: str, filename: str) -> None:
    """
    Export a petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vec : petsc4py.PETSc.Vec
        Vector to be exported.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the vector viewer.
    directory : str
        Directory where to export the vector.
    filename : str
        Name of the file where to export the vector.
    """
    os.makedirs(directory, exist_ok=True)
    viewer = petsc4py.PETSc.Viewer().createBinary(os.path.join(directory, filename + ".dat"), "w", comm)
    viewer.view(vec)
    viewer.destroy()


def export_vectors(
    vecs: typing.List[petsc4py.PETSc.Vec], comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> None:
    """
    Export a list of petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vecs : typing.List[petsc4py.PETSc.Vec]
        Vectors to be exported.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the vector viewer.
    directory : str
        Directory where to export the vector.
    filename : str
        Name of the file where to export the vector.
    """
    os.makedirs(os.path.join(directory, filename), exist_ok=True)

    # Write out length of the list
    def write_length() -> None:
        with open(os.path.join(directory, filename, "length.dat"), "w") as length_file:
            length_file.write(str(len(vecs)))
    on_rank_zero(comm, write_length)

    # Write out the list
    for (index, vec) in enumerate(vecs):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "w", comm)
        viewer.view(vec)
        viewer.destroy()
