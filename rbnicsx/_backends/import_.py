# Copyright (C) 2021-2026 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to import PETSc matrices and vectors."""

import pathlib
import typing

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx.io import on_rank_zero


def import_matrix(
    allocate: typing.Callable[[], petsc4py.PETSc.Mat],  # type: ignore[name-defined]
    comm: mpi4py.MPI.Intracomm, directory: pathlib.Path, filename: str
) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
    """
    Import a petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    allocate
        A callable to allocate the storage.
    comm
        Communicator to be used while creating the matrix viewer.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrix imported from file.
    """
    viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
        str(directory / (filename + ".dat")), "r", comm)
    mat = allocate()
    mat.load(viewer)
    viewer.destroy()
    return mat


def import_matrices(
    allocate: typing.Callable[[], petsc4py.PETSc.Mat],  # type: ignore[name-defined]
    comm: mpi4py.MPI.Intracomm, directory: pathlib.Path, filename: str
) -> list[petsc4py.PETSc.Mat]:  # type: ignore[name-defined]
    """
    Import a list of petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    allocate
        A callable to allocate the storage.
    comm
        Communicator to be used while creating the matrix viewer.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrices imported from file.
    """
    # Read in length of the list
    def read_length() -> int:
        with open(directory / filename / "length.dat") as length_file:
            return int(length_file.readline())
    length = on_rank_zero(comm, read_length)

    # Read in the list
    mats = list()
    for index in range(length):
        viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
            str(directory / filename / (str(index) + ".dat")), "r", comm)
        mat = allocate()
        mat.load(viewer)
        mats.append(mat)
        viewer.destroy()
    return mats


def import_vector(
    allocate: typing.Callable[[], petsc4py.PETSc.Vec],  # type: ignore[name-defined]
    comm: mpi4py.MPI.Intracomm, directory: pathlib.Path, filename: str
) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
    """
    Import a petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    allocate
        A callable to allocate the storage.
    comm
        Communicator to be used while creating the vector viewer.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vector imported from file.
    """
    viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
        str(directory / (filename + ".dat")), "r", comm)
    vec = allocate()
    vec.load(viewer)
    viewer.destroy()
    return vec


def import_vectors(
    allocate: typing.Callable[[], petsc4py.PETSc.Vec],  # type: ignore[name-defined]
    comm: mpi4py.MPI.Intracomm, directory: pathlib.Path, filename: str
) -> list[petsc4py.PETSc.Vec]:  # type: ignore[name-defined]
    """
    Import a list of petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    allocate
        A callable to allocate the storage.
    comm
        Communicator to be used while creating the vector viewer.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vectors imported from file.
    """
    # Read in length of the list
    def read_length() -> int:
        with open(directory / filename / "length.dat") as length_file:
            return int(length_file.readline())
    length = on_rank_zero(comm, read_length)

    # Read in the list
    vecs = list()
    for index in range(length):
        viewer = petsc4py.PETSc.Viewer().createBinary(  # type: ignore[attr-defined]
            str(directory / filename / (str(index) + ".dat")), "r", comm)
        vec = allocate()
        vec.load(viewer)
        vecs.append(vec)
        viewer.destroy()
    return vecs
