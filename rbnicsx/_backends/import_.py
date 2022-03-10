# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to import PETSc matrices and vectors."""

import os
import typing

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx.io import on_rank_zero


def import_matrix(
    allocate: typing.Callable, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> petsc4py.PETSc.Mat:
    """
    Import a petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    allocate : typing.Callable
        A callable to allocate the storage.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the matrix viewer.
    directory : str
        Directory where to import the matrix from.
    filename : str
        Name of the file where to import the matrix from.

    Returns
    -------
    petsc4py.PETSc.Mat
        Matrix imported from file.
    """
    viewer = petsc4py.PETSc.Viewer().createBinary(os.path.join(directory, filename + ".dat"), "r", comm)
    mat = allocate()
    mat.load(viewer)
    viewer.destroy()
    return mat


def import_matrices(
    allocate: typing.Callable, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Mat]:
    """
    Import a list of petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    allocate : typing.Callable
        A callable to allocate the storage.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the matrix viewer.
    directory : str
        Directory where to import the matrix from.
    filename : str
        Name of the file where to import the matrix from.

    Returns
    -------
    typing.List[petsc4py.PETSc.Mat]
        Matrices imported from file.
    """
    # Read in length of the list
    def read_length() -> int:
        with open(os.path.join(directory, filename, "length.dat"), "r") as length_file:
            return int(length_file.readline())
    length = on_rank_zero(comm, read_length)

    # Read in the list
    mats = list()
    for index in range(length):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "r", comm)
        mat = allocate()
        mat.load(viewer)
        mats.append(mat)
        viewer.destroy()
    return mats


def import_vector(
    allocate: typing.Callable, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> petsc4py.PETSc.Vec:
    """
    Import a petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    allocate : typing.Callable
        A callable to allocate the storage.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the vector viewer.
    directory : str
        Directory where to import the vector from.
    filename : str
        Name of the file where to import the vector from.

    Returns
    -------
    petsc4py.PETSc.Vec
        Vector imported from file.
    """
    viewer = petsc4py.PETSc.Viewer().createBinary(os.path.join(directory, filename + ".dat"), "r", comm)
    vec = allocate()
    vec.load(viewer)
    viewer.destroy()
    return vec


def import_vectors(
    allocate: typing.Callable, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Vec]:
    """
    Import a list of petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    allocate : typing.Callable
        A callable to allocate the storage.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the vector viewer.
    directory : str
        Directory where to import the vector from.
    filename : str
        Name of the file where to import the vector from.

    Returns
    -------
    typing.List[petsc4py.PETSc.Vec]
        Vectors imported from file.
    """
    # Read in length of the list
    def read_length() -> int:
        with open(os.path.join(directory, filename, "length.dat"), "r") as length_file:
            return int(length_file.readline())
    length = on_rank_zero(comm, read_length)

    # Read in the list
    vecs = list()
    for index in range(length):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "r", comm)
        vec = allocate()
        vec.load(viewer)
        vecs.append(vec)
        viewer.destroy()
    return vecs
