# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to import dolfinx functions, matrices and vectors."""

import os
import typing

import dolfinx.fem
import mpi4py
import petsc4py

from rbnicsx._backends.import_ import (
    import_matrices as import_matrices_super, import_matrix as import_matrix_super,
    import_vector as import_vector_super, import_vectors as import_vectors_super)
from rbnicsx.io import on_rank_zero


def import_function(function_space: dolfinx.fem.FunctionSpace, directory: str, filename: str) -> dolfinx.fem.Function:
    """
    Import a dolfinx.fem.Function from file.

    Parameters
    ----------
    function_space : dolfinx.fem.FunctionSpace
        Finite element space on which the function to be imported lives.
    directory : str
        Directory where to import the function from.
    filename : str
        Name of the file where to import the function from.

    Returns
    -------
    dolfinx.fem.Function
        Function imported from file.
    """
    comm = function_space.mesh.comm
    function = dolfinx.fem.Function(function_space)
    viewer = petsc4py.PETSc.Viewer().createBinary(os.path.join(directory, filename + ".dat"), "r", comm)
    function.vector.load(viewer)
    viewer.destroy()
    return function


def import_functions(
    function_space: dolfinx.fem.FunctionSpace, directory: str, filename: str
) -> typing.List[dolfinx.fem.Function]:
    """
    Import a list of dolfinx.fem.Function from file.

    Parameters
    ----------
    function_space : dolfinx.fem.FunctionSpace
        Finite element space on which the function to be imported lives.
    directory : str
        Directory where to import the function from.
    filename : str
        Name of the file where to import the function from.

    Returns
    -------
    typing.List[dolfinx.fem.Function]
        Functions imported from file.
    """
    comm = function_space.mesh.comm

    # Read in length of the list
    def read_length() -> int:
        with open(os.path.join(directory, filename, "length.dat"), "r") as length_file:
            return int(length_file.readline())
    length = on_rank_zero(comm, read_length)

    # Read in the list
    functions = list()
    for index in range(length):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "r", comm)
        function = dolfinx.fem.Function(function_space)
        function.vector.load(viewer)
        functions.append(function)
        viewer.destroy()
    return functions


def import_matrix(
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> petsc4py.PETSc.Mat:
    """
    Import a petsc4py.PETSc.Mat assembled by dolfinx from file.

    Parameters
    ----------
    form : dolfinx.fem.FormMetaClass
        The form which is used to assemble the tensors.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the matrix.
    directory : str
        Directory where to import the matrix from.
    filename : str
        Name of the file where to import the matrix from.

    Returns
    -------
    petsc4py.PETSc.Mat
        Matrix imported from file.
    """
    return import_matrix_super(lambda: dolfinx.fem.petsc.create_matrix(form), comm, directory, filename)


def import_matrices(
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Mat]:
    """
    Import a list of petsc4py.PETSc.Mat assembled by dolfinx from file.

    Parameters
    ----------
    form : dolfinx.fem.FormMetaClass
        The form which is used to assemble the tensors.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the matrix.
    directory : str
        Directory where to import the matrix from.
    filename : str
        Name of the file where to import the matrix from.

    Returns
    -------
    typing.List[petsc4py.PETSc.Mat]
        Matrices imported from file.
    """
    return import_matrices_super(lambda: dolfinx.fem.petsc.create_matrix(form), comm, directory, filename)


def import_vector(
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> petsc4py.PETSc.Vec:
    """
    Import a petsc4py.PETSc.Vec assembled by dolfinx from file.

    Parameters
    ----------
    form : dolfinx.fem.FormMetaClass
        The form which is used to assemble the tensors.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the vector.
    directory : str
        Directory where to import the vector from.
    filename : str
        Name of the file where to import the vector from.

    Returns
    -------
    petsc4py.PETSc.Vec
        Vector imported from file.
    """
    return import_vector_super(lambda: dolfinx.fem.petsc.create_vector(form), comm, directory, filename)


def import_vectors(
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Vec]:
    """
    Import a list of petsc4py.PETSc.Vec assembled by dolfinx from file.

    Parameters
    ----------
    form : dolfinx.fem.FormMetaClass
        The form which is used to assemble the tensors.
    comm : mpi4py.MPI.Intracomm
        Communicator to be used while creating the vector.
    directory : str
        Directory where to import the vector from.
    filename : str
        Name of the file where to import the vector from.

    Returns
    -------
    typing.List[petsc4py.PETSc.Vec]
        Vectors imported from file.
    """
    return import_vectors_super(lambda: dolfinx.fem.petsc.create_vector(form), comm, directory, filename)
