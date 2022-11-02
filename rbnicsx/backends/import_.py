# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to import dolfinx functions, matrices and vectors."""

import os
import typing

import dolfinx.fem
import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx._backends.import_ import (
    import_matrices as import_matrices_super, import_matrix as import_matrix_super,
    import_vector as import_vector_super, import_vectors as import_vectors_super)
from rbnicsx.io import on_rank_zero


def import_function(function_space: dolfinx.fem.FunctionSpace, directory: str, filename: str) -> dolfinx.fem.Function:
    """
    Import a dolfinx.fem.Function from file.

    Parameters
    ----------
    function_space
        Finite element space on which the function to be imported lives.
    directory
        Directory where to import the function from.
    filename
        Name of the file where to import the function from.

    Returns
    -------
    :
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
    function_space
        Finite element space on which the function to be imported lives.
    directory
        Directory where to import the function from.
    filename
        Name of the file where to import the function from.

    Returns
    -------
    :
        Functions imported from file.
    """
    comm = function_space.mesh.comm

    # Read in length of the list
    def read_length() -> int:
        with open(os.path.join(directory, filename, "length.dat"), "r") as length_file:
            return int(length_file.readline())
    length = on_rank_zero(comm, read_length)

    # Read in the list
    function_placeholder = dolfinx.fem.Function(function_space)
    functions = list()
    for index in range(length):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "r", comm)
        function = function_placeholder.copy()
        function.vector.load(viewer)
        functions.append(function)
        viewer.destroy()
    return functions


def import_matrix(  # type: ignore[no-any-unimported]
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> petsc4py.PETSc.Mat:
    """
    Import a petsc4py.PETSc.Mat assembled by dolfinx from file.

    Parameters
    ----------
    form
        The form which is used to assemble the tensors.
    comm
        Communicator to be used while creating the matrix.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrix imported from file.
    """
    return import_matrix_super(lambda: dolfinx.fem.petsc.create_matrix(form), comm, directory, filename)


def import_matrices(  # type: ignore[no-any-unimported]
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Mat]:
    """
    Import a list of petsc4py.PETSc.Mat assembled by dolfinx from file.

    Parameters
    ----------
    form
        The form which is used to assemble the tensors.
    comm
        Communicator to be used while creating the matrix.
    directory
        Directory where to import the matrix from.
    filename
        Name of the file where to import the matrix from.

    Returns
    -------
    :
        Matrices imported from file.
    """
    return import_matrices_super(lambda: dolfinx.fem.petsc.create_matrix(form), comm, directory, filename)


def import_vector(  # type: ignore[no-any-unimported]
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> petsc4py.PETSc.Vec:
    """
    Import a petsc4py.PETSc.Vec assembled by dolfinx from file.

    Parameters
    ----------
    form
        The form which is used to assemble the tensors.
    comm
        Communicator to be used while creating the vector.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vector imported from file.
    """
    return import_vector_super(lambda: dolfinx.fem.petsc.create_vector(form), comm, directory, filename)


def import_vectors(  # type: ignore[no-any-unimported]
    form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm, directory: str, filename: str
) -> typing.List[petsc4py.PETSc.Vec]:
    """
    Import a list of petsc4py.PETSc.Vec assembled by dolfinx from file.

    Parameters
    ----------
    form
        The form which is used to assemble the tensors.
    comm
        Communicator to be used while creating the vector.
    directory
        Directory where to import the vector from.
    filename
        Name of the file where to import the vector from.

    Returns
    -------
    :
        Vectors imported from file.
    """
    return import_vectors_super(lambda: dolfinx.fem.petsc.create_vector(form), comm, directory, filename)
