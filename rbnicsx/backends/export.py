# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to export dolfinx functions, matrices and vectors."""

import os
import typing

import dolfinx.fem
import dolfinx.io
import numpy as np
import numpy.typing
import petsc4py.PETSc

from rbnicsx._backends.export import (
    export_matrices as export_matrices_super, export_matrix as export_matrix_super,
    export_vector as export_vector_super, export_vectors as export_vectors_super)


def export_function(function: dolfinx.fem.Function, directory: str, filename: str) -> None:
    """
    Export a dolfinx.fem.Function to file.

    Parameters
    ----------
    function
        Function to be exported.
    directory
        Directory where to export the function.
    filename
        Name of the file where to export the function.
    """
    os.makedirs(directory, exist_ok=True)
    # Export to XDMF file for visualization
    mesh = function.function_space.mesh
    with dolfinx.io.XDMFFile(mesh.comm, os.path.join(directory, filename + ".xdmf"), "w") as xdmf_file:
        xdmf_file.write_mesh(mesh)
        xdmf_file.write_function(function)
    # Export the underlying vector for restart purposes
    export_vector(function.vector, directory, filename)


def export_functions(
    functions: typing.List[dolfinx.fem.Function], indices: np.typing.NDArray[np.float32], directory: str, filename: str
) -> None:
    """
    Export a list of dolfinx.fem.Function to file.

    Parameters
    ----------
    functions
        Functions to be exported.
    indices
        Indices associated to each entry in the list (e.g. time step number or time)
    directory
        Directory where to export the function.
    filename
        Name of the file where to export the function.
    """
    os.makedirs(directory, exist_ok=True)
    # Export to XDMF file for visualization
    mesh = functions[0].function_space.mesh
    with dolfinx.io.XDMFFile(mesh.comm, os.path.join(directory, filename + ".xdmf"), "w") as xdmf_file:
        xdmf_file.write_mesh(mesh)
        for (function, index) in zip(functions, indices):
            xdmf_file.write_function(function, index)
    # Export the underlying vectors for restart purposes
    export_vectors([function.vector for function in functions], directory, filename)


def export_matrix(  # type: ignore[no-any-unimported]
    mat: petsc4py.PETSc.Mat, directory: str, filename: str
) -> None:
    """
    Export a petsc4py.PETSc.Mat assembled by dolfinx to file.

    Parameters
    ----------
    mat
        Matrix to be exported.
    directory
        Directory where to export the matrix.
    filename
        Name of the file where to export the matrix.
    """
    export_matrix_super(mat, mat.comm, directory, filename)


def export_matrices(  # type: ignore[no-any-unimported]
    mats: typing.List[petsc4py.PETSc.Mat], directory: str, filename: str
) -> None:
    """
    Export a list of petsc4py.PETSc.Mat assembled by dolfinx to file.

    Parameters
    ----------
    mats
        Matrices to be exported.
    directory
        Directory where to export the matrix.
    filename
        Name of the file where to export the matrix.
    """
    export_matrices_super(mats, mats[0].comm, directory, filename)


def export_vector(  # type: ignore[no-any-unimported]
    vec: petsc4py.PETSc.Vec, directory: str, filename: str
) -> None:
    """
    Export a petsc4py.PETSc.Vec assembled by dolfinx to file.

    Parameters
    ----------
    vec
        Vector to be exported.
    directory
        Directory where to export the vector.
    filename
        Name of the file where to export the vector.
    """
    export_vector_super(vec, vec.comm, directory, filename)


def export_vectors(  # type: ignore[no-any-unimported]
    vecs: typing.List[petsc4py.PETSc.Vec], directory: str, filename: str
) -> None:
    """
    Export a list of petsc4py.PETSc.Vec assembled by dolfinx to file.

    Parameters
    ----------
    vecs
        Vectors to be exported.
    directory
        Directory where to export the vector.
    filename
        Name of the file where to export the vector.
    """
    export_vectors_super(vecs, vecs[0].comm, directory, filename)
