# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to export dolfinx functions, matrices and vectors."""

import pathlib

import adios4dolfinx
import dolfinx.fem
import dolfinx.io
import numpy as np
import numpy.typing
import petsc4py.PETSc

from rbnicsx._backends.export import (
    export_matrices as export_matrices_super, export_matrix as export_matrix_super,
    export_vector as export_vector_super, export_vectors as export_vectors_super)
from rbnicsx.io import on_rank_zero


def export_function(function: dolfinx.fem.Function, directory: pathlib.Path, filename: str) -> None:
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
    comm = function.function_space.mesh.comm
    visualization_directory = directory / (filename + ".bp")
    checkpointing_directory = directory / (filename + "_checkpoint.bp")
    visualization_directory.mkdir(parents=True, exist_ok=True)
    checkpointing_directory.mkdir(parents=True, exist_ok=True)

    # Export for visualization
    with dolfinx.io.VTXWriter(comm, visualization_directory, function, "bp4") as vtx_file:
        vtx_file.write(0)

    # Export for checkpointing
    adios4dolfinx.write_mesh(function.function_space.mesh, pathlib.Path(checkpointing_directory), "bp4")
    adios4dolfinx.write_function(function, pathlib.Path(checkpointing_directory), "bp4")


def export_functions(
    functions: list[dolfinx.fem.Function], indices: np.typing.NDArray[np.float32],
    directory: pathlib.Path, filename: str
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
    comm = functions[0].function_space.mesh.comm
    visualization_directory = directory / (filename + ".bp")
    checkpointing_directory = directory / (filename + "_checkpoint.bp")
    visualization_directory.mkdir(parents=True, exist_ok=True)
    checkpointing_directory.mkdir(parents=True, exist_ok=True)

    # Export for visualization
    output = functions[0].copy()
    with dolfinx.io.VTXWriter(comm, visualization_directory, output, "bp4") as vtx_file:
        for (function, index) in zip(functions, indices):
            output.x.array[:] = function.x.array
            output.x.scatter_forward()
            vtx_file.write(index)
    del output

    # Export for checkpointing: write out length of the list
    def write_length() -> None:
        with open(checkpointing_directory / "length.dat", "w") as length_file:
            length_file.write(str(len(functions)))
    on_rank_zero(comm, write_length)

    # Export for checkpointing: write out the list
    # Note that here index is an integer counter, rather than an entry of the input array indices.
    for (index, function) in enumerate(functions):
        adios4dolfinx.write_mesh(function.function_space.mesh, checkpointing_directory / str(index), "bp4")
        adios4dolfinx.write_function(function, checkpointing_directory / str(index), "bp4")


def export_matrix(  # type: ignore[no-any-unimported]
    mat: petsc4py.PETSc.Mat, directory: pathlib.Path, filename: str
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
    mats: list[petsc4py.PETSc.Mat], directory: pathlib.Path, filename: str
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
    vec: petsc4py.PETSc.Vec, directory: pathlib.Path, filename: str
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
    vecs: list[petsc4py.PETSc.Vec], directory: pathlib.Path, filename: str
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
