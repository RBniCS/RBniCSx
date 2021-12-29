# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to export functions, matrices and vectors."""

import os
import typing

import dolfinx.fem
import dolfinx.io
import numpy.typing as npt
import petsc4py


def export_function(function: dolfinx.fem.Function, directory: str, filename: str) -> None:
    """
    Export a dolfinx.fem.Function to file.

    Parameters
    ----------
    function : dolfinx.fem.Function
        Function to be exported.
    directory : str
        Directory where to export the function.
    filename : str
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
    functions: typing.List[dolfinx.fem.Function], indices: npt.NDArray[float], directory: str, filename: str
) -> None:
    """
    Export a list of dolfinx.fem.Function to file.

    Parameters
    ----------
    functions : typing.List[dolfinx.fem.Function]
        Functions to be exported.
    indices : numpy.typing.NDArray[float]
        Indices associated to each entry in the list (e.g. time step number or time)
    directory : str
        Directory where to export the function.
    filename : str
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


def export_matrix(mat: petsc4py.PETSc.Mat, directory: str, filename: str) -> None:
    """
    Export a petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mat : petsc4py.PETSc.Mat
        Matrix to be exported.
    directory : str
        Directory where to export the matrix.
    filename : str
        Name of the file where to export the matrix.
    """
    os.makedirs(directory, exist_ok=True)
    viewer = petsc4py.PETSc.Viewer().createBinary(os.path.join(directory, filename + ".dat"), "w", mat.comm)
    viewer.view(mat)
    viewer.destroy()


def export_matrices(mats: typing.List[petsc4py.PETSc.Mat], directory: str, filename: str) -> None:
    """
    Export a list of petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mat : typing.List[petsc4py.PETSc.Mat]
        Matrices to be exported.
    directory : str
        Directory where to export the matrix.
    filename : str
        Name of the file where to export the matrix.
    """
    os.makedirs(os.path.join(directory, filename), exist_ok=True)
    # Write out length of the list
    comm = mats[0].comm
    if comm.rank == 0:
        with open(os.path.join(directory, filename, "length.dat"), "w") as length_file:
            length_file.write(str(len(mats)))
    # Write out the list
    for (index, mat) in enumerate(mats):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "w", comm)
        viewer.view(mat)
        viewer.destroy()


def export_vector(vec: petsc4py.PETSc.Vec, directory: str, filename: str) -> None:
    """
    Export a petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vec : petsc4py.PETSc.Vec
        Vector to be exported.
    directory : str
        Directory where to export the vector.
    filename : str
        Name of the file where to export the vector.
    """
    os.makedirs(directory, exist_ok=True)
    viewer = petsc4py.PETSc.Viewer().createBinary(os.path.join(directory, filename + ".dat"), "w", vec.comm)
    viewer.view(vec)
    viewer.destroy()


def export_vectors(vecs: typing.List[petsc4py.PETSc.Vec], directory: str, filename: str) -> None:
    """
    Export a list of petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vecs : typing.List[petsc4py.PETSc.Vec]
        Vectors to be exported.
    directory : str
        Directory where to export the vector.
    filename : str
        Name of the file where to export the vector.
    """
    os.makedirs(os.path.join(directory, filename), exist_ok=True)
    # Write out length of the list
    comm = vecs[0].comm
    if comm.rank == 0:
        with open(os.path.join(directory, filename, "length.dat"), "w") as length_file:
            length_file.write(str(len(vecs)))
    # Write out the list
    for (index, vec) in enumerate(vecs):
        viewer = petsc4py.PETSc.Viewer().createBinary(
            os.path.join(directory, filename, str(index) + ".dat"), "w", comm)
        viewer.view(vec)
        viewer.destroy()
