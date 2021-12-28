# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to export functions, matrices and vectors."""

import dolfinx.fem
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
    pass  # TODO


def export_matrix(mat: petsc4py.PETSc.Mat, directory: str, filename: str) -> None:
    """
    Export a petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    function : petsc4py.PETSc.Mat
        Matrix to be exported.
    directory : str
        Directory where to export the matrix.
    filename : str
        Name of the file where to export the matrix.
    """
    pass  # TODO


def export_vector(vec: petsc4py.PETSc.Vec, directory: str, filename: str) -> None:
    """
    Export a petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    function : petsc4py.PETSc.Vec
        Vector to be exported.
    directory : str
        Directory where to export the vector.
    filename : str
        Name of the file where to export the vector.
    """
    pass  # TODO
