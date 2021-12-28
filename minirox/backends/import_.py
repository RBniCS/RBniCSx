# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to import functions, matrices and vectors."""

import dolfinx.fem
import petsc4py


def import_function(function: dolfinx.fem.Function, directory: str, filename: str) -> None:
    """
    Import a dolfinx.fem.Function from file.

    Parameters
    ----------
    function : dolfinx.fem.Function
        Function to be filled with imported data.
    directory : str
        Directory where to import the function from.
    filename : str
        Name of the file where to import the function from.
    """
    pass  # TODO


def import_matrix(mat: petsc4py.PETSc.Mat, directory: str, filename: str) -> None:
    """
    Import a petsc4py.PETSc.Mat from file.

    Parameters
    ----------
    function : petsc4py.PETSc.Mat
        Matrix to be filled with imported data.
    directory : str
        Directory where to import the matrix from.
    filename : str
        Name of the file where to import the matrix from.
    """
    pass  # TODO


def import_vector(vec: petsc4py.PETSc.Vec, directory: str, filename: str) -> None:
    """
    Import a petsc4py.PETSc.Vec from file.

    Parameters
    ----------
    function : petsc4py.PETSc.Vec
        Vector to be filled with imported data.
    directory : str
        Directory where to import the vector from.
    filename : str
        Name of the file where to import the vector from.
    """
    pass  # TODO
