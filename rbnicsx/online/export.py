# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to export matrices and vectors."""

import typing

import petsc4py

from rbnicsx._backends.export import (
    export_matrices as export_matrices_super, export_matrix as export_matrix_super,
    export_vector as export_vector_super, export_vectors as export_vectors_super)


def export_matrix(mat: petsc4py.PETSc.Mat, directory: str, filename: str) -> None:
    """
    Export a dense petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mat : petsc4py.PETSc.Mat
        Online matrix to be exported.
    directory : str
        Directory where to export the matrix.
    filename : str
        Name of the file where to export the matrix.
    """
    assert mat.getType() == petsc4py.PETSc.Mat.Type.SEQDENSE
    export_matrix_super(mat, directory, filename)


export_matrix_block = export_matrix


def export_matrices(mats: typing.List[petsc4py.PETSc.Mat], directory: str, filename: str) -> None:
    """
    Export a list of dense petsc4py.PETSc.Mat to file.

    Parameters
    ----------
    mats : typing.List[petsc4py.PETSc.Mat]
        Online matrices to be exported.
    directory : str
        Directory where to export the matrix.
    filename : str
        Name of the file where to export the matrix.
    """
    assert all([mat.getType() == petsc4py.PETSc.Mat.Type.SEQDENSE for mat in mats])
    export_matrices_super(mats, directory, filename)


export_matrices_block = export_matrices


def export_vector(vec: petsc4py.PETSc.Vec, directory: str, filename: str) -> None:
    """
    Export a sequential petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vec : petsc4py.PETSc.Vec
        Online vector to be exported.
    directory : str
        Directory where to export the vector.
    filename : str
        Name of the file where to export the vector.
    """
    assert vec.getType() == petsc4py.PETSc.Vec.Type.SEQ
    export_vector_super(vec, directory, filename)


export_vector_block = export_vector


def export_vectors(vecs: typing.List[petsc4py.PETSc.Vec], directory: str, filename: str) -> None:
    """
    Export a list of sequential petsc4py.PETSc.Vec to file.

    Parameters
    ----------
    vecs : typing.List[petsc4py.PETSc.Vec]
        Online vectors to be exported.
    directory : str
        Directory where to export the vector.
    filename : str
        Name of the file where to export the vector.
    """
    assert all([vec.getType() == petsc4py.PETSc.Vec.Type.SEQ for vec in vecs])
    export_vectors_super(vecs, directory, filename)


export_vectors_block = export_vectors
