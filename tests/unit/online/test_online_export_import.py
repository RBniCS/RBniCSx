# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.export and rbnicsx.online.import_ modules."""

import typing

import numpy as np

import rbnicsx.online


def test_online_export_import_vector(tempdir: str) -> None:
    """Check I/O for an online petsc4py.PETSc.Vec."""
    vector = rbnicsx.online.create_vector(2)
    for i in range(2):
        vector.setValue(i, i + 1)
    vector.view()
    rbnicsx.online.export_vector(vector, tempdir, "vector")

    vector2 = rbnicsx.online.import_vector(2, tempdir, "vector")
    assert np.allclose(vector2.array, vector.array)


def test_online_export_import_vector_block(tempdir: str) -> None:
    """Check I/O for an online petsc4py.PETSc.Vec (block version)."""
    vector = rbnicsx.online.create_vector_block([2, 3])
    for i in range(5):
        vector.setValue(i, i + 1)
    vector.view()
    rbnicsx.online.export_vector(vector, tempdir, "vector")

    vector2 = rbnicsx.online.import_vector_block([2, 3], tempdir, "vector")
    assert np.allclose(vector2.array, vector.array)


def test_online_export_import_vectors(tempdir: str) -> None:
    """Check I/O for a list of online petsc4py.PETSc.Vec."""
    vectors = [rbnicsx.online.create_vector(2) for _ in range(3)]
    for (v, vector) in enumerate(vectors):
        for i in range(2):
            vector.setValue(i, v * 2 + i + 1)
        vector.view()
    rbnicsx.online.export_vectors(vectors, tempdir, "vectors")

    vectors2 = rbnicsx.online.import_vectors(2, tempdir, "vectors")
    assert len(vectors2) == 3
    for (vector, vector2) in zip(vectors, vectors2):
        assert np.allclose(vector2.array, vector.array)


def test_online_export_import_vectors_block(tempdir: str) -> None:
    """Check I/O for a list of online petsc4py.PETSc.Vec (block version)."""
    vectors = [rbnicsx.online.create_vector_block([2, 3]) for _ in range(3)]
    for (v, vector) in enumerate(vectors):
        for i in range(5):
            vector.setValue(i, v * 5 + i + 1)
    rbnicsx.online.export_vectors(vectors, tempdir, "vectors")

    vectors2 = rbnicsx.online.import_vectors_block([2, 3], tempdir, "vectors")
    assert len(vectors2) == 3
    for (vector, vector2) in zip(vectors, vectors2):
        assert np.allclose(vector2.array, vector.array)


def test_online_export_import_matrix(tempdir: str, to_dense_matrix: typing.Callable) -> None:
    """Check I/O for an online petsc4py.PETSc.Mat."""
    matrix = rbnicsx.online.create_matrix(2, 3)
    for i in range(2):
        for j in range(3):
            matrix.setValue(i, j, i * 3 + j + 1)
    matrix.assemble()
    matrix.view()
    rbnicsx.online.export_matrix(matrix, tempdir, "matrix")

    matrix2 = rbnicsx.online.import_matrix(2, 3, tempdir, "matrix")
    assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))


def test_online_export_import_matrix_block(tempdir: str, to_dense_matrix: typing.Callable) -> None:
    """Check I/O for an online petsc4py.PETSc.Mat (block version)."""
    matrix = rbnicsx.online.create_matrix_block([2, 3], [4, 5])
    for i in range(5):
        for j in range(9):
            matrix.setValue(i, j, i * 9 + j + 1)
    matrix.assemble()
    matrix.view()
    rbnicsx.online.export_matrix(matrix, tempdir, "matrix")

    matrix2 = rbnicsx.online.import_matrix_block([2, 3], [4, 5], tempdir, "matrix")
    assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))


def test_online_export_import_matrices(tempdir: str, to_dense_matrix: typing.Callable) -> None:
    """Check I/O for a list of online petsc4py.PETSc.Mat."""
    matrices = [rbnicsx.online.create_matrix(2, 3) for _ in range(4)]
    for (m, matrix) in enumerate(matrices):
        for i in range(2):
            for j in range(3):
                matrix.setValue(i, j, m * 6 + i * 3 + j + 1)
        matrix.assemble()
        matrix.view()
    rbnicsx.online.export_matrices(matrices, tempdir, "matrices")

    matrices2 = rbnicsx.online.import_matrices(2, 3, tempdir, "matrices")
    for (matrix, matrix2) in zip(matrices, matrices2):
        assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))


def test_online_export_import_matrices_block(tempdir: str, to_dense_matrix: typing.Callable) -> None:
    """Check I/O for a list of online petsc4py.PETSc.Mat (block version)."""
    matrices = [rbnicsx.online.create_matrix_block([2, 3], [4, 5]) for _ in range(6)]
    for (m, matrix) in enumerate(matrices):
        for i in range(5):
            for j in range(9):
                matrix.setValue(i, j, m * 45 + i * 9 + j + 1)
        matrix.assemble()
        matrix.view()
    rbnicsx.online.export_matrices(matrices, tempdir, "matrices")

    matrices2 = rbnicsx.online.import_matrices_block([2, 3], [4, 5], tempdir, "matrices")
    for (matrix, matrix2) in zip(matrices, matrices2):
        assert np.allclose(to_dense_matrix(matrix2), to_dense_matrix(matrix))
