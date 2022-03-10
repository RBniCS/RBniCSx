# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.gram_schmidt module."""

import typing

import _pytest.fixtures
import numpy as np
import petsc4py.PETSc
import pytest

import rbnicsx.online


@pytest.fixture
def functions_plain_and_size() -> typing.Tuple[typing.List[petsc4py.PETSc.Vec], int, int]:
    """Generate a list of pairwise linearly independent vectors."""
    vectors = [rbnicsx.online.create_vector(3) for _ in range(4)]
    for i in range(3):
        vectors[0].setValue(i, 1)
        vectors[1].setValue(i, i + 1)
        vectors[2].setValue(i, 2)
        vectors[3].setValue(i, 3 - i)
    return vectors, 3, 3


@pytest.fixture
def functions_block_and_size() -> typing.Tuple[typing.List[petsc4py.PETSc.Vec], typing.List[int], int]:
    """Generate a list of pairwise linearly independent vectors (block version)."""
    vectors = [rbnicsx.online.create_vector_block([3, 4]) for _ in range(4)]
    for i in range(7):
        vectors[0].setValue(i, 1)
        vectors[1].setValue(i, i + 1)
        vectors[2].setValue(i, 2)
        vectors[3].setValue(i, 7 - i)
    return vectors, [3, 4], 7


@pytest.fixture(params=["functions_plain_and_size", "functions_block_and_size"])
def functions_and_size(request: _pytest.fixtures.SubRequest) -> typing.Tuple[
        typing.List[petsc4py.PETSc.Vec], typing.Union[int, typing.List[int]], int]:
    """Parameterize functions generation considering either non-block or block content."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def inner_product() -> typing.Callable:
    """Return a callable that computes the identity matrix."""
    def _(N: typing.Union[int, typing.List[int]]) -> petsc4py.PETSc.Mat:
        """Return the identity matrix."""
        if isinstance(N, int):
            identity = rbnicsx.online.create_matrix(N, N)
        else:
            identity = rbnicsx.online.create_matrix_block(N, N)
        for i in range(identity.size[0]):
            identity.setValue(i, i, 1)
        identity.assemble()
        return identity

    return _


def compute_inner_product(
    inner_product: petsc4py.PETSc.Mat, function_i: petsc4py.PETSc.Vec, function_j: petsc4py.PETSc.Vec
) -> petsc4py.PETSc.ScalarType:
    """Evaluate the inner product between two functions."""
    inner_product_action = rbnicsx.online.matrix_action(inner_product)
    return inner_product_action(function_i)(function_j)


def test_online_gram_schmidt(
    functions_and_size: typing.Tuple[typing.List[petsc4py.PETSc.Vec], typing.Union[int, typing.List[int]], int],
    inner_product: typing.Callable
) -> None:
    """Check rbnicsx.online.gram_schmidt."""
    functions, size, size_int = functions_and_size
    inner_product_matrix = inner_product(size)
    functions_list = rbnicsx.online.FunctionsList(size)
    assert len(functions_list) == 0

    rbnicsx.online.gram_schmidt(functions_list, functions[0], inner_product_matrix)
    assert len(functions_list) == 1
    assert np.isclose(compute_inner_product(inner_product_matrix, functions_list[0], functions_list[0]), 1)
    assert np.allclose(functions_list[0].array, 1 / np.sqrt(size_int))

    rbnicsx.online.gram_schmidt(functions_list, functions[1], inner_product_matrix)
    assert len(functions_list) == 2
    assert np.isclose(compute_inner_product(inner_product_matrix, functions_list[0], functions_list[0]), 1)
    assert np.isclose(compute_inner_product(inner_product_matrix, functions_list[1], functions_list[1]), 1)
    assert np.isclose(compute_inner_product(inner_product_matrix, functions_list[0], functions_list[1]), 0)
    assert np.allclose(functions_list[0].array, 1 / np.sqrt(size_int))
    expected1 = np.arange(1, size_int + 1) - (size_int + 1) / 2
    expected1 /= np.linalg.norm(expected1)
    assert np.allclose(functions_list[1].array, expected1)


def test_online_gram_schmidt_zero(inner_product: typing.Callable) -> None:
    """Check rbnicsx.online.gram_schmidt when adding a linearly dependent function (e.g., zero)."""
    functions_list = rbnicsx.online.FunctionsList(3)
    inner_product_matrix = inner_product(3)
    assert len(functions_list) == 0

    zero = rbnicsx.online.create_vector(3)
    rbnicsx.online.gram_schmidt(functions_list, zero, inner_product_matrix)
    assert len(functions_list) == 0


def test_online_gram_schmidt_block(
    functions_and_size: typing.Tuple[typing.List[petsc4py.PETSc.Vec], typing.Union[int, typing.List[int]]],
    inner_product: typing.Callable
) -> None:
    """Check rbnicsx.online.gram_schmidt_block."""
    functions, size, size_int = functions_and_size
    inner_product_matrix = inner_product(size)
    functions_lists = [rbnicsx.online.FunctionsList(size) for _ in range(2)]
    for functions_list in functions_lists:
        assert len(functions_list) == 0

    rbnicsx.online.gram_schmidt_block(
        functions_lists, [functions[0], functions[2]], [inner_product_matrix, 2 * inner_product_matrix])
    for (functions_list, factor) in zip(functions_lists, [1, 2]):
        assert len(functions_list) == 1
        assert np.isclose(compute_inner_product(
            factor * inner_product_matrix, functions_list[0], functions_list[0]), 1)
        assert np.allclose(functions_list[0].array, 1 / np.sqrt(factor * size_int))

    rbnicsx.online.gram_schmidt_block(
        functions_lists, [functions[1], functions[3]], [inner_product_matrix, 2 * inner_product_matrix])
    for (functions_list, factor, expected1_addend) in zip(
            functions_lists, [1, 2], [np.arange(1, size_int + 1), np.arange(size_int, 0, -1)]):
        assert len(functions_list) == 2
        assert np.isclose(compute_inner_product(
            factor * inner_product_matrix, functions_list[0], functions_list[0]), 1)
        assert np.allclose(functions_list[0].array, 1 / np.sqrt(factor * size_int))
        assert np.isclose(compute_inner_product(
            factor * inner_product_matrix, functions_list[1], functions_list[1]), 1)
        assert np.isclose(compute_inner_product(
            factor * inner_product_matrix, functions_list[0], functions_list[1]), 0)
        expected1 = expected1_addend - (size_int + 1) / 2
        expected1 /= np.sqrt(factor) * np.linalg.norm(expected1)
        assert np.allclose(functions_list[1].array, expected1)
