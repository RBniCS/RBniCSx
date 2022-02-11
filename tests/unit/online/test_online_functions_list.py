# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.online.functions_list module."""

import _pytest.fixtures
import nbvalx.tempfile
import numpy as np
import pytest

import rbnicsx.online


@pytest.fixture
def functions_list_plain() -> rbnicsx.online.FunctionsList:
    """Generate a rbnicsx.online.FunctionsList with two petsc4py.PETSc.Vec entries."""
    vectors = [rbnicsx.online.create_vector(3) for _ in range(2)]
    for (v, vector) in enumerate(vectors):
        for i in range(3):
            vector.setValue(i, (v + 1) * (i + 1))
    functions_list = rbnicsx.online.FunctionsList(3)
    [functions_list.append(vector) for vector in vectors]
    functions_list.first_vector = vectors[0]
    return functions_list


@pytest.fixture
def functions_list_block() -> rbnicsx.online.FunctionsList:
    """Generate a rbnicsx.online.FunctionsList with two petsc4py.PETSc.Vec entries (block version)."""
    vectors = [rbnicsx.online.create_vector_block([3, 4]) for _ in range(2)]
    for (v, vector) in enumerate(vectors):
        for i in range(7):
            vector.setValue(i, (v + 1) * (i + 1))
    functions_list = rbnicsx.online.FunctionsList([3, 4])
    [functions_list.append(vector) for vector in vectors]
    functions_list.first_vector = vectors[0]
    return functions_list


@pytest.fixture(params=["functions_list_plain", "functions_list_block"])
def functions_list(request: _pytest.fixtures.SubRequest) -> rbnicsx.online.FunctionsList:
    """Parameterize rbnicsx.online.FunctionsList considering either non-block or block content."""
    return request.getfixturevalue(request.param)


def test_online_functions_list_shape_plain(functions_list_plain: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.shape in the case non-block content."""
    assert isinstance(functions_list_plain.shape, int)
    assert functions_list_plain.shape == 3


def test_online_functions_list_shape_block(functions_list_block: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.shape in the case of block content."""
    assert isinstance(functions_list_block.shape, list)
    assert all([isinstance(shape_, int) for shape_ in functions_list_block.shape])
    assert len(functions_list_block.shape) == 2
    assert functions_list_block.shape[0] == 3
    assert functions_list_block.shape[1] == 4


def test_online_functions_list_is_block_plain(functions_list_plain: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.is_block in the case non-block content."""
    assert functions_list_plain.is_block is False


def test_online_functions_list_is_block_block(functions_list_block: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.is_block in the case of block content."""
    assert functions_list_block.is_block is True


def test_online_functions_list_duplicate(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.duplicate."""
    functions_list2 = functions_list.duplicate()
    assert len(functions_list2) == 0


def test_online_functions_list_extend(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.extend."""
    functions_list2 = functions_list.duplicate()
    functions_list2.extend(functions_list)
    assert len(functions_list2) == 2
    for i in range(2):
        assert functions_list2[i] == functions_list[i]


def test_online_functions_list_len(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__len__."""
    assert len(functions_list) == 2


def test_online_functions_list_clear(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.clear."""
    functions_list.clear()
    assert len(functions_list) == 0


def test_online_functions_list_iter(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__iter__."""
    for (index, function) in enumerate(functions_list):
        assert np.allclose(function.array, (index + 1) * functions_list.first_vector.array)


def test_online_functions_list_getitem_int(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__getitem__ with integer input."""
    assert np.allclose(functions_list[0].array, functions_list.first_vector.array)
    assert np.allclose(functions_list[1].array, 2 * functions_list.first_vector.array)


def test_online_functions_list_getitem_slice(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__getitem__ with slice input."""
    functions_list2 = functions_list[0:2]
    assert len(functions_list2) == 2
    assert np.allclose(functions_list2[0].array, functions_list.first_vector.array)
    assert np.allclose(functions_list2[1].array, 2 * functions_list.first_vector.array)


def test_online_functions_list_getitem_wrong_type(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__getitem__ with unsupported input."""
    with pytest.raises(RuntimeError):
        functions_list[0, 0]


def test_online_functions_list_setitem(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__setitem__."""
    functions_list[0] = 3 * functions_list.first_vector

    assert np.allclose(functions_list[0].array, 3 * functions_list.first_vector.array)
    assert np.allclose(functions_list[1].array, 2 * functions_list.first_vector.array)


def test_online_functions_list_save_load(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check I/O for a rbnicsx.online.FunctionsList."""
    with nbvalx.tempfile.TemporaryDirectory(functions_list.comm) as tempdir:
        functions_list.save(tempdir, "functions_list")

        functions_list2 = functions_list.duplicate()
        functions_list2.load(tempdir, "functions_list")

        assert len(functions_list2) == 2
        for (function, function2) in zip(functions_list, functions_list2):
            assert np.allclose(function2.array, function.array)


def test_online_functions_list_mul(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__mul__."""
    online_vec = rbnicsx.online.create_vector(2)
    online_vec[0] = 3
    online_vec[1] = 5

    function = functions_list * online_vec
    assert np.allclose(function.array, 13 * functions_list.first_vector.array)


def test_online_functions_list_mul_empty() -> None:
    """Check rbnicsx.online.FunctionsList.__mul__ with empty list."""
    empty_functions_list = rbnicsx.online.FunctionsList(10)

    online_vec = rbnicsx.online.create_vector(0)

    should_be_zero_vector = empty_functions_list * online_vec
    assert np.allclose(should_be_zero_vector.array, 0)
    assert should_be_zero_vector.size == 10


def test_online_functions_list_mul_not_implemented(functions_list: rbnicsx.online.FunctionsList) -> None:
    """Check rbnicsx.online.FunctionsList.__mul__ with an incorrect type."""
    with pytest.raises(TypeError):
        functions_list * None
