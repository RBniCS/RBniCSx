# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.io.on_rank_zero module."""

import mpi4py.MPI
import petsc4py.PETSc
import pytest

import rbnicsx.io


def test_on_rank_zero_no_return() -> None:
    """Check that on_rank_zero runs correctly a function that does not return anything."""
    def no_return_function() -> None:
        return None

    assert rbnicsx.io.on_rank_zero(mpi4py.MPI.COMM_WORLD, no_return_function) is None


def test_on_rank_zero_int_return() -> None:
    """Check that on_rank_zero runs correctly a function that does return something."""
    comm = mpi4py.MPI.COMM_WORLD

    def int_return_function() -> int:
        return comm.rank

    assert rbnicsx.io.on_rank_zero(comm, int_return_function) == 0


def test_on_rank_zero_raise() -> None:
    """Check that on_rank_zero runs correctly a function that raises an error."""
    comm = mpi4py.MPI.COMM_WORLD

    def raise_function() -> None:
        raise RuntimeError

    with pytest.raises(RuntimeError):
        rbnicsx.io.on_rank_zero(comm, raise_function)


def test_on_rank_zero_petsc4py_comm() -> None:
    """Check that on_rank_zero runs correctly when using a petsc4py communicator, rather than a mpi4py one."""
    comm = petsc4py.PETSc.COMM_WORLD  # type: ignore[attr-defined]

    def int_return_function() -> int:
        return comm.rank  # type: ignore[no-any-return]

    assert rbnicsx.io.on_rank_zero(comm, int_return_function) == 0
