# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.io.timer module."""

import time

import mpi4py.MPI
import petsc4py.PETSc
import pytest

import rbnicsx.io


def _expected_measured_time(sleep_time: float, comm: mpi4py.MPI.Intracomm, op: mpi4py.MPI.Op) -> float:
    """
    Compute the expected time measured by the timer.

    Measured time should be the same as sleep time if op == mpi4py.MPI.MAX,
    while should be equal to sleep time * comm.size if if op == mpi4py.MPI.SUM.
    """
    if op == mpi4py.MPI.MAX:
        return sleep_time
    elif op == mpi4py.MPI.SUM:
        return sleep_time * comm.size
    else:
        raise RuntimeError("Invalid MPI operation.")


@pytest.mark.parametrize("comm", [
    mpi4py.MPI.COMM_WORLD, petsc4py.PETSc.COMM_WORLD, mpi4py.MPI.COMM_SELF  # type: ignore[attr-defined]
])
@pytest.mark.parametrize("op", [mpi4py.MPI.MAX, mpi4py.MPI.SUM])
def test_timer(comm: mpi4py.MPI.Intracomm, op: mpi4py.MPI.Op) -> None:
    """Test timer with different communicators and operations."""
    timings: list[float] = [0.0]
    with rbnicsx.io.Timer(comm, op, rbnicsx.io.store_elapsed_time(timings, 0)):
        time.sleep(0.05)
    assert timings[0] >= _expected_measured_time(0.05, comm, op)
