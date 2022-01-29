# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Context manager to time execution of a code and store the result."""

from __future__ import annotations

import time
import types
import typing

import mpi4py
import petsc4py


class Timer(object):
    """
    A context manager to time execution of a code and store the result.

    Parameters
    ----------
    comm : typing.Union[mpi4py.MPI.Intracomm, petsc4py.PETSc.Comm]
        MPI communicator to be used to carry out MPI_Allreduce.
    op : mpi4py.MPI.Op
        MPI operation to be used while carrying out MPI_Allreduce.
    store : typing.Callable
        Callable implementing an action to store the elapsed time.

    Attributes
    ----------
    _comm : mpi4py.MPI.Intracomm
        MPI communicator provided as input.
    _op : mpi4py.MPI.Op
        MPI operation provided as input.
    _store : typing.Callable
        Callable provided as input.
    _start : float
        Time stamp, in fractional seconds, when the context was entered.
    """

    def __init__(
        self, comm: typing.Union[mpi4py.MPI.Intracomm, petsc4py.PETSc.Comm], op: mpi4py.MPI.Op,
        store: typing.Callable
    ) -> None:
        if isinstance(comm, petsc4py.PETSc.Comm):
            comm = comm.tompi4py()

        self._comm = comm
        self._op = op
        self._store = store
        self._start = None

    def __enter__(self) -> Timer:
        """Enter the context and start the timer."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Stop the timer, store the elapsed time and exit the context."""
        elapsed = time.perf_counter() - self._start
        self._start = None
        self._store(self._comm.allreduce(elapsed, op=self._op))


def store_elapsed_time(storage: typing.Iterable[float], index: int) -> typing.Callable:
    """
    Auxiliary function to be passed as third argument to rbnicsx.io.Timer.

    This function handles the frequent case in which the user wants to store
    the elapsed time in an iterable object such as a list or a numpy array.
    """
    def _store_elapsed_time(time: float) -> None:
        storage[index] = time
    return _store_elapsed_time
