# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Execute a function on rank zero and broadcast the result."""

import sys
import typing

import mpi4py.MPI
import petsc4py.PETSc

CallableOutput = typing.TypeVar("CallableOutput")


def on_rank_zero(
    comm: typing.Union[mpi4py.MPI.Intracomm, petsc4py.PETSc.Comm],  # type: ignore[name-defined]
    callable_: typing.Callable[[], CallableOutput]
) -> CallableOutput:
    """Execute a function on rank zero and broadcast the result."""
    if isinstance(comm, petsc4py.PETSc.Comm):  # type: ignore[attr-defined]
        comm = comm.tompi4py()

    return_value: typing.Optional[CallableOutput] = None
    error_raised = False
    error_type: typing.Optional[type[BaseException]] = None
    error_instance_args: typing.Optional[tuple[typing.Any, ...]] = None
    if comm.rank == 0:
        try:
            return_value = callable_()
        except Exception:
            error_raised = True
            error_type, error_instance, _ = sys.exc_info()
            assert isinstance(error_instance, BaseException)
            error_instance_args = error_instance.args
    error_raised = comm.bcast(error_raised, root=0)
    if not error_raised:
        return_value_bcast: CallableOutput = comm.bcast(return_value, root=0)
        return return_value_bcast
    else:
        error_type = comm.bcast(error_type, root=0)
        assert error_type is not None
        assert issubclass(error_type, BaseException)
        error_instance_args = comm.bcast(error_instance_args, root=0)
        assert error_instance_args is not None
        assert isinstance(error_instance_args, tuple)
        raise error_type(*error_instance_args)
