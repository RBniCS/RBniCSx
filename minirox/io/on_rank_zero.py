# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Execute a function on rank zero and broadcast the result."""

import sys
import typing

import mpi4py
import petsc4py


def on_rank_zero(
    comm: typing.Union[mpi4py.MPI.Intracomm, petsc4py.PETSc.Comm], lambda_function: typing.Callable
) -> object:
    """Execute a function on rank zero and broadcast the result."""
    if isinstance(comm, petsc4py.PETSc.Comm):
        comm = comm.tompi4py()

    return_value = None
    error_raised = False
    error_type = None
    error_instance_args = None
    if comm.rank == 0:
        try:
            return_value = lambda_function()
        except Exception:
            error_raised = True
            error_type, error_instance, _ = sys.exc_info()
            error_instance_args = error_instance.args
    error_raised = comm.bcast(error_raised, root=0)
    if not error_raised:
        return_value = comm.bcast(return_value, root=0)
        return return_value
    else:
        error_type = comm.bcast(error_type, root=0)
        error_instance_args = comm.bcast(error_instance_args, root=0)
        raise error_type(*error_instance_args)
