# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox main module."""

# Simplify import of mpi4py.MPI, petsc4py.PETSc and slepc4py.SLEPc in internal modules
# by importing them here once and for all. Internal modules will now only need to import
# the main packages mpi4py, petsc4py and slepc4py.
import mpi4py
import mpi4py.MPI  # noqa: F401
import petsc4py
import petsc4py.PETSc  # noqa: F401
import slepc4py
import slepc4py.SLEPc  # noqa: F401
