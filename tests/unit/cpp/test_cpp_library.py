# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox C++ library."""

import mpi4py
import petsc4py

import minirox.cpp


def test_cpp_backends_frobenius_inner_product() -> None:
    """Test C++ function minirox::backends::frobenius_inner_product."""
    A = petsc4py.PETSc.Mat().createDense((2, 3), comm=mpi4py.MPI.COMM_WORLD)
    A.setUp()
    B = petsc4py.PETSc.Mat().createDense((2, 3), comm=mpi4py.MPI.COMM_WORLD)
    B.setUp()
    for i in range(2):
        for j in range(3):
            A.setValue(i, j, i * 2 + j + 1)
            B.setValue(i, j, j * 3 + i + 1)
    A.assemble()
    B.assemble()
    A.view()
    B.view()

    assert (
        minirox.cpp.cpp_library.backends.frobenius_inner_product(A, B)
        == sum([(i * 2 + j + 1) * (j * 3 + i + 1) for i in range(2) for j in range(3)])
    )
