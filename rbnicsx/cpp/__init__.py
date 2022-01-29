# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""RBniCSx cpp module."""

import os

import mpi4py
import multiphenicsx.cpp

cpp_library = multiphenicsx.cpp.compile_package(
    mpi4py.MPI.COMM_WORLD,
    "rbnicsx",
    os.path.dirname(os.path.abspath(__file__)),
    "backends/frobenius_inner_product.cpp"
)

del mpi4py
del multiphenicsx.cpp

del os
