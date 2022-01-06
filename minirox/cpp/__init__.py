# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox cpp module."""

import os

import mpi4py
import multiphenicsx.cpp

cpp_library = multiphenicsx.cpp.compile_package(
    mpi4py.MPI.COMM_WORLD,
    "minirox",
    os.path.dirname(os.path.abspath(__file__)),
    "backends/frobenius_inner_product.cpp"
)

del mpi4py
del multiphenicsx.cpp

del os
