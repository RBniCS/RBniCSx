# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox cpp module."""

import os

import mpi4py

from minirox.cpp.compile_code import compile_code
from minirox.cpp.compile_package import compile_package

cpp_library = compile_package(
    mpi4py.MPI.COMM_WORLD,
    "minirox",
    os.path.dirname(os.path.abspath(__file__)),
    "backends/frobenius_inner_product.cpp"
)

__all__ = [
    "compile_code",
    "compile_package",
    "cpp_library"
]
