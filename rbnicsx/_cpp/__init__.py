# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""RBniCSx cpp module."""

import os

import mpi4py.MPI

from rbnicsx._cpp.compile_code import compile_code
from rbnicsx._cpp.compile_package import compile_package

# Add source files
sources = [
    "_backends/frobenius_inner_product.cpp",
    "_backends/petsc_error.cpp",
    "_backends/petsc_casters.cpp"
]
try:
    import dolfinx
except ImportError:  # pragma: no cover
    pass
else:  # pragma: no cover
    sources += []  # TODO: any other file in backends which requires dolfinx

# Compile the C++ library
cpp_library = compile_package(
    mpi4py.MPI.COMM_WORLD,
    "rbnicsx",
    os.path.dirname(os.path.abspath(__file__)),
    *sources
)
