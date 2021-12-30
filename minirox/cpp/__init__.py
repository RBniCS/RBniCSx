# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox cpp module."""

from minirox.cpp.compile_code import compile_code
from minirox.cpp.compile_package import compile_package

__all__ = [
    "compile_code",
    "compile_package"
]
