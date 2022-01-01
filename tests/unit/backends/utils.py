# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox.backends.export and minirox.backends.import_ modules."""

import numpy.typing as npt
import petsc4py
import scipy.sparse


def to_dense_matrix(mat: petsc4py.PETSc.Mat) -> npt.NDArray:
    """Convert the local part of a sparse PETSc Mat into a dense numpy ndarray."""
    ai, aj, av = mat.getValuesCSR()
    return scipy.sparse.csr_matrix((av, aj, ai), shape=(mat.getLocalSize()[0], mat.getSize()[1])).toarray()
