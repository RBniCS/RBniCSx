# Copyright (C) 2021-2026 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to perform a step of the Gram-Schmidt process."""


import dolfinx.fem
import dolfinx.mesh
import numpy as np
import petsc4py.PETSc


class SymbolicParameters(dolfinx.fem.Constant):
    """
    A class to store parameters for use inside UFL expressions.

    Parameters
    ----------
    mesh
        Domain of integration of forms which will use the symbolic parameters.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, shape: tuple[int, ...]) -> None:
        super().__init__(mesh, np.zeros(shape, dtype=petsc4py.PETSc.ScalarType))  # type: ignore[arg-type, attr-defined, unused-ignore]
