# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to perform a step of the Gram-Schmidt process."""


import dolfinx.mesh
import numpy as np
import petsc4py.PETSc
import ufl4rom.utils


class SymbolicParameters(ufl4rom.utils.DolfinxNamedConstant):
    """
    A class to store parameters for use inside UFL expressions.

    Parameters
    ----------
    mesh
        Domain of integration of forms which will use the symbolic parameters.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, shape: tuple[int, ...]) -> None:
        super().__init__("mu", np.zeros(shape, dtype=petsc4py.PETSc.ScalarType), mesh)  # type: ignore[attr-defined]
