# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compute the order of magnitude of a number."""

import typing

import numpy as np
import numpy.typing as npt
import petsc4py.PETSc


@typing.overload
def order_of_magnitude(numbers: float | np.float32 | np.float64) -> np.int32:  # pragma: no cover
    ...


@typing.overload
def order_of_magnitude(
    numbers: npt.NDArray[np.float32 | np.float64 | petsc4py.PETSc.RealType]  # type: ignore[name-defined]
            | list[np.float32 | np.float64 | petsc4py.PETSc.RealType]
) -> npt.NDArray[np.int32]:  # pragma: no cover
    ...


def order_of_magnitude(
    numbers: float | np.float32 | np.float64  # type: ignore[name-defined]
             | npt.NDArray[np.float32 | np.float64 | petsc4py.PETSc.RealType]
             | list[np.float32 | np.float64 | petsc4py.PETSc.RealType]
) -> np.int32 | npt.NDArray[np.int32]:
    """Compute the order of magnitude of a number."""
    output: np.int32 | npt.NDArray[np.int32] = np.floor(np.log10(numbers)).astype(np.int32)
    return output
