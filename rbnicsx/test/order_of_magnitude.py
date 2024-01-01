# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compute the order of magnitude of a number."""

import typing

import numpy as np
import numpy.typing
import petsc4py.PETSc


@typing.overload
def order_of_magnitude(numbers: typing.Union[float, np.float32, np.float64]) -> np.int32:  # pragma: no cover
    ...


@typing.overload
def order_of_magnitude(  # type: ignore[no-any-unimported]
    numbers: typing.Union[
        np.typing.NDArray[typing.Union[np.float32, np.float64, petsc4py.PETSc.RealType]],
        typing.List[typing.Union[np.float32, np.float64, petsc4py.PETSc.RealType]]]
) -> np.typing.NDArray[np.int32]:  # pragma: no cover
    ...


def order_of_magnitude(  # type: ignore[no-any-unimported]
    numbers: typing.Union[
        typing.Union[float, np.float32, np.float64],
        np.typing.NDArray[typing.Union[np.float32, np.float64, petsc4py.PETSc.RealType]],
        typing.List[typing.Union[np.float32, np.float64, petsc4py.PETSc.RealType]]]
) -> typing.Union[np.int32, np.typing.NDArray[np.int32]]:
    """Compute the order of magnitude of a number."""
    output: typing.Union[np.int32, np.typing.NDArray[np.int32]] = np.floor(np.log10(numbers)).astype(np.int32)
    return output
