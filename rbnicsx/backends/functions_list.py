# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to wrap a list of dolfinx Functions."""

import pathlib
import sys
import typing

import dolfinx.fem
import numpy as np
import petsc4py.PETSc

from rbnicsx._backends.functions_list import FunctionsList as FunctionsListBase
from rbnicsx.backends.export import export_functions
from rbnicsx.backends.import_ import import_functions

if sys.version_info >= (3, 11):  # pragma: no cover
    import typing as typing_extensions
else:  # pragma: no cover
    import typing_extensions


@typing.final
class FunctionsList(FunctionsListBase[dolfinx.fem.Function]):
    """
    A class wrapping a list of dolfinx Functions.

    Parameters
    ----------
    function_space
        Common finite element space of any Function that will be added to this list.

    Attributes
    ----------
    _function_space
        Finite element space provided as input.
    """

    def __init__(self: typing_extensions.Self, function_space: dolfinx.fem.FunctionSpace) -> None:
        self._function_space: dolfinx.fem.FunctionSpace = function_space
        super().__init__(function_space.mesh.comm)

    @property
    def function_space(self: typing_extensions.Self) -> dolfinx.fem.FunctionSpace:
        """Return the common finite element space of any Function that will be added to this list."""
        return self._function_space

    def duplicate(self: typing_extensions.Self) -> typing_extensions.Self:
        """
        Duplicate this object to a new empty FunctionsList.

        Returns
        -------
        :
            A new FunctionsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        return FunctionsList(self._function_space)

    def _save(self: typing_extensions.Self, directory: pathlib.Path, filename: str) -> None:
        """
        Save this list to file querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to export the list.
        filename
            Name of the file where to export the list.
        """
        export_functions(self._list, np.arange(len(self._list), dtype=float), directory, filename)

    def _load(self: typing_extensions.Self, directory: pathlib.Path, filename: str) -> None:
        """
        Load a list from file into this object querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to import the list from.
        filename
            Name of the file where to import the list from.
        """
        self._list = import_functions(self._function_space, directory, filename)

    def _linearly_combine(self: typing_extensions.Self, other: petsc4py.PETSc.Vec) -> dolfinx.fem.Function:  # type: ignore[no-any-unimported]
        """
        Linearly combine functions in the list using Function's API.

        Parameters
        ----------
        other
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        :
            Function object storing the result of the linear combination.
        """
        if len(self._list) > 0:
            output = self._list[0].copy()
            with output.vector.localForm() as output_local:
                output_local.set(0.0)
            for i in range(other.size):
                output.vector.axpy(other[i], self._list[i].vector)
            output.vector.ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
            return output
        else:
            return dolfinx.fem.Function(self._function_space)
