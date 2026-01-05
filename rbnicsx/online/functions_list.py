# Copyright (C) 2021-2026 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to wrap a list of PETSc Vec which represent solutions to online systems."""

import pathlib
import sys
import typing

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx._backends.functions_list import FunctionsList as FunctionsListBase
from rbnicsx._backends.online_tensors import (
    create_online_vector as create_vector, create_online_vector_block as create_vector_block)
from rbnicsx.online.export import export_vectors, export_vectors_block
from rbnicsx.online.import_ import import_vectors, import_vectors_block

if sys.version_info >= (3, 11):  # pragma: no cover
    import typing as typing_extensions
else:  # pragma: no cover
    import typing_extensions


@typing.final
class FunctionsList(FunctionsListBase[petsc4py.PETSc.Vec]):  # type: ignore[name-defined]
    """
    A class wrapping a list of online PETSc Vec which represent solutions to online systems.

    Parameters
    ----------
    shape
        Shape of the vectors which will be added to the list.

    Attributes
    ----------
    _shape
        Shape provided as input.
    _is_block
        Whether the vector has a block structure or not.
    """

    def __init__(self: typing_extensions.Self, shape: int | list[int]) -> None:
        self._shape: int | list[int] = shape
        if isinstance(shape, list):
            is_block = True
        else:
            is_block = False
        self._is_block: bool = is_block
        # Initialize using COMM_WORLD even though online vectors are on COMM_SELF to identify rank 0 for I/O
        super().__init__(mpi4py.MPI.COMM_WORLD)

    @property
    def shape(self: typing_extensions.Self) -> int | list[int]:
        """Return the shape of the vectors in the list."""
        return self._shape

    @property
    def is_block(self: typing_extensions.Self) -> bool:
        """Return whether the vector has a block structure or not."""
        return self._is_block

    def duplicate(self: typing_extensions.Self) -> typing_extensions.Self:
        """
        Duplicate this object to a new empty FunctionsList.

        Returns
        -------
        :
            A new FunctionsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        return FunctionsList(self._shape)

    def _save(self: typing_extensions.Self, directory: pathlib.Path, filename: str) -> None:
        """
        Save this list to file querying the I/O functions in the online backend.

        Parameters
        ----------
        directory
            Directory where to export the list.
        filename
            Name of the file where to export the list.
        """
        if self._is_block:
            export_vectors_block(self._list, directory, filename)
        else:
            export_vectors(self._list, directory, filename)

    def _load(self: typing_extensions.Self, directory: pathlib.Path, filename: str) -> None:
        """
        Load a list from file into this object querying the I/O functions in the online backend.

        Parameters
        ----------
        directory
            Directory where to import the list from.
        filename
            Name of the file where to import the list from.
        """
        if self._is_block:
            assert isinstance(self._shape, list)
            self._list = import_vectors_block(self._shape, directory, filename)
        else:
            assert isinstance(self._shape, int)
            self._list = import_vectors(self._shape, directory, filename)

    def _linearly_combine(
        self: typing_extensions.Self, other: petsc4py.PETSc.Vec  # type: ignore[name-defined]
    ) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
        """
        Linearly combine functions in the list using petsc4py API.

        Parameters
        ----------
        other
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        :
            Vector storing the result of the linear combination.
        """
        if other.size > 0:
            output = self._list[0].copy()
            output.zeroEntries()
            for i in range(other.size):
                output.axpy(other[i], self._list[i])
            return output
        else:
            if self._is_block:
                assert isinstance(self._shape, list)
                return create_vector_block(self._shape)
            else:
                assert isinstance(self._shape, int)
                return create_vector(self._shape)
