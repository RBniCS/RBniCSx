# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to wrap a list of Functions."""

from __future__ import annotations

import os
import typing

import dolfinx.fem
import petsc4py

from minirox.backends.export import export_function
from minirox.backends.import_ import import_function


class FunctionsList(object):
    """
    A class wrapping a list of Functions.

    Parameters
    ----------
    space : dolfinx.fem.FunctionSpace
        Common finite element space of any Function that will be added to this list.

    Attributes
    ----------
    _space : dolfinx.fem.FunctionSpace
        Finite element space provided as input.
    _list : List[dolfinx.fem.FunctionSpace]
        Internal storage.
    """

    def __init__(self, space: dolfinx.fem.FunctionSpace) -> None:
        self._space = space
        self._list = list()

    def enrich(self, function: dolfinx.fem.Function) -> None:
        """
        Append a dolfinx.fem.Function to the list.

        Parameters
        ----------
        function : dolfinx.fem.Function
            Function to be appended.
        """
        self._list.append(function)

    def clear(self) -> None:
        """Clear the storage."""
        self._list = list()

    def save(self, directory: str, filename: str) -> None:
        """
        Save this list to file.

        Parameters
        ----------
        directory : str
            Directory where to export the list.
        filename : str
            Name of the file where to export the list.
        """
        # Save length
        if self._space.comm.rank == 0:
            with open(os.path.join(directory, filename + ".length"), "w") as length_file:
                length_file.write(str(len(self._list)))
        # Save functions
        for (index, function) in enumerate(self._list):
            export_function(function, directory, filename + "_" + str(index))

    def load(self, directory: str, filename: str) -> None:
        """
        Load a list from file into this object.

        Parameters
        ----------
        directory : str
            Directory where to import the list from.
        filename : str
            Name of the file where to import the list from.
        """
        assert len(self._list) == 0
        # Load length
        if self._space.comm.rank == 0:
            with open(os.path.join(str(directory), filename + ".length"), "r") as length_file:
                length = int(length_file.readline())
        else:
            length = 0
        length = self._space.comm.bcast(length, root=0)
        # Load functions
        for index in range(length):
            function = dolfinx.fem.Function(self._space)
            import_function(function, directory, filename + "_" + str(index))
            self.enrich(function)

    def __mul__(self, other: petsc4py.PETSc.Vec) -> dolfinx.fem.Function:
        """
        Linearly combine functions in the list.

        Parameters
        ----------
        other : petsc4py.PETSc.Vec
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        dolfinx.fem.Function
            Function object storing the result of the linear combination.
        """
        if isinstance(other, petsc4py.PETSc.Vec):
            assert other.getType() == petsc4py.PETSc.Vec.Type.SEQ
            pass  # TODO functions list mul online vector
        else:
            return NotImplemented

    def __len__(self) -> int:
        """Return the number of functions currently stored in the list."""
        return len(self._list)

    def __getitem__(self, key: typing.Union[int, slice]) -> typing.Union[dolfinx.fem.Function, FunctionsList]:
        """
        Extract a single function from the list, or slice the list before its end.

        Parameters
        ----------
        key : int or slice
            Index (if int) or indices (if slice) to be extracted.

        Returns
        -------
        dolfinx.fem.Function or FunctionsList
            Function at position `key` if `key` is an integer, otherwise FunctionsList obtained by
            storing every element at the indices in the slice `key`.
        """
        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, slice):
            assert key.start is None
            assert key.step is None
            assert key.stop is not None
            output = FunctionsList(self._space)
            output._list = self._list[key]
            return output
        else:
            raise NotImplementedError()

    def __setitem__(self, key: int, item: dolfinx.fem.Function) -> None:
        """
        Update the content of the list with the provided function.

        Parameters
        ----------
        key : int
            Index to be updated.
        function : dolfinx.fem.Function
            Function to be stored.
        """
        self._list[key] = item

    def __iter__(self) -> typing.Iterator[dolfinx.fem.Function]:
        """Return an iterator over the list."""
        return self._list.__iter__()
