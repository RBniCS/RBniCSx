# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to wrap a list of Functions."""

from __future__ import annotations

import typing

import dolfinx.fem
import numpy as np
import petsc4py

from rbnicsx.backends.export import export_functions
from rbnicsx.backends.import_ import import_functions


class FunctionsList(object):
    """
    A class wrapping a list of Functions.

    Parameters
    ----------
    function_space : dolfinx.fem.FunctionSpace
        Common finite element space of any Function that will be added to this list.

    Attributes
    ----------
    _function_space : dolfinx.fem.FunctionSpace
        Finite element space provided as input.
    _comm : mpi4py.MPI.Intracomm
        MPI communicator, derived from the finite element space provided as input.
    _list : tpying.List[dolfinx.fem.Function]
        Internal storage.
    """

    def __init__(self, function_space: dolfinx.fem.FunctionSpace) -> None:
        self._function_space = function_space
        self._comm = function_space.mesh.comm
        self._list = list()

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """Return the common finite element space of any Function that will be added to this list."""
        return self._function_space

    def append(self, function: dolfinx.fem.Function) -> None:
        """
        Append a dolfinx.fem.Function to the list.

        Parameters
        ----------
        function : dolfinx.fem.Function
            Function to be appended.
        """
        self._list.append(function)

    def extend(self, functions: typing.Iterable[dolfinx.fem.Function]) -> None:
        """
        Extend the current list with an iterable of dolfinx.fem.Function.

        Parameters
        ----------
        functions : typing.Iterable[dolfinx.fem.Function]
            Functions to be appended.
        """
        self._list.extend(functions)

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
        export_functions(self._list, np.arange(len(self._list), dtype=float), directory, filename)

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
        self._list = import_functions(self._function_space, directory, filename)

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
            assert other.size == len(self._list)
            if other.size == 0:
                return None
            else:
                output = dolfinx.fem.Function(self._function_space)
                for i in range(other.size):
                    output.vector.axpy(other[i], self._list[i].vector)
                output.vector.ghostUpdate(
                    addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
                return output
        else:
            return NotImplemented

    def __len__(self) -> int:
        """Return the number of functions currently stored in the list."""
        return len(self._list)

    def __getitem__(self, key: typing.Union[int, slice]) -> typing.Union[dolfinx.fem.Function, FunctionsList]:
        """
        Extract a single function from the list, or slice the list.

        Parameters
        ----------
        key : typing.Union[int, slice]
            Index (if int) or indices (if slice) to be extracted.

        Returns
        -------
        typing.Union[dolfinx.fem.Function, rbnicsx.backends.FunctionsList]
            Function at position `key` if `key` is an integer, otherwise FunctionsList obtained by
            storing every element at the indices in the slice `key`.
        """
        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, slice):
            output = FunctionsList(self._function_space)
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
        item : dolfinx.fem.Function
            Function to be stored.
        """
        self._list[key] = item

    def __iter__(self) -> typing.Iterator[dolfinx.fem.Function]:
        """Return an iterator over the list."""
        return self._list.__iter__()