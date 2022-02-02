# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to wrap a list of Functions."""

from __future__ import annotations

import abc
import typing

import mpi4py
import petsc4py

Function = typing.TypeVar("Function")
FunctionSpace = typing.TypeVar("FunctionSpace")


class FunctionsList(abc.ABC, typing.Generic[Function, FunctionSpace]):
    """
    A class wrapping a list of Functions.

    Parameters
    ----------
    function_space : FunctionSpace
        Common finite element space of any Function that will be added to this list.
    comm : mpi4py.MPI.Intracomm
        Common MPI communicator that the Function objects will use.

    Attributes
    ----------
    _function_space : FunctionSpace
        Finite element space provided as input.
    _comm : mpi4py.MPI.Intracomm
        MPI communicator, derived from the finite element space provided as input.
    _list : tpying.List[Function]
        Internal storage.
    """

    def __init__(self, function_space: FunctionSpace, comm: mpi4py.MPI.Intracomm) -> None:
        self._function_space = function_space
        self._comm = comm
        self._list = list()

    @property
    def function_space(self) -> FunctionSpace:
        """Return the common finite element space of any Function that will be added to this list."""
        return self._function_space

    @property
    def comm(self) -> str:
        """Return the common MPI communicator that the Function objects will use."""
        return self._comm

    def append(self, function: Function) -> None:
        """
        Append a Function to the list.

        Parameters
        ----------
        function : Function
            Function to be appended.
        """
        self._list.append(function)

    def extend(self, functions: typing.Iterable[Function]) -> None:
        """
        Extend the current list with an iterable of Function.

        Parameters
        ----------
        functions : typing.Iterable[Function]
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
        self._save(directory, filename)

    @abc.abstractmethod
    def _save(self, directory: str, filename: str) -> None:
        """
        Save this list to file querying the I/O functions in the backend.

        Parameters
        ----------
        directory : str
            Directory where to export the list.
        filename : str
            Name of the file where to export the list.
        """
        pass  # pragma: no cover

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
        self._load(directory, filename)

    @abc.abstractmethod
    def _load(self, directory: str, filename: str) -> None:
        """
        Load a list from file into this object querying the I/O functions in the backend.

        Parameters
        ----------
        directory : str
            Directory where to import the list from.
        filename : str
            Name of the file where to import the list from.
        """
        pass  # pragma: no cover

    def __mul__(self, other: petsc4py.PETSc.Vec) -> Function:
        """
        Linearly combine functions in the list.

        Parameters
        ----------
        other : petsc4py.PETSc.Vec
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        Function
            Function object storing the result of the linear combination.
        """
        if isinstance(other, petsc4py.PETSc.Vec):
            assert other.getType() == petsc4py.PETSc.Vec.Type.SEQ
            assert other.size == len(self._list)
            if other.size == 0:
                return None
            else:
                return self._linearly_combine(other)
        else:
            return NotImplemented

    @abc.abstractmethod
    def _linearly_combine(coefficients: petsc4py.PETSc.Vec) -> Function:
        """
        Linearly combine functions in the list using Function's API.

        Parameters
        ----------
        coefficients : petsc4py.PETSc.Vec
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        Function
            Function object storing the result of the linear combination.
        """
        pass  # pragma: no cover

    def __len__(self) -> int:
        """Return the number of functions currently stored in the list."""
        return len(self._list)

    def __getitem__(self, key: typing.Union[int, slice]) -> typing.Union[Function, FunctionsList]:
        """
        Extract a single function from the list, or slice the list.

        Parameters
        ----------
        key : typing.Union[int, slice]
            Index (if int) or indices (if slice) to be extracted.

        Returns
        -------
        typing.Union[Function, rbnicsx._backends.FunctionsList]
            Function at position `key` if `key` is an integer, otherwise FunctionsList obtained by
            storing every element at the indices in the slice `key`.
        """
        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, slice):
            output = type(self)(self._function_space)
            output._list = self._list[key]
            return output
        else:
            raise NotImplementedError()

    def __setitem__(self, key: int, item: Function) -> None:
        """
        Update the content of the list with the provided function.

        Parameters
        ----------
        key : int
            Index to be updated.
        item : Function
            Function to be stored.
        """
        self._list[key] = item

    def __iter__(self) -> typing.Iterator[Function]:
        """Return an iterator over the list."""
        return self._list.__iter__()
