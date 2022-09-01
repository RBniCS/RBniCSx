# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to wrap a list of Functions."""

import abc
import typing

import mpi4py.MPI
import petsc4py.PETSc

Function = typing.TypeVar("Function")
Self = typing.TypeVar("Self", bound="FunctionsList[Function]")  # type: ignore[valid-type]


class FunctionsList(abc.ABC, typing.Generic[Function]):
    """
    A class wrapping a list of Functions.

    Parameters
    ----------
    comm
        Common MPI communicator that the Function objects will use.

    Attributes
    ----------
    _comm
        MPI communicator, derived from the finite element space provided as input.
    _list
        Internal storage.
    """

    def __init__(self: Self, comm: mpi4py.MPI.Intracomm) -> None:
        self._comm: mpi4py.MPI.Intracomm = comm
        self._list: typing.List[Function] = list()

    @property
    def comm(self: Self) -> mpi4py.MPI.Intracomm:
        """Return the common MPI communicator that the Function objects will use."""
        return self._comm

    @abc.abstractmethod
    def duplicate(self: Self) -> Self:
        """
        Duplicate this object to a new empty FunctionsList.

        Returns
        -------
        :
            A new FunctionsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        pass  # pragma: no cover

    def append(self: Self, function: Function) -> None:
        """
        Append a Function to the list.

        Parameters
        ----------
        function
            Function to be appended.
        """
        self._list.append(function)

    def extend(self: Self, functions: typing.Iterable[Function]) -> None:
        """
        Extend the current list with an iterable of Function.

        Parameters
        ----------
        functions
            Functions to be appended.
        """
        self._list.extend(functions)

    def clear(self: Self) -> None:
        """Clear the storage."""
        self._list = list()

    def save(self: Self, directory: str, filename: str) -> None:
        """
        Save this list to file.

        Parameters
        ----------
        directory
            Directory where to export the list.
        filename
            Name of the file where to export the list.
        """
        self._save(directory, filename)

    @abc.abstractmethod
    def _save(self: Self, directory: str, filename: str) -> None:
        """
        Save this list to file querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to export the list.
        filename
            Name of the file where to export the list.
        """
        pass  # pragma: no cover

    def load(self: Self, directory: str, filename: str) -> None:
        """
        Load a list from file into this object.

        Parameters
        ----------
        directory
            Directory where to import the list from.
        filename
            Name of the file where to import the list from.
        """
        assert len(self._list) == 0
        self._load(directory, filename)

    @abc.abstractmethod
    def _load(self: Self, directory: str, filename: str) -> None:
        """
        Load a list from file into this object querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to import the list from.
        filename
            Name of the file where to import the list from.
        """
        pass  # pragma: no cover

    def __mul__(self: Self, other: petsc4py.PETSc.Vec) -> Function:  # type: ignore[no-any-unimported]
        """
        Linearly combine functions in the list.

        Parameters
        ----------
        other
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        :
            Function object storing the result of the linear combination.
        """
        if isinstance(other, petsc4py.PETSc.Vec):
            assert other.getType() == petsc4py.PETSc.Vec.Type.SEQ
            assert other.size == len(self._list)
            return self._linearly_combine(other)
        else:
            return NotImplemented

    @abc.abstractmethod
    def _linearly_combine(self: Self, coefficients: petsc4py.PETSc.Vec) -> Function:  # type: ignore[no-any-unimported]
        """
        Linearly combine functions in the list using Function's API.

        Parameters
        ----------
        coefficients
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        :
            Function object storing the result of the linear combination.
        """
        pass  # pragma: no cover

    def __len__(self: Self) -> int:
        """Return the number of functions currently stored in the list."""
        return len(self._list)

    @typing.overload
    def __getitem__(self: Self, key: int) -> Function:  # pragma: no cover
        """Stub of __getitem__ for type checking. See the concrete implementation below."""
        ...

    @typing.overload
    def __getitem__(self: Self, key: slice) -> Self:  # pragma: no cover
        """Stub of __getitem__ for type checking. See the concrete implementation below."""
        ...

    def __getitem__(self: Self, key: typing.Union[int, slice]) -> typing.Union[Function, Self]:
        """
        Extract a single function from the list, or slice the list.

        Parameters
        ----------
        key
            Index (if int) or indices (if slice) to be extracted.

        Returns
        -------
        :
            Function at position `key` if `key` is an integer, otherwise FunctionsList obtained by
            storing every element at the indices in the slice `key`.
        """
        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, slice):
            output = self.duplicate()
            output._list = self._list[key]
            return output
        else:
            raise NotImplementedError()

    def __setitem__(self: Self, key: int, item: Function) -> None:
        """
        Update the content of the list with the provided function.

        Parameters
        ----------
        key
            Index to be updated.
        item
            Function to be stored.
        """
        self._list[key] = item

    def __iter__(self: Self) -> typing.Iterator[Function]:
        """Return an iterator over the list."""
        return self._list.__iter__()
