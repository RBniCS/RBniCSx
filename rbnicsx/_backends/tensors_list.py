# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to wrap a list of PETSc Mat or Vec."""

import abc
import pathlib
import sys
import typing

import mpi4py.MPI
import petsc4py.PETSc

from rbnicsx.io import on_rank_zero

if sys.version_info >= (3, 11):  # pragma: no cover
    import typing as typing_extensions
else:  # pragma: no cover
    import typing_extensions


class TensorsList(abc.ABC):
    """
    A class wrapping a list of PETSc Mat or Vec.

    Parameters
    ----------
    comm
        Common MPI communicator that the PETSc objects will use.

    Attributes
    ----------
    _comm
        MPI communicator provided as input.
    _list
        Internal storage.
    _type
        A string representing the type of tensors (Mat or Vec) currently stored.
    """

    def __init__(self: typing_extensions.Self, comm: mpi4py.MPI.Intracomm) -> None:
        self._comm: mpi4py.MPI.Intracomm = comm
        self._list: typing.Union[  # type: ignore[no-any-unimported]
            list[petsc4py.PETSc.Mat], list[petsc4py.PETSc.Vec]] = list()
        self._type: typing.Optional[str] = None

    @property
    def comm(self: typing_extensions.Self) -> mpi4py.MPI.Intracomm:
        """Return the common MPI communicator that the PETSc objects will use."""
        return self._comm

    @property
    def type(self: typing_extensions.Self) -> typing.Optional[str]:
        """Return the type of tensors (Mat or Vec) currently stored."""
        return self._type

    @abc.abstractmethod
    def duplicate(self: typing_extensions.Self) -> typing_extensions.Self:
        """
        Duplicate this object to a new empty TensorsList.

        Returns
        -------
        :
            A new TensorsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        pass  # pragma: no cover

    def append(  # type: ignore[no-any-unimported]
        self: typing_extensions.Self, tensor: typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]
    ) -> None:
        """
        Append a PETSc Mat or Vec to the list.

        Parameters
        ----------
        tensor
            Tensor to be appended.
        """
        # Check that tensors of the same type are added
        if isinstance(tensor, petsc4py.PETSc.Mat):
            if self._type is None:
                self._type = "Mat"
            else:
                assert self._type == "Mat"
        elif isinstance(tensor, petsc4py.PETSc.Vec):
            if self._type is None:
                self._type = "Vec"
            else:
                assert self._type == "Vec"
        else:
            raise RuntimeError()

        # Append to storage
        self._list.append(tensor)

    def extend(  # type: ignore[no-any-unimported]
        self: typing_extensions.Self,
        tensors: typing.Union[typing.Iterable[petsc4py.PETSc.Mat], typing.Iterable[petsc4py.PETSc.Vec]]
    ) -> None:
        """
        Extend the current list with an iterable of PETSc Mat or an iterable of Vec.

        Parameters
        ----------
        tensors
            Tensors to be appended.
        """
        for tensor in tensors:
            self.append(tensor)

    def clear(self: typing_extensions.Self) -> None:
        """Clear the storage."""
        self._list = list()

    def save(self: typing_extensions.Self, directory: pathlib.Path, filename: str) -> None:
        """
        Save this list to file.

        Parameters
        ----------
        directory
            Directory where to export the list.
        filename
            Name of the file where to export the list.
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Save type
        def save_type() -> None:
            with open(directory / (filename + ".type"), "w") as type_file:
                if self._type is not None:
                    type_file.write(self._type)
                else:
                    type_file.write("None")
        on_rank_zero(self._comm, save_type)

        # Save tensors
        self._save(directory, filename)

    @abc.abstractmethod
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
        pass  # pragma: no cover

    def load(self: typing_extensions.Self, directory: pathlib.Path, filename: str) -> None:
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

        # Load type
        def load_type() -> str:
            with open(directory / (filename + ".type")) as type_file:
                return type_file.readline()
        self._type = on_rank_zero(self._comm, load_type)

        # Load tensors
        self._load(directory, filename)

    @abc.abstractmethod
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
        pass  # pragma: no cover

    def __mul__(  # type: ignore[no-any-unimported]
        self: typing_extensions.Self, other: petsc4py.PETSc.Vec
    ) -> typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]:
        """
        Linearly combine tensors in the list.

        Parameters
        ----------
        other
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        :
            Tensor object storing the result of the linear combination.
        """
        if isinstance(other, petsc4py.PETSc.Vec):
            assert other.getType() == petsc4py.PETSc.Vec.Type.SEQ
            assert other.size == len(self._list)
            if other.size == 0:
                return None
            else:
                output = self._list[0].copy()
                output.zeroEntries()
                for i in range(other.size):
                    output.axpy(other[i], self._list[i])
                if self._type == "Vec":
                    output.ghostUpdate(
                        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
                return output
        else:
            return NotImplemented

    def __len__(self: typing_extensions.Self) -> int:
        """Return the number of tensors currently stored in the list."""
        return len(self._list)

    @typing.overload
    def __getitem__(self: typing_extensions.Self, key: int) -> typing.Union[  # type: ignore[no-any-unimported]
            petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]:  # pragma: no cover
        ...

    @typing.overload
    def __getitem__(self: typing_extensions.Self, key: slice) -> typing_extensions.Self:  # pragma: no cover
        ...

    def __getitem__(self: typing_extensions.Self, key: typing.Union[int, slice]) -> typing.Union[  # type: ignore[no-any-unimported]
            petsc4py.PETSc.Mat, petsc4py.PETSc.Vec, typing_extensions.Self]:
        """
        Extract a single tensor from the list, or slice the list.

        Parameters
        ----------
        key
            Index (if int) or indices (if slice) to be extracted.

        Returns
        -------
        :
            Tensor at position `key` if `key` is an integer, otherwise TensorsList obtained by
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

    def __setitem__(  # type: ignore[no-any-unimported]
        self: typing_extensions.Self, key: int, tensor: typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]
    ) -> None:
        """
        Update the content of the list with the provided tensor.

        Parameters
        ----------
        key
            Index to be updated.
        item
            Tensor to be stored.
        """
        # Check that tensors of the same type are set
        assert self._type is not None  # since the user must have used .append() to originally add this item
        if isinstance(tensor, petsc4py.PETSc.Mat):
            assert self._type == "Mat"
        elif isinstance(tensor, petsc4py.PETSc.Vec):
            assert self._type == "Vec"
        else:
            raise RuntimeError()

        # Replace storage
        self._list[key] = tensor

    def __iter__(self: typing_extensions.Self) -> typing.Iterator[  # type: ignore[no-any-unimported]
            typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]]:
        """Return an iterator over the list."""
        return self._list.__iter__()
