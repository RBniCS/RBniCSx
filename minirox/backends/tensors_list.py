# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to wrap a list of PETSc Mat or Vec."""

from __future__ import annotations

import os
import typing

import dolfinx.fem
import mpi4py
import petsc4py

from minirox.backends.export import export_matrices, export_vectors
from minirox.backends.import_ import import_matrices, import_vectors


class TensorsList(object):
    """
    A class wrapping a list of PETSc Mat or Vec.

    Parameters
    ----------
    form : dolfinx.fem.Form
        The form which is used to assmemble the tensors.
    comm : mpi4py.MPI.Intracomm
        Common MPI communicator that the PETSc objects will use.

    Attributes
    ----------
    _form : dolfinx.fem.Form
        Form provided as input.
    _comm : mpi4py.MPI.Intracomm
        MPI communicator provided as input.
    _list : tpying.List[typing.Union[petsc4py.PETSc.Mat, PETSc.Vec]]
        Internal storage.
    """

    def __init__(self, form: dolfinx.fem.Form, comm: mpi4py.MPI.Intracomm) -> None:
        self._form = form
        self._comm = comm
        self._list = list()
        self._type = None

    def append(self, tensor: typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]) -> None:
        """
        Append a PETSc Mat or Vec to the list.

        Parameters
        ----------
        tensor : typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]
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
        # Save type
        if self._comm.rank == 0:
            with open(os.path.join(directory, filename + ".type"), "w") as type_file:
                type_file.write(self._type)
        # Save length
        if self._comm.rank == 0:
            with open(os.path.join(directory, filename + ".length"), "w") as length_file:
                length_file.write(str(len(self._list)))
        # Save functions
        if self._type == "Mat":
            export_matrices(self._list, directory, filename)
        elif self._type == "Vec":
            export_vectors(self._list, directory, filename)

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
        # Load type
        type_ = 0
        if self._comm.rank == 0:
            with open(os.path.join(directory, filename + ".type"), "r") as type_file:
                type_ = type_file.readline()
        self._type = self._comm.bcast(type_, root=0)
        # Load length
        length = 0
        if self._comm.rank == 0:
            with open(os.path.join(directory, filename + ".length"), "r") as length_file:
                length = int(length_file.readline())
        length = self._comm.bcast(length, root=0)
        # Load functions
        if self._type == "Mat":
            self._list = import_matrices(self._form, self._comm, directory, filename)
        elif self._type == "Vec":
            self._list = import_vectors(self._form, self._comm, directory, filename)

    def __mul__(self, other: petsc4py.PETSc.Vec) -> typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]:
        """
        Linearly combine tensors in the list.

        Parameters
        ----------
        other : petsc4py.PETSc.Vec
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]
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
                return output
        else:
            return NotImplemented

    def __len__(self) -> int:
        """Return the number of tensors currently stored in the list."""
        return len(self._list)

    def __getitem__(self, key: typing.Union[int, slice]) -> typing.Union[
            petsc4py.PETSc.Mat, petsc4py.PETSc.Vec, TensorsList]:
        """
        Extract a single tensor from the list, or slice the list before its end.

        Parameters
        ----------
        key : typing.Union[int, slice]
            Index (if int) or indices (if slice) to be extracted.

        Returns
        -------
        typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec, TensorsList]
            Tensor at position `key` if `key` is an integer, otherwise TensorsList obtained by
            storing every element at the indices in the slice `key`.
        """
        if isinstance(key, int):
            return self._list[key]
        elif isinstance(key, slice):
            assert key.start is None
            assert key.step is None
            assert key.stop is not None
            output = TensorsList(self._form, self._comm)
            output._list = self._list[key]
            return output
        else:
            raise NotImplementedError()

    def __setitem__(self, key: int, item: typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]) -> None:
        """
        Update the content of the list with the provided tensor.

        Parameters
        ----------
        key : int
            Index to be updated.
        function : typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]
            Tensor to be stored.
        """
        self._list[key] = item

    def __iter__(self) -> typing.Iterator[typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]]:
        """Return an iterator over the list."""
        return self._list.__iter__()
