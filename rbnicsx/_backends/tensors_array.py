# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to wrap an array of PETSc Mat or Vec."""

from __future__ import annotations

import abc
import itertools
import os
import typing

import mpi4py.MPI
import numpy as np
import numpy.typing
import petsc4py.PETSc

from rbnicsx.io import on_rank_zero


class TensorsArray(abc.ABC):
    """
    A class wrapping an array of PETSc Mat or Vec.

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
        Common MPI communicator that the PETSc objects will use.
    shape : typing.Union[int, typing.Tuple[int, int]]
        The shape of the array.

    Attributes
    ----------
    _comm : mpi4py.MPI.Intracomm
        MPI communicator provided as input.
    _array : np.typing.NDArray[typing.Union[petsc4py.PETSc.Mat, PETSc.Vec]]
        Internal storage.
    _type : str
        A string representing the type of tensors (Mat or Vec) currently stored.
    """

    def __init__(self, comm: mpi4py.MPI.Intracomm, shape: typing.Union[int, typing.Tuple[int, int]]) -> None:
        self._comm = comm
        self._array = np.full(shape, fill_value=None, dtype=object)
        self._type = None

    @property
    def comm(self) -> mpi4py.MPI.Intracomm:
        """Return the common MPI communicator that the PETSc objects will use."""
        return self._comm

    @property
    def shape(self) -> typing.List[int]:
        """Return the shape of the array."""
        return self._array.shape

    @property
    def type(self) -> str:
        """Return the type of tensors (Mat or Vec) currently stored."""
        return self._type

    @abc.abstractmethod
    def duplicate(self, shape: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None) -> TensorsArray:
        """
        Duplicate this object to a new empty TensorsArray.

        Parameters
        ----------
        shape : typing.Union[int, typing.Tuple[int, int]]
            The shape of the array. If not passed, the current shape is used

        Returns
        -------
        rbnicsx._backends.TensorsArray
            A new TensorsArray constructed from the same first input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        pass  # pragma: no cover

    def save(self, directory: str, filename: str) -> None:
        """
        Save this array to file.

        Parameters
        ----------
        directory : str
            Directory where to export the array.
        filename : str
            Name of the file where to export the array.
        """
        # Save type
        def save_type() -> None:
            with open(os.path.join(directory, filename + ".type"), "w") as type_file:
                if self._type is not None:
                    type_file.write(self._type)
                else:
                    type_file.write("None")
        on_rank_zero(self._comm, save_type)

        # Save tensors
        self._save(directory, filename)

    @abc.abstractmethod
    def _save(self, directory: str, filename: str) -> None:
        """
        Save this array to file querying the I/O functions in the backend.

        Parameters
        ----------
        directory : str
            Directory where to export the array.
        filename : str
            Name of the file where to export the array.
        """
        pass  # pragma: no cover

    def load(self, directory: str, filename: str) -> None:
        """
        Load an array from file into this object.

        Parameters
        ----------
        directory : str
            Directory where to import the array from.
        filename : str
            Name of the file where to import the array from.
        """
        assert all([tensor is None for tensor in self._array.flat])

        # Load type
        def load_type() -> str:
            with open(os.path.join(directory, filename + ".type"), "r") as type_file:
                return type_file.readline()
        self._type = on_rank_zero(self._comm, load_type)

        # Load tensors
        self._load(directory, filename)

    @abc.abstractmethod
    def _load(self, directory: str, filename: str) -> None:
        """
        Load an array from file into this object querying the I/O functions in the backend.

        Parameters
        ----------
        directory : str
            Directory where to import the array from.
        filename : str
            Name of the file where to import the array from.
        """
        pass  # pragma: no cover

    def contraction(self, *args: petsc4py.PETSc.Vec) -> petsc4py.PETSc.ScalarType:
        """
        Contract entries in the array.

        Parameters
        ----------
        *args : petsc4py.PETSc.Vec
            Arguments of the contraction.

        Returns
        -------
        petsc4py.PETSc.ScalarType
            The result of the contraction.
        """
        # Check that the correct number of arguments has been provided
        if self._type == "Mat":
            assert len(args) == len(self.shape) + 2
        elif self._type == "Vec":
            assert len(args) == len(self.shape) + 1
        else:
            raise RuntimeError()

        # Check dimension compatibility for arguments provided for dimensions up to array shape
        for dim in range(len(self.shape)):
            assert args[dim].getType() == petsc4py.PETSc.Vec.Type.SEQ
            assert args[dim].size == self.shape[dim]

        # Flatten the cartesian product of arguments provided for dimensions up to array shape
        first_args = [
            np.prod(args_tuple) for args_tuple in itertools.product(
                *(arg.array for arg in args[:len(self.shape)]))]
        assert len(first_args) == np.prod(self.shape)

        # Contract first on the dimensions up to array shape
        first_output = self._array[self._array.flat.coords].copy()
        first_output.zeroEntries()
        for (array_it, arg_it) in zip(self._array.flat, first_args):
            first_output.axpy(arg_it, array_it)
        if self._type == "Vec":
            first_output.ghostUpdate(
                addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)

        # Check dimension compatibility for arguments provided after the dimensions up to array shape
        if self._type == "Mat":
            assert isinstance(first_output, petsc4py.PETSc.Mat)
            assert args[-2].size == first_output.size[0]
            assert args[-1].size == first_output.size[1]
        elif self._type == "Vec":
            assert isinstance(first_output, petsc4py.PETSc.Vec)
            assert args[-1].size == first_output.size

        # Contract with the dimensions after array shape
        if self._type == "Mat":
            first_output_dot_last_arg = first_output.createVecLeft()
            first_output.mult(args[-1], first_output_dot_last_arg)
            return first_output_dot_last_arg.dot(args[-2])
        elif self._type == "Vec":
            return first_output.dot(args[-1])

    def __getitem__(
        self, key: typing.Union[int, typing.Tuple[int, int], slice, typing.Tuple[slice, slice]]
    ) -> typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec, TensorsArray]:
        """
        Extract a single tensor from the array, or slice the array.

        Parameters
        ----------
        key : typing.Union[int, slice]
            Index (if int or tuple of ints) or indices (if slice or tuple of slices) to be extracted.

        Returns
        -------
        typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec, rbnicsx._backends.TensorsArray]
            Tensor at position `key` if `key` is an integer, otherwise TensorsArray obtained by
            storing every element at the indices in the slice `key`.
        """
        if isinstance(key, int) or isinstance(key, tuple) and all([isinstance(key_, int) for key_ in key]):
            return self._array[key]
        elif isinstance(key, slice) or isinstance(key, tuple) and all([isinstance(key_, slice) for key_ in key]):
            output_array = self._array[key]
            output = self.duplicate(output_array.shape)
            output._array = output_array
            return output
        else:
            raise NotImplementedError()

    def __setitem__(
        self, key: typing.Union[int, typing.Tuple[int, int]],
        tensor: typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]
    ) -> None:
        """
        Update the content of the array with the provided tensor.

        Parameters
        ----------
        key : typing.Union[int, typing.Tuple[int, int]]
            Index to be updated.
        item : typing.Union[petsc4py.PETSc.Mat, petsc4py.PETSc.Vec]
            Tensor to be stored.
        """
        # Check that tensors of the same type are set
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

        # Replace storage
        # (and delegate checks on the compatibility of the key arguments with the shape of the storage array to numpy)
        self._array[key] = tensor
