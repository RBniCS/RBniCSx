# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to wrap an array of PETSc Mat or Vec used to assemble online systems."""

from __future__ import annotations

import typing

import mpi4py.MPI
import numpy as np
import petsc4py.PETSc

from rbnicsx._backends.online_tensors import create_online_vector as create_vector
from rbnicsx._backends.tensors_array import TensorsArray as TensorsArrayBase
from rbnicsx.online.export import export_matrices, export_matrices_block, export_vectors, export_vectors_block
from rbnicsx.online.import_ import import_matrices, import_matrices_block, import_vectors, import_vectors_block


class TensorsArray(TensorsArrayBase):
    """
    A class wrapping an array of online PETSc Mat or Vec used to assemble online systems.

    Parameters
    ----------
    content_shape
        Shape of the tensors which will be added to the list.
    array_shape
        The shape of the array.

    Attributes
    ----------
    _content_shape
        Shape provided as input.
    _is_block
        Whether the tensor has a block structure or not.
    """

    _vector_with_one_entry = create_vector(1)
    _vector_with_one_entry.setValue(0, 1)

    def __init__(
        self, content_shape: typing.Union[
            int, typing.Tuple[int, int], typing.List[int], typing.Tuple[typing.List[int], typing.List[int]]],
        array_shape: typing.Union[int, typing.Tuple[int, ...]]
    ) -> None:
        self._content_shape: typing.Union[
            int, typing.Tuple[int, int], typing.List[int], typing.Tuple[typing.List[int], typing.List[int]]
        ] = content_shape
        if isinstance(content_shape, list):
            is_block = True  # block vector
        elif isinstance(content_shape, tuple) and all([
                isinstance(content_shape_, list) for content_shape_ in content_shape]):
            is_block = True  # block matrix
        else:
            is_block = False  # plain vector or plain matrix
        self._is_block: bool = is_block
        # Initialize using COMM_WORLD even though online tensors are on COMM_SELF to identify rank 0 for I/O
        super().__init__(mpi4py.MPI.COMM_WORLD, array_shape)

    @property
    def flattened_shape(self) -> typing.Tuple[typing.Union[int, typing.List[int]], ...]:
        """Return the union of the shape of the array and the content shape."""
        if isinstance(self._content_shape, tuple):
            return self.shape + self.content_shape  # type: ignore[operator]
        else:
            return self.shape + (self.content_shape, )  # type: ignore[operator] # noqa: RUF005

    @property
    def content_shape(self) -> typing.Union[
            int, typing.Tuple[int, int], typing.List[int], typing.Tuple[typing.List[int], typing.List[int]]]:
        """Return the shape of the tensors in the array."""
        return self._content_shape

    @property
    def is_block(self) -> bool:
        """Return whether the tensor has a block structure or not."""
        return self._is_block

    def duplicate(
        self, array_shape: typing.Optional[typing.Union[int, typing.Tuple[int, ...]]] = None
    ) -> TensorsArray:
        """
        Duplicate this object to a new empty TensorsArray.

        Parameters
        ----------
        array_shape
            The shape of the array. If not passed, the current shape is used

        Returns
        -------
        :
            A new TensorsArray constructed from the same first input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        if array_shape is None:
            array_shape = self.shape

        return TensorsArray(self._content_shape, array_shape)

    def _save(self, directory: str, filename: str) -> None:
        """
        Save this array to file querying the I/O functions in the online backend.

        Parameters
        ----------
        directory
            Directory where to export the array.
        filename
            Name of the file where to export the array.
        """
        array_flattened = self._array.flatten("C").tolist()

        if self._type == "Mat":
            if self._is_block:
                export_matrices_block(array_flattened, directory, filename)
            else:
                export_matrices(array_flattened, directory, filename)
        elif self._type == "Vec":
            if self._is_block:
                export_vectors_block(array_flattened, directory, filename)
            else:
                export_vectors(array_flattened, directory, filename)
        else:
            raise RuntimeError()

    def _load(self, directory: str, filename: str) -> None:
        """
        Load an array from file into this object querying the I/O functions in the online backend.

        Parameters
        ----------
        directory
            Directory where to import the array from.
        filename
            Name of the file where to import the array from.
        """
        if self._type == "Mat":
            assert isinstance(self._content_shape, tuple)
            assert len(self._content_shape) == 2
            if self._is_block:
                assert isinstance(self._content_shape[0], list)
                assert isinstance(self._content_shape[1], list)
                array_flattened = import_matrices_block(
                    self._content_shape[0], self._content_shape[1], directory, filename)
            else:
                assert isinstance(self._content_shape[0], int)
                assert isinstance(self._content_shape[1], int)
                array_flattened = import_matrices(self._content_shape[0], self._content_shape[1], directory, filename)
        elif self._type == "Vec":
            if self._is_block:
                assert isinstance(self._content_shape, list)
                array_flattened = import_vectors_block(self._content_shape, directory, filename)
            else:
                assert isinstance(self._content_shape, int)
                array_flattened = import_vectors(self._content_shape, directory, filename)
        else:
            raise RuntimeError()

        for (linear_index, tensor) in enumerate(array_flattened):
            self._array[np.unravel_index(linear_index, self.shape)] = tensor

    def contraction(self, *args: petsc4py.PETSc.Vec) -> petsc4py.PETSc.ScalarType:  # type: ignore[no-any-unimported]
        """
        Contract entries in the array.

        Parameters
        ----------
        *args
            Arguments of the contraction.

        Returns
        -------
        :
            The result of the contraction.
        """
        implicit_args_indices = [
            flattened_index
            for (flattened_index, flattened_shape_) in enumerate(self.flattened_shape)
            if flattened_shape_ == 1 and flattened_index >= len(self.shape)]
        if len(implicit_args_indices) > 0 and len(args) < len(self.flattened_shape):
            for i in range(1, len(implicit_args_indices) + 1):
                assert implicit_args_indices[-i] == len(self.flattened_shape) - i
            implicit_args = [self._vector_with_one_entry] * (len(self.flattened_shape) - len(args))
            args = args + tuple(implicit_args)
        return super().contraction(*args)
