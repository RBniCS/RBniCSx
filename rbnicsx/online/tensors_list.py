# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to wrap a list of PETSc Mat or Vec used to assemble online systems."""

import pathlib
import sys
import typing

import mpi4py.MPI

from rbnicsx._backends.tensors_list import TensorsList as TensorsListBase
from rbnicsx.online.export import export_matrices, export_matrices_block, export_vectors, export_vectors_block
from rbnicsx.online.import_ import import_matrices, import_matrices_block, import_vectors, import_vectors_block

if sys.version_info >= (3, 11):  # pragma: no cover
    import typing as typing_extensions
else:  # pragma: no cover
    import typing_extensions


@typing.final
class TensorsList(TensorsListBase):
    """
    A class wrapping a list of online PETSc Mat or Vec used to assemble online systems.

    Parameters
    ----------
    shape
        Shape of the tensors which will be added to the list.

    Attributes
    ----------
    _shape
        Shape provided as input.
    _is_block
        Whether the tensor has a block structure or not.
    """

    def __init__(
        self: typing_extensions.Self, shape: typing.Union[
            int, tuple[int, int], list[int], tuple[list[int], list[int]]]
    ) -> None:
        self._shape: typing.Union[
            int, tuple[int, int], list[int], tuple[list[int], list[int]]] = shape
        if isinstance(shape, list):
            is_block = True  # block vector
        elif isinstance(shape, tuple) and all([isinstance(shape_, list) for shape_ in shape]):
            is_block = True  # block matrix
        else:
            is_block = False  # plain vector or plain matrix
        self._is_block: bool = is_block
        # Initialize using COMM_WORLD even though online tensors are on COMM_SELF to identify rank 0 for I/O
        super().__init__(mpi4py.MPI.COMM_WORLD)

    @property
    def shape(self: typing_extensions.Self) -> typing.Union[
            int, tuple[int, int], list[int], tuple[list[int], list[int]]]:
        """Return the shape of the tensors in the list."""
        return self._shape

    @property
    def is_block(self: typing_extensions.Self) -> bool:
        """Return whether the tensor has a block structure or not."""
        return self._is_block

    def duplicate(self: typing_extensions.Self) -> typing_extensions.Self:
        """
        Duplicate this object to a new empty TensorsList.

        Returns
        -------
        :
            A new TensorsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        return TensorsList(self._shape)

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
        if self._type == "Mat":
            if self._is_block:
                export_matrices_block(self._list, directory, filename)
            else:
                export_matrices(self._list, directory, filename)
        elif self._type == "Vec":
            if self._is_block:
                export_vectors_block(self._list, directory, filename)
            else:
                export_vectors(self._list, directory, filename)
        else:
            raise RuntimeError()

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
        if self._type == "Mat":
            assert isinstance(self._shape, tuple)
            assert len(self._shape) == 2
            if self._is_block:
                assert isinstance(self._shape[0], list)
                assert isinstance(self._shape[1], list)
                self._list = import_matrices_block(self._shape[0], self._shape[1], directory, filename)
            else:
                assert isinstance(self._shape[0], int)
                assert isinstance(self._shape[1], int)
                self._list = import_matrices(self._shape[0], self._shape[1], directory, filename)
        elif self._type == "Vec":
            if self._is_block:
                assert isinstance(self._shape, list)
                self._list = import_vectors_block(self._shape, directory, filename)
            else:
                assert isinstance(self._shape, int)
                self._list = import_vectors(self._shape, directory, filename)
        else:
            raise RuntimeError()
