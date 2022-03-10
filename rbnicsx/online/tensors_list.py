# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to wrap a list of PETSc Mat or Vec used to assemble online systems."""

from __future__ import annotations

import typing

import mpi4py.MPI

from rbnicsx._backends.tensors_list import TensorsList as TensorsListBase
from rbnicsx.online.export import export_matrices, export_matrices_block, export_vectors, export_vectors_block
from rbnicsx.online.import_ import import_matrices, import_matrices_block, import_vectors, import_vectors_block


class TensorsList(TensorsListBase):
    """
    A class wrapping a list of online PETSc Mat or Vec used to assemble online systems.

    Parameters
    ----------
    shape : typing.Union[int, typing.Tuple[int], typing.List[int], typing.Tuple[typing.List[int]]]
        Shape of the tensors which will be added to the list.

    Attributes
    ----------
    _shape : typing.Union[int, typing.Tuple[int], typing.List[int], typing.Tuple[typing.List[int]]]
        Shape provided as input.
    _is_block : bool
        Whether the tensor has a block structure or not.
    _comm : mpi4py.MPI.Intracomm
        MPI world communicator.
    _list : tpying.List[typing.Union[petsc4py.PETSc.Mat, PETSc.Vec]]
        Internal storage.
    _type : str
        A string representing the type of tensors (Mat or Vec) currently stored.
    """

    def __init__(
        self, shape: typing.Union[int, typing.Tuple[int], typing.List[int], typing.Tuple[typing.List[int]]]
    ) -> None:
        self._shape = shape
        if isinstance(shape, list):
            self._is_block = True  # block vector
        elif isinstance(shape, tuple) and all([isinstance(shape_, list) for shape_ in shape]):
            self._is_block = True  # block matrix
        else:
            self._is_block = False  # plain vector or plain matrix
        # Initialize using COMM_WORLD even though online tensors are on COMM_SELF to identify rank 0 for I/O
        super().__init__(mpi4py.MPI.COMM_WORLD)

    @property
    def shape(self) -> typing.Union[int, typing.Tuple[int], typing.List[int], typing.Tuple[typing.List[int]]]:
        """Return the shape of the tensors in the list."""
        return self._shape

    @property
    def is_block(self) -> bool:
        """Return whether the tensor has a block structure or not."""
        return self._is_block

    def duplicate(self) -> TensorsList:
        """
        Duplicate this object to a new empty TensorsList.

        Returns
        -------
        rbnicsx.backends.TensorsList
            A new TensorsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        return TensorsList(self._shape)

    def _save(self, directory: str, filename: str) -> None:
        """
        Save this list to file querying the I/O functions in the online backend.

        Parameters
        ----------
        directory : str
            Directory where to export the list.
        filename : str
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

    def _load(self, directory: str, filename: str) -> None:
        """
        Load a list from file into this object querying the I/O functions in the online backend.

        Parameters
        ----------
        directory : str
            Directory where to import the list from.
        filename : str
            Name of the file where to import the list from.
        """
        if self._type == "Mat":
            if self._is_block:
                self._list = import_matrices_block(*self._shape, directory, filename)
            else:
                self._list = import_matrices(*self._shape, directory, filename)
        elif self._type == "Vec":
            if self._is_block:
                self._list = import_vectors_block(self._shape, directory, filename)
            else:
                self._list = import_vectors(self._shape, directory, filename)
        else:
            raise RuntimeError()
