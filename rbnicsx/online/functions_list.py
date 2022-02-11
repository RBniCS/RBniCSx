# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online backend to wrap a list of PETSc Vec which represent solutions to online systems."""

from __future__ import annotations

import typing

import mpi4py
import petsc4py

from rbnicsx._backends.functions_list import FunctionsList as FunctionsListBase
from rbnicsx._backends.online_tensors import create_online_vector as create_vector
from rbnicsx.online.export import export_vectors, export_vectors_block
from rbnicsx.online.import_ import import_vectors, import_vectors_block


class FunctionsList(FunctionsListBase):
    """
    A class wrapping a list of online PETSc Vec which represent solutions to online systems.

    Parameters
    ----------
    shape : typing.Union[int, typing.List[int]]
        Shape of the vectors which will be added to the list.

    Attributes
    ----------
    _shape : typing.Union[int, typing.List[int]]
        Shape provided as input.
    _is_block : bool
        Whether the vector has a block structure or not.
    _comm : mpi4py.MPI.Intracomm
        MPI world communicator.
    _list : tpying.List[typing.Union[petsc4py.PETSc.Vec]]
        Internal storage.
    """

    def __init__(self, shape: typing.Union[int, typing.List[int]]) -> None:
        self._shape = shape
        if isinstance(shape, list):
            self._is_block = True
        else:
            self._is_block = False
        # Initialize using COMM_WORLD even though online vectors are on COMM_SELF to identify rank 0 for I/O
        super().__init__(mpi4py.MPI.COMM_WORLD)

    @property
    def shape(self) -> typing.Union[int, typing.List[int]]:
        """Return the shape of the vectors in the list."""
        return self._shape

    @property
    def is_block(self) -> bool:
        """Return whether the vector has a block structure or not."""
        return self._is_block

    def duplicate(self) -> FunctionsList:
        """
        Duplicate this object to a new empty FunctionsList.

        Returns
        -------
        rbnicsx.backends.FunctionsList
            A new FunctionsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        return FunctionsList(self._shape)

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
        if self._is_block:
            export_vectors_block(self._list, directory, filename)
        else:
            export_vectors(self._list, directory, filename)

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
        if self._is_block:
            self._list = import_vectors_block(self._shape, directory, filename)
        else:
            self._list = import_vectors(self._shape, directory, filename)

    def _linearly_combine(self, other: petsc4py.PETSc.Vec) -> petsc4py.PETSc.Vec:
        """
        Linearly combine functions in the list using petsc4py API.

        Parameters
        ----------
        other : petsc4py.PETSc.Vec
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        petsc4py.PETSc.Vec
            Vector storing the result of the linear combination.
        """
        if other.size > 0:
            output = self._list[0].copy()
            output.zeroEntries()
            for i in range(other.size):
                output.axpy(other[i], self._list[i])
            return output
        else:
            return create_vector(self.shape)
