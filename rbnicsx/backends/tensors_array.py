# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to wrap an array of PETSc Mat or Vec assembled by dolfinx."""

from __future__ import annotations

import typing

import dolfinx.fem
import mpi4py.MPI
import numpy as np
import numpy.typing

from rbnicsx._backends.tensors_array import TensorsArray as TensorsArrayBase
from rbnicsx.backends.export import export_matrices, export_vectors
from rbnicsx.backends.import_ import import_matrices, import_vectors


class TensorsArray(TensorsArrayBase):
    """
    A class wrapping an array of PETSc Mat or Vec assembled by dolfinx.

    Parameters
    ----------
    form
        The form which is used to assemble the tensors.
    comm
        Common MPI communicator that the PETSc objects will use.
    shape
        The shape of the array.

    Attributes
    ----------
    _form
        Form provided as input.
    """

    def __init__(
        self, form: dolfinx.fem.Form, comm: mpi4py.MPI.Intracomm, shape: typing.Union[int, typing.Tuple[int, int]]
    ) -> None:
        self._form = form
        super().__init__(comm, shape)

    @property
    def form(self) -> dolfinx.fem.Form:
        """Return the form which is used to assemble the tensors."""
        return self._form

    def duplicate(self, shape: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None) -> TensorsArray:
        """
        Duplicate this object to a new empty TensorsArray.

        Parameters
        ----------
        shape
            The shape of the array. If not passed, the current shape is used

        Returns
        -------
        :
            A new TensorsArray constructed from the same first input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        if shape is None:
            shape = self.shape

        return TensorsArray(self._form, self._comm, shape)

    def _save(self, directory: str, filename: str) -> None:
        """
        Save this array to file querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to export the array.
        filename
            Name of the file where to export the array.
        """
        array_flattened = self._array.flatten("C")

        if self._type == "Mat":
            export_matrices(array_flattened, directory, filename)
        elif self._type == "Vec":
            export_vectors(array_flattened, directory, filename)
        else:
            raise RuntimeError()

    def _load(self, directory: str, filename: str) -> None:
        """
        Load an array from file into this object querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to import the array from.
        filename
            Name of the file where to import the array from.
        """
        if self._type == "Mat":
            array_flattened = import_matrices(self._form, self._comm, directory, filename)
        elif self._type == "Vec":
            array_flattened = import_vectors(self._form, self._comm, directory, filename)
        else:
            raise RuntimeError()

        for (linear_index, tensor) in enumerate(array_flattened):
            self._array[np.unravel_index(linear_index, self.shape)] = tensor
