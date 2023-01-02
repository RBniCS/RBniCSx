# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to wrap a list of PETSc Mat or Vec assembled by dolfinx."""

from __future__ import annotations

import dolfinx.fem
import mpi4py.MPI

from rbnicsx._backends.tensors_list import TensorsList as TensorsListBase
from rbnicsx.backends.export import export_matrices, export_vectors
from rbnicsx.backends.import_ import import_matrices, import_vectors


class TensorsList(TensorsListBase):
    """
    A class wrapping a list of PETSc Mat or Vec assembled by dolfinx.

    Parameters
    ----------
    form
        The form which is used to assemble the tensors.
    comm
        Common MPI communicator that the PETSc objects will use.

    Attributes
    ----------
    _form
        Form provided as input.
    """

    def __init__(self, form: dolfinx.fem.FormMetaClass, comm: mpi4py.MPI.Intracomm) -> None:
        self._form: dolfinx.fem.FormMetaClass = form
        super().__init__(comm)

    @property
    def form(self) -> dolfinx.fem.FormMetaClass:
        """Return the form which is used to assemble the tensors."""
        return self._form

    def duplicate(self) -> TensorsList:
        """
        Duplicate this object to a new empty TensorsList.

        Returns
        -------
        :
            A new TensorsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        return TensorsList(self._form, self._comm)

    def _save(self, directory: str, filename: str) -> None:
        """
        Save this list to file querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to export the list.
        filename
            Name of the file where to export the list.
        """
        if self._type == "Mat":
            export_matrices(self._list, directory, filename)
        elif self._type == "Vec":
            export_vectors(self._list, directory, filename)
        else:
            raise RuntimeError()

    def _load(self, directory: str, filename: str) -> None:
        """
        Load a list from file into this object querying the I/O functions in the backend.

        Parameters
        ----------
        directory
            Directory where to import the list from.
        filename
            Name of the file where to import the list from.
        """
        if self._type == "Mat":
            self._list = import_matrices(self._form, self._comm, directory, filename)
        elif self._type == "Vec":
            self._list = import_vectors(self._form, self._comm, directory, filename)
        else:
            raise RuntimeError()
