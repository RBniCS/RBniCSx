# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to wrap a list of dolfinx Functions."""

from __future__ import annotations

import dolfinx.fem
import numpy as np
import petsc4py

from rbnicsx._backends.functions_list import FunctionsList as FunctionsListBase
from rbnicsx.backends.export import export_functions
from rbnicsx.backends.import_ import import_functions


class FunctionsList(FunctionsListBase):
    """
    A class wrapping a list of dolfinx Functions.

    Parameters
    ----------
    function_space : dolfinx.fem.FunctionSpace
        Common finite element space of any Function that will be added to this list.

    Attributes
    ----------
    _function_space : dolfinx.fem.FunctionSpace
        Finite element space provided as input.
    _comm : mpi4py.MPI.Intracomm
        MPI communicator, derived from the finite element space provided as input.
    _list : tpying.List[dolfinx.fem.Function]
        Internal storage.
    """

    def __init__(self, function_space: dolfinx.fem.FunctionSpace) -> None:
        self._function_space = function_space
        super().__init__(function_space.mesh.comm)

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """Return the common finite element space of any Function that will be added to this list."""
        return self._function_space

    def duplicate(self) -> FunctionsList:
        """
        Duplicate this object to a new empty FunctionsList.

        Returns
        -------
        rbnicsx.backends.FunctionsList
            A new FunctionsList constructed from the same input arguments as this object.
            Elements of this object are not copied to the new object.
        """
        return FunctionsList(self._function_space)

    def _save(self, directory: str, filename: str) -> None:
        """
        Save this list to file querying the I/O functions in the backend.

        Parameters
        ----------
        directory : str
            Directory where to export the list.
        filename : str
            Name of the file where to export the list.
        """
        export_functions(self._list, np.arange(len(self._list), dtype=float), directory, filename)

    def _load(self, directory: str, filename: str) -> None:
        """
        Load a list from file into this object querying the I/O functions in the backend.

        Parameters
        ----------
        directory : str
            Directory where to import the list from.
        filename : str
            Name of the file where to import the list from.
        """
        self._list = import_functions(self._function_space, directory, filename)

    def _linearly_combine(self, other: petsc4py.PETSc.Vec) -> dolfinx.fem.Function:
        """
        Linearly combine functions in the list using Function's API.

        Parameters
        ----------
        other : petsc4py.PETSc.Vec
            Vector containing the coefficients of the linear combination.

        Returns
        -------
        dolfinx.fem.Function
            Function object storing the result of the linear combination.
        """
        output = dolfinx.fem.Function(self._function_space)
        for i in range(other.size):
            output.vector.axpy(other[i], self._list[i].vector)
        output.vector.ghostUpdate(
            addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        return output
