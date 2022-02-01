# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Context manager to deform a mesh by a prescribed shape parametrization."""

from __future__ import annotations

import types
import typing

import dolfinx.fem
import dolfinx.mesh


class MeshMotion(object):
    """
    A context manager to deform a mesh by a prescribed shape parametrization.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Mesh to be deformed.
    shape_parametrization : dolfinx.fem.Function
        Shape parametrization interpolated over all points of the mesh.

    Attributes
    ----------
    _mesh : dolfinx.mesh.Mesh
        Mesh provided as input.
    _shape_parametrization : dolfinx.fem.Function
        Interpolated shape parametrization provided as input.
    _reference_coordinates : np.typing.NDArray[float]
        Coordinates of the mesh points in the reference configuration.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, shape_parametrization: dolfinx.fem.Function) -> None:
        assert shape_parametrization.function_space.ufl_element().family() == "Lagrange"
        assert shape_parametrization.function_space.ufl_element().degree() == mesh.geometry.cmap.degree

        self._mesh = mesh
        self._shape_parametrization = shape_parametrization
        self._reference_coordinates = mesh.geometry.x.copy()
        self._identity = None

    @property
    def shape_parametrization(self) -> dolfinx.fem.Function:
        """Return the computed shape parametrization."""
        return self._shape_parametrization

    @property
    def identity(self) -> dolfinx.fem.Function:
        """Interpolate the identity map."""
        if self._identity is None:
            identity = dolfinx.fem.Function(self._shape_parametrization.function_space)
            identity.interpolate(lambda x: x[:self._mesh.topology.dim])
            self._identity = identity
        return self._identity

    @property
    def deformation(self) -> dolfinx.fem.Function:
        """Return the deformation, i.e. the difference between the shape parametrization and the identity map."""
        deformation = dolfinx.fem.Function(self._shape_parametrization.function_space)
        with deformation.vector.localForm() as deformation_local, \
                self._shape_parametrization.vector.localForm() as extended_local, \
                self.identity.vector.localForm() as identity_local:
            deformation_local[:] = extended_local - identity_local
        return deformation

    def __enter__(self) -> MeshMotion:
        """Enter the context and deform the mesh."""
        self._mesh.geometry.x[:, :self._mesh.topology.dim] = self._shape_parametrization.x.array.reshape(
            self._reference_coordinates.shape[0], self._shape_parametrization.function_space.dofmap.index_map_bs)
        return self

    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Reset the mesh to its reference configuration and exit the context."""
        self._mesh.geometry.x[:] = self._reference_coordinates
