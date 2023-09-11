# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.mesh_motion module."""

import typing

import _pytest.fixtures
import dolfinx.fem
import dolfinx.mesh
import mpi4py.MPI
import numpy as np
import numpy.typing
import pytest

import rbnicsx.backends


@pytest.fixture
def mesh_2d() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.fixture
def mesh_3d() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_cube(comm, 2 * comm.size, 2 * comm.size, 2 * comm.size)


@pytest.fixture(params=["mesh_2d", "mesh_3d"])
def mesh(request: _pytest.fixtures.SubRequest) -> dolfinx.mesh.Mesh:
    """Parameterize tests with 2d and 3d meshes."""
    return request.getfixturevalue(request.param)  # type: ignore[no-any-return]


def shape_parametrization_expression(dim: int) -> typing.Callable[
        [np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]:
    """Return a function that computes the analytical expression of the shape parametrization."""

    def _(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
        """Analytical expression to be used in the definition of the shape parametrization."""
        if dim == 2:
            return np.array([x[0] + 2 * x[1]**2 + 3, 4 * x[0]**3 + 5 * x[1]**4 + 6])
        elif dim == 3:
            return np.array([
                x[0] + 2 * x[1]**2 + 3 * x[2]**3 + 4,
                5 * x[0]**4 + 6 * x[1]**5 + 7 * x[2]**6 + 8,
                9 * x[0]**7 + 10 * x[1]**8 + 11 * x[2]**9 + 12
            ])
        else:
            raise RuntimeError("Invalid topological dimension.")

    return _


@pytest.fixture
def shape_parametrization(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.Function:
    """Generate a shape parametrization of the unit square for use in tests in this file."""
    assert len(mesh.geometry.cmaps) == 1
    M = dolfinx.fem.functionspace(mesh, ("Lagrange", mesh.geometry.cmaps[0].degree, (mesh.geometry.dim, )))
    shape_parametrization = dolfinx.fem.Function(M)
    shape_parametrization.interpolate(shape_parametrization_expression(mesh.topology.dim))
    return shape_parametrization


@pytest.fixture
def identity(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.Function:
    """Generate the identity shape parametrization of the unit square for use in tests in this file."""
    assert len(mesh.geometry.cmaps) == 1
    M = dolfinx.fem.functionspace(mesh, ("Lagrange", mesh.geometry.cmaps[0].degree, (mesh.geometry.dim, )))
    identity = dolfinx.fem.Function(M)
    identity.interpolate(lambda x: x[:mesh.topology.dim])
    return identity


def test_backends_mesh_motion_context_manager(
    mesh: dolfinx.mesh.Mesh, shape_parametrization: dolfinx.fem.Function
) -> None:
    """Test that the mesh coordinates change inside the MeshMotion context manager, and reset afterwards."""
    reference_coordinates = mesh.geometry.x.copy()

    with rbnicsx.backends.MeshMotion(mesh, shape_parametrization):
        shape_parametrization_expression_dim = shape_parametrization_expression(mesh.topology.dim)
        expected_deformed_coordinates = shape_parametrization_expression_dim(reference_coordinates.T).T
        assert np.allclose(mesh.geometry.x[:, :mesh.topology.dim], expected_deformed_coordinates)
        assert np.allclose(mesh.geometry.x[:, mesh.topology.dim:], 0)

    assert np.allclose(mesh.geometry.x[:], reference_coordinates)


def test_backends_mesh_motion_shape_parametrization_property(
    mesh: dolfinx.mesh.Mesh, shape_parametrization: dolfinx.fem.Function
) -> None:
    """Test that the shape parametrization property stores the provided function."""
    mesh_motion = rbnicsx.backends.MeshMotion(mesh, shape_parametrization)
    assert mesh_motion.shape_parametrization is shape_parametrization


def test_backends_mesh_motion_identity(
    mesh: dolfinx.mesh.Mesh, shape_parametrization: dolfinx.fem.Function, identity: dolfinx.fem.Function
) -> None:
    """Test that the identity property initializes and stores a function."""
    mesh_motion = rbnicsx.backends.MeshMotion(mesh, shape_parametrization)
    assert mesh_motion._identity is None
    assert isinstance(mesh_motion.identity, dolfinx.fem.Function)
    assert isinstance(mesh_motion._identity, dolfinx.fem.Function)
    assert np.allclose(mesh_motion._identity.vector.array, identity.vector.array)


def test_backends_mesh_motion_deformation(mesh: dolfinx.mesh.Mesh, identity: dolfinx.fem.Function) -> None:
    """Test that the deformation property computes the difference between shape parametrization and identity map."""
    mesh_motion = rbnicsx.backends.MeshMotion(mesh, identity)
    assert np.allclose(mesh_motion.deformation.vector.array, 0)
