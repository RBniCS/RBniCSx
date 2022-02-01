# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.backends.symbolic_parameters module."""

import typing

import dolfinx.mesh
import mpi4py
import numpy as np
import pytest

import rbnicsx.backends


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    comm = mpi4py.MPI.COMM_WORLD
    return dolfinx.mesh.create_unit_square(comm, 2 * comm.size, 2 * comm.size)


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_symbolic_parameters_shape(mesh: dolfinx.mesh.Mesh, shape: typing.Tuple[int]) -> None:
    """Check null initialization of symbolic parameters."""
    mu = rbnicsx.backends.SymbolicParameters(mesh, shape=shape)
    assert mu.ufl_shape == shape
    assert mu.value.shape == shape
    assert np.allclose(mu.value, 0.0)


def test_symbolic_parameters_mesh(mesh: dolfinx.mesh.Mesh) -> None:
    """Check mesh initialization of symbolic parameters."""
    mu = rbnicsx.backends.SymbolicParameters(mesh, shape=())
    assert mu.ufl_domain() == mesh.ufl_domain()


def test_symbolic_parameters_name(mesh: dolfinx.mesh.Mesh) -> None:
    """Check name initialization of symbolic parameters."""
    mu = rbnicsx.backends.SymbolicParameters(mesh, shape=())
    assert mu._name == "mu"
    assert str(mu) == "mu"
