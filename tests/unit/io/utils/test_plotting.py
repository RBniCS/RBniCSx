# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for minirox.io.utils.plotting module."""

import dolfinx.mesh
import mpi4py
import pytest

from minirox.io.utils import plot_mesh


@pytest.fixture
def mesh() -> dolfinx.mesh.Mesh:
    """Generate a unit square mesh for use in tests in this file."""
    return dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 2, 2)


def test_plot_mesh(mesh: dolfinx.mesh.Mesh) -> None:
    """Check that plot_mesh executes without errors."""
    plot_mesh(mesh)
