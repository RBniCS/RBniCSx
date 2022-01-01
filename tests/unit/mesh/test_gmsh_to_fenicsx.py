# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox.mesh.gmsh_to_fenicsx module."""

import dolfinx_utils.test.skips
import gmsh

import minirox.mesh


@dolfinx_utils.test.skips.skip_in_parallel
def test_gmsh_to_fenicsx() -> None:
    """Check that gmsh_to_fenicsx executes without errors."""
    gmsh.initialize()
    gmsh.model.add("mesh_test_gmsh_to_fenicsx")

    mesh_size = 1.0
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
    p1 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, mesh_size)
    p2 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, mesh_size)
    p3 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, mesh_size)
    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p0)
    boundary = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])

    domain = gmsh.model.geo.addPlaneSurface([boundary])

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l0], 1)
    gmsh.model.addPhysicalGroup(1, [l1, l3], 2)
    gmsh.model.addPhysicalGroup(1, [l2], 3)
    gmsh.model.addPhysicalGroup(2, [domain], 1)
    gmsh.model.mesh.generate(2)

    mesh, subdomains, boundaries = minirox.mesh.gmsh_to_fenicsx(gmsh.model, gdim=2)
    gmsh.finalize()
