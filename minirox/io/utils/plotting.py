# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import dolfinx.plot
import numpy as np
import plotly.graph_objects as go
import pyvista


def _dolfinx_to_pyvista_mesh(mesh, dim=None):
    if dim is None:
        dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, dim)
    num_cells = mesh.topology.index_map(dim).size_local
    cell_entities = np.arange(num_cells, dtype=np.int32)
    pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(
        mesh, dim, cell_entities)
    grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)
    return grid


def plot_mesh(mesh):
    if mesh.topology.dim == 1:
        _plot_mesh_plotly(mesh)
    else:
        _plot_mesh_pyvista(mesh)


def _plot_mesh_plotly(mesh):
    fig = go.Figure(data=go.Scatter(
        x=mesh.geometry.x[:, 0], y=np.zeros(mesh.geometry.x.shape[0]),
        line=dict(color="blue", width=2, dash="solid"),
        marker=dict(color="blue", size=10),
        mode="lines+markers"))
    fig.update_xaxes(title_text="x")
    fig.show()


def _plot_mesh_pyvista(mesh):
    grid = _dolfinx_to_pyvista_mesh(mesh)
    plotter = pyvista.PlotterITK()
    plotter.add_mesh(grid)
    plotter.show()


def plot_mesh_entities(mesh, dim, entities):
    assert mesh.topology.dim > 1
    _plot_mesh_entities_pyvista(mesh, dim, entities, np.ones_like(entities))


def plot_mesh_tags(mesh_tags):
    mesh = mesh_tags.mesh
    assert mesh.topology.dim > 1
    _plot_mesh_entities_pyvista(mesh, mesh_tags.dim, mesh_tags.indices, mesh_tags.values)


def _plot_mesh_entities_pyvista(mesh, dim, indices, values):
    num_cells = mesh.topology.index_map(dim).size_local
    all_values = np.zeros(num_cells)
    for (index, value) in zip(indices, values):
        assert value > 0
        all_values[index] = value

    if dim == mesh.topology.dim:
        name = "Subdomains"
    elif dim == mesh.topology.dim - 1:
        name = "Boundaries"
    grid = _dolfinx_to_pyvista_mesh(mesh, dim)
    grid.cell_data[name] = all_values
    grid.set_active_scalars(name)
    plotter = pyvista.PlotterITK()
    plotter.add_mesh(grid)
    plotter.show()


def plot_scalar_field(scalar_field, name, warp_factor=0.0):
    mesh = scalar_field.function_space.mesh
    if mesh.topology.dim == 1:
        _plot_scalar_field_plotly(mesh, scalar_field, name)
    else:
        _plot_scalar_field_pyvista(mesh, scalar_field, name, warp_factor)


def _plot_scalar_field_plotly(mesh, scalar_field, name):
    fig = go.Figure(data=go.Scatter(
        x=mesh.geometry.x[:, 0], y=scalar_field.compute_point_values().reshape(-1),
        line=dict(color="blue", width=2, dash="solid"),
        marker=dict(color="blue", size=10),
        mode="lines+markers"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text=name)
    fig.show()


def _plot_scalar_field_pyvista(mesh, scalar_field, name, warp_factor):
    grid = _dolfinx_to_pyvista_mesh(mesh)
    grid.point_data[name] = scalar_field.compute_point_values()
    grid.set_active_scalars(name)
    plotter = pyvista.PlotterITK()
    if warp_factor != 0.0:
        assert warp_factor > 0.0
        warped = grid.warp_by_scalar(factor=warp_factor)
        plotter.add_mesh(warped)
    else:
        plotter.add_mesh(grid)
    plotter.show()


def plot_vector_field(vector_field, name, glyph_factor=0.0, warp_factor=0.0):
    mesh = vector_field.function_space.mesh
    assert mesh.topology.dim > 1
    _plot_vector_field_pyvista(mesh, vector_field, name, glyph_factor, warp_factor)


def _plot_vector_field_pyvista(mesh, vector_field, name, glyph_factor, warp_factor):
    grid = _dolfinx_to_pyvista_mesh(mesh)
    values = np.zeros((mesh.geometry.x.shape[0], 3))
    values[:, :2] = vector_field.compute_point_values()
    grid.point_data[name] = values
    plotter = pyvista.PlotterITK()
    if glyph_factor == 0.0:
        grid.set_active_vectors(name)
        if warp_factor == 0.0:
            plotter.add_mesh(grid)
        else:
            assert warp_factor > 0.0
            warped = grid.warp_by_vector(factor=warp_factor)
            plotter.add_mesh(warped)
    else:
        assert glyph_factor > 0.0
        assert warp_factor == 0.0
        glyphs = grid.glyph(orient=name, factor=glyph_factor)
        plotter.add_mesh(glyphs)
        grid_background = _dolfinx_to_pyvista_mesh(mesh, mesh.topology.dim - 1)
        plotter.add_mesh(grid_background)
    plotter.show()
