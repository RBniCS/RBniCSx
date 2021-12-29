# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for plotting dolfinx objects with plotly and pyvista."""

import os
import typing

import dolfinx.fem
import dolfinx.mesh
import dolfinx.plot
import numpy as np
import numpy.typing as npt
import petsc4py
import plotly.graph_objects as go
import pyvista


def _dolfinx_to_pyvista_mesh(mesh: dolfinx.mesh.Mesh, dim: int = None) -> pyvista.UnstructuredGrid:
    if dim is None:
        dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, dim)
    num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    cell_entities = np.arange(num_cells, dtype=np.int32)
    pyvista_cells, cell_types = dolfinx.plot.create_vtk_topology(
        mesh, dim, cell_entities)
    return pyvista.UnstructuredGrid(pyvista_cells, cell_types, mesh.geometry.x)


def plot_mesh(mesh: dolfinx.mesh.Mesh) -> None:
    """
    Plot a dolfinx.mesh.Mesh with plotly (in 1D) or pyvista (in 2D or 3D).

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Mesh to be plotted.
    """
    if mesh.topology.dim == 1:
        _plot_mesh_plotly(mesh)
    else:
        _plot_mesh_pyvista(mesh)


def _plot_mesh_plotly(mesh: dolfinx.mesh.Mesh) -> None:
    fig = go.Figure(data=go.Scatter(
        x=mesh.geometry.x[:, 0], y=np.zeros(mesh.geometry.x.shape[0]),
        line=dict(color="blue", width=2, dash="solid"),
        marker=dict(color="blue", size=10),
        mode="lines+markers"))
    fig.update_xaxes(title_text="x")
    if "PYTEST_CURRENT_TEST" not in os.environ:  # pragma: no cover
        fig.show()


def _plot_mesh_pyvista(mesh: dolfinx.mesh.Mesh) -> None:
    grid = _dolfinx_to_pyvista_mesh(mesh)
    plotter = pyvista.PlotterITK()
    plotter.add_mesh(grid)
    if "PYTEST_CURRENT_TEST" not in os.environ:  # pragma: no cover
        plotter.show()


def plot_mesh_entities(mesh: dolfinx.mesh.Mesh, dim: int, entities: npt.NDArray[int]) -> None:
    """
    Plot dolfinx.mesh.Mesh with pyvista, highlighting the provided `dim`-dimensional entities.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Mesh to be plotted. Current implementation is limited to 2D or 3D meshes.
    dim : int
        Dimension of the entities
    entities : numpy.typing.NDArray[int]
        Array containing the IDs of the entities to be highlighted.
    """
    assert mesh.topology.dim > 1
    _plot_mesh_entities_pyvista(mesh, dim, entities, np.ones_like(entities))


def plot_mesh_tags(mesh_tags: dolfinx.mesh.MeshTags) -> None:
    """
    Plot dolfinx.mesh.MeshTags with pyvista.

    Parameters
    ----------
    mesh : dolfinx.mesh.MeshTags
        MeshTags to be plotted. Current implementation is limited to 2D or 3D underlying meshes.
    """
    mesh = mesh_tags.mesh
    assert mesh.topology.dim > 1
    _plot_mesh_entities_pyvista(mesh, mesh_tags.dim, mesh_tags.indices, mesh_tags.values)


def _plot_mesh_entities_pyvista(
    mesh: dolfinx.mesh.Mesh, dim: int, indices: npt.NDArray[int], values: npt.NDArray[int]
) -> None:
    num_cells = mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
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
    if "PYTEST_CURRENT_TEST" not in os.environ:  # pragma: no cover
        plotter.show()


def plot_scalar_field(
    scalar_field: dolfinx.fem.Function, name: str, warp_factor: float = 0.0, part: str = "real"
) -> None:
    """
    Plot a scalar field with plotly (in 1D) or pyvista (in 2D or 3D).

    Parameters
    ----------
    scalar_field : dolfinx.fem.Function
        Function to be plotted, which contains a scalar field.
    name : str
        Name of the quantity stored in the scalar field.
    warp_factor : float, optional
        This argument is ignored for a field on a 1D mesh.
        For a 2D mesh: if provided then the factor is used to produce a warped representation
        the field; if not provided then the scalar field will be plotted on the mesh.
    part : str, optional
        Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
        The argument is ignored when plotting a real field.
    """
    mesh = scalar_field.function_space.mesh
    if mesh.topology.dim == 1:
        _plot_scalar_field_plotly(mesh, scalar_field, name, part)
    else:
        _plot_scalar_field_pyvista(mesh, scalar_field, name, warp_factor, part)


def _plot_scalar_field_plotly(
    mesh: dolfinx.mesh.Mesh, scalar_field: dolfinx.fem.Function, name: str, part: str
) -> None:
    values = scalar_field.compute_point_values().reshape(-1)
    values, name = _extract_part(values, name, part)
    fig = go.Figure(data=go.Scatter(
        x=mesh.geometry.x[:, 0], y=values,
        line=dict(color="blue", width=2, dash="solid"),
        marker=dict(color="blue", size=10),
        mode="lines+markers"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text=name)
    if "PYTEST_CURRENT_TEST" not in os.environ:  # pragma: no cover
        fig.show()


def _plot_scalar_field_pyvista(
    mesh: dolfinx.mesh.Mesh, scalar_field: dolfinx.fem.Function, name: str, warp_factor: float, part: str
) -> None:
    values = scalar_field.compute_point_values()
    values, name = _extract_part(values, name, part)
    grid = _dolfinx_to_pyvista_mesh(mesh)
    grid.point_data[name] = values
    grid.set_active_scalars(name)
    plotter = pyvista.PlotterITK()
    if warp_factor != 0.0:
        assert warp_factor > 0.0
        warped = grid.warp_by_scalar(factor=warp_factor)
        plotter.add_mesh(warped)
    else:
        plotter.add_mesh(grid)
    if "PYTEST_CURRENT_TEST" not in os.environ:  # pragma: no cover
        plotter.show()


def plot_vector_field(
    vector_field: dolfinx.fem.Function, name: str, glyph_factor: float = 0.0, warp_factor: float = 0.0,
    part: str = "real"
) -> None:
    """
    Plot a vector field with pyvista.

    Parameters
    ----------
    vector_field : dolfinx.fem.Function
        Function to be plotted, which contains a vector field.
    name : str
        Name of the quantity stored in the vector field.
    glyph_factor : float, optional
        If provided, the vector field is represented using a gylph, scaled by this factor.
    warp_factor : float, optional
        If provided then the factor is used to produce a warped representation
        the field; if not provided then the magnitude of the vector field will be plotted on the mesh.
        Only used when `glyph_factor` is not provided.
    part : str, optional
        Part of the solution (real or imag) to be plotted. By default, the real part is plotted.
        The argument is ignored when plotting a real field.
    """
    mesh = vector_field.function_space.mesh
    assert mesh.topology.dim > 1
    _plot_vector_field_pyvista(mesh, vector_field, name, glyph_factor, warp_factor, part)


def _plot_vector_field_pyvista(
    mesh: dolfinx.mesh.Mesh, vector_field: dolfinx.fem.Function, name: str, glyph_factor: float,
    warp_factor: float, part: str
) -> None:
    grid = _dolfinx_to_pyvista_mesh(mesh)
    values = np.zeros((mesh.geometry.x.shape[0], 3))
    values[:, :2] = vector_field.compute_point_values()
    values, name = _extract_part(values, name, part)
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
    if "PYTEST_CURRENT_TEST" not in os.environ:  # pragma: no cover
        plotter.show()


def _extract_part(
    values: npt.NDArray[petsc4py.PETSc.ScalarType], name: str, part: str
) -> typing.Tuple[npt.NDArray[float], str]:
    if np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
        if part == "real":
            values = values.real
            name = "real(" + name + ")"
        elif part == "imag":
            values = values.imag
            name = "imag(" + name + ")"
    return values, name
