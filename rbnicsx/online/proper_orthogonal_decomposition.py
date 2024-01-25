# Copyright (C) 2021-2024 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to compute the proper orthogonal decomposition of online objects."""

import typing

import numpy as np
import numpy.typing
import petsc4py.PETSc
import plum

from rbnicsx._backends.proper_orthogonal_decomposition import (
    proper_orthogonal_decomposition_functions as proper_orthogonal_decomposition_functions_super,
    proper_orthogonal_decomposition_functions_block as proper_orthogonal_decomposition_functions_block_super,
    proper_orthogonal_decomposition_tensors as proper_orthogonal_decomposition_tensors_super, real_zero)
from rbnicsx.online.functions_list import FunctionsList
from rbnicsx.online.projection import matrix_action
from rbnicsx.online.tensors_list import TensorsList

# We could have used functools.singledispatch rather than plum, but since rbnicsx.online.projection
# introduces a dependency on plum we also use it here for its better handling in combining docstrings
# and its easier integration with mypy.


@plum.overload
def proper_orthogonal_decomposition(  # type: ignore[no-any-unimported]
    functions_list: FunctionsList, inner_product: petsc4py.PETSc.Mat, N: int = -1,
    tol: petsc4py.PETSc.RealType = real_zero, normalize: bool = True
) -> tuple[
    np.typing.NDArray[petsc4py.PETSc.RealType], FunctionsList, list[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of online snapshots.

    Parameters
    ----------
    functions_list
        Collected snapshots.
    inner_product
        Online matrix which defines the inner product. The resulting modes will be orthonormal
        w.r.t. this inner product.
    N
        Maximum number of modes to be computed. If not provided, it will be set to the number of collected snapshots.
    tol
        Tolerance on the retained energy. If not provided, it will be set to zero.
    normalize : bool, optional
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    :
        A tuple containing:
            1. Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
            2. Retained modes from the snapshots. Only the first few modes are returned, till either the
               maximum number N is reached or the tolerance on the retained energy is fulfilled.
            3. Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
               either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    compute_inner_product = matrix_action(inner_product)

    return proper_orthogonal_decomposition_functions_super(  # type: ignore[return-value]
        functions_list, compute_inner_product, _scale_online_vector, N, tol, normalize)


@plum.overload
def proper_orthogonal_decomposition(  # type: ignore[no-any-unimported] # noqa: F811
    tensors_list: TensorsList, N: int = -1, tol: petsc4py.PETSc.RealType = real_zero, normalize: bool = True
) -> tuple[
    np.typing.NDArray[petsc4py.PETSc.RealType], TensorsList, list[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of online tensors.

    Parameters
    ----------
    tensors_list
        Collected tensors.
    N
        Maximum number of modes to be computed. If not provided, it will be set to the number of collected tensors.
    tol
        Tolerance on the retained energy. If not provided, it will be set to zero.
    normalize
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    :
        A tuple containing:
            1. Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
            2. Retained modes from the tensors. Only the first few modes are returned, till either the
               maximum number N is reached or the tolerance on the retained energy is fulfilled.
            3. Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
               either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    return proper_orthogonal_decomposition_tensors_super(tensors_list, N, tol, normalize)  # type: ignore[return-value]


@plum.dispatch
def proper_orthogonal_decomposition(  # type: ignore[no-untyped-def] # noqa: ANN201, F811
    *args, **kwargs  # noqa: ANN002, ANN003
):
    """Compute the proper orthogonal decomposition of a set of online snapshots or tensors."""
    raise NotImplementedError("The abstract case has not been implemented")  # pragma: no cover


def proper_orthogonal_decomposition_block(  # type: ignore[no-any-unimported]
    functions_lists: typing.Sequence[FunctionsList], inner_products: list[petsc4py.PETSc.Mat],
    N: typing.Union[int, list[int]] = -1,
    tol: typing.Union[petsc4py.PETSc.RealType, list[petsc4py.PETSc.RealType]] = real_zero,
    normalize: bool = True
) -> tuple[
    list[np.typing.NDArray[petsc4py.PETSc.RealType]], list[FunctionsList],
    list[list[petsc4py.PETSc.Vec]]
]:
    """
    Compute the proper orthogonal decomposition of a set of online snapshots, each made of several blocks.

    Parameters
    ----------
    functions_lists
        Collected snapshots. Each snapshot is made of several blocks, defined on possibly different reduced bases.
        The inner FunctionsList contains all snapshots of a single block, while the outer list collects the different
        blocks.
    inner_products
        Online matrices which define the inner products of each block. The resulting modes
        will be orthonormal w.r.t. these inner products.
    N
        Maximum number of modes to be computed. If an integer value is passed then the same maximum number is
        used for each block. To set a different maximum number of modes for each block pass a list of integers.
        If not provided, it will be set to the number of collected snapshots.
    tol
        Tolerance on the retained energy. If a floating point value is passed then the same tolerance is
        used for each block. To set a different tolerance for each block pass a list of floating point numbers.
        If not provided, it will be set to zero.
    normalize
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    :
        A tuple containing:
            1. Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
               The outer list collects the eigenvalues of different blocks.
            2. Retained modes from the snapshots. Only the first few modes are returned, till either the
               maximum number N is reached or the tolerance on the retained energy is fulfilled.
               The outer list collects the retained modes of different blocks.
            3. Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
               either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
               The outer list collects the eigenvectors of different blocks.
    """
    compute_inner_products = [matrix_action(inner_product) for inner_product in inner_products]

    return proper_orthogonal_decomposition_functions_block_super(  # type: ignore[return-value]
        functions_lists, compute_inner_products, _scale_online_vector, N, tol, normalize)


def _scale_online_vector(  # type: ignore[no-any-unimported]
    vector: petsc4py.PETSc.Vec, factor: petsc4py.PETSc.RealType
) -> None:
    """Scale an online petsc4py.PETSc.Vec."""
    vector *= factor
