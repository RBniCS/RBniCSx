# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to compute the proper orthogonal decomposition of dolfinx objects."""

import typing

import dolfinx.fem
import numpy as np
import numpy.typing
import petsc4py.PETSc
import plum

from rbnicsx._backends.proper_orthogonal_decomposition import (
    proper_orthogonal_decomposition_functions as proper_orthogonal_decomposition_functions_super,
    proper_orthogonal_decomposition_functions_block as proper_orthogonal_decomposition_functions_block_super,
    proper_orthogonal_decomposition_tensors as proper_orthogonal_decomposition_tensors_super, real_zero)
from rbnicsx.backends.functions_list import FunctionsList
from rbnicsx.backends.tensors_list import TensorsList

# We could have used functools.singledispatch rather than plum, but since rbnicsx.online.projection
# introduces a dependency on plum we also use it here for its better handling in combining docstrings
# and its easier integration with mypy.


@plum.overload
def proper_orthogonal_decomposition(  # type: ignore[no-any-unimported]
    functions_list: FunctionsList,
    compute_inner_product: typing.Callable[
        [dolfinx.fem.Function], typing.Callable[[dolfinx.fem.Function], petsc4py.PETSc.RealType]],
    N: int = -1, tol: petsc4py.PETSc.RealType = real_zero, normalize: bool = True
) -> typing.Tuple[
    np.typing.NDArray[petsc4py.PETSc.RealType], FunctionsList, typing.List[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of snapshots.

    Parameters
    ----------
    functions_list
        Collected snapshots.
    compute_inner_product
        A callable x(u)(v) to compute the action of the inner product on the trial function u and test function v.
        The resulting modes will be orthonormal w.r.t. this inner product.
        Use rbnicsx.backends.bilinear_form_action to generate the callable x from a UFL form.
    N
        Maximum number of modes to be computed. If not provided, it will be set to the number of collected snapshots.
    tol
        Tolerance on the retained energy. If not provided, it will be set to zero.
    normalize
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
    return proper_orthogonal_decomposition_functions_super(  # type: ignore[return-value]
        functions_list, compute_inner_product, _scale_function, N, tol, normalize)


@plum.overload
def proper_orthogonal_decomposition(  # type: ignore[no-any-unimported] # noqa: F811
    tensors_list: TensorsList, N: int = -1, tol: petsc4py.PETSc.RealType = real_zero, normalize: bool = True
) -> typing.Tuple[
    np.typing.NDArray[petsc4py.PETSc.RealType], TensorsList, typing.List[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of tensors.

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
    """Compute the proper orthogonal decomposition of a set of snapshots or tensors."""
    raise NotImplementedError("The abstract case has not been implemented")  # pragma: no cover


def proper_orthogonal_decomposition_block(  # type: ignore[no-any-unimported]
    functions_lists: typing.Sequence[FunctionsList],
    compute_inner_products: typing.Sequence[
        typing.Callable[[dolfinx.fem.Function], typing.Callable[[dolfinx.fem.Function], petsc4py.PETSc.RealType]]],
    N: typing.Union[int, typing.List[int]] = -1,
    tol: typing.Union[petsc4py.PETSc.RealType, typing.List[petsc4py.PETSc.RealType]] = real_zero,
    normalize: bool = True
) -> typing.Tuple[
    typing.List[np.typing.NDArray[petsc4py.PETSc.RealType]], typing.List[FunctionsList],
    typing.List[typing.List[petsc4py.PETSc.Vec]]
]:
    """
    Compute the proper orthogonal decomposition of a set of snapshots, where each snapshot is made of several blocks.

    Parameters
    ----------
    functions_lists
        Collected snapshots. Each snapshot is made of several blocks, defined on possibly different function spaces.
        The inner FunctionsList contains all snapshots of a single block, while the outer list collects the different
        blocks.
    compute_inner_products
        A list of callables x_i(u_i)(v_i) to compute the action of the inner product on the trial function u_i
        and test function v_i associated to the i-th block.
        The resulting modes will be orthonormal w.r.t. this inner product.
        Use rbnicsx.backends.block_diagonal_bilinear_form_action to generate each callable x_i from a UFL form.
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
    return proper_orthogonal_decomposition_functions_block_super(  # type: ignore[return-value]
        functions_lists, compute_inner_products, _scale_function, N, tol, normalize)


def _scale_function(  # type: ignore[no-any-unimported]
    function: dolfinx.fem.Function, factor: petsc4py.PETSc.RealType
) -> None:
    """Scale a dolfinx Function."""
    with function.vector.localForm() as function_local:
        function_local *= factor
