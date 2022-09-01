# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to perform a step of the Gram-Schmidt process on dolfinx functions."""

import typing

import dolfinx.fem
import numpy as np
import petsc4py.PETSc

from rbnicsx.backends.functions_list import FunctionsList


def gram_schmidt(  # type: ignore[no-any-unimported]
    functions_list: FunctionsList, new_function: dolfinx.fem.Function,
    compute_inner_product: typing.Callable[
        [dolfinx.fem.Function], typing.Callable[[dolfinx.fem.Function], petsc4py.PETSc.ScalarType]]
) -> None:
    """
    Perform a step of the Gram-Schmidt process on functions.

    The input Function is orthonormalized and the result is added to the FunctionsList.

    Parameters
    ----------
    functions_list
        A set of orthonormal functions.
    new_function
        New function to be orthonormalized and added to the set.
    compute_inner_product
        A callable x(u)(v) to compute the action of the inner product on the trial function u and test function v.
        The resulting modes will be orthonormal w.r.t. this inner product.
        Use rbnicsx.backends.bilinear_form_action to generate the callable x from a UFL form.
    """
    orthonormalized = dolfinx.fem.Function(new_function.function_space)
    orthonormalized.x.array[:] = new_function.x.array
    for function_n in functions_list:
        orthonormalized.vector.axpy(
            - compute_inner_product(function_n)(orthonormalized), function_n.vector)
    orthonormalized.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    norm = np.sqrt(compute_inner_product(orthonormalized)(orthonormalized))
    if norm != 0.0:
        with orthonormalized.vector.localForm() as orthonormalized_local:
            orthonormalized_local *= 1.0 / norm
        functions_list.append(orthonormalized)


def gram_schmidt_block(  # type: ignore[no-any-unimported]
    functions_lists: typing.Sequence[FunctionsList], new_functions: typing.Sequence[dolfinx.fem.Function],
    compute_inner_products: typing.Sequence[
        typing.Callable[[dolfinx.fem.Function], typing.Callable[[dolfinx.fem.Function], petsc4py.PETSc.ScalarType]]]
) -> None:
    """
    Perform a step of the Gram-Schmidt process on functions, where each function is made of several blocks.

    Parameters
    ----------
    functions_lists
        A set of orthonormal functions. Each function is made of several blocks, defined on possibly different
        function spaces. The inner FunctionsList contains all orthonormal functions of a single block, while
        the outer list collects the different blocks.
    new_functions
        New functions to be orthonormalized and added to the set.
    compute_inner_products
        A list of callables x_i(u_i)(v_i) to compute the action of the inner product on the trial function u_i
        and test function v_i associated to the i-th block.
        The resulting modes will be orthonormal w.r.t. this inner product.
        Use rbnicsx.backends.block_diagonal_bilinear_form_action to generate each callable x_i from a UFL form.
    """
    assert len(new_functions) == len(functions_lists)
    assert len(compute_inner_products) == len(functions_lists)

    for (functions_list, new_function, compute_inner_product) in zip(
            functions_lists, new_functions, compute_inner_products):
        gram_schmidt(functions_list, new_function, compute_inner_product)
