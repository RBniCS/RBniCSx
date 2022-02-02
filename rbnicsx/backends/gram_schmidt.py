# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to perform a step of the Gram-Schmidt process."""

import typing

import dolfinx.fem
import numpy as np
import petsc4py
import ufl

from rbnicsx.backends.functions_list import FunctionsList
from rbnicsx.backends.projection import bilinear_form_action


def gram_schmidt(functions_list: FunctionsList, new_function: dolfinx.fem.Function, inner_product: ufl.Form) -> None:
    """
    Perform a step of the Gram-Schmidt process on functions.

    The input Function is orthonormalized and the result is added to the FunctionsList.

    Parameters
    ----------
    functions_list : rbnicsx.backends.FunctionsList
        A set of orthonormal functions.
    new_function : dolfinx.fem.Function
        New function to be orthonormalized and added to the set.
    inner_product : ufl.Form
        Bilinear form which defines the inner product. The resulting modes will be orthonormal
        w.r.t. this inner product.
    """
    compute_inner_product = bilinear_form_action(inner_product)

    orthonormalized = dolfinx.fem.Function(new_function.function_space)
    orthonormalized.x.array[:] = new_function.x.array
    for function_n in functions_list:
        orthonormalized.vector.axpy(
            - compute_inner_product(function_n, orthonormalized), function_n.vector)
    orthonormalized.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    norm = np.sqrt(compute_inner_product(orthonormalized, orthonormalized))
    if norm != 0.0:
        with orthonormalized.vector.localForm() as orthonormalized_local:
            orthonormalized_local *= 1.0 / norm
        functions_list.append(orthonormalized)


def gram_schmidt_block(
    functions_lists: FunctionsList, new_functions: typing.List[dolfinx.fem.Function],
    inner_products: typing.List[ufl.Form]
) -> None:
    """
    Perform a step of the Gram-Schmidt process on functions, where each function is made of several blocks.

    Parameters
    ----------
    functions_lists : typing.List[rbnicsx.backends.FunctionsList]
        A set of orthonormal functions. Each function is made of several blocks, defined on possibly different
        function spaces. The inner FunctionsList contains all orthonormal functions of a single block, while
        the outer list collects the different blocks.
    new_functions : typing.List[dolfinx.fem.Function]
        New functions to be orthonormalized and added to the set.
    inner_products : typing.List[ufl.Form]
        Bilinear forms which define the inner products of each block. The resulting modes
        will be orthonormal w.r.t. these inner products.
    """
    assert len(new_functions) == len(functions_lists)
    assert len(inner_products) == len(functions_lists)

    for (inner_product, functions_list, new_function) in zip(inner_products, functions_lists, new_functions):
        gram_schmidt(functions_list, new_function, inner_product)
