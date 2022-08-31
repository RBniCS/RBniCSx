# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to perform a step of the Gram-Schmidt process on online functions."""

import typing

import numpy as np
import petsc4py.PETSc

from rbnicsx.online.functions_list import FunctionsList
from rbnicsx.online.projection import matrix_action


def gram_schmidt(
    functions_list: FunctionsList, new_function: petsc4py.PETSc.Vec, inner_product: petsc4py.PETSc.Mat
) -> None:
    """
    Perform a step of the Gram-Schmidt process on online functions.

    The input Function is orthonormalized and the result is added to the FunctionsList.

    Parameters
    ----------
    functions_list
        A set of orthonormal functions.
    new_function
        New function to be orthonormalized and added to the set.
    inner_product
        Online matrix which defines the inner product.
    """
    compute_inner_product = matrix_action(inner_product)

    orthonormalized = new_function.copy()
    for function_n in functions_list:
        orthonormalized.axpy(
            - compute_inner_product(function_n)(orthonormalized), function_n)
    norm = np.sqrt(compute_inner_product(orthonormalized)(orthonormalized))
    if norm != 0.0:
        orthonormalized *= 1.0 / norm
        functions_list.append(orthonormalized)


def gram_schmidt_block(
    functions_lists: FunctionsList, new_functions: typing.List[petsc4py.PETSc.Vec],
    inner_products: typing.List[petsc4py.PETSc.Mat]
) -> None:
    """
    Perform a step of the Gram-Schmidt process on online functions, where each function is made of several blocks.

    Parameters
    ----------
    functions_lists
        A set of orthonormal functions. Each function is made of several blocks, defined on possibly different
        function spaces. The inner FunctionsList contains all orthonormal functions of a single block, while
        the outer list collects the different blocks.
    new_functions
        New functions to be orthonormalized and added to the set.
    inner_products
        Online matrices which define the inner products of each block.
    """
    assert len(new_functions) == len(functions_lists)
    assert len(inner_products) == len(functions_lists)

    for (inner_product, functions_list, new_function) in zip(inner_products, functions_lists, new_functions):
        gram_schmidt(functions_list, new_function, inner_product)
