# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to project matrices and vectors on the reduced basis."""

import typing

import petsc4py

from rbnicsx._backends.functions_list import FunctionsList
from rbnicsx._backends.online_tensors import BlockMatSubMatrixWrapper, BlockVecSubVectorWrapper


def project_vector(b: petsc4py.PETSc.Vec, L: typing.Callable, B: FunctionsList) -> None:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : typing.Callable
        A callable L(v) to compute the action of the linear form L on the function v.
    B : BackendsList
        Functions spanning the reduced basis space.
    """
    for (n, fun) in enumerate(B):
        b.setValue(  # cannot use setValueLocal due to incompatibility with getSubVector
            n, L(fun), addv=petsc4py.PETSc.InsertMode.ADD)


def project_vector_block(
    b: petsc4py.PETSc.Vec, L: typing.List[typing.Callable], B: typing.List[FunctionsList]
) -> None:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : typing.List[typing.Callable]
        A list of callables L_i(v) to compute the action of the i-th linear form L_i on the function v.
    B : FunctionsList
        Functions spanning the reduced basis space associated to each solution component.
    """
    assert len(L) == len(B)

    N = [len(B_i) for B_i in B]
    assert b.size == sum(N)
    with BlockVecSubVectorWrapper(b, N) as b_wrapper:
        for (i, (b_i, L_i, B_i)) in enumerate(zip(b_wrapper, L, B)):
            project_vector(b_i, L_i, B_i)


def project_matrix(
    A: petsc4py.PETSc.Mat, a: typing.Callable, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList]]
) -> None:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a : typing.Callable
        A callable a(u)(v) to compute the action of the bilinear form a on the trial function u and test function v.
    B : typing.Union[FunctionsList, typing.Tuple[FunctionsList]]
        Functions spanning the reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.
    """
    if isinstance(B, tuple):
        assert len(B) == 2
    else:
        B = (B, B)

    for (n, fun_n) in enumerate(B[1]):
        a_n = a(fun_n)
        for (m, fun_m) in enumerate(B[0]):
            A.setValueLocal(  # cannot use setValue due to incompatibility with getLocalSubMatrix
                m, n, a_n(fun_m), addv=petsc4py.PETSc.InsertMode.ADD)
    A.assemble()


def project_matrix_block(
    A: petsc4py.PETSc.Mat,
    a: typing.List[typing.List[typing.Callable]],
    B: typing.Union[typing.List[FunctionsList], typing.Tuple[typing.List[FunctionsList]]]
) -> None:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a : typing.List[typing.List[typing.Callable]]
        A matrix of callables a_ij(u)(v) to compute the action of the bilinear form a_ij on
        the trial function u and test function v.
    B : typing.Union[typing.List[FunctionsList], typing.Tuple[typing.List[FunctionsList]]]
        Functions spanning the reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.
    """
    if isinstance(B, tuple):
        assert len(B) == 2
    else:
        B = (B, B)
    assert len(B[0]) == len(a)
    assert all(len(row) == len(a[0]) for row in a[1:]), "Matrix of forms has incorrect rows"
    assert len(B[1]) == len(a[0])

    M = [len(B_i) for B_i in B[0]]
    N = [len(B_j) for B_j in B[1]]
    assert A.size == (sum(M), sum(N))
    with BlockMatSubMatrixWrapper(A, M, N) as A_wrapper:
        for (i, j, A_ij) in A_wrapper:
            project_matrix(A_ij, a[i][j], (B[0][i], B[1][j]))
    A.assemble()
