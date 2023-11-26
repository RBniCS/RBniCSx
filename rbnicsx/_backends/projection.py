# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to project matrices and vectors on the reduced basis."""

import typing

import petsc4py.PETSc

from rbnicsx._backends.functions_list import Function, FunctionsList
from rbnicsx._backends.online_tensors import BlockMatSubMatrixWrapper, BlockVecSubVectorWrapper


def project_vector(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: typing.Callable[[Function], petsc4py.PETSc.ScalarType], B: FunctionsList[Function]
) -> None:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    b
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L
        A callable L(v) to compute the action of the linear form L on the function v.
    B
        Functions spanning the reduced basis space.
    """
    for (n, fun) in enumerate(B):
        b.setValue(  # cannot use setValueLocal due to incompatibility with getSubVector
            n, L(fun), addv=petsc4py.PETSc.InsertMode.ADD)


def project_vector_block(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: typing.Sequence[typing.Callable[[Function], petsc4py.PETSc.ScalarType]],
    B: typing.Sequence[FunctionsList[Function]]
) -> None:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    b
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L
        A list of callables L_i(v) to compute the action of the i-th linear form L_i on the function v.
    B
        Functions spanning the reduced basis space associated to each solution component.
    """
    assert len(L) == len(B)

    N = [len(B_i) for B_i in B]
    assert b.size == sum(N)
    with BlockVecSubVectorWrapper(b, N) as b_wrapper:
        for (i, (b_i, L_i, B_i)) in enumerate(zip(b_wrapper, L, B)):
            project_vector(b_i, L_i, B_i)


def project_matrix(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat, a: typing.Callable[[Function], typing.Callable[[Function], petsc4py.PETSc.ScalarType]],
    B: typing.Union[FunctionsList[Function], typing.Tuple[FunctionsList[Function], FunctionsList[Function]]]
) -> None:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    A
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a
        A callable a(u)(v) to compute the action of the bilinear form a on the trial function u and test function v.
    B
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


def project_matrix_block(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat,
    a: typing.Sequence[typing.Sequence[
        typing.Callable[[Function], typing.Callable[[Function], petsc4py.PETSc.ScalarType]]]],
    B: typing.Union[
        typing.Sequence[FunctionsList[Function]],
        typing.Tuple[typing.Sequence[FunctionsList[Function]], typing.Sequence[FunctionsList[Function]]]]
) -> None:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    A
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a
        A matrix of callables a_ij(u)(v) to compute the action of the bilinear form a_ij on
        the trial function u and test function v.
    B
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

    M = [len(B_i) for B_i in B[0]]  # type: ignore[arg-type]
    N = [len(B_j) for B_j in B[1]]  # type: ignore[arg-type]
    assert A.size == (sum(M), sum(N))
    with BlockMatSubMatrixWrapper(A, M, N) as A_wrapper:
        for (i, j, A_ij) in A_wrapper:
            project_matrix(A_ij, a[i][j], (B[0][i], B[1][j]))
    A.assemble()
