# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to project matrices and vectors on the reduced basis."""

import functools
import typing

import dolfinx.fem
import mpi4py
import numpy as np
import petsc4py
import ufl

from minirox.backends.functions_list import FunctionsList


def create_online_vector(N: int) -> petsc4py.PETSc.Vec:
    """
    Create an online vector of the given dimension.

    Parameters
    ----------
    N : int
        Dimension of the vector.

    Returns
    -------
    petsc4py.PETSc.Vec
        Allocated online vector.
    """
    vec = petsc4py.PETSc.Vec().createSeq(N, comm=mpi4py.MPI.COMM_SELF)
    # Attach the identity local-to-global map
    lgmap = petsc4py.PETSc.LGMap().create(np.arange(N, dtype=np.int32), comm=vec.comm)
    vec.setLGMap(lgmap)
    lgmap.destroy()
    # Setup and return
    vec.setUp()
    return vec


def create_online_vector_block(N: typing.List[int]) -> petsc4py.PETSc.Vec:
    """
    Create an online vector of the given block dimensions.

    Parameters
    ----------
    N : typing.List[int]
        Dimension of the blocks of the vector.

    Returns
    -------
    petsc4py.PETSc.Vec
        Allocated online vector.
    """
    return create_online_vector(sum(N))


def create_online_matrix(M: int, N: int) -> petsc4py.PETSc.Mat:
    """
    Create an online matrix of the given dimension.

    Parameters
    ----------
    M, N : int
        Dimension of the matrix.

    Returns
    -------
    petsc4py.PETSc.Mat
        Allocated online matrix.
    """
    mat = petsc4py.PETSc.Mat().createDense((M, N), comm=mpi4py.MPI.COMM_SELF)
    # Attach the identity local-to-global map
    row_lgmap = petsc4py.PETSc.LGMap().create(np.arange(M, dtype=np.int32), comm=mat.comm)
    col_lgmap = petsc4py.PETSc.LGMap().create(np.arange(N, dtype=np.int32), comm=mat.comm)
    mat.setLGMap(row_lgmap, col_lgmap)
    row_lgmap.destroy()
    col_lgmap.destroy()
    # Setup and return
    mat.setUp()
    return mat


def create_online_matrix_block(M: typing.List[int], N: typing.List[int]) -> petsc4py.PETSc.Mat:
    """
    Create an online matrix of the given block dimensions.

    Parameters
    ----------
    M, N : typing.List[int]
        Dimension of the blocks of the matrix.

    Returns
    -------
    petsc4py.PETSc.Mat
        Allocated online matrix.
    """
    return create_online_matrix(sum(M), sum(N))


@functools.singledispatch
def project_vector(L: ufl.Form, B: FunctionsList) -> petsc4py.PETSc.Vec:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    L : ufl.Form
        Linear form to be projected.
    B : minirox.backends.FunctionsList
        Functions spanning the reduced basis space.

    Returns
    -------
    petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
    """
    b = create_online_vector(len(B))
    project_vector(b, L, B)
    return b


@project_vector.register
def _(b: petsc4py.PETSc.Vec, L: ufl.Form, B: FunctionsList) -> None:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : ufl.Form
        Linear form to be projected.
    B : minirox.backends.FunctionsList
        Functions spanning the reduced basis space.
    """
    test, = L.arguments()
    comm = B.function_space.mesh.comm
    for (n, fun) in enumerate(B):
        b.setValue(  # cannot use setValueLocal due to incompatibility with getSubVector
            n,
            comm.allreduce(dolfinx.fem.assemble_scalar(ufl.replace(L, {test: fun})), op=mpi4py.MPI.SUM),
            addv=petsc4py.PETSc.InsertMode.ADD)


@functools.singledispatch
def project_vector_block(L: typing.List[ufl.Form], B: typing.List[FunctionsList]) -> petsc4py.PETSc.Vec:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    L : typing.List[ufl.Form]
        Linear forms to be projected.
    B : typing.List[minirox.backends.FunctionsList]
        Functions spanning the reduced basis space associated to each solution component.

    Returns
    -------
    petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
    """
    b = create_online_vector_block([len(B_i) for B_i in B])
    project_vector_block(b, L, B)
    return b


@project_vector_block.register
def _(b: petsc4py.PETSc.Vec, L: typing.List[ufl.Form], B: typing.List[FunctionsList]) -> None:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : typing.List[ufl.Form]
        Linear forms to be projected.
    B : typing.List[minirox.backends.FunctionsList]
        Functions spanning the reduced basis space associated to each solution component.
    """
    assert len(L) == len(B)

    blocks = np.hstack((0, np.cumsum([len(B_i) for B_i in B])))
    for (i, (L_i, B_i)) in enumerate(zip(L, B)):
        is_i = petsc4py.PETSc.IS().createGeneral(
            np.arange(*blocks[i:i + 2], dtype=np.int32), comm=b.comm)
        b_i = b.getSubVector(is_i)
        project_vector(b_i, L_i, B_i)
        b.restoreSubVector(is_i, b_i)
        is_i.destroy()


@functools.singledispatch
def project_matrix(
    a: ufl.Form, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList]]
) -> petsc4py.PETSc.Mat:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    a : ufl.Form
        Bilinear form to be projected.
    B : typing.Union[minirox.backends.FunctionsList, typing.Tuple[minirox.backends.FunctionsList]]
        Functions spanning the reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.

    Returns
    -------
    petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
    """
    if isinstance(B, tuple):
        assert len(B) == 2
    else:
        B = (B, B)

    A = create_online_matrix(len(B[0]), len(B[1]))
    project_matrix(A, a, B)
    return A


@project_matrix.register
def _(
    A: petsc4py.PETSc.Mat, a: ufl.Form, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList]]
) -> None:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a : ufl.Form
        Bilinear form to be projected.
    B : typing.Union[minirox.backends.FunctionsList, typing.Tuple[minirox.backends.FunctionsList]]
        Functions spanning the reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.
    """
    if isinstance(B, tuple):
        assert len(B) == 2
    else:
        B = (B, B)

    test, trial = a.arguments()
    comm = B[0].function_space.mesh.comm
    for (m, fun_m) in enumerate(B[0]):
        for (n, fun_n) in enumerate(B[1]):
            A.setValueLocal(  # cannot use setValue due to incompatibility with getLocalSubMatrix
                m, n,
                comm.allreduce(
                    dolfinx.fem.assemble_scalar(ufl.replace(a, {test: fun_m, trial: fun_n})),
                    op=mpi4py.MPI.SUM),
                addv=petsc4py.PETSc.InsertMode.ADD)
    A.assemble()


@functools.singledispatch
def project_matrix_block(
    a: typing.List[typing.List[ufl.Form]],
    B: typing.Union[typing.List[FunctionsList], typing.Tuple[typing.List[FunctionsList]]]
) -> petsc4py.PETSc.Mat:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    a : typing.List[typing.List[ufl.Form]]
        Bilinear forms to be projected.
    B : typing.Union[
            typing.List[minirox.backends.FunctionsList],
            typing.Tuple[typing.List[minirox.backends.FunctionsList]]
        ]
        Functions spanning the reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.

    Returns
    -------
    petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
    """
    if isinstance(B, tuple):
        assert len(B) == 2
    else:
        B = (B, B)

    A = create_online_matrix_block([len(B_i) for B_i in B[0]], [len(B_j) for B_j in B[1]])
    project_matrix_block(A, a, B)
    return A


@project_matrix_block.register
def _(
    A: petsc4py.PETSc.Mat,
    a: typing.List[typing.List[ufl.Form]],
    B: typing.Union[typing.List[FunctionsList], typing.Tuple[typing.List[FunctionsList]]]
) -> None:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a : typing.List[typing.List[ufl.Form]]
        Bilinear forms to be projected.
    B : typing.Union[
            typing.List[minirox.backends.FunctionsList],
            typing.Tuple[typing.List[minirox.backends.FunctionsList]]
        ]
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

    row_blocks = np.hstack((0, np.cumsum([len(B_i) for B_i in B[0]])))
    col_blocks = np.hstack((0, np.cumsum([len(B_j) for B_j in B[1]])))
    for (i, (a_i, B_i)) in enumerate(zip(a, B[0])):
        is_i = petsc4py.PETSc.IS().createGeneral(
            np.arange(*row_blocks[i:i + 2], dtype=np.int32), comm=A.comm)
        for (j, (a_ij, B_j)) in enumerate(zip(a_i, B[1])):
            is_j = petsc4py.PETSc.IS().createGeneral(
                np.arange(*col_blocks[j:j + 2], dtype=np.int32), comm=A.comm)
            A_ij = A.getLocalSubMatrix(is_i, is_j)
            project_matrix(A_ij, a_ij, (B_i, B_j))
            A.restoreLocalSubMatrix(is_i, is_j, A_ij)
            is_j.destroy()
        is_i.destroy()
    A.assemble()
