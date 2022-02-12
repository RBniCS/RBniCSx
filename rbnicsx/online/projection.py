# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to project online tensors on a (further reduced) reduced basis."""

import typing

import multipledispatch
import petsc4py

from rbnicsx._backends.online_tensors import (
    BlockMatSubMatrixCopier, BlockVecSubVectorCopier, create_online_matrix as create_matrix,
    create_online_matrix_block as create_matrix_block, create_online_vector as create_vector,
    create_online_vector_block as create_vector_block)
from rbnicsx._backends.projection import (
    project_matrix as project_matrix_super, project_matrix_block as project_matrix_block_super,
    project_vector as project_vector_super, project_vector_block as project_vector_block_super)
from rbnicsx.online.functions_list import FunctionsList

# We need to introduce a dependency on multipledispatch and cannot use functools.singledispatch
# because the type of the first argument does not allow to differentiate between dispatched and
# non-dispatched version, e.g.
#   project_vector(L: petsc4py.PETSc.Vec, B: FunctionsList)
#   project_vector(b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: FunctionsList)
# in which, in both cases, the first argument is petsc4py.PETSc.Vec.

project_vector = multipledispatch.Dispatcher("project_vector")


@project_vector.register(petsc4py.PETSc.Vec, FunctionsList)
def _(L: petsc4py.PETSc.Vec, B: FunctionsList) -> petsc4py.PETSc.Vec:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    L : petsc4py.PETSc.Vec
        Vector to be projected.
    B : rbnicsx.online.FunctionsList
        Functions spanning the (further reduced) reduced basis space.

    Returns
    -------
    petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
    """
    b = create_vector(len(B))
    project_vector(b, L, B)
    return b


@project_vector.register(petsc4py.PETSc.Vec, petsc4py.PETSc.Vec, FunctionsList)
def _(b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: FunctionsList) -> None:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : petsc4py.PETSc.Vec
        Vector to be projected.
    B : rbnicsx.online.FunctionsList
        Functions spanning the (further reduced) reduced basis space.
    """
    project_vector_super(b, vector_action(L), B)


project_vector_block = multipledispatch.Dispatcher("project_vector_block")


@project_vector_block.register(petsc4py.PETSc.Vec, list)
def _(L: petsc4py.PETSc.Vec, B: typing.List[FunctionsList]) -> petsc4py.PETSc.Vec:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    L : petsc4py.PETSc.Vec
        Vector to be projected.
    B : typing.List[rbnicsx.online.FunctionsList]
        Functions spanning the (further reduced) reduced basis space associated to each solution component.

    Returns
    -------
    petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
    """
    b = create_vector_block([len(B_i) for B_i in B])
    project_vector_block(b, L, B)
    return b


@project_vector_block.register(petsc4py.PETSc.Vec, petsc4py.PETSc.Vec, list)
def _(b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: typing.List[FunctionsList]) -> None:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : petsc4py.PETSc.Vec
        Vector to be projected.
    B : typing.List[rbnicsx.online.FunctionsList]
        Functions spanning the (further reduced) reduced basis space associated to each solution component.
    """
    N_L = [B_[0].size for B_ in B]
    with BlockVecSubVectorCopier(L, N_L) as L_copier:
        project_vector_block_super(b, [vector_action(L_) for L_ in L_copier], B)


project_matrix = multipledispatch.Dispatcher("project_matrix")


@project_matrix.register(petsc4py.PETSc.Mat, (FunctionsList, tuple))
def _(a: petsc4py.PETSc.Mat, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList]]) -> petsc4py.PETSc.Mat:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    a : petsc4py.PETSc.Mat
        Matrix to be projected.
    B : typing.Union[rbnicsx.online.FunctionsList, typing.Tuple[rbnicsx.online.FunctionsList]]
        Functions spanning the (further reduced) reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.

    Returns
    -------
    petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
    """
    if isinstance(B, tuple):
        assert len(B) == 2
        (M, N) = (len(B[0]), len(B[1]))
    else:
        M = len(B)
        N = M

    A = create_matrix(M, N)
    project_matrix(A, a, B)
    return A


@project_matrix.register(petsc4py.PETSc.Mat, petsc4py.PETSc.Mat, (FunctionsList, tuple))
def _(
    A: petsc4py.PETSc.Mat, a: petsc4py.PETSc.Mat, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList]]
) -> None:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a : petsc4py.PETSc.Mat
        Matrix to be projected.
    B : typing.Union[rbnicsx.online.FunctionsList, typing.Tuple[rbnicsx.online.FunctionsList]]
        Functions spanning the (further reduced) reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.
    """
    project_matrix_super(A, matrix_action(a), B)


project_matrix_block = multipledispatch.Dispatcher("project_matrix_block")


@project_matrix_block.register(petsc4py.PETSc.Mat, (list, tuple))
def _(
    a: petsc4py.PETSc.Mat,
    B: typing.Union[typing.List[FunctionsList], typing.Tuple[typing.List[FunctionsList]]]
) -> petsc4py.PETSc.Mat:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    a : petsc4py.PETSc.Mat
        Matrix to be projected.
    B : typing.Union[typing.List[rbnicsx.online.FunctionsList], \
                     typing.Tuple[typing.List[rbnicsx.online.FunctionsList]]]
        Functions spanning the (further reduced) reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.

    Returns
    -------
    petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
    """
    if isinstance(B, tuple):
        (M, N) = ([len(B_i) for B_i in B[0]], [len(B_j) for B_j in B[1]])
    else:
        M = [len(B_i) for B_i in B]
        N = M

    A = create_matrix_block(M, N)
    project_matrix_block(A, a, B)
    return A


@project_matrix_block.register(petsc4py.PETSc.Mat, petsc4py.PETSc.Mat, (list, tuple))
def _(
    A: petsc4py.PETSc.Mat,
    a: petsc4py.PETSc.Mat,
    B: typing.Union[typing.List[FunctionsList], typing.Tuple[typing.List[FunctionsList]]]
) -> None:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a : petsc4py.PETSc.Mat
        Matrix to be projected.
    B : typing.Union[
            typing.List[rbnicsx.online.FunctionsList],
            typing.Tuple[typing.List[rbnicsx.online.FunctionsList]]
        ]
        Functions spanning the (further reduced) reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.
    """
    if isinstance(B, tuple):
        (M_a, N_a) = ([B_i[0].size for B_i in B[0]], [B_j[0].size for B_j in B[1]])
    else:
        M_a = [B_i[0].size for B_i in B]
        N_a = M_a
    with BlockMatSubMatrixCopier(a, M_a, N_a) as a_copier:
        matrix_action_a = [[None for _ in range(len(N_a))] for _ in range(len(M_a))]
        for (i, j, a_ij) in a_copier:
            matrix_action_a[i][j] = matrix_action(a_ij)
        project_matrix_block_super(A, matrix_action_a, B)


def vector_action(L: petsc4py.PETSc.Vec) -> typing.Callable:
    """
    Return a callable that represents the action of the dot product between two vectors.

    Parameters
    ----------
    L : petsc4py.PETSc.Vec
        Vector representing a linear form.

    Returns
    -------
    typing.Callable
        A callable that represents the action of L on a vector.
    """

    def _(vec: petsc4py.PETSc.Vec) -> petsc4py.PETSc.ScalarType:
        """
        Compute the action of the dot product between two vectors.

        Parameters
        ----------
        vec : petsc4py.PETSc.Vec
            Vector that should be applied to the linear form.

        Returns
        -------
        petsc4py.PETSc.ScalarType
            Evaluation of the action of the linear form on the provided vector.
        """
        return L.dot(vec)

    return _


def matrix_action(a: petsc4py.PETSc.Mat) -> typing.Callable:
    """
    Return a callable that represents the action of a vector-matrix-vector product.

    Parameters
    ----------
    a : petsc4py.PETSc.Mat
        Matrix representing a bilinear form.

    Returns
    -------
    typing.Callable
        A callable that represents the action of a on a pair of vectors.
    """
    a_dot_vec_1 = a.createVecLeft()

    def _trial_action(vec_1: petsc4py.PETSc.Vec) -> typing.Callable:
        """
        Compute the action of a matrix-vector product.

        Parameters
        ----------
        vec_1 : petsc4py.PETSc.Vec
            Vector that should be applied to the right of the bilinear form, i.e. in the trial space.

        Returns
        -------
        petsc4py.PETSc.ScalarType
            A callable that represents the action of a matrix-vector product.
        """
        a.mult(vec_1, a_dot_vec_1)

        def _test_action(vec_0: petsc4py.PETSc.Vec) -> petsc4py.PETSc.ScalarType:
            """
            Compute the action of a vector-matrix-vector product.

            Parameters
            ----------
            vec_0: petsc4py.PETSc.Vec
                Vector that should be applied to the left of the bilinear form, i.e. in the test space.


            Returns
            -------
            petsc4py.PETSc.ScalarType
                Evaluation of the action of the bilinear form on the provided pair of vectors.
            """
            return vec_0.dot(a_dot_vec_1)

        return _test_action

    return _trial_action
