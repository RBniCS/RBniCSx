# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to project online tensors on a (further reduced) reduced basis."""

import typing

import numpy as np
import petsc4py.PETSc
import plum

from rbnicsx._backends.online_tensors import (
    BlockMatSubMatrixCopier, BlockVecSubVectorCopier, create_online_matrix as create_matrix,
    create_online_matrix_block as create_matrix_block, create_online_vector as create_vector,
    create_online_vector_block as create_vector_block)
from rbnicsx._backends.projection import (
    project_matrix as project_matrix_super, project_matrix_block as project_matrix_block_super,
    project_vector as project_vector_super, project_vector_block as project_vector_block_super)
from rbnicsx.online.functions_list import FunctionsList

# We need to introduce a dependency on a multiple dispatch library and cannot use functools.singledispatch
# because the type of the first argument does not allow to differentiate between dispatched and
# non-dispatched version, e.g.
#   project_vector(L: petsc4py.PETSc.Vec, B: FunctionsList)
#   project_vector(b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: FunctionsList)
# in which, in both cases, the first argument is petsc4py.PETSc.Vec.

project_vector_dispatcher = plum.Dispatcher()


@project_vector_dispatcher
def _project_vector(L: petsc4py.PETSc.Vec, B: FunctionsList) -> petsc4py.PETSc.Vec:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    L
        Vector to be projected.
    B
        Functions spanning the (further reduced) reduced basis space.

    Returns
    -------
    :
        Online vector containing the result of the projection.
    """
    b = create_vector(len(B))
    _project_vector(b, L, B)
    return b


@project_vector_dispatcher  # type: ignore[no-redef]
def _project_vector(b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: FunctionsList) -> None:  # noqa: F811
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    b
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L
        Vector to be projected.
    B
        Functions spanning the (further reduced) reduced basis space.
    """
    project_vector_super(b, vector_action(L), B)


@typing.overload
def project_vector(  # type: ignore[no-any-unimported]
    L: petsc4py.PETSc.Vec, B: FunctionsList
) -> petsc4py.PETSc.Vec:  # pragma: no cover
    """Stub of project_vector for type checking. See the concrete implementation above."""
    ...


@typing.overload
def project_vector(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: FunctionsList
) -> None:  # pragma: no cover
    """Stub of project_vector for type checking. See the concrete implementation above."""
    ...


def project_vector(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Dispatcher of project_vector for type checking. See the concrete implementation above."""
    return _project_vector(*args, **kwargs)


project_vector_block_dispatcher = plum.Dispatcher()


@project_vector_block_dispatcher
def _project_vector_block(L: petsc4py.PETSc.Vec, B: typing.Sequence[FunctionsList]) -> petsc4py.PETSc.Vec:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    L
        Vector to be projected.
    B
        Functions spanning the (further reduced) reduced basis space associated to each solution component.

    Returns
    -------
    :
        Online vector containing the result of the projection.
    """
    b = create_vector_block([len(B_i) for B_i in B])
    _project_vector_block(b, L, B)
    return b


@project_vector_block_dispatcher  # type: ignore[no-redef]
def _project_vector_block(  # noqa: F811
    b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: typing.Sequence[FunctionsList]
) -> None:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    b
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L
        Vector to be projected.
    B
        Functions spanning the (further reduced) reduced basis space associated to each solution component.
    """
    N_L = [B_[0].size for B_ in B]
    with BlockVecSubVectorCopier(L, N_L) as L_copier:
        project_vector_block_super(b, [vector_action(L_) for L_ in L_copier], B)


@typing.overload
def project_vector_block(  # type: ignore[no-any-unimported]
    L: petsc4py.PETSc.Vec, B: typing.Sequence[FunctionsList]
) -> petsc4py.PETSc.Vec:  # pragma: no cover
    """Stub of project_vector_block for type checking. See the concrete implementation above."""
    ...


@typing.overload
def project_vector_block(  # type: ignore[no-any-unimported]
    b: petsc4py.PETSc.Vec, L: petsc4py.PETSc.Vec, B: typing.Sequence[FunctionsList]
) -> None:  # pragma: no cover
    """Stub of project_vector_block for type checking. See the concrete implementation above."""
    ...


def project_vector_block(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Dispatcher of project_vector_block for type checking. See the concrete implementation above."""
    return _project_vector_block(*args, **kwargs)


project_matrix_dispatcher = plum.Dispatcher()


@project_matrix_dispatcher
def _project_matrix(
    a: petsc4py.PETSc.Mat, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList, FunctionsList]]
) -> petsc4py.PETSc.Mat:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    a
        Matrix to be projected.
    B
        Functions spanning the (further reduced) reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.

    Returns
    -------
    :
        Online matrix containing the result of the projection.
    """
    if isinstance(B, tuple):
        assert len(B) == 2
        (M, N) = (len(B[0]), len(B[1]))
    else:
        M = len(B)
        N = M

    A = create_matrix(M, N)
    _project_matrix(A, a, B)
    return A


@project_matrix_dispatcher  # type: ignore[no-redef]
def _project_matrix(  # noqa: F811
    A: petsc4py.PETSc.Mat, a: petsc4py.PETSc.Mat,
    B: typing.Union[FunctionsList, typing.Tuple[FunctionsList, FunctionsList]]
) -> None:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    A
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a
        Matrix to be projected.
    B
        Functions spanning the (further reduced) reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.
    """
    project_matrix_super(A, matrix_action(a), B)


@typing.overload
def project_matrix(  # type: ignore[no-any-unimported]
    a: petsc4py.PETSc.Mat, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList, FunctionsList]]
) -> petsc4py.PETSc.Mat:  # pragma: no cover
    """Stub of project_matrix for type checking. See the concrete implementation above."""
    ...


@typing.overload
def project_matrix(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat, a: petsc4py.PETSc.Mat,
    B: typing.Union[FunctionsList, typing.Tuple[FunctionsList, FunctionsList]]
) -> None:  # pragma: no cover
    """Stub of project_matrix for type checking. See the concrete implementation above."""
    ...


def project_matrix(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Dispatcher of project_matrix for type checking. See the concrete implementation above."""
    return _project_matrix(*args, **kwargs)


project_matrix_block_dispatcher = plum.Dispatcher()


@project_matrix_block_dispatcher
def _project_matrix_block(
    a: petsc4py.PETSc.Mat,
    B: typing.Union[
        typing.Sequence[FunctionsList], typing.Tuple[typing.Sequence[FunctionsList], typing.Sequence[FunctionsList]]]
) -> petsc4py.PETSc.Mat:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    a
        Matrix to be projected.
    B
        Functions spanning the (further reduced) reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.

    Returns
    -------
    :
        Online matrix containing the result of the projection.
    """
    if isinstance(B, tuple):
        (M, N) = ([len(B_i) for B_i in B[0]], [len(B_j) for B_j in B[1]])
    else:
        M = [len(B_i) for B_i in B]
        N = M

    A = create_matrix_block(M, N)
    _project_matrix_block(A, a, B)
    return A


@project_matrix_block_dispatcher  # type: ignore[no-redef]
def _project_matrix_block(  # noqa: F811
    A: petsc4py.PETSc.Mat,
    a: petsc4py.PETSc.Mat,
    B: typing.Union[
        typing.Sequence[FunctionsList], typing.Tuple[typing.Sequence[FunctionsList], typing.Sequence[FunctionsList]]]
) -> None:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    A
        Online matrix containing the result of the projection.
        The matrix is not zeroed before assembly.
    a
        Matrix to be projected.
    B
        Functions spanning the (further reduced) reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.
    """
    if isinstance(B, tuple):
        (M_a, N_a) = ([B_i[0].size for B_i in B[0]], [B_j[0].size for B_j in B[1]])
    else:
        M_a = [B_i[0].size for B_i in B]
        N_a = M_a
    with BlockMatSubMatrixCopier(a, M_a, N_a) as a_copier:
        matrix_action_a = np.zeros((len(N_a), len(M_a)), dtype=object)
        for (i, j, a_ij) in a_copier:
            matrix_action_a[i][j] = matrix_action(a_ij)
        project_matrix_block_super(A, matrix_action_a.tolist(), B)


@typing.overload
def project_matrix_block(  # type: ignore[no-any-unimported]
    a: petsc4py.PETSc.Mat,
    B: typing.Union[
        typing.Sequence[FunctionsList], typing.Tuple[typing.Sequence[FunctionsList], typing.Sequence[FunctionsList]]]
) -> petsc4py.PETSc.Mat:  # pragma: no cover
    """Stub of project_matrix_block for type checking. See the concrete implementation above."""
    ...


@typing.overload
def project_matrix_block(  # type: ignore[no-any-unimported]
    A: petsc4py.PETSc.Mat,
    a: petsc4py.PETSc.Mat,
    B: typing.Union[
        typing.Sequence[FunctionsList], typing.Tuple[typing.Sequence[FunctionsList], typing.Sequence[FunctionsList]]]
) -> None:  # pragma: no cover
    """Stub of project_matrix_block for type checking. See the concrete implementation above."""
    ...


def project_matrix_block(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Dispatcher of project_matrix_block for type checking. See the concrete implementation above."""
    return _project_matrix_block(*args, **kwargs)


def vector_action(  # type: ignore[no-any-unimported]
    L: petsc4py.PETSc.Vec
) -> typing.Callable[[petsc4py.PETSc.Vec], petsc4py.PETSc.ScalarType]:
    """
    Return a callable that represents the action of the dot product between two vectors.

    Parameters
    ----------
    L
        Vector representing a linear form.

    Returns
    -------
    :
        A callable that represents the action of L on a vector.
    """

    def _(vec: petsc4py.PETSc.Vec) -> petsc4py.PETSc.ScalarType:  # type: ignore[no-any-unimported]
        """
        Compute the action of the dot product between two vectors.

        Parameters
        ----------
        vec
            Vector that should be applied to the linear form.

        Returns
        -------
        :
            Evaluation of the action of the linear form on the provided vector.
        """
        return L.dot(vec)

    return _


def matrix_action(  # type: ignore[no-any-unimported]
    a: petsc4py.PETSc.Mat
) -> typing.Callable[[petsc4py.PETSc.Vec], typing.Callable[[petsc4py.PETSc.Vec], petsc4py.PETSc.ScalarType]]:
    """
    Return a callable that represents the action of a vector-matrix-vector product.

    Parameters
    ----------
    a
        Matrix representing a bilinear form.

    Returns
    -------
    :
        A callable that represents the action of a on a pair of vectors.
    """
    a_dot_vec_1 = a.createVecLeft()

    def _trial_action(  # type: ignore[no-any-unimported]
        vec_1: petsc4py.PETSc.Vec
    ) -> typing.Callable[[petsc4py.PETSc.Vec], petsc4py.PETSc.ScalarType]:
        """
        Compute the action of a matrix-vector product.

        Parameters
        ----------
        vec_1
            Vector that should be applied to the right of the bilinear form, i.e. in the trial space.

        Returns
        -------
        :
            A callable that represents the action of a matrix-vector product.
        """
        a.mult(vec_1, a_dot_vec_1)

        def _test_action(vec_0: petsc4py.PETSc.Vec) -> petsc4py.PETSc.ScalarType:  # type: ignore[no-any-unimported]
            """
            Compute the action of a vector-matrix-vector product.

            Parameters
            ----------
            vec_0
                Vector that should be applied to the left of the bilinear form, i.e. in the test space.


            Returns
            -------
            :
                Evaluation of the action of the bilinear form on the provided pair of vectors.
            """
            return vec_0.dot(a_dot_vec_1)

        return _test_action

    return _trial_action
