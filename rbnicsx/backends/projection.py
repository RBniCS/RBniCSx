# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to project UFL forms with arguments on a dolfinx function space on the reduced basis."""

import functools
import typing

import dolfinx.fem
import mpi4py
import petsc4py
import ufl

from rbnicsx._backends.online_tensors import (
    create_online_matrix, create_online_matrix_block, create_online_vector, create_online_vector_block)
from rbnicsx._backends.projection import (
    project_matrix as project_matrix_super, project_matrix_block as project_matrix_block_super,
    project_vector as project_vector_super, project_vector_block as project_vector_block_super)
from rbnicsx.backends.functions_list import FunctionsList


@functools.singledispatch
def project_vector(L: ufl.Form, B: FunctionsList) -> petsc4py.PETSc.Vec:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    L : ufl.Form
        Linear form to be projected.
    B : rbnicsx.backends.FunctionsList
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
    B : rbnicsx.backends.FunctionsList
        Functions spanning the reduced basis space.
    """
    project_vector_super(b, linear_form_action(L), B)


@functools.singledispatch
def project_vector_block(L: typing.List[ufl.Form], B: typing.List[FunctionsList]) -> petsc4py.PETSc.Vec:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    L : typing.List[ufl.Form]
        Linear forms to be projected.
    B : typing.List[rbnicsx.backends.FunctionsList]
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
    B : typing.List[rbnicsx.backends.FunctionsList]
        Functions spanning the reduced basis space associated to each solution component.
    """
    project_vector_block_super(b, [linear_form_action(L_) for L_ in L], B)


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
    B : typing.Union[rbnicsx.backends.FunctionsList, typing.Tuple[rbnicsx.backends.FunctionsList]]
        Functions spanning the reduced basis space. Two different basis of the same space
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

    A = create_online_matrix(M, N)
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
    B : typing.Union[rbnicsx.backends.FunctionsList, typing.Tuple[rbnicsx.backends.FunctionsList]]
        Functions spanning the reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.
    """
    project_matrix_super(A, bilinear_form_action(a), B)


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
    B : typing.Union[typing.List[rbnicsx.backends.FunctionsList], \
                     typing.Tuple[typing.List[rbnicsx.backends.FunctionsList]]]
        Functions spanning the reduced basis space associated to each solution component.
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

    A = create_online_matrix_block(M, N)
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
            typing.List[rbnicsx.backends.FunctionsList],
            typing.Tuple[typing.List[rbnicsx.backends.FunctionsList]]
        ]
        Functions spanning the reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.
    """
    project_matrix_block_super(A, [[bilinear_form_action(a_ij) for a_ij in a_i] for a_i in a], B)


def linear_form_action(L: ufl.Form) -> typing.Callable:
    """
    Return a callable that represents the action of a linear form on a function.

    Parameters
    ----------
    L : ufl.Form
        Linear form to be represented.

    Returns
    -------
    typing.Callable
        A callable that represents the action of L on a function.
    """
    test, = L.arguments()
    comm = test.ufl_function_space().mesh.comm

    def _(fun: dolfinx.fem.Function) -> petsc4py.PETSc.ScalarType:
        """
        Compute the action of a linear form on a function.

        Parameters
        ----------
        fun : dolfinx.fem.Function
            Function to be replaced to the test function.

        Returns
        -------
        petsc4py.PETSc.ScalarType
            Evaluation of the action of L on the provided function.
        """
        return comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.replace(L, {test: fun}))),
            op=mpi4py.MPI.SUM)

    return _


def bilinear_form_action(a: ufl.Form) -> typing.Callable:
    """
    Return a callable that represents the action of a bilinear form on a pair of functions.

    Parameters
    ----------
    a : ufl.Form
        Bilinear form to be represented.

    Returns
    -------
    typing.Callable
        A callable that represents the action of a on a pair of functions.
    """
    test, trial = a.arguments()
    comm = test.ufl_function_space().mesh.comm
    assert trial.ufl_function_space().mesh.comm == comm

    def _(fun_0: dolfinx.fem.Function, fun_1: dolfinx.fem.Function) -> petsc4py.PETSc.ScalarType:
        """
        Compute the action of a bilinear form on a pair of functions.

        Parameters
        ----------
        fun_0 : dolfinx.fem.Function
            Function to be replaced to the test function.
        fun_1 : dolfinx.fem.Function
            Function to be replaced to the trial function.

        Returns
        -------
        petsc4py.PETSc.ScalarType
            Evaluation of the action of a on the provided pair of functions.
        """
        return comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.replace(a, {test: fun_0, trial: fun_1}))),
            op=mpi4py.MPI.SUM)

    return _
