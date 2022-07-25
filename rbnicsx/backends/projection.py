# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to project UFL forms with arguments on a dolfinx function space on the reduced basis."""

import functools
import typing

import dolfinx.fem
import mpi4py.MPI
import numpy as np
import petsc4py.PETSc
import ufl

from rbnicsx._backends.online_tensors import (
    create_online_matrix, create_online_matrix_block, create_online_vector, create_online_vector_block)
from rbnicsx._backends.projection import (
    project_matrix as project_matrix_super, project_matrix_block as project_matrix_block_super,
    project_vector as project_vector_super, project_vector_block as project_vector_block_super)
from rbnicsx.backends.functions_list import FunctionsList


@functools.singledispatch
def project_vector(L: typing.Callable, B: FunctionsList) -> petsc4py.PETSc.Vec:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    L : typing.Callable
        A callable L(v) to compute the action of the linear form L on the function v.
        Use rbnicsx.backends.linear_form_action to generate the callable L from a UFL form.
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
def _(b: petsc4py.PETSc.Vec, L: typing.Callable, B: FunctionsList) -> None:
    """
    Project a linear form onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : typing.Callable
        A callable L(v) to compute the action of the linear form L on the function v.
        Use rbnicsx.backends.linear_form_action to generate the callable L from a UFL form.
    B : rbnicsx.backends.FunctionsList
        Functions spanning the reduced basis space.
    """
    project_vector_super(b, L, B)


@functools.singledispatch
def project_vector_block(L: typing.List[typing.Callable], B: typing.List[FunctionsList]) -> petsc4py.PETSc.Vec:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    L : typing.List[typing.Callable]
        A list of callables L_i(v) to compute the action of the i-th linear form L_i on the function v.
        Use rbnicsx.backends.linear_form_action to generate each callable L_i from a UFL form.
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
def _(b: petsc4py.PETSc.Vec, L: typing.List[typing.Callable], B: typing.List[FunctionsList]) -> None:
    """
    Project a list of linear forms onto the reduced basis.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        Online vector containing the result of the projection.
        The vector is not zeroed before assembly.
    L : typing.List[typing.Callable]
        A list of callables L_i(v) to compute the action of the i-th linear form L_i on the function v.
        Use rbnicsx.backends.linear_form_action to generate each callable L_i from a UFL form.
    B : typing.List[rbnicsx.backends.FunctionsList]
        Functions spanning the reduced basis space associated to each solution component.
    """
    project_vector_block_super(b, L, B)


@functools.singledispatch
def project_matrix(
    a: typing.Callable, B: typing.Union[FunctionsList, typing.Tuple[FunctionsList]]
) -> petsc4py.PETSc.Mat:
    """
    Project a bilinear form onto the reduced basis.

    Parameters
    ----------
    a : typing.Callable
        A callable a(u)(v) to compute the action of the bilinear form a on the trial function u and test function v.
        Use rbnicsx.backends.bilinear_form_action to generate the callable a from a UFL form.
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
        Use rbnicsx.backends.bilinear_form_action to generate the callable a from a UFL form.
    B : typing.Union[rbnicsx.backends.FunctionsList, typing.Tuple[rbnicsx.backends.FunctionsList]]
        Functions spanning the reduced basis space. Two different basis of the same space
        can be provided, e.g. as in Petrov-Galerkin methods.
    """
    project_matrix_super(A, a, B)


@functools.singledispatch
def project_matrix_block(
    a: typing.List[typing.List[typing.Callable]],
    B: typing.Union[typing.List[FunctionsList], typing.Tuple[typing.List[FunctionsList]]]
) -> petsc4py.PETSc.Mat:
    """
    Project a matrix of bilinear forms onto the reduced basis.

    Parameters
    ----------
    a : typing.List[typing.List[typing.Callable]]
        A matrix of callables a_ij(u)(v) to compute the action of the bilinear form a_ij on
        the trial function u and test function v.
        Use rbnicsx.backends.bilinear_form_action to generate each callable a_ij from a UFL form.
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
        Use rbnicsx.backends.bilinear_form_action to generate each callable a_ij from a UFL form.
    B : typing.Union[
            typing.List[rbnicsx.backends.FunctionsList],
            typing.Tuple[typing.List[rbnicsx.backends.FunctionsList]]
        ]
        Functions spanning the reduced basis space associated to each solution component.
        Two different basis of the same space can be provided, e.g. as in Petrov-Galerkin methods.
    """
    project_matrix_block_super(A, a, B)


class FormArgumentsReplacer(object):
    """A wrapper to successive calls to ufl.replace and dolfinx.fem.form."""

    def __init__(
        self, form: ufl.Form, test: typing.Optional[bool] = False, trial: typing.Optional[bool] = False
    ) -> None:
        form_arguments = form.arguments()

        dict_replacement = dict()
        if test:
            test_replacement = dolfinx.fem.Function(form_arguments[0].ufl_function_space())
            dict_replacement[form_arguments[0]] = test_replacement
        else:
            test_replacement = None
        self._test_replacement = test_replacement
        if trial:
            trial_replacement = dolfinx.fem.Function(form_arguments[1].ufl_function_space())
            dict_replacement[form_arguments[1]] = trial_replacement
        else:
            trial_replacement = None
        self._trial_replacement = trial_replacement
        self._form = ufl.replace(form, dict_replacement)
        self._form_cpp = dolfinx.fem.form(self._form)

        self._comm = form_arguments[0].ufl_function_space().mesh.comm
        if len(form_arguments) > 1:
            assert all(
                [form_argument.ufl_function_space().mesh.comm == self._comm for form_argument in form_arguments])

    @property
    def comm(self) -> mpi4py.MPI.Intracomm:
        """Return the common MPI communicator of the mesh of this form."""
        return self._comm

    @property
    def form(self) -> ufl.Form:
        """Return the UFL form, with replacements carried out."""
        return self._form

    @property
    def form_cpp(self) -> dolfinx.fem.FormMetaClass:
        """Return the compiled form, with replacements carried out."""
        return self._form_cpp

    def replace(
        self, test: typing.Optional[typing.Union[dolfinx.fem.Function, ufl.core.expr.Expr]] = None,
        trial: typing.Optional[typing.Union[dolfinx.fem.Function, ufl.core.expr.Expr]] = None
    ) -> None:
        """
        Update the placeholder associated to one or more arguments.

        Parameters
        ----------
        test, trial : typing.Optional[typing.Union[dolfinx.fem.Function, ufl.core.expr.Expr]]
            Expressions to be replaced to the form arguments.
            If the expression is provided as a dolfinx Function, such function will be used as a replacement.
            If the expression is provided as an UFL expression, such expression will first be interpolated
            on the function space and then used as a replacement.
        """
        if test is not None:
            assert self._test_replacement is not None
            if isinstance(test, dolfinx.fem.Function):
                self._copy_dolfinx_function(test, self._test_replacement)
            else:
                assert isinstance(test, ufl.core.expr.Expr)
                self._interpolate_ufl_expression(test, self._test_replacement)
        if trial is not None:
            assert self._trial_replacement is not None
            if isinstance(trial, dolfinx.fem.Function):
                self._copy_dolfinx_function(trial, self._trial_replacement)
            else:
                assert isinstance(trial, ufl.core.expr.Expr)
                self._interpolate_ufl_expression(trial, self._trial_replacement)

    @staticmethod
    def _interpolate_ufl_expression(source: ufl.core.expr.Expr, destination: dolfinx.fem.Function) -> None:
        """Interpolate a field which is provided as a UFL expression."""
        destination.interpolate(
            dolfinx.fem.Expression(source, destination.function_space.element.interpolation_points()))

    @staticmethod
    def _copy_dolfinx_function(source: dolfinx.fem.Function, destination: dolfinx.fem.Function) -> None:
        """Copy a dolfinx Function to the internal storage."""
        with source.vector.localForm() as source_local, destination.vector.localForm() as destination_local:
            source_local.copy(destination_local)


@functools.singledispatch
def linear_form_action(L: ufl.Form, part: typing.Optional[str] = None) -> typing.Callable:
    """
    Return a callable that represents the action of a linear form on a function.

    Parameters
    ----------
    L : ufl.Form
        Linear form to be represented.
    part : typing.Optional[str]
        Optional part (real or complex) to extract from the action result.
        If not provided, no postprocessing of the result will be carried out.

    Returns
    -------
    typing.Callable
        A callable that represents the action of L on a function.
    """
    L_replacement_cpp = FormArgumentsReplacer(L, test=True)

    def _(fun: dolfinx.fem.Function) -> typing.Union[petsc4py.PETSc.ScalarType, petsc4py.PETSc.RealType]:
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
        L_replacement_cpp.replace(test=fun)
        return _extract_part(
            L_replacement_cpp.comm.allreduce(
                dolfinx.fem.assemble_scalar(L_replacement_cpp.form_cpp), op=mpi4py.MPI.SUM),
            part)

    return _


@linear_form_action.register(list)
def _(L: typing.List[ufl.Form], part: typing.Optional[str] = None) -> typing.List[typing.Callable]:
    """
    Return a callable that represents the action of a block linear form on a function.

    Parameters
    ----------
    L : typing.List[ufl.Form]
        Block linear form to be represented.
    part : typing.Optional[str]
        Optional part (real or complex) to extract from the action result.
        If not provided, no postprocessing of the result will be carried out.

    Returns
    -------
    typing.List[typing.Callable]
        A list of callables that represents the action of L on a function.
    """
    return [linear_form_action(L_i) for L_i in L]


@functools.singledispatch
def bilinear_form_action(a: ufl.Form, part: typing.Optional[str] = None) -> typing.Callable:
    """
    Return a callable that represents the action of a bilinear form on a pair of functions.

    Parameters
    ----------
    a : ufl.Form
        Bilinear form to be represented.
    part : typing.Optional[str]
        Optional part (real or complex) to extract from the action result.
        If not provided, no postprocessing of the result will be carried out.

    Returns
    -------
    typing.Callable
        A callable that represents the action of a on a pair of functions.
    """
    a_replacement_cpp = FormArgumentsReplacer(a, test=True, trial=True)

    def _trial_action(fun_1: dolfinx.fem.Function) -> typing.Callable:
        """
        Compute the action of a bilinear form on a function, to be replaced to the trial function.

        Parameters
        ----------
        fun_1 : dolfinx.fem.Function
            Function to be replaced to the trial function.

        Returns
        -------
        typing.Callable
            A callable that represents action of a bilinear form on a function, to be replaced to the trial function.
        """
        a_replacement_cpp.replace(trial=fun_1)

        def _test_action(fun_0: dolfinx.fem.Function) -> typing.Union[
                petsc4py.PETSc.ScalarType, petsc4py.PETSc.RealType]:
            """
            Compute the action of a bilinear form on a pair of functions.

            Parameters
            ----------
            fun_0 : dolfinx.fem.Function
                Function to be replaced to the test function.

            Returns
            -------
            petsc4py.PETSc.ScalarType
                Evaluation of the action of a on the provided pair of functions.
            """
            a_replacement_cpp.replace(test=fun_0)
            return _extract_part(
                a_replacement_cpp.comm.allreduce(
                    dolfinx.fem.assemble_scalar(a_replacement_cpp.form_cpp), op=mpi4py.MPI.SUM),
                part)

        return _test_action

    return _trial_action


@bilinear_form_action.register(list)
def _(
    a: typing.Union[typing.List[ufl.Form], typing.List[typing.List[ufl.Form]]], part: typing.Optional[str] = None
) -> typing.Union[typing.List[typing.Callable], typing.List[typing.List[typing.Callable]]]:
    """
    Return a callable that represents the action of a block bilinear form on a pair of functions.

    Parameters
    ----------
    a : typing.Union[typing.List[ufl.Form], typing.List[typing.List[ufl.Form]]]
        Block bilinear form to be represented.
    part : typing.Optional[str]
        Optional part (real or complex) to extract from the action result.
        If not provided, no postprocessing of the result will be carried out.

    Returns
    -------
    typing.Union[typing.List[typing.Callable], typing.List[typing.List[typing.Callable]]]
        A list or a matrix of callables that represents the action of a on a pair of functions.
    """
    if isinstance(a[0], list):
        return [[bilinear_form_action(a_ij) for a_ij in a_i] for a_i in a]
    else:
        assert isinstance(a[0], ufl.Form)
        return [bilinear_form_action(a_ii) for a_ii in a]


def _extract_part(value: petsc4py.PETSc.ScalarType, part: typing.Optional[str]) -> typing.Union[
        petsc4py.PETSc.ScalarType, petsc4py.PETSc.RealType]:  # pragma: no cover
    if np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating):
        if part == "real":
            return value.real
        elif part == "imag":
            return value.imag
        else:
            assert part is None
            return value
    else:
        assert part in ("real", None)
        return value
