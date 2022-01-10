# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend to compute the proper orthogonal decomposition."""

import functools
import typing

import dolfinx.fem
import mpi4py
import numpy as np
import petsc4py
import slepc4py
import ufl

from minirox.backends.functions_list import FunctionsList
from minirox.backends.projection import create_online_matrix, create_online_vector
from minirox.backends.tensors_list import TensorsList
from minirox.cpp import cpp_library


@functools.singledispatch
def proper_orthogonal_decomposition(
    snapshots: typing.Iterable, N: int, tol: float, normalize: bool = True
) -> typing.Tuple[
    np.typing.NDArray[float], typing.Iterable, typing.List[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of snapshots or tensors.

    Please the dispatched implementation for more details.
    """
    raise RuntimeError("Please run the dispatched implementation.")


@proper_orthogonal_decomposition.register
def _(
    functions_list: FunctionsList, inner_product: ufl.Form, N: int, tol: float, normalize: bool = True
) -> typing.Tuple[
    np.typing.NDArray[float], FunctionsList, typing.List[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of snapshots.

    Parameters
    ----------
    functions_list : minirox.backends.FunctionsList
        Collected snapshots.
    inner_product : ufl.Form
        Bilinear form which defines the inner product. The resulting modes will be orthonormal
        w.r.t. this inner product.
    N : int
        Maximum number of modes to be computed.
    tol : float
        Tolerance on the retained energy.
    normalize : bool, optional
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    np.typing.NDArray[float]
        Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
    minirox.backends.FunctionsList
        Retained modes from the snapshots. Only the first few modes are returned, till either the
        maximum number N is reached or the tolerance on the retained energy is fulfilled.
    typing.List[petsc4py.PETSc.Vec]
        Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
        either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    comm = functions_list.function_space.mesh.comm
    test, trial = inner_product.arguments()

    def compute_inner_product(function_i: dolfinx.fem.Function, function_j: dolfinx.fem.Function) -> float:
        return comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(ufl.replace(inner_product, {test: function_i, trial: function_j}))),
            op=mpi4py.MPI.SUM)

    def scale(function: dolfinx.fem.Function, factor: float) -> None:
        with function.vector.localForm() as function_local:
            function_local *= factor

    eigenvalues, modes, eigenvectors = _solve_eigenvalue_problem(
        functions_list, compute_inner_product, scale, N, tol, normalize)
    modes_wrapped = FunctionsList(functions_list.function_space)
    modes_wrapped.extend(modes)
    return eigenvalues, modes_wrapped, eigenvectors


def proper_orthogonal_decomposition_block(
    functions_lists: FunctionsList, inner_products: typing.List[ufl.Form], N: typing.Union[int, typing.List[int]],
    tol: typing.Union[float, typing.List[float]], normalize: bool = True
) -> typing.Tuple[
    typing.List[np.typing.NDArray[float]], typing.List[FunctionsList], typing.List[typing.List[petsc4py.PETSc.Vec]]
]:
    """
    Compute the proper orthogonal decomposition of a set of snapshots, where each snapshot is made of several blocks.

    Parameters
    ----------
    functions_lists : typing.List[minirox.backends.FunctionsList]
        Collected snapshots. Each snapshot is made of several blocks, defined on possibly different function spaces.
        The inner FunctionsList contains all snapshots of a single block, while the outer list collects the different
        blocks.
    inner_products : typing.List[ufl.Form]
        Bilinear forms which define the inner products of each block. The resulting modes
        will be orthonormal w.r.t. these inner products.
    N : typing.Union[int, typing.List[int]]
        Maximum number of modes to be computed. If an integer value is passed then the same maximum number is
        used for each block. To set a different maximum number of modes for each block pass a list of integers.
    tol : float
        Tolerance on the retained energy. If a floating point value is passed then the same tolerance is
        used for each block. To set a different tolerance for each block pass a list of floating point numbers.
    normalize : bool, optional
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    typing.List[np.typing.NDArray[float]]
        Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
        The outer list collects the eigenvalues of different blocks.
    typing.List[minirox.backends.FunctionsList]
        Retained modes from the snapshots. Only the first few modes are returned, till either the
        maximum number N is reached or the tolerance on the retained energy is fulfilled.
        The outer list collects the retained modes of different blocks.
    typing.List[typing.List[petsc4py.PETSc.Vec]]
        Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
        either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
        The outer list collects the eigenvectors of different blocks.
    """
    assert len(inner_products) == len(functions_lists)
    if isinstance(N, list):
        assert len(N) == len(functions_lists)
    else:
        N = [N for _ in functions_lists]
    if isinstance(tol, list):
        assert len(tol) == len(functions_lists)
    else:
        tol = [tol for _ in functions_lists]

    eigenvalues, modes, eigenvectors = list(), list(), list()
    for (functions_list, inner_product, N_, tol_) in zip(functions_lists, inner_products, N, tol):
        eigenvalues_, modes_, eigenvectors_ = proper_orthogonal_decomposition(
            functions_list, inner_product, N_, tol_, normalize)
        eigenvalues.append(eigenvalues_)
        modes.append(modes_)
        eigenvectors.append(eigenvectors_)
    return eigenvalues, modes, eigenvectors


@proper_orthogonal_decomposition.register
def _(
    tensors_list: TensorsList, N: int, tol: float, normalize: bool = True
) -> typing.Tuple[
    np.typing.NDArray[float], TensorsList, typing.List[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of tensors.

    Parameters
    ----------
    tensors_list : minirox.backends.TensorsList
        Collected tensors.
    N : int
        Maximum number of modes to be computed.
    tol : float
        Tolerance on the retained energy.
    normalize : bool, optional
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    np.typing.NDArray[float]
        Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
    minirox.backends.TensorsList
        Retained modes from the tensors. Only the first few modes are returned, till either the
        maximum number N is reached or the tolerance on the retained energy is fulfilled.
    typing.List[petsc4py.PETSc.Vec]
        Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
        either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    assert tensors_list.type in ("Mat", "Vec")
    if tensors_list.type == "Mat":
        def compute_inner_product(tensor_i: petsc4py.PETSc.Mat, tensor_j: petsc4py.PETSc.Mat) -> float:
            return cpp_library.backends.frobenius_inner_product(tensor_i, tensor_j)

        def scale(tensor: petsc4py.PETSc.Mat, factor: float) -> None:
            tensor *= factor
    elif tensors_list.type == "Vec":
        def compute_inner_product(tensor_i: petsc4py.PETSc.Vec, tensor_j: petsc4py.PETSc.Vec) -> float:
            return tensor_i.dot(tensor_j)

        def scale(tensor: petsc4py.PETSc.Vec, factor: float) -> None:
            with tensor.localForm() as tensor_local:
                tensor_local *= factor

    eigenvalues, modes, eigenvectors = _solve_eigenvalue_problem(
        tensors_list, compute_inner_product, scale, N, tol, normalize)
    modes_wrapped = TensorsList(tensors_list.form, tensors_list.comm)
    modes_wrapped.extend(modes)
    return eigenvalues, modes_wrapped, eigenvectors


def _solve_eigenvalue_problem(
    snapshots: typing.Union[FunctionsList, TensorsList], compute_inner_product: typing.Callable,
    scale: typing.Callable, N: int, tol: float, normalize: bool
) -> typing.Tuple[
    np.typing.NDArray[float],
    typing.Union[
        typing.List[dolfinx.fem.Function], typing.List[petsc4py.PETSc.Mat], typing.List[petsc4py.PETSc.Vec]
    ],
    typing.List[petsc4py.PETSc.Vec]
]:
    """
    Solve the eigenvalue problem for the correlation matrix.

    Parameters
    ----------
    snapshots : typing.Union[minirox.backends.FunctionsList, minirox.backends.TensorsList]
        Collected snapshots.
    compute_inner_product : typing.Callable
        A function that computes the inner product between two snapshots.
    scale : typing.Callable
        A function that rescales a snapshot in place.
    N : int
        Maximum number of eigenvectors to be returned.
    tol : float
        Tolerance on the retained energy.
    normalize : bool, optional
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    np.typing.NDArray[float]
        Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
    typing.Union[
        typing.List[dolfinx.fem.Function], typing.List[petsc4py.PETSc.Mat], typing.List[petsc4py.PETSc.Vec]
    ]
        Retained modes from the snapshots. Only the first few modes are returned, till either the
        maximum number N is reached or the tolerance on the retained energy is fulfilled.
    typing.List[petsc4py.PETSc.Vec]
        Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
        either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    correlation_matrix = create_online_matrix(len(snapshots), len(snapshots))
    for (i, snapshot_i) in enumerate(snapshots):
        for (j, snapshot_j) in enumerate(snapshots):
            correlation_matrix[i, j] = compute_inner_product(snapshot_i, snapshot_j)
    correlation_matrix.assemble()

    eps = slepc4py.SLEPc.EPS().create(correlation_matrix.comm)
    eps.setType(slepc4py.SLEPc.EPS.Type.LAPACK)
    eps.setProblemType(slepc4py.SLEPc.EPS.ProblemType.HEP)
    eps.setWhichEigenpairs(slepc4py.SLEPc.EPS.Which.LARGEST_REAL)
    eps.setFromOptions()
    eps.setOperators(correlation_matrix)
    eps.solve()

    eigenvalues = list()
    for n in range(eps.getConverged()):
        eigenvalue_n = eps.getEigenvalue(n)
        assert np.isclose(eigenvalue_n.imag, 0.0)
        eigenvalues.append(eigenvalue_n.real)

    total_energy = sum([abs(e) for e in eigenvalues])
    retained_energy = np.cumsum([abs(e) for e in eigenvalues])
    if total_energy > 0.0:
        retained_energy = [retained_energy_n / total_energy for retained_energy_n in retained_energy]
    else:
        retained_energy = [1.0 for _ in range(eps.getConverged())]  # trivial case, all snapshots are zero

    N = min(N, eps.getConverged())
    eigenvectors = list()
    for n in range(N):
        eigenvector_n = create_online_vector(correlation_matrix.size[0])
        eps.getEigenvector(n, eigenvector_n)
        eigenvectors.append(eigenvector_n)
        if tol > 0.0 and retained_energy[n] > 1.0 - tol:
            break

    modes = list()
    for eigenvector_n in eigenvectors:
        mode_n = snapshots * eigenvector_n
        if normalize:
            norm_n = np.sqrt(compute_inner_product(mode_n, mode_n))
            if norm_n != 0.0:
                scale(mode_n, 1.0 / norm_n)
        modes.append(mode_n)

    return np.array(eigenvalues), modes, eigenvectors
