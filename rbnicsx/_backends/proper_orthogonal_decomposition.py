# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal backend to compute the proper orthogonal decomposition."""

import typing

import numpy as np
import numpy.typing as npt
import petsc4py.PETSc
import slepc4py.SLEPc

from rbnicsx._backends.functions_list import Function, FunctionsList
from rbnicsx._backends.online_tensors import create_online_matrix, create_online_vector
from rbnicsx._backends.tensors_list import TensorsList
from rbnicsx._cpp import cpp_library

real_zero = petsc4py.PETSc.RealType(0.0)  # type: ignore[attr-defined]


def proper_orthogonal_decomposition_functions(
    functions_list: FunctionsList[Function],
    compute_inner_product: typing.Callable[  # type: ignore[name-defined]
        [Function], typing.Callable[[Function], petsc4py.PETSc.RealType]
    ],
    scale: typing.Callable[[Function, petsc4py.PETSc.RealType], None],  # type: ignore[name-defined]
    N: int = -1, tol: petsc4py.PETSc.RealType = real_zero, normalize: bool = True  # type: ignore[name-defined]
) -> tuple[  # type: ignore[name-defined]
    npt.NDArray[petsc4py.PETSc.RealType], FunctionsList[Function], list[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of snapshots.

    Parameters
    ----------
    functions_list
        Collected snapshots.
    compute_inner_product
        A callable x(u)(v) to compute the action of the inner product on the trial function u and test function v.
        The resulting modes will be orthonormal w.r.t. this inner product.
    scale
        A callable with signature scale(function, factor) to scale any function by a given factor.
    N
        Maximum number of modes to be computed. If not provided, it will be set to the number of collected snapshots.
    tol
        Tolerance on the retained energy. If not provided, it will be set to zero.
    normalize
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    :
        A tuple containing:
            1. Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
            2. Retained modes from the snapshots. Only the first few modes are returned, till either the
               maximum number N is reached or the tolerance on the retained energy is fulfilled.
            3. Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
               either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    eigenvalues, modes, eigenvectors = _solve_eigenvalue_problem(
        functions_list, compute_inner_product, scale, N, tol, normalize)
    modes_wrapped = functions_list.duplicate()
    modes_wrapped.extend(modes)
    return eigenvalues, modes_wrapped, eigenvectors


def proper_orthogonal_decomposition_functions_block(
    functions_lists: typing.Sequence[FunctionsList[Function]],
    compute_inner_products: typing.Sequence[  # type: ignore[name-defined]
        typing.Callable[[Function], typing.Callable[[Function], petsc4py.PETSc.RealType]]
    ],
    scale: typing.Callable[[Function, petsc4py.PETSc.RealType], None],  # type: ignore[name-defined]
    N: typing.Union[int, list[int]] = -1,
    tol: typing.Union[petsc4py.PETSc.RealType, list[petsc4py.PETSc.RealType]] = real_zero,  # type: ignore[name-defined]
    normalize: bool = True
) -> tuple[  # type: ignore[name-defined]
    list[npt.NDArray[petsc4py.PETSc.RealType]], list[FunctionsList[Function]], list[list[petsc4py.PETSc.Vec]]
]:
    """
    Compute the proper orthogonal decomposition of a set of snapshots, where each snapshot is made of several blocks.

    Parameters
    ----------
    functions_lists
        Collected snapshots. Each snapshot is made of several blocks, defined on possibly different function spaces.
        The inner FunctionsList contains all snapshots of a single block, while the outer list collects the different
        blocks.
    compute_inner_products
        A list of callables x_i(u_i)(v_i) to compute the action of the inner product on the trial function u_i
        and test function v_i associated to the i-th block.
        The resulting modes will be orthonormal w.r.t. this inner product.
    scale
        A callable with signature scale(function, factor) to scale any function by a given factor.
    N
        Maximum number of modes to be computed. If an integer value is passed then the same maximum number is
        used for each block. To set a different maximum number of modes for each block pass a list of integers.
        If not provided, it will be set to the number of collected snapshots.
    tol
        Tolerance on the retained energy. If a floating point value is passed then the same tolerance is
        used for each block. To set a different tolerance for each block pass a list of floating point numbers.
        If not provided, it will be set to zero.
    normalize
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    :
        A tuple containing:
            1. Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
               The outer list collects the eigenvalues of different blocks.
            2. Retained modes from the snapshots. Only the first few modes are returned, till either the
               maximum number N is reached or the tolerance on the retained energy is fulfilled.
               The outer list collects the retained modes of different blocks.
            3. Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
               either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
               The outer list collects the eigenvectors of different blocks.
    """
    assert len(compute_inner_products) == len(functions_lists)
    if isinstance(N, list):
        assert len(N) == len(functions_lists)
    else:
        N = [N for _ in functions_lists]
    if isinstance(tol, list):
        assert len(tol) == len(functions_lists)
    else:
        tol = [tol for _ in functions_lists]

    eigenvalues, modes, eigenvectors = list(), list(), list()
    for (functions_list, compute_inner_product, N_, tol_) in zip(functions_lists, compute_inner_products, N, tol):
        eigenvalues_, modes_, eigenvectors_ = proper_orthogonal_decomposition_functions(
            functions_list, compute_inner_product, scale, N_, tol_, normalize)
        eigenvalues.append(eigenvalues_)
        modes.append(modes_)
        eigenvectors.append(eigenvectors_)
    return eigenvalues, modes, eigenvectors


def proper_orthogonal_decomposition_tensors(
    tensors_list: TensorsList, N: int = -1,
    tol: petsc4py.PETSc.RealType = real_zero, normalize: bool = True  # type: ignore[name-defined]
) -> tuple[  # type: ignore[name-defined]
    npt.NDArray[petsc4py.PETSc.RealType], TensorsList, list[petsc4py.PETSc.Vec]
]:
    """
    Compute the proper orthogonal decomposition of a set of tensors.

    Parameters
    ----------
    tensors_list
        Collected tensors.
    N
        Maximum number of modes to be computed. If not provided, it will be set to the number of collected tensors.
    tol
        Tolerance on the retained energy. If not provided, it will be set to zero.
    normalize
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    :
        A tuple containing:
            1. Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
            2. Retained modes from the tensors. Only the first few modes are returned, till either the
               maximum number N is reached or the tolerance on the retained energy is fulfilled.
            3. Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
               either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    assert tensors_list.type in ("Mat", "Vec")
    if tensors_list.type == "Mat":
        def compute_inner_product(
            tensor_j: petsc4py.PETSc.Mat  # type: ignore[name-defined]
        ) -> typing.Callable[[petsc4py.PETSc.Mat], petsc4py.PETSc.RealType]:  # type: ignore[name-defined]
            def _(tensor_i: petsc4py.PETSc.Mat) -> petsc4py.PETSc.RealType:  # type: ignore[name-defined]
                return cpp_library._backends.frobenius_inner_product(tensor_i, tensor_j)

            return _

        def scale(
            tensor: petsc4py.PETSc.Mat, factor: petsc4py.PETSc.RealType  # type: ignore[name-defined]
        ) -> None:
            tensor *= factor
    elif tensors_list.type == "Vec":
        def compute_inner_product(
            tensor_j: petsc4py.PETSc.Vec  # type: ignore[name-defined]
        ) -> typing.Callable[[petsc4py.PETSc.Vec], petsc4py.PETSc.RealType]:  # type: ignore[name-defined]
            def _(tensor_i: petsc4py.PETSc.Vec) -> petsc4py.PETSc.RealType:  # type: ignore[name-defined]
                return tensor_i.dot(tensor_j)

            return _

        def scale(
            tensor: petsc4py.PETSc.Vec, factor: petsc4py.PETSc.RealType  # type: ignore[name-defined]
        ) -> None:
            with tensor.localForm() as tensor_local:
                tensor_local *= factor

    eigenvalues, modes, eigenvectors = _solve_eigenvalue_problem(
        tensors_list, compute_inner_product, scale, N, tol, normalize)
    modes_wrapped = tensors_list.duplicate()
    modes_wrapped.extend(modes)
    return eigenvalues, modes_wrapped, eigenvectors


def _solve_eigenvalue_problem(
    snapshots: typing.Union[FunctionsList[Function], TensorsList],
    compute_inner_product: typing.Union[  # type: ignore[name-defined]
        typing.Callable[[Function], typing.Callable[[Function], petsc4py.PETSc.RealType]],
        typing.Callable[[petsc4py.PETSc.Mat], typing.Callable[[petsc4py.PETSc.Mat], petsc4py.PETSc.RealType]],
        typing.Callable[[petsc4py.PETSc.Vec], typing.Callable[[petsc4py.PETSc.Vec], petsc4py.PETSc.RealType]]
    ],
    scale: typing.Union[  # type: ignore[name-defined]
        typing.Callable[[Function, petsc4py.PETSc.RealType], None],
        typing.Callable[[petsc4py.PETSc.Mat, petsc4py.PETSc.RealType], None],
        typing.Callable[[petsc4py.PETSc.Vec, petsc4py.PETSc.RealType], None],
    ],
    N: int, tol: petsc4py.PETSc.RealType, normalize: bool  # type: ignore[name-defined]
) -> tuple[  # type: ignore[name-defined]
    npt.NDArray[petsc4py.PETSc.RealType],
    typing.Union[
        list[Function], list[petsc4py.PETSc.Mat], list[petsc4py.PETSc.Vec]],
    list[petsc4py.PETSc.Vec]
]:
    """
    Solve the eigenvalue problem for the correlation matrix.

    Parameters
    ----------
    snapshots
        Collected snapshots.
    compute_inner_product
        A function that computes the inner product between two snapshots.
    scale
        A function that rescales a snapshot in place.
    N
        Maximum number of eigenvectors to be returned.
    tol
        Tolerance on the retained energy.
    normalize
        If true (default), the modes are scaled to unit norm.

    Returns
    -------
    :
        A tuple containing:
            1. Eigenvalues of the correlation matrix, largest first. All computed eigenvalues are returned.
            2. Retained modes from the snapshots. Only the first few modes are returned, till either the
               maximum number N is reached or the tolerance on the retained energy is fulfilled.
            3. Eigenvectors of the correlation matrix. Only the first few eigenvectors are returned, till
               either the maximum number N is reached or the tolerance on the retained energy is fulfilled.
    """
    assert N > 0 or N == -1
    if N == -1:
        N = len(snapshots)

    correlation_matrix = create_online_matrix(len(snapshots), len(snapshots))
    for (j, snapshot_j) in enumerate(snapshots):
        compute_inner_product_partial_j = compute_inner_product(snapshot_j)
        for (i, snapshot_i) in enumerate(snapshots):
            correlation_matrix[i, j] = compute_inner_product_partial_j(snapshot_i)
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
        retained_energy = np.array([retained_energy_n / total_energy for retained_energy_n in retained_energy])
    else:
        retained_energy = np.ones(eps.getConverged())  # trivial case, all snapshots are zero

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
            norm_n = np.sqrt(compute_inner_product(mode_n)(mode_n))
            if norm_n != 0.0:
                scale(mode_n, 1.0 / norm_n)
        modes.append(mode_n)

    return np.array(eigenvalues), modes, eigenvectors
