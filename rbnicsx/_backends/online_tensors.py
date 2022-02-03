# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online tensor data structures using PETSc."""

from __future__ import annotations

import contextlib
import types
import typing

import mpi4py
import numpy as np
import petsc4py


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


class VecSubVectorWrapper(object):
    """
    Wrap calls to petsc4py.PETSc.Vec.{getSubVector,restoreSubVector} in a context manager.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        An online vector.
    indices : np.typing.NDArray[np.int32]
        Indices to be extracted.
    """

    def __init__(self, b: petsc4py.PETSc.Vec, indices: np.typing.NDArray[np.int32]) -> None:
        self._b = b
        self._index_set = petsc4py.PETSc.IS().createGeneral(indices, comm=b.comm)
        self._b_sub = None

    def __enter__(self) -> petsc4py.PETSc.Vec:
        """Get subvector on context enter."""
        self._b_sub = self._b.getSubVector(self._index_set)
        return self._b_sub

    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore subvector and clean up index set upon leaving the context."""
        self._b.restoreSubVector(self._index_set, self._b_sub)
        del self._b_sub
        self._index_set.destroy()
        del self._index_set


class VecSubVectorCopier(object):
    """
    A context manager that create copies of a subvector. Caller should de-allocate the returned vector.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        An online vector.
    indices : np.typing.NDArray[np.int32]
        Indices to be extracted.
    """

    def __init__(self, b: petsc4py.PETSc.Vec, indices: np.typing.NDArray[np.int32]) -> None:
        self._b = b
        self._indices = indices

    def __enter__(self) -> petsc4py.PETSc.Vec:
        """Get a copy of the subvector on context enter."""
        b_sub_copy = create_online_vector(len(self._indices))
        b_sub_copy[:] = self._b[self._indices]
        return b_sub_copy

    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Do nothing upon exit."""
        pass


def BlockVecSubVectorContextManager(VecSubVectorContextManager: typing.ContextManager) -> typing.ContextManager:
    """Apply VecSubVectorContextManager to every block of a block vector."""

    class BlockVecSubVectorContextManager(object):
        """
        Apply VecSubVectorContextManager to every block of a block vector.

        Parameters
        ----------
        b : petsc4py.PETSc.Vec
            An online vector.
        N : typing.List[int]
            Dimension of the blocks of the vector.
        """

        def __init__(self, b: petsc4py.PETSc.Vec, N: typing.List[int]) -> None:
            self._b = b
            blocks = np.hstack((0, np.cumsum([N_ for N_ in N])))
            self._indices = [np.arange(*blocks[i:i + 2], dtype=np.int32) for i in range(len(N))]

        def __iter__(self) -> typing.Iterable[petsc4py.PETSc.Vec]:
            """Iterate over blocks."""
            with contextlib.ExitStack() as wrapper_stack:
                for indices_ in self._indices:
                    wrapper = VecSubVectorContextManager(self._b, indices_)
                    yield wrapper_stack.enter_context(wrapper)

        def __enter__(self) -> BlockVecSubVectorContextManager:
            """Return this context manager."""
            return self

        def __exit__(
            self, exception_type: typing.Type[BaseException], exception_value: BaseException,
            traceback: types.TracebackType
        ) -> None:
            """Do nothing upon exit."""
            pass

    return BlockVecSubVectorContextManager


class BlockVecSubVectorWrapper(BlockVecSubVectorContextManager(VecSubVectorWrapper)):
    """
    Wrap an online vector with multiple blocks and iterate over each block.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        An online vector.
    N : typing.List[int]
        Dimension of the blocks of the vector.
    """

    pass


class BlockVecSubVectorCopier(BlockVecSubVectorContextManager(VecSubVectorCopier)):
    """
    Copy an online vector with multiple blocks while iterating over each block.

    Parameters
    ----------
    b : petsc4py.PETSc.Vec
        An online vector.
    N : typing.List[int]
        Dimension of the blocks of the vector.
    """

    pass


class MatSubMatrixWrapper(object):
    """
    Wrap calls to petsc4py.PETSc.Mat.{getLocalSubMatrix,restoreLocalSubMatrix} in a context manager.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        An online matrix.
    row_indices, col_indices : np.typing.NDArray[np.int32]
        A 2-tuple containing the indices to be extracted for each dimension.
    """

    def __init__(
        self, A: petsc4py.PETSc.Mat, row_indices: np.typing.NDArray[np.int32], col_indices: np.typing.NDArray[np.int32]
    ) -> None:
        self._A = A
        self._index_sets = (
            petsc4py.PETSc.IS().createGeneral(row_indices, comm=A.comm),
            petsc4py.PETSc.IS().createGeneral(col_indices, comm=A.comm)
        )
        self._A_sub = None

    def __enter__(self) -> petsc4py.PETSc.Mat:
        """Get submatrix on context enter."""
        self._A_sub = self._A.getLocalSubMatrix(*self._index_sets)
        return self._A_sub

    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Restore submatrix and clean up index sets upon leaving the context."""
        self._A.restoreLocalSubMatrix(*self._index_sets, self._A_sub)
        del self._A_sub
        [index_set.destroy() for index_set in self._index_sets]
        del self._index_sets


class MatSubMatrixCopier(object):
    """
    A context manager that create copies of a submatrix. Caller should de-allocate the returned matrix.

    Parameters
    ----------
    b : petsc4py.PETSc.Mat
        An online matrix.
    row_indices, col_indices : np.typing.NDArray[np.int32]
        A 2-tuple containing the indices to be extracted for each dimension.
    """

    def __init__(
        self, A: petsc4py.PETSc.Mat, row_indices: np.typing.NDArray[np.int32], col_indices: np.typing.NDArray[np.int32]
    ) -> None:
        self._A = A
        self._row_indices = row_indices
        self._col_indices = col_indices

    def __enter__(self) -> petsc4py.PETSc.Mat:
        """Get a copy of the submatrix on context enter."""
        A_sub_copy = create_online_matrix(len(self._row_indices), len(self._col_indices))
        A_sub_copy[:, :] = self._A[self._row_indices, self._col_indices]
        A_sub_copy.assemble()
        return A_sub_copy

    def __exit__(
        self, exception_type: typing.Type[BaseException], exception_value: BaseException,
        traceback: types.TracebackType
    ) -> None:
        """Do nothing upon exit."""
        pass


def BlockMatSubMatrixContextManager(MatSubMatrixContextManager: typing.ContextManager) -> typing.ContextManager:
    """Apply MatSubMatrixContextManager to every block of a block matrix."""

    class BlockMatSubMatrixContextManager(object):
        """
        Apply MatSubMatrixContextManager to every block of a block matrix.

        Parameters
        ----------
        A : petsc4py.PETSc.Mat
            An online matrix.
        M, N : typing.List[int]
            Dimension of the blocks of the matrix.
        """

        def __init__(self, A: petsc4py.PETSc.Mat, M: typing.List[int], N: typing.List[int]) -> None:
            self._A = A
            row_blocks = np.hstack((0, np.cumsum([M_ for M_ in M])))
            col_blocks = np.hstack((0, np.cumsum([N_ for N_ in N])))
            self._row_indices = [np.arange(*row_blocks[i:i + 2], dtype=np.int32) for i in range(len(M))]
            self._col_indices = [np.arange(*col_blocks[j:j + 2], dtype=np.int32) for j in range(len(N))]

        def __iter__(self) -> typing.Iterable[typing.Tuple[int, int, petsc4py.PETSc.Mat]]:
            """Iterate over blocks."""
            with contextlib.ExitStack() as wrapper_stack:
                for (I, row_indices_) in enumerate(self._row_indices):
                    for (J, col_indices_) in enumerate(self._col_indices):
                        wrapper = MatSubMatrixContextManager(self._A, row_indices_, col_indices_)
                        yield (I, J, wrapper_stack.enter_context(wrapper))

        def __enter__(self) -> BlockMatSubMatrixContextManager:
            """Return this context manager."""
            return self

        def __exit__(
            self, exception_type: typing.Type[BaseException], exception_value: BaseException,
            traceback: types.TracebackType
        ) -> None:
            """Do nothing upon exit."""
            pass

    return BlockMatSubMatrixContextManager


class BlockMatSubMatrixWrapper(BlockMatSubMatrixContextManager(MatSubMatrixWrapper)):
    """
    Wrap an online matrix with multiple blocks and iterate over each block.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        An online matrix.
    M, N : typing.List[int]
        Dimension of the blocks of the matrix.
    """

    pass


class BlockMatSubMatrixCopier(BlockMatSubMatrixContextManager(MatSubMatrixCopier)):
    """
    Copy an online matrix with multiple blocks while iterating over each block.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        An online matrix.
    M, N : typing.List[int]
        Dimension of the blocks of the matrix.
    """

    pass
