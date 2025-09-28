# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Online tensor data structures using PETSc."""

import contextlib
import sys
import types
import typing

import mpi4py.MPI
import numpy as np
import numpy.typing as npt
import petsc4py.PETSc

if sys.version_info >= (3, 11):  # pragma: no cover
    import typing as typing_extensions
else:  # pragma: no cover
    import typing_extensions


def create_online_vector(N: int) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
    """
    Create an online vector of the given dimension.

    Parameters
    ----------
    N
        Dimension of the vector.

    Returns
    -------
    :
        Allocated online vector.
    """
    vec = petsc4py.PETSc.Vec().createSeq(N, comm=mpi4py.MPI.COMM_SELF)  # type: ignore[attr-defined]
    # Attach the identity local-to-global map
    lgmap = petsc4py.PETSc.LGMap().create(np.arange(N, dtype=np.int32), comm=vec.comm)  # type: ignore[attr-defined]
    vec.setLGMap(lgmap)
    lgmap.destroy()
    # Setup and return
    vec.setUp()
    return vec


def create_online_vector_block(N: list[int]) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
    """
    Create an online vector of the given block dimensions.

    Parameters
    ----------
    N
        Dimension of the blocks of the vector.

    Returns
    -------
    :
        Allocated online vector.
    """
    return create_online_vector(sum(N))


def create_online_matrix(M: int, N: int) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
    """
    Create an online matrix of the given dimension.

    Parameters
    ----------
    M, N
        Dimension of the matrix.

    Returns
    -------
    :
        Allocated online matrix.
    """
    mat = petsc4py.PETSc.Mat().createDense((M, N), comm=mpi4py.MPI.COMM_SELF)  # type: ignore[attr-defined]
    # Attach the identity local-to-global map
    row_lgmap = petsc4py.PETSc.LGMap().create(np.arange(M, dtype=np.int32), comm=mat.comm)  # type: ignore[attr-defined]
    col_lgmap = petsc4py.PETSc.LGMap().create(np.arange(N, dtype=np.int32), comm=mat.comm)  # type: ignore[attr-defined]
    mat.setLGMap(row_lgmap, col_lgmap)
    row_lgmap.destroy()
    col_lgmap.destroy()
    # Setup and return
    mat.setUp()
    return mat


def create_online_matrix_block(
    M: list[int], N: list[int]
) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
    """
    Create an online matrix of the given block dimensions.

    Parameters
    ----------
    M, N
        Dimension of the blocks of the matrix.

    Returns
    -------
    :
        Allocated online matrix.
    """
    return create_online_matrix(sum(M), sum(N))


class VecSubVectorWrapper(typing.ContextManager[petsc4py.PETSc.Vec]):  # type: ignore[name-defined]
    """
    Wrap calls to petsc4py.PETSc.Vec.{getSubVector,restoreSubVector} in a context manager.

    Parameters
    ----------
    b
        An online vector.
    indices
        Indices to be extracted.
    """

    def __init__(
        self, b: petsc4py.PETSc.Vec, indices: npt.NDArray[np.int32]  # type: ignore[name-defined]
    ) -> None:
        self._b = b
        self._index_set = petsc4py.PETSc.IS().createGeneral(indices, comm=b.comm)  # type: ignore[attr-defined]
        self._b_sub = None

    def __enter__(self) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
        """Get subvector on context enter."""
        self._b_sub = self._b.getSubVector(self._index_set)
        return self._b_sub

    def __exit__(
        self, exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: types.TracebackType | None
    ) -> None:
        """Restore subvector and clean up index set upon leaving the context."""
        self._b.restoreSubVector(self._index_set, self._b_sub)
        del self._b_sub
        self._index_set.destroy()
        del self._index_set


class VecSubVectorCopier(typing.ContextManager[petsc4py.PETSc.Vec]):  # type: ignore[name-defined]
    """
    A context manager that create copies of a subvector. Caller should de-allocate the returned vector.

    Parameters
    ----------
    b
        An online vector.
    indices
        Indices to be extracted.
    """

    def __init__(
        self, b: petsc4py.PETSc.Vec, indices: npt.NDArray[np.int32]  # type: ignore[name-defined]
    ) -> None:
        self._b = b
        self._indices = indices

    def __enter__(self) -> petsc4py.PETSc.Vec:  # type: ignore[name-defined]
        """Get a copy of the subvector on context enter."""
        b_sub_copy = create_online_vector(len(self._indices))
        b_sub_copy[:] = self._b[self._indices]
        return b_sub_copy

    def __exit__(
        self, exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: types.TracebackType | None
    ) -> None:
        """Do nothing upon exit."""
        pass


def BlockVecSubVectorContextManager(
    VecSubVectorContextManager: type[VecSubVectorCopier] | type[VecSubVectorWrapper]
) -> type:
    """Apply VecSubVectorContextManager to every block of a block vector."""

    class BlockVecSubVectorContextManagerImpl(typing.ContextManager["BlockVecSubVectorContextManagerImpl"]):
        """
        Apply VecSubVectorContextManager to every block of a block vector.

        Parameters
        ----------
        b
            An online vector.
        N
            Dimension of the blocks of the vector.
        """

        def __init__(
            self: typing_extensions.Self, b: petsc4py.PETSc.Vec, N: list[int]  # type: ignore[name-defined]
        ) -> None:
            self._b = b
            blocks = np.hstack((0, np.cumsum([N_ for N_ in N]))).astype(np.int32)
            self._indices = [np.arange(blocks[i], blocks[i + 1], dtype=np.int32) for i in range(len(N))]

        def __iter__(self: typing_extensions.Self) -> typing.Iterator[  # type: ignore[name-defined]
                petsc4py.PETSc.Vec]:
            """Iterate over blocks."""
            with contextlib.ExitStack() as wrapper_stack:
                for indices_ in self._indices:
                    wrapper = VecSubVectorContextManager(self._b, indices_)
                    yield wrapper_stack.enter_context(wrapper)

        def __enter__(self: typing_extensions.Self) -> typing_extensions.Self:
            """Return this context manager."""
            return self

        def __exit__(
            self: typing_extensions.Self, exception_type: type[BaseException] | None,
            exception_value: BaseException | None,
            traceback: types.TracebackType | None
        ) -> None:
            """Do nothing upon exit."""
            pass

    return BlockVecSubVectorContextManagerImpl


class BlockVecSubVectorWrapper(BlockVecSubVectorContextManager(VecSubVectorWrapper)):  # type: ignore[misc]
    """
    Wrap an online vector with multiple blocks and iterate over each block.

    Parameters
    ----------
    b
        An online vector.
    N
        Dimension of the blocks of the vector.
    """

    pass


class BlockVecSubVectorCopier(BlockVecSubVectorContextManager(VecSubVectorCopier)):  # type: ignore[misc]
    """
    Copy an online vector with multiple blocks while iterating over each block.

    Parameters
    ----------
    b
        An online vector.
    N
        Dimension of the blocks of the vector.
    """

    pass


class MatSubMatrixWrapper(typing.ContextManager[petsc4py.PETSc.Mat]):  # type: ignore[name-defined]
    """
    Wrap calls to petsc4py.PETSc.Mat.{getLocalSubMatrix,restoreLocalSubMatrix} in a context manager.

    Parameters
    ----------
    A
        An online matrix.
    row_indices, col_indices
        A 2-tuple containing the indices to be extracted for each dimension.
    """

    def __init__(
        self, A: petsc4py.PETSc.Mat,  # type: ignore[name-defined]
        row_indices: npt.NDArray[np.int32], col_indices: npt.NDArray[np.int32]
    ) -> None:
        self._A = A
        self._index_sets = (
            petsc4py.PETSc.IS().createGeneral(row_indices, comm=A.comm),  # type: ignore[attr-defined]
            petsc4py.PETSc.IS().createGeneral(col_indices, comm=A.comm)  # type: ignore[attr-defined]
        )
        self._A_sub = None

    def __enter__(self) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
        """Get submatrix on context enter."""
        self._A_sub = self._A.getLocalSubMatrix(*self._index_sets)
        return self._A_sub

    def __exit__(
        self, exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: types.TracebackType | None
    ) -> None:
        """Restore submatrix and clean up index sets upon leaving the context."""
        self._A.restoreLocalSubMatrix(*self._index_sets, self._A_sub)
        del self._A_sub
        [index_set.destroy() for index_set in self._index_sets]
        del self._index_sets


class MatSubMatrixCopier(typing.ContextManager[petsc4py.PETSc.Mat]):  # type: ignore[name-defined]
    """
    A context manager that create copies of a submatrix. Caller should de-allocate the returned matrix.

    Parameters
    ----------
    b
        An online matrix.
    row_indices, col_indices
        A 2-tuple containing the indices to be extracted for each dimension.
    """

    def __init__(
        self, A: petsc4py.PETSc.Mat,  # type: ignore[name-defined]
        row_indices: npt.NDArray[np.int32], col_indices: npt.NDArray[np.int32]
    ) -> None:
        self._A = A
        self._row_indices = row_indices
        self._col_indices = col_indices

    def __enter__(self) -> petsc4py.PETSc.Mat:  # type: ignore[name-defined]
        """Get a copy of the submatrix on context enter."""
        A_sub_copy = create_online_matrix(len(self._row_indices), len(self._col_indices))
        A_sub_copy[:, :] = self._A[self._row_indices, self._col_indices]
        A_sub_copy.assemble()
        return A_sub_copy

    def __exit__(
        self, exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: types.TracebackType | None
    ) -> None:
        """Do nothing upon exit."""
        pass


def BlockMatSubMatrixContextManager(
    MatSubMatrixContextManager: type[MatSubMatrixCopier] | type[MatSubMatrixWrapper]
) -> type:
    """Apply MatSubMatrixContextManager to every block of a block matrix."""

    class BlockMatSubMatrixContextManagerImpl(typing.ContextManager["BlockMatSubMatrixContextManagerImpl"]):
        """
        Apply MatSubMatrixContextManager to every block of a block matrix.

        Parameters
        ----------
        A
            An online matrix.
        M, N
            Dimension of the blocks of the matrix.
        """

        def __init__(
            self: typing_extensions.Self, A: petsc4py.PETSc.Mat,  # type: ignore[name-defined]
            M: list[int], N: list[int]
        ) -> None:
            self._A = A
            row_blocks = np.hstack((0, np.cumsum([M_ for M_ in M])))
            col_blocks = np.hstack((0, np.cumsum([N_ for N_ in N])))
            self._row_indices = [np.arange(row_blocks[i], row_blocks[i + 1], dtype=np.int32) for i in range(len(M))]
            self._col_indices = [np.arange(col_blocks[j], col_blocks[j + 1], dtype=np.int32) for j in range(len(N))]

        def __iter__(self: typing_extensions.Self) -> typing.Iterator[  # type: ignore[name-defined]
                tuple[int, int, petsc4py.PETSc.Mat]]:
            """Iterate over blocks."""
            with contextlib.ExitStack() as wrapper_stack:
                for (I, row_indices_) in enumerate(self._row_indices):  # noqa: E741
                    for (J, col_indices_) in enumerate(self._col_indices):
                        wrapper = MatSubMatrixContextManager(self._A, row_indices_, col_indices_)
                        yield (I, J, wrapper_stack.enter_context(wrapper))

        def __enter__(self: typing_extensions.Self) -> typing_extensions.Self:
            """Return this context manager."""
            return self

        def __exit__(
            self: typing_extensions.Self, exception_type: type[BaseException] | None,
            exception_value: BaseException | None,
            traceback: types.TracebackType | None
        ) -> None:
            """Do nothing upon exit."""
            pass

    return BlockMatSubMatrixContextManagerImpl


class BlockMatSubMatrixWrapper(BlockMatSubMatrixContextManager(MatSubMatrixWrapper)):  # type: ignore[misc]
    """
    Wrap an online matrix with multiple blocks and iterate over each block.

    Parameters
    ----------
    A
        An online matrix.
    M, N
        Dimension of the blocks of the matrix.
    """

    pass


class BlockMatSubMatrixCopier(BlockMatSubMatrixContextManager(MatSubMatrixCopier)):  # type: ignore[misc]
    """
    Copy an online matrix with multiple blocks while iterating over each block.

    Parameters
    ----------
    A
        An online matrix.
    M, N
        Dimension of the blocks of the matrix.
    """

    pass
