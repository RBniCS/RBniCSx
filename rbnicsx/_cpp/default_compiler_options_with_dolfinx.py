# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Determine default compiler options when dolfinx is available."""  # pragma: no cover

import typing  # pragma: no cover

import dolfinx.jit  # pragma: no cover
import dolfinx.pkgconfig  # pragma: no cover
import dolfinx.wrappers  # pragma: no cover
import numpy as np  # pragma: no cover
import petsc4py.PETSc  # pragma: no cover


def determine_default_compiler_options() -> typing.Dict[str, typing.Union[str, typing.List[str]]]:  # pragma: no cover
    """Determine default compiler options when dolfinx is available."""
    default_compiler_options: typing.Dict[str, typing.Union[str, typing.List[str]]] = dict()

    # C++ components
    dolfinx_pc = dict()
    has_petsc_complex = np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating)
    for (dolfinx_pc_package, scalar_type_check) in zip(
        ("dolfinx", "dolfinx_real", "dolfinx_complex"),
        (True, not has_petsc_complex, has_petsc_complex)
    ):
        if dolfinx.pkgconfig.exists(dolfinx_pc_package) and scalar_type_check:  # type: ignore[no-untyped-call]
            dolfinx_pc.update(dolfinx.pkgconfig.parse(dolfinx_pc_package))  # type: ignore[no-untyped-call]
            break
    assert len(dolfinx_pc) > 0
    default_compiler_options["include_dirs"] = [
        include_dir for include_dir in dolfinx_pc["include_dirs"] if "-NOTFOUND" not in include_dir]
    default_compiler_options["compiler_args"] = ["-std=c++20"] + [
        "-D" + define_macro for define_macro in dolfinx_pc["define_macros"] if "-NOTFOUND" not in define_macro]
    default_compiler_options["library_dirs"] = [
        library_dir for library_dir in dolfinx_pc["library_dirs"] if "-NOTFOUND" not in library_dir]
    default_compiler_options["libraries"] = dolfinx_pc["libraries"]

    # Output directory
    jit_options = dolfinx.jit.get_options()
    default_compiler_options["output_dir"] = str(jit_options["cache_dir"])

    return default_compiler_options
