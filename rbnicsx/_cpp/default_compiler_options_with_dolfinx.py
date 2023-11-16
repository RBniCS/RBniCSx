# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Determine default compiler options when dolfinx is available."""  # pragma: no cover

import typing  # pragma: no cover

import dolfinx.jit  # pragma: no cover
import dolfinx.pkgconfig  # pragma: no cover
import dolfinx.wrappers  # pragma: no cover


def determine_default_compiler_options() -> typing.Dict[str, typing.Union[str, typing.List[str]]]:  # pragma: no cover
    """Determine default compiler options when dolfinx is available."""
    default_compiler_options: typing.Dict[str, typing.Union[str, typing.List[str]]] = dict()

    # C++ components
    dolfinx_pc = dolfinx.pkgconfig.parse("dolfinx")  # type: ignore[no-untyped-call]
    default_compiler_options["include_dirs"] = [
        include_dir for include_dir in dolfinx_pc["include_dirs"] if "-NOTFOUND" not in include_dir]
    default_compiler_options["compiler_args"] = ["-std=c++20"] + [
        "-D" + define_macro for define_macro in dolfinx_pc["define_macros"] if "-NOTFOUND" not in define_macro]
    default_compiler_options["library_dirs"] = [
        library_dir for library_dir in dolfinx_pc["library_dirs"] if "-NOTFOUND" not in library_dir]
    default_compiler_options["libraries"] = dolfinx_pc["libraries"]

    # Output directory
    jit_parameters = dolfinx.jit.get_parameters()
    default_compiler_options["output_dir"] = str(jit_parameters["cache_dir"])

    return default_compiler_options
