# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Determine default compiler options when dolfinx is not available."""  # pragma: no cover

import os  # pragma: no cover
import typing  # pragma: no cover


def determine_default_compiler_options() -> typing.Dict[str, typing.Union[str, typing.List[str]]]:  # pragma: no cover
    """Determine default compiler options when dolfinx is available."""
    default_compiler_options: typing.Dict[str, typing.Union[str, typing.List[str]]] = dict()

    # Get PETSc and SLEPc installation directory from environment variables, if any
    try:
        petsc_dir = os.environ["PETSC_DIR"]
    except KeyError:
        petsc_dir = ""
    else:
        petsc_dir = os.path.join(petsc_dir, os.environ.get("PETSC_ARCH", ""))
    slepc_dir = os.environ.get("SLEPC_DIR", "")

    # C++ components
    default_compiler_options["include_dirs"] = [
        os.path.join(dir_, "include") for dir_ in [petsc_dir, slepc_dir] if dir_ != ""]
    default_compiler_options["library_dirs"] = [
        os.path.join(dir_, "lib") for dir_ in [petsc_dir, slepc_dir] if dir_ != ""]
    default_compiler_options["libraries"] = ["petsc", "slepc"]

    # Output directory
    default_compiler_options["output_dir"] = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "rbnicsx")

    return default_compiler_options
