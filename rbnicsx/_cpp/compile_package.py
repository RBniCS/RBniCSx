# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compile a C++ package using cppimport."""

import glob
import os
import types
import typing

import mpi4py.MPI

from rbnicsx._cpp.compile_code import compile_code


def compile_package(
    comm: mpi4py.MPI.Intracomm, package_name: str, package_root: str,
    *args: str, **kwargs: typing.Union[str, typing.List[str]]
) -> types.ModuleType:
    """Compile a C++ package."""
    # Remove extension from files
    files = [os.path.splitext(f)[0] for f in args]

    # Make sure that there are no duplicate files
    assert len(files) == len(set(files)), (
        "There seems to be duplicate files. Make sure to include in the list only *.cpp files, not *.h ones.")

    # Extract files
    package_headers = [os.path.join(package_root, package_name, f + ".h") for f in files]
    package_sources = [os.path.join(package_root, package_name, f + ".cpp") for f in files]
    assert len(package_headers) == len(package_sources)

    # Make sure that there are no files missing
    for (extension, typename, files_to_check) in zip(
            ["h", "cpp"], ["headers", "sources"], [package_headers, package_sources]):
        all_package_files = set(
            glob.glob(os.path.join(package_root, package_name, "[!wrappers]*", "*." + extension)))
        sorted_package_files = set(files_to_check)
        if len(sorted_package_files) > len(all_package_files):  # pragma: no cover
            raise RuntimeError(
                "Input " + typename + " list contains more files than ones present in the library. "
                + "The files " + str(sorted_package_files - all_package_files) + " seem not to exist.")
        elif len(sorted_package_files) < len(all_package_files):  # pragma: no cover
            raise RuntimeError(
                "Input " + typename + " list is not complete. "
                + "The files " + str(all_package_files - sorted_package_files) + " are missing.")
        else:
            assert sorted_package_files == all_package_files, (
                "Input " + typename + " list contains different files than ones present in the library. "
                + "The files " + str(sorted_package_files - all_package_files) + " seem not to exist.")

    # Extract submodules
    package_submodules = set(
        [os.path.relpath(f, os.path.join(package_root, package_name))
         for f in glob.glob(os.path.join(package_root, package_name, "*/"))])
    package_submodules.remove("wrappers")

    # Extract pybind11 files corresponding to each submodule
    package_pybind11_sources = list()
    for package_submodule in sorted(package_submodules):
        package_pybind11_sources.append(
            os.path.join(package_root, package_name, "wrappers", package_submodule + ".cpp"))

    # Get the main package file
    package_file = os.path.join(package_root, package_name, "wrappers", package_name + ".cpp")

    # Setup sources for compilation
    assert "sources" not in kwargs
    kwargs["sources"] = package_sources + package_pybind11_sources

    # Setup headers for compilation
    assert "dependencies" not in kwargs
    kwargs["dependencies"] = package_headers

    # Compile C++ module and return
    return compile_code(comm, package_name, package_root, package_file, **kwargs)
