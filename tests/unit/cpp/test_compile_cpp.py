# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for minirox.io.compile_code module."""

import os
import typing

import dolfinx.jit
import dolfinx_utils.test.fixtures
import mpi4py

import minirox.cpp
import minirox.io

tempdir = dolfinx_utils.test.fixtures.tempdir


def test_compile_code(tempdir: str) -> None:
    """Compile a simple C++ function."""

    def write_code() -> str:
        code = """
#include <pybind11/pybind11.h>

int multiply(int a, int b) {
    return a * b;
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("multiply", &multiply);
}
"""
        filename = os.path.join(tempdir, "test_compile_code_source.cpp")
        open(filename, "w").write(code)
        return filename
    filename = minirox.io.on_rank_zero(mpi4py.MPI.COMM_WORLD, write_code)

    compile_code = dolfinx.jit.mpi_jit_decorator(minirox.cpp.compile_code)
    cpp_library = compile_code(mpi4py.MPI.COMM_WORLD, "test_compile_code", filename, output_dir=tempdir)
    assert cpp_library.multiply(2, 3) == 6


def test_compile_package(tempdir: str) -> None:
    """Compile a simple C++ package."""

    def write_package_files() -> typing.List[str]:
        package_root = os.path.join(tempdir, "test_compile_package")
        os.makedirs(os.path.join(package_root, "utilities"))
        os.makedirs(os.path.join(package_root, "wrappers"))

        multiply_header_code = """
namespace utilities
{
    int multiply(int a, int b);
}
"""
        multiply_header_file = os.path.join("utilities", "multiply.h")
        open(os.path.join(package_root, multiply_header_file), "w").write(multiply_header_code)

        multiply_source_code = """
#include <test_compile_package/utilities/multiply.h>

int utilities::multiply(int a, int b)
{
    return a * b;
}
"""
        multiply_source_file = os.path.join("utilities", "multiply.cpp")
        open(os.path.join(package_root, multiply_source_file), "w").write(multiply_source_code)

        utilities_wrapper_code = """
#include <pybind11/pybind11.h>

#include <test_compile_package/utilities/multiply.h>

namespace py = pybind11;

namespace wrappers
{
    void utilities(py::module& m)
    {
        m.def("multiply", &utilities::multiply);
    }
}
"""
        utilities_wrapper_file = os.path.join("wrappers", "utilities.cpp")
        open(os.path.join(package_root, utilities_wrapper_file), "w").write(utilities_wrapper_code)

        main_wrapper_code = """
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace wrappers
{
    void utilities(py::module& m);
}

PYBIND11_MODULE(SIGNATURE, m)
{
    py::module utilities = m.def_submodule("utilities", "utilities module");
    wrappers::utilities(utilities);
}
"""
        main_wrapper_file = os.path.join("wrappers", "test_compile_package.cpp")
        open(os.path.join(package_root, main_wrapper_file), "w").write(main_wrapper_code)

        return [multiply_source_file]
    sources = minirox.io.on_rank_zero(mpi4py.MPI.COMM_WORLD, write_package_files)

    cpp_library = minirox.cpp.compile_package(
        mpi4py.MPI.COMM_WORLD, "test_compile_package", tempdir, *sources, output_dir=tempdir)
    assert cpp_library.utilities.multiply(2, 3) == 6
