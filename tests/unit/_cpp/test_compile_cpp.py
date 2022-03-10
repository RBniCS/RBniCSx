# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx._cpp.compile_code module."""

import os

import mpi4py.MPI
import nbvalx.tempfile
import pytest

import rbnicsx._cpp
import rbnicsx.io


def test_compile_code_std() -> None:
    """Compile a simple C++ function which only uses classes from the standard library."""
    comm = mpi4py.MPI.COMM_WORLD

    with nbvalx.tempfile.TemporaryDirectory(comm) as tempdir:
        if comm.rank == 0:
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
            filename = os.path.join(tempdir, "test_compile_code_std_source.cpp")
            open(filename, "w").write(code)
            filename = comm.bcast(filename, root=0)
        else:
            filename = comm.bcast(None, root=0)

        cpp_library = rbnicsx._cpp.compile_code(comm, "test_compile_code_std", tempdir, filename, output_dir=tempdir)
        assert cpp_library.multiply(2, 3) == 6


def test_compile_code_petsc() -> None:
    """Compile a simple C++ function which requires PETSc."""
    comm = mpi4py.MPI.COMM_WORLD

    with nbvalx.tempfile.TemporaryDirectory(comm) as tempdir:
        if comm.rank == 0:
            code = """
#include <petscksp.h>
#include <pybind11/pybind11.h>

int create_ksp() {
    KSP ksp;
    KSPCreate(MPI_COMM_WORLD, &ksp);
    return 0;
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("create_ksp", &create_ksp);
}
"""
            filename = os.path.join(tempdir, "test_compile_code_petsc_source.cpp")
            open(filename, "w").write(code)
            filename = comm.bcast(filename, root=0)
        else:
            filename = comm.bcast(None, root=0)

        cpp_library = rbnicsx._cpp.compile_code(comm, "test_compile_code_petsc", tempdir, filename, output_dir=tempdir)
        assert cpp_library.create_ksp() == 0


def test_compile_code_slepc() -> None:
    """Compile a simple C++ function which requires SLEPc."""
    comm = mpi4py.MPI.COMM_WORLD

    with nbvalx.tempfile.TemporaryDirectory(comm) as tempdir:
        if comm.rank == 0:
            code = """
#include <slepceps.h>
#include <pybind11/pybind11.h>

int create_eps() {
    EPS eps;
    EPSCreate(MPI_COMM_WORLD, &eps);
    return 0;
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("create_eps", &create_eps);
}
"""
            filename = os.path.join(tempdir, "test_compile_code_slepc_source.cpp")
            open(filename, "w").write(code)
            filename = comm.bcast(filename, root=0)
        else:
            filename = comm.bcast(None, root=0)

        cpp_library = rbnicsx._cpp.compile_code(comm, "test_compile_code_slepc", tempdir, filename, output_dir=tempdir)
        assert cpp_library.create_eps() == 0


def test_compile_code_error() -> None:
    """Compile a simple C++ function with an undefined variable, which causes a compilation error."""
    comm = mpi4py.MPI.COMM_WORLD

    with nbvalx.tempfile.TemporaryDirectory(comm) as tempdir:
        if comm.rank == 0:
            code = """
#include <pybind11/pybind11.h>

void compilation_error() {
    undefined_variable;
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("compilation_error", &compilation_error);
}
"""
            filename = os.path.join(tempdir, "test_compile_code_error_source.cpp")
            open(filename, "w").write(code)
            filename = comm.bcast(filename, root=0)
        else:
            filename = comm.bcast(None, root=0)

        with pytest.raises(RuntimeError) as excinfo:
            rbnicsx._cpp.compile_code(
                comm, "test_compile_code_error", tempdir, filename, output_dir=tempdir)
        assert "Compilation failed" in str(excinfo.value)


def test_compile_package() -> None:
    """Compile a simple C++ package."""
    comm = mpi4py.MPI.COMM_WORLD

    with nbvalx.tempfile.TemporaryDirectory(comm) as tempdir:
        if comm.rank == 0:
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

            multiply_source_file = comm.bcast(multiply_source_file, root=0)
        else:
            multiply_source_file = comm.bcast(None, root=0)
        sources = [multiply_source_file]

        cpp_library = rbnicsx._cpp.compile_package(
            comm, "test_compile_package", tempdir, *sources, output_dir=tempdir)
        assert cpp_library.utilities.multiply(2, 3) == 6
