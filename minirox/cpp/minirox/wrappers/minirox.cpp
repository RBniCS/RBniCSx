// Copyright (C) 2021-2022 by the minirox authors
//
// This file is part of minirox.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace minirox_wrappers
{
    void backends(py::module& m);
}

PYBIND11_MODULE(SIGNATURE, m)
{
    // Create module for C++ wrappers
    m.doc() = "minirox Python interface";

    // Create backends submodule
    py::module backends = m.def_submodule("backends", "backends module");
    minirox_wrappers::backends(backends);
}
