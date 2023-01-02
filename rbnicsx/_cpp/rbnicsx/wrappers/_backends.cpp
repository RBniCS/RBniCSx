// Copyright (C) 2021-2023 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include <rbnicsx/_backends/petsc_casters.h>
#include <rbnicsx/_backends/frobenius_inner_product.h>

namespace py = pybind11;

namespace rbnicsx_wrappers
{
    void _backends(py::module& m)
    {
        m.def("frobenius_inner_product", &rbnicsx::_backends::frobenius_inner_product,
              "Frobenius inner product between PETSc Mat objects.");
    }
}
