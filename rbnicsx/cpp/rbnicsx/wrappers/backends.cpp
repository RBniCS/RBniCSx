// Copyright (C) 2021-2022 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <caster_petsc.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include <rbnicsx/backends/frobenius_inner_product.h>

namespace py = pybind11;

namespace rbnicsx_wrappers
{
    void backends(py::module& m)
    {
        m.def("frobenius_inner_product", &rbnicsx::backends::frobenius_inner_product,
              "Frobenius inner product between PETSc Mat objects.");
    }
}
