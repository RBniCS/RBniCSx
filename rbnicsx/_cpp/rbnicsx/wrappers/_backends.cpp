// Copyright (C) 2021-2026 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <nanobind/nanobind.h>

#include <rbnicsx/_backends/frobenius_inner_product.h>
#include <rbnicsx/_backends/petsc_casters.h>

namespace nb = nanobind;

namespace rbnicsx_wrappers
{
void _backends(nb::module_& m)
{
  m.def("frobenius_inner_product", &rbnicsx::_backends::frobenius_inner_product,
        "Frobenius inner product between PETSc Mat objects.");
}
} // namespace rbnicsx_wrappers
