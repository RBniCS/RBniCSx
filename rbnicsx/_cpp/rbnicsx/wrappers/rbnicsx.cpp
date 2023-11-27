// Copyright (C) 2021-2023 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace rbnicsx_wrappers
{
void _backends(nb::module_& m);
}

NB_MODULE(rbnicsx_cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "RBniCSx Python interface";

  // Create internal backends submodule
  nb::module_ _backends
      = m.def_submodule("_backends", "Internal backends module");
  rbnicsx_wrappers::_backends(_backends);
}
