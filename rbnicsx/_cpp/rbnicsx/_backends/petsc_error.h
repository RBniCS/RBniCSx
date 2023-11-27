// Copyright (C) 2021-2023 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <string>

namespace rbnicsx::_backends
{
/// Print error message for PETSc calls that return an error
void petsc_error(int error_code, std::string filename,
                 std::string petsc_function);
} // namespace rbnicsx::_backends
