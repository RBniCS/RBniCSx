// Copyright (C) 2021-2023 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <stdexcept>

#include <petscsys.h>

#include <rbnicsx/_backends/petsc_error.h>

void rbnicsx::_backends::petsc_error(int error_code, std::string filename,
                                     std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Log detailed error info
  throw std::runtime_error(
      "Failed to successfully call PETSc function '" + petsc_function + "' in '"
      + filename + "'. " + "PETSc error code is: " + std ::to_string(error_code)
      + ", " + std::string(desc));
}
