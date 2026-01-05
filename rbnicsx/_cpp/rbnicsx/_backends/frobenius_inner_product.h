// Copyright (C) 2021-2026 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <petscmat.h>

namespace rbnicsx::_backends
{
/// Frobenius inner product between two matrices, with code adapted from the
/// implementation of MatAXPY in PETSc.
PetscReal frobenius_inner_product(Mat a, Mat b);
} // namespace rbnicsx::_backends
