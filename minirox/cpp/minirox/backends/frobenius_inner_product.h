// Copyright (C) 2021-2022 by the minirox authors
//
// This file is part of minirox.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <petscmat.h>

namespace minirox::backends
{
    /// Frobenius inner product between two matrices, with code adapted from the
    /// implementation of MatAXPY in PETSc.
    PetscScalar frobenius_inner_product(Mat a, Mat b);
}
