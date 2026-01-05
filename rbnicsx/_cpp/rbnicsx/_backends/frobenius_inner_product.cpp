// Copyright (C) 2021-2026 by the RBniCSx authors
//
// This file is part of RBniCSx.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <rbnicsx/_backends/frobenius_inner_product.h>
#include <rbnicsx/_backends/petsc_error.h>

PetscReal rbnicsx::_backends::frobenius_inner_product(Mat a, Mat b)
{
  PetscInt start_a, end_a, ncols_a, start_b, end_b, ncols_b;
  PetscErrorCode ierr;
  const PetscInt *cols_a, *cols_b;
  const PetscScalar *vals_a, *vals_b;
  PetscScalar sum(0.);

  ierr = MatGetOwnershipRange(a, &start_a, &end_a);
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MatGetOwnershipRange");
  if (a != b)
  {
    ierr = MatGetOwnershipRange(b, &start_b, &end_b);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatGetOwnershipRange");
  }
  else
  {
    start_b = start_a;
    end_b = end_a;
  }
  if (start_a != start_b)
    petsc_error(ierr, __FILE__, "start_a != start_b");
  if (end_a != end_b)
    petsc_error(ierr, __FILE__, "end_a != end_b");

  for (PetscInt i(start_a); i < end_a; i++)
  {
    ierr = MatGetRow(a, i, &ncols_a, &cols_a, &vals_a);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatGetRow");
    if (a != b)
    {
      ierr = MatGetRow(b, i, &ncols_b, &cols_b, &vals_b);
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "MatGetRow");
      if (ncols_a != ncols_b)
        petsc_error(ierr, __FILE__, "ncols_a != ncols_b");
    }
    else
    {
      ncols_b = ncols_a;
      cols_b = cols_a;
      vals_b = vals_a;
    }
    for (PetscInt j(0); j < ncols_a; j++)
    {
      if (cols_a[j] != cols_b[j])
        petsc_error(ierr, __FILE__, "cols_a[j] != cols_b[j]");
      sum += vals_a[j] * vals_b[j];
    }
    if (a != b)
    {
      ierr = MatRestoreRow(b, i, &ncols_b, &cols_b, &vals_b);
      if (ierr != 0)
        petsc_error(ierr, __FILE__, "MatRestoreRow");
    }
    else
    {
      ncols_b = 0;
      cols_b = NULL;
      vals_b = NULL;
    }
    ierr = MatRestoreRow(a, i, &ncols_a, &cols_a, &vals_a);
    if (ierr != 0)
      petsc_error(ierr, __FILE__, "MatRestoreRow");
  }

  PetscReal output(0.);
  ierr = MPIU_Allreduce(&sum, &output, 1, MPIU_REAL, MPIU_SUM,
                        PetscObjectComm((PetscObject)a));
  if (ierr != 0)
    petsc_error(ierr, __FILE__, "MPIU_Allreduce");
  return output;
}
