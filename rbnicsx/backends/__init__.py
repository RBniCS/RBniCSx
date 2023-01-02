# Copyright (C) 2021-2023 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""RBniCSx backends module."""

from rbnicsx.backends.export import (
    export_function, export_functions, export_matrices, export_matrix, export_vector, export_vectors)
from rbnicsx.backends.functions_list import FunctionsList
from rbnicsx.backends.gram_schmidt import gram_schmidt, gram_schmidt_block
from rbnicsx.backends.import_ import (
    import_function, import_functions, import_matrices, import_matrix, import_vector, import_vectors)
from rbnicsx.backends.mesh_motion import MeshMotion
from rbnicsx.backends.projection import (
    bilinear_form_action, block_bilinear_form_action, block_diagonal_bilinear_form_action, block_linear_form_action,
    FormArgumentsReplacer, linear_form_action, project_matrix, project_matrix_block, project_vector,
    project_vector_block)
from rbnicsx.backends.proper_orthogonal_decomposition import (
    proper_orthogonal_decomposition, proper_orthogonal_decomposition_block)
from rbnicsx.backends.symbolic_parameters import SymbolicParameters
from rbnicsx.backends.tensors_array import TensorsArray
from rbnicsx.backends.tensors_list import TensorsList
