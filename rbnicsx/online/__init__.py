# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""RBniCSx online module."""

# Expose tensor creation functions which are implemented in the internal module to avoid cyclic imports
from rbnicsx._backends.online_tensors import (
    create_online_matrix as create_matrix, create_online_matrix_block as create_matrix_block,
    create_online_vector as create_vector, create_online_vector_block as create_vector_block)
#
# Import functions and classes defined in this module
from rbnicsx.online.export import (
    export_matrices, export_matrices_block, export_matrix, export_matrix_block, export_vector, export_vector_block,
    export_vectors, export_vectors_block)
from rbnicsx.online.functions_list import FunctionsList
from rbnicsx.online.import_ import (
    import_matrices, import_matrices_block, import_matrix, import_matrix_block, import_vector, import_vector_block,
    import_vectors, import_vectors_block)
from rbnicsx.online.tensors_list import TensorsList
