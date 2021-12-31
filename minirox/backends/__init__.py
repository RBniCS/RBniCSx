# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox backends module."""

from minirox.backends.export import (export_function, export_functions, export_matrices, export_matrix, export_vector,
                                     export_vectors)
from minirox.backends.functions_list import FunctionsList
from minirox.backends.import_ import (import_function, import_functions, import_matrices, import_matrix, import_vector,
                                      import_vectors)
from minirox.backends.projection import (create_online_matrix, create_online_matrix_block, create_online_vector,
                                         create_online_vector_block, project_matrix, project_matrix_block,
                                         project_vector, project_vector_block)
from minirox.backends.proper_orthogonal_decomposition import (proper_orthogonal_decomposition,
                                                              proper_orthogonal_decomposition_block)
from minirox.backends.tensors_list import TensorsList

__all__ = [
    "create_online_matrix",
    "create_online_matrix_block",
    "create_online_vector",
    "create_online_vector_block",
    "export_function",
    "export_functions",
    "export_matrices",
    "export_matrix",
    "export_vector",
    "export_vectors",
    "FunctionsList",
    "import_function",
    "import_functions",
    "import_matrices",
    "import_matrix",
    "import_vector",
    "import_vectors",
    "project_matrix",
    "project_matrix_block",
    "project_vector",
    "project_vector_block",
    "proper_orthogonal_decomposition",
    "proper_orthogonal_decomposition_block",
    "TensorsList"
]
