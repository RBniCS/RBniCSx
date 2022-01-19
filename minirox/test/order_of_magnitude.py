# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compute the order of magnitude of a number."""

import numbers

import numpy as np


def order_of_magnitude(number: numbers.Real) -> int:
    """Compute the order of magnitude of a number."""
    return np.floor(np.log10(number)).astype(int)
