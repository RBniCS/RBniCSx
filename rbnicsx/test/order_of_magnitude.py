# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compute the order of magnitude of a number."""

import numbers
import typing

import numpy as np


def order_of_magnitude(number: typing.Union[numbers.Real, typing.Iterable[numbers.Real]]) -> int:
    """Compute the order of magnitude of a number."""
    return np.floor(np.log10(number)).astype(int)
