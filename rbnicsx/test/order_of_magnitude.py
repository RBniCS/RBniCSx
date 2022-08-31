# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compute the order of magnitude of a number."""

import typing

import numpy as np
import numpy.typing


def order_of_magnitude(number: typing.Union[float, typing.Sequence[float]]) -> typing.Union[
        np.int32, np.typing.NDArray[np.int32]]:
    """Compute the order of magnitude of a number."""
    return np.floor(np.log10(number)).astype(np.int32)
