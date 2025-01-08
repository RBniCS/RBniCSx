# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.test.order_of_magnitude module."""

import rbnicsx.test


def test_order_of_magnitude() -> None:
    """Verify rbnicsx.test.order_of_magnitude on some simple cases."""
    assert rbnicsx.test.order_of_magnitude(1.0) == 0
    assert rbnicsx.test.order_of_magnitude(2.5) == 0
    assert rbnicsx.test.order_of_magnitude(25) == 1
    assert rbnicsx.test.order_of_magnitude(25.) == 1
    assert rbnicsx.test.order_of_magnitude(0.1) == -1
    assert rbnicsx.test.order_of_magnitude(0.09) == -2
