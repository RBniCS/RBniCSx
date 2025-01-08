# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.io.text_line module."""

import rbnicsx.io


def test_text_line() -> None:
    """Unit test for TextLine.__str__."""
    greet = "Hello, World!"
    fill = "#"
    text_line = rbnicsx.io.TextLine(greet, fill=fill)
    text_line_str = str(text_line)
    text_line_len = len(text_line_str)
    first_space = text_line_str.find(" ")
    assert first_space >= 0
    last_space = text_line_str.rfind(" ")
    assert first_space >= 0
    assert text_line_str[:first_space] == fill * first_space
    assert text_line_str[last_space + 1:] == fill * (text_line_len - last_space - 1)
    assert text_line_str[first_space + 1:last_space] == greet
