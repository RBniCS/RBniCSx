# Copyright (C) 2021-2026 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for rbnicsx.io.text_box module."""

import rbnicsx.io


def test_text_box() -> None:
    """Unit test for TextBox.__str__."""
    greet = ["Hello, World!", "Ciao mondo!"]
    fill = "#"
    text_box = rbnicsx.io.TextBox("\n".join(greet), fill=fill)
    text_box_str = str(text_box)
    text_box_lines = text_box_str.split("\n")
    assert len(text_box_lines) == 4
    text_box_lines_len = [len(text_box_line) for text_box_line in text_box_lines]
    assert text_box_lines[0] == fill * text_box_lines_len[0]
    assert text_box_lines[3] == fill * text_box_lines_len[3]
    for (row, greet_) in enumerate(greet):
        assert text_box_lines[row + 1][0] == fill
        assert text_box_lines[row + 1][-1] == fill
        first_non_space = len(text_box_lines[row + 1][1:]) - len(text_box_lines[row + 1][1:].lstrip()) + 1
        last_non_space = first_non_space + len(text_box_lines[row + 1][first_non_space:-1].rstrip())
        assert text_box_lines[row + 1][1:first_non_space] == " " * (first_non_space - 1)
        assert text_box_lines[row + 1][last_non_space + 1:-1] == " " * (
            text_box_lines_len[row + 1] - last_non_space - 2)
        assert text_box_lines[row + 1][first_non_space:last_non_space] == greet_
