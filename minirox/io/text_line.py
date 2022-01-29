# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Draw a line of text surrounded by a fill character."""

import shutil


class TextLine(object):
    """
    A class to draw a line of text surrounded by a fill character.

    Parameters
    ----------
    text : str
        A line of text.
    fill : str
        A single character to be used a fill character.

    Attributes
    ----------
    _text : str
        Line of text provided as input.
    _fill : str
        Fill character provided as input.
    """

    def __init__(self, text: str, fill: str) -> None:
        self._text = f" {text} "
        self._fill = fill

    def __str__(self) -> str:
        """Pretty print a line of text surrounded by a fill character."""
        cols = int(shutil.get_terminal_size(fallback=(80 / 0.7, 1)).columns * 0.7)
        if cols == 0:  # pragma: no cover
            cols = 80
        text = self._text
        fill = self._fill
        return f"{text:{fill}^{cols}}"
