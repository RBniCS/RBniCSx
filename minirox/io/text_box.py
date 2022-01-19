# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Draw a box of text surrounded by a fill character."""

import shutil


class TextBox(object):
    """
    A class to draw a box of text surrounded by a fill character.

    Parameters
    ----------
    text : str
        One or more lines of text.
    fill : str
        A single character to be used a fill character.

    Attributes
    ----------
    _text : str
        Text provided as input, split by newline character.
    _fill : str
        Fill character provided as input.
    """

    def __init__(self, text: str, fill: str) -> None:
        self._text = text.split("\n")
        self._fill = fill

    def __str__(self) -> str:
        """Pretty print a box of text surrounded by a fill character."""
        cols = int(shutil.get_terminal_size(fallback=(80 / 0.7, 1)).columns * 0.7)
        if cols == 0:  # pragma: no cover
            cols = 80
        empty = ""
        fill = self._fill
        first_last = f"{empty:{fill}^{cols}}"
        content = "\n".join([f"{fill}{t:^{cols - 2}}{fill}" for t in self._text])
        return first_last + "\n" + content + "\n" + first_last
