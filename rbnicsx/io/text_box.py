# Copyright (C) 2021-2022 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Draw a box of text surrounded by a fill character."""

import shutil
import typing


class TextBox(object):
    """
    A class to draw a box of text surrounded by a fill character.

    Parameters
    ----------
    text
        One or more lines of text.
    fill
        A single character to be used a fill character.

    Attributes
    ----------
    _text
        Text provided as input, split by newline character.
    _fill
        Fill character provided as input.
    """

    def __init__(self, text: str, fill: str) -> None:
        self._text: typing.List[str] = text.split("\n")
        self._fill: str = fill

    def __str__(self) -> str:
        """Pretty print a box of text surrounded by a fill character."""
        cols = int(shutil.get_terminal_size(fallback=(int(80 / 0.7), 1)).columns * 0.7)
        if cols == 0:  # pragma: no cover
            cols = 80
        empty = ""
        fill = self._fill
        first_last = f"{empty:{fill}^{cols}}"
        content = "\n".join([f"{fill}{t:^{cols - 2}}{fill}" for t in self._text])
        return first_last + "\n" + content + "\n" + first_last
