# Copyright (C) 2021-2025 by the RBniCSx authors
#
# This file is part of RBniCSx.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Script to generate tutorials."""

import contextlib
import io
import sys

import pytest

if __name__ == "__main__":  # pragma: no cover
    discard_stdout = io.StringIO()
    with contextlib.redirect_stdout(discard_stdout):
        retcode = pytest.main(["--ipynb-action=create-notebooks", "--collapse", "--work-dir=."])
    if retcode == pytest.ExitCode.NO_TESTS_COLLECTED:
        sys.exit(0)
    else:
        print(discard_stdout.getvalue())
        sys.exit(retcode)
