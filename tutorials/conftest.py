# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
pytest configuration file for tutorials.

This file is mainly responsible for the conversion of Jupyter notebooks into plain python files
for later processing by pytest.
"""

import os
import sys

import nbconvert.exporters
import nbconvert.filters
import nbconvert.preprocessors
import nbformat
import pytest


def pytest_addoption(parser):
    """Add option to write out to notebook files after run."""
    parser.addoption("--nb-write-out", action="store_true", help="write out to notebook files after run")


def pytest_collect_file(path, parent):
    """Collect tutorial files."""
    if path.ext == ".ipynb":
        if not path.basename.startswith("x"):
            return TutorialFile.from_parent(parent=parent, fspath=path)
        else:
            return DoNothingFile.from_parent(parent=parent, fspath=path)


class TutorialFile(pytest.File):
    """Custom file handler for tutorial files."""

    def collect(self):
        """Collect the tutorial file."""
        yield TutorialItem.from_parent(
            parent=self, name="run_tutorial -> " + os.path.relpath(str(self.fspath), str(self.parent.fspath)))


class TutorialItem(pytest.Item):
    """Handle the execution of the tutorial."""

    def __init__(self, name, parent):
        super(TutorialItem, self).__init__(name, parent)

    def runtest(self):
        """Run the tutorial item."""
        os.chdir(self.parent.fspath.dirname)
        sys.path.append(self.parent.fspath.dirname)
        with open(self.parent.fspath) as f:
            nb = nbformat.read(f, as_version=4)
        execute_preprocessor = nbconvert.preprocessors.ExecutePreprocessor()
        try:
            execute_preprocessor.preprocess(nb)
        finally:
            if self.config.getoption("--nb-write-out"):
                with open(self.parent.fspath, "w") as f:
                    nbformat.write(nb, f)

    def reportinfo(self):
        """Report information on the tutorial item."""
        return self.fspath, 0, self.name


class DoNothingFile(pytest.File):
    """Custom file handler to avoid running twice python files explicitly provided on the command line."""

    def collect(self):
        """Do not collect anything on this file."""
        return []
