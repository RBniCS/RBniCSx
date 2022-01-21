# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pytest configuration file for tutorials tests."""

import nbvalx.pytest_hooks_notebooks

pytest_addoption = nbvalx.pytest_hooks_notebooks.addoption
pytest_sessionstart = nbvalx.pytest_hooks_notebooks.sessionstart
pytest_collect_file = nbvalx.pytest_hooks_notebooks.collect_file
pytest_runtest_setup = nbvalx.pytest_hooks_notebooks.runtest_setup
pytest_runtest_makereport = nbvalx.pytest_hooks_notebooks.runtest_makereport
pytest_runtest_teardown = nbvalx.pytest_hooks_notebooks.runtest_teardown
