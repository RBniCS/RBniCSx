# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pytest configuration file for tutorials tests."""

import multiphenicsx.test.notebooks

pytest_addoption = multiphenicsx.test.notebooks.addoption
pytest_collect_file = multiphenicsx.test.notebooks.collect_file
pytest_runtest_setup = multiphenicsx.test.notebooks.runtest_setup
pytest_runtest_makereport = multiphenicsx.test.notebooks.runtest_makereport
pytest_runtest_teardown = multiphenicsx.test.notebooks.runtest_teardown
