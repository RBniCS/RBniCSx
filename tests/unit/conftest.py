# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""pytest configuration file for unit tests."""

import multiphenicsx.test.unit_tests

pytest_runtest_setup = multiphenicsx.test.unit_tests.runtest_setup
pytest_runtest_teardown = multiphenicsx.test.unit_tests.runtest_teardown
