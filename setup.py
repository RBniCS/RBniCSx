# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox setup script."""

import site
import sys

import setuptools

# Workaround for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

setuptools.setup()
