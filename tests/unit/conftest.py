# Copyright (C) 2021 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
pytest configuration file for unit tests.

This file is mainly responsible to call garbage collection and put a MPI barrier after each test.
"""

import gc

import mpi4py


def pytest_runtest_setup(item):
    """Disable garbage collection before running tests."""
    # Do the normal setup
    item.setup()
    # Disable garbage collection
    gc.disable()


def pytest_runtest_teardown(item, nextitem):
    """Force garbage collection and put a MPI barrier after running tests."""
    # Do the normal teardown
    item.teardown()
    # Re-enable garbage collection
    gc.enable()
    # Run garbage gollection
    del item
    gc.collect()
    # Add a MPI barrier in parallel
    mpi4py.MPI.COMM_WORLD.Barrier()
