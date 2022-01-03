# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""
pytest configuration file for notebooks tests.

This file is mainly for starting a ipyparallel Cluster when running notebooks tests in parallel.
"""

import os

import nbformat
import nbval.plugin
import pytest


def pytest_addoption(parser):
    """Add option to set the number of processes."""
    parser.addoption("--np", action="store", type=int, default=1, help="Number of MPI processes to use")
    assert (
        not ("OMPI_COMM_WORLD_SIZE" in os.environ  # OpenMPI
             or "MPI_LOCALNRANKS" in os.environ)), (  # MPICH
        "Please do not start pytest under mpirun. Use the --np pytest option.")


def pytest_collect_file(path, parent):
    """Collect IPython notebooks using a custom pytest nbval hook."""
    opt = parent.config.option
    assert not opt.nbval, "--nbval is implicitly enabled, do not provide it on the command line"
    if path.fnmatch("**/*.ipynb") and not path.fnmatch("**/.ipynb_mpi/*.ipynb"):
        if opt.np > 1:
            # Read in notebook
            with open(path) as f:
                nb = nbformat.read(f, as_version=4)
            # Add the %%px magic to every existing cell
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell.source = "%%px\n" + cell.source
            # Add a cell on top to start a new ipyparallel cluster
            cluster_start_code = f"""import ipyparallel as ipp
cluster = ipp.Cluster(engines="MPI", profile="mpi", n={opt.np})
cluster.start_and_connect_sync()
"""
            cluster_start_cell = nbformat.v4.new_code_cell(cluster_start_code)
            cluster_start_cell.id = "cluster_start"
            nb.cells.insert(0, cluster_start_cell)
            # Add a further cell on top to disable garbage collection
            gc_disable_code = """%%px
import gc
gc.disable()
"""
            gc_disable_cell = nbformat.v4.new_code_cell(gc_disable_code)
            gc_disable_cell.id = "gc_disable"
            nb.cells.insert(1, gc_disable_cell)
            # Add a cell at the end to re-enable garbage collection
            gc_enable_code = """%%px
gc.enable()
gc.collect()
"""
            gc_enable_cell = nbformat.v4.new_code_cell(gc_enable_code)
            gc_enable_cell.id = "gc_enable"
            nb.cells.append(gc_enable_cell)
            # Add a cell at the end to stop the ipyparallel cluster
            cluster_stop_code = """cluster.stop_cluster_sync()
"""
            cluster_stop_cell = nbformat.v4.new_code_cell(cluster_stop_code)
            cluster_stop_cell.id = "cluster_stop"
            nb.cells.append(cluster_stop_cell)
            # Write modified notebook to a temporary file
            mpi_dir = os.path.join(path.dirname, ".ipynb_mpi")
            os.makedirs(mpi_dir, exist_ok=True)
            ipynb_path = path.new(dirname=mpi_dir)
            with open(ipynb_path, "w") as f:
                nbformat.write(nb, str(ipynb_path))
        else:
            ipynb_path = path
        return nbval.plugin.IPyNbFile.from_parent(parent, fspath=ipynb_path)


def pytest_runtest_setup(item):
    """Insert skips on cell failure."""
    # Do the normal setup
    item.setup()
    # If previous cells in a notebook failed skip the rest of the notebook
    if hasattr(item, "_previous_failed") and item.cell.id not in ("gc_enable", "cluster_stop"):
        pytest.skip("A previous cell failed")


def pytest_runtest_makereport(item, call):
    """Determine whether the current cell failed or not."""
    if call.when == "call":
        if call.excinfo:
            if os.path.basename(item.parent.name).startswith("x"):
                call.excinfo._excinfo = (
                    call.excinfo._excinfo[0],
                    pytest.xfail.Exception("One cell in this notebook was expected to fail"),
                    call.excinfo._excinfo[2])
            item._failed = True


def pytest_runtest_teardown(item, nextitem):
    """Propagate cell failure."""
    # Do the normal teardown
    item.teardown()
    # Inform next cell of the notebook of failure of any previous cells
    if hasattr(item, "_failed") or hasattr(item, "_previous_failed"):
        if nextitem is not None and nextitem.name != "Cell 0":
            nextitem._previous_failed = True
