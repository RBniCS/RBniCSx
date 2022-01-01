# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox documentation configuration."""

# Project information
project = "minirox"
copyright = "2021-2022, the minirox authors"
author = "Francesco Ballarin (and contributors)"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode"
]

# Extensions configuration
autodoc_default_options = {
    "exclude-members": "__dict__,__init__,__module__,__weakref__",
    "imported-members": True,
    "members": True,
    "show-inheritance": True,
    "special-members": True,
    "undoc-members": True
}

# Options for HTML output
html_theme = "nature"
