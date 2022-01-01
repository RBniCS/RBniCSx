# Copyright (C) 2021-2022 by the minirox authors
#
# This file is part of minirox.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""minirox setup script."""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()
with open("tutorials/requirements.txt") as f:
    install_requires += f.read().splitlines()

setup(name="minirox",
      description="Reduced order modelling tutorials in FEniCSx",
      long_description="Reduced order modelling tutorials in FEniCSx",
      author="Francesco Ballarin (and contributors)",
      author_email="francesco.ballarin@unicatt.it",
      version="0.0.dev1",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="https://github.com/minirox/minirox",
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      packages=find_packages(),
      include_package_data=True,
      install_requires=install_requires
      )
