## RBniCSx - reduced order modelling in FEniCSx ##

**RBniCSx** contains an implementation in **FEniCSx** of several reduced order modelling techniques for parametrized problems. **RBniCSx** is still at a very early development stage compared to legacy **RBniCS**, and many features are presently missing.

**RBniCSx** is currently developed and maintained at [Università Cattolica del Sacro Cuore](https://www.unicatt.it/) by [Prof. Francesco Ballarin](https://www.francescoballarin.it).

Like all core **FEniCSx** components, **RBniCSx** is freely available under the GNU LGPL, version 3.

### Prerequisites

**RBniCSx** has a few build dependencies that must be installed manually. Follow the appropriate instructions below based on how [`dolfinx`](https://github.com/FEniCS/dolfinx) was installed.

#### If `dolfinx` was installed via conda

```console
conda install -c conda-forge nanobind scikit-build-core
```

#### If `dolfinx` was installed via apt

```console
apt install python3-nanobind python3-scikit-build-core
```

#### If `dolfinx` was installed via Docker or built from source

```console
python3 -m pip install nanobind scikit-build-core[pyproject]
```

### Installation

**RBniCSx** is available on PyPI. Use `pip` extras to install additional dependencies needed for the tutorials.

```console
python3 -m pip install --no-build-isolation 'rbnicsx[backends,tutorials]'
```

### Running tutorials

Tutorials are parameterized, meaning that `01_thermal_block.ipynb` actually contains several cases, e.g. `01_thermal_block[reduction_method=POD Galerkin,online_efficient=True].ipynb` and `01_thermal_block[reduction_method=Reduced Basis,online_efficient=True].ipynb`.
To run the tutorials, first clone the **RBniCSx** repository. Then, ensure you check out the tag that corresponds to the version of **RBniCSx** currently installed. Finally, generate all tutorials according to the parameterization.

```console
git clone https://github.com/RBniCS/RBniCSx.git
cd RBniCSx
RBNICSX_VERSION=$(python3 -c "import rbnicsx; print(rbnicsx.__version__)")
git checkout ${RBNICSX_VERSION}
cd tutorials
python3 generate_tutorials.py
```
