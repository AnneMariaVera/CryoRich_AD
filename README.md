# CryoRich_AD

Author
------
Anne Hermann  
University of Tübingen  
Email: anne.hermann@uni-tuebingen.de  

Short description
-----------------

This repository contains two Jupyter notebooks and utility modules used to solve the Richards' equation coupled with the heat transport equation for partly frozen soils. The project is organised to keep the numerical solver and helper functions inside a small `utils/` package and the interactive work inside notebooks such as `RichardsSolver_AD.ipynb`.

Repository structure
--------------------

- `RichardsSolver_AD.ipynb` - Jupyter notebook containing an example run for a pure Richards' solver (not coupled to the heat transport).
- `CoupledSolver_AD.ipynb` - Jupyter notebook containing an example run for the coupled Richards' and heat transport solver.
- `utils/` - Python package with helper modules:
  - `geometry.py` - main file containing the definition of a Geom() class used for simulations.
  - `petrophysic.py` - petrophysical relations (e.g. water retention, frozen fraction).
  - `volbal.py` - balance equation helper functions.
  - `__init__.py` - package initializer.
- `docu/` - documentation folder containing the mathematical theory guide and figures. This folder will be further developed to include code documentation.

Quick start
-------------------------

1. Create and activate a Conda environment (recommended). Example using conda:

```bash
# from the repository root
conda create -n frozensoil python=3.10 -y
conda activate frozensoil
```

2. Install typical scientific Python packages. There is no `frozensoil.env` in this repo by default, but the notebook and utilities expect packages such as:

```bash
conda install numpy scipy matplotlib jupyterlab ipykernel -c conda-forge
```

This code makes use of automatic-differentiation via JAX. JAX install commands depend on your platform (CPU, GPU, or macOS/Apple Silicon) and on the selected versions of `jax`/`jaxlib`. See the official JAX installation guide for the correct command for your system:

- JAX installation guide: https://github.com/google/jax#installation

Usage notes
-----------

- The utilities codes can be adapted. If you modify modules under `utils/`, restart the notebook kernel.

Documentation
-------------

The `docu/` folder contains a mathematical theory guide (derivations and explanations) and `Figures/`. Over time this folder will be expanded with futher markdown files on code documentation.


TODOs:
------------------------------

- Add a `environment.yml` to pin dependencies.
- Add capillary bundel model
