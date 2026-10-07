# CryoRich_AD

Author
------
Anne Hermann  
University of Tübingen  
Email: anne.hermann@uni-tuebingen.de  

Short description
-----------------

This repository contains several notebooks and utility modules used to solve the Richards' equation coupled with the heat transport equation for partialy frozen soils. The project is organised to keep the numerical solver and helper functions inside the `utils/` folder and the interactive work inside notebooks such as `./notebooks/idealized_cap.ipynb`.

Repository structure
--------------------

- `figures`- containes all published figures (except the overview figure created with Affinity Designer).-
- `input`- interpolations are saved here in pickle files (due to their large size, they can not be pushed to github, use ./notebook/creating_figures/compare_param.ipynb to recreate them), along with the ER5-land derived forcing containg temperature and q_inf data.
- `notebooks/` - here all Jupyter notebooks are stored 
  - `creating_figures` - containing notebooks used for creating the figures stored under `figures/`.
  - `idealized` - containes 5 notebookes (one for each parametrization) that were used to create the idealized simulations
  - `simulation` - containes 5 notebookes (one for each parametrization) that were used to create the ERA5-land simulations
  - `spinup` - containes 5 notebookes (one for each parametrization) that were used to create the spinup simulations
  - `supplementary` - containes two notebooks, one for the simulation run of the synthetic case simulated for the supplementary material and one for creating the figrue and the mass balance calculation. 
- `utils/` - Python package with helper modules:
  - `geometry.py` - main file containing the definition of a Geom() class used for simulations.
  - `parameter_interpolation.py`- interpolation of hydraulic parameterizations along temperature and capillary head.
  - `parse_permafoam_output.py`- containes helper function converting permaFoam output to pickle file.
  - `petrophysic.py` - petrophysical relations (e.g. water retention, frozen fraction).
  - `petrophysics_capillaryBundles.py` - petrophysical relations for the capillary bundle model. 
  - `plotting_utils.py`- python functions that were used to create figures for publication. 
  - `volbal.py` - balance equation helper functions.
  - `volbal_validation.py` - balance equation helper functions (this is the function used for the synthetic case of the supplementary material. It need to be redefined as the BCs differ from the once in volbal.py)
  - `__init__.py` - package initializer.
- `docu/` - documentation folder containing the mathematical theory guide and figures. This folder will be further developed to include code documentation.

Quick start
-------------------------

1. Create a Conda environment from the provided `environment.yml` file (recommended). Example using conda:

```bash
# from the repository root
conda env create -f environment.yml
conda activate cryorich_AD
```
JAX is per default installed for CPU use, check out the JAX installation guide for GPU installation: https://github.com/google/jax#installation

Usage notes
-----------

- The utilities codes can be adapted. If you modify modules under `utils/`, make sure to restart the notebook kernel.
- If you want to rerun the simulations created for the publication, you first need to create the interpolation files with the notebook `compare_param.ipynb` saved under `./notebooks/creating_figrues`.
- First run the spinup period (which will be saved under `./output/spinup`), the output will be used as the initial condition for both the idealized and ERA5 simulations.
- The simulation conducted for supplementary material is compared to a permaFoam simulation. The folder of this simulation conducted in openFoam is saved under `./output/supplementary/permaFoam`.

Documentation
-------------

The `docu/` folder contains a mathematical theory guide (derivations and explanations) and `Figures/`. 