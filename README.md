# CryoRich_AD

Author
------
Anne Hermann  
University of Tübingen  
Email: anne.hermann@uni-tuebingen.de  

Short description
-----------------

This repository contains several notebooks and utility modules used to solve the Richards equation coupled with the heat transport equation for partially frozen soils. The project is organised to keep the numerical solver and helper functions inside the `utils/` folder and the interactive work inside notebooks such as `notebooks/idealized/Idealized_cap.ipynb`.

Repository structure
--------------------

- `figures/` - contains all published figures (except the overview figure created with Affinity Designer).
- `input/` - interpolations are saved here as pickle files (due to their large size, they cannot be pushed to GitHub; use `notebooks/creating_figures/compare_param.ipynb` to recreate them), along with the ERA5-Land derived forcing containing temperature and q_inf data.
- `notebooks/` - all Jupyter notebooks are stored here:
  - `creating_figures/` - contains the notebooks used for creating the figures stored under `figures/`.
  - `idealized/` - contains 5 notebooks (one for each parametrization) that were used to create the idealized simulations.
  - `simulation/` - contains 5 notebooks (one for each parametrization) that were used to create the ERA5-Land simulations.
  - `spinup/` - contains 5 notebooks (one for each parametrization) that were used to create the spinup simulations.
  - `supplementary/` - contains two notebooks: one for running the synthetic case simulated for the supplementary material, and one for creating the figure and calculating the mass balance.
- `utils/` - Python package with helper modules:
  - `geometry.py` - main file containing the definition of the `Geom()` class used for simulations.
  - `parameter_interpolation.py` - interpolation of hydraulic parametrizations along temperature and capillary head.
  - `parse_permafoam_output.py` - contains a helper function converting permaFoam output to a pickle file.
  - `petrophysic.py` - petrophysical relations (e.g. water retention, frozen fraction).
  - `petrophysics_capillaryBundles.py` - petrophysical relations for the capillary bundle model.
  - `plotting_utils.py` - Python functions that were used to create the figures for publication.
  - `volbal.py` - balance equation helper functions.
  - `volbal_validation.py` - balance equation helper functions used for the synthetic case of the supplementary material. They need to be redefined because the BCs differ from the ones in `volbal.py`.
  - `__init__.py` - package initializer.
- `docu/` - documentation folder containing the mathematical theory guide and figures. This folder will be further developed to include code documentation.

Quick start
-----------

Create a Conda environment from the provided `environment.yml` file (recommended):

```bash
# from the repository root
conda env create -f environment.yml
conda activate cryorich_AD
```
By default, JAX is installed for CPU use. For GPU support, see the JAX installation guide: https://github.com/google/jax#installation

Parametrizations
----------------

Five parametrizations of the hydraulic properties of partially frozen soil are compared. The table maps the names used in the code to the labels used in the publication:

| Label | Description | Name in code (`<param>`) |
|-------|-------------|--------------------------|
| (A) | drying assumption | `drying` |
| (B) | drying assumption with impedance factor Ω = 10 | `drying_omega10` |
| (C) | impedance factor Ω = 10 | `imp10` |
| (D) | impedance factor Ω = 50 | `imp50` |
| (E) | capillary bundle model | `cap` |

The same name is used for the `parametrization` variable, the notebook names and all interpolation and output files, e.g. `notebooks/spinup/0Spinup_imp10.ipynb` → `output/spinup/spinup_imp10.pkl`.

Steps to reproduce
------------------

To rerun the simulations of the publication, run the notebooks in the following order (`<param>` stands for one of the five parametrizations):

0. *(optional, the forcings are already included under `input/era5_land_data/Sweden/` named `sim2312_2405.pkl` and `spinup_2308_2311.pkl`)*

ERA5-Land forcing:
   - `input/era5_land_data/download_data.ipynb` downloads ERA5-Land data ([doi:10.24381/cds.e2161bac](https://doi.org/10.24381/cds.e2161bac)). Requests are only possible per month, so the notebook is run once per month by changing `year` and `month`. Two requests are needed per month:
     - skin temperature, snowfall, snowmelt, evaporation from bare soil and total precipitation → saved as `YY_MM.nc`
     - skin temperature, 2 m temperature and soil temperature level 1 → saved as `T_YY_MM.nc`
   - The download assigns a random file name to each download. The files were renamed by hand to the scheme above (e.g. `23_05.nc`, `T_23_05.nc`) and moved to `input/era5_land_data/Sweden/`.
   - `input/era5_land_data/creat_sim_data.ipynb` → `input/era5_land_data/Sweden/spinup_2308_2311.pkl` and `sim_2312_2405.pkl`
1. `notebooks/creating_figures/compare_param.ipynb` → `input/interpolation/interpolation_<param>.pkl`
   (~9 GB per file, ~47 GB in total)
2. `notebooks/spinup/0Spinup_<param>.ipynb` → `output/spinup/`
   (the spinup output is used as the initial condition for both the ERA5 and the idealized simulations)
3. `notebooks/simulation/1Simulate_<param>.ipynb` → `output/simulation/`
   `notebooks/idealized/Idealized_<param>.ipynb` → `output/idealized/`
4. `notebooks/creating_figures/visualize_results.ipynb` → `figures/`
5. Supplementary material: `notebooks/supplementary/synthetic_case.ipynb` → `output/supplementary/`, then `notebooks/supplementary/create_supplementary_plots.ipynb` → `figures/supplementary/`.
   The synthetic case is compared to a permaFoam simulation conducted in OpenFOAM, whose case folder is stored under `output/supplementary/permaFoam/`.

If you modify modules under `utils/`, make sure to restart the notebook kernel.

Documentation
-------------

The `docu/` folder contains a mathematical theory guide (derivations and explanations) and `Figures/`.
