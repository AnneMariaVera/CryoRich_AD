# Numerical simulation of frozen soil hydrology

The notation in this guide follows the publication *Groundwater Recharge or Runoff? Why the Hydraulic Parameterization of Frozen Soils Matters* (Hermann, Drews and Cirpka). Equation numbers of the form "paper eq. (x)" refer to that publication.

**Contents**

- [00 Overview](00_Overview.md) (this file)
- [01 Richards equation](01_Richards_Equation.md)
- [02 Heat transport equation](02_Heat_Transport_Equation.md)
- [03 Hydraulic parameterizations of frozen soil](03_Hydraulic_Parameterizations.md)
- [04 Spatial and temporal discretization](04_Discretization.md)
- [05 Nonlinear solver and time stepping](05_Nonlinear_Solver.md)

## 1. Conceptual overview

![Conceptual model of water flow and heat transport in a partially frozen, sloped soil column](../Figures/concept.png)

Primary variables: pressure head of the pore water ($h_w$) and temperature ($T$).

The effective porosity $n_e$ of a partially frozen soil is the sum of the volumetric contents of ice $\theta_i$, liquid water $\theta_\ell$ and soil gas $\theta_g$ (paper eq. (1)):

$$
n_e = \theta_i + \theta_\ell + \theta_g.
$$

The total water content is $\theta_{tot}=\theta_i+\theta_\ell$.

**Sign convention:** throughout this guide the capillary head is defined by (paper eq. (7))

```math
h_c = \begin{cases}
        -h_w, &\text{if }h_w<0\\
        0, &\text{otherwise,}
    \end{cases}
```

and the total hydraulic head is $h_{tot}=h_w+z$.

## 2. Governing equations

**Richards equation** for water flow in the unsaturated zone (paper eqs. (2) and (3)):

$$\frac{\partial \theta_{tot}}{\partial t}-\nabla \cdot \left(k_{rel}\thinspace K_s\thinspace \nabla(h_w+z)\right)=0\tag{RE}$$

- [01 Richards equation](01_Richards_Equation.md)

**Heat transport equation** for the temperature distribution [1] (paper eq. (4)):

$$\frac{\partial}{\partial t} \left( C_{eq}T - \rho_i H_f \theta_i \right)+ \nabla \cdot \left(\mathbf{q}\thinspace  C_\ell T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0 \tag{HT}$$

- [02 Heat transport equation](02_Heat_Transport_Equation.md)

Both equations are coupled through the liquid and ice contents $\theta_\ell(h_c,T)$, $\theta_i(h_c,T)$ and the relative hydraulic conductivity $k_{rel}(h_c,T)$, which depend on the chosen parameterization, and through the specific discharge $\mathbf{q}$, which transports heat.

- [03 Hydraulic parameterizations of frozen soil](03_Hydraulic_Parameterizations.md)

## 3. Numerical model

- [04 Spatial and temporal discretization](04_Discretization.md): finite-volume formulation, mesh, residuals and boundary conditions
- [05 Nonlinear solver and time stepping](05_Nonlinear_Solver.md): coupled Newton-Raphson scheme, adaptive time stepping and lookup tables

## 4. Forcing (ERA5-Land)

The infiltration rate is computed from the ERA5-Land data as (paper eq. (23))

$$
q_{inf}=P-E-S_{fall}+S_{melt}
$$

with the total precipitation $P$, the evaporation from bare soil $E$, the snowfall $S_{fall}$ and the snowmelt $S_{melt}$. The soil temperature of the first ERA5-Land soil layer (0 cm to 7 cm) is used as surface temperature $T_{surface}$. In the notebooks, both hourly time series are smoothed with a 24-hour moving average. How $q_{inf}$ and $T_{surface}$ enter the model is described in the boundary conditions of [04 Spatial and temporal discretization](04_Discretization.md).

## 5. Model validation

<!-- DRAFT: please check and complete -->

The model is compared to a simulation of the same synthetic case with the permaFoam solver [2] in OpenFOAM, using the impedance-factor parameterization ($\Omega=50$) in both codes. The comparison and a mass balance check are part of the supplementary material (`notebooks/supplementary/`).

<!-- END DRAFT -->

---

[1] A. A. Mohammed, B. L. Kurylyk, E. E. Cey, and M. Hayashi, “Snowmelt Infiltration and Macropore Flow in Frozen Soils: Overview, Knowledge Gaps, and a Conceptual Framework,” _Vadose Zone Journal_, vol. 17, no. 1, p. 180084, 2018, doi: [10.2136/vzj2018.04.0084](https://doi.org/10.2136/vzj2018.04.0084).

[2] L. Orgogozo, T. Xavier, H. Oulbani, and C. Grenier, “Permafrost modelling with OpenFOAM®: New advancements of the permaFoam solver,” _Computer Physics Communications_, vol. 282, p. 108541, Jan. 2023, doi: [10.1016/j.cpc.2022.108541](https://doi.org/10.1016/j.cpc.2022.108541).
