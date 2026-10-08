# Heat transport equation

$$\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\mathbf{q}\thinspace  C_\ell T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0 \tag{HT}
$$

with $T$ in °C and

- $\rho_i\thinspace [kg\thinspace m^{-3}]$, the mass density of ice,
- $H_f\thinspace [m^2s^{-2}]=[J\thinspace kg^{-1}]$, the latent heat of fusion,
- $C_\ell=\rho_\ell c_\ell\thinspace [kg\thinspace m^{-1}s^{-2}°C^{-1}]$, the volumetric heat capacity of liquid water,
- $\mathbf{q}\thinspace [m\thinspace s^{-1}]$, the specific discharge of liquid water from the Richards equation,
- $C_{eq}$ and $\lambda_{eq}$, the equivalent volumetric heat capacity and thermal conductivity (see section 2).

**Unit check:** each term has the unit $\frac{kg}{ms^3} = \frac{J}{sm^3}$.

## 1. Derivation

As local thermal equilibrium between all phases is assumed, we solve for a single integrated temperature field. The heat transport equation is derived from an energy balance, where energy can be transported convectively (advection with the specific discharge) and conductively. After applying the divergence theorem, the energy balance equation is given by

$$
\frac{\partial}{\partial t}\int_V E \thinspace dV+\int_{ V}\nabla\cdot \mathbf{f}\thinspace d V = \int_V S \thinspace dV.
$$

And again for infinitesimally small $V$ we get

$$
\frac{\partial}{\partial t} E +\nabla\cdot \mathbf{f} =  S.
$$

The energy per volume (relative to liquid water at 0 °C) is given by

$$
E = E_{sensible}+E_{latent}=C_{eq}T - \rho_i H_f \theta_i,
$$

where the latent part is negative because the latent heat $H_f$ is released when water freezes.

The energy flux is

$$
\mathbf{f} = \mathbf{f}_{conv}+\mathbf{f}_{cond}=\mathbf{q}\thinspace C_\ell T-\lambda_{eq} \nabla T.
$$

Assuming that there are no internal sources or sinks ($S=0$), this results in

$$
\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\mathbf{q}\thinspace C_\ell T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0. \tag{HT}
$$

## 2. Equivalent thermal properties

The equivalent volumetric heat capacity $C_{eq}$ is given by the volume-weighted arithmetic mean over the ice, liquid, gas and matrix fractions (paper eq. (5))

$$
C_{eq}=\sum_\gamma\rho_\gamma c_\gamma\theta_\gamma=\rho_i c_i\theta_i+\rho_\ell c_\ell\theta_\ell+\rho_g c_g\theta_g+\rho_m c_m(1-n_e)\quad[kg\thinspace m^{-1}s^{-2}°C^{-1}],
$$

and the equivalent thermal conductivity $\lambda_{eq}$ by the corresponding volume-weighted geometric mean (paper eq. (6))

$$
\lambda_{eq}=\prod_\gamma\lambda_\gamma^{\theta_\gamma}=\lambda_i^{\theta_i}\lambda_\ell^{\theta_\ell}\lambda_g^{\theta_g}\lambda_m^{1-n_e}\quad[kg\thinspace m\thinspace s^{-3}°C^{-1}],
$$

with the mass density $\rho_\gamma\thinspace [kg\thinspace m^{-3}]$, the specific heat capacity $c_\gamma\thinspace [m^2s^{-2}°C^{-1}]$ and the thermal conductivity $\lambda_\gamma\thinspace [kg\thinspace m\thinspace s^{-3}°C^{-1}]$ of phase $\gamma$. The gas content is $\theta_g=n_e-\theta_\ell-\theta_i$, and in the code the effective porosity $n_e$ is the saturated water content $\theta_s$.

<!-- DRAFT: values taken from the simulation notebooks (notebooks/spinup, simulation, idealized, supplementary), please check -->

**Values used in the simulations** (set in the notebooks and passed to `Geom.set_thermal_params`):

| Phase | $\rho_\gamma\thinspace [kg\thinspace m^{-3}]$ | $\rho_\gamma c_\gamma\thinspace [J\thinspace m^{-3}K^{-1}]$ | $\lambda_\gamma\thinspace [W\thinspace m^{-1}K^{-1}]$ |
|-------|------|------|-------|
| gas ($g$) | 1.1 | 1228.08 | 0.0262 |
| ice ($i$) | 917 | $1.8952\cdot10^{6}$ | 2.14 |
| liquid water ($\ell$) | 998 | $4.182\cdot10^{6}$ | 0.6 |
| matrix ($m$) | 2650 | $1.92372\cdot10^{6}$ | 2.9 |

and $\rho_iH_f = 3.34\cdot10^{8}\thinspace J\thinspace m^{-3}$, i.e. $H_f\approx3.64\cdot10^{5}\thinspace J\thinspace kg^{-1}$.

See [04 Spatial and temporal discretization](04_Discretization.md) for the details on the discretization of the heat transport equation. 