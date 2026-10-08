# Explanations to the Heat Transport equation

$$\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\mathbf{q}\, C_\ell T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0 \tag{HT}
$$

with $T$ in °C and

- $\rho_i\,[kg\,m^{-3}]$, the mass density of ice,
- $H_f\,[m^2s^{-2}]=[J\,kg^{-1}]$, the latent heat of fusion,
- $C_\ell=\rho_\ell c_\ell\,[kg\,m^{-1}s^{-2}°C^{-1}]$, the volumetric heat capacity of liquid water,
- $\mathbf{q}\,[m\,s^{-1}]$, the specific discharge of liquid water from the Richards equation,
- $C_{eq}$, the equivalent volumetric heat capacity, given by the volume-weighted arithmetic mean over the ice, liquid, gas and matrix fractions (paper eq. (5))

$$
C_{eq}=\sum_\gamma\rho_\gamma c_\gamma\theta_\gamma=\rho_i c_i\theta_i+\rho_\ell c_\ell\theta_\ell+\rho_g c_g\theta_g+\rho_m c_m(1-n_e)\quad[kg\,m^{-1}s^{-2}°C^{-1}],
$$

- $\lambda_{eq}$, the equivalent thermal conductivity, given by the corresponding volume-weighted geometric mean (paper eq. (6))

$$
\lambda_{eq}=\prod_\gamma\lambda_\gamma^{\theta_\gamma}=\lambda_i^{\theta_i}\lambda_\ell^{\theta_\ell}\lambda_g^{\theta_g}\lambda_m^{1-n_e}\quad[kg\,m\,s^{-3}°C^{-1}],
$$

with the mass density $\rho_\gamma\,[kg\,m^{-3}]$, the specific heat capacity $c_\gamma\,[m^2s^{-2}°C^{-1}]$ and the thermal conductivity $\lambda_\gamma\,[kg\,m\,s^{-3}°C^{-1}]$ of phase $\gamma$. The gas content is $\theta_g=n_e-\theta_\ell-\theta_i$, and in the code the effective porosity $n_e$ is the saturated water content $\theta_s$.

**Unit check:** each term has the unit $\frac{kg}{ms^3} = \frac{J}{sm^3}$.

<!-- DRAFT: values taken from the simulation notebooks (notebooks/spinup, simulation, idealized, supplementary), please check -->

**Values used in the simulations** (set in the notebooks and passed to `Geom.set_thermal_params`; the defaults in the code differ):

| Phase | $\rho_\gamma\,[kg\,m^{-3}]$ | $\rho_\gamma c_\gamma\,[J\,m^{-3}K^{-1}]$ | $\lambda_\gamma\,[W\,m^{-1}K^{-1}]$ |
|-------|------|------|-------|
| gas ($g$) | 1.1 | 1228.08 | 0.0262 |
| ice ($i$) | 917 | $1.8952\cdot10^{6}$ | 2.14 |
| liquid water ($\ell$) | 998 | $4.182\cdot10^{6}$ | 0.6 |
| matrix ($m$) | 2650 | $1.92372\cdot10^{6}$ | 2.9 |

and $\rho_iH_f = 3.34\cdot10^{8}\,J\,m^{-3}$, i.e. $H_f\approx3.64\cdot10^{5}\,J\,kg^{-1}$.

<!-- END DRAFT -->

## 1. Derivation

As local thermal equilibrium between all phases is assumed, a single integrated temperature field is solved for. The heat transport equation is derived from an energy balance, where energy can be transported convectively (advection with the specific discharge) and conductively (heat conduction). After applying the divergence theorem, the energy balance equation is given by

$$
\frac{\partial}{\partial t}\int_V E \,dV+\int_{ V}\nabla\cdot \mathbf{f}\,d V = \int_V S \,dV.
$$

And again for infinitesimally small $V$ we get

$$
\frac{\partial}{\partial t} E +\nabla\cdot \mathbf{f} =  S.
$$

The energy per volume (relative to liquid water at 0 °C) is given by

$$
E = E_{sensible}+E_{latent}=C_{eq}T - \rho_i H_f \theta_i,
$$

where the latent part is negative because the latent heat $H_f$ is released when water freezes, so ice holds less energy than liquid water at the same temperature.

The energy flux is

$$
\mathbf{f} = \mathbf{f}_{conv}+\mathbf{f}_{cond}=\mathbf{q}\,C_\ell T-\lambda_{eq} \nabla T.
$$

Assuming that there are no internal sources or sinks ($S=0$), this results in

$$
\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\mathbf{q}\,C_\ell T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0. \tag{HT}
$$

## 2. Apparent heat capacity

Under the assumption of thermal equilibrium (phase change is instantaneous), the ice content is a function of temperature, $\theta_i=\theta_i(T)$. Neglecting the change of $C_{eq}$ with time and changes of $\theta_i$ due to changes in $h_c$, the storage term can be written as

$$
\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)\approx C_{eq}\frac{\partial T}{\partial t} - \rho_i H_f \frac{\partial \theta_i}{\partial T}\frac{\partial T}{\partial t}=\left(  C_{eq} - \rho_i H_f \frac{\partial \theta_i}{\partial T}\right)\frac{\partial T}{\partial t}.
$$

Since $\frac{\partial \theta_i}{\partial T}<0$, the apparent heat capacity $C_{app}=C_{eq} - \rho_i H_f \frac{\partial \theta_i}{\partial T}$ is larger than $C_{eq}$ within the freezing range.

**Note:** the code does not use the apparent heat capacity. It discretises the energy $E$ directly (see [04 Finite-Volume Method](04%20Finite-Volume%20Method.md)), which conserves energy (up to the solver tolerance) and needs none of the assumptions above.
