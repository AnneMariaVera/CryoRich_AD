# Explanations to the Heat Transport equation

$$\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\rho_l c_l \mathbf{q} T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0 \tag{HT}
$$

with $T$ in °C and

- $C_{eq}=\rho_lc_l\theta_l+\rho_ic_i\theta_i+\rho_ac_a\theta_a+\rho_sc_s(1-\theta_s)\;[\frac{J}{m^3K}]=[\frac{kg}{ms^2K}]$, the equivalent volumetric heat capacity, where $\theta_a=\theta_s-\theta_l-\theta_i$ is the air content and the porosity is given by $\theta_s$.
- $\lambda_{eq} = \lambda_{l}^{\theta_l} \lambda_{i}^{\theta_i} \lambda_{a}^{\theta_a} \lambda_{s}^{1-\theta_s}\;[\frac{W}{mK}]=[\frac{kg\, m }{s^3K}]$, the equivalent thermal conductivity (geometric mean).
- $H_f\,[\frac{J}{kg}]$, the latent heat of fusion.
- $\mathbf{q}\,[\frac{m}{s}]$, the water flux from the Richards equation.

**Unit check:** each term has the unit $\frac{kg}{ms^3} = \frac{J}{sm^3}$.

<!-- DRAFT: default values taken from Geom.set_thermal_params, please check -->

**Default values:**

| Phase | Density $\rho\,[kg\,m^{-3}]$ | Specific heat capacity $c\,[J\,kg^{-1}K^{-1}]$ | Thermal conductivity $\lambda\,[W\,m^{-1}K^{-1}]$ |
|-------|------|------|-------|
| liquid water ($l$) | 1000 | 4180 | 0.6 |
| ice ($i$) | 917 | 2100 | 2.2 |
| air ($a$) | 1.1 | 1000 | 0.023 |
| solid ($s$) | 2650 | 800 | 2.5 |

and $H_f = 3.34\cdot10^{5}\,J\,kg^{-1}$.

<!-- END DRAFT -->

## 1. Derivation

The heat transport equation is derived from an energy balance, where energy can be transported convectively (advection with the water flux) and conductively (heat conduction). After applying the divergence theorem, the energy balance equation is given by

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

where the latent part is negative because the latent heat $H_f$ is released when water freezes, so ice holds less energy than liquid water at the same temperature. The energy flux is

$$
\mathbf{f} = \mathbf{f}_{conv}+\mathbf{f}_{cond}=\rho_l c_l \mathbf{q} T-\lambda_{eq} \nabla T.
$$

Assuming that there are no internal sources or sinks ($S=0$), this results in

$$
\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)+ \nabla \cdot \left(\rho_l c_l \mathbf{q} T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0. \tag{HT}
$$

## 2. Apparent heat capacity

Under the assumption of thermal equilibrium (phase change is instantaneous), the ice content is a function of temperature, $\theta_i=\theta_i(T)$. Neglecting the change of $C_{eq}$ with time and changes of $\theta_i$ due to changes in $h_c$, the storage term can be written as

$$
\frac{\partial}{\partial t} \left(  C_{eq}T - \rho_i H_f \theta_i\right)\approx C_{eq}\frac{\partial T}{\partial t} - \rho_i H_f \frac{\partial \theta_i}{\partial T}\frac{\partial T}{\partial t}=\left(  C_{eq} - \rho_i H_f \frac{\partial \theta_i}{\partial T}\right)\frac{\partial T}{\partial t}.
$$

Since $\frac{\partial \theta_i}{\partial T}<0$, the apparent heat capacity $C_{app}=C_{eq} - \rho_i H_f \frac{\partial \theta_i}{\partial T}$ is larger than $C_{eq}$ within the freezing range.

**Note:** the code does not use the apparent heat capacity. It discretises the energy $E$ directly (see [04 Finite-Volume Method](04%20Finite-Volume%20Method.md)), which conserves energy (up to the solver tolerance) and needs none of the assumptions above.
