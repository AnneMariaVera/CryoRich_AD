# Explanations to the Richards Equation

$$
\frac{\partial \theta_{tot}}{\partial t}-\nabla \cdot \left(k_{rel}K_s\nabla(h_w+z)\right)=0\tag{RE}
$$

with the total water content $\theta_{tot}=\theta_\ell+\theta_i$ and the pressure head $h_w$ (paper eqs. (2) and (3)).

## 1. Derivation

The equation is derived from the mass balance equation:

> change of stored mass + net mass flux = total source/sink of mass

In a partially frozen soil, water is stored as liquid and as ice, but only the liquid phase moves:

$$
\frac{\partial}{\partial t}\int_V\left(\rho_\ell\theta_\ell+\rho_i\theta_i\right)dV+\int_{\partial V}\rho_\ell\mathbf{q}\cdot\mathbf{n}\thinspace d\partial V = \int_V\rho_\ell S_0\thinspace dV.
$$

With the divergence theorem and infinitesimally small control volumes we get

$$\frac{\partial (\rho_\ell\theta_\ell+\rho_i\theta_i)}{\partial t}+\nabla \cdot (\rho_\ell\mathbf{q})=\rho_\ell S_0.$$

Applying the chain rule to the divergence leads to

$$
\frac{\partial (\rho_\ell\theta_\ell+\rho_i\theta_i)}{\partial t}+\mathbf{q}\cdot\nabla \rho_\ell+\rho_\ell\nabla \cdot \mathbf{q}=\rho_\ell S_0.
$$

It is reasonable to assume that the density of water does not vary in space ($\nabla\rho_\ell=0$) and that the changes of the densities with time are negligible compared to the changes of the water and ice contents, i.e.

$$
\frac{\theta_\ell}{\rho_\ell}\frac{\partial \rho_\ell}{\partial t}\ll\frac{\partial \theta_\ell}{\partial t}.
$$

In addition, the density difference between ice and liquid water is neglected ($\rho_i\approx\rho_\ell$), so that the volume of water does not change when it freezes. Dividing by $\rho_\ell$ gives

$$
\frac{\partial \theta_{tot}}{\partial t}+\nabla \cdot \mathbf{q}=S_0.
$$

We now also neglect the source/sink term and end up at

$$
\frac{\partial \theta_{tot}}{\partial t}+\nabla \cdot \mathbf{q}=0.\tag{1}
$$

## 2. Darcy's law

According to Harlan (1973), Darcy's law can be applied to describe the specific discharge of liquid water $\mathbf{q}$:

$$
\mathbf{q}=-k_{rel}\frac{K_0}{\mu}\nabla p_{tot}
$$

with the intrinsic permeability $K_0\thinspace [m^2]$, the dynamic viscosity of water $\mu\thinspace [Pa\thinspace s]$ and

$$
p_{tot}=\rho_\ell gz+p_w+p_a
$$

where $p_a$ is the atmospheric pressure, assumed constant throughout the entire domain, and $p_w$ is the pressure of the pore water relative to the atmosphere (negative in the unsaturated zone, positive below the water table).
Note here that

$$
p_w=h_w\rho_\ell g \quad\text{and}\quad K_s=K_0\frac{\rho_\ell g}{\mu}.
$$

This way we end up at

$$
\mathbf{q}=-k_{rel}\frac{K_0}{\mu}\nabla (\rho_\ell gz+p_w)=-k_{rel}K_s\nabla (h_w+z)=-k_{rel}K_s\nabla h_{tot}
$$

with the total hydraulic head $h_{tot}=h_w+z$. The product $K_u=k_{rel}K_s$ is the unsaturated hydraulic conductivity, where both freezing and water content act on $k_{rel}$, while $K_s$ is determined by the soil texture. Plugging the flux back into eq. (1) gives the Richards equation

$$
\frac{\partial \theta_{tot}}{\partial t}-\nabla \cdot \left(k_{rel}K_s\nabla(h_w+z)\right)=0.\tag{RE}
$$

In the frozen soil, $k_{rel}=k_{rel}(h_c,T)$ depends on both capillary head and temperature. The total water content $\theta_{tot}(h_c)$ is given by the soil-retention curve, so the equation can also be expressed as

$$
\frac{\partial \theta_{tot}}{\partial h_w}\frac{\partial h_w}{\partial t}-\nabla \cdot \left(k_{rel}K_s\nabla(h_w+z)\right)=0.
$$

See [02.1 Overview Hydraulic Parameters](02.1%20Overview%20Hydraulic%20Parameters.md) for explanations on $k_{rel}$ and $\theta_{tot}(h_c)$.

---

R. L. Harlan, “Analysis of coupled heat-fluid transport in partially frozen soil,” _Water Resources Research_, vol. 9, pp. 1314–1323, 1973, doi: [10.1029/WR009i005p01314](https://doi.org/10.1029/WR009i005p01314).
