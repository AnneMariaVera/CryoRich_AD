# Explanations to the Richards Equation

$$
\frac{\partial \theta}{\partial t}-\nabla \cdot \left(k_{rel}K_s\nabla(z-h_c)\right)=0\tag{RE}
$$

with the total water content $\theta=\theta_l+\theta_i$ and the capillary head $h_c\geq 0$ (suction).

## 1. Derivation

The equation is derived from the mass balance equation:

> change of stored mass + net mass flux = total source/sink of mass

In a partially frozen soil, water is stored as liquid and as ice, but only the liquid phase moves:

$$
\frac{\partial}{\partial t}\int_V\left(\rho_l\theta_l+\rho_i\theta_i\right)dV+\int_{\partial V}\rho_l\mathbf{q}\cdot\mathbf{n}\,d\partial V = \int_V\rho_l S_0\,dV.
$$

With the divergence theorem and infinitesimally small control volumes we get

$$\frac{\partial (\rho_l\theta_l+\rho_i\theta_i)}{\partial t}+\nabla \cdot (\rho_l\mathbf{q})=\rho_lS_0.$$

Applying the chain rule to the divergence leads to

$$
\frac{\partial (\rho_l\theta_l+\rho_i\theta_i)}{\partial t}+\mathbf{q}\cdot\nabla \rho_l+\rho_l\nabla \cdot \mathbf{q}=\rho_lS_0.
$$

It is reasonable to assume that the density of water does not vary in space ($\nabla\rho_l=0$) and that the changes of the densities with time are negligible compared to the changes of the water and ice contents, i.e.

$$
\frac{\theta_l}{\rho_l}\frac{\partial \rho_l}{\partial t}\ll\frac{\partial \theta_l}{\partial t}.
$$

In addition, the density difference between ice and liquid water is neglected ($\rho_i\approx\rho_l$), so that the volume of water does not change when it freezes. Dividing by $\rho_l$ gives

$$
\frac{\partial \theta}{\partial t}+\nabla \cdot \mathbf{q}=S_0.
$$

We now also neglect the source/sink term and end up at

$$
\frac{\partial \theta}{\partial t}+\nabla \cdot \mathbf{q}=0.\tag{1}
$$

## 2. Darcy's law

The flux is given by Darcy's law

$$
\mathbf{q}=-k_{rel}\frac{K_0}{\mu}\nabla p_{tot}
$$

with the intrinsic permeability $K_0\,[m^2]$, the dynamic viscosity of water $\mu\,[Pa\,s]$ and

$$
p_{tot}=\rho_l gz-p_c+p_a+p_h
$$

where $p_a$ is the atmospheric pressure, assumed constant throughout the entire domain, $p_h$ is the pressure of the overlying water column (0 in the unsaturated zone) and $p_c\geq0$ is the capillary pressure (suction, 0 in the saturated zone).
Note here that

$$
p_c=h_c\rho_l g \quad\text{and}\quad K_s=K_0\frac{\rho_l g}{\mu}.
$$

This way, in the unsaturated zone, we end up at

$$
\mathbf{q}=-k_{rel}\frac{K_0}{\mu}\nabla (\rho_l gz-p_c)=-k_{rel}K_s\nabla (z-h_{c})=-k_{rel}K_s\nabla H
$$

with the total hydraulic head $H=z-h_c$. Plugging the flux back into eq. (1) gives

$$
\frac{\partial \theta}{\partial t}-\nabla \cdot \left(k_{rel}K_s\nabla(z-h_c)\right)=0.\tag{RE}
$$

In the frozen soil, $k_{rel}=k_{rel}(h_c,T)$ depends on both capillary head and temperature. The total water content $\theta(h_c)$ is given by the water retention curve, so the equation can also be expressed as

$$
\frac{\partial \theta}{\partial h_c}\frac{\partial h_c}{\partial t}-\nabla \cdot \left(k_{rel}K_s\nabla(z-h_c)\right)=0.
$$

See [02.1 Overview Hydraulic Parameters](02.1%20Overview%20Hydraulic%20Parameters.md) for explanations on $k_{rel}$ and $\theta(h_c)$.
