# Numerical simulation of frozen soil hydrology

## 1. Conceptual Overview

![Conceptual model of water flow and heat transport in a partially frozen, sloped soil column](../Figures/concept.png)

Primary variables: total hydraulic head ($H = z - h_c$) and temperature ($T$).

**Sign convention:** throughout this guide the capillary head $h_c \geq 0$ denotes suction (positive in the unsaturated zone, zero at saturation). The total hydraulic head is therefore $H = z - h_c$.

## 2. Governing equations

**Richards equation** for water flow in the variably saturated zone, written for the total water content $\theta = \theta_l + \theta_i$:

$$\frac{\partial \theta}{\partial t}-\nabla \cdot \left(k_{rel}(h_c,T)\,K_s\,\nabla(z-h_c)\right)=0\tag{RE}$$

- [02 Explanations to the Richards Equation](02%20Explanations%20to%20the%20Richards%20Equation.md)
- [02.1 Overview Hydraulic Parameters](02.1%20Overview%20Hydraulic%20Parameters.md)

**Heat transport equation** for the temperature distribution [1]:

$$\frac{\partial}{\partial t} \left( C_{eq}T - \rho_i H_f \theta_i \right)+ \nabla \cdot \left(\rho_l c_l \mathbf{q} T \right)- \nabla \cdot \left( \lambda_{eq} \nabla T \right) = 0 \tag{HT}$$

- [03 Explanations to the Heat Transport equation](03%20Explanations%20to%20the%20Heat%20Transport%20equation.md)

Both equations are coupled through the liquid and ice contents $\theta_l(h_c,T)$, $\theta_i(h_c,T)$ and the relative permeability $k_{rel}(h_c,T)$, which depend on the chosen parametrization (see [02.1](02.1%20Overview%20Hydraulic%20Parameters.md)), and through the water flux $\mathbf{q}$, which transports heat.

## 3. Numerical Solver (Finite Volume Method)

For the formulation of the residual functions of both equations see [04 Finite-Volume Method](04%20Finite-Volume%20Method.md).

<!-- DRAFT: section below written from geometry.py (simulate_cap_bundle, newtonRaphson_cap_bundle), please check -->

Both equations are solved fully coupled and fully implicitly. The vector of unknowns is

$$
U=(H, y)^T, \qquad y = \operatorname{arcsinh}\left(\frac{T}{b}\right),\; b = 0.1\,°C,
$$

where the temperature is transformed with $y$ to improve the convergence around the freezing point. The residual vector is $R=(R_H,R_T)^T$ and the coupled Jacobian is

$$
J = \begin{pmatrix}\frac{\partial R_H}{\partial H} & \frac{\partial R_H}{\partial y}\\ \frac{\partial R_T}{\partial H} & \frac{\partial R_T}{\partial y}\end{pmatrix}.
$$

The Jacobian is computed by automatic differentiation (JAX) using its known sparsity pattern (`sparsejac`). In each Newton-Raphson iteration the sparse linear system

$$
J\Delta U=-R
$$

is solved and $U$ is updated. A time step is accepted once each residual component is below either its absolute or its relative tolerance (with respect to the residual of the first iteration).

**Adaptive time stepping:**
- If the Newton-Raphson iteration does not converge within the maximum number of iterations (or the residual becomes NaN/inf), the time step is halved and repeated.
- If it converges within the minimum number of iterations for more than two consecutive time steps, the time step is increased by a factor of 1.2, up to a prescribed maximum time step.

<!-- END DRAFT -->

## 4. Model Validation

<!-- DRAFT: please check and complete -->

The model is compared to a simulation of the same synthetic case with the permaFoam solver [2] in OpenFOAM, using the impedance-factor parametrization in both codes. The comparison and a mass balance check are part of the supplementary material (`notebooks/supplementary/`).

<!-- END DRAFT -->

---

[1] A. A. Mohammed, B. L. Kurylyk, E. E. Cey, and M. Hayashi, “Snowmelt Infiltration and Macropore Flow in Frozen Soils: Overview, Knowledge Gaps, and a Conceptual Framework,” _Vadose Zone Journal_, vol. 17, no. 1, p. 180084, 2018, doi: [10.2136/vzj2018.04.0084](https://doi.org/10.2136/vzj2018.04.0084).

[2] L. Orgogozo, T. Xavier, H. Oulbani, and C. Grenier, “Permafrost modelling with OpenFOAM®: New advancements of the permaFoam solver,” _Computer Physics Communications_, vol. 282, p. 108541, Jan. 2023, doi: [10.1016/j.cpc.2022.108541](https://doi.org/10.1016/j.cpc.2022.108541).
