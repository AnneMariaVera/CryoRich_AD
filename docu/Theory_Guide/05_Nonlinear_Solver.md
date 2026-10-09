# Nonlinear solver and time stepping

For the formulation of the residual functions of both the Richards and heat transport equation see [04 Spatial and temporal discretization](04_Discretization.md).

## 1. Coupled Newton-Raphson scheme

Both equations are solved fully coupled with implicit Euler time integration. In the code, the hydraulic unknown is the total head $h_{tot}=h_w+z$, from which $h_w$ and $h_c$ follow. To improve convergence when gradients along temperature are large (freezing and thawing), the temperature is transformed to

$$
y = \mathrm{arcsinh}\left(\frac{T}{b}\right), \qquad b = 0.1,
$$

prior to the Newton iteration and transformed back upon convergence. The vector of unknowns is thus $U=(h_{tot}, y)^T$, the residual vector is $R=(R_h,R_T)^T$ and the coupled Jacobian is

```math
J = \begin{pmatrix}\frac{\partial R_h}{\partial h_{tot}} & \frac{\partial R_h}{\partial y}\\ \frac{\partial R_T}{\partial h_{tot}} & \frac{\partial R_T}{\partial y}\end{pmatrix}.
```

The Jacobian is computed by automatic differentiation (JAX) using its known sparsity pattern (`sparsejac`). In each Newton-Raphson iteration the sparse linear system

$$
J\Delta U=-R
$$

is solved. The update $U+\alpha\Delta U$ uses a backtracking line search (as in DuMux): starting with $\alpha=1$, $\alpha$ is halved until the residual norm no longer increases (down to $\alpha=10^{-6}$).

**Convergence:** a time step is accepted once each residual (Richards and heat transport equation, evaluated separately) is below either its absolute tolerance or its relative tolerance with respect to the residual of the first iteration.

## 2. Adaptive time stepping

- If the Newton-Raphson iteration does not converge within the maximum number of iterations (or the residual becomes NaN/inf), the time step is halved and repeated.
- If it converges within the minimum number of iterations for more than two consecutive time steps, the time step is increased by a factor of 1.2, up to a prescribed maximum time step.

## 3. Lookup tables

Evaluating the parameterizations (in particular the integrals of the capillary bundle model, see [03 Hydraulic parameterizations of frozen soil](03_Hydraulic_Parameterizations.md)) during the simulation would be too expensive. Therefore $\theta_\ell$ and $k_{rel}$ are precomputed once for each soil type on a grid of capillary heads $h_c$ (from $S\in(0,1)$) and temperatures $T$, and stored as 2D interpolators (`interpax.Interpolator2D`, monotonic interpolation). These lookup tables are saved under `input/interpolation/` and evaluated in each Newton-Raphson iteration. Since the interpolators are differentiable, the Jacobian can still be computed by automatic differentiation.