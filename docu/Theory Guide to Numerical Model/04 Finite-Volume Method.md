# Finite-Volume Method

## 1. General Derivation

Assume the governing equation is given by

$$
\frac{\partial m}{\partial t}+\nabla \cdot \mathbf{q}=0
$$

with a stored quantity $m$ and its flux $\mathbf{q}$. We now look at a single cell $i$ with area $A$:

$$
\int_A\frac{\partial m}{\partial t}\,dA+\int_A\nabla \cdot \mathbf{q}\,dA=0
$$

$$
\Leftrightarrow\frac{\partial}{\partial t}\int_Am\,dA+\int_{\partial A}\mathbf{n} \cdot \mathbf{q}\,d\partial A=0
$$

with the outward unit normal $\mathbf{n}$. For $m$ constant inside the cell and $\mathbf{q}$ constant over each face of the boundary we get

$$
\frac{\partial m}{\partial t}A+\sum_{f\in\{l,r,t,b\}}\ell_f\,q_n^f=0
$$

where $\ell_f$ is the length of face $f$ (left, right, top, bottom) and $q_n^f$ the flux normal to the face (positive out of the cell).

### 1.1 Mesh

<!-- DRAFT: written from Geom.create_grid and volbal.py, please check -->

The domain is a sloped soil column with slope $s$. The mesh is sheared: each row of cells follows the slope ($Z = Z_{flat} + sX$), the left and right faces are vertical, and the top and bottom faces are tilted. All cells have the area $A=\Delta x\Delta z$ and the face lengths

$$
\ell_l=\ell_r=\Delta z, \qquad \ell_t=\ell_b=\widetilde{\Delta x}=\sqrt{\Delta x^2+(s\Delta x)^2}.
$$

For $s=0$ the mesh is the usual rectangular mesh. The upward unit normal of a tilted face is $(-s,1)/\sqrt{1+s^2}$. Two vertically stacked cell centres are separated by $\Delta z$, two neighbouring cell centres in a row by $(\Delta x, s\Delta x)$. The face-normal gradients of a variable $\phi$ are therefore

$$
\frac{\partial\phi}{\partial n}\Big|_{t,b}=\frac{(1+s^2)\frac{\Delta\phi_z}{\Delta z}-s\frac{\partial\phi}{\partial\xi}}{\sqrt{1+s^2}}, \qquad \frac{\partial\phi}{\partial n}\Big|_{l,r}=\frac{\Delta\phi_x}{\Delta x}-s\frac{\partial\phi}{\partial z},
$$

where $\Delta\phi_z$, $\Delta\phi_x$ are the differences between the neighbouring cell centres and $\frac{\partial}{\partial \xi}=\frac{\partial}{\partial x}+s\frac{\partial}{\partial z}$ is the derivative along a row. The cross derivatives ($\frac{\partial\phi}{\partial\xi}$, $\frac{\partial\phi}{\partial z}$) are taken from the previous time step (deferred correction), so the Jacobian keeps its 5-point stencil.

<!-- END DRAFT -->

### 1.2 Time discretisation

The time derivative is approximated with an implicit Euler scheme, i.e. all fluxes are evaluated at the new time level $t+1$. Dividing by the cell area gives the residual of cell $i$

$$
R_i=\frac{m^{t+1}-m^t}{\Delta t}+\frac{\Delta z}{\Delta x\Delta z}\left(q_n^{l,t+1}+q_n^{r,t+1}\right)+\frac{\widetilde{\Delta x}}{\Delta x\Delta z}\left(q_n^{t,t+1}+q_n^{b,t+1}\right)=0.
$$

## 2. Richards Equation

For the Richards equation $m=\theta$ (total water content) and $\mathbf{q} = -k_{rel}K_s\nabla H$ with $H=z-h_c$. The flux over a face between cell $i$ and its neighbour $j$ is

$$
q^{ij}=-k_{rel}^{ij}K_s^{ij}\frac{\partial H}{\partial n}\Big|_{ij}
$$

<!-- DRAFT: written from volbal.vertical_flux_update / horizontal_flux_update, please check -->

with
- $K_s^{ij}=\frac{2K_s^iK_s^j}{K_s^i+K_s^j}$, the harmonic mean of the saturated hydraulic conductivities, and
- $k_{rel}^{ij}$, the relative hydraulic conductivity of the upstream cell (upwinding), i.e. the cell with the higher total head.

$k_{rel}$ and $\theta_l$ are evaluated from the lookup tables (see [02.1](02.1%20Overview%20Hydraulic%20Parameters.md)) at the new time level.

<!-- END DRAFT -->

### 2.1 Boundary Conditions

Notation remark: $[-1,:]$ is the indexing of arrays in Python. The arrays are indexed bottom-up, so $-1$ is the index of the top row.

<!-- DRAFT: written from volbal.volbal_cappillary_bundle and horizontal_flux_update, please check -->

**Top (infiltration, evaporation and runoff):** the prescribed flux $q_{inf}$ (positive downward, from the forcing) is limited by the effective saturated hydraulic conductivity of the top cell, which is reduced by ice:

$$
K_{s,eff}=K_s\,k_{rel}(h_{eff},T), \qquad S_{eff}=\frac{\theta_s-\theta_i-\theta_r}{\theta_s-\theta_r},
$$

where $h_{eff}$ is the capillary head corresponding to $S_{eff}$. The flux over the top face is

$$
q^*=\begin{cases}\min(q_{inf},\,K_{s,eff}) & q_{inf}\geq0 \;\text{(infiltration)}\\ \max(q_{inf},\,-f_{evap}(S_e)\,K_{s,eff}) & q_{inf}<0 \;\text{(evaporation)}\end{cases}
$$

where $f_{evap}$ reduces evaporation in dry soil: $f_{evap}=1$ for $S_e>0.5$, decreasing linearly to $f_{evap}=0$ at $S_e=0.1$.
In addition, no water is allowed to pond on the surface: if the total head of the top cell exceeds its elevation, the excess $q_{H}=\max(H[-1,:]-z[-1,:],0)/\Delta t$ is removed as Horton overland flow. The infiltration flux and the surface runoff are then

$$
q_{infil}=q^*-q_H, \qquad q_{runoff}=\max(q_{inf}-q^*,0)+q_H.
$$

Surface runoff leaves the domain and cannot re-infiltrate further downslope.

**Right (seepage face):** if the boundary cell is unsaturated ($h_c>0$) the right face is a no-flow boundary. If it is saturated, water seeps out with

$$
q^r=-K_s\frac{\partial H}{\partial n}\Big|_r
$$

where the head at the face is prescribed as $H=z$ (atmospheric pressure) at a distance $\Delta x/2$ from the cell centre.

**Left, Bottom:** no-flow boundaries.

<!-- END DRAFT -->

## 3. Heat Transport

For the heat transport equation

$$
m = C_{eq}T-\rho_i H_f \theta_i, \qquad \mathbf{q}_{conv} = \rho_l c_l \mathbf{q} T, \qquad \mathbf{q}_{cond} = -\lambda_{eq} \nabla T.
$$

To make the notation easier to read we separate the residual into different parts:

$$
R_T=R_{storage}+R_{conv,h}+R_{conv,v}+R_{cond,h}+R_{cond,v}
$$

$$
R_{storage}=\frac{m^{t+1}-m^t}{\Delta t}
$$

$$
R_{conv,h}= \frac{1}{\Delta x}\left(q_{conv,n}^{l,t+1}+q_{conv,n}^{r,t+1}\right), \qquad
R_{conv,v}= \frac{\widetilde{\Delta x}}{\Delta x\Delta z}\left(q_{conv,n}^{t,t+1}+q_{conv,n}^{b,t+1}\right)
$$

$$
R_{cond,h}= \frac{1}{\Delta x}\left(q_{cond,n}^{l,t+1}+q_{cond,n}^{r,t+1}\right), \qquad
R_{cond,v}= \frac{\widetilde{\Delta x}}{\Delta x\Delta z}\left(q_{cond,n}^{t,t+1}+q_{cond,n}^{b,t+1}\right)
$$

<!-- DRAFT: written from volbal.py (condflux/convflux updates), please check -->

The water fluxes $\mathbf{q}$ are taken from the Richards equation at the same Newton-Raphson iteration. At the faces
- the thermal conductivity is the harmonic mean $\lambda^{ij}=\frac{2\lambda^i\lambda^j}{\lambda^i+\lambda^j}$, and
- the temperature in the convective flux is taken from the upstream cell (upwinding).

<!-- END DRAFT -->

### 3.1 Boundary Conditions

<!-- DRAFT: written from volbal.vertical_condflux_update / vertical_convflux_update, please check -->

**Top:** the surface temperature $T_{surface}$ is applied at the surface, half a cell ($\Delta z/2$) above the centre of the top cell. The conductive flux is

$$
q_{cond}^{t}=-\lambda_{eq}[-1,:]\frac{T_{surface}-T[-1,:]}{\Delta z/2}\sqrt{1+s^2},
$$

where the factor $\sqrt{1+s^2}$ accounts for the tilted surface. Infiltrating water enters with the temperature $T_{surface}$ (convective flux).

**Right:** no conductive flux. Water leaving through the seepage face carries its heat out of the domain (convective flux with the temperature of the boundary cell).

**Bottom, Left:** $q_{cond}=0$ and $q_{conv}=0$.

<!-- END DRAFT -->
