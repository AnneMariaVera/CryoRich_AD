# Finite-Volume Method

## 1. General Derivation

Assume the governing equation is given by

$$
\frac{\partial m}{\partial t}+\nabla \cdot \mathbf{f}=0
$$

with a stored quantity $m$ and its flux $\mathbf{f}$. We now look at a single cell $i$ with area $A$:

$$
\int_A\frac{\partial m}{\partial t}\,dA+\int_A\nabla \cdot \mathbf{f}\,dA=0
$$

$$
\Leftrightarrow\frac{\partial}{\partial t}\int_Am\,dA+\int_{\partial A}\mathbf{n} \cdot \mathbf{f}\,d\partial A=0
$$

with the outward unit normal $\mathbf{n}$. For $m$ constant inside the cell and $\mathbf{f}$ constant over each face of the boundary we get

$$
\frac{\partial m}{\partial t}A+\sum_{k\in\{l,r,t,b\}}\ell_k\,f_n^k=0
$$

where $\ell_k$ is the length of face $k$ (left, right, top, bottom) and $f_n^k$ the flux normal to the face (positive out of the cell).

### 1.1 Mesh

<!-- DRAFT: written from Geom.create_grid and volbal.py, please check -->

The domain is a hillslope with slope $s$ (paper Figure 3). It is discretised by parallelograms oriented along the surface slope (terrain-following grid cells): each row of cells follows the slope, the left and right faces are vertical, and the top and bottom faces are tilted. All cells have the area $A=\Delta x\Delta z$ and the face lengths

$$
\ell_l=\ell_r=\Delta z, \qquad \ell_t=\ell_b=\overline{\Delta x}=\sqrt{\Delta x^2+(s\Delta x)^2}
$$

($\overline{\Delta x}$ is called `dx_tilde` in the code). For $s=0$ the mesh is the usual rectangular mesh. The upward unit normal of a tilted face is $(-s,1)/\sqrt{1+s^2}$. Two vertically stacked cell centres are separated by $\Delta z$, two neighbouring cell centres in a row by $(\Delta x, s\Delta x)$. The face-normal gradients of a variable $\phi$ are therefore

$$
\frac{\partial\phi}{\partial n}\Big|_{t,b}=\frac{(1+s^2)\frac{\Delta\phi_z}{\Delta z}-s\frac{\partial\phi}{\partial\xi}}{\sqrt{1+s^2}}, \qquad \frac{\partial\phi}{\partial n}\Big|_{l,r}=\frac{\Delta\phi_x}{\Delta x}-s\frac{\partial\phi}{\partial z},
$$

where $\Delta\phi_z$, $\Delta\phi_x$ are the differences between the neighbouring cell centres and $\frac{\partial}{\partial \xi}=\frac{\partial}{\partial x}+s\frac{\partial}{\partial z}$ is the derivative along a row. The cross derivatives ($\frac{\partial\phi}{\partial\xi}$, $\frac{\partial\phi}{\partial z}$) are taken from the previous time step (deferred correction), so the Jacobian keeps its 5-point stencil.

<!-- END DRAFT -->

### 1.2 Time discretisation

Time integration is performed by implicit Euler integration, i.e. all fluxes are evaluated at the new time level $t+1$. Dividing by the cell area gives the residual of cell $i$

$$
R_i=\frac{m^{t+1}-m^t}{\Delta t}+\frac{1}{\Delta x}\left(f_n^{l,t+1}+f_n^{r,t+1}\right)+\frac{\overline{\Delta x}}{\Delta x\Delta z}\left(f_n^{t,t+1}+f_n^{b,t+1}\right)=0.
$$

## 2. Richards Equation

For the Richards equation $m=\theta_{tot}$ and $\mathbf{f}=\mathbf{q} = -k_{rel}K_s\nabla h_{tot}$ with $h_{tot}=h_w+z$. The flux over a face between cell $i$ and its neighbour $j$ is

$$
q^{ij}=-k_{rel}^{ij}K_s^{ij}\frac{\partial h_{tot}}{\partial n}\Big|_{ij}
$$

with
- $K_s^{ij}=\frac{2K_s^iK_s^j}{K_s^i+K_s^j}$, the harmonic mean of the saturated hydraulic conductivities, and
- $k_{rel}^{ij}$, the relative hydraulic conductivity of the upstream cell (upwind scheme), i.e. the cell with the higher total head.

$k_{rel}$ and $\theta_\ell$ are evaluated from the lookup tables (see [02.1](02.1%20Overview%20Hydraulic%20Parameters.md)) at the new time level.

### 2.1 Boundary Conditions

Notation remark: $[-1,:]$ is the indexing of arrays in Python. The arrays are indexed bottom-up, so $-1$ is the index of the top row.

<!-- DRAFT: written from volbal.volbal_cappillary_bundle and horizontal_flux_update, please check -->

**Top (infiltration, evaporation and runoff):** the prescribed infiltration rate $q_{inf}$ (positive downward) is limited by the ice-modified saturated hydraulic conductivity of the top cell:

$$
K_{s,eff}=K_s\,k_{rel}(h_{c,eff},T), \qquad S_{eff}=\frac{\theta_s-\theta_i-\theta_r}{\theta_s-\theta_r},
$$

where $h_{c,eff}$ is the capillary head corresponding to $S_{eff}$. The flux over the top face is

$$
q^*=\begin{cases}\min(q_{inf},\,K_{s,eff}) & q_{inf}\geq0 \;\text{(infiltration)}\\ \max(q_{inf},\,-f_{evap}(S_\ell)\,K_{s,eff}) & q_{inf}<0 \;\text{(evaporation)}\end{cases}
$$

where $f_{evap}$ is a linear scaling function of the liquid water saturation $S_\ell$ of the top cell (Brunner and Simmons, 2012) that reduces evaporation in dry soil: $f_{evap}=1$ for $S_\ell>0.5$, decreasing linearly to $f_{evap}=0$ at $S_\ell=0.1$.
In addition, no water is allowed to pond on the surface: if the total head of the top cell exceeds its elevation, the excess $q_{pond}=\max(h_{tot}[-1,:]-z[-1,:],0)/\Delta t$ is removed. The infiltration flux and the surface runoff are then

$$
q_{infil}=q^*-q_{pond}, \qquad q_{runoff}=\max(q_{inf}-q^*,0)+q_{pond}.
$$

Surface runoff is diagnosed as rejected infiltration: it leaves the domain and is not simulated as overland flow, i.e. it cannot re-infiltrate further downslope.

**Right (free seepage face):** if the boundary cell is unsaturated ($h_c>0$) the right face is a no-flow boundary. If it is saturated, water seeps out with

$$
q^r=-K_s\frac{\partial h_{tot}}{\partial n}\Big|_r
$$

where the head at the face is prescribed as $h_{tot}=z$ (atmospheric pressure, $h_w=0$) at a distance $\Delta x/2$ from the cell centre. In the publication, the outflow through the top half of the seepage face is evaluated as interflow and the outflow through the bottom half as groundwater discharge.

**Left, Bottom:** zero-flux boundaries.

<!-- END DRAFT -->

**Forcing (ERA5-Land):** the infiltration rate is computed from the ERA5-Land data as (paper eq. (23))

$$
q_{inf}=P-E-S_{fall}+S_{melt}
$$

with the total precipitation $P$, the evaporation from bare soil $E$, the snowfall $S_{fall}$ and the snowmelt $S_{melt}$. The soil temperature of the first ERA5-Land soil layer (0 cm to 7 cm) is used as surface temperature. In the notebooks, both hourly time series are smoothed with a 24-hour moving average.

## 3. Heat Transport

For the heat transport equation

$$
m = C_{eq}T-\rho_i H_f \theta_i, \qquad \mathbf{f}_{conv} = \mathbf{q}\, C_\ell T, \qquad \mathbf{f}_{cond} = -\lambda_{eq} \nabla T.
$$

To make the notation easier to read we separate the residual into different parts:

$$
R_T=R_{storage}+R_{conv,h}+R_{conv,v}+R_{cond,h}+R_{cond,v}
$$

$$
R_{storage}=\frac{m^{t+1}-m^t}{\Delta t}
$$

$$
R_{conv,h}= \frac{1}{\Delta x}\left(f_{conv,n}^{l,t+1}+f_{conv,n}^{r,t+1}\right), \qquad
R_{conv,v}= \frac{\overline{\Delta x}}{\Delta x\Delta z}\left(f_{conv,n}^{t,t+1}+f_{conv,n}^{b,t+1}\right)
$$

$$
R_{cond,h}= \frac{1}{\Delta x}\left(f_{cond,n}^{l,t+1}+f_{cond,n}^{r,t+1}\right), \qquad
R_{cond,v}= \frac{\overline{\Delta x}}{\Delta x\Delta z}\left(f_{cond,n}^{t,t+1}+f_{cond,n}^{b,t+1}\right)
$$

The specific discharges $\mathbf{q}$ are taken from the Richards equation at the same Newton-Raphson iteration. At the faces
- the thermal conductivity is harmonically averaged, $\lambda^{ij}=\frac{2\lambda^i\lambda^j}{\lambda^i+\lambda^j}$, and
- the interface temperature of the convective flux is taken from the upstream cell (upwind method).

### 3.1 Boundary Conditions

<!-- DRAFT: written from volbal.vertical_condflux_update / vertical_convflux_update, please check -->

**Top:** the surface temperature $T_{surface}$ is applied at the surface, half a cell ($\Delta z/2$) above the centre of the top cell. The conductive flux is

$$
f_{cond}^{t}=-\lambda_{eq}[-1,:]\frac{T_{surface}-T[-1,:]}{\Delta z/2}\sqrt{1+s^2},
$$

where the factor $\sqrt{1+s^2}$ accounts for the tilted surface. Infiltrating water enters with the temperature $T_{surface}$ (convective flux).

**Right:** zero conductive heat flux. The convective heat flux follows the seepage flow, i.e. water leaving through the seepage face carries its heat out of the domain with the temperature of the boundary cell.

**Bottom, Left:** zero-flux boundaries ($f_{cond}=0$ and $f_{conv}=0$).

<!-- END DRAFT -->

---

P. Brunner and C. T. Simmons, “HydroGeoSphere: A fully integrated, physically based hydrological model,” _Groundwater_, vol. 50, pp. 170–176, 2012, doi: [10.1111/j.1745-6584.2011.00882.x](https://doi.org/10.1111/j.1745-6584.2011.00882.x).
