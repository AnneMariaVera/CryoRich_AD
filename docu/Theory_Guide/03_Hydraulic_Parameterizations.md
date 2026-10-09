# Hydraulic parameterizations of frozen soil

## 1. Unfrozen soil

### 1.1 Soil-retention curve $\theta_{tot}(h_c)$

van Genuchten (paper eq. (8)), with the capillary head $h_c\geq0$, the pore-size-distribution index $n\thinspace [-]$ and $\alpha\thinspace [m^{-1}]$, which is related to the inverse entry-pressure head:

$$
\theta_{tot}=\theta_r+(\theta_s-\theta_r)\left(1+(\alpha h_c)^n\right)^{\frac{1-n}{n}}
$$

$\theta_s$ is the saturated and $\theta_r$ the residual water content. The effective saturation of the total water is (paper eq. (10))

$$
S=\frac{\theta_{tot}-\theta_r}{\theta_s-\theta_r}.
$$

### 1.2 Relative hydraulic conductivity $k_{rel}$

van Genuchten-Mualem (paper eq. (9)):

$$
k_{rel}(S)=\sqrt{S}\left(1-\left(1-S^\frac{n}{n-1}\right)^\frac{n-1}{n}\right)^2
$$

## 2. Partially frozen soil

When the pore water starts freezing, the total water content $\theta_{tot}(h_c)$ from the soil-retention curve is split into a liquid part $\theta_\ell$ and an ice part $\theta_i$, and the relative hydraulic conductivity becomes a function of capillary head and temperature, $k_{rel}(h_c,T)$. 

Five configurations of three parameterizations are compared with this code (see the table in the README for the names used in the code):

| Label | Parameterization | Phase partitioning | $k_{rel}$ |
|-------|-----------------|--------------------|-----------|
| (A) | drying assumption | SFC, eq. (SFC) | $k_{rel}(S_\ell)$ |
| (B) | drying assumption with impedance factor, $\Omega=10$ | SFC, eq. (SFC) | $k_{rel}(S_\ell)\cdot\max(10^{-\Omega\theta_i},10^{-6})$ |
| (C) | impedance factor, $\Omega=10$ | SFC, eq. (SFC) | $k_{rel}(S)\cdot\max(10^{-\Omega\theta_i},10^{-6})$ |
| (D) | impedance factor, $\Omega=50$ | SFC, eq. (SFC) | $k_{rel}(S)\cdot\max(10^{-\Omega\theta_i},10^{-6})$ |
| (E) | capillary bundle model | Gibbs-Thomson | Watanabe & Flury [1], adapted to Mualem |

The labels follow the results figures of the publication (Figures 6, 7, B.1, B.2).

### 2.1 Drying assumption

This approach treats the effect of soil freezing on $k_{rel}$ as equivalent to the effect of drying [2]. It requires that freezing starts from pores with large radius to pores with smaller radius. The liquid water content is calculated with the soil freezing curve (paper eq. (11))

```math
\theta_\ell=\begin{cases}\theta_r+(\theta_{tot}-\theta_r)\exp\left(-\left(\frac{T-T_f}{\omega}\right)^2\right) & T\leq T_f\\ \theta_{tot} & T>T_f\end{cases} \tag{SFC}
```

where $\omega$ is a fitting parameter and $T_f$ the freezing temperature [3]. In the code $T_f = 0\thinspace °C$ and $\omega = 1\thinspace °C$. The ice content is calculated from $\theta_{tot}$ and $\theta_\ell$ through (paper eq. (12))

$$
\theta_i=\theta_{tot}-\theta_\ell.
$$

The relative hydraulic conductivity is then calculated with $k_{rel}(S)$, where $S$ is replaced by the effective saturation of the liquid water (paper eq. (13))

$$
S_\ell=\frac{\theta_\ell-\theta_r}{\theta_s-\theta_r}.
$$

### 2.2 Impedance factor

The reduced connectivity of the liquid phase is accounted for by multiplying the relative hydraulic conductivity with an empirical impedance factor that decreases with increasing ice content (paper eq. (14); [3], [4]):

$$
k_{rel}^{\ast}=k_{rel}\thinspace \max\left(10^{-6},\thinspace 10^{-\Omega\theta_i}\right)
$$

where $\Omega\thinspace [-]$ is the impedance parameter and the lower limit of $10^{-6}$ acts as a numerical floor.

To obtain the ice content $\theta_i$, the same soil freezing curve (SFC) as in the drying assumption is applied. Following Orgogozo et al. [4], the impedance factor is multiplied to $k_{rel}(S)$ evaluated at the total effective saturation $S$ (C, D). Alternatively, it is multiplied to $k_{rel}(S_\ell)$ evaluated at the effective saturation of liquid water $S_\ell$, resulting in a combination of the drying assumption and the impedance factor (B; [5], [6]). Literature values of $\Omega$ range from 5 to 50; $\Omega=10$ and $\Omega=50$ are applied to $k_{rel}(S)$, and $\Omega=10$ is applied to $k_{rel}(S_\ell)$.

### 2.3 Clausius-Clapeyron equation

The soil freezing curve (SFC) used in 2.1 and 2.2 is an empirical parameterization, which is not based on the Clausius-Clapeyron equation. The water content could be related to temperature with the Clausius-Clapeyron equation, which relates temperature to pressure under the assumption of thermal equilibrium. When neglecting the deformation of the soil due to freezing/thawing, the equation is given by

$$
\frac{\partial p_w}{\partial T}\approx \frac{H_f\rho_\ell}{T}
$$

with $T$ in Kelvin. Integrating with respect to $T$ from the melting temperature $T_m$ results in

$$
p_w = H_f\rho_\ell \ln\left(\frac{T}{T_m}\right)\Leftrightarrow h_w=\frac{H_f}{g}\ln\left(\frac{T}{T_m}\right).
$$

This relation is not used in the parameterizations (A)-(D), as we decided on one parametrization, namely the one implemented in permaFoam [4]. The capillary bundle model (E) uses the Gibbs-Thomson equation (see 2.4).

### 2.4 Capillary bundle model

The third parameterization was developed by **Watanabe and Flury** [1] and resolves the geometry of the ice within individual pores. It extends the capillary bundle model underlying the van Genuchten-Mualem parameterization, where the soil is considered as a collection of capillaries with varying radii. Under unfrozen conditions (paper eqs. (15) and (16))

$$
k_{rel}(S)=S^{\frac{1}{2}}\left(\frac{\int_0^S\left(\frac{1}{h_c(S_{\ast})}\right)dS_{\ast}}{\int_0^1\left(\frac{1}{h_c(S_{\ast})}\right)dS_{\ast}}\right)^2, \qquad h_c(S_{\ast})=\frac{1}{\alpha}\left(S_{\ast}^{n/(1-n)}-1\right)^{\frac{1}{n}},
$$

whose analytical solution is the van Genuchten-Mualem (paper eq. (9)). Each effective saturation corresponds to the largest water-filled pore radius (Young-Laplace equation, paper eq. (A.4))

$$
R(S_{\ast})=\frac{2\sigma_{aw}}{\rho_\ell g\thinspace h_c(S_{\ast})}
$$

with the surface tension of water $\sigma_{aw}=0.0756\thinspace N\thinspace m^{-1}$.

For frozen soils, Watanabe and Flury [1] adapted the capillary bundle model accounting for ice formation within the capillaries from the center outwards towards the pore wall. The radius of the ice core is (paper eq. (17))

```math
r_i(R,T)=\begin{cases}R-d(T) & \text{if } R-d(T)>r_{GT}(T)\\ 0 & \text{otherwise,}\end{cases}
```

where $d$ is the thickness of the liquid water film between ice core and pore wall (paper eq. (18))

$$
d(T)=\left(-\frac{A}{6\pi\rho_iH_f}\left(\frac{T_m}{T_m-T}\right)\right)^{\frac{1}{3}}
$$

and $r_{GT}$ is the critical (minimum) radius of the ice core (Gibbs-Thomson, paper eq. (19))

$$
r_{GT}(T)=\frac{\sigma_{ls}}{\rho_iH_f}\left(\frac{T_m}{T_m-T}\right),
$$

with the Hamaker constant $A=-10^{-19.5}\thinspace J$, the surface tension between liquid water and ice $\sigma_{ls}=0.029\thinspace N\thinspace m^{-1}$, the melting temperature $T_m$. Pores of a radius $R\leq r_{GT}(T)+d(T)$ remain ice free, whereas liquid water in pores with a larger radius is restricted to a cylindrical film of thickness $d(T)$.

The saturation of the pore space with liquid water is (paper eq. (20))

$$
S_\ell=\int_0^S\left(1-\frac{r_i^2(R(S_{\ast}),T)}{R^2(S_{\ast})}\right)dS_{\ast},
$$

and the relative hydraulic conductivity for frozen soils is (paper eq. (21))

$$
k_{rel}(S_\ell,T)=S_\ell^{\frac{1}{2}}\left(\frac{\int_0^S\left(S_{\ast}^{n/(1-n)}-1\right)^{-\frac{1}{n}}f(R(S_{\ast}),T)\thinspace dS_{\ast}}{\int_0^1\left(S_{\ast}^{n/(1-n)}-1\right)^{-\frac{1}{n}}dS_{\ast}}\right)^2
$$

with the reduction factor (paper eq. (22))

```math
f(R,T)=\begin{cases}\sqrt{1-\frac{r_i^4(R,T)}{R^4}+\frac{\left(1-\frac{r_i^2(R,T)}{R^2}\right)^2}{\ln(r_i(R,T)/R)}} & \text{if } r_i(R,T)>0\\ 1 & \text{otherwise.}\end{cases}
```

Note that Watanabe and Flury [1] used the parameterization of Burdine [7], while the parameterization of Mualem [8] is used here to ensure consistency with ice-free conditions. The derivation of $f(R,T)$ and the comparison to the original model are given in Appendix A of the publication.

To keep the simulations efficient, $\theta_\ell$ and $k_{rel}$ of all parameterizations are precomputed as lookup tables (see [05 Nonlinear solver and time stepping](05_Nonlinear_Solver.md)).

---

[1] K. Watanabe and M. Flury, “Capillary bundle model of hydraulic conductivity for frozen soil,” _Water Resources Research_, vol. 44, no. 12, p. 2008WR007012, Dec. 2008, doi: [10.1029/2008WR007012](https://doi.org/10.1029/2008WR007012).

[2] R. L. Harlan, “Analysis of coupled heat-fluid transport in partially frozen soil,” _Water Resources Research_, vol. 9, pp. 1314–1323, 1973, doi: [10.1029/WR009i005p01314](https://doi.org/10.1029/WR009i005p01314).

<!-- DRAFT: references [3], [5]-[8] added from memory, please check against the publication -->

[3] J. M. McKenzie, C. I. Voss, and D. I. Siegel, “Groundwater flow with energy transport and water–ice phase change: Numerical simulations, benchmarks, and application to freezing in peat bogs,” _Advances in Water Resources_, vol. 30, no. 4, pp. 966–983, 2007, doi: [10.1016/j.advwatres.2006.08.008](https://doi.org/10.1016/j.advwatres.2006.08.008).

[4] L. Orgogozo, T. Xavier, H. Oulbani, and C. Grenier, “Permafrost modelling with OpenFOAM®: New advancements of the permaFoam solver,” _Computer Physics Communications_, vol. 282, p. 108541, Jan. 2023, doi: [10.1016/j.cpc.2022.108541](https://doi.org/10.1016/j.cpc.2022.108541).

[5] G. S. Taylor and J. N. Luthin, “A model for coupled heat and moisture transfer during soil freezing,” _Canadian Geotechnical Journal_, vol. 15, no. 4, pp. 548–555, 1978, doi: [10.1139/t78-058](https://doi.org/10.1139/t78-058).

[6] S. Endrizzi, S. Gruber, M. Dall'Amico, and R. Rigon, “GEOtop 2.0: simulating the combined energy and water balance at and below the land surface accounting for soil freezing, snow cover and terrain effects,” _Geoscientific Model Development_, vol. 7, no. 6, pp. 2831–2857, 2014, doi: [10.5194/gmd-7-2831-2014](https://doi.org/10.5194/gmd-7-2831-2014).

[7] N. T. Burdine, “Relative permeability calculations from pore size distribution data,” _Journal of Petroleum Technology_, vol. 5, no. 3, pp. 71–78, 1953, doi: [10.2118/225-G](https://doi.org/10.2118/225-G).

[8] Y. Mualem, “A new model for predicting the hydraulic conductivity of unsaturated porous media,” _Water Resources Research_, vol. 12, no. 3, pp. 513–522, 1976, doi: [10.1029/WR012i003p00513](https://doi.org/10.1029/WR012i003p00513).

