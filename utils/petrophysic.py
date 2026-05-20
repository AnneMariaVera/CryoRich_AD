import jax
import jax.numpy as jnp
from functools import partial
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

@jax.jit
def effsat(h_c, alpha, N):
    """
    Calculate effective saturation based on capillary head with van Genuchten parametrisation. 
    """
    h_c_pos = jnp.maximum(h_c, 0.0)  # avoids negative base
    S_e = (1 + (alpha * h_c_pos) ** N) ** ((1 - N) / N)
    # S_e = jnp.where(h_c < 0, 1.0, S_e)  # Saturation = 1 below water table
    return S_e

def effsat_inv(Se,alpha,N):
    h = 1/alpha*(Se**(N/(1-N))-1)**(1/N)
    return h

#@jax.jit
def relperm(S_e,N):
    """
    Calculate relative permeability based on effective saturation and van Genuchten.
    """
    S_e_safe = jnp.clip(S_e, 1e-7, 1.0-1e-7)  # Avoid division by zero
    k_rel = jnp.sqrt(S_e_safe)*(1-(1-S_e_safe**(N/(N-1)))**((N-1)/N))**2
    return k_rel

@jax.jit
def watercontent(h_c, alpha, N, theta_s, theta_r):
    """
    Calculate volumetric water content based on capillary head, alpha, N,
    and the residual and saturated water contents.
    """
    Se = effsat(h_c, alpha, N)
    theta = theta_r + (theta_s - theta_r) * Se
    return theta    

#@jax.jit
def impedance_factor( theta_i, Omega=50.0):
    """
    Calculate impedance factor based on minimum permeability, volumetric frozen water content,
    and exponent omega.
    """
    return jnp.power(10, -Omega*theta_i)

@jax.jit
def phase_separation(theta, theta_r, T, T_freeze=0, omega=1.0):
    """
    volumetric ice content and liquid water content based on total volumetric water content,
    residual water content and Temperature T.
    """
    theta_l = jnp.where(T <= T_freeze, theta_r + (theta - theta_r) * jnp.exp(- jnp.power((T - T_freeze) / omega, 2)),
        theta
    )
    theta_i = theta - theta_l
    return theta_i, theta_l

@jax.jit
def feddes_func(h_c, h_3l=20, h_4=160,eps=0.0001):
    """
    Feddes stress function for root water uptake.
    Returns a factor between 0 and 1 that reduces uptake based on water availability.
    """
    # depth = jnp.where(Z>-0.25,.4,jnp.where(Z>-0.5,.3,jnp.where(Z>-0.75,0.2,jnp.where(Z>-1.0,0.1,0.0))))  # depth factor: 1 at surface, 0.4 below 25cm, 0.7 between 25-75cm
    # return jnp.where(h_c < h_3l, 1.0, jnp.where(h_c < h_4, (h_4 - h_c) / (h_4 - h_3l), 0.0))*0.4
    x = (h_c - h_4) / (h_3l - h_4)
    
    S0 = 0.5 * (1 + jnp.tanh(x / eps))
    S1 = 0.5 * (1 + jnp.tanh((x - 1) / eps))
    
    return x * (S0 - S1) + S1

@jax.jit
def evap_limit(Se, Se_up=0.5, Se_low=0.1):
    """
    Evaporation limit function based on effective saturation.
    Returns a factor between 0 and 1 that reduces evaporation based on soil moisture.
    """
    return jnp.where(Se > Se_up, 1.0, jnp.where(Se > Se_low, (Se - Se_low) / (Se_up - Se_low), 0.0))

@jax.jit
def effective_thermalconductivity(theta_l, theta_i, theta_s, 
                     lambda_s=2.0, lambda_l=0.6, lambda_i=2.2, lambda_a=0.025):
    """
    Geometric mean for thermal conductivity (W/m/K).
    lambda_s : solid matrix (e.g. mineral)
    lambda_w : liquid water
    lambda_i : ice
    """
    return jnp.power(lambda_s, 1-theta_s) * jnp.power(lambda_l, theta_l) * jnp.power(lambda_i, theta_i)* jnp.power(lambda_a, 1 - (1-theta_s + theta_l + theta_i))


@jax.jit
def effective_heatcapacity(theta_l, theta_i, theta_s, rho_l=1000,rho_i=917,rho_s=2650,rho_a=1200, 
                     c_s=800, c_l=4180, c_i=2100, c_a=1000):
    """
    Volumetric heat capacity (J/m3/K).
    c_s : solid matrix (e.g. mineral)
    c_l : liquid water
    c_i : ice
    """
    return c_s * rho_s * (1 - theta_s) + c_l * rho_l * theta_l + c_i * rho_i * theta_i + c_a * rho_a * (theta_s - theta_l - theta_i)

# #@partial(jax.jit, static_argnames=('nbins',))
# def capillary_distribution_scalar(alpha,N,theta_r,theta_s,sigma_w = 7.27e-2, rho_w = 1000, g = 9.81, nbins = 300):
#     dtheta = (theta_s-theta_r)/nbins
#     theta_steps = theta_r + (jnp.arange(nbins) + 0.5) * dtheta
#     Se_steps = (theta_steps-theta_r)/(theta_s-theta_r)
#     h_steps = effsat_inv(Se_steps,alpha,N)
#     R = 2.0 * sigma_w / (rho_w * g * h_steps)
#     n_caps = dtheta/(jnp.pi*R**2)
#     return R, n_caps,dtheta, theta_steps

# def capillary_distribution(alpha, N, theta_r, theta_s,
#                            sigma_w=7.27e-2, rho_w=1000, g=9.81,
#                            nbins=300):
#     """
#     Calculate capillary distribution for a given set of van Genuchten parameters.
#     Parameters
#     ----------
#     alpha : (nx,nz) array
#         van Genuchten alpha parameter (1/m)
#     N : (nx,nz) array
#         van Genuchten N parameter (dimensionless)
#     theta_r : (nx,nz) array
#         residual volumetric water content (m3/m3)
#     theta_s : (nx,nz) array
#         saturated volumetric water content (m3/m3)
#     sigma_w : float
#         surface tension of water (N/m)
#     rho_w : float
#         density of water (kg/m3)
#     g : float
#         gravity constant (m/s2)
#     nbins : int
#         number of discrete capillary pressure bins

#     Returns
#     -------
#     R : (nx,nz,nbins) array
#         capillary radii (m)
#     n_caps : (nx,nz,nbins) array
#         number of capillaries per radius (1/m3)
#     dtheta : (nx,nz) array
#         width of each capillary pressure bin (m3/m3)
#     theta_steps : (nx,nz,nbins) array
#         midpoints of capillary pressure bins (m3/m3)    
#     """
#     orig_shape = alpha.shape

#     # Flatten parameter vectors
#     alpha    = alpha.reshape(-1)
#     N        = N.reshape(-1)
#     theta_r  = theta_r.reshape(-1)
#     theta_s  = theta_s.reshape(-1)

#     # Since θ_r, θ_s are 2d matrices, build inside vmapped fn
#     def single_cap_dist(alpha_i, N_i, th_r_i, th_s_i):
#         dtheta = (th_s_i - th_r_i) / nbins
#         theta_steps = th_r_i + (jnp.arange(nbins)+1) * dtheta
#         jax.debug.print("theta_steps: {x}", x=theta_steps)
#         Se = (theta_steps - th_r_i) / (th_s_i - th_r_i)

#         h = effsat_inv(Se, alpha_i, N_i)       # shape (nbins,)
#         R = 2.0 * sigma_w / (rho_w * g * h)
#         n_caps = dtheta / (jnp.pi * R**2)
#         return R, n_caps, dtheta, theta_steps

#     # vmap over all inputs
#     R, n_caps, dtheta, theta_steps = jax.vmap(single_cap_dist)(
#         alpha, N, theta_r, theta_s
#     )
#     R = R.reshape(*orig_shape, -1)        # (nx, ny, nbins)
#     n_caps = n_caps.reshape(*orig_shape, -1)
#     theta_steps = theta_steps.reshape(*orig_shape, -1)
#     dtheta = dtheta.reshape(orig_shape)   # scalar per grid cell
#     return R, n_caps, dtheta, theta_steps

# def film_thickness(T, Tm = 0, A=-1e-19 , rho_i=917, Lf=3.34e5):
#     """
#     Eq. (8) in W&F. T_K in Kelvin. Returns d (m).
#     d = [ -A / (6 * pi * rho_i * Lf) * (Tm / (Tm - T)) ]^(1/3)
#     Only valid for T < Tm. For T >= Tm returns 0.
#     """
#     T = T+273.15
#     Tm = Tm+273.15
#     delta = jnp.maximum(Tm - T, 0.0)     # ensure argument >= 0 for root
#     arg = -A / (6.0 * jnp.pi * rho_i * Lf) * (Tm / jnp.maximum(delta, 1e-12))
#     # when arg <= 0 produce 0 film thickness
#     d_raw = jnp.where(arg > 0.0, jnp.power(arg, 1.0 / 3.0), 0.0)
#     return d_raw

# @jax.jit
# def critical_radius(T, Tm=0, omega_ls=2.9e-2, rho_i=917, Lf=3.34e5):
#     """
#     Equation (9): rGT = s_ls / (rho_i * Lf) * (Tm/(Tm-T))
#     Valid for T < Tm. For T >= Tm returns +inf (so rGT condition fails -> no core).
#     """
#     T = T+273.15
#     Tm = Tm+273.15
#     delta = jnp.maximum(Tm - T, 0.0)
#     rgt = jnp.where(delta > 0.0, (omega_ls / (rho_i * Lf)) * (Tm / delta), jnp.inf)
#     return rgt

# @jax.jit
# def ice_content(theta, theta_l):
#     return theta-theta_l

# def build_theta_l_k_rel_tables(
#     R, n_caps, theta_steps,
#     alpha, N, theta_r, theta_s,
#     T_min=-20.0, T_max=20.0, dT=0.1,
#     Tm=0, A=-1e-19, rho_i=917, Lf=3.34e5, omega_ls=2.9e-2
# ):
#     # 1D grids
#     T = jnp.arange(T_min, T_max, dT)
#     h_c = effsat_inv(
#         (theta_steps - theta_r) / (theta_s - theta_r),
#         alpha, N
#     )

#     # meshgrid (index space)
#     T_grid, h_c_grid = jnp.meshgrid(T, h_c[0,0,:], indexing="xy")

#     # evaluate fields
#     theta_l, _ = water_content_caps(
#         h_c_grid, T_grid,
#         R, n_caps, theta_steps,
#         alpha, N, theta_r, theta_s,
#         Tm, A, rho_i, Lf, omega_ls
#     )

#     _, k_rel, _ = K_u_caps(
#         h_c_grid, T_grid,
#         R, n_caps, theta_steps,
#         alpha, N, theta_r, theta_s,
#         Tm, A, rho_i, Lf, omega_ls
#     )

#     return T, h_c, theta_l, k_rel

# def make_interp2d_c1(grid, x_grid, y_grid):
#     dx = x_grid[1] - x_grid[0]
#     dy = y_grid[1] - y_grid[0]
#     x0 = x_grid[0]
#     y0 = y_grid[0]

#     def f(x, y):
#         ix = (x - x0) / dx
#         iy = (y - y0) / dy
#         coords = jnp.stack([iy, ix])
#         return map_coordinates(
#             grid,
#             coords,
#             order=3,        # C¹+
#             mode="reflect"
#         )

#     return jax.jit(f)

# def theta_l_k_rel_interp(
#     R, n_caps, theta_steps,
#     alpha, N, theta_r, theta_s,
#     **kwargs
# ):
#     T, h_c, theta_l_tab, k_rel_tab = build_theta_l_k_rel_tables(
#         R, n_caps, theta_steps,
#         alpha, N, theta_r, theta_s,
#         **kwargs
#     )

#     theta_l_fn = make_interp2d_c1(theta_l_tab, T, h_c)
#     k_rel_fn   = make_interp2d_c1(k_rel_tab,   T, h_c)

#     return theta_l_fn, k_rel_fn


# @jax.jit
# def water_content_caps_scalar(h_c,T,R,n_caps,theta_steps,alpha,N,theta_r,theta_s,Tm=0, A=-1e-19 , rho_i=917, Lf=3.34e5, omega_ls=2.9e-2):
#     theta_tot = watercontent(h_c, alpha, N, theta_s, theta_r)
#     d = film_thickness(T,Tm,A,rho_i,Lf)
#     r_GT = critical_radius(T, Tm, omega_ls, rho_i, Lf)
#     ri = jnp.where((theta_steps <= theta_tot) & ((R - d) >= r_GT), R - d, 0.0)
#     R = jnp.where((theta_steps <= theta_tot),R,0)
#     n_caps = jnp.where((theta_steps <= theta_tot),n_caps,0)
#     return jnp.sum(jnp.pi*n_caps*(R**2-ri**2)),jnp.sum(jnp.pi*n_caps*ri**2)

# @jax.jit
# def water_content_caps(h_c,T,R,n_caps,theta_steps,alpha,N,theta_r,theta_s,Tm=0, A=-1e-19 , rho_i=917, Lf=3.34e5, omega_ls=2.9e-2):
#     nz,nx = jnp.shape(h_c)
#     h_c_flat = h_c.flatten()
#     T_flat = T.flatten()
#     alpha = alpha.flatten()
#     N = N.flatten()
#     theta_r = theta_r.flatten()
#     theta_s = theta_s.flatten()
#     theta_l_func = jax.vmap(lambda h, T, alpha, N, theta_r, theta_s: water_content_caps_scalar(h, T, R, n_caps, theta_steps, alpha, N, theta_r, theta_s))
#     theta_l, theta_i=theta_l_func(h_c_flat,T_flat, alpha, N, theta_r, theta_s)
#     return   jnp.reshape(theta_l,(nz,nx)), jnp.reshape(theta_i,(nz,nx))
# @jax.jit
# def K_u_caps_scalar(h_c,T,R,n_caps,theta_steps,alpha,N,theta_r,theta_s,Tm=0, A=-1e-19 , rho_i=917, Lf=3.34e5, omega_ls=2.9e-2,xi=1,rho_w=1000,g=9.81,eta_0=9.62e-7,c=2046):
#     d = film_thickness(T,Tm,A,rho_i,Lf)
#     r_GT = critical_radius(T, Tm, omega_ls, rho_i, Lf)
#     eta = eta_0*jnp.exp(c/(T+273.15))
#     theta_tot = watercontent(h_c, alpha, N, theta_s, theta_r)
#     theta_tot = theta_tot.flatten()
#     ri = jnp.where((theta_steps <= theta_tot) & ((R - d) >= r_GT), R - d, 0.0)
#     K_s_bundle = xi*rho_w*g*jnp.pi/(8*eta)*jnp.sum(n_caps*R**4)
#     R = jnp.where((theta_steps <= theta_tot),R,0)
#     n_caps = jnp.where((theta_steps <= theta_tot),n_caps,0)
#     K_u = xi*rho_w*g*jnp.pi/(8*eta)*jnp.sum(jnp.where(ri>0,n_caps*(R**4-ri**4+(R**2-ri**2)**2/(jnp.log(ri)-jnp.log(R))),n_caps*(R**4)))
#     k_rel = K_u / K_s_bundle
#     return K_u, k_rel, K_s_bundle

# @jax.jit
# def K_u_caps(h_c,T,R,n_caps,theta_steps,alpha,N,theta_r,theta_s,Tm=0, A=-1e-19 , rho_i=917, Lf=3.34e5, omega_ls=2.9e-2,xi=1,rho_w=1000,g=9.81,eta_0=9.62e-7,c=2046):
#     nz,nx = jnp.shape(h_c)
#     h_c_flat = h_c.flatten()
#     T_flat = T.flatten()
#     alpha = alpha.flatten()
#     N = N.flatten()
#     theta_r = theta_r.flatten()
#     theta_s = theta_s.flatten()
#     K_u_func = jax.vmap(lambda h, T, alpha, N, theta_r, theta_s: K_u_caps_scalar(h, T, R, n_caps, theta_steps, alpha, N, theta_r, theta_s,Tm, A, rho_i, Lf, omega_ls,xi,rho_w,g,eta_0,c))
#     K_u, k_rel, K_s_bundle = K_u_func(h_c_flat,T_flat, alpha, N, theta_r, theta_s)
#     return   jnp.reshape(K_u,(nz,nx)), jnp.reshape(k_rel,(nz,nx)), jnp.reshape(K_s_bundle,(nz,nx))