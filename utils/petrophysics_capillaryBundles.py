import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.integrate import cumulative_trapezoid, simpson 


def r_crit(T, A = -10.**(-19.5), rho_i = 917, L_f = 3.34e5, T_m = 273.15, sigma_ls = 0.029):
    '''
    critical radius for ice formation (Gibbs-Thomson effect)
    -----------------------
    T  : temperature [K]
    A  : Hamaker constant [J]
    rho_i : ice density [kg/m^3]
    L_f : latent heat of fusion [J/kg]
    T_m : melting temperature [K]
    sigma_ls : liquid-solid surface tension [N/m]
    -----------------------
    returns: critical radius r_crit at temperature T
    '''

    # vectorized safe implementation (works for scalars and arrays)
    T = jnp.asarray(T)
    # avoid division by zero / negative when T >= T_m by masking
    denom = jnp.where(T < T_m, (T_m - T), 1.0)
    common = jnp.where(T < T_m, T_m / denom, 0.0)
    d = jnp.where(T < T_m, jnp.power(-A / (6.0 * jnp.pi * rho_i * L_f) * common, 1.0 / 3.0), 0.0)
    r_GT = jnp.where(T < T_m, sigma_ls / (rho_i * L_f) * common, jnp.inf)
    return d + r_GT
    
@jax.jit
def r_i(R, T_array, A = -10.**(-19.5), rho_i = 917, L_f = 3.34e5, T_m = 273.15, sigma_ls = 0.029):
    '''
    Vectorized radius of ice core calculation.

    R : 1D array of pore radii (nbins,)
    T_array : 1D array of temperatures (nT,)

    Returns array shape (nT, nbins) with ice core radius (0 where no ice core).
    '''
    R = jnp.asarray(R)
    T = jnp.asarray(T_array)+273.15

    # compute common factor safely for T < T_m
    denom = jnp.where(T < T_m, (T_m - T), 1.0)

    # deltaT = T_m - T
    # denom = jnp.where(T < T_m,
    #                 jnp.sqrt(deltaT**2),
    #                 1.0)
    common = jnp.where(T < T_m, T_m / denom, 0.0)   # = T_m/(T_m-T) for T<T_m, else 0

    # film thickness d(T) and critical radius r_GT(T)
    d_T = jnp.where(T < T_m, jnp.power(-A / (6.0 * jnp.pi * rho_i * L_f) * common, 1.0 / 3.0), 0.0)
    rGT_T = jnp.where(T < T_m, sigma_ls / (rho_i * L_f) * common, jnp.inf)

    # broadcast to (nT, nR)
    # R[None, :] shape (1, nR); d_T[:, None] shape (nT, 1)
    ri_raw = R[None, :] - d_T[:, None]
    # mask where ri_raw > rGT_T[:,None]
    mask = ri_raw > rGT_T[:, None]
    ri_eff = jnp.where(mask, ri_raw, 0.0)
    return ri_eff

def f_q(R,ri):
    '''
    reduction factor in Mualem model due to ice formation
    -----------------------
    R  : pore radius corresponding to Se
    ri : ice radius corresponding to R and T
    -----------------------
    returns: reduction factor f_q
    '''

    out = jnp.ones(len(R))
    mask = jnp.logical_and(R>ri,ri>0)
    out = out.at[mask].set(jnp.sqrt(1 - (ri[mask]/R[mask])**4 + 
                       (1-ri[mask]**2/R[mask]**2)**4/jnp.log(ri[mask]/R[mask])))
    return out

def h_from_S(S, alpha, N):
    '''
    capillary head from effective saturation
    -----------------------
    S        : effective saturation
    alpha    : van-Genuchten parameter alpha
    N        : van-Genuchten parameter N
    -----------------------
    returns: capillary head h_c corresponding to S
    '''
    out = ( (S**(N/(1-N))-1)**(1/N) )/alpha
    return out

def S_from_h(h_c, alpha, N):
    '''
    effective saturation from capillary head
    -----------------------
    h_c      : capillary head [m]
    alpha    : van-Genuchten parameter alpha
    N        : van-Genuchten parameter N
    -----------------------
    returns: effective saturation S corresponding to h_c
    '''
    out = (1+(alpha*h_c)**N)**((1-N)/N)
    return out

def R_from_S(S,alpha,N, sigma_aw=0.0756, rho_w=1000., g=9.81):
    '''
    radius from effective saturation
    -----------------------
    S        : effective saturation
    alpha    : van-Genuchten parameter alpha
    N        : van-Genuchten parameter N
    sigma_aw : air-water surface tension [N/m]
    rho_w    : water density [kg/m^3]
    g        : gravitational acceleration [m/s^2]
    -----------------------
    returns: pore radius R corresponding to S
    '''

    out = jnp.zeros(len(S))
    out = 2*sigma_aw*alpha/rho_w/g*(S**(N/(1-N))-1)**(-1/N)
    return out

def S_e_unfrozen(Se, ri, R):
    """
    unfrozen effective saturation
    -----------------------
    Se : effective saturation
    ri : ice radius corresponding to R and T
    R  : pore radius corresponding to Se
    -----------------------
    returns: unfrozen effective saturation Seu
    """
    Sel = []
    Sei = []
    for i in range(ri.shape[0]):
        integrant_l=jnp.zeros(len(Se))
        integrant_i=jnp.zeros(len(Se))
        mask = (Se>0)
        integrant_l = integrant_l.at[mask].set(1.-ri[i,mask]**2/R[mask]**2)
        integrant_i = integrant_i.at[mask].set(ri[i,mask]**2/R[mask]**2)
        Sel.append(cumulative_trapezoid(integrant_l,Se,initial=0.))
        Sei.append(cumulative_trapezoid(integrant_i,Se,initial=0.))
    return jnp.array(Sel), jnp.array(Sei)

def k_rel_Mualem(Se, Seu, R, ri, N):
    ''' 
    Watanabe-Mualem model for relative permeability with ice effects
    -----------------------
    Se : effective saturation
    Seu: unfrozen effective saturation  
    N  : van-Genuchten parameter N
    R  : pore radius corresponding to Se
    ri : ice radius corresponding to R and T
    -----------------------
    returns: relative permeability k_rel
    '''
    k_rel = []
    for i in range(ri.shape[0]):
        # denominator for Watanabe-Mualam model
        denom = simpson((Se**(N/(1-N))-1)**(-1/N),Se)
        # denom = jnp.where(jnp.isinf(denom), 1., denom)  # avoid division by zero
        # nominator for Watanabe-Mualam model with ice
        nom = cumulative_trapezoid((Se**(N/(1-N))-1)**(-1/N)*f_q(R, ri[i,:]),
                                Se,initial=0.)
        k_rel.append(Seu[i,:]**0.5*(nom/denom)**2)
    return jnp.array(k_rel)

def k_rel_Burdine(Se, Seu,R,ri,N):
    ''' 
    Published Watanabe model for relative permeability with ice effects
    -----------------------
    Se : effective saturation
    Seu: unfrozen effective saturation  
    N  : van-Genuchten parameter N
    R  : pore radius corresponding to Se
    ri : ice radius corresponding to R and T
    -----------------------
    returns: relative permeability k_rel
    '''
    k_rel = []
    for i in range(ri.shape[0]):
        # denominator for Watanabe-Mualam model
        denom = simpson(((Se**(N/(1-N))-1)**(-1/N))**2,Se)
        # nominator for Watanabe-Mualam model with ice
        nom = cumulative_trapezoid(((Se**(N/(1-N))-1)**(-1/N)*f_q(R, ri[i,:]))**2,
                                Se,initial=0.)
        k_rel.append(nom/denom)
    return jnp.array(k_rel)