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
    return S_e

def effsat_inv(Se,alpha,N):
    h = 1/alpha*(Se**(N/(1-N))-1)**(1/N)
    return h

def relperm(S_e,N):
    """
    Calculate relative permeability based on effective saturation and van Genuchten.
    """
    S_e_safe = jnp.clip(S_e, 0.0, 1.0-1e-7)  # Avoid division by zero
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

def impedance_factor( theta_i, Omega=50.0,k_min=1e-6):
    """
    Calculate impedance factor based on minimum permeability, volumetric frozen water content,
    and exponent omega.
    """
    return jnp.maximum(jnp.power(10, -Omega*theta_i), k_min)

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
    Geometric mean for thermal conductivity (W/m/K) based on 
    volumetric fractions:
    theta_l : liquid water content
    theta_i : ice content
    theta_s : saturated water content (= porosity)

    thermal conductivities:
    lambda_s : solid matrix (e.g. mineral) [W/m/K]
    lambda_w : liquid water [W/m/K]
    lambda_i : ice [W/m/K]
    """
    theta_a = theta_s-theta_i-theta_l
    return jnp.power(lambda_s, 1-theta_s) * jnp.power(lambda_l, theta_l) * jnp.power(lambda_i, theta_i)* jnp.power(lambda_a, theta_a)


@jax.jit
def effective_heatcapacity(theta_l, theta_i, theta_s, rho_l=998,rho_i=917,rho_s=2650,rho_a=1.1, 
                     c_s=800, c_l=4180, c_i=2100, c_a=1000):
    """
    Effective volumetric heat capacity, mixing model based on 
    c_s : solid matrix [J/kg/K]
    c_l : liquid water [J/kg/K]
    c_i : ice [J/kg/K]
    c_a : air [J/kg/K]

    rho_s : solid matrix [kg/m³]
    rho_l : liquid water [kg/m³]
    rho_i : ice [kg/m³]
    rho_a : air [kg/m³]

    Unit: c_eff [J/m³/K]=[kg/s²/m/K]
    """
    theta_a = theta_s-theta_i-theta_l
    return c_s * rho_s * (1 - theta_s) + c_l * rho_l * theta_l + c_i * rho_i * theta_i + c_a * rho_a * theta_a