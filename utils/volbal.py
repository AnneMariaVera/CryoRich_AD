import jax.numpy as jnp
import jax
import utils.petrophysic as petro
import utils.petrophysics_capillaryBundles as petro_cap
from functools import partial

eps64 = jnp.finfo(jnp.float64).eps
@partial(jax.jit,static_argnames=(['rho_i', 'H_f', 'rho_l', 'c_l', 'nx', 'nz', 'dx', 'dz']))
def volbal_cappillary_bundle(T_new, T_old, htot_new, htot_old, q_inf, T_surface, dt, 
                             soil,soil_type, Theta_s,Theta_r, alpha, N, Ksat, 
                             rho_i, H_f, rho_l, c_l, Z, nz, nx, dx, dx_tilde, dz, normres=1e-4):
    """
    Calculates the residual of the heat equation with phase change.
    Parameters
    ----------
    T_new, T_old : array (nz, nx)
        Temperature field at t+1 and t.
    htot_new, htot_old : array (nz, nx)
        Hydraulic head fields, used for advection.
    qx, qz : array
        Darcy fluxes at faces (m/s), shape (nz, nx+1) and (nz+1, nx).
    rho_l : float
        Density of liquid water [kg/m³].
    C_l : float
        Specific heat of liquid water [J/(kg K)].
    C_eq : array or float
        Equivalent volumetric heat capacity [J/(m³ K)].
    lambda_eq : array or float
        Effective thermal dtconducnormrestivity [W/(m K)].
    H_f : float
        Latent heat of fusion [J/kg].
    theta_i_new, theta_i_old : array
        Volumetric ice content at t+1 and t.
    X, Z : mesh coordinate arrays.
    dt : float
        Time step [s].
    CN : float
        Crank–Nicolson weighting (0.5 = semi-implicit).

    Returns
    -------
    res_T : array (nz, nx)
        Heat equation residual.
    qT_adv : tuple (qx_adv, qz_adv)
        Advective heat fluxes in x/z direction.
    qT_cond : tuple (qx_cond, qz_cond)
        Conductive heat fluxes in x/z direction.
    """ 


    # ================================  RICHARDS EQUATION  ==================================
    # Capillary head, here we define that hc is greater 0 
    h_c_old = jnp.maximum(Z - htot_old, 0.0)
    h_c_new = jnp.maximum(Z - htot_new, 0.0)    

    # Saturation and water content
    Se_old = petro_cap.S_from_h(h_c_old,alpha,N)
    Theta_old = Se_old * (Theta_s - Theta_r) + Theta_r
    
    Se_new = petro_cap.S_from_h(h_c_new,alpha,N)
    Theta_new = Se_new * (Theta_s - Theta_r) + Theta_r

    # Phase separation between ice and liquid, 
    # Clipping to ensure that both water content and ice content are between residual water content and porosity. Deviations could happen due to interpolation errors in theta_l and theta_i
    theta_l_old = jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape), soil_type[2]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape)),Theta_r,Theta_old) #jnp.where(T_old<=0,jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape), soil_type[2]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape)),Theta_r,Theta_old),Theta_old)
    theta_l_new = jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)),Theta_r,Theta_new) #jnp.where(T_new<=0,jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)),Theta_r,Theta_new),Theta_new)

    theta_i_old = jnp.clip(Theta_old - theta_l_old, normres, Theta_s - Theta_r)
    theta_i_new = jnp.clip(Theta_new - theta_l_new, normres, Theta_s - Theta_r)

    # Relative permeability
    krel_new =jnp.where(soil==1, soil_type[1]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)) #jnp.where(T_new<=0, jnp.where(soil==1, soil_type[1]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)), petro.relperm(Se_new, N))
    krel_new = jnp.clip(krel_new,10*eps64,1)
    # Storage term in residuum
    res_rich = (Theta_new - Theta_old)/dt

    # .....................   Evaporation and runoff handling  ..............................
    # Calculate effective Ksat due to ice for Horton excess flow and evaporation
    S_effective = (Theta_s[-1,:] - theta_i_new[-1,:] - Theta_r[-1,:]) / (Theta_s[-1,:] - Theta_r[-1,:])
    h_eff = petro_cap.h_from_S(S_effective, alpha[-1,:], N[-1,:])
    Ksat_effective = Ksat[-1,:] * soil_type[1]["k_rel_interp"](h_eff, T_new[-1,:])

    q_infil_max = Ksat_effective
    q_evap_max = -Ksat_effective
    
    # Evaporation is limited by saturation (see Hydrogeosphere)
    scaling = petro.evap_limit(Se_new[-1,:])

    # q_infil according to limits defined above
    q_inf_capped = jnp.where(
        q_inf >= 0,
        q_inf,#jnp.minimum(q_inf,q_infil_max),
        jnp.maximum(q_inf,q_evap_max*scaling)
    )

    # The difference between q_inf and the capped q_inf is saved as runoff. 
    # Note here that the runoff can not reinfiltrate, we just save the outflow in a vector.
    q_runoff = jnp.maximum(q_inf-q_inf_capped,0)

    # We additinally need to ensure that no water column builds up above the surface
    # If hw = htot - Z > 0 at surface, this excess becomes runoff
    hw_surface = htot_new[-1, :] - Z[-1, :]
    q_horton = jnp.maximum(hw_surface, 0.0) / dt
    q_runoff += q_horton
    
    # Adjust q_infil to account for excess water 
    q_infil = q_inf_capped - q_horton  

    # q_infil = q_inf
    # q_runoff = jnp.maximum(q_inf - q_infil, 0.0)

    # Initialise flux arrays containing fluxes over cell boundaries
    qx = jnp.zeros((nz, nx + 1)) 
    qz = jnp.zeros((nz + 1, nx))

    # Apply surface flux BC to the residuum and the vertical flux array
    res_rich = res_rich.at[-1, :].add(-q_infil*dx_tilde/(dz*dx))
    qz = qz.at[-1, :].set(-q_infil)

    # ........................  Richards equation flux updates  .........................
    # Update fluxes
    res_v, qz = vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde)
    res_h, qx = horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z)
    res_rich += res_v + res_h

    # Flux at cell center, assumes equispaced grid.
    qx_centered = 0.5 * (qx[:, :-1] + qx[:, 1:])
    qz_centered = 0.5 * (qz[:-1, :] + qz[1:, :])

    # ================================  HEAT EQUATION  ==================================
    # Effective heat capacity and conductivity 
    C_eff_new = petro.effective_heatcapacity(theta_l_new, theta_i_new, Theta_s)
    C_eff_old = petro.effective_heatcapacity(theta_l_old, theta_i_old, Theta_s)

    lambda_eff_new = petro.effective_thermalconductivity(theta_l_new, theta_i_new, Theta_s)

    # Temporal storage term 
    storage_new = (C_eff_new * T_new - rho_i * H_f * theta_i_new)
    storage_old = (C_eff_old * T_old - rho_i * H_f * theta_i_old)
    res_T = (storage_new - storage_old)/dt

    # Convective and conductive flux terms
    res_h_cond = horizontal_condflux_update(lambda_eff_new, T_new, dx, dx_tilde)
    res_v_cond = vertical_condflux_update(lambda_eff_new, T_new, T_surface, dz, dx, dx_tilde)
    res_h_conv = horizontal_convflux_update(qx, c_l, T_new, rho_l, dx)
    res_v_conv = vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde)

    res_T += res_h_conv + res_v_conv +res_v_cond + res_h_cond

    # combine residuals
    res_total = jnp.concatenate([res_rich.ravel(), res_T.ravel()])
    return res_total, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_new, theta_i_new

def vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde):
    """
    Update vertical fluxes and residuals.
    Allows water table above surface - convergence handled in residual via capillary bundle terms.
    """
    # initialize res vector
    res = jnp.zeros_like(htot_new)

    # Calculate the average hydraulic conductivity between cells
    Kint = 2 * Ksat[:-1,:]* Ksat[1:, :] / (Ksat[:-1, :] + Ksat[1:, :])

    # Head difference in z direction
    diff_h_new = htot_new[1:, :] - htot_new[:-1, :] 

    # Calculate relative permeability with upwinding scheme
    kr_new = jnp.where(diff_h_new<0, krel_new[:-1, :], krel_new[1:, :])

    # Internal fluxes over cell faces
    q_val = -Kint * kr_new * diff_h_new / dz
    qz = qz.at[1:-1,:].set(q_val)
    
    # Geometric factor (integration of flux over )
    geofac = dx_tilde / (dz * dx)

    # Add flux from top cell to bottom cell
    res = res.at[:-1, :].add(q_val * geofac)
    # Opposite: substract flux from bottom to top cell
    res = res.at[1:, :].add(-q_val * geofac)
    return res, qz

def horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z):
    """
    Update horizontal fluxes and residuals.
    """
    # initialize res vector
    res = jnp.zeros_like(htot_new)

    # Calculate the average hydraulic conductivity between cells
    Kint = 2 * Ksat[:, 1:] * Ksat[:, :-1] / (Ksat[:, 1:] + Ksat[:, :-1])

    # Head difference in x direction
    diff_h_new = (htot_new[:, 1:] - htot_new[:, :-1]) 

    # Calculate relative permeability with upwinding scheme
    kr_new = jnp.where(diff_h_new>0, krel_new[:, 1:], krel_new[:, :-1])

    # Internal fluxes over cell faces
    q_val = -Kint * kr_new * diff_h_new / dx_tilde

    # Here the geometric factor is dz/(dz*dx)=1/dx
    qx = qx.at[:, 1:-1].set(q_val)
    res = res.at[:, :-1].add(q_val / dx)
    res = res.at[:, 1:].add(-q_val / dx)

    # Apply seepage face BC at the right boundary
    # Unsaturated: 
    #   h_c_new >0 -> set flux to zero (no flow). 
    # Saturated: 
    #   flux based on the head difference between the boundary cell and the atmosphere, where htot=Z.
    q_r = jnp.where(h_c_new[:,-1]>0,jnp.zeros_like(Ksat[:,-1]),(htot_new[:,-1]-Z[:,-1])*2/dx*Ksat[:,-1])

    # set q_r in q_x
    qx = qx.at[:, -1].set(q_r)

    # add seepage flux to residuum
    res = res.at[:, -1].add(q_r / dx)
    return res, qx

def horizontal_condflux_update(lambda_eff_new, T_new, dx, dx_tilde):
    """
    Update horizontal conductive heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)
    # Calculate the average thermal conductivity between cells
    lambda_int_new = 2 * lambda_eff_new[:, 1:] * lambda_eff_new[:, :-1] / (lambda_eff_new[:, 1:] + lambda_eff_new[:, :-1])
    diff_T_new = T_new[:, 1:] - T_new[:, :-1]

    # Internal fluxes over cell faces
    q = -lambda_int_new * diff_T_new / dx_tilde

    # Calculate convective term in residual, the geometric factor is again dz/(dz*dx)=1/dx
    # right side flux
    res = res.at[:, :-1].add(q / dx)
    # left side flux
    res = res.at[:, 1:].add(-q / dx)
    return res

def vertical_condflux_update(lambda_eff_new, T_new, T_surface, dz, dx, dx_tilde):
    """
    Update vertical conductive heat fluxes and residuals.
    Robin-type surface flux: T_surface (air) is treated as a ghost cell
    at half a cell distance (dz/2) from the top cell center.
    """
    res = jnp.zeros_like(T_new)
    # Calculate the average thermal conductivity between cells
    lambda_int_new = 2 * lambda_eff_new[:-1, :] * lambda_eff_new[1:, :] / (lambda_eff_new[:-1, :] + lambda_eff_new[1:, :])
    diff_T_new = T_new[1:, :]-T_new[:-1, :]

    # Internal fluxes over cell faces
    q = -lambda_int_new * diff_T_new / dz

    # Flux from surface ghost cell (air at T_surface, distance dz/2)
    q_surf = -(lambda_eff_new[-1,:]*(T_surface-T_new[-1,:])/(dz/2))

    # Calculate convective term in residual
    # bottom fluxes
    res = res.at[1:, :].add(-q * dx_tilde / (dz*dx))
    # top fluxes
    res = res.at[:-1, :].add(q * dx_tilde / (dz*dx))
    # surface flux
    res = res.at[-1, :].add(q_surf * dx_tilde / (dz*dx))
    return res

def vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde):
    """
    Update vertical conductive heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)

    # Upwinding for interface temperatures
    cond = qz > 0
    T_new = jnp.concatenate([jnp.concatenate([T_new[:1,:],T_new],axis=0),T_surface[None,:]],axis=0)
    T_interface_new = jnp.where(cond, T_new[:-1, :], T_new[1:, :])

    # Calculate cond. heat fluxes
    q_bottom = -qz[:-1, :] * c_l * rho_l * T_interface_new[:-1,:]

    q_top = qz[1:, :] * c_l * rho_l * T_interface_new[1:,:]

    # Calculate convective term in residual
    res = res.at[:, :].add(q_top * dx_tilde / (dz*dx))
    res = res.at[:, :].add(q_bottom * dx_tilde / (dz*dx))
    return res

def horizontal_convflux_update(qx, c_l, T_new, rho_l, dx):
    """
    Update vertical conductive heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)
    # Calculate interface temperatures
    cond = qx > 0
    #qx = qx.at[:,-1].set(0)
    T_new_app = jnp.concatenate([T_new[:,:1],jnp.concatenate([T_new,T_new[:,-1:]],axis=1)],axis=1)
    T_interface_new = jnp.where(cond, T_new_app[:, :-1], T_new_app[:, 1:])

    # Calculate cond. heat fluxes
    q_left = -qx[:, :-1] * c_l * rho_l * T_interface_new[:,:-1]
    q_right = qx[:, 1:] * c_l * rho_l * T_interface_new[:,1:]

    # Calculate convective term in residual
    res = res.at[:, :].add(q_left / dx)
    res = res.at[:, :].add(q_right / dx)
    return res