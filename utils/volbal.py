import jax.numpy as jnp
import jax
import utils.petrophysic as petro
import utils.petrophysics_capillaryBundles as petro_cap
from functools import partial

@partial(jax.jit,static_argnames=(['rho_i', 'H_f', 'rho_l', 'c_l','CN']))
def volbal_coupled_permaFoam(T_new, T_old, htot_new, htot_old, q_inf, T_surface, dt, Theta_s, Theta_r, alpha, N, Ksat, rho_i, H_f, rho_l, c_l, X, Z, CN):
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
        Effective thermal conductivity [W/(m K)].
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
    # Capillary head
    h_c_old = jnp.maximum(Z - htot_old, 0.0)
    h_c_new = jnp.maximum(Z - htot_new, 0.0)

    nz, nx = X.shape
    dz = jnp.abs(Z[1, 0] - Z[0, 0]) 
    dx_tilde = jnp.abs(X[0, 1] - X[0, 0])  
    dx = jnp.sqrt(jnp.square(dx_tilde)+ jnp.square(dz))

    # Effective saturation
    Se_old = petro.effsat(h_c_old, alpha, N)
    Se_new = petro.effsat(h_c_new, alpha, N)

    # Volumetric water content
    Theta_old = Theta_r + (Theta_s - Theta_r) * Se_old
    Theta_new = Theta_r + (Theta_s - Theta_r) * Se_new

    # Phase separation
    theta_i_new, theta_l_new = petro.phase_separation(Theta_new, Theta_r, T_new, T_freeze=0.0, omega=1.0)
    theta_i_old, theta_l_old = petro.phase_separation(Theta_old, Theta_r, T_old, T_freeze=0.0, omega=1.0)

    # Relative permeability
    krel_old = petro.relperm(Se_old, N)
    krel_new = petro.relperm(Se_new, N)

    # Impedance factor 
    imped_new = petro.impedance_factor(theta_i_new)
    imped_old = petro.impedance_factor(theta_i_old)

    # Reduces hydraulic conductivity
    krel_old *= imped_old
    krel_new *= imped_new

    # Initial residual (without div(q))
    res_r = Theta_new - Theta_old

    # Flux arrays
    qx = jnp.zeros((nz, nx + 1))
    qz = jnp.zeros((nz + 1, nx))

#    # maximum infiltration rate
#     S_effective =  (Theta_s-theta_i_old-Theta_r) / (Theta_s - Theta_r)
#     Ksat_effective = Ksat * petro.relperm(S_effective, N) # Ksat * Se_new
#     q_infil_max = Ksat_effective[-1, :]

#     # maximum evaporation rate based on available water and conductivity
#     theta_top = theta_l_old[-1,:] - Theta_r[-1,:]
#     q_evap_available = -theta_top * dz / dt   # max flux that won't drain below residual water
#     q_evap_max = -Ksat_effective[-1,:]
#     root_scaling = petro.feddes_func(h_c_old[-1,:])
#     q_infil = jnp.where(
#         q_inf >= 0,
#         jnp.minimum(q_inf, q_infil_max),
#         jnp.maximum(q_inf, jnp.maximum(q_evap_available, q_evap_max*root_scaling))
#     )
    q_infil = q_inf

    # Excess water becomes runoff
    q_runoff = jnp.maximum(q_inf - q_infil, 0.0)
    
    # Apply surface flux BC 
    res_r = res_r.at[-1, :].add(-q_infil/ dz * dt)
    qz = qz.at[-1, :].set(-q_infil)


    # Update fluxes
    res_v, qz = vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde, dt)
    res_h, qx = horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z, dt)

    res_rich = res_r + res_v + res_h
    # Average face fluxes
    qx_centered = 0.5 * (qx[:, :-1] + qx[:, 1:])
    qz_centered = 0.5 * (qz[:-1, :] + qz[1:, :])

    # ================================  HEAT EQUATION  ==================================
    # Effective heat capacity and conductivity 

    C_eff_new = petro.effective_heatcapacity(theta_l_new, theta_i_new, Theta_s)
    C_eff_old = petro.effective_heatcapacity(theta_l_old, theta_i_old, Theta_s)

    lambda_eff_new = petro.effective_thermalconductivity(theta_l_new, theta_i_new, Theta_s)

    # Temporal storage term 
    storage_new = (C_eff_new * T_new - rho_i * H_f * theta_i_new)/dt
    storage_old = (C_eff_old * T_old - rho_i * H_f * theta_i_old)/dt
    res_T = storage_new - storage_old

    # Convective and conductive flux terms
    res_h_cond = horizontal_condflux_update(lambda_eff_new, T_old, T_new, dx, dt)
    res_v_cond = vertical_condflux_update(lambda_eff_new, T_old, T_new, T_surface, dz, dt)
    res_h_conv = horizontal_convflux_update(qx, c_l, T_new, rho_l, dx, dt)
    res_v_conv = vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dt)
    res_T += res_v_conv + res_h_conv + res_h_cond + res_v_cond
    res_T /= 1e9
    # Combine residuals
    res_total = jnp.concatenate([res_rich.ravel(), res_T.ravel()])
    return res_total, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_new, theta_i_new

@partial(jax.jit,static_argnames=(['rho_i', 'H_f', 'rho_l', 'c_l','CN']))
def volbal_coupled_drying(T_new, T_old, htot_new, htot_old, q_inf, T_surface, dt, Theta_s, Theta_r, alpha, N, Ksat, rho_i, H_f, rho_l, c_l, X, Z, CN):
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
        Effective thermal conductivity [W/(m K)].
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
    # Capillary head
    h_c_old = jnp.maximum(Z - htot_old, 0.0)
    h_c_new = jnp.maximum(Z - htot_new, 0.0)

    nz, nx = X.shape
    dz = jnp.abs(Z[1, 0] - Z[0, 0]) 
    dx_tilde = jnp.abs(X[0, 1] - X[0, 0])  
    dx = jnp.sqrt(jnp.square(dx_tilde)+ jnp.square(dz))

    # Effective saturation
    Se_old = petro.effsat(h_c_old, alpha, N)
    Se_new = petro.effsat(h_c_new, alpha, N)

    # Volumetric water content
    Theta_old = Theta_r + (Theta_s - Theta_r) * Se_old
    Theta_new = Theta_r + (Theta_s - Theta_r) * Se_new

    # Phase separation
    theta_i_new, theta_l_new = petro.phase_separation(Theta_new, Theta_r, T_new, T_freeze=0.0, omega=1.0)
    theta_i_old, theta_l_old = petro.phase_separation(Theta_old, Theta_r, T_old, T_freeze=0.0, omega=1.0)
    
    Se_old = (theta_l_old-Theta_r) / (Theta_s - Theta_r)
    Se_new = (theta_l_new-Theta_r) / (Theta_s - Theta_r)

    # Relative permeability
    krel_new = petro.relperm(Se_new, N)

    # Initial residual (without div(q))
    res_rich = Theta_new - Theta_old

    # Flux arrays
    qx = jnp.zeros((nz, nx + 1))
    qz = jnp.zeros((nz + 1, nx))

#     # maximum infiltration rate
#     S_effective = (Theta_s[-1,:] - theta_i_new[-1,:] - Theta_r[-1,:]) / (Theta_s[-1,:] - Theta_r[-1,:])
#     Ksat_effective = Ksat[-1,:] * petro.relperm(S_effective, N[-1,:]) # Ksat * Se_new
#     q_infil_max = Ksat_effective

#    # maximum evaporation rate based on available water (see Hydrogeosphere)
#     q_evap_max = -Ksat_effective
#     scaling = petro.evap_limit(Se_new[-1,:])
#     q_infil = jnp.where(
#         q_inf >= 0,
#         jnp.minimum(q_inf, q_infil_max),
#         jnp.maximum(q_inf,q_evap_max*scaling)
#     )
    q_infil = q_inf
    # q_infil = jnp.where(abs(q_infil)<1e-15, 0.0, q_infil)
    # Excess water becomes runoff
    q_runoff = jnp.maximum(q_inf - q_infil, 0.0)
    
    # Apply surface flux BC 
    res_rich = res_rich.at[-1, :].add(-q_infil/ dz * dt)
    qz = qz.at[-1, :].set(-q_infil)
    
    # Update fluxes                  
    res_v, qz = vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde, dt)
    res_h, qx = horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z, dt)
    res_rich += res_v + res_h
    # print("richard residual", jnp.max(jnp.linalg.norm(res_rich, ord=jnp.inf)))
    # Average face fluxes
    qx_centered = 0.5 * (qx[:, :-1] + qx[:, 1:])
    qz_centered = 0.5 * (qz[:-1, :] + qz[1:, :])

    # ================================  HEAT EQUATION  ==================================
    # Effective heat capacity and conductivity 

    C_eff_new = petro.effective_heatcapacity(theta_l_new, theta_i_new, Theta_s)
    C_eff_old = petro.effective_heatcapacity(theta_l_old, theta_i_old, Theta_s)

    lambda_eff_new = petro.effective_thermalconductivity(theta_l_new, theta_i_new, Theta_s)
    
    # Temporal storage term 
    storage_new = (C_eff_new * T_new - rho_i * H_f * theta_i_new)/dt
    storage_old = (C_eff_old * T_old - rho_i * H_f * theta_i_old)/dt

    res_T = storage_new - storage_old

    # Convective and conductive flux terms
    res_h_cond = horizontal_condflux_update(lambda_eff_new, T_new, dx, dx_tilde, dt)
    res_v_cond = vertical_condflux_update(lambda_eff_new, T_new, T_surface, dz, dx, dx_tilde, dt)
    res_h_conv = horizontal_convflux_update(qx, c_l, T_new, rho_l, dz, dt)
    res_v_conv = vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde, dt)
    
    res_T += res_v_conv + res_h_conv + res_h_cond + res_v_cond 
    res_T /= 1e6
    # Combine residuals
    res_total = jnp.concatenate([res_rich.ravel(), res_T.ravel()])
    return res_total, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_new, theta_i_new

eps64 = jnp.finfo(jnp.float64).eps
@partial(jax.jit,static_argnames=(['rho_i', 'H_f', 'rho_l', 'c_l', 'nx', 'nz', 'dx', 'dz']))
def volbal_cappillary_bundle(T_new, T_old, htot_new, htot_old, q_inf, T_surface, dt, soil,soil_type, Theta_s,Theta_r, alpha, N, Ksat, rho_i, H_f, rho_l, c_l, X, Z, nz, nx, dx, dx_tilde, dz):
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
        Effective thermal dtconductivity [W/(m K)].
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
    # Capillary head
    h_c_old = jnp.maximum(Z - htot_old, 0.0)
    h_c_new = jnp.maximum(Z - htot_new, 0.0)    

    # Saturation and water content
    Se_old = petro_cap.S_from_h(h_c_old,alpha,N)
    Theta_old = Se_old * (Theta_s - Theta_r) + Theta_r
    
    Se_new = petro_cap.S_from_h(h_c_new,alpha,N)
    Theta_new = Se_new * (Theta_s - Theta_r) + Theta_r

    # Phase separation
    theta_l_old = jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape), soil_type[2]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape)),Theta_r,Theta_old) #jnp.where(T_old<=0,jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape), soil_type[2]["theta_l_interp"](h_c_old.flatten(), T_old.flatten()).reshape(h_c_old.shape)),Theta_r,Theta_old),Theta_old)
    theta_l_new = jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)),Theta_r,Theta_new) #jnp.where(T_new<=0,jnp.clip(jnp.where(soil==1, soil_type[1]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["theta_l_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)),Theta_r,Theta_new),Theta_new)

    theta_i_old = jnp.clip(Theta_old - theta_l_old, 1e-4, Theta_s - Theta_r)
    theta_i_new = jnp.clip(Theta_new - theta_l_new, 1e-4, Theta_s - Theta_r)

    # Relative permeability
    krel_new =jnp.where(soil==1, soil_type[1]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)) #jnp.where(T_new<=0, jnp.where(soil==1, soil_type[1]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape), soil_type[2]["k_rel_interp"](h_c_new.flatten(), T_new.flatten()).reshape(h_c_new.shape)), petro.relperm(Se_new, N))
    krel_new = jnp.clip(krel_new, eps64, 1.0)
    
    # Initial residual (without div(q))
    res_rich = Theta_new - Theta_old

    # Flux arrays
    qx = jnp.zeros((nz, nx + 1))
    qz = jnp.zeros((nz + 1, nx))

    # # maximum infiltration (adapted Horton excess runoff to account for ice)
    # S_effective = (Theta_s[-1,:] - theta_i_old[-1,:] - Theta_r[-1,:]) / (Theta_s[-1,:] - Theta_r[-1,:])
    # h_eff = petro_cap.h_from_S(S_effective, alpha[-1,:], N[-1,:])
    # Ksat_effective = Ksat[-1,:] * soil_type[1]["k_rel_interp"](h_eff, T_old[-1,:])
    # q_infil_max = Ksat_effective

    # # maximum evaporation rate based on available water (see Hydrogeosphere)
    # q_evap_max = -Ksat_effective
    # scaling = petro.evap_limit(Se_old[-1,:])
    # q_infil = jnp.where(
    #     q_inf >= 0,
    #     jnp.minimum(q_inf, q_infil_max),
    #     jnp.maximum(q_inf,q_evap_max*scaling)
    # )
    # q_infil = jnp.where(theta_i_new[-1,:]>0.9*Theta_s[-1,:], 0.0, q_infil)
    q_infil = q_inf
    # Excess water becomes runoff
    q_runoff = jnp.maximum(q_inf - q_infil, 0.0)

    # Apply surface flux BC 
    res_rich = res_rich.at[-1, :].add(-q_infil/dz*dt)
    qz = qz.at[-1, :].set(-q_infil)

    # Update fluxes
    res_v, qz = vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde, dt)
    res_h, qx = horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z, dt)
    res_rich += res_v + res_h

    # Average face fluxes
    qx_centered = 0.5 * (qx[:, :-1] + qx[:, 1:])
    qz_centered = 0.5 * (qz[:-1, :] + qz[1:, :])
# def vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde, dt):
#     """
#     Update vertical conductive heat fluxes and residuals.
#     """
#     res = jnp.zeros_like(T_new)

#     # Calculate interface temperatures
#     # cond = qz[1:-1, :] < 0
#     # T_interface_new = jnp.where(cond, T_new[:-1, :], T_new[1:, :])
#     # T_interface_new = jnp.concatenate([T_interface_new, T_surface[None,:]],axis = 0)
#     # # Calculate cond. heat fluxes
#     # q_bottom = qz[:-1, :] * c_l * rho_l * T_interface_new 
#     # q_top = qz[1:, :] * c_l * rho_l * T_interface_new
#     # Upwind temperature at each vertical face
#     qz_int = qz[1:-1, :]

#     T_face_int = jnp.where(
#         qz_int < 0,
#         T_new[:-1, :],   # flow downward → take from upper cell
#         T_new[1:, :]     # flow upward → take from lower cell
#     )

#     qT_int = qz_int * c_l * rho_l * T_face_int

#     # Calculate convective term in residual
#     # res = res.at[:, :].add(q_top * dt * dx_tilde / (dz * dx))
#     # res = res.at[:, :].add(-q_bottom * dt * dx_tilde / (dz * dx))

#     res = res.at[:, :].add( qT_int[1:, :] * dt * dx_tilde / (dz * dx))
#     res = res.at[:, :].add(-qT_int[:-1, :] * dt * dx_tilde / (dz * dx))
#     return

    # ================================  HEAT EQUATION  ==================================
    # Effective heat capacity and conductivity 
    C_eff_new = petro.effective_heatcapacity(theta_l_new, theta_i_new, Theta_s)
    C_eff_old = petro.effective_heatcapacity(theta_l_old, theta_i_old, Theta_s)

    lambda_eff_new = petro.effective_thermalconductivity(theta_l_new, theta_i_new, Theta_s)

    # Temporal storage term 
    storage_new = (C_eff_new * T_new - rho_i * H_f * theta_i_new)/dt
    storage_old = (C_eff_old * T_old - rho_i * H_f * theta_i_old)/dt
    res_T = (storage_new - storage_old)

    # Convective and conductive flux terms
    res_h_cond = horizontal_condflux_update(lambda_eff_new, T_new, dx, dx_tilde, dt)
    res_v_cond = vertical_condflux_update(lambda_eff_new, T_new, T_surface, dz, dx, dx_tilde, dt)
    res_h_conv = horizontal_convflux_update(qx, c_l, T_new, rho_l, dz, dt)
    res_v_conv = vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde, dt)

    res_T += res_v_conv + res_h_conv + res_h_cond + res_v_cond
    res_T/=1e6
    # combine residuals
    res_total = jnp.concatenate([res_rich.ravel(), res_T.ravel()])
    return res_total, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_new, theta_i_new

def vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde, dt):
    """
    Update vertical fluxes and# residuals.
    """
    # initialize res vector
    res = jnp.zeros_like(htot_new)

    # Calculate the average hydraulic conductivity between cells
    Kint = 2 * Ksat[:-1,:]* Ksat[1:, :] / (Ksat[:-1, :] + Ksat[1:, :])
    diff_h_new = htot_new[1:, :] - htot_new[:-1, :] 

    # Calculate relative permeability with upwinding scheme
    # cond = htot_new[1:, :] < htot_new[:-1, :]
    kr_new = jnp.where(diff_h_new<0, krel_new[:-1, :], krel_new[1:, :])

    # internal fluxes over cell faces
    q_val = -Kint * kr_new * diff_h_new / dz
    qz = qz.at[1:-1,:].set(q_val)
    # qz = qz.at[-1,:].set(q_val[-1,:])
    
    # add flux from top cell to bottom cell
    res = res.at[:-1, :].add(q_val * dt * dx_tilde / (dz * dx))
    # opposite: substract flux from bottom to top cell
    res = res.at[1:, :].add(-q_val * dt * dx_tilde / (dz * dx))
    return res, qz

def horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z, dt):
    """
    Update horizontal fluxes and residuals.
    """
    # initialize res vector
    res = jnp.zeros_like(htot_new)

    # Calculate the average hydraulic conductivity between cells
    Kint = 2 * Ksat[:, 1:] * Ksat[:, :-1] / (Ksat[:, 1:] + Ksat[:, :-1])

    diff_h_new = (htot_new[:, 1:] - htot_new[:, :-1]) 

    # Calculate relative permeability with upwinding scheme
    # cond = htot_new[:, :-1] < htot_new[:, 1:]
    kr_new = jnp.where(diff_h_new>0, krel_new[:, 1:], krel_new[:, :-1])

    q_val = -Kint * kr_new * diff_h_new / dx_tilde

    # Apply seepage face BC at the right boundary
    # If h_c_new at the boundary is positive, set flux to zero (no flow). 
    # Otherwise calculate flux based on the head difference between the last cell and the atmospheric boundary.
    q_r = jnp.where(h_c_new[:,-1]>0,0*Ksat[:,-1],(htot_new[:,-1]-Z[:,-1])*2/dx*Ksat[:,-1])

    qx = qx.at[:, 1:-1].set(q_val)
    res = res.at[:, :-1].add(q_val * dt / dx)
    res = res.at[:, 1:].add(-q_val * dt / dx)

    # seepage face bc 
    qx = qx.at[:, -1].set(q_r)
    res = res.at[:, -1].add(q_r * dt / dx)
    return res, qx

def horizontal_condflux_update(lambda_eff_new, T_new, dx, dx_tilde, dt):
    """
    Update horizontal conductive heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)
    # Calculate the average thermal conductivity between cells
    lambda_int_new = 2 * lambda_eff_new[:, 1:] * lambda_eff_new[:, :-1] / (lambda_eff_new[:, 1:] + lambda_eff_new[:, :-1])
    diff_T_new = T_new[:, 1:] - T_new[:, :-1]

    q_new = -lambda_int_new * diff_T_new / dx
    q = q_new 

    # Calculate convective term in residual
    res = res.at[:, :-1].add(q / dx)
    res = res.at[:, 1:].add(-q / dx)
    return res

    # """
    # Update horizontal conductive heat fluxes and residuals.
    # """
    # res = jnp.zeros_like(T_new)

    # # Calculate the average thermal conductivity between cells
    # lambda_int_new = 2 * lambda_eff_new[:, 1:] * lambda_eff_new[:, :-1] / (lambda_eff_new[:, 1:] + lambda_eff_new[:, :-1])
    # diff_T_new = (T_new[:, 1:] - T_new[:, :-1])

    # q = -lambda_int_new * diff_T_new / dx_tilde

    # # Calculate convective term in residual
    # res = res.at[:, :-1].add(q / dx )
    # res = res.at[:, 1:].add(-q / dx )
    # return res

def vertical_condflux_update(lambda_eff_new, T_new, T_surface, dz, dx, dx_tilde, dt):
    """
    Update horizontal conductive heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)
    # Calculate the average thermal conductivity between cells
    lambda_int_new = 2 * lambda_eff_new[:-1, :] * lambda_eff_new[1:, :] / (lambda_eff_new[:-1, :] + lambda_eff_new[1:, :])
    diff_T_new = -(T_new[:-1, :] - T_new[1:, :])
    # print("diff_T_new =", diff_T_new)
    # print("lambda_int_new =", lambda_int_new)
    # print("dz =", dz)
    q_new = -lambda_int_new * diff_T_new / dz
    q = q_new
    q_surf = (lambda_eff_new[-1,:]*(T_surface-T_new[-1,:])/dz)
    # print("q =", q)
    # print(dt/dz)
    # Calculate convective term in residual
    res = res.at[:-1, :].add(q / dz)
    res = res.at[1:, :].add(-q / dz)
    res = res.at[-1,:].add(-q_surf)
    # print("res_vertical =", res)
    return res

    # """
    # Update horizontal conductive heat fluxes and residuals.
    # """
    # res = jnp.zeros_like(T_new)
    # # Calculate the average thermal conductivity between cells
    # lambda_int_new = 2 * lambda_eff_new[:-1, :] * lambda_eff_new[1:, :] / (lambda_eff_new[:-1, :] + lambda_eff_new[1:, :])
    # diff_T_new = -(T_new[:-1, :] - T_new[1:, :])

    # q = -lambda_int_new * diff_T_new / dz
    # q_surf = -lambda_eff_new[-1,:]*(T_new[-1,:]-T_surface)/dz

    # # Calculate convective term in residual
    # res = res.at[:-1, :].add(q * dx_tilde / (dz * dx))
    # res = res.at[1:, :].add(-q * dx_tilde / (dz * dx) )
    # res = res.at[-1,:].add(-q_surf * dx_tilde / (dz * dx))
    # return res

def vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde, dt):
    """
    Update vertical conductive heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)

    # Calculate interface temperatures
    cond = qz[1:-1, :] < 0
    T_interface_new = jnp.where(cond, T_new[:-1, :], T_new[1:, :])
    T_interface_new = jnp.concatenate([T_interface_new,T_surface[None,:]],axis = 0)
    # Calculate cond. heat fluxes
    q_new = qz[:-1, :] * c_l * rho_l * T_interface_new 
    q_bottom = q_new 

    q_new = qz[1:, :] * c_l * rho_l * T_interface_new
    q_top = q_new

    # Calculate convective term in residual
    res = res.at[:, :].add(q_top / dz)
    res = res.at[:, :].add(-q_bottom / dz)
    return res
    # """
    # Upwinded vertical convective heat flux (finite volume consistent).
    
    # qz: (nz+1, nx)  → face fluxes
    # T_new: (nz, nx) → cell-centered temperatures
    # """

    # res = jnp.zeros_like(T_new)

    # # Interior faces 
    # qz_int = qz[1:-1, :]  # (nz-1, nx)

    # T_face_int = jnp.where(
    #     qz_int > 0,
    #     T_new[1:, :],#[:-1, :],  
    #     T_new[:-1, :]     
    # )

    # qT_int = qz_int * c_l * rho_l * T_face_int

    # # Divergence contribution
    # res = res.at[:-1, :].add(qT_int * dx_tilde / (dz * dx))
    # res = res.at[1:, :].add(-qT_int * dx_tilde / (dz * dx))


    # T_face_top = jnp.where(
    #     qz[-1,:] > 0,
    #     T_new[-1, :],   
    #     T_surface
    # )

    # qT_top = qz[-1, :] * c_l * rho_l * T_face_top
    # res = res.at[-1, :].add(-qT_top * dx_tilde / (dz * dx))
    # return res

def horizontal_convflux_update(qx, c_l, T_new, rho_l, dz, dt):
    """
    Update vertical conductive heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)
    # Calculate interface temperatures
    cond = qx[:, 1:-1] > 0
    T_interface_new = jnp.where(cond, T_new[:, :-1], T_new[:, 1:])
    T_interface_new_left = jnp.concatenate([T_new[:,0:1], T_interface_new], axis=1)
    T_interface_new_right = jnp.concatenate([T_interface_new, T_new[:,-1:]], axis=1)

    # Calculate cond. heat fluxes
    q_new = qx[:, :-1] * c_l * rho_l * T_interface_new_left
    q_left = q_new

    q_new = qx[:, 1:] * c_l * rho_l * T_interface_new_right
    q_right = q_new

    # Calculate convective term in residual
    res = res.at[:, :].add(-q_left / dz)
    res = res.at[:, :].add(q_right / dz)
    return res
    # """
    # Upwinded horizontal convective heat flux (finite volume consistent).

    # qx: (nz, nx+1)  → face fluxes
    # T_new: (nz, nx) → cell-centered temperatures
    # """

    # res = jnp.zeros_like(T_new)

    # # Interior faces
    # qx_int = qx[:, 1:-1]  # (nz, nx-1)

    # T_face_int = jnp.where(
    #     qx_int > 0,
    #     T_new[:, :-1],   # flow left → right → take from left cell
    #     T_new[:, 1:]     # flow right → left → take from right cell
    # )

    # qT_int = qx_int * c_l * rho_l * T_face_int

    # # Divergence contribution
    # res = res.at[:, :-1].add(-qT_int / dz)
    # res = res.at[:, 1:].add( qT_int / dz)

    # return res