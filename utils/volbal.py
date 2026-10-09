import jax.numpy as jnp
import jax
import utils.petrophysic as petro
import utils.petrophysics_capillaryBundles as petro_cap
from functools import partial

eps64 = jnp.finfo(jnp.float64).eps


def normal_gradient_tilted_face(diff_col, dz, s, Gxi_face):
    """
    Face-normal gradient on a top/bottom face of the sheared mesh (Z = Z_flat + s*X).

    Top/bottom faces are tilted, their upward unit normal is (-s, 1)/sqrt(1+s^2).
    ``diff_col/dz`` is the exact d/dz between two vertically stacked cell centers,
    ``Gxi_face`` the along-row derivative d/dxi at the face (rows follow the shear, so
    d/dxi = d/dx + s*d/dz).

    With d/dx = d/dxi - s*d/dz:
        dH/dn = (dH/dz - s*dH/dx)/sqrt(1+s^2)
              = ((1+s^2)*dH/dz - s*dH/dxi)/sqrt(1+s^2)
    """
    return ((1 + s**2) * diff_col / dz - s * Gxi_face) / jnp.sqrt(1 + s**2)


def normal_gradient_vertical_face(diff_row, dx, s, Gz_face):
    """
    Face-normal gradient d/dx on a vertical left/right face of the sheared mesh.

    Neighbouring cell centers in a row are offset by (dx, s*dx), so diff_row/dx is
    d/dxi = d/dx + s*d/dz. The cross part s*d/dz is removed here, leaving the gradient
    normal to the (vertical) face.
    """
    return diff_row / dx - s * Gz_face

def combine_by_soil(soil, soil_type,soil_ids,interp_key, h_c, T):
        field = soil_type[soil_ids[0]][interp_key](h_c.flatten(), T.flatten()).reshape(h_c.shape)
        for soil_id in soil_ids[1:]:
            field = jnp.where(soil == soil_id, soil_type[soil_id][interp_key](h_c.flatten(), T.flatten()).reshape(h_c.shape), field)
        return field

@partial(jax.jit,static_argnames=(['rho_i', 'rho_l', 'rho_a', 'rho_s', 
                                    'c_i', 'c_l', 'c_a', 'c_s',
                                    'lambda_i', 'lambda_l', 'lambda_a', 'lambda_s',
                                    'H_f', 'nx', 'nz', 'dx', 'dz']))
def volbal(T_new, T_old, htot_new, htot_old, q_inf, T_surface, dt, 
                             soil,soil_type, Theta_s,Theta_r, alpha, N, Ksat, H_f,
                             rho_i, rho_l, rho_a, rho_s, 
                             c_i, c_l, c_a, c_s,
                             lambda_i, lambda_l, lambda_a, lambda_s, 
                             Z, nz, nx, dx, dx_tilde, dz):
    """
    Calculates the residual of the coupled Richards and heat equation with phase change
    (fully implicit, sheared finite-volume mesh).

    Parameters
    ----------
    T_new, T_old : array (nz, nx)
        Temperature at t+1 and t [°C].
    htot_new, htot_old : array (nz, nx)
        Total hydraulic head at t+1 and t [m]; primary unknown of the Richards equation.
    q_inf : array (nx,)
        Surface water flux [m/s], positive = infiltration, negative = evaporation.
        Capped by the ice-reduced Ksat; the excess (and any ponding) becomes runoff.
    T_surface : array (nx,)
        Surface temperature, used as Robin-type BC for the heat equation [°C].
    dt : float
        Time step [s].
    soil : array (nz, nx)
        Soil-type index per cell (1 or 2).
    soil_type : dict
        Per soil type: "theta_l_interp" and "k_rel_interp" interpolators in (h_c, T).
    Theta_s, Theta_r, alpha, N : array (nz, nx)
        van Genuchten parameters (porosity, residual water content, alpha [1/m], N [-]).
    Ksat : array (nz, nx)
        Saturated hydraulic conductivity [m/s].
    H_f : float
        Latent heat of fusion [J/kg].
    rho_i, rho_l, rho_a, rho_s : float
        Densities of ice, liquid water, air and solid matrix [kg/m³].
    c_i, c_l, c_a, c_s : float
        Specific heat capacities of ice, liquid water, air and solid matrix [J/(kg K)].
    lambda_i, lambda_l, lambda_a, lambda_s : float
        Thermal conductivities of ice, liquid water, air and solid matrix [W/(m K)].
    Z : array (nz, nx)
        Cell-centre elevations [m]; the mesh shear s is taken from Z.
    nz, nx : int
        Number of cells in z and x direction.
    dx, dz : float
        Grid spacing [m].
    dx_tilde : float
        Length of the tilted top/bottom cell faces, sqrt(dx² + (s·dx)²) [m].

    Returns
    -------
    res_total : array (2*nz*nx,)
        Richards residual followed by heat residual (flattened).
    q_runoff : array (nx,)
        Surface runoff (infiltration excess + saturation excess) [m/s].
    qx, qz : array (nz, nx+1), (nz+1, nx)
        Darcy fluxes at cell faces [m/s].
    qx_centered, qz_centered : array (nz, nx)
        Darcy fluxes interpolated to cell centres [m/s].
    theta_l_new, theta_i_new : array (nz, nx)
        Liquid water and ice content at t+1 [-].
    """ 
    
    soil_ids = list(soil_type.keys())

    # Mesh shear (dZ/dX along a row). s = 0 recovers the flat-mesh discretisation exactly.
    s = (Z[0, 1] - Z[0, 0]) / dx

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
    # Clipping to ensure that both water content and ice content are between residual water content / 0 and porosity. Deviations could happen due to interpolation errors in theta_l and theta_i
    theta_l_old = jnp.clip(combine_by_soil(soil, soil_type, soil_ids, "theta_l_interp", h_c_old, T_old),Theta_r,Theta_old)
    theta_l_new = jnp.clip(combine_by_soil(soil, soil_type, soil_ids, "theta_l_interp", h_c_new, T_new),Theta_r,Theta_new)

    theta_i_old = jnp.clip(Theta_old - theta_l_old, 0, Theta_s)
    theta_i_new = jnp.clip(Theta_new - theta_l_new, 0, Theta_s)

    # Relative permeability
    # Clipping to ensure that krel will not become smaller than machine precision.
    krel_new = combine_by_soil(soil, soil_type, soil_ids, "k_rel_interp", h_c_new, T_new)
    krel_new = jnp.clip(krel_new,10*eps64,1)

    # Storage term in residuum
    res_rich = (Theta_new - Theta_old)/dt

    # .....................   Evaporation and runoff handling  ..............................
    # Calculate effective Ksat due to ice for Horton excess flow and evaporation
    S_effective = (Theta_s[-1,:] - theta_i_new[-1,:] - Theta_r[-1,:]) / (Theta_s[-1,:] - Theta_r[-1,:])

    # h_from_S has an infinite derivative at S = 1 (ice-free surface cell, theta_i = 0), which turns
    # the Jacobian into NaN. Keep S inside the saturation range of the lookup tables.
    S_effective = jnp.clip(S_effective, 1e-6, 1 - 1e-7)
    h_eff = petro_cap.h_from_S(S_effective, alpha[-1,:], N[-1,:])
    Ksat_effective = Ksat[-1,:] * soil_type[1]["k_rel_interp"](h_eff, T_new[-1,:])

    q_infil_max = Ksat_effective
    q_evap_max = -Ksat_effective
    
    # Evaporation is limited by saturation (see Hydrogeosphere)
    Se_l = (theta_l_new[-1,:] - Theta_r[-1,:]) / (Theta_s[-1,:] - Theta_r[-1,:])
    scaling = petro.evap_limit(Se_l)

    # q_infil according to limits defined above
    q_inf_capped = jnp.where(
        q_inf >= 0,
        jnp.minimum(q_inf,q_infil_max),
        jnp.maximum(q_inf,q_evap_max*scaling)
    )

    # The difference between q_inf and the capped q_inf is saved as runoff. 
    # Note here that the runoff can not reinfiltrate, we just save the outflow in a vector.
    q_runoff = jnp.maximum(q_inf-q_inf_capped,0)

    # We additinally need to ensure that no water column builds up above the surface
    # If hw = htot - Z > 0 at surface, this excess becomes runoff (Dunne overland flow)
    hw_surface = htot_new[-1, :] - Z[-1, :]
    q_horton = jnp.maximum(hw_surface, 0.0) / dt
    q_runoff += q_horton
    
    # Adjust q_infil to account for excess water 
    q_infil = q_inf_capped - q_horton  

    # Initialise flux arrays containing fluxes over cell faces
    qx = jnp.zeros((nz, nx + 1)) 
    qz = jnp.zeros((nz + 1, nx))

    # Apply surface flux BC to the residuum and the vertical flux array
    res_rich = res_rich.at[-1, :].add(-q_infil*dx_tilde/(dz*dx))
    qz = qz.at[-1, :].set(-q_infil)

    # ........................  Richards equation flux updates  .........................
    # Cross derivatives needed to turn the centre-to-centre differences into true
    # face-normal gradients on the sheared mesh. They are lagged to the old time level
    # (deferred correction).
    Gz_h_old = jnp.gradient(htot_old, dz, axis=0)   # d/dz along a column (exact)
    Gxi_h_old = jnp.gradient(htot_old, dx, axis=1)  # d/dxi along a (sheared) row

    # Update fluxes
    res_v, qz = vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde, s, Gxi_h_old)
    res_h, qx = horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z, s, Gz_h_old)
    res_rich += res_v + res_h

    # Flux at cell center, assumes equispaced grid.
    qx_centered = 0.5 * (qx[:, :-1] + qx[:, 1:])
    qz_centered = 0.5 * (qz[:-1, :] + qz[1:, :])

    # ================================  HEAT EQUATION  ==================================
    # Effective heat capacity and conductivity 
    C_eff_new = petro.effective_heatcapacity(theta_l_new, theta_i_new, Theta_s, rho_l=rho_l,rho_i=rho_i,rho_s=rho_s,rho_a=rho_a, 
                     c_s=c_s, c_l=c_l, c_i=c_i, c_a=c_a)
    C_eff_old = petro.effective_heatcapacity(theta_l_old, theta_i_old, Theta_s,rho_l=rho_l,rho_i=rho_i,rho_s=rho_s,rho_a=rho_a,
                     c_s=c_s, c_l=c_l, c_i=c_i, c_a=c_a)

    lambda_eff_new = petro.effective_thermalconductivity(theta_l_new, theta_i_new, Theta_s,lambda_s=lambda_s, lambda_l=lambda_l, lambda_i=lambda_i, lambda_a=lambda_a)

    # Temporal storage term 
    storage_new = (C_eff_new * T_new - rho_i * H_f * theta_i_new)
    storage_old = (C_eff_old * T_old - rho_i * H_f * theta_i_old)
    res_T = (storage_new - storage_old)/dt

    # Convective and conductive flux terms
    res_h_cond = horizontal_condflux_update(lambda_eff_new, T_new, dx, dx_tilde, s,
                                            jnp.gradient(T_old, dz, axis=0))
    res_v_cond = vertical_condflux_update(lambda_eff_new, T_new, T_surface, dz, dx, dx_tilde, s,
                                          jnp.gradient(T_old, dx, axis=1))
    res_h_conv = horizontal_convflux_update(qx, c_l, T_new, rho_l, dx)
    res_v_conv = vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde)

    res_T += res_h_conv + res_v_conv +res_v_cond + res_h_cond

    # combine residuals
    res_total = jnp.concatenate([res_rich.ravel(), res_T.ravel()])
    return res_total, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_new, theta_i_new

def vertical_flux_update(qz, Ksat, htot_new, krel_new, dz, dx, dx_tilde, s, Gxi_h):
    """
    Update vertical fluxes and residuals.

    On the sheared mesh the top/bottom faces are tilted, so the flux is driven by the
    face-normal gradient, not by the plain vertical centre difference. The cross term
    (along-row derivative Gxi_h) is part of the implicit solve.
    """
    # initialize res vector
    res = jnp.zeros_like(htot_new)

    # Calculate the average hydraulic conductivity between cells
    Kint = 2 * Ksat[:-1,:]* Ksat[1:, :] / (Ksat[:-1, :] + Ksat[1:, :])

    # Head difference in z direction
    diff_h_new = htot_new[1:, :] - htot_new[:-1, :] 

    # Face-normal gradient (upward normal of the tilted face)
    Gxi_face = 0.5 * (Gxi_h[:-1, :] + Gxi_h[1:, :])
    grad_n = normal_gradient_tilted_face(diff_h_new, dz, s, Gxi_face)

    # Calculate relative permeability with upwinding scheme (on the corrected gradient)
    kr_new = jnp.where(grad_n<0, krel_new[:-1, :], krel_new[1:, :])

    # Internal fluxes over cell faces
    q_val = -Kint * kr_new * grad_n
    qz = qz.at[1:-1,:].set(q_val)
    
    # Geometric factor (integration of flux over)
    geofac = dx_tilde / (dz * dx)

    # Add flux from top cell to bottom cell
    res = res.at[:-1, :].add(q_val * geofac)
    # Opposite: substract flux from bottom to top cell
    res = res.at[1:, :].add(-q_val * geofac)
    return res, qz

def horizontal_flux_update(qx, Ksat, htot_new, h_c_new, krel_new, dx, dx_tilde, Z, s, Gz_h):
    """
    Update horizontal fluxes and residuals.

    Left/right faces are vertical, so the driving gradient is dH/dx. Neighbouring centres in
    a row are offset by (dx, s*dx), so the raw row difference also contains s*dx*dH/dz; that
    part is removed here, using the implicit dH/dz in Gz_h.
    """
    # initialize res vector
    res = jnp.zeros_like(htot_new)

    # Calculate the average hydraulic conductivity between cells
    Kint = 2 * Ksat[:, 1:] * Ksat[:, :-1] / (Ksat[:, 1:] + Ksat[:, :-1])

    # Head difference in x direction
    diff_h_new = (htot_new[:, 1:] - htot_new[:, :-1]) 

    # Face-normal gradient (normal of the vertical face is (1,0))
    Gz_face = 0.5 * (Gz_h[:, :-1] + Gz_h[:, 1:])
    grad_n = normal_gradient_vertical_face(diff_h_new, dx, s, Gz_face)

    # Calculate relative permeability with upwinding scheme (on the corrected gradient)
    kr_new = jnp.where(grad_n>0, krel_new[:, 1:], krel_new[:, :-1])

    # Internal fluxes over cell faces
    q_val = -Kint * kr_new * grad_n

    # Here the geometric factor is dz/(dz*dx)=1/dx
    qx = qx.at[:, 1:-1].set(q_val)
    res = res.at[:, :-1].add(q_val / dx)
    res = res.at[:, 1:].add(-q_val / dx)

    # Apply seepage face BC at the right boundary
    # Unsaturated: 
    #   h_c_new >0 -> set flux to zero (no flow). 
    # Saturated: 
    #   flux based on the head difference between the boundary cell and the atmosphere, where htot=Z.
    
    # The face sits at x_end, i.e. dx/2 further along the row than the cell centre, so its
    # elevation is Z[:,-1] + s*dx/2. The normal distance to the centre is dx/2 (the face is
    # vertical), and the cross term s*dH/dz is removed as for the interior faces.
    # On a seeping face the head is prescribed as H = z, so its along-face derivative dH/dz is
    # exactly 1.
    Z_face = Z[:, -1] + s * dx / 2
    grad_n_bc = normal_gradient_vertical_face(Z_face - htot_new[:, -1], dx / 2, s, 1.0)
    q_r = jnp.where(h_c_new[:,-1]>0, jnp.zeros_like(Ksat[:,-1]), -Ksat[:,-1] * grad_n_bc)

    # set q_r in q_x
    qx = qx.at[:, -1].set(q_r)

    # add seepage flux to residuum
    res = res.at[:, -1].add(q_r / dx)
    return res, qx

def horizontal_condflux_update(lambda_eff_new, T_new, dx, dx_tilde, s, Gz_T):
    """
    Update horizontal conductive heat fluxes and residuals.

    Face-normal gradient dT/dx on the vertical left/right faces; the cross term s*dT/dz is
    solved implicitly (Gz_T).
    """
    res = jnp.zeros_like(T_new)

    # Calculate the average thermal conductivity between cells
    lambda_int_new = 2 * lambda_eff_new[:, 1:] * lambda_eff_new[:, :-1] / (lambda_eff_new[:, 1:] + lambda_eff_new[:, :-1])
    diff_T_new = T_new[:, 1:] - T_new[:, :-1]

    # Internal fluxes over cell faces
    Gz_face = 0.5 * (Gz_T[:, :-1] + Gz_T[:, 1:])
    q = -lambda_int_new * normal_gradient_vertical_face(diff_T_new, dx, s, Gz_face)

    # Calculate conductive term in residual, the geometric factor is again dz/(dz*dx)=1/dx
    # right side flux
    res = res.at[:, :-1].add(q / dx)
    # left side flux
    res = res.at[:, 1:].add(-q / dx)
    return res

def vertical_condflux_update(lambda_eff_new, T_new, T_surface, dz, dx, dx_tilde, s, Gxi_T):
    """
    Update vertical conductive heat fluxes and residuals.
    Robin-type surface flux: T_surface (air) is treated as a ghost cell
    at half a cell distance (dz/2) from the top cell center.

    Top/bottom faces are tilted on the sheared mesh, so the driving gradient is the
    face-normal one; the along-row cross term is solved implicitly (Gxi_T).
    """
    res = jnp.zeros_like(T_new)
    
    # Calculate the average thermal conductivity between cells
    lambda_int_new = 2 * lambda_eff_new[:-1, :] * lambda_eff_new[1:, :] / (lambda_eff_new[:-1, :] + lambda_eff_new[1:, :])
    diff_T_new = T_new[1:, :]-T_new[:-1, :]

    # Internal fluxes over cell faces
    Gxi_face = 0.5 * (Gxi_T[:-1, :] + Gxi_T[1:, :])
    q = -lambda_int_new * normal_gradient_tilted_face(diff_T_new, dz, s, Gxi_face)

    # Flux from surface ghost cell (air at T_surface, distance dz/2).
    # T_surface is uniform along the tilted surface, so its along-face derivative vanishes and
    # the face-normal gradient is sqrt(1+s^2) * dT/dz (equivalently: the perpendicular distance
    # from the top cell centre to the tilted surface is (dz/2)/sqrt(1+s^2)).
    q_surf = -(lambda_eff_new[-1,:]*(T_surface-T_new[-1,:])/(dz/2)) * jnp.sqrt(1 + s**2)

    # Calculate conductive term in residual
    # bottom fluxes
    res = res.at[1:, :].add(-q * dx_tilde / (dz*dx))
    # top fluxes
    res = res.at[:-1, :].add(q * dx_tilde / (dz*dx))
    # surface flux
    res = res.at[-1, :].add(q_surf * dx_tilde / (dz*dx))
    return res

def vertical_convflux_update(qz, c_l, T_new, T_surface, rho_l, dz, dx, dx_tilde):
    """
    Update vertical convective heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)

    # Upwinding for interface temperatures
    cond = qz > 0
    T_new = jnp.concatenate([jnp.concatenate([T_new[:1,:],T_new],axis=0),T_surface[None,:]],axis=0)
    T_interface_new = jnp.where(cond, T_new[:-1, :], T_new[1:, :])

    # Calculate convective heat fluxes
    q_bottom = -qz[:-1, :] * c_l * rho_l * T_interface_new[:-1,:]

    q_top = qz[1:, :] * c_l * rho_l * T_interface_new[1:,:]

    # Calculate convective term in residual
    res = res.at[:, :].add(q_top * dx_tilde / (dz*dx))
    res = res.at[:, :].add(q_bottom * dx_tilde / (dz*dx))
    return res

def horizontal_convflux_update(qx, c_l, T_new, rho_l, dx):
    """
    Update horizontal convective heat fluxes and residuals.
    """
    res = jnp.zeros_like(T_new)
    # Calculate interface temperatures
    cond = qx > 0
    #qx = qx.at[:,-1].set(0)
    T_new_app = jnp.concatenate([T_new[:,:1],jnp.concatenate([T_new,T_new[:,-1:]],axis=1)],axis=1)
    T_interface_new = jnp.where(cond, T_new_app[:, :-1], T_new_app[:, 1:])

    # Calculate convective heat fluxes
    q_left = -qx[:, :-1] * c_l * rho_l * T_interface_new[:,:-1]
    q_right = qx[:, 1:] * c_l * rho_l * T_interface_new[:,1:]

    # Calculate convective term in residual
    res = res.at[:, :].add(q_left / dx)
    res = res.at[:, :].add(q_right / dx)
    return res