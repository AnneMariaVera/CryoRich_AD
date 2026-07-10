"""
Local sensitivity of simulation outputs to the impedance-factor parameter Omega,
computed via forward-mode automatic differentiation (jax.jvp) through the full
coupled Richards + heat solve.

This module is purely additive: it duplicates the small piece of
utils/parameter_interpolation.py that hardcodes Omega and caches results to
disk (interpolate_parameter_permaFoam_10/50), so that Omega can instead be
passed in as a traced JAX scalar. Nothing in geometry.py, volbal.py or
petrophysic.py needs to change -- Omega only ever enters the solve through
soil_type[...]['k_rel_interp'], which is just a data dictionary consumed
by the existing, unmodified simulation loop.
"""
import jax
import jax.numpy as jnp
import interpax as ipx
import utils.petrophysic as petro


def build_soil_type_impedance(Se, T, soil_type, omega, method='monotonic', extrapolate=True):
    """
    Rebuilds the (theta_l, k_rel) interpolation tables for the impedance-factor
    parametrization -- same physics as interpolate_parameter_permaFoam_10/50 in
    utils/parameter_interpolation.py -- but with omega as an explicit,
    differentiable argument and no file caching.

    Returns a new soil_type dict; does not mutate the one passed in.
    """
    idx_safe = jnp.logical_and(Se > 1e-7, Se < 1 - 1e-7)
    Se = Se[idx_safe]

    new_soil_type = {}
    for soil_id, soil_props in soil_type.items():
        alpha = soil_props["alpha"]
        N = soil_props["N"]
        Theta_r = soil_props["Theta_r"]
        Theta_s = soil_props["Theta_s"]

        hc = petro.effsat_inv(Se, alpha, N)
        theta = petro.watercontent(hc, alpha, N, Theta_s, Theta_r)

        k_rel_ar, theta_l_ar = [], []
        for i in range(len(T)):
            theta_i, theta_l = petro.phase_separation(theta, Theta_r, T[i])
            Se_l = (theta - Theta_r) / (Theta_s - Theta_r)
            imp = petro.impedance_factor(theta_i, Omega=omega)
            k_rel = petro.relperm(Se_l, N) * imp
            k_rel_ar.append(k_rel)
            theta_l_ar.append(theta_l)

        k_rel = jnp.stack(k_rel_ar)
        theta_l = jnp.stack(theta_l_ar)

        soil_props_new = dict(soil_props)
        soil_props_new["k_rel_interp"] = ipx.Interpolator2D(
            jnp.sort(hc), T, k_rel[:, ::-1].T, method=method, extrap=extrapolate
        )
        soil_props_new["theta_l_interp"] = ipx.Interpolator2D(
            jnp.sort(hc), T, theta_l[:, ::-1].T, method=method, extrap=extrapolate
        )
        new_soil_type[soil_id] = soil_props_new
    return new_soil_type


def exit_fluxes_vs_omega(omega, geom, Se, T, base_soil_type, z_split=-1.5):
    """
    Runs geom.simulate_cap_bundle() with an impedance parametrization built
    live from `omega`, and returns the q_top/q_bottom/q_surf time series
    (raw fluxes, not yet integrated) as stacked jnp arrays.

    `geom` must already have grid/sparsity/time-discretization/thermal-params/
    newton-params/surface-BCs/initial-conditions set exactly as for a normal
    simulate_cap_bundle() call (see the *_simulations notebooks). Its
    soil_type is overwritten with a fresh, omega-dependent one for this call.
    """
    soil_type = build_soil_type_impedance(Se, T, base_soil_type, omega)
    geom.set_soilType(geom.soil, soil_type)

    (h_total, h_c_total, theta_l_total, theta_i_total, T_total,
     q_runoff_array, qx_array, qz_array, qx_centered_array, qz_centered_array,
     time_array, iter_total, normres_total) = geom.simulate_cap_bundle()

    idx_top = geom.Z[:, -1] >= z_split
    idx_bottom = geom.Z[:, -1] < z_split

    q_top = jnp.stack([jnp.sum(jnp.abs(qx[:, -1][idx_top]) * geom.dz) for qx in qx_array])
    q_bottom = jnp.stack([jnp.sum(jnp.abs(qx[:, -1][idx_bottom]) * geom.dz) for qx in qx_array])
    q_surf = jnp.stack([jnp.sum(jnp.abs(qr)) * geom.dx_tilde for qr in q_runoff_array])
    time_array = jnp.stack(time_array)
    return time_array, q_top, q_bottom, q_surf


def omega_sensitivity(geom, Se, T, base_soil_type, omega0, z_split=-1.5):
    """
    Forward-mode sensitivity d(q_top, q_bottom, q_surf)/d(omega) at omega0,
    for the entire simulated time series, in a single jax.jvp pass.

    Returns (time_array, (q_top, q_bottom, q_surf), (d_qtop, d_qbottom, d_qsurf)).
    """
    def f(omega):
        return exit_fluxes_vs_omega(omega, geom, Se, T, base_soil_type, z_split=z_split)

    (time_array, q_top, q_bottom, q_surf), (_, d_qtop, d_qbottom, d_qsurf) = jax.jvp(
        f, (omega0,), (1.0,)
    )
    return time_array, (q_top, q_bottom, q_surf), (d_qtop, d_qbottom, d_qsurf)
