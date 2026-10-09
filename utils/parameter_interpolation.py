import jax.numpy as jnp
import interpax as ipx
import jax
import utils.petrophysics_capillaryBundles as petro_cap
import utils.petrophysic as petro
import pickle as pkl
import os.path

def interpolate_parameter(Se, T, soil_type, filename, parametrization, method='monotonic', extrapolate=True):
    if parametrization == "cap":
        return interpolate_parameter_cap(Se, T, soil_type, filename, method=method, extrapolate=extrapolate)
    elif parametrization == "drying":
        return interpolate_parameter_drying(Se, T, soil_type, filename, method=method, extrapolate=extrapolate)
    elif parametrization == "imp10":
        return interpolate_parameter_imp10(Se, T, soil_type, filename, method=method, extrapolate=extrapolate)
    elif parametrization == "imp50":
        return interpolate_parameter_imp50(Se, T, soil_type, filename, method=method, extrapolate=extrapolate)
    elif parametrization == "drying_omega10":
        return interpolate_parameter_drying_omega10(Se, T, soil_type, filename, method=method, extrapolate=extrapolate)
    elif parametrization == "permaFoam_synthetic_case":
        return interpolate_parameter_permaFoam_synthetic_case(Se, T, soil_type, filename, method=method, extrapolate=extrapolate)
    else:
        raise ValueError(f"Unknown parametrization: {parametrization}") 
    
def interpolate_parameter_cap(Se, T, soil_type, filename, method='monotonic', extrapolate=True):

    assert filename.endswith(".pkl"), "filename should end with .pkl"

    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            soil_type = pkl.load(f)
        print("Loaded interpolated soil type from file: ", filename)
        return soil_type
    idx_safe = jnp.logical_and(Se>1e-7, Se<1-1e-7)
    Se = Se[idx_safe]
    for soil_id, soil_props in soil_type.items():
        # extract soil properties
        alpha = soil_props["alpha"]
        N = soil_props["N"]
        Theta_r = soil_props["Theta_r"]
        Theta_s = soil_props["Theta_s"]

        # calculate hc from S
        hc = petro_cap.h_from_S(Se,alpha,N) # m

        # calculate R and r_i
        R = petro_cap.R_from_S(Se,alpha,N)
        ri = petro_cap.r_i(R,T)
        
        # Calcualte theta_l and theta_i from Se_l and Se_i
        Se_l, Se_i = petro_cap.S_e_unfrozen(Se, ri, R)
        theta_l = Se_l*(Theta_s-Theta_r)+Theta_r
        theta_i = Se_i*(Theta_s-Theta_r)+Theta_r
        theta_i = jnp.where(theta_i<1e-12, 0, theta_i)
        # theta_l = jnp.append(theta_l,jnp.ones((1,theta_l.shape[-1])))
        if jnp.any(theta_l<0) or jnp.any(theta_l>Theta_s):
            print("Warning: theta_l out of bounds for soil_id ", soil_id)

        # Calculate k_rel from theta_l and theta_i
        k_rel = petro_cap.k_rel_Mualem(Se, Se_l, R,ri, N)
        
        idx_sort = jnp.argsort(hc)

        # interpolate k_rel to the grid
        k_rel_interp = ipx.Interpolator2D(hc[idx_sort], T, k_rel[:,idx_sort].T, method=method,extrap=extrapolate)
        theta_l_interp = ipx.Interpolator2D(hc[idx_sort], T, theta_l[:,idx_sort].T, method=method,extrap=extrapolate)

        soil_type[soil_id]["R"] = R
        soil_type[soil_id]["k_rel_interp"] = k_rel_interp
        soil_type[soil_id]["theta_l_interp"] = theta_l_interp

        del hc, k_rel, theta_l, R, ri, Se_l, Se_i, k_rel_interp, theta_l_interp
    
    with open(filename, "wb") as f:
        pkl.dump(soil_type, f)
    print("Saved interpolated soil type to file: ", filename)
    
    return soil_type

def interpolate_parameter_drying(Se, T, soil_type, filename, method='monotonic', extrapolate=True):
    
    assert filename.endswith(".pkl"), "filename should end with .pkl"

    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            soil_type = pkl.load(f)
        print("Loaded interpolated soil type from file: ", filename)
        return soil_type
    idx_safe = jnp.logical_and(Se>1e-7, Se<1-1e-7)
    Se = Se[idx_safe]
    for soil_id, soil_props in soil_type.items():
        # extract soil properties
        alpha = soil_props["alpha"]
        N = soil_props["N"]
        Theta_r = soil_props["Theta_r"]
        Theta_s = soil_props["Theta_s"]

        # calculate hc from S
        hc = petro.effsat_inv(Se,alpha,N) # m

        # Calcualte theta_l and theta_i from Se_l and Se_i
        theta = petro.watercontent(hc,alpha,N,Theta_s,Theta_r)
        k_rel_ar = []
        theta_l_ar = []
        theta_i_ar = []
        for i in range(len(T)):
            theta_i, theta_l = petro.phase_separation(theta, Theta_r, T[i])
            Se_l = (theta_l-Theta_r)/(Theta_s-Theta_r)
        
            # Calculate k_rel from theta_l and theta_i
            k_rel = petro.relperm(Se_l, N)
            k_rel_ar.append(k_rel)
            theta_l_ar.append(theta_l)
            theta_i_ar.append(theta_i)

        k_rel = jnp.array(k_rel_ar)   
        theta_l = jnp.array(theta_l_ar)
        theta_i = jnp.array(theta_i_ar)

        # interpolate k_rel to the grid
        k_rel_interp = ipx.Interpolator2D(jnp.sort(hc), T, k_rel[:,::-1].T, method=method,extrap=extrapolate)
        theta_l_interp = ipx.Interpolator2D(jnp.sort(hc), T, theta_l[:,::-1].T, method=method,extrap=extrapolate)

        soil_type[soil_id]["k_rel_interp"] = k_rel_interp
        soil_type[soil_id]["theta_l_interp"] = theta_l_interp
        del hc, k_rel, theta_l,Se_l, k_rel_interp, theta_l_interp
    
    with open(filename, "wb") as f:
        pkl.dump(soil_type, f)
    print("Saved interpolated soil type to file: ", filename)
    
    return soil_type

def interpolate_parameter_drying_omega10(Se, T, soil_type, filename, method='monotonic', extrapolate=True):

    assert filename.endswith(".pkl"), "filename should end with .pkl"
 
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            soil_type = pkl.load(f)
        print("Loaded interpolated soil type from file: ", filename)
        return soil_type
    
    idx_safe = jnp.logical_and(Se>1e-7, Se<1-1e-7)
    Se = Se[idx_safe]

    for soil_id, soil_props in soil_type.items():
        # extract soil properties
        alpha = soil_props["alpha"]
        N = soil_props["N"]
        Theta_r = soil_props["Theta_r"]
        Theta_s = soil_props["Theta_s"]

        # calculate hc from S
        hc = petro.effsat_inv(Se,alpha,N) # m

        # Calcualte theta_l and theta_i from Se_l and Se_i
        theta = petro.watercontent(hc,alpha,N,Theta_s,Theta_r)
        k_rel_ar = []
        theta_l_ar = []
        theta_i_ar = []
        for i in range(len(T)):
            theta_i, theta_l = petro.phase_separation(theta, Theta_r, T[i])
            Se_l = (theta_l-Theta_r)/(Theta_s-Theta_r)

            # Calculate k_rel from theta_l and theta_i
            imp = petro.impedance_factor(theta_i, Omega=10)
            k_rel = petro.relperm(Se_l, N)
            k_rel *= imp
            k_rel_ar.append(k_rel)
            theta_l_ar.append(theta_l)
            theta_i_ar.append(theta_i)

        k_rel = jnp.array(k_rel_ar)   
        theta_l = jnp.array(theta_l_ar)
        theta_i = jnp.array(theta_i_ar)

        # interpolate k_rel to the grid
        k_rel_interp = ipx.Interpolator2D(jnp.sort(hc), T, k_rel[:,::-1].T, method=method,extrap=extrapolate)
        theta_l_interp = ipx.Interpolator2D(jnp.sort(hc), T, theta_l[:,::-1].T, method=method,extrap=extrapolate)

        soil_type[soil_id]["k_rel_interp"] = k_rel_interp
        soil_type[soil_id]["theta_l_interp"] = theta_l_interp
        del hc, k_rel, theta_l,Se_l, k_rel_interp, theta_l_interp
    
    with open(filename, "wb") as f:
        pkl.dump(soil_type, f)
    print("Saved interpolated soil type to file: ", filename)
    
    return soil_type

def interpolate_parameter_imp50(Se, T, soil_type, filename, method='monotonic', extrapolate=True):

    assert filename.endswith(".pkl"), "filename should end with .pkl"
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            soil_type = pkl.load(f)
        print("Loaded interpolated soil type from file: ", filename)
        return soil_type
    
    idx_safe = jnp.logical_and(Se>1e-7, Se<1-1e-7)
    Se = Se[idx_safe]

    for soil_id, soil_props in soil_type.items():
        # extract soil properties
        alpha = soil_props["alpha"]
        N = soil_props["N"]
        Theta_r = soil_props["Theta_r"]
        Theta_s = soil_props["Theta_s"]

        # calculate hc from S
        hc = petro.effsat_inv(Se,alpha,N) # m

        # Calcualte theta_l and theta_i from Se_l and Se_i
        theta = petro.watercontent(hc,alpha,N,Theta_s,Theta_r)
        k_rel_ar = []
        theta_l_ar = []
        theta_i_ar = []
        for i in range(len(T)):
            theta_i, theta_l = petro.phase_separation(theta, Theta_r, T[i])
            Se_l = (theta-Theta_r)/(Theta_s-Theta_r)
        
            # Calculate k_rel from theta_l and theta_i
            imp = petro.impedance_factor(theta_i, Omega=50)
            k_rel = petro.relperm(Se_l, N)
            k_rel *= imp
            k_rel_ar.append(k_rel)
            theta_l_ar.append(theta_l)
            theta_i_ar.append(theta_i)

        k_rel = jnp.array(k_rel_ar)
        theta_l = jnp.array(theta_l_ar)
        theta_i = jnp.array(theta_i_ar)

        #theta_i_interp = inter.interpolate_parameter(hc, T, theta_i.T, method='monotonic'
        soil_type[soil_id]["k_rel_interp"] = k_rel_interp
        soil_type[soil_id]["theta_l_interp"] = theta_l_interp
        del hc, k_rel, theta_l,Se_l, k_rel_interp, theta_l_interp
    
    with open(filename, "wb") as f:
        pkl.dump(soil_type, f)
    print("Saved interpolated soil type to file: ", filename)
    
    return soil_type

def interpolate_parameter_imp10(Se, T, soil_type, filename, method='monotonic', extrapolate=True):

    assert filename.endswith(".pkl"), "filename should end with .pkl"
    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            soil_type = pkl.load(f)
        print("Loaded interpolated soil type from file: ", filename)
        return soil_type
    
    idx_safe = jnp.logical_and(Se>1e-7, Se<1-1e-7)
    Se = Se[idx_safe]

    for soil_id, soil_props in soil_type.items():
        # extract soil properties
        alpha = soil_props["alpha"]
        N = soil_props["N"]
        Theta_r = soil_props["Theta_r"]
        Theta_s = soil_props["Theta_s"]

        # calculate hc from S
        hc = petro.effsat_inv(Se,alpha,N) # m

        # Calcualte theta_l and theta_i from Se_l and Se_i
        theta = petro.watercontent(hc,alpha,N,Theta_s,Theta_r)
        k_rel_ar = []
        theta_l_ar = []
        theta_i_ar = []
        for i in range(len(T)):
            theta_i, theta_l = petro.phase_separation(theta, Theta_r, T[i])
            Se_l = (theta-Theta_r)/(Theta_s-Theta_r)
        
            # Calculate k_rel from theta_l and theta_i
            imp = petro.impedance_factor(theta_i, Omega=10)
            k_rel = petro.relperm(Se_l, N)
            k_rel *= imp
            k_rel_ar.append(k_rel)
            theta_l_ar.append(theta_l)
            theta_i_ar.append(theta_i)

        k_rel = jnp.array(k_rel_ar)
        theta_l = jnp.array(theta_l_ar)
        theta_i = jnp.array(theta_i_ar)

        # interpolate k_rel to the grid
        k_rel_interp = ipx.Interpolator2D(jnp.sort(hc), T, k_rel[:,::-1].T, method=method,extrap=extrapolate)
        theta_l_interp = ipx.Interpolator2D(jnp.sort(hc), T, theta_l[:,::-1].T, method=method,extrap=extrapolate)

        #theta_i_interp = inter.interpolate_parameter(hc, T, theta_i.T, method='monotonic'
        soil_type[soil_id]["k_rel_interp"] = k_rel_interp
        soil_type[soil_id]["theta_l_interp"] = theta_l_interp
        del hc, k_rel, theta_l,Se_l, k_rel_interp, theta_l_interp
    
    with open(filename, "wb") as f:
        pkl.dump(soil_type, f)
    print("Saved interpolated soil type to file: ", filename)
    
    return soil_type

def interpolate_parameter_permaFoam_synthetic_case(Se, T, soil_type, filename, method='monotonic', extrapolate=True):

    assert filename.endswith(".pkl"), "filename should end with .pkl"

    if os.path.isfile(filename):
        with open(filename, "rb") as f:
            soil_type = pkl.load(f)
        print("Loaded interpolated soil type from file: ", filename)
        return soil_type
    
    idx_safe = jnp.logical_and(Se>1e-7, Se<1-1e-7)
    Se = Se[idx_safe]

    for soil_id, soil_props in soil_type.items():
        # extract soil properties
        alpha = soil_props["alpha"]
        N = soil_props["N"]
        Theta_r = soil_props["Theta_r"]
        Theta_s = soil_props["Theta_s"]

        # calculate hc from S
        hc = petro.effsat_inv(Se,alpha,N) # m

        # Calcualte theta_l and theta_i from Se_l and Se_i
        theta = petro.watercontent(hc,alpha,N,Theta_s,Theta_r)
        k_rel_ar = []
        theta_l_ar = []
        theta_i_ar = []
        for i in range(len(T)):
            theta_i, theta_l = petro.phase_separation(theta, Theta_r, T[i],T_freeze=0, omega=1)
            Se_l = (theta-Theta_r)/(Theta_s-Theta_r)
        
            # Calculate k_rel from theta_l and theta_i
            imp = petro.impedance_factor(theta_i, Omega=50)
            k_rel = petro.relperm(Se_l, N)
            k_rel *= imp
            k_rel_ar.append(k_rel)
            theta_l_ar.append(theta_l)
            theta_i_ar.append(theta_i)

        k_rel = jnp.array(k_rel_ar)
        theta_l = jnp.array(theta_l_ar)
        theta_i = jnp.array(theta_i_ar)

        # interpolate k_rel to the grid
        k_rel_interp = ipx.Interpolator2D(jnp.sort(hc), T, k_rel[:,::-1].T, method=method,extrap=extrapolate)
        theta_l_interp = ipx.Interpolator2D(jnp.sort(hc), T, theta_l[:,::-1].T, method=method,extrap=extrapolate)

        soil_type[soil_id]["k_rel_interp"] = k_rel_interp
        soil_type[soil_id]["theta_l_interp"] = theta_l_interp
        del hc, k_rel, theta_l,Se_l, k_rel_interp, theta_l_interp
    
    with open(filename, "wb") as f:
        pkl.dump(soil_type, f)
    print("Saved interpolated soil type to file: ", filename)
    
    return soil_type
