import sys
sys.path.append('../')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import utils.petrophysic as petro
from utils.volbal import  volbal_cappillary_bundle
from utils.volbal_validation import volbal_validation
import sparsejac as sparsejac
import interpax as ipx
import time

def timer(func):
    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.time()
        retval = func(*args, **kwargs)
        print(f"{time.time() - start_time:.2f} seconds elapsed")
        return retval
    return inner

class Geom():
    def __init__(self, dx, dz, x_0, x_end, z_0, z_end, slope=0.0):
        self.dx = dx
        self.dz = dz
        self.x_0 = x_0
        self.x_end = x_end
        self.z_0 = z_0
        self.z_end = z_end
        self.slope = slope

    def create_grid(self):
        """
        Create a grid with specified dimensions and slope.
        """
        print("===============   Creating grid   =============== ")


        self.nx = int(jnp.round(abs(self.x_end - self.x_0) / self.dx))
        self.nz = int(jnp.round(abs(self.z_end - self.z_0) / self.dz))

        # Edges for plotting
        X_plot = jnp.linspace(self.x_0, self.x_end, self.nx+1)
        Z_plot = jnp.linspace(self.z_0, self.z_end, self.nz+1)
        self.X_plot, self.Z_plot = jnp.meshgrid(X_plot, Z_plot)
        self.Z_plot = self.Z_plot + self.slope * self.X_plot

        # Centers for computation
        X_centers = X_plot[:-1] + self.dx / 2
        Z_centers = Z_plot[:-1] + self.dz / 2
        self.X, self.Z = jnp.meshgrid(X_centers, Z_centers)
        self.Z = self.Z + self.slope * self.X
        
        # dx_tilde is the length of the top and bottom face of a cell
        self.dx_tilde = jnp.sqrt(self.dx**2+(self.Z[0,1]-self.Z[0,0])**2)

        # Neighboring cells for finite volume discretization
        self.neighbors = []
        # self.color = jnp.empty((self.nz,self.nx))

        for i in range(self.nz):
            for j in range(self.nx):
                cell_neighbors = set()

                # top neighbor
                if i > 0:
                    cell_neighbors.add((i-1)*self.nx + (j))
                # bottom neighbor
                if i < self.nz - 1:
                    cell_neighbors.add((i+1)*self.nx + (j))
                # left neighbor
                if j > 0:
                    cell_neighbors.add(i*self.nx + (j-1))
                # right neighbor
                if j < self.nx - 1:
                    cell_neighbors.add(i*self.nx + (j+1))

                self.neighbors.append(cell_neighbors)
        print("DONE: created grid with ", self.nx, "x", self.nz, "cells.")
        print("=============================================================")

    def set_sparsity_pattern(self):
        """
        Calculates sparsity pattern for fully coupled Jacobian [htot, T].
        """
        print("===============   Calculating coupled sparsity pattern   =============== ")
        self.np = self.nx * self.nz
        self.dof = 2 * self.np  # total number of unknowns (#htot + #T)
        data = []
        indices = []

        # Loop over all cells
        for i in range(self.np):
            # diagonal h-h
            data.append(1)
            indices.append([i, i])
            # diagonal T-T
            data.append(1)
            indices.append([self.np+i, self.np+i])
            # coupling h-T and T-h
            data.append(1)
            indices.append([i, self.np+i])
            data.append(1)
            indices.append([self.np+i, i])

            # FV neighbors (h-h and T-T)
            for j in self.neighbors[i]:
                data.append(1); 
                indices.append([i, self.np+j])     # h-T neighbor
                data.append(1); 
                indices.append([self.np+i, j])     # T-h neighbor
                data.append(1)
                indices.append([i, j])     # h-h
                data.append(1)
                indices.append([self.np+i, self.np+j]) # T-T
        indices = jnp.array(indices)
        order = jnp.lexsort([indices[:,0],indices[:,1]])
 
        indices = indices[order]
        data  =  jnp.array(data)[order]

        self.sparse_pattern = jax.experimental.sparse.BCOO((data, indices),
                                                        shape=(self.dof, self.dof))
        bcsr_sparse = jax.experimental.sparse.BCSR.from_bcoo(self.sparse_pattern)
        self.indices = bcsr_sparse.indices
        self.indptr = bcsr_sparse.indptr
        print("DONE: calculated coupled sparsity pattern with", len(data), "nonzero entries.")
        print("==========================================================================")

    def time_discretization(self, dt=None, t0 = 0, t_end=None, maxdt=None, save_t=None):
        """
        Create a time discretization for the simulation.
        dt = initial time step size                         [s]
        t0 = intial time                                    [s]
        t_end = end time for simulation                     [s]
        maxdt = maximum step size                           [s]
        CN = weighting parameter for impl.and expl. cont.   [s]
        save_t = times for which results are saved.         [s]
                 If you want to save each timestep, choose None.
        """
        print("===============   Creating time discretization   =============== ")
        assert dt != None, "dt is not set. Please set initial dt"
        assert t_end != None, "t_end is not set. Please set the end of the simulation time"
        assert maxdt != None, "t_end is not set. Please set the end of the simulation time"

        self.dt = dt
        self.t0 = t0
        self.t_end = t_end
        self.maxdt = maxdt
        self.nt = int(t_end / dt)
        self.save_t = save_t
        print(f"DONE: created time discretization.")
        print("=============================================================")
   
    def set_soilType(self, soil, soil_type):
        """
        Set soil type for the simulation.
        soil = 2D array of shape (nz, nx) with integer values representing soil types.
        soil_type = dict mapping each integer soil type id present in `soil` to its properties;
        any number of soil types is supported (including a single one).
        """
        assert soil.shape == self.X.shape, f"soil must be of shape ({self.nz}, {self.nx})"
        assert isinstance(soil_type, dict), "soil_type must be a dictionary mapping integer soil types to their properties."
        self.soil = soil
        self.soil_type = soil_type

        # Create array with soil parameters based on soil type, generic over however many
        # soil type ids are present in soil_type (built up by overlaying jnp.where per id).
        soil_ids = list(soil_type.keys())

        def build_param(key, default=None):
            # default=None means the key is required (KeyError if missing); otherwise missing
            # keys fall back to `default` -- used for Theta_WP so soil_type dicts that predate
            # the evapotranspiration sink (PET=0 by default in set_surface_bc, making the sink
            # a no-op regardless of Theta_WP) keep working unchanged.
            get = (lambda sid: soil_type[sid][key])
            param = jnp.full(soil.shape, get(soil_ids[0]))
            for soil_id in soil_ids[1:]:
                param = jnp.where(soil == soil_id, get(soil_id), param)
            return param

        self.Theta_s = build_param("Theta_s")
        self.Theta_r = build_param("Theta_r")
        self.alpha = build_param("alpha")
        self.N = build_param("N")
        self.Ksat = build_param("Ksat")

    def set_cap_distribution(self, nbins=300, sigma_w = 7.27e-2):
        """
        Set capillary distribution parameters for the simulation.
        nbins = number of discrete capillary pressure bins
        """
        self.nbins = nbins  
        self.sigma_w = sigma_w
        R, n_caps,dtheta, theta_steps = petro.capillary_distribution(self.alpha,self.N,self.Theta_r,self.Theta_s,sigma_w = sigma_w, rho_w = 1000, g = 9.81, nbins = 1e2)
        self.R = R[0,0,:]
        self.n_caps = n_caps[0,0,:]
        self.dtheta = dtheta[0,0]
        self.theta_steps = theta_steps[0,0,:]

    def set_thermal_params(self, rho_l = 1000, rho_i = 917, rho_s = 2650, rho_a = 1.1, c_s = 800, c_l = 4180, c_i = 2100, c_a = 1000, lambda_l = 0.6, lambda_i = 2.2, lambda_s = 2.5, lambda_a = 0.023,H_f = 3.34e5):
        """
        Set thermal parameters for the simulation.
        c_l : specific heat capacity liquid water [J/kg/K]
        rho_l : density liquid water [kg/m3]
        c_s : specific heat capacity solid matrix [J/kg/K]
        rho_s : density solid matrix [kg/m3]
        c_i : specific heat capacity ice [J/kg/K]
        rho_i : density ice [kg/m3]
        c_a : specific heat capacity air [J/kg/K]
        rho_a : density air [kg/m3]
        """
        self.c_l = c_l
        self.rho_l = rho_l
        self.lambda_l = lambda_l
        self.c_s = c_s
        self.rho_s = rho_s
        self.lambda_s = lambda_s
        self.c_i = c_i
        self.rho_i = rho_i
        self.lambda_i = lambda_i
        self.c_a = c_a
        self.rho_a = rho_a
        self.lambda_a = lambda_a
        self.H_f = H_f
    
    def set_surface_bc(self, t_bc, q_inf, T_surface,method = "cubic2"):
        """
        Set boundary conditions for the simulation.
        The infiltration and Temperature data points are interpolated for the similation time.
        """
        assert q_inf.shape == (len(t_bc),), f"q_inf must be of shape ({len(t_bc)},)"
        assert T_surface.shape == (len(t_bc),), f"T_surface must be of shape ({len(t_bc)},)"
        self.t_bc = t_bc
        self.q_inf = q_inf
        self.T_surface = T_surface
        self.T_surface_interp = ipx.Interpolator1D(t_bc, T_surface,method=method)
        self.q_inf_interp = ipx.Interpolator1D(t_bc, q_inf,method=method)

    def set_newton_params(self, atol_rich=1e-6, rtol_rich=1e-4, atol_T=1e2, rtol_T=1e-2, max_iter=10, min_iter=5):
        """
        Set parameters for the Newton-Raphson method.
        """
        print("===============   Setting Newton-Raphson parameters   =============== ")
        print(f"Rtol: {rtol_rich,rtol_T}")
        print(f"Max Iterations: {max_iter}")
        print(f"Min Iterations: {min_iter}")

        self.rtol_rich = rtol_rich
        self.rtol_T = rtol_T
        self.atol_rich = atol_rich
        self.atol_T = atol_T
        self.max_iter = max_iter
        self.min_iter = min_iter

        print("DONE: set Newton-Raphson parameters.")
        print("=============================================================")

    def set_initial_conditions(self, htot, h_c, theta_l, theta_i, T_init=None):
        """
        Set initial conditions for the simulation.
        Parameters:
        - htot: Initial hydraulic head.
        - h_c: Initial capillary head.
        - theta: Initial water content.
        """
        print("===============   Setting initial conditions   =============== ")
        assert htot.shape == self.X.shape, f"htot must be of shape ({self.nz}, {self.nx})"
        assert h_c.shape == self.X.shape, f"h_c must be of shape ({self.nz}, {self.nx})"
        assert theta_l.shape == self.X.shape, f"theta_l must be of shape ({self.nz}, {self.nx})"
        assert theta_i.shape == self.X.shape, f"theta_i must be of shape ({self.nz}, {self.nx})"
        assert T_init.shape == self.X.shape, f"T_init must be of shape ({self.nz}, {self.nx})"
        self.initial_conditions = (htot, h_c, theta_l, theta_i, T_init)

        # initialise old variable states
        self.h_tot_old = htot
        self.h_c_old = h_c
        self.theta_l_old = theta_l
        self.theta_i_old = theta_i
        self.T_old = T_init
        print("DONE: set initial conditions.")
        print("=============================================================")

    def simulate_validation(self):
        """
        Fully coupled Richards + heat equation simulation.
        Returns:
            - h_total, h_c_total, theta_total: hydraulic results
            - T_total: temperature results
            - time_array: simulation times
            - iter_total: NR iterations per time step
            - normres_total: residual norm per time step
            - res_last: residual at last step
        """
        # number of parameters 
        # np = int(self.nx * self.nz)

        # initial conditions:
        t = self.t0
        htot = self.h_tot_old
        h_c = self.h_c_old
        theta_l = self.theta_l_old
        theta_i = self.theta_i_old
        T = self.T_old

        # Transform T to y = arcsinh(T/b) to improve convergence for large T variations (e.g. freezing/thawing)
        def T_of_y(y, b=1e-1):
            T = b * jnp.sinh(y)
            return T

        def y_of_T(T, b=1e-1):
            y = jnp.arcsinh(T / b)
            return y

        # Lists for saving simulation outputs. Initial conditions are included as first entry.
        h_total, h_c_total, theta_l_total, theta_i_total = [htot], [h_c], [theta_l], [theta_i]
        T_total = [T]
        iter_total, normres_total = [], []
        time_array = [t]
        
        # Time stepping loop variables
        t_next_save = self.save_t if self.save_t is not None else self.dt
        cons_success = 0
        sparsity = self.sparse_pattern

        # predicator step initial values    
        yT = y_of_T(T)
        y_old = jnp.concatenate([htot.ravel(), yT.ravel()])
        # y_older = y_old.copy()    
        y_new = y_old.copy()
        # dt_old = self.dt
        
        # Coupled residual function for Newton-Raphson solver. 
        def residual_func_cap_bundle(y_new, y_old, dt, soil, soil_type, Theta_s,Theta_r, alpha, N, Ksat, H_f, 
                                     rho_i, rho_l, rho_a, rho_s, 
                                     c_i, c_l, c_a, c_s,
                                     lambda_i, lambda_l, lambda_a, lambda_s,
                                     q_inf, T_surface, Z,  nz, nx, dx, dx_tilde, dz):
            """
            Coupled residual function for [htot, T].
            y_new = [htot_new.ravel(), T_new.ravel()]
            y_old = [htot_old.ravel(), T_old.ravel()]
            """
            np = nx * nz
            htot_new = jnp.reshape(y_new[:np], (nz, nx))
            yT_new   = jnp.reshape(y_new[np:], (nz, nx))

            htot_old = jnp.reshape(y_old[:np], (nz, nx))
            yT_old   = jnp.reshape(y_old[np:], (nz, nx))

            T_new = T_of_y(yT_new)
            T_old = T_of_y(yT_old)

            # Head and Heat residual (advection uses fluxes from h) 
            res_total, _, _, _, _, _, _, _ = volbal_validation(
                    T_new, T_old, htot_new, htot_old, q_inf, T_surface, 
                    dt, soil, soil_type, Theta_s,Theta_r, alpha, N, Ksat, H_f, 
                    rho_i, rho_l, rho_a, rho_s, 
                    c_i, c_l, c_a, c_s,
                    lambda_i, lambda_l, lambda_a, lambda_s,  
                    Z,  nz, nx, dx, dx_tilde, dz
            )
            return res_total
        
        # Sparse Jacobian for coupled system
        jacobian_func = sparsejac.jacfwd(
            residual_func_cap_bundle,
            sparsity=sparsity
        )

        # save initial fluxes
        _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_validation(T, T, htot, htot, self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), 
                                                                                                       self.dt, self.soil, self.soil_type, self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat, self.H_f, 
                                                                                                       self.rho_i, self.rho_l, self.rho_a, self.rho_s, 
                                                                                                       self.c_i, self.c_l, self.c_a, self.c_s,
                                                                                                       self.lambda_i, self.lambda_l, self.lambda_a, self.lambda_s, 
                                                                                                       self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz)
        qz_array = [qz]
        qx_array = [qx]
        qz_centered_array = [qz_centered]
        qx_centered_array = [qx_centered]
        q_runoff_array = [q_runoff]

        print("Starting simulation ...\n" \
        "==============================================================================")
        while t < self.t_end:

            repeat = True
            dt = self.dt
            # Adaptive time stepping loop
            while repeat:
                # dt actually used for this Newton solve (dt may still grow below, but that
                # only takes effect for the *next* step, not the one just solved)
                dt_solved = dt
                # predictor step for T using previous two time steps (for better convergence of coupled system)
                # Call coupled Newton-Raphson
                y_trial, iter_count, norm_rich, norm_T, converged = self.newtonRaphson_cap_bundle(y_old, y_old, t, dt, residual_func_cap_bundle, jacobian_func, self.atol_rich, self.rtol_rich, self.atol_T, self.rtol_T, self.max_iter)
                normres = jnp.sqrt(norm_rich**2 + norm_T**2)

                # Adaptive time step control
                if (iter_count >= self.max_iter and not converged) or jnp.isnan(norm_rich) or jnp.isnan(norm_T) or jnp.isinf(norm_rich) or jnp.isinf(norm_T):
                    dt /= 2
                    if dt < 1e-6:
                        print("Time step too small. Simulation diverging. Returning saved variables.")
                        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total
                    try:
                        print(f"Max iterations reached with ||res_rich||={norm_rich:.3e}, ||res_T||={norm_T:.3e}. Halving dt to {dt}")
                    except TypeError:
                        # see note above on the ||res||_inf print further below
                        print(f"Max iterations reached. Halving dt to {dt}")
                    repeat = True
                    cons_success = 0
                    
                elif iter_count <= self.min_iter and cons_success >= 2:
                    dt = min(dt * 1.2, self.maxdt)
                    # print(f"Few iterations. Doubling dt to {self.dt}")
                    repeat = False
                    cons_success = 0

                elif iter_count <= self.min_iter and cons_success < 2:
                    repeat = False
                    cons_success += 1

                else:
                    cons_success = 0
                    repeat = False
            
            try:
                print(f"Time {t/24/3600:.2f} d, dt={dt_solved:.2e}, NR iterations={iter_count}, ||res||_inf={normres:.3e}")
            except TypeError:
                # normres can be a JVPTracer/JaxprTracer when simulate_cap_bundle is run
                # under jax.jvp/grad (e.g. for parameter sensitivity analysis); its value
                # can't be formatted with :.3e in that case, so just drop it from the line.
                print(f"Time {t/24/3600:.2f} d, dt={dt_solved:.2e}, NR iterations={iter_count}")

            # Extract updated htot and T
            y_new = y_trial
            y_old = y_new.copy()
            htot_new = jnp.reshape(y_new[:self.np], (self.nz, self.nx))
            yT_new = jnp.reshape(y_new[self.np:], (self.nz, self.nx))
            T_new  = T_of_y(yT_new)

            t += dt_solved
            self.dt = dt

            # Store outputs at save times
            if t >= t_next_save:
                h_c = jnp.maximum(self.Z - htot_new, 0.0)
                T_old = T_of_y(jnp.reshape(y_old[self.np:], (self.nz, self.nx)))
                htot_old = jnp.reshape(y_old[:self.np], (self.nz, self.nx))
                                                                                                                
                _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_validation(T_new, T_old,
                                                                                                               htot_new, htot_old, self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), dt_solved,
                                                                                                               self.soil, self.soil_type, self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat, self.H_f,
                                                                                                               self.rho_i, self.rho_l, self.rho_a, self.rho_s,
                                                                                                               self.c_i, self.c_l, self.c_a, self.c_s,
                                                                                                               self.lambda_i, self.lambda_l, self.lambda_a, self.lambda_s,
                                                                                                               self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz)
                h_total.append(htot_new)
                h_c_total.append(h_c)
                theta_l_total.append(theta_l_s)
                theta_i_total.append(theta_i_s)
                T_total.append(T_new)
                iter_total.append(iter_count)
                normres_total.append(normres)
                time_array.append(t)
                q_runoff_array.append(q_runoff)
                qz_array.append(qz)
                qx_array.append(qx)
                qz_centered_array.append(qz_centered)
                qx_centered_array.append(qx_centered)
                t_next_save += self.save_t if self.save_t is not None else self.dt

        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total


    def simulate_cap_bundle(self):
        """
        Fully coupled Richards + heat equation simulation.
        Returns:
            - h_total, h_c_total, theta_total: hydraulic results
            - T_total: temperature results
            - time_array: simulation times
            - iter_total: NR iterations per time step
            - normres_total: residual norm per time step
            - res_last: residual at last step
        """
        # number of parameters 
        # np = int(self.nx * self.nz)

        # initial conditions:
        t = self.t0
        htot = self.h_tot_old
        h_c = self.h_c_old
        theta_l = self.theta_l_old
        theta_i = self.theta_i_old
        T = self.T_old

        # Transform T to y = arcsinh(T/b) to improve convergence for large T variations (e.g. freezing/thawing)
        def T_of_y(y, b=1e-1):
            T = b * jnp.sinh(y)
            return T

        def y_of_T(T, b=1e-1):
            y = jnp.arcsinh(T / b)
            return y

        # Lists for saving simulation outputs. Initial conditions are included as first entry.
        h_total, h_c_total, theta_l_total, theta_i_total = [htot], [h_c], [theta_l], [theta_i]
        T_total = [T]
        iter_total, normres_total = [], []
        time_array = [t]
        
        # Time stepping loop variables
        t_next_save = self.save_t if self.save_t is not None else self.dt
        cons_success = 0
        sparsity = self.sparse_pattern

        # predicator step initial values    
        yT = y_of_T(T)
        y_old = jnp.concatenate([htot.ravel(), yT.ravel()])
        # y_older = y_old.copy()    
        y_new = y_old.copy()
        # dt_old = self.dt
        
        # Coupled residual function for Newton-Raphson solver. 
        def residual_func_cap_bundle(y_new, y_old, dt, soil, soil_type, Theta_s,Theta_r, alpha, N, Ksat, H_f, 
                                     rho_i, rho_l, rho_a, rho_s, 
                                     c_i, c_l, c_a, c_s,
                                     lambda_i, lambda_l, lambda_a, lambda_s,
                                     q_inf, T_surface, Z,  nz, nx, dx, dx_tilde, dz):
            """
            Coupled residual function for [htot, T].
            y_new = [htot_new.ravel(), T_new.ravel()]
            y_old = [htot_old.ravel(), T_old.ravel()]
            """
            np = nx * nz
            htot_new = jnp.reshape(y_new[:np], (nz, nx))
            yT_new   = jnp.reshape(y_new[np:], (nz, nx))

            htot_old = jnp.reshape(y_old[:np], (nz, nx))
            yT_old   = jnp.reshape(y_old[np:], (nz, nx))

            T_new = T_of_y(yT_new)
            T_old = T_of_y(yT_old)

            # Head and Heat residual (advection uses fluxes from h) 
            res_total, _, _, _, _, _, _, _ = volbal_cappillary_bundle(
                    T_new, T_old, htot_new, htot_old, q_inf, T_surface, 
                    dt, soil, soil_type, Theta_s,Theta_r, alpha, N, Ksat, H_f, 
                    rho_i, rho_l, rho_a, rho_s, 
                    c_i, c_l, c_a, c_s,
                    lambda_i, lambda_l, lambda_a, lambda_s,  
                    Z,  nz, nx, dx, dx_tilde, dz
            )
            return res_total
        
        # Sparse Jacobian for coupled system
        jacobian_func = sparsejac.jacfwd(
            residual_func_cap_bundle,
            sparsity=sparsity
        )

        # save initial fluxes
        _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_cappillary_bundle(T, T, htot, htot, self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), 
                                                                                                       self.dt, self.soil, self.soil_type, self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat, self.H_f, 
                                                                                                       self.rho_i, self.rho_l, self.rho_a, self.rho_s, 
                                                                                                       self.c_i, self.c_l, self.c_a, self.c_s,
                                                                                                       self.lambda_i, self.lambda_l, self.lambda_a, self.lambda_s, 
                                                                                                       self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz)
        qz_array = [qz]
        qx_array = [qx]
        qz_centered_array = [qz_centered]
        qx_centered_array = [qx_centered]
        q_runoff_array = [q_runoff]

        print("Starting simulation ...\n" \
        "==============================================================================")
        while t < self.t_end:

            repeat = True
            dt = self.dt
            # Adaptive time stepping loop
            while repeat:
                # dt actually used for this Newton solve (dt may still grow below, but that
                # only takes effect for the *next* step, not the one just solved)
                dt_solved = dt
                # predictor step for T using previous two time steps (for better convergence of coupled system)
                # Call coupled Newton-Raphson
                y_trial, iter_count, norm_rich, norm_T, converged = self.newtonRaphson_cap_bundle(y_old, y_old, t, dt, residual_func_cap_bundle, jacobian_func, self.atol_rich, self.rtol_rich, self.atol_T, self.rtol_T, self.max_iter)
                normres = jnp.sqrt(norm_rich**2 + norm_T**2)

                # Adaptive time step control
                if (iter_count >= self.max_iter and not converged) or jnp.isnan(norm_rich) or jnp.isnan(norm_T) or jnp.isinf(norm_rich) or jnp.isinf(norm_T):
                    dt /= 2
                    if dt < 1e-6:
                        print("Time step too small. Simulation diverging. Returning saved variables.")
                        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total
                    try:
                        print(f"Max iterations reached with ||res_rich||={norm_rich:.3e}, ||res_T||={norm_T:.3e}. Halving dt to {dt}")
                    except TypeError:
                        # see note above on the ||res||_inf print further below
                        print(f"Max iterations reached. Halving dt to {dt}")
                    repeat = True
                    cons_success = 0
                    
                elif iter_count <= self.min_iter and cons_success >= 2:
                    dt = min(dt * 1.2, self.maxdt)
                    # print(f"Few iterations. Doubling dt to {self.dt}")
                    repeat = False
                    cons_success = 0

                elif iter_count <= self.min_iter and cons_success < 2:
                    repeat = False
                    cons_success += 1

                else:
                    cons_success = 0
                    repeat = False
            
            try:
                print(f"Time {t/24/3600:.2f} d, dt={dt_solved:.2e}, NR iterations={iter_count}, ||res||_inf={normres:.3e}")
            except TypeError:
                # normres can be a JVPTracer/JaxprTracer when simulate_cap_bundle is run
                # under jax.jvp/grad (e.g. for parameter sensitivity analysis); its value
                # can't be formatted with :.3e in that case, so just drop it from the line.
                print(f"Time {t/24/3600:.2f} d, dt={dt_solved:.2e}, NR iterations={iter_count}")

            # Extract updated htot and T
            y_new = y_trial
            y_old = y_new.copy()
            htot_new = jnp.reshape(y_new[:self.np], (self.nz, self.nx))
            yT_new = jnp.reshape(y_new[self.np:], (self.nz, self.nx))
            T_new  = T_of_y(yT_new)

            t += dt_solved
            self.dt = dt

            # Store outputs at save times
            if t >= t_next_save:
                h_c = jnp.maximum(self.Z - htot_new, 0.0)
                T_old = T_of_y(jnp.reshape(y_old[self.np:], (self.nz, self.nx)))
                htot_old = jnp.reshape(y_old[:self.np], (self.nz, self.nx))
                                                                                                                
                _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_cappillary_bundle(T_new, T_old,
                                                                                                               htot_new, htot_old, self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), dt_solved,
                                                                                                               self.soil, self.soil_type, self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat, self.H_f,
                                                                                                               self.rho_i, self.rho_l, self.rho_a, self.rho_s,
                                                                                                               self.c_i, self.c_l, self.c_a, self.c_s,
                                                                                                               self.lambda_i, self.lambda_l, self.lambda_a, self.lambda_s,
                                                                                                               self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz)
                h_total.append(htot_new)
                h_c_total.append(h_c)
                theta_l_total.append(theta_l_s)
                theta_i_total.append(theta_i_s)
                T_total.append(T_new)
                iter_total.append(iter_count)
                normres_total.append(normres)
                time_array.append(t)
                q_runoff_array.append(q_runoff)
                qz_array.append(qz)
                qx_array.append(qx)
                qz_centered_array.append(qz_centered)
                qx_centered_array.append(qx_centered)
                t_next_save += self.save_t if self.save_t is not None else self.dt

        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total
    
    def newtonRaphson_cap_bundle(self, y0, y_old, t, dt, residual_func, jacobian_func, atol_rich =1e-6, rtol_rich=1e-4, atol_T=1e2, rtol_T = 1e-2, max_iter=10):
        """
        Newton-Raphson solver for coupled [htot, T].
        A component is accepted once its residual is either below the tolerance
        in absolute terms or has shrunk by the required relative factor since
        the first iterate -- a purely relative test is ill-posed whenever the
        residual at y0 is already small (e.g., a near-equilibrium step), since
        no further relative reduction may be achievable even though the state
        has, in practice, converged.
        """
        def is_converged(norm, norm0, a_tol,r_tol):
            return (norm <= a_tol) | (norm <= r_tol * norm0)

        y_new, norm_rich0, norm_T0 = self.body_fun_cap_bundle(y0, y_old, t, dt, residual_func, jacobian_func)
        norm_rich, norm_T = norm_rich0, norm_T0
        iter_count = 1
        while not (is_converged(norm_rich, norm_rich0, atol_rich, rtol_rich) and is_converged(norm_T, norm_T0,atol_T, rtol_T)) and (iter_count < max_iter):
            y_new, norm_rich, norm_T = self.body_fun_cap_bundle(y_new, y_old, t, dt, residual_func, jacobian_func)
            iter_count += 1
        converged = is_converged(norm_rich, norm_rich0, atol_rich, rtol_rich) and is_converged(norm_T, norm_T0, atol_T, rtol_T)
        return y_new, iter_count, norm_rich, norm_T, converged

    def body_fun_cap_bundle(
        self, y_new, y_old, t, dt,
        residual_func, jacobian_func):

        res = residual_func(
            y_new, y_old, dt, self.soil, self.soil_type,
            self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat,self.H_f,
            self.rho_i, self.rho_l, self.rho_a, self.rho_s, 
            self.c_i, self.c_l, self.c_a, self.c_s,
            self.lambda_i, self.lambda_l, self.lambda_a, self.lambda_s,
            self.q_inf_interp(t)*jnp.ones((self.nx,)),
            self.T_surface_interp(t)*jnp.ones((self.nx,)),
            self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz
        )

        J = jacobian_func(
            y_new, y_old, dt, self.soil, self.soil_type,
            self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat, self.H_f, 
            self.rho_i, self.rho_l, self.rho_a, self.rho_s, 
            self.c_i, self.c_l, self.c_a, self.c_s,
            self.lambda_i, self.lambda_l, self.lambda_a, self.lambda_s,
            self.q_inf_interp(t)*jnp.ones((self.nx,)),
            self.T_surface_interp(t)*jnp.ones((self.nx,)),
            self.Z,  self.nz, self.nx, self.dx, self.dx_tilde, self.dz
        )

        J_csr = jax.experimental.sparse.BCSR.from_bcoo(J)

        # Newton step 
        dy = jax.experimental.sparse.linalg.spsolve(
            J_csr.data, J_csr.indices, J_csr.indptr, -res
        )

        # line search for global convergence as implemented in DuMux: https://dumux.org/docs/doxygen/master/group___newton.html
        alpha = 1.0
        np = len(res)//2
        res_norm = jnp.linalg.norm(res, ord=2)

        while True:

            y_trial = y_new + alpha * dy

            res_trial = residual_func(
                y_trial, y_old, dt, self.soil, self.soil_type,
                self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat,self.H_f, 
                self.rho_i, self.rho_l, self.rho_a, self.rho_s, 
                self.c_i, self.c_l, self.c_a, self.c_s,
                self.lambda_i, self.lambda_l, self.lambda_a, self.lambda_s,
                self.q_inf_interp(t)*jnp.ones((self.nx,)),
                self.T_surface_interp(t)*jnp.ones((self.nx,)),
                self.Z,  self.nz, self.nx, self.dx, self.dx_tilde, self.dz
            )

            trial_norm = jnp.linalg.norm(res_trial, ord=2)

            if trial_norm <= res_norm:
                break

            alpha *= 0.5

            # As in DuMux accept alpha even if it leads to an increase in residual.
            if alpha < 1e-6:
                break

        norm_rich = jnp.linalg.norm(res_trial[:np], ord=2)
        norm_T = jnp.linalg.norm(res_trial[np:], ord=2)
        return y_trial, norm_rich, norm_T