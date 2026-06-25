import sys
sys.path.append('../')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import utils.petrophysic as petro
from utils.volbal import volbal_coupled_drying, volbal_coupled_permaFoam, volbal_cappillary_bundle
import sparsejac as sparsejac
import interpax as ipx
import time
from functools import wraps, partial

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
    
        self.dx_tilde = jnp.abs(self.X[0, 1] - self.X[0, 0])  
        # Neighboring cells for finite volume discretization
        self.neighbors = []
        self.color = jnp.empty((self.nz,self.nx))

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
        Calculates the sparsity pattern for the Jacobian, to be given to sparse.jacfwd.
        """
        print("===============   Calculating sparsity pattern   =============== ")

        N = self.nx*self.nz
        data = []
        indices = []
        for i in range(N):
            data.append(1)  # diagonal always nonzero
            indices.append((i, i))
            for j in self.neighbors[i]:
                data.append(1)  # FV neighbors couple
                indices.append((i, j))
        self.sparse_pattern = jax.experimental.sparse.BCOO((data, indices), shape=(N, N))
        print("DONE: calculated sparsity pattern with ", len(data), "nonzero entries.")
        print("=============================================================")

    def set_sparsity_pattern_coupled(self):
        """
        Calculates sparsity pattern for fully coupled Jacobian [htot, T].
        """
        print("===============   Calculating coupled sparsity pattern   =============== ")
        N = self.nx * self.nz
        N_total = 2 * N  # htot + T
        data = []
        indices = []

        # Loop over all cells
        for i in range(N):
            # diagonal h-h
            data.append(1)
            indices.append([i, i])
            # diagonal T-T
            data.append(1)
            indices.append([N+i, N+i])
            # coupling h-T and T-h
            data.append(1)
            indices.append([i, N+i])
            data.append(1)
            indices.append([N+i, i])

            # FV neighbors (h-h and T-T)
            for j in self.neighbors[i]:
                data.append(1); 
                indices.append([i, N+j])     # h-T neighbor
                data.append(1); 
                indices.append([N+i, j])     # T-h neighbor
                data.append(1)
                indices.append([i, j])     # h-h
                data.append(1)
                indices.append([N+i, N+j]) # T-T
        indices = jnp.array(indices)
        order = jnp.lexsort([indices[:,0],indices[:,1]])
        # order = list(range(1,len(indices)))
        # print(type(order[0]))
        indices = indices[order]
        data  =  jnp.array(data)[order]

        self.sparse_pattern = jax.experimental.sparse.BCOO((data, indices),
                                                        shape=(N_total, N_total))
        bcsr_sparse = jax.experimental.sparse.BCSR.from_bcoo(self.sparse_pattern)
        self.indices = bcsr_sparse.indices
        self.indptr = bcsr_sparse.indptr
        print("DONE: calculated coupled sparsity pattern with", len(data), "nonzero entries.")
        print("==========================================================================")

    def time_discretization(self, dt=None, t_end=None, maxdt=None, CN=None, save_t=None):
        """
        Create a time discretization for the simulation.
        dt = initial time step size                         [s]
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
        assert CN != None, "t_end is not set. Please set the end of the simulation time"

        self.dt = dt
        self.t_end = t_end
        self.maxdt = maxdt
        self.nt = int(t_end / dt)
        self.CN = CN
        self.save_t = save_t
        print(f"DONE: created time discretization.")
        print("=============================================================")
   
    def set_soilType(self, soil, soil_type):
        """
        Set soil type for the simulation.
        soil = 2D array of shape (nz, nx) with integer values representing soil types.
        """
        assert soil.shape == self.X.shape, f"soil must be of shape ({self.nz}, {self.nx})"
        assert isinstance(soil_type, dict), "soil_type must be a dictionary mapping integer soil types to their properties."
        self.soil = soil
        self.soil_type = soil_type

        # Create array with soil parameters based on soil type
        self.Theta_s = jnp.where(self.soil==1, self.soil_type[1]["Theta_s"], self.soil_type[2]["Theta_s"])
        self.Theta_r = jnp.where(self.soil==1, self.soil_type[1]["Theta_r"], self.soil_type[2]["Theta_r"])
        self.alpha = jnp.where(self.soil==1, self.soil_type[1]["alpha"], self.soil_type[2]["alpha"])
        self.N = jnp.where(self.soil==1, self.soil_type[1]["N"], self.soil_type[2]["N"])
        self.Ksat = jnp.where(self.soil==1, self.soil_type[1]["Ksat"], self.soil_type[2]["Ksat"])
        
    def set_vanGenuchten_params(self, Ksat, alpha, N, Theta_r, Theta_s):
        """
        Set van Genuchten parameters for the simulation.
        """
        assert Ksat.shape == self.X.shape, f"Ksat must be of shape ({self.nz}, {self.nx})"
        assert alpha.shape == self.X.shape, f"alpha must be of shape ({self.nz}, {self.nx})"
        assert N.shape == self.X.shape, f"N must be of shape ({self.nz}, {self.nx})"
        assert Theta_r.shape == self.X.shape, f"Theta_r must be of shape ({self.nz}, {self.nx})"
        assert Theta_s.shape == self.X.shape, f"Theta_s must be of shape ({self.nz}, {self.nx})"

        self.Ksat = Ksat
        self.alpha = alpha
        self.N = N
        self.Theta_r = Theta_r
        self.Theta_s = Theta_s

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

    def set_thermal_params(self, rho_l = 1000, rho_i = 917, rho_s = 2650, rho_a = 1200, c_s = 800, c_l = 4180, c_i = 2100, c_a = 1000, H_f = 3.34e5):
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
        self.c_s = c_s
        self.rho_s = rho_s
        self.c_i = c_i
        self.rho_i = rho_i
        self.c_a = c_a
        self.rho_a = rho_a
        self.H_f = H_f
    
    def set_surface_bc(self, t_bc, q_inf, T_surface):
        """
        Set boundary conditions for the simulation.
        """
        assert q_inf.shape == (len(t_bc),), f"q_inf must be of shape ({len(t_bc)},)"
        assert T_surface.shape == (len(t_bc),), f"T_surface must be of shape ({len(t_bc)},)"
        self.t_bc = t_bc
        self.q_inf = q_inf
        self.T_surface = T_surface
        self.T_surface_interp = ipx.Interpolator1D(t_bc, T_surface,method="cubic2")
        self.q_inf_interp = ipx.Interpolator1D(t_bc, q_inf,method="cubic2")

    def set_surface_temperature(self, T_surface):
        """
        Set surface temperature boundary condition.
        """
        assert T_surface.shape == (self.nx,), f"T_surface must be of shape ({self.nx},)"
        self.T_surface = T_surface

    def set_solver_params(self, restol_rich=1e-4, restol_T=1e-2, max_iter=10, min_iter=5, solver='sparse'):
        """
        Set parameters for the Newton-Raphson method.
        """
        print("===============   Setting Newton-Raphson parameters   =============== ")
        print(f"Restol: {restol_rich,restol_T}")
        print(f"Max Iterations: {max_iter}")
        print(f"Min Iterations: {min_iter}")

        if solver == 'newton':
            self.solver = solver
        else:
            raise ValueError("Invalid solver type. Choose 'newton'.") 
        self.restol_rich = restol_rich
        self.restol_T = restol_T
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
        self.h_tot_old = htot
        self.h_c_old = h_c
        self.theta_l_old = theta_l
        self.theta_i_old = theta_i
        self.T_old = T_init
        print("DONE: set initial conditions.")
        print("=============================================================")


    # @partial(jax.jit,static_argnames=["self"])
    def simulate_coupled(self, parametrization="drying"):
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
        np = int(self.nx * self.nz)

        # Concatenate initial conditions: [htot.ravel(), T.ravel()]
        htot = self.h_tot_old
        h_c = self.h_c_old
        theta_l = self.theta_l_old
        theta_i = self.theta_i_old
        T = self.T_old
        y_old = jnp.concatenate([htot.ravel(), T.ravel()])
        y_new = y_old.copy()

        # Lists for storing outputs
        h_total, h_c_total, theta_l_total, theta_i_total = [htot], [h_c], [theta_l], [theta_i]
        T_total = [T]
        iter_total, normres_total = [], []
        time_array = [0]
        
        t = 0
        t_next_save = self.save_t if self.save_t is not None else self.dt
        cons_success = 0
        sparsity = self.sparse_pattern
        
        def residual_func_coupled(y_new, y_old, Theta_s, Theta_r, alpha, N, Ksat, rho_i, H_f, rho_l, c_l, q_inf, T_surface, X, Z, dt, CN, nz, nx):
            """
            Coupled residual function for [htot, T].
            y_new = [htot_new.ravel(), T_new.ravel()]
            y_old = [htot_old.ravel(), T_old.ravel()]
            """
            np = nx * nz
            htot_new = jnp.reshape(y_new[:np], (nz, nx))
            T_new    = jnp.reshape(y_new[np:], (nz, nx))
            htot_old = jnp.reshape(y_old[:np], (nz, nx))
            T_old    = jnp.reshape(y_old[np:], (nz, nx))
            if parametrization == "drying":
                # Head and Heat residual (advection uses fluxes from h) 
                res_total, _, _, _, _, _, _,_ = volbal_coupled_drying(
                        T_new, T_old, htot_new, htot_old, q_inf, T_surface, 
                        dt, Theta_s, Theta_r, alpha, N, Ksat, rho_i, H_f, 
                        rho_l, c_l, X, Z, CN
                )
            elif parametrization == "permaFoam":
                res_total, _, _, _, _, _, _,_ = volbal_coupled_permaFoam(
                        T_new, T_old, htot_new, htot_old, q_inf, T_surface, 
                        dt, Theta_s, Theta_r, alpha, N, Ksat, rho_i, H_f, 
                        rho_l, c_l, X, Z, CN
                )
            return res_total
        # Sparse Jacobian for coupled system
        jacobian_func = sparsejac.jacfwd(
            residual_func_coupled,
            sparsity=sparsity
        )
        
        # save initial fluxes
        if parametrization == "drying":
             _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_coupled_drying(T, jnp.reshape(y_old[:np], (self.nz, self.nx)), htot,jnp.reshape(y_old[np:], (self.nz, self.nx)),  self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), self.dt, self.Theta_s, self.Theta_r, self.alpha, self.N, self.Ksat, self.rho_i, self.H_f, self.rho_l, self.c_l,self.X, self.Z, self.CN)
        elif parametrization == "permaFoam":
             _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_coupled_permaFoam(T, jnp.reshape(y_old[:np], (self.nz, self.nx)), htot,jnp.reshape(y_old[np:], (self.nz, self.nx)),  self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), self.dt, self.Theta_s, self.Theta_r, self.alpha, self.N, self.Ksat, self.rho_i, self.H_f, self.rho_l, self.c_l,self.X, self.Z, self.CN)
        qz_array = [qz]
        qx_array = [qx]
        qz_centered_array = [qz_centered]
        qx_centered_array = [qx_centered]
        q_runoff_array = [q_runoff] 

        print("Starting simulation ...\n" \
        "==============================================================================")
        while t < self.t_end:
            # Adaptive time stepping loop
            repeat = True
            while repeat:
                dt = self.dt
                # Call coupled Newton-Raphson
                y_trial, iter_count, normres = self.newtonRaphson_coupled(y_old, y_old,t, dt, residual_func_coupled, jacobian_func, self.restol, self.max_iter)
                # Extract updated htot and T
                htot_new = jnp.reshape(y_trial[:np], (self.nz, self.nx))
                T_new    = jnp.reshape(y_trial[np:], (self.nz, self.nx))

                # Adaptive time step control
                if (iter_count >= self.max_iter and normres>self.restol) or jnp.isnan(normres) or jnp.isinf(normres):
                    self.dt /= 2
                    if self.dt < 1e-6:
                        print("Time step too small. Simulation diverging. Returning saved variables.")
                        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total
                    print(f"Max iterations reached with norm {normres}. Halving dt to {self.dt}")
                    repeat = True
                    cons_success = 0
                elif iter_count <= self.min_iter and cons_success >= 2:
                    self.dt = min(self.dt * 1.2, self.maxdt)
                    # print(f"Few iterations. Doubling dt to {self.dt}")
                    repeat = False
                    cons_success = 0
                elif iter_count <= self.min_iter:
                    repeat = False
                    cons_success += 1
                else:
                    repeat = False
                    cons_success =0
                
            
            # Update variables for next time step
            y_new = y_trial.copy()
            htot_new = jnp.reshape(y_new[:np], (self.nz, self.nx))
            T_new    = jnp.reshape(y_new[np:], (self.nz, self.nx))
            y_old = y_new.copy()
            t += dt

            # Store outputs at save times
            if t >= t_next_save:
                h_c = jnp.maximum(self.Z - htot_new, 0.0)
                if parametrization == "drying":
                    _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_coupled_drying(T_new, jnp.reshape(y_old[np:], (self.nz, self.nx)), htot_new,jnp.reshape(y_old[:np], (self.nz, self.nx)),  self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), dt, self.Theta_s, self.Theta_r, self.alpha, self.N, self.Ksat, self.rho_i, self.H_f, self.rho_l, self.c_l,self.X, self.Z, self.CN)
                elif parametrization == "permaFoam":
                    _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_coupled_permaFoam(T_new, jnp.reshape(y_old[np:], (self.nz, self.nx)), htot_new,jnp.reshape(y_old[:np], (self.nz, self.nx)),  self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), dt, self.Theta_s, self.Theta_r, self.alpha, self.N, self.Ksat, self.rho_i, self.H_f, self.rho_l, self.c_l,self.X, self.Z, self.CN)
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
                t_next_save += self.save_t if self.save_t is not None else dt
            print(f"Time {t/24/3600:.2f} d, dt={dt:.2e}, NR iterations={iter_count}, ||res||_inf={normres:.3e}")

        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total
    
    #@partial(jax.jit, static_argnames=['self', 'residual_func', 'jacobian_func'])
    def newtonRaphson_coupled(self, y0, y_old, t, dt, residual_func, jacobian_func, restol=1e-4, max_iter=10):
        """
        Newton-Raphson solver for coupled [htot, T] using explicit residual inputs.
        """
        y_new = y0
        normres = 1
        iter_count = 0

        #jax.lax.while_loop(self.cond_fun, self.body_fun_coupled, init_val)
        while (normres > restol) and (iter_count < max_iter):
            y_new, normres = self.body_fun_coupled(y_new, y_old, t, dt, residual_func, jacobian_func)
            iter_count += 1

        return y_new, iter_count, normres
    
    # @partial(jax.jit, static_argnames=['selnewtonRaphson_cap_bundlef', 'residual_func', 'jacobian_func'])
    def body_fun_coupled(self, y_new, y_old, t, dt, residual_func, jacobian_func, maxdh=1e-1,maxdT=1e-1):
        #start_time = time.time()
        res = residual_func(y_new, y_old, self.Theta_s, self.Theta_r, self.alpha, self.N, self.Ksat, self.rho_i, self.H_f, self.rho_l, self.c_l,  self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), self.X, self.Z, dt, self.CN, self.nz, self.nx)
        #print("res ", time.time()-start_time)
        # Compute Jacobian
        #start_time = time.time()
        J = jacobian_func(y_new, y_old, self.Theta_s, self.Theta_r, self.alpha, self.N, self.Ksat, self.rho_i, self.H_f, self.rho_l, self.c_l, self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), self.X, self.Z, dt, self.CN, self.nz, self.nx)
        #print("Jacobian: ", time.time()-start_time)
        J_csr = jax.experimental.sparse.BCSR.from_bcoo(J)
        dy = jax.experimental.sparse.linalg.spsolve(J_csr.data,J_csr.indices,J_csr.indptr,-res)

        # # --- step limiting (your existing limiter) ---
        # dh = dy[:len(dy)//2]
        # dT = dy[len(dy)//2:]
        # # print("max dh:", jnp.max(jnp.abs(dh)), "max dT:", jnp.max(jnp.abs(dT)))
        
        # fac_h = jnp.where(jnp.max(jnp.abs(dh)) > maxdh,
        #                 maxdh / jnp.max(jnp.abs(dh)), 1.0)

        # fac_T = jnp.where(jnp.max(jnp.abs(dT)) > maxdT,
        #                 maxdT / jnp.max(jnp.abs(dT)), 1.0)

        # fac = jnp.minimum(fac_h, fac_T)

        # dy = fac * dy

        # # --- globally convergent Newton (line search) ---
        # delta = 1.0
        # res_norm = jnp.linalg.norm(res, ord=jnp.inf)

        # while True:

        #     y_trial = y_new + delta * dy

        #     res_trial = residual_func(y_trial, y_old, 
        #         self.Theta_s, self.Theta_r, self.alpha, self.N, self.Ksat, 
        #         self.rho_i, self.H_f, self.rho_l, self.c_l,  
        #         self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), 
        #         self.X, self.Z, dt, self.CN, self.nz, self.nx
        #     )

        #     trial_norm = jnp.linalg.norm(res_trial, ord=jnp.inf)

        #     if trial_norm < res_norm:
        #         break

        #     if delta < 1e-4:
        #         break

        #     delta *= 0.5

        # y_new = y_new + delta * dy
        # normres = trial_norm
        y_new = y_new + dy
        normres = jnp.linalg.norm(res, ord=jnp.inf)

        return (y_new, normres)

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
        np = int(self.nx * self.nz)

        # initial conditions:
        t = 0
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
        time_array = [0]
        
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
        def residual_func_cap_bundle(y_new, y_old, dt, soil, soil_type, Theta_s,Theta_r, alpha, N, Ksat, rho_i, H_f, rho_l, c_l, q_inf, T_surface, Z,  nz, nx, dx, dx_tilde, dz):
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
                    dt, soil, soil_type, Theta_s,Theta_r, alpha, N, Ksat, rho_i, H_f, 
                    rho_l, c_l, Z,  nz, nx, dx, dx_tilde, dz
            )
            return res_total
        
        # Sparse Jacobian for coupled system
        jacobian_func = sparsejac.jacfwd(
            residual_func_cap_bundle,
            sparsity=sparsity
        )

        # save initial fluxes
        _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_cappillary_bundle(T, T, htot, htot, self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), 
                                                                                                       self.dt, self.soil, self.soil_type, self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat,
                                                                                                       self.rho_i, self.H_f, self.rho_l, self.c_l,self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz)
        qz_array = [qz]
        qx_array = [qx]
        qz_centered_array = [qz_centered]
        qx_centered_array = [qx_centered]
        q_runoff_array = [q_runoff]

        print("Starting simulation ...\n" \
        "==============================================================================")
        while t < self.t_end:

            repeat = True

            # Adaptive time stepping loop
            while repeat:

                # predictor step for T using previous two time steps (for better convergence of coupled system)
                dt = self.dt
                # Call coupled Newton-Raphson
                y_trial, iter_count, normres_rich,normres_T = self.newtonRaphson_cap_bundle(y_old, y_old, t, dt, residual_func_cap_bundle, jacobian_func, self.restol_rich, self.restol_T, self.max_iter)

                # Adaptive time step control
                if (iter_count >= self.max_iter and (normres_rich>self.restol_rich or normres_T>self.restol_T)) or jnp.isnan(normres_rich) or jnp.isnan(normres_T) or jnp.isinf(normres_rich) or jnp.isinf(normres_T):
                    self.dt /= 2
                    if self.dt < 1e-6:
                        print("Time step too small. Simulation diverging. Returning saved variables.")
                        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total
                    print(f"Max iterations reached with norm_res {normres_rich} and norm_T {normres_T}. Halving dt to {self.dt}")
                    repeat = True
                    cons_success = 0
                    
                elif iter_count <= self.min_iter and cons_success >= 2:
                    self.dt = min(self.dt * 1.2, self.maxdt)
                    # print(f"Few iterations. Doubling dt to {self.dt}")
                    repeat = False
                    cons_success = 0

                elif iter_count <= self.min_iter and cons_success < 2:
                    repeat = False
                    cons_success += 1

                else:
                    cons_success = 0
                    repeat = False
            
            # Extract updated htot and T
            y_new = y_trial
            y_old = y_new.copy()
            htot_new = jnp.reshape(y_new[:np], (self.nz, self.nx))
            yT_new = jnp.reshape(y_new[np:], (self.nz, self.nx))
            T_new  = T_of_y(yT_new)
            t += dt

            # Store outputs at save times
            if t >= t_next_save:
                h_c = jnp.maximum(self.Z - htot_new, 0.0)
                T_old = T_of_y(jnp.reshape(y_old[np:], (self.nz, self.nx)))
                htot_old = jnp.reshape(y_old[:np], (self.nz, self.nx))
                                                                                                                
                _, q_runoff, qx, qz, qx_centered, qz_centered, theta_l_s, theta_i_s = volbal_cappillary_bundle(T_new, T_old, 
                                                                                                               htot_new, htot_old, self.q_inf_interp(t)*jnp.ones((self.nx,)), self.T_surface_interp(t)*jnp.ones((self.nx,)), self.dt, 
                                                                                                               self.soil, self.soil_type, self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat, self.rho_i, self.H_f, self.rho_l, self.c_l,
                                                                                                               self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz)
                h_total.append(htot_new)
                h_c_total.append(h_c)
                theta_l_total.append(theta_l_s)
                theta_i_total.append(theta_i_s)
                T_total.append(T_new)
                iter_total.append(iter_count)
                normres_total.append(normres_rich)
                time_array.append(t)
                q_runoff_array.append(q_runoff)
                qz_array.append(qz)
                qx_array.append(qx)
                qz_centered_array.append(qz_centered)
                qx_centered_array.append(qx_centered)
                t_next_save += self.save_t if self.save_t is not None else self.dt

            
            print(f"Time {t/24/3600:.2f} d, dt={self.dt:.2e}, NR iterations={iter_count}, ||res_rich||_inf={normres_rich:.3e}, ||res_T||_inf={normres_T:.3e}")

        return h_total, h_c_total, theta_l_total,theta_i_total, T_total, q_runoff_array, qx_array,qz_array,qx_centered_array,qz_centered_array, time_array, iter_total, normres_total
    
    def newtonRaphson_cap_bundle(self, y0, y_old, t, dt, residual_func, jacobian_func, restol_rich=1e-4, restol_T = 1e-2, max_iter=10):
        """
        Newton-Raphson solver for coupled [htot, T] using explicit residual inputs.
        """
        # def cond_fun(state):
        #     _, normres, iter_count = state
        #     return jnp.logical_and(normres > restol, iter_count < max_iter)

        # def body_fun(state):
        #     y_new, _, iter_count = state
        #     y_new, normres = self.body_fun_cap_bundle(y_new, y_old, t, dt, residual_func, jacobian_func)
        #     return y_new, normres, iter_count + 1
        # init_state = (y0, 1.0, 0)  # (y_new, normres, iter_count)
        # y_new, normres, iter_count = jax.lax.while_loop(cond_fun, body_fun, init_state)
        # return y_new, iter_count, normres
        y_new0, normres_rich0, normres_T0 = self.body_fun_cap_bundle(y0, y_old, t, dt, residual_func, jacobian_func)
        normres_rich = normres_rich0
        normres_T = normres_T0
        iter_count = 1
        y_new = y_new0
        while ((normres_rich/normres_rich0 > restol_rich) or (normres_T/normres_T0 > restol_T))  and (iter_count < max_iter):
            y_new, normres_rich, normres_T = self.body_fun_cap_bundle(y_new, y_old, t, dt, residual_func, jacobian_func)
            iter_count += 1
        return y_new, iter_count, normres_rich, normres_T

    def body_fun_cap_bundle(
        self, y_new, y_old, t, dt,
        residual_func, jacobian_func,
        c=1e-4, maxdh=1e-1, maxdT=1e-1):

        res = residual_func(
            y_new, y_old, dt, self.soil, self.soil_type,
            self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat,
            self.rho_i, self.H_f, self.rho_l, self.c_l,
            self.q_inf_interp(t)*jnp.ones((self.nx,)),
            self.T_surface_interp(t)*jnp.ones((self.nx,)),
            self.Z, self.nz, self.nx, self.dx, self.dx_tilde, self.dz
        )

        J = jacobian_func(
            y_new, y_old, dt, self.soil, self.soil_type,
            self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat,
            self.rho_i, self.H_f, self.rho_l, self.c_l,
            self.q_inf_interp(t)*jnp.ones((self.nx,)),
            self.T_surface_interp(t)*jnp.ones((self.nx,)),
            self.Z,  self.nz, self.nx, self.dx, self.dx_tilde, self.dz
        )

        J_csr = jax.experimental.sparse.BCSR.from_bcoo(J)

        # Newton step 
        dy = jax.experimental.sparse.linalg.spsolve(
            J_csr.data, J_csr.indices, J_csr.indptr, -res
        )
        # # step limiting
        # dh = dy[:len(dy)//2]
        # dT = dy[len(dy)//2:]
        
        # fac_h = jnp.where(jnp.max(jnp.abs(dh)) > maxdh,
        #                 maxdh / jnp.max(jnp.abs(dh)), 1.0)

        # fac_T = jnp.where(jnp.max(jnp.abs(dT)) > maxdT,
        #                 maxdT / jnp.max(jnp.abs(dT)), 1.0)

        # fac = jnp.minimum(fac_h, fac_T)

        # dy = fac * dy

        # line search for global convergence as implemented in DuMux: https://dumux.org/docs/doxygen/master/group___newton.html
        alpha = 1.0
        np = len(res)//2
        res_norm = jnp.linalg.norm(res, ord=2)

        while True:

            y_trial = y_new + alpha * dy

            res_trial = residual_func(
                y_trial, y_old, dt, self.soil, self.soil_type,
                self.Theta_s,self.Theta_r, self.alpha, self.N, self.Ksat,
                self.rho_i, self.H_f, self.rho_l, self.c_l,
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
                res_rich = jnp.linalg.norm(res[:np], ord=2)
                res_T = jnp.linalg.norm(res[np:], ord=2)
                return y_trial,res_rich,res_T
        # y_trial = y_new +dy
        # trial_norm = jnp.linalg.norm(res, ord=jnp.inf)
        res_rich = jnp.linalg.norm(res[:np], ord=2)
        res_T = jnp.linalg.norm(res[np:], ord=2)
        # normres = res_norm
        return y_trial, res_rich, res_T