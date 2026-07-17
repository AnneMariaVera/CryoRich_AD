import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.animation as animation
from scipy.interpolate import griddata
from scipy.integrate import cumulative_trapezoid
import cmocean.cm as cmo
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from datetime import timedelta
import jax.numpy as jnp


# Plotting settings
cobalt=(0.0, 0.28, 0.67)
persian_red = (0.8,0.2,0.2)
carolina_blue = (0.6,0.73,0.89)
red=(255/255,193/255,7/255)
orange = (216/255,27/255,96/255)
green = (30/255, 136/255, 229/255)
blue = (0, 77/255, 46/255)

N = 256

start = np.array([168, 148, 125]) / 255
end   = np.array([250, 252, 255]) / 255      # icy white with a bit more blue

t = np.linspace(0, 1, N)
t = t

vals = np.ones((N, 4))
vals[:, :3] = (1 - t)[:, None] * start + t[:, None] * end

icy_soil = ListedColormap(vals)

cm = 1/2.54  # centimeters in inches
plt.rcParams['font.size'] = 8

PAGE_WIDTH_CM = 16.46   
PAGE_HEIGHT_CM = 22.41 

PARAMETRIZATION_STYLE = {
    "drying":             {"color": "#0072B2", "linestyle": "-"},        # blue
    "drying_omega10":     {"color": "#56B4E9", "linestyle": "--"},       # sky blue
    "permaFoam_omega10":  {"color": "#D55E00", "linestyle": "-."},       # vermillion
    "permaFoam_omega50":  {"color": "#E69F00", "linestyle": ":"},        # orange
    "cap":                {"color": "#009E73", "linestyle": (0, (3, 1, 1, 1))},  # bluish green
}

def parametrization_style(parametrization):
    return PARAMETRIZATION_STYLE.get(parametrization, {"color": "black", "linestyle": "-"})

def plot_grid(geom):
    """Plot the grid and nodes of the geometry."""
    fig, ax = plt.subplots(figsize=(30*cm, 20*cm))
    ax.set_aspect('equal')
    ax.plot(geom.X,geom.Z,".r")
    ax.plot(geom.X[0,0].T,geom.Z[0,0].T,".r",label="nodes")
    ax.plot(geom.X_plot,geom.Z_plot,"-k")
    ax.plot(geom.X_plot[0,0].T,geom.Z_plot.T[0,0],"-k",label="grid")
    ax.plot(geom.X_plot.T,geom.Z_plot.T,"-k")
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.spines['left'].set_position(('outward', 10))
    ax.title.set_text("Discretization of the domain")
    ax.legend(loc='lower left',facecolor='none',edgecolor='none')
    fig.tight_layout()
    return fig, ax

def plot_sparsity_pattern(geom):
    """Plot the sparsity pattern of jacobian matrix."""
    S = geom.sparse_pattern
    indices = np.array(S.indices)
    data = np.array(S.data)

    # if indices has shape (nnz, 2) -> separate rows/cols
    rows = indices[:, 0].astype(int)
    cols = indices[:, 1].astype(int)

    A_coo = coo_matrix((np.ones_like(data), (rows, cols)), shape=S.shape)  # use ones to show pattern only
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.spy(A_coo, markersize=10)
    return fig, ax

def plot_domain(X, Z, cellValues, colormap="Blues", title=" ", colorbar_label=None):
    """Plot cell values over the geometry."""
    fig, ax = plt.subplots(1,2,figsize=(19*cm, 7*cm), gridspec_kw={'width_ratios': [1, 0.4]})
    ax[1].set_aspect('equal')
    cbar = ax[0].pcolormesh(X, Z, cellValues, cmap=colormap,shading = "gouraud", linewidth=0.05)
    ax[0].plot(X[-1,:],Z[-1,:],"-", color = "#D55E00", linewidth = 2,zorder=1000)

    ax[0].plot(X[:,0],Z[:,0],"-", color = "#56B4E9",linewidth = 2, zorder=1004)
    ax[0].plot(X[:, -1], Z[:, -1], "-", color = "#009E73",linewidth = 2, zorder=1004)
    ax[0].plot(X[0,:], Z[0,:], "-", color = "#56B4E9", linewidth = 2, zorder=1006)
    ax[0].plot(X[-1,:],-.25-0.1*X[-1,:],"-k",zorder=1002) 
    ax[0].plot(X[-1,:],-.5-0.1*X[-1,:],"-k",zorder=1000)


    ax[0].spines[['top','right']].set_visible(False)
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("z [m]")
    eps = 0.1
    ax[0].set_xlim(X.min()-eps, X.max()+eps)
    ax[0].set_ylim(Z.min()-eps, Z.max()+eps)

    ax[0].spines['left'].set_position(('outward', 10))
    if title is not None:
        ax[0].title.set_text(title)

    x = np.array([0-0.001,.025+0.001])
    y_l = -0.1*x
    y_u = -0.1*x+.025
    ax[1].plot(0.0125,0.0125,"o",color=persian_red,markersize=9,zorder=1001)
    ax[1].text(0.0125,0.01,r"primary var.", color=persian_red, zorder=1001, va="top", ha="center")
    ax[1].plot(x,y_l,"-k",zorder=1001)
    ax[1].plot(np.array([0,0.025]),np.array([0,0]),"--k",zorder=1001)
    ax[1].plot(x,y_u,"-k",zorder=1001)
    ax[1].vlines(x=0,ymax=0.025+0.001,ymin=-0.001, color="k", zorder=1002)
    ax[1].vlines(x=0.025,ymax=y_u[-1]+0.001,ymin=y_l[-1]-0.001, color="k", zorder=1003)
    ax[1].text(-0.005,0.0125,r"$\Delta z$", color="k")
    ax[1].text(0.0125,0.0005,r"$\Delta x$", color="k")
    ax[1].text(0.0125,0.025,r"$\overline{\Delta x}$", color="k")
    ax[1].set_axis_off()
    fig.tight_layout()
    return fig, ax

def plot_cellValues(X, Z, cellValues, colormap="Blues", title=" ", colorbar_label=None):
    """Plot cell values over the geometry."""
    fig, ax = plt.subplots(figsize=(30*cm, 20*cm))
    ax.set_aspect('equal')

    cbar = plt.pcolormesh(X, Z, cellValues, cmap=colormap,shading = "gouraud", linewidth=0.05)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.05)
    fig.colorbar(cbar, cax=cax, label=colorbar_label)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    eps = 0.1
    ax.set_xlim(X.min()-eps, X.max()+eps)
    ax.set_ylim(Z.min()-eps, Z.max()+eps)

    ax.spines['left'].set_position(('outward', 10))
    if title is not None:
        ax.title.set_text(title)
    fig.tight_layout()
    return fig, ax

def plot_lineProfile(geom, cellValue1, cellValue2, x_position, title=" ", xlabel1=r' ', xlabel2=r' ', ylabel=r'z [m]'):
    """Plot line profile of Se and kr at a given x_position."""
    fig, ax = plt.subplots(figsize=(8, 6))    

    # Plot Se on the primary x-axis
    ax.plot(cellValue1[:, x_position], geom.Z[:, x_position], ".r")
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel)

    # Set the color of the primary x-axis to red (matching Se plot)
    ax.xaxis.label.set_color('red')
    ax.tick_params(axis='x', colors='red')
    ax.spines['bottom'].set_color('red')

    # Create a twin x-axis for kr
    ax2 = ax.twiny()
    ax2.plot(cellValue2[:, x_position], geom.Z[:, x_position], ".b")
    ax2.set_xlabel(xlabel2)

    # Set the color of the twin x-axis to blue (matching kr plot)
    ax2.xaxis.label.set_color('blue')
    ax2.tick_params(axis='x', colors='blue')
    ax2.spines['top'].set_color('blue')

    ax2.spines[['bottom', 'left', 'right']].set_visible(False)

    # Combine legends from both axes
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)

    ax.title.set_text(f"Line Profile at x = {x_position*geom.dx} m")
    return fig, ax

def parametrization_label(parametrization):
    return {
        "drying": "drying assumption",
        "permaFoam_omega10": r"impedance factor, $\Omega=10$",
        "permaFoam_omega50": r"impedance factor, $\Omega=50$",
        "cap": "capillary bundle model",
        "drying_omega10": r"drying assumption, with imp. factor $\Omega=10$",
    }.get(parametrization)

def plot_interflow(db, parametrization,title=None, ylim=None, relative = False, fig=None, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9*cm,9.5*cm))
    q_top = []
    q_bottom = []
    q_top_rel = []
    q_bottom_rel = []
    q_surf = []
    q_surf_rel = []

    dz = abs(db["Z"][0,1] - db["Z"][0,0])
    dx = abs(db["X"][1,0] - db["X"][0,0])
    dx_tilde = np.sqrt(dx**2 + dz**2)

    idx_top = db["Z"][:,-1] >= -1.5
    idx_bottom = db["Z"][:,-1] < -1.5

    times_days = np.array(db["time_array"]) / 24 / 60 / 60

    for i in range(5,len(db["time_array"])):
        q_surf.append(np.sum(abs(db["q_runoff"][i]))*dx_tilde)
        q_top.append(np.sum(abs(db["qx_array"][i][:,-1][idx_top])*dz))
        q_bottom.append(np.sum(abs(db["qx_array"][i][:,-1][idx_bottom])*dz))

        q_tot = q_top[-1]+q_bottom[-1]+q_surf[-1]
        q_top_rel.append(q_top[-1]/q_tot)
        q_bottom_rel.append(q_bottom[-1]/q_tot)
        q_surf_rel.append(q_surf[-1]/q_tot) 
    q_surf_arr = np.asarray(q_surf)
    has_surf = not np.all(q_surf_arr[~np.isnan(q_surf_arr)] == 0)
    if relative==False:
        if has_surf:
            ax.plot(times_days[5::], q_surf, "-", linewidth = 2, label=r"$q_{surface}$", color=green)
        ax.plot(times_days[5::], q_bottom, ":", linewidth = 2, label=r"$q_{bottom}$", color=persian_red)
        ax.plot(times_days[5::], q_top, "--", linewidth = 2, label=r"$q_{top}$", color=carolina_blue)
    else:
        if has_surf:
            ax.plot(times_days[5::], q_surf_rel, "-", linewidth = 2, label=r"$q_{surface}$", color=green)
        ax.plot(times_days[5::], q_bottom_rel, ":", linewidth = 2, label=r"$q_{bottom}$", color=persian_red)
        ax.plot(times_days[5::], q_top_rel, "--", linewidth = 2, label=r"$q_{top}$", color=carolina_blue)
    ax.spines[['top', 'right']].set_visible(False)

    # Combine legends from both axes
    ax.spines['left'].set_position(('outward', 10))
    #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_ylim(ylim)
    ax.set_xlabel(r"time [days]")
    ax.set_ylabel(r"$Q$ [$m^2s^{-1}$]")
    if standalone:
        fig.legend(loc="center", facecolor='none', edgecolor='none')
    else:
        ax.legend(loc="best", facecolor='none', edgecolor='none')
    ax.set_title(title or parametrization_label(parametrization))
    if standalone:
        fig.tight_layout()
    return fig, ax

def exit_fluxes(db, relative=False):
    """Instantaneous exit fluxes (per unit width) for q_top, q_bottom, q_surf.

    Same fluxes as used by `plot_interflow`. With `relative=True`, each series
    is expressed as its instantaneous share of q_top+q_bottom+q_surf.
    """
    dz = abs(db["Z"][0,1] - db["Z"][0,0])
    dx = abs(db["X"][1,0] - db["X"][0,0])
    dx_tilde = np.sqrt(dx**2 + dz**2)

    idx_top = db["Z"][:,-1] >= -1.5
    idx_bottom = db["Z"][:,-1] < -1.5

    time_s = np.array(db["time_array"])[5:]
    times_days = time_s / 24 / 60 / 60

    q_surf = np.array([np.sum(np.abs(db["q_runoff"][i]))*dx_tilde for i in range(5, len(db["time_array"]))])
    q_top = np.array([np.sum(np.abs(db["qx_array"][i][:,-1][idx_top])*dz) for i in range(5, len(db["time_array"]))])
    q_bottom = np.array([np.sum(np.abs(db["qx_array"][i][:,-1][idx_bottom])*dz) for i in range(5, len(db["time_array"]))])

    if relative:
        q_tot = np.where(q_top + q_bottom + q_surf == 0, np.nan, q_top + q_bottom + q_surf)
        q_surf, q_top, q_bottom = q_surf/q_tot, q_top/q_tot, q_bottom/q_tot

    return times_days, q_top, q_bottom, q_surf

def cumulative_exit_volumes(db, relative=False):
    """Cumulative exit volumes (per unit width) for q_top, q_bottom, q_surf.

    Integrates the raw fluxes from `exit_fluxes` over time (trapezoidal, on the
    true simulation times, since the solver uses adaptive stepping). With
    `relative=True`, each series is expressed as its share of the running
    cumulative total (Q_top+Q_bottom+Q_surf) — a much less noise-sensitive
    partitioning metric than the instantaneous flux ratio, since it isn't
    dominated by short high-flux events (e.g. a single ERA5 snowmelt spike).
    """
    time_s = np.array(db["time_array"])[5:]
    times_days, q_top, q_bottom, q_surf = exit_fluxes(db, relative=False)

    Q_surf = cumulative_trapezoid(q_surf, time_s, initial=0)
    Q_top = cumulative_trapezoid(q_top, time_s, initial=0)
    Q_bottom = cumulative_trapezoid(q_bottom, time_s, initial=0)

    if relative:
        Q_tot = np.where(Q_top + Q_bottom + Q_surf == 0, np.nan, Q_top + Q_bottom + Q_surf)
        Q_surf, Q_top, Q_bottom = Q_surf/Q_tot, Q_top/Q_tot, Q_bottom/Q_tot

    return times_days, Q_top, Q_bottom, Q_surf

def plot_cumulative_interflow(db, parametrization, title=None, ylim=None, relative=False, fig=None, ax=None, legend = False):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9*cm,9.5*cm))

    times_days, Q_top, Q_bottom, Q_surf = cumulative_exit_volumes(db, relative=relative)

    ax.plot(times_days, Q_surf, "-", linewidth = 2, label=r"$Q_{surface}$", color=green)
    ax.plot(times_days, Q_bottom, ":", linewidth = 2, label=r"$Q_{bottom}$", color=persian_red)
    ax.plot(times_days, Q_top, "--", linewidth = 2, label=r"$Q_{top}$", color=carolina_blue)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.set_ylim(ylim)
    ax.set_xlabel(r"time [days]")
    ax.set_ylabel(r"cumulative $Q_i / Q_{tot}$ [-]" if relative else r"cumulative volume [m$^2$]")
    if legend == True:
        if standalone:
            fig.legend(loc="center", facecolor='none', edgecolor='none')
        else:
            ax.legend(loc="best", facecolor='none', edgecolor='none')
    ax.set_title(title or parametrization_label(parametrization))
    if standalone:
        fig.tight_layout()
    return fig, ax

def _plot_flux_comparison(dbs, flux_fn, parametrizations, quantity, relative, ylabel, fig, ax,legend = False):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12*cm,9.5*cm))

    if parametrizations is None:
        parametrizations = list(dbs.keys())

    quantity_index = {"top": 1, "bottom": 2, "surf": 3}[quantity]
    for parametrization in parametrizations:
        times_days, *series = flux_fn(dbs[parametrization], relative=relative)
        style = parametrization_style(parametrization)
        ax.plot(times_days, series[quantity_index-1], linewidth=2,
               label=parametrization_label(parametrization) or parametrization,
               color=style["color"], linestyle=style["linestyle"])

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.set_xlabel(r"time [days]")
    ax.set_ylabel(ylabel)
    if legend == True:
        ax.legend(loc="best", facecolor='none', edgecolor='none')
    if standalone:
        fig.tight_layout()
    return fig, ax

def plot_cumulative_interflow_comparison(dbs, parametrizations=None, quantity="top", relative=True,
                                         title=None, ylim=None, fig=None, ax=None, legend = False):
    """Overlay cumulative exit-volume curves for one flow path across several parametrizations.

    `dbs` is a dict {parametrization: db}. `quantity` selects which path to compare
    ("top"=interflow, "bottom"=GW discharge, "surf"=surface runoff).
    """
    ylabel_quantity = {"top": r"$Q_{top}$", "bottom": r"$Q_{bottom}$", "surf": r"$Q_{surface}$"}[quantity]
    ylabel = f"cumulative {ylabel_quantity}" + (r"$/Q_{tot}$ [-]" if relative else " [m$^2$]")
    fig, ax = _plot_flux_comparison(dbs, cumulative_exit_volumes, parametrizations, quantity, relative, ylabel, fig, ax,legend)
    ax.set_ylim(ylim)
    ax.set_title(title or "")
    return fig, ax

def plot_cumulative_partitioning_grid(dbs, parametrizations=None, relative=False, figsize=None, legend=True):
    """Cumulative exit-volume comparison across parametrizations, one panel
    per flow path, sidmocean.cm.icee by side, with a shared legend centered below all
    panels.

    Includes a surface-runoff panel only if any db in `dbs` has nonzero
    surface runoff; otherwise just interflow and GW discharge.
    """
    if parametrizations is None:
        parametrizations = list(dbs.keys())

    def _has_surf(db):
        q_surf = np.asarray(exit_fluxes(db)[3])
        q_surf = q_surf[~np.isnan(q_surf)]
        return q_surf.size > 0 and np.any(q_surf != 0)

    has_surf = any(_has_surf(db) for db in dbs.values())
    quantities = ["surf", "top", "bottom"] if has_surf else ["top", "bottom"]
    titles = {"surf": "$q_{surface}$",
              "top": "$q_{top}$",
              "bottom": "$q_{bottom}$"}

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm if has_surf else PAGE_WIDTH_CM*2/3*cm, 6*cm)
    fig, axes = plt.subplots(1, len(quantities), figsize=figsize)

    for ax, quantity in zip(axes, quantities):
        plot_cumulative_interflow_comparison(dbs, parametrizations, quantity=quantity, relative=relative,
                                             title=titles[quantity], fig=fig, ax=ax, legend=False)

    fig.tight_layout()
    if legend:
        # fig.tight_layout() already reserved just enough room below the axes
        # for their own x-labels/ticks -- the legend needs to stack on top of
        # that, not replace it.
        labels_margin = fig.subplotpars.bottom

        handles, labels = axes[0].get_legend_handles_labels()
        ncol = min(len(labels), 3)
        leg = fig.legend(handles, labels, loc="lower center", ncol=ncol,
                         facecolor='none', edgecolor='none', bbox_to_anchor=(0.5, 0))
        # Anchored to the figure bottom, so its rendered height (measured after a
        # draw) tells us exactly how much extra room to reserve -- no guessed margin.
        fig.canvas.draw()
        legend_height = leg.get_window_extent(fig.canvas.get_renderer()) \
                           .transformed(fig.transFigure.inverted()).height
        fig.subplots_adjust(bottom=labels_margin + legend_height + 0.02)

    return fig, tuple(axes)

def plot_interflow_comparison(dbs, parametrizations=None, quantity="top", relative=False,
                              title=None, ylim=None, fig=None, ax=None):
    """Overlay instantaneous exit-flux curves for one flow path across several parametrizations.

    `dbs` is a dict {parametrization: db}. `quantity` selects which path to compare
    ("top"=interflow, "bottom"=GW discharge, "surf"=surface runoff). Unlike
    `plot_cumulative_interflow_comparison`, this shows the raw (non-integrated) flux.
    """
    ylabel_quantity = {"top": r"$q_{top}$", "bottom": r"$q_{bottom}$", "surf": r"$q_{surface}$"}[quantity]
    ylabel = f"{ylabel_quantity}" + (r"$/q_{tot}$ [-]" if relative else " [m s$^{-1}$]")
    fig, ax = _plot_flux_comparison(dbs, exit_fluxes, parametrizations, quantity, relative, ylabel, fig, ax)
    ax.set_ylim(ylim)
    ax.set_title(title or "")
    return fig, ax

def plot_streamContours(db, index):
    X = db["X_plot"]
    Z = db["Z_plot"]

    qx = db["qx_array"][index]
    qz = db["qz_array"][index]
    qz = np.append((qz,np.zeros((qz.shape[0],1))), axis=1)
    qx = np.append((np.zeros((1,qx.shape[1]),qx)), axis=0)
    # qz -= 0.1*qx

    chix_integrate_cum = -np.cumsum(qz, axis=1)*abs(Z[1,0]-Z[0,0])
    chiz_integrate_cum = np.cumsum(qx, axis=0)*abs(X[0,1]-X[0,0])
    chi_av = (chix_integrate_cum + chiz_integrate_cum) / 2
    fig, ax = plt.subplots(figsize=(18*cm,10*cm))
    ax.contour(X,Z,chi_av, levels=20, linewidths=0.5)
    return fig,ax

def stream_func(X,Z,qx,qz):
    dz = abs(Z[0,0]-Z[1,0])
    dx_tilde = abs(X[0,0]-X[0,1])
    dx = np.sqrt(dz**2+dx_tilde**2)
    psi_hor = np.cumsum(qz,axis=1)*dx
    psi_ver = np.cumsum(qx,axis=0)*dz
    return psi_hor,psi_ver

def plot_contourFlow(db):
    X = db["X"]
    Z = db["Z"]
    psi_hor, psi_ver = stream_func(X,Z, db["qx_array"], db["qz_array"])
    fig,ax = plt.subplots(figsize = (18*cm,10*cm))
    ax.contourf(X,Z,psi_ver)
    return fig,ax

def plot_streamlines(db, X, Z, qx, qz, slope, field=None, cmap=icy_soil,
                     title=" ", xlabel="x [m]", ylabel="z [m]",
                     density=1.0, color="k", linewidth=1.0, arrowsize=0,cmap_stream="magma",
                     fig=None, ax=None, cax=None):
    """
    Plot streamlines for unsaturated flow on a skewed (parallelogram) grid.

    Parameters
    ----------
    X, Z : 2D arrays
        Physical coordinates of the grid (same shape as qx, qz).
    qx, qz : 2D arrays
        Velocity components at cell centers.
    field : 2D array or None
        Optional scalar field to plot as background (e.g., theta, head).
    cmap : str
        Colormap for field.
    density : float
        Streamline density (higher = more lines).
    color : str
        Streamline color (ignored if field == None).
    linewidth : float
        Width of streamlines.
    arrowsize : float
        Arrow size on streamlines.
    """

    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    qx = np.asarray(qx, dtype=np.float64)
    qz = np.asarray(qz, dtype=np.float64)

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(25*cm, 8*cm))
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 0.02], wspace=0.35)
        ax = fig.add_subplot(gs[0, 0])
        cax0 = fig.add_subplot(gs[0, 1])
    else:
        cax0 = cax

    # ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))

    # Optional scalar field as background
    if field is not None:
        pcm = ax.pcolormesh(X, Z, field,vmin=0,vmax=0.46,shading="gouraud", cmap=cmap)
        if cax0 is not None:
            fig.colorbar(pcm, cax=cax0,label=r"$\theta_i$ [-]")
        else:
            fig.colorbar(pcm, ax=ax,label=r"$\theta_i$ [-]")

    # For skewed / deformed grids: interpolate onto a rectilinear evaluation grid
    Lx = X.max() - X.min()
    Lz = Z.max() - Z.min()

    # Precompute gridstitle_fontsize
    xg = np.linspace(X.min(), X.max(), 200)
    zg = np.linspace(Z.min(), Z.max(), 200)
    Xg, Zg = np.meshgrid(xg, zg)

    start_points = np.c_[
        np.linspace(0.5, 9.5, 20),
        -0.1*np.linspace(0.5, 9.5, 20) - 0.05
    ]

    qx_interp = griddata((X.flatten(), Z.flatten()), qx.flatten(), (Xg, Zg), method="linear")
    qz_interp = griddata((X.flatten(), Z.flatten()), qz.flatten()+slope*qx.flatten(), (Xg, Zg), method="linear")

    # linewidth scaling (still log10-based if you prefer)
    # lw = np.log10(speed) - np.nanmin(np.log10(speed))
    # lw = lw / np.nanmax(lw) * linewidth
    # lw = np.nan_to_num(lw, nan=1e-12)

    # -----------------------------------------
    # Make linewidth depend on speed
    # -----------------------------------------
    # lw = (norm/np.max(norm)) * linewidth
    #lw = np.nan_to_num(lw, nan=0.001)
    stream = ax.streamplot(
        xg, zg, qx_interp, qz_interp,
        density=5,
        color="k",
        linewidth=linewidth,
        arrowsize=arrowsize,
        start_points=start_points,
        integration_direction="forward"
    )
    #fig.colorbar(stream.lines,cax=cax1,label=r"$||q||_2$ [m/s]")
    # Draw boundary
    ax.plot([X[0,0],X[0,-1]],[Z[0,0],Z[0,-1]],"-k")
    ax.plot([X[-1,0],X[0,0]],[Z[-1,0],Z[0,0]],"-k")
    ax.plot([X[-1,0],X[-1,-1]],[Z[-1,0],Z[-1,-1]],"-k")
    ax.plot([X[0,-1],X[-1,-1]],[Z[0,-1],Z[-1,-1]],"-k")

    ax.set_title(title)
    if standalone:
        fig.tight_layout()

    return fig, ax

def _draw_break_marks(ax_top, ax_bottom, size=0.015):
    """Draw the diagonal '//' marks conventionally used to indicate a broken axis,
    at the bottom of `ax_top` and the top of `ax_bottom`."""
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1)
    ax_top.plot((-size, +size), (-size, +size), **kwargs)
    kwargs["transform"] = ax_bottom.transAxes
    ax_bottom.plot((-size, +size), (1 - size, 1 + size), **kwargs)

def plot_interflow_streamlines(db, parametrization, index, slope=-0.1, title=None,
                               ylim=None, relative=False, field_key="theta_i_total",
                               figsize=None, width_ratios=(2, 1), n_per_page=6,
                               title_fontsize=None, annotation_fontsize=None,
                               break_yaxis=None, break_height_ratios=(1, 1)):
    """
    Combined figure: streamlines (left, background field + colorbar) next to
    the interflow partitioning plot (right), with a bold top-left title.

    Defaults to a figure sized so that `n_per_page` of these stack to fillInterflow
    one manuscript text page (e.g. n_per_page=5 -> each figure is 16.46 x 4.48 cm).

    `break_yaxis`: optional ((low_min, low_max), (high_min, high_max)) pair of
    y-limits. When given, the flux (interflow) panel is split into two stacked
    axes sharing the x-axis, with a broken-axis gap between them -- useful when
    an isolated spike (e.g. q_surf during snowmelt) would otherwise dwarf the
    baseline q_top/q_bottom fluxes on a linear axis. `break_height_ratios`
    controls the relative height of (top, bottom) panels.
    """
    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm, PAGE_HEIGHT_CM/n_per_page*cm)
    if title_fontsize is None:
        title_fontsize = 10  # larger than the body text size (rcParams['font.size'] = 10)
    if annotation_fontsize is None:
        annotation_fontsize = 8  # matches body text size

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=list(width_ratios), wspace=0.7)
    gs_stream = gs[0, 0].subgridspec(nrows=1, ncols=2, width_ratios=[1, 0.03], wspace=0.05)
    ax_stream = fig.add_subplot(gs_stream[0, 0])
    cax = fig.add_subplot(gs_stream[0, 1])

    time_days = db["time_array"][index] / 24 / 60 / 60

    plot_streamlines(db, db["X"], db["Z"], db["qx_centered_array"][index], db["qz_centered_array"][index],
                     slope=slope, field=db[field_key][index], linewidth=1,
                     title="", fig=fig, ax=ax_stream, cax=cax)
    ax_stream.text(0.98, 0.97, f"t = {time_days:.0f} days", transform=ax_stream.transAxes,
                  ha="right", va="top", fontsize=annotation_fontsize,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    if break_yaxis is None:
        ax_interflow = fig.add_subplot(gs[0, 1])
        plot_interflow(db, parametrization, ylim=ylim, relative=relative, fig=fig, ax=ax_interflow)
        ax_interflow.set_title("")
        ax_interflow.axvline(time_days, color="black", linestyle="--")
    else:
        ylim_low, ylim_high = break_yaxis
        gs_interflow = gs[0, 1].subgridspec(nrows=2, ncols=1,
                                            height_ratios=list(break_height_ratios), hspace=0.08)
        ax_top = fig.add_subplot(gs_interflow[0, 0])
        ax_bottom = fig.add_subplot(gs_interflow[1, 0], sharex=ax_top)

        plot_interflow(db, parametrization, ylim=ylim_high, relative=relative, fig=fig, ax=ax_top)
        plot_interflow(db, parametrization, ylim=ylim_low, relative=relative, fig=fig, ax=ax_bottom)

        ylabel_text = ax_bottom.get_ylabel()
        ax_top.set_ylabel("")
        ax_bottom.set_ylabel("")

        ax_top.set_title("")
        ax_top.set_xlabel("")
        ax_top.spines['bottom'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        ax_top.axvline(time_days, color="black", linestyle="--")
        # drop the tick nearest the break on each side so they don't crowd/overlap it;
        # tune the *existing* auto locator's prune param rather than replacing it
        # outright, since a fresh MaxNLocator() defaults to nbins=10 (vs. the
        # default AutoLocator's nbins="auto"), which produced far too many ticks.
        ax_top.yaxis.get_major_locator().set_params(prune="lower")
        ax_bottom.yaxis.get_major_locator().set_params(prune="upper")

        # legend outside the axes, above the (topmost) panel, instead of overlapping the data
        handles, labels = ax_top.get_legend_handles_labels()
        legend = ax_top.get_legend()
        if legend is not None:
            legend.remove()
        ax_top.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                     ncol=len(labels), frameon=False, facecolor='none', edgecolor='none',
                     fontsize=annotation_fontsize)

        ax_bottom.set_title("")
        ax_bottom.spines['top'].set_visible(False)
        legend = ax_bottom.get_legend()
        if legend is not None:
            legend.remove()
        ax_bottom.axvline(time_days, color="black", linestyle="--")

        _draw_break_marks(ax_top, ax_bottom)
        ax_interflow = (ax_top, ax_bottom)

    fig.suptitle(title or parametrization_label(parametrization),
                 x=0.02, ha="left", fontsize=title_fontsize, fontweight="bold")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.24, top=0.80)

    if break_yaxis is not None:
        # place a single shared y-label to the left of both panels' tick labels,
        # centered across their combined vertical span.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = []
        for ax in (ax_top, ax_bottom):
            for tick_label in ax.get_yticklabels():
                bboxes.append(tick_label.get_window_extent(renderer=renderer))
            offset_text = ax.yaxis.get_offset_text()
            if offset_text.get_text():
                bboxes.append(offset_text.get_window_extent(renderer=renderer))
        left_px = min(bbox.x0 for bbox in bboxes)
        label_x = fig.transFigure.inverted().transform((left_px, 0))[0] - 0.02

        pos_top, pos_bottom = ax_top.get_position(), ax_bottom.get_position()
        y_center_fig = (pos_top.y1 + pos_bottom.y0) / 2
        fig.text(label_x, y_center_fig, ylabel_text, rotation=90, ha="center", va="center")

    return fig, (ax_stream, ax_interflow, cax)

def plot_Ku(db, parametrization, indices, interp_folder="../input/interpolation/",
           labels=None, slope=-0.1, vmin=1e-18, vmax=1e-4, cmap="magma",
           suptitle=None, figsize=None):
    """
    Unsaturated conductivity K_u = Ksat * k_rel(h_c, T) at given time indices,
    using the parametrization's own cached k_rel interpolation table (the
    same closure the simulation itself used -- drying curve, impedance-Omega,
    or capillary bundle) rather than an approximation. Diagnoses capillary-
    barrier vs. ice-lens contributions to interflow: the loam layer (soil
    type 2) is outlined in dashed cyan.

    The `soil` array is built exactly as in
    notebooks/ERA5_simulations/1Simulate_*.ipynb.
    """
    import pickle
    with open(f"{interp_folder}interpolation_{parametrization}.pkl", "rb") as f:
        soil_type = pickle.load(f)

    times_days = np.array(db["time_array"]) / 86400
    if labels is None:
        labels = [f"day {times_days[idx]:.0f}" for idx in indices]

    X = np.asarray(db["X"], dtype=np.float64)
    Z = np.asarray(db["Z"], dtype=np.float64)

    soil = np.ones_like(X)
    loam_indices = np.where((Z <= -0.25 + slope*X) & (Z >= -0.5 + slope*X))
    soil[loam_indices] = 2

    Ksat_field = np.where(soil == 1, soil_type[1]["Ksat"], soil_type[2]["Ksat"])

    n = len(indices)
    if figsize is None:
        figsize = (13.5*n*cm, 11*cm)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, idx, label in zip(axes, indices, labels):
        hc = np.asarray(db["h_c_total"][idx], dtype=np.float64)
        T = np.asarray(db["T_total"][idx], dtype=np.float64)
        krel = np.where(soil == 1,
                        np.asarray(soil_type[1]["k_rel_interp"](hc.flatten(), T.flatten())).reshape(hc.shape),
                        np.asarray(soil_type[2]["k_rel_interp"](hc.flatten(), T.flatten())).reshape(hc.shape))
        Ku = np.clip(Ksat_field*krel, vmin, vmax)

        pcm = ax.pcolormesh(X, Z, Ku, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                            cmap=cmap, shading="gouraud")
        ax.plot(X[0, :], -0.25 + slope*X[0, :], "--", color="cyan", linewidth=1)
        ax.plot(X[0, :], -0.5 + slope*X[0, :], "--", color="cyan", linewidth=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.set_title(label)
        ax.set_xlabel("x [m]")

    axes[0].set_ylabel("z [m]")
    fig.colorbar(pcm, ax=axes, label=r"$K_u = K_{sat} \cdot k_{rel}(h_c, T)$ [m/s]", shrink=0.8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    return fig, axes

def plot_krel_surface(soil_type, param, hc, T, figsize=None, n_per_row=2, title_fontsize=10):
    """3D surface of log10(k_rel) vs h_c and T for soil type 1 (sand), used to
    compare the closure shape across parametrizations.

    Defaults to a figure sized so that `n_per_row` of these fit side by side
    on one manuscript text page width (e.g. n_per_row=2 -> each figure is
    8.23 x 8.23 cm).
    """
    plt.rcParams['font.size'] = 8

    if figsize is None:
        width = PAGE_WIDTH_CM/n_per_row*cm
        figsize = (width, width)

    X, Y = jnp.meshgrid(hc, T)
    Z = soil_type[1]["k_rel_interp"](X.flatten(), Y.flatten()).reshape(X.shape)
    Z = jnp.log10(Z)
    Z = Z.at[jnp.where(Z < -15)].set(jnp.nan)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    ax.view_init(elev=10, azim=-20, roll=0)
    ax.plot_surface(jnp.log10(X), Y, Z, cmap=cmo.deep)
    ax.set_xticks([-2, -1, 0])
    ax.set_xticklabels([f'$10^{{{i}}}$' for i in range(-2, 1)])
    ax.set_zticks([-15, -10, -5, 0])
    ax.set_zticklabels([f'$10^{{{i}}}$' for i in range(-15, 5, 5)])
    ax.set_xlabel(r'$h_c$ [m]')
    ax.set_ylabel(r'$T$ [°C]')
    ax.set_zlabel(r'$k_{rel}$ [-]')

    label = parametrization_label(param)
    if label:
        ax.text2D(0.02, 0.95, label, transform=ax.transAxes, fontsize=title_fontsize,
                  fontweight="bold", ha="left", va="top")

    fig.tight_layout()
    return fig, ax

def plot_forcing_data(time_array_spin, time_array_sim,
                      skt_array_spin, skt_array_sim,
                      tl1_array_spin, tl1_array_sim,
                      recharge_spin, recharge_sim):
    """Plot ERA5 forcing data (recharge + skin/soil temperature) for the spinup and simulation periods."""
    recharge_sim_plot = np.convolve(recharge_sim, np.ones(24)/24, mode="same")
    recharge_spin_plot = np.convolve(recharge_spin, np.ones(24)/24, mode="same")
    skt_array_sim_plot = np.convolve(skt_array_sim,np.ones(24)/24, mode="same")
    skt_array_spin_plot = np.convolve(skt_array_spin,np.ones(24)/24, mode="same")
    tl1_array_sim_plot = np.convolve(tl1_array_sim,np.ones(24)/24, mode="same")
    tl1_array_spin_plot = np.convolve(tl1_array_spin,np.ones(24)/24, mode="same")


    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_CM*cm,7*cm))
    ax_cp = ax.twinx()
    ax.bar(time_array_spin/24/60/60, recharge_spin_plot, color="#0072B2")
    ax.hlines(0, time_array_spin[0]/24/60/60, time_array_spin[-1]/24/60/60, color="#0072B2", linestyle="--")
    ax.hlines(0, time_array_sim[0]/24/60/60, time_array_sim[-1]/24/60/60, color="#0072B2", linestyle="--")
    ax.bar(time_array_sim/24/60/60, recharge_sim_plot, color="#0072B2")
    ax.set_ylabel("$q_{inf}$ $[m/s]$", color="#0072B2")
    ax.tick_params(axis="y", colors="#0072B2")
    ax.yaxis.set_inverted(True)
    ax_cp.set_ylabel("$T$ $[°C]$", color="#D55E00")
    ax_cp.plot(time_array_spin/24/60/60, skt_array_spin_plot, label="skt", color="#D55E00")
    ax_cp.plot(time_array_spin/24/60/60, tl1_array_spin_plot, label="tl1", color="#E69F00")
    ax_cp.plot(time_array_sim/24/60/60, skt_array_sim_plot, label="skt", color="#D55E00")
    ax_cp.plot(time_array_sim/24/60/60, tl1_array_sim_plot, label="tl1", color="#E69F00")

    ax_cp.set_ylim([-32, max(skt_array_spin)])
    ax_cp.tick_params(axis="y", labelcolor="#D55E00")
    ax_cp.hlines(0, time_array_spin[0]/24/60/60, time_array_spin[-1]/24/60/60, color="#D55E00", linestyle="--")
    ax_cp.hlines(0, time_array_sim[0]/24/60/60, time_array_sim[-1]/24/60/60, color="#D55E00", linestyle="--")
    vline_time = time_array_spin[-1]/24/60/60
    ax.axvline(x=vline_time, color="k", linestyle="--", zorder=100)

    ax_cp.text(0.075, 0.1, "spinup", ha="left", va="top", transform=ax_cp.transAxes,
              color="k", fontweight='bold', fontsize=10, zorder=200)
    ax_cp.text(0.52, 0.1, "simulation", ha="left", va="top", transform=ax_cp.transAxes,
              color="k", fontweight='bold', fontsize=10, zorder=200)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax_cp.spines["left"].set_visible(False)
    ax_cp.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax_cp.spines['right'].set_position(('outward', 10))

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    # Rotated labels have a bigger visual footprint than horizontal text at the
    # same point size, so size them down a notch to read at par with body text.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()

    return fig, (ax, ax_cp)

def create_animation_streamlines(
    X, Z,
    cell1_t, qx_t, qz_t,
    cell2_t, cell3_t,
    recharge, q_inf, skt_array,
    time_array,
    dates=None,
    slope=0,
    time_mode=None,   # NEW: "date" | "days" | None (auto)
    density=1,
    linewidth=2,
    cmaps=("Blues", "bone_r", "coolwarm"),
    clabels=(r" ", r" ", r" "),
    cmap_stream="magma",
    arrowsize=0, 
    dt = None
):
    """Animate scalar fields and streamlines over time with flexible time axis."""

    # Time handling (NEW)
    time_days = (time_array - time_array[0]) / 86400

    # auto-detect mode if not provided
    if time_mode is None:
        time_mode = "date" if dates is not None else "days"

    if time_mode == "date" and dates is not None:
        x_time = dates
        x_full = [dates[0] + timedelta(seconds=float(t)) for t in time_array]
        x_label = "$t$ [date]"
        use_dates = True
    else:
        x_time = time_days
        x_full = time_days
        x_label = "$t$ [days]"
        use_dates = False

    # Helpers
    def compute_global_minmax(arrays):
        return min(np.nanmin(a) for a in arrays), max(np.nanmax(a) for a in arrays)

    def interpolate_flow(qx, qz):
        qx_i = griddata((Xf, Zf), qx.flatten(), (Xg, Zg), method="linear")
        qz_i = griddata((Xf, Zf), (qz + slope * qx).flatten(), (Xg, Zg), method="linear")
        speed = np.sqrt(qx_i**2 + qz_i**2)
        return qx_i, qz_i, speed

    # def compute_linewidth(speed):
    #     lw = np.log10(speed) - np.nanmin(np.log10(speed))
    #     lw = lw / np.nanmax(lw) * linewidth
    #     return np.nan_to_num(lw, nan=0.000000001)

    def draw_domain_outline(ax):
        ax.plot([X[0,0], X[0,-1]], [Z[0,0], Z[0,-1]], "-k")
        ax.plot([X[-1,0], X[0,0]], [Z[-1,0], Z[0,0]], "-k")
        ax.plot([X[-1,0], X[-1,-1]], [Z[-1,0], Z[-1,-1]], "-k")
        ax.plot([X[0,-1], X[-1,-1]], [Z[0,-1], Z[-1,-1]], "-k")

    def compute_linewidth(speed):
        eps = 1e-20

        speed_safe = np.maximum(speed, eps)

        lw = np.log10(speed_safe)
        lw -= np.nanmin(lw)

        maxlw = np.nanmax(lw)

        if maxlw > 0:
            lw = lw / maxlw * linewidth
        else:
            lw = np.ones_like(lw) * 0.2

        return lw
    
    # Figure layout
    fig = plt.figure(figsize=(25*cm, 23*cm))
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 0.03, 0.01],wspace=0.3, hspace=0.25)

    ax_bc = fig.add_subplot(gs[0, 0])
    ax_cp = ax_bc.twinx()

    ax = [fig.add_subplot(gs[i, 0]) for i in range(1, 4)]
    cax00 = fig.add_subplot(gs[1, 1])
    cax01 = fig.add_subplot(gs[1, 2])
    cax1  = fig.add_subplot(gs[2, 1])
    cax2  = fig.add_subplot(gs[3, 1])

    # Top plot (UPDATED)
    dt = np.diff(x_full).mean() if use_dates else np.diff(x_time).mean()
    ax_bc.bar(x_time, recharge,width=dt, color=carolina_blue,label="runoff")
    ax_bc.bar(x_time, q_inf, width=dt, color=cobalt,label="infiltration")
    ax_bc.legend(loc="upper left")
    ax_bc.set_ylabel("flux [m/s]", color=cobalt)
    ax_bc.invert_yaxis()

    ax_cp.plot(x_time, skt_array, color=persian_red)
    ax_cp.set_ylabel("T [°C]", color=persian_red)
    if time_mode == "date":
        ax_cp.axhline(0, linestyle="--", color=persian_red)

    time_indicator = ax_cp.axvline(x_time[0], color="k")

    ax_cp.set_xlabel(x_label)

    if use_dates:
        ax_cp.xaxis.set_major_locator(mdates.MonthLocator())
        ax_cp.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax_cp.get_xticklabels(), rotation=45, ha='right')
        ax_bc.set_xlabel("Time [date]")
    else:
        ax_bc.set_xlabel("Time [d]")

    # Axes formatting
    for a in ax:
        a.set_aspect("equal")
        a.set_xlim([0, 10])
        a.set_ylim([-2, 0])
        a.set_xlabel("x [m]")
        a.set_ylabel("z [m]")
        a.spines[['top', 'right']].set_visible(False)

    for a in ax[:-1]:
        a.xaxis.set_visible(False)
        a.spines["bottom"].set_visible(False)

    # Precompute grids
    Xf, Zf = X.flatten(), Z.flatten()
    xg = np.linspace(X.min(), X.max(), 200)
    zg = np.linspace(Z.min(), Z.max(), 200)
    Xg, Zg = np.meshgrid(xg, zg)

    # Min/max
    vmin1, vmax1 = compute_global_minmax(cell1_t)
    vmin3, vmax3 = compute_global_minmax(cell3_t)

    # Initial plots
    pcm0 = ax[0].pcolormesh(X, Z, cell1_t[0], cmap=cmaps[0],
                             vmin=vmin1, vmax=vmax1, shading="gouraud")
    fig.colorbar(pcm0, cax=cax00, label=clabels[0])

    qx_i, qz_i, speed = interpolate_flow(qx_t[-1], qz_t[-1])
    lw = compute_linewidth(speed)

    norm = matplotlib.colors.Normalize(
        vmin=np.nanmin(speed),
        vmax=np.nanmax(speed)
    )

    start_points = np.c_[
        np.linspace(0.5, 9.5, 20),
        -0.1*np.linspace(0.5, 9.5, 20) - 0.05
    ]

    stream = ax[0].streamplot(
        xg, zg, qx_i, qz_i,
        density=density,
        color=speed,
        cmap=cmap_stream,
        linewidth=lw,
        start_points=start_points,
        norm=norm,
        arrowsize=arrowsize
    )

    fig.colorbar(stream.lines, cax=cax01, label=r"$||q||_2$")

    pcm1 = ax[1].pcolormesh(X, Z, cell2_t[0], cmap=cmaps[1],
                             vmin=vmin1, vmax=vmax1, shading="gouraud")
    fig.colorbar(pcm1, cax=cax1, label=clabels[1])

    pcm2 = ax[2].pcolormesh(X, Z, cell3_t[0], cmap=cmaps[2],
                             vmin=vmin3, vmax=vmax3, shading="gouraud")
    fig.colorbar(pcm2, cax=cax2, label=clabels[2])

    for a in ax:
        draw_domain_outline(a)

    # Animation
    def update(i):
        for coll in ax[0].collections[:]: 
            coll.remove() # removes previous streamlines 
        for pat in ax[0].patches[:]: 
            pat.remove()
        ax[0].pcolormesh(X, Z, cell1_t[i], cmap=cmaps[0],
                        vmin=vmin1, vmax=vmax1, shading="gouraud")

        qx_i, qz_i, speed = interpolate_flow(qx_t[i], qz_t[i])
        lw = compute_linewidth(speed)

        ax[0].streamplot(
            xg, zg, qx_i, qz_i,
            density=density,
            color=speed,
            cmap=cmap_stream,
            linewidth=lw,
            start_points=start_points,
            norm=norm,
            arrowsize=arrowsize
        )

        for coll in ax[1].collections[:]:
            coll.remove()
        for coll in ax[2].collections[:]:
            coll.remove()

        ax[1].pcolormesh(X, Z, cell2_t[i], cmap=cmaps[1],
                        vmin=vmin1, vmax=vmax1, shading="gouraud")

        ax[2].pcolormesh(X, Z, cell3_t[i], cmap=cmaps[2],
                        vmin=vmin3, vmax=vmax3, shading="gouraud")

        time_indicator.set_xdata([x_full[i]])
    if dt is None:
        nt = 1
    else:   
        nt = int(len(x_full)/(int((x_full[-1]- x_full[0])/dt)))
        print(list(range(1,len(x_full),nt)))
    ani = animation.FuncAnimation(
        fig, update,
        frames=range(1,len(x_full),1),
        interval=200
    )

    return ani, fig, ax
