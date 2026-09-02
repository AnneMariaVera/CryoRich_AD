import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import matplotlib.animation as animation
from scipy.interpolate import griddata
from scipy.integrate import cumulative_trapezoid
import cmocean.cm as cmo
import matplotlib.dates as mdates
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

def parametrization_label(parametrization):
    return {
        "drying": "(A) drying assumption",
        "permaFoam_omega10": r"(B) impedance factor, $\Omega=10$",
        "permaFoam_omega50": r"(C) impedance factor, $\Omega=50$",
        "cap": "(D) capillary bundle model",
        "drying_omega10": r"(E) drying assumption, with imp. factor $\Omega=10$",
    }.get(parametrization)

def parametrization_label_leg(parametrization):
    return {
        "drying": "(B) drying assumption",
        "permaFoam_omega10": r"(C) impedance factor, $\Omega=10$",
        "permaFoam_omega50": r"(D) impedance factor, $\Omega=50$",
        "cap": "(A) capillary bundle model",
        "drying_omega10": r"(E) drying assumption, with imp. factor $\Omega=10$",
    }.get(parametrization)

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

def exit_fluxes(db, relative=False):
    """Instantaneous exit fluxes (per unit width) for q_top, q_bottom, q_surf.

    With `relative=True`, each series is expressed as its instantaneous share
    of q_top+q_bottom+q_surf.
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
    partitioning metric than the instantaneous flux ratio, since it isn'tparametrization_label(parametrization) or parametrization
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

def _plot_flux_comparison(dbs, flux_fn, parametrisations, quantity, relative, ylabel, fig, ax,legend = False):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12*cm,9.5*cm))

    if parametrisations is None:
        parametrisations = list(dbs.keys())

    quantity_index = {"top": 1, "bottom": 2, "surf": 3}[quantity]
    for parametrization in parametrisations:
        times_days, *series = flux_fn(dbs[parametrization], relative=relative)
        style = parametrization_style(parametrization)
        ax.plot(times_days, series[quantity_index-1], linewidth=2,
               label=parametrization_label_leg(parametrization) or parametrization,
               color=style["color"], linestyle=style["linestyle"])
        print("values at the end of the simulation for",parametrization)
        print(quantity,jnp.nanmax(series[quantity_index-1]))

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlabel(r"time [days]")
    ax.set_ylabel(ylabel)
    if legend == True:
        ax.legend(loc="best", facecolor='none', edgecolor='none')
    if standalone:
        fig.tight_layout()
    return fig, ax

def plot_exit_flux(dbs, parametrisations=None, quantity="top", relative=True,
                                         title=None, ylim=None, fig=None, ax=None, legend = False):
    """Overlay cumulative exit-volume curves for one flow path across several parametrisations.

    `dbs` is a dict {parametrization: db}. `quantity` selects which path to compare
    ("top"=interflow, "bottom"=GW discharge, "surf"=surface runoff).
    """
    ylabel = f"V" + (r"$/Q_{tot}$ [-]" if relative else " [m$^2$]")
    fig, ax = _plot_flux_comparison(dbs, exit_fluxes, parametrisations, quantity, relative, ylabel, fig, ax,legend)
    ax.set_ylim(ylim)
    ax.set_title(title or "")
    return fig, ax

def plot_cumulative_interflow_comparison(dbs, parametrisations=None, quantity="top", relative=True,
                                         title=None, ylim=None, fig=None, ax=None, legend = False):
    """Overlay cumulative exit-volume curves for one flow path across several parametrisations.

    `dbs` is a dict {parametrization: db}. `quantity` selects which path to compare
    ("top"=interflow, "bottom"=GW discharge, "surf"=surface runoff).
    """
    ylabel = f"V" + (r"$/Q_{tot}$ [-]" if relative else " [m$^2$]")
    fig, ax = _plot_flux_comparison(dbs, cumulative_exit_volumes, parametrisations, quantity, relative, ylabel, fig, ax,legend)
    ax.set_ylim(ylim)
    ax.set_title(title or "")
    return fig, ax

def plot_cumulative_partitioning_grid(dbs, parametrisations=None, relative=False, figsize=None, legend=True):
    """Cumulative exit-volume comparison across parametrisations, one panel
    per flow path, sidmocean.cm.icee by side, with a shared legend centered below all
    panels.

    Includes a surface-runoff panel only if any db in `dbs` has nonzero
    surface runoff; otherwise just interflow and GW discharge.
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())

    def _has_surf(db):
        q_surf = np.asarray(exit_fluxes(db)[3])
        q_surf = q_surf[~np.isnan(q_surf)]
        return q_surf.size > 0 and np.any(q_surf != 0)

    has_surf = any(_has_surf(db) for db in dbs.values())
    quantities = ["surf", "top", "bottom"] if has_surf else ["top", "bottom"]
    titles = {"surf": "surface runoff",
              "top": "interflow",
              "bottom": "groundwater discharge"}

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm if has_surf else PAGE_WIDTH_CM*2/3*cm, 6*cm)
    fig, axes = plt.subplots(1, len(quantities), figsize=figsize)

    for ax, quantity in zip(axes, quantities):
        plot_cumulative_interflow_comparison(dbs, parametrisations, quantity=quantity, relative=relative,
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

def plot_cumulative_partitioning_grid_with_inset(dbs, parametrisations=None, zoom_end=135,
                                                  inset_quantities=("surf", "top"),
                                                  figsize=None, legend=True):
    """Like `plot_cumulative_partitioning_grid`, but panels in `inset_quantities`
    carry a zoomed inset over the pre-melt window (0 to `zoom_end` days).

    Restricting an inset to the pre-melt period sets its y-scale from the
    smallest, earliest-moving parametrisation rather than from whichever one
    later dominates the full-period curve (e.g. drying's late, steep rise in
    interflow) -- otherwise the pre-melt differences between parametrisations
    are squashed flat. Not every panel needs this: groundwater discharge (the
    default omission) already diverges visibly across the full time range, so
    an inset there only magnifies what's already visible rather than revealing
    anything new -- check before adding it back for a different data set.
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())

    titles = {"surf": "surface runoff", "top": "interflow", "bottom": "groundwater discharge"}

    series = {}
    for p in parametrisations:
        times, Qtop, Qbot, Qsurf = cumulative_exit_volumes(dbs[p])
        good = ~(np.isnan(Qtop) | np.isnan(Qbot) | np.isnan(Qsurf))
        series[p] = {"t": times[good], "surf": Qsurf[good], "top": Qtop[good], "bottom": Qbot[good]}

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm, 7*cm)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    ax_surf = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[0, 2])
    axes = {"surf": ax_surf, "top": ax_top, "bottom": ax_bot}

    for quantity, ax in axes.items():
        for p in parametrisations:
            s = series[p]
            style = parametrization_style(p)
            ax.plot(s["t"], s[quantity], linewidth=2, color=style["color"],
                   linestyle=style["linestyle"], label=parametrization_label_leg(p) or p)
        ax.set_xlabel("time [days]")
        ax.set_title(titles[quantity])
        ax.spines[['top', 'right']].set_visible(False)
    ax_surf.set_ylabel(r"V [m$^2$]")

    for quantity in inset_quantities:
        ax = axes[quantity]
        zoom_max = max(series[p][quantity][series[p]["t"] <= zoom_end].max() for p in parametrisations)
        if zoom_max <= 0:
            continue
        axins = inset_axes(ax, width="58%", height="52%",
                           bbox_to_anchor=(0.22, 0.34, 0.78, 0.54), bbox_transform=ax.transAxes,
                           loc="upper left", borderpad=0)
        for p in parametrisations:
            s = series[p]
            style = parametrization_style(p)
            axins.plot(s["t"], s[quantity], linewidth=1.4, color=style["color"], linestyle=style["linestyle"])
        axins.set_xlim(0, zoom_end)
        axins.set_ylim(0, zoom_max * 1.15)
        axins.tick_params(labelsize=6.5)
        axins.set_facecolor("white")
        for spine in axins.spines.values():
            spine.set_edgecolor("0.4")
            spine.set_linewidth(0.8)
        mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5", linewidth=0.7)

    fig.tight_layout()
    if legend:
        labels_margin = fig.subplotpars.bottom
        handles, labels = ax_surf.get_legend_handles_labels()
        ncol = min(len(labels), 3)
        leg = fig.legend(handles, labels, loc="lower center", ncol=ncol,
                         facecolor='none', edgecolor='none', bbox_to_anchor=(0.5, 0))
        fig.canvas.draw()
        legend_height = leg.get_window_extent(fig.canvas.get_renderer()) \
                           .transformed(fig.transFigure.inverted()).height
        fig.subplots_adjust(bottom=labels_margin + legend_height + 0.02)

    return fig, (ax_surf, ax_top, ax_bot, axins)

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

    # For skewed / deformed     X, Y = jnp.meshgrid(hc, T)grids: interpolate onto a rectilinear evaluation grid
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

def plot_streamlines_grid(dbs, index, parametrisations=None, slope=-0.1, field_key="theta_i_total",
                          relative=False, figsize=None, title_fontsize=None, annotation_fontsize=None,
                          cum=True, cum_height_ratio=1.3, legend=True):
    """
    Combined figure: the streamline field for each parametrization stacked in
    rows (one row per parametrization). If `cum` is True (default), a
    cumulative partitioning comparison across parametrisations (one panel per
    flow path, as in `plot_cumulative_partitioning_grid`) is prepended on top,
    with a vline marking the time of `index`, plus a shared legend; if False,
    the figure is just the stacked streamline panels.
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())
    if title_fontsize is None:
        title_fontsize = 10
    if annotation_fontsize is None:
        annotation_fontsize = 8

    n_param = len(parametrisations)

    if cum:
        def _has_surf(db):
            q_surf = np.asarray(exit_fluxes(db)[3])
            q_surf = q_surf[~np.isnan(q_surf)]
            return q_surf.size > 0 and np.any(q_surf != 0)

        has_surf = any(_has_surf(db) for db in dbs.values())
        quantities = ["surf", "top", "bottom"] if has_surf else ["top", "bottom"]
        titles = {"surf": "surface runoff",
                  "top": "interflow",
                  "bottom": "groundwater discharge"}
        ncols = len(quantities)
        legend_height_ratio = 0.35
        n_header_rows = 2
        height_ratios = [cum_height_ratio, legend_height_ratio] + [1]*n_param
    else:
        ncols = 1
        n_header_rows = 0
        height_ratios = [1]*n_param

    if figsize is None:
        header_height = (cum_height_ratio + legend_height_ratio) if cum else 0
        figsize = (PAGE_WIDTH_CM*cm, PAGE_HEIGHT_CM/6*(header_height + n_param)*cm)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=n_header_rows + n_param, ncols=ncols,
                         height_ratios=height_ratios, hspace=0.7, wspace=0.6)

    # `index` is shared across parametrisations (same time step for every db),
    # so any db's time_array gives the time of the vline/annotation.
    time_days = np.asarray(dbs[parametrisations[0]]["time_array"])[index] / 86400

    cum_axes = []
    if cum:
        for col, quantity in enumerate(quantities):
            ax = fig.add_subplot(gs[0, col])
            plot_cumulative_interflow_comparison(dbs, parametrisations, quantity=quantity, relative=relative,
                                                 title=titles[quantity], fig=fig, ax=ax, legend=False)
            ax.axvline(time_days, color="black", linestyle="--")
            cum_axes.append(ax)

    fig.subplots_adjust(left=0.13, right=0.90, bottom=0.05, top=0.96)

    if cum and legend:
        # `fig.legend` (rather than an axes-scoped legend) spans the full figure
        # width, so a wide multi-column legend doesn't overflow the narrower
        # column span of the cumulative/streamline axes. Anchored just below the
        # cumulative row's rendered content (axes + tick labels + x-label),
        # growing down into the blank spacer row.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        cum_bottom = min(ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
                         for ax in cum_axes)

        handles, labels = cum_axes[0].get_legend_handles_labels()
        ncol = min(len(labels), 3)
        fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
                  facecolor='none', edgecolor='none', bbox_to_anchor=(0.5, cum_bottom - 0.01),
                  fontsize=annotation_fontsize)

    stream_axes = []
    ax_stream_ref = None
    last_row = n_header_rows + n_param - 1
    for row, parametrization in enumerate(parametrisations, start=n_header_rows):
        db = dbs[parametrization]
        gs_stream = gs[row, :].subgridspec(nrows=1, ncols=2, width_ratios=[1, 0.025], wspace=0.1)
        ax_stream = fig.add_subplot(gs_stream[0, 0], sharex=ax_stream_ref)
        cax = fig.add_subplot(gs_stream[0, 1])
        if ax_stream_ref is None:
            ax_stream_ref = ax_stream

        plot_streamlines(db, db["X"], db["Z"], db["qx_centered_array"][index], db["qz_centered_array"][index],
                         slope=slope, field=db[field_key][index], linewidth=1,
                         title="", fig=fig, ax=ax_stream, cax=cax)
        ax_stream.set_title(parametrization_label(parametrization) or parametrization,
                            loc="left", fontsize=title_fontsize, fontweight="bold")

        if row < last_row:
            ax_stream.set_xlabel("")
            ax_stream.tick_params(bottom=False, labelbottom=False)
            ax_stream.spines['bottom'].set_visible(False)

        stream_axes.append(ax_stream)

    stream_axes[0].text(0.98, 0.97, f"t = {time_days:.0f} days", transform=stream_axes[0].transAxes,
                        ha="right", va="top", fontsize=annotation_fontsize,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    return fig, (tuple(cum_axes), tuple(stream_axes))

def plot_T_grid(dbs, index, parametrisations=None, figsize=None, title_fontsize=None, annotation_fontsize=None, field_key="T_total", cmap=cmo.thermal):
    """
    plot of temperature distribution within the crosssection for all parametrisaitons.
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())
    if title_fontsize is None:
        title_fontsize = 10
    if annotation_fontsize is None:
        annotation_fontsize = 8

    n_param = len(parametrisations)
    vmin = np.min(dbs["drying"][field_key][index])
    vmax =  np.max(dbs["drying"][field_key][index])
    for parametrisation in parametrisations:
        T = np.array(dbs[parametrisation][field_key][index])
        if np.min(T) < vmin:
            vmin = np.min(T)
        if np.max(T) > vmax:
            vmax = np.max(T)

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm, PAGE_HEIGHT_CM/6*n_param*cm)

    time_days = np.asarray(dbs[parametrisations[0]]["time_array"])[index] / 86400

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=n_param, ncols=1, hspace=0.7)
    fig.subplots_adjust(left=0.13, right=0.90, bottom=0.05, top=0.96)

    T_axes = []
    ax_ref = None
    last_row = n_param - 1
    for row, parametrisation in enumerate(parametrisations):
        gs_T = gs[row, :].subgridspec(nrows=1, ncols=2, width_ratios=[1, 0.025], wspace=0.1)
        ax = fig.add_subplot(gs_T[0, 0], sharex=ax_ref)
        cax = fig.add_subplot(gs_T[0, 1])
        if ax_ref is None:
            ax_ref = ax

        field = np.array(dbs[parametrisation][field_key][index])
        ax.pcolormesh(dbs[parametrisation]["X"], dbs[parametrisation]["Z"], field,
                     vmin=vmin, vmax=vmax, shading="gouraud", cmap=cmap)
        # colorbar built from a plain ScalarMappable (not the gouraud QuadMesh
        # itself), which otherwise can leave a thin uncolored sliver at one end
        # when a row's own data doesn't span the full shared vmin/vmax range.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, cax=cax, label=r"$T$ [°C]")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))

        ax.set_title(parametrization_label(parametrisation) or parametrisation,
                    loc="left", fontsize=title_fontsize, fontweight="bold")

        if row < last_row:
            ax.set_xlabel("")
            ax.tick_params(bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)

        T_axes.append(ax)

    T_axes[0].text(0.98, 0.97, f"t = {time_days:.0f} days", transform=T_axes[0].transAxes,
                   ha="right", va="top", fontsize=annotation_fontsize,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    return fig, tuple(T_axes)


def plot_krel_surface(soil_type, param, hc, T, figsize=None, n_per_row=2, title_fontsize=10):
    """3D surface of log10(k_rel) vs h_c and T for soil type 1 (sand), used to
    compare the closure shape across parametrisations.

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
    ax.plot_surface(jnp.log10(X), Y, Z, cmap=cmo.amp)
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

def plot_theta_l_surface(soil_type, param, hc, T, figsize=None, n_per_row=2, title_fontsize=10):
    """3D surface of log10(k_rel) vs h_c and T for soil type 1 (sand), used to
    compare the closure shape across parametrisations.

    Defaults to a figure sized so that `n_per_row` of these fit side by side
    on one manuscript text page width (e.g. n_per_row=2 -> each figure is
    8.23 x 8.23 cm).
    """
    plt.rcParams['font.size'] = 8

    if figsize is None:
        width = PAGE_WIDTH_CM/n_per_row*cm
        figsize = (width, width)

    X, Y = jnp.meshgrid(hc, T)
    Z = soil_type[1]["theta_l_interp"](X.flatten(), Y.flatten()).reshape(X.shape)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    ax.view_init(elev=10, azim=-20, roll=0)
    ax.plot_surface(np.log10(X), Y, Z, cmap="Blues")
    
    ax.set_yticks([-10, -5, 0, 5,10])
    ax.set_xticks([-2, -1, 0])
    ax.set_xticklabels([f'$10^{{{i}}}$' for i in range(-2, 1)])

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
                      tl1_array_spin, tl1_array_sim,
                      recharge_spin, recharge_sim):
    """Plot ERA5 forcing data (recharge + skin/soil temperature) for the spinup and simulation periods."""
    recharge_sim_plot = np.convolve(recharge_sim, np.ones(24)/24, mode="same")
    recharge_spin_plot = np.convolve(recharge_spin, np.ones(24)/24, mode="same")
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
    ax_cp.plot(time_array_spin/24/60/60, tl1_array_spin_plot, label="tl1", color="#D55E00")
    ax_cp.plot(time_array_sim/24/60/60, tl1_array_sim_plot, label="tl1", color="#D55E00")

    ax_cp.set_ylim([min(tl1_array_sim_plot)-2, max(tl1_array_spin_plot)+2])
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
