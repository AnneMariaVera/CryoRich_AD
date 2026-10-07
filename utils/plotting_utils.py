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

parametrisation_STYLE = {
    "drying":             {"color": "#0072B2", "linestyle": "-"},        # blue
    "drying_omega10":     {"color": "#56B4E9", "linestyle": "--"},       # sky blue
    "permaFoam_omega10":  {"color": "#D55E00", "linestyle": "-."},       # vermillion
    "permaFoam_omega50":  {"color": "#E69F00", "linestyle": ":"},        # orange
    "cap":                {"color": "#009E73", "linestyle": (0, (3, 1, 1, 1))},  # bluish green
}

def parametrisation_style(parametrisation):
    return parametrisation_STYLE.get(parametrisation, {"color": "black", "linestyle": "-"})

def parametrisation_label(parametrisation):
    return {
        "drying": "(A) drying assumption",
        "permaFoam_omega10": r"(B) imp. factor, $\Omega=10$",
        "permaFoam_omega50": r"(C) imp. factor, $\Omega=50$",
        "cap": "(D) capillary bundle model",
        "drying_omega10": r"(E) drying assumption, with imp. factor $\Omega=10$",
    }.get(parametrisation)

def parametrisation_label_leg(parametrisation):
    return {
        "drying": "(A) drying assumption",
        "permaFoam_omega10": r"(C) imp. factor, $\Omega=10$",
        "permaFoam_omega50": r"(D) imp. factor, $\Omega=50$",
        "cap": "(E) capillary bundle model",
        "drying_omega10": r"(B) drying assumption, with imp. factor $\Omega=10$",
    }.get(parametrisation)

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

def exit_fluxes(db, cumulative=False, relative=False):
    """Exit fluxes (per unit width [m²/s]) for q_top, q_bottom, q_surf.

    Outflow through the seepage face is split at z = -1.5 m into interflow (top)
    and groundwater discharge (bottom); surface runoff is summed along the tilted
    top boundary.

    With `cumulative=True`, the fluxes are integrated over the true simulation
    times (trapezoidal, since the solver uses adaptive stepping) into cumulative
    exit volumes [m²]. With `relative=True`, each series is expressed as its
    share of the (running) total q_top+q_bottom+q_surf. The cumulative share is
    a much less noise-sensitive partitioning metric than the instantaneous flux
    ratio, since it isn't dominated by short high-flux events (e.g. a single
    ERA5 snowmelt spike).
    """
    dz = abs(db["Z"][1,0] - db["Z"][0,0])
    dx = abs(db["X"][0,1] - db["X"][0,0])
    dx_tilde = np.sqrt(dx**2 + dz**2)
    idx_top = db["Z"][:,-1] >= -1.5

    steps = range(1, len(db["time_array"]))
    time_s = np.array(db["time_array"])[1:]
    times_days = time_s / 24 / 60 / 60

    # only outflow counts (positive fluxes); NaN is treated as zero
    qx_out = np.array([db["qx_array"][i][:,-1] for i in steps])
    qx_out = np.where(qx_out > 0, qx_out, 0)
    q_top = np.sum(qx_out[:, idx_top], axis=1) * dz
    q_bottom = np.sum(qx_out[:, ~idx_top], axis=1) * dz
    q_surf = np.array([np.sum(np.where(db["q_runoff"][i] > 0, db["q_runoff"][i], 0)) for i in steps]) * dx_tilde

    if cumulative:
        q_top = cumulative_trapezoid(q_top, time_s, initial=0)
        q_bottom = cumulative_trapezoid(q_bottom, time_s, initial=0)
        q_surf = cumulative_trapezoid(q_surf, time_s, initial=0)

    if relative:
        q_tot = np.where(q_top + q_bottom + q_surf == 0, np.nan, q_top + q_bottom + q_surf)
        q_surf, q_top, q_bottom = q_surf/q_tot, q_top/q_tot, q_bottom/q_tot

    return times_days, q_top, q_bottom, q_surf

def plot_flux_comparison(dbs, parametrisations=None, quantity="top", cumulative=True, relative=False,
                         title=None, ylim=None, fig=None, ax=None, legend=False):
    """Overlay the exit flux (or cumulative exit volume) of one flow path across several parametrisations.

    `dbs` is a dict {parametrisation: db}. `quantity` selects which path to compare
    ("top"=interflow, "bottom"=GW discharge, "surf"=surface runoff). If no `ax` is
    given, a new figure is created.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12*cm,9.5*cm))

    if parametrisations is None:
        parametrisations = list(dbs.keys())

    quantity_index = {"top": 1, "bottom": 2, "surf": 3}[quantity]
    for parametrisation in parametrisations:
        series = exit_fluxes(dbs[parametrisation], cumulative=cumulative, relative=relative)
        style = parametrisation_style(parametrisation)
        ax.plot(series[0], series[quantity_index], linewidth=2,
               label=parametrisation_label_leg(parametrisation) or parametrisation,
               color=style["color"], linestyle=style["linestyle"])

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlabel(r"time [days]")
    if cumulative:
        ax.set_ylabel(r"V/V$_{tot}$ [-]" if relative else r"V [m$^2$]")
    else:
        ax.set_ylabel(r"q/q$_{tot}$ [-]" if relative else r"q [m$^2$/s]")
    ax.set_ylim(ylim)
    ax.set_title(title or "")
    if legend:
        ax.legend(loc="best", facecolor='none', edgecolor='none')
    if standalone:
        fig.tight_layout()
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

    quantities = ["surf", "top", "bottom"]
    titles = {"surf": "surface runoff",
              "top": "interflow",
              "bottom": "groundwater discharge"}

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm)
    fig, axes = plt.subplots(1, len(quantities), figsize=figsize)

    for ax, quantity in zip(axes, quantities):
        plot_flux_comparison(dbs, parametrisations, quantity=quantity, relative=relative,
                             title=titles[quantity], fig=fig, ax=ax)
    fig.tight_layout()
    if legend:
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
    """Cumulative exit volumes across parametrisations, one panel per flow path
    (surface runoff, interflow, groundwater discharge). Panels in
    `inset_quantities` carry a zoomed inset over the pre-melt window
    (0 to `zoom_end` days).

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
        times, Qtop, Qbot, Qsurf = exit_fluxes(dbs[p], cumulative=True)
        series[p] = {"t": times, "surf": Qsurf, "top": Qtop, "bottom": Qbot}

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm, 7*cm)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    insets = []

    for ax, quantity in zip(axes, titles):
        for p in parametrisations:
            style = parametrisation_style(p)
            ax.plot(series[p]["t"], series[p][quantity], linewidth=2, color=style["color"],
                    linestyle=style["linestyle"], label=parametrisation_label_leg(p) or p)
        ax.set_xlabel("time [days]")
        ax.set_title(titles[quantity])
        ax.spines[['top', 'right']].set_visible(False)

        if quantity not in inset_quantities:
            continue
        zoom_max = max(series[p][quantity][series[p]["t"] <= zoom_end].max() for p in parametrisations)
        if zoom_max <= 0:
            continue
        axins = inset_axes(ax, width="58%", height="52%",
                           bbox_to_anchor=(0.22, 0.34, 0.78, 0.54), bbox_transform=ax.transAxes,
                           loc="upper left", borderpad=0)
        for p in parametrisations:
            style = parametrisation_style(p)
            axins.plot(series[p]["t"], series[p][quantity], linewidth=1.4,
                       color=style["color"], linestyle=style["linestyle"])
        axins.set_xlim(0, zoom_end)
        axins.set_ylim(0, zoom_max * 1.15)
        axins.tick_params(labelsize=6.5)
        axins.set_facecolor("white")
        for spine in axins.spines.values():
            spine.set_edgecolor("0.4")
            spine.set_linewidth(0.8)
        mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5", linewidth=0.7)
        insets.append(axins)
    axes[0].set_ylabel(r"V [m$^2$]")

    fig.tight_layout()
    if legend:
        # Anchored to the figure bottom, so its rendered height (measured after a
        # draw) tells us exactly how much extra room to reserve -- no guessed margin.
        labels_margin = fig.subplotpars.bottom
        handles, labels = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3),
                         facecolor='none', edgecolor='none', bbox_to_anchor=(0.5, 0))
        fig.canvas.draw()
        legend_height = leg.get_window_extent(fig.canvas.get_renderer()) \
                           .transformed(fig.transFigure.inverted()).height
        fig.subplots_adjust(bottom=labels_margin + legend_height + 0.02)

    return fig, (tuple(axes), tuple(insets))

def plot_streamlines(X, Z, qx, qz, slope, field=None, cmap=icy_soil,
                     title=" ", xlabel="x [m]", ylabel="z [m]",
                     density=5.0, color="k", linewidth=1.0, arrowsize=0,
                     start_points=None, n_seeds=20, n_seeds_base=5, minlength=0.02,
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
    q_min : float or None
        Absolute cut-off on |q| [m/s]. Cells below it carry no visible flow (frozen /
        impermeable soil) and are blanked, so no streamline is drawn through them.
        If None, ``q_rel_min * q_ref`` is used.
    q_rel_min : float
        Relative cut-off used when ``q_min`` is None. streamplot is direction-only, so
        without a cut-off it draws continuous lines through the ice-rich layer even
        where |q| is orders of magnitude below the applied recharge.
    q_ref : float or None
        Flux scale the relative cut-off refers to [m/s]. The applied recharge is the
        meaningful one - it is the same for every parametrisation, so the panels stay
        comparable. ``plot_streamlines_grid`` passes it from ``db["q_inf_interp"]``.
        If None, ``max|q|`` of this panel is used, which is panel-dependent.
    minlength : float
        Shortest trajectory streamplot still draws, in normalised domain units. The
        default of matplotlib (0.1) throws away every path that the frozen layer cuts
        short, so whole panels come out empty.
    start_points : (n, 2) array or None
        Seeds for the forward integration. If None, ``n_seeds`` seeds are placed in the
        top row of cells and ``n_seeds_base`` in the bottom row. The basal seeds are
        needed because a frozen layer cuts the surface streamlines off before they reach
        the base, so the downslope flow feeding the seepage face would not be drawn.
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

    # scalar field of ice content as background

    pcm = ax.pcolormesh(X, Z, field,vmin=0,vmax=0.46,shading="gouraud", cmap=cmap)
    if cax0 is not None:
        fig.colorbar(pcm, cax=cax0,label=r"$\theta_i$ [-]")
    else:
        fig.colorbar(pcm, ax=ax,label=r"$\theta_i$ [-]")

    # grid for streamplot evaluation. The streamplot function can not be evaluted on the sheared mesh, 
    # so we interpolate the fluxes onto a rectilinear grid. 
    xg = np.linspace(X.min(), X.max(), 200)
    zg = np.linspace(Z.min(), Z.max(), 200)
    Xg, Zg = np.meshgrid(xg, zg)

    # Seeds in the top and the bottom row of cells, placed on the actual cell centres of
    # the sheared mesh rather than on a hard-coded offset below the surface.
    if start_points is None:
        inset = abs(Z[1, 0] - Z[0, 0])
        x_top = np.linspace(X.min(), X.max(), n_seeds + 2)[1:-1]
        x_bot = np.linspace(X.min(), X.max(), n_seeds_base + 2)[1:-1]
        start_points = np.concatenate([
            np.column_stack([x_top, Z[-1, 0] + slope * (x_top - X[0, 0]) - inset]),
            np.column_stack([x_bot, Z[0, 0] + slope * (x_bot - X[0, 0]) + inset]),
        ])
    start_points = np.asarray(start_points, dtype=np.float64)

    # streamplot needs the Cartesian components. qx is the flux through the vertical faces,
    # i.e. already the horizontal Darcy component (positive = +x = downslope for s < 0).
    # qz is the flux normal to the tilted top/bottom faces, n = (-s, 1)/sqrt(1+s^2), so
    # qz = (q_z - s*q_x)/sqrt(1+s^2)  ->  q_z = sqrt(1+s^2)*qz + s*qx.
    qz_cart = np.sqrt(1 + slope**2) * qz + slope * qx
    qx_interp = griddata((X.flatten(), Z.flatten()), qx.flatten(), (Xg, Zg), method="linear")
    qz_interp = griddata((X.flatten(), Z.flatten()), qz_cart.flatten(), (Xg, Zg), method="linear")

    stream = ax.streamplot(
        xg, zg, qx_interp, qz_interp,
        density=density,
        color=color,
        linewidth=linewidth,
        arrowsize=arrowsize,
        start_points=start_points,
        integration_direction="forward",
        minlength=minlength,
        zorder=3,
    )

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
                          cum=True, cum_height_ratio=1.3, legend=True, q_rel_min=1e-3):
    """
    Combined figure: the streamline field for each parametrisation stacked in
    rows (one row per parametrisation). If `cum` is True (default), a row with
    the cumulative exit volumes across parametrisations (one panel per flow
    path) is prepended on top, with a vline marking the time of `index`, plus a
    shared legend; if False, the figure is just the stacked streamline panels.
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())
    if title_fontsize is None:
        title_fontsize = 10
    if annotation_fontsize is None:
        annotation_fontsize = 8

    n_param = len(parametrisations)
    quantities = {"surf": "surface runoff", "top": "interflow", "bottom": "groundwater discharge"}
    legend_height_ratio = 0.35

    if cum:
        ncols = len(quantities)
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
    time_days = np.asarray(dbs[parametrisations[0]]["time_array"])[index] / (24 * 60 * 60)

    cum_axes = []
    if cum:
        for col, quantity in enumerate(quantities):
            ax = fig.add_subplot(gs[0, col])
            plot_flux_comparison(dbs, parametrisations, quantity=quantity, relative=relative,
                                 title=quantities[quantity], fig=fig, ax=ax)
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
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False,
                  facecolor='none', edgecolor='none', bbox_to_anchor=(0.5, cum_bottom - 0.01),
                  fontsize=annotation_fontsize)

    stream_axes = []
    last_row = n_header_rows + n_param - 1
    for row, parametrisation in enumerate(parametrisations, start=n_header_rows):
        db = dbs[parametrisation]
        gs_stream = gs[row, :].subgridspec(nrows=1, ncols=2, width_ratios=[1, 0.025], wspace=0.1)
        ax = fig.add_subplot(gs_stream[0, 0], sharex=stream_axes[0] if stream_axes else None)
        cax = fig.add_subplot(gs_stream[0, 1])

        plot_streamlines(db["X"], db["Z"], db["qx_centered_array"][index], db["qz_centered_array"][index],
                         slope=slope, field=db[field_key][index], linewidth=1,
                         title="", fig=fig, ax=ax, cax=cax)
        ax.set_title(parametrisation_label_leg(parametrisation) or parametrisation,
                     loc="left", fontsize=title_fontsize, fontweight="bold")

        if row < last_row:
            ax.set_xlabel("")
            ax.tick_params(bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)

        stream_axes.append(ax)

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

        ax.set_title(parametrisation_label_leg(parametrisation) or parametrisation,
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

    label = parametrisation_label(param)
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

    label = parametrisation_label(param)
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
        qz_i = griddata((Xf, Zf), (np.sqrt(1 + slope**2) * qz + slope * qx).flatten(), (Xg, Zg), method="linear")
        speed = np.sqrt(qx_i**2 + qz_i**2)
        return qx_i, qz_i, speed

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

def make_comparison_figure(X,Z,theta_l_cr, theta_i_cr, T_cr, theta_l_pf, theta_i_pf, T_pf, time, time_index=-1, x_cut=-1, cmap='RdBu_r'):
    fig, axs = plt.subplots(figsize=(PAGE_WIDTH_CM*cm,15*cm), nrows=3, ncols=2, constrained_layout=True)
    fig.suptitle(f"simulation time {np.round(time[time_index]/24/60/60, 0)} d", fontsize=12)

    theta_l_diff = theta_l_cr[time_index][:,:] - theta_l_pf[time_index]
    theta_i_diff = theta_i_cr[time_index][:,:] - theta_i_pf[time_index]
    T_diff = T_cr[time_index][:,:] - T_pf[time_index]

    print(f"max abs theta_l diff: {np.round(np.max(np.abs(theta_l_diff)), 3)}")
    print(f"max abs theta_i diff: {np.round(np.max(np.abs(theta_i_diff)), 3)}")
    print(f"max abs T diff: {np.round(np.max(np.abs(T_diff)), 3)}")

    for ax in axs.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('z [m]')

    ix_cut = np.argmin(np.abs(x_cut - X[0,:]))

    vmax_theta_l = np.max(np.abs(theta_l_diff))
    cbar0 = axs[0,0].pcolormesh(X,Z,theta_l_diff, cmap=cmap, shading='gouraud',vmin=-vmax_theta_l, vmax=vmax_theta_l)
    axs[0,0].axvline(x=X[0,ix_cut], color='k', lw=1.0, ls="--",zorder=10)
    axs[0,0].spines["bottom"].set_visible(False)
    axs[0,0].axes.get_xaxis().set_visible(False)
    axs[0,1].plot(theta_l_cr[time_index][:,ix_cut], Z[:,ix_cut], label='cryorich')
    axs[0,1].plot(theta_l_pf[time_index][:,ix_cut], Z[:,ix_cut],":", label='permaFoam')
    axs[0,1].set_xlabel(r'$\theta_l$ [-]')
    axs[0,1].legend(loc = "upper center", frameon=False)
    fig.colorbar(cbar0, ax=axs[0,0], orientation='vertical', label=r'$\Delta \theta_l$ [-]')

    vmax_theta_i = np.max(np.abs(theta_i_diff))
    cbar1 = axs[1,0].pcolormesh(X,Z,theta_i_diff, cmap=cmap, shading='gouraud',vmin=-vmax_theta_i, vmax=vmax_theta_i)
    axs[1,0].spines["bottom"].set_visible(False)
    axs[1,0].axes.get_xaxis().set_visible(False)
    axs[1,0].axvline(x=X[0,ix_cut], color='k', lw=1.0, ls="--",zorder=10)
    axs[1,1].plot(theta_i_cr[time_index][:,ix_cut], Z[:,ix_cut], label='cryorich')
    axs[1,1].plot(theta_i_pf[time_index][:,ix_cut], Z[:,ix_cut],":", label='permaFoam')
    axs[1,1].set_xlabel(r'$\theta_i$ [-]')
    fig.colorbar(cbar1, ax=axs[1,0], orientation='vertical', label=r'$\Delta \theta_i$ [-]')

    vmax_T = np.max(np.abs(T_diff))
    cbar2 = axs[2,0].pcolormesh(X,Z,T_diff, cmap=cmap, shading='gouraud',vmin=-vmax_T, vmax=vmax_T)
    axs[2,0].axvline(x=X[0,ix_cut], color='k', lw=1.0, ls="--",zorder=10)
    axs[2,0].set_xlabel('x [m]')
    axs[2,1].plot(T_cr[time_index][:,ix_cut], Z[:,ix_cut], label='cryorich')
    axs[2,1].plot(T_pf[time_index][:,ix_cut], Z[:,ix_cut],":", label='permaFoam')
    axs[2,1].set_xlabel('T [°C]')
    fig.colorbar(cbar2, ax=axs[2,0], orientation='vertical', label=r'$\Delta T$ [°C]')

    return fig,axs

def restrict(f, n=2):
    """Average fine nxn blocks onto coarse cells (cell-centred, conservative)."""
    nz, nx = f.shape
    return np.asarray(f).reshape(nz // n, n, nx // n, n).mean(axis=(1, 3))

def make_resolution_figure(X_c,Z_c,X_f,Z_f,theta_l_c, theta_i_c, T_i_c, theta_l_f, theta_i_f, T_i_f, time_c, time_index_c=-1, time_index_f=-1, x_cut=-1, cmap='RdBu_r'):
    fig, axs = plt.subplots(figsize=(PAGE_WIDTH_CM*cm,15*cm), nrows=3, ncols=2, constrained_layout=True)
    fig.suptitle(f"simulation time {np.round(time_c[time_index_c]/24/60/60, 0)} d", fontsize=12)

    if theta_l_c[time_index_c].shape[0] > theta_l_f[time_index_f].shape[0]:
        n = theta_l_c[time_index_c].shape[0] // theta_l_f[time_index_f].shape[0]
        th_l_c = restrict(theta_l_c[time_index_c], n=n)   
        th_l_f = np.asarray(theta_l_f[time_index_f])
        th_i_c = restrict(theta_i_c[time_index_c], n=n) 
        th_i_f = np.asarray(theta_i_f[time_index_f])
        T_c = restrict(T_i_c[time_index_c], n=n)
        T_f = np.asarray(T_i_f[time_index_f])
        X = X_f
        Z = Z_f

    else:
        n = theta_l_f[time_index_f].shape[0] // theta_l_c[time_index_c].shape[0]
        th_l_f = restrict(theta_l_f[time_index_f], n=n)
        th_l_c = np.asarray(theta_l_c[time_index_c])
        th_i_f = restrict(theta_i_f[time_index_f], n=n)
        th_i_c = np.asarray(theta_i_c[time_index_c])
        T_f = restrict(T_i_f[time_index_f], n=n)
        T_c = np.asarray(T_i_c[time_index_c])
        X = X_c
        Z = Z_c
    theta_l_diff = th_l_c - th_l_f
    theta_i_diff = th_i_c - th_i_f
    T_diff  = T_c - T_f    

    print(f"max abs theta_l diff: {np.round(np.max(np.abs(theta_l_diff)), 3)}")
    print(f"max abs theta_i diff: {np.round(np.max(np.abs(theta_i_diff)), 3)}")
    print(f"max abs T diff: {np.round(np.max(np.abs(T_diff)), 3)}")

    for ax in axs.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('z [m]')

    ix_cut = np.argmin(np.abs(x_cut - X[0,:]))

    vmax_theta_l = np.max(np.abs(theta_l_diff))
    cbar0 = axs[0,0].pcolormesh(X,Z,theta_l_diff, cmap=cmap, shading='gouraud',vmin=-vmax_theta_l, vmax=vmax_theta_l)
    axs[0,0].axvline(x=X[0,ix_cut], color='k', lw=1.0, ls="--",zorder=10)
    axs[0,0].spines["bottom"].set_visible(False)
    axs[0,0].axes.get_xaxis().set_visible(False)
    axs[0,1].plot(th_l_c[:,ix_cut], Z[:,ix_cut], label=r'$\Delta x=0.2$m, $\Delta z=0.005$m')
    axs[0,1].plot(th_l_f[:,ix_cut], Z[:,ix_cut],":", label=r'$\Delta x=0.1$m, $\Delta z=0.0025$m')
    axs[0,1].set_xlabel(r'$\theta_l$ [-]')
    axs[0,1].legend(loc = "upper center", frameon=False)
    fig.colorbar(cbar0, ax=axs[0,0], orientation='vertical', label=r'$\Delta \theta_l$ [-]')

    vmax_theta_i = np.max(np.abs(theta_i_diff))
    cbar1 = axs[1,0].pcolormesh(X,Z,theta_i_diff, cmap=cmap, shading='gouraud',vmin=-vmax_theta_i, vmax=vmax_theta_i)
    axs[1,0].spines["bottom"].set_visible(False)
    axs[1,0].axes.get_xaxis().set_visible(False)
    axs[1,0].axvline(x=X[0,ix_cut], color='k', lw=1.0, ls="--",zorder=10)
    axs[1,1].plot(th_i_c[:,ix_cut], Z[:,ix_cut], label=r'$\Delta x=0.2$m, $\Delta z=0.005$m')
    axs[1,1].plot(th_i_f[:,ix_cut], Z[:,ix_cut],":", label=r'$\Delta x=0.1$m, $\Delta z=0.0025$m')
    axs[1,1].set_xlabel(r'$\theta_i$ [-]')
    fig.colorbar(cbar1, ax=axs[1,0], orientation='vertical', label=r'$\Delta \theta_i$ [-]')

    vmax_T = np.max(np.abs(T_diff))
    cbar2 = axs[2,0].pcolormesh(X,Z,T_diff, cmap=cmap, shading='gouraud',vmin=-vmax_T, vmax=vmax_T)
    axs[2,0].axvline(x=X[0,ix_cut], color='k', lw=1.0, ls="--",zorder=10)
    axs[2,0].set_xlabel('x [m]')
    axs[2,1].plot(T_c[:,ix_cut], Z[:,ix_cut], label=r'$\Delta x=0.2$m, $\Delta z=0.005$m')
    axs[2,1].plot(T_f[:,ix_cut], Z[:,ix_cut],":", label=r'$\Delta x=0.1$m, $\Delta z=0.0025$m')
    axs[2,1].set_xlabel('T [°C]')
    fig.colorbar(cbar2, ax=axs[2,0], orientation='vertical', label=r'$\Delta T$ [°C]')

    return fig,axs
