import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import matplotlib.animation as animation
from scipy.interpolate import griddata
from scipy.integrate import cumulative_trapezoid
import cmocean.cm as cmo
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
from datetime import timedelta
import jax.numpy as jnp


# Plotting settings
# Colour-blind safe palette, shared by all figures.
OI_BLUE = "#0072B2"
OI_SKY_BLUE = "#56B4E9"
OI_VERMILLION = "#D55E00"
OI_ORANGE = "#E69F00"
OI_GREEN = "#009E73"
GREY = "0.4"

# Font sizes [pt]: one size for everything, titles one step up, insets one step down.
FONT_SIZE = 8
TITLE_FONTSIZE = 9
SMALL_FONTSIZE = 7

# line widths [pt]
LINEWIDTH = 1.5          # data lines
LINEWIDTH_THIN = 1.0     # inset data lines, reference lines, domain outlines
SPINE_OFFSET = 5         # outward shift of the left spine [pt]

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": TITLE_FONTSIZE,
    "axes.formatter.use_mathtext": True,   # "x10^-6" instead of "1e-6"
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.prop_cycle": matplotlib.cycler(color=[OI_BLUE, OI_VERMILLION, OI_GREEN, OI_ORANGE,
                                                OI_SKY_BLUE]),
    "lines.linewidth": LINEWIDTH,
    "legend.frameon": False,
    "savefig.dpi": 1000,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def style_axes(ax, offset=SPINE_OFFSET):
    """Common 2D axis style: no top/right spine, left spine shifted outward."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_position(('outward', offset))


def panel_title(ax, text):
    """Bold, left-aligned panel title, e.g. '(A) drying assumption'."""
    ax.set_title(text, loc="left", fontsize=TITLE_FONTSIZE, fontweight="bold")


def time_label(ax, time_days):
    """'t = .. days' annotation in the upper right corner of `ax`."""
    ax.text(0.98, 0.97, f"$t$ = {time_days:.0f} days", transform=ax.transAxes,
            ha="right", va="top", fontsize=FONT_SIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))


def shared_legend(fig, ax, **kwargs):
    """Figure-wide legend for the parametrisation lines of `ax` (2 columns, so the
    longest label still fits in one text width)."""
    handles, labels = ax.get_legend_handles_labels()
    return fig.legend(handles, labels, ncol=min(len(labels), 2), frameon=False,
                      handlelength=3, columnspacing=1.5, **kwargs)

# COLORMAP for ice content maps
N = 256
start = np.array([168, 148, 125]) / 255
end   = np.array([250, 252, 255]) / 255      # icy white with a bit more blue

t = np.linspace(0, 1, N)
vals = np.ones((N, 4))
vals[:, :3] = (1 - t)[:, None] * start + t[:, None] * end
icy_soil = ListedColormap(vals)

# Page size used for plots
cm = 1/2.54  # centimeters in inches

PAGE_WIDTH_CM = 16.46   
PAGE_HEIGHT_CM = 22.41 

# Define line style and color for each parametrizations
parametrisation_STYLE = {
    "drying":             {"color": OI_BLUE,       "linestyle": "-"},
    "drying_omega10":     {"color": OI_SKY_BLUE,   "linestyle": "--"},
    "imp10":              {"color": OI_VERMILLION, "linestyle": "-."},
    "imp50":              {"color": OI_ORANGE,     "linestyle": ":"},
    "cap":                {"color": OI_GREEN,      "linestyle": (0, (3, 1, 1, 1))},
}

def parametrisation_style(parametrisation):
    return parametrisation_STYLE.get(parametrisation, {"color": "black", "linestyle": "-"})

def parametrisation_label(parametrisation):
    """Panel labels for the theory-section surface plots."""
    return {
        "drying": "(a) drying assumption",
        "imp10": r"(b) imp. factor, $\Omega=10$",
        "imp50": r"(c) imp. factor, $\Omega=50$",
        "cap": "(d) capillary bundle model",
    }.get(parametrisation)

def parametrisation_label_leg(parametrisation):
    return {
        "drying": "(A) drying assumption",
        "imp10": r"(C) imp. factor, $\Omega=10$",
        "imp50": r"(D) imp. factor, $\Omega=50$",
        "cap": "(E) capillary bundle model",
        "drying_omega10": r"(B) drying assumption, with imp. factor $\Omega=10$",
    }.get(parametrisation)

def plot_domain(X, Z, cellValues, colormap="Blues", title=" "):
    """Create the overview figrue of the test case used in the manuscript."""
    fig, ax = plt.subplots(1,2,figsize=(PAGE_WIDTH_CM*cm, 5*cm), gridspec_kw={'width_ratios': [1, 0.4]})
    ax[1].set_aspect('equal')
    ax[0].pcolormesh(X, Z, cellValues, cmap=colormap,shading = "gouraud", linewidth=0.05)
    ax[0].plot(X[-1,:],Z[-1,:],"-", color = OI_VERMILLION, linewidth = 2,zorder=1000)

    ax[0].plot(X[:,0],Z[:,0],"-", color = OI_SKY_BLUE,linewidth = 2, zorder=1004)
    ax[0].plot(X[:, -1], Z[:, -1], "-", color = OI_GREEN,linewidth = 2, zorder=1004)
    ax[0].plot(X[0,:], Z[0,:], "-", color = OI_SKY_BLUE, linewidth = 2, zorder=1006)
    ax[0].plot(X[-1,:],-.25-0.1*X[-1,:],"-k",linewidth=LINEWIDTH_THIN,zorder=1002)
    ax[0].plot(X[-1,:],-.5-0.1*X[-1,:],"-k",linewidth=LINEWIDTH_THIN,zorder=1000)


    style_axes(ax[0])
    ax[0].set_xlabel("$x$ [m]")
    ax[0].set_ylabel("$z$ [m]")
    eps = 0.1
    ax[0].set_xlim(X.min()-eps, X.max()+eps)
    ax[0].set_ylim(Z.min()-eps, Z.max()+eps)

    if title is not None:
        ax[0].title.set_text(title)

    x = np.array([0-0.001,.025+0.001])
    y_l = -0.1*x
    y_u = -0.1*x+.025
    ax[1].plot(0.0125,0.0125,"o",color='k',markersize=7,zorder=1001)
    ax[1].text(0.0125,0.01,r"primary var.", color='k', zorder=1001, va="top", ha="center")
    ax[1].plot(x,y_l,"-k",linewidth=LINEWIDTH_THIN,zorder=1001)
    ax[1].plot(np.array([0,0.025]),np.array([0,0]),"--k",linewidth=LINEWIDTH_THIN,zorder=1001)
    ax[1].plot(x,y_u,"-k",linewidth=LINEWIDTH_THIN,zorder=1001)
    ax[1].vlines(x=0,ymax=0.025+0.001,ymin=-0.001, color="k", linewidth=LINEWIDTH_THIN, zorder=1002)
    ax[1].vlines(x=0.025,ymax=y_u[-1]+0.001,ymin=y_l[-1]-0.001, color="k", linewidth=LINEWIDTH_THIN, zorder=1003)
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
    style_axes(ax)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$z$ [m]")
    eps = 0.1
    ax.set_xlim(X.min()-eps, X.max()+eps)
    ax.set_ylim(Z.min()-eps, Z.max()+eps)

    if title is not None:
        ax.title.set_text(title)
    fig.tight_layout()
    return fig, ax

def exit_fluxes(db, cumulative=False, relative=False):
    """Exit fluxes (per unit width [m²/s]) for q_top, q_bottom, q_surf.

    Outflow through the seepage face is split at z = -1.5 m into interflow (top)
    and groundwater discharge (bottom); surface runoff is summed along the tilted
    top boundary.

    With `cumulative=True`, the fluxes are integrated over the simulation
    times into cumulative exit volumes [m²]. 
    
    With `relative=True`, each series is expressed as its share of the 
    total q_top+q_bottom+q_surf. 
    """
    dz = abs(db["Z"][1,0] - db["Z"][0,0])
    dx = abs(db["X"][0,1] - db["X"][0,0])
    slope = 0.1
    dx_tilde = dx * np.sqrt(1 + slope**2)

    idx_top = db["Z"][:,-1] >= -1.5

    steps = range(1, len(db["time_array"]))
    time_s = np.array(db["time_array"])[1:]
    times_days = time_s / 24 / 60 / 60

    qx_out = np.array([db["qx_array"][i][:,-1] for i in steps])
    # account only for outflow 
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
    """Create one plot across several parametrisations.

    `dbs` is a dict {parametrisation: db}. `quantity` selects which flux to compare
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
        ax.plot(series[0], series[quantity_index], linewidth=LINEWIDTH,
               label=parametrisation_label_leg(parametrisation) or parametrisation,
               color=style["color"], linestyle=style["linestyle"])

    style_axes(ax)
    ax.set_xlabel(r"time [days]")
    if cumulative:
        ax.set_ylabel(r"$V/V_\mathrm{tot}$ [-]" if relative else r"$V$ [m$^2$]")
    else:
        ax.set_ylabel(r"$q/q_\mathrm{tot}$ [-]" if relative else r"$q$ [m$^2$/s]")
    ax.set_ylim(ylim)
    ax.set_title(title or "")
    if legend:
        ax.legend(loc="best")
    if standalone:
        fig.tight_layout()
    return fig, ax

def plot_cumulative_partitioning_grid(dbs, parametrisations=None, relative=False, figsize=None, legend=True,index=None):
    """Cumulative exit-volume comparison across parametrisations, one panel per flow path.
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())

    quantities = ["surf", "top", "bottom"]
    titles = {"surf": "surface runoff",
              "top": "interflow",
              "bottom": "groundwater discharge"}

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm, 5*cm)
    fig, axes = plt.subplots(1, len(quantities), figsize=figsize)
    if index is not None: 
        time_days = np.asarray(dbs[parametrisations[0]]["time_array"])[index] / (24 * 60 * 60)

    for ax, quantity in zip(axes, quantities):
        plot_flux_comparison(dbs, parametrisations, quantity=quantity, relative=relative,
                             title=titles[quantity], fig=fig, ax=ax)
        if index is not None:
            ax.axvline(time_days, color="k", linestyle="--", linewidth=LINEWIDTH_THIN)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.tight_layout()
    if legend:
        labels_margin = fig.subplotpars.bottom

        leg = shared_legend(fig, axes[0], loc="lower center", bbox_to_anchor=(0.5, 0))
        fig.canvas.draw()
        legend_height = leg.get_window_extent(fig.canvas.get_renderer()) \
                           .transformed(fig.transFigure.inverted()).height
        fig.subplots_adjust(bottom=labels_margin + legend_height + 0.02)

    return fig, tuple(axes)

def plot_cumulative_partitioning_grid_with_inset(dbs, parametrisations=None, zoom_end=150,
                                                  inset_quantities=("surf", "top"),
                                                  figsize=None, legend=True):
    """
    Essentially the same plot is created as with 'plot_cumulative_partitioning_grid', but with insets
    zooming into a defined time span. 
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())

    titles = {"surf": "surface runoff", "top": "interflow", "bottom": "groundwater discharge"}

    series = {}
    for p in parametrisations:
        times, Qtop, Qbot, Qsurf = exit_fluxes(dbs[p], cumulative=True)
        series[p] = {"t": times, "surf": Qsurf, "top": Qtop, "bottom": Qbot}

    if figsize is None:
        figsize = (PAGE_WIDTH_CM*cm, 5*cm)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    insets = []

    for ax, quantity in zip(axes, titles):
        for p in parametrisations:
            style = parametrisation_style(p)
            ax.plot(series[p]["t"], series[p][quantity], linewidth=LINEWIDTH, color=style["color"],
                    linestyle=style["linestyle"], label=parametrisation_label_leg(p) or p)
        ax.set_xlabel("time [days]")
        ax.set_title(titles[quantity])
        style_axes(ax)

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
            axins.plot(series[p]["t"], series[p][quantity], linewidth=LINEWIDTH_THIN,
                       color=style["color"], linestyle=style["linestyle"])
        axins.set_xlim(0, zoom_end)
        axins.set_ylim(0, zoom_max * 1.15)
        axins.tick_params(labelsize=SMALL_FONTSIZE)
        axins.set_facecolor("white")
        for spine in axins.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(GREY)
            spine.set_linewidth(0.8)
        mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec=GREY, linewidth=0.8)
        insets.append(axins)
    axes[0].set_ylabel(r"$V$ [m$^2$]")

    fig.tight_layout()
    if legend:
        labels_margin = fig.subplotpars.bottom
        leg = shared_legend(fig, axes[0], loc="lower center", bbox_to_anchor=(0.5, 0))
        fig.canvas.draw()
        legend_height = leg.get_window_extent(fig.canvas.get_renderer()) \
                           .transformed(fig.transFigure.inverted()).height
        fig.subplots_adjust(bottom=labels_margin + legend_height + 0.02)

    return fig, (tuple(axes), tuple(insets))

def plot_streamlines(X, Z, qx, qz, slope, field=None, cmap=icy_soil,
                     title=" ", xlabel="$x$ [m]", ylabel="$z$ [m]",
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
    slope: float
        slope of the domain
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axes(ax)

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
    # the sheared mesh.
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
    # i.e. already the horizontal Darcy component.
    # qz is the flux normal to the tilted top/bottom faces, n = (-s, 1)/sqrt(1+s^2), so
    # qz = (q_z - s*q_x)/sqrt(1+s^2)  ->  q_z = sqrt(1+s^2)*qz + s*qx.
    qz_cart = np.sqrt(1 + slope**2) * qz + slope * qx
    qx_interp = griddata((X.flatten(), Z.flatten()), qx.flatten(), (Xg, Zg), method="linear")
    qz_interp = griddata((X.flatten(), Z.flatten()), qz_cart.flatten(), (Xg, Zg), method="linear")

    ax.streamplot(
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
    ax.plot([X[0,0],X[0,-1]],[Z[0,0],Z[0,-1]],"-k",linewidth=LINEWIDTH_THIN)
    ax.plot([X[-1,0],X[0,0]],[Z[-1,0],Z[0,0]],"-k",linewidth=LINEWIDTH_THIN)
    ax.plot([X[-1,0],X[-1,-1]],[Z[-1,0],Z[-1,-1]],"-k",linewidth=LINEWIDTH_THIN)
    ax.plot([X[0,-1],X[-1,-1]],[Z[0,-1],Z[-1,-1]],"-k",linewidth=LINEWIDTH_THIN)

    ax.set_title(title)
    if standalone:
        fig.tight_layout()

    return fig, ax

def plot_streamlines_grid(dbs, index, parametrisations=None, slope=-0.1, field_key="theta_i_total",
                          relative=False, figsize=None, title_fontsize=None, annotation_fontsize=None,
                          cum=True, cum_height_ratio=1.3, legend=True):
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
        title_fontsize = TITLE_FONTSIZE
    if annotation_fontsize is None:
        annotation_fontsize = FONT_SIZE

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
            ax.axvline(time_days, color="k", linestyle="--", linewidth=LINEWIDTH_THIN)
            cum_axes.append(ax)

    adjust_kw = dict(left=0.13, right=0.90, bottom=0.05, top=0.96)
    fig.subplots_adjust(**adjust_kw)

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

    time_label(stream_axes[0], time_days)

    if cum and legend:
        to_fig = fig.transFigure.inverted()
        pad = 0.01
        leg = None
        for _ in range(10):
            if leg is not None:
                leg.remove()
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            cum_bottom = min(ax.get_tightbbox(renderer).transformed(to_fig).y0 for ax in cum_axes)
            leg = shared_legend(fig, cum_axes[0], loc="upper center",
                                bbox_to_anchor=(0.5, cum_bottom - pad), fontsize=annotation_fontsize)
            fig.canvas.draw()
            leg_bottom = leg.get_window_extent(renderer).transformed(to_fig).y0
            stream_top = stream_axes[0].get_tightbbox(renderer).transformed(to_fig).y1
            overlap = stream_top - (leg_bottom - pad)
            if overlap <= 0:
                break
            row_height = stream_axes[0].get_position().height
            height_ratios[1] += overlap / row_height
            gs.set_height_ratios(height_ratios)
            fig.subplots_adjust(**adjust_kw) 

    return fig, (tuple(cum_axes), tuple(stream_axes))

def plot_T_grid(dbs, index, parametrisations=None, figsize=None, title_fontsize=None, annotation_fontsize=None, field_key="T_total", cmap=cmo.thermal):
    """
    plot of temperature distribution within the crosssection for all parametrisaitons.
    """
    if parametrisations is None:
        parametrisations = list(dbs.keys())
    if title_fontsize is None:
        title_fontsize = TITLE_FONTSIZE
    if annotation_fontsize is None:
        annotation_fontsize = FONT_SIZE

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
      
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        fig.colorbar(sm, cax=cax, label=r"$T$ [°C]")

        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        style_axes(ax)

        ax.set_title(parametrisation_label_leg(parametrisation) or parametrisation,
                    loc="left", fontsize=title_fontsize, fontweight="bold")

        if row < last_row:
            ax.set_xlabel("")
            ax.tick_params(bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)

        T_axes.append(ax)

    time_label(T_axes[0], time_days)

    return fig, tuple(T_axes)


def plot_krel_surface(soil_type, param, hc, T, figsize=None, n_per_row=2, title_fontsize=TITLE_FONTSIZE):
    """3D surface of log10(k_rel) vs h_c and T for soil type 1 (sand), used to
    compare the closure shape across parametrisations.
    """
    if figsize is None:
        width = (PAGE_WIDTH_CM-3.5*cm)/n_per_row*cm
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
    ax.set_zlabel(r'$k_\mathrm{rel}$ [-]')

    label = parametrisation_label(param)
    if label:
        ax.text2D(0.02, 0.95, label, transform=ax.transAxes, fontsize=title_fontsize,
                  fontweight="bold", ha="left", va="top")

    fig.tight_layout()
    return fig, ax

def plot_theta_l_surface(soil_type, param, hc, T, figsize=None, n_per_row=2, title_fontsize=TITLE_FONTSIZE):
    """3D surface of theta_l vs h_c and T for soil type 1 (sand), used to
    compare the closure shape across parametrisations.
    """
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
    ax.set_zlabel(r'$\theta_{l}$ [-]')

    label = parametrisation_label(param)
    if label:
        ax.text2D(0.02, 0.95, label, transform=ax.transAxes, fontsize=title_fontsize,
                  fontweight="bold", ha="left", va="top")

    fig.tight_layout()
    return fig, ax

def plot_forcing_data_split(time_array_spin, time_array_sim, time_array_ideal,
                            tl1_array_spin, tl1_array_sim, T_ideal,
                            recharge_spin, recharge_sim, recharge_ideal):
    """Plot the forcing as spinup (left) that splits into the ERA5 simulation (top right)
    and the idealized case (bottom right).
    All time arrays are seconds since 1970-01-01 UTC. ERA5 data is smoothed with a 24 h moving average
    and its recharge drawn as bars; the constant idealized recharge is drawn as a line."""
    smooth = lambda a: np.convolve(a, np.ones(24)/24, mode="same")

    fig = plt.figure(figsize=(PAGE_WIDTH_CM*cm, 11*cm))
    duration_spin = time_array_spin[-1] - time_array_spin[0]
    duration_sim = time_array_sim[-1] - time_array_sim[0]
    # 4 rows: right panels span two rows each, spinup spans the middle two -> all panels equally high,
    # spinup centred vertically
    gs = fig.add_gridspec(4, 2, width_ratios=[duration_spin, duration_sim], wspace=1.0, hspace=3.4)
    ax_spin = fig.add_subplot(gs[1:3, 0])
    ax_ideal = fig.add_subplot(gs[0:2, 1])
    ax_sim = fig.add_subplot(gs[2:4, 1])

    panels = [(ax_spin, "spinup", time_array_spin, smooth(recharge_spin), smooth(tl1_array_spin), True),
              (ax_ideal, "idealized", time_array_ideal, recharge_ideal, T_ideal, False),
              (ax_sim, "simulation", time_array_sim, smooth(recharge_sim), smooth(tl1_array_sim), True)]
    axes_cp = []
    for ax, title, time_array, recharge, T, bars in panels:
        t = time_array/24/60/60
        ax_cp = ax.twinx()
        axes_cp.append(ax_cp)
        if bars:
            ax.bar(t, recharge, width=t[1]-t[0], color=OI_BLUE)
        else:
            ax.plot(t, recharge, color=OI_BLUE, linewidth=LINEWIDTH)
        ax.axhline(0, color=OI_BLUE, linestyle="--", linewidth=LINEWIDTH_THIN)
        ax.set_ylabel(r"$q_\mathrm{inf}$ [m/s]", color=OI_BLUE)
        ax.tick_params(axis="y", colors=OI_BLUE)
        if np.ptp(recharge) == 0:   # constant recharge (idealized case): line a third down, clear of T = 0
            ax.set_ylim(min(0, 3*recharge[0]), max(0, 3*recharge[0]))
        ax.set_xlim(t[0], t[-1])
        ax.yaxis.set_inverted(True)

        ax_cp.plot(t, T, color=OI_VERMILLION, linewidth=LINEWIDTH)
        ax_cp.axhline(0, color=OI_VERMILLION, linestyle="--", linewidth=LINEWIDTH_THIN)
        ax_cp.set_ylabel("$T_{surface}$ [°C]", color=OI_VERMILLION)
        ax_cp.tick_params(axis="y", colors=OI_VERMILLION)
        ax_cp.set_ylim([min(T) - 2, max(T) + 2])

        style_axes(ax)
        ax.spines["left"].set_color(OI_BLUE)
        ax_cp.spines[["left", "top", "bottom"]].set_visible(False)
        ax_cp.spines["right"].set_visible(True)
        ax_cp.spines["right"].set_color(OI_VERMILLION)
        ax_cp.spines['right'].set_position(('outward', SPINE_OFFSET))
        panel_title(ax, title)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')

    # spinup branches into simulation and idealized case: lines from the spinup centre to the centres
    # of the right panels, starting/ending just outside the y-axis labels so they don't cross them
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    pad = 0.01
    x_start = to_fig.transform(axes_cp[0].yaxis.label.get_window_extent().get_points())[1, 0] + pad
    y_start = ax_spin.get_position().y0 + ax_spin.get_position().height/2
    for ax in (ax_sim, ax_ideal):
        x_end = to_fig.transform(ax.yaxis.label.get_window_extent().get_points())[0, 0] - pad
        y_end = ax.get_position().y0 + ax.get_position().height/2
        fig.add_artist(ConnectionPatch(xyA=(x_start, y_start), xyB=(x_end, y_end),
                                       coordsA=fig.transFigure, coordsB=fig.transFigure,
                                       color="k", linewidth=0.8))

    return fig, (ax_spin, ax_sim, ax_ideal), axes_cp

def create_animation_streamlines(
    X, Z,
    cell1_t, qx_t, qz_t,
    cell2_t, cell3_t,
    runoff, q_inf, skt_array,
    time_array,
    dates=None,
    slope=0,
    time_mode=None,   
    density=1,
    linewidth=2,
    cmaps=("Blues", "bone_r", "coolwarm"),
    clabels=(r" ", r" ", r" "),
    cmap_stream="magma",
    arrowsize=0, 
    dt = None
):
    """Animate scalar fields and streamlines over time with flexible time axis."""

    # Time handling
    time_days = (time_array - time_array[0]) / 86400

    # auto-detect mode if not provided
    if time_mode is None:
        time_mode = "date" if dates is not None else "days"

    if time_mode == "date" and dates is not None:
        x_time = dates
        x_full = [dates[0] + timedelta(seconds=float(t)) for t in time_array]
        x_label = "date"
        use_dates = True
    else:
        x_time = time_days
        x_full = time_days
        x_label = "time [days]"
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
        ax.plot([X[0,0], X[0,-1]], [Z[0,0], Z[0,-1]], "-k", linewidth=LINEWIDTH_THIN)
        ax.plot([X[-1,0], X[0,0]], [Z[-1,0], Z[0,0]], "-k", linewidth=LINEWIDTH_THIN)
        ax.plot([X[-1,0], X[-1,-1]], [Z[-1,0], Z[-1,-1]], "-k", linewidth=LINEWIDTH_THIN)
        ax.plot([X[0,-1], X[-1,-1]], [Z[0,-1], Z[-1,-1]], "-k", linewidth=LINEWIDTH_THIN)

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
    panel_title(ax_bc, "surface boundary conditions")
    ax_cp = ax_bc.twinx()

    ax = [fig.add_subplot(gs[i, 0]) for i in range(1, 4)]
    cax00 = fig.add_subplot(gs[1, 1])
    cax01 = fig.add_subplot(gs[1, 2])
    cax1  = fig.add_subplot(gs[2, 1])
    cax2  = fig.add_subplot(gs[3, 1])

    # Top plot (UPDATED)
    dt = np.diff(x_full).mean() if use_dates else np.diff(x_time).mean()
    ax_bc.bar(x_time, q_inf, width=dt, color=OI_BLUE,label="infiltration")
    ax_bc.bar(x_time, runoff,width=dt, color=OI_SKY_BLUE,label="runoff")
    ax_bc.legend(loc="upper left")
    ax_bc.set_ylabel("$q$ [m/s]", color=OI_BLUE)
    ax_bc.invert_yaxis()

    ax_cp.plot(x_time, skt_array, color=OI_VERMILLION)
    ax_cp.set_ylabel("$T$ [°C]", color=OI_VERMILLION)
    ax_cp.spines["right"].set_visible(True)
    if time_mode == "date":
        ax_cp.axhline(0, linestyle="--", color=OI_VERMILLION, linewidth=LINEWIDTH_THIN)

    time_indicator = ax_cp.axvline(x_time[0], color="k")

    ax_cp.set_xlabel(x_label)

    if use_dates:
        ax_cp.xaxis.set_major_locator(mdates.MonthLocator())
        ax_cp.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax_cp.get_xticklabels(), rotation=45, ha='right')
    ax_bc.set_xlabel(x_label)

    # Axes formatting
    for a in ax:
        a.set_aspect("equal")
        a.set_xlim([0, 10])
        a.set_ylim([-2, 0])
        a.set_xlabel("$x$ [m]")
        a.set_ylabel("$z$ [m]")
        style_axes(a)

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

    for i in range(1, len(cell1_t)):
        qz_t[i] = np.sqrt(1 + slope**2) * qz_t[i] + slope * qx_t[i]
    qx_i, qz_i, speed = interpolate_flow(qx_t[-1], qz_t[-1])
    lw = compute_linewidth(speed)

    # log10-distributed colormap; 
    norm = matplotlib.colors.LogNorm(
        vmin=np.nanmin(speed[speed > 0]),
        vmax=np.nanmax(speed),
        clip=True
    )

    start_points = np.c_[
        np.linspace(0.5, 9.5, 20),
        -0.1*np.linspace(0.5, 9.5, 20) - 0.05
    ]

    stream = ax[0].streamplot(
        xg, zg, qx_i, qz_i,
        density=density,
        color=np.maximum(speed, norm.vmin),
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
            color=np.maximum(speed, norm.vmin),
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
    """ Comparison figure for supplementary material. Compares our solver with permaFoam."""
    fig, axs = plt.subplots(figsize=(PAGE_WIDTH_CM*cm,15*cm), nrows=3, ncols=2, constrained_layout=True)
    fig.suptitle(f"simulation time {time[time_index]/24/60/60:.0f} days")

    theta_l_diff = theta_l_cr[time_index][:,:] - theta_l_pf[time_index]
    theta_i_diff = theta_i_cr[time_index][:,:] - theta_i_pf[time_index]
    T_diff = T_cr[time_index][:,:] - T_pf[time_index]

    print(f"max abs theta_l diff: {np.round(np.max(np.abs(theta_l_diff)), 3)}")
    print(f"max abs theta_i diff: {np.round(np.max(np.abs(theta_i_diff)), 3)}")
    print(f"max abs T diff: {np.round(np.max(np.abs(T_diff)), 3)}")

    for ax in axs.flatten():
        style_axes(ax)
        ax.set_ylabel('$z$ [m]')

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
    axs[2,0].set_xlabel('$x$ [m]')
    axs[2,1].plot(T_cr[time_index][:,ix_cut], Z[:,ix_cut], label='cryorich')
    axs[2,1].plot(T_pf[time_index][:,ix_cut], Z[:,ix_cut],":", label='permaFoam')
    axs[2,1].set_xlabel('$T$ [°C]')
    fig.colorbar(cbar2, ax=axs[2,0], orientation='vertical', label=r'$\Delta T$ [°C]')

    return fig,axs