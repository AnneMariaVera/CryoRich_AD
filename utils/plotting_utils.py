import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.animation as animation
from scipy.interpolate import griddata
import cmocean
import matplotlib.dates as mdates
import matplotlib as mpl
from datetime import timedelta


# Plotting settings
cobalt=(0.0, 0.28, 0.67)
persian_red = (0.8,0.2,0.2)
carolina_blue = (0.6,0.73,0.89)
red=(255/255,193/255,7/255)
orange = (216/255,27/255,96/255)
green = (30/255, 136/255, 229/255)
blue = (0, 77/255, 46/255)
cm = 1/2.54  # centimeters in inches
plt.rcParams['font.size'] = 12

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
    ax[0].plot(X[-1,:],Z[-1,:],"-", color = orange, linewidth = 2,zorder=1000)
    # ax[0].plot([X[-1,-1],X[-1,-3]],[Z[-1,-1],Z[-1,-3]],"-k",zorder=1001)
    # ax[0].plot([X[-3,-1],X[-3,-3]],[Z[-3,-1],Z[-3,-3]],"-k",zorder=1002)
    # ax[0].plot([X[-1,-3],X[-1,-3]],[Z[-1,-3],Z[-3,-3]],"-k",zorder=1003)
    ax[0].plot(X[:,0],Z[:,0],"-", color = green,linewidth = 2, zorder=1001)
    ax[0].plot(X[:, -1], Z[:, -1], "-", color = blue,linewidth = 2, zorder=1002)
    ax[0].plot(X[0,:], Z[0,:], "-", color = green, linewidth = 2, zorder=1003)
    ax[0].plot(X[-1,:],-.25-0.1*X[-1,:],"-k",zorder=1004) 
    ax[0].plot(X[-1,:],-.5-0.1*X[-1,:],"-k",zorder=1005)
    # ax[0].text(X[-1,0]+0.05,-.5-0.1*X[-1,0]+0.04,"loam", fontsize=12, color="k")
    # ax[0].text(X[-1,0]+0.05,-.5-0.1*X[-1,0]-0.25,"sand", fontsize=12, color="k")

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
    ax[1].text(0.0125,0.0117,r"primary var.", fontsize=12, color=persian_red, zorder=1001, va="top", ha="center")
    ax[1].plot(x,y_l,"-k",zorder=1001)
    ax[1].plot(np.array([0,0.025]),np.array([0,0]),"--k",zorder=1001)
    ax[1].plot(x,y_u,"-k",zorder=1001)
    ax[1].vlines(x=0,ymax=0.025+0.001,ymin=-0.001, color="k", zorder=1002)
    ax[1].vlines(x=0.025,ymax=y_u[-1]+0.001,ymin=y_l[-1]-0.001, color="k", zorder=1003)
    ax[1].text(-0.005,0.0125,r"$\Delta z$", fontsize=12, color="k")
    ax[1].text(0.0125,0.0005,r"$\Delta x$", fontsize=12, color="k")
    ax[1].text(0.0125,0.025,r"$\overline{\Delta x}$", fontsize=12, color="k")
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

def plot_angle_q(db, parametrization):

    times_days = np.array(db["time_array"]) / 24 / 60 / 60

    # Colormap + normalization
    cmap = cmocean.cm.thermal_r
    norm = mpl.colors.Normalize(vmin=times_days.min(),
                                vmax=times_days.max())

    # Colors for each curve
    colors = cmap(norm(times_days))

    fig, ax = plt.subplots(figsize=(8*cm,10.5*cm))

    for i in range(1,len(db["time_array"]),2):
        qx = db["qx_centered_array"][i][:, -1]
        qz = db["qz_centered_array"][i][:, -1]

        q_norm = np.sqrt(qx**2 + qz**2)

        cos_theta = -qz / np.clip(q_norm, 1e-12, None)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angles_deg = np.degrees(angle)
        # q_vector = np.concatenate([db["qx_array"][i][:,-1], db["qz_array"][i][:,-1]]).T
        # angle = angle_between(q_vector, z_vector)
        # angles.append(angle)
        ax.plot(angles_deg,db["Z"][:,-1],".",linewidth=1.5,label=f"t={db['time_array'][i]/24/60/60:.2f} days", color=colors[i])
    
    ax.axhline(-1.25, color='k', linestyle='--', linewidth=1.5)
    ax.axhline(-1.5, color='k', linestyle='--', linewidth=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    # Create colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for older matplotlib versions

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Time [days]")
    # Combine legends from both axes
    ax.spines['left'].set_position(('outward', 10))
    #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_xlim(-5, 185)
    plt.xlabel(r"angle between $\vec{q}$ and vertical [°]")
    plt.ylabel("Depth [m]")
    fig.tight_layout()
    plt.title(f"{parametrization}: x={db['X'][0,-1]+0.025/2:.2f} m")
    return fig, ax


def plot_qx_out(db, parametrization):
    fig, ax = plt.subplots(figsize=(8*cm,10.5*cm))
    # Time values in days
    times_days = np.array(db["time_array"]) / 24 / 60 / 60

    # Colormap + normalization
    cmap = cmocean.cm.thermal_r
    norm = mpl.colors.Normalize(vmin=times_days.min(),
                                vmax=times_days.max())

    # Colors for each curve
    colors = cmap(norm(times_days))

    for i in range(0,len(db["time_array"]),2):
        ax.semilogx(db["qx_array"][i][:,-1],db["Z"][:,-1],".",linewidth=1.5,label=f"t={db['time_array'][i]/24/60/60:.2f} days", color=colors[i])
    ax.axhline(-1.25, color='k', linestyle='--', linewidth=1.5)
    ax.axhline(-1.5, color='k', linestyle='--', linewidth=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    # Create colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for older matplotlib versions

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Time [days]")
    # Combine legends from both axes
    ax.spines['left'].set_position(('outward', 10))
    ax.set_xlim(1e-11,1e-4)
    #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    plt.xlabel(r"$q_x$ [m/s]")
    plt.ylabel("Depth [m]")
    fig.tight_layout()
    plt.title(f"{parametrization}: x={db['X'][0,-1]+0.025/2:.2f} m")
    return fig, ax

def plot_interflow(db, parametrization, ylim=None):
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

        q_tot = q_top[-1]+q_bottom[-1]
        q_top_rel.append(q_top[-1]/q_tot)
        q_bottom_rel.append(q_bottom[-1]/q_tot) 
    ax.plot(times_days[5::], q_surf, "-", linewidth = 2, label=r"$q_{surface}$", color=green)
    ax.plot(times_days[5::], q_bottom, ":", linewidth = 2, label=r"$q_{bottom}$", color=persian_red)
    ax.plot(times_days[5::], q_top, "--", linewidth = 2, label=r"$q_{top}$", color=carolina_blue)
    ax.spines[['top', 'right']].set_visible(False)

    # Combine legends from both axes
    ax.spines['left'].set_position(('outward', 10))
    #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_ylim(ylim)
    plt.xlabel(r"time [days]")
    plt.ylabel(r"$q_i / q_{tot}$ [-]")
    fig.legend(loc="center", facecolor='none', edgecolor='none')
    if parametrization == "drying":
        plt.title(f"drying")
    if parametrization == "permaFoam_omega10":
        plt.title(r"impedance, $\Omega=10$")
    if parametrization == "permaFoam_omega50":
        plt.title(r"impedance, $\Omega=50$")
    if parametrization == "cap":
        plt.title(f"capillary bundle")
    if parametrization == "drying_omega10":
        plt.title(r"drying, $\Omega=10$")
    fig.tight_layout()
    return fig, ax

def plot_streamContours(db, index, parametrization):
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

def plot_contourFlow(db,parametrization):
    X = db["X"]
    Z = db["Z"]
    psi_hor, psi_ver = stream_func(X,Z, db["qx_array"], db["qz_array"])
    fig,ax = plt.subplots(figsize = (18*cm,10*cm))
    ax.contourf(X,Z,psi_ver)
    return fig,ax

def plot_streamlines(db, X, Z, qx, qz, slope, field=None, cmap="Blues",
                     title=" ", xlabel="x [m]", ylabel="z [m]",
                     density=1.0, color="k", linewidth=1.0, arrowsize=0,cmap_stream="magma"):
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

    fig = plt.figure(figsize=(25*cm, 8*cm))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 0.02], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    # ax2 = fig.add_subplot(gs[0, 3])
    # cax2 = fig.add_subplot(gs[0, 4])

    # ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))

    # Optional scalar field as background
    if field is not None:
        pcm = ax.pcolormesh(X, Z, field,vmin=0,vmax=0.46,shading="gouraud", cmap=cmap)
        fig.colorbar(pcm, cax=cax0,label=r"$\theta_l$ [-]")

    # For skewed / deformed grids: interpolate onto a rectilinear evaluation grid
    Lx = X.max() - X.min()
    Lz = Z.max() - Z.min()

    # Precompute grids
    xg = np.linspace(X.min(), X.max(), 200)
    zg = np.linspace(Z.min(), Z.max(), 200)
    Xg, Zg = np.meshgrid(xg, zg)

    start_points = np.c_[
        np.linspace(0.5, 9.5, 20),
        -0.1*np.linspace(0.5, 9.5, 20) - 0.05
    ]

    qx_interp = griddata((X.flatten(), Z.flatten()), qx.flatten(), (Xg, Zg), method="linear")
    qz_interp = griddata((X.flatten(), Z.flatten()), qz.flatten()+slope*qx.flatten(), (Xg, Zg), method="linear")

    speed = np.sqrt(qx_interp**2 + qz_interp**2)

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
    fig.tight_layout()

    return fig, ax

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
