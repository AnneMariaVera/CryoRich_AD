"""
Supplementary-material figure comparing our JAX Richards+heat model against the
permaFoam reference (Orgogozo et al., 2019) for the freeze/thaw validation case.

Panels:
  (a) domain-mean temperature vs time
  (b) domain-mean liquid/ice water content vs time
  (c,d,g) temperature field at the chosen checkpoint -- ours, permaFoam, and their regridded diff
  (e,f,h) ice content field at the chosen checkpoint -- ours, permaFoam, and their regridded diff
  (k,l,m) liquid water content field at the chosen checkpoint -- ours, permaFoam, and diff
  (i,j) vertical profile at the domain mid-column -- T and liquid/ice content, every checkpoint
        shown faint for context, full opacity on whichever one checkpoint_idx selects (the same
        one panels c-m show, so the two panel families always agree)

checkpoint_idx selects which permaFoam checkpoint (index into the sorted list of non-zero
checkpoint times -- 0 is the first quarter-month, -1/default is the last) the field maps (c-m),
figure title, and the highlighted profile curve (i,j) all show; (a,b) always show every
checkpoint's timeseries regardless.

Usage (from evaluate_validation.ipynb, after validation_run.ipynb has saved its output pickle):
    from make_permafoam_comparison_figure import make_comparison_figure

    fig = make_comparison_figure("cryorich_AD_output.pkl", "parsed_fields.pkl")
    fig = make_comparison_figure("cryorich_AD_output.pkl", "parsed_fields.pkl",
                                 checkpoint_idx=0,
                                 out_png="figures/permaFoam_comparison.png",
                                 out_pdf=None)

Both pickles may equally be passed as already-loaded dicts (e.g. the dict returned by
parse_permafoam_output), so a notebook that has just parsed a case need not round-trip
through disk.

Usage (from a terminal, in this folder):
    python make_permafoam_comparison_figure.py --ours cryorich_AD_output.pkl \
                                               --permafoam parsed_fields.pkl --checkpoint -1
"""
from pathlib import Path
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

HERE = Path(__file__).resolve().parent

# Defaults only -- every one of these is overridable per call.
DEFAULT_OURS_PKL = HERE / "cryorich_AD_output.pkl"
DEFAULT_PF_PKL = HERE / "parsed_fields.pkl"
DEFAULT_OUT_PNG = HERE / "permaFoam_comparison.png"
DEFAULT_OUT_PDF = HERE / "permaFoam_comparison.pdf"
DEFAULT_TITLE = "Validation with permaFoam (Orgogozo et al., 2019)"
A4_INCHES = (210 / 25.4, 297 / 25.4)  # DIN A4 portrait

# permaFoam.C: clamps its solved T field to [Tminc, Tmaxc] every Picard iteration, so our own
# T is clipped to the same window before comparing. Pass t_clip=None to compare unclipped.
DEFAULT_T_CLIP = (-4.75, 5.8)

# ---- validated categorical palette (dataviz skill, references/palette.md) ----
BLUE = "#2a78d6"    # slot 1 -- "our model"
ORANGE = "#eb6834"  # slot 2 -- "permaFoam"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


def clip_T(T, t_clip=DEFAULT_T_CLIP):
    """Clip T to permaFoam's own solver window; t_clip=None returns T untouched."""
    T = np.asarray(T)
    return T if t_clip is None else np.clip(T, t_clip[0], t_clip[1])


def load_pickle(source):
    """Accept either a path to a pickle or an already-loaded dict, and return the dict."""
    if isinstance(source, dict):
        return source
    with open(source, "rb") as f:
        return pkl.load(f)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def add_map(fig, gspos, X, Z, field, cmap, vmin, vmax, title, xlim, ylim,
            xticks=None, yticks=None, ylabel=None):
    ax = fig.add_subplot(gspos)
    im = ax.pcolormesh(X, Z, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(title, loc="left", fontsize=7, fontweight="bold")
    ax.set_xlabel("x [m]", fontsize=6.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=6.5)
    ax.tick_params(labelsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax, im


def pf_weighted_mean(field2d, dz_row):
    """permaFoam's mesh is z-graded (blockMeshDict simpleGrading ratio 0.22): cell height shrinks
    from ~2.3 cm at the bottom to ~0.5 cm at the top, a ~4x variation concentrating many thin
    cells near the (warm, thawed) top boundary and few large cells in the (cold) bulk/bottom. A
    plain np.mean() over such a grid is implicitly cell-COUNT-weighted, not volume-weighted, and
    is biased warm by ~0.7-0.9 C at every checkpoint here as a result -- confirmed by comparing
    against both this dz-weighted mean and an independent regrid onto our own uniform grid, which
    agree with each other. Our own grid is uniform, so np.mean() over it is already volume-correct
    and needs no such correction."""
    return float(np.sum(field2d.mean(axis=1) * dz_row) / np.sum(dz_row))


def regrid_to(X_target, Z_target, X_src, Z_src, field_src):
    """Linearly interpolate field_src (on the possibly-different, non-matching src grid) onto
    the target grid's cell centers. Both grids cover the same sheared rectangle, so a plain
    scattered-point linear interpolation (with nearest-neighbour fill for the thin sliver of
    target points that fall just outside the source convex hull, e.g. at corners) is enough --
    no special handling of the shear needed."""
    pts_src = np.column_stack([X_src.ravel(), Z_src.ravel()])
    vals_src = field_src.ravel()
    pts_tgt = np.column_stack([X_target.ravel(), Z_target.ravel()])
    out = griddata(pts_src, vals_src, pts_tgt, method="linear")
    nanmask = np.isnan(out)
    if nanmask.any():
        out[nanmask] = griddata(pts_src, vals_src, pts_tgt[nanmask], method="nearest")
    return out.reshape(X_target.shape)


def column_index(X, x_position=None):
    """Index of the column closest to x_position [m], or the mid-column if x_position is None."""
    X = np.asarray(X)
    if x_position is None:
        return X.shape[1] // 2
    return int(np.argmin(np.abs(X[0, :] - x_position)))


def make_comparison_figure(ours_pkl=DEFAULT_OURS_PKL, pf_pkl=DEFAULT_PF_PKL,
                           checkpoint_idx=-1,
                           out_png=DEFAULT_OUT_PNG, out_pdf=DEFAULT_OUT_PDF,
                           t_clip=DEFAULT_T_CLIP,
                           xlim=None, ylim=None, xticks=None, yticks=None,
                           profile_x=None,
                           title=DEFAULT_TITLE, figsize=A4_INCHES,
                           print_bias=True, dpi=300):
    """Build the permaFoam comparison figure.

    Parameters
    ----------
    ours_pkl : str, Path or dict
        Our own simulation output -- the pickle validation_run.ipynb writes, or that same dict.
        Needs keys "X", "Z", "time_array", "T_total", "theta_l_total", "theta_i_total".
    pf_pkl : str, Path or dict
        The parsed permaFoam reference -- parse_permafoam_output's pickle, or its return value.
    checkpoint_idx : int
        Which permaFoam checkpoint (index into the sorted non-zero checkpoint times, 0=first,
        -1=last) the field maps (c-m), title, and highlighted profile curve show.
    out_png, out_pdf : str, Path or None
        Where to save. Either may be None to skip that format; parent folders are created.
    t_clip : (float, float) or None
        Window our T is clipped to before comparing, matching permaFoam's own solver clamp.
        None disables clipping.
    xlim, ylim : (float, float) or None
        Axis limits shared by every field map. None derives them from the union of both grids,
        so the same limits apply to each panel pair and the panels stay comparable.
    xticks, yticks : sequence or None
        Explicit ticks for the field maps; None leaves matplotlib's own choice (identical across
        panels anyway, since the limits are pinned).
    profile_x : float or None
        x position [m] for the profile panels (i,j). None uses each grid's mid-column.
    title : str
        First line of the figure suptitle; the checkpoint time is appended as a second line.
    figsize : (float, float)
        Figure size in inches. The gridspec fractions are tuned for DIN A4 portrait.
    print_bias : bool
        Print the raw-vs-clipped domain-mean T bias table per checkpoint.
    dpi : int
        Raster resolution for the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ours = load_pickle(ours_pkl)
    pf = load_pickle(pf_pkl)

    t_ours = np.array(ours["time_array"]) / 86400.0  # days
    T_ours_mean = np.array([float(np.mean(clip_T(x, t_clip))) for x in ours["T_total"]])
    thl_ours_mean = np.array([float(np.mean(x)) for x in ours["theta_l_total"]])
    thi_ours_mean = np.array([float(np.mean(x)) for x in ours["theta_i_total"]])

    Z_pf_full = np.array(pf["Z"])
    dz_row_pf = np.gradient(Z_pf_full[:, 0])  # per-row cell height (x-grading is uniform, so this
                                              # is the same for every column) -- see pf_weighted_mean

    pf_times = [t for t in pf["times"] if t != 0 and "T_C" in pf["fields"][t]]
    assert pf_times, "the parsed permaFoam output has no non-zero checkpoint carrying a T field"
    t_pf = np.array(pf_times) / 86400.0
    T_pf_mean = np.array([pf_weighted_mean(np.array(pf["fields"][t]["T_C"]), dz_row_pf) for t in pf_times])
    thl_pf_mean = np.array([pf_weighted_mean(np.array(pf["fields"][t]["thetal"]), dz_row_pf) for t in pf_times])
    thi_pf_mean = np.array([pf_weighted_mean(np.array(pf["fields"][t]["thetag"]), dz_row_pf) for t in pf_times])

    # permaFoam's t=0 field snapshot isn't in the upload (uniform IC, not stored per-cell), but
    # the PDF specifies the *same* homogeneous IC (psi=-0.8m, T=-5C) for both models by
    # construction of the validation case -- so reuse our own t=0 mean (identical by definition)
    # rather than leaving permaFoam's curve appear to start near 0C.
    t_pf = np.concatenate([[0.0], t_pf])
    T_pf_mean = np.concatenate([[T_ours_mean[0]], T_pf_mean])
    thl_pf_mean = np.concatenate([[thl_ours_mean[0]], thl_pf_mean])
    thi_pf_mean = np.concatenate([[thi_ours_mean[0]], thi_pf_mean])

    # Match the chosen permaFoam checkpoint to the closest-in-time "ours" snapshot (same
    # nearest-match approach as the profile panels below) rather than assuming ours[-1] lines
    # up with pf_times[-1] -- that only happened to hold for the default (last) checkpoint.
    t_ours_arr = np.array(ours["time_array"])
    t_end_pf = pf_times[checkpoint_idx]
    i_end_ours = int(np.argmin(np.abs(t_ours_arr - t_end_pf)))

    X_ours, Z_ours = np.array(ours["X"]), np.array(ours["Z"])
    T_ours_end = clip_T(ours["T_total"][i_end_ours], t_clip)
    thl_ours_end = np.array(ours["theta_l_total"][i_end_ours])
    thi_ours_end = np.array(ours["theta_i_total"][i_end_ours])

    X_pf, Z_pf = np.array(pf["X"]), np.array(pf["Z"])
    T_pf_end = np.array(pf["fields"][t_end_pf]["T_C"])
    thl_pf_end = np.array(pf["fields"][t_end_pf]["thetal"])
    thi_pf_end = np.array(pf["fields"][t_end_pf]["thetag"])

    # Regrid permaFoam's (graded, e.g. 85x50) field onto our (uniform, e.g. 40x200) grid so we
    # can take a direct cell-by-cell difference -- the two grids don't share cell centers.
    T_pf_on_ours = regrid_to(X_ours, Z_ours, X_pf, Z_pf, T_pf_end)
    thi_pf_on_ours = regrid_to(X_ours, Z_ours, X_pf, Z_pf, thi_pf_end)
    thl_pf_on_ours = regrid_to(X_ours, Z_ours, X_pf, Z_pf, thl_pf_end)
    T_diff = T_ours_end - T_pf_on_ours
    thi_diff = thi_ours_end - thi_pf_on_ours
    thl_diff = thl_ours_end - thl_pf_on_ours

    # Both grids cover the same absolute domain (matched datum, see geometry setup) to within
    # half a cell width, so pin identical limits on each pair for a direct visual comparison
    # rather than letting matplotlib's autoscaler pick slightly different ranges per panel.
    if xlim is None:
        xlim = (min(X_ours.min(), X_pf.min()), max(X_ours.max(), X_pf.max()))
    if ylim is None:
        ylim = (min(Z_ours.min(), Z_pf.min()), max(Z_ours.max(), Z_pf.max()))

    fig = plt.figure(figsize=figsize)

    # timeseries row (2 columns), map rows (3 columns, 3 rows: T, theta_i, theta_l) and the
    # profile row (2 columns) use independent column grids so the third (difference) map column
    # doesn't force the other panels apart.
    gs_top = fig.add_gridspec(1, 2, top=0.925, bottom=0.805, left=0.09, right=0.80, wspace=0.40)
    gs_maps = fig.add_gridspec(3, 3, top=0.765, bottom=0.425, left=0.06, right=0.91,
                               hspace=0.30, wspace=0.55)
    gs_profile = fig.add_gridspec(1, 2, top=0.375, bottom=0.135, left=0.09, right=0.96, wspace=0.30)

    # --- (a) mean temperature ---
    ax = fig.add_subplot(gs_top[0, 0])
    ax.plot(t_ours, T_ours_mean, color=BLUE, lw=1.3, marker="o", ms=3, label="Our model", zorder=3)
    ax.plot(t_pf[1:], T_pf_mean[1:], color=ORANGE, lw=1.3, marker="o", ms=3, label="permaFoam", zorder=3)
    ax.plot(t_pf[:2], T_pf_mean[:2], color=ORANGE, lw=1.3, zorder=2)  # connector to the shared IC
    ax.plot(t_pf[0], T_pf_mean[0], marker="o", ms=4, mfc="white", mec=ORANGE, mew=1.5, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mean temperature [°C]")
    ax.set_title("(a) Domain-mean temperature", loc="left", fontsize=7.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    style_axes(ax)

    # --- (b) mean liquid/ice content ---
    ax = fig.add_subplot(gs_top[0, 1])
    ax.plot(t_ours, thl_ours_mean, color=BLUE, lw=1.3, marker="o", ms=3, ls="-", label="liquid, our model", zorder=3)
    ax.plot(t_ours, thi_ours_mean, color=BLUE, lw=1.3, marker="o", ms=3, ls="--", label="ice, our model", zorder=3)
    ax.plot(t_pf[1:], thl_pf_mean[1:], color=ORANGE, lw=1.3, marker="o", ms=3, ls="-", label="liquid, permaFoam", zorder=3)
    ax.plot(t_pf[1:], thi_pf_mean[1:], color=ORANGE, lw=1.3, marker="o", ms=3, ls="--", label="ice, permaFoam", zorder=3)
    ax.plot(t_pf[:2], thl_pf_mean[:2], color=ORANGE, lw=1.3, ls="-", zorder=2)
    ax.plot(t_pf[:2], thi_pf_mean[:2], color=ORANGE, lw=1.3, ls="--", zorder=2)
    ax.plot(t_pf[0], thl_pf_mean[0], marker="o", ms=4, mfc="white", mec=ORANGE, mew=1.5, zorder=4)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(r"Mean volumetric content [$-$]")
    ax.set_title("(b) Domain-mean liquid / ice content", loc="left", fontsize=7.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, fontsize=6.5, handlelength=1.8)
    style_axes(ax)

    # --- field maps at the chosen checkpoint ---
    clip_note = " (clipped)" if t_clip is not None else ""
    Tabs = max(abs(T_ours_end.min()), abs(T_ours_end.max()), abs(T_pf_end.min()), abs(T_pf_end.max()))
    ax_c, im_c = add_map(fig, gs_maps[0, 0], X_ours, Z_ours, T_ours_end, "coolwarm", -Tabs, Tabs,
                         f"(c) Our model — T{clip_note}", xlim, ylim, xticks, yticks, ylabel="z [m]")
    ax_d, im_d = add_map(fig, gs_maps[0, 1], X_pf, Z_pf, T_pf_end, "coolwarm", -Tabs, Tabs,
                         "(d) permaFoam — T", xlim, ylim, xticks, yticks)
    cbar = fig.colorbar(im_d, ax=[ax_c, ax_d], orientation="vertical", fraction=0.05, pad=0.02)
    cbar.set_label("T [°C]", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    Tdiff_abs = np.abs(T_diff).max()
    ax_g, im_g = add_map(fig, gs_maps[0, 2], X_ours, Z_ours, T_diff, "RdBu_r", -Tdiff_abs, Tdiff_abs,
                         "(g) Diff. (ours − permaFoam)", xlim, ylim, xticks, yticks)
    cbar_g = fig.colorbar(im_g, ax=ax_g, orientation="vertical", fraction=0.05, pad=0.05)
    cbar_g.set_label("ΔT [°C]", fontsize=6.5)
    cbar_g.ax.tick_params(labelsize=6)

    thi_max = max(thi_ours_end.max(), thi_pf_end.max())
    ax_e, im_e = add_map(fig, gs_maps[1, 0], X_ours, Z_ours, thi_ours_end, "Blues", 0, thi_max,
                         "(e) Our model — $\\theta_i$", xlim, ylim, xticks, yticks, ylabel="z [m]")
    ax_f, im_f = add_map(fig, gs_maps[1, 1], X_pf, Z_pf, thi_pf_end, "Blues", 0, thi_max,
                         "(f) permaFoam — $\\theta_i$", xlim, ylim, xticks, yticks)
    cbar2 = fig.colorbar(im_f, ax=[ax_e, ax_f], orientation="vertical", fraction=0.05, pad=0.02)
    cbar2.set_label(r"$\theta_i$ [$-$]", fontsize=6.5)
    cbar2.ax.tick_params(labelsize=6)

    thidiff_abs = np.abs(thi_diff).max()
    ax_h, im_h = add_map(fig, gs_maps[1, 2], X_ours, Z_ours, thi_diff, "RdBu_r", -thidiff_abs, thidiff_abs,
                         "(h) Diff. (ours − permaFoam)", xlim, ylim, xticks, yticks)
    cbar_h = fig.colorbar(im_h, ax=ax_h, orientation="vertical", fraction=0.05, pad=0.05)
    cbar_h.set_label(r"$\Delta\theta_i$ [$-$]", fontsize=6.5)
    cbar_h.ax.tick_params(labelsize=6)

    # --- liquid water content field maps ---
    thl_max = max(thl_ours_end.max(), thl_pf_end.max())
    ax_m, im_m = add_map(fig, gs_maps[2, 0], X_ours, Z_ours, thl_ours_end, "Blues", 0, thl_max,
                         "(k) Our model — $\\theta_l$", xlim, ylim, xticks, yticks, ylabel="z [m]")
    ax_n, im_n = add_map(fig, gs_maps[2, 1], X_pf, Z_pf, thl_pf_end, "Blues", 0, thl_max,
                         "(l) permaFoam — $\\theta_l$", xlim, ylim, xticks, yticks)
    cbar4 = fig.colorbar(im_n, ax=[ax_m, ax_n], orientation="vertical", fraction=0.05, pad=0.02)
    cbar4.set_label(r"$\theta_l$ [$-$]", fontsize=6.5)
    cbar4.ax.tick_params(labelsize=6)

    thldiff_abs = np.abs(thl_diff).max()
    ax_p, im_p = add_map(fig, gs_maps[2, 2], X_ours, Z_ours, thl_diff, "RdBu_r", -thldiff_abs, thldiff_abs,
                         "(m) Diff. (ours − permaFoam)", xlim, ylim, xticks, yticks)
    cbar_p = fig.colorbar(im_p, ax=ax_p, orientation="vertical", fraction=0.05, pad=0.05)
    cbar_p.set_label(r"$\Delta\theta_l$ [$-$]", fontsize=6.5)
    cbar_p.ax.tick_params(labelsize=6)

    # --- (i,j) vertical profile at the chosen column ---
    # Match each permaFoam checkpoint to the closest-in-time saved "ours" snapshot, rather than
    # assuming ours has exactly one snapshot per pf checkpoint (index i+1) -- that assumption
    # only holds if save_t was passed to geom.time_discretization(); when it wasn't (as here),
    # ours saves nearly every accepted timestep, so a fixed offset silently pulls near-t=0
    # snapshots instead of the intended quarter-month ones.
    save_idx_map = {t: int(np.argmin(np.abs(t_ours_arr - t))) for t in pf_times}
    mid_ours = column_index(X_ours, profile_x)
    mid_pf = column_index(X_pf, profile_x)
    z_ours_mid = Z_ours[:, mid_ours]
    z_pf_mid = Z_pf[:, mid_pf]
    x_mid = X_ours[0, mid_ours]

    # Highlight (full opacity + the legend label) whichever checkpoint panels (c-m) above are
    # actually showing -- previously this was hardcoded to "last checkpoint" regardless of
    # checkpoint_idx, so picking an earlier checkpoint for the maps left the profile panels
    # still emphasizing day-30 data, making the two panel families show different times without
    # any visual indication of the mismatch.
    highlight_k = checkpoint_idx % len(pf_times)
    alphas = np.full(len(pf_times), 0.30)
    alphas[highlight_k] = 1.0

    ax_i = fig.add_subplot(gs_profile[0, 0])
    ax_j = fig.add_subplot(gs_profile[0, 1])
    for k, t in enumerate(pf_times):
        i = save_idx_map[t]
        T_o = clip_T(ours["T_total"][i], t_clip)[:, mid_ours]
        thl_o = np.array(ours["theta_l_total"][i])[:, mid_ours]
        thi_o = np.array(ours["theta_i_total"][i])[:, mid_ours]
        fl = pf["fields"][t]
        T_p = np.array(fl["T_C"])[:, mid_pf]
        thl_p = np.array(fl["thetal"])[:, mid_pf]
        thi_p = np.array(fl["thetag"])[:, mid_pf]

        lbl_o = "Our model" if k == highlight_k else None
        lbl_p = "permaFoam" if k == highlight_k else None
        ax_i.plot(T_o, z_ours_mid, color=BLUE, lw=1.6, alpha=alphas[k], zorder=3, label=lbl_o)
        ax_i.plot(T_p, z_pf_mid, color=ORANGE, lw=1.6, ls="--", alpha=alphas[k], zorder=3, label=lbl_p)

        ax_j.plot(thl_o, z_ours_mid, color=BLUE, lw=1.6, alpha=alphas[k], zorder=3)
        ax_j.plot(thi_o, z_ours_mid, color=BLUE, lw=1.0, ls=":", alpha=alphas[k], zorder=3)
        ax_j.plot(thl_p, z_pf_mid, color=ORANGE, lw=1.6, ls="--", alpha=alphas[k], zorder=3)
        ax_j.plot(thi_p, z_pf_mid, color=ORANGE, lw=1.0, ls=":", alpha=alphas[k], zorder=3)

    ax_i.axvline(0, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax_i.set_xlabel("T [°C]")
    ax_i.set_ylabel("z [m]")
    ax_i.set_title(f"(i) Profile at x = {x_mid:.1f} m — T (ours{clip_note or ' unclipped'})",
                   loc="left", fontsize=7.5, fontweight="bold")
    ax_i.legend(frameon=False, loc="lower right", fontsize=6.5)
    style_axes(ax_i)

    theta_handles = [
        Line2D([0], [0], color=INK, lw=1.6, ls="-", label=r"$\theta_l$"),
        Line2D([0], [0], color=INK, lw=1.0, ls=":", label=r"$\theta_i$"),
    ]
    ax_j.set_xlabel(r"$\theta$ [$-$]")
    ax_j.set_ylabel("z [m]")
    ax_j.set_title(f"(j) Profile at x = {x_mid:.1f} m — liquid / ice content",
                   loc="left", fontsize=7.5, fontweight="bold")
    ax_j.legend(handles=theta_handles, frameon=False, loc="upper left", bbox_to_anchor=(0.05, 1.0), fontsize=6.5)
    style_axes(ax_j)

    t_sim = t_pf[1:][highlight_k]
    fig.suptitle(f"{title}\nt = {t_sim:.2f} d", fontsize=9, fontweight="bold", y=0.975)

    # no bbox_inches="tight": that would crop the canvas and the page would no longer be A4
    for path, kwargs in ((out_png, {"dpi": dpi}), (out_pdf, {})):
        if path is None:
            continue
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white", **kwargs)
        print("saved:", path)

    # --- effect of the postprocessing clip on the domain-mean bias (ours - permaFoam) ---
    if print_bias and t_clip is not None:
        T_ours_mean_raw = np.array([float(np.mean(x)) for x in ours["T_total"]])
        print(f"\nDomain-mean T bias (ours - permaFoam), raw vs. clipped to {t_clip}:")
        print(f"{'t [d]':>8} {'raw':>8} {'clipped':>8}")
        for k in range(1, len(t_pf)):
            j = int(np.argmin(np.abs(t_ours - t_pf[k])))
            print(f"{t_pf[k]:8.1f} {T_ours_mean_raw[j]-T_pf_mean[k]:8.3f} {T_ours_mean[j]-T_pf_mean[k]:8.3f}")

    return fig