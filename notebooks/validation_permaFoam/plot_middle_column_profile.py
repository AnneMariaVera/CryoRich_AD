"""
Diagnostic: vertical T / theta profile at the domain mid-column (x ~ 5.1 m),
comparing our model against permaFoam at each saved checkpoint.

Reads output/Comparision_permaFoam/permaFoam_omega50.pkl and
output/permaFoam_demoCase_reference/parsed_fields.pkl; writes
docu/Figures/permaFoam_midcolumn_profile.png
"""

from pathlib import Path
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OURS_PKL = ROOT / "output/Comparision_permaFoam/permaFoam_omega50.pkl"
PF_PKL = ROOT / "output/permaFoam_demoCase_reference/parsed_fields.pkl"
OUT_PNG = ROOT / "docu/Figures/permaFoam_midcolumn_profile.png"

BLUE = "#2a78d6"
ORANGE = "#eb6834"

PF_TIMES = [657000, 1314000, 1971000, 2628000]


def main():
    with open(OURS_PKL, "rb") as f:
        ours = pkl.load(f)
    with open(PF_PKL, "rb") as f:
        pf = pkl.load(f)

    Z_ours = np.array(ours["Z"])
    Z_pf = np.array(pf["Z"])
    nx_ours = Z_ours.shape[1]
    nx_pf = Z_pf.shape[1]
    mid_ours = nx_ours // 2
    mid_pf = nx_pf // 2

    T_total = [np.array(t) for t in ours["T_total"]]
    theta_l_total = [np.array(t) for t in ours["theta_l_total"]]
    theta_i_total = [np.array(t) for t in ours["theta_i_total"]]
    save_idx_map = {657000: 1, 1314000: 2, 1971000: 3, 2628000: 4}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row")

    for col, t in enumerate(PF_TIMES):
        i = save_idx_map[t]
        z_o = Z_ours[:, mid_ours]
        T_o = T_total[i][:, mid_ours]
        thl_o = theta_l_total[i][:, mid_ours]
        thi_o = theta_i_total[i][:, mid_ours]

        fl = pf["fields"][t]
        z_p = Z_pf[:, mid_pf]
        T_p = np.array(fl["T_C"])[:, mid_pf]
        thl_p = np.array(fl["thetal"])[:, mid_pf]
        thg_p = np.array(fl["thetag"])[:, mid_pf]
        thi_p = thg_p  # permaFoam's "thetag" is the ice-phase content in this naming (Cdg formulation)

        ax = axes[0, col]
        ax.plot(T_o, z_o, color=BLUE, lw=2, label="ours")
        ax.plot(T_p, z_p, color=ORANGE, lw=2, ls="--", label="permaFoam")
        ax.axvline(0, color="0.7", lw=1, ls=":")
        ax.set_title(f"t = {t/86400:.1f} d")
        ax.set_xlabel("T [°C]")
        if col == 0:
            ax.set_ylabel("z [m]")
            ax.legend(frameon=False, loc="lower right")

        ax = axes[1, col]
        ax.plot(thl_o, z_o, color=BLUE, lw=2, label=r"$\theta_l$ ours")
        ax.plot(thl_p, z_p, color=ORANGE, lw=2, ls="--", label=r"$\theta_l$ permaFoam")
        ax.plot(thi_o, z_o, color=BLUE, lw=1, alpha=0.5, label=r"$\theta_i$ ours")
        ax.plot(thi_p, z_p, color=ORANGE, lw=1, alpha=0.5, ls="--", label=r"$\theta_i$ permaFoam")
        ax.set_xlabel(r"$\theta$ [-]")
        if col == 0:
            ax.set_ylabel("z [m]")
            ax.legend(frameon=False, loc="center right", fontsize=8)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Mid-column (x = 5.1 m) vertical profile: ours vs permaFoam")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"saved {OUT_PNG}")
    return fig


if __name__ == "__main__":
    main()
