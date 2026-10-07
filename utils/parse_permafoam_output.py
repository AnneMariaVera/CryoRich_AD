"""
Parse raw permaFoam OpenFOAM output (the numbered time-step field directories, e.g.
<case_dir>/657000/T) into plain (nz, nx) numpy arrays for comparison against our own
simulation output.

Fully self-adapting to whatever case is passed in -- no slope, mesh resolution, or
checkpoint-time constants to hand-edit when the case changes (this file has been re-edited by
hand for three different slopes and two different mesh resolutions across one session; this
version shouldn't need it again):

  - Available checkpoint times are discovered from the case's own numbered subdirectories, not a
    hardcoded TIMES list (adaptive timestepping means the actual output times rarely land on
    the exact quarter-month target anyway).
  - The mesh's cell counts (nx, nz) are read off the "top" and "downSlope" boundary patches'
    own per-face list lengths in a reference timestep, not hardcoded.
  - The cell-center grid (X, Z) is *not* reconstructed from a hand-transcribed blockMeshDict
    (which silently goes stale the moment the case's slope or domain shape changes). Instead,
    Z is recovered exactly from the solved data itself: permaFoam's own solver computes
    `H = psi + z` every step (see the solver source, "Storage of the field of results..."),
    so simply `Z = H - psi`, cell-for-cell, on any reference timestep. X only needs xtot plus
    the now-adaptive nx, since none of these cases have graded/sheared x-spacing.
  - Verified against expected_postProcessing/internal/0/volFieldValue.dat (domain-mean
    T/thetal/thetag match to <0.1%) for the original 20%-slope case, confirming the assumed
    OpenFOAM cell ordering (x fastest, z slowest) is correct; that ordering is a mesh-writer
    convention, not case-specific, so it's left hardcoded.

Usage (from evaluate_validation.ipynb, or any notebook in this folder):
    from parse_permafoam_output import parse_permafoam_output

    pf = parse_permafoam_output(case_dir="/usr/lib/openfoam/.../validationCase")
    pf = parse_permafoam_output(case_dir=..., out_file="parsed_fields.pkl", xtot=10.0)

Usage (from a terminal):
    python parse_permafoam_output.py --case-dir /usr/lib/openfoam/.../validationCase \
                                     --out parsed_fields.pkl
"""
import re
import os
import numpy as np
import pickle as pkl
from pathlib import Path

def discover_times(case_dir):
    """Every integer-named subdirectory of case_dir is a checkpoint OpenFOAM wrote -- t=0 (the IC)
    plus whatever real times the adaptive timestepper actually landed on for the rest, which
    need not be round numbers (e.g. 1970907 instead of a requested 1971000)."""
    times = sorted(int(d) for d in os.listdir(case_dir) if re.fullmatch(r"\d+", d))
    assert times, f"no numbered checkpoint directories found under {case_dir}"
    return times


def parse_scalar_field(path, ncells):
    """Return a flat np.ndarray of the per-cell internalField values.

    A field that is genuinely uniform everywhere (e.g. thetag=0 once a case has fully thawed)
    still has a real value -- it's not missing data -- so it's broadcast to every cell rather
    than dropped, otherwise a comparison against a case that stays uniform at every checkpoint
    (as opposed to merely being uniform in the initial condition) would end up with the field
    absent from every timestep.
    """
    with open(path) as f:
        text = f.read()
    m_uniform = re.search(r'internalField\s+uniform\s+([\-\d.eE]+)\s*;', text)
    if m_uniform:
        return np.full(ncells, float(m_uniform.group(1)))
    m = re.search(
        r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(\s*\n(.*?)\n\)\s*;',
        text, re.S,
    )
    assert m, f"could not parse internalField in {path}"
    n = int(m.group(1))
    vals = np.array([float(x) for x in m.group(2).split()])
    assert len(vals) == n, f"{path}: expected {n} values, got {len(vals)}"
    return vals


def parse_boundary_field(text, patch_name):
    """Return a patch's own per-face nonuniform list (its face count == the mesh's cell count
    along whichever direction that patch spans), or None if the patch has no such list (e.g.
    a uniform/expression-only spec, as every patch has at t=0 before the solver has resolved
    it into concrete per-face values)."""
    m = re.search(re.escape(patch_name) + r"\s*\{(.*?)\n\s*\}", text, re.S)
    if not m:
        return None
    m2 = re.search(r"nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(\s*\n(.*?)\n\)\s*;", m.group(1), re.S)
    if not m2:
        return None
    return np.array([float(x) for x in m2.group(2).split()])


def infer_dims_and_Z(ref_path, x_patch="top", z_patch="downSlope"):
    """From one reference timestep's psi and H field files: nx (the x_patch's face count, since
    that patch spans the x direction), nz (the z_patch's face count, spanning z), and the full
    (nz, nx) cell-center Z array via Z = H - psi."""
    ref_path = str(ref_path)
    with open(ref_path) as f:
        psi_text = f.read()
    top = parse_boundary_field(psi_text, x_patch)
    downslope = parse_boundary_field(psi_text, z_patch)
    assert top is not None, f"{ref_path}: '{x_patch}' patch has no resolved per-face list -- pick a later checkpoint"
    assert downslope is not None, f"{ref_path}: '{z_patch}' patch has no resolved per-face list -- pick a later checkpoint"
    nx, nz = len(top), len(downslope)

    ncells = nx * nz
    psi = parse_scalar_field(ref_path, ncells).reshape(nz, nx)
    H = parse_scalar_field(ref_path.replace("/psi", "/H"), ncells).reshape(nz, nx)
    Z = H - psi
    return nx, nz, Z


def build_X(nx, nz, xtot):
    """Uniform cell-center x coordinates, tiled over all nz rows (no x-grading in these cases)."""
    i_edges = np.linspace(0, 1, nx + 1)
    x_centers = 0.5 * (i_edges[:-1] + i_edges[1:]) * xtot
    return np.tile(x_centers, (nz, 1))


def find_geometry_reference(case_dir, times, x_patch="top", z_patch="downSlope"):
    """Pick the latest checkpoint that has resolved (non-uniform) boundary/field data to infer
    the mesh from -- t=0 (the IC) is often uniform/expression-only, so search from the end.

    Returns (nx, nz, Z, ref_t)."""
    for t in reversed(times):
        p = os.path.join(str(case_dir), str(t), "psi")
        if not os.path.isfile(p) or not os.path.isfile(p.replace("/psi", "/H")):
            continue
        with open(p) as f:
            text = f.read()
        if parse_boundary_field(text, x_patch) is not None:
            nx, nz, Z = infer_dims_and_Z(p, x_patch=x_patch, z_patch=z_patch)
            return nx, nz, Z, t
    raise AssertionError(
        f"no checkpoint under {case_dir} had resolved boundary data to infer the mesh from"
    )


def parse_permafoam_output(case_dir, out_file,
                           fields, xtot,
                           x_patch="top", z_patch="downSlope", verbose=True):
    """Parse a permaFoam case directory into (nz, nx) arrays and (optionally) pickle the result.

    Parameters
    ----------
    case_dir : str or Path
        The OpenFOAM case directory holding the numbered checkpoint subdirectories.
    out_file : str, Path or None
        Where to write the pickle. None skips writing and only returns the dict.
        Relative paths resolve against the caller's working directory.
    fields : sequence of str
        Field file names to read from each checkpoint directory. Missing files are skipped.
    xtot : float
        Domain length in x [m], used to build the cell-center X grid.
    x_patch, z_patch : str
        Boundary patch names spanning x and z, used to infer nx and nz.
    verbose : bool
        Print the inferred mesh and per-checkpoint field list.

    Returns
    -------
    dict with keys "X", "Z" (each (nz, nx)), "times" (list of int, seconds) and
    "fields" ({time: {field_name: (nz, nx) array}}, plus a derived "T_C" = T - 273.15
    wherever "T" was present).
    """
    case_dir = Path(case_dir)
    times = discover_times(case_dir)
    nx, nz, Z, ref_t = find_geometry_reference(case_dir, times, x_patch=x_patch, z_patch=z_patch)
    X = build_X(nx, nz, xtot)
    ncells = nx * nz

    out = {"X": X, "Z": Z, "times": times, "fields": {}}

    for t in times:
        out["fields"][t] = {}
        for field in fields:
            path = case_dir / str(t) / field
            if not path.is_file():
                continue
            vals = parse_scalar_field(path, ncells)
            # OpenFOAM single-hex-block cell ordering: x (nx) fastest, then y(=1), then z (nz) slowest
            out["fields"][t][field] = vals.reshape(nz, nx)
        if "T" in out["fields"][t]:
            out["fields"][t]["T_C"] = out["fields"][t]["T"] - 273.15

    if verbose:
        print(f"inferred mesh: {nx} x {nz} cells, geometry from t={ref_t}")

    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            pkl.dump(out, f)
        if verbose:
            print("saved:", out_file)

    if verbose:
        for t in times:
            print(t, "fields:", list(out["fields"][t].keys()))

    return out
