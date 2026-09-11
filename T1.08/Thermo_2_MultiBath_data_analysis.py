# -*- coding: utf-8 -*-
"""
analyse_absolute_zero.py

Quick-look analysis of the multi-bath ideal-gas / absolute-zero experiment.
Designed to run on the RPi-400 with only numpy + matplotlib.

What it does, for each run (*_full.csv produced by Thermo_2_MultiBath.py):
  - extracts equilibrium plateaus from the State column
  - computes per-plateau mean T, mean p, standard errors, and the
    within-plateau pressure drift dp/dt (the leak fingerprint)
  - fits p = m*T + c (unweighted OLS -- the same line LINEST gives)
  - reports absolute zero x0 = -c/m with the CORRECT inverse-prediction
    uncertainty (carries the c-m covariance; equivalent to full propagation),
    plus R^2 and chi^2/dof about the fit
Across runs it reports the run-to-run scatter, an (internal) weighted mean,
and whether the runs agree within their own quoted errors.

Usage:
    python3 analyse_absolute_zero.py                      # scan default folders
    python3 analyse_absolute_zero.py run1.csv run2.csv    # specific files
    python3 analyse_absolute_zero.py /path/to/data_dir    # scan a directory

Nothing here is meant for the students -- it's your quick-look tool.
"""

import sys
import os
import glob
import csv

import numpy as np
import matplotlib
# Use a non-interactive backend if there's no display (e.g. over SSH),
# otherwise let matplotlib pick the interactive one so plt.show() works.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
#  CONFIG
# =============================================================================
PLATEAU_STATES     = ('HOLDING', 'READY')  # samples that count as equilibrium
MIN_PLATEAU_SAMPLES = 10      # ignore equilibrium runs shorter than this
MERGE_GAP_S         = 5.0     # merge equilibrium runs separated by < this (heals flicker)
LIT_ABS_ZERO        = -273.15 # literature value, deg C, for comparison
OUTPUT_DIR          = "analysis_output"
SHOW_PLOTS          = True    # also pop up windows if a display is available
# =============================================================================


# ---------------------------------------------------------------- data loading
def load_run(csv_path):
    """Read a *_full.csv into arrays. Returns dict of numpy arrays."""
    cols = {"Time_s": [], "Temperature_C": [], "Pressure_Pa": [],
            "Volume_proxy": [], "State": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in cols:
                cols[key].append(row.get(key, ""))
    t = np.array(cols["Time_s"], dtype=float)
    T = np.array(cols["Temperature_C"], dtype=float)
    P = np.array(cols["Pressure_Pa"], dtype=float)
    V = np.array(cols["Volume_proxy"], dtype=float)
    st = np.array([s.strip() for s in cols["State"]])
    return dict(t=t, T=T, P=P, V=V, st=st)


# ------------------------------------------------------------ plateau extraction
def contiguous_runs(mask):
    """(start, stop_inclusive) index pairs for each contiguous True run."""
    mask = np.asarray(mask, dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask, [0]))))
    starts, stops = edges[0::2], edges[1::2] - 1
    return list(zip(starts, stops))


def extract_plateaus(run):
    t, T, P, st = run["t"], run["T"], run["P"], run["st"]
    eq = np.isin(st, PLATEAU_STATES)
    runs = contiguous_runs(eq)

    # merge runs separated by only a short gap (brief SETTLING flicker mid-plateau)
    merged = []
    for s, e in runs:
        if merged and (t[s] - t[merged[-1][1]]) < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    plats = []
    for s, e in merged:
        if (e - s + 1) < MIN_PLATEAU_SAMPLES:
            continue
        sl = slice(s, e + 1)
        tt, TT, PP = t[sl], T[sl], P[sl]
        dpdt = np.polyfit(tt - tt[0], PP, 1)[0] if (tt[-1] - tt[0]) > 0 else np.nan
        plats.append(dict(
            n=len(TT), t0=tt[0], t1=tt[-1],
            Tmean=TT.mean(), Pmean=PP.mean(),
            seT=TT.std(ddof=1) / np.sqrt(len(TT)),
            seP=PP.std(ddof=1) / np.sqrt(len(PP)),
            dpdt=dpdt))
    return plats


# ------------------------------------------------------------------------ fitting
def fit_absolute_zero(Tm, Pm, seP, seT):
    """OLS fit + absolute zero with the inverse-prediction (covariance-correct) SE."""
    n = len(Tm)
    xbar = Tm.mean()
    Sxx = np.sum((Tm - xbar) ** 2)
    m, c = np.polyfit(Tm, Pm, 1)
    resid = Pm - (m * Tm + c)
    s2 = np.sum(resid ** 2) / (n - 2)
    s = np.sqrt(s2)

    Vm = s2 / Sxx
    Vc = s2 * np.sum(Tm ** 2) / (n * Sxx)
    Cov = -s2 * xbar / Sxx

    x0 = -c / m
    # inverse-prediction SE (identical to full propagation incl. covariance)
    sx0 = (s / abs(m)) * np.sqrt(1.0 / n + (x0 - xbar) ** 2 / Sxx)
    # naive eq.(5) without covariance -- reported only to show the difference
    sx0_naive = np.sqrt((1 / m ** 2) * Vc + (c ** 2 / m ** 4) * Vm)

    ss_tot = np.sum((Pm - Pm.mean()) ** 2)
    R2 = 1 - np.sum(resid ** 2) / ss_tot

    sigP_eff = np.sqrt(seP ** 2 + (m * seT) ** 2)
    chi2 = np.sum((resid / sigP_eff) ** 2)
    chi2dof = chi2 / (n - 2)

    return dict(n=n, m=m, sm=np.sqrt(Vm), c=c, sc=np.sqrt(Vc), s=s,
                x0=x0, sx0=sx0, sx0_naive=sx0_naive, R2=R2,
                chi2dof=chi2dof, resid=resid, xbar=xbar)


# ------------------------------------------------------------------- per-run plot
def plot_run(name, plats, fit, out_png):
    Tm = np.array([p["Tmean"] for p in plats])
    Pm = np.array([p["Pmean"] for p in plats])
    seP = np.array([p["seP"] for p in plats])
    order = np.argsort([p["t0"] for p in plats])
    direction = np.zeros(len(plats))          # up/down for residual colouring
    for k in range(1, len(order)):
        direction[order[k]] = np.sign(Tm[order[k]] - Tm[order[k - 1]])
    colmap = {1.0: "#d62728", -1.0: "#1f77b4", 0.0: "grey"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    xx = np.linspace(LIT_ABS_ZERO - 5, Tm.max() + 10, 100)
    ax[0].plot(xx, fit["m"] * xx + fit["c"], "-", color="#1f77b4", lw=1.4)
    ax[0].errorbar(Tm, Pm, yerr=seP, fmt="o", color="k", ms=5, capsize=3)
    ax[0].axhline(0, color="grey", ls=":", lw=0.8)
    ax[0].axvline(LIT_ABS_ZERO, color="green", ls="--", lw=1, label="-273.15")
    ax[0].errorbar([fit["x0"]], [0], xerr=fit["sx0"], fmt="s", color="red",
                   ms=8, capsize=4, label=f"{fit['x0']:.0f} +/- {fit['sx0']:.0f}")
    ax[0].set_xlabel("Temperature (C)"); ax[0].set_ylabel("Pressure (Pa)")
    ax[0].set_title("Extrapolation to p = 0"); ax[0].legend(fontsize=8)

    for i, p in enumerate(plats):
        ax[1].errorbar(p["Tmean"], fit["resid"][i], yerr=p["seP"], fmt="o",
                       ms=7, capsize=3, color=colmap[direction[i]])
    ax[1].axhline(0, color="grey", lw=1)
    ax[1].set_xlabel("Temperature (C)"); ax[1].set_ylabel("Residual (Pa)")
    ax[1].set_title(f"Residuals  (red=up, blue=down)\n"
                    f"R2={fit['R2']:.4f}   chi2/dof={fit['chi2dof']:.0f}")
    fig.suptitle(name, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=120)
    return fig


# --------------------------------------------------------------------- one run
def analyse_run(csv_path):
    name = os.path.basename(csv_path)
    run = load_run(csv_path)
    plats = extract_plateaus(run)
    if len(plats) < 3:
        print(f"  [skip] {name}: only {len(plats)} plateaus found (need >=3)")
        return None

    Tm = np.array([p["Tmean"] for p in plats])
    Pm = np.array([p["Pmean"] for p in plats])
    seT = np.array([p["seT"] for p in plats])
    seP = np.array([p["seP"] for p in plats])
    fit = fit_absolute_zero(Tm, Pm, seP, seT)

    print(f"\n=== {name} ===")
    print(f"  plateaus: {len(plats)}   T span: {Tm.min():.1f} to {Tm.max():.1f} C")
    print(f"  {'#':>2} {'T(C)':>7} {'P(Pa)':>10} {'seP':>7} {'dp/dt(Pa/s)':>12}")
    for i, p in enumerate(sorted(plats, key=lambda q: q["t0"])):
        print(f"  {i:>2d} {p['Tmean']:>7.2f} {p['Pmean']:>10.1f} "
              f"{p['seP']:>7.1f} {p['dpdt']:>+12.2f}")
    print(f"  gradient  m = {fit['m']:.2f} +/- {fit['sm']:.2f} Pa/C")
    print(f"  intercept c = {fit['c']:.1f} +/- {fit['sc']:.1f} Pa")
    print(f"  R^2 = {fit['R2']:.4f}    chi^2/dof = {fit['chi2dof']:.1f}"
          f"  ({'OK' if fit['chi2dof'] < 3 else 'systematic present'})")
    print(f"  ABSOLUTE ZERO = {fit['x0']:.2f} +/- {fit['sx0']:.2f} C"
          f"   (naive eq.5 would say +/- {fit['sx0_naive']:.2f})")
    dev = fit["x0"] - LIT_ABS_ZERO
    print(f"  deviation from -273.15 : {dev:+.2f} C "
          f"({abs(dev)/fit['sx0']:.1f} sigma)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png = os.path.join(OUTPUT_DIR, name.replace(".csv", "") + "_fit.png")
    plot_run(name, plats, fit, png)
    return dict(name=name, x0=fit["x0"], sx0=fit["sx0"],
                chi2dof=fit["chi2dof"], R2=fit["R2"], m=fit["m"])


# ------------------------------------------------------------------- many runs
def summarise_runs(results):
    if len(results) < 2:
        return
    names = [r["name"] for r in results]
    x0 = np.array([r["x0"] for r in results])
    sx0 = np.array([r["sx0"] for r in results])
    N = len(x0)

    unw_mean = x0.mean()
    unw_std = x0.std(ddof=1)               # run-to-run scatter (external)
    unw_sem = unw_std / np.sqrt(N)

    w = 1.0 / sx0 ** 2                      # internal-error weighting
    wmean = np.sum(w * x0) / np.sum(w)
    wmean_err = 1.0 / np.sqrt(np.sum(w))
    chi2_runs = np.sum(((x0 - wmean) / sx0) ** 2) / (N - 1)

    print("\n" + "=" * 60)
    print("RUN-TO-RUN SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']:<32} x0 = {r['x0']:7.1f} +/- {r['sx0']:4.1f} C"
              f"   chi2/dof={r['chi2dof']:.0f}")
    print("-" * 60)
    print(f"  unweighted mean : {unw_mean:.1f} C   "
          f"(scatter {unw_std:.1f}, SEM {unw_sem:.1f})")
    print(f"  weighted mean   : {wmean:.1f} +/- {wmean_err:.1f} C (internal)")
    print(f"  runs agree within quoted errors? chi^2/dof = {chi2_runs:.1f} "
          f"({'yes' if chi2_runs < 2 else 'NO -- errors underestimated'})")
    if chi2_runs > 1:
        print(f"  -> inflated weighted error: "
              f"+/- {wmean_err * np.sqrt(chi2_runs):.1f} C")
    print(f"  literature      : {LIT_ABS_ZERO} C")

    # cross-run figure
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(range(1, N + 1), x0, yerr=sx0, fmt="o", ms=7, capsize=4, color="k")
    ax.axhline(LIT_ABS_ZERO, color="green", ls="--", label="-273.15")
    ax.axhline(wmean, color="red", ls="-", lw=1, label=f"wtd mean {wmean:.0f}")
    ax.fill_between([0.5, N + 0.5],
                    wmean - wmean_err * max(1, np.sqrt(chi2_runs)),
                    wmean + wmean_err * max(1, np.sqrt(chi2_runs)),
                    color="red", alpha=0.15)
    ax.set_xticks(range(1, N + 1))
    ax.set_xticklabels([n.replace("_full.csv", "")[-8:] for n in names],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Absolute zero estimate (C)")
    ax.set_title("Run-to-run variation")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "run_to_run.png"), dpi=120)


# ------------------------------------------------------------------------- main
def find_csvs(paths):
    if not paths:
        # default: look where Thermo_2_MultiBath.py saves, then anywhere below CWD
        paths = ["Pressure vs Temperature data", "."]
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*_full.csv"), recursive=True)
        elif p.endswith(".csv"):
            files.append(p)
    return sorted(set(files))


def main():
    files = find_csvs(sys.argv[1:])
    if not files:
        print("No *_full.csv files found. Pass a file or folder as an argument.")
        return
    print(f"Found {len(files)} run(s).")
    results = [r for r in (analyse_run(f) for f in files) if r]
    summarise_runs(results)
    print(f"\nFigures written to ./{OUTPUT_DIR}/")
    if SHOW_PLOTS and os.environ.get("DISPLAY"):
        plt.show()


if __name__ == "__main__":
    main()
