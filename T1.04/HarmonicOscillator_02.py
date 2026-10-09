"""
PHY1035 Harmonic Oscillator - rattling-jar version

Used for the final part of the practical, with a jar of loose metal pieces
in the box. It records the LED exactly as HarmonicOscillator_01.py does
(camera set-up, tracking and envelope fitting are imported from that script,
so any change made there applies here too), then:

  * asks how many pieces of metal are in the jar (written into the file names),
  * finds where the decay envelope falls through two amplitude levels inside
    the rattling regime (rattle_analysis.py),
  * saves a two-column file for Excel: time shifted so that t = 0 where the
    amplitude falls through the lower level, and ln(amplitude),
  * adds the decay time between the two levels to rattle_summary_<bench>.csv,
  * tells the student straight away if the run can't be used and why.

HarmonicOscillator_01.py and rattle_analysis.py must be in the same folder.

Run from Geany (F5) on the Raspberry Pi. To test without a camera:
    python3 HarmonicOscillator_02.py --simulate
"""

import os
import sys
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import HarmonicOscillator_01 as ho      # camera, tracking, spectrum, envelope
import rattle_analysis as ra

# =============================== Settings ===============================
# Amplitude levels for the decay-time measurement, in pixels at 640 px frame
# width (scaled automatically for other frame sizes). Both must lie in the
# rattling regime on every bench: on bench_00 the rattling stops at ~30-33 px.
A_HI_640 = 60.0
A_LO_640 = 40.0

RATTLE_ROOT = os.path.join(ho.DATA_ROOT, "Rattle data")


# ============================== Input ===================================
def ask_pieces():
    while True:
        text = input("\nHow many pieces of metal are in the jar? ").strip()
        try:
            n = int(text)
            if n >= 0:
                return n
        except ValueError:
            pass
        print("Please type a whole number, e.g. 10")


# ============================ Simulation ================================
def simulated_rattle(n_pieces=10):
    """Synthetic run: hand-held for 2 s, then a release into a decay that is
    strongly damped above a threshold amplitude and lightly damped below."""
    rng = np.random.default_rng(n_pieces)
    fps, f0 = ho.FPS, 4.88
    n = int(ho.ACQ_SECONDS * fps)
    t = np.cumsum(np.full(n, 1 / fps) + rng.normal(0, 0.002, n))
    t -= t[0]
    t_rel, A_th, gam_lo = 2.0, 31.0, 0.028
    gam_rattle = 0.02 * n_pieces                 # extra damping while rattling
    A = np.empty(n)
    A[0] = 85.0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if t[i] < t_rel:
            A[i] = A[0]
            continue
        g = gam_lo + (gam_rattle * (1 - A_th / A[i - 1]) * 6 if A[i - 1] > A_th else 0)
        A[i] = A[i - 1] * np.exp(-g * dt)
    y = 180 + np.where(t < t_rel, -A[0], A * np.cos(2 * np.pi * f0 * (t - t_rel) + np.pi))
    y += rng.normal(0, 0.3, n)
    x = 320 + rng.normal(0, 0.3, n)
    return t, x, y, "simulated"


# ============================== Saving ==================================
def save_rattle(t, x, y, n_pieces, env, t_shift, A_used, decay, status, fy, source):
    now = datetime.now()
    stamp = now.strftime("%H_%M_%S")
    base = f"{stamp}_{ho.BENCH_ID}_pieces_{n_pieces}"

    # Raw data, kept with the other runs (not needed by students)
    raw_folder = os.path.join(ho.DATA_ROOT, now.strftime("%Y_%m_%d"))
    os.makedirs(raw_folder, exist_ok=True)
    with open(os.path.join(raw_folder, base + "_positions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "x_horizontal_px", "y_vertical_px"])
        for row in zip(t, x, y):
            w.writerow([f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}"])
    with open(os.path.join(raw_folder, base + "_envelope.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "amplitude_vertical_px", "amplitude_uncertainty_px",
                    "ln_amplitude", "ln_amplitude_uncertainty",
                    "local_frequency_Hz", "points_in_window"])
        for r in env:
            w.writerow([f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.3f}",
                        f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.4f}", int(r[6])])

    # Two-column file for the students
    rattle_file = None
    if t_shift is not None:
        folder = os.path.join(RATTLE_ROOT, now.strftime("%Y_%m_%d"))
        os.makedirs(folder, exist_ok=True)
        rattle_file = os.path.join(folder, base + "_rattle.csv")
        with open(rattle_file, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_shifted_s", "ln_amplitude"])
            for ts, A in zip(t_shift, A_used):
                w.writerow([f"{ts:.3f}", f"{np.log(A):.4f}"])

    # One line per run
    summary_file = os.path.join(RATTLE_ROOT, f"rattle_summary_{ho.BENCH_ID}.csv")
    os.makedirs(RATTLE_ROOT, exist_ok=True)
    new = not os.path.exists(summary_file)
    with open(summary_file, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "time", "bench", "pieces", "decay_time_s",
                        "start_amplitude_px", "f_vertical_Hz", "status",
                        "timestamps", "rattle_file"])
        w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), ho.BENCH_ID,
                    n_pieces, f"{decay:.3f}" if decay is not None else "",
                    f"{A_used[0]:.1f}" if len(A_used) else "", f"{fy:.4f}", status,
                    source, os.path.basename(rattle_file) if rattle_file else ""])
    return rattle_file, summary_file


# =============================== Plot ===================================
def plot(t, y, env, t_shift, A_used, a_hi, a_lo, decay, n_pieces):
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    ax_t.plot(t, y - np.nanmean(y), ":r.", ms=3)
    if len(env):
        ax_t.errorbar(env[:, 0], env[:, 1], env[:, 2], fmt="k_", ms=10,
                      label="envelope (+/- amplitude)")
        ax_t.errorbar(env[:, 0], -env[:, 1], env[:, 2], fmt="k_", ms=10)
        ax_t.legend(loc="upper right", fontsize=8)
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Vertical position (px)")
    ax_t.set_title(f"Position of the LED ({n_pieces} pieces)")

    if t_shift is not None:
        ax_s.plot(t_shift, np.log(A_used), "ko-", ms=4, lw=0.8)
        for lvl, lab in ((a_hi, "upper level"), (a_lo, "lower level")):
            ax_s.axhline(np.log(lvl), color="b", ls="--", lw=1)
            ax_s.text(t_shift[-1], np.log(lvl),
                      f" {lab}", va="bottom", ha="right", color="b", fontsize=8)
        ax_s.axvline(0, color="grey", lw=0.8)
        title = (f"Decay time between levels: {decay:.3f} s" if decay is not None
                 else "No decay time (see message in the command window)")
    else:
        ax_s.text(0.5, 0.5, "Run could not be aligned\n(see message in the command window)",
                  ha="center", va="center", transform=ax_s.transAxes)
        title = "Not aligned"
    ax_s.set_xlabel("Shifted time (s)")
    ax_s.set_ylabel("ln(amplitude / px)")
    ax_s.set_title(title)
    fig.suptitle("Close this window to finish")
    fig.tight_layout()
    plt.show()


# ================================ Main ==================================
def main():
    plt.close("all")
    if ho.SIMULATE:
        n_pieces = 10
        t, x, y, source = simulated_rattle(n_pieces)
    else:
        cam = ho.open_camera()
        try:
            ho.preview(cam)
            n_pieces = ask_pieces()
            print("\n\nSTART\n")
            t, x, y, source = ho.acquire(cam)
        finally:
            cam.release()

    lost = int(np.sum(~np.isfinite(y)))
    if np.sum(np.isfinite(y)) < 10:
        sys.exit("\nThe LED was not found in the recording. Check it is on and in view.")

    f, amp_y = ho.spectrum(t, y)
    fy, _ = ho.find_peak(f, amp_y)
    env = ho.envelope(t, y, fy)

    scale = ho.FRAME_W / 640
    a_hi, a_lo = A_HI_640 * scale, A_LO_640 * scale
    if len(env):
        t_shift, A_used, decay, status = ra.rattle_analysis(env[:, 0], env[:, 1], a_hi, a_lo)
    else:
        t_shift, A_used, decay, status = None, np.array([]), None, \
            "No oscillation found - repeat the run."

    rattle_file, summary_file = save_rattle(t, x, y, n_pieces, env, t_shift, A_used,
                                            decay, status, fy, source)

    rate = (len(t) - 1) / t[-1]
    print(f"\nRecorded {t[-1]:.1f} s at {rate:.1f} frames/s (timestamps from {source})")
    print(f"Pieces of metal: {n_pieces}    Frequency: {fy:.3f} Hz")
    if lost > 0.02 * len(t):
        print(f"Warning: LED lost in {lost} frames. Check it stays in view.")
    if rate < 0.8 * ho.FPS:
        print(f"Warning: frame rate is low ({rate:.1f} /s).")
    print("\n" + "=" * 60)
    if status == "OK":
        print(f"  Decay time ({a_hi:.0f} -> {a_lo:.0f} px): {decay:.3f} s")
    else:
        print(f"  {status}")
    print("=" * 60)
    if rattle_file:
        print(f"\nFile for Excel:\n  {rattle_file}")
    print(f"Summary of all runs:\n  {summary_file}")

    plot(t, y, env, t_shift, A_used, a_hi, a_lo, decay, n_pieces)


if __name__ == "__main__":
    main()
