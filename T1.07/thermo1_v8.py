# -*- coding: utf-8 -*-
"""
thermo1_v9.py  -  PHY1035 Ideal Gas Law 1 (T1.07)
Original: bs426, March 2025.  v5: October 2026 (leak correction).
v6: one green band per hold, one click per hold.
v8: 8 s holds; uncertainty on every leak-corrected pressure; fitted volumes
    no longer shown to students.
v9: snow leopard splash screen showing the script version at start-up.

Records gas pressure and temperature while the syringe compresses the sealed
apparatus in fixed volume steps. The user then selects each pressure plateau
on a pressure-vs-time plot; each plateau is averaged over a short time window
around the click.

Student procedure
  1. Syringe at 50 mL, all valves closed to atmosphere.
  2. Run this script. Live plots of pressure and temperature vs time appear.
  3. Follow the prompt above the live plot. While the background is pale RED,
     hold the syringe still (a countdown shows how long is left). When it
     turns pale GREEN, push the syringe in by 5 mL in one smooth movement.
     The script detects the compression and turns the background red again.
  4. After the 10th compression's hold, the prompt turns GREEN again and asks
     you to PULL the syringe back out to 50 mL. Do it in one smooth movement,
     then hold still while the prompt counts down (30 s). The pressure should
     return to its starting value; any shortfall is gas that leaked out, and
     is used to correct the pressures for the leak. The live plot then closes
     by itself.
  5. On the analysis plot, pale green bands show where the pressure has
     settled enough to be measured - one band per hold. Left-click once
     inside each band (11 in total, any order). The samples averaged for each click are outlined by a
     rectangle labelled with its volume change; a red outline means the
     pressure was still changing inside that window.
     Right-click undoes the last selection.
  6. Press Enter (or close the window) to save.

Output, in "<DATA_FOLDER>/<timestamp>/" next to this script:
  <timestamp> pressures and delta volumes.csv  columns A-C as the original
                                               script (existing Excel
                                               templates still work), then
                                               D: leak-corrected pressure
                                               E: its uncertainty (1 s.d.)
  <timestamp> plateau details.csv              per plateau: mean, SD, SEM,
                                               drift, temperature, window
  <timestamp> raw data.csv                     every sample recorded
  <timestamp> plateaus.png                     analysis plot with windows
  <timestamp> leak correction.txt              gas lost, leak conductance,
                                               size of the correction
  <timestamp> leak correction.png              whole run, with the return to
                                               the starting pressure

Leak correction (v5)
  Gas escapes at a rate proportional to the overpressure:
      d(pV)/dt = -C (p - p0),   p0 = starting (atmospheric) pressure.
  After the final pull-back the system is at its starting volume again, so
      C = V1 (p0 - p_return) / integral of (p - p0) dt,
  with p_return the pressure the system settles to (fitted asymptote, with
  the small inflow of air while it is below p0 removed). The fraction of the
  starting gas still present at time t is then
      n(t)/n0 = 1 - C * integral_0^t (p - p0) dt / (p0 V1),
  and each sample is corrected to p / (n/n0) before the plateau averages are
  formed. V1 is iterated with the straight-line fit. The uncertainty from the
  correction is propagated from the asymptote fit plus a 0.1 K allowance for
  temperature drift.

Uncertainty on each leak-corrected pressure (column E), 1 standard deviation,
three independent parts added in quadrature:
  - random:     standard error of the mean of the samples in the window
  - settling:   pressure drift across the window / sqrt(12)  (the "true"
                plateau value could be anywhere in that range)
  - leak:       change in that plateau's corrected pressure when the leak
                conductance C is moved by +/- its uncertainty. This part is
                common to all plateaus (fully correlated), not random.
Not included: the absolute calibration of the pressure sensor and of the
syringe graduations (systematic, the same for every point).

Changes from the original
  - Serial reader re-synchronises on the pressure line, so a partial first
    line after the buffer reset can no longer shift the
    distance/pressure/temperature order for the whole run. Malformed lines
    are skipped. Temperature accepted in deg C or K (stored in K).
  - Live plot is pressure and temperature vs TIME, redrawn at a fixed rate
    with the serial buffer drained every loop, so the display no longer lags
    behind the apparatus as the run gets longer.
  - Analysis averages raw samples over a window around each click, giving a
    mean and uncertainty per plateau instead of a cursor level placed by eye.
    Settled regions are shaded to show where it is safe to click.
  - Plateaus are sorted by pressure, so click order no longer matters.
    A warning is printed if pressure order and time order disagree.
  - Output folder is set relative to the script, not Thonny's working dir.
  - Fewer than 11 selections no longer crashes the save: details and raw
    data are always written.
  - v5: final pull-back to 50 mL, automatic end of recording, and a leak
    correction applied to every plateau (extra column in both csv files).
  - v8: column E (uncertainty of the leak-corrected pressure); default hold
    8 s; the fitted V1 is no longer printed or saved (SHOW_V1 = False);
    the "slow recovery" note only appears when the extrapolation is large.
  - v9: splash screen (snow_leopard.jpg, any case of extension, in the
    same folder as this script) showing the version for SPLASH_S seconds
    before recording starts. A missing picture only prints a warning.
  - v6: the detected compression times split the run into 11 holds. Only
    the longest settled region in each hold is shown as a green band (a
    nudge of the plunger part-way through a hold used to split it into two
    bands), and only one click is accepted per hold.
"""

import csv
import os
import re
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Cursor

plt.style.use('dark_background')

# ============================== Settings ===================================

VERSION = "thermo1_v9"       # shown on the splash screen
SPLASH_IMAGE = "snow_leopard.jpg"   # in the same folder as this script
SPLASH_S = 3.0               # how long the splash screen stays up (s)

N_STEPS = 10                 # number of syringe compressions
STEP_ML = 5.0                # volume change per compression (mL)
N_POINTS = N_STEPS + 1       # plateaus to select (including 0 mL)

WINDOW_HALF_S = 2.5          # plateau averaging window: click time +/- this (s)
                             # (~10 samples at the Arduino's 2 Hz output)
MIN_SAMPLES = 5              # minimum samples inside a plateau window
SETTLED_FRACTION = 0.10      # a window counts as settled if the pressure
                             # changes by less than this fraction of a typical
                             # compression step across it. Used for both the
                             # green bands and the red/green click outlines.

HOLD_S = 8.0                 # how long students hold each volume before the
                             # live plot tells them to compress again (s)
V_GUESS_ML = 400.0           # rough system volume, used only to predict the
                             # size of a 5 mL pressure step for step detection
RETURN_HOLD_S = 30.0         # hold after pulling the syringe back to 50 mL
SHOW_V1 = False              # print/save the fitted V1 (raw and leak-corrected).
                             # Useful for validation; set False before students
                             # use the script if they should fit it themselves.
HOLD_COLOUR = (1.0, 0.55, 0.55, 0.28)       # pale red tint: hold steady
COMPRESS_COLOUR = (0.55, 1.0, 0.55, 0.28)   # pale green tint: compress now
REFRESH_HZ = 4               # live plot redraw rate
MARKER_SIZE = 10             # scatter marker size (points^2)
CMAP = 'jet'                 # marker colour map (colour = pressure)
DATA_FOLDER = "Volume vs Pressure"   # matches Task 1.6 in the handout
TOOLS_PATH = "/home/dorkmaster/Desktop/PHY1035/Tools"

# Serial protocol
#   False: three lines per sample - distance, pressure, temperature
#          (current Arduino sketch). The reader identifies the pressure line
#          by its magnitude and takes the next line as temperature.
#   True:  one line per sample - "distance,pressure,temperature"
SINGLE_LINE_PROTOCOL = False

P_RANGE = (2e4, 3e5)         # plausible pressure (Pa), used to identify lines
T_RANGE = (200.0, 400.0)     # plausible temperature (K)
T_RANGE_C = (-40.0, 150.0)   # plausible temperature (deg C); converted to K


# =========================== Serial reading ================================

def extract_floats(text):
    """All numbers in a string (integers, decimals, scientific notation)."""
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    return [float(n) for n in numbers]


def in_range(v, rng):
    return rng[0] < v < rng[1]


def to_kelvin(v):
    """Temperature in K if v looks like K or deg C, else None."""
    if in_range(v, T_RANGE):
        return v
    if in_range(v, T_RANGE_C):
        return v + 273.15
    return None


class SensorReader:
    """Turns the serial stream into (distance, pressure, temperature) samples.

    For the three-line protocol, a sample is emitted only when a line in the
    pressure range is followed immediately by a plausible temperature line;
    the line before the pressure line is taken as distance. Anything else is
    discarded, so the reader recovers from partial or garbled lines instead of
    silently mislabelling every value for the rest of the run.
    """

    def __init__(self, ser):
        self.ser = ser
        self.prev_value = np.nan     # value of the previous line (distance)
        self.pending = None          # (distance, pressure) awaiting temperature

    @staticmethod
    def _parse(raw):
        text = raw.decode('utf-8', errors='ignore').strip()
        vals = extract_floats(text)
        return vals if vals else None

    def feed(self, raw):
        """Process one raw line; return a sample tuple or None."""
        vals = self._parse(raw)
        if vals is None:
            return None

        if SINGLE_LINE_PROTOCOL:
            if len(vals) >= 3 and in_range(vals[1], P_RANGE):
                Tk = to_kelvin(vals[2])
                if Tk is not None:
                    return (vals[0], vals[1], Tk)
            return None

        v = float(np.mean(vals))
        sample = None
        if self.pending is not None:
            d, p = self.pending
            self.pending = None
            Tk = to_kelvin(v)
            if Tk is not None:
                sample = (d, p, Tk)
        if sample is None and in_range(v, P_RANGE):
            self.pending = (self.prev_value, v)
        self.prev_value = v
        return sample

    def read_available(self):
        """Drain everything currently waiting on the port."""
        samples = []
        while self.ser.in_waiting:
            s = self.feed(self.ser.readline())
            if s is not None:
                samples.append(s)
        return samples


# =============================== Helpers ===================================

def script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def safe_nanmedian(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def typical_step(p):
    """Typical pressure rise per compression, from the overall pressure range."""
    lo, hi = np.percentile(p, [1, 99])
    return max((hi - lo) / N_STEPS, 1.0)


def window_stats(t, p, tc):
    """Samples within +/-WINDOW_HALF_S of tc, and the drift across them."""
    mask = np.abs(t - tc) <= WINDOW_HALF_S
    n = int(mask.sum())
    if n < max(MIN_SAMPLES, 3):
        return mask, n, np.nan
    tw, pw = t[mask], p[mask]
    if tw[-1] - tw[0] <= 0:
        return mask, n, np.nan
    drift = float(np.polyfit(tw, pw, 1)[0] * (tw[-1] - tw[0]))
    return mask, n, drift


def settled_regions(t, p, min_width_s=0.5):
    """Time intervals where a click would give a settled averaging window.

    A click time is 'settled' if its window holds enough samples, the fitted
    drift across the window is below SETTLED_FRACTION of a typical step, and
    no single sample-to-sample jump inside the window exceeds that same limit
    (i.e. the window does not straddle a compression).
    Returns a list of (t_start, t_end, p_min, p_max) per region.
    """
    if len(t) < MIN_SAMPLES:
        return []
    limit = SETTLED_FRACTION * typical_step(p)
    ok = np.zeros(len(t), dtype=bool)
    for i, tc in enumerate(t):
        mask, n, drift = window_stats(t, p, tc)
        if n < MIN_SAMPLES or not np.isfinite(drift) or abs(drift) > limit:
            continue
        pw = p[mask]
        if len(pw) > 1 and np.max(np.abs(np.diff(pw))) > limit:
            continue
        ok[i] = True

    regions = []
    i = 0
    while i < len(t):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and ok[j + 1]:
            j += 1
        t0, t1 = t[i], t[j]
        if t1 - t0 >= min_width_s:
            seg = np.abs(t - 0.5 * (t0 + t1)) <= 0.5 * (t1 - t0) + WINDOW_HALF_S
            regions.append((float(t0), float(t1),
                            float(p[seg].min()), float(p[seg].max())))
        i = j + 1
    return regions


def show_splash():
    """Show the snow leopard and the script version for SPLASH_S seconds.

    Used as a quick visual version check on the lab Pis. Linux file names are
    case-sensitive, so any case of the extension (.jpg, .JPG, .jpeg, .png) is
    accepted. A missing or unreadable picture never stops the experiment.
    """
    folder = script_dir()
    stem = os.path.splitext(SPLASH_IMAGE)[0].lower()
    path = None
    try:
        for name in sorted(os.listdir(folder)):
            root, ext = os.path.splitext(name)
            if root.lower() == stem and ext.lower() in ('.jpg', '.jpeg',
                                                        '.png'):
                path = os.path.join(folder, name)
                break
    except OSError:
        pass
    if path is None:
        print(f"WARNING: splash picture '{SPLASH_IMAGE}' not found next to "
              f"this script ({VERSION}). Carrying on without it.")
        return
    try:
        img = plt.imread(path)
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(8, 8 * h / w + 0.8), dpi=100)
        ax = fig.add_axes((0, 0, 1, 8 * h / w / (8 * h / w + 0.8)))
        ax.imshow(img)
        ax.axis('off')
        fig.suptitle(f'PHY1035 Ideal Gas Law 1   -   {VERSION}', fontsize=18,
                     fontweight='bold', color='white', y=0.985)
        plt.show(block=False)
        plt.pause(SPLASH_S)
        plt.close(fig)
    except Exception as e:           # never let the splash stop a run
        print(f"WARNING: could not show splash picture ({e}).")
        plt.close('all')


# ============================ Live acquisition =============================

class StepTracker:
    """Tells students when to hold, compress and pull back, from the pressure.

    States
      'hold'      - hold the syringe still; lasts HOLD_S from the end of the
                    last compression (or from the first sample)
      'compress'  - hold time is up; compress by one step now
      'release'   - all compressions done; pull the syringe back to 50 mL
      'settle'    - hold still at 50 mL for RETURN_HOLD_S (leak measurement)
      'done'      - finished; recording stops automatically

    A compression is detected when the pressure rises more than a quarter of
    the expected step above the pre-compression baseline. It counts as
    finished once the pressure has stopped rising AND the total rise is at
    least 60% of the expected step (so a pause part-way through a push does
    not end it early). The hold timer restarts at that moment, even if the
    students compressed before being told to. The pull-back is detected the
    same way, as a fall of at least 60% of the total rise.
    """

    def __init__(self, n_steps=N_STEPS):
        self.n_steps = n_steps
        self.state = 'hold'
        self.count = 0               # completed compressions
        self.hold_start = None       # time the current hold began
        self.in_push = False
        self.in_release = False
        self.base = None             # baseline pressure before a movement
        self.p_start = None          # pressure before the first compression
        self.recent = []             # last few pressures during a hold
        self.prev_p = None
        self.step_times = []         # (start, end) of each compression
        self.release_times = None    # (start, end) of the pull-back
        self.settle_start = None

    def expected_step(self, p):
        v = V_GUESS_ML - self.count * STEP_ML
        return p * STEP_ML / max(v - STEP_ML, STEP_ML)

    def update(self, t, p):
        """Feed one sample. Returns True if the state changed."""
        old = (self.state, self.count, self.in_push, self.in_release)
        if self.hold_start is None:
            self.hold_start = t

        if self.in_push:
            rise = p - self.base
            still_rising = self.prev_p is not None and p - self.prev_p > 50.0
            if not still_rising and rise > 0.6 * self.expected_step(self.base):
                self.in_push = False
                self.count += 1
                self.step_times.append((self.push_start, t))
                self.hold_start = t
                self.state = 'hold'
                self.recent = [p]
        elif self.in_release:
            drop = self.base - p
            still_falling = self.prev_p is not None and self.prev_p - p > 50.0
            if not still_falling and drop > 0.6 * (self.base - self.p_start):
                self.in_release = False
                self.release_times = (self.release_start, t)
                self.settle_start = t
                self.state = 'settle'
        elif self.state not in ('settle', 'done'):
            if self.recent:
                self.base = float(np.median(self.recent[-3:]))
            if self.base is not None and self.count < self.n_steps and \
                    p - self.base > 0.25 * self.expected_step(self.base):
                if self.count == 0:
                    self.p_start = self.base
                self.in_push = True
                self.push_start = t
                self.recent = []
            elif self.base is not None and self.count >= self.n_steps and \
                    self.base - p > 0.15 * (self.base - self.p_start):
                self.in_release = True
                self.release_start = t
                self.recent = []
            else:
                self.recent = (self.recent + [p])[-5:]

        self.tick(t)
        self.prev_p = p
        return (self.state, self.count, self.in_push, self.in_release) != old

    def tick(self, t):
        """Advance the timers to time t (also called between samples)."""
        if self.in_push or self.in_release or self.hold_start is None:
            return
        if self.state == 'hold' and t - self.hold_start >= HOLD_S:
            self.state = 'release' if self.count >= self.n_steps else 'compress'
        elif self.state == 'settle' and t - self.settle_start >= RETURN_HOLD_S:
            self.state = 'done'

    def message(self, t):
        if self.in_push:
            return f'Compressing...   (step {self.count + 1}/{self.n_steps})'
        if self.in_release:
            return 'Pulling back...'
        if self.state == 'hold':
            left = max(0, int(np.ceil(HOLD_S - (t - self.hold_start))))
            return (f'HOLD STEADY  {left} s   '
                    f'({self.count}/{self.n_steps} compressions done)')
        if self.state == 'compress':
            return (f'COMPRESS BY {STEP_ML:g} mL NOW   '
                    f'(step {self.count + 1}/{self.n_steps})')
        if self.state == 'release':
            return 'PULL the syringe back OUT to 50 mL NOW'
        if self.state == 'settle':
            left = max(0, int(np.ceil(RETURN_HOLD_S - (t - self.settle_start))))
            return f'HOLD STILL at 50 mL  {left} s   (measuring the leak)'
        return 'FINISHED - analysing...'

    def colour(self):
        if self.state == 'done':
            return 'black'
        if self.state in ('compress', 'release') or self.in_push or \
                self.in_release:
            return COMPRESS_COLOUR
        return HOLD_COLOUR


def acquire(ser):
    """Live-plot p and T vs time until the run finishes (or the window is
    closed).

    Returns arrays (t, d, p, T) of raw samples and the StepTracker.
    """
    reader = SensorReader(ser)
    tracker = StepTracker()

    fig, (axp, axt) = plt.subplots(2, 1, sharex=True, figsize=(9, 8), dpi=100,
                                   gridspec_kw={'height_ratios': [3, 1]})
    status = fig.suptitle('Waiting for data...', fontsize=16,
                          fontweight='bold', color='white')
    scat_p = axp.scatter([], [], c=[], s=MARKER_SIZE, cmap=CMAP)
    scat_t = axt.scatter([], [], s=MARKER_SIZE, color='orange')
    axp.set_ylabel('Pressure (Pa)')
    axt.set_ylabel('Temperature (K)')
    axt.set_xlabel('Time (s)')
    for ax in (axp, axt):
        ax.grid(True, color='gray', lw=0.5, alpha=0.5)
    readout = axp.text(0.02, 0.95, 'Waiting for data...',
                       transform=axp.transAxes, va='top', ha='left',
                       fontsize=13,
                       bbox=dict(boxstyle='round', facecolor='black',
                                 alpha=0.7))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show(block=False)
    fig.canvas.flush_events()

    t_list, d_list, p_list, T_list = [], [], [], []
    t0 = time.time()
    last_draw = 0.0
    last_data = time.time()
    last_batch = 0.0
    warned = False

    try:
        while plt.fignum_exists(fig.number):
            new = reader.read_available()
            now = time.time()
            if new:
                # Spread each batch evenly between the previous batch and now,
                # rather than giving every sample in it the same timestamp.
                t_now = now - t0
                t_start = last_batch if t_list else t_now
                times = np.linspace(t_start, t_now, len(new) + 1)[1:]
                last_batch = t_now
                for ts, (d, p, T) in zip(times, new):
                    t_list.append(float(ts))
                    d_list.append(d)
                    p_list.append(p)
                    T_list.append(T)
                    tracker.update(float(ts), p)
                last_data = now
                warned = False
            elif not warned and now - last_data > 5:
                print("WARNING: no sensor data for 5 s - check the connection.")
                warned = True

            if len(t_list) > 1 and now - last_draw >= 1.0 / REFRESH_HZ:
                t = np.asarray(t_list)
                p = np.asarray(p_list)
                T = np.asarray(T_list)
                scat_p.set_offsets(np.column_stack([t, p]))
                scat_p.set_array(p)
                prng = max(p.max() - p.min(), 1.0)
                scat_p.set_clim(p.min() - 0.15 * prng, p.min() + prng)
                scat_t.set_offsets(np.column_stack([t, T]))
                axp.set_xlim(0, max(30.0, t[-1] * 1.05 + 2))
                pad = max(200.0, 0.05 * (p.max() - p.min()))
                axp.set_ylim(p.min() - pad, p.max() + pad)
                tpad = max(0.5, 0.1 * (T.max() - T.min()))
                axt.set_ylim(T.min() - tpad, T.max() + tpad)
                n = min(5, len(p))
                readout.set_text(f'p = {p[-n:].mean():.0f} Pa\n'
                                 f'T = {np.median(T[-n:]):.2f} K')
                # Hold / compress prompt. The hold countdown runs on the
                # latest sample time, so it updates even if data are batched.
                t_status = max(t[-1], now - t0)
                tracker.tick(t_status)
                face = tracker.colour()
                for ax in (axp, axt):
                    ax.set_facecolor(face)
                status.set_text(tracker.message(t_status))
                status.set_color('lime' if tracker.state == 'done' else
                                 'white')
                try:
                    fig.canvas.draw()
                except Exception:
                    break            # window closed mid-draw
                last_draw = now
                if tracker.state == 'done':
                    plt.pause(1.0)   # let students see FINISHED
                    break

            try:
                fig.canvas.flush_events()
            except Exception:
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:          # e.g. serial dropped out mid-run
        print(f"\nAcquisition stopped: {e}")

    plt.close(fig)
    return (np.asarray(t_list), np.asarray(d_list),
            np.asarray(p_list), np.asarray(T_list), tracker)


# ============================ Plateau selection ============================

class PlateauPicker:
    """Click-to-select pressure plateaus, each averaged over a time window."""

    def __init__(self, t, p, T, n_points=N_POINTS, step_times=None):
        self.t, self.p, self.T = t, p, T
        self.n_points = n_points
        # Holds from the detected compressions: hold k runs from the end of
        # compression k to the start of compression k+1 (None: no limits)
        if step_times:
            starts = [float(s) for s, e in step_times] + [float(t.max()) + 1.0]
            ends = [float(t.min()) - 1.0] + [float(e) for s, e in step_times]
            self.bounds = list(zip(ends, starts))
        else:
            self.bounds = None
        self.plateaus = []           # in the order they were added
        self.limit = SETTLED_FRACTION * typical_step(p)

        self.fig, self.axp = plt.subplots(figsize=(10, 8), dpi=100)
        ax = self.axp
        prng = p.max() - p.min()
        # Start the colour scale a little below the data so the lowest
        # pressures are a visible blue rather than near-black navy
        ax.scatter(t, p, c=p, s=MARKER_SIZE, cmap=CMAP, zorder=3,
                   vmin=p.min() - 0.15 * prng, vmax=p.max())
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Pressure (Pa)')
        ax.grid(True, color='gray', lw=0.5, alpha=0.5, zorder=0)

        # Fix the axis limits before adding bands, so they don't rescale it
        ax.set_ylim(p.min() - 0.08 * prng - 200, p.max() + 0.08 * prng + 200)
        ax.set_xlim(t.min() - 2, t.max() + 2)

        # Settled bands: extend well above and below the data in each band
        vpad = 0.35 * typical_step(p)
        self.regions = settled_regions(t, p)
        missing = []
        if self.bounds is not None:
            # Keep only the longest settled region in each hold
            kept = []
            for k, (lo, hi) in enumerate(self.bounds):
                cand = [r for r in self.regions
                        if lo <= 0.5 * (r[0] + r[1]) < hi]
                if cand:
                    kept.append(max(cand, key=lambda r: r[1] - r[0]))
                else:
                    missing.append(k)
            self.regions = kept
        for t0, t1, pmin, pmax in self.regions:
            ax.add_patch(Rectangle((t0, pmin - vpad), t1 - t0,
                                   (pmax - pmin) + 2 * vpad,
                                   facecolor='palegreen', edgecolor='none',
                                   alpha=0.3, zorder=1))
        for k in missing:
            print(f"Note: no settled band in hold {k + 1} "
                  f"({k * STEP_ML:g} mL) - the pressure never settled there. "
                  "Click on the flattest part of that plateau.")
        if self.bounds is None and len(self.regions) != n_points:
            print(f"Note: {len(self.regions)} settled regions found "
                  f"(expected {n_points}). Bands are a guide only - check the "
                  "data before clicking.")

        self.fig.tight_layout(rect=(0, 0, 1, 0.95))

        # Horizontal cursor to help line up clicks with the data
        self.cursor = Cursor(ax, useblit=True, horizOn=True, vertOn=False,
                             color='red', linewidth=1)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_title()

    # ---------------------------------------------------------------- events
    def on_click(self, event):
        if event.inaxes is not self.axp or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is not None and getattr(toolbar, 'mode', ''):
            return                   # zoom/pan active: ignore
        if event.button == 1:
            self.add_plateau(event.xdata)
        elif event.button == 3:
            self.undo()

    def on_key(self, event):
        if event.key == 'enter':
            plt.close(self.fig)

    # ------------------------------------------------------------- selection
    def add_plateau(self, tc):
        if len(self.plateaus) >= self.n_points:
            print(f"Already have {self.n_points} plateaus - right-click to "
                  "undo one first.")
            return
        mask, n, drift = window_stats(self.t, self.p, tc)
        if n < MIN_SAMPLES or not np.isfinite(drift):
            print(f"Only {n} samples within +/-{WINDOW_HALF_S} s of "
                  f"t = {tc:.1f} s - click on a plateau.")
            return
        tw, pw = self.t[mask], self.p[mask]
        lo, hi = float(tw[0]), float(tw[-1])
        hold = None
        if self.bounds is not None:
            inside = [k for k, (b0, b1) in enumerate(self.bounds)
                      if b0 <= tc < b1]
            if not inside:
                print("That is during a compression - click inside a green "
                      "band.")
                return
            hold = inside[0]
            if any(pl['hold'] == hold for pl in self.plateaus):
                print(f"You have already selected a plateau in hold {hold + 1} "
                      "- one click per band. Right-click to undo it first.")
                return
            # Keep the averaging window inside this hold
            b_lo, b_hi = self.bounds[hold]
            if lo < b_lo or hi > b_hi:
                print("That window runs into a compression - click nearer "
                      "the middle of the green band.")
                return
        for pl in self.plateaus:
            if lo <= pl['t_hi'] and hi >= pl['t_lo']:
                print("That window overlaps an existing plateau - "
                      "click further away or right-click to undo.")
                return

        mean = float(pw.mean())
        sd = float(pw.std(ddof=1))
        sem = sd / np.sqrt(n)
        jump = float(np.max(np.abs(np.diff(pw)))) if n > 1 else 0.0
        drifting = abs(drift) > self.limit or jump > self.limit

        # Outline exactly the samples being averaged, padded a little
        y0, y1 = self.axp.get_ylim()
        ypad = 0.01 * (y1 - y0)
        xpad = 0.02 * (hi - lo) + 0.05
        colour = 'red' if drifting else 'white'
        rect = Rectangle((lo - xpad, pw.min() - ypad),
                         (hi - lo) + 2 * xpad, (pw.max() - pw.min()) + 2 * ypad,
                         facecolor=colour, edgecolor=colour, alpha=0.35,
                         lw=1.5, zorder=4)
        self.axp.add_patch(rect)
        mline = self.axp.hlines(mean, lo, hi, colors=colour, lw=1.5, zorder=5)
        label = self.axp.text((lo + hi) / 2, pw.max() + 1.5 * ypad, '',
                              ha='center', va='bottom', fontsize=9,
                              color=colour, fontweight='bold', zorder=5)

        self.plateaus.append(dict(
            mean=mean, sd=sd, n=n, sem=sem, drift=drift, drifting=drifting,
            T=safe_nanmedian(self.T[mask]), hold=hold,
            t_lo=lo, t_hi=hi, t_mid=0.5 * (lo + hi),
            artists=[rect, mline, label], label=label))

        msg = (f"Plateau {len(self.plateaus)}: p = {mean:.0f} +/- {sem:.0f} Pa "
               f"(SD {sd:.0f} Pa, n = {n})")
        if drifting:
            msg += ("  ** pressure still changing in this window - "
                    "click nearer the middle of a green band **")
        print(msg)
        self.relabel()

    def undo(self):
        if not self.plateaus:
            return
        pl = self.plateaus.pop()
        for a in pl['artists']:
            a.remove()
        print(f"Removed plateau at p = {pl['mean']:.0f} Pa")
        self.relabel()

    def relabel(self):
        """Label each plateau with the volume change implied by its rank."""
        for k, pl in enumerate(sorted(self.plateaus, key=lambda q: q['mean'])):
            pl['label'].set_text(f"{k * STEP_ML:g} mL")
        self.update_title()
        self.fig.canvas.draw_idle()

    def update_title(self):
        n = len(self.plateaus)
        if n < self.n_points:
            txt = (f"Click once in each green band ({n}/{self.n_points})    "
                   "Right-click: undo    Enter: save")
            col = 'red'
        else:
            txt = (f"All {self.n_points} plateaus selected - press Enter to "
                   "save (right-click to undo)")
            col = 'lime'
        self.fig.suptitle(txt, fontsize=12, fontweight='bold', color=col)


# ============================ Leak correction ==============================

def cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def asymptote(tt, pp):
    """Pressure the system settles to after the pull-back.

    Fits p = p_inf + B1 exp(-t/tau1) [+ B2 exp(-t/tau2)]: a fast (gas) and an
    optional slow thermal recovery. The second term is used only if it lowers
    the residual variance by more than 10%. Returns p_inf, its standard
    error, and the slowest time constant used.
    """
    def best_fit(taus_list):
        best = None
        for taus in taus_list:
            X = np.column_stack([np.ones_like(tt)] +
                                [np.exp(-tt / tau) for tau in taus])
            c, *_ = np.linalg.lstsq(X, pp, rcond=None)
            r = float(np.sum((pp - X @ c) ** 2))
            if best is None or r < best[0]:
                best = (r, taus, c, X)
        return best

    fast = np.logspace(np.log10(0.5), np.log10(10.0), 30)
    slow = np.logspace(np.log10(10.0), np.log10(60.0), 20)
    one = best_fit([(a,) for a in np.concatenate([fast, slow])])
    two = best_fit([(a, b) for a in fast for b in slow])
    n = len(tt)
    v1, v2 = one[0] / max(n - 3, 1), two[0] / max(n - 5, 1)
    r, taus, c, X = two if v2 < 0.9 * v1 else one
    cov = r / max(n - X.shape[1], 1) * np.linalg.inv(X.T @ X)
    return float(c[0]), float(np.sqrt(cov[0, 0])), float(max(taus))


def line_fit_V1(pm):
    """Straight-line fit of dV against 1/p; returns V1 and its standard error."""
    pm = np.asarray(pm, dtype=float)
    dV = np.arange(len(pm)) * STEP_ML
    X = np.column_stack([np.ones(len(pm)), 1.0 / pm])
    c, *_ = np.linalg.lstsq(X, dV, rcond=None)
    r = dV - X @ c
    dof = max(len(pm) - 2, 1)
    cov = (r @ r / dof) * np.linalg.inv(X.T @ X)
    return float(c[0]), float(np.sqrt(cov[0, 0]))


def leak_correction(t, p, tracker, plateaus):
    """Correct the plateau pressures for gas lost through leaks.

    Needs the pull-back to have been detected and at least 10 s of data after
    it. Returns a dict, or None if the correction cannot be made.
    """
    if not tracker.step_times or tracker.release_times is None:
        return None
    t_rel = tracker.release_times[1]
    aft = t >= t_rel + 1.0
    if aft.sum() < 15:
        return None

    first = tracker.step_times[0][0]
    base = (t >= first - 6.0) & (t < first - 0.5)
    if base.sum() < 3:
        base = t < first - 0.5
    p0 = float(p[base].mean())
    I = cumtrapz(p - p0, t)                      # Pa s
    I_rel = float(np.interp(t_rel, t, I))
    if I_rel <= 0:
        return None

    pls = sorted(plateaus, key=lambda q: q['mean'])
    masks = [(t >= pl['t_lo']) & (t <= pl['t_hi']) for pl in pls]
    raw = [pl['mean'] for pl in pls]
    V1_raw, sV1_raw = line_fit_V1(raw)

    ta, pa = t[aft] - t_rel, p[aft]

    def corrected(C, V1):
        nf = 1.0 - C * I / (p0 * V1)
        return [float((p[m] / nf[m]).mean()) for m in masks]

    V1, C = V1_raw, 0.0
    for _ in range(8):
        # Air leaks back IN while the system sits below p0 after the
        # pull-back; remove that before taking the asymptote.
        deficit = p0 - pa
        back = (C / V1) * cumtrapz(deficit, ta)
        p_inf, p_inf_err, tau_slow = asymptote(ta, pa - back)
        C = V1 * (p0 - p_inf) / I_rel             # mL/s
        pm = corrected(C, V1)
        V1, sV1 = line_fit_V1(pm)

    sC = V1 * np.hypot(p_inf_err, p0 * 0.1 / 300.0) / I_rel
    pm_hi, pm_lo = corrected(C + sC, V1), corrected(C - sC, V1)
    V1_hi, _ = line_fit_V1(pm_hi)
    V1_lo, _ = line_fit_V1(pm_lo)
    sV1_leak = 0.5 * abs(V1_hi - V1_lo)

    # Uncertainty on each corrected plateau pressure (1 s.d.)
    nf = 1.0 - C * I / (p0 * V1)
    u_rand, u_settle, u_leak, u_tot = [], [], [], []
    for k, m in enumerate(masks):
        q, tq = p[m] / nf[m], t[m]
        n = len(q)
        ur = float(np.std(q, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        if n > 2 and tq[-1] > tq[0]:
            drift = float(np.polyfit(tq, q, 1)[0] * (tq[-1] - tq[0]))
        else:
            drift = 0.0
        us = abs(drift) / np.sqrt(12.0)
        ul = 0.5 * abs(pm_hi[k] - pm_lo[k])
        u_rand.append(ur); u_settle.append(us); u_leak.append(ul)
        u_tot.append(float(np.sqrt(ur ** 2 + us ** 2 + ul ** 2)))

    notes = []
    if p_inf > p0:
        notes.append("The pressure returned slightly ABOVE its starting value, "
                     "so no gas loss was detected (the apparatus or room "
                     "probably warmed during the run). The correction is "
                     "negligible.")
    # Only warn if the extrapolation beyond the last 5 s of data is a large
    # part of the measured shortfall
    p_end = float(pa[ta >= ta[-1] - 5.0].mean())
    extrap = p_inf - p_end
    if abs(extrap) > max(30.0, 0.5 * abs(p0 - p_inf)):
        notes.append(f"The pressure was still recovering at the end of the "
                     f"hold (extrapolated a further {extrap:+.0f} Pa), so the "
                     "leak estimate is less certain than usual.")
    return dict(p0=p0, p_inf=p_inf, p_inf_err=p_inf_err,
                frac_lost=(p0 - p_inf) / p0, C=C, sC=sC,
                V1_raw=V1_raw, sV1_raw=sV1_raw, V1=V1, sV1=sV1,
                sV1_leak=sV1_leak, pm_corr=pm, raw=raw,
                u_rand=u_rand, u_settle=u_settle, u_leak=u_leak, u_tot=u_tot,
                t_rel=t_rel, tau_slow=tau_slow, notes=notes)


def leak_report(lc):
    if lc is None:
        return ("No leak correction: the pull-back to 50 mL was not detected "
                "or was not held long enough.")
    shift = lc['pm_corr'][-1] - lc['raw'][-1]
    lines = [
        f"Starting pressure p0:            {lc['p0']:.0f} Pa",
        f"Pressure after pull-back:        {lc['p_inf']:.0f} +/- "
        f"{lc['p_inf_err']:.0f} Pa",
        f"Gas lost during the run:         {lc['frac_lost'] * 100:.3f} %",
        f"Leak conductance C:              {lc['C'] * 1e3:.0f} +/- "
        f"{lc['sC'] * 1e3:.0f} x 1e-3 mL/s",
        f"Correction to the highest pressure: {shift:+.0f} Pa",
    ]
    if SHOW_V1:
        lines += [
            "",
            f"V1 without leak correction:  {lc['V1_raw']:.1f} +/- "
            f"{lc['sV1_raw']:.1f} mL (fit)",
            f"V1 with leak correction:     {lc['V1']:.1f} +/- {lc['sV1']:.1f} mL"
            f" (fit) +/- {lc['sV1_leak']:.1f} mL (correction)",
        ]
    for n in lc['notes']:
        lines += ["", "NOTE: " + n]
    return "\n".join(lines)


def plot_leak(t, p, lc, path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    rng = p.max() - p.min()
    ax.scatter(t, p, c=p, s=MARKER_SIZE, cmap=CMAP,
               vmin=p.min() - 0.15 * rng, vmax=p.max())
    ax.axhline(lc['p0'], color='white', ls='--', lw=1, label='starting pressure')
    ax.axhline(lc['p_inf'], color='orange', ls=':', lw=1.5,
               label='pressure after pull-back')
    ax.axvspan(lc['t_rel'], t[-1], color='palegreen', alpha=0.12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_title(f"Gas lost during the run: {lc['frac_lost'] * 100:.2f} %")
    ins = ax.inset_axes([0.04, 0.58, 0.32, 0.38])
    a = t >= lc['t_rel'] - 2
    ins.scatter(t[a], p[a], s=6, color='cyan')
    ins.axhline(lc['p0'], color='white', ls='--', lw=1)
    ins.axhline(lc['p_inf'], color='orange', ls=':', lw=1.5)
    ins.set_ylim(min(lc['p_inf'], lc['p0']) - 400,
                 max(lc['p_inf'], lc['p0']) + 250)
    ins.set_title('after pull-back (zoomed)', fontsize=8)
    ins.tick_params(labelsize=7)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ================================ Saving ===================================

def save_raw(out_dir, timestr, t, d, p, T):
    path = os.path.join(out_dir, f"{timestr} raw data.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Time (s)', 'Distance', 'Pressure (Pa)', 'Temperature (K)'])
        for row in zip(t, d, p, T):
            w.writerow([f"{row[0]:.3f}", f"{row[1]:.4g}",
                        f"{row[2]:.2f}", f"{row[3]:.3f}"])
    return path


def save_results(out_dir, timestr, plateaus, lc=None):
    pls = sorted(plateaus, key=lambda q: q['mean'])
    n = len(pls)
    if n == 0:
        print("No plateaus selected - only raw data saved.")
        return

    t_order = [pl['t_mid'] for pl in pls]
    if any(b < a for a, b in zip(t_order, t_order[1:])):
        print("WARNING: sorting by pressure does not match time order. "
              "Check for a mis-click, a leak, or a step done out of sequence.")

    dV_cum = [k * STEP_ML for k in range(n)]

    # Detailed per-plateau results
    path = os.path.join(out_dir, f"{timestr} plateau details.csv")
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Cumulative change in volume (mL)', 'Mean pressure (Pa)',
                    'SD (Pa)', 'N samples', 'SEM (Pa)',
                    'Drift across window (Pa)', 'Median temperature (K)',
                    'Window start (s)', 'Window end (s)',
                    'Leak-corrected pressure (Pa)',
                    'Uncertainty in leak-corrected pressure (Pa)',
                    '  of which random (Pa)', '  of which settling (Pa)',
                    '  of which leak correction (Pa)'])
        for k, (dv, pl) in enumerate(zip(dV_cum, pls)):
            if lc:
                extra = [f"{lc['pm_corr'][k]:.2f}", f"{lc['u_tot'][k]:.1f}",
                         f"{lc['u_rand'][k]:.1f}", f"{lc['u_settle'][k]:.1f}",
                         f"{lc['u_leak'][k]:.1f}"]
            else:
                extra = [''] * 5
            w.writerow([f"{dv:g}", f"{pl['mean']:.2f}", f"{pl['sd']:.2f}",
                        pl['n'], f"{pl['sem']:.2f}", f"{pl['drift']:.1f}",
                        f"{pl['T']:.2f}",
                        f"{pl['t_lo']:.2f}", f"{pl['t_hi']:.2f}"] + extra)
    print(f"Saved {path}")

    # Original-format file (unchanged columns, for the existing Excel steps)
    if n != N_POINTS:
        print(f"WARNING: {n} plateaus selected, expected {N_POINTS}. "
              "The 'pressures and delta volumes' file was NOT written - "
              "re-run the analysis or use the plateau details file.")
        return
    path = os.path.join(out_dir, f"{timestr} pressures and delta volumes.csv")
    cols = [[pl['mean'] for pl in pls], [STEP_ML] * n, dV_cum]
    header = ("Pressure (Pa), Incremental change in volume (mL) , "
              "Cummulative change in volume (mL)")
    if lc:
        cols += [lc['pm_corr'], lc['u_tot']]
        header += (", Leak-corrected pressure (Pa), "
                   "Uncertainty in leak-corrected pressure (Pa)")
    np.savetxt(path, np.column_stack(cols), delimiter=",", comments="",
               fmt="%.2f", header=header)
    print(f"Saved {path}")


# ================================= Main ====================================

def main():
    print(f"PHY1035 Ideal Gas Law 1 - {VERSION}")
    show_splash()
    sys.path.append(os.path.abspath(TOOLS_PATH))
    import serial
    from boardFinder import get_serial_connection

    timestr = time.strftime("%Y-%m-%d__%H-%M-%S")
    out_dir = os.path.join(script_dir(), DATA_FOLDER, timestr)

    try:
        ser = get_serial_connection(baud=115200, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open the serial connection: {e}")
        return

    try:
        time.sleep(2)
        ser.reset_input_buffer()
        print("Recording. Hold the syringe still while the plot is RED; "
              "move the syringe when it turns GREEN.\nThe recording stops by "
              "itself after the final pull-back.")
        t, d, p, T, tracker = acquire(ser)
    finally:
        if ser.is_open:
            ser.close()

    if len(t) == 0:
        print("No data recorded.")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"Saved {save_raw(out_dir, timestr, t, d, p, T)}")

    print(f"\nClick inside each of the {N_POINTS} green bands, then press "
          "Enter.")
    # Only the compression part of the run is shown for clicking
    if tracker.release_times is not None:
        keep = t < tracker.release_times[0] - 0.5
    else:
        keep = np.ones(len(t), dtype=bool)
        print("NOTE: the pull-back to 50 mL was not detected - no leak "
              "correction will be possible for this run.")
    picker = PlateauPicker(t[keep], p[keep], T[keep],
                           step_times=tracker.step_times)
    plt.show()
    picker.fig.savefig(os.path.join(out_dir, f"{timestr} plateaus.png"))

    lc = None
    if len(picker.plateaus) == N_POINTS:
        lc = leak_correction(t, p, tracker, picker.plateaus)
    save_results(out_dir, timestr, picker.plateaus, lc)

    text = leak_report(lc)
    print("\n" + text)
    with open(os.path.join(out_dir, f"{timestr} leak correction.txt"),
              'w') as fh:
        fh.write(text + "\n")
    if lc is not None:
        plot_leak(t, p, lc,
                  os.path.join(out_dir, f"{timestr} leak correction.png"))


if __name__ == '__main__':
    main()
