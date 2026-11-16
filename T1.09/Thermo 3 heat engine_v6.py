# -*- coding: utf-8 -*-
"""
Thermo 3 heat engine: acquisition and analysis

Original: bs426, Mar 2025
v6, revised 16 Nov 2026:
  * Labels only (no change to any calculation): the terminal output, plots and
    summary columns now use the same names and symbols as the lab script:
      W_gas    = net work done by the gas (p-V loop area)
      rise     = Delta h_rise, how far the mass rises during the power stroke
      W_lift   = mg x rise, work done lifting the mass
      net lift = Delta h_net, removal height - placement height
      W_useful = mg x net lift, useful work
  * Column names changed. If an older results_summary_v6.csv with the old column
    names exists, it is renamed to results_summary_v6_old_<date-time>.csv and a
    fresh results_summary_v6.csv is started, so the file name in the lab script
    stays correct.
v6 (Nov 2026):
  * Live plot only: p, V and T are shown as a rolling mean of the last
    LIVE_SMOOTH_SAMPLES samples, which looks much cleaner. Everything that is
    calculated (stage detection, p-V area, piston heights, summary) still uses the
    raw 2 Hz samples, and the raw samples are what is saved.
  * Snow leopard splash screen with the version number at start-up
    (snow_leopard.jpg must be in the same folder as this script).
  * Results go to 'results_summary_v6.csv' (same columns as v5).
v5 (Nov 2026):
  * Removed v4's "wait for the gas to settle" checks (they slowed runs without
    addressing the main problem, which is leakage at heavy masses). Back to a
    fixed 5 s countdown before loading and the v3 end-of-cycle test.
  * Height-picker plot: each stage (mass off/on, cold/hot) has its own faint colour.
  * Live p-V and temperature axes recentre on the data (and widen if needed) whenever
    the data reach the edge of the plot.
  * Results go to 'results_summary_v5.csv'; the 'warnings' column flags actions taken
    before their countdown finished.
v3 (Nov 2026):
  * The summary also records the gas expansion on heating (a measure of how much the
    gas was heated), net lift / rise and the closure gap (a leak indicator).
  * Live p-V colour scale and axes are set from the expected pressure step, so the
    colours span the whole cycle (axes widen automatically if the data run off).
  * Guided cycle: an instruction box counts down each stage, then tells the student
    what to do next. The script detects the action from the data (pressure step when
    the mass goes on or off; pressure/volume/temperature change when the calorimeter
    moves between baths) and starts the next stage's countdown automatically. Stage
    times are saved to stages.csv, and an action taken before the countdown finished
    is flagged.
  * Raw 2 Hz samples are saved and analysed. The 10-sample rolling means have been
    removed from the data path, because they smeared every load/unload step into a
    5 s diagonal ramp on the p-V plot.
  * Distance is median-filtered (this removes sensor noise but keeps the steps sharp).
    Pressure is left unfiltered.
  * The sample caught mid-step at each load/unload is excluded from the loop area.
    The pressure sensor reads instantaneously, but the distance sensor averages over
    its timing window, so that one sample mixes the before and after states.
  * Loop area is computed on gauge pressure (p - p0), so if the cycle does not quite
    close, the closing line adds almost nothing.
  * Students click three piston heights (initial, final, placement) using a cursor
    with coloured bands. The detected stages are shaded on the plot to guide them.
  * Each run is appended as one row to 'Heat engine data/results_summary.csv'.

NOTE: .npy files saved by this version hold RAW samples. Files from earlier versions
hold 10-sample rolling means, so don't mix them in the same analysis.
"""

import sys
import os
import re
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.widgets import Cursor

sys.path.append(os.path.abspath("/home/dorkmaster/Desktop/PHY1035/Tools"))
plt.style.use('dark_background')

# ======================= SETTINGS ===================================================
# Position-to-volume calibration from Task 1.8: np.poly1d([gradient, intercept])
volume_fn = np.poly1d([-0.6531, 410.95])

G = 9.81                  # m s^-2
MEDIAN_WINDOW = 5         # samples (2.5 s at 2 Hz); applied to distance only
DROP_AFTER_STEP = 1       # samples excluded after each load/unload pressure step
BAND_HALF_WIDTH_MM = 1.0  # half-height of the selection bands (~sensor resolution)

# Minimum time (s) for each stage before the student is told to take the next action
MIN_BASELINE_S = 5        # mass off, cold bath
MIN_LOADED_COLD_S = 8     # mass on, cold bath (keep this the same for every run)
MIN_LOADED_HOT_S = 15     # mass on, hot bath (power stroke)
MIN_UNLOADED_HOT_S = 3    # mass off, hot bath
MIN_UNLOADED_COLD_S = 10  # mass off, cold bath, before checking that the piston has stopped

# Thresholds used to detect when the calorimeter has been moved between baths
# (any one is enough)
BATH_DP_PA = 200          # pressure step (friction band is usually ~500 Pa)
BATH_DV_ML = 1.5          # volume change (about 2.5 mm of piston travel)
BATH_DT_K = 0.75          # thermocouple change (slow, so used only as a backup)
SETTLED_DV_ML = 0.7       # piston counts as stopped if V changes less than this in 5 s

# To re-analyse a saved run instead of acquiring new data, set this to the run's
# folder, e.g. "Heat engine data/2026-11-12__14-03-22". Leave as None for normal use.
REANALYSE_FOLDER = None

VERSION = 'v6'
LIVE_SMOOTH_SAMPLES = 10  # rolling mean used ONLY for the live plot (10 samples = 5 s)
SPLASH_SECONDS = 3        # how long the snow leopard splash screen is shown
# ====================================================================================


def piston_area():
    """Piston area (m^2) from the calibration gradient (mL/mm)."""
    return abs(volume_fn.coeffs[0]) * 1e-3


def load_threshold(mass_g):
    """Pressure step (Pa) that counts as the mass going on or off: half of mg/A."""
    return max(300.0, 0.5 * (mass_g * 1e-3) * G / piston_area())


# ----------------------------------------------------------------------------------
# Serial helpers
# ----------------------------------------------------------------------------------
def extract_floats(text):
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    return [float(num) for num in numbers]


def first_float(raw_line):
    vals = extract_floats(raw_line.decode('utf-8', errors='ignore').strip())
    return vals[0] if vals else None


def read_frame(ser):
    """Read one (distance, pressure, temperature) frame.

    The Arduino sends distance, pressure and temperature on separate lines. Only the
    pressure (~1e5 Pa) is larger than 5e4, so we keep reading until we find it. This
    re-synchronises automatically if we start reading part-way through a frame.
    """
    prev = None
    for _ in range(6):                       # at most two frames' worth of lines
        val = first_float(ser.readline())
        if val is not None and val > 5e4 and prev is not None:
            temp = first_float(ser.readline())
            if temp is None:
                return None
            return prev, val, temp
        prev = val
    return None


# ----------------------------------------------------------------------------------
# Stage tracking: countdown, instructions and automatic detection of each action
# ----------------------------------------------------------------------------------
class StageTracker:
    STAGES = [
        dict(key='mass_on', min_s=MIN_BASELINE_S,
             hold='Hold the calorimeter in the COLD bath. Do not touch the syringe.',
             act='Gently place the mass on the syringe',
             done='Mass on detected'),
        dict(key='into_hot', min_s=MIN_LOADED_COLD_S,
             hold='Mass ON. Keep the calorimeter in the COLD bath.',
             act='Move the calorimeter into the HOT bath',
             done='Calorimeter in hot bath detected'),
        dict(key='mass_off', min_s=MIN_LOADED_HOT_S,
             hold='Mass ON. Keep the calorimeter in the HOT bath.',
             act='Lift the mass off the syringe',
             done='Mass off detected'),
        dict(key='into_cold', min_s=MIN_UNLOADED_HOT_S,
             hold='Mass OFF. Keep the calorimeter in the HOT bath.',
             act='Move the calorimeter into the COLD bath',
             done='Calorimeter in cold bath detected'),
        dict(key='settled', min_s=MIN_UNLOADED_COLD_S,
             hold='Keep the calorimeter in the COLD bath.',
             act='Waiting for the piston to stop moving...',
             done='Cycle complete'),
    ]

    def __init__(self, mass_g):
        self.thr_load = load_threshold(mass_g)
        self.t, self.p, self.T, self.d, self.Vm = [], [], [], [], []
        self.k = 0                 # index of current stage
        self.start_t = None        # time the current stage started
        self.start_i = 0           # sample index where the current stage started
        self.events = []           # dicts: key, time, stage duration, early flag
        self.flags = []            # e.g. 'into_hot:early'
        self.warning = ''

    @property
    def finished(self):
        return self.k >= len(self.STAGES)

    def update(self, t, d, p, T):
        """Add one sample. Returns the event dict if an action was detected."""
        self.t.append(t)
        self.d.append(d)
        self.p.append(p)
        self.T.append(T)
        self.Vm.append(float(volume_fn(np.median(self.d[-MEDIAN_WINDOW:]))))
        if self.start_t is None:
            self.start_t = t
        if self.finished or not self._detected():
            return None
        stage = self.STAGES[self.k]
        dur = t - self.start_t
        # 1 s tolerance: at 2 Hz an action is only registered at the next sample
        too_soon = dur < stage['min_s'] - 1.0 and stage['key'] != 'settled'
        ev = dict(key=stage['key'], time=round(float(t), 2),
                  stage_duration=round(float(dur), 2), early=bool(too_soon))
        self.events.append(ev)
        if too_soon:
            self.flags.append(f'{stage["key"]}:early')
            self.warning = (f'Warning: that action came {stage["min_s"] - dur:.1f} s '
                            f'too early (stage lasted {dur:.1f} s, minimum '
                            f'{stage["min_s"]} s). Consider repeating this run.')
        elif stage['key'] != 'settled':
            self.warning = ''
        self.k += 1
        self.start_t = t
        self.start_i = len(self.p) - 1
        return ev

    def _ref(self, arr, skip):
        """Samples of arr in this stage, excluding the first `skip` and the latest."""
        seg = arr[self.start_i + skip:len(arr) - 1]
        return seg if len(seg) >= 2 else None

    def _detected(self):
        key = self.STAGES[self.k]['key']
        p, Vm, T = self.p, self.Vm, self.T
        if key == 'mass_on':
            ref = self._ref(p, 0)
            return ref is not None and p[-1] - np.median(ref) > self.thr_load
        if key == 'mass_off':
            ref = self._ref(p, 2)
            return ref is not None and np.median(ref) - p[-1] > self.thr_load
        if key == 'into_hot':          # skip the load step and its settling
            rp, rV, rT = self._ref(p, 2), self._ref(Vm, 2), self._ref(T, 2)
            if rp is None:
                return False
            return (p[-1] - np.median(rp) > BATH_DP_PA or
                    Vm[-1] - min(rV) > BATH_DV_ML or
                    T[-1] - np.median(rT) > BATH_DT_K)
        if key == 'into_cold':         # pressure settles for ~1.5 s after unloading
            rp, rV, rT = self._ref(p, 3), self._ref(Vm, 3), self._ref(T, 3)
            if rp is None:
                return False
            return (np.median(rp) - p[-1] > BATH_DP_PA or
                    max(rV) - Vm[-1] > BATH_DV_ML or
                    max(rT) - T[-1] > BATH_DT_K)
        if key == 'settled':
            if self.t[-1] - self.start_t < MIN_UNLOADED_COLD_S or len(Vm) < 11:
                return False
            return abs(Vm[-1] - Vm[-11]) < SETTLED_DV_ML
        return False

    def message(self, now):
        """(text, colour) for the instruction box at time `now`."""
        if self.finished:
            return ('CYCLE COMPLETE: close this window to analyse the data.', 'green')
        n = len(self.STAGES)
        stage = self.STAGES[self.k]
        header = f'STAGE {self.k + 1} of {n}: {stage["hold"]}'
        if self.events:
            last = self.STAGES[self.k - 1]['done']
            header = f'✓ {last} at {self.events[-1]["time"]:.1f} s\n' + header
        elapsed = now - (self.start_t if self.start_t else now)
        remaining = stage['min_s'] - elapsed
        if remaining > 0 and stage['key'] != 'settled':
            text = f'{header}\nNext: {stage["act"].lower()} in {np.ceil(remaining):.0f} s'
            colour = 'dimgrey'
        elif stage['key'] == 'settled':
            text = f'{header}\n{stage["act"]}'
            colour = 'dimgrey'
        else:
            text = f'{header}\nNOW: {stage["act"].upper()}'
            colour = 'darkgreen'
        if self.warning:
            text += '\n' + self.warning
            colour = 'darkred'
        return text, colour


def save_stages(folder, events):
    with open(os.path.join(folder, 'stages.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['key', 'time', 'stage_duration', 'early'])
        w.writeheader()
        w.writerows(events)


def load_stages(folder):
    path = os.path.join(folder, 'stages.csv')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [dict(key=r['key'], time=float(r['time']),
                     stage_duration=float(r['stage_duration']),
                     early=(r['early'] == 'True')) for r in csv.DictReader(f)]


def robust_mean(values, threshold=2.0):
    """Mean after discarding outliers (> threshold x MAD from the median)."""
    values = np.asarray(values, float)
    median = np.median(values)
    dev = np.abs(values - median)
    mad = np.median(dev)
    keep = dev < threshold if mad == 0 else dev / mad < threshold
    return float(np.mean(values[keep])) if keep.any() else float(median)


def show_splash():
    """Snow leopard splash screen with the script version (basic version control)."""
    here = os.path.dirname(os.path.abspath(__file__))
    img_path = None
    for name in os.listdir(here):
        if name.lower() in ('snow_leopard.jpg', 'snow_leopard.jpeg', 'snow_leopard.png'):
            img_path = os.path.join(here, name)
            break
    if img_path is None:
        print(f"Thermo 3 heat engine {VERSION} (splash image snow_leopard.jpg not found)")
        return
    img = plt.imread(img_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.5, 0.06, f'Thermo 3 heat engine  {VERSION}', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=22, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.6))
    try:
        fig.canvas.manager.set_window_title(f'Thermo 3 heat engine {VERSION}')
    except Exception:
        pass
    plt.show(block=False)
    plt.pause(SPLASH_SECONDS)
    plt.close(fig)


def _recentre(lo, hi, data, grow=1.3, edge=0.05):
    """New (lo, hi) keeping the data centred, or None if no change is needed.

    Nothing changes while all the data sit inside the inner 90% of the window. Once
    they reach the edge, the window is recentred on the data and widened if needed
    (span = max(current span, 1.3 x data range)). It never zooms in.
    """
    dmin, dmax = float(np.min(data)), float(np.max(data))
    span = hi - lo
    if dmin >= lo + edge * span and dmax <= hi - edge * span:
        return None
    span = max(span, grow * (dmax - dmin))
    mid = 0.5 * (dmin + dmax)
    return mid - span / 2, mid + span / 2


def recentre_limits(ax, x=None, y=None):
    """Keep live data roughly centred: recentre/widen an axis when data reach its edge."""
    if x is not None and len(x):
        new = _recentre(*ax.get_xlim(), x)
        if new:
            ax.set_xlim(*new)
    if y is not None and len(y):
        new = _recentre(*ax.get_ylim(), y)
        if new:
            ax.set_ylim(*new)


# ----------------------------------------------------------------------------------
# Acquisition (guided cycle with live p-V and T-t plots; raw samples stored)
# ----------------------------------------------------------------------------------
def acquire(mass_g):
    import serial  # noqa: F401  (needed by boardFinder)
    from boardFinder import get_serial_connection

    cmap = cm.jet

    ser = get_serial_connection(baud=115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)

    print("Reading initial pressure and volume to centre the plot...")
    frame = None
    for _ in range(10):
        frame = read_frame(ser)
        if frame is not None:
            break
    if frame is None:
        print("Warning: no valid data received, using default plot limits")
        d0, p_init = 100.0, 1.0e5
    else:
        d0, p_init, _ = frame
    V_init = volume_fn(d0)
    dp_load = (mass_g * 1e-3) * G / piston_area()       # expected pressure step (Pa)
    # Colour scale spans the cycle's actual pressure range (unloaded -> loaded + friction)
    norm = plt.Normalize(vmin=p_init - 100, vmax=p_init + dp_load + 350)
    p_lo = p_init - 0.3 * dp_load - 300
    p_hi = p_init + 1.3 * dp_load + 400
    V_lo = V_init - (V_init * dp_load / p_init + 10)    # room for the drop on loading
    V_hi = V_init + 40                                  # room for the expansion
    ser.reset_input_buffer()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7.5))
    fig.subplots_adjust(top=0.72, bottom=0.09, left=0.08, right=0.97, wspace=0.25)
    scat = ax1.scatter([], [], c=[], s=12, cmap=cmap, norm=norm)
    ax1.set_xlim(V_lo, V_hi)
    ax1.set_ylim(p_lo, p_hi)
    ax1.set_xlabel('Volume (mL)')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.set_title('Volume vs Pressure')
    temp_line, = ax2.plot([], [], '--or', ms=3)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(270, 330)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature vs Time')
    instr = fig.text(0.5, 0.86, '', ha='center', va='center', fontsize=13,
                     fontweight='bold', color='white', wrap=True,
                     bbox=dict(boxstyle='round,pad=0.6', facecolor='dimgrey'))
    plt.show(block=False)

    tracker = StageTracker(mass_g)
    disp_V, disp_p, disp_T = [], [], []      # smoothed copies for the live plot only
    t_start = time.time()
    last_msg = None
    try:
        while plt.fignum_exists(fig.number):
            now = time.time() - t_start
            redraw = False
            if ser.in_waiting >= 1:
                frame = read_frame(ser)
                if frame is not None:
                    ev = tracker.update(now, *frame)
                    if ev:
                        print(f"{ev['time']:6.1f} s  {ev['key']}"
                              + ("  (EARLY)" if ev['early'] else ""))
                    # display-only smoothing (raw values stay in tracker)
                    n = LIVE_SMOOTH_SAMPLES
                    disp_V.append(float(volume_fn(np.mean(tracker.d[-n:]))))
                    disp_p.append(float(np.mean(tracker.p[-n:])))
                    disp_T.append(robust_mean(tracker.T[-n:]))
                    V = np.array(disp_V)
                    scat.set_offsets(np.column_stack((V, disp_p)))
                    scat.set_array(np.array(disp_p))
                    temp_line.set_data(tracker.t, disp_T)
                    if now > 60:
                        ax2.set_xlim(now - 60, now)
                    recentre_limits(ax1, V, disp_p)
                    recentre_limits(ax2, None, disp_T)
                    redraw = True
            msg = tracker.message(now)
            if msg != last_msg:
                instr.set_text(msg[0])
                instr.get_bbox_patch().set_facecolor(msg[1])
                last_msg = msg
                redraw = True
            if redraw:
                fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if ser.is_open:
            ser.close()
    print("Plot window closed, stopping.")
    if not tracker.finished:
        print("Warning: the cycle was not completed (stopped during stage "
              f"{tracker.k + 1}). Results may not be meaningful.")
    return (np.array(tracker.t), np.array(tracker.d), np.array(tracker.p),
            np.array(tracker.T), tracker.events, tracker.flags)


# ----------------------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------------------
def running_median(x, k):
    k = int(k) | 1                           # force odd
    h = k // 2
    xp = np.pad(np.asarray(x, float), h, mode='edge')
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def find_steps(p, threshold):
    """Indices of the first sample after each pressure step larger than threshold."""
    return np.where(np.abs(np.diff(p)) > threshold)[0] + 1


def loop_work(V_mL, p_gauge):
    """Closed-loop integral of p dV (J). Positive for a clockwise (engine) cycle."""
    Vc = np.append(V_mL, V_mL[0])
    pc = np.append(p_gauge, p_gauge[0])
    return np.sum(0.5 * (pc[1:] + pc[:-1]) * np.diff(Vc)) * 1e-6


def process(times, d, p, mass_g):
    """Returns a dict with filtered signals, the samples used and the loop work."""
    d_med = running_median(d, MEDIAN_WINDOW)
    V = volume_fn(d_med)

    steps = find_steps(p, load_threshold(mass_g))
    first = steps[0] if len(steps) else len(p)
    n_base = first if first >= 3 else min(3, len(p))
    p0 = np.median(p[:n_base])
    d_base = np.median(d_med[:n_base])

    keep = np.ones(len(p), bool)
    for s in steps:
        keep[s:s + DROP_AFTER_STEP] = False

    W = loop_work(V[keep], p[keep] - p0)
    return dict(d_med=d_med, V=V, keep=keep, p0=p0, d_base=d_base,
                height=d_base - d_med, height_raw=d_base - np.asarray(d), W=W,
                n_steps=len(steps))


def plot_pv(res, p, folder):
    V, keep = res['V'], res['keep']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill(V[keep], p[keep], color='lightblue', alpha=0.4, zorder=1)
    ax.plot(V[keep], p[keep], '-', color='white', lw=0.6, alpha=0.6, zorder=2)
    ax.scatter(V[keep], p[keep], c=p[keep], cmap='jet', s=18, zorder=3)
    if (~keep).any():
        ax.scatter(V[~keep], p[~keep], facecolors='none', edgecolors='white', s=40,
                   zorder=4, label='excluded (mid-step sample)')
        ax.legend(loc='lower left', fontsize=9)
    ax.set_xlabel('Volume (mL)')
    ax.set_ylabel('Pressure (Pa)')
    ax.set_title(f'W_gas = net work done by the gas (enclosed area) = {res["W"] * 1e3:.1f} mJ\n'
                 '(close this window to continue)')
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'Filled PV plot.png'), dpi=150)
    return fig


# ----------------------------------------------------------------------------------
# Height picker: three clicks with coloured translucent bands
# ----------------------------------------------------------------------------------
STAGE_SHADING = [  # (label, start event, end event, colour)
    ('mass OFF\ncold', None, 'mass_on', 'deepskyblue'),
    ('mass ON\ncold', 'mass_on', 'into_hot', 'violet'),
    ('mass ON\nhot', 'into_hot', 'mass_off', 'orangered'),
    ('mass OFF\nhot', 'mass_off', 'into_cold', 'gold'),
    ('mass OFF\ncold', 'into_cold', None, 'deepskyblue'),
]
STAGE_ALPHA = 0.18


class HeightPicker:
    STEPS = [
        ('initial',   'lime',
         'Click 1 of 3 (GREEN): LOWEST piston position while the mass is ON'),
        ('final',     'red',
         'Click 2 of 3 (RED): HIGHEST piston position while the mass is ON'),
        ('placement', 'deepskyblue',
         'Click 3 of 3 (BLUE): piston position just BEFORE the mass was placed on'),
    ]

    def __init__(self, t, height, height_raw, folder, events=()):
        self.folder = folder
        self.values = {}
        self.bands = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        ax = self.ax
        self._shade_stages(t, events)
        ax.plot(t, height_raw, '.', color='grey', alpha=0.4, ms=4,
                label='raw sensor reading')
        ax.plot(t, height, 'o-', color='white', ms=4, lw=0.8,
                label='median-filtered (click on these)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Piston height above starting position (mm)')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.2)
        lo, hi = np.nanmin(height_raw), np.nanmax(height_raw)
        ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.45 * (hi - lo))  # room for labels
        self.ytr = ax.get_yaxis_transform()   # x in axes units, y in data units
        w = BAND_HALF_WIDTH_MM
        self.preview = Rectangle((0, height[0] - w), 1, 2 * w, transform=self.ytr,
                                 color=self.STEPS[0][1], alpha=0.25, zorder=0)
        ax.add_patch(self.preview)
        self.readout = ax.text(0.01, 0.90, '', transform=ax.transAxes, va='top',
                               fontsize=11, bbox=dict(boxstyle='round',
                                                      facecolor='black', alpha=0.7))
        self.cursor = Cursor(ax, useblit=False, horizOn=True, vertOn=True,
                             color='white', linewidth=0.8, linestyle='--')
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self._update_title()

    def _shade_stages(self, t, events):
        times = {e['key']: e['time'] for e in events}
        xtr = self.ax.get_xaxis_transform()
        for label, a, b, colour in STAGE_SHADING:
            t0 = t[0] if a is None else times.get(a)
            t1 = t[-1] if b is None else times.get(b)
            if t0 is None or t1 is None:
                continue
            self.ax.axvspan(t0, t1, color=colour, alpha=STAGE_ALPHA, lw=0, zorder=0)
            self.ax.text((t0 + t1) / 2, 0.99, label, transform=xtr, ha='center',
                         va='top', fontsize=9, fontweight='bold', color=colour)

    # ----- UI state -----
    def _n(self):
        return len(self.bands)

    def _update_title(self):
        if self._n() < 3:
            name, colour, text = self.STEPS[self._n()]
            self.ax.set_title(text + '\n(right-click to undo)', color=colour,
                              fontsize=12, fontweight='bold')
            self.preview.set_color(colour)
            self.preview.set_visible(True)
        else:
            self.ax.set_title('All three positions selected: press ENTER to accept, '
                              'or right-click to undo', color='yellow',
                              fontsize=12, fontweight='bold')
            self.preview.set_visible(False)
        self._update_readout()
        self.fig.canvas.draw_idle()

    def _update_readout(self, y=None):
        lines = [f'{n}: {self.values[n]:.1f} mm' for n, _, _ in self.STEPS
                 if n in self.values]
        if 'initial' in self.values and 'final' in self.values:
            lines.append(f'rise (\u0394h_rise) = '
                         f'{self.values["final"] - self.values["initial"]:.1f} mm')
        if 'final' in self.values and 'placement' in self.values:
            lines.append(f'net lift (\u0394h_net) = '
                         f'{self.values["final"] - self.values["placement"]:.1f} mm')
        if y is not None and self._n() < 3:
            lines.insert(0, f'cursor: {y:.1f} mm')
        self.readout.set_text('\n'.join(lines))

    # ----- event handlers -----
    def on_move(self, event):
        if event.inaxes != self.ax or event.ydata is None:
            return
        self.preview.set_y(event.ydata - BAND_HALF_WIDTH_MM)
        self._update_readout(event.ydata)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.ydata is None:
            return
        if event.button == 1 and self._n() < 3:
            name, colour, _ = self.STEPS[self._n()]
            y = event.ydata
            band = Rectangle((0, y - BAND_HALF_WIDTH_MM), 1, 2 * BAND_HALF_WIDTH_MM,
                             transform=self.ytr, color=colour, alpha=0.35, zorder=0)
            self.ax.add_patch(band)
            label = self.ax.text(1.0, y, f' {name}', transform=self.ytr,
                                 color=colour, va='center', ha='left', fontsize=10)
            self.bands.append((band, label))
            self.values[name] = y
            print(f'{name} position = {y:.1f} mm')
        elif event.button == 3 and self._n() > 0:
            band, label = self.bands.pop()
            band.remove()
            label.remove()
            self.values.pop(self.STEPS[self._n()][0], None)
        self._update_title()

    def on_key(self, event):
        if event.key == 'enter' and self._n() == 3:
            if self.values['final'] <= self.values['initial']:
                print('Warning: the FINAL (red) position is not above the INITIAL '
                      '(green) one. Right-click to undo and check your selections.')
                return
            self.fig.savefig(os.path.join(self.folder, 'Piston height selections.png'),
                             dpi=150)
            plt.close(self.fig)

    def run(self):
        plt.show()
        return self.values if len(self.values) == 3 else None


# ----------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------
def append_summary(folder, mass_g, res, sel, events, flags=None):
    summary = os.path.join(os.path.dirname(folder), 'results_summary_v6.csv')
    rise = sel['final'] - sel['initial']
    net = sel['final'] - sel['placement']
    m = mass_g * 1e-3
    grad = abs(volume_fn.coeffs[0])                       # mL per mm
    V_loaded = volume_fn(res['d_base'] - sel['initial'])  # gas volume at start of lift
    expansion = 100 * rise * grad / V_loaded              # % volume increase on heating
    V_start = volume_fn(res['d_base'])
    V_end = np.median(res['V'][-10:])                     # last ~5 s
    gap = V_end - V_start
    dur = {e['key']: e['stage_duration'] for e in events}
    warnings = flags if flags is not None else [e['key'] + ':early'
                                                 for e in events if e['early']]
    row = {
        'run folder': os.path.basename(folder),
        'mass (g)': mass_g,
        'W_gas: net work done by gas, PV area (mJ)': round(float(res['W']) * 1e3, 2),
        'rise (mm)': round(float(rise), 2),
        'W_lift = mg x rise (mJ)': round(float(m * G * rise), 2),
        'net lift (mm)': round(float(net), 2),
        'W_useful = mg x net lift (mJ)': round(float(m * G * net), 2),
        'mass on, cold (s)': dur.get('into_hot', ''),
        'mass on, hot (s)': dur.get('mass_off', ''),
        'warnings': ' '.join(warnings),
        'gas expansion on heating (%)': round(float(expansion), 2),
        'net lift / rise': round(float(net / rise), 3) if rise else '',
        'closure gap (mL)': round(float(gap), 2),
        'baseline pressure p0 (Pa)': round(float(res['p0']), 1),
    }
    if os.path.exists(summary):           # never append under a different header
        with open(summary, newline='') as f:
            header = next(csv.reader(f), [])
        if header != list(row):
            # keep the lab-script file name: archive the old file and start afresh
            old = summary.replace('.csv', time.strftime('_old_%Y-%m-%d__%H-%M-%S.csv'))
            os.rename(summary, old)
            print(f"Note: older summary file (different columns) renamed to {old}")
    new = not os.path.exists(summary)
    with open(summary, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row, summary


def main():
    show_splash()
    print(f"Thermo 3 heat engine {VERSION}")
    while True:
        try:
            mass_g = float(input("Enter the mass value in grams: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    if REANALYSE_FOLDER:
        folder = REANALYSE_FOLDER
        times = np.load(os.path.join(folder, 'times.npy'))
        d = np.load(os.path.join(folder, 'distances.npy'))
        p = np.load(os.path.join(folder, 'pressures.npy'))
        events = load_stages(folder)
        flags = None
    else:
        times, d, p, T, events, flags = acquire(mass_g)
        if len(times) < 10:
            print("Too little data recorded, nothing to analyse.")
            return
        folder = os.path.join(os.getcwd(), 'Heat engine data',
                              time.strftime("%Y-%m-%d__%H-%M-%S"))
        os.makedirs(folder, exist_ok=True)
        for name, arr in [('times', times), ('distances', d), ('pressures', p),
                          ('temperatures', T), ('volumes', volume_fn(d))]:
            np.save(os.path.join(folder, name + '.npy'), arr)
        save_stages(folder, events)

    res = process(times, d, p, mass_g)
    print(f"\nW_gas, net work done by the gas (area of PV loop) = "
          f"{res['W'] * 1e3:.1f} mJ  ({res['W']:.4f} J)")
    if res['W'] < 0:
        print("Warning: negative area. The loop ran anticlockwise; check the "
              "calibration gradient sign and the order of the stages.")
    if res['n_steps'] < 2:
        print("Warning: fewer than two load/unload steps were detected in the pressure "
              "data. Was the mass placed AND removed during the recording?")

    plot_pv(res, p, folder)
    plt.show()

    picker = HeightPicker(times, res['height'], res['height_raw'], folder, events)
    sel = picker.run()
    if sel is None:
        print("Height selection not completed: W_lift and W_useful not saved.")
        return

    row, summary = append_summary(folder, mass_g, res, sel, events, flags)
    print(f"\nRise during the power stroke = {row['rise (mm)']} mm")
    print(f"W_lift, work done lifting the mass (mg x rise) = "
          f"{row['W_lift = mg x rise (mJ)']} mJ")
    print(f"Net lift (removal height - placement height) = {row['net lift (mm)']} mm")
    print(f"W_useful, useful work (mg x net lift) = "
          f"{row['W_useful = mg x net lift (mJ)']} mJ")
    if row['warnings']:
        print(f"Warnings for this run: {row['warnings']}")
    print(f"Gas expansion on heating = {row['gas expansion on heating (%)']} %")
    print(f"Closure gap (end volume - start volume) = {row['closure gap (mL)']} mL"
          "  (a large negative value suggests a leak)")
    print(f"\nResults added to: {summary}")


if __name__ == '__main__':
    main()
