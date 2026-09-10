# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:08:45 2025 (rewritten 2026)

@author: bs426

Ideal gas law / absolute zero experiment -- MULTI-BATH EQUILIBRIUM VERSION.

Measures pressure vs temperature of a fixed volume of gas as the calorimeter is
moved between several water baths at different temperatures. Unlike the previous
transient version, this script is designed for EQUILIBRIUM points: the operator
dwells at each bath until the gas temperature has stabilised, records a short
plateau, then moves to the next bath.

Method (suggested protocol):
  1. Line up water baths spanning your range, e.g. ice water (~0-5 C),
     cool (~15 C), warm (~28 C), hot-tap (~40 C).
  2. With the BLUE valve OPEN, let the sensor settle at room pressure/temperature.
  3. Close the BLUE valve to seal the apparatus.
  4. Run this script. It begins logging immediately.
  5. Let it sit sealed at room temperature for ~15-20 s -- this is your
     leak-immune anchor point (captured automatically as the first plateau).
  6. Place the calorimeter in the FIRST bath. Watch the temperature-vs-time panel:
       - GREY background  = still equilibrating, wait.
       - GREEN background = plateau reached, now recording the dwell.
       - BLUE background  = enough dwell recorded, MOVE to the next bath.
  7. Repeat for each bath, coldest -> hottest, then hottest -> coldest,
     ending back at sealed room temperature (second anchor).
  8. Close the plot window to end the session. ALL raw data is saved --
     no cropping. Analyse the plateaus afterwards (the CSV tags each sample
     with its detected state to make this easy).

Open the BLUE valve to atmosphere once finished.
"""

import sys
import os
sys.path.append(os.path.abspath("/home/dorkmaster/Desktop/PHY1035/Tools"))

import time
import csv
import collections
import re

import serial
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from boardFinder import get_serial_connection

plt.close('all')


# =============================================================================
#  CONFIG -- everything you are likely to want to tweak lives here
# =============================================================================

# --- Equilibrium detection -----------------------------------------------
# The rate of temperature change is estimated by a straight-line fit to the
# temperature over the last SLOPE_WINDOW_SECONDS. When |dT/dt| falls below
# EQUIL_SLOPE_THRESHOLD the gas is treated as equilibrated (background -> green).
# It only counts as "moving again" (background -> grey) once |dT/dt| rises above
# the larger REARM_SLOPE_THRESHOLD -- this hysteresis stops sensor noise from
# flickering the state near the threshold.
EQUIL_SLOPE_THRESHOLD  = 0.05   # deg C / s : below this = equilibrium reached
REARM_SLOPE_THRESHOLD  = 0.15   # deg C / s : above this = calorimeter has moved
SLOPE_WINDOW_SECONDS   = 8.0    # s   : rolling window used to estimate dT/dt
MIN_POINTS_FOR_SLOPE   = 4      #     : need at least this many samples to trust it

# --- Dwell ---------------------------------------------------------------
# How long to hold at a plateau (after equilibrium is reached) before the
# background turns blue to prompt moving to the next bath.
DWELL_TARGET_SECONDS   = 20.0   # s

# --- Display -------------------------------------------------------------
PLOT_REFRESH_INTERVAL  = 0.2    # s : redraw interval (logging is EVERY sample)
SHOW_PRESSURE_PANEL    = True   # also show pressure vs time (leak diagnostic)

# Background colours per state (dark enough for white text/lines to show).
COLOUR_SETTLING = '#2b2b2b'     # grey  : wait, still equilibrating
COLOUR_HOLDING  = '#14532d'     # green : plateau reached, recording
COLOUR_READY    = '#1e3a8a'     # blue  : dwell complete, move to next bath

# Bright status-text colours per state.
TEXTCOL_SETTLING = '#bdbdbd'
TEXTCOL_HOLDING  = '#5eff8f'
TEXTCOL_READY    = '#6fa8ff'

# =============================================================================


def robust_mean(values, threshold=2.0):
    """MAD-filtered mean, robust to the occasional sensor glitch."""
    values = np.array(values).flatten()
    median = np.median(values)
    abs_deviation = np.abs(values - median)
    mad = np.median(abs_deviation)
    if mad == 0:
        filtered = values[abs_deviation < threshold]
    else:
        filtered = values[abs_deviation / mad < threshold]
    if filtered.size == 0:
        return np.mean(values)
    return np.mean(filtered)


def extract_floats(text):
    """Pull all numbers (int/decimal/scientific) out of a serial line."""
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    return [float(num) for num in numbers]


def rolling_slope(times_arr, temps_arr, window_s, min_pts):
    """Return dT/dt (deg C / s) over the last window_s seconds, or None."""
    if len(times_arr) < min_pts:
        return None
    t_now = times_arr[-1]
    mask = times_arr >= (t_now - window_s)
    tt = times_arr[mask]
    TT = temps_arr[mask]
    if len(tt) < min_pts or (tt[-1] - tt[0]) <= 0:
        return None
    # slope of a straight-line fit; subtract t0 for numerical conditioning
    return float(np.polyfit(tt - tt[0], TT, 1)[0])


def update_state(state, onset_time, slope, t_now):
    """
    Three-state machine with hysteresis.
      SETTLING -> HOLDING when |slope| < EQUIL_SLOPE_THRESHOLD
      HOLDING  -> READY    when dwell >= DWELL_TARGET_SECONDS
      HOLDING/READY -> SETTLING when |slope| > REARM_SLOPE_THRESHOLD
    Returns (new_state, new_onset_time, dwell_seconds).
    """
    absslope = abs(slope) if slope is not None else np.inf

    if state == 'SETTLING':
        if slope is not None and absslope < EQUIL_SLOPE_THRESHOLD:
            state = 'HOLDING'
            onset_time = t_now

    elif state == 'HOLDING':
        if absslope > REARM_SLOPE_THRESHOLD:
            state = 'SETTLING'
            onset_time = None
        elif onset_time is not None and (t_now - onset_time) >= DWELL_TARGET_SECONDS:
            state = 'READY'

    elif state == 'READY':
        if absslope > REARM_SLOPE_THRESHOLD:
            state = 'SETTLING'
            onset_time = None

    dwell = (t_now - onset_time) if onset_time is not None else 0.0
    return state, onset_time, dwell


def contiguous_runs(mask):
    """Yield (start_idx, stop_idx) for each contiguous True run in a bool array."""
    mask = np.asarray(mask)
    if mask.size == 0:
        return
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    starts = idx[0::2]
    stops = idx[1::2]
    for s, e in zip(starts, stops):
        yield s, e


# -----------------------------------------------------------------------------
#  Serial setup
# -----------------------------------------------------------------------------
distance    = collections.deque(maxlen=3)
pressure    = collections.deque(maxlen=3)
temperature = collections.deque(maxlen=3)

# Per-sample logs (all saved at the end -- nothing is cropped)
times   = []   # s since logging started
V       = []   # volume proxy (syringe position)
P       = []   # pressure, Pa
T       = []   # temperature, deg C
slopes  = []   # estimated dT/dt, deg C / s (NaN until enough points)
states  = []   # 'SETTLING' / 'HOLDING' / 'READY'

# Live status (updated on each new sample, read by the redraw)
cur_slope = np.nan
cur_state = 'SETTLING'
cur_dwell = 0.0

state = 'SETTLING'
onset_time = None

try:
    ser = get_serial_connection(baud=115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)

    # --- short baseline read to set sensible initial axis limits -------------
    print("Gathering initial baseline measurements...")
    baseline_pressures = []
    baseline_temps = []
    for _ in range(50):
        if ser.in_waiting > 0:
            ser.readline()  # distance (discard for baseline)
            p_line = ser.readline().decode('utf-8').strip()
            t_line = ser.readline().decode('utf-8').strip()
            try:
                baseline_pressures.append(np.mean(extract_floats(p_line)))
                baseline_temps.append(np.mean(extract_floats(t_line)) - 273.15)
            except Exception:
                pass
    p_mean = np.mean(baseline_pressures) if baseline_pressures else 1.0e5
    t_mean = np.mean(baseline_temps) if baseline_temps else 20.0

    print("\nProtocol reminder:")
    print("  seal at room T -> hold ~20 s -> cold bath ... hot bath ... back down")
    print("  GREY = wait   GREEN = recording   BLUE = move to next bath")
    print("  Close the plot window to end and save the session.\n")

    # --- figure --------------------------------------------------------------
    if SHOW_PRESSURE_PANEL:
        fig, (ax_T, ax_P) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    else:
        fig, ax_T = plt.subplots(1, 1, figsize=(11, 6))
        ax_P = None

    temp_line, = ax_T.plot([], [], '-o', color='#ff5555', markersize=4, lw=1.2)
    ax_T.set_ylabel('Temperature (deg C)')
    ax_T.set_ylim(t_mean - 25, t_mean + 45)
    ax_T.set_xlim(0, 60)

    if ax_P is not None:
        pres_line, = ax_P.plot([], [], '-o', color='gold', markersize=4, lw=1.2)
        ax_P.set_ylabel('Pressure (Pa)')
        ax_P.set_xlabel('Time (s)')
        ax_P.set_ylim(p_mean * 0.9, p_mean * 1.08)
    else:
        pres_line = None
        ax_T.set_xlabel('Time (s)')

    # Big status banner
    status_text = fig.text(0.5, 0.98, '', ha='center', va='top',
                           fontsize=15, weight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)

    globalStartTime = time.time()
    last_draw = 0.0

    while True:
        if not plt.fignum_exists(fig.number):
            print("Plot window closed, stopping.")
            break

        got_data = False
        if ser.in_waiting > 0:
            # three lines per frame: distance, pressure, temperature
            distance.append(extract_floats(ser.readline().decode('utf-8').strip()))
            pressure.append(extract_floats(ser.readline().decode('utf-8').strip()))
            temperature.append(extract_floats(ser.readline().decode('utf-8').strip()))

            accumTime = round(time.time() - globalStartTime, 4)
            v_val = np.mean(distance)
            p_val = np.mean(pressure)
            t_val = robust_mean(temperature) - 273.15

            times.append(accumTime)
            V.append(v_val)
            P.append(p_val)
            T.append(t_val)

            # equilibrium detection + state machine
            slope = rolling_slope(np.asarray(times), np.asarray(T),
                                  SLOPE_WINDOW_SECONDS, MIN_POINTS_FOR_SLOPE)
            state, onset_time, dwell = update_state(state, onset_time,
                                                    slope, accumTime)

            slopes.append(slope if slope is not None else np.nan)
            states.append(state)

            cur_slope = slope if slope is not None else np.nan
            cur_state = state
            cur_dwell = dwell
            got_data = True

        # --- throttled redraw (logging above happens every sample) -----------
        now = time.time()
        if now - last_draw >= PLOT_REFRESH_INTERVAL and len(times) > 0:
            temp_line.set_data(times, T)
            if pres_line is not None:
                pres_line.set_data(times, P)

            # autoscale to show the WHOLE session so every plateau stays visible
            tmax = max(times[-1], 60)
            ax_T.set_xlim(0, tmax * 1.02)
            ax_T.relim(); ax_T.autoscale_view(scalex=False, scaley=True)
            if ax_P is not None:
                ax_P.set_xlim(0, tmax * 1.02)
                ax_P.relim(); ax_P.autoscale_view(scalex=False, scaley=True)

            # background colour + status banner per state
            if cur_state == 'HOLDING':
                bg, tc = COLOUR_HOLDING, TEXTCOL_HOLDING
                msg = ("HOLDING  --  recording plateau   "
                       "[{:.1f} / {:.0f} s]     dT/dt = {:+.3f} deg C/s"
                       .format(cur_dwell, DWELL_TARGET_SECONDS, cur_slope))
            elif cur_state == 'READY':
                bg, tc = COLOUR_READY, TEXTCOL_READY
                msg = ("READY  --  MOVE calorimeter to NEXT bath     "
                       "dT/dt = {:+.3f} deg C/s".format(cur_slope))
            else:
                bg, tc = COLOUR_SETTLING, TEXTCOL_SETTLING
                sval = "  --  " if np.isnan(cur_slope) else "{:+.3f}".format(cur_slope)
                msg = ("SETTLING  --  wait for temperature to stabilise     "
                       "dT/dt = {} deg C/s".format(sval))

            fig.patch.set_facecolor(bg)
            ax_T.set_facecolor(bg)
            if ax_P is not None:
                ax_P.set_facecolor(bg)
            status_text.set_text(msg)
            status_text.set_color(tc)

            fig.canvas.draw()
            last_draw = now

        fig.canvas.flush_events()
        if not got_data:
            time.sleep(0.005)

except serial.SerialException as e:
    print("Error: {}".format(e))
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()


# -----------------------------------------------------------------------------
#  Save EVERYTHING -- no cropping
# -----------------------------------------------------------------------------
times_a  = np.array(times)
V_a      = np.array(V)
P_a      = np.array(P)
T_a      = np.array(T)
slopes_a = np.array(slopes)
states_a = np.array(states)

timestr = time.strftime("%Y-%m-%d__%H-%M-%S")
folderName = os.path.join(os.getcwd(), 'Pressure vs Temperature data', timestr)
os.makedirs(folderName, exist_ok=True)

# raw arrays
np.save(os.path.join(folderName, 'times.npy'),        times_a)
np.save(os.path.join(folderName, 'temperatures.npy'), T_a)
np.save(os.path.join(folderName, 'pressures.npy'),    P_a)
np.save(os.path.join(folderName, 'volumes.npy'),      V_a)
np.save(os.path.join(folderName, 'dTdt.npy'),         slopes_a)
np.save(os.path.join(folderName, 'states.npy'),       states_a)

# comprehensive CSV: State column lets you pull equilibrium plateaus directly
csv_path = os.path.join(folderName, timestr + '_full.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["Time_s", "Temperature_C", "Pressure_Pa",
                "Volume_proxy", "dTdt_C_per_s", "State"])
    for i in range(len(times_a)):
        w.writerow(["{:.2f}".format(times_a[i]),
                    "{:.3f}".format(T_a[i]),
                    "{:.2f}".format(P_a[i]),
                    "{:.3f}".format(V_a[i]),
                    "" if np.isnan(slopes_a[i]) else "{:.4f}".format(slopes_a[i]),
                    states_a[i]])

# record the config used, for reproducibility
with open(os.path.join(folderName, 'config_used.txt'), 'w') as f:
    f.write("EQUIL_SLOPE_THRESHOLD  = {}\n".format(EQUIL_SLOPE_THRESHOLD))
    f.write("REARM_SLOPE_THRESHOLD  = {}\n".format(REARM_SLOPE_THRESHOLD))
    f.write("SLOPE_WINDOW_SECONDS   = {}\n".format(SLOPE_WINDOW_SECONDS))
    f.write("MIN_POINTS_FOR_SLOPE   = {}\n".format(MIN_POINTS_FOR_SLOPE))
    f.write("DWELL_TARGET_SECONDS   = {}\n".format(DWELL_TARGET_SECONDS))

print("\nSaved full session to:\n  {}".format(folderName))

# -----------------------------------------------------------------------------
#  Summary plot with detected equilibrium regions shaded
# -----------------------------------------------------------------------------
if len(times_a) > 0:
    fig2, (bx_T, bx_P) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    bx_T.plot(times_a, T_a, '-o', color='#ff5555', markersize=3, lw=1.0)
    bx_T.set_ylabel("Temperature (deg C)", color='#ff5555')
    bx_P.plot(times_a, P_a, '-o', color='gold', markersize=3, lw=1.0)
    bx_P.set_ylabel("Pressure (Pa)", color='gold')
    bx_P.set_xlabel("Time (s)")

    equil_mask = np.isin(states_a, ['HOLDING', 'READY'])
    for s, e in contiguous_runs(equil_mask):
        e = min(e, len(times_a) - 1)
        for bx in (bx_T, bx_P):
            bx.axvspan(times_a[s], times_a[e], color='#5eff8f', alpha=0.18)

    bx_T.set_title("Full session -- shaded = detected equilibrium plateaus")
    fig2.tight_layout()
    fig2.savefig(os.path.join(folderName, 'session_summary.png'), dpi=120)
    plt.show()
