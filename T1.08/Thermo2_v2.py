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

# curve_fit is used only for the optional transient extension (to find each
# relaxation's true baseline). If scipy is missing we fall back gracefully.
try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

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

# --- Non-equilibrium transients (optional extension experiment) ----------
# After each move, the gas relaxes towards the new bath temperature along a
# Newton's-law exponential, T(t) = T_bath + amp*exp(-t/tau). The script saves
# each relaxation as its own small CSV in a "non equilibrium transients"
# subfolder. Each file carries a ready-to-fit ln|T - T_bath| column: the student
# plots that against t_since_move_s and reads the time constant straight off a
# straight-line fit as tau = -1 / slope (Excel SLOPE/LINEST, as in Part 1).
#
# The baseline T_bath is found here by a robust three-parameter fit so the
# student's ln column is clean; if scipy is unavailable the script falls back to
# a simpler estimate and the fits are slightly noisier but still usable.
SAVE_TRANSIENTS        = True   # set False to switch the extension off entirely
TRANSIENT_DROP_S       = 1.2    # s : discard this long right after a move
                                #     (handling jolt + pressure spike + probe lag)
TRANSIENT_MIN_STEP_C   = 3.0    # deg C : ignore 'steps' smaller than this, which
                                #     excludes the initial room-temperature settle
TRANSIENT_TAU_GUESS_S  = 13.0   # s : rough tau, only used to cap each block length
TRANSIENT_TMAX_FACTOR  = 4.0    #     keep up to this many guessed-tau past a move
                                #     (a clean decay that levels off, no long tail)
TRANSIENT_LN_FLOOR_C   = 0.15   # deg C : drop |T - T_bath| below this before ln()
                                #     (noise floor; stops ln of near-zero noise)

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
#  Non-equilibrium transients (optional extension experiment)
#
#  Each move produces one relaxation towards the new bath temperature. We save
#  each as its own small CSV inside a "non equilibrium transients" subfolder, so
#  the main data folder stays clean (students doing Part 1 see only the main CSV
#  and this one subfolder). Each file is ready to fit straight away: a student
#  plots ln_T_minus_Tbath against t_since_move_s and reads the time constant off
#  a straight-line fit as tau = -1 / slope (Excel SLOPE/LINEST, as in Part 1).
#
#  The baseline T_bath is found per block by a robust 3-parameter exponential
#  fit, so the ln() column is clean. A demonstrator-only answer key with every
#  fitted tau and R^2 is written alongside (not needed by students).
# -----------------------------------------------------------------------------
def _fit_tau_baseline(rel_t, TT):
    """Return (T_bath, tau, r2) for TT(t) = T_bath + amp*exp(-t/tau).

    Uses scipy if available (robust free baseline); otherwise falls back to a
    tail-mean baseline plus a log-linear slope. Never raises."""
    tail_n = max(5, len(TT) // 4)
    tbath_guess = float(np.mean(TT[-tail_n:]))
    # physical bounds: the asymptote must lie within (a little beyond) the data,
    # and tau within the window. This stops curve_fit running away to nonsense.
    lo, hi = float(TT.min()), float(TT.max())
    span = hi - lo
    tbath_min, tbath_max = lo - 0.5 * span, hi + 0.5 * span
    tau_max = 10.0 * (rel_t[-1] - rel_t[0] + 1.0)
    if _HAVE_SCIPY:
        try:
            def _model(x, tbath, amp, tau):
                return tbath + amp * np.exp(-x / tau)
            p0 = [tbath_guess, TT[0] - tbath_guess, TRANSIENT_TAU_GUESS_S]
            popt, _ = curve_fit(
                _model, rel_t, TT, p0=p0, maxfev=20000,
                bounds=([tbath_min, -5 * span - 5, 0.1],
                        [tbath_max,  5 * span + 5, tau_max]))
            tbath, amp, tau = float(popt[0]), float(popt[1]), float(popt[2])
            resid = TT - _model(rel_t, *popt)
            ss = np.sum((TT - TT.mean()) ** 2)
            r2 = 1.0 - np.sum(resid ** 2) / ss if ss > 0 else float('nan')
            if 0 < tau < tau_max and abs(amp) > TRANSIENT_LN_FLOOR_C:
                return tbath, tau, r2
        except Exception:
            pass
    # fallback: tail-mean baseline, slope of ln|T - T_bath|
    tbath = tbath_guess
    y = np.abs(TT - tbath)
    g = y > TRANSIENT_LN_FLOOR_C
    if np.count_nonzero(g) >= 4:
        slope, _ = np.polyfit(rel_t[g], np.log(y[g]), 1)
        tau = -1.0 / slope if slope < 0 else float('nan')
    else:
        tau = float('nan')
    return tbath, tau, float('nan')


if SAVE_TRANSIENTS and len(times_a) > 0:
    settle_runs = list(contiguous_runs(states_a == 'SETTLING'))
    hold_runs   = list(contiguous_runs(np.isin(states_a, ['HOLDING', 'READY'])))

    trans_dir = os.path.join(folderName, 'non equilibrium transients')

    seg_no = 0
    answer_key = []   # (file, direction, T_bath, dT_step, n, tau, r2)
    for s, e in settle_runs:
        following = [h for h in hold_runs if h[0] > e]
        if not following:
            continue                      # trailing settle with no plateau after
        hs, he = following[0]
        T_start = float(T_a[s])

        # clip the block to a few time constants past the move (clean decay, no
        # long flat tail), bounded by the end of the following plateau
        t0 = times_a[s]
        t_cap = t0 + TRANSIENT_DROP_S + TRANSIENT_TAU_GUESS_S * TRANSIENT_TMAX_FACTOR
        b = min(he, int(np.searchsorted(times_a, t_cap)))

        rel_t = times_a[s:b + 1] - t0
        TT    = T_a[s:b + 1]
        PP    = P_a[s:b + 1]
        keep  = rel_t >= TRANSIENT_DROP_S
        if np.count_nonzero(keep) < 8:
            continue                      # too short to fit

        rel_t = rel_t[keep] - TRANSIENT_DROP_S   # start clock at first kept sample
        TT, PP = TT[keep], PP[keep]

        # robust baseline + tau for this block
        T_bath, tau, r2 = _fit_tau_baseline(rel_t, TT)

        if abs(T_bath - T_start) < TRANSIENT_MIN_STEP_C:
            continue                      # excludes the initial room-temperature settle

        dT = TT - T_bath
        # clean ln column: blank where |dT| is at/below the noise floor
        ln_dT = [('{:.4f}'.format(np.log(abs(v))) if abs(v) > TRANSIENT_LN_FLOOR_C else '')
                 for v in dT]

        seg_no += 1
        if seg_no == 1:
            os.makedirs(trans_dir, exist_ok=True)

        direction = 'heating' if T_bath > T_start else 'cooling'
        fname = os.path.join(trans_dir, 'transient_{:02d}.csv'.format(seg_no))
        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_since_move_s', 'ln_T_minus_Tbath',
                        'T_minus_Tbath_C', 'Temperature_C', 'Pressure_Pa', 'T_bath_C'])
            for i in range(len(rel_t)):
                w.writerow(['{:.2f}'.format(rel_t[i]),
                            ln_dT[i],
                            '{:.3f}'.format(dT[i]),
                            '{:.3f}'.format(TT[i]),
                            '{:.2f}'.format(PP[i]),
                            '{:.3f}'.format(T_bath)])
        answer_key.append((os.path.basename(fname), direction, T_bath,
                           T_bath - T_start, len(rel_t), tau, r2))

    if seg_no > 0:
        # demonstrator-only answer key (students don't need this)
        key_path = os.path.join(trans_dir, '_answer_key_demonstrator.csv')
        with open(key_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file', 'direction', 'T_bath_C', 'step_dT_C',
                        'n_points', 'tau_s', 'R2'])
            for row in answer_key:
                fn, d, tb, dstep, n, tau, r2 = row
                w.writerow([fn, d, '{:.2f}'.format(tb), '{:+.1f}'.format(dstep),
                            n,
                            '' if tau != tau else '{:.1f}'.format(tau),
                            '' if r2 != r2 else '{:.3f}'.format(r2)])
        taus = [r[5] for r in answer_key if r[5] == r[5]]
        tau_msg = ('mean tau = {:.1f} s'.format(np.mean(taus)) if taus else 'tau: n/a')
        print("Saved {:d} relaxation transients ({}) to:\n  {}".format(
            seg_no, tau_msg, trans_dir))
    else:
        print("No usable relaxation transients found (no clear temperature steps).")

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