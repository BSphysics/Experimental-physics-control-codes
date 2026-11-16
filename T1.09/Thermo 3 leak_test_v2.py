# -*- coding: utf-8 -*-
"""
leak_test.py: measure the gas leak rate of a Thermo 3 heat engine rig

Why: while the mass is on the syringe, the gas is above atmospheric pressure and
slowly leaks out (mostly past the glass piston). Gas that leaks out lowers the
piston, so the measured net lift (Part B) comes out too small. The leak gets faster
as the extra pressure increases, so it matters most for the heaviest masses.

How the test works (about 3 minutes):
  1. Calorimeter resting in the COLD (room-temperature) bath, no mass:
     30 s baseline to measure any slow thermal drift.
  2. Place the test mass (use the heaviest mass students will use).
  3. Mass on for 75 s. The first 15 s are ignored while the piston settles; the
     next 60 s give the leak rate.
  4. Remove the mass: 30 s baseline to measure the drift again.

The cold bath is used throughout because its large heat capacity keeps the gas
temperature steady. In air, handling and draughts change the temperature; in the
hot bath, the gas keeps heating and the bath keeps cooling. Either would swamp a
slow leak.

The amount of gas is tracked as pV (an "equivalent volume" V*p/p_ref). This counts
the leak correctly even if friction holds the piston still and only the pressure
falls.

ISOLATED MODE: answer "y" to "Is the calorimeter isolated?" to run the same test with
the valve to the calorimeter CLOSED. Only the syringe, manifold, pressure sensor and
short tubing then hold gas. Comparing an isolated and a normal test on the same rig
with the same mass shows where the leak is:
  - similar leak rate (mL/s)  -> leak is on the syringe side
  - much lower when isolated  -> leak is in the calorimeter, bung or its tubing
In isolated mode the gas volume is not known from the calibration, so it is
estimated from the loading step using Boyle's law: V = (volume drop) x p / (pressure
step). In normal mode the same estimate is printed as a check against the
calibrated volume.

Results are shown on screen and appended to 'Leak test data/leak_test_log.csv'.
Raw data for each test are saved in 'Leak test data/<date-time>/'.

Nov 2026
"""

import sys
import os
import re
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("/home/dorkmaster/Desktop/PHY1035/Tools"))
plt.style.use('dark_background')

# ======================= SETTINGS ===================================================
# Use the same position-to-volume calibration as 'Thermo 3 heat engine_v3.py'
volume_fn = np.poly1d([-0.6531, 410.95])

G = 9.81
BASELINE_S = 30          # mass off, before
BASELINE_SKIP_S = 10     # ignored at the start of each baseline
LOADED_S = 75            # mass on
LOADED_SKIP_S = 15       # ignored while the piston settles after loading
FINAL_S = 30             # mass off, after

# Pass/fail is judged on the gas lost during a typical "mass on" period of the heat
# engine cycle (~8 s cold + 15 s hot + handling), at the heaviest mass in use.
MASS_ON_PERIOD_S = 30
GOOD_LOSS_ML = 1.0       # below this the leak is negligible (~1.5 mm of piston travel)
MARGINAL_LOSS_ML = 3.0   # below this it is usable, but net lift will read a bit low
# ====================================================================================


def piston_area():
    return abs(volume_fn.coeffs[0]) * 1e-3            # m^2 (gradient in mL/mm)


# ----------------------------------------------------------------------------------
# Serial helpers (same as the heat engine script)
# ----------------------------------------------------------------------------------
def extract_floats(text):
    return [float(n) for n in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)]


def first_float(raw_line):
    vals = extract_floats(raw_line.decode('utf-8', errors='ignore').strip())
    return vals[0] if vals else None


def read_frame(ser):
    """One (distance, pressure, temperature) frame, re-syncing on the pressure line."""
    prev = None
    for _ in range(6):
        val = first_float(ser.readline())
        if val is not None and val > 5e4 and prev is not None:
            temp = first_float(ser.readline())
            return None if temp is None else (prev, val, temp)
        prev = val
    return None


# ----------------------------------------------------------------------------------
# Guided stages
# ----------------------------------------------------------------------------------
class LeakTestStages:
    """Timed 'hold still' stages, plus mass on/off actions detected from pressure."""

    def __init__(self, mass_g, isolated=False):
        self.mass_g = mass_g
        where = ('Calorimeter ISOLATED (valve closed).' if isolated else
                 'Calorimeter resting in the COLD bath.')
        self.thr = max(300.0, 0.5 * (mass_g * 1e-3) * G / piston_area())
        self.stages = [
            dict(key='before', dur=BASELINE_S,
                 text=f'{where} NO mass on the syringe.\n'
                      'Do not touch the apparatus.'),
            dict(key='mass_on', event=True,
                 text=f'NOW: gently place the {mass_g:g} g mass on the syringe'),
            dict(key='loaded', dur=LOADED_S,
                 text=f'Mass ON. {where}\nDo not touch the apparatus.'),
            dict(key='mass_off', event=True,
                 text='NOW: lift the mass off the syringe'),
            dict(key='after', dur=FINAL_S,
                 text=f'Mass OFF. {where}\nDo not touch the apparatus.'),
        ]
        self.k = 0
        self.t, self.d, self.p = [], [], []
        self.start_t = None
        self.start_i = 0
        self.marks = {}              # stage key -> (start time, end time)
        self.disturbed = []          # times of unexpected pressure jumps

    @property
    def finished(self):
        return self.k >= len(self.stages)

    def _advance(self, t):
        key = self.stages[self.k]['key']
        self.marks[key] = (self.start_t, t)
        self.k += 1
        self.start_t = t
        self.start_i = len(self.p) - 1

    def update(self, t, d, p):
        self.t.append(t)
        self.d.append(d)
        self.p.append(p)
        if self.start_t is None:
            self.start_t = t
        if self.finished:
            return
        st = self.stages[self.k]
        jump = len(self.p) > 1 and abs(self.p[-1] - self.p[-2]) > self.thr
        if st.get('event'):
            ref = self.p[self.start_i:-1]
            if len(ref) >= 1:
                dp = self.p[-1] - np.median(ref)
                if (st['key'] == 'mass_on' and dp > self.thr) or \
                   (st['key'] == 'mass_off' and -dp > self.thr):
                    self._advance(t)
        else:
            # a pressure jump during a 'hold still' stage means someone touched the mass
            if jump and len(self.p) - 1 > self.start_i + 1:
                self.disturbed.append(round(t, 1))
            if t - self.start_t >= st['dur']:
                self._advance(t)

    def message(self, now):
        if self.finished:
            return ('TEST COMPLETE: close this window to see the result.', 'darkgreen')
        st = self.stages[self.k]
        n = len(self.stages)
        if st.get('event'):
            text, colour = f'STEP {self.k + 1} of {n}: {st["text"].upper()}', 'darkgreen'
        else:
            left = max(0, st['dur'] - (now - (self.start_t or now)))
            text = f'STEP {self.k + 1} of {n}: {st["text"]}\n{np.ceil(left):.0f} s remaining'
            colour = 'dimgrey'
        if self.disturbed:
            text += '\nWarning: the apparatus was disturbed. Consider repeating the test.'
            colour = 'darkred'
        return text, colour


# ----------------------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------------------
def acquire(mass_g, isolated=False):
    import serial  # noqa: F401
    from boardFinder import get_serial_connection

    ser = get_serial_connection(baud=115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)
    ser.readline()                     # discard a possibly partial line

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.12, right=0.96, hspace=0.12)
    lineV, = ax1.plot([], [], 'o-', color='deepskyblue', ms=3, lw=0.8)
    lineP, = ax2.plot([], [], 'o-', color='orange', ms=3, lw=0.8)
    ax1.set_ylabel('Volume (mL)')
    ax2.set_ylabel('Pressure (Pa)')
    ax2.set_xlabel('Time (s)')
    instr = fig.text(0.5, 0.90, '', ha='center', va='center', fontsize=13,
                     fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.6', facecolor='dimgrey'))
    plt.show(block=False)

    stages = LeakTestStages(mass_g, isolated)
    t0 = time.time()
    last_msg = None
    try:
        while plt.fignum_exists(fig.number):
            now = time.time() - t0
            redraw = False
            if ser.in_waiting >= 1:
                frame = read_frame(ser)
                if frame is not None:
                    stages.update(now, frame[0], frame[1])
                    V = volume_fn(np.array(stages.d))
                    lineV.set_data(stages.t, V)
                    lineP.set_data(stages.t, stages.p)
                    for ax in (ax1, ax2):
                        ax.relim()
                        ax.autoscale_view()
                    redraw = True
            msg = stages.message(now)
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
    return stages


# ----------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------
def fit_slope(t, y):
    """Least-squares slope and its standard error."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = max(1, len(t) - 2)
    resid = y - A @ coef
    se = np.sqrt(np.sum(resid ** 2) / dof / np.sum((t - t.mean()) ** 2))
    return coef[0], se, coef[1], t.mean()


def boyle_volume(t, V, p, marks, w_before):
    """Gas volume (mL, before loading) from the loading step: V1 = dV * p2 / (p2 - p1).

    Uses the mass-off baseline and the 2-5 s just after loading (fast enough that
    little gas has leaked, slow enough that the compression heating has gone).
    """
    t_load = marks['loaded'][0]
    w = (t >= t_load + 2) & (t <= t_load + 5)
    if w.sum() < 3:
        return None
    V1, p1 = np.median(V[w_before]), np.median(p[w_before])
    V2, p2 = np.median(V[w]), np.median(p[w])
    dV, dp = V1 - V2, p2 - p1
    if dV <= 0 or dp <= 0:
        return None
    return dV * p2 / dp


def analyse(t, d, p, marks, mass_g, isolated=False):
    t, d, p = map(np.asarray, (t, d, p))
    V_cal = volume_fn(d)

    def window(key, skip):
        a, b = marks[key]
        return (t >= a + skip) & (t <= b)

    w_before = window('before', BASELINE_SKIP_S)
    w_loaded = window('loaded', LOADED_SKIP_S)
    w_after = window('after', BASELINE_SKIP_S)

    V_boyle = boyle_volume(t, V_cal, p, marks, w_before)
    note = ''
    if not isolated:
        V = V_cal                             # calibrated absolute volume
    elif V_boyle is not None and 5 < V_boyle < 600:
        # isolated: absolute volume from the Boyle estimate, changes from the piston
        V = V_boyle + (V_cal - np.median(V_cal[w_before]))
    else:
        V = V_cal - np.median(V_cal[w_before]) + 50.0
        note = ('Could not estimate the isolated gas volume from the loading step; '
                'only piston movement is reliable (pressure-only leakage under-counted).')

    p_ref = np.median(p[w_loaded])
    Veq = V * p / p_ref                       # amount of gas, as mL at p_ref

    s_b, e_b, *_ = fit_slope(t[w_before], Veq[w_before])
    s_l, e_l, c_l, tm_l = fit_slope(t[w_loaded], Veq[w_loaded])
    s_a, e_a, *_ = fit_slope(t[w_after], Veq[w_after])

    drift = 0.5 * (s_b + s_a)                 # thermal drift with no extra pressure
    leak = -(s_l - drift)                     # mL/s lost because of the extra pressure
    leak_err = np.sqrt(e_l ** 2 + 0.25 * (e_b ** 2 + e_a ** 2))
    dp = p_ref - 0.5 * (np.median(p[w_before]) + np.median(p[w_after]))
    k = leak / (dp / 1000)                    # mL/s per kPa of extra pressure
    k_err = leak_err / (dp / 1000)

    return dict(V=V, Veq=Veq, p_ref=p_ref, V_boyle=V_boyle,
                V_used=float(np.median(V[w_before])), note=note, dp=dp, drift=drift, drift_err=0.5 * np.hypot(e_b, e_a),
                drift_diff=s_a - s_b, leak=leak, leak_err=leak_err, k=k, k_err=k_err,
                fit=(s_l, c_l, tm_l), windows=(w_before, w_loaded, w_after),
                slopes=(s_b, s_l, s_a))


def predicted_losses(k, masses_g):
    A = piston_area()
    rows = []
    for m in masses_g:
        dp_kpa = (m * 1e-3) * G / A / 1000
        loss = k * dp_kpa * MASS_ON_PERIOD_S      # mL
        rows.append((m, dp_kpa, loss, loss / abs(volume_fn.coeffs[0])))
    return rows


def verdict(loss_ml):
    if loss_ml < GOOD_LOSS_ML:
        return 'GOOD', 'lime'
    if loss_ml < MARGINAL_LOSS_ML:
        return 'MARGINAL', 'orange'
    return 'FAIL', 'red'


def plot_result(t, res, mass_g, rig, folder, verdict_text, colour, table, isolated=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    wb, wl, wa = res['windows']
    ax.plot(t, res['Veq'], '.', color='grey', ms=4, label='all data')
    for w, c, lab in [(wb, 'deepskyblue', 'baseline (mass off)'),
                      (wl, 'orange', 'used for leak rate (mass on)'),
                      (wa, 'deepskyblue', None)]:
        ax.plot(np.asarray(t)[w], res['Veq'][w], 'o', color=c, ms=4, label=lab)
    s_l, c_l, tm_l = res['fit']
    tl = np.asarray(t)[wl]
    ax.plot(tl, c_l + s_l * (tl - tm_l), '-', color='white', lw=2,
            label=f'fit: {s_l:+.3f} mL/s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amount of gas, as volume at the loaded pressure (mL)')
    mode = ' (calorimeter ISOLATED)' if isolated else ''
    ax.set_title(f'Leak test{mode}, rig {rig}, {mass_g:g} g: {verdict_text}', color=colour,
                 fontsize=14, fontweight='bold')
    txt = (f"Leak rate at +{res['dp']:.0f} Pa: {res['leak']:.3f} ± {res['leak_err']:.3f} mL/s\n"
           f"Leak per kPa: {res['k']:.3f} ± {res['k_err']:.3f} mL/s/kPa\n"
           f"Thermal drift (mass off): {res['drift']:+.3f} mL/s\n"
           f"Gas volume used: {res['V_used']:.0f} mL\n\n"
           f"Predicted gas lost in {MASS_ON_PERIOD_S} s with the mass on:\n" +
           '\n'.join(f'  {m:4.0f} g: {loss:4.1f} mL  ({mm:4.1f} mm)'
                     for m, _, loss, mm in table))
    ax.text(0.01, 0.02, txt, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, 'leak test result.png'), dpi=150)
    return fig


def main():
    rig = input("Enter the rig / bench number: ").strip() or 'unknown'
    while True:
        try:
            mass_g = float(input("Enter the test mass in grams "
                                 "(use the heaviest mass students will use, e.g. 250): "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        ans = input("Is the calorimeter isolated? [Y/N]: ").strip().lower()
        if ans in ('y', 'n'):
            break
        print("Please type y or n, then press Enter.")
    isolated = ans == 'y'

    print("\nBEFORE STARTING:")
    print("  1. Put the calorimeter in the COLD (room-temperature) water bath and rest it")
    print("     there so it does not need holding. Leave it for at least 1 minute.")
    if isolated:
        print("  2. CLOSE the valve to the calorimeter, so only the syringe side holds gas.")
    else:
        print("  2. Make sure the valve to the calorimeter is OPEN.")
    print("  3. Turn the BLUE valve so the system is open to the atmosphere and set the")
    print("     glass syringe to about 25 mL. Then close the BLUE valve.")
    print("  4. Make sure the test mass is ready, but NOT on the syringe.")
    if isolated:
        print("  Do not touch the syringe or tubing during the test: with the calorimeter")
        print("  isolated the gas volume is small, so hand warmth affects it more.")
    input("Press Enter to start the test (about 3 minutes)...")

    stages = acquire(mass_g, isolated)
    if not stages.finished:
        print("Test not completed: no result saved. Please run it again.")
        return

    t, d, p = np.array(stages.t), np.array(stages.d), np.array(stages.p)
    folder = os.path.join(os.getcwd(), 'Leak test data', time.strftime("%Y-%m-%d__%H-%M-%S"))
    os.makedirs(folder, exist_ok=True)
    for name, arr in [('times', t), ('distances', d), ('pressures', p)]:
        np.save(os.path.join(folder, name + '.npy'), arr)

    res = analyse(t, d, p, stages.marks, mass_g, isolated)
    table = predicted_losses(res['k'], [50, 100, 150, 200, 250, 300])
    loss_test = res['k'] * (res['dp'] / 1000) * MASS_ON_PERIOD_S
    v_text, colour = verdict(loss_test)
    if stages.disturbed:
        v_text += ' (disturbed: repeat)'

    mode = 'calorimeter ISOLATED' if isolated else 'calorimeter connected'
    print(f"\nRig {rig}, {mode}, test mass {mass_g:g} g (extra pressure {res['dp']:.0f} Pa)")
    print(f"Leak rate:      {res['leak']:.3f} ± {res['leak_err']:.3f} mL/s")
    print(f"Leak per kPa:   {res['k']:.3f} ± {res['k_err']:.3f} mL/s per kPa")
    print(f"Thermal drift:  {res['drift']:+.3f} mL/s "
          f"(before vs after differ by {res['drift_diff']:+.3f} mL/s)")
    print(f"Gas lost in a {MASS_ON_PERIOD_S} s mass-on period at {mass_g:g} g: "
          f"{loss_test:.1f} mL")
    vb = res['V_boyle']
    if isolated:
        print(f"Gas volume (estimated from the loading step): "
              f"{'n/a' if vb is None else f'{vb:.0f} mL'}")
    else:
        print(f"Gas volume: {res['V_used']:.0f} mL from calibration, "
              f"{'n/a' if vb is None else f'{vb:.0f} mL'} from the loading step (check)")
    if res['note']:
        print("Note: " + res['note'])
    print(f"VERDICT: {v_text}")
    if abs(res['drift_diff']) > max(0.02, 3 * res['leak_err']):
        print("Note: the drift changed between the start and end of the test. The "
              "calorimeter may not have been settled in the bath; consider repeating.")
    if stages.disturbed:
        print(f"Note: the apparatus was disturbed at t = {stages.disturbed} s.")

    log = os.path.join(os.path.dirname(folder), 'leak_test_log.csv')
    row = {'date-time': os.path.basename(folder), 'rig': rig,
           'calorimeter isolated': 'yes' if isolated else 'no',
           'test mass (g)': mass_g,
           'extra pressure (Pa)': round(float(res['dp']), 0),
           'leak rate (mL/s)': round(float(res['leak']), 4),
           'leak rate error (mL/s)': round(float(res['leak_err']), 4),
           'leak per kPa (mL/s/kPa)': round(float(res['k']), 4),
           'thermal drift (mL/s)': round(float(res['drift']), 4),
           f'loss in {MASS_ON_PERIOD_S} s at test mass (mL)': round(float(loss_test), 2),
           'gas volume used (mL)': round(float(res['V_used']), 1),
           'gas volume from loading step (mL)': '' if vb is None else round(float(vb), 1),
           'verdict': v_text}
    if os.path.exists(log):               # never append under a different header
        with open(log, newline='') as f:
            header = next(csv.reader(f), [])
        if header != list(row):
            log = log.replace('.csv', time.strftime('_%Y-%m-%d__%H-%M-%S.csv'))
    new = not os.path.exists(log)
    with open(log, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"Result added to: {log}")

    plot_result(t, res, mass_g, rig, folder, v_text, colour, table, isolated)
    plt.show()


if __name__ == '__main__':
    main()
