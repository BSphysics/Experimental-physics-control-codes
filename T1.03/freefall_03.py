# -*- coding: utf-8 -*-
"""
freefall_03.py

Freefall-WITH-FAN control script for the PHY1035 practical.

"""
import sys
import os

# Add the Tools folder to the Python path
sys.path.append(os.path.abspath("/home/dorkmaster/Desktop/PHY1035/Tools"))
# make sure the helper module is importable (same dir as this script)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from os import system
import subprocess
import pigpio
import serial, time, csv, os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Cursor
from time import sleep
from datetime import datetime
from boardFinder import get_serial_connection

import freefall_workbook as fw   # v2: extended rows, gated reduced-chi2, auto-t0

os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"

plt.close('all')

# ---------------------------------------------------------------------------
# Fan + acquisition timing (named constants so they are easy to tune).
# ---------------------------------------------------------------------------
PWM_GPIO = 12        # hardware-PWM pin driving the fan
PWM_FREQ = 25000     # Hz

FAN_ON_IDX = 0       # start the fan at the first acquisition sample...
DROP_IDX   = 100     # ...and release ~100 samples later (~2 s at this Arduino's
                     # rate), so the air column is established and there is a
                     # pre-release baseline. Increase if the fan needs longer to
                     # reach steady airflow.
N_SAMPLES  = 500     # total loop length. Deliberately generous: near the fan
                     # threshold a projectile can fall very slowly, and the
                     # window must not clip the fall before the student truncates.

# NOTE: the original freefall_03 turned the fan OFF mid-loop (idx == 300). That
# is a latent bug for THIS experiment: near the threshold a slow fall can still
# be in progress at sample 300, so the fan would cut mid-fall and change the
# condition being measured. The fan now stays on for the whole acquisition and
# is switched off the instant the loop ends.

servo1_start = 1000
servo2_start = 2200
servo1_open = 2000
servo2_open = 2500

# ---------------------------------------------------------------------------
# STEP 0: which projectile, and what fan power, does this run belong to?
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  PHY1035 Freefall (with fan) - new run")
print("=" * 60)

projectile_name = input(
    "\n  Projectile / experiment name\n"
    "  (e.g. 'golf', 'hollow' or 'stress')\n"
    "  > "
).strip()
while not projectile_name:
    projectile_name = input("  Please enter a non-empty name > ").strip()


def _ask_fan_pct():
    """Prompt for an integer fan power 0-100 %, re-asking until valid."""
    while True:
        raw = input(
            "\n  Fan power for this run (whole number, 0-100 %)\n"
            "  (0 = fan off / baseline; sweep this to find the threshold)\n"
            "  > "
        ).strip().rstrip('%').strip()   # tolerate a trailing '%'
        try:
            val = int(raw)
        except ValueError:
            print("  Please enter a whole number between 0 and 100.")
            continue
        if 0 <= val <= 100:
            return val
        print("  Out of range - must be between 0 and 100.")


fan_pct = _ask_fan_pct()

# Fold the fan setting into the routing label so each (projectile, fan%)
# condition gets its OWN workbook and its own five repeats.
run_label = f"{projectile_name}_fan{fan_pct:03d}"

# where the workbooks live (one file per condition)
dateStamp = datetime.now().strftime('%Y_%m_%d')
WORKBOOK_FOLDER = os.path.join(os.getcwd(), "Free fall data", dateStamp, "Excel workbooks")
os.makedirs(WORKBOOK_FOLDER, exist_ok=True)
wb_path = fw.workbook_path(WORKBOOK_FOLDER, run_label)

# report current status of this condition's file so the student knows where they are
if os.path.exists(wb_path):
    import openpyxl
    _ws = openpyxl.load_workbook(wb_path)[fw.SHEET_NAME]
    status = fw.runs_status(_ws)
    filled = [r for r, done in status.items() if done]
    suggested = fw.next_empty_run(_ws)
    print(f"\n  Existing file found for '{run_label}'.")
    print(f"  Runs already saved: {filled if filled else 'none'}")
    if suggested is None:
        print("  NOTE: all 5 runs are already filled. You'll be asked to confirm an overwrite.")
    else:
        print(f"  This drop will be saved as Run {suggested} (unless you choose otherwise).")
else:
    print(f"\n  No file yet for '{run_label}' - this will create it as Run 1.")
    suggested = 1

print(f"\n  Condition: projectile '{projectile_name}' at {fan_pct}% fan.")
print("  Get the projectile in place, then press Enter to arm the drop...")
input()

# ---------------------------------------------------------------------------
# connect to the servos + fan PWM
# ---------------------------------------------------------------------------
pi = pigpio.pi()
sleep(2.2)
pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

# fan PWM (GPIO12 hardware PWM; independent of the 50 Hz servo pins 22/23)
pi.set_PWM_frequency(PWM_GPIO, PWM_FREQ)


def set_fan(percent):
    """Set fan power as a percentage 0-100 (hardware-PWM duty on PWM_GPIO)."""
    if not 0 <= percent <= 100:
        raise ValueError("Fan percent must be between 0 and 100")
    pi.set_PWM_dutycycle(PWM_GPIO, int(percent / 100 * 255))


set_fan(0)   # guarantee the fan is off before we begin

# Open serial connection
ARDser = get_serial_connection(baud=115200, timeout=1)
while True:
    if ARDser.in_waiting:
        line = ARDser.readline().decode("utf-8").strip()
        if line == "READY":
            break
time.sleep(2)
ARDser.reset_input_buffer()
ARDser.reset_output_buffer()
time.sleep(1)

timer, distance = [], []
release_index = None   # robustly record the true release point

print('\n\n Here we go \n\n')
for idx in range(0, N_SAMPLES):

    if idx == FAN_ON_IDX:
        # fan on and left running for the WHOLE fall (see note at top)
        set_fan(fan_pct)

    if idx == DROP_IDX:
        print('\n\n **DROP** \n\n')
        pi.set_servo_pulsewidth(23, servo1_open)
        pi.set_servo_pulsewidth(22, servo2_open)
        # record how many valid readings we had at the moment of release,
        # rather than assuming a fixed offset (which silently breaks if an
        # early serial line was dropped)
        release_index = len(distance)

    try:
        ser_bytes = ARDser.readline()
        try:
            decoded_bytes = (ser_bytes[0:len(ser_bytes) - 2].decode("utf-8")).split(',')
        except (UnicodeDecodeError, IndexError):
            continue
        if len(decoded_bytes) != 2 or decoded_bytes[0] == '' or decoded_bytes[1] == '' or decoded_bytes[1] == 'READY':
            continue
        timer.append(float(decoded_bytes[0]))
        distance.append(float(decoded_bytes[1]))
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
        break

ARDser.close()

# fan off the instant acquisition ends, before any interactive clicking
set_fan(0)

# fall back to the old behaviour if for some reason release wasn't recorded
if release_index is None or release_index < 1:
    release_index = DROP_IDX - 1

delta = release_index
timer = np.asarray(timer)
distances = np.asarray(distance[delta:])

# ---------------------------------------------------------------------------
# interactive truncation (unchanged, title now names the condition)
# ---------------------------------------------------------------------------
norm = plt.Normalize(0, 2000)
fig, ax = plt.subplots()
sc = ax.scatter(timer[delta:] - timer[delta], distances, c=distances, cmap=cm.viridis, norm=norm)
ax = plt.gca()
# auto-scale so slow / fan-slowed projectiles are not silently clipped
_tmax = float((timer[delta:] - timer[delta]).max()) if len(timer) > delta else 1500
ax.set_xlim([0, max(1500, _tmax * 1.05)])
ax.set_ylim([0, 2200])
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Distance (mm)')
ax.set_title(f"{projectile_name} @ {fan_pct}% fan: click the last point still in freefall")

live_cursor = ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
right_cursor = ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, visible=False)

clicks = []

def on_move(event):
    if event.inaxes == ax:
        live_cursor.set_xdata([event.xdata])
        plt.draw()

def on_click(event):
    if event.inaxes != ax:
        return
    x = event.xdata
    clicks.clear()
    clicks.append(x)
    right_cursor.set_xdata([x])
    right_cursor.set_visible(True)
    print(f"Freefall end set at x = {x:.2f}")
    plt.draw()
    plt.close()

cid_move = fig.canvas.mpl_connect('motion_notify_event', on_move)
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# guard against the window being closed without a click
if not clicks:
    print("\n  No point was selected - nothing will be saved for this run.")
    set_fan(0)                                   # (already off; belt and braces)
    pi.set_servo_pulsewidth(23, servo1_start)
    pi.set_servo_pulsewidth(22, servo2_start)
    print('READY')
    sys.exit(0)

t_axis = np.array(timer[delta:] - timer[delta])
idx_end = np.argmin(np.abs(t_axis - clicks[0]))
time_end = t_axis[idx_end]

truncated_time = t_axis[:idx_end]
truncated_distance = distances[:idx_end]

plt.close('all')
pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

# ---------------------------------------------------------------------------
# save raw .npy + truncated .txt (condition stamped into the filenames so raw
# files are self-documenting). Raw data is grouped by projectile; the fan %
# lives in the filename stem.
# ---------------------------------------------------------------------------
folder_path = os.path.join(os.getcwd(), 'Free fall data', dateStamp, projectile_name + ' raw data')
os.makedirs(folder_path, exist_ok=True)
timeStamp = datetime.now().strftime('%H_%M_%S')
stem = fw._safe_name(run_label)   # e.g. golf_fan050

txt_path = os.path.join(folder_path, f"{timeStamp}_{stem}_truncated_data.txt")
np.savetxt(txt_path, np.column_stack((truncated_time, truncated_distance)),
           delimiter=",", header="Time,Distance", comments='')

np.save(os.path.join(folder_path, f"{timeStamp}_{stem}_raw_times.npy"), timer)
np.save(os.path.join(folder_path, f"{timeStamp}_{stem}_raw_distances.npy"), distances)

# ---------------------------------------------------------------------------
# STEP N: confirm, then route into the correct per-condition workbook
# ---------------------------------------------------------------------------
run_to_write = suggested
overwrite = False
if run_to_write is None:
    # file already full - force an explicit choice
    print("\n  All 5 runs already exist for this condition.")
    ans = input("  Type a run number (1-5) to OVERWRITE, or Enter to skip saving to workbook: ").strip()
    if ans.isdigit() and 1 <= int(ans) <= 5:
        run_to_write = int(ans)
        overwrite = True
    else:
        run_to_write = None

if run_to_write is not None:
    print("\n" + "-" * 60)
    print(f"  About to save Run {run_to_write} of 5 for '{run_label}'")
    print(f"  (projectile '{projectile_name}', {fan_pct}% fan, "
          f"{len(truncated_time)} data points) into:")
    print(f"    {wb_path}")
    print("-" * 60)
    confirm = input("  Press Enter to confirm, or type 'x' to discard this run: ").strip().lower()
    if confirm == 'x':
        print("  Discarded - workbook not modified. (Raw .npy/.txt files were still saved.)")
    else:
        try:
            saved_path, saved_run, status = fw.write_run(
                WORKBOOK_FOLDER, run_label,
                truncated_time, truncated_distance,
                run_number=run_to_write, overwrite=overwrite,
            )
            done = [r for r, ok in status.items() if ok]
            print(f"\n  Saved as Run {saved_run}. Runs now complete: {done}")
            if len(done) == 5:
                print("  All 5 repeats collected for this condition - ready to analyse.")
                try:
                    import openpyxl as _ox
                    _t0 = _ox.load_workbook(saved_path)[fw.SHEET_NAME][fw.T0_CELL].value
                    if _t0 is not None:
                        print(f"  Auto-estimated release delay t0 = {float(_t0)*1000:.1f} ms "
                              f"(written to cell {fw.T0_CELL}; adjust if needed).")
                except Exception:
                    pass
        except Exception as e:
            print(f"\n  Could not write to workbook: {e}")
            print("  (Raw .npy/.txt files were still saved, so no data is lost.)")

set_fan(0)   # final safety: fan definitely off
print('\nREADY')
