# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:27:51 2024
@author: BES

Requires freefall_workbook.py in the same folder (or on the path).
"""
import sys
import os

# Add the other folder to the Python path
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
from boardFinder import get_serial_connection

import freefall_workbook as fw   # v2: extended rows, gated reduced-chi2, auto-t0   # <-- new helper module

os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"

plt.close('all')

N_SAMPLES = 150

# ---------------------------------------------------------------------------
# STEP 0: Ask which experiment / projectile this run belongs to.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  PHY1035 Freefall - new run")
print("=" * 60)
projectile_name = input(
    "\n  Projectile / experiment name\n"
    "  (e.g. 'golf', 'hollow' or 'stress')\n"
    "  > "
).strip()
while not projectile_name:
    projectile_name = input("  Please enter a non-empty name > ").strip()

# where the workbooks live (one file per projectile)
from datetime import datetime
dateStamp = datetime.now().strftime('%Y_%m_%d')
WORKBOOK_FOLDER = os.path.join(os.getcwd(), "Free fall data", dateStamp, "Excel workbooks")
os.makedirs(WORKBOOK_FOLDER, exist_ok=True)
wb_path = fw.workbook_path(WORKBOOK_FOLDER, projectile_name)

# report current status of this projectile's file so the student knows where they are
if os.path.exists(wb_path):
    import openpyxl
    _ws = openpyxl.load_workbook(wb_path)[fw.SHEET_NAME]
    status = fw.runs_status(_ws)
    filled = [r for r, done in status.items() if done]
    suggested = fw.next_empty_run(_ws)
    print(f"\n  Existing file found for '{projectile_name}'.")
    print(f"  Runs already saved: {filled if filled else 'none'}")
    if suggested is None:
        print("  NOTE: all 5 runs are already filled. You'll be asked to confirm an overwrite.")
    else:
        print(f"  This drop will be saved as Run {suggested} (unless you choose otherwise).")
else:
    print(f"\n  No file yet for '{projectile_name}' - this will create it as Run 1.")
    suggested = 1

print("\n  Get the projectile in place, then press Enter to arm the drop...")
input()

# ---------------------------------------------------------------------------
# connect to the servo
# ---------------------------------------------------------------------------
servo1_start = 1000
servo2_start = 2200
servo1_open = 2000
servo2_open = 2500

pi = pigpio.pi()
sleep(2.2)
pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

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
release_index = None   # <-- robustly record the true release point

print('\n\n Here we go \n\n')
for idx in range(0, N_SAMPLES):

    if idx == 50:
        print('\n\n **DROP** \n\n')
        pi.set_servo_pulsewidth(23, servo1_open)
        pi.set_servo_pulsewidth(22, servo2_open)
        # record how many valid readings we had at the moment of release,
        # rather than assuming a fixed offset of 49 (which silently breaks
        # if any early serial line was dropped)
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

# fall back to the old behaviour if for some reason release wasn't recorded
if release_index is None or release_index < 1:
    release_index = 49

delta = release_index
timer = np.asarray(timer)
distances = np.asarray(distance[delta:])

# ---------------------------------------------------------------------------
# interactive truncation (unchanged)
# ---------------------------------------------------------------------------
norm = plt.Normalize(0, 2000)
fig, ax = plt.subplots()
sc = ax.scatter(timer[delta:] - timer[delta], distances, c=distances, cmap=cm.viridis, norm=norm)
ax = plt.gca()
# auto-scale so slow / bouncy projectiles are not silently clipped
_tmax = float((timer[delta:] - timer[delta]).max()) if len(timer) > delta else 1500
ax.set_xlim([0, max(1500, _tmax * 1.05)])
ax.set_ylim([0, 2200])
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Distance (mm)')
ax.set_title(f"{projectile_name}: click the last point still in freefall")

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
# save raw .npy + truncated .txt (unchanged behaviour, plus projectile label
# stamped into the filenames so raw files are self-documenting)
# ---------------------------------------------------------------------------
folder_path = os.path.join(os.getcwd(), 'Free fall data', dateStamp, projectile_name + ' raw data')
os.makedirs(folder_path, exist_ok=True)
timeStamp = datetime.now().strftime('%H_%M_%S')
stem = fw._safe_name(projectile_name)

txt_path = os.path.join(folder_path, f"{timeStamp}_{stem}_truncated_data.txt")
np.savetxt(txt_path, np.column_stack((truncated_time, truncated_distance)),
           delimiter=",", header="Time,Distance", comments='')

np.save(os.path.join(folder_path, f"{timeStamp}_{stem}_raw_times.npy"), timer)
np.save(os.path.join(folder_path, f"{timeStamp}_{stem}_raw_distances.npy"), distances)

# ---------------------------------------------------------------------------
# STEP N: confirm, then route into the correct per-projectile workbook
# ---------------------------------------------------------------------------
# decide which run slot
run_to_write = suggested
overwrite = False
if run_to_write is None:
    # file already full - force an explicit choice
    print("\n  All 5 runs already exist for this projectile.")
    ans = input("  Type a run number (1-5) to OVERWRITE, or Enter to skip saving to workbook: ").strip()
    if ans.isdigit() and 1 <= int(ans) <= 5:
        run_to_write = int(ans)
        overwrite = True
    else:
        run_to_write = None

if run_to_write is not None:
    print("\n" + "-" * 60)
    print(f"  About to save Run {run_to_write} of 5 for '{projectile_name}'")
    print(f"  ({len(truncated_time)} data points) into:")
    print(f"    {wb_path}")
    print("-" * 60)
    confirm = input("  Press Enter to confirm, or type 'x' to discard this run: ").strip().lower()
    if confirm == 'x':
        print("  Discarded - workbook not modified. (Raw .npy/.txt files were still saved.)")
    else:
        try:
            saved_path, saved_run, status = fw.write_run(
                WORKBOOK_FOLDER, projectile_name,
                truncated_time, truncated_distance,
                run_number=run_to_write, overwrite=overwrite,
            )
            done = [r for r, ok in status.items() if ok]
            print(f"\n  Saved as Run {saved_run}. Runs now complete: {done}")
            if len(done) == 5:
                print("  All 5 repeats collected for this projectile - ready to analyse.")
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

print('\nREADY')
