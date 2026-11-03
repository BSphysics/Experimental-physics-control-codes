# -*- coding: utf-8 -*-
"""
leak_test.py

Leak-rate test for the PHY1035 thermodynamics apparatus.

Procedure:
    1. Set the piston to approximately maximum volume (~50 mL).
    2. Place the small test weight on top of the piston.
    3. Allow the piston/pressure to settle.
    4. Run this script.
    5. The script records position, pressure and temperature.
    6. Position is smoothed to reduce short-timescale sensor noise.
    7. A linear fit to smoothed volume vs time gives the leak rate.
    8. Results, plots and raw data are saved automatically.

IMPORTANT:
    Change ACCEPTABLE_LEAK_RATE below to the leak rate that you
    decide is acceptable for the apparatus.

Serial data are assumed to arrive in repeating groups of:

    distance
    pressure
    temperature

using the same Arduino setup as the main experiment.
"""

import sys
import os
import re
import time
import csv

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================
# ADD PHY1035 TOOLS FOLDER
# ==============================================================

sys.path.append(
    os.path.abspath("/home/dorkmaster/Desktop/PHY1035/Tools")
)

from boardFinder import get_serial_connection


# ==============================================================
# USER SETTINGS
# ==============================================================

# Length of leak test
TEST_DURATION = 120.0          # seconds

# Time allowed for piston/pressure to settle before measurement
SETTLING_TIME = 10.0           # seconds

# --------------------------------------------------------------
# IMPORTANT:
# This is an example value only.
# Change this to the maximum acceptable leak rate.
# --------------------------------------------------------------

ACCEPTABLE_LEAK_RATE = 5.0    # mL/min


# Position sensor smoothing time.
#
# A 5-second moving average removes rapid sensor noise while
# preserving the slow movement caused by a leak.
#
SMOOTHING_TIME = 5.0           # seconds


# Minimum number of valid readings required
MIN_DATA_POINTS = 20


# ==============================================================
# POSITION -> VOLUME CALIBRATION
# ==============================================================

# Same calibration as your original experiment.
#
# Volume (mL) = -0.6531 * distance + 410.95

volume_fn = np.poly1d([
    -0.6531,
    410.95
])


# ==============================================================
# SERIAL DATA FUNCTIONS
# ==============================================================

def extract_floats(text):
    """
    Extract floating-point numbers from a serial-data string.
    """

    numbers = re.findall(
        r'-?\d+\.?\d*(?:[eE][-+]?\d+)?',
        text
    )

    return [float(num) for num in numbers]


def read_sensor_group(ser):
    """
    Read one complete distance / pressure / temperature group.

    Returns:
        distance, pressure, temperature

    or:
        None
    """

    try:

        # ------------------------------------------------------
        # DISTANCE
        # ------------------------------------------------------

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        distance_values = extract_floats(line)

        if not distance_values:
            return None


        # ------------------------------------------------------
        # PRESSURE
        # ------------------------------------------------------

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        pressure_values = extract_floats(line)

        if not pressure_values:
            return None


        # ------------------------------------------------------
        # TEMPERATURE
        # ------------------------------------------------------

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        temperature_values = extract_floats(line)

        if not temperature_values:
            return None


        # ------------------------------------------------------
        # AVERAGE ANY MULTIPLE VALUES ON EACH LINE
        # ------------------------------------------------------

        distance = np.mean(distance_values)
        pressure = np.mean(pressure_values)
        temperature = np.mean(temperature_values)

        return distance, pressure, temperature


    except Exception:

        return None


# ==============================================================
# MOVING AVERAGE
# ==============================================================

def moving_average_time(times, values, window_seconds):
    """
    Calculate a centred moving average using a time-based window.

    A time-based window is used rather than simply averaging a
    fixed number of samples because the sampling frequency may
    vary slightly.
    """

    times = np.asarray(times)
    values = np.asarray(values)

    smoothed = np.full_like(
        values,
        np.nan,
        dtype=float
    )

    half_window = window_seconds / 2.0


    for i, t in enumerate(times):

        mask = (
            (times >= t - half_window)
            &
            (times <= t + half_window)
        )

        if np.sum(mask) > 0:

            smoothed[i] = np.mean(
                values[mask]
            )


    return smoothed


# ==============================================================
# LINEAR FIT
# ==============================================================

def calculate_linear_fit(times, values):
    """
    Fit values against time.

    Returns:

        slope
        intercept
        R^2
    """

    mask = (
        np.isfinite(times)
        &
        np.isfinite(values)
    )

    t = times[mask]
    v = values[mask]


    if len(t) < MIN_DATA_POINTS:

        raise ValueError(
            "Not enough valid data points."
        )


    # Linear regression
    coefficients = np.polyfit(
        t,
        v,
        1
    )

    slope = coefficients[0]
    intercept = coefficients[1]


    # Calculate fitted values
    fitted = (
        slope * t
        +
        intercept
    )


    # Calculate R^2
    ss_res = np.sum(
        (v - fitted) ** 2
    )

    ss_tot = np.sum(
        (v - np.mean(v)) ** 2
    )


    if ss_tot == 0:

        r_squared = 0.0

    else:

        r_squared = (
            1.0
            -
            ss_res / ss_tot
        )


    return slope, intercept, r_squared


# ==============================================================
# START PROGRAM
# ==============================================================

print()
print("==============================================")
print("          APPARATUS LEAK RATE TEST")
print("==============================================")
print()

print("Procedure:")
print()
print("  1. Set piston to maximum volume (~50 mL).")
print("  2. Place the test weight on the piston.")
print("  3. Allow the piston to settle.")
print("  4. Start this program.")
print()

print(
    f"Test duration       : "
    f"{TEST_DURATION:.0f} s"
)

print(
    f"Settling time       : "
    f"{SETTLING_TIME:.0f} s"
)

print(
    f"Smoothing time      : "
    f"{SMOOTHING_TIME:.1f} s"
)

print(
    f"Acceptance criterion: "
    f"{ACCEPTABLE_LEAK_RATE:.4f} mL/min"
)

print()

input(
    "Press ENTER to begin the leak test..."
)


# ==============================================================
# CONNECT TO ARDUINO
# ==============================================================

try:

    print()
    print("Connecting to apparatus...")


    ser = get_serial_connection(
        baud=115200,
        timeout=1
    )


    time.sleep(2)


    ser.reset_input_buffer()
    ser.reset_output_buffer()


    time.sleep(1)


    print(
        f"Connected on {ser.port}"
    )

    print()


except Exception as e:

    print()
    print(
        "ERROR: Could not connect to Arduino."
    )

    print(e)

    sys.exit(1)


# ==============================================================
# SETTLING PERIOD
# ==============================================================

print(
    f"Allowing {SETTLING_TIME:.0f} seconds "
    "for the piston and pressure to settle..."
)

settling_start = time.time()


settling_volumes = []
settling_pressures = []
settling_temperatures = []


while (
    time.time() - settling_start
    <
    SETTLING_TIME
):

    if ser.in_waiting > 0:

        data = read_sensor_group(ser)


        if data is not None:

            distance, pressure, temperature = data


            settling_volumes.append(
                volume_fn(distance)
            )

            settling_pressures.append(
                pressure
            )

            settling_temperatures.append(
                temperature
            )


    else:

        time.sleep(0.01)


# --------------------------------------------------------------
# Check that data were received
# --------------------------------------------------------------

if len(settling_volumes) == 0:

    print()
    print(
        "ERROR: No valid sensor data received."
    )

    ser.close()

    sys.exit(1)


# --------------------------------------------------------------
# Calculate initial conditions
# --------------------------------------------------------------

initial_volume = np.mean(
    settling_volumes
)

initial_pressure = np.mean(
    settling_pressures
)

initial_temperature = np.mean(
    settling_temperatures
)


print()
print("Initial state:")
print(
    f"  Volume      = "
    f"{initial_volume:.3f} mL"
)

print(
    f"  Pressure    = "
    f"{initial_pressure:.1f} Pa"
)

print(
    f"  Temperature = "
    f"{initial_temperature:.2f} K"
)

print()


# Clear any old serial data before starting the actual test
ser.reset_input_buffer()


# ==============================================================
# ACTUAL LEAK TEST
# ==============================================================

print("==============================================")
print("             TEST NOW RUNNING")
print("==============================================")
print()

print(
    "Do not touch the apparatus."
)

print()


# Arrays for collected data

times = []
distances = []
volumes = []
pressures = []
temperatures = []


test_start = time.time()


try:

    while True:

        elapsed = (
            time.time()
            -
            test_start
        )


        # ------------------------------------------------------
        # Stop after specified duration
        # ------------------------------------------------------

        if elapsed >= TEST_DURATION:

            break


        # ------------------------------------------------------
        # Read sensor
        # ------------------------------------------------------

        if ser.in_waiting > 0:

            data = read_sensor_group(ser)


            if data is not None:

                distance, pressure, temperature = data


                times.append(
                    elapsed
                )

                distances.append(
                    distance
                )

                volumes.append(
                    volume_fn(distance)
                )

                pressures.append(
                    pressure
                )

                temperatures.append(
                    temperature
                )


        else:

            time.sleep(0.005)


        # ------------------------------------------------------
        # Progress display
        # ------------------------------------------------------

        if (
            len(times) > 0
            and
            len(times) % 50 == 0
        ):

            print(
                f"\rElapsed time: "
                f"{elapsed:6.1f} / "
                f"{TEST_DURATION:.1f} s",
                end=""
            )


except KeyboardInterrupt:

    print()
    print()
    print(
        "Test stopped by user."
    )


finally:

    if ser.is_open:

        ser.close()


print()
print()
print(
    "Data collection complete."
)


# ==============================================================
# CONVERT DATA TO NUMPY ARRAYS
# ==============================================================

times = np.asarray(times)

distances = np.asarray(distances)

volumes = np.asarray(volumes)

pressures = np.asarray(pressures)

temperatures = np.asarray(temperatures)


# --------------------------------------------------------------
# Check enough data were collected
# --------------------------------------------------------------

if len(times) < MIN_DATA_POINTS:

    print()

    print(
        "ERROR: Too few valid measurements "
        "were collected."
    )

    print(
        f"Only {len(times)} data points "
        "were obtained."
    )

    sys.exit(1)


# ==============================================================
# SMOOTH THE DATA
# ==============================================================

smooth_volumes = moving_average_time(
    times,
    volumes,
    SMOOTHING_TIME
)


smooth_pressures = moving_average_time(
    times,
    pressures,
    SMOOTHING_TIME
)


smooth_temperatures = moving_average_time(
    times,
    temperatures,
    SMOOTHING_TIME
)


# ==============================================================
# CALCULATE LEAK RATE
# ==============================================================

slope, intercept, r_squared = calculate_linear_fit(
    times,
    smooth_volumes
)


# slope is currently mL/s.
#
# Convert to mL/min:

leak_rate_signed = (
    slope * 60.0
)


# Use the magnitude for the leak-rate result.
#
# This means that whether volume increases or decreases,
# the reported leak rate is positive.

leak_rate = abs(
    leak_rate_signed
)


# ==============================================================
# PRESSURE ANALYSIS
# ==============================================================

pressure_slope, pressure_intercept, pressure_r2 = (
    calculate_linear_fit(
        times,
        smooth_pressures
    )
)


pressure_change = (
    smooth_pressures[-1]
    -
    smooth_pressures[0]
)


pressure_change_percent = (
    100.0
    *
    pressure_change
    /
    np.mean(pressures)
)


# ==============================================================
# VOLUME CHANGE
# ==============================================================

volume_change = (
    smooth_volumes[-1]
    -
    smooth_volumes[0]
)


volume_change_rate = (
    volume_change
    /
    (times[-1] - times[0])
    *
    60.0
)


# ==============================================================
# PASS / FAIL VERDICT
# ==============================================================

if leak_rate <= ACCEPTABLE_LEAK_RATE:

    verdict = "PASS"

    verdict_text = (
        "Leak rate is within the acceptable limit."
    )

else:

    verdict = "FAIL"

    verdict_text = (
        "Leak rate exceeds the acceptable limit."
    )


# ==============================================================
# PRINT RESULTS
# ==============================================================

print()
print("==============================================")
print("              LEAK TEST RESULT")
print("==============================================")
print()

print(
    f"Test duration       : "
    f"{times[-1]:.1f} s"
)

print(
    f"Number of readings  : "
    f"{len(times)}"
)

print()

print(
    f"Initial volume      : "
    f"{smooth_volumes[0]:.3f} mL"
)

print(
    f"Final volume        : "
    f"{smooth_volumes[-1]:.3f} mL"
)

print(
    f"Volume change       : "
    f"{volume_change:.3f} mL"
)

print()

print(
    f"Leak rate           : "
    f"{leak_rate:.4f} mL/min"
)

print(
    f"Acceptance limit    : "
    f"{ACCEPTABLE_LEAK_RATE:.4f} mL/min"
)

print()

print(
    f"Volume-fit R²       : "
    f"{r_squared:.4f}"
)

print()

print(
    f"Pressure change     : "
    f"{pressure_change:.1f} Pa"
)

print(
    f"Pressure change     : "
    f"{pressure_change_percent:.3f} %"
)

print()

print("----------------------------------------------")

print(
    f"                {verdict}"
)

print("----------------------------------------------")

print(
    verdict_text
)

print("==============================================")
print()


# ==============================================================
# CREATE RESULTS FOLDER
# ==============================================================

timestr = time.strftime(
    "%Y-%m-%d__%H-%M-%S"
)


folder_name = os.path.join(
    os.getcwd(),
    "Leak test data",
    timestr
)


os.makedirs(
    folder_name,
    exist_ok=True
)


# ==============================================================
# SAVE RAW DATA
# ==============================================================

np.save(
    os.path.join(
        folder_name,
        "times.npy"
    ),
    times
)


np.save(
    os.path.join(
        folder_name,
        "distances.npy"
    ),
    distances
)


np.save(
    os.path.join(
        folder_name,
        "volumes.npy"
    ),
    volumes
)


np.save(
    os.path.join(
        folder_name,
        "pressures.npy"
    ),
    pressures
)


np.save(
    os.path.join(
        folder_name,
        "temperatures.npy"
    ),
    temperatures
)


# ==============================================================
# SAVE SMOOTHED DATA
# ==============================================================

np.save(
    os.path.join(
        folder_name,
        "smoothed_volumes.npy"
    ),
    smooth_volumes
)


np.save(
    os.path.join(
        folder_name,
        "smoothed_pressures.npy"
    ),
    smooth_pressures
)


# ==============================================================
# SAVE CSV
# ==============================================================

csv_filename = os.path.join(
    folder_name,
    "leak_test_data.csv"
)


with open(
    csv_filename,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)


    writer.writerow([
        "time_s",
        "distance_mm",
        "volume_ml",
        "smoothed_volume_ml",
        "pressure_Pa",
        "smoothed_pressure_Pa",
        "temperature_K"
    ])


    for i in range(len(times)):

        writer.writerow([
            times[i],
            distances[i],
            volumes[i],
            smooth_volumes[i],
            pressures[i],
            smooth_pressures[i],
            temperatures[i]
        ])


# ==============================================================
# SAVE SUMMARY
# ==============================================================

summary_filename = os.path.join(
    folder_name,
    "leak_test_result.txt"
)


with open(
    summary_filename,
    "w"
) as f:

    f.write(
        "APPARATUS LEAK TEST\n"
    )

    f.write(
        "===================\n\n"
    )


    f.write(
        f"Test duration: "
        f"{times[-1]:.2f} s\n"
    )


    f.write(
        f"Number of readings: "
        f"{len(times)}\n"
    )


    f.write(
        f"Initial volume: "
        f"{smooth_volumes[0]:.4f} mL\n"
    )


    f.write(
        f"Final volume: "
        f"{smooth_volumes[-1]:.4f} mL\n"
    )


    f.write(
        f"Volume change: "
        f"{volume_change:.4f} mL\n"
    )


    f.write(
        f"Leak rate: "
        f"{leak_rate:.6f} mL/min\n"
    )


    f.write(
        f"Acceptance limit: "
        f"{ACCEPTABLE_LEAK_RATE:.6f} mL/min\n"
    )


    f.write(
        f"Linear-fit R^2: "
        f"{r_squared:.5f}\n"
    )


    f.write(
        f"Pressure change: "
        f"{pressure_change:.3f} Pa\n"
    )


    f.write(
        f"Pressure change (%): "
        f"{pressure_change_percent:.5f} %\n"
    )


    f.write(
        f"Pressure linear-fit R^2: "
        f"{pressure_r2:.5f}\n"
    )


    f.write(
        f"\nVERDICT: {verdict}\n"
    )


    f.write(
        verdict_text
        +
        "\n"
    )


# ==============================================================
# PLOT 1: VOLUME VS TIME
# ==============================================================

plt.style.use(
    "dark_background"
)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


# Raw data
ax.plot(
    times,
    volumes,
    color="gray",
    alpha=0.35,
    linewidth=0.8,
    label="Raw position"
)


# Smoothed data
ax.plot(
    times,
    smooth_volumes,
    color="cyan",
    linewidth=2,
    label=(
        f"{SMOOTHING_TIME:.1f} s moving average"
    )
)


# Linear fit
fitted_volume = (
    slope * times
    +
    intercept
)


ax.plot(
    times,
    fitted_volume,
    color="red",
    linewidth=2,
    linestyle="--",
    label=(
        f"Linear fit: "
        f"{leak_rate:.4f} mL/min"
    )
)


ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Volume (mL)"
)

ax.set_title(
    "Leak Test — Piston Volume vs Time"
)


ax.legend()


ax.grid(
    True,
    alpha=0.2
)


fig.tight_layout()


plt.savefig(
    os.path.join(
        folder_name,
        "volume_vs_time.png"
    ),
    dpi=300
)


plt.close(fig)


# ==============================================================
# PLOT 2: PRESSURE VS TIME
# ==============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


# Raw pressure
ax.plot(
    times,
    pressures,
    color="gray",
    alpha=0.35,
    linewidth=0.8,
    label="Raw pressure"
)


# Smoothed pressure
ax.plot(
    times,
    smooth_pressures,
    color="orange",
    linewidth=2,
    label=(
        f"{SMOOTHING_TIME:.1f} s moving average"
    )
)


ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Pressure (Pa)"
)

ax.set_title(
    "Leak Test — Pressure vs Time"
)


ax.legend()


ax.grid(
    True,
    alpha=0.2
)


fig.tight_layout()


plt.savefig(
    os.path.join(
        folder_name,
        "pressure_vs_time.png"
    ),
    dpi=300
)


plt.close(fig)


# ==============================================================
# PLOT 3: COMBINED SUMMARY
# ==============================================================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True
)


# --------------------------------------------------------------
# Volume
# --------------------------------------------------------------

ax1.plot(
    times,
    volumes,
    color="gray",
    alpha=0.3,
    linewidth=0.8
)


ax1.plot(
    times,
    smooth_volumes,
    color="cyan",
    linewidth=2
)


ax1.plot(
    times,
    fitted_volume,
    color="red",
    linestyle="--",
    linewidth=2
)


ax1.set_ylabel(
    "Volume (mL)"
)


ax1.set_title(
    f"Leak Test Result: {verdict} — "
    f"{leak_rate:.4f} mL/min"
)


ax1.grid(
    True,
    alpha=0.2
)


# --------------------------------------------------------------
# Pressure
# --------------------------------------------------------------

ax2.plot(
    times,
    pressures,
    color="gray",
    alpha=0.3,
    linewidth=0.8
)


ax2.plot(
    times,
    smooth_pressures,
    color="orange",
    linewidth=2
)


ax2.set_xlabel(
    "Time (s)"
)

ax2.set_ylabel(
    "Pressure (Pa)"
)


ax2.grid(
    True,
    alpha=0.2
)


fig.tight_layout()


plt.savefig(
    os.path.join(
        folder_name,
        "leak_test_summary.png"
    ),
    dpi=300
)


plt.close(fig)


# ==============================================================
# FINISHED
# ==============================================================

print(
    "Results saved to:"
)

print(
    folder_name
)

print()
print(
    "Files created:"
)

print(
    "  leak_test_result.txt"
)

print(
    "  leak_test_data.csv"
)

print(
    "  volume_vs_time.png"
)

print(
    "  pressure_vs_time.png"
)

print(
    "  leak_test_summary.png"
)

print(
    "  *.npy data files"
)

print()
