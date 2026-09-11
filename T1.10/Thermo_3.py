# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:08:45 2025

@author: bs426
"""
import sys
import os

# Add the other folder to the Python path
sys.path.append(os.path.abspath("/home/dorkmaster/Desktop/PHY1035/Tools"))

import serial
import time
import collections
import numpy as np
import re
from os import system
import csv,os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from boardFinder import get_serial_connection

plt.close('all')

volume_fn = np.poly1d([-0.466, 406]) # Calculates volume in mL from distance-to-volume calibration

while True:
    try:
        mass_input = input("Enter the mass value in grams: ")
        mass = float(mass_input)
        break
    except ValueError:
        print("Invalid input. Please enter a number.")


def extract_floats(text):
    # Find all numbers in the string (integers, decimals, scientific notation)
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    # Convert extracted numbers to floats
    return [float(num) for num in numbers]

def robust_mean(values, threshold=2.0):
    values = np.array(values).flatten()  # Flatten list-of-lists
    median = np.median(values)
    abs_deviation = np.abs(values - median)
    mad = np.median(abs_deviation)  # Median Absolute Deviation

    # Define threshold in terms of MAD (robust to outliers)
    if mad == 0:
        filtered = values[abs_deviation < threshold]  # fallback if all values are equal
    else:
        filtered = values[abs_deviation / mad < threshold]

    return np.mean(filtered)

#*******************Configure the live plots***************

blit = True

pMin = 0.995e5
pMax = 1.05e5
pressure_span = pMax - pMin

norm = plt.Normalize(vmin=pMin, vmax=pMax)
x = np.linspace(0,1,32)
cmap = cm.jet

#*******************Setting up serial comms with Arduino***************
start_time = time.time()

distance = collections.deque(maxlen=10)
pressure = collections.deque(maxlen=10)
temperature = collections.deque(maxlen=10)

   
try:
    # Open serial connection
    ser = get_serial_connection(baud=115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)
    #print(f"✅ Connected on {ser.port} at {ser.baudrate} baud")
    
    print("Reading initial pressure to center plot...")
    initial_pressures = []
    for _ in range(10):  # try up to 10 cycles
        if ser.in_waiting > 0:
        # Skip distance line
            line = ser.readline().decode('utf-8').strip()
            _ = extract_floats(line)
        
        # Read pressure line
            line = ser.readline().decode('utf-8').strip()
            #print(line)
            floats = extract_floats(line)
            if floats:
                initial_pressures.append(np.mean(floats))
                break

        # Skip temperature line
            line = ser.readline().decode('utf-8').strip()
            _ = extract_floats(line)
            
        else:
            time.sleep(0.05)
    
    if initial_pressures:
        initial_pressure = np.mean(initial_pressures)
    else:
        print("Warning: No initial pressure received, using default")
        initial_pressure = (pMin + pMax)/2
    
    ser.reset_input_buffer()  # clears everything currently in the buffer
    time.sleep(0.1) 

    # Center pressure axis on initial reading
    pMin_centered = initial_pressure - 0.25*pressure_span
    pMax_centered = initial_pressure + 0.75*pressure_span

    print(f"Initial pressure = {initial_pressure:.1f} Pa, setting y-limits to {pMin_centered:.0f} – {pMax_centered:.0f}")

    live_plotting_enabled = True

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

    # Subplot 1: Scatter plot of T vs P (colored by pressure)
    scat = ax1.scatter([], [], c=[], s=12, cmap=cmap, norm=norm)
    ax1.set_xlim(360, 405)
    ax1.set_ylim(pMin_centered, pMax_centered)
    ax1.set_xlabel('Volume (mL)')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.set_title('Volume vs Pressure')

    # Subplot 2: Pressure vs Time
    temperature_line, = ax2.plot([], [], '--or')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(270, 330)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_title('Temperature vs Time')
    fig.tight_layout()
    fig.canvas.draw()   # first draw before setting data

    p = []
    V = []
    T = []
    d = []
    

    if blit:
        fig.canvas.draw()
        ax1_background = fig.canvas.copy_from_bbox(ax1.bbox)
        ax2_background = fig.canvas.copy_from_bbox(ax2.bbox)
    plt.show(block=False)

    globalStartTime = time.time()
    times=[]

except Exception as e:
    print(f"Error initializing serial or plot: {e}")

try:
    while True:
                
        if ser.in_waiting >=1:
            
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)
            
            distance.append(serialData)
            #print('\n Distance = ' + str(np.round(np.mean(distance),2)))
            
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)           
            pressure.append(serialData)
            #print('\nPressure = ' + str(np.round(np.mean(pressure),2)))
            
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)           
            temperature.append(serialData)
            #print('\nTemperature= ' + str(np.round(np.mean(temperature),2)))
            
            d.append(np.mean(distance))
            p.append(np.mean(pressure))
            T.append(robust_mean(temperature))               
            V.append(volume_fn(np.mean(distance)))
            
            accumTime = np.round(time.time() - globalStartTime,4)
            times.append(accumTime)
            
            if live_plotting_enabled:
                data = np.column_stack((V, p))
                scat.set_offsets(data)
                scat.set_array(np.array(p))
                # Update pressure line plot
                temperature_line.set_data(times, T)
                ax2.set_xlim(max(0, times[-1] - 50), times[-1])
            
            if blit:
                fig.canvas.restore_region(ax1_background)
                fig.canvas.restore_region(ax2_background)
                ax1.draw_artist(scat)
                ax2.draw_artist(temperature_line)
                fig.canvas.blit(ax1.bbox)
                fig.canvas.blit(ax2.bbox)
            else:
                fig.canvas.draw()

            fig.canvas.flush_events()
       
        
        if not plt.fignum_exists(fig.number):
            print("Plot window closed, stopping.")
            break

    
    plt.pause(0.01)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()  # Close serial connection



times = np.array(times)
V = np.array(V)
p = np.array(p)
T = np.array(T)
d = np.array(d)

# Recreate the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

# Plot Volume vs Pressure
ax1.scatter(V, p, c=p, cmap='jet')
ax1.set_xlabel('Volume (mL)')
ax1.set_ylabel('Pressure (Pa)')
ax1.set_title('Volume vs Pressure')

# Plot Temperature vs Time
ax2.plot(times, T, '--or')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Temperature (K)')
ax2.set_title('Temperature vs Time')

fig.tight_layout()

timestr = time.strftime("%Y-%m-%d__%H-%M-%S")
folderName = os.path.join(os.getcwd(), 'Heat engine data', timestr)
os.makedirs(folderName,exist_ok=True)

filename = os.path.join(folderName,'PV plot.png')
plt.savefig(filename)

plt.close('all')

filename = os.path.join(folderName,'pressures.npy')
np.save(filename, p)
filename = os.path.join(folderName,'temperatures.npy')
np.save(filename, T)
filename = os.path.join(folderName, 'times.npy')
np.save(filename, times)
filename = os.path.join(folderName, 'volumes.npy')
np.save(filename, V)
filename = os.path.join(folderName, 'distances.npy')
np.save(filename, d)

if V[0] != V[-1] or p[0] != p[-1]:
    V = np.append(V , V[0])
    p = np.append(p , p[0])

area = 0.5 * np.abs(np.dot(V , np.roll(p,-1)) - np.dot(p , np.roll(V,-1)))*1e-6

print('Estimated work done (from area of PV plot = ' + str(np.round(area,4)) + ' J')


fig, ax = plt.subplots(1, 1)
scat = ax.scatter(V, p, c=p, s=12, cmap=cmap, norm=norm)
ax.set_xlim(360, 405)
ax.set_ylim(pMin_centered, pMax_centered)
ax.set_xlabel('Volume (mL)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title('Volume vs Pressure')
plt.fill(V, p, color='lightblue', alpha=0.5)


filename = os.path.join(folderName,'Filled PV plot.png')
plt.savefig(filename)
plt.show()


from matplotlib.widgets import Cursor

fig, ax = plt.subplots()
ax.plot(times, d, 'og')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Vertical position of syringe top (mm)')
ax.set_title('Select Minimum and maximum vertical positions of mass')


snapped_lines = []
clicked_y = []

cursor = Cursor(ax, useblit=True, linestyle='--', color='red', linewidth=1)

def on_click(event):
    if event.inaxes != ax:
        return
    if event.button == 1:  # left click
        y = event.ydata
        clicked_y.append(y)

        # Draw a horizontal line at this y
        line = ax.axhline(y=y, color='r', linestyle='--', alpha=0.5)
        snapped_lines.append(line)
        fig.canvas.draw_idle()

        # Print messages depending on click order
        if len(clicked_y) == 1:
            print(f"\n Minimum vertical position of mass = {y:.3f} mm")
        elif len(clicked_y) == 2:
            print(f"\n Maximum vertical position of mass = {y:.3f} mm")
            

        # If you only want two points total, disconnect after 2 clicks:
        if len(clicked_y) == 2:
            fig.canvas.mpl_disconnect(cid)

# Connect event
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

mass_heights = np.array(clicked_y)

deltaGPE = (mass_heights[0]-mass_heights[1])*1e-3 * mass * 1e-3 * 9.81

print(f"\n Change in gravitational potential energy = {deltaGPE:.3f} J")