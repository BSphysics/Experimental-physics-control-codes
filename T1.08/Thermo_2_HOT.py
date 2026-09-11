# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:08:45 2025

@author: bs426
This script is designed to measure the change in pressure of
an ideal gas as the temperature changes (volume constant)

Method:

1. Open BLUE valve to vent apparatus to atmosphere for 10 s
2. Close BLUE valve to seal apparatus
3. Run this script
4. Place calorimeter into hot water bath
5. Start timer
6. After 60 s remove calorimeter for hot water bath and place on absorbent material
7. After **60 s** of heating close the plot window
8. In new plot window truncate data after temperature has definitely started to change

"""
import sys
import os
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
plt.style.use('dark_background')
from matplotlib.widgets import Cursor
from boardFinder import get_serial_connection

plt.close('all')

import numpy as np

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

def extract_floats(text):
    # Find all numbers in the string (integers, decimals, scientific notation)
    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', text)
    # Convert extracted numbers to floats
    return [float(num) for num in numbers]

#*******************Configure the live plots***************

blit = True


P = []
V = []
T = []

pMin = 0.90e5
pMax = 1.05e5

norm = plt.Normalize(vmin=pMin, vmax=pMax)
x = np.linspace(0,1,32)
cmap = cm.jet


import matplotlib.ticker as ticker

#*******************Setting up serial comms with Arduino***************

start_time = time.time()

distance = collections.deque(maxlen=3)
pressure = collections.deque(maxlen=3)
temperature = collections.deque(maxlen=3)

try:
    # Open serial connection
    ser = get_serial_connection(baud=115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)
    
    print("Gathering initial baseline measurements...")

    # --- Collect a short burst of data to determine initial ranges ---
    baseline_pressures = []
    baseline_temps = []
    
    for _ in range(100):
        if ser.in_waiting >= 3:
            # Read 3 lines (distance, pressure, temperature)
            ser.readline()  # distance (discard for now)
            p_line = ser.readline().decode('utf-8').strip()
            t_line = ser.readline().decode('utf-8').strip()

            p_val = np.mean(extract_floats(p_line))
            t_val = np.mean(extract_floats(t_line))

            baseline_pressures.append(p_val)
            baseline_temps.append(t_val)

        # --- Compute robust ranges ---
        p_mean = np.mean(baseline_pressures)
        p_std = np.std(baseline_pressures)
        t_mean = np.mean(baseline_temps)
        t_std = np.std(baseline_temps)
        
    #ax.set_ylim(p_mean*0.98, pMax)
    #ax.set_xlim(t_mean-2, t_mean+2)

    print("Plotting data from sensors...")
    
    print("\n Close plot window when measurement has finished")  

    globalStartTime = time.time()
    times=[]
      
    idx=0

    live_plotting_enabled = True
    
  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=False)

    # Subplot 1: Scatter plot of T vs P 
    scat = ax1.scatter([], [], c=[], s=12, cmap=cmap, norm=norm)
    ax1.set_xlim(t_mean-283, t_mean-253)
    ax1.set_ylim(p_mean*0.99, p_mean*1.05)
    ax1.set_xlabel('Temperature (C)')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.set_title('Temperature vs Pressure')

    # Subplot 2: Pressure vs Time
    pressure_line, = ax2.plot([], [], '--or')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(p_mean*0.99, p_mean*1.05)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Pressure (Pa)')
    ax2.set_title('Pressure vs Time')
    ax2.set_xticklabels([])

    fig.tight_layout()
    fig.canvas.draw()   # note that the first draw comes before setting data
    
    P = []
    V = []
    T = []
    times = []

    if blit:
        fig.canvas.draw()
        ax1_background = fig.canvas.copy_from_bbox(ax1.bbox)
        ax2_background = fig.canvas.copy_from_bbox(ax2.bbox)
    plt.show(block=False)
    
    while True:
        
        if not plt.fignum_exists(fig.number):
            print("Plot window closed, stopping.")
            break
                      
        if ser.in_waiting > 0:
            idx+=1
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)           
            distance.append(serialData)
                
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)           
            pressure.append(serialData)
            
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)           
            temperature.append(serialData)
            #print('\nTemperature= ' + str(np.round(np.mean(temperature) - 273.15,2)) + ' deg C')

            V.append(np.mean(distance))
            P.append(np.mean(pressure))
            T.append(robust_mean(temperature)-273.15)            
            
            accumTime = np.round(time.time() - globalStartTime,4)
            times.append(accumTime)
            
            if live_plotting_enabled:
                data = np.column_stack((T, P))
                scat.set_offsets(data)
                scat.set_array(np.array(P))
                # Update pressure line plot
                pressure_line.set_data(times, P)
                ax2.set_xlim(max(0, times[-1] - 50), times[-1])
            
            if blit:
                fig.canvas.restore_region(ax1_background)
                fig.canvas.restore_region(ax2_background)
                ax1.draw_artist(scat)
                ax2.draw_artist(pressure_line)
                fig.canvas.blit(ax1.bbox)
                fig.canvas.blit(ax2.bbox)
            else:
                fig.canvas.draw()

            fig.canvas.flush_events()
            
except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()  # Close serial connection



times = np.array(times)
V = np.array(V)
p = np.array(P)
T = np.array(T)

timestr = time.strftime("%Y-%m-%d__%H-%M-%S")
folderName = os.path.join(os.getcwd(), 'Pressure vs Temperature data', timestr)
os.makedirs(folderName,exist_ok=True)

filename = os.path.join(folderName,'pressures.npy')
np.save(filename, p)
filename = os.path.join(folderName,'temperatures.npy')
np.save(filename, T)
filename = os.path.join(folderName, 'times.npy')
np.save(filename, times)

plt.close('all')

import matplotlib.path as mpath

star = mpath.Path.unit_regular_star(5)
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.plot(times, p, linestyle = '', marker = star, markersize=10,
         markerfacecolor='gold',
         markeredgecolor='gold')
ax1.set_ylabel("Pressure (Pa)", color='gold')

ax2 = ax1.twinx()
ax2.plot(times, T, '--ro', label='Temperature')
ax2.set_ylabel("Temperature (°C)", color='r')
ax2.tick_params(axis='y', labelcolor='r')

filename = os.path.join(folderName,'Pressure and temperature vs time plot.png')
plt.savefig(filename)

# Add a vertical cursor
cursor = Cursor(ax2, useblit=False, horizOn=False, vertOn=True, color='red', linewidth=1)

# Add a text box in the top right
text = ax2.text(0.95, 0.95, '', transform=ax2.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Mouse motion callback to update y-value
def on_mouse_move(event):
    if event.inaxes == ax2 and event.ydata is not None:
        x_val = np.round(event.xdata,2)
        text.set_text(f'Start time = {x_val:.2f} s')
        fig.canvas.draw_idle()
        
clicked_t=[]


def on_mouse_click(event):
    if event.inaxes == ax2 and event.xdata is not None and len(clicked_t) < 1:
        clicked_t.append(event.xdata)
        print(f"Start time = {event.xdata:.2f} s")
        if len(clicked_t) == 1:
            plt.close(fig)
# Connect the event handlers
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_mouse_click)

plt.show()

start_time = np.array(clicked_t)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

start_idx = find_nearest(times, start_time)
stop_idx = find_nearest(times, start_time + 60)
    
headers = "Time,Temperature,Pressure"
data = np.array([times[start_idx : stop_idx] , T[start_idx : stop_idx] , p[start_idx : stop_idx]]).T
# Save to CSV with headers
filename = os.path.join(folderName, timestr + '.csv')
np.savetxt(filename, data, delimiter=",", header=headers, comments="", fmt="%.2f")

