# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:08:45 2025

@author: bs426
This script is designed to measure the pressure of an ideal gas at different volumes
Volume is changed by moving the plastic syringe by 5 mL increments.

Method
Step 1: Place Calorimeter in a water bath (to keep temperature approximately constant)
Step 2: Open valve 1 (blue) to atmosphere
Step 3: Pull back plastic syringe until it is at the 50 mL mark
Step 4: Close valve 1 (blue) to atmosphere
Step 5: Push the syringe down and pull back a few times to mix the air inside the apparatus
Step 6: With syringe set at 50 mL, run this script 
Step 7: Hold syringe at 50 mL for a few seconds and check pressure has settled
Step 8: Push syringe inwards to 45 mL and hold until pressure has settled and temperature return to approximate starting value
Step 9: Repeat step 8 for syringe volumes of 40 mL, 35 mL, 30 mL, 25 mL, 20 mL, 15 mL, 10 mL, 5 mL, 0 mL
Step 10: Close the plot window when all pressure measurements have been acquired
Step 11: Use horizontal cursor to estimate the pressures for each of the syringe volumes (by positioning and left clicking the mouse)


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
fig = plt.figure(figsize = (8,10) , dpi=100)
ax = fig.add_subplot(111)

P = []
V = []
T = []

pMin = 0.90e5
pMax = 1.18e5

norm = plt.Normalize(vmin=pMin, vmax=pMax)
x = np.linspace(0,1,32)
#colormap = plt.cm.jet(x)
cmap = cm.jet
scat = plt.scatter([], [], c=[], s=8,cmap=cmap, norm=norm)

ax.set_ylim(pMin, pMax)
ax.set_xlim(293, 305)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Pa)')
ax.set_title(
    'Close plot window when finished changing volume',
    fontsize=16,          # increase the font size
    fontweight='bold',    # make it bold
    color='red'           # make it red
)


import matplotlib.ticker as ticker

# Enable grid lines and make them denser
ax.grid(True, which='both', axis='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Set the grid line density by controlling the tick intervals
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Adjust x-axis grid interval (e.g., every 1)
ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))  # Adjust y-axis grid interval (e.g., every 5000 Pa)

# Optionally, you can also set minor ticks for even denser gridlines
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Smaller interval for minor ticks on x-axis
ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))  # Smaller interval for minor ticks on y-axis

# Enable minor gridlines
ax.grid(True, which='minor', axis='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

fig.canvas.draw()   # note that the first draw comes before setting data

if blit:
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
plt.show(block=False)


#*******************Setting up serial comms with Arduino***************
  

start_time = time.time()

distance = collections.deque(maxlen=10)
pressure = collections.deque(maxlen=10)
temperature = collections.deque(maxlen=10)

window_size = 5

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
        
    ax.set_ylim(p_mean*0.98, pMax)
    ax.set_xlim(t_mean-2, t_mean+2)

    print("Plotting data from sensors...")
    
    print("\n Close the plot window when measurements have been completed")  

    globalStartTime = time.time()
    times=[]
    
    color_index = 0
    point_colors = []
       
    idx=0
    while True:
        idx+=1
        if not plt.fignum_exists(fig.number):
            print("Figure window closed, stopping.")
            break
                
        if ser.in_waiting >= 1:
            
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

            V.append(np.mean(distance))
            P.append(np.mean(pressure))
            T.append(robust_mean(temperature))
                        
            if idx % 1 ==0:
                data = data = np.column_stack((T , P))
                scat.set_offsets(data)
                scat.set_array(np.array(P)) 

                if blit:
        
                    fig.canvas.restore_region(axbackground)
                    ax.draw_artist(scat)
                    
                    fig.canvas.blit(ax.bbox)

                else:
         
                    fig.canvas.draw()

                fig.canvas.flush_events()
    
            accumTime = np.round(time.time() - globalStartTime,4)
            times.append(accumTime)
            if idx == 1000000:
                idx=0

except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()  # Close serial connection

if 'fig' in locals():
    plt.close(fig)  # Ensure old figure is gone

# Create a new figure for analysis
fig = plt.figure(figsize = (8,10) , dpi=100)
ax = fig.add_subplot(111)
ax.set_title(
    'Click on lowest pressure first',
    fontsize=16,          # increase the font size
    fontweight='bold',    # make it bold
    color='red'           # make it red
)
data = np.column_stack((T, P))
scat = ax.scatter(T, P, c=P, cmap='viridis')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Pa)')

# Add a horizontal cursor
cursor = Cursor(ax, useblit=False, horizOn=True, vertOn=False, color='red', linewidth=1)

# Add a text box in the top left
text = ax.text(0.05, 0.05, '', transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

# Mouse motion callback to update y-value
def on_mouse_move(event):
    if event.inaxes == ax and event.ydata is not None:
        y_val = np.round(event.ydata)
        #text.set_text(f'Pressure = {y_val:.0f} Pa')
        fig.canvas.draw_idle()
        
clicked_y=[]


def on_mouse_click(event):
    if event.inaxes == ax and event.ydata is not None and len(clicked_y) < 11:
        clicked_y.append(event.ydata)
        print(f"Pressure = {event.ydata:.0f} Pa (saved {len(clicked_y)}/11)")
        
        # Update text box with all clicked y-values
        text_str = "Clicked Pressures:\n" + "\n".join(f"p{i+1} = {y:.0f} Pa" for i, y in enumerate(clicked_y))
        text.set_text(text_str)
        fig.canvas.draw_idle()

        # Automatically close the plot after 11 clicks
        if len(clicked_y) == 11:
            plt.close(fig)
# Connect the event handlers
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_mouse_click)

times = np.array(times)
p = np.array(P)
T = np.array(T)

timestr = time.strftime("%Y-%m-%d__%H-%M-%S")
folderName = os.path.join(os.getcwd(), 'Pressure vs Volume data', timestr)
os.makedirs(folderName,exist_ok=True)


filename = os.path.join(folderName,'times.npy')
np.save(filename, times)
filename = os.path.join(folderName,'temperatures.npy')
np.save(filename, T)
filename = os.path.join(folderName,'pressures.npy')
np.save(filename, p)

filename = os.path.join(folderName,'Raw pressure and temperature data.png')
plt.savefig(filename)


plt.show()
pressures = np.array(clicked_y)
deltaVolume = np.ones(len(pressures))*5
cummulativeDeltaVolume = np.arange(0,51,5)

headers = "Pressure (Pa), Incremental change in volume (mL) , Cummulative change in volume (mL)"
data = np.array([pressures , deltaVolume , cummulativeDeltaVolume]).T
# Save to CSV with headers
filename = os.path.join(folderName, timestr + ' pressures and delta volumes.csv')
np.savetxt(filename, data, delimiter=",", header=headers, comments="", fmt="%.2f")
