# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:08:45 2025

@author: bs426
This script is designed to allow us to calibrate the distance measurements made
by the VL531X sensor in terms of volume of the apparatus. Note this requires prior knowledge
of the volume of the apparatus without the syringe attached (see Ideal gas law 1 experiment)

1. Make sure the cap for the glass syringe in positioned underneath the position sensor
2. Open the BLUE valve to the atmosphere and lift the syringe up to 50 mL
3. Run this script
4. Hold the syringe steady at 50 mL and record position data for a few seconds
5. Lower syringe to 40 mL and record position data for a few seconds
6. Repeat at 10 mL intervals until syringe is empty
7. Close the live plot window
8. Use cursor to select the 6 vertical positions where the syringe was held steady
9. Don't adjust the separation between the position sensor and the syringe otherwise this calibration will have to be re-done

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

while True:
    try:
        volume_input = input("Enter the estimated volume of your apparatus in mL: ")
        volume_estimate = float(volume_input)
        break
    except ValueError:
        print("Invalid input. Please enter a number.")



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

blit = False
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)

P = []
d = []
T = []

dMin = 0
dMax = 130

norm = plt.Normalize(vmin = dMin, vmax = dMax)
x = np.linspace(0,1,32)
#colormap = plt.cm.jet(x)
cmap = cm.jet

z = np.linspace(0,1,130)
colors = cmap(norm(z))
scat = plt.scatter([], [], c=[], s=12,cmap=cmap, norm=norm)

ax.set_ylim(0, 150)
ax.set_xlim(0, 50)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (mm)')
#ax.set_title('Close the plot when measurements are complete')
ax.set_title(
    'Close plot window when finished changing volume',
    fontsize=16,          # increase the font size
    fontweight='bold',    # make it bold
    color='red'           # make it red
)

import matplotlib.ticker as ticker

fig.canvas.draw()   # note that the first draw comes before setting data

if blit:
    axbackground = fig.canvas.copy_from_bbox(ax.bbox)
plt.show(block=False)


#*******************Setting up serial comms with Arduino***************
 

start_time = time.time()

distance = collections.deque(maxlen=3)
pressure = collections.deque(maxlen=3)
temperature = collections.deque(maxlen=3)

window_size = 5

text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

try:
    # Open serial connection
    ser = get_serial_connection(baud=115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)

    print("Plotting data from sensors...")
    
    
    globalStartTime = time.time()
    times=[]
    
    color_index = 0
    point_colors = []
       
    idx=0
    initial_distances = []
    live_plotting_enabled = True
    
    while True:
                
        if ser.in_waiting >=1:
            idx+=1
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)
            
            distance.append(serialData)
            if idx <= 10:
                initial_distances.extend(serialData)
                
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)           
            pressure.append(serialData)
            #print('\nPressure = ' + str(np.round(np.mean(pressure),2)))
            
            line = ser.readline().decode('utf-8').strip()  # Read and decode serial data
            serialData = extract_floats(line)           
            temperature.append(serialData)
            #print('\nTemperature= ' + str(np.round(np.mean(temperature),2)))

            d.append(np.mean(distance))
            P.append(np.mean(pressure)*1e-0)
            T.append(robust_mean(temperature))
            
            
            accumTime = np.round(time.time() - globalStartTime,4)
            times.append(accumTime)
                                   
            if live_plotting_enabled:
                data = np.column_stack((times, d))
                scat.set_offsets(data)
                scat.set_array(np.array(d))

                if len(times) >= 25:
                    ax.set_xlim(times[-25], times[-1])
                else:
                    ax.set_xlim(0, max(12.5, times[-1]))  

                if blit:
                    fig.canvas.restore_region(axbackground)
                    ax.draw_artist(scat)
                    fig.canvas.blit(ax.bbox)
                else:
                    fig.canvas.draw()

                fig.canvas.flush_events()
            
        if not plt.fignum_exists(fig.number):
            print("Plot window closed, stopping.")
            break
                
                #print(f"Recent mean: {recent_mean:.2f}, Initial mean: {initial_mean:.2f}, Δ: {abs(recent_mean - initial_mean):.2f}")
            
except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()  # Close serial connection



times = np.array(times)
d = np.array(d)
p = np.array(P)
T = np.array(T)

import time

timestr = time.strftime("%Y-%m-%d__%H-%M-%S")
folderName = os.path.join(os.getcwd(), 'Distance to volume calibration data', timestr)
os.makedirs(folderName,exist_ok=True)


filename = os.path.join(folderName,'distances.npy')
np.save(filename, d)

filename = os.path.join(folderName,'times.npy')
np.save(filename, times)

plt.close('all')
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.scatter(times, d, c=d,s=12,cmap=cmap, norm=norm)
ax1.set_ylabel("Distance (mm)" , fontsize = 20)
ax1.set_xlabel("Time (s)", fontsize = 20)
#ax1.set_title('Click smallest to largest positions')
ax1.set_title(
    'Select smallest to largest positions',
    fontsize=16,          # increase the font size
    fontweight='bold',    # make it bold
    color='red'           # make it red
)

filename = os.path.join(folderName,'distance to volume calibraton raw data.png')
plt.savefig(filename)

# Add a vertical cursor
cursor = Cursor(ax1, useblit=False, horizOn=True, vertOn=False, color='blue', linewidth=1)

# Add a text box in the top right
text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Mouse motion callback to update y-value
def on_mouse_move(event):
    if event.inaxes == ax1 and event.ydata is not None:
        y_val = np.round(event.ydata,2)
        #text.set_text(f'Distance = {y_val:.2f} mm')
        fig.canvas.draw_idle()
        
clicked_d=[]


def on_mouse_click(event):
    if event.inaxes == ax1 and event.ydata is not None and len(clicked_d) < 6:
        clicked_d.append(event.ydata)
        print(f"Distance = {event.ydata:.2f} mm")
        
            # Update text box with all clicked y-values
        text_str = "Clicked positions:\n" + "\n".join(f"p{i+1} = {y:.0f} mm" for i, y in enumerate(clicked_d))
        text.set_text(text_str)
        fig.canvas.draw_idle()
        
        if len(clicked_d) == 6:
            plt.close(fig)
# Connect the event handlers
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_mouse_click)

plt.show()

deltaV = np.arange(50,-1,-10)
totalVolumes = volume_estimate + deltaV
distances = np.asarray(clicked_d)


headers = "Distances (mm),Volumes (mL)"
data = np.array([distances , totalVolumes]).T
# Save to CSV with headers
filename = os.path.join(folderName, timestr + '_distance_volume_calibration.csv')
np.savetxt(filename, data, delimiter=",", header=headers, comments="", fmt="%.2f")
