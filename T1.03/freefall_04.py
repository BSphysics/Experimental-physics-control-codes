# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:27:51 2024

@author:BES
"""
import sys
import os

# Add the other folder to the Python path
sys.path.append(os.path.abspath("/home/dorkmaster/Desktop/PHY1035/Tools"))

from os import system
import subprocess
#import os
import pigpio
import serial,time,csv,os
import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Cursor 
import pigpio
from time import sleep
from boardFinder import get_serial_connection

from matplotlib.widgets import TextBox, Button
import numpy as np
from scipy.interpolate import interp1d
import time

#ramp_file_path_path = os.path.join(os.getcwd(), 'Free fall data', "last_ramp.npz")

RAMP_SAVE_FILE = os.path.join(os.getcwd(), 'Free fall data', "last_ramp.npz")

shape_times = None
shape_voltages = None

# Initial parameters
num_points = 10
duration = 2  # seconds

# Set up figure and axes with better spacing
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.25, 0.85, 0.7])  # [left, bottom, width, height]

# Dark theme styling
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')

if os.path.exists(RAMP_SAVE_FILE):
    data = np.load(RAMP_SAVE_FILE)
    shape_times = data['times']
    shape_voltages = data['voltages']
    num_points = len(shape_times)
    duration = shape_times[-1]
    print("Loaded previous ramp from file.")
else:
    shape_times = np.linspace(0, duration, num_points)
    shape_voltages = np.linspace(0, 100, num_points)

# Always generate initial display values from shape
times = shape_times.copy()
voltages = shape_voltages.copy()

line, = ax.plot(times, voltages, color='cyan', lw=2)
points, = ax.plot(times, voltages, 'o', color='yellow', markersize=8)
dragged_index = None

def update_plot():
    global times, voltages, line, points
    times = np.linspace(0, duration, num_points)
    voltages = np.linspace(0, 100, num_points)
    line.set_xdata(times)
    line.set_ydata(voltages)
    points.set_xdata(times)
    points.set_ydata(voltages)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 100)
    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes != ax:
        return
    x_click, y_click = event.xdata, event.ydata
    distances = np.hypot(times - x_click, voltages - y_click)
    global dragged_index
    dragged_index = np.argmin(distances)

def on_motion(event):
    if dragged_index is None or event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    shape_voltages[dragged_index] = np.clip(y, 0, 100)
    update_displayed_ramp()

def update_displayed_ramp():
    global times, voltages
    interpolator = interp1d(shape_times, shape_voltages, kind='cubic', fill_value="extrapolate")
    times = np.linspace(0, duration, num_points)
    voltages = interpolator(times)
    line.set_xdata(times)
    line.set_ydata(voltages)
    points.set_xdata(times)
    points.set_ydata(voltages)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 100)
    fig.canvas.draw_idle()


def on_release(event):
    global dragged_index
    dragged_index = None

def update_duration(text):
    global duration, shape_times, shape_voltages  
    try:
        new_dur = float(text)
        if new_dur > 0 and new_dur != duration:
            duration = new_dur
            shape_times = np.linspace(0, duration, len(shape_times))
            update_displayed_ramp()
    except ValueError:
        pass


def update_points(text):
    global num_points, shape_times, shape_voltages  
    try:
        new_n = int(text)
        if 1 < new_n <= 100 and new_n != num_points:
            old_interp = interp1d(shape_times, shape_voltages, kind='cubic', fill_value="extrapolate")
            shape_times = np.linspace(0, duration, new_n)
            shape_voltages = old_interp(shape_times)
            num_points = new_n
            update_displayed_ramp()
    except ValueError:
        pass

def send_to_gpio(event):
    print("Sending PWM ramp (simulated):")
    interpolator = interp1d(times, voltages, kind='cubic')
    step = 0.018  # 15 ms
    duration = times[-1] - times[0]
    num_points = int(np.ceil(duration / step)) + 1  # +1 to include endpoint

    fine_times = np.linspace(times[0], times[-1], num_points)
    fine_voltages = interpolator(fine_times)
    button.fine_voltages = fine_voltages
    
       
    np.savez(RAMP_SAVE_FILE, times=times, voltages=voltages)
    print("Saved ramp to file.")
    plt.close()

# TextBoxes below plot
duration_ax = fig.add_axes([0.25, 0.10, 0.1, 0.05], facecolor='gray')
duration_box = TextBox(duration_ax, 'Ramp duration (s)', initial=str(duration))
duration_box.label.set_color('red')      # label color
duration_box.text_disp.set_color('red') # text inside the edit box
duration_box.label.set_fontweight('bold')     # label bold
duration_box.on_submit(update_duration)

points_ax = fig.add_axes([0.5, 0.10, 0.1, 0.05], facecolor='gray')
points_box = TextBox(points_ax, 'Ramp points', initial=str(num_points))
points_box.label.set_color('red')      # label color
points_box.text_disp.set_color('red') # text inside the edit box
points_box.label.set_fontweight('bold')     # label bold
points_box.on_submit(update_points)

button_ax = plt.axes([0.7, 0.1, 0.2, 0.05])
button = Button(button_ax, 'Send to GPIO')
button.fine_voltages = None  # Add this line
button.on_clicked(send_to_gpio)


# Connect plot interactivity
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()

pwm = button.fine_voltages

os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"

plt.close('all')

fanON = 15

fanOFF = 3000

servo1_start = 1050
servo2_start = 2125

servo1_open = 2000
servo2_open = 2500

# connect to the servo 
pi = pigpio.pi()
if not pi.connected:
    print("Could not connect to pigpio daemon")
    exit()

PWM_GPIO = 12  # Use GPIO12 for hardware PWM (pin 32)
FREQ = 25000   
pi.set_PWM_frequency(PWM_GPIO, FREQ)

sleep(2.2)
pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

serialString = ""  # declare a string variable

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

    
timer,distance = [],[]

print('\n\n Here we go \n\n')

idx = 0
a = np.clip(button.fine_voltages*2.55 , a_min=0, a_max=255)
drop_idx = 100
for idx in range(0,700):
    if idx < len(a):
        pi.set_PWM_dutycycle(PWM_GPIO, (min(int(a[idx]),255)))
    
    else:
        pi.set_PWM_dutycycle(PWM_GPIO, 0)                    
    start_time = time.time()
    
    if idx == drop_idx:
        print('\n\n **DROP** \n\n')
        
        pi.set_servo_pulsewidth(23, servo1_open)
        pi.set_servo_pulsewidth(22, servo2_open)

    try:
        ser_bytes = ARDser.readline()
        # print(ser_bytes)
        try:
            decoded_bytes = (ser_bytes[0:len(ser_bytes)-2].decode("utf-8")).split(',')
        except:
            continue
        if len(decoded_bytes)!=2 or decoded_bytes[0]=='' or decoded_bytes[1]=='':
            continue
        #print(decoded_bytes)
        timer.append(float(decoded_bytes[0]))
        distance.append(float(decoded_bytes[1]))
        # print(decoded_bytes[1])
    except KeyboardInterrupt:
        print('My work here is done...')
        break
    
    end_time = time.time()
    iteration_duration = end_time - start_time
    
ARDser.close()

delta = drop_idx
pi.write(PWM_GPIO,0)


timer = np.asarray(timer)
distances = np.asarray(distance[delta:])
norm = plt.Normalize(0,2000)
fig, ax = plt.subplots(figsize = (18,6))
sc = ax.scatter(timer[delta:]-timer[delta],distances, c=distances, cmap=cm.viridis, norm=norm)
ax = plt.gca()
ax.set_ylim([0,2200])
ax.set_xlabel('Time (ms)')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
ax.set_ylabel('Distance (mm)')

# Initialize live cursor line
live_cursor = ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
right_cursor = ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, visible=False)

clicks = []

# Update live cursor position with mouse motion
def on_move(event):
    if event.inaxes == ax:
        live_cursor.set_xdata([event.xdata])
        plt.draw()

# Capture click
def on_click(event):
    if event.inaxes != ax:
        return
    x = event.xdata
    clicks.clear()  # always overwrite previous click
    clicks.append(x)
    right_cursor.set_xdata([x])
    right_cursor.set_visible(True)
    print(f"Freefall end set at x = {x:.2f}")
    plt.draw()
    plt.close()

# Connect events
cid_move = fig.canvas.mpl_connect('motion_notify_event', on_move)
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
time = np.array(timer[delta:]-timer[delta])

idx_end = np.argmin(np.abs(time - clicks[0]))

# Get the actual closest x-values
time_end = time[idx_end]

# Truncate your data before saving
truncated_time = time[:idx_end]
truncated_distance = distances[:idx_end]
truncated_velocity = np.gradient(truncated_distance, truncated_time)
truncated_acceleration = np.gradient(truncated_velocity, truncated_time)

plt.close('all')
pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

from datetime import datetime
dateStamp = datetime.now().strftime('%Y_%m_%d')

folder_path = os.path.join(os.getcwd(), 'Free fall data', dateStamp)
os.makedirs(folder_path,exist_ok=True)

timeStamp = datetime.now().strftime('%H_%M_%S')
file_path = os.path.join(folder_path, timeStamp + '_truncated_data.txt')
# Stack columns together
data = np.column_stack((truncated_time, truncated_distance))
# Save with headers
np.savetxt(file_path, data, delimiter=",", header="Time,Distance", comments='')

file_path = os.path.join(folder_path, timeStamp + '_truncated_data_inc_vel_&_acc.txt')
# Stack columns together
data = np.column_stack((truncated_time, truncated_distance, truncated_velocity, truncated_acceleration))
# Save with headers
np.savetxt(file_path, data, delimiter=",", header="Time, Distance, Velocity, Acceleration", comments='')

file_path = os.path.join(folder_path, timeStamp + '_raw_times.npy')
np.save(file_path , timer)
file_path = os.path.join(folder_path, timeStamp + '_raw_distances.npy')
np.save(file_path , distances)

print ('READY') 

