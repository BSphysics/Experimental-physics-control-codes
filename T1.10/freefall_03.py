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
from scipy.signal import savgol_filter


os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"

plt.close('all')

fanON = 15
fanOFF = 300

fanPower = 50 # Enter value between 0 and 100%


servo1_start = 1050 
servo2_start = 2125

servo1_open = 1700
servo2_open = 2400

# connect to the servo 
pi = pigpio.pi()
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


PWM_GPIO = 12  # Use GPIO12 for hardware PWM
FREQ = 25000   
pi.set_PWM_frequency(PWM_GPIO, FREQ)

def set_voltage(GPIO, percent):
    """
    Set PWM duty cycle on a GPIO pin as a percentage (0-100%).
    """
    if not 0 <= percent <= 100:
        raise ValueError("Percent must be between 0 and 100")
    duty = int(percent / 100 * 255)
    pi.set_PWM_dutycycle(GPIO, duty)


timer,distance = [],[]

print('\n\n Here we go \n\n')
for idx in range(0,300):
    if idx == fanON:
        set_voltage(PWM_GPIO, fanPower)
        
    if idx == fanOFF:
        set_voltage(PWM_GPIO, 0)
    
    if idx == 50:
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
        print('Keyboard Interrupt')
        break

    
ARDser.close()

set_voltage(PWM_GPIO, 0)
pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

delta = 49
timer = np.asarray(timer)
distances = np.asarray(distance[delta:])


norm = plt.Normalize(0,2000)

fig, ax = plt.subplots()
sc = ax.scatter(timer[delta:]-timer[delta],distances, c=distances, cmap=cm.viridis, norm=norm)
ax = plt.gca()
#ax.set_yscale("log")
ax.set_xlim([0, 1200])
ax.set_ylim([0,2200])
ax.set_xlabel('Time (ms)')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
ax.set_ylabel('Distance (mm)')
ax.set_title('Position cursor over final data point and click')

# Initialize live cursor line
live_cursor = ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
import subprocess
import os
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
os.chmod(folder_path, 0o777)

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

os.chmod(file_path, 0o777)

print ('READY') 

