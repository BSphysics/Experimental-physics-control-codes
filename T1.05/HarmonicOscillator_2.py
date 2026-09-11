# Author: Bes
# Date: 13/08/24


import time 
import cv2
cv2.setLogLevel(0)
from os import system
print("OpenCV version:", cv2.__version__)
import numpy as np
import os
from matplotlib import pyplot as plt
import subprocess
from matplotlib.widgets import Cursor 


plt.close('all')

def get_centroid(img):
    ret,thresh = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx,cy

cam_props = {'brightness': -64, 'contrast': 50, 'saturation': 128,'gamma': 40,
             'gain': 1, 'auto_exposure': 1, 'exposure_time_absolute': 1}

for key in cam_props:
    subprocess.call(['v4l2-ctl -d /dev/video0 -c {}={}'.format(key, str(cam_props[key]))],
                    shell=True)

camIdx = 0
cam = cv2.VideoCapture(camIdx)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)


######-------------------------Check Centroid looks ok
plt.ion()
figure, ax = plt.subplots(1,1)
p = ax.imshow(np.zeros((480,640,3), dtype=np.uint8), origin='lower')
ax.axis('off')
ax.set_title('Verify camera is tracking the LED')

# Function to get hexagon vertices around a center
def hexagon_vertices(center, radius, rotation_deg=0):
    cx, cy = center
    vertices = []
    for i in range(6):
        angle = np.deg2rad(60 * i - 30 + rotation_deg)  # rotate by rotation_deg
        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))
        vertices.append((x, y))
    return vertices

colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255)    # Cyan
]

rotation = 0  # starting angle
rotation_speed = 5  # degrees per frame

for idx in range(20):
    print('\n Frame Number = ' + str(idx))
    ret, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    cxr, cyr = get_centroid(img[:,:,0])
    
    # Draw the rotating hexagon
    verts = hexagon_vertices((cxr, cyr), radius=50, rotation_deg=rotation)
    for i in range(6):
        pt1 = verts[i]
        pt2 = verts[(i+1)%6]
        cv2.line(img, pt1, pt2, colors[i], 2)
    
   
    p.set_data(img)
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    time.sleep(0.01)
    
    rotation += rotation_speed  # rotate for next frame

#########--------------------------------------------------
    
freq = input("What driving frequency are you using (Hz)? ")

plt.close('all')

print('\n\n START \n\n')

globalStartTime = time.time()
times = []
cxrs=[]
cyrs=[]

n_frames = 300 #200
times = np.empty(n_frames)
cxrs = np.empty(n_frames)
cyrs = np.empty(n_frames)

for idx in range(0,n_frames):
    #print('\n Frame Number = ' + str(idx))
    ret, img = cam.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    cxr, cyr = get_centroid(img[:,:,0])
    
    times[idx] = time.time() - globalStartTime
    cxrs[idx] = cxr
    cyrs[idx] = cyr

plt.close('all')
plt.ioff()
time = np.array(times)
cxrs = np.array(cxrs)
cyrs = np.array(cyrs)

position = cxrs - np.mean(cxrs)

from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d


t_uniform = np.linspace(min(time), max(time), len(time))

# Interpolate signal onto uniform grid
interp_func = interp1d(time, position, kind='cubic')  # Cubic interpolation
signal_uniform = interp_func(t_uniform)


# Perform FFT
N = len(t_uniform)
T = np.mean(np.diff(t_uniform))  # Approximate uniform sampling interval
fft_values = fft(signal_uniform)
frequencies = fftfreq(N, T)[:N // 2]  # Only positive frequencies
magnitudes = np.abs(fft_values[:N // 2])

magnitudes[0:5]=0
frequencies[0:5]=0

# Find dominant frequency
idxs = np.argsort(magnitudes)[-2:][::-1]
dominant_freqs = frequencies[idxs]

# Plot FFT spectrum
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude (pix)')
ax1.plot(time, position, ':ro')


line=ax2.plot(frequencies, magnitudes,'-b', label="FFT Spectrum")
points=ax2.plot(frequencies, magnitudes,'ob', label="FFT Spectrum")

ax2.set_xlim([0,14])
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_title("Frequency spectrum")
#plt.legend()
#plt.show()

# Select peak position from plot
cursor = Cursor(
    ax2,
    useblit=True,
    color='red',
    linewidth=2,      # thicker lines
    linestyle='--'    # dashed style makes them stand out
)

# Function to capture clicks and print coordinates
def on_click(event):
    if event.inaxes == ax2:
        # Find nearest data point (optional)
        xdata = event.xdata
        ydata = event.ydata
        #print(f"\nPeak frequency = {xdata:.2f} Hz")
        #print(f"\nPeak amplitude = {ydata:.2f} A.U.")

        # Optionally, snap to nearest point in your dataset
        distances = np.hypot(frequencies - xdata, magnitudes - ydata)
        idx = distances.argmin()
        print(f"\nPeak frequency = {frequencies[idx]:.2f} Hz")
        print(f"\nPeak amplitude = {magnitudes[idx]:.2f} A.U.")
        plt.close(fig)

# Connect the click event to the function
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

from datetime import datetime
dateStamp = datetime.now().strftime('%Y_%m_%d')

folder_path = os.path.join(os.getcwd(), 'Driven harmonic oscillator data', dateStamp)
os.makedirs(folder_path,exist_ok=True)

timeStamp = datetime.now().strftime('%H_%M_%S')
file_path1 = os.path.join(folder_path, timeStamp + '_driving_frequency_' + freq + '_Hz_Positions.txt')
file_path2 = os.path.join(folder_path, timeStamp + '_driving_frequency_' + freq + '_Hz_FFT.txt')
# Stack columns together
data1 = np.column_stack((time, position))
data2 = np.column_stack((frequencies, magnitudes))
# Save with headers
np.savetxt(file_path1, data1, delimiter=",", header="Time,Amplitude", comments='')
np.savetxt(file_path2, data2, delimiter=",", header="Frequency,Magnitude", comments='')

cam.release()
