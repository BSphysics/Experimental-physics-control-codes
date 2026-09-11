# Author: Bes
# Date: 13/08/24

import time 
import cv2
cv2.setLogLevel(0)
print("OpenCV version:", cv2.__version__)
import numpy as np
from matplotlib import pyplot as plt
import subprocess

plt.close('all')

result = subprocess.run(
    ['v4l2-ctl', '-d', '/dev/video0', '-c', 'auto_exposure=1'],
    capture_output=True, text=True
)


print("stdout:", result.stdout)
print("stderr:", result.stderr)
print("return code:", result.returncode)


cam_props = {'brightness': 64, 'contrast': 45, 'saturation': 128,'gamma': 37,
             'gain': 1, 'exposure_time_absolute': 1}

for key in cam_props:
    subprocess.call(['v4l2-ctl -d /dev/video0 -c {}={}'.format(key, str(cam_props[key]))],
                    shell=True)

camIdx = 0
cam = cv2.VideoCapture(camIdx)
if not cam.isOpened():
    print("❌ Cannot open camera")
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30) 
    
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 640, 480)
print('\n\n START \n\n')

idx=0
while True:
    print('\n Frame Number = ' + str(idx))
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    idx+=1

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
        print("👋 Window closed, exiting.")
        break
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting on 'q'")
        break


cam.release()
cv2.destroyAllWindows()
