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

 
try:
    # Open serial connection
    ser = get_serial_connection(baud=115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1)
    
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()  # Close serial connection
