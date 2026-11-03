import pigpio
from time import sleep

# Connect to pigpio
pi = pigpio.pi()
sleep(1)

servo1_start = 1050
servo2_start = 2125

pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

# Small wiggle range (microseconds)
wiggle_amount = 50  # adjust if needed, keeps motion small
wiggle_delay = 0.1  # delay between wiggle steps
wiggle_cycles = 5   # how many times to wiggle


# Function to gently wiggle a servo
def wiggle_servo(servo_pin, start_pos):
    for i in range(wiggle_cycles):
        pi.set_servo_pulsewidth(servo_pin, start_pos + wiggle_amount)
        sleep(wiggle_delay)
        pi.set_servo_pulsewidth(servo_pin, start_pos - wiggle_amount)
        sleep(wiggle_delay)
    pi.set_servo_pulsewidth(servo_pin, start_pos)  # return to start

# Wiggle servo 1 first
wiggle_servo(23, servo1_start)
sleep(0.5)

# Wiggle servo 2 next
wiggle_servo(22, servo2_start)
sleep(0.5)


pi.set_servo_pulsewidth(23, servo1_start)
pi.set_servo_pulsewidth(22, servo2_start)

print("Servo test completed")
