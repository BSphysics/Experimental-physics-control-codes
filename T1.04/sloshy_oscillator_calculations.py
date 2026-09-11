
import time 
import numpy as np
import os
from matplotlib import pyplot as plt
import subprocess
from matplotlib.widgets import Cursor 

import numpy as np
from scipy.special import jnp_zeros

def calculate_lowest_sloshing_frequency(container_radius: float, water_depth: float) -> dict:
    """
    Calculates the lowest sloshing mode frequency for a cylindrical container.

    Parameters:
    - container_radius (float): Radius of the cylindrical container in meters.
    - water_depth (float): Depth of the water in meters.

    Returns:
    - dict: Contains angular frequency (rad/s) and frequency in Hertz (Hz).
    """
    if container_radius <= 0 or water_depth <= 0:
        raise ValueError("Container radius and water depth must be greater than zero.")

    # Acceleration due to gravity (m/s^2)
    g = 9.81

    # Get the first zero of the derivative of the Bessel function J_1' (n=1, m=1)
    # scipy's jnp_zeros(n, nt) returns the first nt roots of J_n'(x) = 0
    alpha_11_prime = jnp_zeros(1, 1)[0]

    # Calculate omega squared using the sloshing frequency formula
    # omega^2 = g * (alpha / R) * tanh(alpha * h / R)
    omega_squared = g * (alpha_11_prime / container_radius) * np.tanh(alpha_11_prime * (water_depth / container_radius))
    
    omega = np.sqrt(omega_squared) # Angular frequency in rad/s
    frequency_hz = omega / (2 * np.pi) # Frequency in Hz

    return {
        "angular_frequency_rad_s": omega,
        "frequency_hz": frequency_hz,
        "bessel_root": alpha_11_prime
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters
    R = 12.5e-3  # Radius in meters (e.g., 50 cm)
    h = 30e-3  # Water depth in meters (e.g., 1 meter)

    try:
        results = calculate_lowest_sloshing_frequency(container_radius=R, water_depth=h)
        print(f"Container Radius: {R} m")
        print(f"Water Depth: {h} m")
        print(f"Lowest Sloshing Angular Frequency: {results['angular_frequency_rad_s']:.4f} rad/s")
        print(f"Lowest Sloshing Frequency: {results['frequency_hz']:.4f} Hz")
    except ValueError as e:
        print(f"Error: {e}")
