"""
PHY1035 Driven Harmonic Oscillator - LED tracker  (version 4)

Version check: this version shows a picture of a SNOW LEOPARD for a few
seconds when it starts. If you don't see the snow leopard, you are running an
older version of the script. (Needs snow_leopard.jpg in the same folder.)

Tracks the red LED on the oscillator with the USB camera while the vibration
generator drives it at a known frequency, and measures the amplitude of the
vertical oscillation at that driving frequency. Run once per driving frequency;
each run adds one line (frequency, amplitude) to summary_<bench>.csv.

How a run works
  1. Settling: after a frequency change the motion contains a transient that
     dies away over tens of seconds. The script always waits at least
     MIN_SETTLE_SECONDS (printing progress every PROGRESS_SECONDS), then also
     requires the motion to be steady: it fits the motion in short blocks
     (BLOCK_SECONDS) and watches the complex amplitude (size and phase
     relative to the drive). When the last STABLE_BLOCKS blocks agree to
     within SETTLE_TOL, the oscillator is in steady state.
  2. Measuring: it then records ACQ_SECONDS more and fits a sinusoid at the
     driving frequency to that part only -> amplitude (pixels) +/- uncertainty.
     That part is also split into segments as a final check.
Students can change the frequency and start the script straight away; there
is no need to stop the mass or wait.

Original script: Bes, 13/08/24.  Revised: 12/10/26 (tracking, timing and
spectrum code shared with HarmonicOscillator_01.py; automatic settling).
v4, 13/10/26: snow leopard start-up picture (version check); minimum
60 s settling wait with progress messages before the stability test.
v4.1, 15/10/26: splash picture found whatever the case of its file name
(snow_leopard.JPG etc.); final amplitude shown in bold in a red box.

Run from Geany (Build > Execute, or F5) on the Raspberry Pi. To test without
a camera, run from a terminal:
    python3 drivenHarmonicOscillator_v4.py --simulate F [F_PREVIOUS]
(simulated oscillator: f0 = 3.5 Hz, half-power width 0.02 Hz; drive switched
to F from rest, or from steady state at F_PREVIOUS).
To find the fastest camera settings on a new PC/camera, run:
    python3 drivenHarmonicOscillator_v4.py --benchmark
"""

import os
import sys
import csv
import time
import queue
import threading
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

SIMULATE = "--simulate" in sys.argv
BENCHMARK = "--benchmark" in sys.argv
if not SIMULATE:
    import cv2
    if hasattr(cv2, "setLogLevel"):       # not present in every OpenCV build
        cv2.setLogLevel(0)

# =============================== Settings ===============================
VERSION = "v4.1"
SPLASH_IMAGE = "snow_leopard.jpg"   # shown at start-up so students can check the version
SPLASH_SECONDS = 3.0
BENCH_ID = "bench_00"         # set this on each lab PC (appears in file names)
CAM_INDEX = 0
# Arducam IMX291 (16:9 sensor): its datasheet guarantees 30 fps uncompressed only at
# 640x360. 4:3 sizes such as 640x480 or 320x240 may be slower or cropped.
# Run with --benchmark to test every mode the camera offers.
FRAME_W, FRAME_H = 640, 360
FPS = 30                      # requested frame rate (Nyquist limit = FPS/2)
PIXEL_FORMAT = "YUYV"         # "YUYV" (uncompressed), "MJPG", or None for camera default
N_BUFFERS = 4                 # driver frame buffers. 1 drops frames: while the single
                              # buffer is being read, newly captured frames have nowhere to go
PREVIEW_FRAMES = 20           # frames shown in the tracking check
THRESHOLD = 100               # LED threshold on the red channel (0-255)
EDGE_MARGIN = 5               # warn if the LED centre comes within this many px of the frame edge

# --- settling ---
MIN_SETTLE_SECONDS = 60.0     # always wait at least this long (about 3 amplitude decay
                              # times) before the stability test is allowed to pass
PROGRESS_SECONDS = 5.0        # print a progress message this often during that wait
BLOCK_SECONDS = 4.0          # length of each block fitted while waiting to settle
STABLE_BLOCKS = 4             # this many consecutive blocks must agree...
SETTLE_TOL = 0.01             # ...to within this fraction of the amplitude (size AND phase).
                              # If real runs never settle, loosen to 0.02.
MAX_SETTLE_SECONDS = 150      # give up waiting after this long and measure anyway (with a warning)
# --- measurement ---
ACQ_SECONDS = 20.0            # recording analysed for the amplitude, after settling
N_SEGMENTS = 4                # final check: the analysed recording is split into this many parts
SETTLE_LIMIT = 0.03           # warn if those parts' amplitudes differ by more than this fraction
# --- spectrum ---
F_MIN, F_MAX = 0.5, 14.0      # frequency range searched for peaks (Hz); keep below FPS/2
ZERO_PAD = 8                  # FFT zero-padding factor (finer grid for locating the peak)
SNAP_WINDOW = 0.3             # clicking a spectrum snaps to the highest point within +/- this (Hz)
FREQ_TOL = 0.1                # measured peak searched for within +/- this of the entered frequency
FIT_DF = 0.02                 # frequency search range for the final sinusoid fit (+/- Hz)

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Driven harmonic oscillator data")

CAM_PROPS = {"brightness": -64, "contrast": 50, "saturation": 128, "gamma": 40,
             "gain": 1, "auto_exposure": 1, "exposure_time_absolute": 1,
             # stop the camera lowering its own frame rate (name differs by kernel;
             # whichever one the camera doesn't have is silently ignored)
             "exposure_dynamic_framerate": 0, "exposure_auto_priority": 0}


# ============================ Version check =============================
def show_splash():
    """Show the snow leopard for SPLASH_SECONDS so students can see they are
    running this version. If the picture is missing, say so and carry on."""
    print(f"\nPHY1035 driven harmonic oscillator script, version {VERSION} "
          "(you should see a snow leopard)")
    folder = os.path.dirname(os.path.abspath(__file__))
    # Linux file names are case-sensitive, so also accept e.g. snow_leopard.JPG
    # or snow_leopard.jpeg (cameras and Windows often use capitals).
    stem = os.path.splitext(SPLASH_IMAGE)[0].lower()
    path = os.path.join(folder, SPLASH_IMAGE)
    for name in sorted(os.listdir(folder)):
        root, ext = os.path.splitext(name)
        if root.lower() == stem and ext.lower() in (".jpg", ".jpeg", ".png"):
            path = os.path.join(folder, name)
            break
    try:
        img = plt.imread(path)
    except (FileNotFoundError, OSError, ValueError) as err:
        print(f"Warning: could not show the version picture ({err}).\n"
              f"Put {SPLASH_IMAGE} in the same folder as this script.")
        time.sleep(SPLASH_SECONDS)
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Driven harmonic oscillator - script version {VERSION}", fontsize=14)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(SPLASH_SECONDS)        # keeps the window drawn and responsive
    plt.close(fig)


# ============================== Tracking ================================
def get_centroid(red):
    """Intensity-weighted centroid of the pixels above THRESHOLD.

    Returns sub-pixel (x, y) in image coordinates, or (nan, nan) if no pixel
    is above threshold (LED out of view), rather than crashing.
    """
    w = cv2.subtract(red, THRESHOLD)   # uint8, clips at 0; faster than float maths
    m = cv2.moments(w)
    if m["m00"] < 1e-6:
        return np.nan, np.nan
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def fourcc_str(cam):
    code = int(cam.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def open_camera(width=FRAME_W, height=FRAME_H, fps=FPS, pix_fmt=PIXEL_FORMAT,
                buffers=N_BUFFERS, quiet=False):
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cam.isOpened():
        sys.exit("\nCould not open the camera. Check it is plugged in and that "
                 "cam_test (or another program) is not still using it.")
    if pix_fmt:                           # must be set before the frame size
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pix_fmt))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, fps)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, buffers)
    # Set exposure etc. after opening: opening the device can reset them.
    for key, val in CAM_PROPS.items():
        try:
            subprocess.run(["v4l2-ctl", "-d", f"/dev/video{CAM_INDEX}",
                            "-c", f"{key}={val}"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"Warning: could not set camera property '{key}'")
    if not quiet:
        w, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {w}x{h} {fourcc_str(cam)}, requested {fps} frames/s")
        if (w, h) != (width, height):
            print(f"Note: camera did not accept {width}x{height}; using {w}x{h}.")
    return cam


def hexagon_vertices(center, radius, rotation_deg=0):
    cx, cy = center
    return [(int(cx + radius * np.cos(np.deg2rad(60 * i - 30 + rotation_deg))),
             int(cy + radius * np.sin(np.deg2rad(60 * i - 30 + rotation_deg))))
            for i in range(6)]


HEX_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
               (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def preview(cam):
    """Show a few frames with a marker on the tracked LED."""
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    im = None
    ax.axis("off")
    ax.set_title("Check the camera is tracking the LED")
    rotation, lost = 0, 0
    for _ in range(PREVIEW_FRAMES):
        ok, frame = cam.read()
        if not ok:
            continue
        cx, cy = get_centroid(frame[:, :, 2])
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if im is None:
            im = ax.imshow(img, origin="lower")
        if np.isfinite(cx):
            verts = hexagon_vertices((cx, cy), max(10, img.shape[1] // 13), rotation)
            for i in range(6):
                cv2.line(img, verts[i], verts[(i + 1) % 6], HEX_COLOURS[i], 2)
        else:
            lost += 1
        im.set_data(img)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        rotation += 20
    plt.close(fig)
    plt.ioff()
    if lost:
        print(f"\nWarning: LED not found in {lost} of {PREVIEW_FRAMES} preview frames. "
              "Check the LED is on and in view before continuing.")


def ask_frequency():
    while True:
        text = input("\nWhat driving frequency is set on the function generator (Hz)?\n"
                     "Type it exactly as shown on the display: ")
        text = text.strip().lower().replace("hz", "").strip()
        try:
            f = float(text)
        except ValueError:
            print("Please type a number, e.g. 3.452")
            continue
        if F_MIN <= f <= F_MAX:
            return f
        print(f"This script measures between {F_MIN:g} and {F_MAX:g} Hz. "
              "Check the value on the function generator.")


def camera_frames(cam, stop, info):
    """Yield (t_wall, t_cam, x, y) for each frame until `stop` is set.

    A background thread does nothing but read frames and timestamp them, so
    the camera is serviced at its full rate; centroids are computed here from
    a queue. If processing is ever slower than the camera, frames wait in the
    queue instead of being dropped. y is flipped so that 'up' is positive.
    """
    # Discard frames queued while waiting for input: keep grabbing until a grab
    # has to wait for a genuinely new frame (i.e. the queue is empty).
    for _ in range(3 * N_BUFFERS + 5):
        t_start = time.perf_counter()
        cam.grab()
        if time.perf_counter() - t_start > 0.5 / FPS:
            break
    frames = queue.Queue()

    def reader():
        t0 = time.perf_counter()
        failures = 0
        while not stop.is_set():
            ok, frame = cam.read()
            t = time.perf_counter() - t0
            if not ok:
                failures += 1
                if failures > 20:
                    info["error"] = "Camera stopped returning frames."
                    break
                continue
            frames.put((t, cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        np.ascontiguousarray(frame[:, :, 2])))
        frames.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while True:
        item = frames.get()
        if item is None:
            if info.get("error"):
                raise RuntimeError(info["error"])
            return
        t, tc, red = item
        info["height"] = red.shape[0]
        cx, cy = get_centroid(red)
        yield t, tc, cx, red.shape[0] - 1 - cy


# =========================== Settling monitor ===========================
def complex_amplitude(tt, ss, f):
    """Least-squares fit of a cos + b sin + c + d t at fixed frequency f.

    Returns (C, sigma_A): C = a - i b is the complex amplitude (|C| = amplitude,
    arg C = phase relative to the time origin), sigma_A the uncertainty of |C|
    from the scatter about the fit. Returns None if too few points.
    """
    good = np.isfinite(ss)
    tt, ss = tt[good], ss[good]
    if len(tt) < 8:
        return None
    X = np.column_stack([np.cos(2 * np.pi * f * tt), np.sin(2 * np.pi * f * tt),
                         np.ones_like(tt), tt - tt.mean()])
    coef, *_ = np.linalg.lstsq(X, ss, rcond=None)
    resid = ss - X @ coef
    cov = (resid @ resid / max(len(tt) - 4, 1)) * np.linalg.inv(X.T @ X)
    A = np.hypot(coef[0], coef[1])
    J = coef[:2] / max(A, 1e-12)
    sA = float(np.sqrt(J @ cov[:2, :2] @ J))
    return coef[0] - 1j * coef[1], sA


class SettleMonitor:
    """Fits consecutive BLOCK_SECONDS blocks at the driving frequency.

    In steady state the complex amplitude C (size and phase relative to the
    drive) is constant. A transient at the natural frequency makes C change:
    its size grows or shrinks, and its phase drifts (beats). Using C rather
    than |C| alone catches beats even near their turning points, where the
    size is momentarily flat but the phase is changing fastest.
    """

    def __init__(self, f_drive):
        self.f = f_drive
        self.i0 = 0                  # index of the first frame of the current block
        self.t_mid, self.C, self.sA = [], [], []
        self.dev = np.nan
        self.settled = False

    def feed(self, ts, ys):
        """Call after each new frame (ts, ys are the growing lists).
        Returns a progress line when a block has been completed, else None."""
        if ts[-1] - ts[self.i0] < BLOCK_SECONDS:
            return None
        tt = np.array(ts[self.i0:])
        yy = np.array(ys[self.i0:])
        self.i0 = len(ts)
        r = complex_amplitude(tt, yy, self.f)
        if r is None:
            return f"  {tt[-1]:5.0f} s   LED not found in this block"
        C, sA = r
        self.t_mid.append(tt.mean())
        self.C.append(C)
        self.sA.append(sA)
        line = f"  {tt[-1]:5.0f} s   amplitude {abs(C):7.2f} px"
        if len(self.C) >= STABLE_BLOCKS:
            Cs = np.array(self.C[-STABLE_BLOCKS:])
            Cm = Cs.mean()
            self.dev = np.abs(Cs - Cm).max() / abs(Cm)
            # don't demand better than the noise allows (small amplitudes)
            tol = max(SETTLE_TOL, 3 * max(self.sA[-STABLE_BLOCKS:]) / abs(Cm))
            self.settled = self.dev < tol
            line += f"   still changing by {100 * self.dev:5.1f}%"
        return line

    def blocks(self):
        return np.array(self.t_mid), np.abs(np.array(self.C)), np.array(self.sA)


def settle_and_measure(frames, f_drive):
    """Consume frames: wait for steady state, then record ACQ_SECONDS more.

    Returns arrays t_wall, t_cam, x, y, the index of the first analysed frame,
    the settling time (s), whether it settled, and the monitor.
    """
    t_wall, t_cam, xs, ys = [], [], [], []
    mon = SettleMonitor(f_drive)
    i_meas = None
    waiting = True                       # still inside the minimum wait
    next_progress = PROGRESS_SECONDS
    print(f"\nWaiting {MIN_SETTLE_SECONDS:g} s for the transient to die away "
          "(do not touch the apparatus)...")
    for t, tc, x, y in frames:
        t_wall.append(t)
        t_cam.append(tc)
        xs.append(x)
        ys.append(y)
        if i_meas is None:
            # The monitor fits blocks from the start, so by the end of the
            # minimum wait it already has the blocks the stability test needs.
            line = mon.feed(t_wall, ys)
            if waiting:
                if t >= next_progress:
                    amp = (f"   (amplitude now {abs(mon.C[-1]):.1f} px)"
                           if mon.C else "")
                    print(f"  {next_progress:3.0f} s of {MIN_SETTLE_SECONDS:g} s{amp}")
                    next_progress += PROGRESS_SECONDS
                if t < MIN_SETTLE_SECONDS:
                    continue
                waiting = False
                print("\nMinimum wait done. Now checking the amplitude is steady "
                      f"(blocks of {BLOCK_SECONDS:g} s; up to {MAX_SETTLE_SECONDS:g} s "
                      "in total):")
                if np.isfinite(mon.dev):
                    print(f"  over the last {STABLE_BLOCKS * BLOCK_SECONDS:g} s the "
                          f"amplitude changed by {100 * mon.dev:.1f}%")
            elif line:
                print(line)
            if mon.settled or t >= MAX_SETTLE_SECONDS:
                i_meas = len(t_wall)
                t_settle = t
                if mon.settled:
                    print(f"\nSettled after {t:.0f} s. Recording {ACQ_SECONDS:g} s "
                          "for the amplitude measurement...")
                else:
                    print(f"\nNot settled after {MAX_SETTLE_SECONDS:g} s. Recording "
                          "anyway, but check the warnings at the end.")
        elif t - t_wall[i_meas] >= ACQ_SECONDS:
            break
    if i_meas is None:
        raise RuntimeError("Recording ended before the measurement started.")
    return (np.array(t_wall), np.array(t_cam), np.array(xs), np.array(ys),
            i_meas, t_settle, mon.settled, mon)


# ============================ Camera tools ==============================
def list_camera_modes():
    """(format, width, height, fps) for every mode the camera reports, via v4l2-ctl.
    Returns [] if v4l2-ctl isn't available."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{CAM_INDEX}",
                              "--list-formats-ext"], capture_output=True,
                             text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    modes, fmt, size = [], None, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("[") and "'" in line:
            fmt = line.split("'")[1]
        elif line.startswith("Size:"):
            w, h = line.split()[-1].split("x")
            size = (int(w), int(h))
        elif line.startswith("Interval:") and "fps" in line and fmt and size:
            fps = float(line.split("(")[-1].split()[0])
            modes.append((fmt, size[0], size[1], fps))
    return modes


def benchmark():
    """Measure the frame rate each camera mode actually delivers."""
    modes = list_camera_modes()
    best = {}                                  # (fmt, w, h) -> highest advertised fps
    for f, w, h, r in modes:
        best[(f, w, h)] = max(r, best.get((f, w, h), 0))
    if best:
        print("\nModes reported by the camera (highest frame rate):")
        for f in sorted({k[0] for k in best}):
            print(f"  {f}: " + ", ".join(f"{w}x{h} @ {best[(ff, w, h)]:g}"
                                         for (ff, w, h) in sorted(best) if ff == f))
        tests = [(f, w, h, r) for (f, w, h), r in sorted(best.items())
                 if f in ("YUYV", "MJPG") and w <= 1280]
    else:
        print("\n(v4l2-ctl not found, testing a standard list of modes)")
        tests = [(f, w, h, 30) for f in ("YUYV", "MJPG")
                 for (w, h) in ((640, 480), (640, 360), (320, 240))]

    print("\nTesting (about 3 s per mode)...\n")
    print(f"{'requested':>28} | {'got':>14} | {'read only':>10} | {'read + centroid':>15}")
    for pix, w, h, fps in tests:
        cam = open_camera(w, h, int(round(fps)), pix, N_BUFFERS, quiet=True)
        for _ in range(10):
            cam.grab()
        rates = []
        for do_centroid in (False, True):
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 1.5:
                ok, frame = cam.read()
                if ok:
                    n += 1
                    if do_centroid:
                        get_centroid(frame[:, :, 2])
            rates.append(n / (time.perf_counter() - t0))
        got = (f"{int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
               f"{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))} {fourcc_str(cam)}")
        cam.release()
        print(f"{pix} {w}x{h} @ {fps:g} fps".rjust(28),
              f"| {got:>14} | {rates[0]:>6.1f} /s | {rates[1]:>11.1f} /s")
    print("\nSet FRAME_W, FRAME_H, FPS and PIXEL_FORMAT at the top of the script to the "
          "fastest row whose 'got' matches what was requested.")
    print("To check the driver's own rate without OpenCV:")
    print(f"    v4l2-ctl -d /dev/video{CAM_INDEX} --stream-mmap --stream-count=300")


def choose_timestamps(t_wall, t_cam):
    """Prefer the camera driver's frame timestamps (set when the frame was
    captured) if they look valid; otherwise use the computer clock."""
    span_w = t_wall[-1] - t_wall[0]
    span_c = t_cam[-1] - t_cam[0]
    if (np.all(np.isfinite(t_cam)) and np.all(np.diff(t_cam) > 0)
            and abs(span_c - span_w) < 0.05 * span_w):
        return t_cam - t_cam[0], "camera"
    return t_wall - t_wall[0], "computer clock"


# ============================== Simulation ==============================
SIM_F0, SIM_TAU = 3.5, 16.0   # natural frequency (Hz), amplitude decay time (s)


def sim_response(f):
    """Complex steady-state response of x'' + g x' + w0^2 x = K cos(wt), scaled
    so the amplitude is 60 px at f0."""
    w0, g, w = 2 * np.pi * SIM_F0, 2 / SIM_TAU, 2 * np.pi * f
    return 60 * g * w0 / ((w0**2 - w**2) + 1j * g * w)


def simulated_frames(f_drive, f_prev, info):
    """Synthetic frames: the drive switches to f_drive at t = 0, starting from
    rest (f_prev None) or from steady state at f_prev. Exact solution with
    the transient at the damped natural frequency."""
    rng = np.random.default_rng(1)
    w0, g, w = 2 * np.pi * SIM_F0, 2 / SIM_TAU, 2 * np.pi * f_drive
    wd = np.sqrt(w0**2 - g**2 / 4)
    S = sim_response(f_drive)
    if f_prev is None:
        x0 = v0 = 0.0
    else:
        Sp = sim_response(f_prev)
        x0, v0 = Sp.real, (1j * 2 * np.pi * f_prev * Sp).real
    C = x0 - S.real
    D = (v0 - (1j * w * S).real + g / 2 * C) / wd
    info["height"] = 360
    info["true_amplitude"] = abs(S)
    t = 0.0
    while True:
        y = ((S * np.exp(1j * w * t)).real
             + np.exp(-g * t / 2) * (C * np.cos(wd * t) + D * np.sin(wd * t)))
        y = 180 + y + rng.normal(0, 0.3)
        x = 320 + 0.05 * (y - 180) + rng.normal(0, 0.3)
        if rng.random() < 0.002:
            x = y = np.nan
        yield t, np.nan, x, y          # no camera timestamps -> computer clock
        t += 1 / FPS + rng.normal(0, 0.003)


# ============================== Analysis ================================
def spectrum(t, s):
    """Amplitude spectrum of a (possibly unevenly sampled) signal.

    Removes the mean and any slow drift, resamples onto a uniform time grid,
    applies a Hann window and zero-pads. Returns frequency (Hz) and
    amplitude (pixels, approx. the oscillation amplitude at a peak).
    """
    good = np.isfinite(s)
    t, s = t[good], s[good]
    s = s - np.polyval(np.polyfit(t, s, 1), t)
    n = len(t)
    tu = np.linspace(t[0], t[-1], n)
    su = np.interp(tu, t, s)
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * ZERO_PAD)))
    amp = np.abs(rfft(su * win, nfft)) * 2 / win.sum()
    f = rfftfreq(nfft, tu[1] - tu[0])
    return f, amp


def find_peak(f, amp, lo=F_MIN, hi=F_MAX):
    """Highest point in [lo, hi], refined by a parabolic fit to log amplitude."""
    band = np.where((f >= lo) & (f <= hi))[0]
    i = band[np.argmax(amp[band])]
    if 0 < i < len(amp) - 1:
        a, b, c = np.log(amp[i - 1:i + 2] + 1e-12)
        denom = a - 2 * b + c
        if denom < 0:
            return f[i] + 0.5 * (a - c) / denom * (f[1] - f[0]), amp[i]
    return f[i], amp[i]


def fit_sinusoid(tt, ss, f0, df=FIT_DF, n_try=21):
    """Least-squares fit of a cos + b sin + c + d t to (tt, ss), trying
    frequencies within +/- df of f0 and keeping the best.

    Returns (A, sigma_A, f, model) where A = sqrt(a^2 + b^2) and sigma_A
    comes from the scatter of the data about the fit (standard least-squares
    covariance, propagated to A). Returns None if there are too few points.
    """
    if len(tt) < 8:
        return None
    tc = tt - tt.mean()
    best = None
    for f in f0 + np.linspace(-df, df, n_try):
        X = np.column_stack([np.cos(2 * np.pi * f * tt), np.sin(2 * np.pi * f * tt),
                             np.ones_like(tt), tc])
        coef, *_ = np.linalg.lstsq(X, ss, rcond=None)
        resid = ss - X @ coef
        chi = resid @ resid
        if best is None or chi < best[0]:
            best = (chi, coef, X, f)
    chi, coef, X, f = best
    dof = len(tt) - 5                        # a, b, c, d and the frequency
    cov = (chi / dof) * np.linalg.inv(X.T @ X)
    A = np.hypot(coef[0], coef[1])
    J = coef[:2] / A                         # dA/da, dA/db
    sA = float(np.sqrt(J @ cov[:2, :2] @ J))
    return A, sA, f, X @ coef


def driven_amplitude(t, s, f_drive):
    """Amplitude of the oscillation at the driving frequency.

    1. Locate the spectral peak within +/- FREQ_TOL of the entered frequency
       (guards against small timing-scale errors; a large offset is flagged).
    2. Fit a sinusoid at that frequency to the whole recording -> amplitude.
    3. Fit each of N_SEGMENTS parts separately. In steady state they agree.
       The quoted uncertainty is the standard error of the segment amplitudes,
       or the whole-record fit uncertainty if that is larger.
    """
    good = np.isfinite(s)
    t, s = t[good], s[good]
    f, amp = spectrum(t, s)
    f_meas, _ = find_peak(f, amp, f_drive - FREQ_TOL, f_drive + FREQ_TOL)
    A, sA_fit, f_fit, model = fit_sinusoid(t, s, f_meas)

    seg_A, seg_sA = [], []
    edges = np.linspace(t[0], t[-1], N_SEGMENTS + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t <= hi)
        r = fit_sinusoid(t[m], s[m], f_fit, df=0.0, n_try=1)
        if r is not None:
            seg_A.append(r[0])
            seg_sA.append(r[1])
    seg_A = np.array(seg_A)
    sA_seg = seg_A.std(ddof=1) / np.sqrt(len(seg_A)) if len(seg_A) > 1 else 0.0
    spread = (seg_A.max() - seg_A.min()) / A if len(seg_A) > 1 else 0.0
    # the range of N_SEGMENTS noisy values is typically ~2-3 sigma, so only
    # treat a spread above ~4 sigma as a real change in amplitude
    spread_limit = max(SETTLE_LIMIT, 4 * max(seg_sA, default=0) / A)

    # Largest peak away from the driving frequency (e.g. the natural motion)
    away = (np.abs(f - f_fit) > SNAP_WINDOW) & (f >= F_MIN) & (f <= F_MAX)
    if away.any():
        f2, a2 = find_peak(f, np.where(away, amp, 0))
    else:
        f2, a2 = np.nan, np.nan

    return dict(f=f, amp=amp, f_fit=f_fit, A=A, sA=max(sA_seg, sA_fit),
                seg_A=seg_A, spread=spread, spread_limit=spread_limit, f2=f2, a2=a2,
                t_fit=t, model=model)


# =============================== Saving =================================
def save_results(t, x, y, i_meas, f_drive, res, rate, lost, source,
                 t_settle, settled):
    now = datetime.now()
    folder = os.path.join(DATA_ROOT, now.strftime("%Y_%m_%d"))
    os.makedirs(folder, exist_ok=True)
    base = f"{now.strftime('%H_%M_%S')}_{BENCH_ID}_drive_{f_drive:g}Hz"

    pos_file = os.path.join(folder, base + "_positions.csv")
    with open(pos_file, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "x_horizontal_px", "y_vertical_px", "analysed"])
        for i, row in enumerate(zip(t, x, y)):
            w.writerow([f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}",
                        int(i >= i_meas)])

    keep = res["f"] <= F_MAX
    fft_file = os.path.join(folder, base + "_spectrum.csv")
    with open(fft_file, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_Hz", "amplitude_vertical_px"])
        for row in zip(res["f"][keep], res["amp"][keep]):
            w.writerow([f"{v:.5g}" for v in row])

    summary_file = os.path.join(DATA_ROOT, f"summary_{BENCH_ID}.csv")
    new = not os.path.exists(summary_file)
    with open(summary_file, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "time", "bench", "driving_frequency_Hz",
                        "measured_frequency_Hz", "amplitude_px", "amplitude_uncertainty_px",
                        "segment_spread_percent", "settled", "settling_time_s",
                        "other_peak_Hz", "other_peak_px",
                        "mean_frame_rate_Hz", "frames_without_LED",
                        "timestamps", "positions_file"])
        w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), BENCH_ID,
                    f"{f_drive:g}", f"{res['f_fit']:.4f}",
                    f"{res['A']:.3f}", f"{res['sA']:.3f}", f"{100 * res['spread']:.1f}",
                    "yes" if settled else "no", f"{t_settle:.0f}",
                    f"{res['f2']:.3f}", f"{res['a2']:.3f}",
                    f"{rate:.1f}", lost, source, os.path.basename(pos_file)])
    return pos_file, fft_file, summary_file


# ================================ Plot ==================================
def plot(t, y, i_meas, f_drive, res, mon):
    """Vertical position against time (settling blocks, analysed part with the
    fitted sinusoid) and the spectrum of the analysed part."""
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 5))
    y0 = np.nanmean(y[i_meas:])
    ax_t.axvspan(t[i_meas], t[-1], color="0.9", label="analysed")
    ax_t.plot(t, y - y0, ":r.", ms=2, label="LED position")
    ax_t.plot(res["t_fit"], res["model"] - y0, "-k", lw=0.8,
              label=f"fit at {res['f_fit']:.3f} Hz")
    tb, Ab, sAb = mon.blocks()
    if len(tb):
        # block times are on the computer clock; t may be on the camera clock,
        # but both start at the first frame so the offset is negligible
        ax_t.errorbar(tb, Ab, sAb, fmt="bo", ms=4, label="amplitude while settling")
    ax_t.legend(loc="lower left", fontsize=8)
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Vertical position (px)")
    ax_t.set_title(f"Amplitude = {res['A']:.2f} ± {res['sA']:.2f} px")

    ax_f.plot(res["f"], res["amp"], "-b")
    ax_f.axvline(f_drive, color="k", ls="--", lw=1, label="driving frequency")
    ax_f.set_xlim(0, F_MAX)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Amplitude (px)")
    ax_f.set_title("FFT of the analysed recording")
    ax_f.legend(loc="upper right", fontsize=8)
    # The result, in bold inside a red box above both plots
    fig.text(0.5, 0.935,
             f"Amplitude at {f_drive:g} Hz:  {res['A']:.2f} ± {res['sA']:.2f} px",
             ha="center", va="center", fontsize=20, fontweight="bold", color="black",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2f2",
                       edgecolor="red", linewidth=3))
    fig.text(0.99, 0.01, "Close this window to finish", ha="right", va="bottom",
             fontsize=9, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.86))

    def on_click(event):
        if event.inaxes is ax_f and event.xdata is not None:
            fp, ap = find_peak(res["f"], res["amp"],
                               event.xdata - SNAP_WINDOW, event.xdata + SNAP_WINDOW)
            print(f"\nSelected peak: {fp:.3f} Hz, amplitude {ap:.2f} px")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


# ================================ Main ==================================
def main():
    plt.close("all")
    show_splash()
    if BENCHMARK:
        benchmark()
        return
    info = {}
    if SIMULATE:
        args = sys.argv[sys.argv.index("--simulate") + 1:]
        nums = []
        for a in args[:2]:
            try:
                nums.append(float(a))
            except ValueError:
                break
        f_drive = nums[0] if nums else 3.5
        f_prev = nums[1] if len(nums) > 1 else None
        gen = simulated_frames(f_drive, f_prev, info)
        t_wall, t_cam, x, y, i_meas, t_settle, settled, mon = settle_and_measure(gen, f_drive)
    else:
        cam = open_camera()
        stop = threading.Event()
        try:
            preview(cam)
            f_drive = ask_frequency()
            gen = camera_frames(cam, stop, info)
            t_wall, t_cam, x, y, i_meas, t_settle, settled, mon = \
                settle_and_measure(gen, f_drive)
        finally:
            stop.set()
            time.sleep(0.2)
            cam.release()

    t, source = choose_timestamps(t_wall, t_cam)
    tm, ym = t[i_meas:], y[i_meas:]
    lost = int(np.sum(~np.isfinite(ym)))
    if np.sum(np.isfinite(ym)) < 10:
        sys.exit("\nThe LED was not found in the recording. Check it is on and in view.")
    rate = (len(tm) - 1) / (tm[-1] - tm[0])

    res = driven_amplitude(tm, ym, f_drive)
    pos_file, fft_file, summary_file = save_results(t, x, y, i_meas, f_drive, res, rate,
                                                    lost, source, t_settle, settled)

    print(f"\nRecorded {t[-1]:.0f} s in total at {rate:.1f} frames/s "
          f"(timestamps from {source})")
    result = (f"  Driving frequency: {f_drive:g} Hz",
              f"  AMPLITUDE:         {res['A']:.2f} +/- {res['sA']:.2f} px")
    width = max(len(s) for s in result) + 2
    bold_red, bold, reset = "\033[1;31m", "\033[1m", "\033[0m"
    print("\n" + bold_red + "+" + "=" * width + "+" + reset)
    for s in result:
        print(bold_red + "|" + reset + bold + s.ljust(width) + reset + bold_red + "|" + reset)
    print(bold_red + "+" + "=" * width + "+" + reset)
    if SIMULATE:
        print(f"(true steady-state amplitude {info['true_amplitude']:.2f} px)")

    if not settled:
        print("\nWarning: the oscillation never settled. Check the frequency you typed "
              "matches the display exactly, that the drive amplitude wasn't changed, "
              "and that nothing is touching the apparatus.")
    if res["spread"] > res["spread_limit"]:
        print(f"Warning: the amplitude changed by {100 * res['spread']:.0f}% during the "
              "measurement. Run again.")
    if abs(res["f_fit"] - f_drive) > 0.5 * FREQ_TOL:
        print(f"Warning: the oscillation is at {res['f_fit']:.3f} Hz, not {f_drive:g} Hz. "
              "Check the frequency you typed matches the function generator.")
    if lost > 0.02 * len(tm):
        print(f"Warning: LED lost in {lost} frames. Reduce the driving amplitude "
              "or re-centre the camera.")
    ymin, ymax = np.nanmin(ym), np.nanmax(ym)
    if ymin < EDGE_MARGIN or ymax > info["height"] - 1 - EDGE_MARGIN:
        print("Warning: the LED reaches the edge of the picture. Reduce the driving "
              "amplitude on the power signal generator.")
    if rate < 0.8 * FPS:
        print(f"Warning: frame rate is low ({rate:.1f} /s).")
    print(f"\nData saved to:\n  {pos_file}\n  {fft_file}\n  {summary_file}")

    plot(t, y, i_meas, f_drive, res, mon)


if __name__ == "__main__":
    main()
