"""
PHY1035 Harmonic Oscillator - LED tracker

Tracks the red LED on the oscillator with the USB camera, records its
position against time, and finds the frequency of the vertical oscillation
from the FFT. It also saves the decay envelope of the vertical oscillation.
(The horizontal position is still recorded in the _positions file, but it
is not analysed or plotted.)

Original script: Bes, 13/08/24.  Revised: 06/10/26.

Run from Geany (Build > Execute, or F5) on the Raspberry Pi. To test the
analysis without a camera, run from a terminal:
    python3 HarmonicOscillator_2.py --simulate
To find the fastest camera settings on a new PC/camera, run:
    python3 HarmonicOscillator_2.py --benchmark
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
BENCH_ID = "bench_00"         # set this on each lab PC (appears in file names)
CAM_INDEX = 0
# Arducam IMX291 (16:9 sensor): its datasheet guarantees 30 fps uncompressed only at
# 640x360. 4:3 sizes such as 640x480 or 320x240 may be slower or cropped.
# Run with --benchmark to test every mode the camera offers.
FRAME_W, FRAME_H = 640, 360
FPS = 30                      # requested frame rate
PIXEL_FORMAT = "YUYV"         # "YUYV" (uncompressed), "MJPG", or None for camera default
N_BUFFERS = 4                 # driver frame buffers. 1 drops frames: while the single
                              # buffer is being read, newly captured frames have nowhere to go
ACQ_SECONDS = 30.0            # acquisition length (s); FFT bin spacing = 1/ACQ_SECONDS
PREVIEW_FRAMES = 20           # frames shown in the tracking check
THRESHOLD = 100               # LED threshold on the red channel (0-255)
F_MIN, F_MAX = 0.5, 13.0      # frequency range searched for the peak (Hz)
ZERO_PAD = 8                  # FFT zero-padding factor (finer grid for locating the peak)
SNAP_WINDOW = 0.3             # clicking a spectrum snaps to the highest point within +/- this (Hz)
ENV_CYCLES = 2                # envelope: each fitting window spans this many oscillation
                              # cycles (short enough to resolve a few seconds of rattling)
ENV_DF = 0.1                  # envelope: frequency search range in each window (+/- Hz)
RELEASE_FRACTION = 0.3        # envelope starts at the release: the first clean window
                              # whose amplitude exceeds this fraction of the largest one

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Harmonic oscillator data")

CAM_PROPS = {"brightness": -64, "contrast": 50, "saturation": 128, "gamma": 40,
             "gain": 1, "auto_exposure": 1, "exposure_time_absolute": 1,
             # stop the camera lowering its own frame rate (name differs by kernel;
             # whichever one the camera doesn't have is silently ignored)
             "exposure_dynamic_framerate": 0, "exposure_auto_priority": 0}


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


def ask_mass():
    while True:
        text = input("\nWhat mass are you using (g)? ").strip().lower().rstrip("g").strip()
        try:
            return float(text)
        except ValueError:
            print("Please type a number, e.g. 20")


def acquire(cam):
    """Record LED position for ACQ_SECONDS. Returns t (s), x, y (pixels, y up).

    A background thread does nothing but read frames and timestamp them, so
    the camera is serviced at its full rate; the centroids are computed in
    the main thread from a queue. If processing is ever slower than the
    camera, frames wait in the queue instead of being dropped.
    """
    # Discard frames queued while waiting for input: keep grabbing until a grab
    # has to wait for a genuinely new frame (i.e. the queue is empty).
    for _ in range(3 * N_BUFFERS + 5):
        t_start = time.perf_counter()
        cam.grab()
        if time.perf_counter() - t_start > 0.5 / FPS:
            break
    frames = queue.Queue()
    status = {"error": None}

    def reader():
        t0 = time.perf_counter()
        failures = 0
        while True:
            ok, frame = cam.read()
            t = time.perf_counter() - t0
            if not ok:
                failures += 1
                if failures > 20:
                    status["error"] = "Camera stopped returning frames."
                    break
                continue
            frames.put((t, cam.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        np.ascontiguousarray(frame[:, :, 2])))
            if t >= ACQ_SECONDS:
                break
        frames.put(None)        # tells the main thread we're finished

    threading.Thread(target=reader, daemon=True).start()

    t_wall, t_cam, xs, ys = [], [], [], []
    next_report = 5
    while True:
        item = frames.get()
        if item is None:
            break
        t, tc, red = item
        cx, cy = get_centroid(red)
        t_wall.append(t)
        t_cam.append(tc)
        xs.append(cx)
        ys.append(red.shape[0] - 1 - cy)     # flip so that 'up' is positive
        if t >= next_report:
            print(f"  {next_report:.0f} of {ACQ_SECONDS:.0f} s recorded")
            next_report += 5
    if status["error"]:
        raise RuntimeError(status["error"])
    t, source = choose_timestamps(np.array(t_wall), np.array(t_cam))
    return t, np.array(xs), np.array(ys), source


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


def simulated():
    """Synthetic data for testing without a camera."""
    rng = np.random.default_rng(1)
    n = int(ACQ_SECONDS * FPS)
    t = np.cumsum(np.full(n, 1 / FPS) + rng.normal(0, 0.003, n))
    t -= t[0]
    env = np.exp(-t / 20)
    y = 240 + 80 * env * np.cos(2 * np.pi * 3.517 * t) + rng.normal(0, 0.3, n)
    x = 320 + 4 * env * np.cos(2 * np.pi * 2.083 * t + 1) + rng.normal(0, 0.3, n)
    lost = rng.choice(n, 5, replace=False)
    x[lost] = y[lost] = np.nan
    return t, x, y, "simulated"


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


def fit_window(tt, ss, f0):
    """Least-squares fit of a cos + b sin + c to one window, trying frequencies
    within +/- ENV_DF of f0. Returns (A, sigma_A, f) or None."""
    if len(tt) < 8:                          # too few points (e.g. LED lost)
        return None
    best = None
    for f in f0 + np.linspace(-ENV_DF, ENV_DF, 41):
        X = np.column_stack([np.cos(2 * np.pi * f * tt),
                             np.sin(2 * np.pi * f * tt), np.ones_like(tt)])
        coef, *_ = np.linalg.lstsq(X, ss, rcond=None)
        resid = ss - X @ coef
        chi = resid @ resid
        if best is None or chi < best[0]:
            best = (chi, coef, X, f)
    chi, coef, X, f = best
    dof = len(tt) - 4                        # a, b, c and the frequency
    cov = (chi / dof) * np.linalg.inv(X.T @ X)
    A = np.hypot(coef[0], coef[1])
    J = coef[:2] / A                         # dA/da, dA/db
    sA = float(np.sqrt(J @ cov[:2, :2] @ J))
    return A, sA, f


def envelope(t, s, f0):
    """Oscillation amplitude in consecutive, non-overlapping windows of
    ENV_CYCLES oscillation cycles, starting at the release.

    In each window the data are least-squares fitted with
        s(t) = a cos(2 pi f t) + b sin(2 pi f t) + c,
    trying frequencies within +/- ENV_DF of f0 and keeping the best, so a small
    drift in frequency doesn't bias the amplitude. The amplitude is
    A = sqrt(a^2 + b^2). Its uncertainty comes from the scatter of the data
    about the fit in that window (standard least-squares covariance,
    propagated to A). Also returns ln A and its uncertainty sigma_A / A,
    which is what a straight-line fit of ln A against t needs.

    Release detection: if the recording was started before the mass was set
    oscillating, the early windows contain either no oscillation or the hand
    pulling the mass down. The envelope therefore starts at the first window
    that is both large (A > RELEASE_FRACTION x the largest A) and clean
    (sigma_A < 5% of A, i.e. a good sinusoid). A run that is already
    oscillating when recording starts is unaffected.

    Returns an array with columns:
        t_mid, A, sigma_A, lnA, sigma_lnA, f_local, n_points
    """
    good = np.isfinite(s)
    t, s = t[good], s[good]
    width = ENV_CYCLES / f0
    rows = []
    start = t[0]
    while start + width <= t[-1] + 1e-9:
        m = (t >= start) & (t < start + width)
        r = fit_window(t[m], s[m], f0)
        if r is not None:
            A, sA, f = r
            rows.append((t[m].mean(), A, sA, np.log(A), sA / A, f, m.sum()))
        start += width
    rows = np.array(rows)
    if len(rows) == 0:
        return rows
    A, sA = rows[:, 1], rows[:, 2]
    ok = (A > RELEASE_FRACTION * A.max()) & (sA < 0.05 * A)
    first = int(np.argmax(ok)) if ok.any() else 0
    return rows[first:]


# =============================== Saving =================================
def save_results(t, x, y, mass, results, source):
    now = datetime.now()
    folder = os.path.join(DATA_ROOT, now.strftime("%Y_%m_%d"))
    os.makedirs(folder, exist_ok=True)
    base = f"{now.strftime('%H_%M_%S')}_{BENCH_ID}_mass_{mass:g}g"

    pos_file = os.path.join(folder, base + "_positions.csv")
    with open(pos_file, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "x_horizontal_px", "y_vertical_px"])
        for row in zip(t, x, y):
            w.writerow([f"{row[0]:.4f}", f"{row[1]:.3f}", f"{row[2]:.3f}"])

    f, ay_ = results["f"], results["amp_y"]
    keep = f <= 1.5 * F_MAX
    fft_file = os.path.join(folder, base + "_spectrum.csv")
    with open(fft_file, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_Hz", "amplitude_vertical_px"])
        for row in zip(f[keep], ay_[keep]):
            w.writerow([f"{v:.5g}" for v in row])

    env = results["env"]
    env_file = os.path.join(folder, base + "_envelope.csv")
    with open(env_file, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "amplitude_vertical_px", "amplitude_uncertainty_px",
                    "ln_amplitude", "ln_amplitude_uncertainty",
                    "local_frequency_Hz", "points_in_window"])
        for r in env:
            w.writerow([f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.3f}",
                        f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.4f}", int(r[6])])

    summary_file = os.path.join(DATA_ROOT, f"summary_{BENCH_ID}.csv")
    new = not os.path.exists(summary_file)
    with open(summary_file, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "time", "bench", "mass_g",
                        "f_vertical_Hz", "amplitude_vertical_px",
                        "duration_s", "mean_frame_rate_Hz", "frames_without_LED",
                        "timestamps", "positions_file", "envelope_file"])
        w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), BENCH_ID,
                    f"{mass:g}",
                    f"{results['fy']:.4f}", f"{results['Ay']:.2f}",
                    f"{t[-1]:.2f}", f"{results['rate']:.1f}", results["lost"],
                    source, os.path.basename(pos_file), os.path.basename(env_file)])
    return pos_file, env_file, summary_file


# ================================ Plot ==================================
def plot(t, y, results):
    """Vertical position against time (with the envelope) and its spectrum."""
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 5))
    ax_t.plot(t, y - np.nanmean(y), ":r.", ms=3)
    env = results["env"]
    if len(env):
        ax_t.errorbar(env[:, 0], env[:, 1], env[:, 2], fmt="k_", ms=10,
                      label="envelope (+/- amplitude)")
        ax_t.errorbar(env[:, 0], -env[:, 1], env[:, 2], fmt="k_", ms=10)
        ax_t.legend(loc="upper right", fontsize=8)
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Vertical position (px)")
    ax_t.set_title("Position of the LED")
    ax_f.plot(results["f"], results["amp_y"], "-b")
    ax_f.axvline(results["fy"], color="k", ls="--", lw=1)
    ax_f.set_xlim(0, F_MAX)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Amplitude (px)")
    ax_f.set_title(f"FFT: peak at {results['fy']:.3f} Hz")
    fig.suptitle("Close this window when you have recorded the frequency")
    fig.tight_layout()

    def on_click(event):
        if event.inaxes is ax_f and event.xdata is not None:
            fp, _ = find_peak(results["f"], results["amp_y"],
                              event.xdata - SNAP_WINDOW, event.xdata + SNAP_WINDOW)
            print(f"\nSelected peak = {fp:.3f} Hz")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


# ================================ Main ==================================
def main():
    plt.close("all")
    if BENCHMARK:
        benchmark()
        return
    if SIMULATE:
        mass = 50.0
        t, x, y, source = simulated()
    else:
        cam = open_camera()
        try:
            preview(cam)
            mass = ask_mass()
            print("\n\nSTART\n")
            t, x, y, source = acquire(cam)
        finally:
            cam.release()

    lost = int(np.sum(~np.isfinite(x)))
    if np.sum(np.isfinite(x)) < 10:
        sys.exit("\nThe LED was not found in the recording. Check it is on and in view.")

    f, amp_y = spectrum(t, y)
    fy, Ay = find_peak(f, amp_y)
    # Envelope of the vertical motion. Note: the plot shows the signal with its
    # mean removed, while the envelope fit also removes it via its constant term.
    env = envelope(t, y, fy)
    results = dict(f=f, amp_y=amp_y, fy=fy, Ay=Ay,
                   lost=lost, rate=(len(t) - 1) / t[-1], env=env)

    pos_file, env_file, summary_file = save_results(t, x, y, mass, results, source)

    print(f"\nRecorded {t[-1]:.1f} s at {results['rate']:.1f} frames/s "
          f"(timestamps from {source})")
    print(f"Vertical peak: {fy:.3f} Hz")
    if lost > 0.02 * len(t):
        print(f"Warning: LED lost in {lost} frames. Check it stays in view.")
    if results["rate"] < 0.8 * FPS:
        print(f"Warning: frame rate is low ({results['rate']:.1f} /s).")
    if Ay < 3 * FRAME_W / 640:
        print("Warning: the oscillation is very small. Displace the mass a bit more.")
    if fy < F_MIN + 0.1 or fy > F_MAX - 0.1:
        print("Warning: the peak is at the edge of the search range; check the plot.")
    print(f"\nData saved to:\n  {pos_file}\n  {env_file}\n  {summary_file}")

    plot(t, y, results)


if __name__ == "__main__":
    main()
