"""
PHY1035 Driven Harmonic Oscillator - RESONANCE FINDER

A quick search tool, NOT a measurement script. It tracks the LED continuously
and, every UPDATE_SECONDS, reports the frequency and amplitude of the motion
over the last WINDOW_SECONDS. Change the frequency on the function generator
while it runs and watch the amplitude: the live plot builds up amplitude
against frequency so the resonance shows up as you sweep.

Use it to find roughly where the resonance is, then take the real data with
drivenHarmonicOscillator_v4.py (springs) or _v5.py (elastic bands).

Why the readings are only approximate
  * After every frequency change the motion contains a transient at the
    natural frequency as well as the response at the drive frequency. Until
    it dies away the reading wobbles (beats). A reading is marked "steady"
    only when the readings over the last STEADY_SPAN seconds agree with it in
    frequency and amplitude; only steady readings are drawn as filled points.
      - Elastic bands (heavy damping): steady within a few seconds.
      - Springs (light damping): can take a minute or more. Use coarse steps
        (about 0.05 Hz) and wait for "steady" before reading.
  * The script does not know the drive frequency: it finds the strongest
    oscillation in each window itself (spectrum peak, refined by a sine fit).
    A 4 s window cannot separate two frequencies closer than about 0.25 Hz,
    so a transient close to the drive frequency shows up as a wobbling
    amplitude rather than as a second peak.

Run from Geany (F5). Stop by closing the plot window (or Ctrl+C in the
terminal). The readings are saved to a csv file when it stops.
To test without a camera, from a terminal:
    python3 resonance_finder.py --simulate [F0] [TAU]
(simulated oscillator, natural frequency F0 Hz and amplitude decay time TAU s;
the drive is stepped through the resonance automatically).

Written 14/10/26, sharing camera/tracking code with drivenHarmonicOscillator_v5.py.
v2, 14/10/26: plot window turns pale green when the reading is steady, pale red when not.
v3, 14/10/26: (1) reports the drive-frequency motion even when its 2nd harmonic is
stronger (seen with the elastic bands near 6.6 Hz, where a mode near 13.2 Hz
amplifies the harmonic); (2) the steadiness test can no longer be loosened by a
second motion inflating the fit uncertainty (falling readings during the
parametric swinging were being marked steady).
"""

import os
import sys
import csv
import time
import queue
import threading
import subprocess
from bisect import bisect_left
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

SIMULATE = "--simulate" in sys.argv
if not SIMULATE:
    import cv2
    if hasattr(cv2, "setLogLevel"):       # not present in every OpenCV build
        cv2.setLogLevel(0)

# =============================== Settings ===============================
VERSION = "resonance finder v3"
SPLASH_IMAGE = "snow_leopard.jpg"   # shown at start-up so students can check the version
SPLASH_SECONDS = 3.0
BENCH_ID = "bench_00"         # set this on each lab PC (appears in file names)
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360   # same camera settings as drivenHarmonicOscillator_v5.py
FPS = 30
PIXEL_FORMAT = "YUYV"
N_BUFFERS = 4
PREVIEW_FRAMES = 20
THRESHOLD = 100               # LED threshold on the red channel (0-255)

# --- live readings ---
WINDOW_SECONDS = 4.0          # each reading uses the last this-many seconds of motion
UPDATE_SECONDS = 2.0          # a new reading this often (windows overlap)
STEADY_SPAN = 4.0             # a reading is "steady" if all readings from the last this-many
STEADY_DF = 0.01              # seconds agree with it: frequency within this (Hz)...
STEADY_DA = 0.03              # ...and amplitude within this fraction (or 3 sigma, if larger)
STEADY_NOISE_CAP = 0.3        # ...but the 3-sigma allowance is capped at this (px): sigma also
                              # grows when a second motion (e.g. swinging) is present, and must
                              # not then hide a real change in amplitude
HARMONIC_RATIO = 0.25         # if the strongest peak has a peak at half its frequency at least
                              # this fraction of its height, the two are a fundamental and its
                              # 2nd harmonic (or a drive and a half-frequency swing): see
                              # choose_fundamental()
COLOUR_STEADY = "#d9f2d9"     # plot window background when the reading is steady (pale green)
COLOUR_CHANGING = "#f8d7d7"   # ...and when it is changing or the LED is lost (pale red)
MIN_GOOD_FRACTION = 0.8       # skip a window if the LED was found in fewer frames than this
# --- spectrum ---
F_MIN, F_MAX = 0.5, 14.0      # frequency range searched (Hz); keep below FPS/2
ZERO_PAD = 8                  # FFT zero-padding factor
FIT_DF = 0.02                 # default frequency search range for sine fits (+/- Hz)

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Driven harmonic oscillator data")

CAM_PROPS = {"brightness": -64, "contrast": 50, "saturation": 128, "gamma": 40,
             "gain": 1, "auto_exposure": 1, "exposure_time_absolute": 1,
             "exposure_dynamic_framerate": 0, "exposure_auto_priority": 0}


# ======================= Shared with drivenHarmonicOscillator_v5 =======================
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



# ============================ Live readings =============================
def valid_camera_times(tc, tw):
    """True if the camera timestamps in this window look usable."""
    if not np.all(np.isfinite(tc)) or not np.all(np.diff(tc) > 0):
        return False
    span_w = tw[-1] - tw[0]
    return abs((tc[-1] - tc[0]) - span_w) < 0.05 * span_w


def choose_fundamental(f, amp, f_peak, a_peak, f_prev):
    """Decide which frequency to report when the spectrum has two peaks in a
    1:2 ratio (the strongest, f_peak, plus one at half or twice it).

    That pattern has two possible causes, which one window cannot tell apart:
      * the drive frequency with a strong 2nd harmonic at twice it, or
      * the drive frequency with a parametric swing at half of it.
    The drive frequency is changed in small steps, so report the candidate
    nearer the previous reading. With no previous reading, take the lower
    frequency when the strongest peak is the upper one (a harmonic is the
    common case) and otherwise the strongest. Returns (frequency, note).
    """
    others = []
    for f_other in (f_peak / 2, 2 * f_peak):
        if F_MIN <= f_other <= F_MAX:
            _, a_other = find_peak(f, amp, f_other - 0.15, f_other + 0.15)
            if a_other >= HARMONIC_RATIO * a_peak:
                others.append(f_other)
    if not others:
        return f_peak, ""
    cands = [f_peak] + others
    if f_prev is not None and np.isfinite(f_prev):
        f_use = min(cands, key=lambda c: abs(np.log(c / f_prev)))
    else:
        f_use = f_peak / 2 if f_peak / 2 in others else f_peak
    f_other = max(c for c in cands if c != f_use) if f_use < max(cands) else min(cands)
    if f_other > f_use:
        return f_use, f"(also motion at {f_other:.2f} Hz: 2nd harmonic)"
    return f_use, f"(also motion at {f_other:.2f} Hz: half-frequency swinging?)"


def measure_window(tw, tc, y, f_prev=None):
    """Frequency and amplitude of the oscillation in one window.

    The spectrum peak gives the frequency to within about one bin
    (1/WINDOW_SECONDS); if it is accompanied by a peak at half its frequency,
    choose_fundamental() decides which to report. Two sine fits (coarse, then
    fine frequency grid) refine the frequency and give the amplitude.
    Returns (f, A, sigma_A, note), or None if the LED was lost.
    """
    good = np.isfinite(y)
    if good.sum() < max(20, MIN_GOOD_FRACTION * WINDOW_SECONDS * FPS):
        return None
    t = tc if valid_camera_times(tc, tw) else tw
    t, y = t[good], y[good]
    t = t - t[0]
    f, amp = spectrum(t, y)
    f_peak, a_peak = find_peak(f, amp, F_MIN, F_MAX)
    f_use, note = choose_fundamental(f, amp, f_peak, a_peak, f_prev)
    r = fit_sinusoid(t, y, f_use, df=0.15, n_try=31)             # 0.01 Hz grid
    if r is None:
        return None
    r = fit_sinusoid(t, y, r[2], df=0.01, n_try=21)              # 0.001 Hz grid
    A, sA, f_fit, _ = r
    return f_fit, A, sA, note


class Readings:
    """List of readings with the steadiness test."""

    def __init__(self):
        self.t, self.f, self.A, self.sA, self.steady = [], [], [], [], []

    def add(self, t, f, A, sA):
        # Steady = every reading from the last STEADY_SPAN seconds agrees with
        # this one. Comparing over several seconds (not just with the previous
        # reading) catches a slow ring-up, which changes little per update.
        steady = False
        if self.t and self.t[0] <= t - STEADY_SPAN:
            i0 = bisect_left(self.t, t - STEADY_SPAN)
            if self.t[i0] > t - STEADY_SPAN and i0 > 0:
                i0 -= 1                      # include one reading at least STEADY_SPAN old
            f_prev = np.array(self.f[i0:])
            A_prev = np.array(self.A[i0:])
            dA_tol = max(STEADY_DA * A,
                         min(3 * max([sA] + self.sA[i0:]), STEADY_NOISE_CAP))
            steady = bool(np.all(np.isfinite(A_prev))
                          and np.all(np.abs(f_prev - f) < STEADY_DF)
                          and np.all(np.abs(A_prev - A) < dA_tol))
        for lst, v in ((self.t, t), (self.f, f), (self.A, A), (self.sA, sA),
                       (self.steady, steady)):
            lst.append(v)
        return steady

    def add_lost(self, t):
        for lst, v in ((self.t, t), (self.f, np.nan), (self.A, np.nan),
                       (self.sA, np.nan), (self.steady, False)):
            lst.append(v)

    def arrays(self):
        return (np.array(self.t), np.array(self.f), np.array(self.A),
                np.array(self.sA), np.array(self.steady, dtype=bool))

    def best(self):
        """Steady reading with the largest amplitude (or None)."""
        t, f, A, sA, st = self.arrays()
        if not st.any():
            return None
        i = np.flatnonzero(st)[np.argmax(A[st])]
        return f[i], A[i]


# =============================== Live plot ==============================
class LivePlot:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_f, self.ax_t) = plt.subplots(1, 2, figsize=(13, 5.5))
        self.fig.set_facecolor(COLOUR_CHANGING)
        self.closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        ax = self.ax_f
        self.sc_unsteady = ax.scatter([], [], s=18, facecolors="none",
                                      edgecolors="0.6", label="changing (transient)")
        self.sc_steady = ax.scatter([], [], s=30, c="tab:blue", label="steady")
        self.sc_latest = ax.scatter([], [], s=140, facecolors="none",
                                    edgecolors="red", linewidths=2, label="latest")
        ax.set_xlabel("Frequency of the motion (Hz)")
        ax.set_ylabel("Amplitude (px)")
        ax.set_title("Amplitude against frequency")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

        ax = self.ax_t
        (self.line_A,) = ax.plot([], [], "-", color="tab:blue", lw=1)
        self.sc_t_steady = ax.scatter([], [], s=14, c="tab:blue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (px)")
        ax.set_title("Amplitude against time (wait for it to level off)")
        ax.grid(alpha=0.3)
        self.ax_f2 = ax.twinx()
        (self.line_f,) = self.ax_f2.plot([], [], "-", color="tab:orange", lw=1)
        self.ax_f2.set_ylabel("Frequency (Hz)", color="tab:orange")
        self.fig.tight_layout(rect=(0, 0, 1, 0.9))
        self.suptitle = self.fig.suptitle("Waiting for the first reading...", fontsize=15)
        plt.show(block=False)
        plt.pause(0.05)

    def _on_close(self, _event):
        self.closed = True

    @staticmethod
    def _limits(ax, x, y, ymin0=True):
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return
        x, y = x[ok], y[ok]
        dx = max(x.max() - x.min(), 0.05)
        ax.set_xlim(x.min() - 0.08 * dx, x.max() + 0.08 * dx)
        top = y.max() * 1.1 + 1e-9
        ax.set_ylim(0 if ymin0 else y.min() - 0.1 * (top - y.min()), top)

    def update(self, readings):
        t, f, A, sA, st = readings.arrays()
        ok = np.isfinite(A)
        self.sc_steady.set_offsets(np.c_[f[st], A[st]] if st.any() else np.empty((0, 2)))
        uns = ok & ~st
        self.sc_unsteady.set_offsets(np.c_[f[uns], A[uns]] if uns.any() else np.empty((0, 2)))
        if ok[-1]:
            self.sc_latest.set_offsets([[f[-1], A[-1]]])
        self._limits(self.ax_f, f, A)

        self.line_A.set_data(t, A)
        self.sc_t_steady.set_offsets(np.c_[t[st], A[st]] if st.any() else np.empty((0, 2)))
        self.line_f.set_data(t, f)
        self._limits(self.ax_t, t, A)
        if ok.any():
            lo, hi = np.nanmin(f), np.nanmax(f)
            pad = max(0.02, 0.1 * (hi - lo))
            self.ax_f2.set_ylim(lo - pad, hi + pad)

        if ok[-1]:
            state = "STEADY" if st[-1] else "changing - wait"
            text = f"{f[-1]:.3f} Hz    amplitude {A[-1]:.1f} px    ({state})"
        else:
            text = "LED not found - check it is on and in view"
        best = readings.best()
        if best:
            text += f"\nlargest steady amplitude so far: {best[1]:.1f} px at {best[0]:.3f} Hz"
        self.suptitle.set_text(text)
        # whole window pale green when steady, pale red otherwise (the plot
        # areas stay white so the points remain easy to see)
        self.fig.set_facecolor(COLOUR_STEADY if (ok[-1] and st[-1]) else COLOUR_CHANGING)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ============================== Main loop ===============================
def run(frames, live=True):
    """Consume frames, make a reading every UPDATE_SECONDS, until stopped."""
    tw, tc, ys = [], [], []
    readings = Readings()
    plot = LivePlot() if live else None
    next_update = WINDOW_SECONDS
    keep = int(3 * WINDOW_SECONDS * FPS)          # frames kept in memory
    print(f"\nReadings every {UPDATE_SECONDS:g} s over the last {WINDOW_SECONDS:g} s. "
          "Change the frequency whenever you like.\nClose the plot window "
          "(or press Ctrl+C here) to stop.\n")
    print("   time     frequency    amplitude")
    try:
        for t, t_cam, _x, y in frames:
            tw.append(t)
            tc.append(t_cam)
            ys.append(y)
            if t < next_update:
                continue
            next_update += UPDATE_SECONDS
            i0 = bisect_left(tw, t - WINDOW_SECONDS)
            valid_f = [v for v in readings.f if np.isfinite(v)]
            r = measure_window(np.array(tw[i0:]), np.array(tc[i0:]), np.array(ys[i0:]),
                               valid_f[-1] if valid_f else None)
            if r is None:
                readings.add_lost(t)
                print(f"  {t:5.0f} s    LED not found")
            else:
                f, A, sA, note = r
                steady = readings.add(t, f, A, sA)
                print(f"  {t:5.0f} s   {f:7.3f} Hz   {A:7.1f} px   "
                      f"{'steady' if steady else 'changing'}  {note}")
            if len(tw) > 2 * keep:                   # keep memory bounded
                del tw[:-keep], tc[:-keep], ys[:-keep]
            if plot:
                plot.update(readings)
                if plot.closed:
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    return readings, plot


def save_readings(readings):
    t, f, A, sA, st = readings.arrays()
    if not len(t):
        return None
    now = datetime.now()
    folder = os.path.join(DATA_ROOT, now.strftime("%Y_%m_%d"))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{now.strftime('%H_%M_%S')}_{BENCH_ID}_resonance_finder.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "frequency_Hz", "amplitude_px",
                    "amplitude_uncertainty_px", "steady"])
        for row in zip(t, f, A, sA, st):
            w.writerow([f"{row[0]:.1f}", f"{row[1]:.4f}", f"{row[2]:.3f}",
                        f"{row[3]:.3f}", "yes" if row[4] else "no"])
    return path


# ============================== Simulation ==============================
def simulated_frames(f0, tau, info):
    """Driven, damped oscillator integrated numerically (so transients after
    each frequency change are real). The drive steps from f0 - 0.3 Hz to
    f0 + 0.3 Hz in 0.05 Hz steps, DWELL seconds each, then the run ends."""
    rng = np.random.default_rng(1)
    w0, g = 2 * np.pi * f0, 2 / tau            # g = 2 x amplitude decay rate
    F = 40 * w0 * g                            # 40 px at resonance
    dwell = max(14.0, 3 * tau)
    steps = f0 + np.arange(-0.30, 0.3001, 0.05)
    info["schedule"] = (steps, dwell)
    n_sub = 10
    dt = 1 / FPS / n_sub
    x = v = phase = t = 0.0

    def acc(x, v, ph):
        return F * np.cos(ph) - g * v - w0 ** 2 * x

    for fd in steps:
        wd = 2 * np.pi * fd
        t_end = t + dwell
        while t < t_end:
            for _ in range(n_sub):                 # RK4
                k1x, k1v = v, acc(x, v, phase)
                k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, phase + 0.5 * dt * wd)
                k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, phase + 0.5 * dt * wd)
                k4x, k4v = v + dt * k3v, acc(x + dt * k3x, v + dt * k3v, phase + dt * wd)
                x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
                v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
                phase += dt * wd
                t += dt
            y = 180 + x + rng.normal(0, 0.3)
            yield t, np.nan, 320.0, y          # no camera timestamps -> computer clock


# ================================ Main ==================================
def main():
    plt.close("all")
    show_splash()
    info = {}
    if SIMULATE:
        args = sys.argv[sys.argv.index("--simulate") + 1:]
        nums = []
        for a in args[:2]:
            try:
                nums.append(float(a))
            except ValueError:
                break
        f0 = nums[0] if nums else 5.08
        tau = nums[1] if len(nums) > 1 else 20.0
        print(f"\nSimulation: natural frequency {f0:g} Hz, amplitude decay time {tau:g} s "
              f"(Q = {np.pi * f0 * tau:.0f})")
        live = os.environ.get("MPLBACKEND", "").lower() != "agg"
        readings, plot = run(simulated_frames(f0, tau, info), live=live)
    else:
        cam = open_camera()
        stop = threading.Event()
        try:
            preview(cam)
            input("\nSet the function generator to your first frequency, "
                  "then press Enter to start.")
            readings, plot = run(camera_frames(cam, stop, info))
        finally:
            stop.set()
            time.sleep(0.2)
            cam.release()

    path = save_readings(readings)
    best = readings.best()
    print()
    if best:
        print(f"Largest steady amplitude: {best[1]:.1f} px at {best[0]:.3f} Hz.")
        print("The resonance is near there. Now measure the resonance curve properly with "
              "drivenHarmonicOscillator_v4.py (springs) or _v5.py (elastic bands).")
    else:
        print("No steady readings. Wait for the amplitude to level off after "
              "each frequency change.")
    if path:
        print(f"\nReadings saved to:\n  {path}")
    if plot and not plot.closed:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
