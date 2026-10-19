"""
PHY1035 Driven Harmonic Oscillator - LIVE SPECTRUM DEMO

A demonstration tool, NOT a measurement script. It tracks the LED (vertical
motion only) and continuously shows
  * top:    position against time for the last TRACE_SECONDS, with the part
            that goes into the spectrum shaded;
  * bottom: the amplitude spectrum (FFT) of the last WINDOW_SECONDS of motion,
            linear amplitude by default, with the strongest peaks labelled.
After a frequency change (or a flick of the mass) the transient shows up as a
second peak at the natural frequency, which fades out while the peak at the
drive frequency stays.

Window length is a trade-off, which you can change live (up/down keys):
  * a Hann-windowed T-second window separates two peaks cleanly only if
    they are more than about 3/T apart (4 s -> 0.75 Hz, 6 s -> 0.5 Hz);
  * each spectrum averages over T seconds, so a transient that decays in a
    time short compared with T is smeared and hard to see.
  So the natural peak shows clearly when the decay time is longer than the
  window AND the drive is detuned by more than 3/T. Springs (light damping)
  with the drive about 0.6 Hz off resonance and a 6 s window are ideal.
  With the elastic bands the transient dies in a few seconds: use a short
  window (2 s) and detune by 1.5 Hz or more.

Keys (click on the plot window first):
  up / down     longer / shorter FFT window
  z             zoom on the strongest peak (press again for the full range)
  left / right  move the zoomed range
  l             linear / log amplitude
  space         freeze / unfreeze the display
  q             quit (or close the window)

Run from Geany (F5). To test without a camera, from a terminal:
    python3 drivenHarmonicOscillator_FFT_demo.py --simulate [F0] [TAU] [FD]
(natural frequency F0 Hz, amplitude decay time TAU s, drive FD Hz; the
oscillator starts at rest and is flicked every so often).
    springs: --simulate 5.08 20 5.7        bands: --simulate 6.6 2 8.2

Written 19/10/26, sharing camera/tracking code with drivenHarmonicOscillator_v5.py.
v2, 19/10/26: waterfall removed; linear amplitude by default.
"""

import os
import sys
import time
import queue
import threading
import subprocess
from bisect import bisect_left

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

SIMULATE = "--simulate" in sys.argv
if not SIMULATE:
    import cv2
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)

# =============================== Settings ===============================
VERSION = "FFT demo v2"
SPLASH_IMAGE = "snow_leopard.jpg"
SPLASH_SECONDS = 3.0
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360
FPS = 30
PIXEL_FORMAT = "YUYV"
N_BUFFERS = 4
PREVIEW_FRAMES = 20
THRESHOLD = 100

WINDOW_SECONDS = 6.0          # starting FFT window length (s)
WINDOW_CHOICES = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
UPDATE_SECONDS = 0.1          # redraw at most this often (slower if the computer can't keep up)
TRACE_SECONDS = 10.0          # length of the position-time trace (s)
F_MIN, F_MAX = 0.5, 14.0      # full frequency range (Hz); keep below FPS/2
F_STEP = 0.02                 # frequency grid of the display (Hz)
ZOOM_HALF_WIDTH = 1.5         # half-width of the zoomed range (Hz)
ZERO_PAD = 16                 # FFT zero padding (only smooths the plotted curve)
N_PEAK_LABELS = 3             # label up to this many peaks...
PEAK_DB_BELOW = 26            # ...within this many dB of the strongest
MIN_GOOD_FRACTION = 0.5       # skip the spectrum if the LED was found in fewer frames than this

CAM_PROPS = {"brightness": -64, "contrast": 50, "saturation": 128, "gamma": 40,
             "gain": 1, "auto_exposure": 1, "exposure_time_absolute": 1,
             "exposure_dynamic_framerate": 0, "exposure_auto_priority": 0}
KEYS = ["up", "down", "left", "right", "z", "l", " "]


# ======================= Shared with drivenHarmonicOscillator_v5 =======================
def show_splash():
    print(f"\nPHY1035 driven harmonic oscillator script, version {VERSION} "
          "(you should see a snow leopard)")
    folder = os.path.dirname(os.path.abspath(__file__))
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
        print(f"Warning: could not show the version picture ({err}).")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Driven harmonic oscillator - script version {VERSION}", fontsize=14)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(SPLASH_SECONDS)
    plt.close(fig)


def get_centroid(red):
    w = cv2.subtract(red, THRESHOLD)
    m = cv2.moments(w)
    if m["m00"] < 1e-6:
        return np.nan, np.nan
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def fourcc_str(cam):
    code = int(cam.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def open_camera():
    cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cam.isOpened():
        sys.exit("\nCould not open the camera. Check it is plugged in and that "
                 "no other program is using it.")
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PIXEL_FORMAT))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cam.set(cv2.CAP_PROP_FPS, FPS)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, N_BUFFERS)
    for key, val in CAM_PROPS.items():
        try:
            subprocess.run(["v4l2-ctl", "-d", f"/dev/video{CAM_INDEX}",
                            "-c", f"{key}={val}"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"Warning: could not set camera property '{key}'")
    w, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {w}x{h} {fourcc_str(cam)}, requested {FPS} frames/s")
    return cam


def hexagon_vertices(center, radius, rotation_deg=0):
    cx, cy = center
    return [(int(cx + radius * np.cos(np.deg2rad(60 * i - 30 + rotation_deg))),
             int(cy + radius * np.sin(np.deg2rad(60 * i - 30 + rotation_deg))))
            for i in range(6)]


HEX_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
               (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def preview(cam):
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
        print(f"\nWarning: LED not found in {lost} of {PREVIEW_FRAMES} preview frames.")


def camera_frames(cam, stop, info):
    """Yield (t_wall, t_cam, y, backlog) for each frame until `stop` is set.

    A background thread only reads and timestamps frames, so the camera is
    serviced at full rate whatever the display is doing; `backlog` is the
    number of frames still waiting, so the display can skip redraws rather
    than fall behind. y is flipped so that 'up' is positive.
    """
    for _ in range(3 * N_BUFFERS + 5):          # discard stale buffered frames
        t_start = time.perf_counter()
        cam.grab()
        if time.perf_counter() - t_start > 0.5 / FPS:
            break
    frames = queue.Queue()

    def reader():
        info["t0"] = t0 = time.perf_counter()
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
        _cx, cy = get_centroid(red)
        yield t, tc, red.shape[0] - 1 - cy, frames.qsize()


def valid_camera_times(tc, tw):
    if len(tc) < 2 or not np.all(np.isfinite(tc)) or not np.all(np.diff(tc) > 0):
        return False
    span_w = tw[-1] - tw[0]
    return abs((tc[-1] - tc[0]) - span_w) < 0.05 * span_w


def spectrum_on_grid(t, y, grid):
    """Amplitude spectrum (px) of y(t), interpolated onto the frequency grid.

    Removes the mean and any drift, resamples onto a uniform time grid,
    applies a Hann window and zero-pads. A sinusoid of amplitude A gives a
    peak of height ~A.
    """
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    n = len(t)
    tu = np.linspace(t[0], t[-1], n)
    yu = np.interp(tu, t, y)
    win = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * ZERO_PAD)))
    amp = np.abs(rfft(yu * win, nfft)) * 2 / win.sum()
    f = rfftfreq(nfft, tu[1] - tu[0])
    return np.interp(grid, f, amp)


# =============================== Display ===============================
class SpectrumDisplay:
    """Two-panel live display, redrawn by blitting (only the changing
    lines and labels are redrawn) so it keeps up on a Raspberry Pi."""

    def __init__(self, window=WINDOW_SECONDS):
        for name in plt.rcParams:                 # free our keys from matplotlib's defaults
            if name.startswith("keymap."):
                plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in KEYS]
        self.window = window
        self.log_amp = False
        self.paused = False
        self.closed = False
        self.xlim = (F_MIN, F_MAX)
        self.grid = np.arange(F_MIN, F_MAX + F_STEP / 2, F_STEP)
        self.amp = None
        self.render_dt = None
        self.t_render = None
        self.t_status = -np.inf
        self.status_dirty = True
        self.bgs = None
        self.need_full_draw = True

        plt.ion()
        fig = self.fig = plt.figure(figsize=(12, 7.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.8],
                              left=0.08, right=0.97, top=0.93, bottom=0.11, hspace=0.35)
        ax = self.ax_tr = fig.add_subplot(gs[0])
        ax.set_xlim(-TRACE_SECONDS, 0)
        ax.set_xlabel("Time (s, 0 = now)")
        ax.set_ylabel("Height (px)")
        ax.grid(alpha=0.3)
        self.trace_half = 5.0
        ax.set_ylim(-self.trace_half, self.trace_half)
        self.span = ax.axvspan(-self.window, 0, color="tab:orange", alpha=0.15, lw=0)
        (self.line_tr,) = ax.plot([], [], "-", color="tab:blue", lw=1, animated=True)
        self.status = fig.text(0.08, 0.955, "", fontsize=10, va="bottom", animated=True)

        ax = self.ax_sp = fig.add_subplot(gs[1])
        ax.set_ylabel("Amplitude (px)")
        ax.grid(alpha=0.3, which="both")
        (self.line_sp,) = ax.plot([], [], "-", color="tab:blue", lw=1.5, animated=True)
        (self.peak_marks,) = ax.plot([], [], "v", color="tab:red", ms=8, animated=True)
        self.peak_text = [ax.text(0, 0, "", ha="center", va="bottom", fontsize=11,
                                  color="tab:red", animated=True, clip_on=True)
                          for _ in range(N_PEAK_LABELS)]
        self.sp_top = 10.0
        self._set_amp_scale()
        # artists redrawn on every update, by region (see _blit)
        self.regions = {"trace": (self.ax_tr, [self.line_tr]),
                        "spectrum": (self.ax_sp, [self.line_sp, self.peak_marks,
                                                  *self.peak_text]),
                        "status": (None, [self.status])}

        self.ax_sp.set_xlabel("Frequency (Hz)")
        self.ax_sp.set_xlim(*self.xlim)
        fig.text(0.5, 0.01, "keys:  up/down window length    z zoom    left/right move zoom"
                 "    l linear/log    space freeze    q quit",
                 ha="center", fontsize=9, color="0.3")
        fig.canvas.mpl_connect("draw_event", self._on_draw)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        fig.canvas.mpl_connect("close_event", self._on_close)
        plt.show(block=False)
        plt.pause(0.1)

    # ---------- helpers ----------
    def _set_amp_scale(self):
        ax = self.ax_sp
        if self.log_amp:
            ax.set_yscale("log")
            ax.set_ylim(self.sp_top * 1e-4, self.sp_top)
        else:
            ax.set_yscale("linear")
            ax.set_ylim(0, self.sp_top)
        self.need_full_draw = True

    def _set_xlim(self, lo, hi):
        lo, hi = max(F_MIN, lo), min(F_MAX, hi)
        self.xlim = (lo, hi)
        self.ax_sp.set_xlim(lo, hi)
        self.need_full_draw = True

    def _region_bbox(self, name):
        ax = self.regions[name][0]
        if ax is not None:
            return ax.bbox
        from matplotlib.transforms import Bbox       # status strip above the trace
        fb = self.fig.bbox
        return Bbox.from_extents(fb.x0, self.ax_tr.bbox.y1 + 2, fb.x1, fb.y1)

    def _on_draw(self, _event):
        canvas = self.fig.canvas
        self.bgs = {name: canvas.copy_from_bbox(self._region_bbox(name))
                    for name in self.regions}
        for _ax, arts in self.regions.values():
            for a in arts:
                self.fig.draw_artist(a)


    def _on_close(self, _event):
        self.closed = True

    def _on_key(self, event):
        k = event.key
        if k in ("up", "down"):
            i = int(np.argmin(np.abs(np.array(WINDOW_CHOICES) - self.window)))
            i = min(i + 1, len(WINDOW_CHOICES) - 1) if k == "up" else max(i - 1, 0)
            self.window = WINDOW_CHOICES[i]
            self.span.remove()
            self.span = self.ax_tr.axvspan(-self.window, 0, color="tab:orange",
                                           alpha=0.15, lw=0)
            self.need_full_draw = True
        elif k == "z":
            if self.xlim != (F_MIN, F_MAX) or self.amp is None:
                self._set_xlim(F_MIN, F_MAX)
            else:
                fc = self.grid[np.nanargmax(self.amp)]
                self._set_xlim(fc - ZOOM_HALF_WIDTH, fc + ZOOM_HALF_WIDTH)
        elif k in ("left", "right") and self.xlim != (F_MIN, F_MAX):
            step = 0.25 * (1 if k == "right" else -1)
            lo, hi = self.xlim
            if F_MIN <= lo + step and hi + step <= F_MAX:
                self._set_xlim(lo + step, hi + step)
        elif k == "l":
            self.log_amp = not self.log_amp
            self._set_amp_scale()
        elif k == " ":
            self.paused = not self.paused
            self.status.set_text(self.status.get_text() + ("   FROZEN" if self.paused else ""))
            self.status_dirty = True
            self._blit()

    def _blit(self):
        """Redraw what changed: the trace and spectrum every time, the status
        line only when its text changes."""
        canvas = self.fig.canvas
        if self.need_full_draw or self.bgs is None:
            self.need_full_draw = False
            canvas.draw()                      # triggers _on_draw
        else:
            names = ["trace", "spectrum"]
            if self.status_dirty:
                names.append("status")
            for name in names:
                canvas.restore_region(self.bgs[name])
                for a in self.regions[name][1]:
                    self.fig.draw_artist(a)
                canvas.blit(self._region_bbox(name))
        self.status_dirty = False
        canvas.flush_events()

    # ---------- per-update ----------
    def update(self, tw, tc, y, lag=None):
        if self.paused:
            self.fig.canvas.flush_events()
            return
        now = time.perf_counter()
        if self.t_render is not None:
            dt = now - self.t_render
            self.render_dt = dt if self.render_dt is None else 0.8 * self.render_dt + 0.2 * dt
        self.t_render = now
        t_now = tw[-1]

        # position-time trace
        i0 = bisect_left(tw, t_now - TRACE_SECONDS)
        yt = y[i0:] - np.nanmean(y[i0:]) if np.isfinite(y[i0:]).any() else y[i0:]
        self.line_tr.set_data(tw[i0:] - t_now, yt)
        need = 1.15 * np.nanmax(np.abs(yt)) if np.isfinite(yt).any() else self.trace_half
        if need > self.trace_half or need < 0.4 * self.trace_half:
            self.trace_half = max(1.5 * need, 2.0)
            self.ax_tr.set_ylim(-self.trace_half, self.trace_half)
            self.need_full_draw = True

        # spectrum of the last `window` seconds
        i0 = bisect_left(tw, t_now - self.window)
        good = np.isfinite(y[i0:])
        span = tw[-1] - tw[i0]
        ok = span >= 1.0 and good.sum() >= MIN_GOOD_FRACTION * span * FPS
        if ok:
            t = tc[i0:] if valid_camera_times(tc[i0:], tw[i0:]) else tw[i0:]
            self.amp = spectrum_on_grid(t[good] - t[good][0], y[i0:][good], self.grid)
            self._update_spectrum()
        else:
            self.amp = None
            self.line_sp.set_data([], [])
            self.peak_marks.set_data([], [])
            for txt in self.peak_text:
                txt.set_text("")

        if now - self.t_status >= 1.0 or not ok:      # status: at most once a second
            msg = (f"FFT window {self.window:g} s (shaded): separates peaks more than "
                   f"~{3 / self.window:.2f} Hz apart")
            if self.render_dt:
                msg += f"    |    {1 / self.render_dt:.0f} updates/s"
            if lag is not None:
                msg += f", delay {lag:.2f} s"
            if not ok:
                msg += "    |    LED NOT FOUND" if span >= 1.0 else "    |    collecting data..."
            if msg != self.status.get_text():
                self.status.set_text(msg)
                self.status_dirty = True
            self.t_status = now
        self._blit()

    def _update_spectrum(self):
        amp = self.amp
        self.line_sp.set_data(self.grid, amp)
        lo, hi = np.searchsorted(self.grid, self.xlim)
        a = amp[lo:hi]
        amax = a.max()
        if self.log_amp:
            if 1.5 * amax > self.sp_top or 1.5 * amax < self.sp_top / 30:
                self.sp_top = 10 ** np.ceil(np.log10(1.5 * amax))
                self._set_amp_scale()
        elif amax > self.sp_top or amax < 0.6 * self.sp_top:
            self.sp_top = 1.25 * amax
            self._set_amp_scale()
        # label the strongest peaks
        idx, _ = find_peaks(np.log10(a + 1e-9), prominence=0.3,
                            height=np.log10(amax) - PEAK_DB_BELOW / 20)
        idx = idx[np.argsort(a[idx])[::-1][:N_PEAK_LABELS]] + lo
        self.peak_marks.set_data(self.grid[idx], amp[idx] * (1.25 if self.log_amp else 1.03))
        for j, txt in enumerate(self.peak_text):
            if j < len(idx):
                i = idx[j]
                ytxt = amp[i] * 1.6 if self.log_amp else amp[i] + 0.05 * self.sp_top
                txt.set_position((self.grid[i], ytxt))
                txt.set_text(f"{self.grid[i]:.2f} Hz")
            else:
                txt.set_text("")


# ============================== Main loop ===============================
def run(frames, display, info, realtime=True):
    tw, tc, ys = [], [], []
    keep = int((max(WINDOW_CHOICES[-1], TRACE_SECONDS) + 2) * FPS)
    last = -np.inf
    try:
        for t, t_cam, y, backlog in frames:
            tw.append(t)
            tc.append(t_cam)
            ys.append(y)
            if display.closed:
                break
            now = time.perf_counter() if realtime else t
            if backlog > 1 or now - last < UPDATE_SECONDS:
                continue
            last = now
            if len(tw) > 2 * keep:
                del tw[:-keep], tc[:-keep], ys[:-keep]
            lag = time.perf_counter() - info["t0"] - t if "t0" in info else None
            display.update(np.array(tw), np.array(tc), np.array(ys), lag)
    except KeyboardInterrupt:
        print("\nStopped.")


# ============================== Simulation ==============================
def simulated_frames(f0, tau, fd, realtime=True, duration=np.inf):
    """Driven, damped oscillator, starting from rest when the drive is
    switched on and flicked (velocity kick) every KICK seconds."""
    rng = np.random.default_rng(1)
    w0, wd, g = 2 * np.pi * f0, 2 * np.pi * fd, 2 / tau
    F = 15 * np.hypot(w0 ** 2 - wd ** 2, g * wd)     # 15 px steady-state amplitude
    kick_every = max(30.0, 5 * tau)
    n_sub, dt = 10, 1 / FPS / 10
    x = v = t = 0.0
    next_kick = kick_every
    t_wall0 = time.perf_counter()

    def acc(x, v, t):
        return F * np.cos(wd * t) - g * v - w0 ** 2 * x

    while t < duration:
        for _ in range(n_sub):
            k1x, k1v = v, acc(x, v, t)
            k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t + dt / 2)
            k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t + dt / 2)
            k4x, k4v = v + dt * k3v, acc(x + dt * k3x, v + dt * k3v, t + dt)
            x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            t += dt
        if t >= next_kick:
            v += w0 * 20.0                            # flick: ~20 px at f0
            next_kick += kick_every
        if realtime:
            wait = t - (time.perf_counter() - t_wall0)
            if wait > 0:
                time.sleep(wait)
        yield t, np.nan, 180 + x + rng.normal(0, 0.3), 0


# ================================ Main ==================================
def main():
    plt.close("all")
    show_splash()
    info = {}
    if SIMULATE:
        args = sys.argv[sys.argv.index("--simulate") + 1:]
        nums = []
        for a in args[:3]:
            try:
                nums.append(float(a))
            except ValueError:
                break
        f0, tau, fd = (nums + [5.08, 20.0, 5.7][len(nums):])[:3]
        print(f"\nSimulation: natural frequency {f0:g} Hz, decay time {tau:g} s, "
              f"drive {fd:g} Hz (starts from rest, flicked every {max(30, 5 * tau):g} s)")
        display = SpectrumDisplay()
        run(simulated_frames(f0, tau, fd), display, info, realtime=True)
    else:
        cam = open_camera()
        stop = threading.Event()
        try:
            preview(cam)
            input("\nPress Enter to start the live spectrum.")
            display = SpectrumDisplay()
            run(camera_frames(cam, stop, info), display, info)
        finally:
            stop.set()
            time.sleep(0.2)
            cam.release()


if __name__ == "__main__":
    main()
