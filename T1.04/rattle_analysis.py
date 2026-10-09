"""
Rattle-damping analysis for the PHY1035 harmonic oscillator practical.

For one run's envelope (time, amplitude) this:
  1. discards everything before the largest envelope point (the release transient),
  2. finds the time at which the envelope first falls through an upper level A_HI
     and a lower level A_LO (interpolated in ln A, and requiring the envelope to
     stay below the level for a few points so a noise blip can't trigger it),
  3. returns a shifted time axis with t = 0 where the envelope crosses A_LO,
     plus the decay time between the two levels.

Every failure mode returns a plain-English status message instead of crashing,
so the run is still saved and the student is told what to do.

Can be called from the main script after each run, or run on its own:
    python3 rattle_analysis.py *_envelope.csv
which adds a 'time_shifted_s' column to each file (written as *_shifted.csv)
and prints one summary line per file.
"""
import sys
import numpy as np

# --- Settings (pixels; check these suit each bench's camera distance) ---------
A_HI = 60.0   # upper level: must be inside the rattling regime and below the
              # smallest amplitude students reliably start with
A_LO = 40.0   # lower level: still in the rattling regime, above the transition
              # (~30-33 px on bench 00); also used as the t = 0 reference
N_CONFIRM = 3  # number of consecutive points that must lie below a level


def _crossing_time(t, A, level, n_confirm=N_CONFIRM):
    """Time of the first downward crossing of `level`, interpolated in ln A.
    Returns None if the envelope never starts above the level or never
    falls (and stays) below it."""
    if A[0] < level:
        return None
    below = A < level
    for i in range(1, len(A)):
        if below[i:i + n_confirm].all() and len(below[i:i + n_confirm]) == n_confirm:
            # straight-line fit of ln A vs t over the 4 points around the
            # crossing (2 above, 2 below), so one noisy point can't move it much
            j0, j1 = max(0, i - 2), min(len(A), i + 2)
            slope, icpt = np.polyfit(t[j0:j1], np.log(A[j0:j1]), 1)
            if slope >= 0:
                return t[i]
            return (np.log(level) - icpt) / slope
    return None


def rattle_analysis(t, A, a_hi=A_HI, a_lo=A_LO):
    """Return (t_shifted, A_used, decay_time_s, status).

    t_shifted is None if the run can't be aligned; decay_time_s is None if the
    upper level wasn't reached. status is a short message for the student."""
    t = np.asarray(t, float)
    A = np.asarray(A, float)
    ok = np.isfinite(t) & np.isfinite(A) & (A > 0)
    t, A = t[ok], A[ok]
    if len(A) < 2 * N_CONFIRM:
        return None, A, None, "Too little envelope data - repeat the run."

    # drop the release transient: start from the largest envelope point
    i0 = int(np.argmax(A))
    t, A = t[i0:], A[i0:]

    t_lo = _crossing_time(t, A, a_lo)
    if t_lo is None:
        if A[0] < a_lo:
            return None, A, None, (f"Starting amplitude ({A[0]:.1f} px) is below "
                                   f"{a_lo:.0f} px - pull the box down further and repeat.")
        return None, A, None, "Envelope never settled below the lower level - repeat the run."

    t_shifted = t - t_lo
    t_hi = _crossing_time(t, A, a_hi)
    if t_hi is None:
        return t_shifted, A, None, (f"Aligned OK, but the starting amplitude ({A[0]:.1f} px) "
                                    f"is below {a_hi:.0f} px, so no decay time. "
                                    "Pull the box down further for a decay-time measurement.")
    if t_hi >= t_lo:
        return t_shifted, A, None, "Decay time could not be measured - repeat the run."
    return t_shifted, A, t_lo - t_hi, "OK"


def _process_file(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t_sh, A, decay, status = rattle_analysis(d["time_s"], d["amplitude_vertical_px"])
    if t_sh is not None:
        out = path.replace(".csv", "_shifted.csv")
        np.savetxt(out, np.column_stack([t_sh, A]), delimiter=",", fmt="%.4f",
                   header="time_shifted_s,amplitude_vertical_px", comments="")
    dt = f"{decay:.3f} s" if decay is not None else "-"
    print(f"{path.split('/')[-1]}: decay {A_HI:.0f}->{A_LO:.0f} px = {dt} | {status}")
    return decay


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if not p.endswith("_shifted.csv"):
            _process_file(p)
