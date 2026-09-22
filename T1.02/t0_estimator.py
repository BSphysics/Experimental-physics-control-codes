"""
t0 auto-estimation from raw freefall data.

Physical basis (established during the PHY1035 analysis work):
a passive projectile can never fall FURTHER than ideal free-fall at any
instant, so for the model s = 0.5 g (t - t0)^2 every point obeys

    0.5 g (t_i - t0)^2 >= s_i   =>   t0 <= t_i - sqrt(2 s_i / g)   (s_i > 0)

The release delay t0 is a property of the servo, not the ball, so it must
come out the same (~0.1 s) for a dense ball and a very draggy one. That is
exactly why this constraint is the right tool: drag only ever makes the ball
fall SHORT of free-fall, which pushes t_i - sqrt(2 s_i/g) *up*, so the lower
envelope of those bounds is set by the least-draggy (earliest) points and is
insensitive to how much drag develops later. A least-squares fit of the
free-fall model, by contrast, drives t0 far too late on a draggy ball
(observed: 0.28 s on the hollow plastic ball) because it tries to bend a
free-fall parabola through a non-free-fall curve - do NOT use LSQ here.

Two things wreck a naive minimum of the bounds, and both are handled below:
  * Random positive distance noise (and, on this IR sensor, a ranging bias
    that makes part of the mid-flight appear to BEAT free-fall) produce a few
    bounds below the true t0. Fix: restrict to an early displacement window
    (below the sensor's apparent-overshoot region) and take a low percentile
    of the bounds rather than the raw minimum, and estimate from the 5-run
    MEAN trajectory (sqrt(5) less noise), not run by run.
  * Points still in the pre-release baseline have s dominated by noise, where
    sqrt(2 s/g) is meaningless. Fix: a displacement floor.
"""
import numpy as np

G = 9.81


def estimate_t0_from_raw(times_s, distances_m,
                         min_disp_m=0.03, max_disp_m=0.20,
                         low_pct=10.0, min_points=4):
    """
    Estimate the release-delay t0 (seconds) from one trajectory (ideally the
    5-run mean) via the "never beats free-fall" constraint on the early arc.

    Parameters
    ----------
    times_s : array, seconds, zeroed so the acquisition window starts at 0
    distances_m : array, metres, zeroed so pre-release baseline ~ 0
    min_disp_m : ignore points below this displacement (pre-release noise floor)
    max_disp_m : only use points below this displacement - the early arc that
                 is still close to free fall for any projectile, and below the
                 region where this sensor's ranging bias fakes an overshoot.
                 Capped at 0.6 * peak displacement for short falls.
    low_pct : percentile of the in-window bounds to take as t0 (a robust lower
              envelope: rejects the odd noise dip without chasing the raw min).
    min_points : if the window holds fewer than this, widen to every point
                 above min_disp_m before giving up.

    Returns
    -------
    t0 estimate in seconds (>= 0), or None if not estimable.
    """
    t = np.asarray(times_s, dtype=float)
    s = np.asarray(distances_m, dtype=float)
    if t.size == 0 or s.size == 0:
        return None

    s_peak = np.nanmax(s)
    if not np.isfinite(s_peak) or s_peak <= min_disp_m:
        return None

    hi = min(max_disp_m, 0.6 * s_peak)
    window = (s > min_disp_m) & (s < hi)
    if window.sum() < min_points:
        window = s > min_disp_m
    if window.sum() < min_points:
        return None

    bounds = t[window] - np.sqrt(2.0 * np.clip(s[window], 0.0, None) / G)
    t0 = float(np.percentile(bounds, low_pct))
    return max(0.0, t0)


def estimate_t0_multi(runs, **kwargs):
    """
    Combine several runs (list of (times_s, distances_m)) into one t0 estimate.

    The bound-based estimator wants low noise, so we build the row-aligned mean
    trajectory across the runs (the same averaging the spreadsheet does, ~sqrt(N)
    less noise) and estimate once on that, rather than estimating per run and
    taking a median of noisy minima.
    """
    runs = [(np.asarray(t, float), np.asarray(s, float)) for t, s in runs if len(t) >= 5]
    if not runs:
        return None

    max_len = max(len(t) for t, _ in runs)
    t_mean, s_mean = [], []
    for i in range(max_len):
        ts = [t[i] for t, _ in runs if i < len(t)]
        ss = [s[i] for _, s in runs if i < len(s)]
        if ts:
            t_mean.append(sum(ts) / len(ts))
            s_mean.append(sum(ss) / len(ss))
    return estimate_t0_from_raw(np.array(t_mean), np.array(s_mean), **kwargs)
