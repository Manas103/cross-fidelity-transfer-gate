"""Procedural route generation: a piecewise-curvature centerline, integrated
into a world-frame polyline once per episode and shared by both fidelity
tiers (same route, same initial conditions; see sim.py for what differs)."""
import math
import numpy as np

DS = 0.25  # route sampling resolution, meters


def generate_route(rng, n_segments=6, seg_len_range=(8.0, 15.0), curvature_range=0.05):
    """Returns a dict of arrays (x, y, theta, s) sampled every DS meters."""
    seg_lengths = rng.uniform(seg_len_range[0], seg_len_range[1], size=n_segments)
    seg_curvatures = rng.uniform(-curvature_range, curvature_range, size=n_segments)
    boundaries = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = boundaries[-1]

    s_samples = np.arange(0.0, total_length, DS)
    curvature_at_s = np.interp(
        s_samples,
        (boundaries[:-1] + boundaries[1:]) / 2.0,
        seg_curvatures,
    )

    theta = np.concatenate([[0.0], np.cumsum(curvature_at_s[:-1] * DS)])
    x = np.concatenate([[0.0], np.cumsum(np.cos(theta[:-1]) * DS)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(theta[:-1]) * DS)])

    return dict(x=x, y=y, theta=theta, s=s_samples, total_length=float(s_samples[-1]))


def nearest_index(route, ego_x, ego_y, hint_idx, search_window=80):
    """Find the route sample nearest (ego_x, ego_y), searching near hint_idx
    (the ego moves forward along the route, so a local search is enough and
    is far cheaper than a full scan every step)."""
    lo = max(0, hint_idx - 5)
    hi = min(len(route["x"]), hint_idx + search_window)
    xs = route["x"][lo:hi]
    ys = route["y"][lo:hi]
    d2 = (xs - ego_x) ** 2 + (ys - ego_y) ** 2
    return lo + int(np.argmin(d2))


def lateral_and_heading_error(route, idx, ego_x, ego_y, ego_theta):
    rx, ry, rtheta = route["x"][idx], route["y"][idx], route["theta"][idx]
    dx, dy = ego_x - rx, ego_y - ry
    lateral = -math.sin(rtheta) * dx + math.cos(rtheta) * dy
    heading_err = math.atan2(math.sin(ego_theta - rtheta), math.cos(ego_theta - rtheta))
    return lateral, heading_err


def local_frame_points(route, idx, ego_x, ego_y, ego_theta, lookahead_m):
    """Route points within [idx, idx + lookahead] transformed into the ego's
    local frame (x forward, y left-positive)."""
    span = int(lookahead_m / DS) + 2
    lo, hi = idx, min(len(route["x"]), idx + span)
    rx, ry = route["x"][lo:hi], route["y"][lo:hi]
    dx, dy = rx - ego_x, ry - ego_y
    c, s = math.cos(ego_theta), math.sin(ego_theta)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return lx, ly
