"""Two-tier kinematic bicycle simulator sharing one route generator.

Tier A ("fast"): plain kinematic bicycle, clean rasterized frame.
Tier B ("fidelity"): adds a tire-slip lateral perturbation to the dynamics,
plus rolling-shutter row shear and speed-scaled exposure noise to the frame.
Both tiers are driven by the same route, the same initial pose, and the
same random seed, so a rollout pair is exactly paired.
"""
import math
import numpy as np

from . import route as R

DT = 0.1
MAX_STEPS = 220
WHEELBASE = 2.8
STEER_LIMIT = 0.5
CRUISE_SPEED = 6.0
ACCEL_LIMIT = 2.0
OFFROAD_THRESHOLD = 2.5  # meters of lateral error

IMG_SIZE = 48
LOOKAHEAD_M = 24.0
LATERAL_HALF_RANGE_M = 6.0
RENDER_HALF_WIDTH_M = 2.0

SLIP_UNDERSTEER_COEFF = 0.5  # tire slip: fraction the yaw response is understeered by, ~ coeff * v^2 * |steer|
ROLLING_SHUTTER_PERIOD = 0.04  # seconds, full-frame readout time
ROLLING_SHUTTER_GAIN = 500.0    # columns of shift per (rad/s * s)
EXPOSURE_NOISE_BASE = 0.12
EXPOSURE_NOISE_SPEED_COEF = 0.04


def step_bicycle(state, steer, accel, dt=DT, tire_slip=False):
    """Clean kinematic bicycle step. With tire_slip, the yaw response to a
    given steer command is understeered (reduced) at speed, as real tires
    do past their linear slip range; the lateral/forward position update
    is otherwise unchanged (slip is a steering-to-yaw-rate effect here, not
    an extra translation), so a policy that has never seen it under-turns
    and carries that heading error forward rather than having it cancelled
    in the next frame (see README, "Design deep-dives")."""
    x, y, theta, v = state
    steer = max(-STEER_LIMIT, min(STEER_LIMIT, steer))
    yaw_rate = (v / WHEELBASE) * math.tan(steer)
    if tire_slip:
        understeer_scale = 1.0 / (1.0 + SLIP_UNDERSTEER_COEFF * v * v * abs(steer))
        yaw_rate *= understeer_scale
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = theta + yaw_rate * dt
    v_new = max(0.0, v + accel * dt)
    return (x_new, y_new, theta_new, v_new)


def render_frame(route, idx, ego_x, ego_y, ego_theta, omega, rolling_shutter=False):
    """Render a clean top-down corridor frame, then optionally apply the
    rolling-shutter row shear. omega is the ego's current angular velocity.
    Exposure noise is added by the caller (Episode.frame), since its scale
    depends on ego speed, which this function is not given."""
    lx, ly = R.local_frame_points(route, idx, ego_x, ego_y, ego_theta, LOOKAHEAD_M)
    order = np.argsort(lx)
    lx, ly = lx[order], ly[order]

    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    row_forward = np.linspace(LOOKAHEAD_M, 0.0, IMG_SIZE)  # row 0 = far, row -1 = near
    if len(lx) >= 2:
        centerline = np.interp(row_forward, lx, ly, left=ly[0], right=ly[-1])
    else:
        centerline = np.zeros(IMG_SIZE)

    col_lateral = np.linspace(-LATERAL_HALF_RANGE_M, LATERAL_HALF_RANGE_M, IMG_SIZE)
    for row in range(IMG_SIZE):
        band = np.abs(col_lateral - centerline[row]) < RENDER_HALF_WIDTH_M
        img[row, band] = 1.0

    if rolling_shutter:
        # Each row is treated as captured at a slightly later time than the
        # one above it; the whole frame drifts sideways by that much turn
        # rate times elapsed readout time, in columns.
        for row in range(IMG_SIZE):
            t_row = (row / IMG_SIZE) * ROLLING_SHUTTER_PERIOD
            shift_cols = int(round(omega * t_row * ROLLING_SHUTTER_GAIN))
            if shift_cols != 0:
                img[row] = np.roll(img[row], shift_cols)

    return img


class Episode:
    def __init__(self, route, rng, tire_slip=False, rolling_shutter=False, exposure_noise=False):
        self.route = route
        self.rng = rng
        self.tire_slip = tire_slip
        self.rolling_shutter = rolling_shutter
        self.exposure_noise = exposure_noise
        self.ego = (route["x"][0], route["y"][0], route["theta"][0], CRUISE_SPEED * 0.5)
        self.idx = 0
        self.progress_s = 0.0
        self.t = 0
        self.done = False
        self.off_road = False
        self.reached_goal = False
        self.last_omega = 0.0

    def frame(self):
        img = render_frame(
            self.route, self.idx, self.ego[0], self.ego[1], self.ego[2], self.last_omega,
            rolling_shutter=self.rolling_shutter,
        )
        if self.exposure_noise:
            v = self.ego[3]
            noise_std = EXPOSURE_NOISE_BASE + EXPOSURE_NOISE_SPEED_COEF * v
            img = np.clip(img + self.rng.normal(0.0, noise_std, size=img.shape).astype(np.float32), 0.0, 1.0)
        return img

    def lateral_heading_error(self):
        return R.lateral_and_heading_error(self.route, self.idx, self.ego[0], self.ego[1], self.ego[2])

    def step(self, steer):
        if self.done:
            return True
        x, y, theta, v = self.ego
        accel = ACCEL_LIMIT * np.sign(CRUISE_SPEED - v) if abs(CRUISE_SPEED - v) > 0.1 else 0.0
        new_ego = step_bicycle(self.ego, steer, accel, tire_slip=self.tire_slip)
        self.last_omega = (new_ego[2] - theta) / DT
        self.ego = new_ego
        self.idx = R.nearest_index(self.route, self.ego[0], self.ego[1], self.idx)
        self.progress_s = max(self.progress_s, self.route["s"][self.idx])
        lat_err, _ = self.lateral_heading_error()
        self.t += 1

        if abs(lat_err) > OFFROAD_THRESHOLD:
            self.off_road = True
            self.done = True
        elif self.progress_s >= self.route["total_length"] - R.DS:
            self.reached_goal = True
            self.done = True
        elif self.t >= MAX_STEPS:
            self.done = True
        return self.done

    def route_completion(self):
        return min(1.0, self.progress_s / self.route["total_length"])
