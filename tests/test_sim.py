import math
import numpy as np
from src import sim as S
from src import route as R


def test_straight_line_matches_closed_form():
    state = (0.0, 0.0, 0.0, 5.0)
    for _ in range(20):
        state = S.step_bicycle(state, 0.0, 0.0)
    t = 20 * S.DT
    assert math.isclose(state[0], 5.0 * t, rel_tol=1e-9)
    assert math.isclose(state[1], 0.0, abs_tol=1e-9)


def test_tire_slip_reduces_yaw_rate_but_not_position_update():
    """Understeer should only change theta's increment, never x,y directly
    within the same step (see sim.step_bicycle docstring)."""
    state = (1.0, 2.0, 0.3, 6.0)
    steer = 0.3
    clean = S.step_bicycle(state, steer, 0.0, tire_slip=False)
    slipped = S.step_bicycle(state, steer, 0.0, tire_slip=True)
    assert math.isclose(clean[0], slipped[0], rel_tol=1e-9)
    assert math.isclose(clean[1], slipped[1], rel_tol=1e-9)
    assert abs(slipped[2] - state[2]) < abs(clean[2] - state[2])


def test_tire_slip_is_a_no_op_at_zero_steer():
    state = (0.0, 0.0, 0.0, 6.0)
    clean = S.step_bicycle(state, 0.0, 0.0, tire_slip=False)
    slipped = S.step_bicycle(state, 0.0, 0.0, tire_slip=True)
    assert clean == slipped


def test_render_frame_is_binary_and_in_range():
    rng = np.random.default_rng(0)
    route = R.generate_route(rng)
    img = S.render_frame(route, 0, route["x"][0], route["y"][0], route["theta"][0], 0.0)
    assert img.shape == (S.IMG_SIZE, S.IMG_SIZE)
    assert set(np.unique(img)).issubset({0.0, 1.0})


def test_render_frame_centerline_is_centered_when_on_route():
    rng = np.random.default_rng(0)
    route = R.generate_route(rng, n_segments=1, seg_len_range=(50.0, 50.0), curvature_range=0.0)
    img = S.render_frame(route, 0, 0.0, 0.0, 0.0, 0.0)
    mid_col = S.IMG_SIZE // 2
    # the corridor should be lit around the image's lateral center for a
    # straight route with the ego exactly on it
    assert img[0, mid_col] == 1.0
    assert img[0, 0] == 0.0
    assert img[0, -1] == 0.0


def test_off_road_ends_episode_and_caps_completion_below_one():
    rng = np.random.default_rng(3)
    route = R.generate_route(rng)
    ep = S.Episode(route, rng, tire_slip=False, rolling_shutter=False, exposure_noise=False)
    done = False
    steps = 0
    while not done and steps < 60:
        done = ep.step(S.STEER_LIMIT)  # steer hard and constantly, should run off-road
        steps += 1
    assert ep.off_road
    assert not ep.reached_goal
    assert ep.route_completion() < 1.0


def test_reached_goal_requires_progress_past_route_length():
    rng = np.random.default_rng(4)
    route = R.generate_route(rng, n_segments=1, seg_len_range=(5.0, 5.0), curvature_range=0.0)
    ep = S.Episode(route, rng, tire_slip=False, rolling_shutter=False, exposure_noise=False)
    done = False
    steps = 0
    while not done and steps < S.MAX_STEPS:
        done = ep.step(0.0)  # drive straight down a short straight route
        steps += 1
    assert ep.reached_goal
    assert not ep.off_road
