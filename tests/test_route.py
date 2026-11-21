import math
import numpy as np
from src import route as R


def test_zero_curvature_is_a_straight_line():
    rng = np.random.default_rng(0)
    route = R.generate_route(rng, n_segments=1, seg_len_range=(50.0, 50.0), curvature_range=0.0)
    assert math.isclose(route["y"][-1], 0.0, abs_tol=1e-6)
    assert math.isclose(route["theta"][-1], 0.0, abs_tol=1e-6)
    assert math.isclose(route["x"][-1], route["total_length"], rel_tol=1e-3)


def test_constant_curvature_matches_closed_form_circle():
    """A constant curvature k traces a circle of radius 1/k; after going a
    quarter circle the heading should have turned by pi/2."""
    rng = np.random.default_rng(0)
    k = 0.02
    quarter_len = (math.pi / 2) / k
    route = R.generate_route(rng, n_segments=1, seg_len_range=(quarter_len, quarter_len), curvature_range=k)
    # force the single segment's curvature to exactly k (generate_route draws
    # it randomly in [-k, k], so pin it directly for this closed-form check)
    route["theta"] = np.concatenate([[0.0], np.cumsum(np.full(len(route["s"]) - 1, k) * R.DS)])
    assert math.isclose(route["theta"][-1], math.pi / 2, rel_tol=0.02)


def test_route_generation_is_deterministic_given_seed():
    r1 = R.generate_route(np.random.default_rng(42))
    r2 = R.generate_route(np.random.default_rng(42))
    assert np.allclose(r1["x"], r2["x"])
    assert np.allclose(r1["y"], r2["y"])


def test_nearest_index_finds_the_closest_sample():
    rng = np.random.default_rng(1)
    route = R.generate_route(rng)
    idx = R.nearest_index(route, route["x"][20], route["y"][20], hint_idx=20)
    assert idx == 20


def test_lateral_error_is_zero_on_the_centerline():
    rng = np.random.default_rng(1)
    route = R.generate_route(rng)
    idx = 30
    lat, head = R.lateral_and_heading_error(
        route, idx, route["x"][idx], route["y"][idx], route["theta"][idx])
    assert math.isclose(lat, 0.0, abs_tol=1e-9)
    assert math.isclose(head, 0.0, abs_tol=1e-9)


def test_lateral_error_sign_matches_left_positive_convention():
    rng = np.random.default_rng(1)
    route = R.generate_route(rng)
    idx = 30
    rtheta = route["theta"][idx]
    # offset 1m to the left of the route heading
    off_x = route["x"][idx] - math.sin(rtheta) * 1.0
    off_y = route["y"][idx] + math.cos(rtheta) * 1.0
    lat, _ = R.lateral_and_heading_error(route, idx, off_x, off_y, rtheta)
    assert math.isclose(lat, 1.0, rel_tol=1e-6)
