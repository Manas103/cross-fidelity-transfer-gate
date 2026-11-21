import numpy as np
import torch
from . import route as R
from . import sim as S
from .expert import expert_steer
from .policy import act


def collect_bc_episodes(n_episodes, rng):
    """Teacher-forced expert rollouts on Tier A (clean), the only tier the
    policy is trained on."""
    frames, steers = [], []
    for _ in range(n_episodes):
        rt = R.generate_route(rng)
        ep = S.Episode(rt, rng, tire_slip=False, rolling_shutter=False, exposure_noise=False)
        done = False
        while not done:
            frames.append(ep.frame())
            a = expert_steer(ep)
            steers.append(a)
            done = ep.step(a)
    return (np.array(frames, dtype=np.float32)[:, None, :, :],
            np.array(steers, dtype=np.float32))


def rollout_once(model, route, rng, tire_slip, rolling_shutter, exposure_noise):
    ep = S.Episode(route, rng, tire_slip=tire_slip, rolling_shutter=rolling_shutter,
                   exposure_noise=exposure_noise)
    done = False
    while not done:
        frame = ep.frame()
        steer = act(model, frame)
        done = ep.step(steer)
    return ep.route_completion(), ep.reached_goal


def paired_eval(model, seeds, tier_a_kwargs, tier_b_kwargs):
    """For each seed, generate ONE route and initial condition, then run it
    under both tier settings. Returns, for each tier, an array of route
    completion fractions and an array of reached_goal booleans, all
    index-aligned (paired across tiers by seed)."""
    completions_a, completions_b = [], []
    reached_a, reached_b = [], []
    for seed in seeds:
        rng_route = np.random.default_rng(seed)
        route = R.generate_route(rng_route)
        rng_a = np.random.default_rng(seed + 1_000_000)
        rng_b = np.random.default_rng(seed + 1_000_000)  # identical noise stream as tier A's own
        ca, ra = rollout_once(model, route, rng_a, **tier_a_kwargs)
        cb, rb = rollout_once(model, route, rng_b, **tier_b_kwargs)
        completions_a.append(ca)
        completions_b.append(cb)
        reached_a.append(ra)
        reached_b.append(rb)
    return (np.array(completions_a), np.array(completions_b),
            np.array(reached_a), np.array(reached_b))
