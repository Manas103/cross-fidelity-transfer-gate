"""End-to-end experiment: train a CNN steering policy on the clean (Tier A)
simulator only, then measure its closed-loop route completion on both tiers
over 600 paired rollouts, run a sequential significance readout on the gap,
and attribute the gap to each Tier-B factor with a one-at-a-time ablation.

Writes docs/results.json and docs/run_log.txt.
"""
import json
import time
import numpy as np
import torch

from src import sim as S
from src import train as T
from src.policy import CNNPolicy
from src.dataset import collect_bc_episodes, paired_eval
from src.sequential import sequential_readout

MASTER_SEED = 7
N_BC_EPISODES = 150
N_BC_EPOCHS = 15
N_EVAL_ROLLOUTS = 600
N_ABLATION_ROLLOUTS = 300

TIER_A = dict(tire_slip=False, rolling_shutter=False, exposure_noise=False)
TIER_B_FULL = dict(tire_slip=True, rolling_shutter=True, exposure_noise=True)
ABLATION_ARMS = {
    "tire_slip_only": dict(tire_slip=True, rolling_shutter=False, exposure_noise=False),
    "rolling_shutter_only": dict(tire_slip=False, rolling_shutter=True, exposure_noise=False),
    "exposure_noise_only": dict(tire_slip=False, rolling_shutter=False, exposure_noise=True),
}

EVAL_SEEDS = list(range(500000, 500000 + N_EVAL_ROLLOUTS))
ABLATION_SEEDS = list(range(600000, 600000 + N_ABLATION_ROLLOUTS))


def pct(x):
    return round(100.0 * float(x), 1)


def main():
    t0 = time.time()
    log = []
    torch.manual_seed(MASTER_SEED)
    rng = np.random.default_rng(MASTER_SEED)

    frames, steers = collect_bc_episodes(N_BC_EPISODES, rng)
    log.append(f"BC dataset: {len(frames)} frames from {N_BC_EPISODES} Tier-A expert episodes")

    model = CNNPolicy()
    bc_log = []
    T.fit(model, frames, steers, epochs=N_BC_EPOCHS, log=bc_log)
    log.append(f"BC training: final loss {bc_log[-1]['loss']:.5f} (epoch 0 loss {bc_log[0]['loss']:.5f})")

    comp_a, comp_b, reached_a, reached_b = paired_eval(model, EVAL_SEEDS, TIER_A, TIER_B_FULL)
    log.append(f"paired eval: Tier A mean completion {pct(comp_a.mean())}%, "
               f"Tier B (full) mean completion {pct(comp_b.mean())}%")

    seq = sequential_readout(reached_a, reached_b, checkpoint_every=50)
    log.append(f"sequential readout: first significant at n={seq['first_significant_n']}, "
               f"stayed significant after: {seq['stayed_significant_after_first']}")

    ablation_results = {}
    for name, kwargs in ABLATION_ARMS.items():
        a2, b2, _, _ = paired_eval(model, ABLATION_SEEDS, TIER_A, kwargs)
        drop_points = pct(a2.mean()) - pct(b2.mean())
        ablation_results[name] = dict(
            tier_a_mean_pct=pct(a2.mean()), arm_mean_pct=pct(b2.mean()), drop_points=round(drop_points, 1))
        log.append(f"ablation {name}: Tier A {pct(a2.mean())}% vs arm {pct(b2.mean())}% "
                   f"(drop {round(drop_points, 1)} points, n={N_ABLATION_ROLLOUTS})")

    total_gap_points = pct(comp_a.mean()) - pct(comp_b.mean())
    attributed_sum = sum(v["drop_points"] for v in ablation_results.values())

    results = {
        "seed": MASTER_SEED,
        "n_bc_episodes": N_BC_EPISODES,
        "n_eval_rollouts": N_EVAL_ROLLOUTS,
        "n_ablation_rollouts_per_arm": N_ABLATION_ROLLOUTS,
        "tier_a_route_completion_pct": pct(comp_a.mean()),
        "tier_b_full_route_completion_pct": pct(comp_b.mean()),
        "total_gap_points": round(total_gap_points, 1),
        "sequential_readout": seq,
        "ablation": ablation_results,
        "attributed_points_sum": round(attributed_sum, 1),
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    with open("docs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("docs/run_log.txt", "w") as f:
        f.write("\n".join(log) + "\n")
        f.write(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "sequential_readout"}, indent=2))


if __name__ == "__main__":
    main()
