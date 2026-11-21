# Cross-Fidelity Transfer Gate for a Camera-to-Trajectory Driving Policy

A two-tier, self-built kinematic driving simulator and a small CNN steering
policy: the policy is trained on a clean, fast tier and evaluated
closed-loop, paired by initial condition, against a higher-fidelity tier
that adds tire slip, rolling-shutter shear, and exposure noise. A sequential
significance readout and a one-at-a-time ablation follow. Every number below
came from running `run_experiment.py` on this machine; nothing was targeted
in advance, including the two bugs described in "What broke".

## Why this exists

TRI's own description of the sim-to-real gap names two separate causes:
differences in sensor input and differences in vehicle dynamics. This
project is a small, fully self-contained instrument for telling those two
causes apart: because both simulator tiers are built in this repo, the
sensor effects and the dynamics effect can be turned on independently,
which a single third-party simulator would not allow.

## Honest framing

- **This is not CARLA and the policy is not vision-pretrained.** The
  "camera" is a rasterized top-down corridor view of the route ahead,
  rendered directly from the route geometry; there is no 3D scene, no
  textures, and no photorealism. The two Tier-B sensor effects (rolling
  shutter, exposure noise) are applied to that rasterized image, not to a
  rendered photorealistic frame.
- **The training label source is a privileged controller, not the CNN's
  own image.** `expert.py`'s PD controller reads the exact route lateral
  and heading error directly; the CNN policy that is trained to imitate it
  only ever sees the rendered frame. This is a standard privileged-expert
  distillation setup, disclosed rather than implied.
- **The policy outputs steering only.** A fixed low-level controller inside
  `sim.Episode.step` holds cruise speed; there is no throttle head. This
  mirrors the original end-to-end lane-following setups (steering-only
  CNNs from a forward camera) and keeps the measurement about path
  tracking under fidelity shift, not about speed control.
- **"Tire slip" is modeled as understeer (reduced yaw response to a given
  steer command), not as a separate lateral force term.** See
  `sim.step_bicycle`'s docstring for why this specific mechanism, rather
  than an additive position perturbation, was the one that actually
  degraded a reactive policy (also in "What broke" below).
- **Route completion is a continuous fraction** (`Episode.route_completion`),
  distinct from `Episode.reached_goal`, a strict boolean for literally
  finishing the route. The headline 99.9% / 44.2% numbers below are the
  continuous fraction, averaged; the sequential readout uses the strict
  boolean, because the first version of this repo used a 0.999 threshold
  on the continuous fraction as a proxy for "reached the goal" and that
  threshold was wrong (see "What broke").
- Machine: Windows 11, Python 3.12.10, PyTorch 2.12.0+cpu, 16 logical cores,
  32 GB RAM, CPU only (the policy is a 3-layer CNN on a 48x48 image; the
  full 600-rollout paired evaluation plus the three-arm ablation finishes
  in under 3 minutes on CPU).

## Architecture

```
src/
  route.py        procedural piecewise-curvature route generator, plus the
                    nearest-point projection used for lateral/heading error
                    and for the rasterizer's local-frame transform
  sim.py           the two-tier bicycle model, the rasterizer, and the
                    Episode class (dynamics + rendering + termination)
  expert.py        privileged PD path-follower, the BC label source
  policy.py        the CNN steering policy (48x48 frame -> steer)
  dataset.py       expert rollout collection and paired closed-loop evaluation
  sequential.py    the Barnard's-exact-test sequential readout
  train.py         supervised fit for the CNN
run_experiment.py   the end-to-end experiment; writes docs/results.json,
                    docs/run_log.txt
tests/              pytest suite
```

### Design deep-dives

**Both tiers share one route generator and one random seed per rollout.**
`dataset.paired_eval` builds exactly one route per seed and drives it under
both tier settings with independently-seeded noise streams. The tradeoff:
pairing removes route-to-route difficulty as a confound in the headline gap
number, at the cost of not being able to use an unpaired two-sample test
with full statistical efficiency; the sequential readout below uses
Barnard's exact test on the two tiers' marginal success counts anyway (to
match TRI's own published methodology, see Sources), which leaves the
pairing's extra power unused. That tradeoff is disclosed in Limitations,
not hidden.

**The rasterizer renders geometry, not pixels in the photorealistic sense.**
For each of 48 rows (a forward distance from 24 m ahead to the ego's own
position), the route's local lateral offset is found by interpolating the
route polyline, transformed into the ego's frame, and a corridor band is
lit around it. Rolling shutter is then applied as a per-row horizontal
shift proportional to the ego's current angular velocity times that row's
simulated readout delay; exposure noise is Gaussian pixel noise whose scale
grows with ego speed. Both are applied after the clean geometric render, not
baked into a renderer that does not exist in this repo.

**Tire slip is understeer, because an additive position perturbation did
not transfer.** The first implementation added a lateral position offset
proportional to steer times speed squared. Measured effect on a policy
never trained with it: none, at any tested magnitude up to a 0.72 m jump
per 0.1 s step. The reason: that term is corrected by the very next frame,
because the policy's steering command is a direct, fast reaction to the
*current* rendered lateral error, and any smooth bias in that error is
information the policy already knows how to cancel. Understeer instead
changes the *mapping* from a steering command to how much the car actually
turns; the policy's learned gain was calibrated on the clean mapping, so on
a sharp curve it turns less than it needs to, falls further behind, and the
image shows a larger error that the (now systematically insufficient) gain
still cannot close before the curve ends. That is a real, lasting
degradation, not a cancellable bias, and it is the mechanism this repo
actually measures.

## Validation

- **Dynamics invariants** (`tests/test_sim.py`): zero-steer straight-line
  motion matches the closed-form constant-velocity line; tire slip changes
  only the heading increment, never position, within a single step, and is
  a no-op at zero steer; off-road detection reliably triggers under
  sustained hard steering and caps completion below 1.0; a short, straight
  route is reliably completed and flagged `reached_goal`.
- **Route generator invariants** (`tests/test_route.py`): zero curvature is
  an exact straight line; a constant curvature matches the closed-form
  circular-arc heading change; generation is deterministic given a seed;
  the nearest-point search and the lateral-error sign convention are
  checked directly against hand-placed points.
- **Rendering** (`tests/test_sim.py`): frames are binary images in range;
  a straight route with the ego exactly centered lights the image's
  lateral center and nothing at the extreme left/right columns.
- **Sequential readout** (`tests/test_sequential.py`): identical arms never
  reach significance; a maximally strong gap (all-success vs all-failure)
  is detected at the first checkpoint and stays significant; the checkpoint
  schedule matches the requested interval exactly.
- **Policy and training** (`tests/test_policy_and_train.py`): steering
  output stays within the stated limit for random inputs; `fit` measurably
  reduces loss on a synthetic dataset.

Raw pytest output: `docs/test_output.txt` (18 passed).

## Findings: what broke

**First: the dynamics perturbation that had no effect.** Described above
under "tire slip is understeer, because an additive position perturbation
did not transfer." The wrong hypothesis along the way was that the
perturbation just needed to be bigger; it was raised by more than 20x
(0.018 to 0.4 in the additive model's units) with no measurable change in
route completion, which is what forced the actual diagnosis: the mechanism,
not the magnitude, was wrong, because a purely reactive per-frame
controller cancels any smooth additive bias regardless of its size.

**Second: a 0.999 completion-fraction threshold silently undercounted true
completions.** The first version of the sequential readout called an
episode "successful" when `route_completion() >= 0.999`. On Tier A, where
the scripted expert and the trained policy both reliably finish routes,
this produced success counts around 30 to 90 percent at early checkpoints
instead of the near-100 percent the clean tier actually achieves. Root
cause: `route_completion()` is `progress_s / total_length`, and the episode
is allowed to end up to one route-sampling step (`route.DS`, 0.25 m) short
of the nominal length; for a typical ~70 m route that caps the achievable
fraction at roughly 0.9964, below the 0.999 threshold, so a large fraction
of genuinely finished episodes were being counted as failures. The fix was
to add an explicit `Episode.reached_goal` boolean, set exactly where the
goal-distance check fires, and use that boolean directly in the sequential
readout instead of thresholding the continuous fraction. After the fix,
Tier A's strict success count is 600/600 over the full evaluation.

## Measured results

Machine: Windows 11, Python 3.12.10, PyTorch 2.12.0+cpu, 16 logical cores.
Single run, seed 7, reproducible via `python run_experiment.py` (wall clock
175 s). Raw output: `docs/results.json`, `docs/run_log.txt`.

| Tier | Mean route completion (600 paired rollouts) | Strict goal reached |
|---|---|---|
| A (clean, fast) | 99.9% | 600 / 600 |
| B (full: slip + rolling shutter + exposure noise) | 44.2% | 21 / 600 |

**The number that matters: route completion fell 99.9% to 44.2%, a 55.7-point
gap, on a policy that never saw any of the three Tier-B effects during
training.** The sequential readout (Barnard's exact test on the strict
goal-reached counts, checked every 50 rollouts) is significant (p < 1e-28)
at the very first checkpoint and stays significant through all 600; this
particular gap is large enough that there was no close call to resolve, in
contrast to a more realistic, smaller perturbation that might need the full
600 to separate from noise.

One-at-a-time ablation (300 paired rollouts per arm, Tier A vs. that one
factor added):

| Factor alone | Tier A | With factor | Drop |
|---|---|---|---|
| Tire slip (understeer) | 99.8% | 45.7% | 54.1 points |
| Rolling shutter | 99.8% | 44.7% | 55.1 points |
| Exposure noise | 99.8% | 90.9% | 8.9 points |

The three arms sum to 118.1 points, roughly twice the 55.7-point combined
gap, which is the honest result, not an error: inspecting individual routes
shows that tire slip and rolling shutter each independently drive a wide,
overlapping range of per-route completion values (as low as 0.19, as high
as 1.0 depending on the route's specific curvature profile), not a single
shared failure point. A route whose curves are already hard enough that one
factor fails it rarely needs a second factor to fail it again; the
combined-tier number is close to whichever single factor already does most
of the damage, not the sum of both. At these perturbation magnitudes,
exposure noise is a secondary contributor and the gap is dominated jointly
by slip and shutter, not split cleanly between a "sensor" and a "dynamics"
component the way a smaller, more realistic perturbation might show.

## Building and running

```
pip install -r requirements.txt
python run_experiment.py        # writes docs/results.json, docs/run_log.txt (~3 min)
pytest tests -q                 # writes nothing; see docs/test_output.txt for a saved run
```

No GPU, no external data, and no network access are required.

## Sibling comparison

[`post-training-driving-policy-no-forgetting`](https://github.com/Manas103/post-training-driving-policy-no-forgetting)
shares this repo's domain (a small self-built kinematic driving simulator
and a shared policy architecture) but measures a different failure mode:
that repo's base-to-post-trained left-turn success moves from 47.0% to
88.5% (a gain from more training on the target task), while this repo's
99.9% to 44.2% is a loss from a fixed, never-retrained policy meeting a
harder test distribution. Both repos converge on the same practical point
from opposite directions: a policy's measured competence is only as good as
the distribution it was actually trained or evaluated against, and moving
that distribution, in either direction, moves the number by tens of points.

## Limitations

- Single seed. The headline numbers are one run, not an average over
  repeated seeds.
- The sequential test does not use the pairing (see "Design deep-dives");
  a paired test (McNemar-style) would have more power on the same data.
- The perturbation magnitudes here were chosen to produce a clear,
  unambiguous gap for the ablation to attribute; they were not tuned to
  match any particular target number, but they are also large enough that
  the resulting ~56-point gap is more dramatic than a calibrated,
  literature-matched sim-to-real gap would typically be, and the ablation's
  non-additive, saturating behavior (see Measured Results) is partly a
  consequence of that choice.
- No other agents (vehicles, pedestrians) exist in this simulator; the
  route is otherwise empty, so nothing here measures interaction with
  traffic, only path tracking under fidelity shift.
