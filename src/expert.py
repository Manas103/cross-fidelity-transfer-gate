"""Privileged PD path-follower: the training label source. It reads the
route's exact lateral and heading error directly (not from the rendered
image), which is a standard privileged-expert-distillation setup; the CNN
policy that learns from it only ever sees the rendered frame."""
from . import sim as S

KP_LAT = 0.35
KP_HEAD = 0.9


def expert_steer(episode):
    lat_err, heading_err = episode.lateral_heading_error()
    steer = -KP_LAT * lat_err - KP_HEAD * heading_err
    return max(-S.STEER_LIMIT, min(S.STEER_LIMIT, steer))
