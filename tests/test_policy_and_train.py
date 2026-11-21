import numpy as np
import torch
from src import sim as S
from src.policy import CNNPolicy, act
from src.train import fit


def test_action_bounds_respected():
    torch.manual_seed(0)
    model = CNNPolicy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        frame = (rng.random((S.IMG_SIZE, S.IMG_SIZE)) > 0.5).astype(np.float32)
        steer = act(model, frame)
        assert -S.STEER_LIMIT - 1e-6 <= steer <= S.STEER_LIMIT + 1e-6


def test_fit_reduces_loss():
    torch.manual_seed(0)
    model = CNNPolicy()
    rng = np.random.default_rng(0)
    frames = rng.random((200, 1, S.IMG_SIZE, S.IMG_SIZE)).astype(np.float32)
    steers = (frames.mean(axis=(1, 2, 3)) - 0.5).astype(np.float32) * S.STEER_LIMIT
    log = []
    fit(model, frames, steers, epochs=10, log=log)
    assert log[-1]["loss"] < log[0]["loss"]
