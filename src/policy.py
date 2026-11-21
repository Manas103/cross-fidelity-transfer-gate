"""A small CNN driving policy: rendered frame in, steering angle out.
Throttle is not learned (see README, "Honest framing"); a fixed low-level
speed controller inside sim.Episode.step holds cruise speed."""
import torch
import torch.nn as nn
from . import sim as S


class CNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),   # 48 -> 24
            nn.Conv2d(8, 16, 5, stride=2, padding=2), nn.ReLU(),  # 24 -> 12
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),  # 12 -> 6
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (N, 1, H, W)
        z = self.conv(x)
        raw = self.head(z)[:, 0]
        return torch.tanh(raw) * S.STEER_LIMIT


def act(model, frame_np):
    with torch.no_grad():
        x = torch.from_numpy(frame_np).float().unsqueeze(0).unsqueeze(0)
        return float(model(x)[0])
