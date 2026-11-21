import torch
import torch.nn as nn


def fit(model, frames, steers, epochs=15, lr=1e-3, batch_size=256, log=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    frames_t = torch.from_numpy(frames).float()
    steers_t = torch.from_numpy(steers).float()
    n = frames_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(frames_t[idx])
            loss = nn.functional.mse_loss(pred, steers_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        if log is not None:
            log.append(dict(epoch=epoch, loss=total / n))
    return model
