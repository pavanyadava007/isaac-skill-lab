"""Behaviour-cloning policies: MLP and BC-RNN (GRU), with the input/output normalization stored inside.

Action head: 4 continuous outputs (TCP dx, dy, dz, dyaw; MSE on standardized targets) + 1 gripper logit
(binary cross-entropy on open/close). The executed gripper command is +1 (open) if the logit > 0 else -1.
"""

from __future__ import annotations

import torch
from torch import nn

CONT = 4  # continuous action dims


class Normalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: torch.Tensor) -> None:
        self.mean.copy_(x.mean(0))
        std = x.std(0)
        # a feature that is constant in the training data (e.g. cube size without DR) is only centred, not
        # scaled: dividing by a tiny std would turn any test-time variation into a huge input
        self.std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std.clamp_min(1e-4)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


class MLPPolicy(nn.Module):
    kind = "mlp"

    def __init__(self, obs_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.obs_dim, self.hidden, self.layers = obs_dim, hidden, layers
        self.obs_norm = Normalizer(obs_dim)
        self.act_norm = Normalizer(CONT)
        mods, d = [], obs_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, CONT + 1))
        self.net = nn.Sequential(*mods)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (..., obs_dim) -> (..., 5) raw head
        return self.net(self.obs_norm(obs))

    def config(self) -> dict:
        return {"kind": self.kind, "obs_dim": self.obs_dim, "hidden": self.hidden, "layers": self.layers}


class RNNPolicy(nn.Module):
    """BC-RNN as in robomimic: GRU over windows of `horizon` steps, hidden state reset every `horizon` steps."""

    kind = "rnn"

    def __init__(self, obs_dim: int, hidden: int = 256, horizon: int = 10):
        super().__init__()
        self.obs_dim, self.hidden, self.horizon = obs_dim, hidden, horizon
        self.obs_norm = Normalizer(obs_dim)
        self.act_norm = Normalizer(CONT)
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, CONT + 1)

    def forward(self, obs: torch.Tensor, h: torch.Tensor | None = None):  # obs (B, T, obs_dim)
        y, h = self.gru(self.enc(self.obs_norm(obs)), h)
        return self.head(y), h

    def config(self) -> dict:
        return {"kind": self.kind, "obs_dim": self.obs_dim, "hidden": self.hidden, "horizon": self.horizon}


def build(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    return {"mlp": MLPPolicy, "rnn": RNNPolicy}[kind](**cfg)


def head_to_action(model: nn.Module, raw: torch.Tensor) -> torch.Tensor:
    cont = model.act_norm.inverse(raw[..., :CONT])
    grip = torch.where(raw[..., CONT:] > 0, 1.0, -1.0)
    return torch.cat([cont, grip], dim=-1)


def bc_loss(model: nn.Module, raw: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, dict]:
    target = model.act_norm(act[..., :CONT])
    mse = ((raw[..., :CONT] - target) ** 2).mean()
    label = (act[..., CONT] > 0).float()
    bce = nn.functional.binary_cross_entropy_with_logits(raw[..., CONT], label)
    acc = ((raw[..., CONT] > 0).float() == label).float().mean()
    return mse + bce, {"mse": float(mse), "bce": float(bce), "grip_acc": float(acc)}


class PolicyRunner:
    """Stateful wrapper used at rollout time (hidden state for the RNN, reset every horizon steps)."""

    def __init__(self, model: nn.Module, num_envs: int, device: str | torch.device):
        self.model = model.to(device).eval()
        self.n, self.device = num_envs, device
        self.t = 0
        self.h = None

    def reset(self) -> None:
        self.t = 0
        self.h = None

    @torch.inference_mode()
    def act(self, flat_obs: torch.Tensor) -> torch.Tensor:
        if self.model.kind == "mlp":
            return head_to_action(self.model, self.model(flat_obs))
        if self.t % self.model.horizon == 0:
            self.h = None
        raw, self.h = self.model(flat_obs[:, None, :], self.h)
        self.t += 1
        return head_to_action(self.model, raw[:, 0])


def save(path, model: nn.Module, meta: dict) -> None:
    torch.save({"config": model.config(), "state_dict": model.state_dict(), "meta": meta}, path)


def load(path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build(ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device), ck["meta"]
