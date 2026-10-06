import torch

from skilllab import models
from skilllab.task import OBS_DIM


def test_mlp_save_load_and_action(tmp_path):
    m = models.build({"kind": "mlp", "obs_dim": OBS_DIM, "hidden": 32, "layers": 2})
    m.obs_norm.fit(torch.randn(100, OBS_DIM))
    m.act_norm.fit(torch.randn(100, 4) * 0.01)
    models.save(tmp_path / "m.pt", m, {"run": "x"})
    m2, meta = models.load(tmp_path / "m.pt")
    x = torch.randn(3, OBS_DIM)
    assert torch.allclose(m(x), m2(x)) and meta["run"] == "x"
    a = models.PolicyRunner(m2, 3, "cpu").act(x)
    assert a.shape == (3, 5) and set(a[:, 4].tolist()) <= {-1.0, 1.0}


def test_rnn_runner_resets_hidden_every_horizon():
    m = models.build({"kind": "rnn", "obs_dim": OBS_DIM, "hidden": 16, "horizon": 3})
    r = models.PolicyRunner(m, 2, "cpu")
    x = torch.randn(2, OBS_DIM)
    first = r.act(x)
    r.act(x)
    r.act(x)
    again = r.act(x)  # t = 3: hidden state reset, same input -> same output as the first step
    assert torch.allclose(first, again)


def test_bc_loss_finite():
    m = models.build({"kind": "mlp", "obs_dim": OBS_DIM})
    act = torch.cat([torch.randn(8, 4), torch.sign(torch.randn(8, 1))], dim=-1)
    loss, parts = models.bc_loss(m, m(torch.randn(8, OBS_DIM)), act)
    assert torch.isfinite(loss) and 0 <= parts["grip_acc"] <= 1


def test_normalizer_constant_feature_not_blown_up():
    n = models.Normalizer(2)
    x = torch.stack([torch.randn(50), torch.full((50,), 0.05)], dim=-1)
    n.fit(x)
    assert float(n.std[1]) == 1.0
    assert abs(float(n(torch.tensor([[0.0, 0.04]]))[0, 1])) < 0.02
