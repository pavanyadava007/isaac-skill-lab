import math

import torch

from skilllab.config import DR, MAX_DPOS, NO_DR
from skilllab.expert import APPROACH, DESCEND, ScriptedExpert
from skilllab.task import build_obs, flatten_obs, placed, sample_episode, sample_sizes, wrap_quarter, yaw_from_quat


def test_sampling_deterministic_and_in_range():
    a, b = sample_episode(123, DR), sample_episode(123, DR)
    assert a == b
    for s in range(200):
        e = sample_episode(s, DR)
        assert DR.cube_x[0] <= e["cube_x"] <= DR.cube_x[1]
        assert math.hypot(e["cube_x"] - e["target_x"], e["cube_y"] - e["target_y"]) >= DR.min_cube_target_dist
        assert DR.cube_mass[0] <= e["cube_mass"] <= DR.cube_mass[1]
    assert sample_episode(5, NO_DR)["cube_mass"] == NO_DR.cube_mass[0]
    sizes = sample_sizes(100, 0, DR)
    assert sizes.min() >= 0.04 and sizes.max() <= 0.06


def test_wrap_and_yaw():
    x = torch.tensor([0.0, math.pi / 2, math.pi / 4 + 0.1, -math.pi])
    w = wrap_quarter(x)
    assert torch.all(w >= -math.pi / 4) and torch.all(w < math.pi / 4)
    assert abs(float(w[1])) < 1e-6
    yaw = 0.3
    q = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
    assert float(yaw_from_quat(q)[0]) == pytest_approx(yaw)


def pytest_approx(v):
    import pytest

    return pytest.approx(v, abs=1e-5)


def _obs(tcp, cube, grip=0.08, target=(0.5, 0.2), size=0.05):
    t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
    return build_obs(t(tcp), torch.zeros(1), t(grip), t(cube), torch.zeros(1), t(target), t(size))


def test_obs_layout():
    o = _obs([0.45, 0, 0.25], [0.5, 0.1, 0.025])
    assert flatten_obs(o).shape == (1, 16)
    assert torch.allclose(o["cube_rel_tcp"], torch.tensor([[0.05, 0.1, -0.225]]))


def test_expert_first_steps():
    ex = ScriptedExpert(1, "cpu")
    a = ex.act(_obs([0.45, 0, 0.25], [0.5, 0.1, 0.025]))
    assert ex.phase.item() == APPROACH
    assert a.shape == (1, 5) and a[0, 4] > 0  # open
    assert torch.all(a[0, :3].abs() <= MAX_DPOS + 1e-9)
    ex.act(_obs([0.5, 0.1, 0.15], [0.5, 0.1, 0.025]))  # aligned above the cube at hover height
    assert ex.phase.item() == DESCEND


def test_success_check():
    size = torch.tensor([0.05])
    ok = placed(torch.tensor([[0.5, 0.2, 0.025]]), torch.tensor([[0.51, 0.2]]), size, torch.tensor([0.08]),
                torch.tensor([[0.5, 0.2, 0.15]]))
    held = placed(torch.tensor([[0.5, 0.2, 0.025]]), torch.tensor([[0.51, 0.2]]), size, torch.tensor([0.05]),
                  torch.tensor([[0.5, 0.2, 0.02]]))
    off = placed(torch.tensor([[0.6, 0.2, 0.025]]), torch.tensor([[0.51, 0.2]]), size, torch.tensor([0.08]),
                 torch.tensor([[0.5, 0.2, 0.15]]))
    assert bool(ok) and not bool(held) and not bool(off)
