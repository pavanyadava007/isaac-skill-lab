import numpy as np

from skilllab.dataset import demo_names, env_args, flat_obs, load_demos, write_demos
from skilllab.task import OBS_DIM, OBS_KEYS


def _demo(t, seed):
    rng = np.random.default_rng(seed)
    return {"obs": {k: rng.normal(size=(t, w)).astype(np.float32) for k, w in OBS_KEYS},
            "actions": rng.normal(size=(t, 5)).astype(np.float32), "attrs": {"seed": seed, "cube_mass": 0.1}}


def test_roundtrip(tmp_path):
    demos = [_demo(7 + i, i) for i in range(20)]
    p = tmp_path / "d.hdf5"
    write_demos(p, demos, {"env_name": "x"}, valid_frac=0.1)
    names = demo_names(p)
    assert len(names) == 20 and names[2] == "demo_2"
    train, valid = demo_names(p, "train"), demo_names(p, "valid")
    assert len(valid) == 2 and not set(train) & set(valid) and len(train) + len(valid) == 20
    back = load_demos(p, ["demo_3"])[0]
    np.testing.assert_allclose(back["actions"], demos[3]["actions"])
    assert back["attrs"]["seed"] == 3
    assert flat_obs(back).shape == (10, OBS_DIM)
    assert env_args(p)["env_name"] == "x"


def test_robomimic_layout(tmp_path):
    import h5py

    p = tmp_path / "d.hdf5"
    write_demos(p, [_demo(5, 0)], {"env_name": "x"}, valid_frac=0.0)
    with h5py.File(p) as f:
        g = f["data/demo_0"]
        assert g.attrs["num_samples"] == 5
        assert set(g.keys()) == {"obs", "actions", "rewards", "dones"}
        assert g["dones"][-1] == 1 and g["rewards"][-1] == 1.0
        assert f["data"].attrs["total"] == 5
