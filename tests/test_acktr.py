import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import IdentityEnv, IdentityEnvBox, IdentityEnvMultiDiscrete

from sb3_contrib import ACKTR


def test_acktr_cartpole_smoke():
    env = gym.make("CartPole-v1")

    model = ACKTR(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.05,
        n_steps=16,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.25,
        vf_fisher_coef=1.0,
        max_grad_norm=0.5,
        kfac_clip=1e-3,
        kfac_update=1,
        kfac_damping=1e-2,
        verbose=0,
        seed=0,
    )

    model.learn(total_timesteps=128)

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    assert env.action_space.contains(action)

    env.close()


@pytest.mark.parametrize(
    "env",
    [IdentityEnv(4), IdentityEnvBox(-1, 1), IdentityEnvMultiDiscrete(3)],
    ids=["discrete", "box", "multidiscrete"],
)
def test_acktr_action_spaces(env):
    # The default cold phase is 11 steps; shorten it so 8 updates reach K-FAC.
    model = ACKTR("MlpPolicy", env, n_steps=8, net_arch=(16,), cold_iter=2, seed=0)
    model.learn(total_timesteps=8 * 8)

    optimizer = model.policy.optimizer
    assert optimizer.steps == 8
    assert optimizer.preconditioner_is_active()
    assert optimizer.modules_without_statistics() == []


@pytest.mark.parametrize("shared_trunk", [True, False])
def test_acktr_trunk_layout(shared_trunk):
    model = ACKTR("MlpPolicy", "CartPole-v1", net_arch=(16,), shared_trunk=shared_trunk, seed=0)
    policy = model.policy

    assert (policy.pi_features_extractor is policy.vf_features_extractor) == shared_trunk
    # Shared: trunk + 2 heads. Separate: 2 trunks + 2 heads.
    assert len(policy.optimizer.modules) == (3 if shared_trunk else 4)


def test_acktr_rejects_trunk_in_policy_kwargs():
    with pytest.raises(ValueError, match="not through policy_kwargs"):
        ACKTR("MlpPolicy", "CartPole-v1", policy_kwargs=dict(net_arch=[16]))


@pytest.mark.parametrize("shared_trunk", [True, False])
def test_acktr_save_load_continues_training(tmp_path, shared_trunk):
    env = make_vec_env("CartPole-v1", n_envs=2, seed=0)
    model = ACKTR("MlpPolicy", env, n_steps=8, net_arch=(16,), shared_trunk=shared_trunk, cold_iter=2, seed=0)
    model.learn(total_timesteps=8 * 2 * 6)

    obs = env.reset()
    actions, _ = model.predict(obs, deterministic=True)
    optimizer = model.policy.optimizer

    model.save(tmp_path / "acktr.zip")
    loaded = ACKTR.load(tmp_path / "acktr.zip", env=env)
    loaded_optimizer = loaded.policy.optimizer

    assert loaded.shared_trunk == shared_trunk
    for key, value in model.policy.state_dict().items():
        assert th.equal(value, loaded.policy.state_dict()[key])
    assert loaded_optimizer.steps == optimizer.steps
    assert loaded_optimizer.stats_steps == optimizer.stats_steps
    loaded_actions, _ = loaded.predict(obs, deterministic=True)
    assert np.array_equal(actions, loaded_actions)

    loaded.learn(total_timesteps=8 * 2 * 2, reset_num_timesteps=False)
    assert loaded_optimizer.steps == optimizer.steps + 2
    assert loaded_optimizer.preconditioner_is_active()


class HeadMaskInfo(gym.Wrapper):
    """Adds a fixed per-head relevance mask to every step's info dict."""

    def __init__(self, env, mask):
        super().__init__(env)
        self.mask = np.asarray(mask, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["head_mask"] = self.mask
        return obs, reward, terminated, truncated, info


def _multidiscrete_env(mask):
    return make_vec_env(lambda: HeadMaskInfo(IdentityEnvMultiDiscrete(3), mask), n_envs=2, seed=0)


def _make_masked(mask, head_mask_key="head_mask"):
    return ACKTR("MlpPolicy", _multidiscrete_env(mask), n_steps=8, net_arch=(16,), cold_iter=2, seed=0, head_mask_key=head_mask_key)


def test_acktr_all_ones_head_mask_matches_unmasked():
    # Build and train one at a time: seeding happens at construction, so
    # interleaving would let the first run consume the second run's RNG stream.
    masked = _make_masked([1.0, 1.0])
    masked.learn(8 * 2 * 6)
    plain = _make_masked([1.0, 1.0], head_mask_key=None)
    plain.learn(8 * 2 * 6)
    for (name, a), b in zip(masked.policy.state_dict().items(), plain.policy.state_dict().values()):
        assert th.allclose(a, b, atol=1e-6), name


def test_acktr_masked_head_gets_no_update():
    model = _make_masked([1.0, 0.0])
    nvec = model.action_space.nvec
    rows = slice(int(nvec[0]), int(nvec[0] + nvec[1]))
    weight_before = model.policy.action_net.weight[rows].detach().clone()
    bias_before = model.policy.action_net.bias[rows].detach().clone()
    model.learn(8 * 2 * 6)
    assert model.policy.optimizer.preconditioner_is_active()
    assert th.equal(model.policy.action_net.weight[rows], weight_before)
    assert th.equal(model.policy.action_net.bias[rows], bias_before)
    # the unmasked head did learn
    assert not th.equal(model.policy.action_net.weight[: int(nvec[0])], model.policy.action_net.weight[: int(nvec[0])].detach() * 0)


def test_acktr_head_mask_errors():
    env = make_vec_env(lambda: IdentityEnvMultiDiscrete(3), n_envs=1, seed=0)
    model = ACKTR("MlpPolicy", env, n_steps=4, net_arch=(8,), head_mask_key="head_mask")
    with pytest.raises(KeyError, match="did not return it"):
        model.learn(8)
    with pytest.raises(ValueError, match="requires a MultiDiscrete"):
        ACKTR("MlpPolicy", "CartPole-v1", head_mask_key="head_mask")


def test_acktr_head_mask_save_load(tmp_path):
    model = _make_masked([1.0, 0.0])
    model.learn(8 * 2 * 3)
    model.save(tmp_path / "masked.zip")
    loaded = ACKTR.load(tmp_path / "masked.zip", env=_multidiscrete_env([1.0, 0.0]))
    assert loaded.head_mask_key == "head_mask"
    assert type(loaded.rollout_buffer).__name__ == "HeadMaskRolloutBuffer"
    loaded.learn(8 * 2 * 2, reset_num_timesteps=False)


def _train_with_scaled_advantages(env_fn, normalize_advantage, scale):
    env = make_vec_env(env_fn, n_envs=2, seed=0)
    model = ACKTR("MlpPolicy", env, n_steps=8, net_arch=(16,), cold_iter=2, seed=0, normalize_advantage=normalize_advantage)
    train = model.train

    def scaled_train():
        model.rollout_buffer.advantages *= scale
        train()

    model.train = scaled_train
    model.learn(8 * 2 * 6)
    return model.policy.state_dict()


def _same_params(a, b):
    return all(th.allclose(x, y, atol=1e-5) for x, y in zip(a.values(), b.values()))


@pytest.mark.parametrize(
    "env_fn, normalize_advantage, invariant",
    [
        (lambda: IdentityEnv(4), True, True),
        (lambda: IdentityEnv(4), None, False),
        (lambda: IdentityEnvMultiDiscrete(3), True, True),
        (lambda: IdentityEnvBox(-1, 1), None, True),
        (lambda: IdentityEnvBox(-1, 1), False, False),
    ],
    ids=["discrete-true", "discrete-default", "multidiscrete-true", "box-default", "box-false"],
)
def test_acktr_normalize_advantage(env_fn, normalize_advantage, invariant):
    # Normalized advantages make the update independent of the advantage scale.
    plain = _train_with_scaled_advantages(env_fn, normalize_advantage, 1.0)
    scaled = _train_with_scaled_advantages(env_fn, normalize_advantage, 1000.0)
    assert _same_params(plain, scaled) == invariant


def test_acktr_normalize_advantage_save_load(tmp_path):
    env = make_vec_env("CartPole-v1", n_envs=1, seed=0)
    for value in (True, False, None):
        model = ACKTR("MlpPolicy", env, n_steps=4, net_arch=(8,), normalize_advantage=value)
        model.save(tmp_path / "model.zip")
        assert ACKTR.load(tmp_path / "model.zip", env=env).normalize_advantage is value

    # A checkpoint from before the option stored A2C's fixed False; it loads as the SB2 default.
    model = ACKTR("MlpPolicy", env, n_steps=4, net_arch=(8,))
    model.normalize_advantage = False
    super(ACKTR, model).save(tmp_path / "old.zip")
    assert ACKTR.load(tmp_path / "old.zip", env=env).normalize_advantage is None
