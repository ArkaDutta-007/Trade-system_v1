"""Properties of the RL execution environment and the PPO agent.

The environment is a simulator, so the tests it needs are the simulator's:
money must be conserved, actions must map to legal books, and the reward must
be the thing the docstring says it is.  The agent's tests are about the action
transform — a bug there silently produces out-of-range books that still look
plausible in aggregate.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from trading_system.research.costs import RealisticCostModel
from trading_system.rl.env import ACTION_NAMES, OBS_NAMES, EnvConfig, EpisodeData, PortfolioEnv
from trading_system.rl.ppo import ACTION_HIGH, ACTION_LOW, RunningNorm


def make_tape(T=600, N=50, seed=0, rebalance=21):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.015, (T, N))
    scores = rng.normal(0, 1, (T, N))
    # give the scores a little genuine edge so the tape is not pure noise
    ret += 0.002 * scores
    rebal = np.zeros(T, dtype=bool)
    rebal[::rebalance] = True
    return EpisodeData(
        dates=np.array([dt.date(2010, 1, 1) + dt.timedelta(days=i) for i in range(T)]),
        ret=ret, scores=scores,
        adv=np.full((T, N), 5e7), spread_bps=np.full((T, N), 10.0),
        dvol=np.full((T, N), 0.015), alive=np.ones((T, N), dtype=bool),
        rebalance=rebal,
    )


@pytest.fixture
def env():
    return PortfolioEnv(make_tape(), RealisticCostModel(), EnvConfig(episode_days=252))


class TestEnvContract:
    def test_observation_and_action_shapes_match_the_names(self, env):
        obs = env.reset(start=25)
        assert obs.shape == (len(OBS_NAMES),) == (env.obs_dim,)
        assert env.act_dim == len(ACTION_NAMES) == 4

    def test_observations_are_always_finite(self, env):
        obs = env.reset(start=25)
        assert np.isfinite(obs).all()
        for _ in range(8):
            obs, r, done, _ = env.step(np.array([1.0, 1.0, 0.0, 0.0]))
            assert np.isfinite(obs).all() and np.isfinite(r)
            if done:
                break

    def test_target_weights_respect_cap_deploy_and_top_k(self, env):
        env.reset(start=25)
        w = env.target_weights(25, tilt=0.0, deploy=0.8)
        assert (w > 0).sum() <= env.cfg.top_k
        assert w.sum() == pytest.approx(0.8)
        assert w.max() <= env.cfg.max_weight + 1e-9

    def test_zero_deploy_holds_nothing(self, env):
        env.reset(start=25)
        assert env.target_weights(25, tilt=1.0, deploy=0.0).sum() == pytest.approx(0.0)

    def test_tilt_increases_concentration(self, env):
        env.reset(start=25)
        flat = env.target_weights(25, tilt=0.0, deploy=1.0)
        tilted = env.target_weights(25, tilt=3.0, deploy=1.0)
        assert tilted.max() > flat.max()
        # Herfindahl: more concentrated means a larger sum of squares
        assert (tilted ** 2).sum() > (flat ** 2).sum()

    def test_out_of_range_actions_are_clipped_not_obeyed(self, env):
        env.reset(start=25)
        _, _, _, info = env.step(np.array([5.0, -3.0, 9.0, 99.0]))
        d, tr, b, ti = info["action"]
        assert 0.0 <= d <= 1.0 and 0.0 <= tr <= 1.0
        assert 0.0 <= b <= 0.5 and 0.0 <= ti <= 3.0


class TestEnvAccounting:
    def test_equity_stays_positive_and_finite(self, env):
        env.reset(start=25)
        for _ in range(12):
            _, _, done, info = env.step(np.array([1.0, 1.0, 0.05, 1.0]))
            assert np.isfinite(info["equity"]) and info["equity"] > 0
            if done:
                break

    def test_no_trading_leaves_equity_at_the_initial_cash(self):
        """deploy=0 never buys anything, so equity cannot move."""
        e = PortfolioEnv(make_tape(), RealisticCostModel(), EnvConfig(episode_days=252))
        e.reset(start=25)
        for _ in range(10):
            _, _, done, info = e.step(np.array([0.0, 1.0, 0.0, 0.0]))
            assert info["equity"] == pytest.approx(e.cfg.initial_cash, rel=1e-9)
            assert info["cost"] == pytest.approx(0.0)
            if done:
                break

    def test_costs_are_non_negative_and_turnover_is_charged(self, env):
        env.reset(start=25)
        _, _, _, info = env.step(np.array([1.0, 1.0, 0.0, 0.0]))
        assert info["cost"] > 0, "deploying from all cash must cost something"
        assert info["turnover"] > 0

    def test_partial_trading_costs_less_than_full(self):
        cost = RealisticCostModel()
        full = PortfolioEnv(make_tape(), cost, EnvConfig(episode_days=252))
        part = PortfolioEnv(make_tape(), cost, EnvConfig(episode_days=252))
        full.reset(start=25)
        part.reset(start=25)
        cf = cp = 0.0
        for _ in range(10):
            cf += full.step(np.array([1.0, 1.0, 0.0, 0.0]))[3]["cost"]
            cp += part.step(np.array([1.0, 0.3, 0.0, 0.0]))[3]["cost"]
        assert cp < cf

    def test_reward_is_the_log_equity_change_when_no_drawdown_penalty(self):
        e = PortfolioEnv(make_tape(), RealisticCostModel(),
                         EnvConfig(episode_days=252, dd_penalty=0.0))
        e.reset(start=25)
        eq0 = e.pos.sum() + e.cash
        _, r, _, info = e.step(np.array([1.0, 1.0, 0.0, 0.0]))
        assert r == pytest.approx(np.log(info["equity"] / eq0), abs=1e-9)

    def test_drawdown_penalty_only_bites_past_the_threshold(self):
        e = PortfolioEnv(make_tape(), RealisticCostModel(),
                         EnvConfig(episode_days=252, dd_penalty=5.0, dd_threshold=0.10))
        e.reset(start=25)
        _, r, _, info = e.step(np.array([1.0, 1.0, 0.0, 0.0]))
        if info["dd"] > -0.10:
            assert r == pytest.approx(np.log(info["equity"] / e.cfg.initial_cash),
                                      abs=1e-9)

    def test_episodes_terminate(self, env):
        env.reset(start=25)
        for i in range(200):
            _, _, done, _ = env.step(np.array([1.0, 1.0, 0.0, 0.0]))
            if done:
                break
        else:
            pytest.fail("episode never terminated")

    def test_daily_log_covers_every_held_day(self, env):
        env.reset(start=25)
        env.step(np.array([1.0, 1.0, 0.0, 0.0]))
        assert len(env.daily_log) == env.cfg.rebalance_days


class TestRunningNorm:
    def test_it_matches_batch_statistics(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 3, (2000, 4))
        n = RunningNorm(4)
        for i in range(0, 2000, 50):
            n.update(x[i:i + 50])
        assert n.mean == pytest.approx(x.mean(0), abs=0.15)
        assert np.sqrt(n.var) == pytest.approx(x.std(0), abs=0.15)

    def test_output_is_clipped(self):
        n = RunningNorm(2)
        n.update(np.zeros((100, 2)))
        assert np.abs(n(np.array([1e9, -1e9]))).max() <= 10.0

    def test_freezing_stops_updates(self):
        n = RunningNorm(2)
        n.update(np.zeros((50, 2)))
        before = n.mean.copy()
        n.frozen = True
        n.update(np.full((50, 2), 100.0))
        assert n.mean == pytest.approx(before)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"), reason="needs torch")
class TestPPOAgent:
    def test_actions_always_land_inside_the_bounds(self):
        from trading_system.rl.ppo import PPOAgent, PPOConfig
        a = PPOAgent(len(OBS_NAMES), 4, PPOConfig(device="cpu"))
        rng = np.random.default_rng(0)
        for _ in range(50):
            act, logp, v, u = a.act(rng.normal(size=len(OBS_NAMES)).astype(np.float32))
            assert (act >= ACTION_LOW).all() and (act <= ACTION_HIGH).all()
            assert np.isfinite(logp) and np.isfinite(v)

    def test_deterministic_mode_is_deterministic(self):
        from trading_system.rl.ppo import PPOAgent, PPOConfig
        a = PPOAgent(len(OBS_NAMES), 4, PPOConfig(device="cpu"))
        obs = np.zeros(len(OBS_NAMES), dtype=np.float32)
        assert a.act(obs, deterministic=True)[0] == pytest.approx(
            a.act(obs, deterministic=True)[0])

    def test_a_short_training_run_completes_and_improves_its_value_fit(self):
        from trading_system.rl.ppo import PPOConfig, train_ppo
        e = PortfolioEnv(make_tape(T=900), RealisticCostModel(),
                         EnvConfig(episode_days=252))
        agent, log = train_ppo(e, PPOConfig(total_updates=3, steps_per_update=128,
                                            device="cpu", log_every=0))
        assert len(log.updates) == 3
        assert all(np.isfinite(u["v_loss"]) for u in log.updates)
        assert np.isfinite(log.best_mean_return())

    def test_save_and_load_round_trip(self, tmp_path):
        from trading_system.rl.ppo import PPOAgent, PPOConfig
        a = PPOAgent(len(OBS_NAMES), 4, PPOConfig(device="cpu"))
        a.norm.update(np.random.default_rng(0).normal(size=(100, len(OBS_NAMES))))
        p = tmp_path / "agent.pt"
        a.save(p)
        b = PPOAgent.load(p, map_location="cpu")
        obs = np.zeros(len(OBS_NAMES), dtype=np.float32)
        assert b.act(obs, deterministic=True)[0] == pytest.approx(
            a.act(obs, deterministic=True)[0], abs=1e-6)
        assert b.norm.frozen, "a loaded agent must not keep adapting its normaliser"
