"""Compact PPO for the portfolio-execution environment.

Written out rather than pulled from a library for three reasons: the action
space is four bounded scalars (so nothing generic is needed), the whole agent is
under 10k parameters (so training is dominated by the environment, not the
network), and an auditable file is worth more here than a dependency whose
defaults would need checking anyway.

Choices that matter for this problem rather than for benchmarks:

* **Squashed Gaussian actions.**  A ``tanh``-squashed Gaussian rescaled into
  each action's bounds means the policy can never emit an invalid book, and the
  log-probability correction for the squash is applied exactly.
* **Reward is already log wealth**, so no reward scaling or clipping is used —
  rescaling it would silently change the risk preference being optimised.
* **Observation normalisation from running statistics**, frozen at evaluation.
  Market state features live on wildly different scales (a drawdown near 0.1, an
  annualised vol near 0.2, a turnover near 1.0).
* **Early stopping on approximate KL**, the standard guard against a single
  update destroying a policy that took many episodes to find.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)

__all__ = ["PPOConfig", "PPOAgent", "RunningNorm", "train_ppo", "collect_rollout"]

# bounds for (deploy, trade_rate, band, tilt) — see rl.env.ACTION_NAMES
ACTION_LOW = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 0.5, 3.0], dtype=np.float32)


@dataclass
class PPOConfig:
    hidden: int = 64
    n_layers: int = 2
    lr: float = 3e-4
    gamma: float = 0.995          # a step is one rebalance period; discount gently
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs_per_update: int = 10
    minibatch: int = 256
    target_kl: float = 0.02
    steps_per_update: int = 2048
    total_updates: int = 200
    seed: int = 0
    device: str | None = None
    log_every: int = 10


class RunningNorm:
    """Welford running mean/variance for observation normalisation."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.frozen = False

    def update(self, x: np.ndarray) -> None:
        if self.frozen:
            return
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -10, 10).astype(np.float32)


def _build_ac(obs_dim: int, act_dim: int, cfg: PPOConfig):
    import torch
    import torch.nn as nn

    def mlp(out_dim, final_gain):
        layers, d = [], obs_dim
        for _ in range(cfg.n_layers):
            lin = nn.Linear(d, cfg.hidden)
            nn.init.orthogonal_(lin.weight, np.sqrt(2))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = cfg.hidden
        head = nn.Linear(d, out_dim)
        nn.init.orthogonal_(head.weight, final_gain)
        nn.init.zeros_(head.bias)
        return nn.Sequential(*layers, head)

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = mlp(act_dim, 0.01)     # small init -> near-neutral start
            self.v = mlp(1, 1.0)
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        def dist(self, obs):
            mu = self.pi(obs)
            std = self.log_std.clamp(-5, 1).exp()
            return torch.distributions.Normal(mu, std)

        def value(self, obs):
            return self.v(obs).squeeze(-1)

    torch.manual_seed(cfg.seed)
    return ActorCritic()


def _squash(u, low, high):
    """tanh-squash raw Gaussian samples into ``[low, high]``."""
    import torch
    return low + (torch.tanh(u) + 1.0) * 0.5 * (high - low)


def _squash_logp(dist, u, span):
    """Exact log-density after the tanh squash and affine rescale.

    ``d/du [low + (tanh(u)+1)/2 * span] = (1 - tanh(u)^2) * span/2``, so the
    change-of-variables term is the log of that Jacobian.
    """
    import torch
    logp = dist.log_prob(u).sum(-1)
    corr = (torch.log1p(-torch.tanh(u).pow(2) + 1e-6)
            + torch.log(span * 0.5)).sum(-1)
    return logp - corr


class PPOAgent:
    """Policy + value network with the PPO update."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None):
        import torch
        self.cfg = cfg or PPOConfig()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.device = torch.device(
            self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ac = _build_ac(obs_dim, act_dim, self.cfg).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.norm = RunningNorm(obs_dim)
        self.low = torch.as_tensor(ACTION_LOW[:act_dim]).to(self.device)
        self.high = torch.as_tensor(ACTION_HIGH[:act_dim]).to(self.device)
        self.n_params = sum(p.numel() for p in self.ac.parameters())

    # ── acting ──
    def act(self, obs: np.ndarray, deterministic: bool = False):
        import torch
        o = torch.as_tensor(self.norm(obs)[None, :]).to(self.device)
        with torch.no_grad():
            d = self.ac.dist(o)
            u = d.mean if deterministic else d.sample()
            a = _squash(u, self.low, self.high)
            logp = _squash_logp(d, u, self.high - self.low)
            v = self.ac.value(o)
        return (a.squeeze(0).cpu().numpy(), float(logp.item()), float(v.item()),
                u.squeeze(0).cpu().numpy())

    def policy_fn(self):
        """A deterministic ``obs -> action`` closure for evaluation."""
        def f(obs):
            return self.act(obs, deterministic=True)[0]
        return f

    def save(self, path) -> None:
        import torch
        torch.save({
            "state_dict": self.ac.state_dict(),
            "norm": {"mean": self.norm.mean, "var": self.norm.var,
                     "count": self.norm.count},
            "cfg": self.cfg.__dict__,
            "obs_dim": self.obs_dim, "act_dim": self.act_dim,
        }, path)

    @classmethod
    def load(cls, path, map_location=None):
        import torch
        blob = torch.load(path, map_location=map_location, weights_only=False)
        cfg = PPOConfig(**blob["cfg"])
        a = cls(blob["obs_dim"], blob["act_dim"], cfg)
        a.ac.load_state_dict(blob["state_dict"])
        a.norm.mean = blob["norm"]["mean"]
        a.norm.var = blob["norm"]["var"]
        a.norm.count = blob["norm"]["count"]
        a.norm.frozen = True
        return a

    # ── learning ──
    def update(self, batch: dict) -> dict:
        import torch
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"]).to(self.device)
        u = torch.as_tensor(batch["u"]).to(self.device)
        old_logp = torch.as_tensor(batch["logp"]).to(self.device)
        adv = torch.as_tensor(batch["adv"]).to(self.device)
        ret = torch.as_tensor(batch["ret"]).to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        stats = {"kl": 0.0, "clipfrac": 0.0, "pi_loss": 0.0, "v_loss": 0.0,
                 "entropy": 0.0, "epochs": 0}
        rng = np.random.default_rng(cfg.seed)

        for ep in range(cfg.epochs_per_update):
            rng.shuffle(idx)
            kls = []
            for s in range(0, n, cfg.minibatch):
                mb = idx[s: s + cfg.minibatch]
                if len(mb) < 8:
                    continue
                d = self.ac.dist(obs[mb])
                logp = _squash_logp(d, u[mb], self.high - self.low)
                ratio = (logp - old_logp[mb]).exp()
                a = adv[mb]
                pi_loss = -torch.min(
                    ratio * a,
                    ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v_loss = (self.ac.value(obs[mb]) - ret[mb]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kls.append((old_logp[mb] - logp).mean().item())
                stats["pi_loss"] = float(pi_loss.item())
                stats["v_loss"] = float(v_loss.item())
                stats["entropy"] = float(ent.item())
                stats["clipfrac"] = float(
                    ((ratio - 1).abs() > cfg.clip).float().mean().item())
            stats["epochs"] = ep + 1
            stats["kl"] = float(np.mean(kls)) if kls else 0.0
            if stats["kl"] > cfg.target_kl:
                break
        return stats


def collect_rollout(env, agent: PPOAgent, n_steps: int) -> dict:
    """Run ``env`` until ``n_steps`` transitions are gathered; return a GAE batch."""
    obs_buf, u_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    ep_returns, ep_equities = [], []

    o = env.reset()
    ep_r = 0.0
    for _ in range(n_steps):
        agent.norm.update(o[None, :])
        a, logp, v, u = agent.act(o)
        o2, r, done, info = env.step(a)
        obs_buf.append(agent.norm(o))
        u_buf.append(u)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        done_buf.append(done)
        ep_r += r
        o = o2
        if done:
            ep_returns.append(ep_r)
            ep_equities.append(info["equity"])
            ep_r = 0.0
            o = env.reset()

    _, _, last_v, _ = agent.act(o)
    rew = np.array(rew_buf, dtype=np.float32)
    val = np.array(val_buf + [last_v], dtype=np.float32)
    dn = np.array(done_buf, dtype=np.float32)

    cfg = agent.cfg
    adv = np.zeros_like(rew)
    gae = 0.0
    for t in reversed(range(len(rew))):
        nonterm = 1.0 - dn[t]
        delta = rew[t] + cfg.gamma * val[t + 1] * nonterm - val[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
        adv[t] = gae
    ret = adv + val[:-1]

    return {
        "obs": np.array(obs_buf, dtype=np.float32),
        "u": np.array(u_buf, dtype=np.float32),
        "logp": np.array(logp_buf, dtype=np.float32),
        "adv": adv.astype(np.float32),
        "ret": ret.astype(np.float32),
        "ep_returns": ep_returns,
        "ep_equities": ep_equities,
    }


@dataclass
class TrainLog:
    updates: list[dict] = field(default_factory=list)

    def best_mean_return(self) -> float:
        vals = [u["mean_ep_return"] for u in self.updates
                if np.isfinite(u["mean_ep_return"])]
        return max(vals) if vals else float("-inf")


def train_ppo(env, cfg: PPOConfig | None = None, agent: PPOAgent | None = None):
    """Train a policy on ``env``.  Returns ``(agent, log)``."""
    cfg = cfg or PPOConfig()
    agent = agent or PPOAgent(env.obs_dim, env.act_dim, cfg)
    log = TrainLog()
    logger.info(f"PPO: {agent.n_params:,} params on {agent.device}, "
                f"{cfg.total_updates} updates x {cfg.steps_per_update} steps")

    for it in range(cfg.total_updates):
        batch = collect_rollout(env, agent, cfg.steps_per_update)
        stats = agent.update(batch)
        rec = {
            "update": it,
            "mean_ep_return": (float(np.mean(batch["ep_returns"]))
                               if batch["ep_returns"] else float("nan")),
            "n_episodes": len(batch["ep_returns"]),
            **stats,
        }
        log.updates.append(rec)
        if cfg.log_every and it % cfg.log_every == 0:
            logger.info(
                f"  upd {it:>4}  ep_ret {rec['mean_ep_return']:+.4f} "
                f"({rec['n_episodes']} eps)  kl {stats['kl']:.4f} "
                f"clip {stats['clipfrac']:.2f}  ent {stats['entropy']:+.2f}")
    return agent, log
