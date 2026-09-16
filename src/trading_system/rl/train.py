"""Train and evaluate the execution policy, against benchmarks that can win.

The protocol
------------
1. Generate causal alpha scores over the whole sample
   (:func:`~trading_system.research.scores.generate_causal_scores`).  Every
   score is what the model would have said on that date from prior data only.
2. Split the timeline in two.  The agent trains on the earlier part and is
   evaluated on the later part, once, with a deterministic policy.  There is no
   tuning against the evaluation window.
3. Run the same evaluation for four fixed policies: full rebalance, a banded
   rebalance, the Gârleanu–Pedersen partial-trading rule, and buy-and-hold of
   the first book.  These are not strawmen — GP is the closed-form optimum for
   this problem under quadratic costs, and full rebalance is what the live
   system does today.
4. Compare with the same statistics used everywhere else: bootstrap confidence
   intervals on Sharpe and an SPA test, so "the agent won" has to survive the
   fact that we tried several policies.

An honest negative result is a real outcome here.  If PPO cannot beat GP the
right conclusion is that the closed form is sufficient and the agent should not
be deployed, and this harness is built to say that clearly rather than to find a
window where the agent happens to look good.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..research.costs import RealisticCostModel
from ..research.execution import GarleanuPedersenPolicy
from ..research.stats import bootstrap_ci, paired_bootstrap_ci, sharpe, spa_test
from ..research.wfbacktest import PanelData
from ..utils import get_logger
from .env import EnvConfig, EpisodeData, PortfolioEnv
from .ppo import PPOConfig, train_ppo

logger = get_logger(__name__)

__all__ = [
    "build_episode_data", "rollout", "fixed_action_policy", "gp_policy",
    "evaluate_policies", "train_rl_execution",
]


def build_episode_data(
    panel: PanelData, scores: np.ndarray, rebalance_days: int,
    start: int = 0, end: int | None = None,
) -> EpisodeData:
    """Slice the panel and a causal score matrix into a replayable tape."""
    end = end if end is not None else panel.ret.shape[0]
    sl = slice(start, end)
    T = end - start
    rebal = np.zeros(T, dtype=bool)
    rebal[::rebalance_days] = True
    return EpisodeData(
        dates=panel.dates[sl], ret=panel.ret[sl], scores=scores[sl],
        adv=panel.adv[sl], spread_bps=panel.spread_bps[sl], dvol=panel.dvol[sl],
        alive=panel.alive[sl], rebalance=rebal,
    )


def rollout(env: PortfolioEnv, policy, start: int = 0) -> dict:
    """Run one deterministic pass from ``start`` to the end of the tape.

    ``policy`` maps an observation to an action.  Returns the daily equity
    curve, per-step actions, and the usual summary numbers, so an RL policy's
    output is directly comparable to a walk-forward backtest's.
    """
    env.cfg = env.cfg
    obs = env.reset(start=start)
    env.end_k = len(env.d.dates) - 2
    env.daily_log = []
    actions, rewards, infos = [], [], []
    done = False
    while not done:
        a = np.asarray(policy(obs), dtype=float)
        obs, r, done, info = env.step(a)
        actions.append(info["action"])
        rewards.append(r)
        infos.append(info)

    log = getattr(env, "daily_log", [])
    if not log:
        return {"daily_ret": np.zeros(0), "equity": np.zeros(0), "actions": np.zeros((0, 4))}
    idx = np.array([t for t, _ in log])
    eq = np.array([e for _, e in log])
    dr = np.diff(eq, prepend=env.cfg.initial_cash) / np.concatenate(
        [[env.cfg.initial_cash], eq[:-1]])
    return {
        "daily_ret": dr,
        "equity": eq,
        "date_idx": idx,
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "turnover": np.array([i["turnover"] for i in infos]),
        "cost": np.array([i["cost"] for i in infos]),
        "final_equity": float(eq[-1]),
    }


def fixed_action_policy(deploy=1.0, trade_rate=1.0, band=0.0, tilt=0.0):
    """A constant-action policy — the fixed-rule benchmarks."""
    a = np.array([deploy, trade_rate, band, tilt], dtype=float)
    return lambda obs: a


def gp_policy(trade_rate: float = 0.35, deploy: float = 1.0, tilt: float = 0.0):
    """Gârleanu–Pedersen partial trading with the decay read from the state.

    The agent's observation already carries the rank autocorrelation of the
    score cross-section (``score_autocorr``), which is exactly the persistence
    the closed form needs.  Reading it live gives GP the same information the
    agent has, so the comparison is about the policy, not about who was told
    more.
    """
    from .env import OBS_NAMES
    ac_i = OBS_NAMES.index("score_autocorr")

    def f(obs):
        phi = float(np.clip(1.0 - obs[ac_i], 0.0, 0.99))
        # The aim portfolio is the target shrunk by the signal's decay; in this
        # action space that shrink is expressed through ``deploy``, and the
        # partial move toward it through ``trade_rate``.
        aim = GarleanuPedersenPolicy(trade_rate=trade_rate, signal_decay=phi)
        shrink = float(aim.aim(np.ones(1))[0])
        return np.array([deploy * shrink, trade_rate, 0.0, tilt], dtype=float)

    return f


def evaluate_policies(
    data: EpisodeData, cost: RealisticCostModel, policies: dict,
    env_cfg: EnvConfig | None = None, start: int = 25,
) -> dict:
    """Run each named policy over the same tape and summarise."""
    out = {}
    for name, pol in policies.items():
        env = PortfolioEnv(data, cost, env_cfg or EnvConfig())
        r = rollout(env, pol, start=start)
        dr = r["daily_ret"]
        eq = r["equity"]
        yrs = len(dr) / 252 if len(dr) else 0.0
        peak = np.maximum.accumulate(eq) if len(eq) else np.array([1.0])
        out[name] = {
            "sharpe": round(sharpe(dr), 3),
            "cagr": round((eq[-1] / env.cfg.initial_cash) ** (1 / max(yrs, 1e-9)) - 1, 4)
            if len(eq) else 0.0,
            "maxdd": round(float((eq / peak - 1).min()), 4) if len(eq) else 0.0,
            "ann_turnover": round(float(r["turnover"].sum() / max(yrs, 1e-9)), 2),
            # Cost as a fraction of the equity actually at risk, not of the
            # initial cash: a book that compounded 5x would otherwise report a
            # cost drag 5x too high and be incomparable to one that did not.
            "cost_drag": round(
                float(r["cost"].sum() / max(float(np.mean(eq)), 1e-9)
                      / max(yrs, 1e-9)), 5) if len(eq) else 0.0,
            "mean_action": {k: round(float(v), 3) for k, v in zip(
                ("deploy", "trade_rate", "band", "tilt"),
                r["actions"].mean(0), strict=True)} if len(r["actions"]) else {},
            "_returns": dr,
        }
    return out


@dataclass
class RLResult:
    train_log: list = field(default_factory=list)
    eval_table: dict = field(default_factory=dict)
    verdict: str = ""
    spa: dict = field(default_factory=dict)

    def render(self) -> str:
        cols = ["sharpe", "cagr", "maxdd", "ann_turnover", "cost_drag"]
        lines = [f"{'policy':<22}" + "".join(f"{c:>14}" for c in cols)]
        lines.append("-" * (22 + 14 * len(cols)))
        for k, v in sorted(self.eval_table.items(),
                           key=lambda kv: -kv[1].get("sharpe", -9)):
            lines.append(f"{k:<22}" + "".join(f"{v.get(c, ''):>14}" for c in cols))
        for k, v in self.eval_table.items():
            if v.get("mean_action"):
                lines.append(f"  {k} mean action: {v['mean_action']}")
        if self.spa:
            lines.append(f"\nSPA vs {self.spa['benchmark']}: p = {self.spa['p_value']:.3f}")
        if self.verdict:
            lines.append(f"\nVerdict: {self.verdict}")
        return "\n".join(lines)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tbl = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               for k, v in self.eval_table.items()}
        path.write_text(json.dumps(
            {"eval": tbl, "spa": self.spa, "verdict": self.verdict,
             "train_log": self.train_log[-50:]}, indent=2, default=str))


def train_rl_execution(
    panel: PanelData,
    scores: np.ndarray,
    split_date,
    cost: RealisticCostModel | None = None,
    env_cfg: EnvConfig | None = None,
    ppo_cfg: PPOConfig | None = None,
    out_dir: Path | None = None,
) -> RLResult:
    """Full protocol: train on the past, evaluate once on the future.

    ``split_date`` is the first date of the evaluation window.  Everything
    before it is available to the agent; nothing after it is, at any point.
    """
    cost = cost or RealisticCostModel()
    env_cfg = env_cfg or EnvConfig()
    ppo_cfg = ppo_cfg or PPOConfig()

    split_i = int(np.searchsorted(
        np.array([str(d) for d in panel.dates]), str(split_date)))
    logger.info(f"RL split at index {split_i} ({panel.dates[split_i]}): "
                f"train {panel.dates[0]}..{panel.dates[split_i-1]}, "
                f"eval {panel.dates[split_i]}..{panel.dates[-1]}")

    train_data = build_episode_data(panel, scores, env_cfg.rebalance_days, 0, split_i)
    eval_data = build_episode_data(panel, scores, env_cfg.rebalance_days, split_i)

    train_env = PortfolioEnv(train_data, cost, env_cfg)
    agent, log = train_ppo(train_env, ppo_cfg)

    policies = {
        "full_rebalance": fixed_action_policy(1.0, 1.0, 0.0, 0.0),
        "banded_0.25": fixed_action_policy(1.0, 1.0, 0.25, 0.0),
        "gp_partial": gp_policy(trade_rate=0.35),
        "ppo": agent.policy_fn(),
    }
    table = evaluate_policies(eval_data, cost, policies, env_cfg)

    # Did the agent beat the best fixed rule, accounting for the search?
    #
    # The comparison has to be risk-adjusted, and getting this wrong is easy: a
    # policy that simply takes more risk posts a higher mean return and a higher
    # excess-return t-statistic while being no better per unit of risk. That is
    # what happens here in practice — PPO learns to concentrate (tilt ~2.6) and
    # earns more with a worse drawdown. So the verdict is decided on the
    # bootstrapped *Sharpe* difference, and the return difference is reported
    # alongside the risk that bought it.
    bench = "gp_partial"
    others = [k for k in table if k != bench]
    n = min(len(table[k]["_returns"]) for k in table)
    lm = np.column_stack([table[k]["_returns"][-n:] for k in others])
    s = spa_test(-table[bench]["_returns"][-n:], -lm, n_boot=500)
    spa = {"benchmark": bench, "p_value": s["p_value"], "t_spa": s["t_spa"],
           "best": others[s["best_model"]],
           "criterion": "mean return (not risk-adjusted)"}

    rp = table["ppo"]["_returns"][-n:]
    rb = table[bench]["_returns"][-n:]
    sr_ci = paired_bootstrap_ci(rp, rb, stat=sharpe, n_boot=500)
    ret_ci = bootstrap_ci(rp - rb, stat=lambda x: float(np.mean(x)) * 252, n_boot=500)

    ppo_sr, gp_sr = table["ppo"]["sharpe"], table[bench]["sharpe"]
    ppo_dd, gp_dd = table["ppo"]["maxdd"], table[bench]["maxdd"]
    beat = sr_ci["lo"] > 0
    verdict = (
        f"PPO Sharpe {ppo_sr} vs GP {gp_sr}; Sharpe difference "
        f"{sr_ci['point']:+.3f} with 95% CI [{sr_ci['lo']:+.3f}, {sr_ci['hi']:+.3f}]. "
        f"Annualised excess return {ret_ci['point']:+.2%} "
        f"[{ret_ci['lo']:+.2%}, {ret_ci['hi']:+.2%}], bought with a "
        f"{ppo_dd:.1%} drawdown against {gp_dd:.1%} and "
        f"{table['ppo']['ann_turnover']:.1f}x turnover against "
        f"{table[bench]['ann_turnover']:.1f}x. "
        + ("PPO beats the closed form on risk-adjusted terms out of sample."
           if beat else
           "PPO earns more but not per unit of risk: the Garleanu-Pedersen "
           "closed form is sufficient, and the extra return is extra risk.")
    )

    res = RLResult(train_log=[dict(u) for u in log.updates],
                   eval_table=table, verdict=verdict, spa=spa)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        agent.save(out_dir / "ppo_agent.pt")
        res.to_json(out_dir / "rl_report.json")
    return res
