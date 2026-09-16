"""Portfolio execution environment for reinforcement learning.

What the agent is asked to do — and what it is not
--------------------------------------------------
It does **not** pick stocks.  Ranking names from features is a supervised
problem with a dense label and a clean objective; RL applied to it throws away
the label and estimates the same thing from a far noisier signal.  Nothing in
the literature supports doing that, and this system already has calibrated
cross-sectional forecasters.

It **does** decide how to act on those forecasts through time: how hard to
deploy, how far to trade toward the target, how wide a band to leave, how
concentrated to be.  That problem is genuinely sequential — today's trade sets
tomorrow's starting portfolio and tomorrow's cost — and it is path dependent in
ways a per-period optimiser cannot express: square-root impact, participation
caps that carry an unfilled order forward, and drawdown state.

That framing also fixes the benchmark.  With *quadratic* costs this problem has
a closed-form answer (Gârleanu & Pedersen 2013: aim in front of the target,
trade partially toward the aim), implemented in
:class:`~trading_system.research.execution.GarleanuPedersenPolicy`.  The agent
must beat that policy, not a naive full-rebalance strawman.  If it cannot, the
honest conclusion is that the closed form is enough — and the environment is
built so that conclusion is reachable.

Action space (4 bounded scalars, not N weights)
-----------------------------------------------
Emitting one weight per name makes the action space grow with the universe and
the agent spends its samples rediscovering the ranking it was already given.
Instead the action reparameterises a fixed, sane book construction:

    0  deploy      in [0, 1]     fraction of equity put at risk
    1  trade_rate  in [0, 1]     how far to move toward the target book
    2  band        in [0, 0.5]   no-trade band, as a fraction of target weight
    3  tilt        in [0, 3]     concentration: weight ∝ rank_score ** tilt

Every action maps to a long-only, capped, fully-specified book, so an untrained
agent is merely mediocre rather than catastrophic — and the learned policy stays
inspectable: four numbers you can plot against market state.

Reward
------
Log return of equity after costs, minus a penalty on drawdown below a
threshold.  Log wealth is the right accumulator (it sums to terminal log
wealth), it is naturally risk-averse, and it does not need an arbitrary
volatility target baked in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["EpisodeData", "PortfolioEnv", "EnvConfig", "OBS_NAMES", "ACTION_NAMES"]

ACTION_NAMES = ("deploy", "trade_rate", "band", "tilt")

OBS_NAMES = (
    "gross", "cash_frac", "n_pos_frac", "last_turnover", "book_vol_20",
    "drawdown", "score_dispersion", "score_autocorr", "mkt_ret_20",
    "mkt_vol", "breadth", "median_spread", "median_particip", "t_frac",
)


@dataclass
class EpisodeData:
    """Precomputed, replayable market tape for the environment.

    Built once from a causal scoring pass, then replayed millions of times.  The
    scores must be *out-of-sample with respect to the alpha model* — generated
    by an inner purged walk-forward inside the RL training window — otherwise
    the agent learns to act on a signal quality it will never see live.
    """

    dates: np.ndarray        # (T,)
    ret: np.ndarray          # (T, N) daily simple returns
    scores: np.ndarray       # (T, N) alpha scores, NaN where not scored
    adv: np.ndarray          # (T, N)
    spread_bps: np.ndarray   # (T, N)
    dvol: np.ndarray         # (T, N)
    alive: np.ndarray        # (T, N) bool
    rebalance: np.ndarray    # (T,) bool

    def __post_init__(self):
        self.T, self.N = self.ret.shape


@dataclass
class EnvConfig:
    rebalance_days: int = 21
    episode_days: int = 504          # ~2y per episode
    top_k: int = 20
    max_weight: float = 0.10
    min_dollar_volume: float = 2_000_000.0
    initial_cash: float = 1_000_000.0
    dd_penalty: float = 2.0          # reward penalty per unit drawdown beyond...
    dd_threshold: float = 0.15       # ...this drawdown
    seed: int = 0


class PortfolioEnv:
    """Single-book environment stepped one *rebalance period* at a time.

    Stepping per rebalance rather than per day keeps episodes short (24 steps
    per simulated year at a monthly cadence) so PPO sees many complete episodes,
    while the daily returns inside each step are still accrued exactly.
    """

    def __init__(self, data: EpisodeData, cost_model, cfg: EnvConfig | None = None):
        self.d = data
        self.cost = cost_model
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.rebal_idx = np.nonzero(data.rebalance)[0]
        self.obs_dim = len(OBS_NAMES)
        self.act_dim = len(ACTION_NAMES)
        self._reset_state()

    # ── state ──
    def _reset_state(self):
        self.daily_log: list[tuple[int, float]] = []
        self.pos = np.zeros(self.d.N)
        self.cash = self.cfg.initial_cash
        self.peak = self.cfg.initial_cash
        self.last_turnover = 0.0
        self.recent_rets: list[float] = []
        self.prev_scores: np.ndarray | None = None

    def reset(self, start: int | None = None) -> np.ndarray:
        """Begin a new episode at a random (or given) rebalance date."""
        usable = self.rebal_idx[
            (self.rebal_idx > 25)
            & (self.rebal_idx < self.d.T - self.cfg.episode_days - 5)]
        if len(usable) == 0:
            usable = self.rebal_idx[:1]
        self.k = int(start if start is not None else self.rng.choice(usable))
        self.start_k = self.k
        self.end_k = min(self.k + self.cfg.episode_days, self.d.T - 2)
        self._reset_state()
        return self._obs()

    # ── book construction from an action ──
    def target_weights(self, k: int, tilt: float, deploy: float) -> np.ndarray:
        s = self.d.scores[k].copy()
        liq = (self.d.adv[k] >= self.cfg.min_dollar_volume) & self.d.alive[k]
        s = np.where(liq & np.isfinite(s), s, np.nan)
        ok = np.isfinite(s)
        w = np.zeros(self.d.N)
        if ok.sum() < 5:
            return w
        k_eff = min(self.cfg.top_k, int(ok.sum()))
        idx = np.flatnonzero(ok)
        pick = idx[np.argsort(-s[idx])[:k_eff]]

        # rank-based tilt: exponent 0 -> equal weight, larger -> concentrated
        r = np.arange(k_eff, 0, -1, dtype=float) / k_eff
        raw = r ** max(tilt, 1e-3)
        w[pick] = raw / raw.sum()

        for _ in range(8):
            over = w > self.cfg.max_weight
            if not over.any():
                break
            excess = (w[over] - self.cfg.max_weight).sum()
            w[over] = self.cfg.max_weight
            free = (w > 0) & ~over
            if not free.any():
                break
            w[free] += excess * w[free] / w[free].sum()
        tot = w.sum()
        return w * (deploy / tot) if tot > 0 else w

    # ── observation ──
    def _obs(self) -> np.ndarray:
        k = min(self.k, self.d.T - 1)
        eq = max(self.pos.sum() + self.cash, 1e-6)
        s = self.d.scores[k]
        ok = np.isfinite(s)

        if self.prev_scores is not None:
            m = ok & np.isfinite(self.prev_scores)
            if m.sum() > 10:
                a = s[m].argsort().argsort().astype(float)
                b = self.prev_scores[m].argsort().argsort().astype(float)
                ac = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 else 0.0
            else:
                ac = 0.0
        else:
            ac = 0.0

        lo = max(0, k - 20)
        mkt = self.d.ret[lo:k + 1]
        alive_now = self.d.alive[k]
        held = np.abs(self.pos) > 1e-6
        want = np.abs(self.pos).sum() * 0.2
        particip = np.median(want / np.maximum(self.d.adv[k][alive_now], 1.0)) \
            if alive_now.any() else 0.0

        book_vol = float(np.std(self.recent_rets[-20:])) * np.sqrt(252) \
            if len(self.recent_rets) >= 5 else 0.0

        return np.array([
            self.pos.sum() / eq,
            self.cash / eq,
            held.sum() / max(self.cfg.top_k, 1),
            self.last_turnover,
            book_vol,
            eq / self.peak - 1.0,
            float(np.nanstd(s[ok])) if ok.sum() > 2 else 0.0,
            ac,
            float(np.nanmean(mkt)) * 20 if mkt.size else 0.0,
            float(np.nanstd(mkt)) * np.sqrt(252) if mkt.size else 0.0,
            float(np.nanmean(self.d.ret[k][alive_now] > 0)) if alive_now.any() else 0.5,
            float(np.median(self.d.spread_bps[k][alive_now])) / 100 if alive_now.any() else 0.2,
            float(np.clip(particip, 0, 1)),
            (self.k - self.start_k) / max(self.end_k - self.start_k, 1),
        ], dtype=np.float32)

    # ── step ──
    def step(self, action: np.ndarray):
        """Apply one rebalance decision, then hold to the next rebalance date."""
        from ..research.execution import apply_no_trade_bands

        a = np.asarray(action, dtype=float)
        deploy = float(np.clip(a[0], 0.0, 1.0))
        trade_rate = float(np.clip(a[1], 0.0, 1.0))
        band = float(np.clip(a[2], 0.0, 0.5))
        tilt = float(np.clip(a[3], 0.0, 3.0))

        k = self.k
        eq0 = max(self.pos.sum() + self.cash, 1e-6)
        target_w = self.target_weights(k, tilt, deploy)
        cur_w = self.pos / eq0

        moved = apply_no_trade_bands(target_w, cur_w, band, band * 0.5)
        blended = cur_w + trade_rate * (moved - cur_w)
        desired = blended * eq0 - self.pos
        desired = np.where(self.d.alive[k], desired, -self.pos)

        filled = self.cost.fillable_notional(desired, self.d.adv[k])
        costs = self.cost.cost_notional(
            filled, self.d.adv[k], self.d.spread_bps[k], self.d.dvol[k])
        self.pos = self.pos + filled
        self.cash = self.cash - float(filled.sum()) - float(costs.sum())
        self.last_turnover = float(np.abs(filled).sum() / eq0)
        self.prev_scores = self.d.scores[k].copy()

        # hold to the next rebalance, accruing daily
        nxt = min(k + self.cfg.rebalance_days, self.end_k)
        prev_eq = self.pos.sum() + self.cash
        for t in range(k + 1, nxt + 1):
            self.pos = self.pos * (1.0 + self.d.ret[t])
            self.pos[~self.d.alive[t]] = 0.0
            e = self.pos.sum() + self.cash
            self.recent_rets.append(e / max(prev_eq, 1e-9) - 1.0)
            self.daily_log.append((t, e))
            prev_eq = e

        eq1 = max(self.pos.sum() + self.cash, 1e-9)
        self.peak = max(self.peak, eq1)
        dd = eq1 / self.peak - 1.0

        reward = float(np.log(max(eq1, 1e-9) / max(eq0, 1e-9)))
        if dd < -self.cfg.dd_threshold:
            reward -= self.cfg.dd_penalty * (abs(dd) - self.cfg.dd_threshold)

        self.k = nxt
        done = self.k >= self.end_k or eq1 <= 0.05 * self.cfg.initial_cash
        info = {"equity": eq1, "turnover": self.last_turnover, "dd": dd,
                "cost": float(costs.sum()), "action": (deploy, trade_rate, band, tilt)}
        return self._obs(), reward, done, info

