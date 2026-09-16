"""Calibration of the robustness statistics.

These are the tests that matter most in the whole research package, because
every conclusion is filtered through them. The check is not "does the function
run" but "does it say *no* when the answer is no": fed pure noise and a search
over many configurations, each statistic has to refuse to be impressed.
"""
from __future__ import annotations

import numpy as np
import pytest

from trading_system.research.stats import (
    bootstrap_ci, deflated_sharpe, paired_bootstrap_ci, pbo_cscv,
    probabilistic_sharpe, sharpe, spa_test, stationary_bootstrap,
)


@pytest.fixture(scope="module")
def noise_matrix():
    """200 skill-less strategies over 10 years of daily returns."""
    return np.random.default_rng(1).normal(0, 0.01, (2500, 200))


class TestSharpe:
    def test_annualisation(self):
        r = np.full(504, 0.001)
        assert sharpe(r) == 0.0, "zero variance has no Sharpe"
        rng = np.random.default_rng(0)
        x = rng.normal(0.0004, 0.01, 100_000)
        assert sharpe(x) == pytest.approx(0.0004 / 0.01 * np.sqrt(252), rel=0.1)

    def test_degenerate_inputs(self):
        assert sharpe(np.array([])) == 0.0
        assert sharpe(np.array([0.01])) == 0.0


class TestDeflatedSharpe:
    def test_the_best_of_many_noise_strategies_is_not_credited(self, noise_matrix):
        """This is the whole point: best-of-N noise looks good and must not pass."""
        srs = noise_matrix.mean(0) / noise_matrix.std(0, ddof=1)
        best = int(np.argmax(srs))
        naive = srs[best] * np.sqrt(252)
        assert naive > 0.4, "best-of-200 noise should look superficially decent"
        d = deflated_sharpe(noise_matrix[:, best], srs)
        assert d["dsr"] < 0.6, f"DSR {d['dsr']:.3f} credited a skill-less winner"

    def test_a_real_signal_scores_above_the_best_noise_winner(self, noise_matrix):
        """The ordering is the invariant; the level depends on how wide the search was.

        A Sharpe-1.3 signal over ten years is genuinely marginal once you have
        searched 200 configurations, and the statistic says so (DSR near 0.5).
        That is the correction working, not a failure. What must always hold is
        that the real signal outranks the luckiest noise strategy.
        """
        rng = np.random.default_rng(7)
        real = rng.normal(0.0008, 0.01, 2500)
        srs = noise_matrix.mean(0) / noise_matrix.std(0, ddof=1)
        best_noise = noise_matrix[:, int(np.argmax(srs))]
        assert deflated_sharpe(real, srs)["dsr"] > deflated_sharpe(best_noise, srs)["dsr"]

    def test_a_strong_signal_clears_the_haircut_outright(self, noise_matrix):
        rng = np.random.default_rng(7)
        strong = rng.normal(0.0020, 0.01, 2500)    # Sharpe ~3 annualised
        srs = noise_matrix.mean(0) / noise_matrix.std(0, ddof=1)
        assert deflated_sharpe(strong, srs)["dsr"] > 0.95

    def test_a_wider_search_makes_the_same_signal_less_credible(self, noise_matrix):
        rng = np.random.default_rng(7)
        real = rng.normal(0.0008, 0.01, 2500)
        srs = noise_matrix.mean(0) / noise_matrix.std(0, ddof=1)
        narrow = deflated_sharpe(real, srs[:5])["dsr"]
        wide = deflated_sharpe(real, srs)["dsr"]
        assert narrow > wide

    def test_more_trials_raise_the_bar(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.0004, 0.01, 2500)
        few = deflated_sharpe(r, rng.normal(0, 0.02, 5))
        many = deflated_sharpe(r, rng.normal(0, 0.02, 500))
        assert many["sr0_annual"] > few["sr0_annual"]


class TestPBO:
    def test_pure_noise_shows_heavy_overfitting(self, noise_matrix):
        out = pbo_cscv(noise_matrix, n_blocks=12)
        assert out["pbo"] > 0.4, "selecting among noise must look overfit"
        assert out["median_oos_rank"] < 0.55

    def test_a_dominant_strategy_is_selected_reliably(self):
        rng = np.random.default_rng(5)
        R = rng.normal(0, 0.01, (2500, 20))
        R[:, 3] += 0.0015                       # one genuinely better column
        out = pbo_cscv(R, n_blocks=12)
        assert out["pbo"] < 0.2
        assert out["median_oos_rank"] > 0.8

    def test_it_refuses_a_series_that_is_too_short(self):
        with pytest.raises(ValueError):
            pbo_cscv(np.random.default_rng(0).normal(size=(10, 5)))


class TestBootstrap:
    def test_stationary_bootstrap_preserves_autocorrelation(self):
        rng = np.random.default_rng(0)
        x = np.zeros(3000)
        for t in range(1, 3000):
            x[t] = 0.85 * x[t - 1] + rng.normal()     # strongly autocorrelated
        draws = stationary_bootstrap(x, n_boot=40, mean_block=40, rng=rng)

        def ac1(v):
            return float(np.corrcoef(v[:-1], v[1:])[0, 1])

        kept = np.mean([ac1(d) for d in draws])
        shuffled = np.mean([ac1(rng.permutation(x)) for _ in range(40)])
        assert kept > 0.5, "block resampling must retain serial dependence"
        assert abs(shuffled) < 0.1

    def test_confidence_interval_brackets_the_point_estimate(self):
        r = np.random.default_rng(2).normal(0.0006, 0.01, 2500)
        ci = bootstrap_ci(r, n_boot=300)
        assert ci["lo"] < ci["point"] < ci["hi"]

    def test_a_zero_mean_series_cannot_be_called_profitable(self):
        r = np.random.default_rng(11).normal(0, 0.01, 2500)
        assert bootstrap_ci(r, n_boot=300)["lo"] < 0


class TestPairedBootstrap:
    def test_pairing_tightens_the_interval_for_correlated_series(self):
        """Two strategies on the same dates share most of their variance.

        Resampling them independently would treat that shared move as risk in
        the difference and make every comparison inconclusive.
        """
        rng = np.random.default_rng(0)
        common = rng.normal(0, 0.012, 2500)
        a = common + rng.normal(0.0004, 0.002, 2500)
        b = common + rng.normal(0, 0.002, 2500)
        paired = paired_bootstrap_ci(a, b, n_boot=300)
        assert paired["point"] > 0
        assert paired["lo"] > 0, "a real risk-adjusted edge should be detected"
        assert (paired["hi"] - paired["lo"]) < 1.0

    def test_two_identical_series_show_no_difference(self):
        r = np.random.default_rng(4).normal(0.0005, 0.01, 1500)
        out = paired_bootstrap_ci(r, r.copy(), n_boot=200)
        assert out["point"] == pytest.approx(0.0, abs=1e-9)
        assert out["lo"] == pytest.approx(0.0, abs=1e-9)

    def test_it_separates_more_risk_from_more_skill(self):
        """A leveraged copy earns more but is not better per unit of risk."""
        rng = np.random.default_rng(6)
        base = rng.normal(0.0004, 0.01, 2500)
        levered = base * 2.0
        out = paired_bootstrap_ci(levered, base, n_boot=200)
        assert out["point"] == pytest.approx(0.0, abs=1e-6)


class TestSPA:
    def test_many_noise_models_do_not_beat_a_zero_benchmark(self, noise_matrix):
        out = spa_test(np.zeros(2500), -noise_matrix[:, :50], n_boot=200)
        assert out["p_value"] > 0.10, (
            f"SPA p={out['p_value']:.3f} credited a search over 50 noise models")

    def test_a_genuinely_better_model_is_detected(self):
        rng = np.random.default_rng(9)
        bench = rng.normal(0, 0.01, 2500)
        models = rng.normal(0, 0.01, (2500, 10))
        models[:, 4] += 0.0012
        out = spa_test(-bench, -models, n_boot=300)
        assert out["p_value"] < 0.10
        assert out["best_model"] == 4

    def test_probabilistic_sharpe_accounts_for_sample_length(self):
        rng = np.random.default_rng(8)
        short = rng.normal(0.0005, 0.01, 60)
        long = rng.normal(0.0005, 0.01, 6000)
        assert probabilistic_sharpe(long) > probabilistic_sharpe(short)
