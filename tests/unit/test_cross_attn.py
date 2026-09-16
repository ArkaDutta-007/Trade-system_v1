"""Properties of the cross-sectional attention forecaster.

The architectural claim is specific: attention over the asset axis lets a name's
score depend on the rest of the cross-section, which a row-wise model cannot do.
These tests check that the claim holds mechanically (permutation equivariance,
no positional leakage) and empirically (it learns a signal that is only visible
from the cross-section, where a GBM cannot).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from trading_system.models.cross_attn import (  # noqa: I001
    CrossAttnAlpha, CrossAttnConfig, CrossSectionalAttention, build_date_batches,
    rank_gauss, rank_gauss_2d, torch_available,
)

pytestmark = pytest.mark.skipif(not torch_available(), reason="needs torch")


class TestRankGauss:
    def test_output_is_standardised_and_bounded(self):
        x = np.exp(np.random.default_rng(0).normal(0, 3, 500))   # very skewed
        z = rank_gauss(x)
        assert abs(z.mean()) < 0.05
        assert 0.9 < z.std() < 1.1
        assert np.abs(z).max() < 4

    def test_it_is_monotone(self):
        x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        z = rank_gauss(x)
        assert list(np.argsort(z)) == list(np.argsort(x))

    def test_outliers_cannot_dominate(self):
        a = rank_gauss(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        b = rank_gauss(np.array([1.0, 2.0, 3.0, 4.0, 1e9]))
        assert a == pytest.approx(b), "only the ranks matter, never the magnitude"

    def test_degenerate_input_does_not_blow_up(self):
        assert np.isfinite(rank_gauss(np.array([np.nan, np.nan, 1.0]))).all()
        assert len(rank_gauss(np.array([1.0]))) == 1

    def test_vectorised_form_matches_the_column_wise_one(self):
        """The 2-D version is the hot path; it must be identical, not merely close."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(360, 68))
        X[rng.random(X.shape) < 0.05] = np.nan
        wide = rank_gauss_2d(X)
        narrow = np.column_stack([rank_gauss(X[:, j]) for j in range(X.shape[1])])
        assert wide == pytest.approx(narrow, abs=1e-9)

    def test_vectorised_form_handles_all_nan_and_constant_columns(self):
        X = np.column_stack([
            np.full(50, np.nan),
            np.full(50, 3.0),
            np.arange(50.0),
        ])
        out = rank_gauss_2d(X)
        assert np.isfinite(out).all()
        assert (out[:, 0] == 0).all(), "an all-NaN column contributes nothing"
        assert out[:, 2].argsort().tolist() == list(range(50))


class TestArchitecture:
    def test_parameter_count_lands_near_the_documented_plateau(self):
        """Kelly et al. find out-of-sample gains flat by ~25k parameters."""
        m = CrossSectionalAttention(68, CrossAttnConfig(device="cpu"))
        assert 5_000 < m.n_params < 60_000

    def test_scores_are_permutation_equivariant(self):
        """Relabelling the tickers must permute the scores, never change them.

        This is the whole reason there is no positional encoding: the daily
        cross-section is a set. If this fails, the model has learned something
        about row order, which is an artefact of the data layout.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 12)).astype(np.float32)
        m = CrossSectionalAttention(12, CrossAttnConfig(device="cpu", dropout=0.0))
        s1 = m.predict_one(X)
        perm = rng.permutation(40)
        s2 = m.predict_one(X[perm])
        assert s2 == pytest.approx(s1[perm], abs=1e-4)

    def test_a_name_score_depends_on_the_other_names(self):
        """The cross-asset channel must actually carry information.

        A row-wise model would give the same score for an identical feature row
        regardless of what else is in the cross-section. This one must not.
        """
        rng = np.random.default_rng(1)
        m = CrossSectionalAttention(8, CrossAttnConfig(device="cpu", dropout=0.0))
        target_row = rng.normal(size=(1, 8)).astype(np.float32)
        ctx_a = rng.normal(size=(30, 8)).astype(np.float32)
        ctx_b = rng.normal(loc=3.0, size=(30, 8)).astype(np.float32)
        s_a = m.predict_one(np.vstack([target_row, ctx_a]))[0]
        s_b = m.predict_one(np.vstack([target_row, ctx_b]))[0]
        assert abs(s_a - s_b) > 1e-6


class TestBatching:
    def test_dates_are_grouped_and_normalised_within_each_date(self):
        rows = []
        for d in range(5):
            for t in range(30):
                rows.append({"date": dt.date(2020, 1, 1) + dt.timedelta(days=d),
                             "ticker": f"T{t}", "f0": float(t) * (d + 1),
                             "f1": float(t % 7), "y": float(t)})
        df = pl.DataFrame(rows)
        Xs, ys, ds, ctx = build_date_batches(df, ["f0", "f1"], "y")
        assert len(Xs) == len(ys) == len(ds) == 5
        for X in Xs:
            assert X.shape == (30, 2)
            # f0's scale changes tenfold across dates; after within-date ranking
            # every date must look the same
            assert abs(X[:, 0].mean()) < 0.05
        assert np.allclose(Xs[0][:, 0], Xs[4][:, 0], atol=1e-5)

    def test_short_cross_sections_are_dropped(self):
        rows = [{"date": dt.date(2020, 1, 1), "ticker": f"T{t}", "f0": 1.0, "y": 1.0}
                for t in range(5)]
        Xs, _, _, _ = build_date_batches(pl.DataFrame(rows), ["f0"], "y")
        assert Xs == []


class TestLearning:
    """A signal only a cross-asset model can see.

    ``regime_t`` is the cross-sectional mean of feature 1 on date t.  A row-wise
    learner sees one asset's ``f1``, never the mean, so it cannot recover the
    sign flip; attention across assets can.
    """

    @staticmethod
    def _panel(n_dates=260, n_assets=100, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        d0 = dt.date(2015, 1, 1)
        for t in range(n_dates):
            X = rng.normal(size=(n_assets, 4))
            regime = X[:, 1].mean() * 3.0
            y = (X[:, 0] - np.median(X[:, 0])) * regime + rng.normal(0, 0.5, n_assets)
            for i in range(n_assets):
                rows.append({"date": d0 + dt.timedelta(days=t), "ticker": f"T{i:03d}",
                             **{f"f{j}": float(X[i, j]) for j in range(4)},
                             "y": float(y[i])})
        return pl.DataFrame(rows)

    @pytest.mark.slow
    def test_it_beats_a_row_wise_gbm_on_a_cross_sectional_signal(self):
        from scipy.stats import spearmanr

        from trading_system.research.alphas import GBMAlpha

        df = self._panel()
        cut = df["date"].min() + dt.timedelta(days=200)
        tr, te = df.filter(pl.col("date") < cut), df.filter(pl.col("date") >= cut)
        feat = [f"f{j}" for j in range(4)]

        def mean_ic(predict):
            ics = []
            for (_d,), g in te.group_by(["date"], maintain_order=True):
                r, _ = spearmanr(g["y"].to_numpy(), predict(g))
                if np.isfinite(r):
                    ics.append(r)
            return float(np.mean(ics))

        gbm = GBMAlpha("lgbm")
        gbm.fit(tr, feat, "y")
        ic_gbm = mean_ic(lambda g: gbm.predict(g, feat))

        ca = CrossAttnAlpha(CrossAttnConfig(epochs=50, patience=10, verbose=False,
                                            device="cpu"), n_seeds=1)
        ca.fit(tr, feat, "y")
        ic_ca = mean_ic(lambda g: ca.predict(g, feat))

        assert abs(ic_gbm) < 0.10, "a row-wise model cannot see the regime"
        assert ic_ca > ic_gbm + 0.05, f"cross-attn {ic_ca:.3f} vs gbm {ic_gbm:.3f}"
