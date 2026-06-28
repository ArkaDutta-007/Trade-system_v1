"""V2: SHAP waterfall visualization utilities.

Provides SHAP computation and a custom matplotlib waterfall chart that
is compatible with Streamlit's st.pyplot() (avoids shap.plots.waterfall
which requires IPython display machinery).

Usage::

    from trading_system.monitoring.shap_viz import compute_shap_waterfall, render_shap_waterfall_fig

    shap_data = compute_shap_waterfall(model_registry_path, features_df, "AAPL")
    fig = render_shap_waterfall_fig(shap_data)
    st.pyplot(fig)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import get_logger

logger = get_logger(__name__)


def compute_shap_waterfall(
    model_registry_path: Path | str,
    features: "pl.DataFrame",
    ticker: str,
    top_n: int = 12,
) -> dict[str, Any] | None:
    """Compute SHAP values for a specific ticker using the latest ensemble model.

    Supports tree models (LGBM, XGB, RF, GBM) via TreeExplainer and
    linear models (Ridge, ElasticNet) via LinearExplainer.

    Parameters
    ----------
    model_registry_path:
        Path to the reports/models/ directory.
    features:
        Gold feature matrix (pl.DataFrame).
    ticker:
        Ticker symbol to explain (e.g. "AAPL").
    top_n:
        Number of top features to return (by |SHAP value|).

    Returns
    -------
    dict with keys:
        base_value (float): mean model prediction
        shap_values (list[float]): SHAP contributions for top_n features
        feature_names (list[str]): feature names (sorted by |SHAP|)
        feature_values (list[float]): actual feature values for that row
        prediction (float): model's prediction for this ticker
        ticker (str): the ticker
        as_of (str): date of the features used
    Returns None if SHAP computation fails.
    """
    try:
        import shap
        import numpy as np
        import polars as pl
        from ..models.model_registry import load_latest_model

        model_registry_path = Path(model_registry_path)
        ensemble, meta = load_latest_model(model_registry_path)
        feat_cols = meta.get("feature_columns", [])
        if not feat_cols:
            logger.warning("No feature_columns in model metadata.")
            return None

        ticker = ticker.upper()
        last_date = features["date"].max()
        row = features.filter(
            (pl.col("ticker") == ticker) & (pl.col("date") == last_date)
        )
        if row.is_empty():
            logger.warning(f"No features for {ticker} on {last_date}.")
            return None

        available_cols = [c for c in feat_cols if c in row.columns]
        X = row.select(available_cols).to_numpy()

        # Try to get the best underlying tree model from the ensemble
        tree_model = None
        for model_name in ["lgbm", "xgb", "hist_gbm", "extra_trees", "rf"]:
            m = getattr(ensemble, "_models", {}).get(model_name)
            if m is not None:
                tree_model = m
                break

        # Bare tree model in the registry (not wrapped in an ensemble) — use directly
        if tree_model is None and not hasattr(ensemble, "_models"):
            cls = type(ensemble).__name__.lower()
            if any(k in cls for k in ("lgbm", "xgb", "boosting", "forest", "tree", "gbm")):
                tree_model = ensemble

        if tree_model is not None:
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X)
            base_value = float(explainer.expected_value)
            if isinstance(shap_values, list):
                sv = shap_values[0][0]  # multi-output: take first output
            else:
                sv = shap_values[0]
        else:
            # Fall back to linear model or KernelExplainer
            linear_model = None
            for model_name in ["ridge", "elastic_net", "huber", "bayesian_ridge"]:
                m = getattr(ensemble, "_models", {}).get(model_name)
                if m is not None:
                    linear_model = m
                    break
            if linear_model is not None:
                explainer = shap.LinearExplainer(linear_model, X, feature_perturbation="correlation_dependent")
                shap_values = explainer.shap_values(X)
                base_value = float(explainer.expected_value)
                sv = shap_values[0]
            else:
                logger.warning("No compatible model found for SHAP computation.")
                return None

        # Get model prediction
        prediction = float(ensemble.predict(X)[0])

        # Sort features by |SHAP value|
        abs_shap = np.abs(sv)
        sorted_idx = np.argsort(abs_shap)[::-1][:top_n]

        return {
            "ticker": ticker,
            "as_of": str(last_date),
            "base_value": base_value,
            "prediction": prediction,
            "feature_names": [available_cols[i] for i in sorted_idx],
            "shap_values": [float(sv[i]) for i in sorted_idx],
            "feature_values": [float(X[0][i]) for i in sorted_idx],
        }

    except Exception as e:
        logger.warning(f"SHAP computation failed for {ticker}: {e}")
        return None


def render_shap_waterfall_fig(
    shap_data: dict[str, Any],
    figsize: tuple[float, float] | None = None,
    dark_theme: bool = True,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Render a waterfall chart from shap_data as a matplotlib Figure.

    Compatible with st.pyplot() in Streamlit without IPython dependency.

    Parameters
    ----------
    shap_data:
        Output dict from compute_shap_waterfall().
    figsize:
        Matplotlib figure size (width, height) in inches.
        Defaults to auto-sized based on number of features.
    dark_theme:
        Use dark background (matches Streamlit dark theme).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    names = shap_data["feature_names"]
    values = shap_data["shap_values"]
    feat_vals = shap_data["feature_values"]
    base = shap_data.get("base_value", 0.0)
    prediction = shap_data.get("prediction", base + sum(values))
    ticker = shap_data.get("ticker", "")
    as_of = shap_data.get("as_of", "")

    n = len(names)
    bg_color = "#0e1117" if dark_theme else "white"
    text_color = "white" if dark_theme else "black"
    pos_color = "#2dc653"  # green
    neg_color = "#e63946"  # red
    base_color = "#aaaaaa"

    # Auto-size height based on number of features so labels never overlap
    if figsize is None:
        figsize = (10, max(6, n * 0.55 + 2))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Build waterfall: start at base_value, accumulate SHAP contributions
    # Reverse order: top feature at top of chart
    rev_names = list(reversed(names))
    rev_values = list(reversed(values))
    rev_feat_vals = list(reversed(feat_vals))

    bar_colors = []
    bar_labels = []
    lefts = []
    running_fwd = base
    for v, fn, fv in zip(rev_values, rev_names, rev_feat_vals):
        lefts.append(running_fwd)
        bar_colors.append(pos_color if v >= 0 else neg_color)
        bar_labels.append(f"{fn}={fv:.3f}  ({'+' if v >= 0 else ''}{v:.4f})")
        running_fwd += v

    y_pos = list(range(n))
    ax.barh(
        y_pos,
        rev_values,
        left=lefts,
        color=bar_colors,
        height=0.65,
        edgecolor=bg_color,
        linewidth=0.5,
    )

    # Base value line
    ax.axvline(base, color=base_color, linestyle="--", linewidth=1.0, alpha=0.7, label=f"Base: {base:.4f}")
    # Prediction line
    ax.axvline(prediction, color="#f4a261", linestyle="-", linewidth=1.5, alpha=0.9, label=f"Prediction: {prediction:.4f}")

    # Feature labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=8, color=text_color)

    # Spine cleanup
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=text_color, labelsize=8)
    ax.tick_params(axis="y", left=False)
    ax.xaxis.label.set_color(text_color)
    ax.set_xlabel("SHAP Value (impact on prediction)", color=text_color, fontsize=9)

    title = f"SHAP Waterfall — {ticker}  (as of {as_of})"
    ax.set_title(title, color=text_color, fontsize=11, fontweight="bold", pad=12)

    legend = ax.legend(
        handles=[
            mpatches.Patch(color=pos_color, label="Positive impact"),
            mpatches.Patch(color=neg_color, label="Negative impact"),
            mpatches.Patch(color=base_color, label=f"Base: {base:.4f}"),
            mpatches.Patch(color="#f4a261", label=f"Prediction: {prediction:.4f}"),
        ],
        loc="lower right",
        fontsize=7,
        framealpha=0.3,
        labelcolor=text_color,
    )
    legend.get_frame().set_facecolor(bg_color)

    # Reserve left margin so feature labels aren't clipped
    fig.subplots_adjust(left=0.45)
    fig.tight_layout()
    return fig


def compute_shap_summary_df(
    model_registry_path: Path | str,
    features: "pl.DataFrame",
    top_n: int = 20,
) -> "pl.DataFrame":
    """Compute mean |SHAP| across the universe for feature importance ranking.

    Returns a pl.DataFrame with columns: feature, mean_abs_shap.
    Falls back to returning an empty DataFrame on failure.
    """
    try:
        import shap
        import numpy as np
        import polars as pl
        from ..models.model_registry import load_latest_model

        model_registry_path = Path(model_registry_path)
        ensemble, meta = load_latest_model(model_registry_path)
        feat_cols = meta.get("feature_columns", [])
        if not feat_cols:
            return pl.DataFrame(schema={"feature": pl.Utf8, "mean_abs_shap": pl.Float64})

        # Use last date's cross-section
        last_date = features["date"].max()
        X = features.filter(pl.col("date") == last_date)
        available = [c for c in feat_cols if c in X.columns]
        X_np = X.select(available).to_numpy()

        tree_model = None
        for name in ["lgbm", "xgb", "hist_gbm", "extra_trees", "rf"]:
            m = getattr(ensemble, "_models", {}).get(name)
            if m is not None:
                tree_model = m
                break

        if tree_model is None:
            return pl.DataFrame(schema={"feature": pl.Utf8, "mean_abs_shap": pl.Float64})

        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_np)
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1][:top_n]
        return pl.DataFrame({
            "feature": [available[i] for i in sorted_idx],
            "mean_abs_shap": [float(mean_abs[i]) for i in sorted_idx],
        })
    except Exception as e:
        logger.warning(f"SHAP summary computation failed: {e}")
        import polars as pl
        return pl.DataFrame(schema={"feature": pl.Utf8, "mean_abs_shap": pl.Float64})
