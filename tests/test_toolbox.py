import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from eda_toolbox import (
    missingness_report,
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_numeric_distributions,
    profile_dataframe,
    suggest_next_steps,
)


def test_profile_and_missingness_report_cover_numeric_and_empty_data() -> None:
    data = pd.DataFrame({"score": [1.0, np.nan, 3.0], "group": ["a", "b", "a"]})

    profile = profile_dataframe(data).set_index("column")
    missing = missingness_report(data)

    assert profile.loc["score", "missing_count"] == 1
    assert profile.loc["score", "mean"] == 2.0
    assert np.isnan(profile.loc["group", "mean"])
    assert missing.loc["score", "missing_pct"] == pytest.approx(100 / 3)

    empty = profile_dataframe(pd.DataFrame(columns=["x"]))
    assert empty.loc[0, "missing_pct"] == 0


def test_suggestions_flag_duplicate_missing_skew_and_correlation() -> None:
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 100.0, 1.0],
            "related": [2.0, 4.0, 6.0, 8.0, 200.0, 2.0],
            "label": ["a", None, "b", "b", "c", "a"],
        }
    )

    suggestions = suggest_next_steps(data)
    findings = " ".join(suggestion.finding for suggestion in suggestions)

    assert "valeurs manquantes" in findings
    assert "dupliquée" in findings
    assert "asymétrique" in findings
    assert "Corrélation linéaire élevée" in findings


def test_visualizations_return_figures_and_validate_inputs() -> None:
    data = pd.DataFrame({"value": [1, 2, 3], "group": ["a", "b", "a"]})

    assert plot_numeric_distributions(data).axes
    assert plot_categorical_counts(data, "group").axes
    with pytest.raises(ValueError, match="Au moins deux"):
        plot_correlation_heatmap(data)
    with pytest.raises(ValueError, match="Colonnes absentes"):
        plot_numeric_distributions(data, ["missing"])
    with pytest.raises(ValueError, match="strictement positif"):
        plot_categorical_counts(data, "group", top_n=0)


def test_empty_data_and_invalid_thresholds_are_explicit() -> None:
    assert suggest_next_steps(pd.DataFrame())[0].priority == "high"
    with pytest.raises(ValueError, match="missing_threshold"):
        suggest_next_steps(pd.DataFrame({"x": [1]}), missing_threshold=2)
