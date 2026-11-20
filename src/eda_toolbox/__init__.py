"""Outils d'exploration et de visualisation de données tabulaires."""

from eda_toolbox.profile import missingness_report, profile_dataframe
from eda_toolbox.recommendations import AnalysisSuggestion, suggest_next_steps
from eda_toolbox.visualization import (
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_numeric_distributions,
)

__all__ = [
    "AnalysisSuggestion",
    "missingness_report",
    "plot_categorical_counts",
    "plot_correlation_heatmap",
    "plot_numeric_distributions",
    "profile_dataframe",
    "suggest_next_steps",
]
