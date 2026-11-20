"""Visualisations exploratoires Matplotlib."""

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from eda_toolbox._validation import require_dataframe


def plot_numeric_distributions(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    bins: int = 30,
) -> Figure:
    """Crée un histogramme par variable numérique et retourne la figure."""
    require_dataframe(data)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins doit être un entier strictement positif.")

    if columns is None:
        selected = list(data.select_dtypes(include="number").columns)
    else:
        selected = columns
        absent = [column for column in selected if column not in data.columns]
        if absent:
            raise ValueError(f"Colonnes absentes : {', '.join(absent)}.")
        non_numeric = [
            column
            for column in selected
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if non_numeric:
            raise ValueError(
                f"Colonnes non numériques : {', '.join(non_numeric)}."
            )

    if not selected:
        raise ValueError("Aucune colonne numérique à représenter.")

    column_count = min(3, len(selected))
    row_count = math.ceil(len(selected) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5 * column_count, 3.5 * row_count),
        squeeze=False,
    )
    for axis, column in zip(axes.flat, selected):
        axis.hist(data[column].dropna(), bins=bins, edgecolor="white")
        axis.set(title=str(column), xlabel=str(column), ylabel="Fréquence")
    for axis in list(axes.flat)[len(selected) :]:
        axis.set_visible(False)
    figure.tight_layout()
    return figure


def plot_correlation_heatmap(data: pd.DataFrame) -> Figure:
    """Visualise les corrélations de Pearson entre colonnes numériques."""
    require_dataframe(data)
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("Au moins deux colonnes numériques sont nécessaires.")

    correlations = numeric.corr()
    figure, axis = plt.subplots(
        figsize=(max(5, len(correlations.columns)), max(4, len(correlations.columns) * 0.8))
    )
    image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    labels = [str(column) for column in correlations.columns]
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title("Corrélations de Pearson")
    figure.colorbar(image, ax=axis, label="Corrélation")
    figure.tight_layout()
    return figure


def plot_categorical_counts(
    data: pd.DataFrame,
    column: str,
    *,
    top_n: int = 10,
) -> Figure:
    """Trace les modalités les plus fréquentes d'une colonne catégorielle."""
    require_dataframe(data)
    if column not in data.columns:
        raise ValueError(f"Colonne absente : {column}.")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n doit être un entier strictement positif.")
    if pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError("La colonne doit être catégorielle, pas numérique.")

    counts = data[column].value_counts(dropna=False).head(top_n)
    figure, axis = plt.subplots(figsize=(max(6, len(counts) * 0.7), 4))
    axis.bar(counts.index.astype(str), counts.values)
    axis.set(title=f"Modalités les plus fréquentes — {column}", xlabel=column, ylabel="Effectif")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    return figure
