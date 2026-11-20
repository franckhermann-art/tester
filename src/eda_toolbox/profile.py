"""Profils statistiques et rapport de valeurs manquantes."""

import numpy as np
import pandas as pd

from eda_toolbox._validation import require_dataframe


def profile_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Retourne un résumé typé de chaque colonne du jeu de données."""
    require_dataframe(data)
    row_count = len(data)
    records: list[dict[str, object]] = []

    for column in data.columns:
        values = data[column]
        numeric = pd.api.types.is_numeric_dtype(values)
        record: dict[str, object] = {
            "column": column,
            "dtype": str(values.dtype),
            "non_null_count": int(values.notna().sum()),
            "missing_count": int(values.isna().sum()),
            "missing_pct": float(values.isna().mean() * 100) if row_count else 0.0,
            "unique_count": int(values.nunique(dropna=True)),
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }
        if numeric:
            record.update(
                mean=float(values.mean()) if values.notna().any() else np.nan,
                median=float(values.median()) if values.notna().any() else np.nan,
                std=float(values.std()) if values.notna().any() else np.nan,
            )
        records.append(record)

    return pd.DataFrame.from_records(
        records,
        columns=[
            "column",
            "dtype",
            "non_null_count",
            "missing_count",
            "missing_pct",
            "unique_count",
            "mean",
            "median",
            "std",
        ],
    )


def missingness_report(data: pd.DataFrame) -> pd.DataFrame:
    """Retourne le nombre et le pourcentage de valeurs manquantes par colonne."""
    require_dataframe(data)
    row_count = len(data)
    missing_count = data.isna().sum()
    return pd.DataFrame(
        {
            "missing_count": missing_count.astype("int64"),
            "missing_pct": (
                data.isna().mean().mul(100) if row_count else 0.0
            ),
        }
    )
