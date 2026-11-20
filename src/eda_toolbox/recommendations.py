"""Pistes d'analyse dérivées de constats descriptifs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eda_toolbox._validation import require_dataframe


@dataclass(frozen=True)
class AnalysisSuggestion:
    """Constat descriptif et prochaine étape d'analyse suggérée."""

    priority: str
    finding: str
    recommendation: str


def suggest_next_steps(
    data: pd.DataFrame,
    *,
    missing_threshold: float = 0.05,
    correlation_threshold: float = 0.8,
) -> list[AnalysisSuggestion]:
    """Propose des vérifications exploratoires, sans conclure sur la causalité."""
    require_dataframe(data)
    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold doit être compris entre 0 et 1.")
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold doit être compris entre 0 et 1.")

    if data.empty:
        return [
            AnalysisSuggestion(
                "high",
                "Le jeu de données ne contient aucune ligne.",
                "Vérifier la source, les filtres appliqués et le chargement des données.",
            )
        ]

    suggestions: list[AnalysisSuggestion] = []
    missing_ratio = data.isna().mean()
    missing_columns = missing_ratio[missing_ratio > 0]
    if not missing_columns.empty:
        affected = int((missing_ratio >= missing_threshold).sum())
        suggestions.append(
            AnalysisSuggestion(
                "high" if affected else "medium",
                f"{len(missing_columns)} colonne(s) contiennent des valeurs manquantes.",
                "Étudier leur mécanisme et leur répartition avant de choisir une stratégie de traitement.",
            )
        )

    duplicate_count = int(data.duplicated().sum())
    if duplicate_count:
        suggestions.append(
            AnalysisSuggestion(
                "medium",
                f"{duplicate_count} ligne(s) dupliquée(s) ont été détectées.",
                "Vérifier si ces répétitions sont attendues avant toute déduplication.",
            )
        )

    numeric = data.select_dtypes(include="number")
    skewed = [
        str(column)
        for column in numeric.columns
        if numeric[column].nunique(dropna=True) > 2
        and abs(float(numeric[column].skew())) > 1
    ]
    if skewed:
        suggestions.append(
            AnalysisSuggestion(
                "low",
                f"Distribution très asymétrique détectée pour : {', '.join(skewed)}.",
                "Examiner les histogrammes, les valeurs extrêmes et, si pertinent, une transformation.",
            )
        )

    categorical = data.select_dtypes(exclude="number")
    high_cardinality = [
        str(column)
        for column in categorical.columns
        if categorical[column].nunique(dropna=True) / len(data) > 0.5
    ]
    if high_cardinality:
        suggestions.append(
            AnalysisSuggestion(
                "low",
                f"Cardinalité élevée pour : {', '.join(high_cardinality)}.",
                "Vérifier s'il s'agit d'identifiants ou de catégories à regrouper avant une analyse par modalité.",
            )
        )

    if numeric.shape[1] >= 2:
        correlations = numeric.corr().abs()
        upper_triangle = correlations.where(
            np.triu(np.ones(correlations.shape, dtype=bool), k=1)
        )
        pairs = [
            (str(column), str(row), float(value))
            for column in upper_triangle.columns
            for row, value in upper_triangle[column].items()
            if pd.notna(value) and value >= correlation_threshold
        ]
        if pairs:
            pair_names = ", ".join(f"{left}/{right}" for left, right, _ in pairs)
            suggestions.append(
                AnalysisSuggestion(
                    "medium",
                    f"Corrélation linéaire élevée observée entre : {pair_names}.",
                    "Vérifier la redondance des variables et le contexte métier ; une corrélation n'implique pas une causalité.",
                )
            )

    if not suggestions:
        suggestions.append(
            AnalysisSuggestion(
                "low",
                "Aucun signal exploratoire courant n'a été détecté par les contrôles intégrés.",
                "Poursuivre avec des questions métier, une analyse de la cible et la validation des hypothèses.",
            )
        )
    return suggestions
