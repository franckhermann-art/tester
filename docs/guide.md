# Guide d'utilisation

## Préparer les données

La bibliothèque accepte un `pandas.DataFrame`. Les méthodes ne modifient pas les données en entrée.

```python
import pandas as pd

data = pd.read_csv("donnees.csv")
```

## Explorer la structure

```python
from eda_toolbox import missingness_report, profile_dataframe

profile = profile_dataframe(data)
missing = missingness_report(data)
```

`profile_dataframe` retourne une ligne par colonne avec son type, ses effectifs non nuls, le taux de valeurs manquantes, sa cardinalité et, pour les variables numériques, la moyenne, la médiane et l'écart-type.

`missingness_report` retourne le nombre et le pourcentage de valeurs absentes par colonne. Le rapport inclut aussi les colonnes sans valeur manquante.

## Visualiser les variables

```python
import matplotlib.pyplot as plt

from eda_toolbox import (
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_numeric_distributions,
)

plot_numeric_distributions(data, bins=25)
plot_correlation_heatmap(data)
plot_categorical_counts(data, "region", top_n=8)
plt.show()
```

Les fonctions renvoient des figures Matplotlib pour permettre leur affichage, leur personnalisation ou leur sauvegarde. Les histogrammes ignorent les valeurs manquantes. Le graphique catégoriel conserve les valeurs manquantes comme une modalité. La heatmap utilise les corrélations de Pearson et ignore les colonnes non numériques.

## Orienter la suite de l'analyse

```python
from eda_toolbox import suggest_next_steps

for suggestion in suggest_next_steps(data):
    print(suggestion.priority, suggestion.finding)
    print(suggestion.recommendation)
```

Les recommandations signalent quelques cas fréquents : données absentes, lignes dupliquées, asymétrie prononcée, forte cardinalité ou corrélations élevées. Les seuils de valeurs manquantes et de corrélation peuvent être adaptés avec `missing_threshold` et `correlation_threshold`.

Ces signaux sont descriptifs. Ils ne déterminent pas à eux seuls une méthode d'imputation, la suppression d'observations, la sélection de variables ou une relation causale. Il faut confronter les résultats aux objectifs, au domaine métier et aux hypothèses de l'étude.

## Erreurs d'entrée

Les fonctions lèvent `TypeError` si la donnée fournie n'est pas un DataFrame, et `ValueError` si les colonnes demandées ou les paramètres ne conviennent pas. Les visualisations renvoient une erreur explicite lorsqu'elles n'ont pas suffisamment de colonnes à représenter.
