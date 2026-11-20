# EDA Toolbox

Une petite bibliothèque Python pour explorer un jeu de données tabulaire, visualiser ses principales caractéristiques et obtenir des pistes pour la suite de l'analyse.

Les recommandations sont des indicateurs exploratoires : elles ne remplacent ni la connaissance métier, ni une validation statistique.

## Installation

Depuis la racine du projet :

```bash
python -m pip install -e .
```

Pour lancer les tests :

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Démarrage rapide

```python
import pandas as pd
import matplotlib.pyplot as plt

from eda_toolbox import (
    missingness_report,
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_numeric_distributions,
    profile_dataframe,
    suggest_next_steps,
)

data = pd.read_csv("donnees.csv")

print(profile_dataframe(data))
print(missingness_report(data))

for suggestion in suggest_next_steps(data):
    print(f"[{suggestion.priority}] {suggestion.finding}")
    print(f"  Suite proposée : {suggestion.recommendation}")

plot_numeric_distributions(data)
plot_correlation_heatmap(data)
plot_categorical_counts(data, "categorie")
plt.show()
```

Les fonctions de visualisation retournent des objets `matplotlib.figure.Figure` et ne déclenchent pas elles-mêmes l'affichage.

## Fonctionnalités

- Profil des colonnes : type, valeurs manquantes, cardinalité et statistiques numériques.
- Rapport des valeurs manquantes.
- Recommandations exploratoires sur les valeurs manquantes, doublons, distributions, cardinalités et corrélations.
- Histogrammes des variables numériques, heatmap de corrélation et graphiques des catégories les plus fréquentes.
- Validation explicite des entrées et messages d'erreur descriptifs.

## Documentation

Consultez le [guide d'utilisation](docs/guide.md) pour les détails, exemples et limites d'interprétation.

## Développement

Python 3.10 ou plus récent est requis. Les dépendances principales sont pandas, NumPy et Matplotlib.