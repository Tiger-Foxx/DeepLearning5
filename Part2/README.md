# Partie 2 : Séquence-à-Séquence avec Mémoire

## Description

Ce dossier contient l'implémentation d'un modèle Sequence-to-Sequence (Seq2Seq) pour la prédiction de séries temporelles synthétiques.

## Architecture

- **Encodeur** : Bi-LSTM (Capture du contexte passé et futur local).
- **Décodeur** : LSTM + Cross-Attention (Génération autoregressive avec focalisation sur l'encodage).
- **Attention** : Scaled Dot-Product Attention.

## Structure

- `data/generator.py` : Générateur de données synthétiques (somme de sinusoïdes + tendance).
- `src/model.py` : Définition du modèle Seq2Seq avec Attention.
- `experiments/train_mlflow.py` : Script d'entraînement, de logging MLflow et d'analyse de l'Attention Span.

## Exécution

Pour entraîner le modèle et visualiser l'attention :

```bash
python Part2/experiments/train_mlflow.py
```

Les résultats (modèles, métriques, heatmaps d'attention) seront enregistrés dans un dossier `output` et gérés via MLflow (dossier `mlruns` à la racine).
