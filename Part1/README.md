# Partie 1 : Attention de Base

## Description

Ce dossier contient l'implémentation d'une couche d'attention personnalisée (`SimpleAttention`) couplée à un réseau de neurones récurrent (GRU).

## Structure

- `src/attention_layer.py` : Classe `SimpleAttention` implémentant le mécanisme d'attention (Produt scalaire + tanh + Softmax).
- `src/model.py` : Architecture du modèle (Input -> GRU -> Attention -> Dense).
- `experiments/run_experiment.py` : Script de test et validation du modèle.

## Exécution

Pour lancer l'expérience de vérification :

```bash
python Part1/experiments/run_experiment.py
```

Le script :

1. Génère des données synthétiques.
2. Entraîne le modèle sur quelques époques.
3. Extrait et vérifie les poids d'attention (forme et somme = 1).
4. Sauvegarde un tracé des poids d'attention dans `experiments/results/attention_plot.png`.
