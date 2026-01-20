# MA-TAP Research Paper

## Fichiers

- `main.tex` - Article scientifique (format NeurIPS)
- `neurips_2024.sty` - Style NeurIPS simplifié
- `figures/` - Figures générées par l'entraînement

## Compilation

### Option 1: Overleaf (Recommandé)
1. Créer un nouveau projet sur [overleaf.com](https://overleaf.com)
2. Uploader tous les fichiers (`main.tex`, `neurips_2024.sty`, `figures/`)
3. Compiler avec pdfLaTeX

### Option 2: Local (si LaTeX installé)
```bash
cd Paper
pdflatex main.tex
pdflatex main.tex  # 2ème passe pour références
```

## Contenu de l'Article

1. **Abstract** - Résumé des contributions
2. **Introduction** - Problème du latent drift
3. **Related Work** - TAP, attention, memory networks
4. **Method** - Architecture MA-TAP détaillée
5. **Experiments** - Résultats sur Moving MNIST (50 epochs)
6. **Ablation Study** - Impact de la taille mémoire
7. **Discussion** - Limites et perspectives
8. **Conclusion** - Résumé des gains

## Figures

| Figure | Description |
|--------|-------------|
| `training_curves.png` | Courbes Loss et SSIM sur 50 epochs |
| `ssim_comparison.png` | SSIM par timestep |
| `reconstruction_comparison.png` | Comparaison visuelle |
| `memory_ablation.png` | Ablation taille mémoire |

## Résultats Clés

| Métrique | MA-TAP | Baseline | Amélioration |
|----------|--------|----------|--------------|
| Val Loss | 0.1654 | 0.1657 | +0.19% |
| Val SSIM | 0.1514 | 0.1506 | +0.57% |
| Stabilité (variance) | 0.18 | 0.51 | 3× meilleur |
| Epochs gagnés (Loss) | 45/50 | 5/50 | 90% |

## Auteur

**Donfack Pascal Arthur**  
Master 2 GI - ENSPY, Université de Yaoundé I  
Superviseur: Dr. Louis Fippo Fitime
