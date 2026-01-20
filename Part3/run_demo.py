"""
Script de Démonstration Rapide - MA-TAP
=======================================

Script simple pour tester rapidement que tout fonctionne.

Usage:
------
    python run_demo.py
"""

import os
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import tensorflow as tf

print("=" * 60)
print("MA-TAP Research - Demo Rapide")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
print()

# ==============================================================================
# 1. TEST DES IMPORTS
# ==============================================================================
print("[1/5] Test des imports...")

from src.matap_cell import MATAPCell, VanillaGRUCell
from src.models import MATAPModel, BaselineTAPModel, count_parameters
from data.dataset import BouncingBallGenerator, MovingMNISTGenerator

print("  ✓ Tous les imports OK")

# ==============================================================================
# 2. GÉNÉRATION DE DONNÉES
# ==============================================================================
print("\n[2/5] Génération de données...")

# Utiliser BouncingBall pour la démo (plus rapide)
data_gen = BouncingBallGenerator(seq_length=20, num_balls=2)
train_data = data_gen.generate_batch(32)
test_data = data_gen.generate_batch(8)

print(f"  ✓ Train data shape: {train_data.shape}")
print(f"  ✓ Test data shape: {test_data.shape}")

# ==============================================================================
# 3. CRÉATION DES MODÈLES
# ==============================================================================
print("\n[3/5] Création des modèles...")

matap_model = MATAPModel(latent_dim=64, memory_size=10, num_heads=4)
baseline_model = BaselineTAPModel(latent_dim=64)

# Build les modèles
dummy_input = tf.zeros((1, 20, 64, 64, 1))
_ = matap_model(dummy_input)
_ = baseline_model(dummy_input)

print(f"  ✓ MA-TAP Model: {count_parameters(matap_model):,} paramètres")
print(f"  ✓ Baseline Model: {count_parameters(baseline_model):,} paramètres")

# ==============================================================================
# 4. ENTRAÎNEMENT RAPIDE (3 epochs)
# ==============================================================================
print("\n[4/5] Entraînement rapide (3 epochs)...")

optimizer = tf.keras.optimizers.Adam(1e-3)

for epoch in range(3):
    with tf.GradientTape() as tape:
        reconstructed, z_true, z_pred, _ = matap_model(train_data, training=True)
        
        # Loss
        loss_rec = tf.reduce_mean(tf.keras.losses.binary_crossentropy(train_data, reconstructed))
        loss_latent = tf.reduce_mean(tf.square(z_true[:, 1:] - z_pred[:, :-1]))
        total_loss = loss_rec + 0.1 * loss_latent
    
    gradients = tape.gradient(total_loss, matap_model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, matap_model.trainable_variables))
    
    print(f"  Epoch {epoch+1}: Loss = {total_loss.numpy():.4f}")

print("  ✓ Entraînement OK")

# ==============================================================================
# 5. TEST DE GÉNÉRATION
# ==============================================================================
print("\n[5/5] Test de génération autorégressive...")

context = test_data[:, :10]  # 10 frames de contexte
future = matap_model.generate_future(context, num_future=10, training=False)

print(f"  ✓ Context shape: {context.shape}")
print(f"  ✓ Generated future shape: {future.shape}")

# Calcul SSIM sur les frames générées
ground_truth = test_data[:, 10:20]
ssim_values = []
for t in range(10):
    ssim = tf.image.ssim(ground_truth[:, t], future[:, t], max_val=1.0)
    ssim_values.append(tf.reduce_mean(ssim).numpy())

print(f"  ✓ SSIM moyen sur frames générées: {np.mean(ssim_values):.4f}")

# ==============================================================================
# RÉSUMÉ
# ==============================================================================
print("\n" + "=" * 60)
print("DEMO TERMINÉE AVEC SUCCÈS!")
print("=" * 60)
print("""
Prochaines étapes:
1. Entraîner le modèle complet:
   python experiments/train.py --model matap --epochs 50

2. Comparer avec baseline:
   python experiments/train.py --model baseline --epochs 50

3. Évaluer et générer les figures:
   python experiments/evaluate.py --compare_models --run_ablation

4. Pour utiliser Moving MNIST (plus lent mais réaliste):
   python experiments/train.py --model matap --epochs 50
   (Moving MNIST est utilisé par défaut)

5. Pour debug rapide avec BouncingBall:
   python experiments/train.py --model matap --epochs 20 --use_bouncing_ball
""")
