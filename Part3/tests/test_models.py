"""
Tests Unitaires - MA-TAP Research
=================================

Tests pour valider le bon fonctionnement des composants.

Usage:
------
    python -m pytest tests/test_models.py -v
    python tests/test_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
import keras

# Imports locaux
from src.matap_cell import MATAPCell, VanillaGRUCell
from src.models import MATAPModel, BaselineTAPModel, SpatialEncoder, SpatialDecoder
from data.dataset import MovingMNISTGenerator, BouncingBallGenerator


def test_matap_cell():
    """Test de la cellule MA-TAP."""
    print("\n[TEST] MATAPCell...")
    
    latent_dim = 64
    memory_size = 10
    batch_size = 4
    
    cell = MATAPCell(latent_dim=latent_dim, memory_size=memory_size, num_heads=4)
    
    # Test des dimensions d'entrée/sortie
    inputs = tf.random.normal((batch_size, latent_dim))
    initial_states = cell.get_initial_state(batch_size=batch_size)
    
    assert len(initial_states) == 2, "Doit retourner 2 états (h, memory)"
    assert initial_states[0].shape == (batch_size, latent_dim), f"h shape incorrect: {initial_states[0].shape}"
    assert initial_states[1].shape == (batch_size, memory_size * latent_dim), f"memory shape incorrect"
    
    # Test forward pass
    output, new_states = cell(inputs, initial_states)
    
    assert output.shape == (batch_size, latent_dim), f"Output shape incorrect: {output.shape}"
    assert len(new_states) == 2, "Doit retourner 2 nouveaux états"
    
    print("  ✓ Dimensions correctes")
    print("  ✓ Forward pass OK")
    print("[TEST] MATAPCell: PASSED ✓")


def test_vanilla_gru_cell():
    """Test de la cellule GRU vanilla (baseline)."""
    print("\n[TEST] VanillaGRUCell...")
    
    latent_dim = 64
    batch_size = 4
    
    cell = VanillaGRUCell(latent_dim=latent_dim)
    
    inputs = tf.random.normal((batch_size, latent_dim))
    initial_states = cell.get_initial_state(batch_size=batch_size)
    
    assert len(initial_states) == 1, "Doit retourner 1 état"
    
    output, new_states = cell(inputs, initial_states)
    
    assert output.shape == (batch_size, latent_dim), f"Output shape incorrect: {output.shape}"
    
    print("  ✓ Forward pass OK")
    print("[TEST] VanillaGRUCell: PASSED ✓")


def test_spatial_encoder():
    """Test de l'encodeur spatial."""
    print("\n[TEST] SpatialEncoder...")
    
    latent_dim = 64
    batch_size = 4
    
    encoder = SpatialEncoder(latent_dim=latent_dim)
    
    # Input: image 64x64
    x = tf.random.uniform((batch_size, 64, 64, 1), 0, 1)
    z = encoder(x, training=False)
    
    assert z.shape == (batch_size, latent_dim), f"Latent shape incorrect: {z.shape}"
    
    print("  ✓ Encode 64x64 -> latent_dim")
    print("[TEST] SpatialEncoder: PASSED ✓")


def test_spatial_decoder():
    """Test du décodeur spatial."""
    print("\n[TEST] SpatialDecoder...")
    
    latent_dim = 64
    batch_size = 4
    
    decoder = SpatialDecoder(latent_dim=latent_dim)
    
    z = tf.random.normal((batch_size, latent_dim))
    x_rec = decoder(z, training=False)
    
    assert x_rec.shape == (batch_size, 64, 64, 1), f"Reconstructed shape incorrect: {x_rec.shape}"
    assert tf.reduce_min(x_rec) >= 0 and tf.reduce_max(x_rec) <= 1, "Output doit être dans [0, 1]"
    
    print("  ✓ Decode latent_dim -> 64x64")
    print("  ✓ Output dans [0, 1]")
    print("[TEST] SpatialDecoder: PASSED ✓")


def test_matap_model():
    """Test du modèle MA-TAP complet."""
    print("\n[TEST] MATAPModel...")
    
    latent_dim = 64
    memory_size = 10
    batch_size = 4
    seq_length = 20
    
    model = MATAPModel(latent_dim=latent_dim, memory_size=memory_size)
    
    # Input: séquence vidéo
    x = tf.random.uniform((batch_size, seq_length, 64, 64, 1), 0, 1)
    
    reconstructed, z_true, z_pred, final_states = model(x, training=False)
    
    assert reconstructed.shape == x.shape, f"Reconstructed shape incorrect: {reconstructed.shape}"
    assert z_true.shape == (batch_size, seq_length, latent_dim), f"z_true shape incorrect"
    assert z_pred.shape == (batch_size, seq_length, latent_dim), f"z_pred shape incorrect"
    assert len(final_states) == 2, "Doit retourner 2 états finaux"
    
    print("  ✓ Forward pass complet")
    print("  ✓ Reconstruction shape OK")
    print("  ✓ Latent sequences shape OK")
    
    # Test génération autorégressive
    context = x[:, :10]
    future = model.generate_future(context, num_future=10, training=False)
    
    assert future.shape == (batch_size, 10, 64, 64, 1), f"Future shape incorrect: {future.shape}"
    
    print("  ✓ Génération autorégressive OK")
    print("[TEST] MATAPModel: PASSED ✓")


def test_baseline_model():
    """Test du modèle Baseline."""
    print("\n[TEST] BaselineTAPModel...")
    
    latent_dim = 64
    batch_size = 4
    seq_length = 20
    
    model = BaselineTAPModel(latent_dim=latent_dim)
    
    x = tf.random.uniform((batch_size, seq_length, 64, 64, 1), 0, 1)
    
    reconstructed, z_true, z_pred, final_states = model(x, training=False)
    
    assert reconstructed.shape == x.shape
    assert len(final_states) == 1, "Baseline doit retourner 1 état final"
    
    # Test génération
    context = x[:, :10]
    future = model.generate_future(context, num_future=10, training=False)
    
    assert future.shape == (batch_size, 10, 64, 64, 1)
    
    print("  ✓ Forward pass OK")
    print("  ✓ Génération OK")
    print("[TEST] BaselineTAPModel: PASSED ✓")


def test_bouncing_ball_generator():
    """Test du générateur Bouncing Ball."""
    print("\n[TEST] BouncingBallGenerator...")
    
    gen = BouncingBallGenerator(seq_length=20, num_balls=2)
    
    # Test single sequence
    seq = gen.generate_sequence()
    assert seq.shape == (20, 64, 64, 1), f"Sequence shape incorrect: {seq.shape}"
    assert seq.min() >= 0 and seq.max() <= 1, "Values doivent être dans [0, 1]"
    
    # Test batch
    batch = gen.generate_batch(8)
    assert batch.shape == (8, 20, 64, 64, 1)
    
    print("  ✓ Génération single OK")
    print("  ✓ Génération batch OK")
    print("[TEST] BouncingBallGenerator: PASSED ✓")


def test_moving_mnist_generator():
    """Test du générateur Moving MNIST."""
    print("\n[TEST] MovingMNISTGenerator...")
    
    gen = MovingMNISTGenerator(seq_length=20, num_digits=2)
    
    # Test single sequence
    seq = gen.generate_sequence()
    assert seq.shape == (20, 64, 64, 1), f"Sequence shape incorrect: {seq.shape}"
    
    # Test batch
    batch = gen.generate_batch(4)
    assert batch.shape == (4, 20, 64, 64, 1)
    
    print("  ✓ Données MNIST chargées")
    print("  ✓ Génération OK")
    print("[TEST] MovingMNISTGenerator: PASSED ✓")


def test_training_step():
    """Test d'un pas d'entraînement."""
    print("\n[TEST] Training Step...")
    
    model = MATAPModel(latent_dim=64, memory_size=10)
    optimizer = keras.optimizers.Adam(1e-3)
    
    # Données
    x = tf.random.uniform((4, 20, 64, 64, 1), 0, 1)
    
    # Forward + Backward
    with tf.GradientTape() as tape:
        reconstructed, z_true, z_pred, _ = model(x, training=True)
        
        # Loss reconstruction
        loss_rec = tf.reduce_mean(keras.losses.binary_crossentropy(x, reconstructed))
        
        # Loss latente
        loss_latent = tf.reduce_mean(tf.square(z_true[:, 1:] - z_pred[:, :-1]))
        
        total_loss = loss_rec + 0.1 * loss_latent
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Vérifier que les gradients existent
    none_grads = sum(1 for g in gradients if g is None)
    assert none_grads == 0, f"{none_grads} gradients sont None!"
    
    # Appliquer les gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"  ✓ Loss calculée: {total_loss.numpy():.4f}")
    print(f"  ✓ Gradients OK ({len(gradients)} variables)")
    print("  ✓ Optimizer step OK")
    print("[TEST] Training Step: PASSED ✓")


def run_all_tests():
    """Exécute tous les tests."""
    print("=" * 60)
    print("MA-TAP Research - Suite de Tests")
    print("=" * 60)
    
    tests = [
        test_matap_cell,
        test_vanilla_gru_cell,
        test_spatial_encoder,
        test_spatial_decoder,
        test_matap_model,
        test_baseline_model,
        test_bouncing_ball_generator,
        test_moving_mnist_generator,
        test_training_step,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[TEST] {test_fn.__name__}: FAILED ✗")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Résultats: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
