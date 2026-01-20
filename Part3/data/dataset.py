"""
Dataset Moving MNIST et Générateurs de Données
==============================================

Contient:
- MovingMNISTGenerator: Générateur de séquences Moving MNIST
- BouncingBallGenerator: Générateur simplifié (debug/tests rapides)
- Utilitaires de chargement et prétraitement

Auteur: DONFACK Pascal
Date: Janvier 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path


class MovingMNISTGenerator:
    """
    Générateur de séquences Moving MNIST.
    
    Crée des séquences vidéo de chiffres MNIST en mouvement avec rebonds
    sur les bords de l'image. Supporte plusieurs chiffres par séquence.
    
    Paramètres:
    -----------
    image_size : int
        Taille de l'image carrée (défaut: 64x64)
    digit_size : int
        Taille des chiffres MNIST redimensionnés (défaut: 28)
    num_digits : int
        Nombre de chiffres par séquence (défaut: 2)
    seq_length : int
        Longueur des séquences générées (défaut: 20)
    deterministic : bool
        Si True, utilise une seed fixe pour reproductibilité
    """
    
    def __init__(self, image_size=64, digit_size=28, num_digits=2, 
                 seq_length=20, deterministic=False, seed=42):
        self.image_size = image_size
        self.digit_size = digit_size
        self.num_digits = num_digits
        self.seq_length = seq_length
        self.deterministic = deterministic
        self.seed = seed
        
        # Charger MNIST
        self._load_mnist()
        
        if deterministic:
            np.random.seed(seed)
    
    def _load_mnist(self):
        """Charge et prétraite les données MNIST."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalisation [0, 1]
        self.mnist_train = x_train.astype(np.float32) / 255.0
        self.mnist_test = x_test.astype(np.float32) / 255.0
        self.train_labels = y_train
        self.test_labels = y_test
        
        print(f"[MovingMNIST] Loaded {len(self.mnist_train)} train, {len(self.mnist_test)} test digits")
    
    def _get_random_digit(self, use_test=False):
        """Retourne un chiffre MNIST aléatoire."""
        data = self.mnist_test if use_test else self.mnist_train
        idx = np.random.randint(len(data))
        digit = data[idx]
        
        # Redimensionnement si nécessaire
        if self.digit_size != 28:
            digit = tf.image.resize(
                digit[..., np.newaxis], 
                (self.digit_size, self.digit_size)
            ).numpy().squeeze()
        
        return digit
    
    def _generate_trajectory(self, seq_length):
        """Génère une trajectoire avec rebonds."""
        # Position initiale
        x = np.random.randint(0, self.image_size - self.digit_size)
        y = np.random.randint(0, self.image_size - self.digit_size)
        
        # Vitesse (pixels par frame)
        speed = np.random.uniform(2, 5)
        angle = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        positions = []
        for _ in range(seq_length):
            positions.append((int(x), int(y)))
            
            # Mise à jour avec rebonds
            x += vx
            y += vy
            
            # Rebond horizontal
            if x < 0:
                x = -x
                vx = -vx
            elif x > self.image_size - self.digit_size:
                x = 2 * (self.image_size - self.digit_size) - x
                vx = -vx
            
            # Rebond vertical
            if y < 0:
                y = -y
                vy = -vy
            elif y > self.image_size - self.digit_size:
                y = 2 * (self.image_size - self.digit_size) - y
                vy = -vy
        
        return positions
    
    def generate_sequence(self, use_test=False):
        """
        Génère une séquence Moving MNIST.
        
        Returns:
            sequence: np.array de shape (seq_length, image_size, image_size, 1)
        """
        sequence = np.zeros((self.seq_length, self.image_size, self.image_size, 1), 
                           dtype=np.float32)
        
        for _ in range(self.num_digits):
            digit = self._get_random_digit(use_test=use_test)
            trajectory = self._generate_trajectory(self.seq_length)
            
            for t, (x, y) in enumerate(trajectory):
                # Superposition additive avec clipping
                x_end = min(x + self.digit_size, self.image_size)
                y_end = min(y + self.digit_size, self.image_size)
                d_x_end = x_end - x
                d_y_end = y_end - y
                
                sequence[t, y:y_end, x:x_end, 0] = np.clip(
                    sequence[t, y:y_end, x:x_end, 0] + digit[:d_y_end, :d_x_end],
                    0, 1
                )
        
        return sequence
    
    def generate_batch(self, batch_size, use_test=False):
        """
        Génère un batch de séquences.
        
        Returns:
            batch: np.array de shape (batch_size, seq_length, image_size, image_size, 1)
        """
        batch = np.zeros((batch_size, self.seq_length, self.image_size, self.image_size, 1),
                        dtype=np.float32)
        
        for i in range(batch_size):
            batch[i] = self.generate_sequence(use_test=use_test)
        
        return batch
    
    def create_tf_dataset(self, num_samples, batch_size=32, use_test=False, shuffle=True):
        """
        Crée un tf.data.Dataset pour l'entraînement.
        
        Returns:
            tf.data.Dataset avec batches de séquences
        """
        def generator():
            for _ in range(num_samples):
                yield self.generate_sequence(use_test=use_test)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(
                shape=(self.seq_length, self.image_size, self.image_size, 1),
                dtype=tf.float32
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, num_samples))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


class BouncingBallGenerator:
    """
    Générateur de données simplifié avec balles rebondissantes.
    
    Plus rapide que Moving MNIST, utile pour debug et tests rapides.
    """
    
    def __init__(self, image_size=64, ball_radius=3, num_balls=1, 
                 seq_length=20, seed=None):
        self.image_size = image_size
        self.ball_radius = ball_radius
        self.num_balls = num_balls
        self.seq_length = seq_length
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_sequence(self):
        """Génère une séquence de balles rebondissantes."""
        sequence = np.zeros((self.seq_length, self.image_size, self.image_size, 1),
                           dtype=np.float32)
        
        for _ in range(self.num_balls):
            # Position et vitesse initiales
            x = np.random.randint(self.ball_radius + 5, self.image_size - self.ball_radius - 5)
            y = np.random.randint(self.ball_radius + 5, self.image_size - self.ball_radius - 5)
            vx = np.random.choice([-3, -2, 2, 3])
            vy = np.random.choice([-3, -2, 2, 3])
            
            for t in range(self.seq_length):
                # Dessiner la balle
                yy, xx = np.ogrid[:self.image_size, :self.image_size]
                mask = ((xx - x)**2 + (yy - y)**2) <= self.ball_radius**2
                sequence[t, :, :, 0] = np.clip(
                    sequence[t, :, :, 0] + mask.astype(np.float32),
                    0, 1
                )
                
                # Mise à jour avec rebonds
                x += vx
                y += vy
                
                if x <= self.ball_radius or x >= self.image_size - self.ball_radius:
                    vx = -vx
                    x = np.clip(x, self.ball_radius, self.image_size - self.ball_radius)
                if y <= self.ball_radius or y >= self.image_size - self.ball_radius:
                    vy = -vy
                    y = np.clip(y, self.ball_radius, self.image_size - self.ball_radius)
        
        return sequence
    
    def generate_batch(self, batch_size):
        """Génère un batch de séquences."""
        batch = np.zeros((batch_size, self.seq_length, self.image_size, self.image_size, 1),
                        dtype=np.float32)
        for i in range(batch_size):
            batch[i] = self.generate_sequence()
        return batch
    
    def create_tf_dataset(self, num_samples, batch_size=32, shuffle=True):
        """Crée un tf.data.Dataset."""
        def generator():
            for _ in range(num_samples):
                yield self.generate_sequence()
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(
                shape=(self.seq_length, self.image_size, self.image_size, 1),
                dtype=tf.float32
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, num_samples))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


# ==============================================================================
# UTILITAIRES DE PRÉTRAITEMENT
# ==============================================================================

def split_context_target(sequence, context_len=10):
    """
    Sépare une séquence en contexte et cible.
    
    Args:
        sequence: Tensor de shape (B, T, H, W, C) ou (T, H, W, C)
        context_len: Nombre de frames de contexte
        
    Returns:
        context: Frames de contexte
        target: Frames cibles à prédire
    """
    if len(sequence.shape) == 5:  # Batched
        context = sequence[:, :context_len]
        target = sequence[:, context_len:]
    else:  # Single sequence
        context = sequence[:context_len]
        target = sequence[context_len:]
    
    return context, target


def create_shifted_pairs(z_seq):
    """
    Crée des paires (z_t, z_{t+1}) pour l'entraînement de la dynamique.
    
    Args:
        z_seq: Séquence latente (B, T, D)
        
    Returns:
        z_input: (B, T-1, D) - États d'entrée
        z_target: (B, T-1, D) - États cibles
    """
    z_input = z_seq[:, :-1, :]
    z_target = z_seq[:, 1:, :]
    return z_input, z_target


# ==============================================================================
# TEST RAPIDE
# ==============================================================================

if __name__ == "__main__":
    print("=== Test des Générateurs de Données ===\n")
    
    # Test BouncingBall (rapide)
    print("1. Test BouncingBallGenerator...")
    bb_gen = BouncingBallGenerator(seq_length=20, num_balls=2)
    bb_batch = bb_gen.generate_batch(4)
    print(f"   Shape: {bb_batch.shape}")
    print(f"   Range: [{bb_batch.min():.2f}, {bb_batch.max():.2f}]")
    
    # Test MovingMNIST
    print("\n2. Test MovingMNISTGenerator...")
    mnist_gen = MovingMNISTGenerator(seq_length=20, num_digits=2)
    mnist_batch = mnist_gen.generate_batch(4)
    print(f"   Shape: {mnist_batch.shape}")
    print(f"   Range: [{mnist_batch.min():.2f}, {mnist_batch.max():.2f}]")
    
    # Test tf.data.Dataset
    print("\n3. Test tf.data.Dataset...")
    dataset = bb_gen.create_tf_dataset(num_samples=100, batch_size=16)
    for batch in dataset.take(1):
        print(f"   Batch shape: {batch.shape}")
    
    print("\n=== Tests terminés avec succès ===")
