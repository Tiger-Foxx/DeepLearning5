"""
Script d'Entraînement avec MLflow - MA-TAP Research
===================================================

Script complet pour entraîner les modèles MA-TAP et Baseline
avec suivi des métriques via MLflow.

Auteur: DONFACK Pascal
Date: Janvier 2026

Usage:
------
    python train.py --model matap --epochs 50 --memory_size 10
    python train.py --model baseline --epochs 50
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras

# Ajout du path parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MATAPModel, BaselineTAPModel, count_parameters
from data.dataset import MovingMNISTGenerator, BouncingBallGenerator, split_context_target


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG = {
    'latent_dim': 64,
    'memory_size': 10,
    'num_heads': 4,
    'seq_length': 20,
    'context_length': 10,
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 1e-3,
    'num_train_samples': 2000,
    'num_val_samples': 400,
    'use_moving_mnist': True,  # False = BouncingBall (plus rapide)
    'num_digits': 2,
    'dropout_rate': 0.1,
}


# ==============================================================================
# FONCTIONS DE LOSS
# ==============================================================================

def reconstruction_loss(y_true, y_pred):
    """Binary Cross-Entropy pour reconstruction d'images."""
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))


def latent_prediction_loss(z_true, z_pred):
    """MSE pour prédiction dans l'espace latent."""
    # On compare z_pred[t] avec z_true[t+1]
    z_true_shifted = z_true[:, 1:, :]
    z_pred_aligned = z_pred[:, :-1, :]
    return tf.reduce_mean(tf.square(z_true_shifted - z_pred_aligned))


def ssim_metric(y_true, y_pred):
    """SSIM moyen sur une séquence."""
    # Reshape pour traiter frame par frame
    B, T = tf.shape(y_true)[0], tf.shape(y_true)[1]
    y_true_flat = tf.reshape(y_true, (B * T, 64, 64, 1))
    y_pred_flat = tf.reshape(y_pred, (B * T, 64, 64, 1))
    
    ssim = tf.image.ssim(y_true_flat, y_pred_flat, max_val=1.0)
    return tf.reduce_mean(ssim)


# ==============================================================================
# CLASSE D'ENTRAÎNEMENT
# ==============================================================================

class Trainer:
    """Classe pour gérer l'entraînement et l'évaluation."""
    
    def __init__(self, model, config, experiment_name="MA-TAP-Research"):
        self.model = model
        self.config = config
        self.experiment_name = experiment_name
        
        # Optimiseur avec learning rate schedule
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config['learning_rate'],
            decay_steps=config['epochs'] * (config['num_train_samples'] // config['batch_size'])
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Métriques
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.val_loss = keras.metrics.Mean(name='val_loss')
        self.train_ssim = keras.metrics.Mean(name='train_ssim')
        self.val_ssim = keras.metrics.Mean(name='val_ssim')
        
        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ssim': [], 'val_ssim': [],
            'latent_loss': []
        }
    
    @tf.function
    def train_step(self, batch):
        """Un pas d'entraînement."""
        with tf.GradientTape() as tape:
            reconstructed, z_true, z_pred, _ = self.model(batch, training=True)
            
            # Loss combinée
            loss_rec = reconstruction_loss(batch, reconstructed)
            loss_latent = latent_prediction_loss(z_true, z_pred)
            total_loss = loss_rec + 0.1 * loss_latent
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Gradient clipping pour stabilité
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Métriques
        ssim = ssim_metric(batch, reconstructed)
        
        return total_loss, loss_latent, ssim
    
    @tf.function
    def val_step(self, batch):
        """Un pas de validation."""
        reconstructed, z_true, z_pred, _ = self.model(batch, training=False)
        
        loss_rec = reconstruction_loss(batch, reconstructed)
        loss_latent = latent_prediction_loss(z_true, z_pred)
        total_loss = loss_rec + 0.1 * loss_latent
        
        ssim = ssim_metric(batch, reconstructed)
        
        return total_loss, ssim
    
    def train(self, train_dataset, val_dataset, epochs=None):
        """Boucle d'entraînement complète."""
        if epochs is None:
            epochs = self.config['epochs']
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Reset métriques
            self.train_loss.reset_state()
            self.val_loss.reset_state()
            self.train_ssim.reset_state()
            self.val_ssim.reset_state()
            
            # Entraînement
            for batch in train_dataset:
                loss, lat_loss, ssim = self.train_step(batch)
                self.train_loss.update_state(loss)
                self.train_ssim.update_state(ssim)
            
            # Validation
            for batch in val_dataset:
                loss, ssim = self.val_step(batch)
                self.val_loss.update_state(loss)
                self.val_ssim.update_state(ssim)
            
            # Logging
            train_loss = self.train_loss.result().numpy()
            val_loss = self.val_loss.result().numpy()
            train_ssim = self.train_ssim.result().numpy()
            val_ssim = self.val_ssim.result().numpy()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ssim'].append(train_ssim)
            self.history['val_ssim'].append(val_ssim)
            
            # MLflow logging
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_ssim': train_ssim,
                'val_ssim': val_ssim
            }, step=epoch)
            
            # Affichage
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train SSIM: {train_ssim:.4f} | Val SSIM: {val_ssim:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarde du meilleur modèle
                output_path = Path(__file__).parent / 'outputs' / 'best_model.weights.h5'
                output_path.parent.mkdir(exist_ok=True)
                self.model.save_weights(str(output_path))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        return self.history


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main(args):
    """Point d'entrée principal."""
    
    # Mise à jour de la config avec les arguments
    config = DEFAULT_CONFIG.copy()
    config['model_type'] = args.model
    config['epochs'] = args.epochs
    config['memory_size'] = args.memory_size
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['use_moving_mnist'] = not args.use_bouncing_ball
    
    print("=" * 60)
    print("MA-TAP Research Training")
    print("=" * 60)
    print(f"Model: {args.model.upper()}")
    print(f"Epochs: {config['epochs']}")
    print(f"Memory Size: {config['memory_size']}")
    print(f"Dataset: {'Moving MNIST' if config['use_moving_mnist'] else 'Bouncing Ball'}")
    print("=" * 60)
    
    # Création des dossiers
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Générateur de données
    if config['use_moving_mnist']:
        data_gen = MovingMNISTGenerator(
            seq_length=config['seq_length'],
            num_digits=config['num_digits']
        )
    else:
        data_gen = BouncingBallGenerator(
            seq_length=config['seq_length'],
            num_balls=2
        )
    
    print("\nGénération des datasets...")
    train_dataset = data_gen.create_tf_dataset(
        num_samples=config['num_train_samples'],
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # Pour validation: use_test uniquement pour MovingMNIST
    if config['use_moving_mnist']:
        val_dataset = data_gen.create_tf_dataset(
            num_samples=config['num_val_samples'],
            batch_size=config['batch_size'],
            shuffle=False,
            use_test=True
        )
    else:
        val_dataset = data_gen.create_tf_dataset(
            num_samples=config['num_val_samples'],
            batch_size=config['batch_size'],
            shuffle=False
        )
    
    # Création du modèle
    print("\nCréation du modèle...")
    if args.model == 'matap':
        model = MATAPModel(
            latent_dim=config['latent_dim'],
            memory_size=config['memory_size'],
            num_heads=config['num_heads'],
            dropout_rate=config['dropout_rate']
        )
    else:
        model = BaselineTAPModel(
            latent_dim=config['latent_dim'],
            dropout_rate=config['dropout_rate']
        )
    
    # Build du modèle
    dummy_input = tf.zeros((1, config['seq_length'], 64, 64, 1))
    _ = model(dummy_input)
    
    num_params = count_parameters(model)
    print(f"Paramètres du modèle: {num_params:,}")
    
    # Configuration MLflow
    mlflow.set_experiment(f"Part3-{args.model.upper()}")
    
    with mlflow.start_run(run_name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log des paramètres
        mlflow.log_params({
            'model_type': args.model,
            'latent_dim': config['latent_dim'],
            'memory_size': config['memory_size'] if args.model == 'matap' else 0,
            'num_heads': config['num_heads'] if args.model == 'matap' else 0,
            'seq_length': config['seq_length'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'num_parameters': num_params,
            'dataset': 'moving_mnist' if config['use_moving_mnist'] else 'bouncing_ball'
        })
        
        # Entraînement
        trainer = Trainer(model, config)
        print("\nDémarrage de l'entraînement...")
        history = trainer.train(train_dataset, val_dataset)
        
        # Sauvegarde finale
        model.save_weights(str(output_dir / f'{args.model}_final.weights.h5'))
        mlflow.log_artifact(str(output_dir / f'{args.model}_final.weights.h5'))
        
        # Métriques finales
        final_metrics = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_ssim': history['train_ssim'][-1],
            'final_val_ssim': history['val_ssim'][-1],
            'best_val_loss': min(history['val_loss']),
            'best_val_ssim': max(history['val_ssim'])
        }
        mlflow.log_metrics(final_metrics)
        
        print("\n" + "=" * 60)
        print("Entraînement terminé!")
        print(f"Best Val Loss: {final_metrics['best_val_loss']:.4f}")
        print(f"Best Val SSIM: {final_metrics['best_val_ssim']:.4f}")
        print("=" * 60)
    
    return history


# ==============================================================================
# POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MA-TAP or Baseline model")
    parser.add_argument('--model', type=str, default='matap', 
                        choices=['matap', 'baseline'],
                        help='Model type: matap or baseline')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--memory_size', type=int, default=10,
                        help='Memory buffer size (MA-TAP only)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--use_bouncing_ball', action='store_true',
                        help='Use BouncingBall dataset instead of Moving MNIST')
    
    args = parser.parse_args()
    main(args)
