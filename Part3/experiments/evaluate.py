"""
Script d'Évaluation et Étude d'Ablation - MA-TAP Research
=========================================================

Compare les performances de MA-TAP vs Baseline sur différentes
longueurs de séquences et génère les courbes d'ablation.

Auteur: DONFACK Pascal
Date: Janvier 2026

Usage:
------
    python evaluate.py --run_ablation
    python evaluate.py --compare_models
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import mlflow

# Ajout du path parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MATAPModel, BaselineTAPModel, count_parameters
from data.dataset import MovingMNISTGenerator, BouncingBallGenerator, split_context_target


# ==============================================================================
# MÉTRIQUES D'ÉVALUATION
# ==============================================================================

def compute_ssim_over_time(y_true, y_pred):
    """
    Calcule le SSIM frame par frame pour analyser la dégradation temporelle.
    
    Args:
        y_true: (B, T, H, W, C)
        y_pred: (B, T, H, W, C)
    
    Returns:
        ssim_per_timestep: Liste de SSIM moyens pour chaque timestep
    """
    B, T = y_true.shape[0], y_true.shape[1]
    ssim_per_timestep = []
    
    for t in range(T):
        ssim = tf.image.ssim(y_true[:, t], y_pred[:, t], max_val=1.0)
        ssim_per_timestep.append(tf.reduce_mean(ssim).numpy())
    
    return ssim_per_timestep


def compute_mse_over_time(y_true, y_pred):
    """Calcule le MSE frame par frame."""
    T = y_true.shape[1]
    mse_per_timestep = []
    
    for t in range(T):
        mse = tf.reduce_mean(tf.square(y_true[:, t] - y_pred[:, t]))
        mse_per_timestep.append(mse.numpy())
    
    return mse_per_timestep


def compute_psnr_over_time(y_true, y_pred):
    """Calcule le PSNR frame par frame."""
    T = y_true.shape[1]
    psnr_per_timestep = []
    
    for t in range(T):
        psnr = tf.image.psnr(y_true[:, t], y_pred[:, t], max_val=1.0)
        psnr_per_timestep.append(tf.reduce_mean(psnr).numpy())
    
    return psnr_per_timestep


def compute_latent_drift(model, sequences, context_len=10):
    """
    Mesure la dérive latente au cours de la génération.
    
    Quantifie à quel point z_t s'éloigne de z_0 au fil du temps.
    """
    B, T = sequences.shape[0], sequences.shape[1]
    
    # Encodage de la séquence complète
    z_true = model.encode_sequence(sequences, training=False)
    
    # Génération autorégressive à partir du contexte
    context = sequences[:, :context_len]
    future = model.generate_future(context, num_future=T-context_len, training=False)
    
    # Encodage des frames générées
    z_generated = model.encode_sequence(future, training=False)
    
    # Distance par rapport au premier état
    z_0 = z_true[:, context_len:context_len+1, :]  # Premier état après contexte
    
    drift_true = []
    drift_generated = []
    
    for t in range(T - context_len):
        # Drift vrai
        dist_true = tf.reduce_mean(tf.norm(z_true[:, context_len+t:context_len+t+1] - z_0, axis=-1))
        drift_true.append(dist_true.numpy())
        
        # Drift généré
        if t < z_generated.shape[1]:
            dist_gen = tf.reduce_mean(tf.norm(z_generated[:, t:t+1] - z_0, axis=-1))
            drift_generated.append(dist_gen.numpy())
    
    return drift_true, drift_generated


# ==============================================================================
# ÉTUDE D'ABLATION
# ==============================================================================

class AblationStudy:
    """Classe pour mener l'étude d'ablation."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
        # Générateur de données
        if config.get('use_moving_mnist', True):
            self.data_gen = MovingMNISTGenerator(
                seq_length=config['seq_length'],
                num_digits=config.get('num_digits', 2)
            )
        else:
            self.data_gen = BouncingBallGenerator(
                seq_length=config['seq_length'],
                num_balls=2
            )
    
    def run_comparison(self, matap_model, baseline_model, num_samples=100):
        """
        Compare MA-TAP et Baseline sur les mêmes données.
        """
        print("\n=== Comparaison MA-TAP vs Baseline ===")
        
        # Génération des données de test
        test_data = self.data_gen.generate_batch(num_samples)
        context_len = self.config.get('context_length', 10)
        
        results = {
            'matap': {'ssim': [], 'mse': [], 'psnr': []},
            'baseline': {'ssim': [], 'mse': [], 'psnr': []}
        }
        
        # Évaluation MA-TAP
        print("Évaluation MA-TAP...")
        rec_matap, _, _, _ = matap_model(test_data, training=False)
        results['matap']['ssim'] = compute_ssim_over_time(test_data, rec_matap.numpy())
        results['matap']['mse'] = compute_mse_over_time(test_data, rec_matap.numpy())
        results['matap']['psnr'] = compute_psnr_over_time(test_data, rec_matap.numpy())
        
        # Évaluation Baseline
        print("Évaluation Baseline...")
        rec_baseline, _, _, _ = baseline_model(test_data, training=False)
        results['baseline']['ssim'] = compute_ssim_over_time(test_data, rec_baseline.numpy())
        results['baseline']['mse'] = compute_mse_over_time(test_data, rec_baseline.numpy())
        results['baseline']['psnr'] = compute_psnr_over_time(test_data, rec_baseline.numpy())
        
        self.results['comparison'] = results
        return results
    
    def run_memory_size_ablation(self, memory_sizes=[5, 10, 15, 20], num_samples=100):
        """
        Étudie l'impact de la taille de la mémoire sur les performances.
        """
        print("\n=== Ablation: Taille de la Mémoire ===")
        
        test_data = self.data_gen.generate_batch(num_samples)
        results = {}
        
        for mem_size in memory_sizes:
            print(f"\nMemory Size = {mem_size}")
            
            # Création du modèle avec cette taille de mémoire
            model = MATAPModel(
                latent_dim=self.config['latent_dim'],
                memory_size=mem_size,
                num_heads=self.config.get('num_heads', 4)
            )
            
            # Build
            dummy = tf.zeros((1, self.config['seq_length'], 64, 64, 1))
            _ = model(dummy)
            
            # Note: En production, il faudrait entraîner chaque modèle
            # Ici on évalue sur un modèle non entraîné pour la démo
            rec, _, _, _ = model(test_data, training=False)
            
            ssim_scores = compute_ssim_over_time(test_data, rec.numpy())
            results[mem_size] = {
                'ssim': ssim_scores,
                'mean_ssim': np.mean(ssim_scores),
                'final_ssim': ssim_scores[-1] if ssim_scores else 0
            }
            
            print(f"  Mean SSIM: {results[mem_size]['mean_ssim']:.4f}")
        
        self.results['memory_ablation'] = results
        return results
    
    def run_sequence_length_study(self, seq_lengths=[10, 20, 30, 40], num_samples=50):
        """
        Étudie la robustesse aux longues séquences.
        """
        print("\n=== Étude: Robustesse aux Longues Séquences ===")
        
        results = {'matap': {}, 'baseline': {}}
        
        for seq_len in seq_lengths:
            print(f"\nSequence Length = {seq_len}")
            
            # Générateur avec cette longueur
            if self.config.get('use_moving_mnist', True):
                gen = MovingMNISTGenerator(seq_length=seq_len, num_digits=2)
            else:
                gen = BouncingBallGenerator(seq_length=seq_len, num_balls=2)
            
            test_data = gen.generate_batch(num_samples)
            
            # Modèles
            matap = MATAPModel(latent_dim=64, memory_size=10)
            baseline = BaselineTAPModel(latent_dim=64)
            
            # Build
            dummy = tf.zeros((1, seq_len, 64, 64, 1))
            _ = matap(dummy)
            _ = baseline(dummy)
            
            # Évaluation
            rec_m, _, _, _ = matap(test_data, training=False)
            rec_b, _, _, _ = baseline(test_data, training=False)
            
            ssim_m = compute_ssim_over_time(test_data, rec_m.numpy())
            ssim_b = compute_ssim_over_time(test_data, rec_b.numpy())
            
            results['matap'][seq_len] = ssim_m
            results['baseline'][seq_len] = ssim_b
            
            print(f"  MA-TAP Final SSIM: {ssim_m[-1]:.4f}")
            print(f"  Baseline Final SSIM: {ssim_b[-1]:.4f}")
        
        self.results['seq_length_study'] = results
        return results


# ==============================================================================
# VISUALISATION
# ==============================================================================

def plot_ssim_comparison(results, save_path=None):
    """Trace les courbes SSIM MA-TAP vs Baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    timesteps = range(len(results['matap']['ssim']))
    
    ax.plot(timesteps, results['matap']['ssim'], 'b-', linewidth=2, 
            label='MA-TAP (Ours)', marker='o', markersize=4)
    ax.plot(timesteps, results['baseline']['ssim'], 'r--', linewidth=2, 
            label='Baseline (GRU)', marker='s', markersize=4)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('Temporal Coherence: SSIM over Time', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    plt.show()
    return fig


def plot_memory_ablation(results, save_path=None):
    """Trace l'impact de la taille de mémoire."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Courbes SSIM temporelles
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for (mem_size, data), color in zip(results.items(), colors):
        timesteps = range(len(data['ssim']))
        ax1.plot(timesteps, data['ssim'], color=color, linewidth=2, 
                label=f'K={mem_size}')
    
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('SSIM', fontsize=12)
    ax1.set_title('SSIM over Time (varying Memory Size K)', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Barplot du SSIM final
    mem_sizes = list(results.keys())
    final_ssims = [results[k]['final_ssim'] for k in mem_sizes]
    
    bars = ax2.bar(range(len(mem_sizes)), final_ssims, color='steelblue')
    ax2.set_xticks(range(len(mem_sizes)))
    ax2.set_xticklabels([f'K={k}' for k in mem_sizes])
    ax2.set_ylabel('Final SSIM', fontsize=12)
    ax2.set_title('Final Frame SSIM vs Memory Size', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    plt.show()
    return fig


def plot_generated_sequences(model, data_gen, num_sequences=3, context_len=10, 
                             num_future=10, save_path=None):
    """Visualise des séquences générées."""
    sequences = data_gen.generate_batch(num_sequences)
    context = sequences[:, :context_len]
    ground_truth = sequences[:, context_len:context_len+num_future]
    
    # Génération
    generated = model.generate_future(context, num_future=num_future, training=False)
    
    # Visualisation
    fig, axes = plt.subplots(num_sequences * 2, num_future + context_len, 
                             figsize=(2 * (num_future + context_len), 4 * num_sequences))
    
    for seq_idx in range(num_sequences):
        # Ligne 1: Ground Truth
        for t in range(context_len + num_future):
            ax = axes[seq_idx * 2, t]
            if t < context_len:
                ax.imshow(context[seq_idx, t, :, :, 0], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Ctx {t}', fontsize=8)
            else:
                ax.imshow(ground_truth[seq_idx, t-context_len, :, :, 0], cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'GT {t}', fontsize=8)
            ax.axis('off')
        
        # Ligne 2: Generated
        for t in range(context_len + num_future):
            ax = axes[seq_idx * 2 + 1, t]
            if t < context_len:
                ax.imshow(context[seq_idx, t, :, :, 0], cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(generated[seq_idx, t-context_len, :, :, 0].numpy(), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'Gen {t}', fontsize=8)
            ax.axis('off')
    
    plt.suptitle('Top: Ground Truth | Bottom: Generated', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")
    
    plt.show()
    return fig


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def main(args):
    """Point d'entrée principal."""
    
    config = {
        'latent_dim': 64,
        'memory_size': 10,
        'num_heads': 4,
        'seq_length': 20,
        'context_length': 10,
        'use_moving_mnist': not args.use_bouncing_ball,
        'num_digits': 2
    }
    
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("MA-TAP Research - Évaluation et Ablation")
    print("=" * 60)
    
    # Chemins vers les poids entraînés
    matap_weights = output_dir / 'matap_best.weights.h5'
    baseline_weights = output_dir / 'baseline_best.weights.h5'
    
    if args.run_ablation or args.compare_models:
        ablation = AblationStudy(config)
        
        if args.compare_models:
            print("\n>>> Comparaison des modèles...")
            
            # Création des modèles
            matap = MATAPModel(latent_dim=64, memory_size=10)
            baseline = BaselineTAPModel(latent_dim=64)
            
            # Build
            dummy = tf.zeros((1, config['seq_length'], 64, 64, 1))
            _ = matap(dummy)
            _ = baseline(dummy)
            
            # Chargement des poids entraînés
            if matap_weights.exists():
                matap.load_weights(str(matap_weights))
                print(f"✓ Poids MA-TAP chargés depuis: {matap_weights}")
            else:
                print(f"⚠ Poids MA-TAP non trouvés: {matap_weights}")
            
            if baseline_weights.exists():
                baseline.load_weights(str(baseline_weights))
                print(f"✓ Poids Baseline chargés depuis: {baseline_weights}")
            else:
                print(f"⚠ Poids Baseline non trouvés: {baseline_weights}")
            
            print(f"MA-TAP params: {count_parameters(matap):,}")
            print(f"Baseline params: {count_parameters(baseline):,}")
            
            # Comparaison
            results = ablation.run_comparison(matap, baseline, num_samples=args.num_samples)
            
            # Visualisation
            plot_ssim_comparison(results, save_path=str(output_dir / 'ssim_comparison.png'))
        
        if args.run_ablation:
            print("\n>>> Étude d'ablation...")
            
            # Ablation taille mémoire
            mem_results = ablation.run_memory_size_ablation(
                memory_sizes=[5, 10, 15, 20],
                num_samples=args.num_samples
            )
            plot_memory_ablation(mem_results, save_path=str(output_dir / 'memory_ablation.png'))
            
            # Étude longueur séquence
            seq_results = ablation.run_sequence_length_study(
                seq_lengths=[10, 20, 30],
                num_samples=args.num_samples // 2
            )
    
    if args.visualize:
        print("\n>>> Visualisation des générations...")
        
        if config['use_moving_mnist']:
            data_gen = MovingMNISTGenerator(seq_length=20, num_digits=2)
        else:
            data_gen = BouncingBallGenerator(seq_length=20, num_balls=2)
        
        model = MATAPModel(latent_dim=64, memory_size=10)
        dummy = tf.zeros((1, 20, 64, 64, 1))
        _ = model(dummy)
        
        plot_generated_sequences(
            model, data_gen, 
            num_sequences=3, context_len=10, num_future=10,
            save_path=str(output_dir / 'generated_sequences.png')
        )
    
    print("\n" + "=" * 60)
    print("Évaluation terminée!")
    print(f"Résultats sauvegardés dans: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Ablate MA-TAP models")
    parser.add_argument('--run_ablation', action='store_true',
                        help='Run ablation study on memory size')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare MA-TAP vs Baseline')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize generated sequences')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of test samples')
    parser.add_argument('--use_bouncing_ball', action='store_true',
                        help='Use BouncingBall dataset (faster)')
    
    args = parser.parse_args()
    
    # Si aucun argument, tout exécuter
    if not (args.run_ablation or args.compare_models or args.visualize):
        args.compare_models = True
        args.run_ablation = True
        args.visualize = True
    
    main(args)
