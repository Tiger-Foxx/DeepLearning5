"""
Memory-Augmented Time-Aware Path Cell (MA-TAP)
================================================

Cette cellule RNN personnalisée hybride une dynamique GRU locale avec une
mémoire épisodique différentiable pour combattre le "Latent Drift".

Auteur: DONFACK Pascal
Date: Janvier 2026
"""

import tensorflow as tf
import keras
from keras import layers


class MATAPCell(layers.Layer):
    """
    Memory-Augmented Time-Aware Path Cell.
    
    Cette cellule remplace la transition standard z_t -> z_{t+1}.
    Elle intègre :
    1. Une dynamique GRU locale pour le mouvement fluide (haute fréquence).
    2. Un mécanisme d'attention Multi-Head sur un buffer mémoire pour la cohérence globale.
    3. Une porte de fusion adaptative (Gating) qui dose l'intervention de la mémoire.
    
    Paramètres:
    -----------
    latent_dim : int
        Dimension de l'espace latent (utilisé pour tous les composants).
    memory_size : int
        Nombre de slots dans le buffer mémoire (K états passés).
    num_heads : int
        Nombre de têtes d'attention.
    dropout_rate : float
        Taux de dropout pour la régularisation.
    """
    
    def __init__(self, latent_dim, memory_size=10, num_heads=4, dropout_rate=0.1, **kwargs):
        super(MATAPCell, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # --- Composants Architecturaux ---
        
        # 1. Projection d'entrée pour uniformiser les dimensions
        self.input_proj = layers.Dense(latent_dim, name="input_projection")
        
        # 2. Dynamique Locale (GRU) - units = latent_dim pour cohérence
        self.gru = layers.GRUCell(latent_dim, name="gru_dynamics")
        
        # 3. Attention Rétrospective Multi-Head
        # Query = état courant, Key/Value = mémoire
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=latent_dim // num_heads,
            value_dim=latent_dim // num_heads,
            dropout=dropout_rate,
            name="retrospective_attention"
        )
        
        # 4. Normalisation et projection du contexte
        self.layer_norm_attn = layers.LayerNormalization(name="norm_attention")
        self.layer_norm_out = layers.LayerNormalization(name="norm_output")
        self.context_proj = layers.Dense(latent_dim, activation='tanh', name="context_projection")
        
        # 5. Gate de fusion adaptative
        # Entrée: concat(gru_out, context) -> sortie: alpha ∈ [0,1]^latent_dim
        self.gate_dense = layers.Dense(latent_dim, activation='sigmoid', name="fusion_gate")
        
        # 6. Projection pour l'écriture en mémoire
        self.memory_write_proj = layers.Dense(latent_dim, name="memory_write_proj")
        
        # --- Gestion de l'État Keras RNN ---
        # State structure: [h_t (latent_dim), memory_buffer_flat (memory_size * latent_dim)]
        self.state_size = [latent_dim, memory_size * latent_dim]
        self.output_size = latent_dim

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Initialise les états (h et mémoire) à zéro."""
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.float32
            
        init_h = tf.zeros((batch_size, self.latent_dim), dtype=dtype)
        init_memory = tf.zeros((batch_size, self.memory_size * self.latent_dim), dtype=dtype)
        return [init_h, init_memory]

    def call(self, inputs, states, training=None):
        """
        Forward pass de la cellule MA-TAP.
        
        Args:
            inputs: z_t de shape (batch, latent_dim) ou (batch, input_dim)
            states: [h_{t-1}, memory_{t-1}_flat]
            training: Mode entraînement pour le dropout
            
        Returns:
            output: h_t corrigé de shape (batch, latent_dim)
            new_states: [h_t, memory_t_flat]
        """
        h_prev, memory_flat = states
        batch_size = tf.shape(inputs)[0]
        
        # 0. Projection d'entrée pour garantir latent_dim
        z_t = self.input_proj(inputs)
        
        # 1. Reshape de la mémoire aplatie -> (Batch, K, latent_dim)
        memory = tf.reshape(memory_flat, (batch_size, self.memory_size, self.latent_dim))
        
        # 2. Dynamique Locale (Proposition de mouvement via GRU)
        gru_out, [h_candidate] = self.gru(z_t, [h_prev], training=training)
        
        # 3. Attention Rétrospective (Correction de Dérive)
        # Query: état courant (B, 1, D), Key/Value: mémoire (B, K, D)
        query = tf.expand_dims(gru_out, axis=1)  # (B, 1, latent_dim)
        
        # Calcul de l'attention avec masque optionnel pour les slots vides
        context = self.attention(
            query=query, 
            value=memory, 
            key=memory,
            training=training
        )  # (B, 1, latent_dim)
        
        context = tf.squeeze(context, axis=1)  # (B, latent_dim)
        context = self.layer_norm_attn(context)
        context_proj = self.context_proj(context)
        
        # 4. Fusion Adaptative (Gating)
        # alpha proche de 1 : forte confiance en la mémoire (ex: occlusion, dérive)
        # alpha proche de 0 : forte confiance en la dynamique GRU (ex: mouvement simple)
        gate_input = tf.concat([gru_out, context_proj], axis=-1)
        alpha = self.gate_dense(gate_input)  # (B, latent_dim)
        
        # Combinaison pondérée
        h_corrected = (1.0 - alpha) * gru_out + alpha * context_proj
        h_corrected = self.layer_norm_out(h_corrected)
        
        # 5. Mise à jour de la Mémoire (Rolling Buffer FIFO)
        # On ajoute le nouvel état à la fin et on supprime le plus ancien
        new_entry = self.memory_write_proj(z_t)  # (B, latent_dim)
        new_entry = tf.expand_dims(new_entry, axis=1)  # (B, 1, latent_dim)
        
        # FIFO: memory[:, 1:, :] + new_entry
        new_memory = tf.concat([memory[:, 1:, :], new_entry], axis=1)  # (B, K, latent_dim)
        
        # Aplatissement pour compatibilité Keras RNN
        new_memory_flat = tf.reshape(new_memory, (batch_size, self.memory_size * self.latent_dim))
        
        return h_corrected, [h_corrected, new_memory_flat]

    def get_config(self):
        """Configuration pour la sérialisation."""
        config = super(MATAPCell, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'memory_size': self.memory_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


class VanillaGRUCell(layers.Layer):
    """
    Cellule GRU Vanilla (Baseline) pour comparaison ablation.
    
    Identique interface que MATAPCell mais sans mémoire ni attention.
    """
    
    def __init__(self, latent_dim, **kwargs):
        super(VanillaGRUCell, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        self.input_proj = layers.Dense(latent_dim, name="input_projection")
        self.gru = layers.GRUCell(latent_dim, name="gru_dynamics")
        self.layer_norm = layers.LayerNormalization(name="norm_output")
        
        self.state_size = [latent_dim]
        self.output_size = latent_dim

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.float32
        return [tf.zeros((batch_size, self.latent_dim), dtype=dtype)]

    def call(self, inputs, states, training=None):
        h_prev = states[0]
        z_t = self.input_proj(inputs)
        gru_out, [h_new] = self.gru(z_t, [h_prev], training=training)
        h_new = self.layer_norm(h_new)
        return h_new, [h_new]

    def get_config(self):
        config = super(VanillaGRUCell, self).get_config()
        config.update({'latent_dim': self.latent_dim})
        return config
