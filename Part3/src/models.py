"""
Modèles pour la Partie 3 - MA-TAP Research
==========================================

Contient:
- SpatialEncoder: Frame -> Latent
- SpatialDecoder: Latent -> Frame  
- MATAPModel: Modèle complet avec mémoire augmentée
- BaselineTAPModel: Modèle baseline (GRU vanilla) pour ablation

Auteur: DONFACK Pascal
Date: Janvier 2026
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from .matap_cell import MATAPCell, VanillaGRUCell


# ==============================================================================
# COMPOSANTS SPATIAUX (Encoder / Decoder)
# ==============================================================================

class SpatialEncoder(keras.Model):
    """
    Encodeur Spatial: Frame (64x64x1) -> Vecteur Latent (latent_dim)
    
    Architecture CNN simple mais efficace pour Moving MNIST.
    """
    
    def __init__(self, latent_dim=64, **kwargs):
        super(SpatialEncoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        self.conv1 = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")
        self.bn3 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256, activation="relu")
        self.dropout = layers.Dropout(0.2)
        self.fc_out = layers.Dense(latent_dim)

    def call(self, x, training=False):
        # x: (B, 64, 64, 1)
        x = self.bn1(self.conv1(x), training=training)
        x = self.bn2(self.conv2(x), training=training)
        x = self.bn3(self.conv3(x), training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        z = self.fc_out(x)
        return z  # (B, latent_dim)


class SpatialDecoder(keras.Model):
    """
    Décodeur Spatial: Vecteur Latent (latent_dim) -> Frame (64x64x1)
    
    Architecture transposée du CNN encoder.
    """
    
    def __init__(self, latent_dim=64, **kwargs):
        super(SpatialDecoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        
        self.fc1 = layers.Dense(256, activation="relu")
        self.fc2 = layers.Dense(8 * 8 * 128, activation="relu")
        self.reshape = layers.Reshape((8, 8, 128))
        
        self.deconv1 = layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="relu")
        self.bn1 = layers.BatchNormalization()
        
        self.deconv2 = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")
        self.bn2 = layers.BatchNormalization()
        
        self.deconv3 = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu")
        self.bn3 = layers.BatchNormalization()
        
        self.output_conv = layers.Conv2D(1, 3, padding="same", activation="sigmoid")

    def call(self, z, training=False):
        # z: (B, latent_dim)
        x = self.fc1(z)
        x = self.fc2(x)
        x = self.reshape(x)
        x = self.bn1(self.deconv1(x), training=training)
        x = self.bn2(self.deconv2(x), training=training)
        x = self.bn3(self.deconv3(x), training=training)
        out = self.output_conv(x)
        return out  # (B, 64, 64, 1)


# ==============================================================================
# MODÈLE MA-TAP COMPLET
# ==============================================================================

class MATAPModel(keras.Model):
    """
    Memory-Augmented Time-Aware Path Model.
    
    Architecture complète pour la génération/prédiction de séquences vidéo.
    Encoder -> MA-TAP Dynamics -> Decoder
    
    Paramètres:
    -----------
    latent_dim : int
        Dimension de l'espace latent.
    memory_size : int
        Nombre de slots dans le buffer mémoire.
    num_heads : int
        Nombre de têtes d'attention dans MA-TAP.
    prediction_horizon : int
        Nombre de frames à prédire dans le futur.
    """
    
    def __init__(self, latent_dim=64, memory_size=10, num_heads=4, 
                 prediction_horizon=10, dropout_rate=0.1, **kwargs):
        super(MATAPModel, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.memory_size = memory_size
        self.prediction_horizon = prediction_horizon
        
        # Composants spatiaux
        self.encoder = SpatialEncoder(latent_dim)
        self.decoder = SpatialDecoder(latent_dim)
        
        # Dynamique temporelle MA-TAP
        self.matap_cell = MATAPCell(
            latent_dim=latent_dim,
            memory_size=memory_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        self.rnn = layers.RNN(self.matap_cell, return_sequences=True, return_state=True)
        
        # Prédicteur de l'état latent suivant
        self.predictor = keras.Sequential([
            layers.Dense(latent_dim * 2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(latent_dim)
        ], name="latent_predictor")
    
    def encode_sequence(self, frames, training=False):
        """Encode une séquence de frames en séquence latente."""
        # frames: (B, T, H, W, C)
        B = tf.shape(frames)[0]
        T = tf.shape(frames)[1]
        
        # Reshape pour encodage batch
        flat_frames = tf.reshape(frames, (B * T, 64, 64, 1))
        z_flat = self.encoder(flat_frames, training=training)
        z_seq = tf.reshape(z_flat, (B, T, self.latent_dim))
        
        return z_seq
    
    def decode_sequence(self, z_seq, training=False):
        """Décode une séquence latente en séquence de frames."""
        # z_seq: (B, T, latent_dim)
        B = tf.shape(z_seq)[0]
        T = tf.shape(z_seq)[1]
        
        z_flat = tf.reshape(z_seq, (B * T, self.latent_dim))
        frames_flat = self.decoder(z_flat, training=training)
        frames = tf.reshape(frames_flat, (B, T, 64, 64, 1))
        
        return frames
    
    def call(self, inputs, training=False):
        """
        Forward pass complet.
        
        Args:
            inputs: Séquence vidéo (B, T, 64, 64, 1)
            training: Mode entraînement
            
        Returns:
            reconstructed: Frames reconstruites (B, T, 64, 64, 1)
            z_true: Séquence latente vraie (B, T, latent_dim)
            z_pred: Séquence latente prédite (B, T, latent_dim)
            final_states: États finaux pour génération autorégressive
        """
        B = tf.shape(inputs)[0]
        
        # 1. Encodage spatial
        z_seq = self.encode_sequence(inputs, training=training)
        
        # 2. Dynamique temporelle MA-TAP
        initial_states = self.matap_cell.get_initial_state(batch_size=B)
        h_seq, final_h, final_memory = self.rnn(z_seq, initial_state=initial_states, training=training)
        
        # 3. Prédiction des états latents futurs
        z_pred = self.predictor(h_seq)
        
        # 4. Décodage spatial (reconstruction)
        reconstructed = self.decode_sequence(z_pred, training=training)
        
        return reconstructed, z_seq, z_pred, [final_h, final_memory]
    
    def generate_future(self, context_frames, num_future=10, training=False):
        """
        Génère des frames futures de manière autorégressive.
        
        Args:
            context_frames: Frames de contexte (B, T_ctx, 64, 64, 1)
            num_future: Nombre de frames à générer
            training: Mode
            
        Returns:
            future_frames: Frames générées (B, num_future, 64, 64, 1)
        """
        B = tf.shape(context_frames)[0]
        
        # Encodage du contexte
        z_ctx = self.encode_sequence(context_frames, training=training)
        
        # Passage dans le RNN pour obtenir l'état final
        initial_states = self.matap_cell.get_initial_state(batch_size=B)
        _, final_h, final_memory = self.rnn(z_ctx, initial_state=initial_states, training=training)
        
        # Génération autorégressive
        states = [final_h, final_memory]
        current_h = final_h  # État caché actuel
        
        future_z = []
        for _ in range(num_future):
            # Prédiction du prochain état latent à partir de l'état caché
            # Expand pour passer dans le predictor qui attend (B, T, D)
            h_expanded = tf.expand_dims(current_h, axis=1)  # (B, 1, D)
            z_next_pred = self.predictor(h_expanded)  # (B, 1, D)
            z_next_pred = tf.squeeze(z_next_pred, axis=1)  # (B, D)
            future_z.append(z_next_pred)
            
            # Mise à jour via MA-TAP avec le z prédit
            h_out, states = self.matap_cell(z_next_pred, states, training=training)
            current_h = h_out
        
        future_z_seq = tf.stack(future_z, axis=1)  # (B, num_future, latent_dim)
        future_frames = self.decode_sequence(future_z_seq, training=training)
        
        return future_frames
    
    def get_attention_weights(self, inputs, training=False):
        """Extrait les poids d'attention pour visualisation."""
        # Note: Nécessite une modification de MATAPCell pour stocker les poids
        # Pour l'instant, retourne un placeholder
        return None


# ==============================================================================
# MODÈLE BASELINE (ABLATION)
# ==============================================================================

class BaselineTAPModel(keras.Model):
    """
    Baseline TAP Model (sans mémoire) pour étude d'ablation.
    
    Architecture identique à MATAPModel mais avec GRU vanilla.
    """
    
    def __init__(self, latent_dim=64, prediction_horizon=10, dropout_rate=0.1, **kwargs):
        super(BaselineTAPModel, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.prediction_horizon = prediction_horizon
        
        # Composants spatiaux (partagés)
        self.encoder = SpatialEncoder(latent_dim)
        self.decoder = SpatialDecoder(latent_dim)
        
        # Dynamique temporelle GRU vanilla
        self.gru_cell = VanillaGRUCell(latent_dim)
        self.rnn = layers.RNN(self.gru_cell, return_sequences=True, return_state=True)
        
        # Prédicteur identique
        self.predictor = keras.Sequential([
            layers.Dense(latent_dim * 2, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(latent_dim)
        ], name="latent_predictor")
    
    def encode_sequence(self, frames, training=False):
        B = tf.shape(frames)[0]
        T = tf.shape(frames)[1]
        flat_frames = tf.reshape(frames, (B * T, 64, 64, 1))
        z_flat = self.encoder(flat_frames, training=training)
        z_seq = tf.reshape(z_flat, (B, T, self.latent_dim))
        return z_seq
    
    def decode_sequence(self, z_seq, training=False):
        B = tf.shape(z_seq)[0]
        T = tf.shape(z_seq)[1]
        z_flat = tf.reshape(z_seq, (B * T, self.latent_dim))
        frames_flat = self.decoder(z_flat, training=training)
        frames = tf.reshape(frames_flat, (B, T, 64, 64, 1))
        return frames
    
    def call(self, inputs, training=False):
        B = tf.shape(inputs)[0]
        
        # 1. Encodage
        z_seq = self.encode_sequence(inputs, training=training)
        
        # 2. Dynamique temporelle (GRU vanilla)
        initial_states = self.gru_cell.get_initial_state(batch_size=B)
        h_seq, final_h = self.rnn(z_seq, initial_state=initial_states, training=training)
        
        # 3. Prédiction
        z_pred = self.predictor(h_seq)
        
        # 4. Reconstruction
        reconstructed = self.decode_sequence(z_pred, training=training)
        
        return reconstructed, z_seq, z_pred, [final_h]
    
    def generate_future(self, context_frames, num_future=10, training=False):
        B = tf.shape(context_frames)[0]
        
        z_ctx = self.encode_sequence(context_frames, training=training)
        initial_states = self.gru_cell.get_initial_state(batch_size=B)
        _, final_h = self.rnn(z_ctx, initial_state=initial_states, training=training)
        
        states = [final_h]
        current_h = final_h
        
        future_z = []
        for _ in range(num_future):
            # Expand pour passer dans le predictor qui attend (B, T, D)
            h_expanded = tf.expand_dims(current_h, axis=1)  # (B, 1, D)
            z_next_pred = self.predictor(h_expanded)  # (B, 1, D)
            z_next_pred = tf.squeeze(z_next_pred, axis=1)  # (B, D)
            future_z.append(z_next_pred)
            
            h_out, states = self.gru_cell(z_next_pred, states, training=training)
            current_h = h_out
        
        future_z_seq = tf.stack(future_z, axis=1)
        future_frames = self.decode_sequence(future_z_seq, training=training)
        
        return future_frames


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def create_model(model_type='matap', latent_dim=64, memory_size=10, **kwargs):
    """Factory pour créer le modèle approprié."""
    if model_type == 'matap':
        return MATAPModel(latent_dim=latent_dim, memory_size=memory_size, **kwargs)
    elif model_type == 'baseline':
        return BaselineTAPModel(latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Compte le nombre de paramètres d'un modèle."""
    return sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
