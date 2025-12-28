import tensorflow as tf
from tensorflow.keras import layers, models, Input
from .attention_layer import SimpleAttention

def build_gru_attention_model(input_shape, hidden_dim=64, output_dim=1):
    """
    Builds the model: Input -> GRU (return_sequences=True) -> SimpleAttention -> Dense
    """
    inputs = Input(shape=input_shape)
    
    # GRU layer must return sequences for attention to work on all hidden states
    gru_out = layers.GRU(hidden_dim, return_sequences=True)(inputs)
    
    # Custom Attention Layer
    # Returns [context_vector, attention_weights]
    # We only need the context vector for the downstream Dense layer
    context_vector, attention_weights = SimpleAttention()(gru_out)
    
    # Dense layer for final prediction
    outputs = layers.Dense(output_dim)(context_vector)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="GRU_Attention_Model")
    return model
