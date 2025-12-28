import tensorflow as tf
from tensorflow.keras import layers, models, Input
from .attention_layer import SimpleAttention

def build_gru_attention_model(input_shape, hidden_dim=64, output_dim=1):
    """
    Builds the model: Input -> GRU (return_sequences=True) -> SimpleAttention -> Dense
    """
    inputs = Input(shape=input_shape)
    
    
    gru_out = layers.GRU(hidden_dim, return_sequences=True)(inputs)
    
    
    
    
    context_vector, attention_weights = SimpleAttention()(gru_out)
    
    
    outputs = layers.Dense(output_dim)(context_vector)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="GRU_Attention_Model")
    return model
