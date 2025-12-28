import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_seq2seq_model(input_shape, output_len, output_dim, hidden_dim=64):
    """
    Builds a Seq2Seq model with Bi-LSTM Encoder and LCD-Attention Decoder.
    
    Args:
        input_shape: (input_seq_len, feature_dim)
        output_len: length of the output sequence to generate
        output_dim: feature dimension of output
        hidden_dim: dimension of LSTM units
    """
    
    # --- ENCODER ---
    # Input: (batch, input_len, features)
    encoder_inputs = Input(shape=input_shape, name="encoder_input")
    
    # Bi-LSTM Encoder
    # We need return_sequences=True for Attention (to attend to all states)
    # We need return_state=True to initialize Decoder
    encoder = layers.Bidirectional(layers.LSTM(hidden_dim // 2, return_sequences=True, return_state=True), name="bi_lstm_encoder")
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    
    # Concatenate states to pass to decoder (which is likely uni-directional or matching dim)
    # Forward and Backward states concatenation
    state_h = layers.Concatenate()([forward_h, backward_h])
    state_c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    # --- DECODER ---
    # In training, we usually use teacher forcing: Input = Target shifted by 1.
    # But for a simple "vector output" model (predicting the whole future sequence at once is one way, 
    # but Seq2Seq usually implies step-by-step).
    # Keras functional API doesn't support dynamic unrolling easily without custom loops.
    # However, if we know output_len fixed, we can use RepeatingVector or just providing expected decoder inputs.
    
    # We will assume "decoder_inputs" are provided during training (Teacher Forcing).
    # Shape: (batch, output_len, features)
    decoder_inputs = Input(shape=(output_len, output_dim), name="decoder_input")
    
    # Decoder LSTM
    # return_sequences=True to get hidden state at each step
    decoder_lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    # --- CROSS-ATTENTION ---
    # Query: Decoder Outputs (what we are generating)
    # Value/Key: Encoder Outputs (source memory)
    # Returns: Context Vector sequence matching Decoder length
    
    attention_layer = layers.Attention(use_scale=True, name="cross_attention")
    # Note: Attention(use_scale=True) implements Scaled Dot-Product Attention
    
    # Keras Attention call args: [query, value, key(optional)]
    # We want to attend TO encoder_outputs FROM decoder_outputs
    context_vector = attention_layer([decoder_outputs, encoder_outputs])
    
    # Concatenate Decoder Output and Context Vector
    # This combines the RNN's internal prediction with the relevant source info
    decoder_combined_context = layers.Concatenate(axis=-1, name="concat_context")([decoder_outputs, context_vector])
    
    # Final Dense 
    output_layer = layers.Dense(output_dim, name="output_dense")
    decoder_pred = output_layer(decoder_combined_context)
    
    # Model
    model = models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred, name="Seq2Seq_Attention_Model")
    
    return model
