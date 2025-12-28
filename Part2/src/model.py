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
    
    
    
    encoder_inputs = Input(shape=input_shape, name="encoder_input")
    
    
    
    
    encoder = layers.Bidirectional(layers.LSTM(hidden_dim // 2, return_sequences=True, return_state=True), name="bi_lstm_encoder")
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    
    
    
    state_h = layers.Concatenate()([forward_h, backward_h])
    state_c = layers.Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]
    
    
    
    
    
    
    
    
    
    
    decoder_inputs = Input(shape=(output_len, output_dim), name="decoder_input")
    
    
    
    decoder_lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    
    
    
    
    
    attention_layer = layers.Attention(use_scale=True, name="cross_attention")
    
    
    
    
    context_vector = attention_layer([decoder_outputs, encoder_outputs])
    
    
    
    decoder_combined_context = layers.Concatenate(axis=-1, name="concat_context")([decoder_outputs, context_vector])
    
    
    output_layer = layers.Dense(output_dim, name="output_dense")
    decoder_pred = output_layer(decoder_combined_context)
    
    
    model = models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred, name="Seq2Seq_Attention_Model")
    
    return model
