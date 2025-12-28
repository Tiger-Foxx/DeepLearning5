import os
import sys
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import seaborn as sns

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Part2.data.generator import SyntheticDataGenerator
from Part2.src.model import build_seq2seq_model

def get_attention_map(model, X_input, decoder_input):
    """
    Extracts attention weights from the model for a given input.
    """
    # Create a sub-model that outputs the attention weights
    # Input: [enc_in, dec_in]
    # We need to find the specific Attention layer output weights.
    # Standard Keras Attention layer does NOT return weights in call() by default unless needed?
    # Actually, keras.layers.Attention call() returns (batch, query_len, value_dim). It applies the weights internally.
    # To get weights, we might need to subclass or inspect. 
    # WAIT, Keras Attention layer has no `return_attention_scores` argument in older versions, 
    # but in newer TF it might.
    # If not available, we can access `scores` if we rebuilt logic, but using `MultiHeadAttention` has `return_attention_scores`.
    # `layers.Attention` (Dot product) is simpler. 
    # Let's check if we can inspect it. 
    # If strict TP requirement "Visualiser", we need those weights.
    # The clean provided "Part 1" code implemented manual attention. 
    # Maybe I should use a Custom Layer like Part 1 but for Cross Attention?
    # Or just rely on `MultiHeadAttention(..., use_bias=True, attention_axes=...)` which supports `return_attention_scores`.
    # Let's try to stick to the plan: use `layers.Attention`. 
    # If it doesn't support returning weights easily, I'll switch to a custom one or MultiHead.
    # Actually, `layers.Attention` documentation says it returns just the tensor.
    # To get weights, I might need to implement the attention math manually or use a slightly more complex Keras layer.
    
    # For robust "Master 2" level code, implementing a "VisualizableAttention" layer inheriting from `layers.Layer` or `layers.Attention` is best.
    pass

# Redefining Model building to return attention weights for visualization if needed, 
# or I will create a separate model for inference that outputs weights.
# Let's perform a trick: The `layers.Attention` layer in Keras is essentially:
# scores = query * key
# weights = softmax(scores)
# output = weights * value
# I will use a custom wrapper or just calculate it manually in the analysis step.
# For now, let's assume I can re-calculate attention weights given the trained embeddings/states.
# BUT, the `layers.Attention` has no trainable weights if use_scale=False (except scale if true).
# So I can just run the math manually on the outputs of the LSTM layers to visualize!
# Perfect. No need to change the model structure.

def compute_attention_weights_manually(model, X_sample, dec_sample):
    # 1. Get Encoder Outputs
    encoder_input = model.input[0]
    encoder_output_layer = model.get_layer("bi_lstm_encoder")
    # We need to run the graph up to encoder outputs
    # Create funcs
    enc_model = tf.keras.Model(inputs=model.input[0], outputs=model.get_layer("bi_lstm_encoder").output[0])
    
    # 2. Get Decoder Outputs (before attention)
    # This is tricky because Decoder takes initial state from Encoder.
    # We need the whole chain.
    # Let's just create a model that outputs [cross_attention_inputs]
    # The input to Attention layer is [decoder_outputs, encoder_outputs]
    
    # Let's find the attention layer input tensors.
    att_layer = model.get_layer("cross_attention")
    # We can create a model that outputs the inputs to this layer.
    # But Keras functional graph is static.
    
    # Alternative: Model outputs multiple things.
    # Let's update `build_seq2seq_model` in `src/model.py` to optionally return attention scores?
    # No, keep model clean.
    
    # Let's just extract the intermediate tensors:
    # inputs: [enc, dec]
    # outputs: [att_layer_input_query, att_layer_input_value] -> These are [decoder_lstm_out, encoder_lstm_out]
    
    lstm_dec = model.get_layer("decoder_lstm").output[0]
    lstm_enc = model.get_layer("bi_lstm_encoder").output[0]
    
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=[lstm_dec, lstm_enc])
    queries, values = intermediate_model.predict([X_sample, dec_sample])
    
    # Now compute attention manually: softmax(Q . K^T / sqrt(d))
    # Q: (batch, dec_len, dim)
    # K = V: (batch, enc_len, dim)
    
    d = queries.shape[-1]
    scores = np.matmul(queries, values.transpose(0, 2, 1)) # (batch, dec, enc)
    scores = scores / np.sqrt(d)
    
    # Apply softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    return weights

def train():
    # MLflow Setup
    mlflow.set_experiment("TP5_Part2_Seq2Seq")
    
    with mlflow.start_run():
        # Params
        INPUT_LEN = 50
        OUTPUT_LEN = 20
        HIDDEN_DIM = 64
        EPOCHS = 10
        BATCH_SIZE = 32
        
        mlflow.log_params({
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        })
        
        # Data
        gen = SyntheticDataGenerator(input_len=INPUT_LEN, output_len=OUTPUT_LEN, n_samples=1000)
        X, y = gen.get_data()
        
        # Teacher Forcing Inputs:
        # Decoder Input should be [StartToken, y_0, y_1, ...]
        # Here for simplicity (continuous values), we'll use:
        # decoder_input[:, 0] = 0 (or constant)
        # decoder_input[:, 1:] = y[:, :-1]
        # And predict y.
        
        decoder_input = np.zeros_like(y)
        decoder_input[:, 1:, :] = y[:, :-1, :]
        
        # Model
        model = build_seq2seq_model(input_shape=(INPUT_LEN, 1), output_len=OUTPUT_LEN, output_dim=1, hidden_dim=HIDDEN_DIM)
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        history = model.fit([X, decoder_input], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
        
        # Log metrics
        for i, loss in enumerate(history.history['loss']):
            mlflow.log_metric("train_loss", loss, step=i)
        for i, val_loss in enumerate(history.history['val_loss']):
            mlflow.log_metric("val_loss", val_loss, step=i)
            
        # Analysis: Attention Span
        print("Analyzing Attention Span...")
        sample_idx = 0
        X_sample = X[sample_idx:sample_idx+1]
        dec_sample = decoder_input[sample_idx:sample_idx+1]
        
        att_weights = compute_attention_weights_manually(model, X_sample, dec_sample) # (1, dec_len, enc_len)
        att_matrix = att_weights[0]
        
        # Plot Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(att_matrix, cmap='viridis')
        plt.xlabel("Encoder Steps")
        plt.ylabel("Decoder Steps")
        plt.title("Attention Weights Heatmap")
        
        os.makedirs("Part2/experiments/outputs", exist_ok=True)
        heatmap_path = "Part2/experiments/outputs/attention_heatmap.png"
        plt.savefig(heatmap_path)
        mlflow.log_artifact(heatmap_path)
        print(f"Artifact saved: {heatmap_path}")
        
        # Save Model
        model_path = "Part2/experiments/outputs/model.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    train()
