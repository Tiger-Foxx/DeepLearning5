import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Add current directory to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import build_gru_attention_model
from src.attention_layer import SimpleAttention

def generate_dummy_data(n_samples=1000, seq_len=50, input_dim=10):
    """
    Generates dummy random data for testing the pipeline.
    Target is a simple function of the input to ensure trainability.
    """
    X = np.random.randn(n_samples, seq_len, input_dim)
    # Target: sum of the first feature across time, just to have something to learn
    y = np.sum(X[:, :, 0], axis=1)
    return X, y

def run_experiment():
    print("running Part 1 Experiment...")
    
    # Parameters
    SEQ_LEN = 50
    INPUT_DIM = 10
    HIDDEN_DIM = 32
    BATCH_SIZE = 32
    EPOCHS = 5
    
    # Data
    print("Generating dummy data...")
    X, y = generate_dummy_data(n_samples=1000, seq_len=SEQ_LEN, input_dim=INPUT_DIM)
    
    # Model check
    print("Building model...")
    model = build_gru_attention_model((SEQ_LEN, INPUT_DIM), hidden_dim=HIDDEN_DIM)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Train
    print("Training model...")
    history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
    
    # Test Attention output
    print("Testing attention mechanism...")
    test_sample = X[:1] # (1, 50, 10)
    
    # Create a sub-model to extract attention weights
    # The original model outputs only the Dense result. 
    # We need to tap into the attention layer.
    
    # Re-instantiate a model that returns attention weights for visualization
    inputs = model.input
    # Find the attention layer output
    # Layer 0: Input, Layer 1: GRU, Layer 2: SimpleAttention
    # SimpleAttention call returns (context, weights)
    # Keras functional API allows accessing layer outputs, but since custom layer returns a tuple, 
    # we might need to be careful.
    
    # Let's inspect layers
    gru_out = model.layers[1].output
    context, weights = model.layers[2](gru_out)
    
    debug_model = tf.keras.models.Model(inputs=inputs, outputs=weights)
    attention_weights = debug_model.predict(test_sample)
    
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Sum of weights: {np.sum(attention_weights)}")
    
    assert attention_weights.shape == (1, SEQ_LEN, 1), f"Expected shape (1, {SEQ_LEN}, 1), got {attention_weights.shape}"
    assert np.isclose(np.sum(attention_weights), 1.0), f"Attention weights should sum to 1, got {np.sum(attention_weights)}"
    
    print("Part 1 Experiment completed successfully.")
    
    # Save a plot of attention weights
    plt.figure()
    plt.plot(attention_weights[0, :, 0])
    plt.title("Attention Weights for a random sample")
    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    os.makedirs("Part1/experiments/results", exist_ok=True)
    plt.savefig("Part1/experiments/results/attention_plot.png")
    print("Attention plot saved to Part1/experiments/results/attention_plot.png")

if __name__ == "__main__":
    run_experiment()
