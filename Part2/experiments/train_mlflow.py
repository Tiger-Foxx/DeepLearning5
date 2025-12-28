import os
import sys
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Part2.data.generator import SyntheticDataGenerator
from Part2.src.model import build_seq2seq_model

def compute_attention_weights_manually(model, X_sample, dec_sample):
    
    encoder_input = model.input[0]
    encoder_output_layer = model.get_layer("bi_lstm_encoder")
    
    
    enc_model = tf.keras.Model(inputs=model.input[0], outputs=model.get_layer("bi_lstm_encoder").output[0])
    
    
    
    
    
    
    
    
    att_layer = model.get_layer("cross_attention")
    
    
    
    
    
    
    
    
    
    
    
    lstm_dec = model.get_layer("decoder_lstm").output[0]
    lstm_enc = model.get_layer("bi_lstm_encoder").output[0]
    
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=[lstm_dec, lstm_enc])
    queries, values = intermediate_model.predict([X_sample, dec_sample])
    
    
    
    
    
    d = queries.shape[-1]
    scores = np.matmul(queries, values.transpose(0, 2, 1)) 
    scores = scores / np.sqrt(d)
    
    
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    return weights

def train():
    
    mlflow.set_experiment("TP5_Part2_Seq2Seq")
    
    with mlflow.start_run():
        
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
        
        
        gen = SyntheticDataGenerator(input_len=INPUT_LEN, output_len=OUTPUT_LEN, n_samples=1000)
        X, y = gen.get_data()
        
        
        
        
        
        
        
        
        decoder_input = np.zeros_like(y)
        decoder_input[:, 1:, :] = y[:, :-1, :]
        
        
        model = build_seq2seq_model(input_shape=(INPUT_LEN, 1), output_len=OUTPUT_LEN, output_dim=1, hidden_dim=HIDDEN_DIM)
        model.compile(optimizer='adam', loss='mse')
        
        
        history = model.fit([X, decoder_input], y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
        
        
        for i, loss in enumerate(history.history['loss']):
            mlflow.log_metric("train_loss", loss, step=i)
        for i, val_loss in enumerate(history.history['val_loss']):
            mlflow.log_metric("val_loss", val_loss, step=i)
            
        
        print("Analyzing Attention Span...")
        sample_idx = 0
        X_sample = X[sample_idx:sample_idx+1]
        dec_sample = decoder_input[sample_idx:sample_idx+1]
        
        att_weights = compute_attention_weights_manually(model, X_sample, dec_sample) 
        att_matrix = att_weights[0]
        
        
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
        
        
        model_path = "Part2/experiments/outputs/model.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    train()
