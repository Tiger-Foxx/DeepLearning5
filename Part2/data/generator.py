import numpy as np

class SyntheticDataGenerator:
    def __init__(self, input_len=50, output_len=20, n_samples=1000, features=1):
        self.input_len = input_len
        self.output_len = output_len
        self.n_samples = n_samples
        self.features = features

    def generate_signal(self, length):
        # Generate a sum of sinusoids + trend
        t = np.arange(length)
        
        # Random parameters
        freq1 = np.random.uniform(0.01, 0.1)
        freq2 = np.random.uniform(0.05, 0.2)
        offset = np.random.uniform(-1, 1)
        
        signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.01 * t * offset
        
        # Add noise
        noise = np.random.normal(0, 0.1, length)
        return signal + noise

    def get_data(self):
        X = []
        y = []
        
        total_len = self.input_len + self.output_len
        
        for _ in range(self.n_samples):
            # Generate a time series
            full_seq = self.generate_signal(total_len)
            
            # Split into input and target
            # X: past, y: future
            X.append(full_seq[:self.input_len])
            y.append(full_seq[self.input_len:])
            
        X = np.array(X)[..., np.newaxis] # Add feature dim if 1
        y = np.array(y)[..., np.newaxis]
        
        # If features > 1, repeat or generate more signals (keeping it simple for now with 1 feature)
        if self.features > 1:
            # Simple hack: just repeat the signal with some noise for other features
            # Or generate distinct signals.
            # Let's generat distinct signals
             X_multi = [X]
             y_multi = [y]
             for _ in range(self.features - 1):
                 X_tmp = []
                 y_tmp = []
                 for _ in range(self.n_samples):
                     full_seq = self.generate_signal(total_len)
                     X_tmp.append(full_seq[:self.input_len])
                     y_tmp.append(full_seq[self.input_len:])
                 X_multi.append(np.array(X_tmp)[..., np.newaxis])
                 y_multi.append(np.array(y_tmp)[..., np.newaxis])
             
             X = np.concatenate(X_multi, axis=-1)
             y = np.concatenate(y_multi, axis=-1)

        return X, y
