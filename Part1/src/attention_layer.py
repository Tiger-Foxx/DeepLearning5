import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is (batch_size, seq_len, hidden_dim)
        # Weight matrix W: (hidden_dim, 1)
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        # Bias b: (seq_len, 1) according to TP instructions
        # Note: This makes the layer dependent on a fixed sequence length for the bias term.
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        
        # 1. Compute scores
        # e = tanh(x . W + b)
        # x . W -> (batch, seq, hidden) . (hidden, 1) = (batch, seq, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)
        
        # 2. Compute weights using (softmax over time axis)
        # a = softmax(e)
        a = K.softmax(e, axis=1)
        
        # 3. Compute context vector
        # context = sum(a * x)
        # a: (batch, seq, 1), x: (batch, seq, hidden)
        # broadcast multiplication -> (batch, seq, hidden)
        # sum over seq axis -> (batch, hidden)
        output = x * a
        context_vector = K.sum(output, axis=1)
        
        return context_vector, a
