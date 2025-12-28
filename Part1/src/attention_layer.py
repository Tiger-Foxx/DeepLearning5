import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        
        
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        
        
        
        
        
        e = K.tanh(K.dot(x, self.W) + self.b)
        
        
        
        a = K.softmax(e, axis=1)
        
        
        
        
        
        
        output = x * a
        context_vector = K.sum(output, axis=1)
        
        return context_vector, a
