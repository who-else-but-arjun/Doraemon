from keras.layers import Layer, Conv2D, Add, Activation, Dropout
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization 

class ReflectionPadding2D(Layer):
    """Reflection Padding Layer (custom implementation)"""
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        return tf.pad(x, [[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]], mode='REFLECT')

def res_block(x, filters, use_dropout=False):
    """Residual block with Instance Normalization."""
    res = ReflectionPadding2D((1, 1))(x)
    res = Conv2D(filters, (3, 3), padding='valid')(res)
    res = InstanceNormalization()(res)
    res = Activation('relu')(res)
    
    res = ReflectionPadding2D((1, 1))(res)
    res = Conv2D(filters, (3, 3), padding='valid')(res)
    res = InstanceNormalization()(res)
    
    if use_dropout:
        res = Dropout(0.5)(res)
    
    return Add()([x, res])

