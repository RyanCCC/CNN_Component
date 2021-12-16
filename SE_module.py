import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation, Reshape, Conv2D, BatchNormalization
import numpy as np

class Squeeze_excitation_layer(tf.keras.Model):
    def __init__(self, filter_sq):
        super().__init__()
        self.filter_sq = filter_sq
    
    def call(self, inputs):
        channel = inputs.shape[-1]
        squeeze = GlobalAveragePooling2D()(inputs)
        excitation = Dense(channel//self.filter_sq)(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(channel)(excitation)
        excitation = Activation('sigmoid')(excitation)
        # reshape excitation: 1*1*input.shape[-1]
        excitation = Reshape((1, 1, channel))(excitation)
        # 获得通道权重
        outputs = inputs*excitation
        return outputs

def SEBottleneck(input, filter_sq=16, stride=1):
    residual = inputs
    se_module = Squeeze_excitation_layer(16)

    x = Conv2D(16, kernel_size=1)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, kernel_size=3, strides=stride, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, kernel_size=1)(input)
    x = BatchNormalization()(x)
    x = se_module(x)

    output = x+residual
    output = Activation('relu')(output)

    return output
    



SE_module = Squeeze_excitation_layer(16)
inputs = np.zeros((1, 32, 32, 32), dtype=np.float32)
out_shape = SE_module(inputs).shape
print(out_shape)


inputs = np.zeros((1, 32, 32, 32), dtype=np.float32)
SEB = SEBottleneck(inputs)
print(SEB.shape)

