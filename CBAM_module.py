import tensorflow  as tf
from tensorflow.keras import backend as K
import tensorflow.keras.layers  as layer
import numpy as np

class CBAM_module(tf.keras.Model):
    def __init__(self, ratio=16, name=''):
        super().__init__()
        self._ratio = ratio
        self._name = name
    
    def channel_attention(self, input):
        channel = input.shape[-1]
        # 同时进行avg pooling和 max pooling
        avg_pool = layer.GlobalAveragePooling2D()(input)
        max_pool = layer.GlobalAveragePooling2D()(input)
        avg_pool = layer.Reshape((1, 1, channel))(avg_pool)
        max_pool = layer.Reshape((1, 1, channel))(max_pool)

        # 对pooling结果经过两层全连接层，第一层核数量为input的通道数//ratio，第二层则恢复到原通道数
        avg_pool = layer.Dense(channel//self._ratio, activation='relu', kernel_initializer='he_normal', name=self._name)(avg_pool)
        max_pool = layer.Dense(channel//self._ratio, activation='relu',kernel_initializer='he_normal', name=self._name)(max_pool)

        avg_pool = layer.Dense(channel, activation='relu', kernel_initializer='he_normal', name=self._name)(avg_pool)
        max_pool = layer.Dense(channel, activation='relu', kernel_initializer='he_normal', name=self._name)(max_pool)

        # 对avg_pool与max_pool相加做激活，得到(batchsize, 1,1 channel)的tensor，作为权重与input相乘
        output = layer.Add()([avg_pool, max_pool])
        output = layer.Activation('sigmoid')(output)
        output = layer.multiply([input, output])
        return output
    
    def spatial_attention(self, input, kernel_size=7):
        avg_pool = layer.Lambda(lambda x:K.mean(x,axis=3, keepdims=True))(input)
        max_pool = layer.Lambda(lambda x:K.max(x,axis=3, keepdims=True))(input)

        concat_feature = layer.Concatenate(axis=3)([avg_pool, max_pool])
        output =layer.Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					kernel_initializer='he_normal')(concat_feature)
        output = layer.Activation('sigmoid')(output)
        output = layer.multiply([input, output])
        return output
    
    def call(self, input):
        cbam_feature = self.channel_attention(input)
        cbam_feature = self.spatial_attention(cbam_feature)
        return cbam_feature

CBAM_module = CBAM_module()
inputs = np.zeros((1, 32, 32, 32), dtype=np.float32)
out_shape = CBAM_module(inputs).shape
print(out_shape)