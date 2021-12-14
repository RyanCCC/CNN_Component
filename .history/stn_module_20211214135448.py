import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Input,Dropout
'''
以DenseNet为例，添加stn模块
'''

def create_localizaton_head(inputs):
    x = Conv2D(14, (5, 5), padding='valid', activation='relu')(inputs)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = Conv2D(32, (5, 5), padding='valid', activation='relu')(x)
    x = MaxPooling2D((2,2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(120,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(84, activation='relu')(x)
    # 6 elements to describe the transformation
    x = Dense(6, activation='linear', kernel_initializer='zeros', 
        bias_initializer=lambda shape, dtype:tf.constant([1,0,0,0,1,0], dtype=dtype))(x)
    return tf.keras.Model(inputs, x)

def stn_module(input_shape):
    localication_head = create_localizaton_head
