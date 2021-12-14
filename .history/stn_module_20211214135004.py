import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Input,Dropout
'''
以DenseNet为例，添加stn模块
'''

def create_localizaton_head(inputs):
    x = Conv2D(14, (5, 5), padding='valid', activation='relu')(inputs)
    x = MaxPooling2D((2, 2), strides=2)(x)
    


def stn_module(inputs):

