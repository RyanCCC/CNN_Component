import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Input,Dropout
'''
以DenseNet为例，添加stn模块
'''

def stn_model(inputs)