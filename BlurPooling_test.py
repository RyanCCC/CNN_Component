import  BlurPooling as pooling
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np

def test_pooling_model(input_shape, pooling_type):
    input = Input(input_shape)
    layer_pool = eval('pooling.'+pooling_type)()(input)
    layer_flattern = Flatten()(layer_pool)
    output = Dense(1)(layer_flattern)
    model = Model(input, output)
    model.summary()
    return model

if __name__=='__main__':
    poolingtype = [
        'MaxBlurPooling1D',
        'MaxBlurPooling2D',
        'AverageBlurPooling1D',
        'AverageBlurPooling2D',
        'BlurPool2D',
        'BlurPool1D'
    ]
    for item in poolingtype:
        if '2D' in item:
            input_shape = (224, 224, 3)
            model = test_pooling_model(input_shape, item)
            model.predict([np.random.random((1, 224, 224, 3))])
        else:
            input_shape = (224, 3)
            model = test_pooling_model(input_shape, item)
            model.predict([np.random.random((1, 224, 3))])

