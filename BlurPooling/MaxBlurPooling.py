import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np

class MaxBlurPooling1D(Layer):
    def __init__(self, pool_size=2, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self._pool_size = pool_size
        self._kernel_size = kernel_size
        self._avg_kernel = None
        self._blur_kernel = None
    
    def build(self, input_shape):
        if self._kernel_size == 3:
            blur_kernel = np.array([2, 4, 2])
        elif self._kernel_size == 5:
            blur_kernel = np.array([6, 24, 36, 24, 6])
        else:
            raise ValueError
        blur_kernel = blur_kernel/np.sum(blur_kernel)
        blur_kernel = np.repeat(blur_kernel, input_shape[2])
        blur_kernel = np.reshape(blur_kernel, (self._kernel_size, 1, input_shape[2], 1))
        blur_init = tf.keras.initializers.constant(blur_kernel)

        self._blur_kernel = self.add_weight(name='blur_kernel', shape=(self._kernel_size, 1, input_shape[2], 1), initializer=blur_init, trainable=False)
        super(MaxBlurPooling1D,self).build(input_shape)
    
    def call(self, x):
        x = tf.nn.pool(x, (self._pool_size, ), strides=(1, ), padding='SAME', pooling_type='MAX',data_format='NWC')
        x = K.expand_dims(x, axis=-2)
        x = K.depthwise_conv2d(x, self._blur_kernel, padding='same', strides=(self._pool_size, self._pool_size))
        x = K.squeeze(x, axis=-2)
        return x
    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1]/2)), input_shape[2]

class MaxBlurPooling2D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(MaxBlurPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            blur_kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            blur_kernel = blur_kernel / np.sum(blur_kernel)
        elif self.kernel_size == 5:
            blur_kernel = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            blur_kernel = blur_kernel / np.sum(blur_kernel)
        else:
            raise ValueError

        blur_kernel = np.repeat(blur_kernel, input_shape[3])

        blur_kernel = np.reshape(blur_kernel, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init =tf.keras.initializers.constant(blur_kernel)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(MaxBlurPooling2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = tf.nn.pool(x, (self.pool_size, self.pool_size),
                       strides=(1, 1), padding='SAME', pooling_type='MAX', data_format='NHWC')
        x = K.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]