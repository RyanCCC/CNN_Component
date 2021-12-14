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

def generate_normalized_homo_meshgrids(inputs):
    # for x, y in grid, -1 <=x,y<=1
    batch_size = tf.shape(inputs)[0]
    _, H, W,_ = inputs.shape
    x_range = tf.range(W)
    y_range = tf.range(H) 
    x_mesh, y_mesh = tf.meshgrid(x_range, y_range)
    x_mesh = (x_mesh/W-0.5)*2
    y_mesh = (y_mesh/H-0.5)*2
    y_mesh = tf.reshape(y_mesh, (*y_mesh.shape,1))
    x_mesh = tf.reshape(x_mesh, (*x_mesh.shape,1))
    ones_mesh = tf.ones_like(x_mesh)
    homogeneous_grid = tf.concat([x_mesh, y_mesh, ones_mesh],-1)
    homogeneous_grid = tf.reshape(homogeneous_grid, (-1, 3,1))
    homogeneous_grid = tf.dtypes.cast(homogeneous_grid, tf.float32)
    homogeneous_grid = tf.expand_dims(homogeneous_grid, 0)
    return tf.tile(homogeneous_grid, [batch_size, 1,1,1])

def transform_grids(transformations, grids, inputs):
    with tf.name_scope("transform_grids"):
        trans_matrices=tf.reshape(transformations, (-1, 2,3))
        batch_size = tf.shape(trans_matrices)[0]
        gs = tf.squeeze(grids, -1)

        reprojected_grids = tf.matmul(trans_matrices, gs, transpose_b=True)
        # transform grid range from [-1,1) to the range of [0,1)
        reprojected_grids = (tf.linalg.matrix_transpose(reprojected_grids) + 1)*0.5
        _, H, W, _ = inputs.shape
        reprojected_grids = tf.math.multiply(reprojected_grids, [W, H])

        return reprojected_grids

def spatial_transform_input(inputs, transormations):
    grids = generate_normalized_homo_meshgrids(inputs)
    reprojected_grids = transform_grids(transormations, grids,inputs)
    result = bilinear_sample(inputs, reprojected_grids)
    return result

def stn_module(input_shape):
    inputs = Input(input_shape)
    localication_head = create_localizaton_head(inputs)

