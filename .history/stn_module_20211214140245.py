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

def generate_four_neighbors_from_reprojection(inputs, reprojected_grids):
    _, H, W, _ = inputs.shape
    
    x, y = tf.split(reprojected_grids, 2, axis=-1)

    x1 = tf.floor(x)
    x1 = tf.dtypes.cast(x1, tf.int32)

    x2 = x1 + tf.constant(1) 

    y1 = tf.floor(y)
    y1 = tf.dtypes.cast(y1, tf.int32)
    y2 = y1 + tf.constant(1) 
    
    y_max = tf.constant(H - 1, dtype=tf.int32)
    x_max = tf.constant(W - 1, dtype=tf.int32)
    zero = tf.zeros([1], dtype=tf.int32)

    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)
    x2_safe = tf.clip_by_value(x2, zero, x_max)
    y2_safe = tf.clip_by_value(y2, zero, y_max)
    return x1_safe, y1_safe, x2_safe, y2_safe

def bilinear_sample(inputs, reprojected_grids):
    x1, y1, x2, y2 = generate_four_neighbors_from_reprojection(inputs, reprojected_grids)
    x1y1 = tf.concat([y1,x1],-1)
    x1y2 = tf.concat([y2,x1],-1)
    x2y1 = tf.concat([y1,x2],-1)
    x2y2 = tf.concat([y2,x2],-1)

    pixel_x1y1 = tf.gather_nd(inputs, x1y1, batch_dims=1)
    pixel_x1y2 = tf.gather_nd(inputs, x1y2, batch_dims=1)
    pixel_x2y1 = tf.gather_nd(inputs, x2y1, batch_dims=1)
    pixel_x2y2 = tf.gather_nd(inputs, x2y2, batch_dims=1)
    x, y = tf.split(reprojected_grids, 2, axis=-1)
    wx = tf.concat([tf.dtypes.cast(x2, tf.float32) - x, x -tf.dtypes.cast(x1, tf.float32)],-1)
    wx = tf.expand_dims(wx, -2)
    wy = tf.concat([tf.dtypes.cast(y2, tf.float32) - y, y - tf.dtypes.cast(y1, tf.float32)],-1)
    wy = tf.expand_dims(wy, -1)
    Q = tf.concat([pixel_x1y1, pixel_x1y2, pixel_x2y1, pixel_x2y2], -1)
    Q_shape = tf.shape(Q)
    Q = tf.reshape(Q, (Q_shape[0], Q_shape[1],2,2))
    Q = tf.cast(Q, tf.float32)

    r = wx@Q@wy
    _, H, W, channels = inputs.shape

    r = tf.reshape(r, (-1,H,W,1))
    return r

def spatial_transform_input(inputs, transormations):
    grids = generate_normalized_homo_meshgrids(inputs)
    reprojected_grids = transform_grids(transormations, grids,inputs)
    result = bilinear_sample(inputs, reprojected_grids)
    return result

def stn_module(inputs):
    localication_head = create_localizaton_head(inputs)
    x = spatial_transform_input(inputs, localication_head.output)
    return x

def model(input_shape):
    inputs = Input(input_shape)
    inputs_stn = stn_module(inputs)
    x = Conv2D(6, (3,3),padding='valid',activation="relu")(inputs_stn)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3,3),padding='valid',activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(10)(x)
    return tf.keras.Model(inputs, x)

input_shape = (28, 28, 3)
st_model = model(input_shape)
st_model.summary()


