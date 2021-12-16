import tensorflow as tf
from tensorflow.keras.layers import Activation,Reshape, Lambda
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.merge import add, dot


def _convND(input, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"
    if rank == 3:
        x = Conv1D(channels, 1, padding='same', kernel_initializer = 'he_normal')(input)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    return x


def non_local_block(input, intermediate_dim=None, compression=2, mode='embedded', add_residual=True):
    '''
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3(temporal), 4(spatial) or 5(spatio-temporal)

    Arguments:
        input:input tensor
        intermediate_dim: The dimension of the intermediate representation
        compression: None or positive integer.
        mode: Mode of operation
        add_residual: Boolean value to decide if the residual connection should be added or not.
    
    Returns:
        a tensor of same shape of input
    '''
    # 获取通道数所在的维度
    channel_dim =1 if K.image_data_format() == 'channel_first' else -1
    input_shape = K.int_shape(input)
    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1
    
     # check rank and calculate the input shape
    if len(input_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = input_shape

    elif len(input_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = input_shape
        else:
            batchsize, dim1, dim2, channels = input_shape

    elif len(input_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = input_shape
        else:
            batchsize, dim1, dim2, dim3, channels = input_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')
    
    if intermediate_dim is None:
        intermediate_dim=channels//2

        if intermediate_dim<1:
            intermediate_dim=1
    else:
        intermediate_dim = int(intermediate_dim)
        if intermediate_dim<1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')
    
    # instantiation
    if mode == 'gaussian':
        x1 = Reshape((-1, channels))(input)
        x2 = Reshape((-1, channels))(input)
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)
    elif mode == 'dot':
        # theta path
        theta = _convND(input, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(input, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)
    else:
        # theta path
        theta = _convND(input, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(input, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)
    
    # g path
    g = _convND(input, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([input, y])

    return y
