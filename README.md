# CNN_Component
基于Tensorflow2卷积神经网络即插即用模块实现

## STN
主要参考：https://xeonqq.github.io/machine%20learning/spatial-transformer-networks/ 。简单的理论部分可以参考我的博客：https://blog.csdn.net/u012655441/article/details/121919291 。STN结构如下图所示：
![image](https://user-images.githubusercontent.com/27406337/145952361-5d738cbc-ca73-40ce-bd89-4244b81358d6.png)
里面包括三个组件：
- **Localization net**：该网络可以是卷积神经网络或者是全连接神经网络，它们有个特点是最后一层是一个回归层，主要生成6个值表示仿射变换的参数θ。
- **Grid Generator**：它首先在目标图像V上生成一个网格，网格的每个点刚好对应目标图像中每个像素的像素坐标。然后它使用Localization net生成的θ来变换网格。
- **Sampler**：变换后的网格就像源图像U上的遮罩，它检索遮罩下的像素。然而，变换的网格不再包含整数值，因此对源图像U执行双线性插值，以获得变换网格下的估计像素值。

### Localization Net

Localization Net输入为\[批量大小、高度、宽度、通道]的输入图像，并为每个维度的输入图像生成转换参数。转换的维度为\[batch_size，6]。
```python
def create_localization_head(inputs):
    x = Conv2D(14, (5,5),padding='valid',activation="relu")(inputs)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = Conv2D(32, (5,5), padding='valid',activation="relu")(x)
    x = MaxPooling2D((2, 2),strides=2)(x)
    x = Flatten()(x)    
    x = Dense(120, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(84, activation='relu')(x)
    x = Dense(6, activation="linear", kernel_initializer="zeros",
              bias_initializer=lambda shape, dtype: tf.constant([1,0,0,0,1,0], dtype=dtype))(x) # 6 elements to describe the transformation
    return tf.keras.Model(inputs, x)
```

### Grid Generator

在网格生成器中，必须注意，变换θ应用于从目标图像V而不是源图像U生成的网格，在图像处理领域称为逆映射。另一方面，如果我们将源图像U转换为目标图像V，这个过程称为前向映射。

**正向映射**迭代输入图像的每个像素，为其计算新坐标，并将其值复制到新位置。但新坐标可能不在输出图像的边界内，也可能不是整数。通过在复制像素值之前检查计算的坐标，前一个问题很容易解决。第二个问题通过将最近的整数指定给x′和y′并将其用作变换像素的输出坐标来解决。问题在于，每个输出像素可能会被寻址多次或根本不寻址（后一种情况会导致“孔”，其中输出图像中的像素没有赋值）。**逆映射**迭代输出图像的每个像素，并使用逆变换确定输入图像中必须从中采样值的位置。在这种情况下，确定的位置也可能不在输入图像的边界内，也可能不是整数。但是输出图像没有孔。

```python
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
```

在```generate_normalized_homo_meshgrid```s函数中，给定输入维数，我们可以生成一个```meshgrid```。然后在[-1,1]之间对网格网格进行规格化，以便相对于图像中心执行旋转或平移。每个网格还扩展了第三维，并填充了第三维，因此被称为均质网格，在以下变换网格中更方便地执行变换。

在变换网格中，我们将从本地化网络生成的变换应用到从generate_normalized_homo_meshgrids生成的网格上，以获得重新```reprojected_grids```。变换后，```reprojected_grids```将重新缩放回输入图像的宽度和高度范围内。

### Sampler
```python
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
```

## Non-Local
![image](https://user-images.githubusercontent.com/27406337/146329854-5e1f5d7c-b69d-493e-8f88-60019b0eaae8.png)


## Deformable Convolution

参考：https://github.com/kastnerkyle/deform-conv
