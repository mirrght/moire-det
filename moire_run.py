# 卷积层C1过滤输入图像，具有32个大小为7 x 7，跨度为1像素的卷积核。池化层2x2。然后取三个特征图像卷积后的最大值。亮度的卷积图像与最大值的结果相乘。
# C2-C4有16个大小为3×3的内核，跨度为1像素。S2汇集了合并后的特征，跨度为4。全连接层FC1有32个神经元，FC2有1个神经元。输出层的激活函数是softmax函数。

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Add, \
    Multiply, Maximum


def createModel(height, width, depth, num_classes):
    kernel_size_1 = 7  # 7x7 kernels
    kernel_size_2 = 3  # 3x3 kernels
    pool_size = 2  # 2x2 pooling
    conv_depth_1 = 32  # 每个卷积层 32 个核
    conv_depth_2 = 16  # 之后16个核
    drop_prob_1 = 0.25  # 卷积层 dropout 概率0.25
    drop_prob_2 = 0.5  # 全连接层 dropout 概率 0.5
    hidden_size = 16  # 全连接层神经元

    inpV = Input(shape=(height, width, depth))
    inpML = Input(shape=(height, width, depth))
    inpHH = Input(shape=(height, width, depth))
    inpHS = Input(shape=(height, width, depth))

    conv_1_V = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpV)
    conv_1_ML = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpML)
    conv_1_HH = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHH)
    conv_1_HS = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), padding='same', activation='relu')(inpHS)
    pool_1_V = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_V)
    pool_1_ML = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_ML)
    pool_1_HH = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HH)
    pool_1_HS = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1_HS)

    avg_ML_HH_HS = Maximum()([pool_1_ML, pool_1_HH, pool_1_HS])
    inp_merged = Multiply()([pool_1_V, avg_ML_HH_HS])
    C4 = Convolution2D(conv_depth_1, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(inp_merged)
    S2 = MaxPooling2D(pool_size=(4, 4))(C4)
    drop_1 = Dropout(drop_prob_1)(S2)
    C5 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(drop_1)
    S3 = MaxPooling2D(pool_size=(pool_size, pool_size))(C5)
    C6 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2), padding='same', activation='relu')(S3)
    S4 = MaxPooling2D(pool_size=(pool_size, pool_size))(C6)
    drop_2 = Dropout(drop_prob_1)(S4)

    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(inputs=[inpV, inpML, inpHH, inpHS], outputs=out)

    return model

