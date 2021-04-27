from Dropblock import *
from keras.layers import *
# from layer import *
from attention_module import *


def BatchActivate(x):
    x = BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                           beta_initializer='zero', gamma_initializer='one')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block_dropblock(x, filters, size, strides=(1, 1), padding='same', activation=True, keep_prob=0.9,
                                block_size=7, df='channels_last'):
    x = Conv2D(filters, size, strides=strides, padding=padding, data_format=df)(x)
    x = DropBlock2D(block_size=block_size, keep_prob=keep_prob, data_format=df)(x)
    if activation:
        x = BatchActivate(x)
    return x

# 双残差块
def residual_drop_block(blockInput, num_filters=16, batch_activate=False, keep_prob=0.9, block_size=7, df='channels_last'):
    x = BatchActivate(blockInput)
    x = convolution_block_dropblock(x, num_filters, (3, 3), keep_prob=keep_prob, block_size=block_size, df=df)
    x = convolution_block_dropblock(x, num_filters, (3, 3), activation=False, keep_prob=keep_prob, block_size=block_size, df=df)

    # print("blockInput.get_shape() :", blockInput.get_shape())
    # print("x.get_shape() :", x.get_shape())
    if blockInput.get_shape().as_list()[-1] != x.get_shape().as_list()[-1]:
        blockInput = Conv2D(num_filters, (1, 1), activation=None, padding="same", data_format=df)(blockInput)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def RCAB(input, batch_activate=True, block_size=7, keep_prob=0.9, df='channels_last'):
    num_filters = input.get_shape().as_list()[-1]
    f = BatchActivate(input)
    f = convolution_block_dropblock(f, num_filters, (3, 3), keep_prob=keep_prob, block_size=block_size, df=df)
    f = convolution_block_dropblock(f, num_filters, (3, 3), activation=False, keep_prob=keep_prob, block_size=block_size, df=df)
    x = meca_block(f)
    result = add([input, x])
    if batch_activate:
        result = BatchActivate(result)
    return result
