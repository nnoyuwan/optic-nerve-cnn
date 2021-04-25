from layer import *
from keras.optimizers import *
from keras.models import *
from attention_module import *


def CARUNet_modified(input_size=(3, 256, 256), start_neurons=32, keep_prob=0.8, block_size=7):
    inputs = Input(input_size)
    # print(inputs.shape())
    # contract path
    conv1 = residual_drop_block(inputs, start_neurons * 1, False, block_size=block_size, keep_prob=keep_prob)
    conv1 = RCAB(conv1, keep_prob=keep_prob, block_size=block_size)
    pool1 = MaxPooling2D((2, 2), data_format="channels_first")(conv1)
    print("pool1", pool1.get_shape())

    conv2 = residual_drop_block(pool1, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    conv2 = RCAB(conv2, keep_prob=keep_prob, block_size=block_size)
    pool2 = MaxPooling2D((2, 2), data_format="channels_first")(conv2)
    print("pool2", pool2.get_shape())

    conv3 = residual_drop_block(pool2, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    conv3 = RCAB(conv3, keep_prob=keep_prob, block_size=block_size)
    pool3 = MaxPooling2D((2, 2), data_format="channels_first")(conv3)
    print("pool3", pool3.get_shape())

    conv4 = residual_drop_block(pool3, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    conv4 = RCAB(conv4, keep_prob=keep_prob, block_size=block_size)
    pool4 = MaxPooling2D((2, 2), data_format="channels_first")(conv4)
    print("pool4", pool4.get_shape())

    # bottom
    convm = residual_drop_block(pool4, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    convm = RCAB(convm, keep_prob=keep_prob, block_size=block_size)
    print("convm", convm.get_shape())

    # expansive path
    deconv4 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), data_format="channels_first", padding="same")(convm)
    # print("deconv4", deconv4.get_shape())
    uconv4 = Concatenate(axis=3)([deconv4, meca_block(conv4)])
    uconv4 = residual_drop_block(uconv4, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    uconv4 = RCAB(uconv4, keep_prob=keep_prob, block_size=block_size)


    deconv3 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), data_format="channels_first", padding="same")(uconv4)
    # print("deconv3", deconv3.get_shape())
    uconv3 = Concatenate(axis=3)([deconv3, meca_block(conv3)])
    uconv3 = residual_drop_block(uconv3, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    uconv3 = RCAB(uconv3, keep_prob=keep_prob, block_size=block_size)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), data_format="channels_first", padding="same")(uconv3)
    uconv2 = Concatenate(axis=3)([deconv2, meca_block(conv2)])
    uconv2 = residual_drop_block(uconv2, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    uconv2 = RCAB(uconv2, keep_prob=keep_prob, block_size=block_size)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), data_format="channels_first", padding="same")(uconv2)
    uconv1 = Concatenate(axis=3)([deconv1, meca_block(conv1)])
    uconv1 = residual_drop_block(uconv1, start_neurons * 1, False, block_size=block_size, keep_prob=keep_prob)
    uconv1 = RCAB(uconv1, keep_prob=keep_prob, block_size=block_size)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(input=inputs, output=output_layer)

    # model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model
