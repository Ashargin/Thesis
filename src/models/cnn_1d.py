from keras.layers import Conv1D, Activation, Input, Flatten
from keras.models import Model
from math import log2


# def CNN1DOld(input_shape=(None, 297), features=64):
#     input_layer = Input(input_shape)
#
#     # Conv
#     x = Conv1D(
#         features,
#         kernel_size=5,
#         dilation_rate=2,
#         strides=1,
#         padding="same",
#     )(input_layer)
#     x = Activation("relu")(x)
#     x = Conv1D(
#         features,
#         kernel_size=5,
#         dilation_rate=2,
#         strides=1,
#         padding="same",
#     )(x)
#     x = Activation("relu")(x)
#
#     # Regressor
#     x = Conv1D(
#         1,
#         kernel_size=1,
#         strides=1,
#         padding="same",
#     )(x)
#     x = Activation("sigmoid")(x)
#     out = Flatten()(x)
#
#     model = Model(input_layer, out)
#
#     return model


def CNN1D(input_shape=(None, 297), max_dil=512, features=64, reverse_dil=True):
    input_layer = Input(input_shape)

    # Conv
    dilations = [1] + [2**i for i in range(int(log2(max_dil)) + 1)]
    if reverse_dil:
        dilations = list(reversed(dilations))
    x = input_layer
    for dil in dilations:
        x = Conv1D(
            features,
            kernel_size=3,
            dilation_rate=dil,
            strides=1,
            padding="same",
        )(x)
        x = Activation("relu")(x)

    # Regressor
    x = Conv1D(
        1,
        kernel_size=1,
        strides=1,
        padding="same",
    )(x)
    x = Activation("sigmoid")(x)
    out = Flatten()(x)

    model = Model(input_layer, out)

    return model
