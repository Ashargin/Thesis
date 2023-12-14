from keras.layers import (
    Dense,
    Conv1D,
    BatchNormalization,
    Activation,
    AveragePooling1D,
    GlobalAveragePooling1D,
    Input,
    Concatenate,
    Add,
    UpSampling1D,
    Multiply,
    Flatten,
    ZeroPadding1D,
)
from keras.models import Model


def cbr(x, out_layer, kernel, stride, dilation):
    x = Conv1D(
        out_layer,
        kernel_size=kernel,
        dilation_rate=dilation,
        strides=stride,
        padding="same",
    )(x)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def se_block(x_in, layer_n):
    x = GlobalAveragePooling1D()(x_in)
    x = Dense(layer_n // 8, activation="relu")(x)
    x = Dense(layer_n, activation="sigmoid")(x)
    x_out = Multiply()([x_in, x])
    return x_out


def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = Add()([x_in, x])
    return x


def Unet1D(input_shape=(500, 297), layer_n=64, kernel_size=7, depth=2):
    input_layer = Input(input_shape)

    input_layer_1 = AveragePooling1D(5)(input_layer)
    input_layer_2 = AveragePooling1D(25)(input_layer)

    # Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 2, kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])
    x = cbr(x, layer_n * 3, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 3, kernel_size, 1)
    out_2 = x

    x = Concatenate()([x, input_layer_2])
    x = cbr(x, layer_n * 4, kernel_size, 5, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 4, kernel_size, 1)

    # Decoder
    x = UpSampling1D(5)(x)
    x = Concatenate()([x, out_2])
    x = cbr(x, layer_n * 3, kernel_size, 1, 1)

    x = UpSampling1D(5)(x)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n * 2, kernel_size, 1, 1)

    x = UpSampling1D(5)(x)
    x = Concatenate()([x, out_0])
    x = cbr(x, layer_n, kernel_size, 1, 1)

    # Regressor
    x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = Activation("sigmoid")(x)
    out = Flatten()(x)

    model = Model(input_layer, out)

    return model
