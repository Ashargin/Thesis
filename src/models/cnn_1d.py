from keras.layers import Conv1D, Activation, Input, Flatten
from keras.models import Model


def CNN1D(input_shape=(None, 297)):
    input_layer = Input(input_shape)

    # Conv
    x = Conv1D(
        64,
        kernel_size=5,
        dilation_rate=2,
        strides=1,
        padding="same",
    )(input_layer)
    x = Activation("relu")(x)
    x = Conv1D(
        64,
        kernel_size=5,
        dilation_rate=2,
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
