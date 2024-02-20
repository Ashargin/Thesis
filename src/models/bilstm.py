from keras.layers import Bidirectional, LSTM, Dense, Activation, Input, Flatten
from keras.models import Model


def BiLSTM(input_shape=(None, 297), features=32):
    input_layer = Input(input_shape)

    # LSTM
    x = Bidirectional(LSTM(features, return_sequences=True))(input_layer)
    x = Bidirectional(LSTM(features, return_sequences=True))(x)

    # Regressor
    x = Dense(1)(x)
    x = Activation("sigmoid")(x)
    out = Flatten()(x)

    model = Model(input_layer, out)

    return model
