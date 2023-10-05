import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# from keras_nlp import layers as layers_nlp


class TransformerModel:
    has_transformer = True

    def __init__(self):
        self.model = self.get_model()

    def get_model(self):
        # Create classifier model using transformer layer
        transformer_ff_dim = 64  # Feed forward network size inside transformer
        num_heads = 8  # Number of attention heads
        dropout_rate = 0.1
        middle_dense_dim = 16

        inputs = layers.Input(shape=(None, 297))
        # inputs = layers.Input(shape=(None, 768))
        transformed = (
            layers_nlp.TransformerEncoder(
                transformer_ff_dim, num_heads, dropout=dropout_rate
            )(inputs)
            if self.has_transformer
            else inputs
        )

        drop1 = layers.Dropout(dropout_rate)(transformed)
        dense1 = layers.Dense(middle_dense_dim, activation="relu")(drop1)
        drop2 = layers.Dropout(dropout_rate)(dense1)

        # Old cuts with independant probability at each position
        dense2 = layers.Dense(1, activation="sigmoid")(drop2)
        pred_cuts = layers.Flatten()(dense2)

        # New cuts with ensemble probability (softmax)
        # dense2 = layers.Dense(1, activation='linear')(drop2)
        # flattened = layers.Flatten()(dense2)
        # pred_cuts = layers.Softmax()(flattened)

        # Outer pred
        # global_pooled = layers.GlobalAveragePooling1D()(transformed)
        # drop1 = layers.Dropout(dropout_rate)(global_pooled)
        # dense1 = layers.Dense(middle_dense_dim, activation='relu')(drop1)
        # drop2 = layers.Dropout(dropout_rate)(dense1)
        # pred_outer = layers.Dense(1, activation='sigmoid')(drop2)

        # Train and evaluate
        model = keras.Model(inputs=inputs, outputs=pred_cuts)

        return model


class NonTransformerModel(TransformerModel):
    has_transformer = False


def min_distance_to_cut_loss(y_true, y_pred):
    loss_array = np.abs(
        y_true.numpy() - np.arange(y_pred.shape[1]).reshape((-1, 1))
    ).min(axis=1)
    loss_array = loss_array**2

    return y_pred * loss_array


def inv_exp_distance_to_cut_loss(y_true, y_pred, lbda=0.5):
    loss_array = np.abs(
        y_true.numpy() - np.arange(y_pred.shape[1]).reshape((-1, 1))
    ).min(axis=1)
    loss_array = np.exp(-lbda * loss_array)

    return (y_pred - loss_array) ** 2
