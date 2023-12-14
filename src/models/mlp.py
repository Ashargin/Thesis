from keras.layers import Input, Dropout, Dense, Flatten, Softmax, GlobalAveragePooling1D

# from keras_nlp.layers import TransformerEncoder
from keras.models import Model


def MLP(input_shape=(None, 297), with_transformer=False):
    # Create classifier model using transformer layer
    transformer_ff_dim = 64  # Feed forward network size inside transformer
    num_heads = 8  # Number of attention heads
    dropout_rate = 0.1
    middle_dense_dim = 16

    inputs = Input(shape=input_shape)
    transformed = (
        TransformerEncoder(transformer_ff_dim, num_heads, dropout=dropout_rate)(inputs)
        if with_transformer
        else inputs
    )

    drop1 = Dropout(dropout_rate)(transformed)
    dense1 = Dense(middle_dense_dim, activation="relu")(drop1)
    drop2 = Dropout(dropout_rate)(dense1)

    # Old cuts with independant probability at each position
    dense2 = Dense(1, activation="sigmoid")(drop2)
    pred_cuts = Flatten()(dense2)

    # New cuts with ensemble probability (softmax)
    # dense2 = Dense(1, activation='linear')(drop2)
    # flattened = Flatten()(dense2)
    # pred_cuts = Softmax()(flattened)

    # Outer pred
    # global_pooled = GlobalAveragePooling1D()(transformed)
    # drop1 = Dropout(dropout_rate)(global_pooled)
    # dense1 = Dense(middle_dense_dim, activation='relu')(drop1)
    # drop2 = Dropout(dropout_rate)(dense1)
    # pred_outer = Dense(1, activation='sigmoid')(drop2)

    # Train and evaluate
    model = Model(inputs=inputs, outputs=pred_cuts)

    return model
