import os
import pickle
import numpy as np
from tensorflow import keras
from models import TransformerModel, NonTransformerModel, min_distance_to_cut_loss, inv_exp_distance_to_cut_loss

# Load model
model = NonTransformerModel()
model.model.compile(optimizer='adam', loss=inv_exp_distance_to_cut_loss, metrics=['accuracy'],
                                                                            run_eagerly=True)


# Dataloader
def data_generator(folder_path, max_len=1500):
    files = os.listdir(folder_path)
    np.random.shuffle(files)

    seq_mat, cuts_mat, outer = None, None, None
    i = 0
    while True:
        with open(os.path.join(folder_path, files[i % len(files)]), 'rb') as infile:
            seq_mat, cuts_mat, outer = pickle.load(infile)

        seq_mat = seq_mat.toarray()
        cuts_mat = cuts_mat.toarray()
        # outer = outer.toarray()

        cuts_mat = np.where(cuts_mat.ravel() == 1)[0].astype(float) # cut indices

        i += 1

        if seq_mat.shape[0] > max_len:
            continue

        yield seq_mat.reshape((1, -1, 297)), cuts_mat.reshape((1, -1))


# Fit model
train_path = r'resources/data/formatted_train'
test_path = r'resources/data/formatted_test'
n_train = len(os.listdir(train_path))
n_test = len(os.listdir(test_path))
history = model.model.fit(data_generator(train_path),
                          validation_data=data_generator(test_path),
                          steps_per_epoch = 1267,
                          epochs = 100,
                          validation_steps = 305,)

model.model.save(r'resources/models/model')

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
X = np.arange(len(loss))
plt.plot(X, loss, label='loss')
plt.plot(X, val_loss, label='val_loss')
plt.legend()
plt.show()

my_model = keras.models.load_model(r'resources/models/model',
                                   custom_objects={'inv_exp_distance_to_cut_loss':
                                                   inv_exp_distance_to_cut_loss})


def plot_cut_probabilities():
    datagen = data_generator(test_path)
    seq_mat, cuts_mat = next(datagen)
    preds = my_model(seq_mat).numpy().ravel()
    y_true = np.zeros_like(preds)
    y_true[cuts_mat.ravel().astype(int)] = 1
    y_true /= y_true.sum()

    X = np.arange(len(preds)) + 1
    plt.plot(X, y_true, 'o', label='true')
    plt.plot(X, preds, label='preds')

    # peaks = signal.find_peaks(preds, height=0.33, distance=15)[0]
    # for x in peaks:
    #     plt.plot([x, x], [0, 1], color='black')
    # print(peaks)

    plt.legend()
    plt.show()
