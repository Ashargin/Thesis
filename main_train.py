import os
import pickle5 as pickle
import numpy as np
import pandas as pd
from tensorflow import keras
import torch
from transformers import AutoTokenizer, AutoModel
from scipy import signal

from models import TransformerModel, NonTransformerModel, min_distance_to_cut_loss, \
                                                          inv_exp_distance_to_cut_loss
from utils import seq2kmer

# Load model
model = NonTransformerModel()
model.model.compile(optimizer='adam', loss=inv_exp_distance_to_cut_loss, metrics=['accuracy'],
                                                                            run_eagerly=True)


# Dataloader
def motif_cache_data_generator(folder_path, max_len=None):
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
        if max_len is not None and seq_mat.shape[0] > max_len:
            continue

        yield seq_mat.reshape((1, seq_mat.shape[0], 297)), cuts_mat.reshape((1, cuts_mat.shape[0]))


def dnabert_data_generator(csv_path, max_len=None):
    dnabert_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6",
                                                        trust_remote_code=True)
    dnabert_encoder = AutoModel.from_pretrained("zhihan1996/DNA_bert_6",
                                                        trust_remote_code=True)

    df = pd.read_csv(csv_path, index_col=0)
    df = df.sample(frac=1.)

    seq_mat, cuts_mat, outer = None, None, None
    i = 0
    while True:
        seq = df.iloc[i].seq
        cuts = [int(x) for x in df.iloc[i].cuts[1:-1].split()]

        i += 1
        if max_len is not None and len(seq) > max_len:
            continue

        tokenized = dnabert_tokenizer(seq2kmer(seq.replace('U', 'T'), k=6),
                                        padding='longest',
                                        pad_to_multiple_of=512,
                                        )
        encoded = dnabert_encoder(torch.tensor([tokenized['input_ids']]).view(-1, 512),
                                  torch.tensor([tokenized['attention_mask']]).view(-1, 512))
        seq_mat, pooled_seq_mat = encoded[0], encoded[1]

        tokens_len = len(seq) - 3
        seq_mat = np.vstack(seq_mat.detach().numpy())[:tokens_len]
        seq_mat = np.vstack([seq_mat[0], seq_mat[0], seq_mat, seq_mat[-1]]) ###### fix size ?
        # pooled_seq_mat = np.mean(pooled_seq_mat.detach().numpy(), axis=0)
        cuts_mat = np.array(cuts)

        # explore LLMs / generatives / T5

        yield seq_mat.reshape((1, len(seq), 768)), cuts_mat.reshape((1, len(cuts)))


# Fit model
train_path = r'resources/data/formatted_train'
test_path = r'resources/data/formatted_test'
# train_csv_path = r'resources\data\train.csv'
# test_csv_path = r'resources\data\test.csv'
n_train = len(os.listdir(train_path))
n_test = len(os.listdir(test_path))
history = model.model.fit(motif_cache_data_generator(train_path),
                          validation_data=motif_cache_data_generator(test_path),
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

my_model = keras.models.load_model(r'resources/models/model', compile=False)
my_model.compile(optimizer='adam', loss=inv_exp_distance_to_cut_loss, metrics=['accuracy'],
                                                                            run_eagerly=True)
test_datagen = motif_cache_data_generator(test_path)


def plot_cut_probabilities():
    seq_mat, cuts_mat = next(test_datagen)
    preds = my_model(seq_mat).numpy().ravel()
    y_true = np.zeros_like(preds)
    y_true[cuts_mat.ravel().astype(int)] = 1
    y_true /= y_true.sum()

    X = np.arange(len(preds)) + 1
    plt.plot(X, y_true, 'o', label='true')
    plt.plot(X, preds, label='preds')

    peaks = signal.find_peaks(preds, height=0.28, distance=12)[0]
    for x in peaks:
        plt.plot([x, x], [0, 1], color='black')
    print(peaks)

    plt.legend()
    plt.show()
