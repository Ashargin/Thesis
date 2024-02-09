import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

# import torch
# from transformers import AutoTokenizer, AutoModel
from scipy import signal
from pathlib import Path

from src.models.mlp import MLP
from src.models.cnn_1d import CNN1D
from src.models.loss import inv_exp_distance_to_cut_loss

# from src.utils import seq2kmer

MAX_MOTIFS = 50
MAX_DIL = 2048

# Load model
model = CNN1D(input_shape=(None, MAX_MOTIFS + 4), max_dil=MAX_DIL)
model.compile(
    optimizer="adam",
    loss=inv_exp_distance_to_cut_loss,
    metrics=["accuracy"],
    run_eagerly=True,
)


# Dataloader
def motif_cache_data_generator(folder_path, max_motifs=MAX_MOTIFS, max_len=None):
    files = os.listdir(folder_path)
    np.random.shuffle(files)

    df_motifs = pd.read_csv(Path("resources/motif_seqs.csv"), index_col=0)
    df_motifs = df_motifs[df_motifs.time < 0.012].reset_index(drop=True)
    max_motifs = df_motifs.shape[0] if max_motifs is None else max_motifs
    used_index = (
        df_motifs.sort_index().sort_values("time").index[:max_motifs].sort_values()
    )
    used_index = [0, 1, 2, 3] + (used_index + 4).to_list()  # add one-hot

    seq_mat, cuts_mat, outer = None, None, None
    i = 0
    while True:
        with open(folder_path / files[i % len(files)], "rb") as infile:
            seq_mat, cuts_mat, outer = pickle.load(infile)

        seq_mat = seq_mat.toarray()
        cuts_mat = cuts_mat.toarray()
        # outer = outer.toarray()

        seq_mat = seq_mat[:, used_index]  # keep top max_motifs motifs
        cuts_mat = np.where(cuts_mat.ravel() == 1)[0].astype(float)  # cut indices

        i += 1
        if max_len is not None and seq_mat.shape[0] > max_len:
            continue

        yield seq_mat.reshape((1, seq_mat.shape[0], max_motifs + 4)), cuts_mat.reshape(
            (1, cuts_mat.shape[0])
        )


def dnabert_data_generator(csv_path, max_len=None):
    dnabert_tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNA_bert_6", trust_remote_code=True
    )
    dnabert_encoder = AutoModel.from_pretrained(
        "zhihan1996/DNA_bert_6", trust_remote_code=True
    )

    df = pd.read_csv(csv_path, index_col=0)
    df = df.sample(frac=1.0)

    seq_mat, cuts_mat, outer = None, None, None
    i = 0
    while True:
        seq = df.iloc[i].seq
        cuts = [int(x) for x in df.iloc[i].cuts[1:-1].split()]

        i += 1
        if max_len is not None and len(seq) > max_len:
            continue

        tokenized = dnabert_tokenizer(
            seq2kmer(seq.replace("U", "T"), k=6),
            padding="longest",
            pad_to_multiple_of=512,
        )
        encoded = dnabert_encoder(
            torch.tensor([tokenized["input_ids"]]).view(-1, 512),
            torch.tensor([tokenized["attention_mask"]]).view(-1, 512),
        )
        seq_mat, pooled_seq_mat = encoded[0], encoded[1]

        tokens_len = len(seq) - 3
        seq_mat = np.vstack(seq_mat.detach().numpy())[:tokens_len]
        seq_mat = np.vstack(
            [seq_mat[0], seq_mat[0], seq_mat, seq_mat[-1]]
        )  ###### fix size ?
        # pooled_seq_mat = np.mean(pooled_seq_mat.detach().numpy(), axis=0)
        cuts_mat = np.array(cuts)

        # explore LLMs / generatives / T5

        yield seq_mat.reshape((1, len(seq), 768)), cuts_mat.reshape((1, len(cuts)))


# Fit model
train_path = Path("resources/data_splits/train_sequencewise")
test_path = Path("resources/data_splits/test_sequencewise")
# train_csv_path = Path("resources/data/train.csv")
# test_csv_path = Path("resources/data/test.csv")
# n_train = len(os.listdir(train_path))
# n_test = len(os.listdir(test_path))
history = model.fit(
    motif_cache_data_generator(train_path),
    validation_data=motif_cache_data_generator(test_path),
    steps_per_epoch=1267,
    epochs=10,
    validation_steps=305,
)

model.save(Path(f"resources/models/CNN1D_sequencewise_{MAX_MOTIFS}motifs{MAX_DIL}dil"))

import matplotlib.pyplot as plt

loss = history.history["loss"]
val_loss = history.history["val_loss"]
X = np.arange(len(loss))
plt.plot(X, loss, label="Train loss")
plt.plot(X, val_loss, label="Validation loss")
plt.legend()
plt.xlim(([0, 100]))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss curve (sequence-wise train / test split)")
plt.savefig(
    rf"resources/png/training_curve_cnn_sequencewise_{MAX_MOTIFS}motifs{MAX_DIL}dil.png"
)
plt.show()

my_model = keras.models.load_model(Path("resources/models/model_motifs"), compile=False)
my_model.compile(
    optimizer="adam",
    loss=inv_exp_distance_to_cut_loss,
    metrics=["accuracy"],
    run_eagerly=True,
)
test_datagen = motif_cache_data_generator(Path("resources/data/temptest"))


def plot_cut_probabilities():
    seq_mat, cuts_mat = next(test_datagen)
    preds = my_model(seq_mat).numpy().ravel()
    cuts_mat = cuts_mat.ravel().astype(int)

    X = np.arange(len(preds)) + 1
    for i, x in enumerate(X[cuts_mat]):
        plt.plot(
            [x, x],
            [0, 1],
            color="black",
            linewidth=1.5,
            label="True cut points" if i == 0 else "",
        )
    plt.plot(X, preds, color="tab:orange", label="Predicted probabilities to cut")

    peaks = signal.find_peaks(preds, height=0.28, distance=12)[0]
    plt.plot(X[peaks], preds[peaks], "o", color="tab:blue", label="Selected cut points")

    plt.xlim([X[0], X[-1]])
    plt.ylim([0, 1])

    plt.title(
        "Predicted cutting probabilities and selected cut points\ncompared to true cut points"
    )
    plt.legend()
    plt.show()
